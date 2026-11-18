"""vidkit: the drawing, narration and rendering layer shared by every video under Video/.

A video is a folder holding a `script.yaml` (the cues) and, optionally, a `visuals.py`
(that video's own generated pictures). `build_video.py` reads the script, makes the
narration, draws or loads every cue's picture and writes one MP4 plus an SRT.

Everything is drawn in a 1920x1080 *reference* coordinate system and scaled to the
output size by `Frame`, so a quick preview at half size draws exactly the same picture.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import wave
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REF_W, REF_H = 1920, 1080
SAMPLE_RATE = 44100
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------
# Palette (RGB). Dark background, so the pictures read on a monitor and in a headset alike.
# ---------------------------------------------------------------------------------------------
BG = (16, 20, 29)
PANEL = (26, 32, 45)
INK = (236, 239, 244)
MUTED = (150, 160, 178)
FAINT = (78, 88, 106)
ACCENT = (255, 176, 59)      # amber: the thing to look at
CYAN = (86, 204, 242)
GREEN = (110, 214, 142)
RED = (240, 96, 96)
MAGENTA = (232, 112, 210)
GROUND = (35, 47, 38)        # grass round the city
ASPHALT = (52, 56, 64)
FOOTPATH = (88, 92, 100)
BUILDING = (122, 136, 160)
BUILDING_EDGE = (182, 194, 214)
HIDDEN = (110, 116, 128)

# One colour per drone, so a drone and its screen can be matched by eye.
DRONE_HUES = [(255, 120, 98), (255, 176, 59), (240, 226, 90), (140, 220, 110), (86, 214, 180),
              (86, 190, 242), (120, 140, 255), (186, 120, 255), (240, 112, 200), (200, 200, 210)]

FONT_FILES = {
    "regular": ["segoeui.ttf", "arial.ttf"],
    "semibold": ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"],
    "bold": ["segoeuib.ttf", "arialbd.ttf"],
    "light": ["segoeuil.ttf", "segoeui.ttf", "arial.ttf"],
}
_font_cache: dict = {}


def font(size: float, weight: str = "regular") -> ImageFont.FreeTypeFont:
    key = (int(round(size)), weight)
    if key not in _font_cache:
        f = None
        for name in FONT_FILES.get(weight, FONT_FILES["regular"]):
            for root in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu"):
                p = os.path.join(root, name)
                if os.path.exists(p):
                    f = ImageFont.truetype(p, key[0])
                    break
            if f:
                break
        _font_cache[key] = f or ImageFont.load_default()
    return _font_cache[key]


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def smoothstep(e0, e1, x):
    """GLSL smoothstep: 0 below e0, 1 above e1, eased between."""
    if e1 == e0:
        return 1.0 if x >= e1 else 0.0
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def ease(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def mix(c0, c1, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c0, c1))


# ---------------------------------------------------------------------------------------------
# Frame: one picture, drawn in reference pixels.
# ---------------------------------------------------------------------------------------------
_SHIFT = 4
_ONE = 1 << _SHIFT


class Frame:
    """An RGB image addressed in 1920x1080 reference pixels.

    Shapes go straight onto the array with OpenCV (anti-aliased, sub-pixel); text is queued
    and drawn with PIL in `finish`, so every text call in a frame costs one conversion.
    """

    def __init__(self, k: float = 1.0, bg=BG, img: np.ndarray | None = None):
        self.k = k
        self.W, self.H = int(round(REF_W * k)), int(round(REF_H * k))
        if img is not None:
            self.img = img
        else:
            self.img = np.empty((self.H, self.W, 3), np.uint8)
            self.img[:] = bg
        self._texts = []

    # -- coordinates -------------------------------------------------------------------------
    def _p(self, pts):
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2) * self.k * _ONE
        return np.round(a).astype(np.int32)

    def _w(self, w):
        return max(1, int(round(w * self.k)))

    def _bbox(self, pts, pad=4):
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2) * self.k
        x0 = int(max(0, math.floor(a[:, 0].min()) - pad))
        y0 = int(max(0, math.floor(a[:, 1].min()) - pad))
        x1 = int(min(self.W, math.ceil(a[:, 0].max()) + pad + 1))
        y1 = int(min(self.H, math.ceil(a[:, 1].max()) + pad + 1))
        return x0, y0, x1, y1

    def _alpha(self, pts, draw, alpha, pad=4):
        """Run draw(img, offset) at full opacity on a copy of the region, then blend it in."""
        if alpha >= 0.999:
            draw(self.img, (0, 0))
            return
        if alpha <= 0.001:
            return
        x0, y0, x1, y1 = self._bbox(pts, pad)
        if x1 <= x0 or y1 <= y0:
            return
        sub = self.img[y0:y1, x0:x1]
        ov = np.ascontiguousarray(sub.copy())
        draw(ov, (x0, y0))
        cv2.addWeighted(ov, alpha, sub, 1 - alpha, 0, dst=sub)

    @staticmethod
    def _off(p, off):
        return p - np.array([off[0] * _ONE, off[1] * _ONE], np.int32)

    # -- shapes ------------------------------------------------------------------------------
    def poly(self, pts, color, alpha=1.0, outline=None, width=2.0):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            return
        P = self._p(pts)
        w = self._w(width)

        def draw(img, off):
            q = self._off(P, off)
            if color is not None:
                cv2.fillPoly(img, [q], color, cv2.LINE_AA, _SHIFT)
            if outline is not None:
                cv2.polylines(img, [q], True, outline, w, cv2.LINE_AA, _SHIFT)
        self._alpha(pts, draw, alpha, pad=w + 2)

    def polyline(self, pts, color, width=2.0, closed=False, alpha=1.0):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            return
        P = self._p(pts)
        w = self._w(width)

        def draw(img, off):
            cv2.polylines(img, [self._off(P, off)], closed, color, w, cv2.LINE_AA, _SHIFT)
        self._alpha(pts, draw, alpha, pad=w + 2)

    def dashed(self, pts, color, width=2.0, dash=14.0, gap=10.0, closed=False, alpha=1.0, phase=0.0):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        seg = np.diff(pts, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0], np.cumsum(L)])
        total = cum[-1]
        s = -phase % (dash + gap)
        s -= dash + gap
        while s < total:
            a, b = max(0.0, s), min(total, s + dash)
            if b > a:
                self.polyline([_along(pts, cum, a), _along(pts, cum, b)], color, width, alpha=alpha)
            s += dash + gap

    def line(self, p0, p1, color, width=2.0, alpha=1.0):
        self.polyline([p0, p1], color, width, alpha=alpha)

    def circle(self, c, r, color, alpha=1.0, width=None):
        """Filled when width is None, else an outline of that width."""
        c = np.asarray(c, float)
        P = self._p(c)[0]
        R = int(round(r * self.k * _ONE))
        w = -1 if width is None else self._w(width)
        bb = [c - r - (width or 0), c + r + (width or 0)]

        def draw(img, off):
            q = self._off(P[None], off)[0]
            cv2.circle(img, (int(q[0]), int(q[1])), max(1, R), color, w, cv2.LINE_AA, _SHIFT)
        self._alpha(bb, draw, alpha)

    def arc(self, c, r, a0_deg, a1_deg, color, width=2.0, alpha=1.0, n=48):
        """Arc in screen angles (0 = right, 90 = down)."""
        a = np.radians(np.linspace(a0_deg, a1_deg, n))
        self.polyline(np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)], color, width, alpha=alpha)

    def wedge(self, c, r, a0_deg, a1_deg, color, alpha=1.0, n=36, outline=None):
        a = np.radians(np.linspace(a0_deg, a1_deg, n))
        pts = np.vstack([[c], np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]])
        self.poly(pts, color, alpha, outline=outline)

    def arrow(self, p0, p1, color, width=4.0, head=16.0, alpha=1.0, double=False):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.hypot(*d)
        if L < 1e-6:
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        h = min(head, L * 0.6)
        shaft_end = p1 - u * h * 0.8
        shaft_start = p0 + (u * h * 0.8 if double else 0)
        self.polyline([shaft_start, shaft_end], color, width, alpha=alpha)
        self.poly([p1, p1 - u * h + n * h * 0.55, p1 - u * h - n * h * 0.55], color, alpha)
        if double:
            self.poly([p0, p0 + u * h + n * h * 0.55, p0 + u * h - n * h * 0.55], color, alpha)

    def rect(self, x, y, w, h, color, alpha=1.0, radius=0.0, outline=None, width=2.0):
        self.poly(rounded_rect(x, y, w, h, radius), color, alpha, outline=outline, width=width)

    def image(self, img_rgb, x, y, w, h, alpha=1.0, mode="cover"):
        """Paste an RGB array into the reference rect (x, y, w, h)."""
        X, Y = int(round(x * self.k)), int(round(y * self.k))
        Wd, Hd = max(1, int(round(w * self.k))), max(1, int(round(h * self.k)))
        sh, sw = img_rgb.shape[:2]
        if mode == "cover":
            s = max(Wd / sw, Hd / sh)
        else:
            s = min(Wd / sw, Hd / sh)
        nw, nh = max(1, int(round(sw * s))), max(1, int(round(sh * s)))
        r = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if mode == "cover":
            cx, cy = (nw - Wd) // 2, (nh - Hd) // 2
            r = r[cy:cy + Hd, cx:cx + Wd]
            ox, oy = X, Y
        else:
            ox, oy = X + (Wd - nw) // 2, Y + (Hd - nh) // 2
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(self.W, ox + r.shape[1]), min(self.H, oy + r.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        src = r[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        dst = self.img[y0:y1, x0:x1]
        if alpha >= 0.999:
            dst[:] = src
        else:
            cv2.addWeighted(src, alpha, dst, 1 - alpha, 0, dst=dst)

    # -- text --------------------------------------------------------------------------------
    def text(self, s, xy, size=36, color=INK, weight="regular", anchor="la", max_width=None,
             align="left", alpha=1.0, line_spacing=1.18):
        """Queue text. anchor is PIL's two letters (l/m/r, a/m/d...). max_width wraps, in ref px."""
        if not s or alpha <= 0.001:
            return
        self._texts.append((s, xy, size, color, weight, anchor, max_width, align, alpha, line_spacing))

    def text_size(self, s, size=36, weight="regular", max_width=None, line_spacing=1.18):
        f = font(size, weight)
        lines = wrap(s, f, max_width) if max_width else s.split("\n")
        w = max((f.getlength(l) for l in lines), default=0)
        h = size * line_spacing * len(lines)
        return w, h

    def finish(self) -> np.ndarray:
        if not self._texts:
            return self.img
        base = Image.fromarray(self.img)
        need_layer = any(t[8] < 0.999 for t in self._texts)
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0)) if need_layer else None
        d_base = ImageDraw.Draw(base)
        d_layer = ImageDraw.Draw(layer) if layer is not None else None
        for (s, xy, size, color, weight, anchor, max_width, align, alpha, ls) in self._texts:
            f = font(size * self.k, weight)
            lines = wrap(s, f, max_width * self.k) if max_width else s.split("\n")
            lh = size * self.k * ls
            x, y = xy[0] * self.k, xy[1] * self.k
            total_h = lh * len(lines)
            v = anchor[1] if len(anchor) > 1 else "a"
            if v == "m":
                y -= total_h / 2
            elif v in ("d", "b"):
                y -= total_h
            h_anchor = anchor[0]
            draw = d_base if alpha >= 0.999 else d_layer
            fill = tuple(color) + ((int(255 * alpha),) if alpha < 0.999 else ())
            for i, line in enumerate(lines):
                draw.text((x, y + i * lh), line, font=f, fill=fill, anchor=h_anchor + "a")
        if layer is not None:
            base = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
        self._texts = []
        self.img = np.asarray(base).copy()
        return self.img


def _along(pts, cum, s):
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(pts) - 2))
    L = cum[i + 1] - cum[i]
    t = 0 if L <= 0 else (s - cum[i]) / L
    return pts[i] + (pts[i + 1] - pts[i]) * t


def wrap(s, f, max_w):
    out = []
    for para in s.split("\n"):
        words = para.split(" ")
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip()
            if f.getlength(cand) <= max_w or not cur:
                cur = cand
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out


def rounded_rect(x, y, w, h, r, n=8):
    r = max(0.0, min(r, w / 2, h / 2))
    if r <= 0:
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)):
        for a in np.radians(np.linspace(a0, a0 + 90, n)):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


# ---------------------------------------------------------------------------------------------
# World view: Unity (x, z) metres onto reference pixels. Yaw is Unity's: 0 = +z, 90 = +x.
# ---------------------------------------------------------------------------------------------
@dataclass
class View:
    center: tuple          # world (x, z) at the screen centre point
    scale: float           # reference px per metre
    screen: tuple = (960, 540)
    up_yaw: float = 0.0    # this world heading points up the screen

    def __post_init__(self):
        a = math.radians(self.up_yaw)
        self._h = np.array([math.sin(a), math.cos(a)])
        self._r = np.array([math.cos(a), -math.sin(a)])

    def __call__(self, pts):
        p = np.asarray(pts, float)
        d = p - np.asarray(self.center, float)
        sx = self.screen[0] + self.scale * (d @ self._r)
        sy = self.screen[1] - self.scale * (d @ self._h)
        return np.stack([sx, sy], axis=-1)

    def screen_angle(self, yaw_deg):
        """The screen angle (0 = right, 90 = down, degrees) a world heading points along."""
        return yaw_deg - self.up_yaw - 90.0


def yaw_vec(yaw_deg):
    a = math.radians(yaw_deg)
    return np.array([math.sin(a), math.cos(a)])


def vec_yaw(v):
    return math.degrees(math.atan2(v[0], v[1]))


# ---------------------------------------------------------------------------------------------
# Narration: Windows SAPI (offline), or the user's own recording if one is in narration/.
# ---------------------------------------------------------------------------------------------
def tts_key(text, voice, rate):
    return hashlib.sha1(f"{voice}|{rate}|{text}".encode("utf-8")).hexdigest()[:16]


def synthesize(jobs, cache_dir):
    """jobs: [{text, voice, rate, out}]. Runs the SAPI script once for all missing ones."""
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    if not todo:
        return
    os.makedirs(cache_dir, exist_ok=True)
    jf = os.path.join(cache_dir, "jobs.json")
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(todo, f, ensure_ascii=False)
    ps1 = os.path.join(TOOLS_DIR, "tts_sapi.ps1")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1, jf],
                       capture_output=True, text=True)
    missing = [j["out"] for j in todo if not os.path.exists(j["out"])]
    if r.returncode != 0 or missing:
        raise RuntimeError(f"TTS failed ({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")


ELEVEN_KEY_FILE = os.path.join(os.path.expanduser("~"), ".elevenlabs_api_key")


def eleven_key():
    """The ElevenLabs secret key (sk_...): $ELEVENLABS_API_KEY, else ~/.elevenlabs_api_key. Never in the repo."""
    k = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not k and os.path.exists(ELEVEN_KEY_FILE):
        with open(ELEVEN_KEY_FILE, encoding="utf-8") as f:
            k = f.read().strip()
    return k or None


def eleven_synthesize(jobs, cache_dir, key):
    """jobs: [{text, voice_id, model, settings, out}] -> 44.1 kHz mono WAVs. Raises on any API error."""
    import urllib.request
    import urllib.error
    for j in jobs:
        if os.path.exists(j["out"]):
            continue
        body = {"text": j["text"], "model_id": j["model"], "voice_settings": j.get("settings") or {}}
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{j['voice_id']}?output_format=mp3_44100_128",
            data=json.dumps(body).encode("utf-8"),
            headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                audio = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ElevenLabs {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        mp3 = os.path.splitext(j["out"])[0] + ".mp3"
        with open(mp3, "wb") as f:
            f.write(audio)
        to_pcm(mp3, j["out"])


def to_pcm(src, dst):
    """Any audio file -> 44.1 kHz mono 16-bit WAV (for the user's own narration takes)."""
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-sample_fmt", "s16", dst], check=True)


def read_wav(path) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit audio")
    a = np.frombuffer(raw, np.int16).astype(np.float32)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        t = np.arange(int(len(a) * SAMPLE_RATE / sr)) * sr / SAMPLE_RATE
        a = np.interp(t, np.arange(len(a)), a)
    return trim_silence(a)


def trim_silence(a, thresh=250):
    idx = np.nonzero(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a[:0]
    pad = int(0.03 * SAMPLE_RATE)
    return a[max(0, idx[0] - pad): min(len(a), idx[-1] + pad)]


def write_wav(path, a):
    a = np.clip(a, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(a.tobytes())


# ---------------------------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------------------------
def caption_chunks(text, max_chars=92):
    """Sentences, split again at commas/dashes when longer than max_chars."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.findall(r"[^.!?]+[.!?]*[\"')]*", text)
    out = []
    for s in (x.strip() for x in sentences):
        if not s:
            continue
        while len(s) > max_chars:
            cut = -1
            for m in re.finditer(r"[,;:\u2014-]\s", s):
                if 20 < m.end() <= max_chars:
                    cut = m.end()
            if cut < 0:
                cut = s.rfind(" ", 0, max_chars)
                if cut <= 0:
                    break
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def time_chunks(chunks, start, dur):
    """Each chunk gets time in proportion to its length."""
    total = sum(len(c) + 8 for c in chunks) or 1
    t, out = start, []
    for c in chunks:
        d = dur * (len(c) + 8) / total
        out.append((t, t + d, c))
        t += d
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def draw_caption(fr: Frame, text, alpha=1.0):
    size = 38
    maxw = 1500
    f = font(size, "regular")
    lines = wrap(text, f, maxw)
    w = max(f.getlength(l) for l in lines)
    h = size * 1.25 * len(lines)
    x, y = 960 - w / 2, 1032 - h
    fr.rect(x - 22, y - 12, w + 44, h + 22, (8, 10, 14), alpha=0.72 * alpha, radius=10)
    fr.text(text, (960, y), size, INK, anchor="ma", max_width=maxw, align="center", alpha=alpha,
            line_spacing=1.25)


# ---------------------------------------------------------------------------------------------
# Footage
# ---------------------------------------------------------------------------------------------
class Footage:
    """Sequential reader returning RGB frames by source time; holds the last frame at the end."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = n / self.fps if n > 0 else 0.0
        self.idx = -1
        self.last = None

    def at(self, t):
        want = max(0, int(t * self.fps))
        if want < self.idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            self.idx = want - 1
        while self.idx < want:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx += 1
            self.last = f
        if self.last is None:
            return None
        return cv2.cvtColor(self.last, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


def probe_duration(path):
    try:
        f = Footage(path)
        d = f.duration
        f.close()
        return d
    except IOError:
        return 0.0


# ---------------------------------------------------------------------------------------------
# Generic visuals, available to every video (a video's visuals.py adds its own).
# ---------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    t: float                 # seconds into the cue
    dur: float               # cue length, seconds
    k: float                 # output / reference scale
    params: dict
    cue: dict
    state: dict = field(default_factory=dict)   # per-cue cache, kept across its frames
    root: str = ""           # the video's folder

    @property
    def u(self):
        """Progress through the cue, 0..1, remapped by params['span'] = [a, b] if given."""
        u = self.t / self.dur if self.dur > 0 else 1.0
        span = self.params.get("span")
        if span:
            u = span[0] + (span[1] - span[0]) * u
        return clamp01(u)

    def frame(self, bg=BG):
        return Frame(self.k, bg)


def v_title(ctx: Ctx):
    p = ctx.params
    fr = ctx.frame()
    a = ease(ctx.t / 0.8)
    # soft vignette band
    fr.rect(0, 380, 1920, 330, PANEL, alpha=0.55)
    fr.rect(120, 470, 10, 150, ACCENT, alpha=a)
    fr.text(p.get("title", ""), (160, 455), p.get("size", 78), INK, "semibold", alpha=a, max_width=1640)
    if p.get("subtitle"):
        fr.text(p["subtitle"], (162, 560), 40, MUTED, "regular", alpha=a, max_width=1600)
    if p.get("footer"):
        fr.text(p["footer"], (160, 980), 28, FAINT, alpha=a)
    return fr


def v_text(ctx: Ctx):
    """A heading and bullet points, revealed one by one across the cue (or all at once)."""
    p = ctx.params
    fr = ctx.frame()
    fr.text(p.get("heading", ""), (140, 120), 64, INK, "semibold")
    fr.rect(140, 205, 120, 6, ACCENT)
    items = p.get("bullets", [])
    reveal = p.get("reveal", True)
    y = 270
    n = max(1, len(items))
    for i, b in enumerate(items):
        if reveal:
            t0 = 0.4 + (ctx.dur * 0.75) * i / n
            a = ease((ctx.t - t0) / 0.5)
        else:
            a = 1.0
        if a <= 0:
            break
        fr.circle((158, y + 26), 8, ACCENT, alpha=a)
        fr.text(b, (190, y), 42, INK, max_width=1580, alpha=a)
        _, h = fr.text_size(b, 42, max_width=1580)
        y += h + 30
    return fr


def v_image(ctx: Ctx):
    p = ctx.params
    fr = ctx.frame(bg=(0, 0, 0))
    path = os.path.join(ctx.root, p["file"])
    if "img" not in ctx.state:
        im = cv2.imread(path)
        ctx.state["img"] = None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if ctx.state["img"] is not None:
        fr.image(ctx.state["img"], 0, 0, 1920, 1080, mode=p.get("fit", "contain"))
    return fr


def v_blank(ctx: Ctx):
    return ctx.frame()


GENERIC = {"title": v_title, "text": v_text, "image": v_image, "blank": v_blank}
