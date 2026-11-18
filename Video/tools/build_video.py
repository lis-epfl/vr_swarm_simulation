"""Build a video from its script.

    python Video/tools/build_video.py Video/participant_instructions            # full 1080p
    python Video/tools/build_video.py Video/participant_instructions --preview  # half size, 15 fps
    python Video/tools/build_video.py Video/participant_instructions --stills   # one PNG per cue
    python Video/tools/build_video.py Video/participant_instructions --only ctl_yaw,ctl_pitch

Reads <video>/script.yaml, writes <video>/build/<name>.mp4, <name>.srt and shots.md (the Unity
recordings the script asks for, and which are still missing). Run it with the `stitching`
miniconda env (it needs numpy, opencv, Pillow and PyYAML) and ffmpeg on PATH.

Per cue, in order of precedence:
  * narration: <video>/narration/<cue id>.wav (or .mp3/.m4a) if present, else Windows TTS of `say`;
  * picture: `footage.file` from <video>/footage/ if present, else the generated `visual`
    with a PLACEHOLDER banner naming the shot to record.
See Video/README.md for every field.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vidkit as vk  # noqa: E402

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def load_visuals(video_dir):
    visuals, overlays = dict(vk.GENERIC), dict(GENERIC_OVERLAYS)
    path = os.path.join(video_dir, "visuals.py")
    if os.path.exists(path):
        sys.path.insert(0, video_dir)
        spec = importlib.util.spec_from_file_location("video_visuals", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        visuals.update(getattr(mod, "VISUALS", {}))
        overlays.update(getattr(mod, "OVERLAYS", {}))
    return visuals, overlays


# ---------------------------------------------------------------------------------------------
# Overlays every video can use
# ---------------------------------------------------------------------------------------------
def o_label(fr: vk.Frame, ov, t, a):
    """Text, on a dark box unless box: false (it is usually over footage)."""
    x, y = ov.get("pos", (960, 120))
    size, weight, anchor = ov.get("size", 44), ov.get("weight", "semibold"), ov.get("anchor", "ma")
    mw = ov.get("max_width", 1700)
    if ov.get("box", True):
        w, h = fr.text_size(ov["text"], size, weight, max_width=mw)
        x0 = x - w / 2 if anchor[0] == "m" else (x - w if anchor[0] == "r" else x)
        fr.rect(x0 - 16, y - 8, w + 32, h + 12, (8, 10, 14), alpha=0.65 * a, radius=10)
    fr.text(ov["text"], (x, y), size, tuple(ov.get("color", vk.INK)), weight, anchor=anchor, alpha=a, max_width=mw)


def o_callout(fr: vk.Frame, ov, t, a):
    """A pill with text, gently pulsing, e.g. 'Press C2'."""
    x, y = ov.get("pos", (960, 160))
    size = ov.get("size", 48)
    col = tuple(ov.get("color", vk.ACCENT))
    w, h = fr.text_size(ov["text"], size, "semibold")
    pulse = 1 + 0.04 * np.sin(t * 6)
    pw, ph = (w + 70) * pulse, (h + 30) * pulse
    fr.rect(x - pw / 2, y - ph / 2, pw, ph, col, alpha=0.95 * a, radius=ph / 2)
    fr.text(ov["text"], (x, y - h / 2 - 2), size, (20, 20, 24), "semibold", anchor="ma", alpha=a)


GENERIC_OVERLAYS = {"label": o_label, "callout": o_callout}


# ---------------------------------------------------------------------------------------------
# Planning: narration, durations, captions
# ---------------------------------------------------------------------------------------------
def plan(video_dir, settings, cues, build_dir, only=None):
    lead_d, tail_d = settings.get("lead", 0.5), settings.get("tail", 0.8)
    tts_dir = os.path.join(build_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    narr_dir = os.path.join(video_dir, "narration")

    # Narration engine: ElevenLabs when asked for and a key is available, else Windows SAPI.
    engine = settings.get("tts", "sapi")
    key = vk.eleven_key() if engine == "elevenlabs" else None
    if engine == "elevenlabs" and not key:
        print("[tts] no ElevenLabs key (set ELEVENLABS_API_KEY or ~/.elevenlabs_api_key): using Windows TTS")
        engine = "sapi"

    def job_for(c, out_dir, eng):
        text = c["say"].strip()
        if eng == "elevenlabs":
            ev = settings.get("elevenlabs", {})
            vid, model = c.get("voice_id", ev.get("voice_id")), ev.get("model", "eleven_multilingual_v2")
            vs = ev.get("settings", {})
            h = vk.tts_key(text, f"11|{vid}|{model}|{json.dumps(vs, sort_keys=True)}", 0)
            return {"text": text, "voice_id": vid, "model": model, "settings": vs,
                    "out": os.path.join(out_dir, h + ".wav")}
        v, r = c.get("voice", settings.get("voice", "Microsoft Hazel Desktop")), c.get("rate", settings.get("rate", 0))
        return {"text": text, "voice": v, "rate": r, "out": os.path.join(out_dir, vk.tts_key(text, v, r) + ".wav")}

    planned = []
    for c in cues:
        if c.get("skip"):
            continue
        if only and c["id"] not in only:
            continue
        p = dict(c)
        own = None
        for ext in AUDIO_EXTS:
            f = os.path.join(narr_dir, c["id"] + ext)
            if os.path.exists(f):
                own = f
                break
        p["_own"] = own
        if own:
            p["_wav"] = os.path.join(tts_dir, f"own_{c['id']}.wav")
            vk.to_pcm(own, p["_wav"])
        elif c.get("say"):
            p["_job"] = job_for(c, tts_dir, engine)
            p["_wav"] = p["_job"]["out"]
        else:
            p["_wav"] = None
        planned.append(p)

    jobs = [p["_job"] for p in planned if p.get("_job")]
    new = [j for j in jobs if not os.path.exists(j["out"])]
    if new:
        print(f"[tts] {len(new)} new narration clips ({engine})")
        if engine == "elevenlabs":
            try:
                vk.eleven_synthesize(new, tts_dir, key)
            except RuntimeError as e:
                print(f"[tts] {e}")
                print("[tts] falling back to Windows TTS for this build")
                for p in planned:
                    if p.get("_job") and not os.path.exists(p["_job"]["out"]):
                        p["_job"] = job_for(p, tts_dir, "sapi")
                        p["_wav"] = p["_job"]["out"]
                vk.synthesize([p["_job"] for p in planned if p.get("_job") and "voice" in p["_job"]], tts_dir)
        else:
            vk.synthesize(new, tts_dir)

    for p in planned:
        p["_audio"] = vk.read_wav(p["_wav"]) if p["_wav"] else np.zeros(0, np.float32)
        speech = len(p["_audio"]) / vk.SAMPLE_RATE
        lead = p.get("lead", lead_d)
        tail = p.get("tail", tail_d)
        p["_lead"], p["_speech"] = lead, speech
        need = lead + speech + tail if speech > 0 else 0.0
        fo = p.get("footage")
        p["_footage_path"] = None
        if fo and fo.get("file"):
            fp = os.path.join(video_dir, "footage", fo["file"])
            if os.path.exists(fp):
                p["_footage_path"] = fp
        d = p.get("duration")
        if d == "clip":
            if p["_footage_path"]:
                clip = (vk.probe_duration(p["_footage_path"]) - fo.get("start", 0)) / fo.get("speed", 1.0)
                if fo.get("end"):
                    clip = (fo["end"] - fo.get("start", 0)) / fo.get("speed", 1.0)
            else:
                clip = p.get("fallback_duration", 20.0)
            dur = max(clip, need)
        else:
            dur = max(float(d or 0), need, 1.0)
        p["_dur"] = dur
        p["_captions"] = vk.time_chunks(vk.caption_chunks(p.get("caption", p.get("say", "")) or ""),
                                        lead, speech) if speech > 0 else []
    return planned


def shot_list(video_dir, planned_all, out_path):
    lines = ["# Unity recordings this video needs", "",
             "Generated by build_video.py from script.yaml. Drop each file into `footage/` under the",
             "name shown; the next build uses it in place of the schematic.", "",
             "| cue | file | status | what to record |", "|---|---|---|---|"]
    n_missing = 0
    for c in planned_all:
        for key in ("footage", "inset"):
            fo = c.get(key)
            if not fo or not fo.get("file"):
                continue
            ok = os.path.exists(os.path.join(video_dir, "footage", fo["file"]))
            n_missing += not ok
            lines.append(f"| {c['id']} | `{fo['file']}` | {'found' if ok else '**missing**'} | "
                         f"{' '.join(str(fo.get('shot', '')).split())} |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n_missing


# ---------------------------------------------------------------------------------------------
# Rendering one frame of one cue
# ---------------------------------------------------------------------------------------------
class CueRenderer:
    def __init__(self, video_dir, cue, visuals, overlays, k, settings):
        self.cue, self.k, self.settings = cue, k, settings
        self.visuals, self.overlays = visuals, overlays
        self.video_dir = video_dir
        self.state = {}
        self.inset_state = {}
        fo = cue.get("footage") or {}
        self.footage = vk.Footage(cue["_footage_path"]) if cue.get("_footage_path") else None
        ins = cue.get("inset")
        self.inset = None
        self.inset_cfg = ins
        if ins and ins.get("file"):
            fp = os.path.join(video_dir, "footage", ins["file"])
            if os.path.exists(fp):
                self.inset = vk.Footage(fp)
        self.placeholder = bool(fo.get("file")) and self.footage is None

    def _visual(self, name, params, t, state):
        fn = self.visuals.get(name)
        if fn is None:
            raise KeyError(f"cue {self.cue['id']}: no visual called '{name}'")
        ctx = vk.Ctx(t=t, dur=self.cue["_dur"], k=self.k, params=params or {}, cue=self.cue, state=state,
                     root=self.video_dir)
        return fn(ctx)

    def frame(self, t):
        c = self.cue
        fo = c.get("footage") or {}
        rect = fo.get("rect")
        if self.footage is not None and not rect:
            fr = vk.Frame(self.k, bg=(0, 0, 0))
            img = self._crop(self.footage.at(self._src(fo, t)), fo)
            if img is not None:
                fr.image(img, 0, 0, 1920, 1080, mode=fo.get("fit", "contain"))
        else:
            # The drawn picture, which a footage `rect` then sits in (a panel beside the controller, say).
            fr = self._visual(c.get("visual", "blank"), c.get("params"), t, self.state)
            if self.footage is not None:
                img = self._crop(self.footage.at(self._src(fo, t)), fo)
                if img is not None:
                    self._panel(fr, img, rect, fo.get("label"))

        if self.inset_cfg:
            self._draw_inset(fr, t)

        for ov in c.get("overlays", []) or []:
            at, until = ov.get("at", 0.0), ov.get("until", c["_dur"])
            if until < 0:
                until = c["_dur"] + until
            if not (at <= t <= until):
                continue
            a = min(vk.ease((t - at) / 0.3), vk.ease((until - t) / 0.3) if until < c["_dur"] - 0.05 else 1.0)
            fn = self.overlays.get(ov.get("kind", "label"))
            if fn is None:
                raise KeyError(f"cue {c['id']}: no overlay called '{ov.get('kind')}'")
            fn(fr, ov, t - at, a)

        if self.settings.get("show_section", True) and c.get("section"):
            w, _ = fr.text_size(c["section"].upper(), 24, "semibold")
            fr.rect(44, 36, w + 32, 44, (8, 10, 14), alpha=0.55, radius=8)
            fr.text(c["section"].upper(), (60, 44), 24, vk.MUTED, "semibold")

        if self.placeholder and self.settings.get("placeholder_banner", True):
            msg = f"PLACEHOLDER: schematic until Unity shot '{fo['file']}' is recorded"
            w, _ = fr.text_size(msg, 22, "semibold")
            fr.rect(1880 - w - 28, 30, w + 28, 40, (150, 40, 40), alpha=0.85, radius=8)
            fr.text(msg, (1866, 38), 22, vk.INK, "semibold", anchor="ra")

        if self.settings.get("captions", True):
            for (a0, a1, txt) in c["_captions"]:
                if a0 <= t < a1:
                    vk.draw_caption(fr, txt)
                    break
        return fr.finish()

    @staticmethod
    def _src(fo, t):
        """Source time in a recording: start + t x speed, held at `end` if the cue outlasts it."""
        src = fo.get("start", 0) + t * fo.get("speed", 1.0)
        return min(src, fo["end"]) if fo.get("end") is not None else src

    @staticmethod
    def _crop(img, fo):
        """`crop: [x, y, w, h]` as fractions of the recording: zoom into part of it (e.g. the one screen in the
        interface camera). Overlays placed from a recording's own coordinates do not know about it."""
        cr = fo.get("crop")
        if img is None or not cr:
            return img
        h, w = img.shape[:2]
        x0, y0 = int(round(cr[0] * w)), int(round(cr[1] * h))
        x1, y1 = int(round((cr[0] + cr[2]) * w)), int(round((cr[1] + cr[3]) * h))
        return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]

    @staticmethod
    def _panel(fr, img, rect, label=None):
        x, y, w, h = rect
        fr.rect(x - 4, y - 4, w + 8, h + 8, vk.INK, alpha=0.9, radius=6)
        fr.image(img, x, y, w, h, mode="cover")
        if label:
            fr.text(label, (x + w / 2, y - 52), 32, vk.INK, "semibold", anchor="ma")

    def _draw_inset(self, fr, t):
        ins = self.inset_cfg
        if ins.get("rect"):
            x, y, w, h = ins["rect"]
        else:
            scale = ins.get("scale", 0.32)
            w, h = 1920 * scale, 1080 * scale
            corner = ins.get("corner", "tr")
            m = 40
            x = 1920 - w - m if corner.endswith("r") else m
            y = m + 50 if corner.startswith("t") else 1080 - h - m - 110
        if self.inset is not None:
            img = self._crop(self.inset.at(self._src(ins, t)), ins)
            if img is None:
                return
        elif ins.get("fallback"):
            sub = self._visual(ins["fallback"], ins.get("params"), t, self.inset_state)
            img = sub.finish()
        else:
            return
        if ins.get("rect") and ins.get("label_above"):
            self._panel(fr, img, (x, y, w, h), ins["label"])
            return
        fr.rect(x - 4, y - 4, w + 8, h + 8, vk.INK, alpha=0.9, radius=6)
        fr.image(img, x, y, w, h, mode="cover")
        if ins.get("label"):
            fr.rect(x, y + h - 40, w, 40, (0, 0, 0), alpha=0.55)
            fr.text(ins["label"], (x + 14, y + h - 36), 24, vk.INK, "semibold")

    def close(self):
        if self.footage:
            self.footage.close()
        if self.inset:
            self.inset.close()


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir")
    ap.add_argument("--preview", action="store_true", help="half size at 15 fps, fast")
    ap.add_argument("--scale", type=float, default=None, help="output size / 1920x1080")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--only", default=None, help="comma-separated cue ids")
    ap.add_argument("--stills", action="store_true", help="write one PNG per cue (at --at) instead of a video")
    ap.add_argument("--at", default="0.65", help="for --stills: fraction(s) of each cue, comma-separated")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    video_dir = os.path.abspath(args.video_dir)
    with open(os.path.join(video_dir, "script.yaml"), encoding="utf-8") as f:
        script = yaml.safe_load(f)
    settings = script.get("video", {})
    cues = script["cues"]
    ids = [c["id"] for c in cues]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        sys.exit(f"duplicate cue ids: {sorted(dup)}")
    if args.no_captions:
        settings["captions"] = False
    name = settings.get("name", os.path.basename(video_dir))
    build_dir = os.path.join(video_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    if only and only - set(ids):
        sys.exit(f"unknown cue ids: {sorted(only - set(ids))}")

    k = args.scale or (0.5 if args.preview else settings.get("scale", 1.0))
    fps = args.fps or (15 if args.preview else settings.get("fps", 30))
    visuals, overlays = load_visuals(video_dir)

    planned = plan(video_dir, settings, cues, build_dir, only)
    n_missing = shot_list(video_dir, [c for c in cues if not c.get("skip")], os.path.join(build_dir, "shots.md"))
    total = sum(p["_dur"] for p in planned)
    print(f"[plan] {len(planned)} cues, {total / 60:.1f} min, {n_missing} Unity shot(s) still missing "
          f"(see build/shots.md)")

    if args.stills:
        sd = os.path.join(build_dir, "stills")
        os.makedirs(sd, exist_ok=True)
        fracs = [float(x) for x in args.at.split(",")]
        for p in planned:
            r = CueRenderer(video_dir, p, visuals, overlays, k, settings)
            for fr_ in fracs:
                img = r.frame(fr_ * p["_dur"])
                suffix = "" if len(fracs) == 1 else f"_{int(fr_ * 100):02d}"
                import cv2
                cv2.imwrite(os.path.join(sd, f"{p['id']}{suffix}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            r.close()
        print(f"[stills] {sd}")
        return

    # Audio: one track, each cue's narration at its lead-in.
    parts, srt, t0 = [], [], 0.0
    for p in planned:
        n = int(round(p["_dur"] * fps)) / fps
        p["_n"] = int(round(p["_dur"] * fps))
        buf = np.zeros(int(round(n * vk.SAMPLE_RATE)), np.float32)
        a = p["_audio"]
        s0 = int(p["_lead"] * vk.SAMPLE_RATE)
        m = min(len(a), len(buf) - s0)
        if m > 0:
            buf[s0:s0 + m] = a[:m] * settings.get("gain", 1.0)
        parts.append(buf)
        for (a0, a1, txt) in p["_captions"]:
            srt.append((t0 + a0, t0 + a1, txt))
        t0 += n
    audio = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    suffix = "" if not (args.preview or only) else ("_preview" if args.preview else "_part")
    out = args.out or os.path.join(build_dir, f"{name}{suffix}.mp4")
    wav = os.path.join(build_dir, f"{name}{suffix}_audio.wav")
    vk.write_wav(wav, audio)
    # Not <name>.srt: players load that beside <name>.mp4 and show it over the burned-in captions.
    srt_path = os.path.join(os.path.dirname(out), "subtitles_" + os.path.splitext(os.path.basename(out))[0] + ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (a0, a1, txt) in enumerate(srt, 1):
            f.write(f"{i}\n{vk.srt_time(a0)} --> {vk.srt_time(a1)}\n{txt}\n\n")

    W, H = int(round(vk.REF_W * k)), int(round(vk.REF_H * k))
    W -= W % 2
    H -= H % 2
    cmd = [vk.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium" if not args.preview else "veryfast",
           "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
           "-movflags", "+faststart", "-shortest", out]
    final_out = out
    out = os.path.splitext(out)[0] + ".partial.mp4"
    cmd[-1] = out
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    xf = int(round(settings.get("crossfade", 0.35) * fps))
    fade_in, fade_out = int(0.6 * fps), int(0.9 * fps)
    total_frames = sum(p["_n"] for p in planned)
    gi = 0
    prev_last = None
    t_start = time.time()
    try:
        for p in planned:
            r = CueRenderer(video_dir, p, visuals, overlays, k, settings)
            tc = time.time()
            img = None
            for i in range(p["_n"]):
                img = r.frame(i / fps)
                if img.shape[0] != H or img.shape[1] != W:
                    img = img[:H, :W]
                if prev_last is not None and i < xf and p.get("transition", "fade") == "fade":
                    a = (i + 1) / (xf + 1)
                    img = (img.astype(np.float32) * a + prev_last.astype(np.float32) * (1 - a)).astype(np.uint8)
                if gi < fade_in:
                    img = (img.astype(np.float32) * (gi / fade_in)).astype(np.uint8)
                if gi >= total_frames - fade_out:
                    img = (img.astype(np.float32) * ((total_frames - 1 - gi) / fade_out)).astype(np.uint8)
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
                gi += 1
            prev_last = img
            r.close()
            el = time.time() - tc
            print(f"  {p['id']:<22} {p['_dur']:5.1f} s  {'footage' if r.footage else ('PLACEHOLDER' if r.placeholder else 'drawn'):<11}"
                  f" {p['_n'] / max(el, 1e-6):5.1f} fps")
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode == 0:
        try:
            os.replace(out, final_out)
            out = final_out
        except PermissionError:
            # The previous video is open in a player: keep this one beside it under a time-stamped name.
            stamped = os.path.splitext(final_out)[0] + time.strftime("_%Y%m%d_%H%M") + ".mp4"
            os.replace(out, stamped)
            print(f"[note] {os.path.basename(final_out)} is open elsewhere; wrote {os.path.basename(stamped)}")
            out = stamped
    print(f"[done] {out}  ({total / 60:.1f} min, rendered in {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
