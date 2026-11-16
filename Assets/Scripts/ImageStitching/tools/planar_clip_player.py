"""
Watch PLANAR stitch the recorded real-drone clips live, in Unity, from a small control window.

What it drives
--------------
For the selected clip it runs two processes, exactly the ones a live flight would have:

- DJI_Swarm's ``AOS server/clip_replay.py --clip <label> --loop``, which writes the clip into
  ``DroneFeedSharedMemory`` as if the drones were flying (same blocks, poses, capture skew and
  rate, plus the clip's standoff in the trailer);
- ``StitcherThreading.py``, which Unity's DJIScene feeds through ``ImageSharing`` -> the block map,
  and whose panorama Unity draws on the curved screen.

Every clip change starts a **fresh stitcher**, and that is load-bearing, not tidiness: the learned
aligner (``planar_align`` / ``planar_track``) carries its lock between solves, so a stitcher that
saw the previous clip keeps rejecting the new one's solutions as implausible and holds the old
alignment. Both processes are stopped with the window, and any left over from an earlier run are
stopped on start, because two replays are two producers on one single-producer map.

The window shows the clip playing, a countdown (auto-advance, default 25 s a clip), the
stitcher's state, its latest ``[PLANAR]`` views and align lines, and Unity's frame rate. Previous /
Next / double-click switch clips; *Restart stitcher* is for after leaving and re-entering Play,
since the stitcher exits on purpose 5 s after Unity's heartbeat stops.

Usage
-----
Open ``Assets/Scenes/DJIScene.unity`` and press Play (it already has PLANAR, Multiband, both
estimator switches and FormationRelative), then, in the ``stitching`` env::

    python tools/planar_clip_player.py            # from the first clip
    python tools/planar_clip_player.py --start 4  # from clip 4
    python tools/planar_clip_player.py --seconds 60 --dji-swarm D:/DJI_Swarm

The DJI_Swarm repo is expected beside this one (``../DJI_Swarm``) unless ``--dji-swarm`` or
``DJI_SWARM_DIR`` says otherwise. Clips flagged unusable by ``clip_replay.py --list`` (a starved
video link) are left out of ``CLIPS``, as in ``planar_stream_bench.CLIPS``.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
STITCH_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(STITCH_DIR, "..", "..", ".."))
REPLAY_DIR = None       # set in main(): <DJI_Swarm>/AOS server

CLIPS = [
    ("MED_facade_stationary_1", "facade, 3 drones holding"),
    ("MED_facade_stationary_2", "facade, 3 drones holding"),
    ("MED_facade_5drone_short", "facade, 5 drones (last ~4 s past the roofline)"),
    ("grass_nadir_long", "down, 3 drones over grass"),
    ("grass_nadir_short_contracting", "down, 3 drones closing in"),
    ("nadir_6drone_30m", "down, 6 drones at 30 m"),
    ("nadir_6drone_40m", "down, 6 drones at 40 m"),
]

ENV = dict(os.environ, PYTHONUNBUFFERED="1")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def kill_strays():
    """Stop replay/stitcher processes left by an earlier run (two producers would fight)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'clip_replay|StitcherThreading' } | "
             "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
            capture_output=True, timeout=20, creationflags=NO_WINDOW)
        return out.returncode == 0
    except Exception:
        return False


class Proc:
    def __init__(self, name, args, cwd, sink):
        self.name = name
        self.p = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env=ENV, text=True, encoding="utf-8", errors="replace",
                                  creationflags=NO_WINDOW)
        threading.Thread(target=self._pump, args=(sink,), daemon=True).start()

    def _pump(self, sink):
        for line in self.p.stdout:
            sink.put((self.name, self.p.pid, line.rstrip()))
        sink.put((self.name, self.p.pid, None))

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.alive():
            self.p.terminate()
            try:
                self.p.wait(8)
            except subprocess.TimeoutExpired:
                self.p.kill()


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.replay = None
        self.stitch = None
        self.index = None
        self.started = 0.0
        root.title("PLANAR clip player")
        root.geometry("760x520")
        root.minsize(620, 440)

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        self.now = tk.StringVar(value="Nothing playing")
        self.desc = tk.StringVar(value="Unity must be in Play in DJIScene")
        ttk.Label(top, textvariable=self.now, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(top, textvariable=self.desc, font=("Segoe UI", 10)).pack(anchor="w")

        bar = ttk.Frame(root, padding=(10, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="◀ Previous", command=self.prev).pack(side="left")
        ttk.Button(bar, text="Next ▶", command=self.next).pack(side="left", padx=6)
        ttk.Button(bar, text="Restart clip", command=self.restart).pack(side="left")
        ttk.Button(bar, text="Restart stitcher", command=self.restart_stitcher).pack(side="left", padx=6)
        ttk.Button(bar, text="Stop", command=self.stop_all).pack(side="left")
        self.auto = tk.BooleanVar(value=True)
        self.secs = tk.IntVar(value=SECONDS)
        ttk.Spinbox(bar, from_=10, to=600, increment=5, width=5, textvariable=self.secs).pack(side="right")
        ttk.Label(bar, text="s each").pack(side="right", padx=(0, 4))
        ttk.Checkbutton(bar, text="Auto-advance every", variable=self.auto).pack(side="right")

        mid = ttk.Frame(root, padding=10)
        mid.pack(fill="both", expand=True)
        self.lst = tk.Listbox(mid, height=len(CLIPS), font=("Segoe UI", 10), activestyle="none",
                              exportselection=False)
        for name, d in CLIPS:
            self.lst.insert("end", f"  {name}   —   {d}")
        self.lst.pack(fill="x")
        self.lst.bind("<Double-Button-1>", lambda e: self.play(self.lst.curselection()[0])
                      if self.lst.curselection() else None)
        ttk.Label(mid, text="Double-click a clip to play it.", foreground="#666").pack(anchor="w", pady=(2, 8))

        stat = ttk.LabelFrame(mid, text="Stitcher", padding=8)
        stat.pack(fill="both", expand=True)
        self.state = tk.StringVar(value="—")
        self.views = tk.StringVar(value="")
        self.align = tk.StringVar(value="")
        self.fps = tk.StringVar(value="")
        self.state_lbl = ttk.Label(stat, textvariable=self.state, font=("Segoe UI", 11, "bold"))
        self.state_lbl.pack(anchor="w")
        for v in (self.views, self.align, self.fps):
            ttk.Label(stat, textvariable=v, wraplength=700, justify="left").pack(anchor="w", pady=2)

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.tick)

    # ---- process control -------------------------------------------------------------
    def _start_stitcher(self):
        if self.stitch:
            self.stitch.stop()
        self.state.set("stitcher loading…")
        self.state_lbl.configure(foreground="#a60")
        self.views.set(""); self.align.set("")
        self.stitch = Proc("stitch", [PY, "-u", "StitcherThreading.py"], STITCH_DIR, self.q)

    def play(self, i):
        i %= len(CLIPS)
        self.stop_all(quiet=True)
        self.index = i
        name, d = CLIPS[i]
        self.now.set(f"{i + 1}/{len(CLIPS)}  {name}")
        self.desc.set(d)
        self.lst.selection_clear(0, "end")
        self.lst.selection_set(i)
        self.lst.see(i)
        self.replay = Proc("replay", [PY, "-u", "clip_replay.py", "--clip", name, "--loop",
                                      "--no-unity-watch"], REPLAY_DIR, self.q)
        self.root.after(800, self._start_stitcher)
        self.started = time.time()

    def next(self):
        self.play(0 if self.index is None else self.index + 1)

    def prev(self):
        self.play(0 if self.index is None else self.index - 1)

    def restart(self):
        if self.index is not None:
            self.play(self.index)

    def restart_stitcher(self):
        if self.index is not None:
            self._start_stitcher()
            self.started = time.time()

    def stop_all(self, quiet=False):
        for p in (self.stitch, self.replay):
            if p:
                p.stop()
        self.stitch = self.replay = None
        if not quiet:
            self.index = None
            self.now.set("Stopped")
            self.state.set("—")
            self.state_lbl.configure(foreground="")

    def close(self):
        self.stop_all(quiet=True)
        self.root.destroy()

    # ---- UI loop ------------------------------------------------------------------------
    def tick(self):
        while True:
            try:
                who, pid, line = self.q.get_nowait()
            except queue.Empty:
                break
            cur = self.stitch if who == "stitch" else self.replay
            if cur is None or cur.p.pid != pid:
                continue        # output of a process already replaced
            if who == "stitch":
                self._stitch_line(line)
            elif line is None:
                self.desc.set(CLIPS[self.index][1] + "   (replay exited!)")

        if self.index is not None:
            el = time.time() - self.started
            if self.auto.get():
                left = max(0, self.secs.get() - el)
                self.now.set(f"{self.index + 1}/{len(CLIPS)}  {CLIPS[self.index][0]}   ({left:.0f} s)")
                if left <= 0:
                    self.next()
            else:
                self.now.set(f"{self.index + 1}/{len(CLIPS)}  {CLIPS[self.index][0]}   ({el:.0f} s)")
        self.root.after(250, self.tick)

    def _stitch_line(self, line):
        if line is None:
            self.state.set("stitcher stopped — is Unity in Play? Then press Restart stitcher")
            self.state_lbl.configure(foreground="#c00")
            return
        s = line.split("] ", 1)[1] if line.startswith("[up ") and "] " in line else line
        if "FATAL" in line:
            self.align.set(line)
        elif "Waiting for Unity metadata" in line:
            self.state.set("waiting for Unity (press Play in DJIScene)")
        elif "[PLANAR] align" in s:
            self.align.set(s.replace("[PLANAR] ", ""))
            self.state.set("aligned / tracking" if "holding" not in s else "holding last alignment")
            self.state_lbl.configure(foreground="#070" if "holding" not in s else "#a60")
        elif s.startswith("[PLANAR]") and "views" in s:
            self.views.set(s.replace("[PLANAR] ", ""))
            if self.state.get().startswith("stitcher loading"):
                self.state.set("stitching (aligner starting)")
        elif s.startswith("Unity ") and "fps" in s:
            self.fps.set(s)


SECONDS = 25


def main():
    global REPLAY_DIR, SECONDS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1, help="clip number to start on (1-based)")
    ap.add_argument("--seconds", type=int, default=SECONDS, help="auto-advance period (default 25)")
    ap.add_argument("--dji-swarm", default=os.environ.get("DJI_SWARM_DIR",
                                                          os.path.join(REPO_ROOT, "..", "DJI_Swarm")),
                    help="DJI_Swarm checkout (default ../DJI_Swarm or $DJI_SWARM_DIR)")
    args = ap.parse_args()
    REPLAY_DIR = os.path.abspath(os.path.join(args.dji_swarm, "AOS server"))
    if not os.path.isfile(os.path.join(REPLAY_DIR, "clip_replay.py")):
        sys.exit(f"clip_replay.py not found in {REPLAY_DIR}; pass --dji-swarm")
    SECONDS = args.seconds

    kill_strays()
    root = tk.Tk()
    app = App(root)
    root.after(300, lambda: app.play(args.start - 1))
    root.mainloop()


if __name__ == "__main__":
    main()
