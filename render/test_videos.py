"""Short videos of the design's test cases, from the existing simulations -> out/videos/*.mp4

uv run python render/test_videos.py [transfer] [floor] [cap]

  transfer  sim/transfer_compare.py: a 1.5 kg bottle, 10 N squeeze per finger, tongue vs lift (same world)
  floor     sim/floor_pick.py: a credit card, scoop and sweep vs a two-finger grasp (free-floating hand and scoop)
  cap       CAP pick in EgoGym on our kinematics with grey images; first run
            third_party/EgoGym/.venv/bin/python experiments/cap_zero_shot/run_eval.py --task pick --robot capnear \
              --episodes 6 --workers 1 --anchor frozen --image gray --record --tag video_gray
The door video comes from `sim/fridge_door.py --video stumpyaw full always 35 0.6`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sim"), str(ROOT / "cad")]
OUT = ROOT / "out" / "videos"


def recorder(lookat, distance, azimuth, elevation, label, w=640, h=480, fps=25, see_through=()):
    """see_through: geom names drawn nearly transparent (rendering only; the trial's model copy, not its physics)."""
    st = dict(fps=fps, next=0.0, frames=[], r=None)

    def grab(m, d):
        if st["r"] is None:
            st["r"] = mujoco.Renderer(m, h, w)
            for nm in see_through:
                m.geom_rgba[m.geom(nm).id][3] = 0.12
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        st["r"].update_scene(d, camera=cam)
        im = Image.fromarray(st["r"].render())
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, h - 28, w, h), fill=(20, 20, 24))
        dr.text((10, h - 21), f"t={d.time:4.1f}s  {label}", fill=(240, 240, 240))
        return np.asarray(im)
    st["grab"] = grab
    return st


def write(frames, path, fps=25):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".frames")
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(tmp / f"f{i:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "slow", str(path)], check=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    print(f"{path.name}: {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB")


def side_by_side(a, b):
    n = max(len(a), len(b))
    a = a + [a[-1]] * (n - len(a))
    b = b + [b[-1]] * (n - len(b))
    return [np.concatenate([x, y], axis=1) for x, y in zip(a, b)]


def transfer():
    import transfer_compare as TC
    obj = next(k for k in TC.OBJECTS if k.startswith("1.5"))
    # first world in which lifting loses the bottle (the same world for both strategies)
    for i in range(30):
        r = TC.trial((obj, "lift", 10.0, False, i))
        if not r["ok"]:
            break
    clips = []
    for mode in ("tongue", "lift"):
        TC.VIDEO = recorder((TC.SHELF_EDGE - 0.06, TC.Y0, TC.SHELF + 0.06), 0.95, 100, -22,
                            f"{obj}, {mode}: " + ("slid onto the tongue" if mode == "tongue" else "lifted by the wrist"),
                            see_through=("fr_right", "fr_top", "fr_door"))
        r = TC.trial((obj, mode, 10.0, False, i))
        clips.append(TC.VIDEO["frames"])
        print(mode, r["why"])
        TC.VIDEO = None
    write(side_by_side(*clips), OUT / "transfer_tongue_vs_lift.mp4")


def floor():
    import floor_pick as FP
    obj = "credit card"
    clips = []
    for st in ("scoop", "grasp"):
        FP.VIDEO = recorder((0.0, 0.0, 0.02), 0.45, 135, -30, f"{obj} on tile: {st}")
        r = FP.trial((obj, "tile", st, 0))
        clips.append(FP.VIDEO["frames"])
        print(st, r)
        FP.VIDEO = None
    write(side_by_side(*clips), OUT / "floor_scoop_vs_grasp.mp4")


def cap():
    """EgoGym's CAP pick recordings (run_eval.py --record --image gray: worker 0, episodes 0-1): wrist view turned grey
    as the policy saw it, next to the outside view, 2x, labelled."""
    src = ROOT / "experiments/cap_zero_shot/videos"
    res = __import__("json").loads((ROOT / "experiments/cap_zero_shot/results/pick_capnear_video_gray.json").read_text())
    eps = {r["ep"]: r for r in res["episodes_detail"] if r["worker"] == 0 and (src / f"pick_capnear_w0_e{r['ep']}.mp4").exists()}
    frames = []
    for ep in sorted(eps, key=lambda e: not eps[e]["success"]):       # the success first
        tmp = OUT / f"_cap{ep}"
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src / f"pick_capnear_w0_e{ep}.mp4"),
                        str(tmp / "f%04d.png")], check=True)
        r = eps[ep]
        lab = f"CAP pick, grey wrist image (left): {r['object'].rsplit('_', 1)[0]} - {'picked' if r['success'] else 'missed'}"
        clip = []
        for f in sorted(tmp.glob("*.png")):
            im = Image.open(f).convert("RGB")
            ego = im.crop((0, 0, 224, 224)).convert("L").convert("RGB")
            im.paste(ego, (0, 0))
            im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
            canvas = Image.new("RGB", (im.width, im.height + 28), (20, 20, 24))
            canvas.paste(im, (0, 0))
            ImageDraw.Draw(canvas).text((10, im.height + 7), lab, fill=(240, 240, 240))
            clip.append(np.asarray(canvas))
            f.unlink()
        tmp.rmdir()
        frames += clip + [clip[-1]] * 6
    write(frames, OUT / "cap_pick_grey.mp4", fps=4)


if __name__ == "__main__":
    what = sys.argv[1:] or ["transfer", "floor", "cap"]
    if "transfer" in what:
        transfer()
    if "floor" in what:
        floor()
    if "cap" in what:
        cap()
