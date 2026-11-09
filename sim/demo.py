"""Scripted fridge -> bed run in MuJoCo, with real actuator limits.

uv run python sim/demo.py            # simulate, print a report, write out/demo.mp4
uv run python sim/demo.py --no-video # report only

This is open-loop: joint targets come from the same IK used for the renders.
It answers "is the motion physically consistent with the hardware limits?",
not "can it do this autonomously" (no perception here).
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cad"))
import params as P  # noqa: E402
import scene as SC  # noqa: E402

SIM = ROOT / "sim"
OUT = ROOT / "out"
JOINTS = ["base_x", "base_y", "base_yaw", "lift", "drawer", "j1", "j2", "j3", "j4", "grip_l"]   # the scripted arm (a)
SHELF = 0.950
FRIDGE_X = 0.72 - P.AXLE_SHIFT / 1000   # fridge front, world x (robot stops at x = 0)
BOX_FACE = FRIDGE_X + 0.058         # box front face (knob side), world x
SHOULDER_Y = P.ARM_Y[0] / 1000      # arm a (right) shoulder
# everything the scripts don't drive: arm a's wrist roll stays at 0, arm b stays parked beside the tray
EXTRA = {"j5": 0.0, **{k: math.radians(v) for k, v in SC.PARK_B.items() if k.startswith("j")},
         "gripb": SC.PARK_B["gripb"] / 1000}
PAD_HALF = 0.030                      # finger pads are 60 mm long; grip the 18 mm knob with their last ~15 mm
GRASP_X = BOX_FACE - PAD_HALF - 0.003  # pad centre target


# ------------------------------------------------------------------ kinematics (robot frame, metres, degrees)
def ik(wx, wy, heading=0.0):
    x = wx - P.SHOULDER_X / 1000
    y = wy - SHOULDER_Y
    l1, l2 = P.L1 / 1000, P.L2 / 1000
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = max(-1.0, min(1.0, c2))
    q2 = math.acos(c2)                               # elbow-left branch, j2 > 0
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    j1, j2 = math.degrees(q1), math.degrees(q2)
    return j1, j2, heading - j1 - j2


def arm(wx, wy, heading=0.0, pitch=0.0, grip=0.004):
    j1, j2, j3 = ik(wx, wy, heading)
    return dict(j1=j1, j2=j2, j3=j3, j4=pitch, grip_l=grip)


TIP = P.PAD_X / 1000                        # pad centre, measured from the pitch axis
PARK = dict(j1=SC.PARK_A["j1"], j2=SC.PARK_A["j2"], j3=SC.PARK_A["j3"], j4=90.0, grip_l=0.004)   # hand down beside the tray
FOLD = arm(0.30, SHOULDER_Y, heading=0.0, pitch=0.0, grip=0.004)                                  # hand forward, folded


def init_extra(m, d):
    """Set the joints and actuators the scripts don't drive (arm a roll, arm b) to their parked values."""
    for name, v in EXTRA.items():
        a = m.actuator(name)
        jn = m.joint(m.actuator_trnid[a.id][0]).name
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v
        d.ctrl[a.id] = v
    d.qpos[m.jnt_qposadr[m.joint("grip_rb").id]] = EXTRA["gripb"]


def set_ctrl(d, q):
    d.ctrl[:len(JOINTS)] = q


def minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# Real speed limits (deg/s, m/s), enforced on every scripted segment (v0.2, 26 Oct). Until then the scripts
# could ask the lift for 16 cm/s and a 26 cm drop in 0.7 s. A min-jerk segment peaks at 1.875x its mean speed.
VMAX = dict(
    lift=0.06,       # the quiet speed the design runs at (stepper max ~0.13 m/s)
    base_x=0.25, base_y=0.25, base_yaw=40.0,
    drawer=0.03,     # STS3215 in wheel mode on a GT2 20T: 45 rpm no-load = 30 mm/s
    j1=60.0,         # STS3215 270 deg/s no-load, through 3:1, ~2/3 under load
    j2=90.0,         # through 2:1
    j3=180.0, j4=150.0, j5=180.0,
    grip_l=0.15,
)
MINJERK_PEAK = 1.875


def min_duration(frm, to, fn=None):
    """Shortest min-jerk duration that keeps every joint under VMAX (samples the path if it is a function)."""
    if fn is None:
        pts = [frm, to]
    else:
        pts = [frm] + [fn(k / 12) for k in range(1, 13)]
    need = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        need += max((abs(b[k] - a[k]) / v for k, v in VMAX.items() if k in a and k in b), default=0.0)
    return MINJERK_PEAK * need


class Script:
    """Builds a timeline of joint targets. Angles in degrees, lengths in metres.
    Durations are stretched where a segment would need more than the real joint speeds (VMAX)."""

    realistic = True
    stretched = 0.0          # total seconds added by the speed limits, for reports

    def __init__(self, start):
        self.state = dict(start)
        self.segments = []      # (t0, t1, from, to, fn)
        self.t = 0.0
        self.marks = []
        self.stretched = 0.0

    def _dur(self, dur, frm, to, fn=None):
        if not self.realistic:
            return dur
        need = min_duration(frm, to, fn)
        if need > dur:
            self.stretched += need - dur
            return need
        return dur

    def move(self, dur, mark=None, **targets):
        frm = dict(self.state)
        to = {**self.state, **targets}
        dur = self._dur(dur, frm, to)
        self.segments.append((self.t, self.t + dur, frm, to, None))
        if mark:
            self.marks.append((self.t, mark))
        self.t += dur
        self.state = to

    def cart(self, dur, w0, w1, heading=0.0, mark=None, **extra):
        """Straight-line wrist motion (robot frame) with fixed heading."""
        frm = dict(self.state)
        def fn(s, frm=frm):
            wx = w0[0] + (w1[0] - w0[0]) * s
            wy = w0[1] + (w1[1] - w0[1]) * s
            return {**frm, **extra, **arm(wx, wy, heading, extra.get("j4", frm["j4"]), extra.get("grip_l", frm["grip_l"]))}
        to = fn(1.0)
        dur = self._dur(dur, frm, to, fn)
        self.segments.append((self.t, self.t + dur, frm, to, fn))
        if mark:
            self.marks.append((self.t, mark))
        self.t += dur
        self.state = to

    def wait(self, dur, mark=None):
        self.move(dur, mark=mark)

    def at(self, t):
        for t0, t1, frm, to, fn in self.segments:
            if t0 <= t < t1:
                s = minjerk((t - t0) / (t1 - t0))
                if fn is not None:
                    return fn(s)
                return {k: frm[k] + (to[k] - frm[k]) * s for k in frm}
        return dict(self.state)


def build_script():
    start = dict(base_x=-0.9, base_y=0.0, base_yaw=0.0, lift=0.30, drawer=0.0, **PARK)
    S = Script(start)
    S.wait(0.5, "start: tray low, arms parked")
    S.move(4.0, "lift the tray to shelf height", lift=SHELF - 0.002)
    S.move(2.0, "hand swings up beside the tray", j4=0.0)
    S.move(3.0, "arm unfolds (away from the fridge: the CAP hand is long)", **FOLD)
    S.move(4.0, "drive to the fridge (door already open)", base_x=0.0)
    S.move(2.5, "drawer out 240 mm", drawer=P.DRAWER_STROKE / 1000)
    pre = (GRASP_X - 0.03 - TIP, 0.0)
    S.move(2.5, **arm(*pre, grip=0.020))
    grasp = (GRASP_X - TIP, 0.0)
    S.cart(1.5, pre, grasp, grip_l=0.020, mark="reach the knob")
    S.move(1.0, "close gripper", grip_l=-0.006)
    pull_end = (GRASP_X - 0.222 - TIP, 0.0)
    S.cart(5.0, grasp, pull_end, grip_l=-0.006, mark="pull the box onto the drawer")
    S.move(0.8, "release", grip_l=0.020)
    # the CAP hand is long: the wrist now sits beside its own shoulder, so the hand leaves to the left
    # (away from the shoulder), fingers closed and tilted up 5 deg, clear of the box's way back
    back = (pull_end[0] - 0.05, 0.0)
    S.cart(1.0, pull_end, back, grip_l=0.020)
    S.move(0.6, grip_l=0.004, j4=-5.0)
    side = (back[0], 0.10)
    S.cart(2.0, back, side, grip_l=0.004, j4=-5.0, mark="hand out to the left")
    S.move(3.0, "drawer in, box rides along", drawer=0.0)
    front = (0.42, 0.10)
    S.cart(1.5, side, front, grip_l=0.004, j4=-5.0)
    S.cart(2.5, front, (0.42, -0.25), grip_l=0.004, j4=-5.0, mark="hand crosses beyond the box to the right")
    S.move(3.0, "back away", base_x=-0.6)
    S.move(2.5, "arm folds back beside the carriage", **{**PARK, "j4": 0.0})
    S.move(1.5, "hand down", j4=90.0)
    S.move(3.0, "tray down to transport/bed height", lift=0.64)
    S.move(3.0, "turn", base_yaw=90.0)
    S.move(4.5, "drive to the bed", base_y=1.0 + P.AXLE_SHIFT / 1000)
    S.move(3.0, "drawer out over the mattress", drawer=P.DRAWER_STROKE / 1000)
    S.wait(2.5, "done: meal within reach")
    return S


def to_ctrl(state):
    v = []
    for j in JOINTS:
        x = state[j]
        if j in ("base_yaw", "j1", "j2", "j3", "j4"):
            x = math.radians(x)
        v.append(x)
    return np.array(v)


def run(video=True, fps=30, speed=2.0, width=960, height=600, box_mass=None, box_mu=None):
    m = mujoco.MjModel.from_xml_path(str(SIM / "tischlein.xml"))
    if box_mass is not None:
        bid = m.body("meal_box").id
        scale = box_mass / m.body_mass[bid]
        m.body_mass[bid] = box_mass
        m.body_inertia[bid] *= scale
    if box_mu is not None:
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == m.body("meal_box").id and m.geom_contype[g]:
                m.geom_friction[g][0] = box_mu
        for g in range(m.ngeom):
            if m.geom(g).name.startswith("shelf"):
                m.geom_friction[g][0] = box_mu
    d = mujoco.MjData(m)
    S = build_script()
    jid = {j: m.joint(j).id for j in JOINTS}
    q0 = to_ctrl(S.at(0.0))
    for j, val in zip(JOINTS, q0):
        d.qpos[m.jnt_qposadr[jid[j]]] = val
    d.qpos[m.jnt_qposadr[m.joint("grip_r").id]] = q0[-1]
    set_ctrl(d, q0)
    init_extra(m, d)
    mujoco.mj_forward(m, d)

    act_names = [m.actuator(i).name for i in range(m.nu)]
    frc_lim = np.array([m.actuator_forcerange[i][1] for i in range(m.nu)])
    peak = np.zeros(m.nu)
    peak_t = np.zeros(m.nu)
    sat_time = np.zeros(m.nu)

    box_id = m.body("meal_box").id
    drawer_id = m.body("drawer").id

    renderer = mujoco.Renderer(m, height, width) if video else None
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    frames_dir = OUT / "demo_frames"
    if video:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for p in frames_dir.glob("*.png"):
            p.unlink()
    frame_dt = speed / fps
    next_frame = 0.0
    nframe = 0
    dt = m.opt.timestep
    T = S.t
    t_away = [t for t, mk in S.marks if mk.startswith("back away")][0]
    t_pull, t_in = [t for t, mk in S.marks if mk.startswith("pull the box")][0], [t for t, mk in S.marks if mk.startswith("drawer in")][0]
    log = []
    while d.time < T:
        set_ctrl(d, to_ctrl(S.at(d.time)))
        mujoco.mj_step(m, d)
        f = np.abs(d.actuator_force)
        upd = f > peak
        peak[upd] = f[upd]
        peak_t[upd] = d.time
        sat_time += (f >= 0.98 * frc_lim) * dt
        if video and d.time >= next_frame:
            next_frame += frame_dt
            bx, by = d.qpos[m.jnt_qposadr[jid["base_x"]]], d.qpos[m.jnt_qposadr[jid["base_y"]]]
            s = min(1.0, max(0.0, (d.time - t_away) / 6.0))
            cam.lookat[:] = (0.30 * (1 - s) + (bx + 0.05) * s, -0.05 * (1 - s) + (by + 0.45) * s, 0.80 - 0.15 * s)
            cam.distance = 2.5 + 0.4 * s
            cam.azimuth = 32 - 22 * s
            cam.elevation = -22
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            from PIL import Image, ImageDraw
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            label = ""
            for tm, mk in S.marks:
                if tm <= d.time:
                    label = mk
            dr.rectangle((0, height - 34, width, height), fill=(20, 20, 24))
            dr.text((12, height - 26), f"t = {d.time:5.1f} s   ({speed:.0f}x)   {label}", fill=(240, 240, 240))
            im.save(frames_dir / f"f{nframe:05d}.png")
            nframe += 1
        if abs(d.time - t_pull) < dt / 2 or abs(d.time - t_in) < dt / 2 or abs(d.time - T + 0.01) < dt / 2:
            bp = d.xpos[box_id]
            dp = d.xpos[drawer_id]
            log.append((round(d.time, 1), bp.copy(), dp.copy()))

    # ---------------------------------------------------------------- report
    bad = [i for i in range(len(d.warning)) if d.warning[i].number]
    if bad or not np.all(np.isfinite(d.qpos)):
        print("SIM WARNINGS:", [(mujoco.mjtWarning(i).name, d.warning[i].number, d.warning[i].lastinfo) for i in bad])
    print("\nActuator peak load (fraction of the real limit):")
    for i, n in enumerate(act_names):
        print(f"  {n:9s} peak {peak[i]:7.2f} / {frc_lim[i]:7.2f}  = {100*peak[i]/frc_lim[i]:5.1f}%   at t={peak_t[i]:5.1f}s   saturated {sat_time[i]:.2f}s")
    # final box position relative to the drawer (tray) frame
    R = d.xmat[drawer_id].reshape(3, 3)
    rel = R.T @ (d.xpos[box_id] - d.xpos[drawer_id])
    on_tray_x = (P.TRAY_X0 / 1000 + 0.005 <= rel[0] - 0.0) and (rel[0] + 0.18 <= (P.TRAY_X0 + P.TRAY_DEPTH) / 1000 + 0.003)
    print(f"\nBox in drawer frame at the end: x={rel[0]*1000:.0f} mm (tray {P.TRAY_X0}..{P.TRAY_X0+P.TRAY_DEPTH}), "
          f"y={rel[1]*1000:.0f} mm, z={rel[2]*1000:.0f} mm above tray surface")
    print("Box fully on the tray:", on_tray_x and abs(rel[1]) < 0.08 and -0.01 < rel[2] < 0.01)
    print(f"Run time {T:.0f} s with real joint speeds (lift 6 cm/s); the speed limits added {S.stretched:.0f} s to the script")
    for t, bp, dp in log:
        print(f"  t={t:5.1f}s box world {np.round(bp*1000).astype(int)}  drawer origin {np.round(dp*1000).astype(int)}")
    if video and nframe:
        mp4 = OUT / "demo.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow", str(mp4)], check=True)
        print(f"\nvideo -> {mp4} ({mp4.stat().st_size/1e6:.1f} MB, {nframe} frames)")


STRESS = ((0.8, 0.35), (1.5, 0.5), (2.0, 0.35), (2.0, 0.5), (2.5, 0.35), (3.0, 0.35), (2.5, 0.6))

if __name__ == "__main__":
    if "--stress" in sys.argv:
        for mass, mu in STRESS:
            print(f"\n===== box {mass} kg, friction {mu} =====")
            run(video=False, box_mass=mass, box_mu=mu)
    else:
        run(video="--no-video" not in sys.argv)
