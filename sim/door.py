"""Can Tischlein open a fridge door? MuJoCo test with a magnetic-seal model.

uv run python sim/door.py            # sweep seal strength x floor traction, print a table
uv run python sim/door.py --video    # one run (35 N seal, mu 0.6) -> out/door.mp4

Door model (all assumptions, to be replaced by measurements):
  - 600 mm wide, 8 kg, hinge on the left (+y) front corner, opens outward to -x
  - seal: closing torque F_b * r_handle at 0 deg, fading to 0 when the handle gap reaches 8 mm
    (F_b = the force you would read on a luggage scale hooked to the handle)
  - self-closing 1.0 N*m below 85 deg; detent holds it open beyond 92 deg
Robot: base forces capped at floor traction (mu * load on the drive wheels),
arm/gripper capped at the real servo limits (same model as demo.py).

Strategy tested (no fridge modification):
  1. grip the vertical handle bar with the arm almost straight, so the pull runs through the bearings
  2. back the base off ~5 cm -> seal breaks or the wheels stall
  3. follow the handle's arc with the arm while the base slides ~40 cm toward the hinge side
  4. let go past the detent
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
import params as P  # noqa: E402
from demo import JOINTS, init_extra, ik, minjerk, set_ctrl, to_ctrl  # noqa: E402

SIM = ROOT / "sim"
OUT = ROOT / "out"
FX = 0.72 - P.AXLE_SHIFT / 1000   # fridge front, world x
HINGE = np.array([FX, 0.30])
H_LOCAL = np.array([-0.11, -0.53])     # handle bar axis in the door frame
R_H = float(np.linalg.norm(H_LOCAL))
BAR_Z = 1.175
TIP = P.PAD_X / 1000          # pitch axis -> finger pad centre
SHOULDER = P.SHOULDER_X / 1000
ROBOT_MASS = 13.2


def door_xml():
    return f"""
    <body name="door" pos="{FX} 0.30 0">
      <joint name="door" type="hinge" axis="0 0 1" range="-120 0" damping="0.6" frictionloss="0.25" armature="0.05"/>
      <inertial pos="-0.035 -0.30 1.35" mass="8" diaginertia="0.86 0.62 0.25"/>
      <geom type="box" pos="-0.038 -0.30 1.340" size="0.035 0.298 0.455" rgba="0.93 0.93 0.91 1"/>
      <geom type="box" pos="-0.0875 -0.53 1.06" size="0.0175 0.012 0.012" rgba="0.6 0.6 0.62 1"/>
      <geom type="box" pos="-0.0875 -0.53 1.29" size="0.0175 0.012 0.012" rgba="0.6 0.6 0.62 1"/>
      <geom name="bar" type="cylinder" pos="-0.11 -0.53 {BAR_Z}" size="0.015 0.125" rgba="0.6 0.6 0.62 1" friction="1.0 0.02 0.002"/>
    </body>"""


def build_model(traction_n):
    xml = (SIM / "tischlein.xml").read_text()
    xml = re.sub(r'<geom name="fr_door"[^>]*/>', door_xml(), xml)
    # drop the meal box / bottle / bed: not needed here
    xml = re.sub(r'<body name="meal_box".*?</body>', "", xml, flags=re.S)
    xml = re.sub(r'<body name="bottle".*?</body>', "", xml, flags=re.S)
    xml = xml.replace('<position name="base_x" joint="base_x" kp="40000" kv="3000" forcerange="-1500 1500"/>',
                      f'<position name="base_x" joint="base_x" kp="40000" kv="3000" forcerange="{-traction_n} {traction_n}"/>')
    xml = xml.replace('<position name="base_y" joint="base_y" kp="40000" kv="3000" forcerange="-1500 1500"/>',
                      f'<position name="base_y" joint="base_y" kp="40000" kv="3000" forcerange="{-traction_n} {traction_n}"/>')
    (SIM / "_door.xml").write_text(xml)
    return mujoco.MjModel.from_xml_path(str(SIM / "_door.xml"))


def handle_world(phi):
    """Handle bar centre for door opening angle phi (rad, positive = open)."""
    c, s = math.cos(-phi), math.sin(-phi)
    return HINGE + np.array([c * H_LOCAL[0] - s * H_LOCAL[1], s * H_LOCAL[0] + c * H_LOCAL[1]])


def arm_for_tip(base, tip, heading_deg, grip):
    bx, by = base
    wx = tip[0] - bx - TIP * math.cos(math.radians(heading_deg))
    wy = tip[1] - by - TIP * math.sin(math.radians(heading_deg))
    j1, j2, j3 = ik(wx, wy, heading_deg)
    return dict(j1=j1, j2=j2, j3=j3, j4=0.0, grip_l=grip)


def plan(phi_target_deg=100.0, base_shift=0.42, follow=False):
    """Timeline of joint targets as a function of t. Returns (fn, T, marks)."""
    h0 = handle_world(0.0)
    r_pull = 0.43                                       # nearly straight arm
    bx0 = h0[0] - TIP - SHOULDER - r_pull
    by0 = h0[1]
    lift = BAR_Z - (P.L2_ABOVE_TRAY - P.PITCH_DROP) / 1000
    pre = (h0[0] - 0.03, h0[1])
    segs = []

    def add(dur, fn, mark):
        t0 = segs[-1][1] if segs else 0.0
        segs.append((t0, t0 + dur, fn, mark))

    start = dict(base_x=bx0, base_y=by0, base_yaw=0.0, lift=lift, drawer=0.0,
                 **arm_for_tip((bx0, by0), pre, 0.0, 0.020))
    grip_close = -0.008
    add(1.0, lambda s: start, "hand in front of the handle")
    add(1.5, lambda s: {**start, **arm_for_tip((bx0, by0), (pre[0] + 0.03 * s, pre[1]), 0.0, 0.020)}, "reach the bar")
    held = {**start, **arm_for_tip((bx0, by0), h0, 0.0, 0.020)}
    add(1.0, lambda s: {**held, "grip_l": 0.020 + (grip_close - 0.020) * s}, "grip")
    held = {**held, "grip_l": grip_close}
    back = 0.05
    def backoff(s):
        phi = back * s / R_H
        base = (bx0 - back * s, by0)
        return {**held, "base_x": base[0], **arm_for_tip(base, handle_world(phi), -math.degrees(phi), grip_close)}
    add(3.0, backoff, "back off: break the seal")
    bx1 = bx0 - back

    phi0 = math.degrees(back / R_H)        # where the door is once the base has backed off

    def swing(s):
        phi = math.radians(phi0 + (phi_target_deg - phi0) * s)
        tip = handle_world(phi)
        if follow:
            # base keeps the wrist at a comfortable spot (330 mm ahead of the shoulder):
            # needs free floor on the hinge side of the fridge
            hd = -phi
            wrist = tip - TIP * np.array([math.cos(hd), math.sin(hd)])
            w0 = handle_world(math.radians(phi0)) - TIP * np.array([math.cos(-math.radians(phi0)), math.sin(-math.radians(phi0))])
            rel0 = np.array([w0[0] - bx1 - SHOULDER, w0[1] - by0])
            rel = rel0 + (np.array([0.33, 0.0]) - rel0) * min(1.0, 4 * s)
            base = (wrist[0] - SHOULDER - rel[0], wrist[1] - rel[1])
        else:
            base = (bx1 + 0.04 * s, by0 + base_shift * s)
        return {**held, "base_x": base[0], "base_y": base[1], **arm_for_tip(base, tip, -math.degrees(phi), grip_close)}
    add(10.0, swing, "follow the arc, base slides toward the hinge side")
    end = swing(1.0)
    add(1.0, lambda s: {**end, "grip_l": grip_close + (0.020 - grip_close) * s}, "let go past the detent")
    rel = {**end, "grip_l": 0.020}
    add(3.0, lambda s: rel, "door stays open?")
    T = segs[-1][1]
    reach = []
    for k in range(51):
        st = swing(k / 50)
        j2 = math.radians(st["j2"])
        reach.append(2 * 0.22 * math.cos(j2 / 2))
    plan.max_reach = max(reach)

    def at(t):
        for t0, t1, fn, mark in segs:
            if t0 <= t < t1:
                return fn(minjerk((t - t0) / (t1 - t0)))
        return segs[-1][2](1.0)
    marks = [(t0, mark) for t0, _, _, mark in segs]
    return at, T, marks


def seal_torque(phi, f_break):
    gap = phi * R_H
    seal = f_break * R_H * max(0.0, 1.0 - gap / 0.008)
    if phi < math.radians(85):
        close = 1.0 * min(1.0, phi / math.radians(5))
    elif phi < math.radians(92):
        close = 1.0 * (math.radians(92) - phi) / math.radians(7)
    else:
        close = -0.5
    return seal + close                    # positive = closing (joint q -> 0)


def run(f_break, mu, video=False, fps=30, speed=1.0, follow=False):
    traction = mu * 0.6 * ROBOT_MASS * 9.81
    m = build_model(traction)
    d = mujoco.MjData(m)
    at, T, marks = plan(follow=follow)
    jid = {j: m.joint(j).id for j in JOINTS}
    door_q = m.jnt_qposadr[m.joint("door").id]
    door_v = m.jnt_dofadr[m.joint("door").id]
    q0 = to_ctrl(at(0.0))
    for j, v in zip(JOINTS, q0):
        d.qpos[m.jnt_qposadr[jid[j]]] = v
    d.qpos[m.jnt_qposadr[m.joint("grip_r").id]] = q0[-1]
    set_ctrl(d, q0)
    init_extra(m, d)
    mujoco.mj_forward(m, d)
    names = [m.actuator(i).name for i in range(m.nu)]
    lim = np.array([m.actuator_forcerange[i][1] for i in range(m.nu)])
    peak = np.zeros(m.nu)
    max_phi = 0.0
    phi_after_break = None
    renderer = mujoco.Renderer(m, 540, 860) if video else None
    frames = OUT / "door_frames"
    if video:
        frames.mkdir(parents=True, exist_ok=True)
        for p in frames.glob("*.png"):
            p.unlink()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.15, 0.25, 0.95)
    cam.distance, cam.azimuth, cam.elevation = 3.3, 25, -24
    nxt, nf = 0.0, 0
    while d.time < T:
        set_ctrl(d, to_ctrl(at(d.time)))
        phi = -d.qpos[door_q]
        d.qfrc_applied[door_v] = seal_torque(phi, f_break)
        mujoco.mj_step(m, d)
        peak = np.maximum(peak, np.abs(d.actuator_force))
        max_phi = max(max_phi, phi)
        if phi_after_break is None and d.time > 6.5:
            phi_after_break = phi
        if video and d.time >= nxt:
            nxt += speed / fps
            renderer.update_scene(d, camera=cam)
            from PIL import Image, ImageDraw
            im = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(im)
            label = [mk for t0, mk in marks if t0 <= d.time][-1]
            dr.rectangle((0, 506, 860, 540), fill=(20, 20, 24))
            dr.text((10, 514), f"t={d.time:4.1f}s  door {math.degrees(phi):5.1f} deg  seal {f_break:.0f} N  floor mu {mu}  |  {label}", fill=(240, 240, 240))
            im.save(frames / f"f{nf:05d}.png")
            nf += 1
    final = math.degrees(-d.qpos[door_q])
    b_end = at(T - 0.01)
    b_start = at(0.0)
    travel = (b_end["base_x"] - b_start["base_x"], b_end["base_y"] - b_start["base_y"])
    ok = final > 88 and np.all(np.isfinite(d.qpos))
    res = dict(f_break=f_break, mu=mu, traction=traction, broke=math.degrees(phi_after_break or 0) > 1.0,
               max_deg=math.degrees(max_phi), final_deg=final, ok=ok, travel=travel,
               peaks={n: (peak[i], lim[i]) for i, n in enumerate(names)})
    if video and nf:
        mp4 = OUT / "door.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(mp4)], check=True)
        res["video"] = str(mp4)
    return res


if __name__ == "__main__":
    follow = "--follow" in sys.argv
    if "--video" in sys.argv:
        r = run(35, 0.6, video=True, follow=follow)
        print(r["video"], f"final {r['final_deg']:.0f} deg")
        sys.exit()
    plan(follow=follow)
    print(f"max arm reach used along the arc: {plan.max_reach*1000:.0f} mm of {P.L1+P.L2} mm ({'base follows the handle' if follow else 'base slides 42 cm sideways'})")
    print(f"{'seal N':>6} {'mu':>4} {'traction N':>10} {'seal broke':>10} {'max deg':>8} {'final deg':>9}  ok   J1%  J2%  J3%  grip%")
    for f_break in (15, 25, 35, 45, 55, 65):
        for mu in (0.4, 0.6, 0.8):
            r = run(f_break, mu, follow=follow)
            pk = r["peaks"]
            pct = lambda n: 100 * pk[n][0] / pk[n][1]
            print(f"{f_break:6.0f} {mu:4.1f} {r['traction']:10.0f} {str(r['broke']):>10} {r['max_deg']:8.1f} {r['final_deg']:9.1f}  "
                  f"{'yes' if r['ok'] else 'NO ':3s} {pct('j1'):4.0f} {pct('j2'):4.0f} {pct('j3'):4.0f} {pct('grip'):5.0f}  base moved {r['travel'][0]*100:+.0f}/{r['travel'][1]*100:+.0f} cm")
