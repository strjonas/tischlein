"""Opening a screw-top jar: one arm, one arm + a clamp, two arms. v0.2 robot, real joint speeds and limits.

uv run python sim/jar.py            # sweep lid torque x grip force for the three set-ups
uv run python sim/jar.py --video two 1.0 20    # one run -> out/jar_two.mp4

The jar (80 mm, 0.5 kg) stands on a table. The right hand grips the lid from the side, level (pitch 0), and turns
it by rotating the hand about the lid's axis: the flat arm can do that (x, y and heading are its three planar
degrees of freedom), in 60 deg strokes: grip, turn, let go, turn back, again. Done when the lid has turned a full
turn. The lid's breakaway torque is a parameter (an opened jar ~0.3-1 N*m; a factory-sealed one 1.5-3+ N*m).
  one          : the jar just stands on the table (non-slip mat, mu 0.6). Nothing else holds it.
  one_clamp    : the jar sits in a clamp on the table (an under-shelf jar holder or a printed V-clamp).
  two          : the left hand holds the jar body, 7 cm lower, hand tilted 17 deg nose-down.
Grip force is the CAP gripper's squeeze per finger (its XL330 servo; not measured, swept 10-30 N). Physics says the
lid turns only if 2 x mu x squeeze x lid radius exceeds the lid torque: ~0.84 N*m per 10 N of squeeze here.
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
import fridge_door as F  # noqa: E402  (robot helpers: IK, timeline with real speeds, tuck poses)

OUT = ROOT / "out"
TABLE_Z = 0.72
JAR = np.array([0.60, 0.0])          # jar axis, robot frame = world (the base stays put)
JAR_R, JAR_H, LID_R, LID_H = 0.040, 0.120, 0.042, 0.015
LID_Z = TABLE_Z + JAR_H + LID_H / 2
BODY_Z = TABLE_Z + 0.055             # where the left hand holds the body
TRAY_Z = LID_Z - F.PITCH_ABOVE_TRAY  # right hand level at the lid
PITCH_B = math.degrees(math.asin((LID_Z - BODY_Z) / F.TIP))
TURN = 2 * math.pi                   # success: one full turn


def jar_xml(clamp):
    weld = ""
    return f"""
    <geom name="table" type="box" pos="0.80 0 {TABLE_Z - 0.02:.3f}" size="0.33 0.5 0.02" rgba="0.62 0.47 0.33 1" friction="0.6 0.01 0.001"/>
    <body name="jar" pos="{JAR[0]} {JAR[1]} {TABLE_Z}">
      {'' if clamp else '<freejoint name="jar_free"/>'}
      <inertial pos="0 0 0.05" mass="0.5" diaginertia="0.0009 0.0009 0.0004"/>
      <geom type="cylinder" pos="0 0 {JAR_H / 2}" size="{JAR_R} {JAR_H / 2}" rgba="0.80 0.88 0.92 0.75" friction="0.6 0.01 0.001"/>
      <body name="lid" pos="0 0 {JAR_H + LID_H / 2}">
        <joint name="lid" type="hinge" axis="0 0 1" damping="0.02" frictionloss="1.0"/>
        <inertial pos="0 0 0" mass="0.03" diaginertia="2e-5 2e-5 3e-5"/>
        <geom type="cylinder" size="{LID_R} {LID_H / 2}" rgba="0.85 0.65 0.15 1" friction="0.9 0.01 0.001"/>
        <geom type="box" pos="{LID_R - 0.004} 0 {LID_H / 2 + 0.001}" size="0.006 0.003 0.001" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
      </body>
    </body>{weld}"""


def build_model(setup, grip_n):
    xml = (F.SIM / "tischlein.xml").read_text()
    for pat in (r'<geom name="fr_[^"]*"[^>]*/>', r'<geom name="shelf\d"[^>]*/>', r'<geom name="bed_[^"]*"[^>]*/>',
                r'<geom name="(mattress|duvet|pillow)"[^>]*/>'):
        xml = re.sub(pat, "", xml)
    for nm in ("meal_box", "bottle"):
        xml = re.sub(rf'<body name="{nm}".*?</body>', "", xml, flags=re.S)
    xml = xml.replace("</worldbody>", jar_xml(setup == "one_clamp") + "\n  </worldbody>")
    # the CAP gripper squeezes with its XL330, not 100 N; foam pads on glass/metal. The actuator drives both
    # (coupled) fingers, so it needs twice the per-finger squeeze.
    xml = xml.replace('<position name="grip" joint="grip_l" kp="8000" kv="60" forcerange="-100 100"/>',
                      f'<position name="grip" joint="grip_l" kp="8000" kv="60" forcerange="-{2 * grip_n} {2 * grip_n}"/>')
    xml = xml.replace('<position name="gripb" joint="grip_lb" kp="8000" kv="60" forcerange="-100 100"/>',
                      f'<position name="gripb" joint="grip_lb" kp="8000" kv="60" forcerange="-{2 * grip_n} {2 * grip_n}"/>')
    xml = xml.replace('<default class="pad"><geom group="3" friction="1.2 0.02 0.002"',
                      '<default class="pad"><geom group="3" friction="1.0 0.02 0.002"')
    if setup != "two":
        xml = F.ghost_arm_b(xml)
    path = F.SIM / f"_jar_{setup}.xml"
    path.write_text(xml)
    return mujoco.MjModel.from_xml_path(str(path))


def lid_pose(h_deg):
    """Right hand's pads on the lid centre, hand heading h (deg, world)."""
    return F.arm_for((0.0, 0.0, 0.0), JAR, math.radians(h_deg), 0.0, "a")


def plan(setup, strokes=9, stroke=60.0, h0=10.0):
    g_open, g_lid, g_body = 0.045, 0.020, 0.016
    start = {**F.base_state((0.0, 0.0, 0.0)), "lift": TRAY_Z, "drawer": 0.0, **F.PARK["a"], **F.PARK["b"]}
    T = F.Timeline(start)
    # hands travel nose-up (-15 deg: pads 6 cm higher, clear of the lid), drop to the working pitch 10 cm out,
    # then slide in along their heading
    if setup == "two":
        hb = math.radians(-70.0)
        far_b = JAR - 0.10 * np.array([math.cos(hb), math.sin(hb)])
        T.to(3.0, "left hand up", **{**F.side_pose("b", g_open), "j4b": -15.0})
        T.to(2.0, "left hand behind the jar", **F.arm_for((0, 0, 0), far_b, hb, g_open, "b", -15.0))
        T.to(1.0, "left hand down", j4b=PITCH_B)
        T.add(1.2, lambda s: F.arm_for((0, 0, 0), far_b + (JAR - far_b) * s, hb, g_open, "b", PITCH_B), "around the jar")
        T.to(0.6, "left hand holds the jar", gripb=g_body)
    h = math.radians(h0)
    far_a = JAR - 0.10 * np.array([math.cos(h), math.sin(h)])
    T.to(3.0, "right hand up", **{**F.side_pose("a", g_open), "j4": -15.0})
    T.to(2.0, "right hand behind the lid", **F.arm_for((0, 0, 0), far_a, h, g_open, "a", -15.0))
    T.to(1.0, "right hand level", j4=0.0)
    T.add(1.2, lambda s: F.arm_for((0, 0, 0), far_a + (JAR - far_a) * s, h, g_open, "a"), "around the lid")
    for k in range(strokes):
        T.to(0.5, f"grip the lid ({k + 1})", grip_l=g_lid)
        T.add(1.5, lambda s: {**lid_pose(h0 + stroke * s), "grip_l": g_lid}, f"turn {stroke:.0f} deg ({k + 1})")
        T.to(0.4, "let go", grip_l=g_open)
        T.add(1.5, lambda s: {**lid_pose(h0 + stroke * (1 - s)), "grip_l": g_open}, "turn back")
    T.add(1.0, lambda s: {}, "done?")
    return T


def run(setup, lid_nm, grip_n, video=False, fps=30):
    m = build_model(setup, grip_n)
    d = mujoco.MjData(m)
    T = plan(setup)
    aid = {nm: m.actuator(nm).id for nm in F.ACT}
    jq = {nm: m.jnt_qposadr[m.actuator_trnid[aid[nm]][0]] for nm in F.ACT}
    lid_q = m.jnt_qposadr[m.joint("lid").id]
    lid_v = m.jnt_dofadr[m.joint("lid").id]

    def vec(st):
        v = np.zeros(m.nu)
        for nm, key in F.ACT.items():
            v[aid[nm]] = math.radians(st[key]) if key in F.ANGLES else st[key]
        return v
    vmax = np.zeros(m.nu)
    for nm, key in F.ACT.items():
        vmax[aid[nm]] = math.radians(F.VM[key]) if key in F.ANGLES else F.VM[key]
    q0 = vec(T.at(0.0))
    for nm in F.ACT:
        d.qpos[jq[nm]] = q0[aid[nm]]
    d.ctrl[:] = q0
    mujoco.mj_forward(m, d)
    jar_id = m.body("jar").id
    jar0 = d.xpos[jar_id].copy()
    lim = np.array([m.actuator_forcerange[i][1] for i in range(m.nu)])
    peak = np.zeros(m.nu)
    cmd = q0.copy()
    dt = m.opt.timestep
    renderer = mujoco.Renderer(m, 540, 860) if video else None
    frames = OUT / "jar_frames"
    if video:
        frames.mkdir(parents=True, exist_ok=True)
        for p in frames.glob("*.png"):
            p.unlink()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.50, 0.0, 0.80)
    cam.distance, cam.azimuth, cam.elevation = 1.3, 200, -30
    nxt, nf = 0.0, 0
    while d.time < T.t:
        ang = abs(d.qpos[lid_q])
        # thread: breakaway torque for the first 20 deg (seal + static thread friction), then ~a third of it
        m.dof_frictionloss[lid_v] = lid_nm if ang < math.radians(20) else max(0.1, 0.35 * lid_nm)
        tgt = vec(T.at(d.time))
        cmd = cmd + np.clip(tgt - cmd, -vmax * dt * 1.05, vmax * dt * 1.05)
        d.ctrl[:] = cmd
        mujoco.mj_step(m, d)
        peak = np.maximum(peak, np.abs(d.actuator_force))
        if not np.all(np.isfinite(d.qpos)):
            break
        if video and d.time >= nxt:
            nxt += 1.0 / fps
            renderer.update_scene(d, camera=cam)
            from PIL import Image, ImageDraw
            im = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(im)
            mk = [mk for t0, mk in T.marks() if t0 <= d.time][-1]
            dr.rectangle((0, 506, 860, 540), fill=(20, 20, 24))
            dr.text((10, 514), f"t={d.time:4.1f}s  lid {math.degrees(d.qpos[lid_q]):6.0f} deg  {setup}, lid {lid_nm} N*m, "
                    f"grip {grip_n} N  |  {mk}", fill=(240, 240, 240))
            im.save(frames / f"f{nf:05d}.png")
            nf += 1
    lid_deg = math.degrees(d.qpos[lid_q])
    moved = float(np.linalg.norm(d.xpos[jar_id][:2] - jar0[:2]))
    upright = float(d.xmat[jar_id].reshape(3, 3)[2, 2])
    jar_turn = 0.0
    if setup == "one":
        R = d.xmat[jar_id].reshape(3, 3)
        jar_turn = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    ok = lid_deg >= math.degrees(TURN) - 1 and upright > 0.95 and moved < 0.03
    res = dict(setup=setup, lid_nm=lid_nm, grip_n=grip_n, ok=ok, lid_deg=lid_deg, jar_moved=moved, jar_turn=jar_turn,
               upright=upright, duration=T.t, peaks={m.actuator(i).name: 100 * peak[i] / lim[i] for i in range(m.nu)})
    if video and nf:
        mp4 = OUT / f"jar_{setup}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(mp4)], check=True)
        res["video"] = str(mp4)
    return res


def fmt(r):
    pk = r["peaks"]
    return (f"{r['setup']:>9} {r['lid_nm']:5.1f} {r['grip_n']:5.0f}  {r['lid_deg']:6.0f} {'yes' if r['ok'] else 'NO':>4}  "
            f"{100 * r['jar_moved']:5.1f} {r['jar_turn']:6.0f} {pk['j1']:4.0f} {pk['j2']:4.0f} {pk['j3']:4.0f} "
            f"{pk.get('j3b', 0):4.0f} {r['duration']:5.0f}")


HEADER = ("    setup lid N*m grip N  lid deg  ok?  jar moved cm  jar turned deg  J1% J2% J3% J3b%  time s")

if __name__ == "__main__":
    if "--video" in sys.argv:
        i = sys.argv.index("--video")
        r = run(sys.argv[i + 1], float(sys.argv[i + 2]), float(sys.argv[i + 3]), video=True)
        print(HEADER)
        print(fmt(r), r.get("video"))
        sys.exit()
    print(f"jar {2 * JAR_R * 1000:.0f} mm, 0.5 kg; lid {2 * LID_R * 1000:.0f} mm; left hand pitch {PITCH_B:.0f} deg; "
          f"success = lid turned {math.degrees(TURN):.0f} deg with the jar upright and in place")
    print(HEADER)
    rows = []
    for setup, lid, grip in product(("one", "one_clamp", "two"), (0.3, 0.6, 1.0, 1.5, 2.0), (10, 20, 30)):
        r = run(setup, lid, grip)
        rows.append(r)
        print(fmt(r), flush=True)
