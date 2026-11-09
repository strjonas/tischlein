"""Cloth, for real: can this hand take a garment out of the drum? Garments are soft handfuls, which the rigid-body world of
the other scenarios cannot represent.

    .venv/bin/python -m arena.laundry --trials 12 --workers 6     # -> arena/results/laundry.json

A separate small scene, because a flex cloth costs far more than a rigid flat: the washer drum with lifters, a small shirt's worth
of flex cloth with bending and damping, dropped in so it lies as it falls, and a hand on a straight reach with wrist pitch, moved
at the robot's real speeds. The hand goes in level through the porthole, tips down, closes where the cloth lies highest, pulls
back. Judged: is the garment out? The boom stops at the door lip, so without pitch the hand cannot reach the drum floor at all.

Cloth parameters are from the MuJoCo cloth examples, not measured; a real snag on a zip or a sleeve inside a sleeve is not
modelled. Read this as a mechanical check of reach and pinch through the porthole, not as a laundry success rate.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

OUT = Path(__file__).resolve().parent / "results"
DRUM_R, DRUM_L, OPEN_R = 0.25, 0.36, 0.16


def scene(seed, reach_m=0.60, stroke=0.10, jaw_len=0.06, grip_N=80.0, nx=14, ny=14, mass=0.20, pitch_max=1.5, press=None):
    rng = np.random.default_rng(seed)
    n = 16
    g = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rd = DRUM_R + 0.01
        g.append(f'<geom name="drum{i}" type="box" pos="{DRUM_L / 2} {rd * math.cos(a):.4f} {rd * math.sin(a):.4f}" size="{DRUM_L / 2} 0.01 {math.pi * rd / n + 0.008:.4f}" euler="{a:.4f} 0 0" rgba="0.7 0.72 0.75 1"/>')
        rm = (OPEN_R + DRUM_R) / 2 + 0.03
        g.append(f'<geom name="gasket{i}" type="box" pos="-0.005 {rm * math.cos(a):.4f} {rm * math.sin(a):.4f}" size="0.005 {(DRUM_R - OPEN_R) / 2 + 0.03:.4f} {math.pi * rm / n + 0.008:.4f}" euler="{a:.4f} 0 0" rgba="0.3 0.32 0.35 1"/>')
    for i, a in enumerate((math.pi / 2 + 2.094, math.pi / 2 - 2.094, math.pi / 2)):
        rr = DRUM_R - 0.02
        g.append(f'<geom name="lifter{i}" type="box" pos="{DRUM_L / 2} {rr * math.cos(a):.4f} {rr * math.sin(a):.4f}" size="{DRUM_L / 2 - 0.01} 0.02 0.012" euler="{a:.4f} 0 0" rgba="0.7 0.72 0.75 1"/>')
    g.append(f'<geom name="back" type="box" pos="{DRUM_L + 0.01} 0 0" size="0.01 {DRUM_R} {DRUM_R}" rgba="0.7 0.72 0.75 1"/>')
    px, py = rng.uniform(0.17, 0.20), rng.uniform(-0.04, 0.04)       # a 30 cm cloth: a small shirt as it lies in the drum
    roll = rng.uniform(-1.2, 1.2)
    sp = 0.30 / nx
    xml = f"""<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
    <size memory="400M"/><visual><global offwidth="900" offheight="600"/></visual>
    <default><geom contype="1" conaffinity="1" friction="0.5 0.005 0.0001" solref="0.004 1"/></default>
    <worldbody><light pos="0 0 1"/>{''.join(g)}
      <body name="hand" pos="-0.45 0 0"><joint name="x" type="slide" axis="1 0 0" range="-0.1 {reach_m + 0.05}" armature="0.2" damping="20"/><joint name="z" type="slide" axis="0 0 1" range="-0.25 0.15" armature="0.2" damping="20"/>
        <geom name="boom" type="box" pos="-0.05 0 0" size="0.19 0.02 0.02" contype="2" conaffinity="1" mass="0.5"/>
        <body name="wrist" pos="0.14 0 0"><joint name="wp" type="hinge" axis="0 1 0" range="-0.2 {pitch_max}" armature="0.02" damping="1"/>
          <geom name="palm" type="box" pos="0.03 0 0" size="0.025 0.04 0.025" mass="0.2" rgba="0.85 0.25 0.2 1" contype="2" conaffinity="1"/>
          <body name="ja" pos="0.06 0 0"><joint name="ja" type="slide" axis="0 1 0" range="0 {stroke / 2}" damping="10" armature="0.05"/><geom name="pad_a" type="box" pos="{jaw_len / 2} 0.003 0" size="{jaw_len / 2} 0.003 0.018" mass="0.04" friction="1.2 0.01 0.001" rgba="0.1 0.1 0.1 1" solref="0.008 1"/></body>
          <body name="jb" pos="0.06 0 0"><joint name="jb" type="slide" axis="0 -1 0" range="0 {stroke / 2}" damping="10" armature="0.05"/><geom name="pad_b" type="box" pos="{jaw_len / 2} -0.003 0" size="{jaw_len / 2} 0.003 0.018" mass="0.04" friction="1.2 0.01 0.001" rgba="0.1 0.1 0.1 1" solref="0.008 1"/></body>
          <site name="tip" pos="{0.06 + jaw_len} 0 0" size="0.005"/>
        </body>
      </body>
      <flexcomp name="cloth" type="grid" count="{nx} {ny} 1" spacing="{sp:.4f} {sp:.4f} {sp:.4f}" pos="{px:.4f} {py:.4f} 0.10" euler="{roll * 0.5:.3f} {rng.uniform(-0.5, 0.5):.3f} 0" dim="2" radius="0.004" mass="{mass}" rgba="0.8 0.45 0.55 1">
        <contact condim="3" solref="0.008 1" selfcollide="none" friction="0.8 0.005 0.0001" contype="1" conaffinity="3"/><edge equality="false" damping="0.03"/>
        <elasticity young="2e4" poisson="0.2" thickness="8e-4" damping="2e-3"/></flexcomp>
    </worldbody>
    <equality><joint joint1="ja" joint2="jb" polycoef="0 1 0 0 0"/></equality>
    <actuator><position name="ax" joint="x" kp="4000" kv="400" forcerange="-60 60"/><position name="az" joint="z" kp="8000" kv="600" forcerange="-200 200"/>
    <position name="ag" joint="ja" kp="{grip_N / 0.002:.0f}" dampratio="1" forcerange="-{grip_N} {grip_N}"/>
    <position name="ap" joint="wp" kp="80" dampratio="1" forcerange="-3 3"/></actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def run(job):
    t0 = time.time()
    seed = job["seed"]
    p = job.get("params", {})
    m = scene(seed, **p)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    jaw_open = p.get("stroke", 0.10) / 2
    d.ctrl[1] = -0.05
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "z")]] = -0.05
    d.ctrl[2] = jaw_open
    vid = m.flex_vertadr[0]
    nv = m.flex_vertnum[0]

    def cloth():
        return d.flexvert_xpos[vid:vid + nv]

    # let it fall and settle
    for _ in range(int(1.2 / dt)):
        mujoco.mj_step(m, d)
    pts = cloth().copy()
    # a depth camera finds the highest cloth inside reach: noise 5 mm. A pinch needs a fold, so it aims at the top of the pile
    rng = np.random.default_rng(seed)
    inside = (pts[:, 0] > 0.02) & (pts[:, 0] < DRUM_L)
    k = np.argsort(-pts[:, 2] * inside)[:12]
    tgt = pts[k].mean(0) + rng.normal(0, 0.005, 3)
    jaw = p.get("jaw_len", 0.06)
    th = min(1.25, p.get("pitch_max", 1.5))
    L = 0.06 + jaw * 0.8
    x_t = tgt[0] - L * math.cos(th) + 0.45 - 0.14
    if x_t > p.get("reach_m", 0.60):
        return dict(job=job, ok=False, why="beyond reach", x=round(float(tgt[0]), 3), wall=round(time.time() - t0, 1))
    z_t = tgt[2] + L * math.sin(th) - p.get("press", 0.010)     # fingertips pressed in: closing then gathers a fold

    def go(x, z, grip, pitch, T):
        x0, z0, g0, p0 = d.ctrl[0], d.ctrl[1], d.ctrl[2], d.ctrl[3]
        steps = max(int(T / dt), 1)
        for i in range(steps):
            s = (i + 1) / steps
            d.ctrl[0], d.ctrl[1], d.ctrl[2], d.ctrl[3] = x0 + (x - x0) * s, z0 + (z - z0) * s, g0 + (grip - g0) * s, p0 + (pitch - p0) * s
            mujoco.mj_step(m, d)

    go(x_t, -0.05, jaw_open, 0.0, abs(x_t) / 0.15 + 0.3)     # in through the porthole at telescope speed, level
    go(x_t, -0.05, jaw_open, th, 1.2)                         # wrist down inside the drum
    go(x_t, z_t, jaw_open, th, 1.2)                           # down onto the cloth
    blocked = float(d.qpos[1] - z_t)
    go(x_t, z_t, -0.02, th, 1.5)                              # close
    held = float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ja")]] * 2)
    go(x_t, -0.02, -0.02, th, 1.0)                            # lift
    go(x_t, -0.02, -0.02, 0.2, 1.0)                           # wrist back
    go(-0.10, -0.02, -0.02, 0.2, (x_t + 0.10) / 0.12)         # pull out at 12 cm/s
    pts = cloth()
    out = float((pts[:, 0] < 0.0).mean())
    return dict(job=job, ok=bool(out > 0.8), out=round(out, 2), held=round(held, 4), blocked_cm=round(blocked * 100, 1), x=round(float(tgt[0]), 3), z=round(float(tgt[2]), 3), wall=round(time.time() - t0, 1), nan=bool(np.isnan(d.qpos).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    groups = {"80 N, 10 cm hand, 0.60 m reach, wrist pitch": {}, "no wrist pitch": dict(pitch_max=0.05), "30 N hand": dict(grip_N=30.0), "0.45 m reach": dict(reach_m=0.45), "20 cm hand": dict(stroke=0.20, jaw_len=0.09), "no press": dict(press=0.002)}
    jobs = [dict(group=g, seed=s, params=p) for g, p in groups.items() for s in range(a.trials)]
    with Pool(a.workers) as p:
        R = p.map(run, jobs, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "laundry.json").write_text(json.dumps(R, indent=0))
    for g in groups:
        rs = [r for r in R if r["job"]["group"] == g]
        print(f"{g:34s} garment out {sum(r['ok'] for r in rs):2d}/{len(rs)}   beyond reach {sum(r.get('why') == 'beyond reach' for r in rs)}   mean share out {np.mean([r.get('out', 0) for r in rs]):.2f}   wall {np.mean([r['wall'] for r in rs]):.0f} s")


if __name__ == "__main__":
    main()
