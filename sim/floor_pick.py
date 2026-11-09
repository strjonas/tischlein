"""Dropped things off the floor: two-finger grasp vs scoop and sweep (27 Oct 2026).

uv run python sim/floor_pick.py [N]          # N trials per object x floor x strategy (default 20) -> out/floor_pick.txt

People with motor impairments drop things ~5.5 times a day and wait up to two hours for help (Georgia Tech, ALS
patients). Georgia Tech's Dusty showed a dustpan-like end effector (a thin wedge plus a sweeping finger) picks
objects off the floor in 94.7 % of 680 trials. Here the idea is to get that from parts Tischlein already has:
a thin plate (the drawer's tongue, or a scoop on the carriage) lowered onto the floor, and the CAP hand pitched
down, its two fingertips used as the broom.

Free-floating hand and scoop (kinematics-agnostic: the lift goes to the floor and the wrist pitches down), each on a
soft vertical spring so the fingertips rest on the floor with ~1.6 N (touch-calibrated). Same errors for both strategies:
object position estimate sigma 5 mm, its heading sigma 8 deg; floors: tile (mu 0.3), wood (0.45), carpet-like (0.8).

  grasp: fingers across the object's narrow side, lowered until the rounded tips (3 mm radius) touch the floor,
         squeeze 15 N per finger, lift.
  scoop: plate in front of the object (robot side), fingertips behind it with a gap a bit narrower than the
         object, sweep it onto the plate while the plate advances, then lift both together.
Objects from the top of the ALS retrieval list: TV remote, phone, glasses case, pill bottle (lying), credit card,
coin, keys, a rolled sock. All rigid: a real card bends and a sock squashes, so the grasp numbers for those are
optimistic, and the scoop numbers for the sock are uncertain.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# render/test_videos.py sets this to {"fps", "next", "frames", "grab": fn(m, d) -> image} to record one trial
VIDEO = None
OUT = ROOT / "out"

# name: (geom type, size (MuJoCo half-sizes), mass kg, euler for the resting pose)
OBJECTS = {
    "TV remote": ("box", (0.090, 0.0225, 0.011), 0.12, (0, 0, 0)),
    "phone": ("box", (0.075, 0.036, 0.004), 0.18, (0, 0, 0)),
    "glasses case": ("box", (0.078, 0.030, 0.018), 0.08, (0, 0, 0)),
    "pill bottle (lying)": ("cylinder", (0.017, 0.035, 0), 0.04, (0, 90, 0)),
    "credit card": ("box", (0.0428, 0.027, 0.0004), 0.005, (0, 0, 0)),
    "coin": ("cylinder", (0.012, 0.001, 0), 0.0075, (0, 0, 0)),
    "keys": ("box", (0.035, 0.015, 0.005), 0.05, (0, 0, 0)),
    "rolled sock": ("box", (0.050, 0.035, 0.020), 0.05, (0, 0, 0)),
}
FLOORS = {"tile": 0.3, "wood": 0.45, "carpet": 0.8}
SQUEEZE = 15.0                    # N per finger
TIP_HALF = 0.006                  # finger tip half-width (12 mm fingers); the tips are rounded, 3 mm radius


def footprint(name):
    """(length along x, width along y, height) of the object as it lies."""
    t, s, _, e = OBJECTS[name]
    if t == "box":
        return 2 * s[0], 2 * s[1], 2 * s[2]
    if e[1] == 90:                 # cylinder on its side, axis along x
        return 2 * s[1], 2 * s[0], 2 * s[0]
    return 2 * s[0], 2 * s[0], 2 * s[1]


def xml(name, mu):
    t, s, mass, e = OBJECTS[name]
    L, W, H = footprint(name)
    size = " ".join(f"{v:.5f}" for v in s if v)
    return f"""
<mujoco model="floor_pick">
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="5"/>
  <default><geom solref="0.004 1" solimp="0.95 0.99 0.0005" condim="4"/></default>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" friction="{mu} 0.01 0.0002"/>
    <body name="obj" pos="0 0 {H / 2 + 0.0005:.5f}" euler="{e[0]} {e[1]} {e[2]}">
      <freejoint/>
      <geom name="o" type="{t}" size="{size}" mass="{mass}" friction="{mu} 0.01 0.0002" rgba="0.85 0.35 0.25 1" contype="3" conaffinity="3"/>
    </body>
    <!-- hand and scoop are driven through actuated joints (x, y, z, yaw), not mocap: a mocap parent has no
         velocity, so friction against the object would never engage. The hand's z axis is soft (400 N/m): it is
         the touch spring that lets the fingertips rest on the floor with ~1.6 N. -->
    <body name="hand" pos="0 0 0">
      <joint name="hx" type="slide" axis="1 0 0" damping="20"/>
      <joint name="hy" type="slide" axis="0 1 0" damping="20"/>
      <joint name="hz" type="slide" axis="0 0 1" damping="8"/>
      <joint name="hyaw" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom name="palm" type="box" size="0.03 0.07 0.015" pos="0 0 0.09" mass="0.25" contype="0" conaffinity="0"/>
      <body name="fl" pos="0 0 0">
        <joint name="fl" type="slide" axis="0 1 0" range="0 0.07" damping="2"/>
        <geom name="tl" type="box" size="{TIP_HALF} 0.006 0.0385" pos="0 0.006 0.0415" mass="0.02" friction="0.8 0.02 0.001"/>
        <geom name="tl_tip" type="capsule" size="0.003 {TIP_HALF}" pos="0 0.003 0.003" euler="0 90 0" mass="0.002" friction="0.8 0.02 0.001"/>
      </body>
      <body name="fr" pos="0 0 0">
        <joint name="fr" type="slide" axis="0 -1 0" range="0 0.07" damping="2"/>
        <geom name="tr" type="box" size="{TIP_HALF} 0.006 0.0385" pos="0 -0.006 0.0415" mass="0.02" friction="0.8 0.02 0.001"/>
        <geom name="tr_tip" type="capsule" size="0.003 {TIP_HALF}" pos="0 -0.003 0.003" euler="0 90 0" mass="0.002" friction="0.8 0.02 0.001"/>
      </body>
    </body>
    <body name="scoop" pos="0 0 0">
      <joint name="sx" type="slide" axis="1 0 0" damping="20"/>
      <joint name="sy" type="slide" axis="0 1 0" damping="20"/>
      <joint name="sz" type="slide" axis="0 0 1" damping="20"/>
      <joint name="syaw" type="hinge" axis="0 0 1" damping="0.5"/>
      <!-- 1.2 mm plate rising 3 deg towards its back wall; its front edge is ground to a feather edge that lies on
           the floor, modelled as the plate sitting 0.2 mm into the floor and colliding only with the object -->
      <body euler="0 3 0">
        <geom name="plate" type="box" size="0.11 0.09 0.0006" pos="-0.11 0 0.0006" mass="0.15" friction="0.5 0.01 0.0002" contype="2" conaffinity="2"/>
        <geom name="wall_l" type="box" size="0.11 0.001 0.008" pos="-0.11 0.09 0.008" mass="0.01" contype="2" conaffinity="2"/>
        <geom name="wall_r" type="box" size="0.11 0.001 0.008" pos="-0.11 -0.09 0.008" mass="0.01" contype="2" conaffinity="2"/>
        <geom name="wall_b" type="box" size="0.001 0.09 0.015" pos="-0.22 0 0.015" mass="0.01" contype="2" conaffinity="2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="grip" joint="fl" kp="3000" kv="30" forcerange="{-SQUEEZE} {SQUEEZE}"/>
    <position name="grip_r" joint="fr" kp="3000" kv="30" forcerange="{-SQUEEZE} {SQUEEZE}"/>
    <position name="hx" joint="hx" kp="20000" kv="300"/>
    <position name="hy" joint="hy" kp="20000" kv="300"/>
    <position name="hz" joint="hz" kp="400" kv="15"/>
    <position name="hyaw" joint="hyaw" kp="200" kv="4"/>
    <position name="sx" joint="sx" kp="20000" kv="300"/>
    <position name="sy" joint="sy" kp="20000" kv="300"/>
    <position name="sz" joint="sz" kp="20000" kv="300"/>
    <position name="syaw" joint="syaw" kp="200" kv="4"/>
  </actuator>
</mujoco>"""


class Plan:
    """Piecewise-linear mocap targets (x, y, z, yaw) for hand and scoop, plus the finger opening."""

    def __init__(self, hand, scoop, grip):
        self.k = [(0.0, np.array(hand, float), np.array(scoop, float), grip)]

    def to(self, dur, hand=None, scoop=None, grip=None):
        t, h, s, g = self.k[-1]
        self.k.append((t + dur, np.array(hand, float) if hand is not None else h,
                       np.array(scoop, float) if scoop is not None else s, g if grip is None else grip))

    @property
    def T(self):
        return self.k[-1][0]

    def at(self, t):
        for (t0, h0, s0, g0), (t1, h1, s1, g1) in zip(self.k, self.k[1:]):
            if t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                a = a * a * (3 - 2 * a)
                return h0 + (h1 - h0) * a, s0 + (s1 - s0) * a, g0 + (g1 - g0) * a
        _, h, s, g = self.k[-1]
        return h, s, g


def quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def plan_grasp(name, est):
    """est = (x, y, yaw) of the object. Fingers close across its width; tips pressed 4 mm into the floor spring."""
    L, W, H = footprint(name)
    x, y, yaw = est
    open_ = min(0.07, W / 2 + 0.015)
    up = 0.12
    p = Plan((x, y, up, yaw), (-0.6, 0, 0.3, 0), open_)
    p.to(0.6, hand=(x, y, -0.004, yaw))                       # down until the tips rest on the floor
    p.to(0.6, grip=-0.01)                                     # squeeze
    p.to(0.8, hand=(x, y, 0.10, yaw))                         # lift
    p.to(1.0)
    return p


def plan_scoop(name, est):
    """Plate's front edge 30 mm in front of the object (robot side, -x along the object's long axis), fingertips
    15 mm behind it; sweep the object onto the plate while the plate advances 30 mm; lift both."""
    L, W, H = footprint(name)
    x, y, yaw = est
    c, s = math.cos(yaw), math.sin(yaw)

    def pt(d):                                                # point d metres along the object's long axis
        return x + d * c, y + d * s

    gap = max(0.004, min(0.05, W / 2 - 0.008))                # broom narrower than the object
    front0, front1 = pt(-L / 2 - 0.030), pt(-L / 2 + 0.000)
    hb0 = pt(L / 2 + 0.015 + TIP_HALF)
    hb1 = pt(-L / 2 - 0.22 + L + 0.02 + TIP_HALF)             # tips end ~2 cm in front of the plate's back wall
    zs = -0.0014                                              # plate's top front edge 0.2 mm below the floor
    p = Plan((*hb0, 0.12, yaw), (*front0, 0.10, yaw), gap)
    p.to(0.6, scoop=(*front0, zs, yaw))                       # plate onto the floor
    p.to(0.6, hand=(*hb0, -0.004, yaw))                       # fingertips onto the floor behind the object
    p.to(1.6, hand=(*hb1, -0.004, yaw), scoop=(*front1, zs, yaw))   # sweep (the tips pass over the plate)
    p.to(0.8, hand=(*hb1, 0.06, yaw), scoop=(*front1, 0.06, yaw))   # lift together
    p.to(1.0)
    return p


def trial(args):
    name, floor, strategy, i = args
    rng = np.random.default_rng(1000 * list(OBJECTS).index(name) + 100 * list(FLOORS).index(floor) + i)
    m = mujoco.MjModel.from_xml_string(xml(name, FLOORS[floor]))
    d = mujoco.MjData(m)
    L, W, H = footprint(name)
    yaw = rng.uniform(-math.pi, math.pi)
    qa = m.jnt_qposadr[m.joint(0).id]
    q0 = d.qpos[qa + 3:qa + 7].copy()
    qz = quat_z(yaw)
    qn = np.zeros(4)
    mujoco.mju_mulQuat(qn, qz, q0)
    d.qpos[qa + 3:qa + 7] = qn
    est = (rng.normal(0, 0.005), rng.normal(0, 0.005), yaw + math.radians(rng.normal(0, 8)))
    plan = plan_grasp(name, est) if strategy == "grasp" else plan_scoop(name, est)
    act = {n: m.actuator(n).id for n in ("grip", "grip_r", "hx", "hy", "hz", "hyaw", "sx", "sy", "sz", "syaw")}
    jq = {n: m.jnt_qposadr[m.joint(n).id] for n in ("hx", "hy", "hz", "hyaw", "sx", "sy", "sz", "syaw", "fl", "fr")}
    sag = 0.29 * 9.81 / 400                                   # the soft z axis carries the hand's weight

    def command(t):
        h, s_, g = plan.at(t)
        for k, v in zip(("hx", "hy", "hz", "hyaw"), (h[0], h[1], h[2] + sag, h[3])):
            d.ctrl[act[k]] = v
        for k, v in zip(("sx", "sy", "sz", "syaw"), s_):
            d.ctrl[act[k]] = v
        d.ctrl[act["grip"]] = d.ctrl[act["grip_r"]] = g
        return h, s_, g

    h, s_, g = command(0.0)
    for k, v in zip(("hx", "hy", "hz", "hyaw"), h):
        d.qpos[jq[k]] = v
    for k, v in zip(("sx", "sy", "sz", "syaw"), s_):
        d.qpos[jq[k]] = v
    d.qpos[jq["fl"]] = d.qpos[jq["fr"]] = g
    mujoco.mj_forward(m, d)
    while d.time < plan.T:
        command(d.time)
        mujoco.mj_step(m, d)
        if VIDEO is not None and d.time >= VIDEO["next"]:
            VIDEO["next"] += 1.0 / VIDEO["fps"]
            VIDEO["frames"].append(VIDEO["grab"](m, d))
        if not np.all(np.isfinite(d.qpos)):
            return dict(name=name, floor=floor, strategy=strategy, ok=False, why="unstable")
    z = d.xpos[m.body("obj").id][2]
    ok = z > 0.03                                             # off the floor, held or on the lifted plate
    return dict(name=name, floor=floor, strategy=strategy, ok=bool(ok), z=float(z))


if __name__ == "__main__":
    import multiprocessing as mp
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
    jobs = [(o, f, st, i) for o in OBJECTS for f in FLOORS for st in ("grasp", "scoop") for i in range(N)]
    with mp.get_context("spawn").Pool(6) as pool:
        res = pool.map(trial, jobs, chunksize=8)
    cell = defaultdict(list)
    for r in res:
        cell[(r["name"], r["strategy"])].append(r["ok"])
        cell[("all", r["strategy"])].append(r["ok"])
        cell[(r["floor"], r["strategy"])].append(r["ok"])
    lines = [f"Floor pickup (sim/floor_pick.py): {N} trials per object x floor x strategy, all three floors pooled", "",
             f"{'object':22s}{'grasp':>8s}{'scoop':>8s}"]
    for o in list(OBJECTS) + ["all"]:
        g, s = cell[(o, "grasp")], cell[(o, "scoop")]
        lines.append(f"{o:22s}{100 * np.mean(g):7.0f}%{100 * np.mean(s):7.0f}%")
    lines += ["", f"{'floor':22s}{'grasp':>8s}{'scoop':>8s}"]
    for f in FLOORS:
        lines.append(f"{f + f' (mu {FLOORS[f]})':22s}{100 * np.mean(cell[(f, 'grasp')]):7.0f}%{100 * np.mean(cell[(f, 'scoop')]):7.0f}%")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "floor_pick.txt").write_text(txt + "\n")
