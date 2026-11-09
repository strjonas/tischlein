"""Does the arena's physics match the textbook where a textbook answer exists?

    .venv/bin/python -m arena.tests.test_physics            # prints a table, writes arena/results/physics_validation.json

Each test isolates one mechanism in the arena's own contact settings (assemble.PHYS). A failed row here means a number in
the leaderboard is a solver artifact, so these run before anything else. Every test also runs at half and a quarter of
the timestep: a result that moves with the timestep is not physics.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from .. import assemble
from ..assemble import PHYS, header

OUT = Path(__file__).resolve().parents[1] / "results" / "v2"
G = 9.81


def scene(body, p=None, extra=""):
    return mujoco.MjModel.from_xml_string(header(dict(PHYS, **(p or {}))) + body + "\n  </worldbody>\n" + extra + "</mujoco>\n")


def run(m, d, T, f=None):
    n = int(T / m.opt.timestep)
    for i in range(n):
        if f:
            f(i * m.opt.timestep)
        mujoco.mj_step(m, d)


def incline(mu, p=None):
    """Block on a tilting plane: slips at tan(angle) = mu. Returns the slip angle and the creep just below it."""
    res = None
    creep = None
    for deg in np.arange(math.degrees(math.atan(mu)) - 4, math.degrees(math.atan(mu)) + 4, 0.5):
        a = math.radians(deg)
        body = f"""<body euler="0 {a} 0"><geom name="plane" class="world" type="box" size="1 0.3 0.01" pos="0 0 -0.01" friction="{mu} 0.005 0.0001"/></body>
        <body name="b" pos="{0.1 * math.sin(a) * 0:.3f} 0 0"><freejoint name="b"/><geom name="obj" class="obj" type="box" size="0.05 0.05 0.03" mass="0.5" friction="{mu} 0.005 0.0001"/></body>"""
        m = scene(body, p)
        d = mujoco.MjData(m)
        d.qpos[:3] = [0.03 * math.sin(a), 0, 0.03 * math.cos(a) + 0.0005]
        d.qpos[3:7] = [math.cos(a / 2), 0, math.sin(a / 2), 0]
        run(m, d, 0.3)
        p0 = d.qpos[:3].copy()
        run(m, d, 1.0)
        v = float(np.linalg.norm(d.qpos[:3] - p0))
        if v > 0.02 and res is None:
            res = deg
        if v <= 0.02:
            creep = v
    return res, creep


def tip_force(h, a, mass=20.0, p=None):
    """Box pushed at height h, half-width a: tips at F = m g a / h. Binary search on a ramped force."""
    body = f"""<geom name="floor" class="world" type="box" size="3 3 0.05" pos="0 0 -0.05" friction="1.5 0.005 0.0001"/>
    <body name="b" pos="0 0 {h + 0.1}"><freejoint name="b"/><geom name="obj" class="rob" type="box" size="{a} 0.2 {h + 0.1}" mass="{mass}" friction="1.5 0.005 0.0001"/></body>"""
    m = scene(body, p)
    F = None
    d = mujoco.MjData(m)
    run(m, d, 0.3)
    b = 1
    F0 = mass * G * a / h
    for i in range(int(20 / m.opt.timestep)):
        Ft = F0 * 0.7 + F0 * 0.03 * i * m.opt.timestep               # slow ramp: tipping takes time to show
        d.xfrc_applied[b, 0] = Ft
        d.xfrc_applied[b, 4] = Ft * (h - (h + 0.1)) * 1.0           # torque so the force acts at height h
        mujoco.mj_step(m, d)
        if d.xmat[b].reshape(3, 3)[2, 2] < math.cos(math.radians(0.5)):
            F = Ft
            break
    return F, mass * G * a / h


def pinch_hold(mu, N, p=None):
    """Object squeezed between two pads with force N each side: holds up to 2 mu N. Returns the load at which it slips."""
    body = f"""<body name="padl" pos="0 0.021 0.5" ><joint name="sl" type="slide" axis="0 1 0" damping="50"/><geom name="pad_a" class="pad" type="box" size="0.02 0.002 0.02" mass="0.1"/></body>
    <body name="padr" pos="0 -0.021 0.5"><joint name="sr" type="slide" axis="0 1 0" damping="50"/><geom name="pad_b" class="pad" type="box" size="0.02 0.002 0.02" mass="0.1"/></body>
    <body name="o" pos="0 0 0.5"><freejoint name="o"/><geom name="obj" class="obj" type="box" size="0.05 0.019 0.05" mass="0.1"/></body>"""
    extra = f"""<contact><pair geom1="pad_a" geom2="obj" condim="4" friction="{mu} {mu} 0.005 0.0001 0.0001" solref="{(p or PHYS).get('solref', PHYS['solref']) * 2} 1" solimp="0.9 0.99 0.001 0.5 2"/>
    <pair geom1="pad_b" geom2="obj" condim="4" friction="{mu} {mu} 0.005 0.0001 0.0001" solref="{(p or PHYS).get('solref', PHYS['solref']) * 2} 1" solimp="0.9 0.99 0.001 0.5 2"/></contact>"""
    m = scene(body, p, extra)
    d = mujoco.MjData(m)
    bo = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "o")
    bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "padl")
    br = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "padr")
    z0 = None
    slip_at, creep = None, None
    dt = m.opt.timestep
    for i in range(int(14 / dt)):
        t = i * dt
        d.xfrc_applied[bl, 1] = -N
        d.xfrc_applied[br, 1] = N
        d.xfrc_applied[bl, 2] = 0.1 * G
        d.xfrc_applied[br, 2] = 0.1 * G
        L = max(0.0, (t - 1.0)) * 2 * mu * N / 10.0                 # ramps through the limit in 10 s
        d.xfrc_applied[bo, 2] = -L
        mujoco.mj_step(m, d)
        if t > 0.9 and z0 is None:
            z0 = d.xpos[bo][2]
        if z0 is not None and z0 - d.xpos[bo][2] > 0.005 and slip_at is None:
            slip_at = L + 0.1 * G
            break
    return slip_at, 2 * mu * N


def step_climb(r, h, W=30.0, Fmax=None, p=None):
    """Rigid wheel of radius r, axle load W, pushed at its axle against a step h: climbs at F = W tan(a), cos(a) = 1 - h/r."""
    a = math.acos(1 - h / r)
    want = W * math.tan(a)
    body = f"""<geom name="floor" class="world" type="box" size="3 1 0.05" pos="0 0 -0.05" friction="0.8 0.005 0.0001"/>
    <geom name="step" class="world" type="box" size="1 1 {h / 2}" pos="1.2 0 {h / 2}" friction="0.8 0.005 0.0001"/>
    <body name="ax" pos="{0.2 - r - 0.02 + 0.0} 0 {r}"><joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="{W / G}" diaginertia="0.01 0.01 0.01"/>
      <body name="w"><joint name="spin" type="hinge" axis="0 1 0" damping="0.0005"/><geom name="wheel" class="rob" contype="2" conaffinity="1" condim="3" type="ellipsoid" size="{r} 0.01 {r}" mass="0.05" friction="0.8 0.005 0.0001" solref="0.01 1"/></body>
    </body>"""
    m = scene(body, p)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    F_climb = None
    for i in range(int(12 / dt)):
        t = i * dt
        F = 1.0 if t < 1.0 else (t - 1.0) * want / 5.0              # rolls up to the step, then ramps past the textbook value
        d.xfrc_applied[1, 0] = F
        mujoco.mj_step(m, d)
        if d.qpos[1] > 0.3 * h:
            F_climb = F
            break
    return F_climb, want


def penetration(F=5.0, p=None):
    body = """<geom name="floor" class="world" type="box" size="3 3 0.05" pos="0 0 -0.05"/>
    <body name="b" pos="0 0 0.02"><freejoint name="b"/><geom name="obj" class="obj" type="box" size="0.02 0.02 0.02" mass="0.05"/></body>"""
    m = scene(body, p)
    d = mujoco.MjData(m)
    d.xfrc_applied[1, 2] = -F
    run(m, d, 1.0)
    return 0.02 - float(d.qpos[2])


def main():
    rows = []

    def row(name, got, want, tol, unit):
        ok = got is not None and abs(got - want) <= tol
        rows.append(dict(test=name, got=None if got is None else round(float(got), 3), want=round(float(want), 3), tol=tol, unit=unit, ok=bool(ok)))
        print(f"  {'ok ' if ok else 'BAD'} {name:46s} got {got if got is None else round(float(got), 3)!s:>8} want {want:7.3f} {unit}")

    print("control: friction pinch without noslip, as the 30 Oct model ran. Holds far less than 2 mu N:")
    s, want = pinch_hold(0.6, 40.0, dict(noslip=0, impratio=10))
    print(f"    mu 0.6: {s:.1f} N of {want:.0f} N")
    for label, p in (("dt 2 ms", None), ("dt 1 ms", dict(timestep=0.001)), ("dt 0.5 ms", dict(timestep=0.0005))):
        print(label)
        for mu in (0.25, 0.5):
            a, c = incline(mu, p)
            row(f"incline slips at atan(mu), mu {mu} [{label}]", a, math.degrees(math.atan(mu)), 1.5, "deg")
        F, want = tip_force(0.5, 0.15, p=p)
        row(f"tips at m g a / h [{label}]", F, want, want * 0.06, "N")
        for mu, N in ((0.6, 40.0), (0.3, 40.0)):
            s, want = pinch_hold(mu, N, p)
            row(f"pinch holds 2 mu N, mu {mu} [{label}]", s, want, want * 0.12, "N")
        for r, h in ((0.0375, 0.015), (0.05, 0.015), (0.0825, 0.015)):
            F, want = step_climb(r, h, p=p)
            row(f"wheel r {int(r * 1000)} mm climbs {int(h * 1000)} mm at W tan a [{label}]", F, want, want * 0.20, "N")
        row(f"penetration under 5 N [{label}]", penetration(5.0, p) * 1000, 0.0, 0.5, "mm")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physics_validation.json").write_text(json.dumps(rows, indent=1))
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows) - len(bad)}/{len(rows)} within tolerance")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
