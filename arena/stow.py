"""A collision-free travelling pose: prefer narrow width as well as turning radius.

Radius alone is rotationally indifferent and used to park the reacher sideways,
making an otherwise narrow differential-drive robot unable to enter a doorway.
"""
from __future__ import annotations

import mujoco
import numpy as np
import hashlib
from dataclasses import asdict

from .assemble import header, CLS
from .robot import Robot, robot_mjcf

_cache = {}


def fit(r: Robot, n=1500):
    key = hashlib.sha256(repr(asdict(r)).encode()).hexdigest()
    if key in _cache:
        for a, h in zip(r.chain, _cache[key]):
            a.home = h
        return r
    xml = header() + '<geom name="floor" class="world" type="plane" size="5 5 0.1"/>\n' + robot_mjcf(r) + "\n  </worldbody>\n</mujoco>\n"
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    rob = np.array([n_.startswith(("rob_", "arm_", "pad_")) and not n_.startswith(("rob_tyre", "rob_cast")) for n_ in names])
    axes = []
    for a in r.chain:
        nm = [a.name] + [f"{a.name}{k}" for k in range(2, a.stages + 1)]
        axes.append([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, x)] for x in nm])
    rng = np.random.default_rng(0)
    lo = np.array([a.lo for a in r.chain])
    hi = np.array([a.hi for a in r.chain])
    best, bq = 1e9, None
    q_b = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base")]

    def radius():
        pts = []
        for g in np.nonzero(rob)[0]:
            c = d.geom_xpos[g]
            R = d.geom_xmat[g].reshape(3, 3)
            s = m.geom_size[g]
            s = s if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else np.array([s[0]] * 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append(c + R @ (np.array([sx, sy, 0]) * s))
        pts = np.array(pts)
        radius = float(np.max(np.hypot(*pts[:, :2].T)))
        width = float(pts[:, 1].max()-pts[:, 1].min())
        return radius + .75*width

    for it in range(n):
        q = lo + (hi - lo) * rng.uniform(0, 1, len(lo)) if it else np.array([a.home for a in r.chain])
        if it % 3 == 0 and it:                  # fully retracted slides are the usual answer
            q = np.where([a.kind == "slide" and a.name != "lift" for a in r.chain], lo, q)
        for k, a in enumerate(r.chain):
            if a.name == "lift":
                q[k] = min(max(0.60, a.lo), a.hi)
        d.qpos[:] = m.qpos0
        d.qpos[q_b:q_b + 3] = (0, 0, 0.002)
        for k, ax in enumerate(axes):
            for qa in ax:
                d.qpos[qa] = q[k] / len(ax)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        hit = False
        for c in range(d.ncon):
            g1, g2 = d.contact.geom1[c], d.contact.geom2[c]
            if rob[g1] and rob[g2] and d.contact.dist[c] < -0.0015:
                hit = True
                break
            if (rob[g1] != rob[g2]) and d.contact.dist[c] < 0.0 and "floor" in (names[g1], names[g2]):
                hit = True
                break
        if hit:
            continue
        rr = radius()
        if rr < best - 1e-4:
            best, bq = rr, q.copy()
    if bq is not None:
        for a, h in zip(r.chain, bq):
            a.home = float(h)
        _cache[key] = [float(x) for x in bq]
    else:
        r.stow_valid = False
    return r
