"""Drag onto the tray vs lift and carry: same robot, same grasp, same errors (27 Oct 2026).

uv run python sim/transfer_compare.py [N]       # N trials per cell (default 40) -> out/transfer_compare.txt

The research question: can tray-supported transfer (slide the object onto the drawer; the shelf and the drawer
carry the weight) match lifting it (the grip carries the weight) with cheap hardware? This isolates the physics
part of that question. Everything is the same in both strategies except whether the lift rises before the pull:

  drag:  grip the object low (pads ~35 mm above the shelf), pull it across the 14 mm gap onto the drawer (set
         6 mm below the shelf), let go.
  lift:  the same grip, then the wrist pitches up 6 deg (the object rises ~25 mm relative to the drawer, which rides
         on the same carriage), the lift rises 25 mm only when there is a lip; carried over the drawer, set down
         by pitching back, let go.

Objects (upright cylinders, the arm's right hand): a can (0.35 kg), a 0.5 l bottle (0.55 kg), a 1 l bottle
(1.05 kg), a 1.5 kg bottle or jar. Squeeze per finger: 10 N or 25 N (the CAP gripper's real squeeze isn't
published; its XL330 servo suggests the low end).

Randomised per trial, as in sim/robustness.py: object position (2-5 cm behind the shelf edge, +-3 cm sideways),
camera estimate (sigma 4 mm), lift calibration (+-4 mm), servo stiffness (50-100 %), shoulder/elbow zero offsets
(+-1 deg), mass (+-10 %), object-shelf friction 0.25-0.55, pad friction 0.5-1.0 (a cold bottle is often wet).
Success: upright (< 10 deg) and fully on the drawer after it has slid back in. Scripted, not learned.
"""
from __future__ import annotations

import copy
import math
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# render/test_videos.py sets this to {"fps", "next", "frames", "grab": fn(m, d) -> image} to record one trial
VIDEO = None
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
import params as P  # noqa: E402
from demo import FOLD, JOINTS, SHELF, TIP, Script, arm, init_extra, set_ctrl, to_ctrl  # noqa: E402

SIM, OUT = ROOT / "sim", ROOT / "out"
SHELF_EDGE = 0.640                    # shelf front edge, world x (from the model)
DRAWER_MID_X = 0.50                   # where the object should end up on the extended drawer
Y0 = -0.04                            # nominal object position sideways (right arm)
OBJECTS = {                           # radius m, height m, mass kg
    "can 0.35 kg": (0.033, 0.115, 0.35),
    "0.5 l bottle": (0.033, 0.19, 0.55),
    "1 l bottle": (0.038, 0.24, 1.05),
    "1.5 kg bottle": (0.038, 0.26, 1.50),
}
OPEN = 0.045                          # widest finger opening (per side, joint units)
TRACE = False
MODES = ("drag", "tongue", "lift")


# the "shovel": a 0.6 mm spring-steel tongue, 40 mm long, ground to a feather edge that lies on the shelf. Modelled as
# a 1.6 deg wedge whose tip top sits 0.3 mm below the shelf surface; it touches only the object, not the shelf.
TONGUE_LEN, TONGUE_T, TONGUE_DEG = 0.040, 0.0006, 1.6
TONGUE_RISE = TONGUE_LEN * math.sin(math.radians(TONGUE_DEG)) - 0.0003    # drawer top above the shelf: ~0.8 mm
_CACHE = {}


def build(obj, lip, tongue=False):
    key = (obj, lip, tongue)
    if key not in _CACHE:
        _CACHE[key] = _build(obj, lip, tongue)
    return copy.deepcopy(_CACHE[key])


def _build(obj, lip, tongue):
    spec = mujoco.MjSpec.from_file(str(SIM / "tischlein.xml"))
    if tongue:                        # the "shovel": a thin tongue on the drawer front that lies on the shelf
        top = next(g for g in spec.body("drawer").geoms if abs(g.size[0] - 0.125) < 1e-3 and abs(g.size[2] - 0.015) < 1e-3)
        a = math.radians(TONGUE_DEG)
        cx = top.pos[0] + top.size[0] + TONGUE_LEN / 2 * math.cos(a) - 0.001
        cz = top.pos[2] + top.size[2] - TONGUE_T / 2 - TONGUE_LEN / 2 * math.sin(a)
        spec.body("drawer").add_geom(name="tongue", type=mujoco.mjtGeom.mjGEOM_BOX,
                                     size=[TONGUE_LEN / 2, 0.11, TONGUE_T / 2], pos=[cx, 0, cz],
                                     quat=[math.cos(a / 2), 0, math.sin(a / 2), 0], rgba=[0.6, 0.6, 0.65, 1],
                                     contype=2, conaffinity=2)
    r, h, mass = OBJECTS[obj]
    b = spec.body("bottle")
    b.mass = mass
    b.ipos = [0, 0, h / 2]
    b.inertia = [mass * (3 * r * r + h * h) / 12, mass * (3 * r * r + h * h) / 12, mass * r * r / 2]
    g_body, g_cap = b.geoms[0], b.geoms[1]
    g_body.contype = g_body.conaffinity = 3        # also touches the tongue (collision group 2)
    g_body.size = [r, h / 2, 0]
    g_body.pos = [0, 0, h / 2]
    g_cap.pos = [0, 0, h + 0.01]
    g_cap.size = [0.012, 0.01, 0]
    g_cap.contype = 0
    g_cap.conaffinity = 0
    if lip:                           # a 20 mm rail along the shelf front, like a door rack
        spec.worldbody.add_geom(name="lip", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.002, 0.25, 0.010],
                                pos=[SHELF_EDGE + 0.002, 0, SHELF + 0.010], rgba=[0.8, 0.8, 0.8, 1])
    m = spec.compile()
    g = m.geom("shelf1").id                   # the tall bottle compartment: no shelf right above this one
    m.geom_contype[g] = m.geom_conaffinity[g] = 0
    m.geom_rgba[g][3] = 0.0
    return m


def script(c_est, lift_err, lift_by, mode, lip=False):
    """c_est: estimated object centre (x, y). The pads close on it from the robot side, hand pointing +x.
    drag: the drawer sits 6 mm below the shelf (so +-4 mm calibration error never leaves a step up).
    lift: the drawer rides on the same carriage, so raising the lift gives no clearance over it; the wrist pitch
          does: 6 deg up raises the object ~25 mm relative to the drawer. The lift rises only to clear a lip.
          Over the drawer the wrist pitches back down to set the object on it before letting go."""
    pad = c_est
    grasp = (pad[0] - TIP, pad[1])
    pre = (grasp[0] - 0.09, grasp[1])
    if lip:                                                   # pads must clear the 20 mm rail on the way in
        z0 = SHELF + 0.008 + lift_err
    elif mode == "tongue":                                    # lowered until the tongue touches the shelf
        z0 = SHELF + TONGUE_RISE + lift_err
    else:
        z0 = SHELF + lift_err - (0.006 if mode == "drag" else 0.002)
    S = Script(dict(base_x=0.0, base_y=0.0, base_yaw=0.0, lift=z0, drawer=P.DRAWER_STROKE / 1000, **FOLD))
    S.wait(0.3)
    S.move(2.0, **arm(*pre, heading=0.0, grip=OPEN))
    S.cart(1.5, pre, grasp, heading=0.0, grip_l=OPEN)
    S.move(0.8, grip_l=-0.010)                                # close until the squeeze limit
    pitch = 0.0
    if mode == "lift":
        pitch = -6.0                                          # negative = up
        S.move(1.2, lift=z0 + lift_by, j4=pitch)
    end = (grasp[0] - (pad[0] - DRAWER_MID_X), grasp[1])
    S.cart(3.0, grasp, end, heading=0.0, grip_l=-0.010, j4=pitch)
    if mode == "lift":
        S.move(1.0, j4=1.0)                                   # set it down (a little past level: the wrist sags)
    S.move(0.6, grip_l=OPEN)                                  # let go
    S.move(0.4, j4=0.0)
    back = (end[0] - 0.10, end[1])
    S.cart(1.0, end, back, heading=0.0, grip_l=OPEN)          # fingers slide back off the object
    side = (back[0], 0.20)
    S.cart(1.5, back, side, heading=0.0, grip_l=0.004)
    if mode == "lift" and lift_by:
        S.move(1.0, lift=z0)
    S.move(2.5, drawer=0.0)
    S.wait(1.0)
    return S


def trial(args):
    obj, mode, squeeze, lip, i = args
    rng = np.random.default_rng(zlib.crc32(f"{obj}|{lip}|{i}".encode()))   # same world for both strategies
    m = build(obj, lip, tongue=(mode == "tongue"))
    d = mujoco.MjData(m)
    r, h, mass0 = OBJECTS[obj]
    bid = m.body("bottle").id
    mass = mass0 * rng.uniform(0.9, 1.1)
    m.body_mass[bid] = mass
    mu_obj, mu_pad = rng.uniform(0.25, 0.55), rng.uniform(0.5, 1.0)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid:
            m.geom_friction[g][0] = mu_obj
        elif m.geom(g).name.startswith("shelf") or m.geom(g).name == "lip" or m.geom_bodyid[g] == m.body("drawer").id:
            m.geom_friction[g][0] = 0.2                       # contact uses the larger: object-shelf = mu_obj
        elif m.geom_bodyid[g] in (m.body("finger_l").id, m.body("finger_r").id):
            m.geom_friction[g][0] = mu_pad
    ga = m.actuator("grip").id
    m.actuator_forcerange[ga] = [-2 * squeeze, 2 * squeeze]  # coupled fingers: each squeezes with half the force
    stiff = rng.uniform(0.5, 1.0)
    for name in ("j1", "j2", "j3", "j4"):
        a = m.actuator(name).id
        m.actuator_gainprm[a][0] *= stiff
        m.actuator_biasprm[a][1] *= stiff
    off = np.zeros(len(JOINTS))
    off[JOINTS.index("j1")] = math.radians(rng.uniform(-1, 1))
    off[JOINTS.index("j2")] = math.radians(rng.uniform(-1, 1))
    lift_err = rng.uniform(-0.004, 0.004)
    if mode == "tongue":
        lift_err *= 0.125                                     # set by touch: +-0.5 mm
    cx = SHELF_EDGE + r + rng.uniform(0.02, 0.05)
    cy = Y0 + rng.uniform(-0.03, 0.03)
    c_est = (cx + rng.normal(0, 0.004), cy + rng.normal(0, 0.004))
    S = script(c_est, lift_err, 0.025 if lip else 0.0, mode, lip)
    jid = {j: m.joint(j).id for j in JOINTS}
    q0 = to_ctrl(S.at(0.0))
    for j, v in zip(JOINTS, q0):
        d.qpos[m.jnt_qposadr[jid[j]]] = v
    d.qpos[m.jnt_qposadr[m.joint("grip_r").id]] = q0[-1]
    qa = m.jnt_qposadr[m.joint(m.body_jntadr[bid]).id]
    d.qpos[qa:qa + 3] = (cx, cy, SHELF + 0.0005)
    d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    qb = m.jnt_qposadr[m.joint(m.body_jntadr[m.body("meal_box").id]).id]
    d.qpos[qb:qb + 3] = (-3.0, 3.0, 0.05)                      # the meal box sits where the bottle goes: move it away
    set_ctrl(d, q0)
    init_extra(m, d)
    mujoco.mj_forward(m, d)
    j4a = m.actuator("j4").id
    tau4, moved, slipped = 0.0, False, False
    while d.time < S.t:
        set_ctrl(d, to_ctrl(S.at(d.time)) + off)
        mujoco.mj_step(m, d)
        tau4 = max(tau4, abs(d.actuator_force[j4a]))
        if VIDEO is not None and d.time >= VIDEO["next"]:
            VIDEO["next"] += 1.0 / VIDEO["fps"]
            VIDEO["frames"].append(VIDEO["grab"](m, d))
        if TRACE and int(d.time / m.opt.timestep) % 500 == 0:
            zz = d.xmat[bid].reshape(3, 3)[:, 2]
            cons = sorted({m.body(m.geom_bodyid[c.geom1] if m.geom_bodyid[c.geom2] == bid else m.geom_bodyid[c.geom2]).name
                           for c in d.contact[:d.ncon] if bid in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])})
            print(f"{d.time:5.2f} obj {np.round(d.xpos[bid], 3)} tilt {math.degrees(math.acos(min(1, zz[2]))):5.1f} "
                  f"grip {d.qpos[m.jnt_qposadr[m.joint('grip_l').id]]:.3f} {cons}")
        if not moved and d.xpos[bid][0] < cx - 0.03:
            moved = True
        if not np.all(np.isfinite(d.qpos)):
            return dict(ok=False, why="sim unstable", obj=obj, mode=mode, squeeze=squeeze, lip=lip, tau4=tau4)
    zax = d.xmat[bid].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(max(-1, min(1, zax[2]))))
    dr = m.body("drawer").id
    p = d.xmat[dr].reshape(3, 3).T @ (d.xpos[bid] - d.xpos[dr])
    tray_x0, tray_x1 = P.TRAY_X0 / 1000, (P.TRAY_X0 + P.TRAY_DEPTH) / 1000
    on = (tray_x0 + r - 0.003 <= p[0] <= tray_x1 - r + 0.006) and abs(p[1]) <= P.TRAY_WIDTH / 2000 - r + 0.003 \
        and abs(p[2] - 0.015) < 0.03
    ok = on and tilt < 10
    if ok:
        why = "ok"
    elif not moved:
        why = "never moved (missed or slipped at once)"
    elif tilt >= 10:
        why = "tipped over"
    else:
        why = "dropped / left behind"
    return dict(ok=ok, why=why, obj=obj, mode=mode, squeeze=squeeze, lip=lip, tau4=tau4)


if __name__ == "__main__":
    import multiprocessing as mp
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 40
    jobs = [(o, mode, sq, False, i) for o in OBJECTS for mode in MODES for sq in (10, 25) for i in range(N)]
    jobs += [("0.5 l bottle", mode, sq, True, i) for mode in MODES for sq in (10, 25) for i in range(N)]
    with mp.get_context("spawn").Pool(6) as pool:
        res = pool.map(trial, jobs, chunksize=4)
    cells = defaultdict(list)
    for r in res:
        cells[(r["obj"] + (" behind a 20 mm lip" if r["lip"] else ""), r["squeeze"], r["mode"])].append(r)
    lines = [f"Drag onto the drawer vs lift and carry (sim/transfer_compare.py), {N} trials per cell, scripted",
             "", f"{'object':34s}{'squeeze/finger':>15s}{'drag':>8s}{'drag+tongue':>13s}{'lift':>8s}"
             f"   wrist pitch peak, drag / tongue / lift (STS3250: rated 1.6, peak 4.9 N.m)"]
    keys = sorted({(k[0], k[1]) for k in cells}, key=lambda k: (list(OBJECTS).index(k[0].split(' behind')[0]) + (10 if 'lip' in k[0] else 0), k[1]))
    for o, sq in keys:
        cs = [cells[(o, sq, md)] for md in MODES]
        rate = [100 * sum(r["ok"] for r in c) / len(c) for c in cs]
        tau = [np.median([r["tau4"] for r in c]) for c in cs]
        lines.append(f"{o:34s}{sq:>12d} N{rate[0]:7.0f}%{rate[1]:12.0f}%{rate[2]:7.0f}%   "
                     + " / ".join(f"{t:.1f}" for t in tau) + " N.m")
    lines += ["", "Failure modes:"]
    for k in sorted(cells, key=str):
        c = Counter(r["why"] for r in cells[k] if not r["ok"])
        if c:
            lines.append(f"  {k[0]}, {k[1]} N, {k[2]}: " + ", ".join(f"{n}x {w}" for w, n in c.most_common()))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "transfer_compare.txt").write_text(txt + "\n")
