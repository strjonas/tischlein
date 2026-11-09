"""Monte Carlo robustness of the fridge -> drawer transfer.

uv run python sim/robustness.py [N]

Each trial randomises what will be wrong in reality:
  box placement on the shelf (x +-25 mm, y +-30 mm, yaw +-8 deg), mass 0.2-2.0 kg, friction 0.25-0.55,
  camera pose estimate error (sigma 4 mm, 2 deg), tray height calibration (+-4 mm),
  servo stiffness (50-100 % of nominal gain) and joint zero offsets on J1/J2 (+-1 deg).
The controller only sees the *estimated* box pose. Success = box fully on the retracted tray.
This is still open-loop after the grasp: no force sensing, no re-grasp. It tests the mechanism and
the tolerance of the motion, not perception or learning.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from collections import Counter

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
import params as P  # noqa: E402
from demo import FOLD, JOINTS, SHELF, TIP, Script, arm, init_extra, set_ctrl, to_ctrl  # noqa: E402

SIM = ROOT / "sim"
FX = 0.72 - P.AXLE_SHIFT / 1000
BOX_FACE0 = FX + 0.058
PAD_HALF = 0.030
TRAY_X1 = (P.TRAY_X0 + P.TRAY_DEPTH) / 1000


# where the box's front face should end up (far end 10 mm inside the drawer front). v0 value 0.556 m; the frame
# moved back with the v0.2 axle shift, and until 26 Oct this constant didn't, so the pull stopped 12 cm short.
FINAL_FACE_X = 0.556 - P.AXLE_SHIFT / 1000


def script(knob_xy_est, yaw_est, lift_err):
    """Grip the knob with the last ~15 mm of the pads, pull straight back to an absolute target."""
    h = yaw_est                                  # hand heading follows the estimated box yaw
    c, s = math.cos(math.radians(h)), math.sin(math.radians(h))
    face = (knob_xy_est[0] + 0.009 * c, knob_xy_est[1] + 0.009 * s)
    pad = (face[0] - (PAD_HALF + 0.003) * c, face[1] - (PAD_HALF + 0.003) * s)
    start = dict(base_x=0.0, base_y=0.0, base_yaw=0.0, lift=SHELF - 0.002 + lift_err,
                 drawer=P.DRAWER_STROKE / 1000, **FOLD)
    S = Script(start)
    S.wait(0.3)
    pre = (pad[0] - 0.03 * c - TIP * c, pad[1] - 0.03 * s - TIP * s)
    grasp = (pad[0] - TIP * c, pad[1] - TIP * s)
    S.move(2.0, **arm(*pre, heading=h, grip=0.020))
    S.cart(1.2, pre, grasp, heading=h, grip_l=0.020)
    S.move(0.8, grip_l=-0.006)
    pull = face[0] - FINAL_FACE_X
    end = (grasp[0] - pull, grasp[1])
    S.cart(4.0, grasp, end, heading=h, grip_l=-0.006)
    S.move(0.6, grip_l=0.020)
    back = (end[0] - 0.025 * c, end[1] - 0.025 * s)
    S.cart(0.6, end, back, heading=h, grip_l=0.020)
    S.back, S.h = back, h
    return S


def regrasp_script(state, back, knob_xy_est, yaw_est, extra):
    """Second try: grip the knob where the camera now sees it and pull `extra` metres further."""
    h = yaw_est
    c, s = math.cos(math.radians(h)), math.sin(math.radians(h))
    face = (knob_xy_est[0] + 0.009 * c, knob_xy_est[1] + 0.009 * s)
    pad = (face[0] - (PAD_HALF + 0.003) * c, face[1] - (PAD_HALF + 0.003) * s)
    S = Script(state)
    grasp = (pad[0] - TIP * c, pad[1] - TIP * s)
    pre = (grasp[0] - 0.03 * c, grasp[1] - 0.03 * s)
    S.cart(0.8, back, pre, heading=h, grip_l=0.020)
    S.cart(0.8, pre, grasp, heading=h, grip_l=0.020)
    S.move(0.8, grip_l=-0.006)
    end = (grasp[0] - extra, grasp[1])
    S.cart(1.5, grasp, end, heading=h, grip_l=-0.006)
    S.move(0.6, grip_l=0.020)
    nb = (end[0] - 0.025 * c, end[1] - 0.025 * s)
    S.cart(0.6, end, nb, heading=h, grip_l=0.020)
    S.back, S.h = nb, h
    return S


def finish_script(state, back, h):
    """v0.2 (long CAP hand): after the pull the wrist sits beside its own shoulder, so the hand leaves to the left
    (away from the shoulder), fingers closed and tilted up 5 deg, as in sim/demo.py; then the drawer comes in."""
    S = Script(state)
    S.move(0.6, grip_l=0.004, j4=-5.0)
    # far enough left that neither the hand nor its iPhone (11.5 cm to the hand's right, above the box's top edge
    # height) is in the box's way when the drawer comes in: boxes sit up to 3 cm left of centre
    side = (back[0], 0.20)
    S.cart(2.0, back, side, heading=0.0, grip_l=0.004, j4=-5.0)
    S.move(2.5, drawer=0.0)
    S.wait(1.0)
    return S


def trial(rng, m_base, closed_loop=True):
    m = m_base
    d = mujoco.MjData(m)
    # --- randomise the world
    dx, dy, dyaw = rng.uniform(-0.0, 0.025), rng.uniform(-0.03, 0.03), rng.uniform(-8, 8)
    mass, mu = rng.uniform(0.2, 2.0), rng.uniform(0.25, 0.55)
    bid = m.body("meal_box").id
    m.body_mass[bid] = mass
    m.body_inertia[bid] = np.array([0.0013, 0.0024, 0.0030]) * mass / 0.8
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_contype[g]:
            m.geom_friction[g][0] = mu if m.geom_size[g][0] > 0.02 else 1.0
        if m.geom(g).name.startswith("shelf"):
            m.geom_friction[g][0] = mu
    stiff = rng.uniform(0.5, 1.0)
    for name in ("j1", "j2", "j3", "j4"):
        a = m.actuator(name).id
        m.actuator_gainprm[a][0] = m_base_gain[name] * stiff
        m.actuator_biasprm[a][1] = -m_base_gain[name] * stiff
    zero = {"j1": rng.uniform(-1, 1), "j2": rng.uniform(-1, 1)}
    lift_err = rng.uniform(-0.004, 0.004)
    # true box pose (front face centre); knob centre is 9 mm in front of the face
    bx, by = BOX_FACE0 + dx, dy
    c, s = math.cos(math.radians(dyaw)), math.sin(math.radians(dyaw))
    knob_true = (bx - 0.009 * c, by - 0.009 * s)
    knob_est = (knob_true[0] + rng.normal(0, 0.004), knob_true[1] + rng.normal(0, 0.004))
    yaw_est = dyaw + rng.normal(0, 2.0)
    S = script(knob_est, yaw_est, lift_err)
    # --- initial state
    jid = {j: m.joint(j).id for j in JOINTS}
    q0 = to_ctrl(S.at(0.0))
    for j, v in zip(JOINTS, q0):
        d.qpos[m.jnt_qposadr[jid[j]]] = v
    d.qpos[m.jnt_qposadr[m.joint("grip_r").id]] = q0[-1]
    qa = m.jnt_qposadr[m.joint(m.body_jntadr[bid]).id]
    # body frame origin is the front-face centre at shelf height; rotate about it
    d.qpos[qa:qa + 3] = (bx, by, SHELF)
    d.qpos[qa + 3:qa + 7] = (math.cos(math.radians(dyaw) / 2), 0, 0, math.sin(math.radians(dyaw) / 2))
    set_ctrl(d, q0)
    init_extra(m, d)
    mujoco.mj_forward(m, d)
    off = np.zeros(len(JOINTS))
    off[JOINTS.index("j1")] = math.radians(zero["j1"])
    off[JOINTS.index("j2")] = math.radians(zero["j2"])
    moved = False

    def play(Sx):
        nonlocal moved
        t0 = d.time
        while d.time - t0 < Sx.t:
            set_ctrl(d, to_ctrl(Sx.at(d.time - t0)) + off)
            mujoco.mj_step(m, d)
            if not moved and d.xpos[bid][0] < bx - 0.05:
                moved = True
            if not np.all(np.isfinite(d.qpos)):
                return False
        return True

    if not play(S):
        return dict(ok=False, mode="sim unstable", mass=mass, mu=mu)
    state, back, h = dict(S.state), S.back, S.h
    retries = 0
    while closed_loop and retries < 2:
        # wrist camera looks at the box: far end vs drawer front, knob pose (with the same noise model)
        xm = d.xmat[bid].reshape(3, 3)
        far = max((d.xpos[bid] + xm @ np.array([0.18, cy, 0.0]))[0] for cy in (-0.06, 0.06))
        front = d.xpos[m.body("drawer").id][0] + TRAY_X1
        overhang = far + rng.normal(0, 0.004) - (front - 0.008)
        if overhang <= 0 or not moved:
            break
        yaw_now = math.degrees(math.atan2(xm[1, 0], xm[0, 0]))
        kc = d.xpos[bid] + xm @ np.array([-0.009, 0.0, 0.0])
        k_est = (kc[0] + rng.normal(0, 0.004), kc[1] + rng.normal(0, 0.004))
        S2 = regrasp_script(state, back, k_est, yaw_now + rng.normal(0, 2.0), overhang + 0.006)
        if not play(S2):
            return dict(ok=False, mode="sim unstable", mass=mass, mu=mu)
        state, back, h = dict(S2.state), S2.back, S2.h
        retries += 1
    if not play(finish_script(state, back, h)):
        return dict(ok=False, mode="sim unstable", mass=mass, mu=mu)
    # evaluate in drawer frame
    dr = m.body("drawer").id
    R = d.xmat[dr].reshape(3, 3)
    xmat = d.xmat[bid].reshape(3, 3)
    corners = []
    for cx in (0.0, 0.18):
        for cy in (-0.06, 0.06):
            p = d.xpos[bid] + xmat @ np.array([cx, cy, 0.0])
            corners.append(R.T @ (p - d.xpos[dr]))
    corners = np.array(corners)
    inside = (corners[:, 0].min() >= P.TRAY_X0 / 1000 - 0.002 and corners[:, 0].max() <= TRAY_X1 + 0.004
              and abs(corners[:, 1]).max() <= P.TRAY_WIDTH / 2000 + 0.002)
    flat = abs(corners[:, 2]).max() < 0.012
    if inside and flat:
        mode = "ok"
    elif not moved:
        mode = "missed grasp (box never moved)"
    elif not flat:
        mode = "box tilted / on the lip"
    else:
        mode = "box not fully on tray"
    return dict(ok=inside and flat, mode=mode, mass=mass, mu=mu, dyaw=dyaw, dy=dy, dx=dx, stiff=stiff, retries=retries)


def _worker(args):
    """One trial in its own process, seeded by its index (so results don't depend on the number of workers)."""
    global m_base_gain
    i, closed = args
    m = mujoco.MjModel.from_xml_path(str(SIM / "tischlein.xml"))
    m_base_gain = {n: float(m.actuator_gainprm[m.actuator(n).id][0]) for n in ("j1", "j2", "j3", "j4")}
    return trial(np.random.default_rng(7000 + i), m, closed_loop=closed)


if __name__ == "__main__":
    import multiprocessing as mp
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    closed = "--open-loop" not in sys.argv
    with mp.get_context("spawn").Pool(6) as pool:
        res = pool.map(_worker, [(i, closed) for i in range(N)])
    ok = sum(r["ok"] for r in res)
    print(f"mode: {'closed loop (camera check + up to 2 re-grasps)' if closed else 'open loop'}")
    print(f"re-grasps used: {Counter(r.get('retries', 0) for r in res)}")
    print(f"{ok}/{N} transfers succeeded ({100*ok/N:.0f}%)")
    for mode, k in Counter(r["mode"] for r in res).most_common():
        print(f"  {k:3d}  {mode}")
    fails = [r for r in res if not r["ok"]]
    if fails:
        print("failures (mass kg, mu, yaw deg, dy mm, dx mm, stiffness):")
        for r in fails[:15]:
            print(f"  {r['mass']:.2f} {r['mu']:.2f} {r.get('dyaw', 0):+5.1f} {1000*r.get('dy', 0):+5.0f} {1000*r.get('dx', 0):+4.0f} {r.get('stiff', 0):.2f}  {r['mode']}")
    heavy = [r for r in res if r["mass"] > 1.5]
    if heavy:
        print(f"boxes > 1.5 kg: {sum(r['ok'] for r in heavy)}/{len(heavy)}")
