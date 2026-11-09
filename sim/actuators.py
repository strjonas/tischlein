"""Are cheap bus servos too jerky and imprecise? What fixes it?

uv run python sim/actuators.py          # -> out/actuators.txt (+ out/actuators_traces.npz for the figure)

Cheap Feetech STS servos are commonly described as jerky, of limited precision and without force feedback;
the usual remedies are a trajectory generator, measured joint state and current limits. This simulates the flat
arm's three yaw joints (shoulder J1, elbow J2, wrist yaw J3) with the effects behind that description:

  STS3215:  12 V DC motor behind a 1:345 gearbox (2.94 N.m stall, 45 rpm), magnetic encoder on the output
            shaft (4096 steps/rev), internal P-D position loop (full torque at a 20-step error), +-1 step dead
            band, static/Coulomb gearbox friction, no integral term (Feetech default I = 0).
            The gearbox backlash sits *inside* that loop (the encoder is after the gearbox), so the servo
            corrects it at rest. Its dynamic effect is hunting: in a first version of this script, which modelled
            the gap explicitly, the belt-driven wrist hunted +-0.5-0.7 deg at rest with default gains until the
            gains were halved (as LeRobot does: P 32 -> 16). That needs tuning on the bench; it is left out here.
  Stepper:  closed-loop NEMA23 (shoulder) / long NEMA17 (elbow), both on the carriage; stiffness = holding torque
            x 50 rotor teeth; 14-bit motor encoder, P-I-D, torque falling with speed, the driver's own ramp.
  Belt:     GT2 6 mm, stiffness from its glass cords (~3e5 N/m per span, two spans), plus lost motion at the
            joint *outside* every loop (pulley bore, set screw, printed parts): 0.1 deg total, an estimate.
  Spring:   a torsion spring or elastic band preloads a joint one way, so the lost motion never opens up while
            the load stays below the preload (a horizontal joint has no gravity to do it).

Each joint runs on its own with the arm's inertia about it (decoupled, fine at these speeds); the tip comes from
forward kinematics. Command modes:
  raw:      the goal jumps to each new target (policy at 10 Hz, or one jump for a long move); STS speed and
            acceleration registers at maximum. What most hobby setups do.
  planned:  a trajectory layer streams smooth (minimum-jerk) set points at 100 Hz under the real joint speed
            limits. On the robot: Ruckig or the like.

All parameters are datasheet values or estimates. A dial indicator on the real arm would measure them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import params as P  # noqa: E402

OUT = ROOT / "out"
DT = 1e-4
TICK = 2 * math.pi / 4096                       # STS3215 encoder step at the servo output
L1, L2, L3 = P.L1 / 1000, P.L2 / 1000, P.WRIST_TO_TIP / 1000

STS = dict(tau=P.STS3215_PEAK_NM, w0=P.STS3215_NOLOAD_RPM * 2 * math.pi / 60,
           J=0.004,                     # estimate: rotor ~3.4e-8 kg m2 x 345^2, at the output
           fc=0.10, fs=0.14,            # estimate: gearbox friction, N.m at the output
           K=P.STS3215_PEAK_NM / (20 * TICK),   # full torque at a 20-step error
           db=1)
# closed-loop steppers: holding torque N.m, rotor inertia kg m2; stiffness = holding torque x 50 rotor teeth
NEMA = {"17": (P.NEMA17_HOLD_NM, 5.4e-6), "17L": (0.59, 8.2e-6), "23": (1.26, 4.8e-5)}
STP = dict(fc=0.01, fs=0.012, enc=2 * math.pi / 16384, w_ref=1000 * 2 * math.pi / 60)


def belt_k(teeth, span=0.15, width=6):
    """Torsional stiffness at a joint pulley with `teeth` GT2 teeth, two belt spans. 3e5 N/m per 0.15 m span of
    6 mm belt is an estimate from the glass cords (+-2x); stiffness grows with width and with the pulley radius^2."""
    r = teeth * 0.002 / (2 * math.pi)
    return 2 * 3e5 * (width / 6) * (0.15 / span) * r * r


def link_inertia(payload):
    """Arm inertia about each joint at mid reach (point masses from params.M) + payload at the tip, kg m2."""
    m1, m2, mh = P.M["arm_L1"], P.M["arm_L2"], P.M["wrist_gripper"]
    return np.array([m1 * 0.15**2 + m2 * 0.30**2 + mh * 0.45**2 + payload * 0.62**2,
                     m2 * 0.11**2 + mh * 0.30**2 + payload * 0.45**2,
                     0.012 + payload * L3**2])


# per joint: (actuator, ratio, transmission stiffness at the joint N.m/rad, lost motion deg total, preload N.m)
# actuator "sts" = STS3215, "stp17"/"stp17L"/"stp23" = closed-loop NEMA17 / long NEMA17 / NEMA23
VARIANTS = {
    "STS direct on each joint": [("sts", 1, 3000.0, 0.05, 0.0)] * 3,
    "STS + belts (Tischlein plan)": [("sts", 3, belt_k(60), 0.1, 0.0), ("sts", 2, belt_k(40), 0.1, 0.0),
                                     ("sts", 1, 3000.0, 0.05, 0.0)],
    # same ratios, 30T->90T and 30T->60T pulleys, 9 mm belts (3.4x stiffer), springs against the lost motion
    "STS + stiffer belts + springs": [("sts", 3, belt_k(90, width=9), 0.1, 0.6), ("sts", 2, belt_k(60, width=9), 0.1, 0.4),
                                      ("sts", 1, 3000.0, 0.05, 0.15)],
    # both motors on the carriage, the elbow driven through the upper arm; 20T->100T, 9 mm belts
    "Closed-loop steppers at J1/J2": [("stp23", 5, belt_k(100, width=9), 0.1, 0.0),
                                      ("stp17L", 5, belt_k(100, span=0.22, width=9), 0.1, 0.0),
                                      ("sts", 1, 3000.0, 0.05, 0.0)],
}


# ------------------------------------------------------------------ kinematics
def fk(q):
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2) + L3 * np.cos(q1 + q2 + q3)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2) + L3 * np.sin(q1 + q2 + q3)
    return np.stack([x, y], -1)


def ik(p, heading=0.0):
    wx, wy = p[0] - L3 * math.cos(heading), p[1] - L3 * math.sin(heading)
    c2 = (wx * wx + wy * wy - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    q2 = math.acos(max(-1, min(1, c2)))
    q1 = math.atan2(wy, wx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return np.array([q1, q2, heading - q1 - q2])


def jac(q):
    J = np.zeros((2, 3))
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = 1e-6
        J[:, i] = (fk(q + dq) - fk(q - dq)) / 2e-6
    return J


P0 = np.array([0.36 + L3, -0.03])               # tip at mid reach, hand pointing forward
Q0 = ik(P0)


def minjerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


VMAX = np.radians([60.0, 90.0, 180.0])         # joint speeds the planner keeps under (as in sim/demo.py)


def plan_duration(qa, qb):
    return float(np.max(1.875 * np.abs(qb - qa) / VMAX))


# ------------------------------------------------------------------ joint simulator (3 lanes at once)
class Arm:
    """States in joint units: motor side qm/wm, link ql/wl. Motor -> transmission spring with lost motion -> link."""

    def __init__(self, variant, payload=0.0, fric_scale=None):
        spec = VARIANTS[variant]
        self.stp = np.array([s[0].startswith("stp") for s in spec])
        self.tau_stp = np.array([NEMA[s[0][3:]][0] if s[0].startswith("stp") else 1.0 for s in spec])
        J_stp = np.array([NEMA[s[0][3:]][1] if s[0].startswith("stp") else 1.0 for s in spec])
        N = np.array([float(s[1]) for s in spec])
        self.N = N
        self.k = np.array([s[2] for s in spec])
        self.gap = np.radians([s[3] / 2 for s in spec])
        self.pre = np.array([s[4] for s in spec])
        self.Jl = link_inertia(payload)
        self.Jm = np.where(self.stp, J_stp, STS["J"]) * N**2
        f = np.ones(3) if fric_scale is None else np.asarray(fric_scale)
        self.fc = np.where(self.stp, STP["fc"], STS["fc"]) * N * f
        self.fs = np.where(self.stp, STP["fs"], STS["fs"]) * N * f
        self.res = np.where(self.stp, STP["enc"], TICK) / N
        self.K = np.where(self.stp, 50 * self.tau_stp, STS["K"]) * N**2      # servo stiffness at the joint
        Jt = self.Jm + self.Jl
        self.D = 2 * 0.8 * np.sqrt(self.K * Jt)                              # well-damped servo
        self.c = 2 * 0.05 * np.sqrt(self.k * self.Jl)                        # belt damping, light
        self.Ki = np.where(self.stp, 5 * self.K, 0.0)                        # closed-loop stepper drivers have I
        self.integ = np.zeros(3)
        self.qm, self.wm, self.ql, self.wl = (np.zeros(3) for _ in range(4))
        self.g_int, self.v_int = None, np.zeros(3)

    def start(self, q):
        q = np.asarray(q, float)
        s = np.sign(self.pre)
        self.ql = q.copy()
        self.qm = q + s * self.gap + self.pre / self.k                       # springs already pressing
        self.wm[:] = 0
        self.wl[:] = 0
        self.integ[:] = 0
        self.g_int, self.v_int = None, np.zeros(3)

    def step(self, goal, tau_ext):
        # joint zeros are calibrated with the springs fitted, so the link lands on the goal at rest
        goal = goal + np.sign(self.pre) * self.gap + self.pre / self.k
        if self.g_int is None:
            self.g_int = goal.copy()
        dist = goal - self.g_int                                             # stepper driver's own ramp
        v_des = np.sign(dist) * np.minimum(3.0, np.sqrt(2 * 30.0 * np.abs(dist)))
        self.v_int += np.clip(v_des - self.v_int, -30 * DT, 30 * DT)
        self.g_int = self.g_int + self.v_int * DT
        goal = np.where(self.stp, self.g_int, goal)
        # transmission
        d = self.qm - self.ql
        s = np.sign(d)
        tau = np.where(np.abs(d) > self.gap, self.k * (d - s * self.gap) + self.c * (self.wm - self.wl), 0.0)
        # servo (encoder on the motor side, before the lost motion)
        e = goal - np.round(self.qm / self.res) * self.res
        e = np.where(~self.stp & (np.abs(e) <= STS["db"] * self.res * 1.0001), 0.0, e)
        imax = 0.3 * self.N * self.tau_stp / np.maximum(self.Ki, 1e-9)    # integral may add <= 30 % torque
        self.integ = np.clip(self.integ + e * DT, -imax, imax)
        tau_p = self.K * e + self.Ki * self.integ
        w_out = self.wm * self.N
        sts_lim = self.N * STS["tau"]
        tau_sts = sts_lim * np.clip((tau_p - self.D * self.wm) / sts_lim - w_out / STS["w0"], -1, 1)  # back-EMF
        stp_lim = self.N * self.tau_stp * np.maximum(0.3, 1 - np.abs(w_out) / STP["w_ref"])
        # the stepper's rotor is so light that its damping term must be integrated implicitly
        dirn0 = np.sign(np.where(self.wm != 0, self.wm, tau_p - tau))
        w_imp = (self.Jm * self.wm + DT * (tau_p - tau - self.fc * dirn0)) / (self.Jm + self.D * DT)
        tau_stp = tau_p - self.D * w_imp
        sat = np.abs(tau_stp) > stp_lim
        tau_m = np.where(self.stp, np.clip(tau_stp, -stp_lim, stp_lim), tau_sts)
        # motor side with static friction
        net = tau_m - tau
        stuck = (np.abs(self.wm) < 1e-3) & (np.abs(net) < self.fs)
        dirn = np.sign(np.where(self.wm != 0, self.wm, net))
        w_exp = self.wm + (net - self.fc * dirn) / self.Jm * DT
        w_new = np.where(self.stp & ~sat, w_imp, w_exp)
        self.wm = np.where(stuck, 0.0, w_new)
        # link: transmission, spring (pulls towards -), outside load, small bearing friction
        acc = (tau - self.pre + tau_ext - 0.02 * np.tanh(self.wl / 0.01) - 0.01 * self.wl) / self.Jl
        self.wl = self.wl + acc * DT
        self.qm = self.qm + self.wm * DT
        self.ql = self.ql + self.wl * DT


def run(variant, goals_fn, T, payload=0.0, force_fn=None, fric_scale=None, q_start=Q0):
    arm = Arm(variant, payload, fric_scale)
    arm.start(q_start)
    ts, qs = [], []
    for k in range(int(T / DT)):
        t = k * DT
        f = np.zeros(3) if force_fn is None else jac(arm.ql).T @ force_fn(t)
        arm.step(goals_fn(t), f)
        if k % 10 == 0:
            ts.append(t)
            qs.append(arm.ql.copy())
    return np.array(ts), np.array(qs)


def tip_acc(ts, p):
    v = np.gradient(p, ts, axis=0)
    a = np.gradient(v, ts, axis=0)
    w = np.ones(10) / 10                                          # 10 ms average: what a hand or a cup feels
    return np.array([np.convolve(a[:, i], w, mode="same") for i in range(2)]).T


# ------------------------------------------------------------------ scenarios
def s1_slow_align(variant, mode):
    """Line up on a knob: 2 cm sideways at 1 cm/s, new policy target every 0.1 s (1 mm apart)."""
    pa, pb = P0, P0 + np.array([0.0, 0.02])

    def target(t):
        return pa + (pb - pa) * np.clip((t - 0.2) / 2.0, 0, 1)

    if mode == "raw":
        def g(t):
            return ik(target(math.floor(t * 10) / 10 + 0.1))
    else:                                  # a policy chunk holds the next targets, so the layer can blend them
        def g(t):
            k0 = math.floor(t * 10) / 10
            return ik(target(k0) + (target(k0 + 0.1) - target(k0)) * ((t - k0) / 0.1))
    ts, qs = run(variant, g, 3.0, payload=0.3)
    p = fk(qs)
    ideal = np.array([target(t) for t in ts])
    m = (ts > 0.4) & (ts < 2.2)
    err = np.linalg.norm(p[m] - ideal[m], axis=1)
    a = tip_acc(ts, p)
    v = np.linalg.norm(np.gradient(p, ts, axis=0), axis=1)[m]
    stops = int(np.sum((v[1:] < 0.001) & (v[:-1] >= 0.001)))     # tip stops (< 1 mm/s) while it should move
    return dict(track_rms_mm=1000 * float(np.sqrt(np.mean(err**2))),
                a_peak=float(np.max(np.linalg.norm(a[ts > 0.2], axis=1))), stops=stops), (ts, a)


def s2_fast_reach(variant, mode, payload=0.3):
    """25 cm reach with 0.3 kg in the hand (a full bottle), then hold."""
    pb = P0 + np.array([-0.15, 0.20])
    qa, qb = Q0, ik(pb)
    Tm = plan_duration(qa, qb)
    if mode == "raw":
        def g(t):
            return qb if t >= 0.1 else qa
    else:
        def g(t):
            return qa + (qb - qa) * minjerk((t - 0.1) / Tm)
    ts, qs = run(variant, g, Tm + 1.6, payload=payload)
    p = fk(qs)
    a = tip_acc(ts, p)
    err = np.linalg.norm(p - pb, axis=1)
    u = (pb - fk(qa)) / np.linalg.norm(pb - fk(qa))
    over = 1000 * float(max(0.0, np.max((p - pb) @ u)))
    inside = err < 0.002
    settle = next((ts[i] - 0.1 for i in range(len(ts)) if inside[i:].all()), float("nan"))
    return dict(a_peak=float(np.max(np.linalg.norm(a, axis=1))), overshoot_mm=over, settle_s=float(settle),
                move_s=Tm, err_final_mm=1000 * float(err[-1])), (ts, a)


def s3_load(variant):
    """Hold at mid reach; 5 N sideways at the tip (+y), then 5 N the other way, then release."""
    def F(t):
        return np.array([0.0, 5.0 if 0.3 <= t < 1.3 else (-5.0 if 1.3 <= t < 2.3 else 0.0)])
    ts, qs = run(variant, lambda t: Q0, 3.0, force_fn=F)
    p = fk(qs)
    p0 = fk(Q0)
    plus = p[(ts > 1.1) & (ts < 1.3)].mean(0)
    minus = p[(ts > 2.1) & (ts < 2.3)].mean(0)
    rest = p[ts > 2.9].mean(0)
    return dict(plus_mm=1000 * float(np.linalg.norm(plus - p0)), minus_mm=1000 * float(np.linalg.norm(minus - p0)),
                span_mm=1000 * float(np.linalg.norm(plus - minus)), after_mm=1000 * float(np.linalg.norm(rest - p0)))


def s4_bidirectional(variant, cycles=6, seed=3):
    """Back to one point from +3 cm and from -3 cm sideways, `cycles` times each; friction +-20 % per try."""
    rng = np.random.default_rng(seed)
    finals = []
    for k in range(2 * cycles):
        pa = P0 + np.array([0.0, 0.03 if k % 2 == 0 else -0.03])
        qa = ik(pa)
        Tm = max(0.4, plan_duration(qa, Q0))
        ts, qs = run(variant, lambda t, qa=qa, Tm=Tm: qa + (Q0 - qa) * minjerk(t / Tm), Tm + 0.6,
                     fric_scale=rng.uniform(0.8, 1.2, 3), q_start=qa)
        finals.append(fk(qs[-1]))
    f = np.array(finals)
    return dict(spread_mm=1000 * float(np.max(np.linalg.norm(f[:, None] - f[None], axis=2))),
                mean_err_mm=1000 * float(np.mean(np.linalg.norm(f - P0, axis=1))))


CALIB = np.array([0.003, -0.002])              # the arm's own calibration is 3.6 mm off at this pose


def s5_vision(variant, sigma=0.0015, seed=5, use_camera=True):
    """After the planned 25 cm reach, the wrist camera sees the tip-to-target error (sigma 1.5 mm, 10 Hz,
    100 ms late) and corrects 60 % of anything over 2 mm per frame; stops when the mean of the last three frames
    is under 1.5 mm, or after 2 s."""
    rng = np.random.default_rng(seed)
    pb = P0 + np.array([-0.15, 0.20])
    qa, q_cmd = Q0, ik(pb + CALIB)
    Tm = plan_duration(qa, q_cmd)
    arm = Arm(variant, 0.3)
    arm.start(qa)
    goal = qa.copy()
    t, frames, hist, seen, t_done = 0.0, 0, [], [], None
    while t < Tm + 0.1 + 2.0:
        if t < Tm + 0.1:
            goal = qa + (q_cmd - qa) * minjerk((t - 0.1) / Tm)
        elif use_camera and t >= Tm + 0.1 + 0.1 * frames:
            frames += 1
            if hist:
                meas = hist[-1] + rng.normal(0, sigma, 2)
                seen.append(meas)
                if len(seen) >= 3 and np.linalg.norm(np.mean(seen[-3:], axis=0)) < 0.0015:
                    t_done = t
                    break
                if np.linalg.norm(meas) > 0.002:
                    goal = goal + np.linalg.pinv(jac(arm.ql)) @ (-0.6 * meas)
            hist.append(fk(arm.ql) - pb)
        arm.step(goal, np.zeros(3))
        t += DT
    for _ in range(int(0.3 / DT)):
        arm.step(goal, np.zeros(3))
    return dict(err_mm=1000 * float(np.linalg.norm(fk(arm.ql) - pb)),
                extra_s=float((t_done if t_done else t) - (Tm + 0.1)))


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    lines = []

    def say(x=""):
        print(x)
        lines.append(x)

    say("Flat-arm actuator study (sim/actuators.py). Tip at mid reach, "
        f"{1000*np.linalg.norm(fk(Q0)):.0f} mm from the shoulder; 3 yaw joints.")
    rows, traces = [], {}
    for v in VARIANTS:
        r = {"v": v}
        for mode in ("raw", "planned"):
            m1, tr1 = s1_slow_align(v, mode)
            m2, tr2 = s2_fast_reach(v, mode)
            r[mode] = (m1, m2)
            traces[f"{v}|{mode}|s1"] = np.column_stack([tr1[0], tr1[1]])
            traces[f"{v}|{mode}|s2"] = np.column_stack([tr2[0], tr2[1]])
        r["load"], r["bidir"] = s3_load(v), s4_bidirectional(v)
        r["cam"], r["nocam"] = s5_vision(v), s5_vision(v, use_camera=False)
        rows.append(r)
        print("  done:", v)

    say()
    say("1) Slow alignment, 2 cm sideways at 1 cm/s, 0.3 kg in the hand; new policy target every 0.1 s")
    say(f"   {'':28s}{'mode':>8s}{'tracking RMS mm':>17s}{'peak tip acc m/s2':>19s}{'tip stops':>11s}")
    for r in rows:
        for mode in ("raw", "planned"):
            m = r[mode][0]
            say(f"   {r['v'][:28]:28s}{mode:>8s}{m['track_rms_mm']:17.2f}{m['a_peak']:19.2f}{m['stops']:11d}")
    say()
    say("2) Fast reach, 25 cm with 0.3 kg in the hand, then hold")
    say(f"   {'':28s}{'mode':>8s}{'peak tip acc m/s2':>19s}{'overshoot mm':>14s}{'within 2 mm after s':>21s}{'final mm':>10s}")
    for r in rows:
        for mode in ("raw", "planned"):
            m = r[mode][1]
            st = f"{m['settle_s']:.2f}" if math.isfinite(m["settle_s"]) else "never"
            say(f"   {r['v'][:28]:28s}{mode:>8s}{m['a_peak']:19.2f}{m['overshoot_mm']:14.1f}{st:>21s}{m['err_final_mm']:10.2f}")
    say(f"   (planned move: {rows[0]['planned'][1]['move_s']:.2f} s at the joint speed limits; raw: as fast as the actuator goes)")
    say()
    say("3) Stiffness: 5 N sideways at the tip, then 5 N the other way, then released (pushing a box, a door)")
    say(f"   {'':28s}{'+5 N mm':>9s}{'-5 N mm':>9s}{'span mm':>9s}{'after release mm':>18s}")
    for r in rows:
        m = r["load"]
        if max(m["plus_mm"], m["minus_mm"]) > 50:
            say(f"   {r['v'][:28]:28s}   pushed away: a joint's torque limit is below what 5 N at this reach needs")
        else:
            say(f"   {r['v'][:28]:28s}{m['plus_mm']:9.1f}{m['minus_mm']:9.1f}{m['span_mm']:9.1f}{m['after_mm']:18.1f}")
    say()
    say("4) Bidirectional repeatability: back to one point from +3 cm and -3 cm, 6 times each, friction +-20 %")
    say(f"   {'':28s}{'max spread mm':>15s}{'mean error mm':>15s}")
    for r in rows:
        m = r["bidir"]
        say(f"   {r['v'][:28]:28s}{m['spread_mm']:15.2f}{m['mean_err_mm']:15.2f}")
    say()
    say("5) Wrist camera closes the loop after the planned reach (camera sigma 1.5 mm; arm calibration 3.6 mm off)")
    say(f"   {'':28s}{'no camera mm':>14s}{'with camera mm':>16s}{'extra time s':>14s}")
    for r in rows:
        say(f"   {r['v'][:28]:28s}{r['nocam']['err_mm']:14.1f}{r['cam']['err_mm']:16.1f}{r['cam']['extra_s']:14.2f}")
    (OUT / "actuators.txt").write_text("\n".join(lines) + "\n")
    np.savez(OUT / "actuators_traces.npz", **traces)
    return rows


if __name__ == "__main__":
    main()
