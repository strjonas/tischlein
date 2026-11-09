"""Can the robot feel the meal touch down, and does it need a new sensor for that?

Build11 lets go where the plan says the surface is. The robot has no sensor that says the
support has taken the weight. Here the same
physics as dyn11.py (same model, servos, friction and random errors) runs the put-down
into the microwave with the surface higher or lower than the robot believes (e, -10 to
+10 mm: belt stretch, a plate sitting high on its roller ring, a drifted lift count), and
compares ways of deciding when to let go, all in the same trials:

  plan40   status quo: lower at 40 mm/s to the planned height, open
  plan10   the same at 10 mm/s
  lift_sX  lower at 10 mm/s until the lift servo's load reading drops by 60 % of the meal's
           weight (3 readings in a row at 50 Hz), then open; X = reading noise, % of stall
  pitch_sX the same with the wrist-pitch servo's load reading (the meal's torque about it)
  cell     the same with a load cell between the wrist and the hand (0.1 N noise)

Every guarded descent stops without letting go if nothing is felt 15 mm below the plan.
After letting go the rest of the plan (open, fork down, withdraw) shifts by the height
where the touch was felt. The servo readings are the simulated actuator forces plus
Gaussian noise and 0.1 % steps; how noisy the real STS3215 reading is is the bench question.

    uv run python designs/breakfast-cell/build11/senses11.py [seeds per menu, default 3]
    uv run python designs/breakfast-cell/build11/senses11.py --sweep   # contact stiffness and timestep
"""
from pathlib import Path
import json
import math
import sys
import time
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dyn11 as DY  # noqa: E402
P11, X, M, S = DY.P11, DY.X, DY.M, DY.S

E_MM = (-10, -5, -2, 0, 2, 5, 10)
MENUS = (('jar', 'small'), ('box', 'small'))
NOISE = (.5, 2.0, 5.0)                       # servo load reading noise, % of stall per reading
CELL_SD = .1                                 # N, load cell noise per reading
V_GUARD = .010                               # m/s, guarded descent
SAMPLE = 2                                   # control ticks per reading (50 Hz)
CONFIRM = 3                                  # readings in a row over the threshold
FRAC = .6                                    # share of the meal's weight that must move to the support
OVER = .015                                  # m below the plan with nothing felt: stop, keep holding
STRATEGIES = ['plan40', 'plan10'] + [f'{s}_s{n:g}' for s in ('lift', 'pitch') for n in NOISE] + ['cell']


def put_down_indices(wp):
    """The planned put-down: k_open is the first waypoint after the grip lets go, k_end the
    end of the vertical lowering just before it, k_up where that lowering starts."""
    held = [w['held'] for w in wp]
    k_open = next(k for k in range(1, len(wp)) if held[k - 1] and not held[k])
    k_end = k_open - 1
    while k_end > 0 and abs(wp[k_end]['q'][6] - wp[k_end - 1]['q'][6]) > 1e-6:
        k_end -= 1                               # skip any jaw motion that precedes the opening
    k_up = k_end
    while k_up > 0:
        a, b = np.array(wp[k_up - 1]['q']), np.array(wp[k_up]['q'])
        if a[0] - b[0] > 1e-6 and np.all(np.abs(a[1:6] - b[1:6]) < 1e-4):
            k_up -= 1                            # the lowering moves the lift only
        else:
            break
    return k_up, k_end, k_open


class Rig:
    """dyn11's dynamic model with the readings a guarded put-down needs."""

    def __init__(self, obj, kind, solref=None, timestep=None):
        self.D = DY.Dyn(obj, kind)
        m = self.D.m
        if solref:
            m.geom_solref[:, 0] = solref
        if timestep:
            m.opt.timestep = timestep
        self.hand = m.body('wrist_pitch').id
        self.pitch_j = m.joint('pitch').id
        self.snap = mujoco.MjData(m)
        self.f6 = np.zeros(6)

    def tick(self, goal, env, squeeze, offsets):
        D, m, d = self.D, self.D.m, self.D.d
        q = goal + offsets
        if squeeze:
            q[6] += DY.OVERCLOSE
        for i, j in enumerate(S.ARM):
            d.ctrl[D.act[j]] = q[i]
        for k, v in env.items():
            if k in D.act:
                d.ctrl[D.act[k]] = v
        for _ in range(int(round(DY.DT_CTRL / m.opt.timestep))):
            d.qpos[D.jadr['cw']] = M.CW_C - d.qpos[D.jadr['lift']]
            d.qvel[D.dadr['cw']] = 0
            d.xfrc_applied[D.car, 2] = DY.CW_FORCE
            mujoco.mj_step(m, d)

    def support_force(self):
        D, m, d = self.D, self.D.m, self.D.d
        tot = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (D.is_obj[g1] and not D.is_robot[g2] and not D.is_obj[g2]) or (D.is_obj[g2] and not D.is_robot[g1] and not D.is_obj[g1]):
                mujoco.mj_contactForce(m, d, i, self.f6)
                tot += self.f6[0]
        return tot

    def readings(self):
        """Noise-free lift force (N), pitch torque (N m) and the force the wrist carries up (N)."""
        D, m, d = self.D, self.D.m, self.D.d
        mujoco.mj_rnePostConstraint(m, d)
        return (float(d.actuator_force[D.act['lift']]), float(d.actuator_force[D.act['pitch']]),
                float(d.cfrc_int[self.hand][5]))

    def meal_torque(self, W):
        """The meal's weight as a torque about the pitch axis: what the robot's model predicts
        the pitch servo stops carrying when the support takes it."""
        d = self.D.d
        r = d.xipos[self.D.obj_body] - d.xanchor[self.pitch_j]
        return abs(float(np.cross(r, [0, 0, -W]) @ d.xaxis[self.pitch_j]))

    def follow(self, wp, offsets, squeeze, log):
        R = DY.reference(wp, settle=0.0)
        held = [wp[i]['held'] for i in R['i0']]
        for n in range(len(R['t'])):
            env = DY.env_at(wp, R['i0'][n], R['i1'][n], R['frac'][n])
            self.tick(R['goal'][n].copy(), env, squeeze and held[n], offsets)
            log(held[n])


def plan_take(D, seq, p):
    """dyn11's step 1 plan (take, into the microwave) and its true placement target."""
    obj, kind = D.obj, D.kind
    o = X.OBJECTS[obj]
    fork = o['mode'] == 'fork'
    start, psi0 = P11.START[(obj, kind)]
    psi0 = math.radians(psi0) if psi0 is not None else math.pi / 2
    w0 = DY.segment_start(seq, 'take')
    true_pos = np.array(start) + p['place']
    true_psi = psi0 + p['place_yaw']
    seen_pos, seen_psi = true_pos + p['seen'], true_psi + p['seen_yaw']
    c = P11.Cell(kind, obj)
    c.env.update(w0['env'])
    P = P11.Planner(c, np.array(w0['q']))
    P.park = np.array(seq['wp'][0]['q'])
    if fork:
        P.take(seen_pos, seen_psi, out_y=-.12 - o['L'] / 2, rails=True, lift=.015)
    else:
        P.take(seen_pos, math.pi / 2, out_y=-.12)
    mc = c.f['mw_centre']
    tb = np.array([mc[0], mc[1], c.f['plate_z'] + X.RAILS['h'] + .0005])
    L = o['L'] if fork else .1
    t_start = tb + [0, 0, c.cfg['tool_h'] + .015]; t_start[1] = -L / 2 - .03
    P.put_down(tb, math.pi / 2, t_start)
    wp = [dict(t=w['t'], q=list(w['q']), env=w['env'], held=w['held']) for w in P.wp]
    rails = np.r_[true_pos[0], true_pos[1], true_pos[2] - X.RAILS['h'], M.quat_z(true_psi - math.pi / 2)] if fork else None
    return wp, true_pos, true_psi, rails, tb


def branch(rig, wp, k, offsets, e, strategy, W, tau_meal, tb, rng):
    """From the snapshot over the spot: the robot believes the surface is e lower than it
    really is (its lift runs e low), lowers, decides when to let go, finishes the plan."""
    D, m, d = rig.D, rig.D.m, rig.D.d
    mujoco.mj_copyData(d, m, rig.snap)
    k_up, k_end, _ = k
    env = dict(wp[k_up]['env'])
    q_up = np.array(wp[k_up]['q'], float)
    z_t = wp[k_end]['q'][0]
    off = offsets.copy()
    peak = [0.0]
    trace = dict(z=[], held=[])

    def log(held):
        peak[0] = max(peak[0], rig.support_force())
        trace['z'].append(float(d.xpos[D.obj_body][2])); trace['held'].append(bool(held))

    for n in range(30):                                   # 0.3 s: the error appears while it hovers
        off[0] = offsets[0] - e * (n + 1) / 30
        rig.tick(q_up.copy(), env, True, off); log(True)
    for _ in range(20):                                   # and settles
        rig.tick(q_up.copy(), env, True, off); log(True)
    kind = 'plan' if strategy.startswith('plan') else strategy.split('_')[0]
    sd = float(strategy.split('_s')[1]) / 100 if '_s' in strategy else 0.0
    v = .040 if strategy == 'plan40' else V_GUARD
    goal = q_up.copy()
    base, over, felt, stopped, t, n = [], 0, None, False, 0.0, 0
    cap = {'lift': X.DRIVES['lift'][2], 'pitch': X.DRIVES['pitch'][2]}
    expect = {'lift': W, 'pitch': tau_meal, 'cell': W}
    while True:
        goal[0] = max(goal[0] - v * DY.DT_CTRL, z_t - (OVER if kind in ('lift', 'pitch', 'cell') else 0))
        rig.tick(goal.copy(), env, True, off); log(True)
        t += DY.DT_CTRL; n += 1
        if kind == 'plan':
            if goal[0] <= z_t + 1e-9:
                break
            continue
        if n % SAMPLE:
            continue
        lf, pt, wc = rig.readings()
        if kind == 'cell':
            s = wc + rng.normal(0, CELL_SD)
        else:
            raw = lf if kind == 'lift' else pt
            s = raw + rng.normal(0, sd * cap[kind])
            s = round(s / (.001 * cap[kind])) * .001 * cap[kind]
        if .1 <= t <= .3:
            base.append(s)
            continue
        if t < .3:
            continue
        b = float(np.mean(base))
        moved = (b - s) if kind in ('lift', 'cell') else abs(abs(b) - abs(s))
        over = over + 1 if moved >= FRAC * expect[kind] else 0
        if over >= CONFIRM:
            felt = goal[0]
            break
        if goal[0] <= z_t - OVER + 1e-9:
            stopped = True
            break
    if stopped:
        for _ in range(60):
            rig.tick(goal.copy(), env, True, off); log(True)
        pos, quat = D.obj_pose()
        return dict(outcome='stopped, still holding', press_N=round(peak[0], 1), drop_mm=0.0,
                    felt_mm=None, place_err_mm=None, tilt_deg=round(DY.tilt_deg(quat), 1))
    shift = (felt - z_t) if felt is not None else 0.0
    rest = [dict(w, q=[w['q'][0] + shift] + list(w['q'][1:])) for w in wp[k_end:]]
    rest[0]['q'] = list(goal)
    n_hold = len(trace['held'])
    rig.follow(rest, off, True, log)
    for _ in range(60):
        rig.tick(np.array(rest[-1]['q'], float), rest[-1]['env'], False, off); log(False)
    pos, quat = D.obj_pose()
    z = np.array(trace['z']); held = np.array(trace['held'])
    i_open = n_hold + int(np.argmax(~held[n_hold:])) if (~held[n_hold:]).any() else len(z) - 1
    drop = max(0.0, float(z[i_open] - z[-1]))
    err = float(np.linalg.norm((pos - tb)[:2]))
    tilt = DY.tilt_deg(quat)
    ok = pos[2] - tb[2] > -.02 and tilt < 5 and err < .015
    return dict(outcome='placed' if ok else ('tipped' if tilt >= 5 else 'off target'), press_N=round(peak[0], 1),
                drop_mm=round(drop * 1000, 1), felt_mm=None if felt is None else round((felt - z_t) * 1000, 1),
                place_err_mm=round(err * 1000, 1), tilt_deg=round(tilt, 1))


def signal_check(rig, wp, k, offsets, W, tau_meal):
    """Noise-free readings while lowering (10 mm/s, surface where believed): how much each
    reading moves when the support takes the meal, against its own wobble while lowering."""
    D, m, d = rig.D, rig.D.m, rig.D.d
    mujoco.mj_copyData(d, m, rig.snap)
    k_up, k_end, _ = k
    env = dict(wp[k_up]['env'])
    goal = np.array(wp[k_up]['q'], float)
    z_t = wp[k_end]['q'][0]
    rows, goals = [], []
    while goal[0] > z_t - .006:
        goal[0] -= V_GUARD * DY.DT_CTRL
        rig.tick(goal.copy(), env, True, offsets)
        rows.append((*rig.readings(), rig.support_force()))
        goals.append(goal[0])
    a = np.array(rows)
    t = np.arange(len(a)) * DY.DT_CTRL
    touch = int(np.argmax(a[:, 3] > .05 * W)) if (a[:, 3] > .05 * W).any() else len(a) - 1
    pre = a[(t >= .1) & (t < t[touch] - .05)]
    full = int(np.argmax(a[:, 3] > .95 * W)) if (a[:, 3] > .95 * W).any() else len(a) - 1
    after = min(full + 15, len(a) - 1)                  # 0.15 s later, once the servo has caught up
    out = {}
    for i, (name, ref) in enumerate((('lift', W), ('pitch', tau_meal), ('cell', W))):
        moved = abs(float(np.mean(pre[:, i]) - a[after, i]))
        out[name] = dict(moves=round(moved, 3), expected=round(ref, 3), wobble_sd=round(float(np.std(pre[:, i])), 3),
                         of_stall=round(moved / X.DRIVES[name][2], 4) if name in X.DRIVES else None)
    out['touch_to_full_mm'] = round((full - touch) * DY.DT_CTRL * V_GUARD * 1000, 2)
    out['touch_at_mm'] = round((goals[touch] - z_t) * 1000, 1)    # lift goal at first touch, above (+) the plan's end
    return out


def prepare(rig, seq, p):
    """Runs the take to the point over the microwave and keeps that state as the snapshot."""
    D = rig.D
    D.set_params(p)
    wp, true_pos, true_psi, rails, tb = plan_take(D, seq, p)
    k = put_down_indices(wp)
    D.place(true_pos, M.quat_z(true_psi - math.pi / 2), wp[0]['q'], wp[0]['env'])
    if rails is not None:
        D.set_rails(rails)
        mujoco.mj_forward(D.m, D.d)
    rig.follow(wp[:k[0] + 1], p['offsets'], True, lambda h: None)
    for _ in range(40):                                    # settle over the spot
        rig.tick(np.array(wp[k[0]]['q'], float), wp[k[0]]['env'], True, p['offsets'])
    mujoco.mj_copyData(rig.snap, D.m, D.d)
    return wp, k, tb


def sweep():
    """Do the conclusions hold with softer or stiffer contacts and a finer timestep?"""
    seqs = json.loads((HERE / 'plan11.json').read_text())['sequences']
    out = []
    for solref, ts in ((.002, None), (.004, None), (.008, None), (.004, .001)):
        for obj, kind in MENUS:
            rig = Rig(obj, kind, solref=solref, timestep=ts)
            rng = np.random.default_rng(7)
            for s in range(3):
                p = DY.draw(obj, rng, nominal=(s == 0))
                wp, k, tb = prepare(rig, seqs[f'{obj}_{kind}'], p)
                W = p['mass'] * 9.81
                tau = rig.meal_torque(W)
                for e in (-5, 5):
                    for st in ('plan40', 'lift_s2', 'pitch_s2', 'cell'):
                        if st.startswith('pitch') and X.OBJECTS[obj]['mode'] == 'fork':
                            continue
                        r = branch(rig, wp, k, p['offsets'], e / 1000, st, W, tau, tb, np.random.default_rng(10 * s + e + 50))
                        r.update(solref=solref, timestep=ts or .002, menu=f'{obj}_{kind}', seed=s, e_mm=e, strategy=st)
                        out.append(r)
                        print(r, flush=True)
    (HERE / 'senses11-sweep.json').write_text(json.dumps(out, indent=1))


def main():
    if '--sweep' in sys.argv:
        return sweep()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seqs = json.loads((HERE / 'plan11.json').read_text())['sequences']
    res = dict(settings=dict(e_mm=E_MM, noise_pct_of_stall=NOISE, cell_sd_N=CELL_SD, v_guard_mm_s=V_GUARD * 1000,
                             sample_hz=round(1 / (SAMPLE * DY.DT_CTRL)), confirm=CONFIRM, frac=FRAC, over_mm=OVER * 1000))
    for obj, kind in MENUS:
        key = f'{obj}_{kind}'
        t0 = time.time()
        rig = Rig(obj, kind)
        D = rig.D
        rng = np.random.default_rng(7)
        runs, signals = [], []
        for s in range(n_seeds):
            p = DY.draw(obj, rng, nominal=(s == 0))
            wp, k, tb = prepare(rig, seqs[key], p)
            W = p['mass'] * 9.81
            tau = rig.meal_torque(W)
            sig = signal_check(rig, wp, k, p['offsets'], W, tau)
            signals.append(sig)
            print(key, 'seed', s, 'signal', sig, flush=True)
            for e in E_MM:
                for st in STRATEGIES:
                    if st.startswith('pitch') and X.OBJECTS[obj]['mode'] == 'fork':
                        continue                           # the box sits over the pitch axis: no torque to feel
                    r = branch(rig, wp, k, p['offsets'], e / 1000, st, W, tau, tb,
                               np.random.default_rng(1000 * s + 10 * (e + 20) + STRATEGIES.index(st)))
                    r.update(seed=s, e_mm=e, strategy=st)
                    runs.append(r)
                    print(key, s, f'e={e:+d}', st, r, flush=True)
        summary = {}
        for st in STRATEGIES:
            rs = [r for r in runs if r['strategy'] == st]
            if not rs:
                continue
            summary[st] = dict(
                placed=sum(r['outcome'] == 'placed' for r in rs), stopped=sum(r['outcome'].startswith('stopped') for r in rs), n=len(rs),
                worst_drop_mm=max(r['drop_mm'] for r in rs), worst_press_N=max(r['press_N'] for r in rs),
                by_e={e: dict(drop_mm=max(r['drop_mm'] for r in rs if r['e_mm'] == e),
                              press_N=max(r['press_N'] for r in rs if r['e_mm'] == e),
                              outcomes=sorted({r['outcome'] for r in rs if r['e_mm'] == e})) for e in E_MM})
        res[key] = dict(signals=signals, summary=summary, runs=runs, runtime_s=round(time.time() - t0, 1))
        (HERE / 'senses11-results.json').write_text(json.dumps(res, indent=1))
        print(key, 'done', round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
