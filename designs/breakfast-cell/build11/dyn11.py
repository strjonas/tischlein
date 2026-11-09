"""Does it really work? The object handling of every menu in physics, with random errors.

The same build11 geometry as plan11.py, now dynamic (model11.mjcf11(dynamic=True)):
gravity; the seven servos as position actuators whose torque is capped at the STS3215's
stall torque at each joint (the jaw at its rated torque, as a torque-limit setting);
counterweight on the lift; friction on every contact; the object a free body that only
the plate, the jaw and its supports hold. Doors, flap and turntable follow the plan.

Each trial:
1. Draws errors: where the object really stands (on rails: +-10 mm along, +-1.5 mm and
   +-1.5 deg across; the jar anywhere +-10 mm), what the camera sees (2 mm, 1 deg sd),
   weight, friction (wet glass, frost), servo stiffness (60-100 %), joint calibration
   (0.3 deg, 0.5 mm sd), and the turntable's stop angle (180 deg turns +- 8 deg).
2. Plans 'take, into the microwave' from the perceived pose (plan11's planner), then runs
   it in physics.
3. Turns the turntable (with the settled object), perceives it again, plans 'hot take,
   serve', runs it.
Pass: the object is never dropped or knocked over, lands on the serving ribs within
15 mm and 8 deg of the target, upright; no servo above stall for more than 0.2 s.

    uv run python designs/breakfast-cell/build11/dyn11.py [trials per menu, default 30]
"""
from pathlib import Path
import fcntl
import json
import math
import sys
import time
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan11 as P11  # noqa: E402
import model11 as X  # noqa: E402
M, S = P11.M, P11.S

DT_CTRL = .01
RAMP = .25              # s, the servos' acceleration ramp: progress along the path is smoothed over this window
CORNER = 20.0           # deg: at every sharper bend of the planned joint path the arm comes to a stop
STIFF_CAL = .8          # servo stiffness the lag compensation is calibrated for (trials draw 60-100 %)
OVERCLOSE = .15         # rad past contact: the jaw squeezes at its torque limit
CW_FORCE = 1.25 * 9.81
MENUS = [('jar', 'large'), ('box', 'large'), ('tray', 'large'), ('jar', 'small'), ('box', 'small')]
VMAX = np.array([.06, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])


def lag_lead():
    """Seconds each servo trails a moving goal (kv / kp at the calibrated stiffness); the host
    sends goal + lead * planned speed, as you would after timing each servo on the bench."""
    return np.array([X.DRIVES[j][1] / (X.DRIVES[j][0] * STIFF_CAL) for j in S.ARM])


def reference(wp, settle=0.0, tail=.6, lead=True):
    """Servo goals that follow the planned path itself. The planner's dense waypoints are
    a polyline in joint space; smoothing each joint separately would cut its corners (by
    about a centimetre at the hand), so instead the progress along the polyline is smoothed,
    and the arm dwells for one ramp at every bend sharper than CORNER (it decelerates into
    the bend and accelerates out of it). Goals lead the plan by each servo's lag.
    Returns control times, goals, planned joints and the waypoint index and fraction per tick."""
    Q = np.array([w['q'] for w in wp])
    ts = np.array([w['t'] for w in wp]) - wp[0]['t']
    d = np.diff(Q, axis=0) / VMAX
    nrm = np.linalg.norm(d, axis=1)
    corner = np.zeros(len(wp), bool)
    last = None
    for i in range(len(d)):
        if nrm[i] < 1e-7:
            continue
        u = d[i] / nrm[i]
        if last is not None and u @ last < math.cos(math.radians(CORNER)):
            corner[i] = True
        last = u
    times, sig, shift = [], [], settle
    for i in range(len(wp)):
        times.append(ts[i] + shift); sig.append(float(i))
        if corner[i] or i == 0:
            shift += RAMP
            times.append(ts[i] + shift); sig.append(float(i))
    times.append(times[-1] + tail); sig.append(float(len(wp) - 1))
    times, sig = np.array(times), np.array(sig)
    tgrid = np.arange(0, times[-1], DT_CTRL)
    s = np.interp(tgrid, times, sig)
    k = max(1, int(RAMP / DT_CTRL))
    pad = np.r_[np.repeat(s[:1], k), s, np.repeat(s[-1:], k)]
    s = np.convolve(pad, np.ones(k) / k, 'same')[k:-k]
    i0 = np.clip(np.floor(s).astype(int), 0, len(wp) - 1)
    i1 = np.clip(i0 + 1, 0, len(wp) - 1)
    fr = (s - i0)[:, None]
    plan = Q[i0] + (Q[i1] - Q[i0]) * fr
    goal = plan.copy()
    if lead:
        goal += lag_lead() * np.gradient(plan, DT_CTRL, axis=0)
    return dict(t=tgrid, goal=goal, plan=plan, i0=i0, i1=i1, frac=fr[:, 0], n_corners=int(corner.sum()))


def env_at(wp, i0, i1, fr):
    a, b = wp[i0]['env'], wp[i1]['env']
    return {k: a[k] + (b[k] - a[k]) * fr for k in a}


class Dyn:
    def __init__(self, obj, kind):
        X.configure(obj)
        xml, f, _ = X.mjcf11(kind, obj, dynamic=True)
        self.m = mujoco.MjModel.from_xml_string(xml)
        X.joint_tuning(self.m)
        # a microwave turntable is a 5 rpm synchronous motor behind a large gearbox: its reflected
        # inertia keeps the plate smooth (without it the stiff position drive chatters under the meal)
        self.m.dof_armature[self.m.jnt_dofadr[self.m.joint('turntable').id]] = .1
        self.d = mujoco.MjData(self.m)
        self.f, self.obj, self.kind = f, obj, kind
        m = self.m
        self.jadr = {m.joint(i).name: int(m.jnt_qposadr[i]) for i in range(m.njnt)}
        self.dadr = {m.joint(i).name: int(m.jnt_dofadr[i]) for i in range(m.njnt)}
        self.act = {m.actuator(i).name[2:]: i for i in range(m.nu)}
        self.obj_body = m.body('jar').id
        self.car = m.body('carriage').id
        self.gripper = m.body('wrist_pitch').id
        self.base_kp = m.actuator_gainprm[:, 0].copy()
        self.base_mass = m.body_mass[self.obj_body]
        self.base_inertia = m.body_inertia[self.obj_body].copy()
        names = [m.geom(i).name for i in range(m.ngeom)]
        self.gname = names
        body_of = [m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)]
        self.is_obj = np.array([b == 'jar' for b in body_of])
        robot_bodies = S.ROBOT_BODIES | {'jaw'}
        self.is_robot = np.array([b in robot_bodies for b in body_of])
        self.groups = dict(
            plate=[i for i, n in enumerate(names) if n in ('plate', 'nub_top', 'nub_low')],
            pads=[i for i, n in enumerate(names) if n.startswith('jaw_v')],
            jawpetg=[i for i, n in enumerate(names) if n in ('jaw_finger', 'jaw_link', 'jaw_hub', 'palm')],
            shelf=[i for i, n in enumerate(names) if n.startswith('f_shelf') and 'rail' not in n] + [names.index('f_bottom')] if 'f_bottom' in names else [],
            rails=[i for i, n in enumerate(names) if 'rail' in n],
            ribs=[i for i, n in enumerate(names) if 'rib' in n],
            plate_glass=[i for i, n in enumerate(names) if n == 'm_plate'],
            obj=[i for i, n in enumerate(names) if self.is_obj[i]],
        )

    def set_params(self, p):
        m = self.m
        for g in range(m.ngeom):
            m.geom_friction[g, 0] = .4
        for k, mu in p['mu'].items():
            for g in self.groups[k]:
                m.geom_friction[g, 0] = mu
        for g in self.groups['obj']:
            m.geom_friction[g, 0] = .05                    # pair friction = max of the two: the partner decides
        m.body_mass[self.obj_body] = p['mass']
        m.body_inertia[self.obj_body] = self.base_inertia * p['mass'] / self.base_mass
        for j, i in self.act.items():
            if j in X.DRIVES:
                kp = self.base_kp[i] * p['stiff'].get(j, 1.0)
                m.actuator_gainprm[i, 0] = kp
                m.actuator_biasprm[i, 1] = -kp

    def place(self, pos, quat, q, env):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        for i, n in enumerate(S.ARM):
            d.qpos[self.jadr[n]] = q[i]
        for k, v in env.items():
            if k in self.jadr:
                d.qpos[self.jadr[k]] = v
        a = self.jadr['jar']
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, quat
        self.set_rails(np.r_[0, 0, -4, 1, 0, 0, 0])
        mujoco.mj_forward(self.m, d)

    def set_rails(self, pose):
        k = self.m.body('rails').mocapid[0]
        self.d.mocap_pos[k], self.d.mocap_quat[k] = pose[:3], pose[3:]

    def run(self, wp, offsets, squeeze, rails_pose, settle=.4):
        """Follow the waypoints (dense plan) in physics. Returns a trace."""
        m, d = self.m, self.d
        R = reference(wp, settle=settle)
        tgrid, ref = R['t'], R['goal']
        held = np.array([wp[i]['held'] for i in R['i0']])
        env = [env_at(wp, a, b, f) for a, b, f in zip(R['i0'], R['i1'], R['frac'])]
        caps = {j: X.DRIVES[j][2] for j in X.DRIVES}
        if rails_pose is not None:
            self.set_rails(rails_pose)
            mujoco.mj_forward(m, d)
        sat = {j: 0.0 for j in caps}
        peak = {j: 0.0 for j in caps}
        trace = dict(t=[], obj=[], rel=[], held=[])
        hold_sum, hold_n = 0.0, 0
        nsub = int(round(DT_CTRL / m.opt.timestep))
        G = self.gripper
        for n, t in enumerate(tgrid):
            q = ref[n] + offsets
            if held[n] and squeeze:
                q[6] += OVERCLOSE
            for i, j in enumerate(S.ARM):
                d.ctrl[self.act[j]] = q[i]
            for kk, v in env[n].items():
                if kk in self.act:
                    d.ctrl[self.act[kk]] = v
            for _ in range(nsub):
                d.qpos[self.jadr['cw']] = M.CW_C - d.qpos[self.jadr['lift']]
                d.qvel[self.dadr['cw']] = 0
                d.xfrc_applied[self.car, 2] = CW_FORCE
                mujoco.mj_step(m, d)
            for j in caps:
                f = abs(d.actuator_force[self.act[j]])
                peak[j] = max(peak[j], f / caps[j])
                if f >= caps[j] * .999:
                    sat[j] += DT_CTRL
            if held[n]:
                hold_sum += abs(d.actuator_force[self.act['pitch']]); hold_n += 1
            if n % 5 == 0:
                po = d.xpos[self.obj_body].copy()
                R = d.xmat[G].reshape(3, 3)
                trace['t'].append(t); trace['obj'].append(po.tolist()); trace['held'].append(bool(held[n]))
                trace['rel'].append((R.T @ (po - d.xpos[G])).tolist())
        return dict(trace=trace, peak={j: round(v, 3) for j, v in peak.items()}, stalled_s={j: round(v, 2) for j, v in sat.items()},
                    pitch_hold_Nm=round(hold_sum / max(hold_n, 1), 3))

    def obj_pose(self):
        a = self.jadr['jar']
        return self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()


def tilt_deg(quat):
    w, x, y, z = quat
    zz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1, min(1, zz))))


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ one trial
def draw(obj, rng, nominal=False):
    o = X.OBJECTS[obj]
    u = (lambda lo, hi: (lo + hi) / 2) if nominal else (lambda lo, hi: float(rng.uniform(lo, hi)))
    nz = (lambda sd: 0.0) if nominal else (lambda sd: float(rng.normal(0, sd)))
    frost = obj == 'tray'
    p = dict(
        mass=u(*{'jar': (.70, .80), 'box': (1.57, 1.87), 'tray': (.40, .46)}[obj]),
        mu=dict(plate=u(.12, .20) if frost else u(.20, .30), pads=u(.25, .40) if frost else u(.35, .70),
                jawpetg=u(.20, .30), shelf=u(.20, .40), rails=u(.12, .20) if frost else u(.20, .30), ribs=u(.60, .90),
                plate_glass=u(.20, .30)),
        stiff={j: u(.6, 1.0) for j in X.DRIVES},
        place=np.array([nz(.006) if o['mode'] == 'fork' else u(-.01, .01), u(-.01, .01) if o['mode'] == 'fork' else u(-.01, .01), 0]),
        place_yaw=0.0 if o['mode'] == 'side' else u(-math.radians(1.5), math.radians(1.5)),
        seen=np.array([nz(.002), nz(.002), 0]), seen_yaw=nz(math.radians(1.0)),
        seen2=np.array([nz(.002), nz(.002), 0]), seen2_yaw=nz(math.radians(1.0)),
        offsets=np.array([nz(.0005)] + [nz(math.radians(.3)) for _ in range(5)] + [0.0]),
        tt=u(-math.radians(8), math.radians(8)),
    )
    if o['mode'] == 'fork':
        p['place'][0] = float(np.clip(p['place'][0], -.0015, .0015)) if not nominal else 0.0   # rails hold it across
    return p


def segment_start(seq, label):
    wp = seq['wp']
    i = next(k for k, w in enumerate(wp) if w['label'] == label)
    return wp[i]


def trial(D, seq, p):
    obj, kind = D.obj, D.kind
    o = X.OBJECTS[obj]
    fork = o['mode'] == 'fork'
    start, psi0 = P11.START[(obj, kind)]
    psi0 = math.radians(psi0) if psi0 is not None else math.pi / 2
    out = dict()
    # ---- 1. take, into the microwave
    w0 = segment_start(seq, 'take')
    true_pos = np.array(start) + p['place']
    true_psi = psi0 + p['place_yaw']
    seen_pos, seen_psi = true_pos + p['seen'], true_psi + p['seen_yaw']
    c = P11.Cell(kind, obj)
    c.env.update(w0['env'])
    P = P11.Planner(c, np.array(w0['q']))
    P.park = np.array(seq['wp'][0]['q'])
    try:
        if fork:
            P.take(seen_pos, seen_psi, out_y=-.12 - o['L'] / 2, rails=True, lift=.006 if obj == 'tray' else .015)
        else:
            P.take(seen_pos, math.pi / 2, out_y=-.12)
        mc = c.f['mw_centre']
        tb = np.array([mc[0], mc[1], c.f['plate_z'] + X.RAILS['h'] + .0005])
        L = o['L'] if fork else .1
        t_start = tb + [0, 0, c.cfg['tool_h'] + .015]; t_start[1] = -L / 2 - .03
        P.put_down(tb, math.pi / 2, t_start)
    except RuntimeError as e:
        return dict(ok=False, why=f'plan 1: {e}')
    wp1 = [dict(t=w['t'], q=list(w['q']), env=w['env'], held=w['held']) for w in P.wp]
    rails = np.r_[true_pos[0], true_pos[1], true_pos[2] - X.RAILS['h'], M.quat_z(true_psi - math.pi / 2)] if fork else None
    D.place(true_pos, M.quat_z(true_psi - math.pi / 2), wp1[0]['q'], wp1[0]['env'])
    if rails is not None:
        D.set_rails(rails)
        mujoco.mj_forward(D.m, D.d)
    r1 = D.run(wp1, p['offsets'], squeeze=True, rails_pose=rails)
    pos1, quat1 = D.obj_pose()
    out['seg1'] = dict(peak=r1['peak'], stalled_s=r1['stalled_s'], pitch_hold_Nm=r1['pitch_hold_Nm'])
    ev = judge(r1['trace'], pos1, quat1, tb, math.pi / 2, fork, 'microwave')
    out['seg1'].update(ev)
    if not ev['ok']:
        return dict(ok=False, why=f'take/into microwave: {ev["why"]}', **out)
    # ---- 2. the turntable turns; hot take, serve
    w2 = segment_start(seq, 'take_hot')
    turn = 3 * math.pi + (p['tt'] if fork else 1.3)
    ctr = np.r_[c.f['mw_centre'], 0]
    r = pos1 - ctr
    ca, sa = math.cos(turn), math.sin(turn)
    pos2 = ctr + np.array([r[0] * ca - r[1] * sa, r[0] * sa + r[1] * ca, r[2]])
    yaw2 = yaw_of(quat1) + turn
    quat2 = M.quat_z(yaw2)
    env2 = dict(w2['env']); env2['turntable'] = turn
    try:
        P2, sb, cage = plan_hot(obj, kind, np.array(w2['q']), env2, pos2, p['seen2'], p['seen2_yaw'], np.array(seq['wp'][0]['q']))
    except RuntimeError as e:
        return dict(ok=False, why=f'plan 2: {e}', **out)
    wp2 = [dict(t=w['t'], q=list(w['q']), env=w['env'], held=w['held']) for w in P2.wp]
    D.place(pos2, quat2, wp2[0]['q'], wp2[0]['env'])
    r2 = D.run(wp2, p['offsets'], squeeze=cage is None, rails_pose=None)
    pos3, quat3 = D.obj_pose()
    out['seg2'] = dict(peak=r2['peak'], stalled_s=r2['stalled_s'], pitch_hold_Nm=r2['pitch_hold_Nm'])
    ev = judge(r2['trace'], pos3, quat3, sb, math.pi / 2, fork, 'serving spot')
    out['seg2'].update(ev)
    if not ev['ok']:
        return dict(ok=False, why=f'hot take/serve: {ev["why"]}', **out)
    stall = max(max(v for j, v in r['stalled_s'].items() if j != 'grip') for r in (r1, r2))   # the jaw squeezes at its limit by design
    if stall > .2:
        return dict(ok=False, why=f'a servo at stall for {stall:.1f} s', **out)
    return dict(ok=True, **out)


def plan_hot(obj, kind, q0, env, pos, seen_err, seen_yaw, park):
    """After heating the camera sees the meal (at pos, seen with error seen_err) and the
    turntable (env['turntable'], seen to seen_yaw): plan the hot take and the serve from there.
    The fork follows the rib channel, so only the box's place along it matters.
    Returns the planner, the serving target and the jaw's cage angle (tray) or None."""
    o = X.OBJECTS[obj]
    fork = o['mode'] == 'fork'
    c2 = P11.Cell(kind, obj)
    c2.env.update(env)
    ctr = np.r_[c2.f['mw_centre'], 0]
    seen = np.asarray(pos, float) + seen_err
    if fork:
        psi_hot = wrap(env['turntable'] + math.pi / 2 + seen_yaw)
        if math.cos(psi_hot - math.pi / 2) < 0:
            psi_hot = wrap(psi_hot + math.pi)
        u = np.array([math.cos(psi_hot), math.sin(psi_hot), 0])
        seen = ctr + u * float((seen - ctr) @ u) + [0, 0, seen[2]]
    else:
        psi_hot = math.pi / 2
    P2 = P11.Planner(c2, q0)
    P2.park = park
    cage = X.jaw_angle_to(X.PLATE['face'] - o['H'], (X.PALM, X.PALM + o['L'])) - .08 if obj == 'tray' else None
    P2.step('take_hot', 'Seen again after heating: fork in along the ribs, take it out' if fork else 'Seen again after heating: take the hot jar out')
    P2.take(seen, psi_hot, out_y=-(o['L'] if fork else .1) / 2 - .03, grip=cage)
    sb = np.array([X.SERVE[0], X.SERVE[1], c2.f['mw_top'] + X.RAILS['h'] + .0005])
    s_start = sb + [0, 0, c2.cfg['tool_h'] + .045]; s_start[1] = sb[1] - .12
    P2.step('serve', 'Set it on the serving ribs on top of the microwave' if fork else 'Stand it on top of the microwave')
    P2.put_down(sb, math.pi / 2, s_start, grip=cage)
    return P2, sb, cage


def judge(tr, pos, quat, target, psi_t, fork, where):
    rel = np.array(tr['rel']); held = np.array(tr['held']); obj = np.array(tr['obj'])
    slip = 0.0
    if held.any():
        hr = rel[held]
        k = min(len(hr) - 1, 8)                    # reference once the grasp has settled
        slip = float(np.max(np.linalg.norm(hr[k:] - hr[k], axis=1))) if len(hr) > k + 1 else 0.0
    err = float(np.linalg.norm((pos - target)[:2]))
    dz = float(pos[2] - target[2])
    yaw_err = 0.0
    if fork:
        yaw_err = abs(math.degrees(wrap(yaw_of(quat) - (psi_t - math.pi / 2))))
        yaw_err = min(yaw_err, 180 - yaw_err)
    tilt = tilt_deg(quat)
    ok, why = True, ''
    if dz < -.02:
        ok, why = False, 'dropped'
    elif tilt > 5:
        ok, why = False, f'tipped over ({tilt:.0f} deg)'
    elif slip > .015:
        ok, why = False, f'slid {slip * 1000:.0f} mm in the hand'
    elif err > .015 or yaw_err > 8:
        ok, why = False, f'landed {err * 1000:.0f} mm / {yaw_err:.0f} deg off the {where}'
    return dict(ok=ok, why=why, place_err_mm=round(err * 1000, 1), yaw_err_deg=round(yaw_err, 1),
                tilt_deg=round(tilt, 1), slip_mm=round(slip * 1000, 1))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
    only = [a for a in sys.argv[1:] if not a.isdigit()]
    seqs = json.loads((HERE / 'plan11.json').read_text())['sequences']
    res_path = HERE / 'dyn11-results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() and only else {}
    for obj, kind in MENUS:
        key = f'{obj}_{kind}'
        if (only and key not in only) or key not in seqs:
            continue
        t0 = time.time()
        D = Dyn(obj, kind)
        seq = seqs[key]
        rng = np.random.default_rng(11)
        trials = []
        for i in range(n + 1):
            p = draw(obj, rng, nominal=(i == 0))
            D.set_params(p)
            r = trial(D, seq, p)
            r['params'] = dict(mass=round(p['mass'], 3), mu={k: round(v, 3) for k, v in p['mu'].items()},
                               place_mm=[round(v * 1000, 1) for v in p['place'][:2]], tt_deg=round(math.degrees(p['tt']), 1))
            trials.append(r)
            print(key, 'nominal' if i == 0 else i, 'OK' if r['ok'] else 'FAIL ' + r['why'],
                  {s: {k: r[s][k] for k in ('place_err_mm', 'slip_mm', 'tilt_deg')} for s in ('seg1', 'seg2') if s in r}, flush=True)
        rand = trials[1:]
        peaks = {}
        for r in trials:
            for s in ('seg1', 'seg2'):
                for j, v in r.get(s, {}).get('peak', {}).items():
                    peaks[j] = max(peaks.get(j, 0), v)
        holds = [r[s]['pitch_hold_Nm'] for r in trials for s in ('seg1', 'seg2') if s in r]
        results[key] = dict(trials=len(rand), passed=sum(r['ok'] for r in rand), nominal_ok=trials[0]['ok'],
                            failures=[r['why'] for r in rand if not r['ok']], peak_of_stall=peaks,
                            pitch_hold_mean_Nm=dict(max=max(holds), of_rated=round(max(holds) / .981, 2)) if holds else None,
                            runtime_s=round(time.time() - t0, 1), detail=trials)
        print(key, f"{results[key]['passed']}/{len(rand)} passed", flush=True)
        with open(HERE / '.dyn11.lock', 'w') as lock:   # menus run as parallel processes merge into one file
            fcntl.flock(lock, fcntl.LOCK_EX)
            cur = json.loads(res_path.read_text()) if only and res_path.exists() else {}
            cur[key] = results[key]
            results = cur
            res_path.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
