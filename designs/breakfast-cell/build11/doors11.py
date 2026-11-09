"""The whole morning in physics: does the hand really move the doors and carry the meal?

Replays a whole planned morning (plan11.json) on the dynamic model (model11.mjcf11(dynamic=True))
with the fridge door, the microwave door and the freezer flap free on their hinges and the jar
or box a free body. The doors move only when the hand pushes or pulls them through their pins,
the meal only when the hand lifts it. The turntable follows the plan. Servo goals follow the
planned path itself (dyn11.reference): the progress along the path is smoothed, the arm stops
at sharp bends, and each servo's lag is fed forward as calibrated on the bench.

Doors: mass, hinge friction and damping; the fridge seal pulls the door shut below 2 deg
(force at the outer pin, drawn 10-25 N, or set); the microwave door springs ajar to 10 deg after
its key (the planned key press stands in for it) and its catch resists the last 3 deg (8-20 N);
the flap falls open under its own weight past the tipping point onto its 95 deg stop and its
catch holds it shut below 3 deg (3-8 N at the loop). Servos as in dyn11.py. On a door pin the
jaw closes to its planned angle at its rated torque: a cage round the pin, the plate takes the
force. Holding the meal, the jaw squeezes at its rated torque. Meal: weight and friction drawn
as in dyn11.py (wet glass); it starts exactly where the plan puts it (dyn11.py tests placement
errors and re-planning from the camera).

Pass: every door step ends within 5 deg of the planned angle, both doors and the flap end shut
(< 1 deg), no servo other than the jaw sits at stall for more than 0.2 s, and the meal ends on
the serving ribs within 15 mm and 8 deg, upright, never slipping 15 mm in the hand.

    uv run python designs/breakfast-cell/build11/doors11.py [runs per menu, default 15] [menu ...]
    uv run python designs/breakfast-cell/build11/doors11.py --sweep     (seal force sweep, stock and STS3250 wrist yaw)
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
import dyn11 as Y  # noqa: E402
import model11 as X  # noqa: E402
P11 = Y.P11
M, S = P11.M, P11.S
PIN_SQUEEZE = 0.0   # rad past the planned jaw angle on a door pin, at stall torque. 0: the jaw closes to the planned angle
                    # at its rated torque, a cage round the pin (squeezing at stall fixed the pin where the plan did not expect it
                    # and the arm fought the hinge: 3-8 s of stalled wrist pitch per morning)
PIN_SOFT = 1.0      # shoulder and elbow stiffness while holding a door pin, share of normal (the servos' P gain register)
DOOR_STEPS = {'mw_open': 'mw_door', 'fridge_open': 'fridge_door', 'fridge_wide': 'fridge_door', 'flap_open': 'flap',
              'flap_close': 'flap', 'mw_close': 'mw_door', 'fridge_close': 'fridge_door', 'mw_reopen': 'mw_door', 'mw_shut': 'mw_door'}


class Track:
    """How far the arm strays from where the plan puts it at the same point of its path: the
    planned joints go on a kinematic copy, and the corners of every forearm and hand part
    are compared (the jaw is left out: it closes past the plan on purpose to squeeze)."""
    BODIES = ('forearm', 'wrist_yaw', 'wrist_roll', 'wrist_pitch')

    def __init__(self, m, D, free):
        self.m, self.D, self.dk = m, D, mujoco.MjData(m)
        self.geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in self.BODIES]
        self.box = [g for g in self.geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        self.signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        self.worst = {}

    def pts(self, d):
        out = [d.geom_xpos[self.geoms]]
        for g in self.box:
            out.append(d.geom_xpos[g] + (self.signs * self.m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T)
        return np.vstack(out)

    def check(self, d, q_plan, label):
        dk = self.dk
        dk.qpos[:] = d.qpos
        for i, j in enumerate(S.ARM):
            dk.qpos[self.D.jadr[j]] = q_plan[i]
        dk.qpos[self.D.jadr['cw']] = M.CW_C - q_plan[0]
        mujoco.mj_kinematics(self.m, dk)
        e = float(np.max(np.linalg.norm(self.pts(d) - self.pts(dk), axis=1)))
        self.worst[label] = max(self.worst.get(label, 0.0), e)


def draw(rng, nominal):
    u = (lambda lo, hi: (lo + hi) / 2) if nominal else (lambda lo, hi: float(rng.uniform(lo, hi)))
    nz = (lambda sd: 0.0) if nominal else (lambda sd: float(rng.normal(0, sd)))
    return dict(door_kg=u(4.0, 7.0), seal_N=u(10, 25), fric=u(.1, .5), pin_mu=u(.3, .6), mw_kg=u(1.5, 2.5), mw_catch_N=u(8, 20),
                flap_kg=u(.15, .3), flap_catch_N=u(3, 8), stiff={j: u(.6, 1.0) for j in X.DRIVES},
                offsets=np.array([nz(.0005)] + [nz(math.radians(.3)) for _ in range(5)] + [0.0]),
                tt=math.radians(6.0) if nominal else float(rng.uniform(-math.radians(8), math.radians(8))),
                place_u=np.zeros(3) if nominal else rng.uniform(-1, 1, 3),
                seen2=np.array([nz(.002), nz(.002), 0]), seen2_yaw=nz(math.radians(1.0)),
                meal=dict(mass={k: u(*v) for k, v in dict(jar=(.70, .80), box=(1.57, 1.87), tray=(.40, .46)).items()},
                          mu=dict(plate=u(.20, .30), pads=u(.35, .70), jawpetg=u(.20, .30), shelf=u(.20, .40), rails=u(.20, .30),
                                  ribs=u(.60, .90), plate_glass=u(.20, .30))))


def run(seq, obj, kind, p, trace=None, squeeze=PIN_SQUEEZE, soft=PIN_SOFT, meal=True, record=None, live=False):
    """One whole morning.
    live=False: the plan runs open-loop up to the hot take; there the robot sees the meal and the
    turntable again (camera error drawn), re-plans the hot take and the serve (dyn11.plan_hot),
    joins the plan with a checked joint move and runs the rest of it.
    live=True: the robot runs plan11.morning() itself: before every step and every door grasp it
    runs what it has planned so far in physics, then looks (door angles from the pins seen to
    2 mm, the turntable to 1 deg, the meal to 2 mm and 1 deg) and plans on from that. The meal
    starts off its spot by the drawn placement error.
    record: a list that receives (time, label, qpos) at the viewer's frame rate."""
    D = Y.Dyn(obj, kind)
    m, d = D.m, D.d
    fork = X.OBJECTS[obj]['mode'] == 'fork'
    D.set_params(dict(mu=p['meal']['mu'], mass=p['meal']['mass'][obj] if meal else D.base_mass, stiff=p['stiff']))
    names = [m.geom(i).name for i in range(m.ngeom)]
    for i, n in enumerate(names):
        if n in ('f_bar', 'f_bar2', 'm_bar') or n.startswith('flap_tab'):
            m.geom_friction[i, 0] = p['pin_mu']
    free = [j for j in ('fridge_door', 'mw_door', 'flap') if j in D.jadr]
    for j in free:                                     # unlock: no actuator force on these hinges
        a = D.act[j]
        m.actuator_gainprm[a, 0] = 0; m.actuator_biasprm[a, 1] = 0; m.actuator_biasprm[a, 2] = 0
        dof = D.dadr[j]
        m.dof_damping[dof], m.dof_frictionloss[dof], m.dof_armature[dof] = .3, p['fric'] if j != 'flap' else .02, .01
    for body, kg in (('f_door', p['door_kg']), ('m_door', p['mw_kg']), ('flap', p['flap_kg'])):
        if body in [m.body(i).name for i in range(m.nbody)]:
            b = m.body(body).id
            m.body_inertia[b] *= kg / m.body_mass[b]
            m.body_mass[b] = kg
    mujoco.mj_setConst(m, d)
    wp = real_heating(seq['wp'], p['tt'] if fork else None)
    w0 = wp[0]
    u = p['place_u']                                   # the meal stands off its spot: along its rails, or anywhere for a jar
    p['place'] = np.array([u[0] * .0015, u[1] * .010, 0]) if fork else np.array([u[0] * .010, u[1] * .010, 0])
    p['place_yaw'] = u[2] * math.radians(1.5) if fork else 0.0
    if meal:
        start = np.array(w0['jar'][:3]) + (p['place'] if live else 0)
        yaw0 = Y.yaw_of(w0['jar'][3:]) + (p['place_yaw'] if live else 0.0)
        D.place(start, M.quat_z(yaw0), np.array(w0['q']), w0['env'])
        if fork:                                       # the rails under the object's start, as plan11.Cell.put places them
            D.set_rails(np.r_[start[0], start[1], start[2] - X.RAILS['h'], M.quat_z(yaw0)])
        mujoco.mj_forward(m, d)
    else:
        D.place(np.array([0, 0, -3.0]), np.array([1., 0, 0, 0]), np.array(w0['q']), w0['env'])
    S_ = dict(t=0.0, stall={j: 0.0 for j in X.DRIVES}, dev={}, ajar_until=None, rel=[], rel_n=[], n=0, rec_next=0.0,
              track=Track(m, D, free), kp0={j: float(m.actuator_gainprm[D.act[j], 0]) for j in ('shoulder', 'elbow')},
              seal_arm=D.f['W'] - .05)        # seal_N is the straight pull at the outer pin, 5 cm from the free edge

    def drive(seg, tail=.3):
        seg = cap_turntable(seg)
        R = Y.reference(seg, tail=tail)
        for k in range(len(R['t'])):
            i0, i1, fr = R['i0'][k], R['i1'][k], R['frac'][k]
            step(R['goal'][k], R['plan'][k], seg[i0]['label'], Y.env_at(seg, i0, i1, fr), meal and seg[i0]['held'], k + 1 >= len(R['t']) or seg[R['i0'][k + 1]]['label'] != seg[i0]['label'])

    def step(goal, planned, label, env, held, last):
        t = S_['t']
        q = goal + p['offsets']
        on_pin = DOOR_STEPS.get(label) in ('fridge_door', 'mw_door') and abs(planned[6] - M.GRIP['bar']) < .02
        if held:
            q[6] += Y.OVERCLOSE                        # the jaw squeezes the meal at its rated torque
        ga = D.act['grip']
        if on_pin and squeeze:                         # (squeezing a pin at stall: kept as an option, PIN_SQUEEZE)
            q[6] += squeeze
            m.actuator_forcerange[ga] = [-2.94, 2.94]
        else:
            m.actuator_forcerange[ga] = [-X.DRIVES['grip'][2], X.DRIVES['grip'][2]]
        for j in ('shoulder', 'elbow'):                # a softer arm on a door pin (P gain register), 1.0 = off
            a = D.act[j]
            kp = S_['kp0'][j] * (soft if on_pin else 1.0)
            m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1] = kp, -kp
        for i, j in enumerate(S.ARM):
            d.ctrl[D.act[j]] = q[i]
        d.ctrl[D.act['turntable']] = env['turntable']
        # the microwave key: while the plan lets the door spring ajar, its spring pushes it to 10 deg
        if label in ('mw_open', 'mw_reopen') and env['mw_door'] > 1e-3 and d.qpos[D.jadr['mw_door']] < math.radians(9):
            S_['ajar_until'] = t + .5
        for _ in range(int(round(Y.DT_CTRL / m.opt.timestep))):
            d.qpos[D.jadr['cw']] = M.CW_C - d.qpos[D.jadr['lift']]
            d.qvel[D.dadr['cw']] = 0
            d.xfrc_applied[D.car, 2] = Y.CW_FORCE
            th = d.qpos[D.jadr['fridge_door']]
            d.qfrc_applied[D.dadr['fridge_door']] = -p['seal_N'] * S_['seal_arm'] * max(0.0, 1 - th / math.radians(2))
            th = d.qpos[D.jadr['mw_door']]
            tq = -p['mw_catch_N'] * .29 * max(0.0, 1 - th / math.radians(3))
            if S_['ajar_until'] is not None and t < S_['ajar_until'] and th < math.radians(10):
                tq = p['fric'] + .15                  # spring: the door pops open by itself, about 10 deg
            d.qfrc_applied[D.dadr['mw_door']] = tq
            if 'flap' in D.jadr:
                th = d.qpos[D.jadr['flap']]
                d.qfrc_applied[D.dadr['flap']] = -p['flap_catch_N'] * .07 * max(0.0, 1 - th / math.radians(3))
            mujoco.mj_step(m, d)
        for j, cap in ((j, X.DRIVES[j][2]) for j in X.DRIVES):
            if j != 'grip' and abs(d.actuator_force[D.act[j]]) >= cap * .999:
                S_['stall'][j] += Y.DT_CTRL
        if trace is not None:
            trace.append((round(t, 2), label, {j: round(math.degrees(d.qpos[D.jadr[j]]), 1) for j in free},
                          {j: round(math.degrees(env[j]), 1) for j in free if j in env},
                          {j: round(abs(d.actuator_force[D.act[j]]) / X.DRIVES[j][2], 2) for j in X.DRIVES}))
        if S_['n'] % 5 == 0:
            S_['track'].check(d, planned, label)
            if held:
                Rg = d.xmat[D.gripper].reshape(3, 3)
                S_['rel'].append(Rg.T @ (d.xpos[D.obj_body] - d.xpos[D.gripper])); S_['rel_n'].append(S_['n'])
        if record is not None and t >= S_['rec_next'] - 1e-9:
            record.append((round(t, 3), label, [round(float(v), 4) for v in d.qpos]))
            S_['rec_next'] += 1 / P11.FPS
        jn = DOOR_STEPS.get(label)
        if jn and jn in D.jadr and last:
            S_['dev'][label] = round(math.degrees(d.qpos[D.jadr[jn]] - env[jn]), 1)
        S_['t'] += Y.DT_CTRL
        S_['n'] += 1

    k_hot = next((k for k, w in enumerate(wp) if w['label'] == 'take_hot'), None)
    k_shut = next((k for k, w in enumerate(wp) if w['label'] == 'mw_shut'), None)
    replanned = None
    if live:
        replanned, wp = live_morning(obj, kind, p, D, drive, fork)
    elif not meal or k_hot is None:
        drive(wp)
    else:
        drive(wp[:k_hot + 1])
        pos, quat = D.obj_pose()                       # the camera sees the meal and the turntable again
        env = {j: float(d.qpos[D.jadr[j]]) if j in D.jadr else v for j, v in wp[k_hot]['env'].items()}
        try:
            P2, _, _ = Y.plan_hot(obj, kind, np.array(wp[k_hot]['q']), env, pos, p['seen2'], p['seen2_yaw'], np.array(w0['q']))
            P2.step('mw_shut', 'Back to the plan: close the microwave door and park')
            P2.joint_to(np.array(wp[k_shut]['q']))
        except RuntimeError as e:
            replanned = f'hot take not planned: {e}'
        if replanned is None:
            seg = [dict(t=w['t'], q=list(w['q']), env=w['env'], held=w['held'], label=w['label']) for w in P2.wp]
            drive(seg, tail=0.0)
            rest = [dict(w, env=dict(w['env'], turntable=env['turntable'])) for w in wp[k_shut:]]
            drive(rest)
    final = {j: round(math.degrees(d.qpos[D.jadr[j]]), 1) for j in free}
    worst_stall = max(S_['stall'].values())
    ok = replanned is None and all(abs(v) <= 5 for v in S_['dev'].values()) and all(abs(v) < 1 for v in final.values()) and worst_stall <= .2
    meal_res = judge_meal(D, wp, S_['rel'], S_['rel_n'], fork) if meal and replanned is None else None
    if meal_res is not None and not meal_res['ok']:
        ok = False
    return dict(ok=ok, meal=meal_res, replan_error=replanned, step_end_error_deg=S_['dev'], final_deg=final,
                stalled_s={j: round(v, 2) for j, v in S_['stall'].items() if v},
                off_path_mm={k: round(v * 1000, 1) for k, v in S_['track'].worst.items()}, duration_s=round(S_['t'], 1),
                params={k: (round(v, 3) if isinstance(v, float) else None) for k, v in p.items() if isinstance(v, float)})


TT_SPEED = .55     # rad/s, a microwave turntable's 5 rpm
SEE_PIN = .002     # m: the wrist camera sees a door pin this well before the hand goes for it


def live_morning(obj, kind, p, D, drive, fork):
    """plan11.morning() with a planner that senses: returns (error or None, the waypoints run)."""
    d, rng = D.d, np.random.default_rng(abs(hash((p['seal_N'], p['tt']))) % 2**32)
    done = dict(k=0)

    def sense(P):
        seg = P.wp[done['k']:]
        if len(seg) > 1:
            drive([dict(t=w['t'], q=list(w['q']), env=w['env'], held=w['held'], label=w['label']) for w in seg], tail=0.0)
            done['k'] = len(P.wp) - 1
        waited = 0.0                                   # a door still swinging (popped ajar, let go of) is watched until it stops
        while waited < 3.0 and any(abs(d.qvel[D.dadr[j]]) > math.radians(.5) for j in ('fridge_door', 'mw_door') if j in D.dadr):
            w = P.wp[-1]
            drive([dict(t=0.0, q=list(w['q']), env=w['env'], held=w['held'], label=w['label']),
                   dict(t=.25, q=list(w['q']), env=w['env'], held=w['held'], label=w['label'])], tail=0.0)
            waited += .25
        c = P.c
        for j, r in (('fridge_door', D.f['W'] - .05), ('mw_door', .29), ('flap', .065)):
            if j in c.env and j in D.jadr:              # a shut door is flush: that it is shut is plain to see
                th = float(d.qpos[D.jadr[j]])
                c.env[j] = 0.0 if abs(th) < math.radians(.5) else th + float(rng.normal(0, SEE_PIN / r))
        c.env['turntable'] = float(d.qpos[D.jadr['turntable']] + rng.normal(0, math.radians(1)))
        if not c.held:
            pos, quat = D.obj_pose()
            c.jar_pos = pos + np.r_[rng.normal(0, .002, 2), 0]
            c.jar_quat = M.quat_z(Y.yaw_of(quat) + rng.normal(0, math.radians(1)))

    class Live(P11.Planner):
        dry = False

        def step(self, label, text):
            if self.wp:
                sense(self)
            super().step(label, text)

        def swing(self, door, bar, th1, need, score=None):
            if not self.dry:
                sense(self)
            self.starts.append((len(self.wp), bar))
            got = super().swing(door, bar, th1, need, score)
            self.last_swing = (door, bar, got, need)
            return got

        def release_handle(self):
            """Let go, look at the door, and if it ended more than 1 deg off where the swing put it
            (the caged pin lags the hand), take the pin again and finish the swing (twice at most)."""
            super().release_handle()
            if self.dry:
                self.released.append(len(self.wp))
            if self.dry or getattr(self, 'last_swing', None) is None:
                return
            door, bar, got, need = self.last_swing
            self.last_swing = None
            for _ in range(2):
                sense(self)
                if abs(self.c.env[door] - got) < math.radians(1):
                    break
                snap = (self.q.copy(), len(self.wp), self.t)
                try:
                    super().swing(door, bar, got, max(need, math.radians(2)))
                    super().release_handle()
                except RuntimeError:                   # no grasp from here (e.g. at the pin's reach limit): the next
                    self.q, self.t = snap[0], snap[2]  # step plans from the door as seen
                    del self.wp[snap[1]:]
                    break

        def hand_over(self, outer_deg):
            """Planned dry (it tries hand-over angles and rolls failed tries back), then run up
            to letting go of the inner pin and look: a door that ended more than 1 deg off is
            taken again by the inner pin. Only then the way out of the door's wedge is planned,
            from where the door really is (a door 2 deg short of the plan stood in the dry plan's
            way out, and the hand pushed it back open)."""
            self.starts, self.released, self.dry = [], [], True
            try:
                at = super().hand_over(outer_deg)
            finally:
                self.dry = False
            k_out = max(i for i, bar in self.starts if bar == 'f_bar' and i < len(self.wp))
            k_in = max(i for i, bar in self.starts if bar == 'f_bar2' and i < k_out)
            k = min(i for i in self.released if k_in < i <= k_out) - 1
            del self.wp[k + 1:]
            w = self.wp[-1]
            self.q, self.t = np.array(w['q']), w['t']
            self.c.env.update(w['env'])
            def regrip(deg):                           # the inner pin again, the door to deg, let go and look
                snap = (self.q.copy(), len(self.wp), self.t)
                self.inner = True
                try:
                    P11.Planner.swing(self, 'fridge_door', 'f_bar2', math.radians(deg), math.radians(1))
                    P11.Planner.release_handle(self)
                    return True
                except RuntimeError:
                    self.q, self.t = snap[0], snap[2]
                    del self.wp[snap[1]:]
                    return False
                finally:
                    self.inner = False
                    sense(self)

            sense(self)                                # did the door stay where the inner pin left it?
            if abs(self.c.env['fridge_door'] - math.radians(at)) >= math.radians(2):
                regrip(at)
            for push in (0, 2, 4):                     # a door that ended short can wedge the hand: push it back out
                if push and not regrip(at + push):
                    continue
                try:
                    self.joint_to(self.park)
                    break
                except RuntimeError:
                    if push == 4:
                        raise
            self.swing('fridge_door', 'f_bar', 0.0, math.radians(.5))
            self.release_handle()
            return at

    Live.starts = []
    try:
        c, P, out = P11.morning(obj, kind, planner=Live, residual=p['tt'])
    except RuntimeError as e:
        return f'{P11.Planner.last}: {e}', None
    seg = P.wp[done['k']:]
    if len(seg) > 1:
        drive([dict(t=w['t'], q=list(w['q']), env=w['env'], held=w['held'], label=w['label']) for w in seg])
    return None, [dict(jar=list(w['jar'])) for w in P.wp]


def cap_turntable(wp):
    """Waypoints re-timed so the turntable never turns faster than TT_SPEED."""
    out, shift = [], 0.0
    for k, w in enumerate(wp):
        if k:
            dt = w['t'] - wp[k - 1]['t']
            dth = abs(w['env']['turntable'] - wp[k - 1]['env']['turntable'])
            if dt > 0 and dth / dt > TT_SPEED:
                shift += dth / TT_SPEED - dt
        out.append(dict(w, t=w['t'] + shift))
    return out


def real_heating(wp, residual=None):
    """The plan shows heating sped up (3.5 turns in 4 s); physics turns the turntable at 5 rpm.
    residual: where the Shelly cut-off really stops it, past the half turn (rad); None keeps the plan's."""
    heat = [k for k, w in enumerate(wp) if w['label'] == 'heat']
    th0, th1 = wp[heat[0] - 1]['env']['turntable'], wp[heat[-1]]['env']['turntable']
    new1 = th1 if residual is None else 3 * math.pi + residual
    out, shift = [], 0.0
    for k, w in enumerate(wp):
        env = dict(w['env'])
        if k >= heat[0]:
            env['turntable'] = th0 + (w['env']['turntable'] - th0) * (new1 - th0) / (th1 - th0) if k <= heat[-1] else new1
        if k:
            dt = w['t'] - wp[k - 1]['t']
            dth = abs(env['turntable'] - out[-1]['env']['turntable'])
            if dt > 0 and dth / dt > TT_SPEED:
                shift += dth / TT_SPEED - dt
        out.append(dict(w, t=w['t'] + shift, env=env))
    return out


def judge_meal(D, wp, rel, rel_held, fork):
    """Where the meal ended against where the plan put it, and the worst slip in the hand
    during any one carry (each carry is judged from its own settled grip)."""
    pos, quat = D.obj_pose()
    target, tq = np.array(wp[-1]['jar'][:3]), np.array(wp[-1]['jar'][3:])
    slip, runs = 0.0, []
    for r, n in zip(rel, rel_held):
        if runs and n - runs[-1][-1][0] <= 5:
            runs[-1].append((n, r))
        else:
            runs.append([(n, r)])
    for run_ in runs:
        if len(run_) > 9:
            ref = run_[8][1]
            slip = max(slip, max(float(np.linalg.norm(r - ref)) for _, r in run_[8:]))
    err = float(np.linalg.norm((pos - target)[:2]))
    dz = float(pos[2] - target[2])
    yaw_err = abs(math.degrees(Y.wrap(Y.yaw_of(quat) - Y.yaw_of(tq)))) if fork else 0.0
    yaw_err = min(yaw_err, 180 - yaw_err)
    tilt = Y.tilt_deg(quat)
    why = ('dropped' if dz < -.02 else f'tipped over ({tilt:.0f} deg)' if tilt > 5 else
           f'slid {slip * 1000:.0f} mm in the hand' if slip > .015 else
           f'landed {err * 1000:.0f} mm / {yaw_err:.0f} deg off the serving spot' if err > .015 or yaw_err > 8 else '')
    return dict(ok=not why, why=why, place_err_mm=round(err * 1000, 1), yaw_err_deg=round(yaw_err, 1), tilt_deg=round(tilt, 1),
                slip_mm=round(slip * 1000, 1))


YAW = dict(stock=None, sts3250=4.90)     # wrist yaw stall torque, N·m: STS3215 (2.94) or STS3250 (50 kg·cm)


def use_yaw(option):
    """Set the wrist yaw servo for the runs that follow (the model is built per run)."""
    kp, kv, cap, arm, damp, fl = X.DRIVES_STOCK['yaw']
    cap2 = YAW[option] or cap
    X.DRIVES['yaw'] = (kp * cap2 / cap, kv * cap2 / cap, cap2, arm, damp, fl)


def sweep():
    """Seal force at the outer pin against the arm: nominal everything else, jar morning, 90 cm."""
    seq = json.loads((HERE / 'plan11.json').read_text())['sequences']['jar_large']
    out = {}
    for option in YAW:
        use_yaw(option)
        out[option] = []
        for seal in (10, 14, 17.5, 20, 23, 26, 29, 32):
            p = draw(np.random.default_rng(7), True); p['seal_N'] = float(seal)
            r = run(seq, 'jar', 'large', p, live=True)
            opened = abs(r['step_end_error_deg'].get('fridge_open', 99)) <= 5
            out[option].append(dict(seal_N=seal, opened=opened, ok=r['ok']))
            print(option, seal, 'opened' if opened else 'stuck', 'OK' if r['ok'] else 'FAIL', flush=True)
    use_yaw('stock')
    return out


def main():
    """doors11.py [runs per menu] [menu ...] [--yaw sts3250] [--cal 0.5] [--sweep]: named menus merge into the saved results.
    --cal scales the drawn joint zero errors (0.3 deg, 0.5 mm) by the factor: 0.5 is a robot calibrated to 0.15 deg."""
    res_path = HERE / 'doors11-results.json'
    if '--sweep' in sys.argv:
        r = sweep()
        with open(HERE / '.doors11.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cur = json.loads(res_path.read_text()) if res_path.exists() else {}
            cur['sweep'] = r
            res_path.write_text(json.dumps(cur, indent=1))
        return
    option = sys.argv[sys.argv.index('--yaw') + 1] if '--yaw' in sys.argv else 'stock'
    use_yaw(option)
    cal = float(sys.argv[sys.argv.index('--cal') + 1]) if '--cal' in sys.argv else 1.0   # joint zero errors scaled by this
    args = [a for a in sys.argv[1:] if a not in ('--yaw', option, '--cal') and not (cal != 1.0 and a == sys.argv[sys.argv.index('--cal') + 1])]
    n = int(args[0]) if args and args[0].isdigit() else 15
    only = [a for a in args if not a.isdigit()]
    seqs = json.loads((HERE / 'plan11.json').read_text())['sequences']
    for key in ('jar_large', 'box_large', 'tray_large', 'jar_small', 'box_small'):
        if (only and key not in only) or key not in seqs:
            continue
        seq = seqs[key]
        obj, kind = seq['obj'], seq['kind']
        rng = np.random.default_rng(7)
        t0 = time.time()
        runs = []
        for i in range(n + 1):
            p = draw(rng, nominal=(i == 0))
            p['offsets'] = p['offsets'] * cal
            r = run(seq, obj, kind, p, live=True)
            runs.append(r)
            print(key, option, 'nominal' if i == 0 else i, 'OK' if r['ok'] else 'FAIL', r['meal'] and r['meal']['why'], r['replan_error'],
                  {k: v for k, v in r['step_end_error_deg'].items() if abs(v) > 5}, r['stalled_s'], flush=True)
        res = dict(runs=n, passed=sum(r['ok'] for r in runs[1:]), nominal_ok=runs[0]['ok'], yaw=option, cal=cal, runtime_s=round(time.time() - t0, 1),
                   off_path_mm_max=max(max(r['off_path_mm'].values()) for r in runs), detail=runs)
        name = (key if option == 'stock' else f'{key}+{option}') + ('' if cal == 1.0 else '+cal')
        with open(HERE / '.doors11.lock', 'w') as lock:  # menus run as parallel processes merge into one file
            fcntl.flock(lock, fcntl.LOCK_EX)
            cur = json.loads(res_path.read_text()) if res_path.exists() else {}
            cur[name] = res
            res_path.write_text(json.dumps(cur, indent=1))


if __name__ == '__main__':
    main()
