"""Whole mornings on the build11 model (model11.py), planned at real servo speeds and checked
in every frame with build10's collision rule (3 mm between the robot and everything,
itself included; the held object may touch only what it rests on).

Menus:
- jar:  Weck jar, side grip (build10's morning with build11's hand); 90 and 50 cm fridge.
- box:  1 L IKEA glass box on rails, fork (hand rolled -90 deg); fridge door to 100 deg by
        the inner pin; ribs on the turntable; 90 and 50 cm fridge.
- tray: frozen lasagna tray on rails in the freezer compartment, fork; the flap opened by
        its loop tab and pushed shut; 90 cm fridge only (the 50 cm one has no freezer).

Heating turns the turntable. For the box and tray the Shelly plug cuts the power when the
turntable is at 0 or 180 deg (ribs are symmetric), so the object comes back at 90 deg
plus a residual; the morning uses +6 deg, dyn11.py samples +-8 deg.

Also the shelf (and freezer) coverage of each object with this hand.
Writes plan11.json (frames for the viewer and the dynamic check) and plan11-results.json.

    uv run python designs/breakfast-cell/build11/plan11.py
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
sys.path.insert(0, str(HERE.parent / 'build10'))
import model11 as X  # noqa: E402
import model3d as M  # noqa: E402
import sequence as S  # noqa: E402

ORIG_SOLUTIONS = S.solutions
ROLL = {'jar': 0.0}
FPS = 15
TT_RESIDUAL = math.radians(6.0)
PIN_CLEAR = .08                  # under a 90 mm door pin before rising onto it / after letting go
PLATE_PULL = .85                 # breaking the seal: the plate must face the pull at least this much (cosine)
SEAL_ZONE = math.radians(5)      # ... while the door is still within this of shut
SIDE_CLEAR = .008                # the plate passes 8 mm from a jar until the jaw closes (4 mm let calibration error push jars away)
UNDER = .003                     # fork this far under the object's bottom on the way in and out
FLAP_STOP = math.radians(95)      # the open flap rests on its stop a little below level
MW_OPEN = 75.0                   # beyond this the pin can't be gripped afresh to close the door
PIN_TOUCH = .0015                # the fingers may touch the pin in use (its fit is computed to the mm)
PIN_SIDE = .006                  # rising onto a pin, the plate stays this far off it (the pin centred in the open jaw)
DOOR_CLEAR = .008                # going for a door pin and moving the door, the hand keeps this from the doors (they settle and flex)
HELD_CLEAR = .003                # a carried meal keeps this from everything but the hand and the surface it is set on
DOOR_FACES = ('f_door_slab', 'f_door_racks', 'm_door_slab')
PIN_DOOR = ('f_bar', 'f_bar2', 'm_bar')
START = {   # object's bottom centre at the start of each menu, and its heading
    ('jar', 'large'): ((-.16, .15, .37 + .0005), None), ('jar', 'small'): ((-.16, .15, .21 + .0005), None),
    ('box', 'large'): ((-.405, .175, .37 + X.RAILS['h'] + .0005), 90.0), ('box', 'small'): ((-.275, .175, .21 + X.RAILS['h'] + .0005), 90.0),
    ('tray', 'large'): ((-.43, .19, X.FREEZER['floor'] + .01 + X.RAILS['h'] + .0005), 90.0),
}
NEEDS_100 = {'jar', 'box', 'tray'}


def pin_open():
    """Jaw angle while rising onto a door pin."""
    return M.GRIP['bar'] - .18          # about 12 mm either side of the pin


def openings():
    """Jaw openings to try on the way in: wide, 60 %, and just clear (about 15 mm at the tip)."""
    return (M.GRIP['open'], M.GRIP['open'] * .6, M.GRIP['jar'] - .12)


def solutions(g, psi, tool, grip, roll=None, pitch=0.0):
    r = ROLL.get(tool, 0.0) if roll is None else roll
    return ORIG_SOLUTIONS(g, psi, tool, grip, r, pitch)


S.solutions = solutions


class Cell(S.Cell):
    def __init__(self, kind, obj):
        self.obj = obj
        self.cfg = X.configure(obj)
        ROLL['jar'] = self.cfg['roll']
        xml, f, _ = X.mjcf11(kind, obj)
        self.kind, self.f, self.xml = kind, f, xml
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.adr = {self.m.joint(i).name: int(self.m.jnt_qposadr[i]) for i in range(self.m.njnt)}
        self.gname = [self.m.geom(i).name for i in range(self.m.ngeom)]
        self.gbody = [self.m.body(self.m.geom_bodyid[i]).name for i in range(self.m.ngeom)]
        self.is_jar = np.array([b == 'jar' for b in self.gbody])
        self.is_hand = np.array([b in S.HAND_BODIES for b in self.gbody])
        sup = S.SUPPORTS + ('m_top_rib', 'flap_slab')
        self.is_support = np.array([any(n.startswith(s) for s in sup) for n in self.gname])
        self.is_robot = np.array([b in S.ROBOT_BODIES for b in self.gbody])
        self.pitch = self.m.body('wrist_pitch').id
        self.env = dict(fridge_door=0.0, mw_door=0.0, turntable=0.0)
        if 'flap' in self.adr:
            self.env['flap'] = 0.0
        self.held = False
        self.jar_pos, self.jar_quat = np.array([0, 0, -3.0]), np.array([1., 0, 0, 0])
        self.rails = np.r_[0, 0, -4.0, 1, 0, 0, 0]
        self._strict = []
        self.fingers = {self.gname[i] for i in range(self.m.ngeom) if self.m.geom_contype[i] == 8}
        self.meal = {self.gname[i] for i in range(self.m.ngeom) if self.is_jar[i]}
        self.supports = {self.gname[i] for i in range(self.m.ngeom) if self.is_support[i]} | \
            {n for n in self.gname if 'rail' in n or 'rib' in n}
        self.allow = {}          # handle geom -> how far the fingers may press into it (m), while it is in use
        self.at_door = False     # going for a door pin or moving the door: DOOR_CLEAR from the doors (they settle and flex)

    def contacts(self):
        """sequence.py's contacts, less the fingers touching the one pin, key or tab in use."""
        con = super().contacts()
        if not self.allow or not con:
            return con
        keep, strict = [], []
        for (dd, a, b), s in zip(con, self._strict):
            h = b if a in self.fingers else a if b in self.fingers else None
            if h in self.allow and dd > -self.allow[h]:
                continue
            keep.append((dd, a, b)); strict.append(s)
        self._strict = strict
        return keep

    def ok(self, q, env=None):
        """sequence.py's rule (3 mm from everything, the held or resting meal excepted), and while
        the hand moves a door by its pin, DOOR_CLEAR from that door: doors lag and flex."""
        if not S.limits_ok(q):
            return False
        self.apply(q, env)
        con = self.contacts()
        moving = self.at_door or any(k in PIN_DOOR for k in self.allow)
        for (dd, a, b), strict in zip(con, self._strict):
            if strict:
                lim = DOOR_CLEAR if moving and (a in DOOR_FACES or b in DOOR_FACES) else S.CLEAR
            elif self.held and (a in self.meal) != (b in self.meal) and (b if a in self.meal else a) not in self.supports:
                lim = HELD_CLEAR                         # a carried meal keeps clear of all but what it is set on
            else:
                lim = -1e-4
            if dd <= lim:
                return False
        return True

    def apply(self, q, env=None):
        d = self.d
        for i, n in enumerate(S.ARM):
            d.qpos[self.adr[n]] = q[i]
        for k, v in (env or self.env).items():
            d.qpos[self.adr[k]] = v
        d.qpos[self.adr['cw']] = M.CW_C - q[0]
        a = self.adr['rails']
        d.qpos[a:a + 7] = self.rails
        mujoco.mj_kinematics(self.m, d)
        if self.held:
            R = d.xmat[self.pitch].reshape(3, 3)
            tool = d.xpos[self.pitch] + R @ np.array([M.JAR_IN_PITCH, 0, 0])
            heading = math.atan2(R[1, 0], R[0, 0])
            pos = tool - [0, 0, self.cfg['tool_h']]
            quat = M.quat_z(heading - math.pi / 2)
        else:
            pos, quat = self.jar_pos, self.jar_quat
        a = self.adr['jar']
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, quat
        mujoco.mj_kinematics(self.m, d)
        return pos, quat

    def put(self, base, psi, rails=False):
        self.held = False
        self.jar_pos = np.asarray(base, float)
        self.jar_quat = M.quat_z(psi - math.pi / 2)
        if rails:
            self.rails = np.r_[base[0], base[1], base[2] - X.RAILS['h'], self.jar_quat]


def heading_of(quat):
    return 2 * math.atan2(quat[3], quat[0]) + math.pi / 2


class Planner(S.Planner):
    inner = False
    last = None

    def step(self, label, text):
        Planner.last = label
        super().step(label, text)

    def _push(self, dt, q=None):
        super()._push(dt, q)
        self.wp[-1]['rails'] = self.c.rails.copy()

    def _door_normal(self, door, th):
        n = np.array([-math.sin(th), -math.cos(th), 0.0])
        return -n if self.inner else n

    def ok_on(self, bar, q, env=None):
        """Collision-free with the fingers closed on this pin."""
        saved, self.c.allow = self.c.allow, {bar: PIN_TOUCH}
        try:
            return self.c.ok(q, env)
        finally:
            self.c.allow = saved

    def pin_approach(self, q):
        """For a grip q on a pin: the pose beside the pin at grip height with the jaw open, the
        plate PIN_SIDE off the pin (the pin centred in the opening) or, where the door leaves no
        room for that, nearer down to brushing it (the pin's printed foot is chamfered); and the
        pose PIN_CLEAR under it. Returns (under, beside, grip point, heading, side) or None."""
        psi = float(q[1] - math.pi / 2 + q[2] + q[3])
        y = np.array([-math.sin(psi), math.cos(psi), 0])
        self.q, q_saved = q, self.q
        g = self.tool_point('bar')
        self.q = q_saved
        bar = next(iter(self.c.allow), None)
        for side in (PIN_SIDE, PIN_SIDE / 2, PIN_SIDE / 4, 0.0):
            saved, self.c.allow = self.c.allow, ({} if side > S.CLEAR + PIN_TOUCH else self.c.allow)
            try:
                qs = self.choose(g + side * y, psi, 'bar', pin_open(), q, max_jump=math.radians(10))
            finally:
                self.c.allow = saved
            if qs is None:
                continue
            qo = qs.copy(); qo[0] -= PIN_CLEAR
            saved, self.c.allow = self.c.allow, {}
            try:
                if self.c.ok(qo):
                    return qo, qs, g, psi, side
            finally:
                self.c.allow = saved
        return None

    def take_pin(self, bar, approach):
        """Under the pin, rise beside it, slide the plate onto it, close the jaw."""
        qo, qs, g, psi, side = approach
        self.joint_to(qo)
        if side <= S.CLEAR + PIN_TOUCH:
            self.c.allow = {bar: PIN_TOUCH}
        self.joint_to(qs, v_scale=.5)
        self.c.allow = {bar: PIN_TOUCH}
        y = np.array([-math.sin(psi), math.cos(psi), 0])
        if side > 0 and not self.line(g + side * y, g, psi, psi, 'bar', pin_open(), v=S.V_FINE):
            raise RuntimeError(f'cannot slide onto {bar}')
        self.grip_to(M.GRIP['bar'])

    def joint_to(self, target, v_scale=1.0, check=True):
        """sequence.py's straight joint move (with its lift-first, arm-first and tucked detours);
        when all of those collide, a bidirectional random tree in joint space (RRT-connect)
        with the same collision rule, then shortcut smoothing."""
        try:
            return super().joint_to(target, v_scale, check)
        except RuntimeError:
            if not check:
                raise
        path = None
        for seed in range(3):
            path = self.rrt(self.q.copy(), np.asarray(target, float), iters=2500, seed=seed)
            if path is not None:
                break
        if path is None:
            raise RuntimeError(f'joint move blocked in step "{self.label}"')
        prev = self.q
        for q in path[1:]:
            n = max(2, int(np.max(np.abs(q - prev) / np.array([.002, .009, .009, .009, .009, .009, .02]))) + 1)
            for s_ in np.linspace(0, 1, n)[1:]:
                qq = prev + (q - prev) * s_
                self._push(S.tdist(qq, self.q) / v_scale, qq)
            prev = q
        return True

    def rrt(self, a, b, iters=4000, seed=0):
        rng = np.random.default_rng(seed)
        lo = np.array([S.M.LIMITS[n][0] for n in S.ARM]); hi = np.array([S.M.LIMITS[n][1] for n in S.ARM])
        lo[6], hi[6] = min(a[6], b[6]), max(a[6], b[6]) + 1e-9         # jaw only between its two end angles
        scale = np.array([.01, .05, .05, .05, .05, .05, .05])         # one step: 1 cm lift, 3 deg joints

        def free(q0, q1):
            n = max(2, int(np.max(np.abs(q1 - q0) / np.array([.002, .009, .009, .009, .009, .009, .02]))) + 1)
            return all(self.c.ok(q0 + (q1 - q0) * t) for t in np.linspace(0, 1, n)[1:])

        if not self.c.ok(b):
            return None
        T = [[a], [b]]
        par = [[-1], [-1]]

        def extend(side, r):
            tree = np.array(T[side])
            k = int(np.argmin(np.sum(((tree - r) / scale) ** 2, axis=1)))
            d = (r - tree[k]) / scale
            nd = np.max(np.abs(d))
            new = r.copy() if nd <= 1 else tree[k] + d / nd * scale
            if not free(tree[k], new):
                return None
            T[side].append(new); par[side].append(k)
            return 'reached' if nd <= 1 else 'advanced'

        def chain(side, k):
            out = []
            while k >= 0:
                out.append(T[side][k]); k = par[side][k]
            return out

        for it in range(iters):
            A = it % 2
            r = rng.uniform(lo, hi) if rng.random() > .1 else (b if A == 0 else a)
            if extend(A, r) is None:
                continue
            q_new = T[A][-1]
            while True:                                  # connect the other tree to q_new
                st = extend(1 - A, q_new)
                if st != 'advanced':
                    break
            if st == 'reached':
                ia, ib = (len(T[0]) - 1, len(T[1]) - 1)
                path = chain(0, ia)[::-1] + chain(1, ib)
                path = [q for i, q in enumerate(path) if i == 0 or np.max(np.abs(q - path[i - 1])) > 1e-12]
                i = 0                                    # shortcut
                while i < len(path) - 2:
                    j = len(path) - 1
                    while j > i + 1 and not free(path[i], path[j]):
                        j -= 1
                    path = path[:i + 1] + path[j:]
                    i += 1
                return path
        return None

    def slide_out(self):
        """After letting go of the inner pin: the hand slides along the door's inner face towards
        its free edge, a few cm off the face, until the arm is clear of the wedge between door
        and cabinet; heading fixed, jaw open."""
        th = self.c.env['fridge_door']
        u = np.array([math.cos(th), -math.sin(th), 0])               # hinge -> free edge
        nrm = np.array([math.sin(th), math.cos(th), 0])               # off the inner face
        psi = self.heading()
        p = self.tool_point('bar')
        lead = math.atan2(u[1], u[0])                                  # hand turning to lead the way out
        for off in (.03, .05, .015):
            for dist in (.40, .34, .28):
                for psi1 in (psi, lead, lead + .5, lead - .5, lead + 1.0):
                    psi1 = psi + (psi1 - psi + math.pi) % (2 * math.pi) - math.pi
                    snap = (self.q.copy(), len(self.wp), self.t)
                    a = p + off * nrm
                    if self.line(p, a, psi, psi, 'bar', M.GRIP['open'], v=S.V_FINE * 2) and \
                            self.line(a, a + dist * u, psi, psi1, 'bar', M.GRIP['open'], v=S.V_CART):
                        return True
                    self.q, self.t = snap[0], snap[2]
                    del self.wp[snap[1]:]
        raise RuntimeError('cannot slide out of the door wedge')

    def release_moves(self, q, drop=None):
        """Let go of a door pin without opening the jaw on it: the closed hand drops until the
        pin has slid out of the bottom of the fingers (a jaw opened against the pin sweeps its tip
        past the plate's and can knock the door), then backs 4 cm straight away from the door."""
        drop = PIN_CLEAR if drop is None else drop
        q = np.asarray(q, float).copy()
        out = []
        for k in range(1, 11):
            qq = q.copy(); qq[0] -= drop * k / 10
            if not self.c.ok(qq):
                return None
            out.append(qq)
        psi = float(q[1] - math.pi / 2 + q[2] + q[3])
        h = np.array([math.cos(psi), math.sin(psi), 0])
        self.c.apply(out[-1])
        yb = self.c.m.body('wrist_yaw').id
        p0 = self.c.d.xpos[yb] + self.c.d.xmat[yb].reshape(3, 3) @ M.TOOL['bar']
        self._n_drop = len(out)
        saved, self.c.allow = self.c.allow, {}
        try:
            for back in (.04, .02):                   # back off if there is room; later moves open the jaw (checked)
                prev, extra = out[-1], []
                for k in range(1, 6):
                    qq = self.choose(p0 - back / 5 * k * h, psi, 'bar', q[6], prev, max_jump=math.radians(10))
                    if qq is None:
                        break
                    extra.append(qq); prev = qq
                else:
                    return out + extra
            if not self.c.ok(out[-1]):
                return None                           # still touching the pin after the drop
            return out
        finally:
            self.c.allow = saved

    def release_handle(self):
        moves = self.release_moves(self.q)
        if moves is None:
            raise RuntimeError('cannot let go of the handle')
        prev = self.q
        for k, q in enumerate(moves):
            if k == self._n_drop:
                self.c.allow = {}
            self._push(max(.005 / S.V_FINE, S.tdist(q, prev)), q)
            prev = q
        self.c.allow = {}

    def retreat(self, dist=.08):
        """Back straight out along the hand, jaw open, e.g. after letting go of the inner pin."""
        psi = self.heading()
        h = np.array([math.cos(psi), math.sin(psi), 0])
        p = self.tool_point('bar')
        for d in (dist, dist / 2):
            if self.line(p, p - d * h, psi, psi, 'bar', M.GRIP['open'], v=S.V_FINE * 2):
                return True
        return False

    def hand_over(self, outer_deg):
        """Inner pin back from 100 deg, then the outer pin takes over: tries handing over 6 deg
        inside the outer pin's reach limit first (a door that ends a little off can still be
        gripped), then other angles. Returns the angle used."""
        for deg in (outer_deg - 6, outer_deg - 12, outer_deg, outer_deg - 18, outer_deg - 24, outer_deg - 30):   # margin first
            snap = (self.q.copy(), len(self.wp), self.t, dict(self.c.env))
            try:
                self.inner = True
                self.swing('fridge_door', 'f_bar2', math.radians(deg), math.radians(2))
                self.release_handle()
                self.inner = False
                self.joint_to(self.park)          # out of the door's wedge (random-tree search if needed)
                at = round(math.degrees(self.c.env['fridge_door']), 1)
                self.swing('fridge_door', 'f_bar', 0.0, math.radians(.5))
                self.release_handle()
                return at
            except RuntimeError:
                self.inner = False
                self.q, self.t = snap[0], snap[2]
                del self.wp[snap[1]:]
                self.c.env.update(snap[3])
        raise RuntimeError('no hand-over from the inner to the outer pin')

    def swing(self, door, bar, th1, need, score=None):
        self.c.at_door = True
        try:
            return self._swing(door, bar, th1, need, score)
        finally:
            self.c.at_door = False

    def _swing(self, door, bar, th1, need, score=None):
        """Grasp the pin and swing the door towards th1; tries the grasp candidates in turn and
        keeps the first whose swing reaches within `need` (rad) of th1. Returns the angle reached.
        Opening (pulling against the seal, or swinging a heavy door on): the hand is turned so the
        rigid plate lies on the door side of the pin and takes the pull; the jaw's servo would
        give way at about 7 N. Closing: the plate pushes the last few degrees against the catch."""
        th = self.c.env[door]
        self.c.apply(self.q)
        g = self.c.point(bar)
        n_out = self._door_normal(door, th)
        n_move = np.array([-math.sin(th), -math.cos(th), 0.0])         # where the door's points go when it opens
        opening = th1 > th
        base = math.atan2(-n_out[1], -n_out[0])
        cands = []
        for dl in np.radians(np.arange(-180, 180, 5)):
            psi = base + dl
            y_hand = np.array([-math.sin(psi), math.cos(psi), 0])
            if opening and th < SEAL_ZONE and y_hand @ n_move > -PLATE_PULL:      # breaking the seal
                continue
            if not opening and th1 < SEAL_ZONE and y_hand @ n_move < PLATE_PULL - 1:  # closing: plate on the closing side
                continue
            for q in solutions(g, psi, 'bar', M.GRIP['bar']):
                if self.ok_on(bar, q):
                    self.c.allow = {bar: PIN_TOUCH}
                    ap = self.pin_approach(q)
                    self.c.allow = {}
                    if ap is not None:            # grips that leave the pin room on the way up come first
                        gap = 0 if ap[4] >= PIN_SIDE else 1 if ap[4] > 0 else 2
                        cands.append(((gap, score(q) if score else abs((dl + math.pi) % (2 * math.pi) - math.pi)), ap, q))
        best = None
        for _, ap, q in sorted(cands, key=lambda x: x[0])[:12]:
            snap = (self.q.copy(), len(self.wp), self.t, dict(self.c.env))
            try:
                self.take_pin(bar, ap)
                if door == 'fridge_door' and bar == 'f_bar' and th < SEAL_ZONE:
                    self.seal_loads = S.door_loads(self.c, self.q, door, bar)
                    psi = self.heading()
                    self.seal_plate_cos = round(float(np.array([-math.sin(psi), math.cos(psi), 0]) @ n_move), 2)
                got = self.door_arc(door, bar, self.c.env[door], th1)
                if abs(got - th1) <= need:
                    return got
                if best is None or abs(got - th1) < abs(best - th1):
                    best = got
            except RuntimeError:
                pass
            self.c.allow = {}
            self.q, self.t = snap[0], snap[2]
            del self.wp[snap[1]:]
            self.c.env.update(snap[3])
        raise RuntimeError(f'{bar}: no grasp swings the door to {math.degrees(th1):.0f} deg (best {None if best is None else round(math.degrees(best))})')

    def door_arc(self, door, bar, th0, th1, tool='bar', grip=None, dth=math.radians(2), deltas=None):
        """Hand on the pin while the door turns from th0 to th1 (sequence.py's beam search over
        hand heading and elbow branch, minimal joint travel, at most 12 deg per joint per layer).
        Build11 changes: after the first layer the door angles snap to a fixed 2 deg grid, so
        a door seen a fraction of a degree off plans the same way; and the plate rule: within
        SEAL_ZONE of shut the rigid plate faces the force (the seal's pull when opening, the
        catch's push when closing), so the jaw's servo never carries it. The pin is a cylinder:
        the caged hand may turn round it. Stops where the hand can still let go of the pin."""
        grip = M.GRIP['bar'] if grip is None else grip
        deltas = np.radians(np.arange(-100, 101, 5)) if deltas is None else deltas
        sgn = 1 if th1 >= th0 else -1
        g0 = (math.floor if sgn > 0 else math.ceil)(th0 / dth + sgn * 1e-6) * dth
        ths = [th0] + [g for g in np.arange(g0 + sgn * dth, th1, sgn * dth) if sgn * (th1 - g) > 1e-6] + [th1]
        layers = []
        for th in ths:
            env = dict(self.c.env); env[door] = th
            self.c.apply(self.q, env)
            g = self.c.point(bar)
            n_out = self._door_normal(door, th)
            psi0 = math.atan2(-n_out[1], -n_out[0])
            F = sgn * np.array([-math.sin(th), -math.cos(th), 0.0])
            layer = []
            for dl in deltas:
                psi = psi0 + dl
                if th < SEAL_ZONE and np.array([-math.sin(psi), math.cos(psi), 0]) @ F > -PLATE_PULL:
                    continue
                for q in solutions(g, psi, tool, grip):
                    if self.c.ok(q, env):
                        layer.append((q, env))
            if not layer:
                break
            layers.append(layer)
        if not layers:
            return th0
        cost = [S.tdist(q, self.q) if np.max(np.abs(q[1:4] - self.q[1:4])) < math.radians(12) else np.inf for q, _ in layers[0]]
        back = []
        for L in range(1, len(layers)):
            new, bk = [], []
            for q, _ in layers[L]:
                best = (np.inf, -1)
                for k, (qp, _) in enumerate(layers[L - 1]):
                    if cost[k] == np.inf or np.max(np.abs(q[1:4] - qp[1:4])) > math.radians(12):
                        continue
                    c = cost[k] + S.tdist(q, qp)
                    if c < best[0]:
                        best = (c, k)
                new.append(best[0]); bk.append(best[1])
            if all(c == np.inf for c in new):
                break
            cost = new
            back.append(bk)
        k = int(np.argmin(cost))
        if cost[k] == np.inf:
            return th0
        path = [k]
        for L in range(len(back), 0, -1):
            k = back[L - 1][k]
            path.append(k)
        path.reverse()
        keep_env = dict(self.c.env)                    # stop where the hand can still let go of the pin
        while len(path) > 1:
            q_end, env_end = layers[len(path) - 1][path[-1]]
            self.c.env.update(env_end)
            if self.release_moves(q_end) is not None:
                break
            path.pop()
        self.c.env.update(keep_env)
        last = len(path) - 1
        prev = self.q
        hinge = self.c.d.xpos[self.c.m.body('f_door' if door == 'fridge_door' else 'm_door').id][:2]
        radius = float(np.linalg.norm(self.c.point(bar)[:2] - hinge))
        for L, k in enumerate(path):
            q, env = layers[L][k]
            self.c.env.update(env)
            arc = abs(ths[L] - (ths[L - 1] if L else th0)) * radius
            self._push(max(arc / S.V_FINE, S.tdist(q, prev)), q)
            prev = q
        return float(ths[last])

    def press(self, key, travel=.003, stand=.04):
        """A nub on a front-panel key, pressed straight in: the top nub, else the low one."""
        self.c.apply(self.q)
        k = self.c.point(key)
        face = k.copy(); face[1] = -.008
        for tool in ('tip', 'tip_low'):
            for dl in np.radians([0, 10, -10, 20, -20, 30, -30]):
                psi = math.pi / 2 + dl
                h = np.array([math.cos(psi), math.sin(psi), 0])
                pre = face - stand * h
                qa = self.choose(pre, psi, tool, M.GRIP['tip'], self.q)
                if qa is None:
                    continue
                snap = (self.q.copy(), len(self.wp), self.t)
                try:
                    self.joint_to(qa)
                    self.c.allow = {key: travel + PIN_TOUCH}
                    if self.line(pre, face + travel * h, psi, psi, tool, M.GRIP['tip'], v=S.V_FINE) and \
                            self.line(face + travel * h, face - .08 * h, psi, psi, tool, M.GRIP['tip'], v=S.V_FINE * 2):
                        self.c.allow = {}
                        return
                except RuntimeError:
                    pass
                self.c.allow = {}
                self.q, self.t = snap[0], snap[2]
                del self.wp[snap[1]:]
        raise RuntimeError(f'cannot press {key}')

    # -------------------------------------------------------------- objects
    def pre_dist(self):
        return X.OBJECTS[self.c.obj]['L'] / 2 + .075 if self.c.cfg['mode'] == 'fork' else .09

    def take(self, base, psi, out_y, rails=False, grip=None, lift=.015):
        """Open hand in along psi, close (clamp, or for a hot tray a cage 5 mm off the top),
        lift 15 mm, out along the hand or straight to the front."""
        grip = M.GRIP['jar'] if grip is None else grip
        h = np.array([math.cos(psi), math.sin(psi), 0])
        self.c.put(base, psi, rails)
        g = self.c.jar_pos + [0, 0, self.c.cfg['tool_h']]
        if self.c.cfg['mode'] == 'fork':
            g = g - [0, 0, UNDER]                    # the fork slides in just under the bottom, then lifts it
            lift += UNDER
        else:                                        # the plate stays 4 mm clear of the jar; closing, the jaw slides it over
            g = g + SIDE_CLEAR * np.array([-math.sin(psi), math.cos(psi), 0])
        pre = g - self.pre_dist() * h
        for opening, stage in [(o, st) for st in (0.0, .12, .22) for o in openings()]:
            far = pre - stage * h                       # line up further out first, then come straight in
            qa = self.choose(far, psi, 'jar', opening, self.q)
            if qa is None:
                continue
            snap = (self.q.copy(), len(self.wp), self.t)
            try:
                self.joint_to(qa)
                if (stage == 0 or self.line(far, pre, psi, psi, 'jar', opening, v=S.V_CART)) and \
                        self.line(pre, g, psi, psi, 'jar', opening, v=S.V_FINE):
                    self.grip_to(grip)
                    self.c.held = True
                    up = g + [0, 0, lift]
                    if self.line(g, up, psi, psi, 'jar', grip, v=S.V_FINE):
                        back = up - h * ((up[1] - out_y) / h[1])
                        front = up.copy(); front[1] = out_y
                        for end in (back, front):
                            if self.line(up, end, psi, psi, 'jar', grip, v=S.V_FINE * 2):
                                return psi
            except RuntimeError:
                pass
            self.c.held = False
            self.q, self.t = snap[0], snap[2]
            del self.wp[snap[1]:]
        raise RuntimeError('cannot take the object')

    def put_down(self, base, psi, start, grip=None):
        """From start (in front of the spot) in to 15 mm over it, down, let go, out. If the straight
        way in is blocked (a door a little less open than planned), starts 1-2 cm to the side,
        higher, or turned a few degrees are tried. Last, the way in goes straight (heading 90 deg)
        and the meal turns to psi only over the last 4 cm, inside: a box turned to a rib channel
        that stopped 5-8 deg off would otherwise catch the open microwave door with a corner."""
        grip = M.GRIP['jar'] if grip is None else grip
        h = np.array([math.cos(psi), math.sin(psi), 0])
        g = np.asarray(base) + [0, 0, self.c.cfg['tool_h']]
        up = g + [0, 0, .015]
        tries = [(dx, dz, dpsi, False) for dx, dz, dpsi in (
            (0, 0, 0), (.01, 0, 0), (-.01, 0, 0), (0, .01, 0), (.02, 0, 0), (-.02, 0, 0),
            (0, 0, math.radians(5)), (0, 0, -math.radians(5)), (.01, .01, 0), (-.01, .01, 0))]
        if abs(psi - math.pi / 2) > math.radians(1):
            tries += [(dx, dz, math.pi / 2 - psi, True) for dx, dz in ((0, 0), (0, .01), (.01, 0), (-.01, 0))]
        for dx, dz, dpsi, inside in tries:
            st = np.asarray(start) + [dx, 0, dz]
            if inside:                                  # straight in along the spot's own approach line
                u0 = np.array([0.0, 1.0, 0.0])
                st = np.r_[up[0], up[1] - abs(up[1] - start[1]), start[2]] + [dx, 0, dz]
                mid = up - u0 * .04 + [dx, 0, dz]
            done = False
            for qa in sorted((q for q in solutions(st, psi + dpsi, 'jar', grip) if self.c.ok(q)),
                             key=lambda q: S.tdist(q, self.q)):         # every elbow branch, nearest first
                snap = (self.q.copy(), len(self.wp), self.t)
                try:
                    self.joint_to(qa)
                    if inside:
                        way = self.line(st, mid, psi + dpsi, psi + dpsi, 'jar', grip, v=S.V_FINE * 2) and \
                            self.line(mid, up, psi + dpsi, psi, 'jar', grip, v=S.V_FINE)
                    else:
                        way = self.line(st, up, psi + dpsi, psi, 'jar', grip, v=S.V_FINE * 2)
                    if way and self.line(up, g + [0, 0, .0005], psi, psi, 'jar', grip, v=S.V_FINE):
                        done = True
                        break
                except RuntimeError:
                    pass
                self.q, self.t = snap[0], snap[2]
                del self.wp[snap[1]:]
            if done:
                break
        else:
            raise RuntimeError('put-down path blocked')
        pos, quat = self.c.apply(self.q)
        self.c.put(pos, psi)
        fork = self.c.cfg['mode'] == 'fork'
        opening = M.GRIP['jar'] - .12 if fork else M.GRIP['open'] * .6
        self.grip_to(opening)
        if fork:                                     # down off the object's bottom, so the fork slides out free
            g2 = g + [0, 0, .0005 - UNDER - .001]
            if not self.line(g + [0, 0, .0005], g2, psi, psi, 'jar', opening, v=S.V_FINE):
                raise RuntimeError('cannot lower the fork after the put-down')
            g = g2
        if not self.line(g, g - self.pre_dist() * h, psi, psi, 'jar', opening, v=S.V_FINE):
            raise RuntimeError('cannot withdraw after the put-down')

    # -------------------------------------------------------------- freezer flap
    def tab_point(self, th):
        f = self.c.f
        c, s = math.cos(th), math.sin(th)
        p = f['flap_tab']
        return f['flap_hinge'] + np.array([p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c])

    def flap_open(self, th1=math.radians(90), dth=math.radians(3)):
        """Top nub into the loop from above, then down with the lift: the loop and the flap
        turn about the hinge while the nub follows the loop (hand heading fixed, into the fridge)."""
        psi = math.pi / 2
        p0 = self.tab_point(0.0)
        above = p0 + [0, 0, .025]
        qa = self.choose(above, psi, 'tip', M.GRIP['tip'], self.q)
        if qa is None:
            raise RuntimeError('cannot reach above the flap tab')
        self.joint_to(qa)
        self.c.allow = {'flap_tab': PIN_TOUCH, 'flap_tab_floor': PIN_TOUCH}
        if not self.line(above, p0, psi, psi, 'tip', M.GRIP['tip'], v=S.V_FINE):
            raise RuntimeError('cannot hook the flap tab')
        prev = self.q
        th = 0.0
        while th < th1 - 1e-9:
            th = min(th1, th + dth)
            env = dict(self.c.env); env['flap'] = th
            g = self.tab_point(th)
            q = None
            for qq in solutions(g, psi, 'tip', M.GRIP['tip']):
                if np.max(np.abs(qq[1:4] - prev[1:4])) < math.radians(12) and self.c.ok(qq, env):
                    q = qq
                    break
            if q is None:
                th -= dth
                if th < math.radians(75):
                    raise RuntimeError(f'flap stuck at {math.degrees(th):.0f} deg')
                break
            self.c.env['flap'] = th
            self._push(max(dth * .12 / S.V_FINE, S.tdist(q, prev)), q)
            prev = q
        p = self.tab_point(self.c.env['flap'])
        for up in ([0, 0, .03], [0, -.03, .02], [0, -.04, 0]):
            if self.line(p, p + up, psi, psi, 'tip', M.GRIP['tip'], v=S.V_FINE):
                self.c.allow = {}
                for back in (.15, .10, .06):                # clear of the flap and the cabinet
                    if self.line(p + up, p + up - [0, back, 0], psi, psi, 'tip', M.GRIP['tip'], v=S.V_CART):
                        break
                pulled = math.degrees(self.c.env['flap'])
                th0 = self.c.env['flap']
                for k in range(1, 6):                      # past its tipping point the flap drops onto its stop
                    self.c.env['flap'] = th0 + (FLAP_STOP - th0) * k / 5
                    self._push(.06)
                return pulled
        raise RuntimeError('cannot leave the flap tab')

    def flap_close(self, dth=math.radians(3)):
        """The top nub (the plate's upper edge) under the open flap, 1 cm in from its free edge,
        then the flap pushed up and in: the nub follows the flap's underside as it turns."""
        psi = math.pi / 2
        f = self.c.f
        h = X.FREEZER['flap_h'] - .01

        def contact(th):
            c, s_ = math.cos(th), math.sin(th)
            under = np.array([0, -h * s_, h * c])                     # a point on the underside face (local y = 0)
            n = np.array([0, -c, -s_])                                 # the underside's outward normal
            return f['flap_hinge'] + under + .009 * n

        th = self.c.env['flap']
        p = contact(th)
        below = p + [0, 0, -.04]
        qa = self.choose(below, psi, 'tip', M.GRIP['tip'], self.q)
        if qa is None:
            raise RuntimeError('cannot reach under the flap')
        self.joint_to(qa)
        if not self.line(below, p, psi, psi, 'tip', M.GRIP['tip'], v=S.V_FINE):
            raise RuntimeError('cannot touch the flap')
        prev = self.q
        while th > 1e-9:
            th = max(0.0, th - dth)
            env = dict(self.c.env); env['flap'] = th
            g = contact(th)
            q = None
            for qq in solutions(g, psi, 'tip', M.GRIP['tip']):
                if np.max(np.abs(qq[1:4] - prev[1:4])) < math.radians(12) and self.c.ok(qq, env):
                    q = qq
                    break
            if q is None:
                raise RuntimeError(f'cannot push the flap past {math.degrees(th):.0f} deg')
            self.c.env['flap'] = th
            self._push(max(dth * .12 / S.V_FINE, S.tdist(q, prev)), q)
            prev = q
        p = contact(0.0)
        if not self.line(p, p + [0, -.08, 0], psi, psi, 'tip', M.GRIP['tip'], v=S.V_FINE):
            raise RuntimeError('cannot back off the flap')

    # -------------------------------------------------------------- heating
    def heat(self, turn):
        """The turntable turns by `turn` (rad); an object standing on it turns with it."""
        f = self.c.f
        ctr = np.r_[f['mw_centre'], 0]
        n = 30
        on_tt = not self.c.held and abs(self.c.jar_pos[2] - (f['plate_z'] + X.RAILS['h'])) < .01
        p0, h0 = self.c.jar_pos.copy(), heading_of(self.c.jar_quat)
        for k in range(1, n + 1):
            a = turn * k / n
            self.c.env['turntable'] = a
            if on_tt:
                r = p0 - ctr
                rp = np.array([r[0] * math.cos(a) - r[1] * math.sin(a), r[0] * math.sin(a) + r[1] * math.cos(a), r[2]])
                self.c.jar_pos = ctr + rp
                self.c.jar_quat = M.quat_z(h0 + a - math.pi / 2)
            self._push(4.0 / n)


def park_pose(c):
    best = None
    for x, y in ((-.05, .20), (-.10, .22), (0.0, .18), (-.15, .25), (-.20, .15)):
        for psi in np.radians(np.arange(-180, 180, 15)):
            for q in solutions(np.array([x, y, M.LIFT[1] - .03 + M.HAND_DZ]), psi, 'jar', M.GRIP['jar']):
                if not c.ok(q):
                    continue
                a = q[1] - math.pi / 2
                E = M.S + M.L1 * np.array([math.cos(a), math.sin(a)])
                Y = E + M.L2 * np.array([math.cos(a + q[2]), math.sin(a + q[2])])
                score = max(E[0], Y[0]) + 5 * max(0.0, -.03 - min(E[1], Y[1]))
                if best is None or score < best[0]:
                    best = (score, q)
    return best[1]


def morning(obj, kind, planner=None, residual=None):
    """The whole morning as a script of skills. planner: a Planner subclass, e.g. one that runs
    each step in physics and senses before the next (doors11.run, live=True). residual: where
    the Shelly cut-off stops the turntable past the half turn (rad), the plan's +6 deg if None."""
    c = Cell(kind, obj)
    f = c.f
    o = X.OBJECTS[obj]
    fork = c.cfg['mode'] == 'fork'
    (start, psi0) = START[(obj, kind)]
    psi0 = math.radians(psi0) if psi0 is not None else None
    c.put(np.array(start), psi0 or math.pi / 2, rails=fork)
    park = park_pose(c)
    P = (planner or Planner)(c, park)
    P.park = park
    out = dict(obj=obj, kind=kind)

    P.step('mw_open', 'Press the microwave door key with a nub; the door springs ajar; rise onto its pin and swing it open')
    P.press('m_release')
    for th in np.radians(np.linspace(0, 10, 6))[1:]:
        c.env['mw_door'] = th
        P._push(.08)
    out['mw_door_deg'] = round(math.degrees(P.swing('mw_door', 'm_bar', math.radians(MW_OPEN), math.radians(5))), 1)
    P.release_handle()

    P.step('fridge_open', 'Rise onto the outer door pin, break the seal, swing the door to the end of reach')
    out['outer_pin_deg'] = round(math.degrees(P.swing('fridge_door', 'f_bar', math.radians(110), math.radians(70),
                                                      score=lambda q: max(S.door_loads(c, q, 'fridge_door', 'f_bar', (16.0,))['16N'].values()))), 1)
    out['seal_loads'] = P.seal_loads
    out['seal_plate_cos'] = P.seal_plate_cos           # -1: the plate takes the whole pull
    P.release_handle()
    if obj in NEEDS_100:
        P.step('fridge_wide', 'Inner pin on the door rack: swing the door on to 100 deg')
        P.inner = True
        out['inner_pin_deg'] = round(math.degrees(P.swing('fridge_door', 'f_bar2', math.radians(100), math.radians(8))), 1)
        P.release_handle()
        P.retreat()
        P.inner = False
    if obj == 'tray':
        P.step('flap_open', 'Hook the freezer flap\'s loop with the top nub and lower the lift: the flap folds down')
        out['flap_deg'] = round(P.flap_open(), 1)

    P.step('take', 'Fork under the tray on its rails, jaw onto the top, lift 15 mm, out' if obj == 'tray' else
           'Fork under the box on its rails, jaw onto the lid, lift 15 mm, out' if obj == 'box' else
           'Grip the jar at mid-height, lift 15 mm, out')
    if fork:                                     # in the freezer: 6 mm clears the rails, 15 mm would touch its roof
        P.take(c.jar_pos.copy(), psi0, out_y=-.12 - o['L'] / 2, rails=True, lift=.006 if obj == 'tray' else .015)
    else:
        P.take(c.jar_pos.copy(), math.pi / 2, out_y=-.12)

    P.step('to_microwave', 'Into the microwave: onto the ribs on the turntable' if fork else 'Into the microwave, onto the turntable centre')
    mc = f['mw_centre']
    lift = X.RAILS['h']                       # every object stands on the ribs in the microwave and on top
    tb = np.array([mc[0], mc[1], f['plate_z'] + lift + .0005])
    L = o['L'] if fork else .1
    psi_in = math.pi / 2
    if fork:              # into the rib channel, wherever the turntable stopped last time (seen to 1 deg)
        psi_in = (c.env['turntable'] + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        if math.cos(psi_in - math.pi / 2) < 0:
            psi_in += math.pi
    u_in = np.array([math.cos(psi_in), math.sin(psi_in), 0])
    t_start = tb + [0, 0, c.cfg['tool_h'] + .015] - u_in * (tb[1] + L / 2 + .03) / u_in[1]
    P.put_down(tb, psi_in, t_start)

    if obj == 'tray':
        P.step('flap_close', 'Push the freezer flap up and shut with the low nub')
        P.flap_close()
    P.step('mw_close', 'Swing the microwave door shut by its pin, press start')
    out['mw_door_closed'] = P.swing('mw_door', 'm_bar', 0.0, math.radians(.5)) < math.radians(.5)
    P.release_handle()
    P.press('m_start')

    P.step('fridge_close', 'Inner pin swings the door back to about 45 deg, the outer pin shuts it' if obj in NEEDS_100 else
           'Swing the fridge door shut by its pin')
    if obj in NEEDS_100:
        out['handover_deg'] = P.hand_over(out['outer_pin_deg'])
        out['fridge_door_closed'] = c.env['fridge_door'] < math.radians(.5)
    else:
        out['fridge_door_closed'] = P.swing('fridge_door', 'f_bar', 0.0, math.radians(.5)) < math.radians(.5)
        P.release_handle()

    P.step('heat', 'Heat (sped up). The Shelly cuts the power with the turntable at 0 or 180 deg' if fork else 'Heat (sped up)')
    P.heat(2 * math.pi + math.pi + ((TT_RESIDUAL if residual is None else residual) if fork else 1.3))

    P.step('mw_reopen', 'Press the door key, swing the door open again')
    P.press('m_release')
    for th in np.radians(np.linspace(0, 10, 6))[1:]:
        c.env['mw_door'] = th
        P._push(.08)
    P.swing('mw_door', 'm_bar', math.radians(out['mw_door_deg']), math.radians(5))
    P.release_handle()

    P.step('take_hot', 'Fork in along the ribs (turned 180 deg plus a few), take it out' if fork else 'Take the hot jar out')
    # the fork follows the rib channel, seen from the turntable's angle; the box anywhere along it
    psi_hot = c.env['turntable'] + math.pi / 2 if fork else math.pi / 2
    psi_hot = (psi_hot + math.pi) % (2 * math.pi) - math.pi
    if fork and math.cos(psi_hot - math.pi / 2) < 0:
        psi_hot += math.pi                                        # the ribs are symmetric: enter from the front
    if fork:
        ctr, u = np.r_[f['mw_centre'], 0], np.array([math.cos(psi_hot), math.sin(psi_hot), 0])
        c.jar_pos = ctr + u * float((c.jar_pos - ctr) @ u) + [0, 0, c.jar_pos[2]]
    cage = X.jaw_angle_to(X.PLATE['face'] - o['H'], (X.PALM, X.PALM + o['L'])) - .08 if obj == 'tray' else None
    P.take(c.jar_pos.copy(), psi_hot, out_y=t_start[1], grip=cage)
    out['hot_heading_deg'] = round(math.degrees(psi_hot), 1)

    P.step('serve', 'Set it on the serving ribs on top of the microwave' if fork else 'Stand it on top of the microwave')
    sb = np.array([X.SERVE[0], X.SERVE[1], f['mw_top'] + lift + .0005])
    s_start = sb + [0, 0, c.cfg['tool_h'] + .045]; s_start[1] = sb[1] - .12
    P.put_down(sb, math.pi / 2, s_start, grip=cage)

    P.step('mw_shut', 'Close the microwave door and park')
    P.swing('mw_door', 'm_bar', 0.0, math.radians(.5))
    P.release_handle()
    P.joint_to(park)
    P.step('done', 'Parked')
    return c, P, out


def frames(c, P):
    ts = np.array([w['t'] for w in P.wp])
    out, labels, held = [], [], []
    for t in np.arange(0, ts[-1] + 1e-9, 1 / FPS):
        k = int(np.searchsorted(ts, t, side='right') - 1)
        k2 = min(k + 1, len(ts) - 1)
        a, b = P.wp[k], P.wp[k2]
        s = 0 if k2 == k or ts[k2] == ts[k] else (t - ts[k]) / (ts[k2] - ts[k])
        q = a['q'] + (b['q'] - a['q']) * s
        env = {kk: a['env'][kk] + (b['env'][kk] - a['env'][kk]) * s for kk in a['env']}
        jar = a['jar'] if not (a['held'] and b['held']) else a['jar'] + (b['jar'] - a['jar']) * s
        qpos = np.zeros(c.m.nq)
        for i, n in enumerate(S.ARM):
            qpos[c.adr[n]] = q[i]
        for kk, v in env.items():
            qpos[c.adr[kk]] = v
        qpos[c.adr['cw']] = M.CW_C - q[0]
        qpos[c.adr['jar']:c.adr['jar'] + 7] = jar
        qpos[c.adr['rails']:c.adr['rails'] + 7] = a['rails']
        out.append([round(float(v), 4) for v in qpos])
        labels.append(a['label'])
        held.append(bool(a['held']))
    return out, labels, held


def waypoints(P):
    """Dense plan for the dynamic replay: time, joint targets, appliance angles, held flag."""
    return [dict(t=round(w['t'], 4), q=[round(float(v), 5) for v in w['q']], env={k: round(float(v), 5) for k, v in w['env'].items()},
                 held=bool(w['held']), label=w['label'], jar=[round(float(v), 5) for v in w['jar']],
                 rails=[round(float(v), 5) for v in w['rails']]) for w in P.wp]


# ------------------------------------------------------------------ coverage with this hand
def takeable(c, base, headings, rails, out_y=-.12, lift=.015):
    P = Planner(c, np.zeros(7))
    pre = P.pre_dist()
    for deg in headings:
        psi = math.radians(deg)
        h = np.array([math.cos(psi), math.sin(psi), 0])
        if h[1] < .05:
            continue
        c.put(base, psi, rails)
        g = c.jar_pos + [0, 0, c.cfg['tool_h']]
        for opening in openings():
            for q in solutions(g, psi, 'jar', opening):
                c.held = False
                if not c.ok(q) or not _path(c, g, g - pre * h, psi, opening, q, False):
                    continue
                qc = q.copy(); qc[6] = M.GRIP['jar']
                up = g + [0, 0, lift]
                back = up - h * ((up[1] - out_y) / h[1])
                front = up.copy(); front[1] = out_y
                if _path(c, up, back, psi, M.GRIP['jar'], qc, True) or _path(c, up, front, psi, M.GRIP['jar'], qc, True):
                    c.held = False
                    return deg
    c.held = False
    return None


def _path(c, a, b, psi, grip, q_ref, held, step=.01):
    c.held = held
    prev = q_ref
    for s in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step) + 1)):
        g = a + (b - a) * s
        best = None
        for q in solutions(g, psi, 'jar', grip):
            if np.max(np.abs(q[1:4] - prev[1:4])) > math.radians(20):
                continue
            if c.ok(q):
                best = q
                break
        if best is None:
            return False
        prev = best
    return True


def coverage(obj, kind, door_deg):
    c = Cell(kind, obj)
    c.env['fridge_door'] = math.radians(door_deg)
    if obj == 'tray':
        c.env['flap'] = math.radians(90)
    f, o = c.f, X.OBJECTS[obj]
    W, D, t, y0 = f['W'], f['D'], f['t'], f['y0']
    fork = c.cfg['mode'] == 'fork'
    if fork:
        hw, hl, side = o['W'] / 2, o['L'] / 2, .01
        xs = np.arange(-W + t + hw + side, -t - hw - side + 1e-9, .04)
    else:                                         # build10's grid (sequence.coverage), so the numbers compare
        hl = 0.0
        xs = np.arange(-W + t + .06, -t - .06 + 1e-9, .04)
    if obj == 'tray':
        levels = {'freezer': (X.FREEZER['floor'] + .01, np.arange(X.FREEZER['front'] + .015 + hl, D - .05 - hl + 1e-9, .04))}
    else:
        levels = {f'{z:.2f}': (z, np.arange(y0 + (.01 if fork else .06) + hl, D - .05 - hl - (0 if fork else .06) + 1e-9, .04))
                  for z in (f['shelves'] if kind == 'large' or fork else f['shelves'][1:] + [f['shelves'][0]])}
    res = {}
    for lev, (z, ys) in levels.items():
        pts = [(float(x), float(y), z + (X.RAILS['h'] if fork else 0) + .0005) for x in xs for y in ys]
        heads = [90] if fork else list(range(90, 181, 15)) + list(range(75, -1, -15))
        ok = [takeable(c, np.array(p), heads, fork, lift=.006 if obj == 'tray' else .015) is not None for p in pts]
        res[lev] = dict(spots=len(pts), takeable=sum(ok))
    n = sum(v['spots'] for v in res.values())
    res['share'] = round(sum(v['takeable'] for v in res.values()) / n, 3)
    return res


def main():
    """All menus, or only those named (e.g. jar_large box_small), merged into the saved files."""
    t0 = time.time()
    only = [a for a in sys.argv[1:] if not a.startswith('--')]
    res_path, seq_path = HERE / 'plan11-results.json', HERE / 'plan11.json'
    results = json.loads(res_path.read_text()) if only and res_path.exists() else {}
    seqs = json.loads(seq_path.read_text())['sequences'] if only and seq_path.exists() else {}
    menus = [('jar', 'large'), ('box', 'large'), ('tray', 'large'), ('jar', 'small'), ('box', 'small')]
    for obj, kind in menus:
        key = f'{obj}_{kind}'
        if only and key not in only:
            continue
        try:
            c, P, out = morning(obj, kind)
        except RuntimeError as e:
            results[key] = dict(error=f'{Planner.last}: {e}')
            seqs.pop(key, None)
            print(key, 'FAILED in', Planner.last, '-', e, flush=True)
            continue
        out = S.summarize(c, P, out)
        out.pop('pitch_hold', None)
        fr, labels, held = frames(c, P)
        seqs[key] = dict(obj=obj, kind=kind, steps=P.steps, labels=labels, held=held, frames=fr, nq=int(c.m.nq), wp=waypoints(P))
        if '--no-coverage' not in sys.argv:
            out['coverage'] = coverage(obj, kind, 100.0 if obj in NEEDS_100 else out['outer_pin_deg'])
        elif key in results and 'coverage' in results[key]:
            out['coverage'] = results[key]['coverage']
        results[key] = {k: v for k, v in out.items() if k not in ('steps',)}
        print(key, json.dumps({k: v for k, v in out.items() if k not in ('steps', 'penetrations', 'closest_by_step')}, default=float), flush=True)
    with open(HERE / '.plan11.lock', 'w') as lock:      # menus run as parallel processes merge into the same files
        fcntl.flock(lock, fcntl.LOCK_EX)
        if only:
            cur_r = json.loads(res_path.read_text()) if res_path.exists() else {}
            cur_s = json.loads(seq_path.read_text())['sequences'] if seq_path.exists() else {}
            for key in only:
                if key in results:
                    cur_r[key] = results[key]
                if key in seqs:
                    cur_s[key] = seqs[key]
                else:
                    cur_s.pop(key, None)
            results, seqs = cur_r, cur_s
        results['runtime_s'] = round(time.time() - t0, 1)
        res_path.write_text(json.dumps(results, indent=1, default=float))
        seq_path.write_text(json.dumps(dict(fps=FPS, sequences=seqs), separators=(',', ':')))


if __name__ == '__main__':
    main()
