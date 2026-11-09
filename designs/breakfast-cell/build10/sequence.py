"""The whole breakfast morning on the detailed build10 model, and the shelf coverage
re-run with real part shapes.

Plans every motion with the arm's own inverse kinematics, at servo speeds
(lift 60 mm/s, shoulder and elbow 57 deg/s through their belts, wrist 115 deg/s),
and checks every frame in MuJoCo's collision detector with the detailed geometry:
penetration anywhere is a failure, and the closest approach per step is reported.
Doors move only while the hand holds their handle bar.

Kinematic replay: it checks geometry, reach, joint limits, timing and static loads.
It does not simulate grip friction, door springs, servo play or a slipping jar.
Writes sequence.json (frames) and sequence-results.json.
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
import model3d as M  # noqa: E402

ARM = M.JOINTS
VMAX = np.array([.06, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])     # m/s, rad/s
V_CART, V_FINE = .10, .04                              # free and contact moves, m/s
FPS = 15
C018_STALL, BELT = 2.942, 3 * .8
SUPPORTS = ('f_shelf', 'm_plate', 'm_top', 'm_panel', 'f_crisper')
CLEAR = .003       # the robot keeps 3 mm from everything, itself included (jar contacts excepted)
ROBOT_BODIES = {'carriage', 'upper_arm', 'forearm', 'wrist_yaw', 'wrist_roll', 'wrist_pitch', 'jaw'}
HAND_BODIES = {'wrist_pitch', 'jaw', 'wrist_roll'}


class Cell:
    def __init__(self, kind='large'):
        xml, meta, f, _ = M.mjcf(kind)
        self.kind, self.f, self.xml = kind, f, xml
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.adr = {self.m.joint(i).name: int(self.m.jnt_qposadr[i]) for i in range(self.m.njnt)}
        self.gname = [self.m.geom(i).name for i in range(self.m.ngeom)]
        self.gbody = [self.m.body(self.m.geom_bodyid[i]).name for i in range(self.m.ngeom)]
        self.is_jar = np.array([b == 'jar' for b in self.gbody])
        self.is_hand = np.array([b in HAND_BODIES for b in self.gbody])
        self.is_support = np.array([any(n.startswith(s) for s in SUPPORTS) for n in self.gname])
        self.is_robot = np.array([b in ROBOT_BODIES for b in self.gbody])
        self.pitch = self.m.body('wrist_pitch').id
        self.env = dict(fridge_door=0.0, mw_door=0.0, turntable=0.0)
        self.held = False
        self.jar_pos = np.array([-.16, .15, f['shelves'][1] + .0005])
        self.jar_quat = np.array([1., 0, 0, 0])

    def apply(self, q, env=None):
        d = self.d
        for i, n in enumerate(ARM):
            d.qpos[self.adr[n]] = q[i]
        for k, v in (env or self.env).items():
            d.qpos[self.adr[k]] = v
        d.qpos[self.adr['cw']] = M.CW_C - q[0]
        mujoco.mj_kinematics(self.m, d)
        if self.held:
            R = d.xmat[self.pitch].reshape(3, 3)
            pos = d.xpos[self.pitch] + R @ np.array([M.JAR_IN_PITCH, 0, -M.JAR['grip']])
            quat = d.xquat[self.pitch].copy()
        else:
            pos, quat = self.jar_pos, self.jar_quat
        a = self.adr['jar']
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, quat
        mujoco.mj_kinematics(self.m, d)
        return pos, quat

    def contacts(self):
        mujoco.mj_collision(self.m, self.d)
        n = self.d.ncon
        if n == 0:
            return []
        c = self.d.contact
        g1, g2, dist = np.array(c.geom1[:n]), np.array(c.geom2[:n]), np.array(c.dist[:n])
        keep = dist < M.ROBOT_MARGIN - 1e-6
        jar1, jar2 = self.is_jar[g1], self.is_jar[g2]
        if self.held:                                   # the jar sits in the fingers
            keep &= ~((jar1 & self.is_hand[g2]) | (jar2 & self.is_hand[g1]))
        else:                                           # resting on its support
            keep &= ~(((jar1 & self.is_support[g2]) | (jar2 & self.is_support[g1])) & (dist > -.002))
        idx = np.nonzero(keep)[0]
        self._strict = [(not (jar1[k] or jar2[k])) and (self.is_robot[g1[k]] or self.is_robot[g2[k]]) for k in idx]
        return [(float(dist[k]), self.gname[g1[k]], self.gname[g2[k]]) for k in idx]

    def ok(self, q, env=None):
        if not limits_ok(q):
            return False
        self.apply(q, env)
        con = self.contacts()
        return all(dd > (CLEAR if strict else -1e-4) for (dd, _, _), strict in zip(con, self._strict))

    def point(self, geom):
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.geom_xpos[self.m.geom(geom).id].copy()


def limits_ok(q):
    return all(M.LIMITS[n][0] - 1e-9 <= q[i] <= M.LIMITS[n][1] + 1e-9 for i, n in enumerate(ARM))


def tdist(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / VMAX))


def solutions(g, psi, tool, grip, roll=0.0, pitch=0.0):
    return [np.r_[s, roll, pitch, grip] for s in M.ik(g, psi, tool)]


class Planner:
    def __init__(self, cell, q0):
        self.c = cell
        self.q = np.asarray(q0, float)
        self.wp = []            # dense waypoints: (time step, q, env, held, jar pos, quat, label)
        self.steps = []
        self.t = 0.0
        self.label = 'start'
        self._push(0.0)

    # ---------------------------------------------------------------- bookkeeping
    def _push(self, dt, q=None):
        if q is not None:
            self.q = np.asarray(q, float)
        self.t += dt
        pos, quat = self.c.apply(self.q)
        self.wp.append(dict(t=self.t, q=self.q.copy(), env=dict(self.c.env), held=self.c.held,
                            jar=np.r_[pos, quat], label=self.label, con=self.c.contacts()))

    def step(self, label, text):
        self.steps.append(dict(start=round(self.t, 2), label=label, text=text))
        self.label = label

    def wait(self, sec, spin=0.0):
        n = max(1, int(sec * FPS / 4))
        for _ in range(n):
            self.c.env['turntable'] += spin / n
            self._push(sec / n)

    # ---------------------------------------------------------------- motion primitives
    def choose(self, g, psi, tool, grip, prev, max_jump=None):
        best = None
        for q in solutions(g, psi, tool, grip):
            if not self.c.ok(q):
                continue
            j = tdist(q, prev)
            if max_jump is not None and np.max(np.abs(q[1:4] - prev[1:4])) > max_jump:
                continue
            if best is None or j < best[0]:
                best = (j, q)
        return None if best is None else best[1]

    def joint_to(self, target, v_scale=1.0, check=True):
        """Straight line in joint space, checked every 0.5 deg / 2 mm; falls back to
        lift-first or arm-first orderings when the direct move collides."""
        a, b = self.q.copy(), np.asarray(target, float)
        orders = [[b]]
        mid1 = a.copy(); mid1[0] = b[0]
        mid2 = b.copy(); mid2[0] = a[0]
        orders += [[mid1, b], [mid2, b]]
        for hub in self.hubs():                         # detour through a tucked pose in front
            h1, h2 = hub.copy(), hub.copy()
            h1[0], h2[0] = a[0], b[0]
            h1[4:] = a[4:]; h2[4:] = b[4:]
            orders += [[h1, h2, b]]
        for path in orders:
            pts, cur, good = [], a, True
            for tgt in path:
                n = max(2, int(np.max(np.abs(tgt - cur) / np.array([.002, .009, .009, .009, .009, .009, .02]))) + 1)
                for s in np.linspace(0, 1, n)[1:]:
                    q = cur + (tgt - cur) * s
                    if check and not self.c.ok(q):
                        good = False
                        break
                    pts.append(q)
                if not good:
                    break
                cur = tgt
            if good:
                prev = a
                for q in pts:
                    self._push(tdist(q, prev) / v_scale, q)
                    prev = q
                return True
        raise RuntimeError(f'joint move blocked in step "{self.label}"')

    def hubs(self):
        """Tucked arm poses in front of the mast (any lift height), used as detours."""
        if not hasattr(self, '_hubs'):
            self._hubs = []
            for x, y in ((.02, -.24), (-.06, -.30), (.10, -.30), (-.14, -.36), (.04, -.40)):
                for deg in (-90, -45, -135, 0, 180):
                    for q in solutions(np.array([x, y, .6]), math.radians(deg), 'jar', M.GRIP['jar']):
                        if limits_ok(q):
                            self._hubs.append(q)
        return self._hubs

    def line(self, p0, p1, psi0, psi1, tool, grip, v=V_CART, max_jump=math.radians(8)):
        """Straight Cartesian path of the tool point; heading interpolated."""
        n = max(2, int(np.linalg.norm(np.subtract(p1, p0)) / .004) + 1)
        qs = []
        prev = self.q
        for s in np.linspace(0, 1, n)[1:]:
            g = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * s
            q = self.choose(g, psi0 + (psi1 - psi0) * s, tool, grip, prev, max_jump)
            if q is None:
                return False
            qs.append((q, np.linalg.norm(np.subtract(p1, p0)) / (n - 1)))
            prev = q
        prev = self.q
        for q, ds in qs:
            self._push(max(ds / v, tdist(q, prev)), q)
            prev = q
        return True

    def grip_to(self, value):
        q = self.q.copy(); q[6] = value
        self.joint_to(q)

    def door_arc(self, door, bar, th0, th1, tool='bar', grip=M.GRIP['bar'], dth=math.radians(2), deltas=None):
        """Hand on the handle bar while the door turns from th0 to th1. A beam search over
        hand heading and elbow branch keeps the arm continuous; returns the angle reached."""
        deltas = np.radians(np.arange(-70, 71, 5)) if deltas is None else deltas
        sgn = 1 if th1 >= th0 else -1
        ths = list(np.arange(th0, th1, sgn * dth)) + [th1]
        layers = []
        for th in ths:
            env = dict(self.c.env); env[door] = th
            self.c.apply(self.q, env)
            g = self.c.point(bar)
            n_out = self._door_normal(door, th)
            psi0 = math.atan2(-n_out[1], -n_out[0])        # hand points into the door face
            layer = []
            for dl in deltas:
                for q in solutions(g, psi0 + dl, tool, grip):
                    if self.c.ok(q, env):
                        layer.append((q, env))
            if not layer:
                break
            layers.append(layer)
        # dynamic programming over layers: minimal total joint travel, max 12 deg per step
        if not layers:
            return th0
        cost = [tdist(q, self.q) if np.max(np.abs(q[1:4] - self.q[1:4])) < math.radians(12) else np.inf for q, _ in layers[0]]
        back = []
        for L in range(1, len(layers)):
            new, bk = [], []
            for q, _ in layers[L]:
                best = (np.inf, -1)
                for k, (qp, _) in enumerate(layers[L - 1]):
                    if cost[k] == np.inf or np.max(np.abs(q[1:4] - qp[1:4])) > math.radians(12):
                        continue
                    c = cost[k] + tdist(q, qp)
                    if c < best[0]:
                        best = (c, k)
                new.append(best[0]); bk.append(best[1])
            if all(c == np.inf for c in new):
                break
            cost = new
            back.append(bk)
        last = len(back)
        k = int(np.argmin(cost))
        path = [k]
        for L in range(last, 0, -1):
            k = back[L - 1][k]
            path.append(k)
        path.reverse()
        # stop where the hand can still let go of the bar
        keep_env = dict(self.c.env)
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
            self._push(max(arc / V_FINE, tdist(q, prev)), q)
            prev = q
        return float(ths[last])

    def _door_normal(self, door, th):
        return np.array([-math.sin(th), -math.cos(th), 0.0])

    # ---------------------------------------------------------------- composite skills
    def grasp_handle(self, door, bar, rise=.05, score=None):
        """Open hand under the pin, rise onto it with the lift, close. score(q) ranks the
        candidate headings (default: closest to pointing straight at the door)."""
        th = self.c.env[door]
        self.c.apply(self.q)
        g = self.c.point(bar)
        n_out = self._door_normal(door, th)
        base = math.atan2(-n_out[1], -n_out[0])
        cands = []
        for dl in np.radians(np.arange(-70, 71, 5)):
            for q in solutions(g, base + dl, 'bar', M.GRIP['bar']):
                if self.c.ok(q):
                    qo = q.copy(); qo[6] = M.GRIP['open']; qo[0] -= rise
                    if self.c.ok(qo):
                        cands.append(((score(q) if score else abs(dl)), qo, q))
        for _, qo, q in sorted(cands, key=lambda x: x[0]):
            snap = (self.q.copy(), len(self.wp), self.t)
            try:
                self.joint_to(qo)
                up = qo.copy(); up[0] = q[0]
                self.joint_to(up, v_scale=.5)
                self.grip_to(M.GRIP['bar'])
                return q
            except RuntimeError:
                self.q, self.t = snap[0], snap[2]
                del self.wp[snap[1]:]
        raise RuntimeError(f'cannot grasp {bar}')

    def release_moves(self, q, drop=.05):
        """Jaw open, then down off the pin with the lift. Dry run."""
        q = np.asarray(q, float).copy(); q[6] = M.GRIP['open']
        out = [q]
        for k in range(1, 11):
            qq = q.copy(); qq[0] -= drop * k / 10
            if not self.c.ok(qq):
                return None
            out.append(qq)
        return out

    def release_handle(self):
        moves = self.release_moves(self.q)
        if moves is None:
            raise RuntimeError('cannot let go of the handle')
        prev = self.q
        for q in moves:
            self._push(max(.005 / V_FINE, tdist(q, prev)), q)
            prev = q

    def heading(self):
        return float(self.q[1] - math.pi / 2 + self.q[2] + self.q[3])

    def tool_point(self, tool):
        self.c.apply(self.q)
        d = self.c.d
        yb = self.c.m.body('wrist_yaw').id
        return d.xpos[yb] + d.xmat[yb].reshape(3, 3) @ M.TOOL[tool]

    def press(self, key, travel=.003, stand=.04):
        """Fixed fingertip on a front-panel key, pressed straight in."""
        self.c.apply(self.q)
        k = self.c.point(key)
        face = k.copy(); face[1] = -.008
        for dl in np.radians([0, 10, -10, 20, -20, 30, -30]):
            psi = math.pi / 2 + dl
            h = np.array([math.cos(psi), math.sin(psi), 0])
            pre = face - stand * h
            qa = self.choose(pre, psi, 'tip', M.GRIP['tip'], self.q)
            if qa is None:
                continue
            snap = (self.q.copy(), len(self.wp), self.t)
            try:
                self.joint_to(qa)
                if self.line(pre, face + travel * h, psi, psi, 'tip', M.GRIP['tip'], v=V_FINE) and \
                        self.line(face + travel * h, pre, psi, psi, 'tip', M.GRIP['tip'], v=V_FINE):
                    return
            except RuntimeError:
                pass
            self.q, self.t = snap[0], snap[2]
            del self.wp[snap[1]:]
        raise RuntimeError(f'cannot press {key}')

    def pick(self, base, out_y=-.12, lift=.015, psi_pref=None):
        """Grasp the jar standing at base (its bottom centre) and take it out to y = out_y."""
        g = np.asarray(base) + [0, 0, M.JAR['grip']]
        cands = list(self.jar_headings(base, out_y, lift))
        if psi_pref is not None:
            cands = [(psi_pref, how) for _, how in cands[:2]] + cands
        for (psi, how), opening in [(c, o) for o in (M.GRIP['open'], M.GRIP['open'] * .6) for c in cands]:
            h = np.array([math.cos(psi), math.sin(psi), 0])
            pre = g - .09 * h
            qa = self.choose(pre, psi, 'jar', opening, self.q)
            if qa is None:
                continue
            snap = (self.q.copy(), len(self.wp), self.t)
            try:
                self.joint_to(qa)
                ok = self.line(pre, g, psi, psi, 'jar', opening, v=V_FINE)
                if ok:
                    self.grip_to(M.GRIP['jar'])
                    self.c.held = True
                    up = g + [0, 0, lift]
                    ok = self.line(g, up, psi, psi, 'jar', M.GRIP['jar'], v=V_FINE)
                    for a, b in how(up, h):
                        ok = ok and self.line(a, b, psi, psi, 'jar', M.GRIP['jar'], v=V_FINE * 2)
                if ok:
                    return psi
            except RuntimeError:
                pass
            self.c.held = False
            self.q, self.t = snap[0], snap[2]
            del self.wp[snap[1]:]
        raise RuntimeError('cannot pick the jar')

    def jar_headings(self, base, out_y, lift):
        def back(up, h):
            end = up - h * ((up[1] - out_y - M.JAR['r_top']) / max(h[1], 1e-3))
            return [(up, end)]

        def front(up, h):
            end = up.copy(); end[1] = out_y + 0
            return [(up, end)]
        for deg in list(range(90, 181, 10)) + list(range(80, -1, -10)):
            for how in (back, front):
                yield math.radians(deg), how

    def place(self, base, psi_pref=None, lift=.015, approach_from=None):
        """Carry the held jar to stand at base: in from approach_from (straight), set down, let go, back out."""
        g = np.asarray(base) + [0, 0, M.JAR['grip']]
        up = g + [0, 0, lift]
        cands = [psi_pref] if psi_pref is not None else []
        cands += [math.radians(dg) for dg in list(range(90, 181, 10)) + list(range(80, -1, -10))]
        for psi in cands:
            h = np.array([math.cos(psi), math.sin(psi), 0])
            start = approach_from(up, h) if approach_from else up - .10 * h
            qa = self.choose(start, psi, 'jar', M.GRIP['jar'], self.q)
            if qa is None:
                continue
            snap = (self.q.copy(), len(self.wp), self.t)
            try:
                self.joint_to(qa)
                ok = self.line(start, up, psi, psi, 'jar', M.GRIP['jar'], v=V_FINE * 2) and \
                    self.line(up, g + [0, 0, .0005], psi, psi, 'jar', M.GRIP['jar'], v=V_FINE)
                if ok:
                    pos, quat = self.c.apply(self.q)          # where the held jar is now
                    self.c.held = False
                    self.c.jar_pos, self.c.jar_quat = pos.copy(), quat.copy()
                    self.grip_to(M.GRIP['open'] * .6)
                    ok = self.line(g, g - .09 * h, psi, psi, 'jar', M.GRIP['open'] * .6, v=V_FINE)
                if ok:
                    return psi
            except RuntimeError:
                pass
            self.c.held = True
            self.q, self.t = snap[0], snap[2]
            del self.wp[snap[1]:]
        raise RuntimeError('cannot place the jar')


# -------------------------------------------------------------------- the morning
def door_loads(cell, q, door, bar, forces=(16.0, 30.0)):
    """Joint torque at the seal break, |r x F| / stall, with the real pose (vertical axes)."""
    cell.apply(q)
    d, m = cell.d, cell.m
    p = cell.point(bar)
    n_out = np.array([-math.sin(cell.env[door]), -math.cos(cell.env[door]), 0])
    axes = dict(shoulder=(d.xpos[m.body('upper_arm').id], C018_STALL * BELT),
                elbow=(d.xpos[m.body('forearm').id], C018_STALL * BELT),
                yaw=(d.xpos[m.body('wrist_yaw').id], C018_STALL))
    out = {}
    for F in forces:
        Fv = F * n_out
        out[f'{F:.0f}N'] = {k: round(abs((p - o)[0] * Fv[1] - (p - o)[1] * Fv[0]) / lim, 3) for k, (o, lim) in axes.items()}
    return out


def morning(kind='large'):
    c = Cell(kind)
    f = c.f
    hz, z0 = f['handle_z'], f['z0']
    best = None                                  # high, folded over the back of the microwave, inside the board's width
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
    park = best[1]
    P = Planner(c, park)
    out = dict(kind=kind, notes=[])

    P.step('mw_open', 'Press the microwave door key; the door springs ajar; rise onto its pin and swing it open')
    P.press('m_release')
    for th in np.radians(np.linspace(0, 10, 6))[1:]:
        c.env['mw_door'] = th
        P._push(.08)
    P.grasp_handle('mw_door', 'm_bar')
    mw_open = P.door_arc('mw_door', 'm_bar', c.env['mw_door'], math.radians(110))
    out['mw_door_deg'] = round(math.degrees(mw_open), 1)
    P.release_handle()

    P.step('fridge_open', 'Rise onto the fridge door pin, break the seal, swing the door open')
    P.grasp_handle('fridge_door', 'f_bar', score=lambda q: max(door_loads(c, q, 'fridge_door', 'f_bar', (16.0,))['16N'].values()))
    out['seal_loads'] = door_loads(c, P.q, 'fridge_door', 'f_bar')
    fr_open = P.door_arc('fridge_door', 'f_bar', 0.0, math.radians(110))
    out['fridge_door_deg'] = round(math.degrees(fr_open), 1)
    P.release_handle()

    P.step('pick', 'Reach into the fridge, grip the jar at mid-height, lift 15 mm, take it out')
    base = c.jar_pos.copy()
    out['pick_heading_deg'] = round(math.degrees(P.pick(base)), 1)

    P.step('to_microwave', 'Carry the jar up to the microwave and set it on the turntable centre')
    mc = f['mw_centre']
    tbase = np.array([mc[0], mc[1], f['plate_z'] + .0005])
    def from_front(up, h):
        s = up.copy(); s[1] = -.10
        return s
    psi_mw = P.place(tbase, approach_from=from_front)
    out['place_heading_deg'] = round(math.degrees(psi_mw), 1)

    P.step('mw_close', 'Swing the microwave door shut by its pin, press start')
    P.grasp_handle('mw_door', 'm_bar')
    shut = P.door_arc('mw_door', 'm_bar', c.env['mw_door'], 0.0)
    out['mw_door_closed'] = shut < math.radians(.5)
    P.release_handle()
    P.press('m_start')

    P.step('fridge_close', 'Swing the fridge door shut by its pin')
    P.grasp_handle('fridge_door', 'f_bar')
    shut = P.door_arc('fridge_door', 'f_bar', c.env['fridge_door'], 0.0)
    out['fridge_door_closed'] = shut < math.radians(.5)
    P.release_handle()

    P.step('heat', 'Heat (shown sped up; the real time is set by the recipe)')
    P.wait(4.0, spin=2 * math.pi * 1.2)

    P.step('mw_reopen', 'Press the door key, swing the door open again')
    P.press('m_release')
    for th in np.radians(np.linspace(0, 10, 6))[1:]:
        c.env['mw_door'] = th
        P._push(.08)
    P.grasp_handle('mw_door', 'm_bar')
    P.door_arc('mw_door', 'm_bar', c.env['mw_door'], mw_open)
    P.release_handle()

    P.step('serve', 'Take the hot jar out and stand it on top of the microwave')
    P.pick(c.jar_pos.copy(), out_y=-.10, psi_pref=psi_mw)
    sbase = np.array([-.12, .12, f['mw_top'] + .0005])
    def from_above_front(up, h):
        s = up - .10 * h; s[2] = up[2] + .03
        return s
    out['serve_heading_deg'] = round(math.degrees(P.place(sbase, approach_from=from_above_front)), 1)

    P.step('mw_shut', 'Close the microwave door and park')
    P.grasp_handle('mw_door', 'm_bar')
    P.door_arc('mw_door', 'm_bar', c.env['mw_door'], 0.0)
    P.release_handle()
    P.joint_to(park)
    P.step('done', 'Parked')
    return c, P, out


def summarize(c, P, out):
    worst = {}
    pen = []
    for w in P.wp:
        for dist, a, b in w['con']:
            key = (w['label'], ) + tuple(sorted((a, b)))
            if key not in worst or dist < worst[key]:
                worst[key] = dist
            if dist < -1e-4:
                pen.append(dict(t=round(w['t'], 2), step=w['label'], pair=[a, b], depth_mm=round(-dist * 1e3, 1)))
    by_step = {}
    for (step, a, b), dist in worst.items():
        cur = by_step.get(step)
        if cur is None or dist < cur['clearance_mm'] / 1e3:
            by_step[step] = dict(clearance_mm=round(dist * 1e3, 1), pair=[a, b])
    out['duration_s'] = round(P.t, 1)
    out['penetrations'] = pen[:40]
    out['n_penetrating_frames'] = len({p['t'] for p in pen})
    out['closest_by_step'] = by_step
    out['steps'] = P.steps
    jc = M.JAR_IN_PITCH
    tau = 9.81 * (M.JAR['m'] * jc + .12 * .045)
    out['pitch_hold'] = dict(lever_m=jc, torque_Nm=round(tau, 3), of_rated=round(tau / .981, 2), of_stall=round(tau / 2.942, 2))
    return out


def frames(c, P):
    """Resample the dense waypoints to FPS: full qpos per frame."""
    ts = np.array([w['t'] for w in P.wp])
    out, labels = [], []
    adr = c.adr
    for t in np.arange(0, ts[-1] + 1e-9, 1 / FPS):
        k = int(np.searchsorted(ts, t, side='right') - 1)
        k2 = min(k + 1, len(ts) - 1)
        a, b = P.wp[k], P.wp[k2]
        s = 0 if k2 == k or ts[k2] == ts[k] else (t - ts[k]) / (ts[k2] - ts[k])
        q = a['q'] + (b['q'] - a['q']) * s
        env = {n: a['env'][n] + (b['env'][n] - a['env'][n]) * s for n in a['env']}
        jar = a['jar'] + (b['jar'] - a['jar']) * s
        jar[3:] /= np.linalg.norm(jar[3:])
        qpos = np.zeros(c.m.nq)
        for i, n in enumerate(ARM):
            qpos[adr[n]] = q[i]
        for n, v in env.items():
            qpos[adr[n]] = v
        qpos[adr['cw']] = M.CW_C - q[0]
        qpos[adr['jar']:adr['jar'] + 7] = jar
        out.append([round(float(v), 4) for v in qpos])
        labels.append(a['label'])
    return out, labels


# -------------------------------------------------------------------- coverage, detailed
def coverage(kind, door_deg, step=.04):
    """The shelf test on the detailed model: open hand in, close, lift 15 mm and a
    straight path out (back along the approach, or straight to the front), door at door_deg."""
    c = Cell(kind)
    f = c.f
    c.env['fridge_door'] = math.radians(door_deg)
    W, D, t, y0 = f['W'], f['D'], f['t'], f['y0']
    res = {}
    for z in f['shelves'] if kind == 'large' else f['shelves'][1:] + [f['shelves'][0]]:
        pts = [(float(x), float(y), z) for x in np.arange(-W + t + .06, -t - .06 + 1e-9, step)
               for y in np.arange(y0 + .06, D - .05 - .06 + 1e-9, step)]
        okc = 0
        for p in pts:
            okc += reachable(c, np.array(p) + [0, 0, .0005])
        res[f'{z:.2f}'] = dict(reachable=okc, of=len(pts), share=round(okc / len(pts), 3))
    tot = sum(v['reachable'] for v in res.values()); n = sum(v['of'] for v in res.values())
    return dict(by_shelf=res, share=round(tot / n, 3), door_deg=door_deg)


def reachable(c, base, out_y=-.12):
    c.jar_pos = np.asarray(base, float)
    g = c.jar_pos + [0, 0, M.JAR['grip']]
    for deg in range(0, 360, 15):
        psi = math.radians(deg)
        h = np.array([math.cos(psi), math.sin(psi), 0])
        if h[1] < .05:
            continue
        for q in [qq for o in (M.GRIP['open'], M.GRIP['open'] * .6) for qq in solutions(g, psi, 'jar', o)]:
            c.held = False
            if not c.ok(q):
                continue
            qc = q.copy(); qc[6] = M.GRIP['jar']
            if not _path(c, g - .07 * h, g, psi, q[6], q, held=False):
                continue
            up = g + [0, 0, .015]
            back = up - h * ((up[1] - out_y) / h[1])
            front = up.copy(); front[1] = out_y
            if _path(c, up, back, psi, M.GRIP['jar'], qc, held=True) or _path(c, up, front, psi, M.GRIP['jar'], qc, held=True):
                c.held = False
                return True
    c.held = False
    return False


def _path(c, a, b, psi, grip, q_ref, held, step=.01):
    c.held = held
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    prev = q_ref
    for s in np.linspace(0, 1, n):
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


def main():
    t0 = time.time()
    results, seqs = {}, {}
    for kind in ('large', 'small'):
        c, P, out = morning(kind)
        out = summarize(c, P, out)
        fr, labels = frames(c, P)
        seqs[kind] = dict(steps=P.steps, labels=labels, frames=fr, nq=int(c.m.nq))
        if '--no-coverage' not in sys.argv:
            out['coverage'] = coverage(kind, out['fridge_door_deg'])
            out['coverage_door_90'] = coverage(kind, 90.0)
        results[kind] = out
        print(kind, json.dumps({k: v for k, v in out.items() if k not in ('steps', 'penetrations', 'closest_by_step')}), flush=True)
    results['runtime_s'] = round(time.time() - t0, 1)
    (HERE / 'sequence-results.json').write_text(json.dumps(results, indent=1))
    (HERE / 'sequence.json').write_text(json.dumps(dict(fps=FPS, adr=c.adr, sequences=seqs), separators=(',', ':')))


if __name__ == '__main__':
    main()
