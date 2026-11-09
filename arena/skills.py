"""The reference controller: one set of skills for every robot. It sees only sensors.

A bad script must not count as bad hardware, so every entry gets the same code unless it brings its own (Robot may later
carry a policy). The controller knows what a real robot would know: the mapped floor plan, its own kinematics at nominal
values, where appliances are (mapped, +-1 cm), and the object list. It does not know where things are, what they
weigh, how slippery they are, or where it stands; it localizes against the lidar and looks with its cameras.

Skills are generators: each `yield` is one 20 ms control tick. plan() composes them, retries on detected failure, and
gives up honestly, because a robot that knows it failed is worth more than one that does not.

Whole-body motion: base forward, base yaw and every chain axis are columns of one Jacobian, so the same servo drives a
telescoping arm, a robot whose base is its reach axis, and a six-axis arm. Sideways is not a column: a diff drive has none.
"""
from __future__ import annotations

import heapq
import math

import mujoco
import numpy as np
from scipy import ndimage

from . import assemble, objects, reach
from .contacts import world_cands
from .reach import CELL, HEADINGS, Contact, rz
from .sim import CTRL_DT, Obs, World
from .rotation_error import tool_error

POLICIES = {
    "reference": set(),
    "local": {"local"},
    "carry": {"carry"},
    "placement": {"placement"},
    "vision": {"vision"},
    "improved": {"local", "carry", "placement", "vision", "paths"},
}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Localizer:
    """Odometry from the wheels, corrected by matching lidar ranges against the mapped walls and furniture."""

    def __init__(self, built, pose, lidar_xy):
        self.x, self.y, self.th = pose
        self.b = built
        self.lx, self.ly = lidar_xy
        self.w_prev = None
        m = built.m
        res = 0.02
        self.res = res
        X, Y = int(10.6 / res), int(8.8 / res)
        occ = np.zeros((X, Y), bool)
        z = built.robot.base.lidar[2]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0 or m.geom_contype[g] == 0 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            c, s = m.geom_pos[g], m.geom_size[g]
            if abs(m.geom_quat[g, 0]) < 0.999 or c[2] - s[2] > z or c[2] + s[2] < z:
                continue
            i0, i1 = int((c[0] - s[0] + 0.3) / res), int(math.ceil((c[0] + s[0] + 0.3) / res))
            j0, j1 = int((c[1] - s[1] + 0.1) / res), int(math.ceil((c[1] + s[1] + 0.1) / res))
            occ[max(i0, 0):i1, max(j0, 0):j1] = True
        self.dist = np.minimum(ndimage.distance_transform_edt(~occ) * res, 0.30)
        self.a = np.arange(360) * 2 * math.pi / 360
        self.quality = 1.0
        self.frozen = False

    def predict(self, wheels, track, r):
        if self.w_prev is None:
            self.w_prev = wheels.copy()
            return
        dl, dr = (wheels - self.w_prev) * r
        self.w_prev = wheels.copy()
        ds, dth = (dl + dr) / 2, (dr - dl) / track
        self.x += ds * math.cos(self.th + dth / 2)
        self.y += ds * math.sin(self.th + dth / 2)
        self.th += dth

    def _cost(self, x, y, th, r, a):
        c, s = np.cos(th + a), np.sin(th + a)
        lx = x + self.lx * math.cos(th) - self.ly * math.sin(th)
        ly = y + self.lx * math.sin(th) + self.ly * math.cos(th)
        px, py = lx + r * c, ly + r * s
        i = np.clip(((px + 0.3) / self.res).astype(int), 0, self.dist.shape[0] - 1)
        j = np.clip(((py + 0.1) / self.res).astype(int), 0, self.dist.shape[1] - 1)
        return float(np.mean(self.dist[i, j]))

    def correct(self, scan):
        if self.frozen:                                   # final approach: wheels only, a pose that jumps 10 times a
            return                                        # second turns into commanded velocity and then into force
        ok = np.isfinite(scan) & (scan < 8.0) & (scan > 0.08)
        if ok.sum() < 40:
            return
        r, a = scan[ok], self.a[ok]
        best = (self._cost(self.x, self.y, self.th, r, a), 0.0, 0.0, 0.0)
        # a good match needs only a small refinement; a bad one (wheels slipped, robot shoved) searches wider
        levels = ((0.02, 0.012), (0.007, 0.004)) if best[0] < 0.035 else ((0.10, 0.20), (0.04, 0.07), (0.015, 0.025), (0.007, 0.008))
        for step, ang in levels:
            bx, by, bt = best[1], best[2], best[3]
            for dx in (-step, 0, step):
                for dy in (-step, 0, step):
                    for dt in (-ang, 0, ang):
                        c = self._cost(self.x + bx + dx, self.y + by + dy, self.th + bt + dt, r, a)
                        if c < best[0]:
                            best = (c, bx + dx, by + dy, bt + dt)
        gain = 0.35 if self.quality < 0.035 else 0.7
        self.x += gain * best[1]
        self.y += gain * best[2]
        self.th += gain * best[3]
        self.quality = best[0]


class Controller:
    def __init__(self, world: World, belief: assemble.Built, task, T=None, initial_pose=None, policy="improved"):
        self.task = task                     # the world is used once, below, for the pose it starts from: the dock it knows
        self.T = T or reach.Tier1(belief)    # the robot's own model of itself and its home: nominal, no noise
        self.r = belief.robot
        self.B = belief
        self.obs = None
        self.n = self.T.n
        # The caller supplies a nominal dock/planned pose, never a simulator pose.
        x, y, yaw = initial_pose if initial_pose is not None else self.T.dock_pose()
        lid = self.r.base.lidar
        self.loc = Localizer(belief, (x, y, yaw), lid[:2])
        self.mem = {}                         # last estimate of each object: world position, time
        self.log = []
        self.q_cmd = np.array([a.home for a in self.r.chain])
        self.home_q = self.q_cmd.copy()
        self.grip = 0.05
        self.vmax = np.array([a.speed * 0.7 for a in self.r.chain])
        self.result = None
        self.gen = None
        self.cam_site = {c.name: mujoco.mj_name2id(belief.m, mujoco.mjtObj.mjOBJ_SITE, f"cam_{c.name}") for c in self.r.cams}
        self.cam_cfg = {c.name: c for c in self.r.cams}
        self.d = self.T.d
        self.held = None
        self.hint = None
        self.carry_payload = 0.0
        self.options = POLICIES[policy]
        self.policy = policy
        self.stage = "startup"
        self.events = []
        self.failure = None
        self.held_transform = None
        self.pose_history = []
        # numbers a learner may tune (arena.learn); the defaults are the reference controller
        self.tune = dict(pre=0.09, look_up=0.12, in_speed=0.04, in_guard=0.6, drop_extra=0.0)

    def mark(self, stage, **details):
        self.stage = stage
        self.events.append(dict(t=round(self.obs.t, 2) if self.obs else 0, stage=stage, **details))

    def fail(self, reason, message):
        self.failure = reason
        self.log.append(message)
        self.events.append(dict(t=round(self.obs.t, 2), stage=self.stage, failure=reason))

    # ------------------------------------------------------------ belief --
    def pose(self):
        return self.loc.x, self.loc.y, self.loc.th

    def _belief(self):
        x, y, th = self.pose()
        self.T._set(x, y, th, self.obs.q, tilt=(self.obs.pitch, self.obs.roll))

    def ee(self):
        self._belief()
        return self.d.site_xpos[self.T.ee].copy(), self.d.site_xmat[self.T.ee].reshape(3, 3).copy()

    def update(self, ob: Obs):
        self.obs = ob
        b = self.r.base
        self.loc.predict(ob.wheels, 2 * b.wheel_y, b.wheel_r)
        if ob.scan is not None:
            self.loc.correct(ob.scan)
        self.pose_history.append((ob.t, self.pose(), ob.q.copy(), (ob.pitch, ob.roll)))
        self.pose_history = self.pose_history[-40:]
        for det in ob.dets:
            if "vision" in self.options and self.pose_history:
                _, pose, q, tilt = min(self.pose_history, key=lambda row: abs(row[0]-det.t))
                self.T._set(*pose, q, tilt=tilt)
            else:
                self._belief()
            site = self.cam_site[det.cam]
            c = self.cam_cfg[det.cam]
            p = self.d.site_xpos[site].copy()
            R = self.d.site_xmat[site].reshape(3, 3)
            look = R @ (np.array(c.look) / np.linalg.norm(c.look))
            up = R @ np.array([0, 0, 1.0])
            right = np.cross(look, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, look)
            w = p + right * det.pos[0] + up * det.pos[1] + look * det.pos[2]
            m = self.mem.get(det.name)
            yaw = det.yaw
            if m is not None and np.linalg.norm(m[0] - w) < 0.06 and ob.t - m[1] < 3.0:
                w = m[0] + 0.35 * (w - m[0])          # low-pass over repeated looks
                if "vision" in self.options:
                    yaw = _wrap(m[2]+.35*_wrap(yaw-m[2]))
            self.mem[det.name] = (w, ob.t, yaw, det.cam, det.vis, det.t)

    # ------------------------------------------------------------- moves --
    def wait(self, s):
        for _ in range(int(s / CTRL_DT)):
            yield dict(q=self.q_cmd)

    def move_q(self, q, speed=1.0, guard=None, grip=None, squeeze=False):
        """Straight line in joint space at the speed limits, smooth at both ends. guard: stop on force (fraction of rating)."""
        q0 = self.q_cmd.copy()
        q = np.clip(q, self.T.lo, self.T.hi)
        dt = float(np.max(np.abs(q - q0) / (self.vmax * speed))) * 1.875
        dt = max(dt, 0.08)
        n = max(int(dt / CTRL_DT), 2)
        for i in range(1, n + 1):
            s = i / n
            u = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            self.q_cmd = q0 + (q - q0) * u
            vel = (q - q0) * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / dt
            if guard and i > 3 and self._forced(guard):
                self.q_cmd = np.array(self.obs.q)
                return False
            yield dict(q=self.q_cmd, qd=vel, grip=self.grip if grip is None else grip, squeeze=squeeze)
        return True

    def _gravity(self):
        """Motor forces the robot's own model expects at this pose: current sensing is only useful after subtracting these."""
        T = self.T
        self._belief()
        T.d.qvel[:] = 0
        T.d.qacc[:] = 0
        res = np.zeros(T.m.nv)
        mujoco.mj_rne(T.m, T.d, 0, res)
        return np.array([sum(res[v] for v in ax["v"]) / len(ax["v"]) for ax in T.axes])

    def _forced(self, frac):
        """Force on any axis beyond gravity at this pose and what it read when tared: the hand touched something."""
        hold = getattr(self, "_tau0", None)
        if hold is None:
            return False
        # what current sensing can resolve: about 40 N on a slide, 2 N m on a hinge, at frac = 1
        # a screw's friction reverses with direction, so an axis is judged above three times its own friction
        lim = frac * np.array([40.0 if a.kind == "slide" else 2.0 for a in self.r.chain]) + 3 * np.array([a.friction for a in self.r.chain])
        r = np.abs(self.obs.tau - self._gravity() - hold)
        self._r = 0.75 * getattr(self, "_r", r) + 0.25 * r  # 100 ms low-pass: current ripple and acceleration are not contact
        r = self._r
        if np.any(r > lim):
            self.log.append(f"guard {self.r.chain[int(np.argmax(r / lim))].name} {float(np.max(r / lim)):.1f} x limit")
            return True
        return False

    def stow(self, keep_lift=None, grip=None):
        """Tuck the arm: reach first, then the rest; the lift stays where the hand is safe."""
        q = self.home_q.copy()
        names = [a.name for a in self.r.chain]
        if self.held and "carry" in self.options:
            for i, name in enumerate(names):
                if name in ("pitch", "roll", "wpitch", "wroll") or name.startswith("j"):
                    q[i] = self.q_cmd[i]
        if keep_lift is not None and "lift" in names:
            q[names.index("lift")] = keep_lift
        if "paths" in self.options:
            if grip is not None:
                self.grip = grip
            ok = yield from self.arm_to(q)
            if not ok:
                self.fail("stow_path", "no checked path to travelling posture")
            return ok
        if "lift" in names:                                # up first: tucking a low hand runs it into the deck
            k = names.index("lift")
            if self.q_cmd[k] < q[k]:
                up = self.q_cmd.copy()
                up[k] = q[k]
                yield from self.move_q(up, grip=grip)
        yield from self.move_q(q, grip=grip)

    def open_hand(self, width=None):
        width = width if width is not None else min(self.r.grip.stroke, 0.13)
        self.grip = width
        stable = 0
        for _ in range(int(self.r.grip.stroke / self.r.grip.speed / CTRL_DT)+100):
            yield dict(q=self.q_cmd, grip=width)
            stable = stable+1 if self.obs.grip >= width-.002 else 0
            if stable >= 3:
                return True
        self.fail("gripper_did_not_open", "measured jaw opening did not reach release target")
        return False

    def close_hand(self):
        """Close until the squeeze builds up or the jaws meet. Returns the opening: 0 means nothing is in the hand."""
        self.grip = 0.0
        last = None
        for i in range(int(self.r.grip.stroke / self.r.grip.speed / CTRL_DT) + 25):
            yield dict(q=self.q_cmd, grip=0.0, squeeze=True)
            if self.obs.grip_f > 0.8 * self.r.grip.force and i > 8:
                break
        for _ in range(15):
            yield dict(q=self.q_cmd, grip=0.0, squeeze=True)
        return self.obs.grip

    # ------------------------------------------------------------- drive --
    def path(self, goal, near=3):
        """A* over (x, y, heading): straight moves and turns on the spot, as a diff drive moves."""
        T = self.T
        free = T.free
        if not hasattr(T, "room"):                        # clearance per heading: paths keep to the middle of doorways
            T.room = np.stack([ndimage.distance_transform_edt(free[h]) for h in range(HEADINGS)]) * CELL
        room = T.room
        x, y, th = self.pose()
        a = 2 * math.pi / HEADINGS
        s0 = (int((x + 0.2) / CELL), int(y / CELL), int(round(th / a)) % HEADINGS)
        g = (int((goal[0] + 0.2) / CELL), int(goal[1] / CELL), int(round(goal[2] / a)) % HEADINGS)
        steps = [(int(round(math.cos(h * a) * 3)), int(round(math.sin(h * a) * 3))) for h in range(HEADINGS)]

        def h(n):
            return math.hypot(n[0] - g[0], n[1] - g[1]) * CELL

        openl = [(h(s0), 0.0, s0, None)]
        came, cost = {}, {s0: 0.0}
        found = None
        tries = 0
        while openl and tries < 150000:
            tries += 1
            f, c, n, par = heapq.heappop(openl)
            if n in came:
                continue
            came[n] = par
            if abs(n[0] - g[0]) <= near and abs(n[1] - g[1]) <= near:
                found = n
                break
            nb = []
            dx, dy = steps[n[2]]
            nb.append(((n[0] + dx, n[1] + dy, n[2]), 3 * CELL))
            nb.append(((n[0] - dx, n[1] - dy, n[2]), 3 * CELL * 2.5))
            nb.append(((n[0], n[1], (n[2] + 1) % HEADINGS), 0.25))
            nb.append(((n[0], n[1], (n[2] - 1) % HEADINGS), 0.25))
            for m, w in nb:
                if not (0 <= m[0] < free.shape[1] and 0 <= m[1] < free.shape[2]) or m in came:
                    continue
                if m[2] != n[2]:
                    hturn = n[2] if m[2] == (n[2]+1)%HEADINGS else m[2]
                    if hasattr(T, "turn_free") and not T.turn_free[hturn, n[0], n[1]]:
                        continue
                elif any(not free[m[2], int(round(n[0]+(m[0]-n[0])*t)), int(round(n[1]+(m[1]-n[1])*t))] for t in (.33, .67)):
                    continue
                w = w * (1.0 + 3.0 * max(0.0, 0.15 - room[m[2], m[0], m[1]]) / 0.15 + (0.6 if m[2] % 2 else 0.0))
                # the start may sit a little off the lattice, but only straight moves are excused: no pivoting into a wall
                if not free[m[2], m[0], m[1]] and not (m[2] == s0[2] and abs(m[0] - s0[0]) <= 3 and abs(m[1] - s0[1]) <= 3):
                    continue
                nc = c + w
                if nc < cost.get(m, 1e18):
                    cost[m] = nc
                    heapq.heappush(openl, (nc + h(m), nc, m, n))
        if found is None:
            return None
        out = []
        n = found
        while n is not None:
            out.append(((n[0] + 0.5) * CELL - 0.2, (n[1] + 0.5) * CELL, n[2] * a))
            n = came[n]
        return out[::-1]

    def goto(self, gx, gy, gyaw, tol=0.02, tol_yaw=0.02, lift=None, contact=None):
        """Follow the path, then a final approach: turn to the goal, roll in, turn to the heading. lift: raise the hand to
        this height before the last stretch, so it passes over the worktop instead of into it."""
        if self.held and "carry" in self.options:
            self.T.home_q = self.q_cmd.copy()
            self.T._grid()
            if hasattr(self.T, "room"):
                del self.T.room
        self.arrival_contact_sol = None
        def ready():
            if contact is None or "local" not in self.options:
                return False
            target = np.asarray(contact.cands[0][0])
            if np.linalg.norm(target[:2]-np.array(self.pose()[:2])) > self.T.span+.1:
                return False
            sol = self.T.solve(contact, seeds=0, current=np.r_[self.pose(), self.obs.q])
            if sol["ok"] and sol.get("local"):
                self.arrival_contact_sol = sol
                return True
            return False
        if ready():
            return True
        p = self.path((gx, gy, gyaw)) or self.path((gx, gy, gyaw), near=9)
        if p is None:
            x, y, _ = self.pose()
            if math.hypot(gx - x, gy - y) < 0.6:          # the goal cell is tight: edge in directly
                return (yield from self._final(gx, gy, gyaw, tol, tol_yaw))
            self.log.append(f"no route to ({gx:.2f}, {gy:.2f})")
            return False
        vmax = min(self.r.base.vmax, 0.30)
        wps = self._simplify(p)
        for k, (wx, wy, wth, rev) in enumerate(wps):
            if lift is not None and math.hypot(wx - gx, wy - gy) < 0.9:
                names = [a.name for a in self.r.chain]
                if "lift" in names:
                    q = self.q_cmd.copy()
                    q[names.index("lift")] = lift
                    yield from self.move_q(q)
                lift = None
            ok = yield from self._line(wx, wy, vmax, rev)
            if not ok:
                return ready()
            if ready():
                return True
        if lift is not None:
            names = [a.name for a in self.r.chain]
            if "lift" in names:
                q = self.q_cmd.copy()
                q[names.index("lift")] = lift
                yield from self.move_q(q)
        return True if ready() else (yield from self._final(gx, gy, gyaw, tol, tol_yaw))

    def _simplify(self, p):
        out = []
        for i in range(1, len(p)):
            a, b = p[i - 1], p[i]
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            if d < 1e-6:
                continue
            bearing = math.atan2(b[1] - a[1], b[0] - a[0])
            rev = abs(_wrap(bearing - a[2])) > 1.5
            if out and abs(_wrap(out[-1][2] - a[2])) < 1e-6 and out[-1][3] == rev:
                out[-1] = (b[0], b[1], a[2], rev)
            else:
                out.append((b[0], b[1], a[2], rev))
        return out

    def _turn(self, th, wmax=None):
        wmax = wmax or self.r.base.wmax * 0.7
        for _ in range(int(14 / CTRL_DT)):
            e = _wrap(th - self.loc.th)
            if abs(e) < 0.015 and abs(self.obs.qd[0] * 0) < 1:
                return True
            yield dict(q=self.q_cmd, v=0.0, w=float(np.clip(2.2 * e, -wmax, wmax)))
        return False

    def _line(self, gx, gy, vmax, rev=False, tol=0.04):
        x, y, th = self.pose()
        d = math.atan2(gy - y, gx - x) + (math.pi if rev else 0.0)
        ok = yield from self._turn(d)
        if not ok:
            return False
        sgn = -1.0 if rev else 1.0
        last = (self.loc.x, self.loc.y)
        stuck = 0
        for i in range(int(60 / CTRL_DT)):
            x, y, th = self.pose()
            e = math.hypot(gx - x, gy - y)
            if e < tol:
                return True
            head = math.atan2(gy - y, gx - x) + (math.pi if rev else 0.0)
            eh = _wrap(head - th)
            # keep rolling through a yaw kick: stopping on a threshold leaves a wheel jammed against the step
            v = sgn * min(vmax, 0.05 + 0.6 * e) * max(0.0, math.cos(eh)) if abs(eh) < 0.7 else 0.0
            yield dict(q=self.q_cmd, v=v, w=float(np.clip(2.0 * eh, -0.8, 0.8)))
            if i % 50 == 49:                               # 1 s: did it move at all?
                moved = math.hypot(self.loc.x - last[0], self.loc.y - last[1])
                last = (self.loc.x, self.loc.y)
                stuck = stuck + 1 if moved < 0.01 else 0
                if stuck >= 4:
                    self.log.append("drive: stuck")
                    return False
        return False

    def _final(self, gx, gy, gyaw, tol, tol_yaw):
        for attempt in range(3):
            x, y, th = self.pose()
            if math.hypot(gx - x, gy - y) > tol:
                rev = abs(_wrap(math.atan2(gy - y, gx - x) - gyaw)) > 1.6
                ok = yield from self._line(gx, gy, 0.10, rev, tol=tol * 0.7)
                if not ok:
                    return False
            ok = yield from self._turn(gyaw, 0.5)
            if not ok:
                return False
            yield from self._hold(0.4)
            x, y, th = self.pose()
            if math.hypot(gx - x, gy - y) < tol * 1.5 and abs(_wrap(gyaw - th)) < tol_yaw * 1.5:
                return True
        self.log.append("drive: final pose tolerance not reached")
        return False

    def _hold(self, s):
        for _ in range(int(s / CTRL_DT)):
            yield dict(q=self.q_cmd, v=0.0, w=0.0)

    # -------------------------------------------------------------- look --
    def see(self, name, wait=1.0, settle=0.3):
        """A fresh look, standing still. An image is converted with the arm pose of its capture time, and the camera
        delivers 100 ms late: only images taken after the arm stopped count, or a moving hand misplaces the target."""
        t0 = self.obs.t
        for _ in range(int(settle / CTRL_DT)):
            yield dict(q=self.q_cmd)
        t1 = self.obs.t
        self.mem.pop(name, None)
        while self.obs.t - t0 < wait + settle:
            m = self.mem.get(name)
            if m and m[5] >= t1:
                return m
            yield dict(q=self.q_cmd)
        return None

    def scan_for(self, name, hint=None):
        """Turn on the spot in 40 deg steps until it shows up; if a hint says where, walk round it and look again."""
        for ring in range(3):
            th0 = self.loc.th
            for k in range(9):
                m = yield from self.see(name, 0.7)
                if m:
                    return m
                yield from self._turn(th0 + (k + 1) * 0.7, 0.6)
            hint = hint or self.hint
            if hint is None:
                return None
            yield from self.look_near(hint[0] + (0.5 if ring == 0 else -0.5), hint[1] + (-0.3 if ring == 0 else 0.3), radius=0.9)
        return None

    def look_pose(self, target):
        """Arm pose that puts the target well inside a camera's view, base where it stands. Searched, not scripted, so
        a hand camera, a carriage camera and a mast camera are all served by the same code."""
        rng = np.random.default_rng(int(self.obs.t * 50) % 2 ** 31)
        x, y, th = self.pose()
        best, bq = 1e9, None
        T = self.T
        T._allow, T._allow_pref = set(), ()
        tgt = np.asarray(target, float)
        for k in range(160):
            q = self.home_q.copy() if k == 0 else self.q_cmd + rng.normal(0, 0.35, self.n) * (T.hi - T.lo) * 0.5 if k % 2 else T.lo + (T.hi - T.lo) * rng.uniform(0, 1, self.n)
            q = np.clip(q, T.lo, T.hi)
            T._set(x, y, th, q)
            for name, c in self.cam_cfg.items():
                site = self.cam_site[name]
                p = self.d.site_xpos[site]
                R = self.d.site_xmat[site].reshape(3, 3)
                look = R @ (np.array(c.look) / np.linalg.norm(c.look))
                v = tgt - p
                dist = float(np.linalg.norm(v))
                if dist < c.near + 0.15 or dist > c.far * 0.85:
                    continue
                off = math.degrees(math.acos(float(np.clip(look @ (v / dist), -1, 1))))
                if off > min(c.fov) * 0.35:
                    continue
                cost = off * 0.02 + dist * 0.5 + 0.3 * float(np.abs((q - self.home_q) / (T.hi - T.lo + 1e-6)).sum())
                if cost < best and not T._clear(None, None, False):
                    best, bq = cost, q.copy()
        return bq

    def aim(self, target):
        q = self.look_pose(target)
        if q is not None:
            yield from self.arm_to(q)
            yield from self._hold(0.3)
        return q is not None

    def look_near(self, hx, hy, radius=1.1, z=0.0):
        """Drive to where the hinted place is in view of a camera: a free spot about a metre away, facing it."""
        x, y, th = self.pose()
        best = None
        any_h = self.T.access.any(0)
        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            px, py = hx + radius * math.cos(a), hy + radius * math.sin(a)
            i, j = int((px + 0.2) / CELL), int(py / CELL)
            if not (0 <= i < any_h.shape[0] and 0 <= j < any_h.shape[1]) or not any_h[i, j]:
                continue
            d = math.hypot(px - x, py - y)
            if best is None or d < best[0]:
                best = (d, px, py)
        if best is None:
            return False
        yaw = math.atan2(hy - best[2], hx - best[1]) - self._cam_bearing()
        yield from self.stow()
        ok = yield from self.goto(best[1], best[2], yaw, tol=0.05, tol_yaw=0.05)
        if ok:
            yield from self.aim((hx, hy, z))
        return ok

    def _cam_bearing(self):
        """Which way the widest-ranging camera looks, relative to the base, with the arm stowed."""
        c = max(self.r.cams, key=lambda k: k.far)
        self.T._set(0, 0, 0, self.home_q)
        R = self.d.site_xmat[self.cam_site[c.name]].reshape(3, 3)
        v = R @ np.array(c.look, float)
        return math.atan2(v[1], v[0])

    def fridge_open(self):
        """With the kit: ask, wait for it. Without: the handle by hand (see tier 1 for whether the body can)."""
        if "fridge_opener" in self.r.kits:
            for _ in range(int(9 / CTRL_DT)):
                yield dict(q=self.q_cmd, kit="fridge_open")
            return True
        self.log.append("fridge: no opener, handle skill not implemented for this body")
        return False

    def fridge_close(self):
        if "fridge_opener" in self.r.kits:
            for _ in range(int(9 / CTRL_DT)):
                yield dict(q=self.q_cmd, kit="fridge_close")
            return True
        return False

    def microwave_door(self, opening):
        """Observe, pinch the free edge and follow its hinge arc with the actual hand.

        Door angle is estimated from camera detections; no simulator joint value
        or door encoder is available to this controller.
        """
        W = self.B.world
        self.mark("microwave_open" if opening else "microwave_close")
        if opening:
            if "mw_pusher" not in self.r.kits:
                self.fail("missing_button_skill", "microwave release requires button pusher")
                return False
            yield dict(q=self.q_cmd, kit="mw_release")
            yield from self.wait(2)
        hinge = np.array([*W.doors["mw_door"]["hinge"], self.B.home.counter_h+.14])
        self.hint = tuple(hinge[:2])
        if not (yield from self.look_near(hinge[0]+.17, hinge[1], z=hinge[2])):
            self.fail("door_search_navigation", "cannot reach microwave observation pose")
            return False
        seen = yield from self.scan_for("mw_door")
        if not seen:
            self.fail("door_not_seen", "microwave door not seen")
            return False
        angle = float(np.clip(seen[2], -1.75, 0))
        target = -1.66 if opening else 0.0
        def point(theta):
            R = reach.rz(theta)
            return hinge+R@np.array([W.doors["mw_door"]["width"]-.025, -.0175, 0]), -R[:, 0], R[:, 1]
        p, app, jaw = point(angle)
        contact = Contact("microwave door", "hot meal", "pull", [(p, app, jaw, .035)], "mw_door", force=(0, 3, 0), doors={"mw_door": angle})
        sol = self.plan_contact(contact)
        if not sol["ok"]:
            self.fail("door_grasp_plan_"+sol["why"], "cannot plan microwave door grasp: "+sol["detail"])
            return False
        yield from self.stow()
        if not (yield from self.goto(*sol["base"], lift=self._lift_of(sol["q"]))):
            self.fail("door_navigation", "cannot drive to microwave door")
            return False
        yield from self.open_hand(.06)
        if not (yield from self.arm_to(np.array(sol["q"]))):
            self.fail("door_approach", "microwave edge approach blocked")
            return False
        yield from self.close_hand()
        for theta in np.linspace(angle, target, 12)[1:]:
            p, app, jaw = point(theta)
            if not (yield from self.servo_to(p, app, jaw, speed=.05, timeout=4)):
                self.fail("door_arc", "could not follow microwave hinge arc")
                yield from self.open_hand()
                return False
        yield from self.open_hand(.06)
        yield from self._clear_hand(.08)
        yield from self.stow()
        return True

    # ---------------------------------------------------------- manipulate --
    def plan_contact(self, c: Contact, seeds=24):
        """Pose for this interaction in the robot's own model; a harder search before giving up."""
        rng = np.random.default_rng(int(self.obs.t * 100) % 2 ** 31)
        current = np.r_[self.pose(), self.obs.q] if "local" in self.options else None
        sol = self.T.solve(c, seeds=seeds, rng=rng, current=current)
        return sol if sol["ok"] else self.T.solve(c, seeds=seeds * 3, rng=rng)

    def arm_to(self, q, guard=None):
        """Lift first or last, whichever keeps the hand clear: check both orders in the model, take the free one."""
        names = [a.name for a in self.r.chain]
        i = names.index("lift") if "lift" in names else None
        q0 = self.q_cmd.copy()
        if i is None:
            return (yield from self.move_q(q, guard=guard))
        rising = q[i] > q0[i]
        qa = q0.copy()
        qa[i] = q[i]
        qb = q.copy()
        qb[i] = q0[i]
        first = [qa, q] if rising else [qb, q]
        alt = [qb, q] if rising else [qa, q]
        for seq in (first, alt):
            if self._path_free([q0] + seq):
                for t in seq:
                    ok = yield from self.move_q(t, guard=guard)
                    if not ok:
                        return False
                return True
        if "paths" in self.options:
            from .motion_planner import connect
            lo, hi = self.T.lo.copy(), self.T.hi.copy()
            if self.held:
                for k, a in enumerate(self.r.chain):
                    if a.name in ("pitch", "roll", "wpitch", "wroll"):
                        lo[k], hi[k] = min(q0[k], q[k]), max(q0[k], q[k])+1e-7
            path = connect(q0, q, lo, hi, lambda v: self._path_free([v, v]), seed=int(self.obs.t*100))
            if path:
                self.events.append(dict(t=round(self.obs.t, 2), stage=self.stage, planned_waypoints=len(path)))
                for target in path[1:]:
                    if not (yield from self.move_q(target, guard=guard)):
                        return False
                return True
            self.log.append("arm: no checked joint path ("+getattr(self, "_path_hit", "unknown")+")")
            return False
        ok = yield from self.move_q(q, guard=guard)
        return ok

    def _path_free(self, way):
        x, y, th = self.pose()
        self.T._allow, self.T._allow_pref = set(), ()
        for a, b in zip(way[:-1], way[1:]):
            for s in ([0] if np.array_equal(a, b) else np.linspace(0, 1, 8)):
                self.T._set(x, y, th, a + (b - a) * s)
                hit = self.T._clear(None, None, True)
                if hit:
                    self._path_hit = hit
                    return False
        return True

    def servo_to(self, p, a, j, speed=0.05, tol=0.004, guard=None, timeout=15.0, track=None, base=True):
        """Whole-body servo to a point with the tool along a, jaws along +-j. track(): fresher target, or None."""
        n = self.n
        T = self.T
        lo, hi = T.lo, T.hi
        t0 = self.obs.t
        lam = 1e-3
        W = np.concatenate([[6.0 if base else 1e3, 3.0 if base else 1e3], np.ones(n)])
        cmd_v, cmd_w = 0.0, 0.0
        dq_f = np.zeros(n)
        self.loc.frozen = True
        while self.obs.t - t0 < timeout:
            if track:
                new = track()
                if new is not None:
                    p = new
            self._belief()
            d = self.d
            P = d.site_xpos[T.ee].copy()
            R = d.site_xmat[T.ee].reshape(3, 3)
            xt, yt = R[:, 0], R[:, 1]
            ep = p - P
            er = tool_error(R, a, j)
            err = float(np.linalg.norm(ep))
            if err < tol and float(np.linalg.norm(er)) < 0.12:
                yield from self._hold(0.05)
                self.loc.frozen = False
                return True
            if guard and self._forced(guard):
                self.loc.frozen = False
                return False
            mujoco.mj_jacSite(self.B.m, d, T.jp, T.jr, T.ee)
            Jp = np.zeros((3, 2 + n))
            Jr = np.zeros((3, 2 + n))
            c, s = math.cos(self.loc.th), math.sin(self.loc.th)
            r = P - np.array([self.loc.x, self.loc.y, 0.0])
            Jp[:, 0] = (c, s, 0)
            Jp[:, 1] = (-r[1], r[0], 0)
            Jr[2, 1] = 1.0
            for i, ax in enumerate(T.axes):
                k = len(ax["v"])
                Jp[:, 2 + i] = sum(T.jp[:, v] for v in ax["v"]) / k
                Jr[:, 2 + i] = sum(T.jr[:, v] for v in ax["v"]) / k
            if j is not None:
                Aor, eor = Jr, er
            else:
                Aor, eor = (np.eye(3) - np.outer(xt, xt)) @ Jr, er
            A = np.vstack([Jp, 0.3 * Aor]) / W[None, :]
            e = np.concatenate([ep, 0.3 * eor])
            sc = min(1.0, speed / max(err, 1e-4)) * 2.5 * CTRL_DT
            du = np.linalg.solve(A.T @ A + lam * np.eye(2 + n), A.T @ e) / W
            vb, wb = du[0] * sc / CTRL_DT, du[1] * sc / CTRL_DT
            dq = du[2:] * sc
            dq = np.clip(dq, -self.vmax * CTRL_DT, self.vmax * CTRL_DT)
            dq_f += 0.25 * (dq - dq_f)
            dq = dq_f
            self.q_cmd = np.clip(self.q_cmd + dq, lo, hi)
            cmd_v += 0.25 * (float(np.clip(vb, -0.08, 0.08)) - cmd_v)
            cmd_w += 0.25 * (float(np.clip(wb, -0.3, 0.3)) - cmd_w)
            yield dict(q=self.q_cmd, qd=dq / CTRL_DT, v=cmd_v, w=cmd_w)
        self.loc.frozen = False
        return False

    def tare(self):
        """Remember the motor forces at rest: the guard reads what changes from here."""
        s = np.zeros(self.n)
        for _ in range(10):
            yield dict(q=self.q_cmd)
            s += self.obs.tau - self._gravity()
        self._tau0 = s / 10
        self._r = np.zeros(self.n)

    def grasp_obj(self, name, cat, cands_fn, region_z, attempts=3, kind="grasp"):
        """See it, plan, drive, look again from close by, re-plan if it is not where it seemed, commit, grip, check.
        A first look from 2 m can be 10 cm off; the hand camera at 20 cm is good to millimetres. Returns True when
        something is in the hand."""
        g = self.r.grip
        for att in range(attempts):
            m = self.mem.get(name)
            if not m:
                self.mark("search", obj=name, attempt=att+1)
                m = yield from self.scan_for(name)
                if not m:
                    self.fail("not_found", f"{name}: not found")
                    return False
            pos = m[0].copy()
            yaw = m[2]
            sol = None
            for look in range(4):
                pos[2] = region_z(pos)
                planned = pos.copy()
                planned_yaw = yaw
                c = cands_fn(pos, yaw)
                self.mark("plan_grasp", obj=name, attempt=att+1)
                sol = self.plan_contact(c)
                if not sol["ok"]:
                    self.fail("grasp_plan_"+sol["why"], f"{name}: no plan ({sol['why']} {sol['detail']})")
                    return False
                bx, by, byaw = sol["base"]
                q = np.array(sol["q"])
                x, y, th = self.pose()
                far = math.hypot(bx - x, by - y)
                if far > 0.04 or abs(_wrap(byaw - th)) > 0.08:
                    self.mark("drive_to_grasp", distance=round(far, 3))
                    if far > 0.6:
                        yield from self.stow()
                        ok = yield from self.goto(bx, by, byaw, lift=self._lift_of(q))
                    else:                                  # a short shuffle: lift the hand clear and go, a full stow costs a slow lift a minute
                        yield from self._clear_hand()
                        ok = yield from self._final(bx, by, byaw, 0.02, 0.02)
                    if not ok:
                        self.fail("drive_to_grasp", f"{name}: could not drive to the pose")
                        return False
                x, y, th = self.pose()
                self.T._set(bx, by, byaw, q)
                pg = self.d.site_xpos[self.T.ee].copy()
                R = self.d.site_xmat[self.T.ee].reshape(3, 3)
                app, jaw = R[:, 0].copy(), R[:, 1].copy()
                tn = self.tune
                pre = pg - app * tn["pre"]
                # a hand camera sees best from a hand's length back and above: look from there, then come in
                look_p = pre + np.array([0, 0, tn["look_up"]])
                self.mark("observe_grasp")
                moved = yield from self.arm_to(self._pose_at(x, y, th, q, look_p, app, jaw))
                if not moved and "paths" in self.options:
                    self.fail("observation_path", f"{name}: observation approach collides")
                    return False
                yield from self.open_hand(g.stroke)
                yield from self.tare()
                yield from self.servo_to(look_p, app, jaw, speed=0.10, guard=1.5, timeout=6)
                fresh = yield from self.see(name, 1.2)
                if not fresh:
                    yield from self.servo_to(look_p + np.array([0, 0, 0.10]), app, None, speed=0.10, guard=1.5, timeout=4)
                    fresh = yield from self.see(name, 1.2)
                if not fresh:
                    self.log.append(f"{name}: lost sight of it")
                    self.mem[name] = (pos, self.obs.t, yaw, "memory", 0.0, self.obs.t)
                    break
                moved = float(np.linalg.norm((fresh[0] - pos)[:2]))
                pos, yaw = fresh[0].copy(), fresh[2]
                yaw_moved = abs(_wrap(fresh[2]-planned_yaw))
                if moved < (.015 if "vision" in self.options else .025) and ("vision" not in self.options or yaw_moved < math.radians(8)):
                    break
                self.log.append(f"{name}: {moved * 100:.0f} cm from where it seemed, planning again")
            off = pos - planned                 # what the last look moved it by
            off[2] = 0.0
            # go down first, then in: the hand arrives at the object along its approach, not over the top of it
            yield from self.tare()
            self.mark("pregrasp")
            ok = yield from self.servo_to(pre + off, app, jaw, speed=0.06, guard=2.0, timeout=9)
            if ok:
                yield from self.tare()
                self.mark("final_approach")
                ok = yield from self.servo_to(pg + off + np.array([0, 0, tn["drop_extra"]]), app, jaw, speed=tn["in_speed"], guard=tn["in_guard"], timeout=10)
                if not ok and "paths" in self.options:
                    self.fail("final_approach_blocked", f"{name}: final approach blocked")
                    yield from self.open_hand()
                    yield from self.stow()
                    self.mem.pop(name, None)
                    continue
            else:
                self.fail("pregrasp_blocked", f"{name}: blocked on the way to the pre-grasp")
                yield from self.open_hand()
                yield from self.stow()
                self.mem.pop(name, None)
                continue
            self.mark("close_gripper")
            yield from self.close_hand()
            yield from self.tare()
            P, Rn = self.ee()
            grasp_offset = Rn.T @ (np.r_[pos[:2], region_z(pos)]-P)
            grasp_rotation = Rn.T @ reach.rz(yaw)
            self.mark("lift")
            yield from self.servo_to(P + np.array([0, 0, 0.06]), Rn[:, 0], None, speed=0.04, timeout=5, base=False)
            yield from self._hold(0.3)
            fresh = self.mem.get(name)
            visual_lift = None
            if fresh and fresh[5] > self.obs.t-.4:
                visual_lift = fresh[0][2]-objects.CATALOG[cat].size[2]/2-region_z(pos)
            held = self.obs.grip > .002 and self.obs.grip_f > .15*g.force
            if visual_lift is not None and "vision" in self.options:
                held = held and visual_lift > .015
            if held:
                self.held = name
                if fresh and fresh[5] > self.obs.t-.4 and "placement" in self.options:
                    origin = fresh[0]-np.array([0, 0, objects.CATALOG[cat].size[2]/2])
                    P1, R1 = self.ee()
                    grasp_offset = R1.T@(origin-P1)
                    grasp_rotation = R1.T@reach.rz(fresh[2])
                self.held_transform = (grasp_offset, grasp_rotation)
                self.T.held = (name, grasp_offset, grasp_rotation)
                self.failure = None
                self.mark("grasp_claimed", width=round(self.obs.grip, 4))
                return True
            self.fail("empty_grasp", f"{name}: empty hand on attempt {att + 1}")
            yield from self.open_hand()
            yield from self.stow()
            self.mem.pop(name, None)
        return False

    def _local(self, p):
        return np.array(p, float)

    def _clear_hand(self, up=0.08):
        names = [a.name for a in self.r.chain]
        if "lift" in names:
            k = names.index("lift")
            q = self.q_cmd.copy()
            q[k] = min(q[k] + up, self.T.hi[k])
            yield from self.move_q(q)

    def _lift_of(self, q, extra=0.0):
        names = [a.name for a in self.r.chain]
        return min(float(q[names.index("lift")]) + extra, float(self.T.hi[names.index("lift")])) if "lift" in names else None

    def _pose_at(self, bx, by, byaw, q, p, a, j):
        """Joint values for a hand pose near the planned one, base where it stands."""
        sol = self.T._ik(np.concatenate([[bx, by, byaw], q]), p, a, j, fix_base=True)
        return sol[0][3:] if sol[0] is not None else np.array(q, float)

    def place_obj(self, name, cat, world_pos, yaw, attempts=1):
        """Plan a placement near world_pos, carry it there, set it down until the table pushes back, let go.
        Spots: the requested one, then others within the person's reach."""
        sol = None
        self.mark("plan_place", obj=name)
        z = world_pos[2]
        for dx, dy in ((0, 0), (0.08, 0.12), (0.08, -0.14), (-0.20, 0.0), (0.14, 0.0)):
            p = np.array(world_pos, float) + (dx, dy, 0)
            cs = world_cands(cat, p, yaw, lambda a: a[2] > 0.5)
            exact = "placement" in self.options and self.held_transform is not None
            if exact:
                # Plan a short, physically released drop above the support.
                # Solving directly on it with millimetre IK error rejected
                # otherwise valid poses as interpenetrating the table.
                p[2] += .012
                offset, rel_rotation = self.held_transform
                rotation = reach.rz(yaw) @ rel_rotation.T
                cs = [(p-rotation@offset, rotation[:, 0], rotation[:, 1], self.obs.grip, 0.0)]
            c = Contact("place", "x", "place", cs, cat, (tuple(p), yaw), objects.CATALOG[cat].mass)
            c.exact_tool = exact
            if exact:
                c.tol_deg = 5
            sol = self.plan_contact(c)
            if sol["ok"]:
                break
        if not sol["ok"]:
            self.fail("place_plan_"+sol["why"], f"{name}: no place plan ({sol['why']} {sol['detail']})")
            return False
        bx, by, byaw = sol["base"]
        q = np.array(sol["q"])
        self.mark("carry", obj=name)
        yield from self.stow(keep_lift=None, grip=0.0)
        ok = yield from self.goto(bx, by, byaw, lift=self._lift_of(q, 0.05), contact=c)
        if not ok:
            self.fail("drive_to_place", f"{name}: could not drive to the place pose")
            return False
        if self.arrival_contact_sol:
            sol = self.arrival_contact_sol
            bx, by, byaw = sol["base"]
            q = np.array(sol["q"])
        self.T._set(bx, by, byaw, q)
        pt = self.d.site_xpos[self.T.ee].copy()
        R = self.d.site_xmat[self.T.ee].reshape(3, 3)
        app = R[:, 0].copy()
        q_hi = q.copy()
        names = [a.name for a in self.r.chain]
        if "lift" in names:
            q_hi[names.index("lift")] = min(q[names.index("lift")] + 0.05, self.T.hi[names.index("lift")])
        self.mark("place_approach")
        if not (yield from self.arm_to(q_hi)) and "paths" in self.options:
            self.fail("place_approach_path", "no checked approach to placement")
            return False
        yield from self.tare()
        low = pt + np.array([0, 0, -0.004 if exact else 0.004])
        yield from self.servo_to(low, app, None, speed=0.03, guard=0.35, timeout=8, base=False)
        yield from self._hold(0.2)
        self.mark("release")
        if not (yield from self.open_hand(min(self.obs.grip + 0.04, self.r.grip.stroke))):
            return False
        yield from self._hold(.35)
        self.held = None
        self.held_transform = None
        self.T.held = None
        self.T.home_q = self.home_q.copy()
        P, Rn = self.ee()
        back = P - Rn[:, 0] * 0.10 + np.array([0, 0, 0.02])
        yield from self.servo_to(back, Rn[:, 0], Rn[:, 1], speed=0.05, timeout=6, base=False)
        self.mark("retreat")
        yield from self.stow()
        self.failure = None
        self.mark("delivery_claimed")
        return True

    # -------------------------------------------------------------- run --
    def act(self, ob):
        """One tick: sensors in, commands out."""
        self.update(ob)
        if self.gen is None:
            self.gen = self.task(self)
        try:
            c = next(self.gen)
        except StopIteration as e:
            self.result = e.value if e.value is not None else True
            c = dict(q=self.q_cmd)
        return c
