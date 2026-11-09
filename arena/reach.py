"""Tier 1: can this body do it at all? Reach, fit, access, load, tipping, traction, for every interaction a home demands.

One solver for every robot: inverse kinematics over the chain's axes plus a free planar base (x, y, yaw). A diff drive can
reach any pose it has room for, and a separate check says whether it has: a flood fill of (x, y, heading) from the charging
dock through the real doorways, with the robot's real footprint, gives the poses it can actually get to.

Fails are named, because the reason is the design lesson:
    stroke     the object does not fit between the jaws
    reach      no joint values put the hand there with the needed orientation
    access     the pose exists but the base cannot get there
    collision  the body hits something or comes closer than 5 mm
    motor      an axis would exceed its continuous rating
    tip        centre of pressure closer than 1 cm to the edge
    traction   the wheels would slip
    grip       friction cannot hold the load

These are necessary conditions, computed without dynamics: passing says nothing about contact, perception or control.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy import ndimage

from . import assemble, objects
from .homes import FLOOR_MU, Home

CELL = 0.03
HEADINGS = 8
CLEAR = 0.005


@dataclass
class Contact:
    """One physical interaction. Candidates: world-frame (point, approach, jaw axis or None, width)."""
    name: str
    cls: str                                # need class of the value model
    kind: str                               # grasp | place | press | pull
    cands: list
    obj: str = None                         # catalog name of the object, if any
    obj_pose: tuple = None                  # (pos, yaw) in the world
    mass: float = 0.0
    force: tuple = (0, 0, 0)                # force on the hand from the world while holding this pose
    doors: dict = field(default_factory=dict)
    group: str = None                       # alternatives: the group passes if any member does
    tol_deg: float = 20.0
    exact_tool: bool = False                 # placing with the actual estimated grasp transform


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Tier1:
    def __init__(self, built: assemble.Built, objs=()):
        self.b = built
        self.r = built.robot
        m = self.m = built.m
        self.d = mujoco.MjData(m)
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qb = m.jnt_qposadr[J("base")]
        self.vb = m.jnt_dofadr[J("base")]
        self.axes = []
        for a in self.r.chain:
            names = [a.name] + [f"{a.name}{k}" for k in range(2, a.stages + 1)]
            self.axes.append(dict(a=a, q=[m.jnt_qposadr[J(n)] for n in names], v=[m.jnt_dofadr[J(n)] for n in names]))
        self.n = len(self.axes)
        self.lo = np.array([x["a"].lo for x in self.axes])
        self.hi = np.array([x["a"].hi for x in self.axes])
        self.ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.names = assemble.geom_owner(m)
        n = np.array(self.names, dtype=object)
        self.g_rob = np.array([x.startswith(("rob_", "arm_", "pad_")) for x in n])
        self.g_wheel = np.array([x.startswith(("rob_tyre", "rob_cast")) for x in n])
        self.g_pad = np.array([x.startswith(("pad_", "arm_jaw")) for x in n])
        self.g_obj = np.array([x.startswith("obj_") for x in n])
        self.g_person = np.array([x.startswith("per_") for x in n])
        for i in np.nonzero(self.g_rob & ~self.g_wheel)[0]:
            m.geom_margin[i] = CLEAR
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.jaw = float(self.r.grip.jaw_len / 2)
        mujoco.mj_resetData(m, self.d)
        self._door0 = {k: m.jnt_qposadr[J(k)] for k in built.world.doors}
        self.obj_q = {i: m.jnt_qposadr[J(i)] for i, _ in built.names}
        self.home_q = np.array([x["a"].home for x in self.axes])
        self._set(0, 0, 0, self.home_q)
        self.total = m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot")]
        self.rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.support = self._support()
        self.t_home = self._tool_bearing()
        self.span = self._span()
        self.memo = []
        self._grid()

    # ------------------------------------------------------------ state --
    def _set(self, x, y, yaw, q, doors=None, tilt=None):
        d = self.d
        a = self.qb
        d.qpos[a:a + 3] = [x, y, 0.0]
        d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        if tilt:                                         # the base nods on its sprung legs: the IMU knows by how much
            t = np.zeros(4)
            qp, qr = np.array([math.cos(tilt[0] / 2), 0, math.sin(tilt[0] / 2), 0]), np.array([math.cos(tilt[1] / 2), math.sin(tilt[1] / 2), 0, 0])
            mujoco.mju_mulQuat(t, d.qpos[a + 3:a + 7].copy(), qp)
            o = np.zeros(4)
            mujoco.mju_mulQuat(o, t, qr)
            d.qpos[a + 3:a + 7] = o
        for i, ax in enumerate(self.axes):
            v = q[i] / len(ax["q"])
            for qa in ax["q"]:
                d.qpos[qa] = v
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        if getattr(self, "held", None):
            name, offset, rel_rotation = self.held
            P = d.site_xpos[self.ee]
            R = d.site_xmat[self.ee].reshape(3, 3)
            adr = self.obj_q[name]
            d.qpos[adr:adr+3] = P+R@offset
            mujoco.mju_mat2Quat(d.qpos[adr+3:adr+7], (R@rel_rotation).ravel())
            mujoco.mj_kinematics(self.m, d)

    def _tool_bearing(self):
        R = self.d.site_xmat[self.ee].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def _span(self):
        rng = np.random.default_rng(0)
        best = 0.0
        for _ in range(200):
            self._set(0, 0, 0, self.lo + (self.hi - self.lo) * rng.uniform(0, 1, self.n))
            best = max(best, float(np.hypot(*self.d.site_xpos[self.ee][:2])))
        return best + 0.05

    def climbs(self):
        """Over the home's threshold from rest? Two conditions, both checked against the dynamics in tests/test_physics.py.
        1. A wheel of radius r takes W tan(a) at the edge of a step h, with cos(a) = 1 - h/r; the drive wheels push with mu N.
        2. A rigid chassis on 3+ wheels in a line pivots on its rear caster and lifts the drive wheels off the floor,
           unless the legs give: travel >= h |x_rear| / (x_front + |x_rear|). Power wheelchairs are sprung for this reason."""
        h = self.b.home.threshold
        if h <= 0:
            return True
        b = self.r.base
        rc = b.caster_r
        if h >= rc:
            return False
        W = self.total * 9.81
        front = [c for c in b.casters if c[0] > b.wheel_x] or [(0, 0)]
        rear = [c for c in b.casters if c[0] < b.wheel_x]
        load = W * (1 - b.drive_share) / len(b.casters)
        a = math.acos(1 - h / rc)
        need = len(front) * load * math.tan(a)
        if rear and front:
            xf, xr = max(c[0] for c in front) - b.wheel_x, b.wheel_x - min(c[0] for c in rear)
            if b.travel < h * xr / (xf + xr):
                return False
        return need <= FLOOR_MU[self.b.home.floor] * b.drive_share * W * 0.85

    def _support(self):
        b = self.r.base
        pts = [(b.wheel_x, b.wheel_y), (b.wheel_x, -b.wheel_y)] + list(b.casters)
        c = np.mean(pts, axis=0)
        return sorted(pts, key=lambda p: math.atan2(p[1] - c[1], p[0] - c[0]))

    # ------------------------------------------------------------- floor --
    def _grid(self):
        """Occupancy from the world's fixed geoms, at two height bands, and the (x, y, heading) reachable set."""
        m = self.m
        X, Y = int(10.4 / CELL), int(8.6 / CELL)
        low, high = np.zeros((X, Y), bool), np.zeros((X, Y), bool)
        self._set(0, 0, 0, self.home_q)
        top = float(max(self.d.geom_xpos[g][2] + (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 0) for g in range(m.ngeom) if self.g_rob[g]))
        base_top = max(self.r.base.z[1], (self.r.deck.z + self.r.deck.fence) if self.r.deck and self.r.deck.parent == "robot" else 0)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0 or m.geom_contype[g] == 0 or self.names[g] == "floor" or self.names[g].startswith("landing"):
                continue
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            c, s = m.geom_pos[g], m.geom_size[g]
            if abs(m.geom_quat[g, 0]) < 0.999:
                continue
            z0, z1 = c[2] - s[2], c[2] + s[2]
            if z0 > top + 0.03:                             # lintels are above it
                continue
            if self.names[g].startswith("thr_") and (self.climbs() or "ramps" in self.r.kits):
                continue
            if z1 < 0.021 and not self.names[g].startswith("thr_"):       # a rug: the wheels roll onto it
                continue
            i0, i1 = int((c[0] - s[0] + 0.2) / CELL), int(math.ceil((c[0] + s[0] + 0.2) / CELL))
            j0, j1 = int((c[1] - s[1]) / CELL), int(math.ceil((c[1] + s[1]) / CELL))
            high[max(i0, 0):i1, max(j0, 0):j1] = True
            if z0 < base_top:
                low[max(i0, 0):i1, max(j0, 0):j1] = True
        self.low, self.high = low, high
        x0, x1, y0, y1 = self.r.footprint()
        # the lattice rotates in 45 deg steps on a 3 cm grid: a pivot between two free headings needs one cell to spare
        k = CLEAR + CELL
        self.fp = (x0 - k, x1 + k, y0 - k, y1 + k)
        sb = self._stowed_box()
        self.fp_high = (sb[0] - CELL, sb[1] + CELL, sb[2] - CELL, sb[3] + CELL)
        free = np.zeros((HEADINGS, X, Y), bool)
        for h in range(HEADINGS):
            a = h * 2 * math.pi / HEADINGS
            free[h] = ~ndimage.binary_dilation(low, self._foot(a, self.fp)) & ~ndimage.binary_dilation(high, self._foot(a, self.fp_high))
        from .navigation import maps
        free, self.turn_free = maps(self)
        self.free = free
        # flood from the dock
        dock = self.dock_pose()
        reach = np.zeros_like(free)
        i, j = int((dock[0] + 0.2) / CELL), int(dock[1] / CELL)
        h0 = int(round(dock[2] / (2 * math.pi / HEADINGS))) % HEADINGS
        if not free[h0, i, j]:                              # the dock pose itself may touch by a cell: start nearby
            ok = np.argwhere(free[h0, max(i - 6, 0):i + 7, max(j - 6, 0):j + 7])
            if len(ok):
                i, j = i - 6 + ok[0][0], j - 6 + ok[0][1]
        reach[h0, i, j] = True
        step = [(int(round(2.4 * math.cos(h * 2 * math.pi / HEADINGS))), int(round(2.4 * math.sin(h * 2 * math.pi / HEADINGS)))) for h in range(HEADINGS)]
        for _ in range(400):
            new = reach.copy()
            for h in range(HEADINGS):
                sx, sy = step[h]
                new[h] |= np.roll(np.roll(reach[h], sx, 0), sy, 1) | np.roll(np.roll(reach[h], -sx, 0), -sy, 1)
                new[h] |= (reach[(h + 1) % HEADINGS] & self.turn_free[h]) | (reach[(h - 1) % HEADINGS] & self.turn_free[(h-1)%HEADINGS])
            new &= free
            if (new == reach).all():
                break
            reach = new
        # the flood steps 7 cm at a time; cells between its steps are as reachable as their neighbours
        for h in range(HEADINGS):
            reach[h] = ndimage.binary_dilation(reach[h], np.ones((5, 5), bool)) & free[h]
        self.access = reach
        # the last metre is driven with the arm raised to its working height: a hand may pass over a worktop that the
        # stowed arm would hit, so only the base has to fit. Roll on from anywhere reachable, along the heading
        base = np.zeros_like(free)
        for h in range(HEADINGS):
            base[h] = ~ndimage.binary_dilation(low, self._foot(h * 2 * math.pi / HEADINGS, self.fp))
        near = reach.copy()
        for _ in range(12):
            new = near.copy()
            for h in range(HEADINGS):
                sx, sy = step[h]
                new[h] |= (np.roll(np.roll(near[h], sx, 0), sy, 1) | np.roll(np.roll(near[h], -sx, 0), -sy, 1)) & base[h]
            near = new
        for h in range(HEADINGS):
            near[h] = (ndimage.binary_dilation(near[h], np.ones((5, 5), bool)) & base[h]) | near[h]
        self.work = near

    def _stowed_box(self):
        """Bounding rectangle of the stowed robot: an arm that sticks out sweeps a wider circle than its base."""
        self._set(0, 0, 0, self.home_q)
        d, m = self.d, self.m
        pts = []
        for g in range(m.ngeom):
            if self.g_rob[g] and not self.g_wheel[g]:
                c = d.geom_xpos[g]
                R = d.geom_xmat[g].reshape(3, 3)
                s = m.geom_size[g]
                if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                    s = np.array([s[0], s[0], s[0]])
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        pts.append(c + R @ (np.array([sx, sy, 0]) * s))
        pts = np.array(pts)
        return float(pts[:, 0].min()) - CLEAR, float(pts[:, 0].max()) + CLEAR, float(pts[:, 1].min()) - CLEAR, float(pts[:, 1].max()) + CLEAR

    def _foot(self, a, fp=None):
        x0, x1, y0, y1 = fp or self.fp
        r = int(math.ceil(max(abs(x0), abs(x1), abs(y0), abs(y1)) * 1.5 / CELL))
        gi, gj = np.mgrid[-r:r + 1, -r:r + 1] * CELL
        c, s = math.cos(a), math.sin(a)
        u, v = c * gi + s * gj, -s * gi + c * gj
        mask = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        return mask[::-1, ::-1]                             # dilation adds the structure: robot at p hits a if a - p is in it

    def dock_pose(self):
        """A robot backs onto the charging plate: how far out it stands depends on its own length."""
        p = self.b.world.ports["dock"]
        return p[0], 4.40 - self.fp_high[0], p[2]

    def reachable(self, x, y, yaw):
        i, j = int((x + 0.2) / CELL), int(y / CELL)
        if not (0 <= i < self.access.shape[1] and 0 <= j < self.access.shape[2]):
            return False
        h = int(round(yaw / (2 * math.pi / HEADINGS))) % HEADINGS
        for hh in (h, (h + 1) % HEADINGS, (h - 1) % HEADINGS):
            if self.work[hh, max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].any():
                return True
        return False

    def _near(self, p, rng, n):
        """Base positions the robot can really drive to, within arm's length of p."""
        any_h = self.work.any(0)
        i0, j0 = int((p[0] + 0.2) / CELL), int(p[1] / CELL)
        r = int(self.span / CELL) + 2
        lo_i, lo_j = max(i0 - r, 0), max(j0 - r, 0)
        ij = np.argwhere(any_h[lo_i:i0 + r + 1, lo_j:j0 + r + 1])
        if not len(ij):
            return []
        xy = (ij + (lo_i, lo_j)) * CELL + (-0.2 + CELL / 2, CELL / 2)
        dist = np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1])
        keep = xy[dist <= self.span]
        if not len(keep):
            return []
        idx = rng.choice(len(keep), size=min(n, len(keep)), replace=False)
        return [(float(keep[k, 0]), float(keep[k, 1]), 0.0) for k in idx]

    def covered(self):
        """Floor area the base can reach, m2 (doors decide this)."""
        return float(self.access.any(0).sum() * CELL * CELL)

    # ---------------------------------------------------------------- IK --
    def _ik(self, u0, p, a, j, tip=0.0, tol=0.0035, fix_base=False):
        """u = (x, y, yaw, axes...). Target: tool point p, tool +X along a, tool Y along +-j (if given)."""
        m, d = self.m, self.d
        u = u0.copy()
        lam = 1e-3
        n = 3 + self.n
        lo = np.concatenate([u0[:3] if fix_base else [-1e9, -1e9, -1e9], self.lo])
        hi = np.concatenate([u0[:3] if fix_base else [1e9, 1e9, 1e9], self.hi])
        W = np.concatenate([[1e3, 1e3, 1e3] if fix_base else [1.5, 1.5, 0.5], np.ones(self.n)])
        tol_a = getattr(self, "_tol", 20.0)
        good = None                     # first acceptable pose; keep going while the wrist can still square up
        last = 1e9
        for it in range(60):
            self._set(u[0], u[1], u[2], u[3:])
            P = d.site_xpos[self.ee]
            R = d.site_xmat[self.ee].reshape(3, 3)
            xt, yt = R[:, 0], R[:, 1]
            if tip:
                q = P + xt * tip
            else:
                q = P
            ep = p - q
            er = np.cross(xt, a)
            ejv = 0.0
            if j is not None:
                ja = yt if yt @ j > 0 else -yt
                ej = np.cross(ja, j)
            else:
                ej = np.zeros(3)
            err = float(np.linalg.norm(ep))
            ang = math.degrees(math.asin(min(1.0, float(np.linalg.norm(er)))))
            if xt @ a < 0:
                ang = 180 - ang
            if j is not None:
                ejv = math.degrees(math.asin(min(1.0, float(np.linalg.norm(ej)))))
            if err < tol and ang < tol_a * 0.8 and ejv < tol_a:
                good = (u.copy(), err, ang, ejv)
                if (ang < 2.0 and ejv < 3.0) or ang + ejv > last - 0.05:
                    return good
                last = ang + ejv
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.ee)
            Jp = np.zeros((3, n))
            Jr = np.zeros((3, n))
            base = d.xpos[self.rid].copy()
            r = q - base
            Jp[:, 0] = (1, 0, 0)
            Jp[:, 1] = (0, 1, 0)
            Jp[:, 2] = (-r[1], r[0], 0)
            Jr[2, 2] = 1.0
            for i, ax in enumerate(self.axes):
                k = len(ax["v"])
                Jp[:, 3 + i] = sum(self.jp[:, v] for v in ax["v"]) / k
                Jr[:, 3 + i] = sum(self.jr[:, v] for v in ax["v"]) / k
            if tip:
                # tool point ahead of the site: add w x (xt tip)
                Jp = Jp + np.stack([np.cross(Jr[:, c], xt * tip) for c in range(n)], axis=1)
            # orientation: the rotation that aligns the tool. Without a jaw axis, spin about the fingers is free
            if j is None:
                Pr = np.eye(3) - np.outer(xt, xt)
                Aor, eor = Pr @ Jr, er
            else:
                Aor, eor = Jr, er + xt * (xt @ ej)
            A = np.vstack([Jp, 0.4 * Aor]) / W[None, :]
            e = np.concatenate([ep, 0.4 * eor])
            du = np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ e)
            du = du / W
            s = min(1.0, 0.25 / max(np.abs(du[:2]).max(), 1e-9), 0.5 / max(abs(du[2]), 1e-9))
            u = np.clip(u + s * du, lo, hi)
        return good if good else (None, err, ang, ejv)

    # ----------------------------------------------------------- collide --
    def _clear(self, c: Contact, doors, held_ok):
        d, m = self.d, self.m
        mujoco.mj_collision(m, d)
        n = d.ncon
        if not n:
            return None
        g = d.contact.geom[:n]
        a, b = g[:, 0], g[:, 1]
        dist = d.contact.dist[:n]
        whl = self.g_wheel[a] | self.g_wheel[b]
        for k in np.nonzero((self.g_rob[a] | self.g_rob[b]) & ~whl)[0]:
            ga, gb = int(a[k]), int(b[k])
            if self.g_rob[gb]:
                ga, gb = gb, ga
            na, nb = self.names[ga], self.names[gb]
            if self.g_rob[gb]:                              # the arm against its own base: real penetration only
                if dist[k] < -0.002:
                    return f"self: {na} / {nb}"
                continue
            if self.g_obj[gb] and held_ok and self.g_pad[ga]:
                continue
            if self.g_pad[ga] and dist[k] > -0.001:         # fingertips may graze the surface they pick from
                continue
            if na == "arm_palm" and self.g_obj[gb] and held_ok and dist[k] > -0.001:
                continue
            if nb in self._allow or (self.g_pad[ga] and self._allow_pref and nb.startswith(self._allow_pref)):
                continue
            if nb.startswith(("rug", "thr_", "doormat")):
                continue
            return f"{na} / {nb}"
        if getattr(self, "held", None):
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.held[0])
            held = m.geom_bodyid == body
            for k in np.nonzero((held[a] | held[b]) & ~(self.g_rob[a] | self.g_rob[b]))[0]:
                if dist[k] < -.001:
                    return f"payload: {self.names[a[k]]} / {self.names[b[k]]}"
        return None

    # ------------------------------------------------------------ statics --
    def _statics(self, u, c: Contact, mu_floor):
        """Holding this pose with the payload and the contact force: motors, tipping, traction."""
        m, d = self.m, self.d
        self._set(u[0], u[1], u[2], u[3:])
        mujoco.mj_comPos(m, d)
        # joint loads from gravity on the robot's own parts
        d.qvel[:] = 0
        d.qacc[:] = 0
        res = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, res)
        P = d.site_xpos[self.ee].copy()
        F = np.array(c.force, float) + np.array([0, 0, -9.81 * c.mass])
        mujoco.mj_jacSite(m, d, self.jp, self.jr, self.ee)
        tau = -(res - 0) + self.jp.T @ F            # generalized force the world applies; the motor balances it
        worst = (0.0, "")
        for ax in self.axes:
            a = ax["a"]
            f = abs(sum(tau[v] for v in ax["v"])) / len(ax["v"])
            f = max(0.0, f - a.friction) if a.selflock else f
            u_ = f / a.force
            if u_ > worst[0]:
                worst = (u_, a.name)
        # centre of pressure in the base frame
        mt = self.total
        cw = d.subtree_com[self.rid].copy()
        Rb = rz(u[2])
        pb = np.array([u[0], u[1], 0.0])
        c_b = Rb.T @ (cw - pb)
        p_b = Rb.T @ (P - pb)
        Fe = Rb.T @ F                                # includes the payload as a force at the hand
        Mg = mt * 9.81
        Rz = Mg - Fe[2]
        if Rz <= 1:
            return worst, -1.0, 0.0
        rx = (Mg * c_b[0] + p_b[2] * Fe[0] - p_b[0] * Fe[2]) / Rz
        ry = (Mg * c_b[1] - p_b[1] * Fe[2] + p_b[2] * Fe[1]) / Rz
        marg = _margin(np.array([rx, ry]), self.support)
        h = float(np.linalg.norm(Fe[:2]))
        trac = h / (mu_floor * self.r.base.drive_share * Rz)
        return worst, marg, trac

    def solve(self, c: Contact, seeds=24, rng=None, current=None):
        """Best pose for this interaction, or the furthest stage any attempt got to."""
        rng = rng or np.random.default_rng(0)
        if c.kind == "tray":
            return self._tray(c)
        order = ["stroke", "reach", "access", "collision", "motor", "tip", "traction", "grip"]
        best = dict(ok=False, why="stroke", detail="")
        self._tol = c.tol_deg
        g = self.r.grip
        mu_f = FLOOR_MU[self.b.home.floor]
        self._allow, self._allow_pref = set(), ()
        if c.kind in ("press", "pull"):
            self._allow_pref = tuple(c.obj.split(",")) if c.obj else ()
        for i, v in self._door0.items():
            self.d.qpos[v] = c.doors.get(i, 0.0)
        for inst, qa in self.obj_q.items():
            self.d.qpos[qa:qa + 3] = (0, 0, -3)
        if c.kind in ("grasp", "place") and c.obj_pose is not None and c.obj in self.obj_q:
            qa = self.obj_q[c.obj]
            p, yaw = c.obj_pose
            self.d.qpos[qa:qa + 3] = p
            self.d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for cand in c.cands:
            p, a, j, width = cand[:4]
            depth = cand[4] if len(cand) > 4 else 0.0
            p, a = np.asarray(p, float), np.asarray(a, float)
            j = None if j is None else np.asarray(j, float)
            if width is not None and width + 0.003 > g.stroke:
                continue
            best = _max(best, dict(ok=False, why="reach", detail=f"{c.kind}"), order)
            tip = self.jaw if c.kind == "press" else 0.0
            # a thing deeper than the jaws is held by its near side: the palm stops 5 mm short of it
            pp = p - a * max(0.0, depth / 2 + 0.005 - g.jaw_len / 2)
            # the hand may not go through what the object rests on: a horizontal hand rides up by half its jaw
            # height, a downward hand by half its jaw length (fingertips 2 mm over the surface)
            if c.kind in ("grasp", "place") and c.obj_pose is not None and not c.exact_tool:
                z0 = c.obj_pose[0][2]
                if abs(a[2]) < 0.5:
                    pp[2] = max(pp[2], z0 + max(g.jaw_h, g.palm[2]) / 2 + 0.005)
                else:
                    pp[2] = max(pp[2], z0 + abs(a[2]) * g.jaw_len / 2 + 0.005)
            ah = a[:2] / (np.linalg.norm(a[:2]) + 1e-9) if np.linalg.norm(a[:2]) > 0.2 else None
            # A local arm solution avoids gratuitous base pivots beside furniture.
            # An endpoint is accepted only after the same collisions/statics checks.
            if current is not None:
                candidates = [np.array(current, float)]
                for _ in range(4):
                    u0 = candidates[0].copy()
                    u0[3:] = np.clip(u0[3:]+rng.normal(0, 0.08, self.n)*(self.hi-self.lo), self.lo, self.hi)
                    candidates.append(u0)
                for u0 in candidates:
                    local = self._ik(u0, pp, a, j, tip=tip, fix_base=True)
                    if local[0] is None:
                        continue
                    u = local[0]
                    self._set(*u[:3], u[3:])
                    if self._clear(c, c.doors, c.kind in ("grasp", "place")):
                        continue
                    load, margin, traction = self._statics(u, c, mu_f)
                    obj = objects.CATALOG.get(c.obj) if c.obj else None
                    mu = assemble.PAD_MU[g.pad][obj.material] if obj else 0.6
                    if load[0] > 1 or margin < .01 or traction > .8 or 2*mu*g.force < 1.5*c.mass*9.81:
                        continue
                    return dict(ok=True, why="", detail="local arm solution", base=u[:3].tolist(), q=u[3:].tolist(),
                                load=load[0], margin=margin, trac=traction, local=True)
            spots = self._near(pp, rng, seeds)
            warm = [(np.r_[w[0] + pp - w[1], w[2]]) for w in self.memo[-6:]]
            for s in range(len(spots) + len(warm)):
                if s < len(warm):
                    u0 = warm[s][[0, 1, 3]].tolist() + warm[s][4:].tolist()
                    u0 = np.array(u0)
                else:
                    bx, by, yaw = spots[s - len(warm)]
                    if s % 3 == 2:                          # a curled arm's stowed heading says nothing: try any heading too
                        yaw = rng.uniform(-math.pi, math.pi)
                    elif ah is not None:
                        yaw = math.atan2(ah[1], ah[0]) - self.t_home + rng.uniform(-0.4, 0.4)
                    else:
                        yaw = math.atan2(pp[1] - by, pp[0] - bx) - self.t_home + rng.uniform(-0.6, 0.6)
                    q0 = self.lo + (self.hi - self.lo) * rng.uniform(0.2, 0.8, self.n)
                    for k, ax in enumerate(self.axes):
                        if ax["a"].name in ("lift",):
                            q0[k] = np.clip(pp[2] + 0.05, self.lo[k], self.hi[k])
                    u0 = np.concatenate([[bx, by, yaw], q0])
                sol = self._ik(u0, pp, a, j, tip=tip)
                if sol[0] is None:
                    continue
                u = sol[0]
                best = _max(best, dict(ok=False, why="access", detail=f"{u[:3].round(2).tolist()}"), order)
                if not self.reachable(u[0], u[1], u[2]):
                    # the IK drifted into furniture: hold the base where the seed stood and let the arm alone try
                    held = self._ik(np.concatenate([u0[:3], sol[0][3:]]), pp, a, j, tip=tip, fix_base=True) if s >= len(warm) else (None,)
                    if held[0] is None or not self.reachable(held[0][0], held[0][1], held[0][2]):
                        continue
                    u = held[0]
                best = _max(best, dict(ok=False, why="collision", detail=""), order)
                self._set(u[0], u[1], u[2], u[3:])
                hit = self._clear(c, c.doors, c.kind in ("grasp", "place"))
                if hit:
                    best["detail"] = hit
                    continue
                wst, marg, trac = self._statics(u, c, mu_f)
                res = dict(ok=False, why="motor", detail=f"{wst[1]} {wst[0]:.2f}", base=u[:3].round(3).tolist(), q=u[3:].round(4).tolist(), margin=round(marg, 3), load=round(wst[0], 2), trac=round(trac, 2))
                if wst[0] > 1.0:
                    best = _max(best, res, order)
                    continue
                res["why"], res["detail"] = "tip", f"margin {marg * 100:.1f} cm"
                if marg < 0.01:
                    best = _max(best, res, order)
                    continue
                res["why"], res["detail"] = "traction", f"{trac:.2f} of the limit"
                if trac > 0.8:
                    best = _max(best, res, order)
                    continue
                if c.mass:
                    obj = objects.CATALOG.get(c.obj.split("#")[0]) if c.obj else None
                    mu = assemble.PAD_MU[g.pad][obj.material] if obj else 0.6
                    res["why"], res["detail"] = "grip", f"{2 * mu * g.force / (c.mass * 9.81):.1f} x weight"
                    if 2 * mu * g.force < 1.5 * c.mass * 9.81:
                        best = _max(best, res, order)
                        continue
                res.update(ok=True, why="", detail="")
                self.memo.append((np.array([u[0], u[1], 0.0]), pp.copy(), u[2:].copy()))
                return res
        return best


    def _tray(self, c: Contact):
        """A person sets the load on the robot's deck or tray: it has to fit, and the robot must stay upright."""
        dk = self.r.deck
        if dk is None:
            return dict(ok=False, why="stroke", detail="no deck or tray")
        sz = objects.CATALOG[c.obj].size
        w, l = dk.x[1] - dk.x[0], dk.y[1] - dk.y[0]
        if min(sz[0], sz[1]) > min(w, l) + 0.05 or max(sz[0], sz[1]) > max(w, l) + 0.08:
            return dict(ok=False, why="stroke", detail=f"tray {w:.2f} x {l:.2f}")
        return dict(ok=True, why="", detail="")


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _max(a, b, order):
    return b if order.index(b["why"]) >= order.index(a["why"]) else a


def _margin(p, poly):
    best = 1e9
    c = np.mean(poly, axis=0)
    for i in range(len(poly)):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        e = b - a
        nrm = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        if (c - a) @ nrm < 0:
            nrm = -nrm
        best = min(best, float((p - a) @ nrm))
    return best
