"""Tier 2 runtime: one trial of one robot in one home, with real contacts.

The boundary that matters: World holds the truth. A controller gets only Obs (encoders, motor current, lidar, camera
detections), and only what a real sensor could deliver: ranges through ray casts, detections that need a line of sight
and a field of view. It sends only motor commands. judge.py reads the truth and scores.

What runs each 20 ms control tick:
  motors      torque-speed line, peak vs continuous, heating; PID in the firmware at 500 Hz, in MuJoCo's actuator
  appliances  magnetic fridge seal, door springs, microwave and dishwasher latches, buttons
  sensors     encoders (quantized, one tick late), current (10 % noise), IMU, lidar at 10 Hz, cameras at 10 Hz, 100 ms late
  judge       contact forces, tilt, speed limits, drops, spills

Domain randomization draws every uncertain number per trial (DR). Ranges are estimates until a bench says otherwise.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import assemble, objects, appliance_judge
from .homes import Home
from .robot import Robot

CTRL_DT = 0.02
RAY_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)     # group 0 only: the lidar does not see the robot's own parts

# name: (low, high). 1.0 = nominal for the scale factors
DR = dict(mu_pad=(0.75, 1.25), mu_obj=(0.8, 1.25), mu_floor=(0.8, 1.2), obj_mass=(0.85, 1.15), seal=(0.8, 1.25),
          door_damp=(0.7, 1.5), motor_friction=(0.8, 1.5), motor_force=(0.85, 1.05), wheel_r=(0.995, 1.005), wheel_track=(0.99, 1.01),
          cam_mount_deg=(0.0, 1.0), lidar_bias=(0.0, 0.02), latency_ticks=(1, 4))


@dataclass
class Detection:
    name: str
    t: float                  # capture time
    pos: np.ndarray           # in the camera frame
    yaw: float                # object's yaw about its own vertical, in the camera frame (noisy)
    cam: str
    vis: float                # fraction of the object seen


@dataclass
class Obs:
    t: float = 0.0
    q: np.ndarray = None      # axis positions, encoder
    qd: np.ndarray = None
    tau: np.ndarray = None    # motor force estimate from current
    grip: float = 0.0         # opening between the pads
    grip_f: float = 0.0       # squeeze estimate
    wheels: np.ndarray = None # wheel angles, rad
    tilt: float = 0.0
    pitch: float = 0.0        # base attitude from the IMU, rad
    roll: float = 0.0
    scan: np.ndarray = None   # new ranges, or None
    scan_t: float = 0.0
    dets: list = field(default_factory=list)


class World:
    def __init__(self, built: assemble.Built, seed=0, dr=True, noise=1.0, dt_scale=1):
        self.b, self.r, self.home = built, built.robot, built.home
        self.m = built.m
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.noise = noise
        m = self.m
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        A = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        S = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        self.jq = lambda n: m.jnt_qposadr[J(n)]
        self.jv = lambda n: m.jnt_dofadr[J(n)]
        self.bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.rid = self.bid("robot")
        self.ax = []
        for a in self.r.chain:
            nm = [a.name] + [f"{a.name}{k}" for k in range(2, a.stages + 1)]
            act = A(f"a_{a.name}")
            self.ax.append(dict(a=a, q=[self.jq(n) for n in nm], v=[self.jv(n) for n in nm], act=act, c=m.actuator_ctrladr[act]))
        self.na = len(self.ax)
        self.a_grip, self.c_grip = A("a_grip"), m.actuator_ctrladr[A("a_grip")]
        self.c_wheel = [m.actuator_ctrladr[A(f"a_wheel_{s}")] for s in "lr"]
        self.q_wheel = [self.jq(f"wheel_{s}") for s in "lr"]
        self.v_wheel = [self.jv(f"wheel_{s}") for s in "lr"]
        self.q_jaw = [self.jq("jaw_a"), self.jq("jaw_b")]
        self.ee = S("ee")
        self.q_base, self.v_base = self.jq("base"), self.jv("base")
        self.lidar = S("lidar")
        self.cams = {c.name: (c, S(f"cam_{c.name}")) for c in self.r.cams}
        n = assemble.geom_owner(m)
        self.gname = n
        arr = np.array(n, dtype=object)
        self.g_rob = np.array([x.startswith(("rob_", "arm_", "pad_")) for x in arr])
        self.g_wheel = np.array([x.startswith(("rob_tyre", "rob_cast")) for x in arr])
        self.g_person = np.array([x.startswith("per_") for x in arr])
        self.g_obj = np.array([x.startswith("obj_") for x in arr])
        self.g_pad = np.array([x.startswith(("pad_", "arm_jaw")) for x in arr])
        self.g_group = m.geom_group.copy()
        for i in np.nonzero(self.g_rob)[0]:
            m.geom_group[i] = 3
        self.obj = {i: dict(cat=c, o=objects.CATALOG[c], q=self.jq(i), v=self.jv(i), b=self.bid(i), pts=objects.surface_points(objects.CATALOG[c])) for i, c in built.names}
        self.landmarks = {}
        for name in ("mw_door", "fr_door"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name+"_slab")
            center, size = m.geom_pos[g], m.geom_size[g]
            points = [center.copy()]+[center+size*np.array([x, 0, z])*.8 for x in (-1, 1) for z in (-1, 1)]
            self.landmarks[name] = dict(b=self.bid(name), pts=points)
        # door joints and latches
        self.dq = {k: self.jq(k) for k in ("fr_door", "mw_door", "dw_door")}
        self.dv = {k: self.jv(k) for k in ("fr_door", "mw_door", "dw_door")}
        E = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n)
        self.eq = {k: E(f"{k}_latch") for k in ("fr_door", "mw_door", "dw_door")}
        self.btn = {k: self.jq(f"btn_{k}") for k in ("open", "start")}
        self.hand_b = self.bid("hand")
        self.t = 0.0
        self.tick = 0
        self.kits = set(self.r.kits)
        self.p = self._draw(dr)
        self._apply_dr()
        self.state = dict(fr_open_cmd=0.0, mw_running=False, mw_heated=0.0, mw_started=[], btn_prev=dict(open=False, start=False), fridge_seal=True, dw_seal=True)
        self.q_cmd = np.array([a["a"].home for a in self.ax])
        self.qd_cmd = np.zeros(self.na)
        self.grip_cmd = 0.04
        self.grip_force_cmd = None
        self.wheel_tgt = np.zeros(2)
        self.heat = np.zeros(self.na + 1)
        self.energy = 0.0
        self.q_hist = deque(maxlen=8)
        self.det_queue = deque()
        self.mon = dict(max_person=0.0, max_bump=0.0, max_tilt=0.0, bumps=0, overspeed=0.0, spill=False, spill_margin=1e9, tip=False, max_load={}, drops=[], peak_heat=0.0, stalled=False, diverged=False, contact_peaks={})
        self.record = None

    # --------------------------------------------------------- randomize --
    def _draw(self, dr):
        p = {}
        for k, (lo, hi) in DR.items():
            nominal = 1.0 if lo <= 1.0 <= hi else lo
            p[k] = float(self.rng.uniform(lo, hi)) if dr else nominal
        p["latency_ticks"] = int(round(p["latency_ticks"])) if dr else 2
        return p

    def _apply_dr(self):
        m, p = self.m, self.p
        m.pair_friction[:, :2] *= p["mu_pad"]
        for i, o in self.obj.items():
            m.body_mass[o["b"]] *= p["obj_mass"]
            m.body_inertia[o["b"]] *= p["obj_mass"]
        for g in np.nonzero(self.g_obj)[0]:
            m.geom_friction[g, 0] *= p["mu_obj"]
        for s in "lr":
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"rob_tyre_{s}")
            m.geom_friction[g, 0] *= p["mu_floor"]
        for a in self.ax:
            m.actuator_forcerange[a["act"]] *= p["motor_force"]
            for v in a["v"]:
                m.dof_frictionloss[v] *= p["motor_friction"]
        for k in ("fr_door", "mw_door", "dw_door"):
            m.dof_damping[self.dv[k]] *= p["door_damp"]
        self.frc0 = np.array([m.actuator_forcerange[a["act"], 1] for a in self.ax])
        self.seal = self.home.seal_N * p["seal"]
        mujoco.mj_setConst(m, self.d)                  # every edit above changes constants the solver depends on
        # the robot's own calibration is off by a little: mount error of each camera, a fixed lidar bias
        self.cam_err = {}
        for c in self.cams:
            a = self.rng.normal(0, 1, 3)
            self.cam_err[c] = a / np.linalg.norm(a) * math.radians(p["cam_mount_deg"]) * self.noise
        self.lidar_bias = self.rng.uniform(-1, 1) * p["lidar_bias"] * self.noise

    # ------------------------------------------------------------- set-up --
    def place_robot(self, x, y, yaw, q=None):
        d, m = self.d, self.m
        a = self.q_base
        d.qpos[a:a + 3] = [x, y, 0.0015]
        d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, ax in enumerate(self.ax):
            v = ax["a"].home if q is None else float(q[i])
            for qa in ax["q"]:
                d.qpos[qa] = v / len(ax["q"])
        d.qpos[self.q_jaw[0]] = d.qpos[self.q_jaw[1]] = 0.02

    def place(self, name, pos, yaw=0.0):
        o = self.obj[name]
        d = self.d
        d.qpos[o["q"]:o["q"] + 3] = pos
        d.qpos[o["q"] + 3:o["q"] + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[o["v"]:o["v"] + 6] = 0

    def start(self, settle=1.0, doors=None):
        """Latch the doors, set the controllers to where the joints are, let everything come to rest."""
        d, m = self.d, self.m
        for k in self.eq:
            d.eq_active[self.eq[k]] = 1
        for k, ang in (doors or {}).items():
            d.qpos[self.dq[k]] = ang
            d.eq_active[self.eq[k]] = 0
            if k == "fr_door":
                self.state["fr_open_cmd"] = abs(ang)
        mujoco.mj_forward(m, d)
        self.q_cmd = self.true_q()
        self.q_last = self.q_cmd.copy()
        self.c_axes = np.array([a["c"] for a in self.ax])
        self.grip_cmd = self.true_grip()
        self.wheel_tgt = np.array([d.qpos[q] for q in self.q_wheel])
        for _ in range(int(settle / CTRL_DT)):
            self._apply(np.zeros(2), np.zeros(self.na))
            self._appliances()
            mujoco.mj_step(m, d, nstep=int(round(CTRL_DT / m.opt.timestep)))
        self.t = 0.0
        self.energy = 0.0

    # -------------------------------------------------------------- truth --
    def true_q(self):
        return np.array([sum(self.d.qpos[q] for q in a["q"]) for a in self.ax])

    def true_qd(self):
        return np.array([sum(self.d.qvel[v] for v in a["v"]) for a in self.ax])

    def true_grip(self):
        return float(self.d.qpos[self.q_jaw[0]] + self.d.qpos[self.q_jaw[1]])

    def base_pose(self):
        a = self.q_base
        w, x, y, z = self.d.qpos[a + 3:a + 7]
        return float(self.d.qpos[a]), float(self.d.qpos[a + 1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def obj_pos(self, name):
        return self.d.xpos[self.obj[name]["b"]].copy()

    def obj_mat(self, name):
        return self.d.xmat[self.obj[name]["b"]].reshape(3, 3).copy()

    # ------------------------------------------------------------- motors --
    def _apply(self, wheel_v, qd_ff):
        """Commands to the firmware loops, force limits from the torque-speed line and heat."""
        d, m = self.d, self.m
        v = self.true_qd()
        for i, a in enumerate(self.ax):
            s = a["a"]
            f0 = self.frc0[i]
            lim = f0 * max(0.15, 1.0 - abs(v[i]) / (s.speed * 1.2))
            if self.heat[i] > 1.0:
                lim = min(lim, s.force)                  # hot: the driver falls back to its continuous rating
            m.actuator_forcerange[a["act"]] = [-lim, lim]
            d.ctrl[a["c"]] = self.q_cmd[i]
            d.ctrl[a["c"] + 1] = self.qd_cmd[i]
        d.ctrl[self.c_grip], d.ctrl[self.c_grip + 1] = self.grip_cmd, 0.0
        r = self.r.base.wheel_r * self.p["wheel_r"]
        for k in range(2):
            self.wheel_tgt[k] += wheel_v[k] * CTRL_DT
            act = d.qpos[self.q_wheel[k]]
            self.wheel_tgt[k] = act + np.clip(self.wheel_tgt[k] - act, -0.3, 0.3)
            d.ctrl[self.c_wheel[k]], d.ctrl[self.c_wheel[k] + 1] = self.wheel_tgt[k], wheel_v[k]

    def command(self, q=None, qd=None, v=0.0, w=0.0, grip=None, squeeze=False, kit=None):
        """What a controller may do: joint targets, base speed, gripper, a kit command."""
        if q is not None:
            self.q_cmd = np.clip(q, [a["a"].lo for a in self.ax], [a["a"].hi for a in self.ax])
        self.qd_cmd = np.zeros(self.na) if qd is None else qd
        if grip is not None:
            step = self.r.grip.speed * CTRL_DT
            tgt = grip - 0.03 if squeeze else grip
            # A blocked, squeezing hand cannot accumulate centimetres of
            # trajectory lag. Real firmware restarts opening from its encoder.
            self.grip_cmd = float(np.clip(self.grip_cmd, self.true_grip()-.004, self.true_grip()+.004))
            self.grip_cmd += float(np.clip(tgt - self.grip_cmd, -step, step))
        b = self.r.base
        track = 2 * b.wheel_y * self.p["wheel_track"]
        wl, wr = (v - w * track / 2) / b.wheel_r, (v + w * track / 2) / b.wheel_r
        self._wheel_v = np.array([wl, wr])
        if kit:
            self._kit(kit)

    def _kit(self, k):
        if k == "fridge_open" and "fridge_opener" in self.kits:
            self.state["fr_open_cmd"] = 1.65
        elif k == "fridge_close" and "fridge_opener" in self.kits:
            self.state["fr_open_cmd"] = 0.0
        elif k == "mw_release" and "mw_pusher" in self.kits:
            self._mw_release()
        elif k == "mw_start" and "mw_pusher" in self.kits:
            self._mw_start()

    # --------------------------------------------------------- appliances --
    def door_angle(self, k):
        return abs(float(self.d.qpos[self.dq[k]]))

    def _eq_force(self, k):
        d = self.d
        idx = np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == self.eq[k]))[0]
        return float(d.efc_force[idx[0]]) if len(idx) else 0.0

    def _mw_release(self):
        d = self.d
        if d.eq_active[self.eq["mw_door"]]:
            d.eq_active[self.eq["mw_door"]] = 0

    def _mw_start(self):
        if self.d.eq_active[self.eq["mw_door"]]:
            self.state["mw_started"].append(self.t)
            self.state["mw_running"] = True

    def _appliances(self):
        d, m, st = self.d, self.m, self.state
        # ---- fridge: magnet gasket. Latched until the pull at the handle passes the seal, then gone within 1 cm of
        # travel; below closes_below the door falls shut; a detent holds it near 95 deg
        dn = "fr_door"
        sgn = -1.0 if self.home.hinge == "west" else 1.0
        al = self.door_angle(dn)
        tau = 0.0
        latched = d.eq_active[self.eq[dn]]
        cmd_open = st["fr_open_cmd"]
        if latched:
            fz = self._eq_force(dn)
            if abs(fz) > self.seal * 0.55 or cmd_open > 0.02:
                d.eq_active[self.eq[dn]] = 0
                latched = 0
        if not latched:
            if "fridge_opener" in self.kits:
                e = cmd_open - al
                tau = float(np.clip(30.0 * e - 12.0 * abs(d.qvel[self.dv[dn]]) * np.sign(d.qvel[self.dv[dn]]) * sgn, -18, 18))
                d.qfrc_applied[self.dv[dn]] = sgn * tau
            else:
                back = 0.0
                if al < math.radians(self.home.closes_below):
                    back = 1.4 * (1 - al / math.radians(self.home.closes_below)) + 0.2
                back += self.seal * 0.55 * math.exp(-al / 0.018) * 0.5
                if math.radians(90) < al < math.radians(99):
                    back -= 0.5
                d.qfrc_applied[self.dv[dn]] = -sgn * back
            if al < 0.006 and cmd_open < 0.02 and abs(d.qvel[self.dv[dn]]) < 0.4:
                d.eq_active[self.eq[dn]] = 1
                d.qfrc_applied[self.dv[dn]] = 0.0
        # ---- microwave: pops on its button, closes against a 10 N catch
        dn = "mw_door"
        al = self.door_angle(dn)
        if not d.eq_active[self.eq[dn]]:
            t = 0.0
            if al < math.radians(30):
                t += 0.40 * (1 - al / math.radians(30))
            if al < math.radians(3) and st.get("mw_armed", False):
                t -= 2.2
            if al > math.radians(92):
                t += 0.3
            d.qfrc_applied[self.dv[dn]] = -t
            st.setdefault("mw_armed", False)
            if al > math.radians(4):
                st["mw_armed"] = True
            if st["mw_armed"] and al < 0.006:
                d.eq_active[self.eq[dn]] = 1
                d.qfrc_applied[self.dv[dn]] = 0.0
                st["mw_armed"] = False
        else:
            d.qfrc_applied[self.dv[dn]] = 0.0
        if st["mw_running"]:
            if d.eq_active[self.eq[dn]]:
                st["mw_heated"] += CTRL_DT
                appliance_judge.accumulate(self, CTRL_DT)
            else:
                st["mw_running"] = False
        for k in ("open", "start"):
            on = d.qpos[self.btn[k]] >= 0.0015
            if on and not st["btn_prev"][k]:
                (self._mw_release if k == "open" else self._mw_start)()
            st["btn_prev"][k] = on
        # ---- dishwasher: 6 kg door, 80 % counterbalanced, 25 N seal
        dn = "dw_door"
        al = self.door_angle(dn)
        if d.eq_active[self.eq[dn]]:
            if abs(self._eq_force(dn)) > 25.0 * 0.62:
                d.eq_active[self.eq[dn]] = 0
        else:
            d.qfrc_applied[self.dv[dn]] = -(0.8 * 6.0 * 9.81 * 0.33 * math.sin(al))
            if al < 0.006:
                d.eq_active[self.eq[dn]] = 1
                d.qfrc_applied[self.dv[dn]] = 0.0

    # ------------------------------------------------------------ sensors --
    def _observe(self):
        d, rng, nz = self.d, self.rng, self.noise
        ob = Obs(t=self.t)
        q, qd = self.true_q(), self.true_qd()
        self.q_hist.append((q.copy(), qd.copy()))
        q0, qd0 = self.q_hist[max(0, len(self.q_hist) - 2)]
        res = np.array([2e-5 if a["a"].kind == "slide" else 2 * math.pi / 4096 / 4 for a in self.ax])
        ob.q = np.round(q0 / res) * res if nz else q0
        ob.qd = qd0
        F = np.array([d.actuator_force[a["act"]] for a in self.ax])
        ob.tau = F * (1 + 0.10 * nz * rng.normal(0, 1, self.na)) + rng.normal(0, 0.01, self.na) * self.frc0 * nz
        ob.grip = float(self.true_grip() + rng.normal(0, 3e-4) * nz)
        ob.grip_f = abs(float(d.actuator_force[self.a_grip]))
        ob.wheels = np.array([d.qpos[q] for q in self.q_wheel])
        R = d.xmat[self.rid].reshape(3, 3)
        ob.tilt = math.degrees(math.acos(min(1.0, R[2, 2])))
        ob.pitch = math.atan2(-R[2, 0], R[2, 2]) + rng.normal(0, math.radians(0.1)) * nz
        ob.roll = math.atan2(R[2, 1], R[2, 2]) + rng.normal(0, math.radians(0.1)) * nz
        if self.tick % 5 == 0:
            ob.scan = self._lidar()
            ob.scan_t = self.t
        if self.tick % 5 == 2:
            self._capture()
        while self.det_queue and self.det_queue[0][0] <= self.t:
            ob.dets.append(self.det_queue.popleft()[1])
        return ob

    def _lidar(self):
        """360 beams, 1 deg, +-1.5 cm (RPLIDAR C1 class) in a plane at the sensor height, seen from the robot."""
        d, m = self.d, self.m
        n = 360
        p = d.site_xpos[self.lidar].copy()
        R = d.site_xmat[self.lidar].reshape(3, 3)
        a = np.arange(n) * 2 * math.pi / n
        vec = (R @ np.stack([np.cos(a), np.sin(a), np.zeros(n)])).T.reshape(-1)
        gid, dist = np.zeros(n, np.int32), np.zeros(n)
        mujoco.mj_multiRay(m, d, p, vec, RAY_GROUP, 1, -1, gid, dist, None, n, 12.0)
        dist = np.where(dist < 0, np.nan, dist)
        return dist + self.rng.normal(0, 0.015, n) * self.noise + self.lidar_bias

    def _capture(self):
        """Every camera looks for every object in view. Line of sight is ray-cast to points on the object: nothing is
        known about an object nobody can see, whatever a script says."""
        d, m, rng, nz = self.d, self.m, self.rng, self.noise
        for name, (c, site) in self.cams.items():
            if self.home.dark and c.kind == "rgb":
                continue
            p = d.site_xpos[site].copy()
            R = d.site_xmat[site].reshape(3, 3)
            look = R @ (np.array(c.look) / np.linalg.norm(c.look))
            up = R @ np.array([0, 0, 1.0])
            right = np.cross(look, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, look)
            for oname, o in {**self.obj, **self.landmarks}.items():
                c0 = d.xpos[o["b"]]
                if c0[2] < -1:
                    continue
                Ro = d.xmat[o["b"]].reshape(3, 3)
                v0 = c0 + Ro @ o["pts"][0] - p
                dist = float(np.linalg.norm(v0))
                if dist > c.far * 1.1:
                    continue
                seen = 0
                for pt in o["pts"]:
                    w = c0 + Ro @ pt
                    v = w - p
                    dd = float(np.linalg.norm(v))
                    x, y, z = v @ right, v @ up, v @ look
                    if z <= 0 or dd < c.near or dd > c.far:
                        continue
                    if abs(math.atan2(x, z)) > math.radians(c.fov[0] / 2) or abs(math.atan2(y, z)) > math.radians(c.fov[1] / 2):
                        continue
                    gid = np.array([-1], dtype=np.int32)
                    fr = mujoco.mj_ray(m, d, p, v / dd, None, 1, self.rid, gid)     # the arm can block the view, the housing does not
                    if fr < 0 or fr > dd - 0.02 or m.geom_bodyid[gid[0]] == o["b"]:
                        seen += 1
                vis = seen / len(o["pts"])
                if vis < 0.35:
                    continue
                # position error grows with range and shrinks with what is seen; 1 % of detections are simply wrong
                sigma = (0.002 + 0.008 * dist) / max(vis, 0.3) * nz
                if self.home.dark:
                    sigma *= 1.6
                err = rng.normal(0, sigma, 3)
                if rng.uniform() < 0.01 * nz:
                    err = err + rng.normal(0, 0.05, 3)
                tgt = c0 + Ro @ o["pts"][0]
                cam_R = np.stack([right, up, look], axis=1)
                dq = np.zeros(4)
                mujoco.mju_axisAngle2Quat(dq, self.cam_err[name] / (np.linalg.norm(self.cam_err[name]) + 1e-12), float(np.linalg.norm(self.cam_err[name])))
                M = np.zeros(9)
                mujoco.mju_quat2Mat(M, dq)
                pc = cam_R.T @ (tgt - p)
                pc = M.reshape(3, 3) @ pc + err
                yaw = math.atan2(Ro[1, 0], Ro[0, 0]) + rng.normal(0, math.radians(6)) * nz
                det = Detection(oname, self.t, pc, yaw, name, vis)
                self.det_queue.append((self.t + 0.04 + 0.02 * self.p["latency_ticks"], det))

    def cam_pose_nominal(self, belief):
        return None

    # --------------------------------------------------------------- step --
    def step(self):
        """One control tick."""
        m, d = self.m, self.d
        self._apply(getattr(self, "_wheel_v", np.zeros(2)), self.qd_cmd)
        self._appliances()
        # motor firmware interpolates between commands; a 20 ms staircase at this stiffness is 75 N of ripple
        n = int(round(CTRL_DT / m.opt.timestep))
        q0, q1 = self.q_last, self.q_cmd
        for k in range(1, 6):
            d.ctrl[self.c_axes] = q0 + (q1 - q0) * (k / 5)
            mujoco.mj_step(m, d, nstep=n // 5)
        self.q_last = np.array(q1, float)
        self.t += CTRL_DT
        # MuJoCo resets the state when it blows up and carries on from the origin. That must never pass for a result
        if not self.mon["diverged"] and (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number or d.warning[mujoco.mjtWarning.mjWARN_BADQVEL].number or d.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number):
            self.mon["diverged"] = True
        self.tick += 1
        self._monitor()
        if self.record is not None and self.tick % 10 == 0:
            self.record.append((self.t, d.qpos.copy()))
        return self._observe()

    def _monitor(self):
        d, mn = self.d, self.mon
        n = d.ncon
        if n:
            g = d.contact.geom[:n]
            a, b = g[:, 0], g[:, 1]
            f = np.abs(d.efc_force[d.contact.efc_address[:n]])
            rob = self.g_rob[a] | self.g_rob[b]
            per = self.g_person[a] | self.g_person[b]
            if np.any(rob & per):
                mn["max_person"] = max(mn["max_person"], float(f[rob & per].max()))
            wheel = self.g_wheel[a] | self.g_wheel[b]
            obj = self.g_obj[a] | self.g_obj[b]
            pad = self.g_pad[a] | self.g_pad[b]
            hit = rob & ~wheel & ~obj & ~per & ~pad
            for k in np.nonzero(rob & ~wheel & (f > 10))[0]:
                key = " / ".join(sorted([self.gname[a[k]], self.gname[b[k]]]))
                mn["contact_peaks"][key] = max(mn["contact_peaks"].get(key, 0), round(float(f[k]), 2))
            if np.any(hit):
                top = float(f[hit].max())
                mn["max_bump"] = max(mn["max_bump"], top)
                if top > 40:
                    mn["bumps"] += 1
        R = d.xmat[self.rid].reshape(3, 3)
        tilt = math.degrees(math.acos(min(1.0, R[2, 2])))
        mn["max_tilt"] = max(mn["max_tilt"], tilt)
        if tilt > 12.0:
            mn["tip"] = True
        # heat and energy: (force / continuous rating)^2, one minute time constant
        F = np.array([abs(d.actuator_force[a["act"]]) for a in self.ax])
        cont = np.array([a["a"].force for a in self.ax])
        self.heat[:self.na] += CTRL_DT * ((F / cont) ** 2 - self.heat[:self.na]) / 60.0
        mn["peak_heat"] = max(mn["peak_heat"], float(self.heat.max()))
        self.energy += CTRL_DT * float(np.abs(F * self.true_qd()).sum())
        # liquids: the free surface tilts with the bowl and with its acceleration. Spills when it reaches the rim
        for name, o in self.obj.items():
            liq = o["o"].liquid
            if not liq or o.get("empty"):
                continue
            # cacc is not populated by this sensor-free model unless RNE post-
            # constraint is explicitly requested. Differentiate free-body world
            # velocity instead; smooth on a labelled 100 ms quasi-static scale.
            vel = d.qvel[o["v"]:o["v"]+3].copy()
            acc = (vel-o.get("prev_vel", vel))/CTRL_DT
            o["prev_vel"] = vel
            o["acc"] = o.get("acc", np.zeros(3))+.167*(acc-o.get("acc", np.zeros(3)))
            Ro = d.xmat[o["b"]].reshape(3, 3)
            effective_g = Ro.T@(np.array([0, 0, -9.81])-o["acc"])
            slope = float(np.linalg.norm(effective_g[:2]))/max(-effective_g[2], 1e-3)
            rise = liq["level"] + liq["r"] * slope
            mn["spill_margin"] = min(mn["spill_margin"], liq["rim"] - rise)
            if rise > liq["rim"] and float(np.linalg.norm(d.qvel[o["v"]:o["v"] + 3])) > 1e-3:
                mn["spill"] = True
