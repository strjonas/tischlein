"""Tischlein's real kinematics inside EgoGym, so the pretrained CAP policies can be run on it.

EgoGym's "cap" robot is a floating gripper with an iPhone: whatever pose the policy asks for, it gets.
Here the same gripper sits on Tischlein: diff-drive base, belt lift, flat SCARA arm (2 x 220 mm),
wrist yaw + pitch, and optionally a wrist roll. Every pose the policy commands is passed through a
whole-body IK first. The gripper only goes where the robot can actually put it, so the policy sees
(and has to live with) what our kinematics allow.

What is modelled:
- joint limits and link lengths from params.py,
- the base may drive along its *current* heading and turn a little during the episode, never slide sideways
  (a diff drive can do both; each step's translation is integrated along the mid-step heading, so no sideways slip),
- per-step limits (one policy step ~0.5 s on the real robot): lift 3 cm, base 5 cm / 5 deg; shoulder 45 deg and
  elbow 60 deg (STS3215 at ~270 deg/s through the 3:1 and 2:1 belts), wrist joints 90 deg (direct drive),
- the mast and tray must not run into the table or cabinet (tray may pass over a table if it is above it),
- v3: the arm links (elbow, wrist) may only pass over a table above its top, and stay in front of a cabinet front.
What is not: link dynamics, link-vs-object collisions, servo slop, and the camera seeing our own arm.
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
import egogym  # noqa: F401  (registers the stock tasks)
from egogym.embodiments import ROBOT_REGISTRY
from egogym.embodiments.grippers.cap import CAPGripper
from egogym.tasks.pick import PickTask
from egogym.tasks.open import OpenTask
from egogym.tasks.close import CloseTask
from egogym.utils import get_pose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import params as P  # noqa: E402

mm = 1e-3
MAST_FRONT = (P.MAST_X + P.MAST_W_X / 2) * mm
SHOULDER_X = P.SHOULDER_X * mm
TRAY_FRONT = (P.TRAY_X0 + P.TRAY_DEPTH) * mm
PITCH_ABOVE_TRAY = (P.L2_ABOVE_TRAY - P.PITCH_DROP) * mm
L1, L2 = P.L1 * mm, P.L2 * mm
Z_MIN, Z_MAX = P.TRAY_Z_MIN * mm, P.TRAY_Z_MAX * mm
GRIP_BACK = 0.05        # CAP gripper origin sits 5 cm in front of its back face
BRACKET = 0.02          # pitch axis -> gripper back face
ROLL_LEN = 0.045        # an STS3215 roll stage adds about this much
# tool frame (x forward, y left, z up) -> CAP gripper frame (x lateral, y forward, z up)
M_TOOL_TO_GRIP = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def near_start_pose(target, back=0.30, up=0.25):
    """Camera pose `back` m in front of and `up` m above the target, looking at it (bench protocol start)."""
    cam = np.array(target) + np.array([0.0, -back, up])
    f = np.array(target) - cam
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up_img = np.cross(right, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, up_img, -f])     # MuJoCo cameras look along -z, y up
    T[:3, 3] = cam
    return T


class CAPNear(CAPGripper):
    """Ideal floating gripper, but starting ~40 cm from the target like the bench protocol."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.context_fn = None

    def prepare(self, desired_pose, step_fn=None):
        super().prepare(near_start_pose(self.context_fn()["target"]), step_fn)


class TischleinCAP(CAPGripper):
    """CAP gripper carried by Tischlein. q = [s, dth, z_tray, j1, j2, j3, j4, (j5)]."""

    has_roll = False
    move_base = True
    roll_prior = None       # rad; if set, the roll joint is pulled toward 0 and only follows large roll requests
    start_near = False      # start ~40 cm from the target (bench protocol) instead of EgoGym's far start
    lock_lift = False       # bench rig without a lift: the carriage stays at its start height
    start_reach = 0.32      # horizontal shoulder -> pitch-axis distance in the start pose
    park_reach = None       # frozen base: None = as close as the furniture allows; "start" = where the start pose is comfortable
    link_check = True       # v3: elbow/wrist vs table top and cabinet front (v2 checked only mast and tray)
    anchor_err = None       # v3: callable(rng, T_cam, p_obj) -> 3D error (m) added to the contact anchor, fixed per episode
    # per policy step
    STEP = dict(s=0.05, dth=np.radians(5), z=0.03, arm=np.radians([45, 60, 90, 90, 90]))

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.name = "cap"   # reuse the CAP gripper model files
        self.context_fn = None
        self.q = None
        self.ik_log = []

    # ---------- kinematics ----------
    def n_q(self):
        return 8 if self.has_roll else 7

    def bounds(self):
        lo = [-1.5, np.radians(-30), Z_MIN, np.radians(-150), np.radians(-155), np.radians(-180), np.radians(-20)]
        hi = [1.5, np.radians(30), Z_MAX, np.radians(150), np.radians(155), np.radians(180), np.radians(100)]
        if not self.move_base:
            lo[0], hi[0], lo[1], hi[1] = -1e-6, 1e-6, -1e-6, 1e-6
        if self.lock_lift and getattr(self, "z_lock", None) is not None:
            lo[2], hi[2] = self.z_lock - 1e-6, self.z_lock + 1e-6
        if self.has_roll:
            lo.append(np.radians(-180))
            hi.append(np.radians(180))
        return np.array(lo), np.array(hi)

    def base_pose(self, q):
        """Diff drive: q[0] is the path length driven, q[1] the heading change, both since the episode start.
        The translation since the last committed step goes along the mid-step heading, so the base never
        slides sideways relative to its wheels (v2 translated along the *initial* heading: wrong once it turned)."""
        bx0, by0, th0 = self.base_ref
        ds = q[0] - self.q_ref[0]
        dth = q[1] - self.q_ref[1]
        hm = th0 + dth / 2
        return bx0 + ds * np.cos(hm), by0 + ds * np.sin(hm), th0 + dth

    def commit(self, q):
        """Accept q as the robot's new state: the base pose it implies becomes the reference for the next step."""
        self.base_ref = np.array(self.base_pose(q))
        self.q_ref = np.array(q, dtype=float)
        self.q = q

    def fk(self, q):
        """-> base xy, heading, tray z, gripper rotation (world), grasp centre, gripper origin."""
        bx, by, th = self.base_pose(q)
        zt = q[2]
        sx, sy = bx + SHOULDER_X * np.cos(th), by + SHOULDER_X * np.sin(th)
        a1 = th + q[3]
        a2 = a1 + q[4]
        psi = a2 + q[5]
        wx = sx + L1 * np.cos(a1) + L2 * np.cos(a2)
        wy = sy + L1 * np.sin(a1) + L2 * np.sin(a2)
        p_pitch = np.array([wx, wy, zt + PITCH_ABOVE_TRAY])
        R_hand = rz(psi) @ ry(q[6])
        if self.has_roll:
            R_tool = R_hand @ rx(q[7])
            reach = BRACKET + ROLL_LEN + GRIP_BACK
        else:
            R_tool = R_hand
            reach = BRACKET + GRIP_BACK
        p_grip = p_pitch + R_hand[:, 0] * reach
        R_grip = R_tool @ M_TOOL_TO_GRIP
        p_gc = p_grip + R_grip @ self.gc_in_grip
        return bx, by, th, zt, R_grip, p_gc, p_grip

    def residual(self, q, R_des, p_gc_des, q_prev):
        bx, by, th, zt, R_g, p_gc, _ = self.fk(q)
        r = [(p_gc - p_gc_des) / 0.005,
             (R_g[:, 1] - R_des[:, 1]) / 0.03,      # approach direction
             (R_g[:, 2] - R_des[:, 2]) / 0.15]      # roll about the approach axis
        # regularise: prefer the arm over base motion, stay away from joint limits
        r.append(np.array([(q[0] - q_prev[0]) / 0.3, (q[1] - q_prev[1]) / 0.3, (q[2] - q_prev[2]) / 0.5]))
        r.append((q[3:] - q_prev[3:]) / 2.0)
        if self.has_roll and self.roll_prior:
            r.append(np.array([q[7] / self.roll_prior]))
        r.append(self.obstacle_penalty(bx, by, th, zt, q))
        return np.concatenate(r)

    def link_points(self, bx, by, th, q):
        """Elbow and wrist (pitch axis) positions in the world, xy only."""
        sx, sy = bx + SHOULDER_X * np.cos(th), by + SHOULDER_X * np.sin(th)
        a1 = th + q[3]
        a2 = a1 + q[4]
        ex, ey = sx + L1 * np.cos(a1), sy + L1 * np.sin(a1)
        return [(ex, ey), (ex + L2 * np.cos(a2), ey + L2 * np.sin(a2))]

    def obstacle_penalty(self, bx, by, th, zt, q):
        c = self.ctx
        fwd = np.array([np.cos(th), np.sin(th)])
        mast_front = np.array([bx, by]) + MAST_FRONT * fwd
        tray_front = np.array([bx, by]) + TRAY_FRONT * fwd
        pen = []
        # forearm plane (lowest link) above the tray surface; v3: links must clear the furniture too
        z_links = zt + P.L2_ABOVE_TRAY * mm - 0.02
        if c["kind"] == "table":
            # mast must stay behind the table edge; tray may overhang only above the table top
            pen.append(max(0.0, mast_front[1] - (c["edge_y"] - 0.02)) / 0.002)
            over = tray_front[1] - c["edge_y"]
            low = (c["top_z"] + 0.015) - zt
            pen.append(max(0.0, min(over, low)) / 0.002)
            if self.link_check:
                for _, ly in self.link_points(bx, by, th, q):
                    pen.append(max(0.0, min(ly - c["edge_y"], (c["top_z"] + 0.01) - z_links)) / 0.002)
        else:
            # cabinet or drawer front: tray front stays 1 cm in front of it
            pen.append(max(0.0, tray_front[1] - (c["front_y"] - 0.01)) / 0.002)
            pen.append(0.0)
            if self.link_check:
                # elbow stays in front of the shut front; the wrist may reach to it (the hand goes in, not the arm)
                (_, ey), (_, wy) = self.link_points(bx, by, th, q)
                pen.append(max(0.0, ey - (c["front_y"] - 0.03)) / 0.002)
                pen.append(max(0.0, wy - c["front_y"]) / 0.002)
        return np.array(pen)

    def solve(self, T_cam_des, q0, step_limits=True):
        T_grip_des = T_cam_des @ self.T_grip_cam
        R_des = T_grip_des[:3, :3]
        p_gc_des = T_grip_des[:3, 3] + R_des @ self.gc_in_grip
        lo, hi = self.bounds()
        if step_limits:
            st = self.STEP
            d = np.concatenate([[st["s"], st["dth"], st["z"]], st["arm"][: self.n_q() - 3]])
            lo, hi = np.maximum(lo, q0 - d), np.minimum(hi, q0 + d)
        x0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
        sol = least_squares(self.residual, x0, bounds=(lo, hi), args=(R_des, p_gc_des, q0),
                            method="trf", max_nfev=60, xtol=1e-6, ftol=1e-6)
        q = sol.x
        _, _, _, _, R_g, p_gc, p_grip = self.fk(q)
        pos_err = np.linalg.norm(p_gc - p_gc_des)
        ang_err = np.degrees(np.arccos(np.clip((np.trace(R_g.T @ R_des) - 1) / 2, -1, 1)))
        app_err = np.degrees(np.arccos(np.clip(R_g[:, 1] @ R_des[:, 1], -1, 1)))
        T_grip = np.eye(4)
        T_grip[:3, :3], T_grip[:3, 3] = R_g, p_grip
        return q, T_grip @ self.T_cam_in_grip, dict(pos=pos_err, ang=ang_err, app=app_err)

    # ---------- EgoGym hooks ----------
    def setup_mj(self, model, data):
        super().setup_mj(model, data)
        import mujoco
        mujoco.mj_forward(model, data)
        T_g = get_pose(data, "base", "body")
        T_c = get_pose(data, "camera_body", "body")
        T_gc = get_pose(data, "grasping_center", "site")
        self.T_cam_in_grip = np.linalg.inv(T_g) @ T_c
        self.T_grip_cam = np.linalg.inv(self.T_cam_in_grip)
        self.gc_in_grip = (np.linalg.inv(T_g) @ T_gc)[:3, 3]
        self.q = None

    def place_base(self, T_cam):
        """Park the robot square to the furniture, in line with the target, far enough back to reach the start pose."""
        self.ctx = self.context_fn()
        T_grip = T_cam @ self.T_grip_cam
        p_gc = T_grip[:3, 3] + T_grip[:3, :3] @ self.gc_in_grip
        th = np.pi / 2                       # the tasks put the furniture towards +y
        x = self.ctx["target"][0]
        # shoulder start_reach behind the hand's pitch axis
        reach = BRACKET + GRIP_BACK + (ROLL_LEN if self.has_roll else 0) + self.gc_in_grip[1]
        y = p_gc[1] - reach - self.start_reach - SHOULDER_X
        if self.ctx["kind"] == "table":
            closest = self.ctx["edge_y"] - 0.02 - MAST_FRONT
        else:
            closest = self.ctx["front_y"] - 0.01 - TRAY_FRONT
        # a base that won't move during the motion parks as close as the furniture allows;
        # the start pose is then the nearest one the arm can reach from there
        if not self.move_base and self.park_reach != "start":
            y = closest                      # frozen base parked as close as the furniture allows
        else:
            y = min(y, closest)              # park so the start pose sits at a comfortable reach
        self.base0 = np.array([x, y, th])
        self.base_ref = self.base0.copy()
        self.q_ref = np.zeros(self.n_q())
        # start the arm folded forward, elbow bent left
        q0 = np.zeros(self.n_q())
        q0[2] = np.clip(p_gc[2] - PITCH_ABOVE_TRAY, Z_MIN, Z_MAX)
        q0[3], q0[4], q0[5] = np.radians(-40), np.radians(80), np.radians(-40)
        return q0

    def prepare(self, desired_pose, step_fn=None):
        self.z_lock = None
        if self.start_near:
            desired_pose = near_start_pose(self.context_fn()["target"])
        if self.add_noise_init:
            desired_pose = desired_pose.copy()
            desired_pose[:3, 3] += self.np_random.uniform(-0.03, 0.03, size=3)
            e = R.from_matrix(desired_pose[:3, :3]).as_euler("xyz")
            e += self.np_random.uniform([-0.1, -0.05, -0.1], [0.1, 0.05, 0.1])
            desired_pose[:3, :3] = R.from_euler("xyz", e).as_matrix()
        q0 = self.place_base(desired_pose)
        q = q0
        for _ in range(8):               # converge without step limits: this is the robot's ready pose
            q, T_cam, err = self.solve(desired_pose, q, step_limits=False)
            self.commit(q)
        if self.lock_lift:
            self.z_lock = float(q[2])
        self.ik_log = [dict(t=0, **err)]
        self.last_teleop_pose = T_cam
        self._set_mocap(T_cam)
        step_fn(self.control_steps)

    def _set_mocap(self, T):
        self.data.mocap_pos[0] = T[:3, 3]
        self.data.mocap_quat[0] = np.roll(R.from_matrix(T[:3, :3]).as_quat(), 1)

    def move(self, desired_pose):
        q, T_cam, err = self.solve(desired_pose, self.q, step_limits=True)
        self.commit(q)
        self.ik_log.append(dict(t=len(self.ik_log), **err))
        self._set_mocap(T_cam)


class Tischlein5(TischleinCAP):
    """Today's design: wrist yaw + pitch."""
    has_roll = False


class Tischlein6(TischleinCAP):
    """Design change: add a wrist roll servo."""
    has_roll = True


class Tischlein6Selective(TischleinCAP):
    """Wrist roll that stays near 0 unless the policy asks for a clear turn (handles), instead of chasing every wobble."""
    has_roll = True
    roll_prior = 0.35


class BenchLift(TischleinCAP):
    """Bench rig: arm + lift on a stand at the table edge, no base. Selective roll. Starts near the target."""
    has_roll = True
    roll_prior = 0.35
    move_base = False
    start_near = True
    park_reach = "start"
    start_reach = 0.14      # start with the arm folded, so all of its 0.44 m reach is left for the approach


class BenchNoLift(BenchLift):
    """Bench rig without the lift: carriage fixed at its start height."""
    lock_lift = True


class Tischlein6Near(TischleinCAP):
    """Full robot (base moves), selective roll, starting near the target: the reference for the bench rigs."""
    has_roll = True
    roll_prior = 0.35
    start_near = True


class Tischlein6Fixed(TischleinCAP):
    """Wrist roll, but the base stays parked during the motion."""
    has_roll = True
    move_base = False


ROBOT_REGISTRY.update({"tisch5": Tischlein5, "tisch6": Tischlein6, "tisch6sel": Tischlein6Selective,
                       "tisch6fixed": Tischlein6Fixed, "capnear": CAPNear, "tisch6near": Tischlein6Near,
                       "benchlift": BenchLift, "benchnolift": BenchNoLift})


# ---------- tasks that tell the robot where the furniture is ----------
def _bind(task):
    if isinstance(task.robot, (TischleinCAP, CAPNear)):
        task.robot.context_fn = lambda: task.tischlein_context()


class TPick(PickTask):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _bind(self)

    def tischlein_context(self):
        return dict(kind="table", edge_y=-(self.spread + 0.24), top_z=0.76,
                    target=self.object.get_pose(self.data)[:3, 3].copy())


class _Furniture:
    def tischlein_context(self):
        import mujoco
        h = self.object.get_handle_pose(self.data)[:3, 3].copy()
        # the front plane is where the handle sits when the door/drawer is shut
        j = self.data.joint("task_joint")
        q_now = float(j.qpos[0])
        j.qpos[0] = self.object.initial_joint if self.object.initial_joint is not None else 0.0
        mujoco.mj_kinematics(self.model, self.data)
        h_shut = self.object.get_handle_pose(self.data)[:3, 3].copy()
        j.qpos[0] = q_now
        mujoco.mj_kinematics(self.model, self.data)
        return dict(kind="front", front_y=h_shut[1] + 0.04, target=h)


class TOpen(_Furniture, OpenTask):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _bind(self)


class TClose(_Furniture, CloseTask):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _bind(self)


for _id, _cls in [("Egogym-Pick-v0", TPick), ("Egogym-Open-v0", TOpen), ("Egogym-Close-v0", TClose)]:
    gym.registry.pop(_id, None)
    gym.register(id=_id, entry_point=_cls)
