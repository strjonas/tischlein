"""World + robot + objects + physics settings -> one MuJoCo model.

Physics settings live here and nowhere else (PHYS). They were chosen by tests/test_physics.py, not by feel:
elliptic friction, implicitfast, multi-point contact for flat convex pairs. Friction is a property of a pair of
materials, so grip pads get an explicit pair with each object: MuJoCo's default takes the higher of the two geoms'
values, which made the old "pad friction 0.2" act as 0.3 and could not express a wet bowl in a rubber hand.

No PID is hidden in the world: all forces reach the robot through its actuators, clamped at the ratings of its parts.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import homes, objects
from .homes import FLOOR_MU, Home
from .robot import Robot, robot_mjcf

# High impratio is needed for the tested pinch-friction accuracy. The old
# impratio=10 control holds only 56-69% of 2 mu N; see tests/test_physics.py.
PHYS = dict(timestep=0.002, solref=0.004, solimp=(0.95, 0.99, 0.001, 0.5, 2.0), impratio=50.0, noslip=0, multiccd=True, iterations=100, obj_armature=5e-4, solver="Newton", cone="elliptic", pad_sol=2.0)

# static friction of a gripper pad against each material; estimates from tyre-rubber and TPU tables, DR varies them
PAD_MU = dict(tpu=dict(ceramic=0.7, glass=0.6, pet=0.6, plastic=0.6, paper=0.65, steel=0.55),
              silicone=dict(ceramic=0.9, glass=0.8, pet=0.8, plastic=0.8, paper=0.8, steel=0.7),
              pla=dict(ceramic=0.25, glass=0.2, pet=0.25, plastic=0.25, paper=0.35, steel=0.22))
# object on a household surface (wood, glass shelf, laminate)
OBJ_MU = dict(ceramic=0.30, glass=0.25, pet=0.35, plastic=0.35, paper=0.40, steel=0.25)

CLS = {"world": (1, 14), "person": (1, 14), "rob": (2, 13), "arm": (4, 15), "pad": (4, 15), "obj": (8, 15), "vis": (0, 0)}


def header(p=PHYS, name="arena"):
    defaults = "\n".join(f'    <default class="{k}"><geom contype="{c}" conaffinity="{a}" solref="{p["solref"]} 1" '
                         f'solimp="{" ".join(map(str, p["solimp"]))}" friction="0.35 0.005 0.0001" condim="4"/></default>'
                         for k, (c, a) in CLS.items())
    flag = '<flag multiccd="enable"/>' if p["multiccd"] else ""
    return f"""<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{p['timestep']}" gravity="0 0 -9.81" integrator="implicitfast" cone="{p['cone']}" solver="{p['solver']}" impratio="{p['impratio']}" noslip_iterations="{p['noslip']}" iterations="{p['iterations']}" ls_iterations="{max(50, p['iterations'] // 2)}">{flag}</option>
  <size memory="256M"/>
  <visual><global offwidth="1600" offheight="1000"/><headlight ambient="0.45 0.45 0.45" diffuse="0.35 0.35 0.35"/></visual>
  <default>
{defaults}
    <default class="tyre"><geom contype="2" conaffinity="1" condim="3" solref="0.015 1" friction="0.5 0.005 0.0001" priority="2"/></default>
    <default class="caster"><geom contype="2" conaffinity="1" condim="3" solref="0.01 1" friction="0.5 0.005 0.0001" priority="2"/></default>
  </default>
  <worldbody>
    <light pos="2 6 2.4" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light pos="6 6 2.4" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
"""


@dataclass
class Built:
    m: object
    spec: object
    world: homes.B
    robot: Robot
    home: Home
    names: list
    digest: str
    phys: dict = None


def _pid(spec, name, target, kp, frc, gear=1.0, ki=0.0, trn=mujoco.mjtTrn.mjTRN_JOINT):
    a = spec.add_actuator(name=name, target=target, trntype=trn)
    a.set_to_pid(kp=kp, kv=-1, dampratio=1.0, ki=ki, imax=frc if ki > 0 else -1)
    a.gear[0] = gear
    a.forcelimited = True
    a.forcerange = [-frc, frc]
    return a


def build(robot: Robot = None, home: Home = None, objs=(), kits=(), phys=None):
    """objs: names from objects.CATALOG, or (instance name, catalog name). kits add to what the robot brings."""
    home = home or homes.HOMES["A"]
    p = dict(PHYS, **(phys or {}))
    kits = tuple(sorted(set(kits) | set(robot.kits if robot else ())))
    w = homes.build(home, kits)
    body = list(w.lines)
    pairs = []
    if robot:
        body.append(robot_mjcf(robot))
    names = []
    for o in objs:
        inst, cat = o if isinstance(o, tuple) else (o, o)
        body.append(objects.mjcf(objects.CATALOG[cat], inst))
        names.append((inst, cat))
    xml = header(p) + "\n".join("    " + l for l in "\n".join(body).split("\n")) + "\n  </worldbody>\n</mujoco>\n"
    spec = mujoco.MjSpec.from_string(xml)
    spec.memory = 256 * 2 ** 20
    mu_floor = FLOOR_MU[home.floor]
    for g in spec.geoms:
        n = g.name or ""
        if n.startswith("rob_tyre") or n.startswith("rob_cast"):
            g.friction = [mu_floor if n.startswith("rob_tyre") else 0.8, 0.005, 0.0001]
    if robot:
        _robot_spec(spec, robot)
    _world_spec(spec, w)
    # friction by material pair
    for inst, cat in names:
        o = objects.CATALOG[cat]
        for g in spec.geoms:
            if (g.name or "").startswith(f"obj_{inst}_"):
                g.friction = [OBJ_MU[o.material], 0.005, 0.0001]
        if robot:
            mu = PAD_MU[robot.grip.pad][o.material]
            for g in spec.geoms:
                n = g.name or ""
                if n.startswith(f"obj_{inst}_"):
                    for pad in ("pad_a", "pad_b"):
                        q = spec.add_pair(name=f"p_{pad}_{n}", geomname1=pad, geomname2=n)
                        q.condim = 4
                        q.friction = [mu, mu, 0.005, 0.0001, 0.0001]
                        q.solref = [p["solref"] * p["pad_sol"], 1.0]       # rubber: softer than ceramic on wood
                        q.solimp = [0.9, 0.99, 0.001, 0.5, 2.0]
    m = spec.compile()
    dig = hashlib.sha256((spec.to_xml() + repr(sorted(p.items()))).encode()).hexdigest()
    # very light things squeezed by a strong gripper make the contact solve ill-conditioned: a small rotational armature
    # keeps it well posed without changing how they rest, slip or are held (tests/test_physics.py)
    if p["obj_armature"]:
        for inst, _ in names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, inst)
            a = m.jnt_dofadr[j]
            m.dof_armature[a + 3:a + 6] = p["obj_armature"]
        mujoco.mj_setConst(m, mujoco.MjData(m))        # masses changed: solver constants must follow, or the solve is wrong
    return Built(m, spec, w, robot, home, names, dig, p)


def _robot_spec(spec, r: Robot):
    """Motors, stage couplings, caster springs. Gains: full continuous force at 2 mm or 0.3 deg of error."""
    for a in r.chain:
        tgt = a.name
        gear = float(a.stages)
        if a.stages > 1:
            for k in range(2, a.stages + 1):
                e = spec.add_equality(name=f"couple_{a.name}{k}", type=mujoco.mjtEq.mjEQ_JOINT, name1=f"{a.name}{k}", name2=a.name)
                e.data[:5] = [0, 1, 0, 0, 0]
                e.objtype = mujoco.mjtObj.mjOBJ_JOINT
        kp = a.peak / (0.002 if a.kind == "slide" else math.radians(0.3))
        # ki = kp would add 150 N per second for each mm of tracking lag: windup that reads like a collision
        _pid(spec, f"a_{a.name}", tgt, kp, a.peak, gear=gear, ki=0.3 * kp)
    g = r.grip
    jo = spec.add_tendon(name="jaw_open")
    jo.wrap_joint("jaw_a", 1.0)
    jo.wrap_joint("jaw_b", 1.0)
    # the jaws move together, one drive sets the opening; a float would go here
    e = spec.add_equality(name="jaw_sync", type=mujoco.mjtEq.mjEQ_JOINT, name1="jaw_b", name2="jaw_a")
    e.data[:5] = [0, 1, 0, 0, 0]
    e.objtype = mujoco.mjtObj.mjOBJ_JOINT
    _pid(spec, "a_grip", "jaw_open", g.force / 0.002, g.force, trn=mujoco.mjtTrn.mjTRN_TENDON)
    for side in "lr":
        _pid(spec, f"a_wheel_{side}", f"wheel_{side}", 400.0, r.base.torque * 1.5, ki=400.0)
    # spring legs of the casters: preload carries (1 - drive_share) of the weight
    n = len(r.base.casters)
    # estimated total mass; refined after compile by assemble.tune_casters
    tot = r.base.mass + sum(p[0] for p in r.base.parts) + sum(a.mass for a in r.chain) + g.mass
    pre = (1 - r.base.drive_share) * tot * 9.81 / n
    tr = r.base.travel
    for i in range(n):
        j = spec.joint(f"caster{i}_z")
        j.stiffness = [pre / tr, 0, 0]
        j.springref = -tr
        j.damping = [300.0, 0, 0]
        j.armature = 0.05


def _world_spec(spec, w: homes.B):
    """Door friction and inertia; buttons on springs; soft mattress."""
    door_m = dict(fr_door=8.0, mw_door=1.2, dw_door=6.0)
    for nm, kg in door_m.items():
        b = spec.body(nm)
        # a door's inertia comes from its slab: uniform plate about its hinge
        b.explicitinertial = False
        for g in b.geoms:
            if g.name.endswith("slab"):
                g.mass = kg
            elif g.contype:
                g.mass = 0.02
        j = spec.joint(nm)
        j.damping = [0.5 if nm != "mw_door" else 0.05, 0, 0]
        j.frictionloss = 0.3 if nm == "fr_door" else 0.05
        j.armature = 0.01
        a = spec.add_equality(name=f"{nm}_latch", type=mujoco.mjtEq.mjEQ_JOINT, name1=nm)
        a.data[:5] = [0, 0, 0, 0, 0]
        a.objtype = mujoco.mjtObj.mjOBJ_JOINT
        a.solref = [0.01, 1.0]
        a.active = False
    for nm in ("btn_open", "btn_start"):
        j = spec.joint(nm)
        j.stiffness = [5.0 / 0.0115, 0, 0]
        j.springref = -0.010
        j.damping = [5.0, 0, 0]
        j.armature = 0.01
        spec.body(nm).add_geom(name=nm + "_m", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], mass=0.01, contype=0, conaffinity=0, rgba=[0, 0, 0, 0])
    spec.geom("bed_mattress").solref = [0.03, 1.0]


def geom_owner(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]


def ground_robot(b: Built, d, x, y, yaw, settle=False):
    """Put the robot on the floor at (x, y, yaw) with the arm stowed."""
    m = b.m
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base")
    a = m.jnt_qposadr[j]
    d.qpos[a:a + 3] = [x, y, 0.0015]
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), yaw)
    d.qpos[a + 3:a + 7] = q
    for ax in b.robot.chain:
        for k in range(ax.stages):
            nm = ax.name if k == 0 else f"{ax.name}{k + 1}"
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            d.qpos[m.jnt_qposadr[jid]] = ax.home / ax.stages
    mujoco.mj_kinematics(m, d)
