"""Fridge door, v0.2 robot: one arm vs two arms, at real joint speeds, with the robot's own collision shapes.

uv run python sim/fridge_door.py                 # sweep: strategy x door type x self-closing x seal x floor
uv run python sim/fridge_door.py --quick         # one case per strategy and door
uv run python sim/fridge_door.py --video one upper cam25 35 0.6     # one run -> out/fridge_door_<...>.mp4

Compared with a simulation that lets the re-grasp drop the lift 26 cm in 0.7 s (the design runs the lift at
6 cm/s, ~4.3 s for that), has no collision shapes for the robot, and uses one door that starts 0.885 m above
the floor so that the base can drive underneath it, this one has:
  - every joint is held to its real speed (sim/demo.py VMAX: lift 6 cm/s, J1 60 deg/s, ...); a rate limiter on
    the actuator targets enforces it and reports any lag;
  - the base, legs, wheels, mast, carriage, face, tray and both arms' links collide with the door and cabinet;
  - two doors: "upper" (fridge-freezer, door from 0.885 m) and "full" (single tall door from 4 cm, the base
    cannot pass under it);
  - self-closing behaviour: "cam25" (closes itself below 25 deg, otherwise stays), "cam60" (below 60 deg),
    "always" (closes from anywhere: no hold-open). Hold-open detent beyond 92 deg for the cam doors.

Strategies:
  one: one arm (a, right). Grip the handle low on the bar, back off 5 cm (seal), pull to ~35 deg on the handle,
       then a quick re-grasp onto the free edge at the same height (no lift move), pull the edge round while the
       base reverses and turns (diff-drive planner, full footprint vs door), lift lowered on the way at 6 cm/s.
  two: arm b (left) does the handle and keeps holding while arm a takes the free edge: a hand-over, so the door
       can't escape whatever its self-closing does. Then arm a pulls it round as above; arm b parks.
Seal and floor model as sim/door.py: the seal breaks if it is weaker than the wheels' traction.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "cad"))
import params as P  # noqa: E402
import scene as SC  # noqa: E402
from demo import MINJERK_PEAK, VMAX, minjerk  # noqa: E402

SIM = ROOT / "sim"
OUT = ROOT / "out"
mm = 1e-3
G = 9.81

# ------------------------------------------------------------------ robot geometry (robot frame, metres)
SX = P.SHOULDER_X * mm
ARM_Y = {"a": P.ARM_Y[0] * mm, "b": P.ARM_Y[1] * mm}
L1, L2 = P.L1 * mm, P.L2 * mm
TIP = P.PAD_X * mm                      # pitch axis -> finger pad centre
PITCH_ABOVE_TRAY = (P.L2_ABOVE_TRAY - P.PITCH_DROP) * mm
PITCH_DROP_M = P.PITCH_DROP * mm
REACH_MAX = 0.415                       # shoulder -> wrist, a little short of L1 + L2 (stay off the singularity)
ROBOT_MASS = 15.8                       # calc.py, v0.2 without payload
DRIVE_SHARE = 0.75                      # share of the weight on the drive wheels (calc.py, axle 12 cm forward)
TRAY_X0, TRAY_X1 = P.TRAY_X0 * mm, (P.TRAY_X0 + P.TRAY_DEPTH) * mm
TRAY_HW = P.TRAY_WIDTH / 2 * mm

VM = dict(VMAX)
VM.update({k + "b": VMAX[k] for k in ("j1", "j2", "j3", "j4", "j5")}, gripb=VMAX["grip_l"])
ANGLES = {"base_yaw", "j1", "j2", "j3", "j4", "j5", "j1b", "j2b", "j3b", "j4b", "j5b"}
# actuator name -> state key
ACT = {"base_x": "base_x", "base_y": "base_y", "base_yaw": "base_yaw", "lift": "lift", "drawer": "drawer",
       "j1": "j1", "j2": "j2", "j3": "j3", "j4": "j4", "grip": "grip_l", "j5": "j5",
       "j1b": "j1b", "j2b": "j2b", "j3b": "j3b", "j4b": "j4b", "j5b": "j5b", "gripb": "gripb"}


def robot_boxes(tray_z, drawer=0.0):
    """Top-view boxes of the robot body (robot frame): ((x0, y0), (x1, y1), z0, z1). Arms are separate."""
    bw = (P.TRACK / 2 + P.WHEEL_W / 2) * mm
    return [
        ((P.BASE_X_REAR * mm, -bw), (P.BASE_X_FRONT * mm, bw), 0.0, 0.26),                 # frame, wheels, electronics
        ((P.BASE_X_FRONT * mm, -0.215), (P.FRONT_CASTER_X * mm + 0.05, 0.215), 0.0, 0.15),  # straddle legs + casters
        (((P.MAST_X - 20) * mm, -0.015), ((P.MAST_X + 20) * mm, 0.015), 0.0, 1.86),      # mast (+ lamp above)
        ((0.095, -0.20), (0.16, 0.20), tray_z - 0.06, tray_z + 0.34),                        # carriage, beam, face
        ((-0.06, -0.27), (0.23, 0.27), tray_z - 0.26, tray_z + 0.05),                        # tucked arms, hands hang here
        ((TRAY_X0, -TRAY_HW - 0.03), (TRAY_X1 + drawer, TRAY_HW + 0.03), tray_z - 0.05, tray_z + 0.02),  # tray + arms
    ]


def rob_geoms_xml():
    """Collision shapes for the robot itself (contype 2: they hit the door and fridge, not each other or the floor)."""
    bw = (P.TRACK / 2) * mm
    rw = P.WHEEL_D / 2 * mm
    base = f"""
      <geom class="rob" type="box" pos="{(P.BASE_X_REAR + P.BASE_X_FRONT) / 2 * mm:.4f} 0 0.18" size="{(P.BASE_X_FRONT - P.BASE_X_REAR) / 2 * mm:.4f} 0.19 0.08"/>
      <geom class="rob" type="cylinder" pos="0 {bw:.4f} {rw:.4f}" size="{rw:.4f} 0.0275" euler="90 0 0"/>
      <geom class="rob" type="cylinder" pos="0 {-bw:.4f} {rw:.4f}" size="{rw:.4f} 0.0275" euler="90 0 0"/>
      <geom class="rob" type="box" pos="{(P.BASE_X_FRONT + P.FRONT_CASTER_X) / 2 * mm:.4f} 0.2 0.12" size="{(P.FRONT_CASTER_X - P.BASE_X_FRONT) / 2 * mm:.4f} 0.012 0.02"/>
      <geom class="rob" type="box" pos="{(P.BASE_X_FRONT + P.FRONT_CASTER_X) / 2 * mm:.4f} -0.2 0.12" size="{(P.FRONT_CASTER_X - P.BASE_X_FRONT) / 2 * mm:.4f} 0.012 0.02"/>
      <geom class="rob" type="cylinder" pos="{P.FRONT_CASTER_X * mm:.4f} 0.2 0.065" size="0.05 0.065"/>
      <geom class="rob" type="cylinder" pos="{P.FRONT_CASTER_X * mm:.4f} -0.2 0.065" size="0.05 0.065"/>
      <geom class="rob" type="box" pos="{P.MAST_X * mm:.4f} 0 0.92" size="0.02 0.012 0.90"/>
      <geom class="rob" type="cylinder" pos="{P.MAST_X * mm:.4f} 0 1.86" size="0.115 0.04"/>"""
    carriage = """
        <geom class="rob" type="box" pos="0.108 0 0.07" size="0.008 0.12 0.13"/>
        <geom class="rob" type="box" pos="0.136 0 0.16" size="0.02 0.20 0.012"/>
        <geom class="rob" type="box" pos="0.11 0 0.26" size="0.006 0.04 0.078"/>
        <geom class="rob" type="box" pos="0.25 0.163 -0.025" size="0.135 0.01 0.01"/>
        <geom class="rob" type="box" pos="0.25 -0.163 -0.025" size="0.135 0.01 0.01"/>"""
    # arm links: class "rob_a" / "rob_b" also collide with the other arm's links (not with their own arm or body)
    upper = f"""
          <geom class="rob_X" type="cylinder" pos="0 0 0.03" size="0.04 0.035"/>
          <geom class="rob_X" type="capsule" fromto="0 0 0 {L1:.3f} 0 0" size="0.022"/>"""
    fore = f"""
            <geom class="rob_X" type="capsule" fromto="0 0 0 {L2:.3f} 0 0" size="0.02"/>"""
    wrist = """
              <geom class="rob_X" type="box" pos="0 0 -0.035" size="0.025 0.03 0.045"/>"""
    return base, carriage, upper, fore, wrist


# ------------------------------------------------------------------ doors
FX = 0.72 - P.AXLE_SHIFT * mm       # fridge front, world x
HY = 0.30                           # hinge y (left side, +y)
DOOR_W = 0.60
MID_LX = -0.038                     # door mid-plane, door-local x
EDGE_LY = -0.598                    # free edge, door-local y
H_LOCAL = np.array([-0.11, -0.53])  # handle bar axis, door-local
R_H = float(np.linalg.norm(H_LOCAL))
DOORS = {
    "upper": dict(z0=0.885, z1=1.795, bar=(1.05, 1.30), grip_z=1.08, mass=8.0),   # fridge on top of a freezer
    "full": dict(z0=0.04, z1=1.795, bar=(0.92, 1.30), grip_z=0.95, mass=10.0),    # one tall door
}
CLOSING = {"cam25": 25.0, "cam60": 60.0, "always": 999.0}


def door_xml(kind, mass=None):
    D = DOORS[kind]
    z0, z1 = D["z0"], D["z1"]
    zc, hz = (z0 + z1) / 2, (z1 - z0) / 2
    b0, b1 = D["bar"]
    m = D["mass"] if mass is None else mass
    h, t = 2 * hz, 0.07                                     # plate inertia about its own centre
    ixx, iyy, izz = m * (DOOR_W ** 2 + h * h) / 12, m * (t * t + h * h) / 12, m * (DOOR_W ** 2 + t * t) / 12
    return f"""
    <body name="door" pos="{FX:.4f} {HY} 0">
      <joint name="door" type="hinge" axis="0 0 1" range="-125 0" damping="0.6" frictionloss="0.25" armature="0.05"/>
      <inertial pos="-0.035 -0.30 {zc:.3f}" mass="{m}" diaginertia="{ixx:.3f} {iyy:.3f} {izz:.3f}"/>
      <geom class="dr" type="box" pos="{MID_LX} -0.30 {zc:.4f}" size="0.035 0.298 {hz:.4f}" rgba="0.93 0.93 0.91 1"/>
      <geom class="dr" type="box" pos="-0.0875 -0.53 {b0 + 0.01:.3f}" size="0.0175 0.012 0.012" rgba="0.6 0.6 0.62 1"/>
      <geom class="dr" type="box" pos="-0.0875 -0.53 {b1 - 0.01:.3f}" size="0.0175 0.012 0.012" rgba="0.6 0.6 0.62 1"/>
      <geom class="dr" name="bar" type="cylinder" pos="-0.11 -0.53 {(b0 + b1) / 2:.3f}" size="0.015 {(b1 - b0) / 2:.3f}" rgba="0.6 0.6 0.62 1" friction="1.0 0.02 0.002"/>
    </body>"""


def ghost_arm_b(xml):
    """The one-arm robot (arm a only): arm b stays in the model but collides with nothing and
    is hidden in renders."""
    i0 = xml.index('<body name="upper_arm_b"')
    i1 = xml.index("\n      </body>\n    </body>", i0)
    blk = xml[i0:i1]

    def fix(mt):
        tag = mt.group(0)
        tag = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", tag)
        return tag.replace("<geom ", '<geom contype="0" conaffinity="0" group="5" ', 1)
    blk = re.sub(r"<geom [^>]*/>", fix, blk)
    return xml[:i0] + blk + xml[i1:]


# ------------------------------------------------------------------ the "stump"
# Arm b keeps its shoulder and elbow but loses wrist pitch, roll, gripper and phone. In their place a printed hook
# hangs from the wrist: a shank that passes beside the handle bar and a tip that sits in the gap behind it, so pulling
# the arm back pulls the bar (no squeeze, no friction grip). Hook point = where the bar sits = the full hand's pad
# centre (TIP ahead of the wrist axis, at the pitch-axis height), so the planner's targets stay the same.
#   stumprigid: no wrist servo; the hook is fixed to the forearm at an angle chosen per plan. A 2-DoF arm can't
#               keep it square to a door that turns 25 deg (heading swing +-13-16 deg): it jams (kept as a record)
#   stump:      no wrist servo; the hook sits on a free swivel (bearing + weak return spring) at the wrist, so it
#               lines up with the pull like a leash. The wrist follows the full hand's wrist path, which keeps the
#               swivel-to-bar line on the door normal
#   stumpyaw:   one wrist yaw servo (STS3215) turns the hook
STUMPS = ("stumprigid", "stump", "stumpyaw")
SWIVEL_K = 0.3           # N*m/rad return spring of the swivel hook (a hairpin spring)
SWIVEL_LIM = 0.3         # N*m it can give at most
HOOK_IN = 0.075          # the hook slides in sideways from the hinge side: tip clears the bar by 2.5 cm before
HOOK_TIP_X = 0.0255      # tip centre behind the hook point (bar r 15 mm + 5 mm half thickness + ~5 mm slack)
HOOK_SHANK_Y = 0.025     # shank beside the bar (inner face 1 mm off a 30 mm bar centred on the axis)
HOOK_END_Y = -0.030      # open end of the tip, 15 mm past the bar


def stump_geoms_xml():
    """Hook in the wrist_b frame (x along the heading, z = 0 at the forearm plane). Collides like arm b's links."""
    z = -PITCH_DROP_M
    sx, tx = TIP + 0.026, TIP + HOOK_TIP_X
    # The corner between shank and tip sits on the swivel's axis (the bar centre at y = 0): pulled from the
    # swivel, a bar off that axis walks along the tip, toward the corner if it starts on the shank side, off the
    # open end otherwise (first version, shank at y = 4 cm: the bar escaped at ~15 deg every time).
    parts = [("capsule", f'fromto="0 0 -0.02 0 0 {z:.3f}" size="0.012"'),
             ("capsule", f'fromto="0 0 {z:.3f} 0.10 {HOOK_SHANK_Y} {z:.3f}" size="0.009"'),
             ("capsule", f'fromto="0.10 {HOOK_SHANK_Y} {z:.3f} {sx:.4f} {HOOK_SHANK_Y} {z:.3f}" size="0.009"'),
             ("box", f'pos="{tx:.4f} {(HOOK_SHANK_Y + HOOK_END_Y) / 2:.4f} {z:.3f}" '
                     f'size="0.005 {(HOOK_SHANK_Y - HOOK_END_Y) / 2:.4f} 0.014"')]
    out = []
    for typ, attr in parts:
        out.append(f'<geom class="rob_b" type="{typ}" {attr}/>')
        out.append(f'<geom type="{typ}" {attr} contype="0" conaffinity="0" group="1" mass="0" rgba="0.93 0.55 0.2 1"/>')
    return "\n              " + "\n              ".join(out)


def make_stump(xml, kind):
    """Arm b -> stump: hand, tool and fingers ghosted and nearly massless, hook added to the wrist. For the fixed
    stump the wrist yaw actuator becomes a stiff, strong stand-in for a rigid joint (it is only ever held still)."""
    i0 = xml.index('<body name="hand_b"')
    i1 = xml.index("\n            </body>\n          </body>\n        </body>", i0)
    blk = xml[i0:i1]

    def fix(mt):
        tag = mt.group(0)
        tag = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", tag)
        return tag.replace("<geom ", '<geom contype="0" conaffinity="0" group="5" ', 1)
    blk = re.sub(r"<geom [^>]*/>", fix, blk)
    blk = re.sub(r'mass="[0-9.]+"', 'mass="0.002"', blk)
    xml = xml[:i0] + blk + xml[i1:]
    j = xml.index('<joint name="j3b"')
    j = xml.index("/>", j) + 2
    xml = xml[:j] + stump_geoms_xml() + xml[j:]
    if kind == "stumprigid":
        xml = re.sub(r'<position name="j3b" joint="j3b"[^>]*/>',
                     '<position name="j3b" joint="j3b" kp="5000" kv="50" forcerange="-200 200"/>', xml)
    elif kind == "stump":
        xml = re.sub(r'<position name="j3b" joint="j3b"[^>]*/>',
                     f'<position name="j3b" joint="j3b" kp="{SWIVEL_K}" kv="{SWIVEL_K / 30:.4f}" '
                     f'forcerange="{-SWIVEL_LIM} {SWIVEL_LIM}"/>', xml)
    return xml


def build_model(kind, traction, arms=2, door_mass=None, tag="", stump=None):
    xml = (SIM / "tischlein.xml").read_text()
    xml = xml.replace('<default class="pad">',
                      '<default class="rob"><geom contype="2" conaffinity="0" group="3" rgba="0.2 0.8 0.3 0.3"/></default>\n'
                      '    <default class="rob_a"><geom contype="2" conaffinity="8" group="3" rgba="0.2 0.8 0.3 0.3"/></default>\n'
                      '    <default class="rob_b"><geom contype="10" conaffinity="0" group="3" rgba="0.2 0.8 0.3 0.3"/></default>\n'
                      '    <default class="dr"><geom contype="1" conaffinity="3"/></default>\n    <default class="pad">')
    xml = re.sub(r'<geom name="fr_door"[^>]*/>', door_xml(kind, door_mass), xml)
    for nm in ("fr_left", "fr_right", "fr_back", "fr_top", "fr_freezer"):
        xml = re.sub(rf'(<geom name="{nm}" )', r'\1conaffinity="3" ', xml)
    if kind == "full":   # one tall door: no freezer drawer front sticking out below it
        xml = re.sub(r'<geom name="fr_freezer"[^>]*/>', "", xml)
        xml = xml.replace("</worldbody>", f'<geom name="fr_low" type="box" conaffinity="3" pos="{FX + 0.33:.3f} 0 0.44" '
                          f'size="0.32 0.3 0.44" rgba="0.93 0.93 0.91 1"/>\n  </worldbody>')
    for nm in ("meal_box", "bottle"):
        xml = re.sub(rf'<body name="{nm}".*?</body>', "", xml, flags=re.S)
    base, carriage, upper, fore, wrist = rob_geoms_xml()
    xml = xml.replace('      <body name="carriage"', base + '\n      <body name="carriage"', 1)
    xml = xml.replace('        <body name="drawer"', carriage + '\n        <body name="drawer"', 1)
    for sfx, arm in (("", "a"), ("_b", "b")):
        cls = lambda x: x.replace("rob_X", f"rob_{arm}")
        xml = xml.replace(f'          <body name="forearm{sfx}"', cls(upper) + f'\n          <body name="forearm{sfx}"', 1)
        xml = xml.replace(f'            <body name="wrist{sfx}"', cls(fore) + f'\n            <body name="wrist{sfx}"', 1)
        xml = xml.replace(f'              <body name="hand{sfx}"', cls(wrist) + f'\n              <body name="hand{sfx}"', 1)
    for ax in ("base_x", "base_y"):
        xml = xml.replace(f'<position name="{ax}" joint="{ax}" kp="40000" kv="3000" forcerange="-1500 1500"/>',
                          f'<position name="{ax}" joint="{ax}" kp="40000" kv="3000" forcerange="{-traction:.1f} {traction:.1f}"/>')
    if arms == 1:
        xml = ghost_arm_b(xml)
    if stump:
        xml = make_stump(xml, stump)
        tag = f"_{stump}{tag}"
    path = SIM / f"_fridge_{kind}_{arms}{tag}.xml"
    path.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(path))
    if tag:                       # per-process copies (sim/door_policy.py workers) are only needed while loading
        path.unlink()
    return m


def door_pt(phi, lx, ly):
    c, s = math.cos(-phi), math.sin(-phi)
    return np.array([FX, HY]) + np.array([c * lx - s * ly, s * lx + c * ly])


def handle_world(phi):
    return door_pt(phi, *H_LOCAL)


def edge_grip(phi, depth=0.035):
    """Pad target on the free edge and the hand heading (along the door, toward the hinge)."""
    tip = door_pt(phi, MID_LX, EDGE_LY + depth)
    d = door_pt(phi, MID_LX, 0.0) - door_pt(phi, MID_LX, -1.0)
    return tip, math.atan2(d[1], d[0])


def door_torque(phi, f_break, closing):
    """Seal (fades over 8 mm of handle gap) + self-closing + hold-open detent. Positive closes the door."""
    gap = phi * R_H
    seal = f_break * R_H * max(0.0, 1.0 - gap / 0.008)
    lim = math.radians(CLOSING[closing])
    if phi < lim:
        close = 1.0 * min(1.0, phi / math.radians(5))
    elif closing == "always" or phi < math.radians(92):
        close = 0.0
    else:
        close = -0.5
    if closing == "always":
        close = 1.2 * min(1.0, phi / math.radians(5))          # self-closing hinges: no hold-open anywhere
    return seal + close


# ------------------------------------------------------------------ kinematics
def ik(wx, wy, heading, arm="a"):
    """Wrist-axis target in the robot frame -> (j1, j2, j3) deg. Arm a elbow-left, arm b elbow-right."""
    x, y = wx - SX, wy - ARM_Y[arm]
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    q2 = math.acos(c2) * (1 if arm == "a" else -1)
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    j1, j2 = math.degrees(q1), math.degrees(q2)
    j3 = (heading - j1 - j2 + 180) % 360 - 180
    return j1, j2, j3


def to_robot(base, p):
    bx, by, yaw = base
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.asarray(p) - np.array([bx, by])
    return np.array([c * r[0] + s * r[1], -s * r[0] + c * r[1]])


def arm_for(base, tip_world, heading_world, grip, arm="a", pitch=0.0):
    """Joint targets (deg, m) that put this arm's pad centre on a world point with a world heading."""
    t = to_robot(base, tip_world)
    h = heading_world - base[2]
    reach = TIP * math.cos(math.radians(pitch))       # pitched hands reach less far (pads drop TIP * sin)
    wx, wy = t[0] - reach * math.cos(h), t[1] - reach * math.sin(h)
    j1, j2, j3 = ik(wx, wy, math.degrees(h), arm)
    k = "" if arm == "a" else "b"
    return {f"j1{k}": j1, f"j2{k}": j2, f"j3{k}": j3, f"j4{k}": pitch, f"j5{k}": 0.0,
            ("grip_l" if arm == "a" else "gripb"): grip}


def hook_ik(base, tip_world, c_deg):
    """Fixed stump: shoulder + elbow put the hook point on a world point; the heading is whatever results.
    Returns (j1b, j2b, heading in the robot frame), degrees. The forearm + fixed hook act as one rigid link."""
    t = to_robot(base, tip_world)
    x, y = t[0] - SX, t[1] - ARM_Y["b"]
    c = math.radians(c_deg)
    vx, vy = L2 + TIP * math.cos(c), TIP * math.sin(c)
    le, dl = math.hypot(vx, vy), math.atan2(vy, vx)
    c2 = (x * x + y * y - L1 * L1 - le * le) / (2 * L1 * le)
    p2 = -math.acos(max(-1.0, min(1.0, c2)))                     # arm b: elbow-right, as in ik()
    q1 = math.atan2(y, x) - math.atan2(le * math.sin(p2), L1 + le * math.cos(p2))
    q2 = p2 - dl
    j1, j2 = math.degrees(q1), math.degrees(q2)
    return j1, j2, _wrap(j1 + j2 + c_deg)


def hook_c_for(base, tip_world, heading_world):
    """Fixed hook angle that gives the wanted heading at this one pose."""
    want = math.degrees(heading_world - base[2])
    c = 0.0
    for _ in range(30):
        c = _wrap(c + _wrap(want - hook_ik(base, tip_world, c)[2]))
    return c


def hold_for(strategy, base, tip_world, heading_world, grip, arm, hook_c=0.0):
    """Joint targets for the arm that holds the handle: a gripping hand, or a stump's hook."""
    if strategy == "stumprigid":
        j1, j2, _ = hook_ik(base, tip_world, hook_c)
        return {"j1b": j1, "j2b": j2, "j3b": hook_c, "j4b": 0.0, "j5b": 0.0, "gripb": 0.0}
    st = arm_for(base, tip_world, heading_world, grip, arm)
    if strategy == "stump":
        st["j3b"] = hook_c                      # the swivel's spring rest angle; the joint itself floats
    if strategy in STUMPS:
        st.update(j4b=0.0, j5b=0.0, gripb=0.0)
    return st


def hook_heading_err(st, base, heading_world):
    """deg between the hook's actual heading and the heading wanted (perpendicular to the door)."""
    return abs(_wrap(st["j1b"] + st["j2b"] + st["j3b"] - math.degrees(heading_world - base[2])))


def wrist_dist(base, tip_world, heading_world, arm):
    t = to_robot(base, tip_world)
    h = heading_world - base[2]
    w = t - TIP * np.array([math.cos(h), math.sin(h)])
    return float(np.linalg.norm(w - np.array([SX, ARM_Y[arm]])))


def _wrap(a):
    return (a + 180) % 360 - 180


# Idle arm: tucked along its side of the carriage, upper arm pointing back, forearm forward, hand hanging down
# at x ~0.19 m. cad/scene.py's park points the forearm back from an elbow 0.35 m ahead of the shoulder; that
# blocks the other arm's work space in front (both arms move in the same horizontal planes), so the door sim tucks.
# The iPhone sticks out 11.5 cm to the hand's right (tool -y), so the hand's heading keeps it outboard: forward for
# the right arm, backward for the left one (otherwise the hanging phone lands on the tray).
PARK = {"a": {**{k: float(v) for k, v in SC.TUCK_A.items() if k.startswith("j")}, "grip_l": SC.TUCK_A["grip"] * mm},
        "b": {**{k: float(v) for k, v in SC.TUCK_B.items() if k.startswith("j")}, "gripb": SC.TUCK_B["gripb"] * mm}}
# on the way between work and the tuck: hand level, pointing outboard, beside the carriage
SIDE = {"a": dict(wrist=(0.22, -0.30), heading=-90.0), "b": dict(wrist=(0.22, 0.30), heading=90.0)}


def side_pose(arm, grip=0.004):
    k = "" if arm == "a" else "b"
    w = SIDE[arm]
    j1, j2, j3 = ik(*w["wrist"], w["heading"], arm)
    return {f"j1{k}": j1, f"j2{k}": j2, f"j3{k}": j3, f"j4{k}": 0.0, f"j5{k}": 0.0,
            ("grip_l" if arm == "a" else "gripb"): grip}


def tuck(T, arm, mark):
    """Move an arm from its work pose to the tuck: out to the side, fold back, hand down."""
    k = "" if arm == "a" else "b"
    T.to(1.5, f"{mark}: out to the side", **side_pose(arm))
    T.to(1.5, f"{mark}: fold", **{kk: v for kk, v in PARK[arm].items() if not kk.startswith("j4")}, **{f"j4{k}": 0.0})
    T.to(1.0, f"{mark}: hand down", **{f"j4{k}": 90.0})


def stump_points(j1, j2, j3):
    """Top view, robot frame: shoulder, elbow, wrist, along the shank, hook corner, tip end of the stump."""
    q1, q2 = math.radians(j1), math.radians(j2)
    h = math.radians(j1 + j2 + j3)
    u, n = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    s = np.array([SX, ARM_Y["b"]])
    e = s + L1 * np.array([math.cos(q1), math.sin(q1)])
    w = e + L2 * np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
    return [s, e, w, w + 0.10 * u + HOOK_SHANK_Y * n, w + (TIP + 0.026) * u + HOOK_SHANK_Y * n,
            w + (TIP + HOOK_TIP_X) * u + HOOK_END_Y * n]


_STUMP_TUCK = {}


def stump_tuck_pose(strategy, hook_c):
    """Fold the stump inside the carriage's left half (where the tucked hand hangs), clear of arm a's tuck."""
    key = (strategy, round(hook_c, 1))
    if key not in _STUMP_TUCK:
        A = arm_polyline({**PARK["a"], "j4": 0.0}, "a")
        best = None
        j3s = range(-180, 180, 15) if strategy == "stumpyaw" else [hook_c]
        for j1 in range(0, 151, 5):               # arm b's own side and elbow branch, as TUCK_B (no flip on the way)
            for j2 in range(-155, 1, 5):
                for j3 in j3s:
                    pts = stump_points(j1, j2, j3)
                    out = sum(max(0, -0.06 - p[0]) + max(0, p[0] - 0.23) + max(0, 0.03 - p[1]) + max(0, p[1] - 0.27)
                              for p in pts[1:])
                    gap = min(seg_dist(pts[i], pts[i + 1], A[k], A[k + 1]) for i in range(5) for k in range(3))
                    cost = out + 2 * max(0.0, 0.05 - gap)
                    if best is None or cost < best[0]:
                        best = (cost, dict(j1b=float(j1), j2b=float(j2), j3b=float(j3)))
        _STUMP_TUCK[key] = best
    return _STUMP_TUCK[key]


def stump_tuck(T, strategy, hook_c):
    cost, pose = stump_tuck_pose(strategy, hook_c)
    j1, j2, j3 = ik(*SIDE["b"]["wrist"], SIDE["b"]["heading"], "b")   # out to the left side first, as tuck() does
    T.to(1.5, "stump out to the side", j1b=j1, j2b=j2, j3b=(j3 if strategy == "stumpyaw" else hook_c))
    T.to(1.5, "stump folds", **pose, j4b=0.0, j5b=0.0)
    return cost


def arm_polyline(st, arm):
    """Top view, robot frame: shoulder, elbow, wrist, pad centre of one arm for a joint state (deg)."""
    k = "" if arm == "a" else "b"
    q1, q2, q3 = (math.radians(st[f"j{i}{k}"]) for i in (1, 2, 3))
    s = np.array([SX, ARM_Y[arm]])
    e = s + L1 * np.array([math.cos(q1), math.sin(q1)])
    w = e + L2 * np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
    reach = TIP * math.cos(math.radians(st[f"j4{k}"]))
    t = w + reach * np.array([math.cos(q1 + q2 + q3), math.sin(q1 + q2 + q3)])
    return [s, e, w, t]


def seg_dist(p, q, r, s):
    """Min distance between 2D segments pq and rs."""
    def pt_seg(a, b, c):
        ab = c - b
        t = np.clip(np.dot(a - b, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        return np.linalg.norm(a - (b + t * ab))
    d1, d2 = q - p, s - r
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) > 1e-12:
        t = ((r - p)[0] * d2[1] - (r - p)[1] * d2[0]) / den
        u = ((r - p)[0] * d1[1] - (r - p)[1] * d1[0]) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    return min(pt_seg(p, r, s), pt_seg(q, r, s), pt_seg(r, p, q), pt_seg(s, p, q))


def arms_gap(st):
    """Smallest top-view distance between the two arms (links + hand), in m. Both arms work in the same planes."""
    A, B = arm_polyline(st, "a"), arm_polyline(st, "b")
    return min(seg_dist(A[i], A[i + 1], B[j], B[j + 1]) for i in range(3) for j in range(3))


# ------------------------------------------------------------------ planning (vectorised over candidate poses)
def door_points(phi, n=31):
    a, b = door_pt(phi, MID_LX, -0.002), door_pt(phi, MID_LX, EDGE_LY)
    return a[None, :] + (b - a)[None, :] * np.linspace(0, 1, n)[:, None]


def clearance_v(poses, phi, tray_z, kind):
    """Min distance (m) between the door panel and the robot body for many base poses (N, 3). <0 = overlap.
    Also keeps the robot body out of the fridge cabinet (fridge front; freezer drawer 7 cm proud below 0.88 m)."""
    D = DOORS[kind]
    poses = np.atleast_2d(poses)
    bx, by, yaw = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    c, s = np.cos(yaw), np.sin(yaw)
    pts = door_points(phi)                                   # (M, 2)
    rx = pts[None, :, 0] - bx
    ry = pts[None, :, 1] - by
    qx, qy = c * rx + s * ry, -s * rx + c * ry               # (N, M) door points in the robot frame
    best = np.full(len(poses), 9.0)
    for (x0, y0), (x1, y1), z0, z1 in robot_boxes(tray_z):
        cx = np.array([x0, x1, x1, x0])
        cy = np.array([y0, y0, y1, y1])
        wx = bx + c * cx[None, :] - s * cy[None, :]            # (N, 4) corners in the world
        wy = by + s * cx[None, :] + c * cy[None, :]
        front = FX - (0.075 if (kind == "upper" and z0 < 0.88) else 0.005)
        inside_w = np.abs(wy) < DOOR_W / 2 + 0.03
        best = np.minimum(best, np.where(inside_w, front - wx, 9.0).min(axis=1))
        if z1 < D["z0"] or z0 > D["z1"]:
            continue
        dx = np.maximum(np.maximum(x0 - qx, 0), qx - x1)
        dy = np.maximum(np.maximum(y0 - qy, 0), qy - y1)
        inside = (dx == 0) & (dy == 0)
        dist = np.where(inside, -0.01, np.hypot(dx, dy)) - 0.036
        best = np.minimum(best, dist.min(axis=1))
    return best


def clearance(base, phi, tray_z, kind):
    return float(clearance_v(np.array([base]), phi, tray_z, kind)[0])


def wrist_dist_v(poses, tip_world, heading_world, arm, reach=TIP):
    bx, by, yaw = poses[:, 0], poses[:, 1], poses[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    rx, ry = tip_world[0] - bx, tip_world[1] - by
    tx, ty = c * rx + s * ry, -s * rx + c * ry
    h = heading_world - yaw
    wx, wy = tx - reach * np.cos(h), ty - reach * np.sin(h)
    return np.hypot(wx - SX, wy - ARM_Y[arm])


def plan_swing(start, phis, tray_zs, kind, arm="a", reach_goal=0.33, need=0.02, reach=TIP):
    """Greedy diff-drive base path: per door angle choose (ds, dyaw) so the holding arm stays within reach and the
    robot clear of the door and fridge. Returns poses, feasible, worst (reach margin, clearance)."""
    B, pose = [], np.array(start, dtype=float)
    feasible, worst_r, worst_c = True, 9.0, 9.0
    DS, DY = np.meshgrid(np.linspace(-0.05, 0.05, 21), np.radians(np.linspace(-10, 10, 21)))
    DS, DY = DS.ravel(), DY.ravel()
    for phi, tz in zip(phis, tray_zs):
        tip, hd = edge_grip(phi)
        ym = pose[2] + DY / 2
        cand = np.stack([pose[0] + DS * np.cos(ym), pose[1] + DS * np.sin(ym), pose[2] + DY], axis=1)
        r = wrist_dist_v(cand, tip, hd, arm, reach)
        cl = clearance_v(cand, phi, tz, kind)
        pen = np.maximum(0, r - REACH_MAX) * 400 + np.maximum(0, need - cl) * 400 + np.maximum(0, 0.16 - r) * 400
        cost = pen + 4 * DS ** 2 + 0.3 * DY ** 2 + 2 * (r - reach_goal) ** 2
        cost[(r > REACH_MAX + 0.03) | (r < 0.12)] = np.inf
        i = int(np.argmin(cost))
        if not np.isfinite(cost[i]):
            feasible = False
            B.append(pose.copy())
            continue
        pose = cand[i]
        B.append(pose.copy())
        worst_r, worst_c = min(worst_r, REACH_MAX - r[i]), min(worst_c, cl[i])
        if REACH_MAX - r[i] < -1e-3 or cl[i] < need - 2e-3:
            feasible = False
    return np.array(B), feasible, (worst_r, worst_c)


MEAS = {"phi": 0.0}     # door angle as the robot perceives it (camera), set every control step
# one-arm re-grasp path: pull the hand back, swing the wrist round the front at this radius, turn onto the door
# line this far beyond the free edge (tuned with a clearance check: >= 3 cm from the door everywhere)
REGRASP = dict(back=0.15, r_front=0.14, pre=0.14)


class Timeline:
    """Segments of joint-state targets; durations stretched to the real joint speeds."""

    def __init__(self, start):
        self.state = dict(start)
        self.segs = []          # (t0, t1, fn, mark, raw)  fn(s) -> state; raw: no min-jerk shaping
        self.t = 0.0

    def need(self, fn, n=24):
        """Duration (s) at which the fastest part of the path just reaches the joint speed limits, for a path
        traversed at constant speed in s. Callers scale it by the min-jerk peak factor."""
        pts = [fn(k / n) for k in range(n + 1)]
        worst = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            worst = max(worst, max((abs(b[k] - a[k]) / v for k, v in VM.items() if k in a and k in b), default=0.0))
        return worst * n

    def add(self, dur, fn, mark, raw=False, nominal=None):
        """fn(s) -> dict of targets for s in [0, 1]. nominal: same path for the speed check when fn is closed-loop."""
        frm = dict(self.state)
        f = (lambda s, fn=fn, frm=frm: {**frm, **fn(s)})
        chk = f if nominal is None else (lambda s, fn=nominal, frm=frm: {**frm, **fn(s)})
        need = self.need(chk) * (1.0 if raw else MINJERK_PEAK)
        dur = max(dur, need)
        self.segs.append((self.t, self.t + dur, f, mark, raw))
        self.t += dur
        self.state = f(1.0)
        return dur

    def to(self, dur, mark, **targets):
        frm = dict(self.state)
        return self.add(dur, lambda s: {k: frm[k] + (v - frm[k]) * s for k, v in targets.items()}, mark)

    def at(self, t):
        for t0, t1, f, mark, raw in self.segs:
            if t0 <= t < t1:
                s = (t - t0) / (t1 - t0)
                return f(s if raw else minjerk(s))
        return self.segs[-1][2](1.0)

    def marks(self):
        return [(t0, mk) for t0, _, _, mk, _ in self.segs]


def base_state(base):
    return dict(base_x=base[0], base_y=base[1], base_yaw=math.degrees(base[2]))


def plan(strategy, kind, closing, by0=-0.08, phi_pull=None, n=26, phi_end=102.0, handover2=False, dx0=0.0, yaw0=0.0,
         hook_c=None):
    """Build the whole timeline for one strategy/door. Returns (Timeline, info)."""
    D = DOORS[kind]
    stump = strategy in STUMPS
    grip_z = D["grip_z"] if not stump else sum(D["bar"]) / 2   # the hook goes mid-bar, clear of the standoffs
    z_handle = grip_z - PITCH_ABOVE_TRAY            # tray height with the pads at the grip height
    if kind == "upper":
        z_swing = D["z0"] + 0.030 - PITCH_ABOVE_TRAY  # tray rides 1 cm below the door, fingers on its bottom 3 cm
    else:
        z_swing = z_handle                            # tall door: nothing passes underneath, stay at grip height
    h_arm = "a" if strategy == "one" else "b"         # the arm on the handle
    e_arm = "a"                                       # the arm on the free edge
    g_open, g_shut, g_edge = 0.040, -0.008, 0.030     # edge is 70 mm: pads at ~35 mm stay loose (a hook, not a squeeze)
    kg_h = "grip_l" if h_arm == "a" else "gripb"
    h0 = handle_world(0.0)
    # base square to the fridge; the handle arm's wrist ~0.30 m from its shoulder
    w0 = h0 - np.array([TIP, 0.0])
    bx0 = w0[0] - SX - math.sqrt(max(0.0, 0.30 ** 2 - (w0[1] - by0 - ARM_Y[h_arm]) ** 2))
    base0 = (bx0 + dx0, by0, yaw0)          # dx0, yaw0: how far off the robot actually parked (learning sim)
    cy0, sy0 = math.cos(yaw0), math.sin(yaw0)
    pre = h0 + np.array([-0.05, 0.0])
    if hook_c is None:
        hook_c = {"stumprigid": lambda: hook_c_for(base0, h0, 0.0),
                  "stump": lambda: arm_for(base0, h0, 0.0, 0.0, "b")["j3b"]}.get(strategy, lambda: 0.0)()
    hold = lambda b, p, ph: hold_for(strategy, b, p, -ph, g_shut, h_arm, hook_c)

    def held(b, ph):
        return hold(b, handle_world(ph), ph)
    if not stump:
        start = {**base_state(base0), "lift": z_handle, "drawer": 0.0, **PARK["a"], **PARK["b"],
                 **arm_for(base0, pre, 0.0, g_open, h_arm)}
        T = Timeline(start)
        T.add(0.5, lambda s: {}, "hand in front of the handle (arrived with the base)")
        T.add(1.0, lambda s: arm_for(base0, pre + [0.05 * s, 0], 0.0, g_open, h_arm), "reach the bar")
        T.to(0.6, "grip", **{kg_h: g_shut})
    else:
        # hook in from the hinge side: in front of the door left of the bar, forward so the tip is behind the bar's
        # plane, then sideways until the bar sits in the hook's corner (door-local points)
        L_pre = (H_LOCAL[0] - 0.07, H_LOCAL[1] + HOOK_IN)
        L_in = (H_LOCAL[0], H_LOCAL[1] + HOOK_IN)
        start = {**base_state(base0), "lift": z_handle, "drawer": 0.0, **PARK["a"], **PARK["b"],
                 **hold(base0, door_pt(0.0, *L_pre), 0.0)}
        T = Timeline(start)
        T.add(0.5, lambda s: {}, "hook in front of the door (arrived with the base)")
        pa, pb_ = door_pt(0.0, *L_pre), door_pt(0.0, *L_in)
        T.add(1.0, lambda s: hold(base0, pa + (pb_ - pa) * s, 0.0), "hook past the bar")
        T.add(1.0, lambda s: hold(base0, pb_ + (h0 - pb_) * s, 0.0), "hook slides behind the bar")
    back = 0.05
    p_back = back / R_H

    def seal(s):
        phi = p_back * s
        b = (base0[0] - back * s * cy0, base0[1] - back * s * sy0, yaw0)
        return {**base_state(b), **held(b, phi)}
    T.add(1.5, seal, "base backs off 5 cm: seal breaks")
    base1 = (base0[0] - back * cy0, base0[1] - back * sy0, yaw0)
    if phi_pull is None:
        phi_pull = 35.0 if strategy == "one" else 25.0
    pp = math.radians(phi_pull)
    # pull on the handle: arm, plus base reversing if the arm runs out of reach
    phh = np.radians(np.linspace(math.degrees(p_back), phi_pull, 8))
    bases = [base1]
    for ph in phh[1:]:
        b = bases[-1]
        for _ in range(80):          # the handle comes toward the robot: back off to keep the arm comfortable
            r = wrist_dist(b, handle_world(ph), -ph, h_arm)
            if r < 0.24:
                b = (b[0] - 0.01 * cy0, b[1] - 0.01 * sy0, yaw0)
            elif r > REACH_MAX - 0.01:
                b = (b[0] + 0.01 * cy0, b[1] + 0.01 * sy0, yaw0)
            else:
                break
        bases.append(b)
    BH = np.array(bases)

    def pull(s):
        k = s * (len(phh) - 1)
        i = min(int(k), len(phh) - 2)
        f = k - i
        b = tuple(BH[i] + (BH[i + 1] - BH[i]) * f)
        ph = phh[i] + (phh[i + 1] - phh[i]) * f
        return {**base_state(b), **held(b, ph)}
    T.add(2.0, pull, f"pull the handle to {phi_pull:.0f} deg")
    herr, herr_s = 0.0, []                           # rigid stump: hook heading vs the door normal while hooked
    if strategy == "stumprigid":
        for fn, phx in ((seal, lambda u: p_back * u), (pull, None)):
            for u in np.linspace(0, 1, 21):
                st = fn(u)
                b = (st["base_x"], st["base_y"], math.radians(st["base_yaw"]))
                ph = phx(u) if phx else None
                if ph is None:
                    k = u * (len(phh) - 1)
                    i = min(int(k), len(phh) - 2)
                    ph = phh[i] + (phh[i + 1] - phh[i]) * (k - i)
                herr = max(herr, hook_heading_err(st, b, -ph))
                herr_s.append(_wrap(st["j1b"] + st["j2b"] + st["j3b"] - math.degrees(-ph - b[2])))
    base2 = tuple(BH[-1])
    h1 = handle_world(pp)
    e1, hd1 = edge_grip(pp)
    if strategy == "one":
        # quick re-grasp: open, straight off the bar, round the edge at 10 cm from the door's outer face, then onto
        # the edge along the door. Closed loop: the targets follow the door angle the camera sees, because a
        # self-closing door moves as soon as it is released.
        T.to(0.25, "release the handle", grip_l=g_open)

        def pad_world(w_rob, h_rob):
            """Pad centre (world) and world heading for a wrist position and heading given in the robot frame."""
            c, s_ = math.cos(base2[2]), math.sin(base2[2])
            pr = np.asarray(w_rob) + TIP * np.array([math.cos(math.radians(h_rob)), math.sin(math.radians(h_rob))])
            return np.array(base2[:2]) + np.array([c * pr[0] - s_ * pr[1], s_ * pr[0] + c * pr[1]]), \
                math.radians(h_rob) + base2[2]

        # The hand starts left of its shoulder (the handle moved left as the door opened) and must end right of it,
        # pointing along the door: the wrist swings round the front of the shoulder with the hand pointing right
        # (clear of the door), then the hand turns onto the door line beyond the edge. No path goes through the
        # shoulder (singular: L1 = L2).
        # first the base reverses 4 cm with the arm frozen: the open hand slides straight off the bar. Then it
        # turns/drives to the pose (searched) from which the free edge is comfortably in reach and the robot body
        # keeps 3 cm from the door; the hand stays where it is relative to the robot.
        hx0 = handle_world(pp)
        w_off = to_robot(base2, hx0) - TIP * np.array([math.cos(-pp - base2[2]), math.sin(-pp - base2[2])])
        base2 = (base2[0] - 0.04 * math.cos(base2[2]), base2[1] - 0.04 * math.sin(base2[2]), base2[2])
        T.to(0.6, "base reverses 4 cm: hand off the bar", **base_state(base2))
        sh = np.array([SX, ARM_Y["a"]])
        e_pp, hd_pp = edge_grip(pp)
        best3 = None
        for dd in np.linspace(-0.08, 0.10, 19):
            for dyw in np.radians(np.linspace(-20, 20, 21)):
                ym = base2[2] + dyw / 2
                b3 = (base2[0] + dd * math.cos(ym), base2[1] + dd * math.sin(ym), base2[2] + dyw)
                pre_edge = e_pp - 0.08 * np.array([math.cos(hd_pp), math.sin(hd_pp)])
                r_pre = wrist_dist(b3, pre_edge, hd_pp, "a")
                r_e = wrist_dist(b3, e_pp, hd_pp, "a")
                cl = clearance(b3, pp, z_handle, kind)
                # the hand, still where it left the bar (robot frame), must stay clear of the door as the base moves
                c, s_ = math.cos(b3[2]), math.sin(b3[2])
                hand = np.array(b3[:2]) + np.array([c * (w_off[0] + 0.12) - s_ * w_off[1], s_ * (w_off[0] + 0.12) + c * w_off[1]])
                d_hand = min(np.linalg.norm(hand - q) for q in door_points(pp, 15))
                cost = (r_pre - 0.33) ** 2 + 0.5 * (r_e - 0.33) ** 2 + 0.02 * dd ** 2 + 0.002 * dyw ** 2
                cost += 50 * max(0.0, 0.03 - cl) + 50 * max(0.0, r_e - (REACH_MAX - 0.02)) + 50 * max(0.0, 0.08 - d_hand)
                if best3 is None or cost < best3[0]:
                    best3 = (cost, b3)
        base2 = best3[1]
        T.to(1.0, "base to the re-grasp pose", **base_state(base2))
        w_front = sh + REGRASP["r_front"] * np.array([math.cos(math.radians(-15)), math.sin(math.radians(-15))])

        def regrasp(s, meas=True):
            phx = MEAS["phi"] if meas else pp
            e, hde = edge_grip(phx)
            pre_e = e - REGRASP["pre"] * np.array([math.cos(hde), math.sin(hde)])
            hde_r = math.degrees(hde - base2[2])
            w_pre = to_robot(base2, pre_e) - TIP * np.array([math.cos(math.radians(hde_r)), math.sin(math.radians(hde_r))])
            h0r = math.degrees(-pp - base2[2])
            bounds = np.array([0.0, 0.15, 0.55, 0.78, 1.0])
            k = min(int(np.searchsorted(bounds, s, side="right") - 1), 3)
            u = (s - bounds[k]) / (bounds[k + 1] - bounds[k])
            hdir = np.array([math.cos(math.radians(h0r)), math.sin(math.radians(h0r))])
            w_back = w_off - REGRASP["back"] * hdir          # hand pulled straight back first
            if k == 0:
                return arm_for(base2, *pad_world(w_off + (w_back - w_off) * u, h0r), g_open, "a")
            if k == 1:      # wrist round the front of the shoulder, hand turning to point right
                a0 = math.atan2(*(w_back - sh)[::-1])
                a1 = math.radians(-15)
                r0, r1 = np.linalg.norm(w_back - sh), REGRASP["r_front"]
                ang, rad = a0 + (a1 - a0) * u, r0 + (r1 - r0) * u
                w = sh + rad * np.array([math.cos(ang), math.sin(ang)])
                return arm_for(base2, *pad_world(w, h0r + (-90.0 - h0r) * u), g_open, "a")
            if k == 2:      # wrist to the pre-grasp spot, hand still pointing right
                w = w_front + (w_pre - w_front) * u
                return arm_for(base2, *pad_world(w, -90.0), g_open, "a")
            # hand turns onto the door line (pads beyond the free edge)
            return arm_for(base2, *pad_world(w_pre, -90.0 + (hde_r + 90.0) * u), g_open, "a")
        T.add(1.0, regrasp, "re-grasp: hand round to the free edge", nominal=lambda s: regrasp(s, meas=False))
        T.add(0.6, lambda s: arm_for(base2, *(lambda e_h: (e_h[0] - REGRASP["pre"] * (1 - s) * np.array([math.cos(e_h[1]), math.sin(e_h[1])]), e_h[1]))(edge_grip(MEAS["phi"])), g_open, "a"),
              "slide onto the edge", nominal=lambda s: arm_for(base2, edge_grip(pp)[0] - REGRASP["pre"] * (1 - s) * np.array([math.cos(edge_grip(pp)[1]), math.sin(edge_grip(pp)[1])]), edge_grip(pp)[1], g_open, "a"))
        T.add(0.25, lambda s: arm_for(base2, *edge_grip(MEAS["phi"]), g_open + (g_edge - g_open) * s, "a"),
              "hook the edge", nominal=lambda s: arm_for(base2, *edge_grip(pp), g_open + (g_edge - g_open) * s, "a"))
    else:
        # hand-over: arm a takes the edge while arm b still holds the handle; no hurry
        pre_e = e1 - 0.08 * np.array([math.cos(hd1), math.sin(hd1)])
        T.to(1.5, "right hand to the free edge", **arm_for(base2, pre_e, hd1, g_open, "a"))
        T.add(1.0, lambda s: arm_for(base2, pre_e + (e1 - pre_e) * s, hd1, g_open, "a"), "around the edge")
        T.to(0.4, "hook the edge", grip_l=g_edge)
        if stump:
            # unhook: slide toward the hinge until the tip clears the bar, then back off the door
            hk, hin = handle_world(pp), door_pt(pp, *L_in)
            hout = door_pt(pp, H_LOCAL[0] - 0.08, H_LOCAL[1] + HOOK_IN)
            T.add(0.8, lambda s: hold(base2, hk + (hin - hk) * s, pp), "left hook slides out from behind the bar")
            T.add(0.8, lambda s: hold(base2, hin + (hout - hin) * s, pp), "left hook back off the door")
            stump_tuck(T, strategy, hook_c)
        else:
            T.to(0.4, "left hand lets go of the handle", gripb=g_open)
            hb = handle_world(pp)
            away = hb + 0.08 * np.array([-math.cos(pp), math.sin(pp)])
            T.add(0.8, lambda s: arm_for(base2, hb + (away - hb) * s, -pp, g_open, "b"), "left hand back")
            tuck(T, "b", "left arm tucks")
    def swing_part(base_s, phi0, phi1, z0, z1, arm, pitch, mark, sec_per_deg=0.18):
        """Plan the base for the edge pull phi0 -> phi1 (deg) with one arm hooked, add it to the timeline."""
        nn = max(4, int(round((phi1 - phi0) / 3)))
        ph = np.radians(np.linspace(phi0, phi1, nn + 1)[1:])
        dur = sec_per_deg * (phi1 - phi0)
        t_lift = MINJERK_PEAK * abs(z1 - z0) / VM["lift"]
        fr = min(1.0, t_lift / dur) if abs(z1 - z0) > 1e-4 else 0.0
        zs = [z0 + (z1 - z0) * (min(1.0, ((k + 1) / nn) / fr) if fr > 0 else 1.0) for k in range(nn)]
        reach = TIP * math.cos(math.radians(pitch))
        B, ok_, (wr_, wc_) = plan_swing(np.array(base_s), ph, zs, kind, arm, reach=reach)
        Bf = np.vstack([np.array(base_s), B])
        phf = np.concatenate([[math.radians(phi0)], ph])
        zf = np.concatenate([[z0], zs])
        kg = "grip_l" if arm == "a" else "gripb"

        def fn(s_):
            k = s_ * nn
            i = min(int(k), nn - 1)
            f = k - i
            b = tuple(Bf[i] + (Bf[i + 1] - Bf[i]) * f)
            phx = phf[i] + (phf[i + 1] - phf[i]) * f
            tip, hd = edge_grip(phx)
            return {**base_state(b), "lift": zf[i] + (zf[i + 1] - zf[i]) * f, **arm_for(b, tip, hd, g_edge, arm, pitch)}
        T.add(dur, fn, mark, raw=True)
        return tuple(Bf[-1]), ok_, wr_, wc_

    if strategy == "one":
        # the door may have drifted while released: the hooked hand first pulls it back to the planned angle
        T.add(0.8, lambda s: arm_for(base2, *edge_grip(MEAS["phi"] + (pp - MEAS["phi"]) * min(1.0, 1.5 * s)), g_edge, "a"),
              "pull the edge back to the planned angle",
              nominal=lambda s: arm_for(base2, *edge_grip(pp), g_edge, "a"))
    if strategy == "one" or not handover2:
        base_e, ok, wr, wc = swing_part(base2, phi_pull, phi_end, z_handle, z_swing, "a", 0.0,
                                        "pull the edge round; base reverses and turns")
        last = "a"
    else:
        # the edge travels from the robot's right to its left: hand it from the right hand to the left one at
        # ~55 deg, so neither arm has to sweep across the other (both work in the same horizontal planes)
        phi_mid = 40.0
        base_m, ok1, wr1, wc1 = swing_part(base2, phi_pull, phi_mid, z_handle, z_swing, "a", 0.0,
                                           "pull the edge round (right hand)")
        pb = -15.0 if kind == "upper" else 30.0        # left hand hooks 6 cm above / 12 cm below the right one
        em, hdm = edge_grip(math.radians(phi_mid))
        pre_m = em - 0.08 * np.array([math.cos(hdm), math.sin(hdm)])
        T.to(2.0, "left hand to the edge", **arm_for(base_m, pre_m, hdm, g_open, "b", pb))
        T.add(1.0, lambda s: arm_for(base_m, pre_m + (em - pre_m) * s, hdm, g_open, "b", pb), "left hand around the edge")
        T.to(0.4, "left hand hooks the edge", gripb=g_edge)
        T.to(0.4, "right hand lets go", grip_l=g_open)
        ba = em + 0.06 * np.array([-math.cos(hdm), -math.sin(hdm)])
        T.add(0.8, lambda s: arm_for(base_m, em + (ba - em) * s, hdm, g_open, "a"), "right hand back")
        tuck(T, "a", "right arm tucks")
        base_e, ok2, wr2, wc2 = swing_part(base_m, phi_mid, phi_end, z_swing, z_swing, "b", pb,
                                           "pull the edge round (left hand)")
        ok, wr, wc = ok1 and ok2, min(wr1, wr2), min(wc1, wc2)
        last = "b"
    kl = "grip_l" if last == "a" else "gripb"
    p_last = 0.0 if last == "a" else (-15.0 if kind == "upper" else 30.0)
    if closing == "always":
        T.add(3.0, lambda s: {}, "hold it open (it would close if released)")
    else:
        T.to(0.6, "let go past the detent", **{kl: g_open})
        tip_end, hd_end = edge_grip(math.radians(phi_end))
        T.add(1.0, lambda s: arm_for(base_e, tip_end - 0.06 * s * np.array([math.cos(hd_end), math.sin(hd_end)]),
                                     hd_end, g_open, last, p_last), "hand back off the edge")
        T.add(3.0, lambda s: {}, "does it stay open?")
    info = dict(plan_ok=ok, reach_margin=wr, clearance=wc, base_end=(base_e[0] - base0[0], base_e[1] - base0[1],
                math.degrees(base_e[2])), z_handle=z_handle, z_swing=z_swing, hook_c=hook_c, hook_heading_err=herr,
                hook_err_range=(min(herr_s), max(herr_s)) if herr_s else (0.0, 0.0))
    return T, info


_PLANS = {}


def best_plan(strategy, kind, closing):
    """Try a few sideways start offsets of the base and keep the plan with the best swing clearance."""
    key = (strategy, kind, closing)
    if key not in _PLANS:
        best = None
        # one arm: right shoulder a little right of the handle; two arms: left shoulder in line with it
        if strategy == "stumprigid":
            by0s = []
            for by0 in np.arange(-0.40, -0.05, 0.025):
                hc = None
                for _ in range(3):                     # centre the fixed angle on the path's heading swing
                    _, i0 = plan(strategy, kind, closing, by0=by0, hook_c=hc)
                    lo, hi = i0["hook_err_range"]
                    hc = i0["hook_c"] - (lo + hi) / 2
                by0s.append((max(hi - lo, 0) / 2, by0, hc))
            cands = [(b, c) for _, b, c in sorted(by0s)[:3]]
        else:
            cands = [(b, None) for b in ((-0.07, -0.05, -0.03, 0.0) if strategy == "one" else (-0.25, -0.20, -0.15))]
        for by0, hc in cands:
            T, info = plan(strategy, kind, closing, by0=by0, hook_c=hc)
            gap = min(arms_gap(T.at(t)) for t in np.arange(0.0, T.t, 0.1)) if strategy != "one" else 9.0
            score = (info["plan_ok"], gap > 0.05, -round(info["hook_heading_err"] / 5),
                     min(info["clearance"], 0.05) + min(info["reach_margin"], 0.03))
            info.update(by0=by0, plan_arms_gap=gap)
            if best is None or score > best[0]:
                best = (score, T, info)
        _PLANS[key] = best[1:]
    return _PLANS[key]


# ------------------------------------------------------------------ physics
def run(strategy, kind, closing, f_break, mu, video=False, fps=30, label="", view="persp", log=None):
    traction = mu * DRIVE_SHARE * ROBOT_MASS * G
    m = build_model(kind, traction, arms=1 if strategy == "one" else 2,
                    stump=strategy if strategy in STUMPS else None, tag=f"_{os.getpid()}")
    d = mujoco.MjData(m)
    T, info = best_plan(strategy, kind, closing)
    aid = {nm: m.actuator(nm).id for nm in ACT}
    jq = {nm: m.jnt_qposadr[m.actuator_trnid[aid[nm]][0]] for nm in ACT}
    door_q = m.jnt_qposadr[m.joint("door").id]
    door_v = m.jnt_dofadr[m.joint("door").id]

    def vec(st):
        v = np.zeros(m.nu)
        for nm, key in ACT.items():
            x = st[key]
            v[aid[nm]] = math.radians(x) if key in ANGLES else x
        return v
    vmax = np.zeros(m.nu)
    for nm, key in ACT.items():
        vmax[aid[nm]] = math.radians(VM[key]) if key in ANGLES else VM[key]
    q0 = vec(T.at(0.0))
    for nm in ACT:
        d.qpos[jq[nm]] = q0[aid[nm]]
    d.qpos[m.jnt_qposadr[m.joint("grip_r").id]] = q0[aid["grip"]]
    d.qpos[m.jnt_qposadr[m.joint("grip_rb").id]] = q0[aid["gripb"]]
    d.ctrl[:] = q0
    mujoco.mj_forward(m, d)
    lim = np.array([m.actuator_forcerange[i][1] for i in range(m.nu)])
    peak = np.zeros(m.nu)
    sat = np.zeros(m.nu)
    lag = 0.0                      # largest rate-limiter clip (the plan asking for more than the joint speed)
    rob_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] & 2}
    gap = 9.0                      # closest approach of the two arms to each other (top view)
    door_body = m.body("door").id
    hit, hit_t = 0.0, 0.0          # largest contact force between the robot's own shapes and the door / fridge
    t_swing = [t0 for t0, mk in T.marks() if mk.startswith("pull the edge")][0]
    max_phi = 0.0
    min_phi_after_pull = 9.0
    t_pull_end = [t0 for t0, mk in T.marks() if mk.startswith("release the handle") or mk.startswith("right hand to")][0]
    t_regrasp = [t1 - t0 for (t0, mk), (t1, _) in zip(T.marks(), T.marks()[1:]) if mk.startswith("re-grasp")]
    renderer = mujoco.Renderer(m, 540, 860) if video else None
    frames = OUT / "fridge_door_frames"
    if video:
        frames.mkdir(parents=True, exist_ok=True)
        for p in frames.glob("*.png"):
            p.unlink()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.15, 0.05, 0.85)
    cam.distance, cam.azimuth, cam.elevation = 3.2, 35, -38
    vopt = mujoco.MjvOption()
    if view == "top":            # debugging: straight down, collision shapes shown
        cam.lookat[:] = (0.25, 0.05, 1.0)
        cam.distance, cam.azimuth, cam.elevation = 2.4, 180, -89.9
        vopt.geomgroup[3] = 1
    nxt, nf = 0.0, 0
    cmd = q0.copy()
    f6 = np.zeros(6)
    dt = m.opt.timestep
    while d.time < T.t:
        phi = -d.qpos[door_q]
        MEAS["phi"] = phi
        st_now = T.at(d.time)
        tgt = vec(st_now)
        if strategy != "one" and int(d.time / dt) % 40 == 0:
            gap = min(gap, arms_gap(st_now))
        step = np.clip(tgt - cmd, -vmax * dt * 1.05, vmax * dt * 1.05)
        lag = max(lag, float(np.max(np.abs(tgt - cmd - step) / np.maximum(vmax, 1e-9))))
        cmd = cmd + step
        d.ctrl[:] = cmd
        d.qfrc_applied[door_v] = door_torque(phi, f_break, closing)
        mujoco.mj_step(m, d)
        if log is not None and int(d.time / dt) % 200 == 0:
            log.append((d.time, math.degrees(phi), {nm: math.degrees(d.qpos[jq[nm]]) if ACT[nm] in ANGLES else d.qpos[jq[nm]]
                                                    for nm in ("j1b", "j2b", "j3b")},
                        {nm: math.degrees(cmd[aid[nm]]) for nm in ("j1b", "j2b", "j3b")}))
        frac = np.abs(d.actuator_force) / lim
        sat += (frac > 0.98) * dt
        if d.time > t_swing + 0.3:
            peak = np.maximum(peak, np.abs(d.actuator_force))
        max_phi = max(max_phi, phi)
        if d.time > t_pull_end:
            min_phi_after_pull = min(min_phi_after_pull, phi)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in rob_geoms) != (g2 in rob_geoms):
                other = g2 if g1 in rob_geoms else g1
                if m.geom_bodyid[other] == door_body or m.geom_bodyid[other] == 0:
                    mujoco.mj_contactForce(m, d, i, f6)
                    if abs(f6[0]) > hit:
                        hit, hit_t = abs(f6[0]), d.time
        if video and d.time >= nxt:
            nxt += 1.0 / fps
            renderer.update_scene(d, camera=cam, scene_option=vopt)
            from PIL import Image, ImageDraw
            im = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(im)
            mk = [mk for t0, mk in T.marks() if t0 <= d.time][-1]
            dr.rectangle((0, 506, 860, 540), fill=(20, 20, 24))
            dr.text((10, 514), f"t={d.time:4.1f}s  door {math.degrees(phi):5.1f} deg  {label}  |  {mk}", fill=(240, 240, 240))
            im.save(frames / f"f{nf:05d}.png")
            nf += 1
    final = math.degrees(-d.qpos[door_q])
    ok = final > 88 and bool(np.all(np.isfinite(d.qpos)))
    res = dict(strategy=strategy, kind=kind, closing=closing, f_break=f_break, mu=mu, ok=ok, final=final,
               max_deg=math.degrees(max_phi), min_after_pull=math.degrees(min_phi_after_pull), duration=T.t,
               regrasp_s=(t_regrasp[0] if t_regrasp else 0.0),
               lag=lag, hit=hit, hit_at=[mk for t0, mk in T.marks() if t0 <= hit_t][-1], arms_gap=gap, peaks={m.actuator(i).name: 100 * peak[i] / lim[i] for i in range(m.nu)},
               sat=max(sat[aid[k]] for k in ("j1", "j2", "j3", "j1b", "j2b", "j3b")), **info)
    if video and nf:
        mp4 = OUT / f"fridge_door_{strategy}_{kind}_{closing}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(mp4)], check=True)
        res["video"] = str(mp4)
    return res


def fmt(r):
    pk = r["peaks"]
    arm = max(pk[k] for k in ("j1", "j2", "j3", "j1b", "j2b", "j3b"))
    return (f"{r['strategy']:>4} {r['kind']:>5} {r['closing']:>6} {r['f_break']:4.0f} {r['mu']:4.1f} "
            f"{r['max_deg']:6.0f} {r['final']:6.0f} {'yes' if r['ok'] else 'NO ':>4} "
            f"{'ok' if r['plan_ok'] else 'NO':>4} {100 * r['clearance']:5.1f} {r['hit']:6.0f} {r['lag']:5.2f} "
            f"{arm:5.0f} {pk['base_x']:5.0f} {r['sat']:5.2f} {r['duration']:5.0f} {100 * r['arms_gap']:5.1f} {100 * r['by0']:+4.0f}  "
            f"{100 * r['base_end'][0]:+4.0f}/{100 * r['base_end'][1]:+4.0f} cm {r['base_end'][2]:+4.0f} deg")


HEADER = ("strat  door  close seal   mu   max  final  ok? plan  clr  hit N  lag   arm%  base%  sat s  time  arms  by0  base end\n"
          "  (max/final: door deg; plan clr: planned door clearance cm; hit: largest robot-body contact force on door/fridge;\n"
          "   lag: largest speed-limiter clip in s; arm%: peak arm torque during the swing; sat: s at an arm torque limit;\n"
          "   arms: closest approach of the two arms in cm, top view; by0: sideways start offset of the base in cm)")

if __name__ == "__main__":
    if "--video" in sys.argv:
        i = sys.argv.index("--video")
        st, kd, cl, fb, mu = sys.argv[i + 1:i + 6]
        r = run(st, kd, cl, float(fb), float(mu), video=True, label=f"{st} arm(s), {kd} door, {cl}")
        print(fmt(r))
        print(r.get("video"))
        sys.exit()
    strats = ("one", "two")
    if "--only" in sys.argv:          # e.g. --only stumpyaw  or  --only stump,stumprigid
        strats = tuple(sys.argv[sys.argv.index("--only") + 1].split(","))
    if "--quick" in sys.argv:
        cases = list(product(strats, ("upper", "full"), ("cam25", "cam60", "always"), (35,), (0.6,)))
    else:
        cases = list(product(strats, ("upper", "full"), ("cam25", "cam60", "always"), (15, 35, 55), (0.4, 0.6, 0.8)))
    print(HEADER)
    from multiprocessing import Pool
    with Pool(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1) as pool:
        rows = []
        for r in pool.starmap(run, cases):
            rows.append(r)
            print(fmt(r), flush=True)
    print("\nsuccess by strategy x door x closing (over seal x floor):")
    for st, kd, cl in product(strats, ("upper", "full"), ("cam25", "cam60", "always")):
        sel = [r for r in rows if (r["strategy"], r["kind"], r["closing"]) == (st, kd, cl)]
        if sel:
            print(f"  {st:>4} {kd:>5} {cl:>6}: {sum(r['ok'] for r in sel)}/{len(sel)}")
