"""Build a MuJoCo model of Tischlein v0.2 from the CAD (visual meshes) plus simple collision shapes.

Two arms. Arm a (right) keeps the v0 joint names (j1..j4, grip_l/grip_r) plus j5 (wrist roll), so the
scripted sims still drive it; arm b (left) has the same joints with a "b" suffix. The first ten actuators
are the v0 ones in the v0 order; the new ones (j5, arm b) come after them.

uv run python sim/build_mjcf.py   -> sim/tischlein.xml + sim/meshes/*.stl

Actuator force limits are the real ones from the spec sheets (peak, after belt
reductions), so a task that needs more torque than the hardware has will visibly fail.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad"))
sys.path.insert(0, str(ROOT))

from build123d import Compound, export_stl  # noqa: E402

import params as P  # noqa: E402
import robot as R  # noqa: E402

SIM = ROOT / "sim"
MESH = SIM / "meshes"
mm = 0.001

# real limits (N·m or N), see params / calc.py
TAU = dict(
    j1=P.STS3215_PEAK_NM * P.J1_RATIO,
    j2=P.STS3215_PEAK_NM * P.J2_RATIO,
    j3=P.STS3215_PEAK_NM,
    j4=P.STS3250_PEAK_NM,
    j5=P.STS3215_PEAK_NM,
)
GRIP_MAX = P.CAP_OPEN / 2


def cap_box(x0, x1, y0, y1, z0, z1):
    """Collision box given in the CAP gripper frame (mm) -> (pos, half-size) in the tool frame (m)."""
    tx = (P.CAP_ORIGIN_X + y0, P.CAP_ORIGIN_X + y1)
    ty = (-x1, -x0)
    tz = (z0 - P.CAP_AXIS_Z, z1 - P.CAP_AXIS_Z)
    pos = [sum(v) / 2 * mm for v in (tx, ty, tz)]
    half = [(v[1] - v[0]) / 2 * mm for v in (tx, ty, tz)]
    return pos, half


def export_link_meshes():
    MESH.mkdir(parents=True, exist_ok=True)
    out = {}
    for link, items in R.links().items():
        groups = defaultdict(list)
        for label, shape, rgb, printable in items:
            groups[tuple(round(c, 3) for c in rgb)].append(shape)
        out[link] = []
        for i, (rgb, shapes) in enumerate(groups.items()):
            name = f"{link}_{i}"
            export_stl(Compound(children=shapes), str(MESH / f"{name}.stl"), tolerance=0.3, angular_tolerance=0.3)
            out[link].append((name, rgb))
    return out


def f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def vis(link, meshes):
    return "\n".join(
        f'<geom class="vis" mesh="{n}" rgba="{f(*rgb, 1)}"/>' for n, rgb in meshes[link])


def arm_xml(arm, meshes):
    """Upper arm -> forearm -> wrist -> hand (pitch) -> tool (roll) -> fingers, for arm 'a' (v0 names) or 'b'."""
    sfx = "" if arm == "a" else "_b"          # body names
    j = (lambda n: n) if arm == "a" else (lambda n: n + "b")
    ys = P.ARM_Y[0 if arm == "a" else 1] * mm
    v = lambda link: vis(f"{link}_{arm}", meshes)
    lo, hi = P.PITCH_RANGE
    body, holder, phone = cap_box(-32.5, 32.5, -64.3, 48.7, -5.3, 33.0), cap_box(-26, 16, -64.3, -30, 33, 142), \
        cap_box(-31.8, 114.9, -41.0, -32.5, 43.6, 114.1)
    gl, gr = ("grip_l", "grip_r") if arm == "a" else ("grip_lb", "grip_rb")
    pad = P.PAD_X * mm
    return f"""
        <body name="upper_arm{sfx}" pos="{f(P.SHOULDER_X*mm, ys, P.L1_ABOVE_TRAY*mm)}">
          <joint name="{j('j1')}" type="hinge" axis="0 0 1" range="-150 150" armature="0.05"/>
          <inertial pos="0.11 0 0" mass="{P.M['arm_L1']+0.1}" diaginertia="0.002 0.004 0.004"/>
          {v("upper_arm")}
          <body name="forearm{sfx}" pos="{f(P.L1*mm, 0, (P.L2_ABOVE_TRAY-P.L1_ABOVE_TRAY)*mm)}">
            <joint name="{j('j2')}" type="hinge" axis="0 0 1" range="-155 155" armature="0.02"/>
            <inertial pos="0.12 0 -0.01" mass="{P.M['arm_L2']+0.06}" diaginertia="0.001 0.003 0.003"/>
            {v("forearm")}
            <body name="wrist{sfx}" pos="{f(P.L2*mm, 0, 0)}">
              <joint name="{j('j3')}" type="hinge" axis="0 0 1" range="-180 180" armature="0.005"/>
              <inertial pos="-0.01 0 -0.06" mass="0.12" diaginertia="0.0003 0.0003 0.0003"/>
              {v("wrist")}
              <body name="hand{sfx}" pos="0 0 {-P.PITCH_DROP*mm}">
                <joint name="{j('j4')}" type="hinge" axis="0 1 0" range="{lo} {hi}" armature="0.008"/>
                <inertial pos="0.045 -0.01 0" mass="0.16" diaginertia="0.0004 0.0004 0.0004"/>
                {v("hand")}
                <geom class="col" type="box" pos="0.0455 -0.0125 0" size="0.0175 0.0225 0.0124"/>
                <body name="tool{sfx}" pos="0 0 0">
                  <joint name="{j('j5')}" type="hinge" axis="1 0 0" range="-180 180" armature="0.005"/>
                  <inertial pos="0.13 -0.02 0.02" mass="0.45" diaginertia="0.0015 0.0015 0.0012"/>
                  {v("tool")}
                  <geom class="col" type="box" pos="{f(*body[0])}" size="{f(*body[1])}"/>
                  <geom class="col" type="box" pos="{f(*holder[0])}" size="{f(*holder[1])}"/>
                  <geom class="col" type="box" pos="{f(*phone[0])}" size="{f(*phone[1])}"/>
                  <body name="finger_l{sfx}" pos="0 0 0">
                    <joint name="{gl}" type="slide" axis="0 1 0" range="-0.01 {GRIP_MAX*mm}" damping="2"/>
                    <inertial pos="{f(pad, 0.018, 0)}" mass="0.03" diaginertia="5e-5 5e-5 5e-5"/>
                    {v("finger_l")}
                    <geom class="pad" type="box" pos="{f(pad, 0.018, 0)}" size="0.03 0.006 0.015"/>
                  </body>
                  <body name="finger_r{sfx}" pos="0 0 0">
                    <joint name="{gr}" type="slide" axis="0 -1 0" range="-0.01 {GRIP_MAX*mm}" damping="2"/>
                    <inertial pos="{f(pad, -0.018, 0)}" mass="0.03" diaginertia="5e-5 5e-5 5e-5"/>
                    {v("finger_r")}
                    <geom class="pad" type="box" pos="{f(pad, -0.018, 0)}" size="0.03 0.006 0.015"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>"""


def build(meshes):
    tray_x0 = P.TRAY_X0 * mm
    tray_x1 = (P.TRAY_X0 + P.TRAY_DEPTH) * mm
    tw = P.TRAY_WIDTH / 2 * mm
    assets = "\n".join(f'<mesh name="{n}" file="{n}.stl" scale="0.001 0.001 0.001"/>'
                       for ms in meshes.values() for n, _ in ms)
    xml = f"""
<mujoco model="tischlein_v0_2">
  <compiler angle="degree" meshdir="meshes" inertiafromgeom="false" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="pyramidal" noslip_iterations="4"/>
  <size memory="200M"/>
  <visual>
    <global offwidth="1600" offheight="1000" azimuth="210" elevation="-25"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <map znear="0.01"/>
  </visual>
  <default>
    <geom friction="0.6 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.001" condim="4"/>
    <default class="vis"><geom type="mesh" contype="0" conaffinity="0" group="1" mass="0"/></default>
    <default class="col"><geom group="3" rgba="0.9 0.2 0.2 0.25"/></default>
    <default class="pad"><geom group="3" friction="1.2 0.02 0.002" rgba="0.2 0.5 0.9 0.4"/></default>
    <joint damping="0.5" armature="0.01"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.94 0.95 0.97" rgb2="0.78 0.80 0.84" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.80 0.76" rgb2="0.78 0.76 0.72" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    {assets}
  </asset>
  <worldbody>
    <light pos="0.5 -1 3" dir="-0.2 0.3 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="-2 2 3" dir="0.5 -0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor"/>
    <body name="base" pos="0 0 0">
      <joint name="base_x" type="slide" axis="1 0 0" damping="50"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="50"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="20"/>
      <inertial pos="-0.06 0 0.2" mass="11.8" diaginertia="0.7 0.7 0.5"/>
      {vis("base", meshes)}
      <body name="carriage" pos="0 0 0">
        <joint name="lift" type="slide" axis="0 0 1" range="{P.TRAY_Z_MIN*mm} {P.TRAY_Z_MAX*mm}" damping="30"/>
        <inertial pos="{f((P.YOKE_X0+40)*mm, 0, 0.12)}" mass="{P.M['carriage'] + 2 * P.M['arm_shoulder']}" diaginertia="0.02 0.01 0.02"/>
        {vis("carriage", meshes)}
        <body name="drawer" pos="0 0 0">
          <joint name="drawer" type="slide" axis="1 0 0" range="0 {P.DRAWER_STROKE*mm}" damping="10"/>
          <inertial pos="{f((P.TRAY_X0+125)*mm, 0, -0.01)}" mass="{P.M['tray']}" diaginertia="0.004 0.003 0.006"/>
          {vis("drawer", meshes)}
          <geom class="col" type="box" pos="{f((tray_x0+tray_x1)/2, 0, -0.015)}" size="{f((tray_x1-tray_x0)/2, tw, 0.015)}" friction="0.45 0.01 0.001"/>
          <geom class="col" type="box" pos="{f((tray_x0+tray_x1-0.025)/2, tw-0.0025, 0.0075)}" size="{f((tray_x1-tray_x0-0.025)/2, 0.0025, 0.0075)}"/>
          <geom class="col" type="box" pos="{f((tray_x0+tray_x1-0.025)/2, -tw+0.0025, 0.0075)}" size="{f((tray_x1-tray_x0-0.025)/2, 0.0025, 0.0075)}"/>
          <geom class="col" type="box" pos="{f(tray_x0+0.0025, 0, 0.0075)}" size="{f(0.0025, tw, 0.0075)}"/>
        </body>
        {arm_xml("a", meshes)}
        {arm_xml("b", meshes)}
      </body>
    </body>
    {props_xml()}
  </worldbody>
  <equality>
    <joint joint1="grip_r" joint2="grip_l" polycoef="0 1 0 0 0" solref="0.005 1"/>
    <joint joint1="grip_rb" joint2="grip_lb" polycoef="0 1 0 0 0" solref="0.005 1"/>
  </equality>
  <actuator>
    <position name="base_x" joint="base_x" kp="40000" kv="3000" forcerange="-1500 1500"/>
    <position name="base_y" joint="base_y" kp="40000" kv="3000" forcerange="-1500 1500"/>
    <position name="base_yaw" joint="base_yaw" kp="4000" kv="400" forcerange="-500 500"/>
    <position name="lift" joint="lift" kp="40000" kv="2500" forcerange="-200 200"/>
    <position name="drawer" joint="drawer" kp="4000" kv="200" forcerange="-60 60"/>
    <position name="j1" joint="j1" kp="200" kv="8" forcerange="{-TAU['j1']:.2f} {TAU['j1']:.2f}"/>
    <position name="j2" joint="j2" kp="150" kv="5" forcerange="{-TAU['j2']:.2f} {TAU['j2']:.2f}"/>
    <position name="j3" joint="j3" kp="40" kv="1" forcerange="{-TAU['j3']:.2f} {TAU['j3']:.2f}"/>
    <position name="j4" joint="j4" kp="60" kv="1.5" forcerange="{-TAU['j4']:.2f} {TAU['j4']:.2f}"/>
    <position name="grip" joint="grip_l" kp="8000" kv="60" forcerange="-100 100"/>
    <position name="j5" joint="j5" kp="30" kv="0.8" forcerange="{-TAU['j5']:.2f} {TAU['j5']:.2f}"/>
    <position name="j1b" joint="j1b" kp="200" kv="8" forcerange="{-TAU['j1']:.2f} {TAU['j1']:.2f}"/>
    <position name="j2b" joint="j2b" kp="150" kv="5" forcerange="{-TAU['j2']:.2f} {TAU['j2']:.2f}"/>
    <position name="j3b" joint="j3b" kp="40" kv="1" forcerange="{-TAU['j3']:.2f} {TAU['j3']:.2f}"/>
    <position name="j4b" joint="j4b" kp="60" kv="1.5" forcerange="{-TAU['j4']:.2f} {TAU['j4']:.2f}"/>
    <position name="j5b" joint="j5b" kp="30" kv="0.8" forcerange="{-TAU['j5']:.2f} {TAU['j5']:.2f}"/>
    <position name="gripb" joint="grip_lb" kp="8000" kv="60" forcerange="-100 100"/>
  </actuator>
</mujoco>
"""
    return xml


# ------------------------------------------------------------------ world props
FRIDGE_X = 0.72 - P.AXLE_SHIFT * mm   # fridge front, world (robot stops at x = 0 facing +x)
SHELF = 0.950
BOX0 = (FRIDGE_X + 0.058, 0.0)    # meal box front face at the shelf front


def b(name, pos, size, rgba, extra=""):
    return f'<geom name="{name}" type="box" pos="{f(*pos)}" size="{f(*size)}" rgba="{f(*rgba)}" {extra}/>'


def props_xml():
    W, D, H = 0.60, 0.65, 1.85
    fx = FRIDGE_X
    white = (0.93, 0.93, 0.91, 1)
    glass = (0.75, 0.88, 0.92, 0.55)
    g = []
    g.append(b("fr_left", (fx + D / 2, W / 2 - 0.0175, H / 2), (D / 2, 0.0175, H / 2), white))
    g.append(b("fr_right", (fx + D / 2, -W / 2 + 0.0175, H / 2), (D / 2, 0.0175, H / 2), white))
    g.append(b("fr_back", (fx + D - 0.02, 0, H / 2), (0.02, W / 2, H / 2), white))
    g.append(b("fr_top", (fx + D / 2, 0, H - 0.02), (D / 2, W / 2, 0.02), white))
    g.append(b("fr_freezer", (fx + D / 2 - 0.035, 0, 0.44), (D / 2 + 0.035, W / 2, 0.44), white))
    for i, z in enumerate((SHELF, 1.15, 1.35, 1.55)):
        g.append(b(f"shelf{i}", (fx + 0.04 + (D - 0.085) / 2, 0, z - 0.012), ((D - 0.085) / 2, W / 2 - 0.035, 0.012), glass, 'friction="0.3 0.01 0.001"'))
    # door, opened 150 degrees, hinge on the +y front corner
    th = math.radians(-150)
    hx, hy = fx, W / 2
    cx = hx + (-0.035) * math.cos(th) - (-W / 2) * math.sin(th)
    cy = hy + (-0.035) * math.sin(th) + (-W / 2) * math.cos(th)
    g.append(f'<geom name="fr_door" type="box" pos="{f(cx, cy, 0.88 + (H - 0.9) / 2)}" size="{f(0.035, W / 2, (H - 0.9) / 2)}" euler="0 0 {math.degrees(th):.1f}" rgba="{f(*white)}"/>')
    # bed: long axis along world x, robot approaches from -y... placed for the final robot pose
    bx0, bx1 = -1.55 - P.AXLE_SHIFT * mm, 0.45 - P.AXLE_SHIFT * mm
    by0 = 1.0 + 0.36
    g.append(b("bed_frame", ((bx0 + bx1) / 2, by0 + 0.475, 0.325), ((bx1 - bx0) / 2, 0.475, 0.045), (0.55, 0.40, 0.28, 1)))
    for k, (lx, ly) in enumerate([(lx, ly) for lx in (bx0 + 0.03, bx1 - 0.03) for ly in (by0 + 0.03, by0 + 0.92)]):
        g.append(b(f"bed_leg{k}", (lx, ly, 0.14), (0.015, 0.015, 0.14), (0.55, 0.40, 0.28, 1)))
    g.append(b("mattress", ((bx0 + bx1) / 2, by0 + 0.475, 0.46), ((bx1 - bx0) / 2, 0.475, 0.09), (0.93, 0.93, 0.95, 1)))
    g.append(b("duvet", ((bx0 + bx1) / 2 - 0.2, by0 + 0.475, 0.57), ((bx1 - bx0) / 2 - 0.2, 0.46, 0.02), (0.55, 0.62, 0.70, 1)))
    g.append(b("pillow", (bx1 - 0.17, by0 + 0.475, 0.6), (0.15, 0.3, 0.06), (0.97, 0.97, 0.97, 1)))
    # meal box (free), knob on its front face (-x)
    x0, y0 = BOX0
    g.append(f"""<body name="meal_box" pos="{f(x0, y0, SHELF)}">
      <freejoint/>
      <inertial pos="0.09 0 0.03" mass="0.8" diaginertia="0.0013 0.0024 0.0030"/>
      <geom type="box" pos="0.09 0 0.035" size="0.09 0.06 0.035" rgba="0.85 0.30 0.25 1" friction="0.35 0.01 0.001"/>
      <geom type="box" pos="0.09 0 0.069" size="0.091 0.061 0.003" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.009 0 0.035" size="0.009 0.015 0.015" rgba="0.85 0.30 0.25 1" friction="1 0.02 0.002"/>
      <geom type="cylinder" pos="-0.002 0 0.035" size="0.02 0.002" euler="0 90 0" rgba="0.85 0.30 0.25 1" contype="0" conaffinity="0"/>
    </body>""")
    g.append(f"""<body name="bottle" pos="{f(fx + 0.16, 0.19, SHELF)}">
      <freejoint/>
      <inertial pos="0 0 0.1" mass="0.55" diaginertia="0.002 0.002 0.0003"/>
      <geom type="cylinder" pos="0 0 0.095" size="0.033 0.095" rgba="0.35 0.70 0.95 0.9"/>
      <geom type="cylinder" pos="0 0 0.21" size="0.015 0.02" rgba="0.35 0.70 0.95 0.9"/>
    </body>""")
    return "\n".join(g)


if __name__ == "__main__":
    meshes = export_link_meshes()
    xml = build(meshes)
    (SIM / "tischlein.xml").write_text(xml)
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(SIM / "tischlein.xml"))
    print(f"ok: nq={m.nq} nu={m.nu} ngeom={m.ngeom} nmesh={m.nmesh}")
