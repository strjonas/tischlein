"""Tischlein v0.2 — parametric CAD (build123d).

The robot is described as a kinematic tree of links. Each link function returns
a list of (label, shape, rgb, printable) in that link's own frame, so the same
geometry feeds the assembly renders, the MuJoCo model and the print files.

v0.2: two identical arms ("a" on the right at y = ARM_Y[0], "b" on the left), a wrist roll stage,
and the CAP robot gripper (NYU design, driven by its own Dynamixel XL330) with an iPhone on top.
The CAP gripper's own parts are printed from its published STLs (see print/README.md); here it is
an envelope for clearance, renders and the simulation.

Link frames (all z up), per arm s in {a, b}:
  base         world frame of the robot (origin on the floor under the drive axle)
  carriage     base + (0, 0, lift)            lift = tray surface height
  upper_arm_s  carriage + (SHOULDER_X, ARM_Y, L1_ABOVE_TRAY), rotated j1 about z
  forearm_s    upper_arm + (L1, 0, L2-L1 plane offset), rotated j2 about z
  wrist_s      forearm + (L2, 0, 0), rotated j3 about z
  hand_s       wrist + (0, 0, -PITCH_DROP), rotated j4 about y (+90 = pointing down)
  tool_s       hand rotated j5 about x (wrist roll); the roll axis runs through the fingers' mid-plane
  finger_*_s   tool + (0, +/-grip, 0)  (parallel stand-in for the CAP's angular jaws)
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

from build123d import (Align, Axis, Box, Color, Compound, Cone, Cylinder, Location, Plane, Pos, Rot,
                       Shape, import_step)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import params as P  # noqa: E402

MIN = Align.MIN
MAX = Align.MAX
CEN = Align.CENTER

# colours
PRINT_A = (0.93, 0.55, 0.20)    # orange PETG: structural prints
PRINT_B = (0.22, 0.24, 0.27)    # dark PETG: covers, brackets
TPU = (0.20, 0.55, 0.85)
ALU = (0.75, 0.77, 0.80)
STEEL = (0.55, 0.57, 0.60)
RUBBER = (0.10, 0.10, 0.11)
SERVO = (0.12, 0.12, 0.14)
WOOD = (0.80, 0.68, 0.50)
PCB = (0.10, 0.45, 0.25)
RED = (0.85, 0.10, 0.10)
CAP_WHITE = (0.94, 0.94, 0.94)
CAP_BLUE = (0.25, 0.55, 0.75)
PHONE = (0.12, 0.12, 0.13)
LAMP = (0.97, 0.94, 0.86)

# fasteners / fits
M3 = 3.4
M4 = 4.5
M5 = 5.5
BEARING_6000 = dict(od=26.0, id=10.0, w=8.0)
SHAFT_D = 10.0


# ------------------------------------------------------------------ primitives
def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    z0, z1 = sorted((z0, z1))
    return Pos(x0, y0, z0) * Box(x1 - x0, y1 - y0, z1 - z0, align=(MIN, MIN, MIN))


def cyl_z(x, y, z0, z1, d):
    return Pos(x, y, z0) * Cylinder(d / 2, z1 - z0, align=(CEN, CEN, MIN))


def cyl_y(x, z, y0, y1, d):
    return Pos(x, y0, z) * (Rot(-90, 0, 0) * Cylinder(d / 2, y1 - y0, align=(CEN, CEN, MIN)))


def cyl_x(y, z, x0, x1, d):
    return Pos(x0, y, z) * (Rot(0, 90, 0) * Cylinder(d / 2, x1 - x0, align=(CEN, CEN, MIN)))


def extrusion(length, w=20, h=20, along="x"):
    """V-slot-ish aluminium profile, centred on its axis, starting at 0."""
    body = Box(length, w, h, align=(MIN, CEN, CEN))
    slot = 6.2
    for k in range(int(w // 20)):
        yc = -w / 2 + 10 + 20 * k
        body -= box(-1, length + 1, yc - slot / 2, yc + slot / 2, h / 2 - 1.8, h / 2 + 1)
        body -= box(-1, length + 1, yc - slot / 2, yc + slot / 2, -h / 2 - 1, -h / 2 + 1.8)
    for k in range(int(h // 20)):
        zc = -h / 2 + 10 + 20 * k
        body -= box(-1, length + 1, w / 2 - 1.8, w / 2 + 1, zc - slot / 2, zc + slot / 2)
        body -= box(-1, length + 1, -w / 2 - 1, -w / 2 + 1.8, zc - slot / 2, zc + slot / 2)
    if along == "z":
        body = Rot(0, -90, 0) * body
    elif along == "y":
        body = Rot(0, 0, 90) * body
    return body


@lru_cache(maxsize=1)
def _servo_raw():
    s = import_step(str(ROOT / "vendor/so-arm100/STS3215_03a.step"))
    # STEP: output axis at (12.5, 0) along +z, horn top at z = 18.7
    return Pos(-12.5, 0, -18.7) * s


def servo(loc: Location | None = None):
    """STS3215 in its local frame: output axis = z, horn face at z = 0, body toward -x and -z."""
    s = _servo_raw()
    return s if loc is None else loc * s


# Servo envelope in local frame (for pockets): x [-35.2, 10.2], y [-12.4, 12.4], z [-38.1, -2.8]
SERVO_POCKET = dict(x0=-35.6, x1=10.6, y0=-12.8, y1=12.8, z0=-38.5, z1=-2.5)
SERVO_TOP_HOLES = [(-7.5, 10.2), (-7.5, -10.2), (-28.2, 10.2), (-28.2, -10.2)]  # M2 into cover


def servo_pocket(loc: Location):
    p = SERVO_POCKET
    return loc * box(p["x0"], p["x1"], p["y0"], p["y1"], p["z0"], p["z1"] + 30)


def gt2_pulley(teeth, width=6, flange=True):
    pd = P.gt2_pd(teeth)
    p = Cylinder(pd / 2 - 0.25, width, align=(CEN, CEN, MIN))
    if flange:
        p += Pos(0, 0, -1) * Cylinder(pd / 2 + 2, 1, align=(CEN, CEN, MIN))
        p += Pos(0, 0, width) * Cylinder(pd / 2 + 2, 1, align=(CEN, CEN, MIN))
    hub = Pos(0, 0, -1) * Cylinder(max(5.0, pd / 4), width + 8, align=(CEN, CEN, MIN))
    return (p + hub) - Pos(0, 0, -5) * Cylinder(SHAFT_D / 2 if teeth > 24 else 2.5, 40, align=(CEN, CEN, MIN))


def belt_loop(c1, c2, r1, r2, z, width=6, t=1.4):
    """Closed belt between two pulley centres (x, y) at height z, as a thin wall (visual)."""
    (x1, y1), (x2, y2) = c1, c2
    d = math.hypot(x2 - x1, y2 - y1)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    a = Pos(0, 0, 0) * Cylinder(r1 + t, width, align=(CEN, CEN, MIN)) - Cylinder(r1, width + 1, align=(CEN, CEN, MIN))
    b = Pos(d, 0, 0) * (Cylinder(r2 + t, width, align=(CEN, CEN, MIN)) - Cylinder(r2, width + 1, align=(CEN, CEN, MIN)))
    loop = a + b
    # straight runs (approximate tangents for unequal pulleys)
    loop += box(0, d, r1, r1 + t, 0, width) if r1 == r2 else _tangent_runs(r1, r2, d, width, t)
    if r1 == r2:
        loop += box(0, d, -r1 - t, -r1, 0, width)
    return Pos(x1, y1, z) * (Rot(0, 0, ang) * loop)


def _tangent_runs(r1, r2, d, width, t):
    beta = math.asin((r1 - r2) / d)
    runs = None
    for s in (1, -1):
        p1 = (r1 * math.sin(beta) * 1, s * r1 * math.cos(beta))
        p2 = (d + r2 * math.sin(beta), s * r2 * math.cos(beta))
        L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        a = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        run = Pos(p1[0], p1[1], 0) * (Rot(0, 0, a) * box(0, L, -t / 2 + s * t / 2, t / 2 + s * t / 2, 0, width))
        runs = run if runs is None else runs + run
    return runs


def mgn12_block():
    return box(-22.7, 22.7, -13.5, 13.5, 0, P.BLOCK_H + 5)  # 45.4 long, 27 wide


def bearing_6000(z0):
    o = BEARING_6000
    return cyl_z(0, 0, z0, z0 + o["w"], o["od"]) - cyl_z(0, 0, z0 - 1, z0 + o["w"] + 1, o["id"])


# ------------------------------------------------------------------ base link
def link_base():
    out = []
    zr0 = P.CASTER_D + 16 + 12       # frame rails above the casters' mounting plates
    zr1 = zr0 + 20
    deck_t = 6
    rail_y = P.TRACK / 2 - P.WHEEL_W / 2 - 30

    # frame: side rails, front and rear cross rails
    for s in (1, -1):
        out.append(("frame_side", Pos(P.BASE_X_REAR, s * rail_y, zr0 + 10) * extrusion(P.BASE_X_FRONT - P.BASE_X_REAR - 20), ALU, False))
    out.append(("frame_cross", Pos(P.BASE_X_REAR + 10, -rail_y + 10, zr0 + 10) * extrusion(2 * rail_y - 20, along="y"), ALU, False))
    # front cross rail runs out to the straddle legs
    fy = P.FRONT_CASTER_Y + 10
    out.append(("frame_cross_front", Pos(P.BASE_X_FRONT - 10, -fy, zr0 + 10) * extrusion(2 * fy, along="y"), ALU, False))
    for s_ in (1, -1):
        leg_len = P.FRONT_CASTER_X + 25 - P.BASE_X_FRONT
        out.append(("straddle_leg_2020", Pos(P.BASE_X_FRONT, s_ * P.FRONT_CASTER_Y, zr0 + 10) * extrusion(leg_len), ALU, False))
    deck = box(P.BASE_X_REAR, P.BASE_X_FRONT - 20, -rail_y - 10, rail_y + 10, zr1, zr1 + deck_t)
    deck -= box(P.MAST_X - 30, P.BASE_X_FRONT + 1, -45, 45, zr1 - 1, zr1 + deck_t + 1)  # notch at the mast
    out.append(("deck_plywood", deck, WOOD, False))

    # hub motors (hoverboard), axles and printed axle-clamp adapters
    for s in (1, -1):
        wheel = cyl_y(0, P.WHEEL_D / 2, -P.WHEEL_W / 2, P.WHEEL_W / 2, P.WHEEL_D)
        wheel -= cyl_y(0, P.WHEEL_D / 2, P.WHEEL_W / 2 - 4, P.WHEEL_W / 2 + 1, P.WHEEL_D - 30)
        out.append(("hub_motor", Pos(0, s * P.TRACK / 2, 0) * wheel, RUBBER, False))
        out.append(("hub_motor_hub", Pos(0, s * P.TRACK / 2, 0) * cyl_y(0, P.WHEEL_D / 2, -P.WHEEL_W / 2 - 0.5, P.WHEEL_W / 2 + 0.5, P.WHEEL_D - 34), STEEL, False))
        axle_y0, axle_y1 = sorted((s * (P.TRACK / 2 - P.WHEEL_W / 2), s * (rail_y - 10)))
        out.append(("axle", cyl_y(0, P.WHEEL_D / 2, axle_y0, axle_y1, 16), STEEL, False))
        clamp = box(-22, 22, s * rail_y - 12, s * rail_y + 12, P.WHEEL_D / 2 - 14, zr0)
        clamp -= cyl_y(0, P.WHEEL_D / 2, s * rail_y - 13, s * rail_y + 13, 16.3)
        for xh in (-15, 15):
            clamp -= cyl_z(xh, s * rail_y, P.WHEEL_D / 2 - 15, zr0 + 1, M5)
        out.append(("axle_clamp_adapter", clamp, PRINT_B, True))

    # casters: one under each straddle leg (sprung pads), one sprung rear caster
    def caster(x, y):
        c = P.CASTER_D
        wheel = cyl_y(x + 18, c / 2, y - 12, y + 12, c)
        fork = box(x - 2, x + 28, y - 16, y + 16, c / 2, c + 12)
        plate = box(x - 22, x + 22, y - 22, y + 22, c + 12, c + 16)
        return wheel, fork + plate
    for y in (P.FRONT_CASTER_Y, -P.FRONT_CASTER_Y):
        w, f = caster(P.FRONT_CASTER_X, y)
        out.append(("caster_wheel", w, RUBBER, False))
        out.append(("caster_fork", f, STEEL, False))
        pad = box(P.FRONT_CASTER_X - 25, P.FRONT_CASTER_X + 25, y - 25, y + 25, P.CASTER_D + 16, zr0)
        out.append(("front_caster_pad_tpu", pad, TPU, True))
    w, f = caster(P.REAR_CASTER_X, 0)
    out.append(("caster_wheel", w, RUBBER, False))
    out.append(("caster_fork", f, STEEL, False))
    spring_mount = box(P.REAR_CASTER_X - 25, P.REAR_CASTER_X + 25, -25, 25, P.CASTER_D + 16, zr0 + 20)
    spring_mount -= box(P.REAR_CASTER_X - 26, P.REAR_CASTER_X + 26, -10.2, 10.2, zr0, zr0 + 21)
    out.append(("rear_caster_spring_mount", spring_mount, PRINT_A, True))

    # battery, electronics, e-stop, lidar (on the deck: they moved back with the axle shift)
    zt = zr1 + deck_t
    A = P.AXLE_SHIFT
    out.append(("battery_hoverboard_36v", box(-235 - A, -60 - A, -45, 45, zt, zt + 70), (0.15, 0.15, 0.17), False))
    ebox = box(-235 - A, -60 - A, 52, rail_y + 8, zt, zt + 55) - box(-233 - A, -62 - A, 54, rail_y + 6, zt + 2, zt + 56)
    out.append(("electronics_box", ebox, PRINT_B, True))
    out.append(("pi_and_drivers", box(-225 - A, -80 - A, 60, rail_y, zt + 5, zt + 25), PCB, False))
    out.append(("estop", cyl_z(-200 - A, -95, zt, zt + 30, 40) + cyl_z(-200 - A, -95, zt + 30, zt + 42, 34), RED, False))
    post = cyl_z(-140 - A, 0, zt + 70, zt + 120, 30)
    out.append(("lidar_post", post, PRINT_B, True))
    out.append(("lidar_rplidar_c1", cyl_z(-140 - A, 0, zt + 120, zt + 160, 56), (0.05, 0.05, 0.05), False))

    # mast: 2040 + MGN12 rail on its front face
    mx0 = P.MAST_X - P.MAST_W_X / 2
    mast = box(mx0, mx0 + 40, -10, 10, P.MAST_Z0, P.MAST_Z0 + P.MAST_LEN)
    for xc in (mx0 + 10, mx0 + 30):
        for s in (1, -1):
            mast -= box(xc - 3.1, xc + 3.1, s * 10 - 1.8 if s > 0 else -11, s * 10 + 1.8 if s < 0 else 11, P.MAST_Z0 - 1, P.MAST_Z0 + P.MAST_LEN + 1)
    mast -= box(mx0 + 37, mx0 + 41, -3.1, 3.1, P.MAST_Z0 - 1, P.MAST_Z0 + P.MAST_LEN + 1)
    mast -= box(mx0 - 1, mx0 + 3, -3.1, 3.1, P.MAST_Z0 - 1, P.MAST_Z0 + P.MAST_LEN + 1)
    out.append(("mast_2040", mast, ALU, False))
    rail_z0, rail_z1 = 15, P.MAST_Z0 + P.MAST_LEN - 140
    out.append(("rail_mgn12", box(mx0 + 40, mx0 + 40 + P.RAIL_H, -6, 6, rail_z0, rail_z1), STEEL, False))
    # mast foot: two printed gussets onto the front cross rail
    for s in (1, -1):
        g = box(mx0 - 45, mx0, s * 10 - (0 if s > 0 else 6), s * 10 + (6 if s > 0 else 0), zr0 - 30, zr1 + 60)
        g -= box(mx0 - 46, mx0 - 20, s * 10 - 8, s * 10 + 8, zr1 + 20, zr1 + 61)
        out.append(("mast_foot_gusset", g, PRINT_A, True))

    # lift drive: NEMA17 lying on the deck, 3:1 to a 20T lift pulley on the mast side
    ys = -10 - 9                      # belt plane beside the mast (right side)
    lift_pd = P.gt2_pd(P.LIFT_PULLEY_T)
    zp = 165                          # lower lift pulley axis height
    xp = P.MAST_X + 6.4 - lift_pd / 2  # pulley centred so its front strand is at x = MAST_X + 6.4... (visual)
    xp = P.MAST_X
    motor_x, motor_z = xp - 100, zt + 21
    out.append(("nema17", box(motor_x - 21, motor_x + 21, -95, -55, motor_z - 21, motor_z + 21), (0.15, 0.15, 0.15), False))
    out.append(("nema17_mount", box(motor_x - 25, motor_x + 25, -55, -50, zt, motor_z + 25), PRINT_A, True))
    out.append(("lift_pulley_20T", Pos(xp, ys - 3, zp) * (Rot(-90, 0, 0) * gt2_pulley(20)), ALU, False))
    out.append(("lift_reduction_60T", Pos(xp, -44, zp) * (Rot(-90, 0, 0) * gt2_pulley(60)), ALU, False))
    out.append(("motor_pulley_20T", Pos(motor_x, -44, motor_z) * (Rot(-90, 0, 0) * gt2_pulley(20)), ALU, False))
    lower = box(xp - 22, xp + 22, -52, -10, zp - 22, zp + 22) - cyl_y(xp, zp, -53, -9, 26.1)
    lower -= box(xp - 13, xp + 13, -40, -16, zp - 30, zp + 30)
    out.append(("lift_lower_bearing_block", lower, PRINT_A, True))
    ztop = P.MAST_Z0 + P.MAST_LEN - 30
    out.append(("lift_top_idler", Pos(xp, ys - 3, ztop) * (Rot(-90, 0, 0) * gt2_pulley(20)), ALU, False))
    top = box(xp - 22, xp + 22, -30, -10, ztop - 20, P.MAST_Z0 + P.MAST_LEN) - box(xp - 13, xp + 13, -26, -12, ztop - 21, ztop + 13)
    top = box(xp - 22, xp + 22, -30, -10, ztop - 20, P.MAST_Z0 + P.MAST_LEN + 8)
    top -= box(xp - 14, xp + 14, -27, -12, ztop - 21, ztop + 14)
    out.append(("lift_top_block", top, PRINT_A, True))
    # lift belt (two strands) and reduction belt (visual)
    for xs in (xp - lift_pd / 2 - 0.7, xp + lift_pd / 2 + 0.7):
        out.append(("lift_belt", box(xs - 0.7, xs + 0.7, ys - 3, ys + 3, zp, ztop), RUBBER, False))

    # mast top: lamp (reading/night light, and light for the cameras), constant-force spring balancer
    zm = P.MAST_Z0 + P.MAST_LEN
    holder = cyl_z(P.MAST_X, 0, zm + 8, zm + 40, 36) + box(mx0 - 4, mx0 + 44, -14, 14, zm - 30, zm + 12)
    holder -= box(mx0 - 0.2, mx0 + 40.2, -10.2, 10.2, zm - 31, zm + 0.1)          # socket over the mast end
    out.append(("lamp_holder", holder, PRINT_B, True))
    out.append(("lamp_led_disk", cyl_z(P.MAST_X, 0, zm + 40, zm + 46, 60), (1.0, 0.93, 0.75), False))
    shade = Pos(P.MAST_X, 0, zm + 60) * Cone(P.LAMP_D / 2, P.LAMP_D / 2 - 45, 150)
    shade -= Pos(P.MAST_X, 0, zm + 60) * Cone(P.LAMP_D / 2 - 2, P.LAMP_D / 2 - 47, 151)
    out.append(("lamp_shade", shade, LAMP, False))
    # tool balancer hanging behind the mast, its cable runs down to the carriage
    bz = zm - 60
    hanger = box(mx0 - 40, mx0, -12, 12, bz + 30, bz + 60) - box(mx0 - 30, mx0 - 10, -13, 13, bz + 38, bz + 52)
    out.append(("balancer_hanger", hanger, PRINT_B, True))
    out.append(("spring_balancer_3_5kg", cyl_x(0, bz, mx0 - 75, mx0 - 5, 95), (0.85, 0.25, 0.15), False))
    return out


# ------------------------------------------------------------------ carriage link
SHOULDER_HOUSING_Z = (P.L1_ABOVE_TRAY + 14, P.L1_ABOVE_TRAY + 62)
J1_BELT_Z = SHOULDER_HOUSING_Z[1] + 6       # pulley bottom
J1_C = 58.6                                  # centre distance for a 200 mm GT2 loop, 20T/60T


def j1_servo_axis():
    """J1 servo axis for the right arm, relative to its shoulder in y: straight outboard, clear of the shoulder beam."""
    return (P.SHOULDER_X, -J1_C)


def link_carriage():
    out = []
    x0 = P.CARRIAGE_X0
    x1 = x0 + P.CARRIAGE_T
    ztop = SHOULDER_HOUSING_Z[1] - 5
    plate = box(x0, x1, -80, 80, -28, ztop)
    for zc in (35, 35 + P.LIFT_BLOCK_SPACING):
        for dy in (-10, 10):
            for dz in (-10, 10):
                plate -= cyl_x(dy, zc + dz, x0 - 1, x1 + 1, M3)
    plate -= box(x0 - 1, x1 + 1, -30, 30, 70, 140)  # weight-saving window / cable pass
    zb = sum(SHOULDER_HOUSING_Z) / 2
    for yb in (-60, -20, 20, 60):
        for dz in (-10, 10):
            plate -= cyl_x(yb, zb + dz, x0 - 1, x1 + 1, M5)    # shoulder beam bolts
    # belt clamp tab reaching back beside the mast to the front strand of the lift belt,
    # and an eye for the balancer cable behind the mast
    tab = box(P.MAST_X + 2, x1, -30, -15, ztop - 35, ztop)
    plate += tab
    eye = box(P.MAST_X - P.MAST_W_X / 2 - 34, x0, 14, 26, ztop - 20, ztop) - cyl_y(P.MAST_X - P.MAST_W_X / 2 - 24, ztop - 10, 13, 27, 6)
    plate += eye
    out.append(("carriage_plate", plate, PRINT_A, True))
    for zc in (35, 35 + P.LIFT_BLOCK_SPACING):
        out.append(("mgn12h_block", Pos(x0 - P.BLOCK_H - 5 + 5, 0, zc) * (Rot(0, 90, 0) * Rot(0, 0, 90) * Box(45.4, 27, P.BLOCK_H + 5, align=(CEN, CEN, MAX))), STEEL, False))

    # tray frame: 2020 yoke behind the drawer + two 2020 side arms carrying full-extension slides
    ay = P.ARM_RAIL_Y
    arm_x1 = P.TRAY_X0 + P.TRAY_DEPTH
    out.append(("yoke_2020", Pos(P.YOKE_X0 + 10, -ay - 10, -15) * extrusion(2 * ay + 20, along="y"), ALU, False))
    for sgn in (1, -1):
        out.append(("tray_arm_2020", Pos(P.YOKE_X0 + 20, sgn * ay, -15) * extrusion(arm_x1 - P.YOKE_X0 - 20), ALU, False))
        # slide, outer member (250 mm, 12.7 x 27): visual
        yi = sgn * (P.TRAY_WIDTH / 2 + 6.35)
        out.append(("slide_250_outer", box(P.TRAY_X0, arm_x1, yi - 6.35, yi + 6.35, -25, 2), STEEL, False))
        sp = box(P.TRAY_X0 + 5, arm_x1 - 5, sgn * (P.TRAY_WIDTH / 2 + 12.7), sgn * (ay - 10), -25, 2)
        out.append(("slide_spacer", sp, PRINT_B, True))
    corner = box(x1, x1 + 30, -80, 80, -30, -5) - Pos(0, 0, 0) * box(x1 + 2, x1 + 31, -60, 60, -26, -4)
    out.append(("yoke_bracket", corner, PRINT_B, True))
    # drawer drive: STS3215 in wheel mode under the right arm, GT2 loop along the arm
    dloc = Location((P.YOKE_X0 + 45, -ay, -25 - 2), (180, 0, 0))
    out.append(("servo_drawer", servo(dloc), SERVO, False))
    out.append(("pulley_20T_drawer", Pos(P.YOKE_X0 + 45, -ay, -25 - 2 - 7) * gt2_pulley(20), ALU, False))
    out.append(("idler_drawer", Pos(arm_x1 - 15, -ay, -25 - 2 - 7) * gt2_pulley(20), ALU, False))
    out.append(("belt_drawer", belt_loop((P.YOKE_X0 + 45, -ay), (arm_x1 - 15, -ay), P.gt2_pd(20) / 2, P.gt2_pd(20) / 2, -25 - 2 - 7), RUBBER, False))
    dm = box(P.YOKE_X0 + 10, P.YOKE_X0 + 60, -ay - 16, -ay + 16, -32, -25) - servo_pocket(dloc)
    out.append(("drawer_servo_mount", dm, PRINT_B, True))

    # 2040 shoulder beam across the plate; both shoulder housings and J1 brackets clamp onto it
    zh0, zh1 = SHOULDER_HOUSING_Z
    beam_len = 2 * (max(abs(y) for y in P.ARM_Y) + 70)
    zb = (zh0 + zh1) / 2
    out.append(("shoulder_beam_2040", Pos(x1 + 10, -beam_len / 2, zb) * extrusion(beam_len, w=20, h=40, along="y"), ALU, False))
    arm_a = carriage_arm_parts(P.ARM_Y[0])
    out += [(f"{lab}", shp, rgb, pr) for lab, shp, rgb, pr in arm_a]
    out += [(f"{lab}_b" if pr else lab, shp.mirror(Plane.XZ), rgb, pr) for lab, shp, rgb, pr in arm_a]

    # face: the old phone, upright in front of the plate top, above the arms' sweep
    fz = P.FACE_Z_ABOVE_TRAY
    fw, fh, ft = P.FACE_W, P.FACE_H, P.FACE_T
    mount = box(x0, x1 + ft + 6, -fw / 2 - 5, fw / 2 + 5, ztop - 30, fz + fh / 2 + 4)
    mount -= box(x1 + 2, x1 + ft + 7, -fw / 2 - 0.5, fw / 2 + 0.5, fz - fh / 2 - 0.5, fz + fh / 2 + 5)  # phone slot
    mount -= box(x1 + ft + 1, x1 + ft + 7, -fw / 2 + 4, fw / 2 - 4, fz - fh / 2 + 6, fz + fh / 2 + 5)    # screen window
    mount -= box(x0 - 1, x1 + 1, -fw / 2 + 8, fw / 2 - 8, ztop - 31, ztop + 1)                           # over the plate top
    out.append(("face_mount", mount, PRINT_B, True))
    out.append(("face_phone", box(x1 + 2, x1 + 2 + ft, -fw / 2, fw / 2, fz - fh / 2, fz + fh / 2), PHONE, False))
    for s_ in (1, -1):
        out.append(("face_eye", cyl_x(s_ * 16, fz + 12, x1 + 2 + ft, x1 + 3 + ft, 18), (0.95, 0.85, 0.60), False))
    return out


def carriage_arm_parts(ys):
    """Shoulder housing (two 6000 bearings) and the J1 servo + 20T/60T belt for the arm whose shoulder is at y = ys.
    Built for the right arm (ys < 0); the left arm is its mirror image."""
    out = []
    x1 = P.CARRIAGE_X0 + P.CARRIAGE_T
    xb = x1 + 20                                   # front face of the shoulder beam
    zh0, zh1 = SHOULDER_HOUSING_Z
    sx = P.SHOULDER_X
    hous = box(xb, sx + 22, ys - 26, ys + 26, zh0, zh1)
    hous += box(xb, xb + 6, ys - 45, ys + 45, zh0, zh1)
    hous -= cyl_z(sx, ys, zh0 - 1, zh1 + 1, 14)
    hous -= cyl_z(sx, ys, zh0 - 1, zh0 + 8, 26.1)
    hous -= cyl_z(sx, ys, zh1 - 8, zh1 + 1, 26.1)
    for yh in (ys - 36, ys + 36):
        for zh in (zh0 + 14, zh1 - 14):
            hous -= cyl_x(yh, zh, xb - 1, xb + 7, M5)       # into T-nuts in the beam
    out.append(("shoulder_housing", hous, PRINT_A, True))
    out.append(("bearing_6000", Pos(sx, ys, 0) * bearing_6000(zh0), STEEL, False))
    out.append(("bearing_6000", Pos(sx, ys, 0) * bearing_6000(zh1 - 8), STEEL, False))

    # J1 servo on a slotted bracket, outboard of the shoulder, horn up, 20T driving the 60T on the shoulder shaft
    sxs, dy = j1_servo_axis()
    sys_ = ys + dy                                 # dy < 0: outboard for the right arm
    horn_z = J1_BELT_Z - 2
    loc = Location((sxs, sys_, horn_z), (0, 0, 90))  # servo body points further outboard
    out.append(("servo_J1", servo(loc), SERVO, False))
    brk = box(xb, sxs + 16, sys_ - 40, sys_ + 14, horn_z - 44, horn_z - 38)          # floor under servo
    brk += box(xb, xb + 6, sys_ - 40, sys_ + 14, horn_z - 44, horn_z - 2)              # back wall on the beam
    brk -= servo_pocket(loc)
    for zh in (horn_z - 30, horn_z - 14):
        brk -= box(xb - 1, xb + 7, sys_ - 14 - 2, sys_ - 14 + 2 + 6, zh - M4 / 2, zh + M4 / 2)  # tension slots
    out.append(("j1_servo_bracket", brk, PRINT_B, True))
    out.append(("pulley_20T_J1", Pos(sxs, sys_, horn_z) * gt2_pulley(20), ALU, False))
    out.append(("belt_J1_200", belt_loop((sxs, sys_), (sx, ys), P.gt2_pd(20) / 2, P.gt2_pd(60) / 2, horn_z), RUBBER, False))
    return out


# ------------------------------------------------------------------ drawer (the tray)
def link_drawer():
    """Tray on full-extension slides. Retracted position (drawer = 0) in the carriage frame."""
    out = []
    tx0, tx1 = P.TRAY_X0, P.TRAY_X0 + P.TRAY_DEPTH
    for s in (1, -1):
        y_out = s * P.TRAY_WIDTH / 2
        ya, yb = sorted((0, y_out))
        half = box(tx0, tx1, ya, yb, -P.TRAY_T, 0)
        lip_y0, lip_y1 = sorted((y_out, y_out - s * 5))
        half += box(tx0, tx1 - 25, lip_y0, lip_y1, 0, P.TRAY_LIP)                   # side lip
        half += box(tx0, tx0 + 5, ya, yb, 0, P.TRAY_LIP)                             # back lip
        half += box(tx0 + 5, tx1 - 5, lip_y0, lip_y1, -25, -P.TRAY_T)                # slide skirt
        wedge = Pos(tx1, ya - 1, -P.TRAY_T) * (Rot(0, -12, 0) * box(-40, 0, 0, yb - ya + 2, -10, 0))
        half -= wedge                                                                # thin leading edge
        for xh in (tx0 + 40, tx0 + 120, tx0 + 200):
            half -= cyl_z(xh, s * 12, -P.TRAY_T - 1, 1, M3)                           # seam strap holes
        for xh in (tx0 + 30, tx0 + 125, tx0 + 220):
            half -= cyl_y(xh, -15, lip_y0 - 1, lip_y1 + 1, M4)                        # slide screws
        out.append((f"tray_half_{'left' if s > 0 else 'right'}", half, PRINT_A, True))
        yi = s * (P.TRAY_WIDTH / 2 + 6.35)
        out.append(("slide_250_inner", box(tx0 + 3, tx1 - 3, yi - 3, yi + 3, -22, -1), STEEL, False))
    out.append(("tray_seam_strap", box(tx0 + 25, tx1 - 40, -22, 22, -P.TRAY_T - 3, -P.TRAY_T), PRINT_B, True))
    out.append(("tray_belt_clamp", box(tx0 + 10, tx0 + 40, -P.ARM_RAIL_Y - 4, -P.TRAY_WIDTH / 2 + 2, -40, -25), PRINT_B, True))
    return out


# ------------------------------------------------------------------ upper arm (L1)
J2_C = 69.7     # 200 mm loop, 20T/40T


def link_upper_arm():
    out = []
    # shaft fixed in the root clamp, through the shoulder bearings, 60T on top
    z_rel_belt = (J1_BELT_Z - 2) - P.L1_ABOVE_TRAY
    out.append(("shaft_10_shoulder", cyl_z(0, 0, -12, z_rel_belt + 10, SHAFT_D), STEEL, False))
    out.append(("pulley_60T_J1", Pos(0, 0, z_rel_belt) * gt2_pulley(60), ALU, False))
    root = box(-22, 45, -16, 16, -12, 12) - cyl_z(0, 0, -13, 13, SHAFT_D + 0.2)
    root -= box(20, 46, -10.15, 10.15, -10.15, 10.15)                        # 2020 socket
    root -= box(-23, -6, -1, 1, -13, 13)                                      # pinch slot
    root -= cyl_y(-15, 0, -17, 17, M4)
    out.append(("l1_root_clamp", root, PRINT_A, True))
    out.append(("l1_2020", Pos(20, 0, 0) * extrusion(P.L1 - 40), ALU, False))
    # elbow block with two 6000 bearings; L2 hangs 40 mm lower
    ex = P.L1
    blk = box(ex - 45, ex + 22, -18, 18, -26, 12) + cyl_z(ex, 0, -26, 26, 36)
    blk -= box(ex - 46, ex - 20, -10.15, 10.15, -10.15, 10.15)
    blk -= cyl_z(ex, 0, -27, 27, 14)
    blk -= cyl_z(ex, 0, -27, -18, 26.1)
    blk -= cyl_z(ex, 0, 18, 27, 26.1)
    out.append(("elbow_block", blk, PRINT_A, True))
    out.append(("bearing_6000", Pos(ex, 0, 0) * bearing_6000(-26), STEEL, False))
    out.append(("bearing_6000", Pos(ex, 0, 0) * bearing_6000(18), STEEL, False))
    # J2 servo saddle on top of L1, horn up
    horn_z = 10 + 38.5
    sx = ex - J2_C
    loc = Location((sx, 0, horn_z), (0, 0, 180))
    out.append(("servo_J2", servo(loc), SERVO, False))
    saddle = box(sx - 16, sx + 40, -16, 16, 10, 16) + box(sx - 16, sx + 40, -16, -11, -10, 16) + box(sx - 16, sx + 40, 11, 16, -10, 16)
    saddle -= box(sx - 17, sx + 41, -13.1, 13.1, 9, 12)  # seat
    for xh in (sx - 6, sx + 30):
        saddle -= box(xh - M5 / 2 - 3, xh + M5 / 2 + 3, -2.75, 2.75, 9, 17)  # tension slot down to the T-nut
    out.append(("j2_servo_saddle", saddle, PRINT_B, True))
    out.append(("pulley_20T_J2", Pos(sx, 0, horn_z) * gt2_pulley(20), ALU, False))
    out.append(("pulley_40T_J2", Pos(ex, 0, horn_z) * gt2_pulley(40), ALU, False))
    out.append(("belt_J2_200", belt_loop((sx, 0), (ex, 0), P.gt2_pd(20) / 2, P.gt2_pd(40) / 2, horn_z), RUBBER, False))
    return out


# ------------------------------------------------------------------ forearm (L2)
def link_forearm():
    out = []
    dz = P.L1_ABOVE_TRAY - P.L2_ABOVE_TRAY      # elbow axis rises to the L1 plane
    out.append(("shaft_10_elbow", cyl_z(0, 0, -12, dz + 38.5 + 10 + 10, SHAFT_D), STEEL, False))
    root = box(-22, 45, -16, 16, -12, 12) - cyl_z(0, 0, -13, 13, SHAFT_D + 0.2)
    root -= box(20, 46, -10.15, 10.15, -10.15, 10.15)
    root -= box(-23, -6, -1, 1, -13, 13)
    root -= cyl_y(-15, 0, -17, 17, M4)
    out.append(("l2_root_clamp", root, PRINT_A, True))
    out.append(("l2_2020", Pos(20, 0, 0) * extrusion(P.L2 - 50), ALU, False))
    # wrist block: L2 socket + hanging yaw servo (horn down)
    wx = P.L2
    blk = box(wx - 45, wx + 16, -16, 16, -12, 12)
    blk -= box(wx - 46, wx - 30, -10.15, 10.15, -10.15, 10.15)
    loc = Location((wx, 0, -48), (180, 0, 0))
    blk = box(wx - 45, wx + 14, -16, 16, -10, 12)
    blk -= box(wx - 46, wx - 30, -10.15, 10.15, -10.15, 10.15)
    blk += box(wx - 38, wx + 14, -16, -13, -48 + 6, -10) + box(wx - 38, wx + 14, 13, 16, -48 + 6, -10)  # cheeks
    blk -= servo_pocket(Location((wx, 0, -48), (180, 0, 0)))
    out.append(("wrist_block", blk, PRINT_A, True))
    out.append(("servo_J3_yaw", servo(loc), SERVO, False))
    return out


# ------------------------------------------------------------------ wrist (yaw output)
def link_wrist():
    out = []
    zh = -48
    zp = -P.PITCH_DROP
    # horn plate + bracket that carries the pitch servo (axis along y, horn toward +y)
    loc = Location((0, -3.5 + 16 - 16, zp), (-90, 0, 0))  # local z -> world +y
    loc = Location((0, 16.0, zp), (-90, 0, 0))
    br = cyl_z(0, 0, zh - 5, zh, 26)
    br += box(-36, 11, -22.5, 13.5, zp + 12.6, zh - 3)
    br -= servo_pocket(loc)
    out.append(("wrist_bracket", br, PRINT_B, True))
    out.append(("servo_J4_pitch", servo(loc), SERVO, False))
    return out


# ------------------------------------------------------------------ hand (pitch output) + roll stage
def roll_servo_loc():
    """Roll STS3215 in the hand frame: output axis along +x through the pitch axis, horn face at ROLL_HORN_X,
    body behind the horn and toward -y."""
    return Location((P.ROLL_HORN_X, 0, 0)) * Location((0, 0, 0), (0, 90, 0)) * Location((0, 0, 0), (0, 0, 90))


def link_hand():
    out = []
    # side plates on the pitch horn (+y) and idler (-y), joined by a cradle around the roll servo
    plate_hs = box(-12, 34, 17, 21, -14, 14) + cyl_y(0, 0, 17, 21, 28)
    plate_id = box(-12, 34, -41, -37, -14, 14) + cyl_y(0, 0, -41, -37, 28)
    loc = roll_servo_loc()
    cradle = box(P.ROLL_BRACKET, P.ROLL_HORN_X - 4, -41, 21, -16, 16) - servo_pocket(loc)
    cradle -= cyl_x(0, 0, P.ROLL_HORN_X - 6, P.ROLL_HORN_X + 1, 24)          # horn clearance
    out.append(("roll_bracket", plate_hs + plate_id + cradle, PRINT_A, True))
    out.append(("servo_J5_roll", servo(loc), SERVO, False))
    return out


# ------------------------------------------------------------------ tool (roll output): CAP gripper + iPhone
CAP_LOC = Location((P.CAP_ORIGIN_X, 0, -P.CAP_AXIS_Z), (0, 0, -90))   # CAP frame (x lateral, y forward) in the tool frame


def cap_box(x0, x1, y0, y1, z0, z1):
    """Box given in the CAP gripper's own frame (from EgoGym's model, mm), placed in the tool frame."""
    return CAP_LOC * box(x0, x1, y0, y1, z0, z1)


# CAP bottom plate mounting holes, measured from its published STL (third_party/cap_gripper/bottom-plate.stl):
# 2x M3 through the bottom face with hex-nut traps above, 52.1 mm apart, 27.0 mm in front of the back face.
# The back face itself has no holes. In the CAP frame: x = +-26.05, y = -64.3 + 27.0, bottom face z = -5.3.
CAP_HOLES = [(-26.05, -37.3), (26.05, -37.3)]
CAP_BOTTOM_Z = -5.3


def link_tool():
    out = []
    # printed L-bracket: roll horn (4x M2.5 on r = 7 mm) -> a tongue under the CAP bottom plate, 2x M3 from below
    # into the plate's own nut traps (screw heads in counterbores)
    xa0, xa1 = P.ROLL_HORN_X, P.ROLL_HORN_X + P.CAP_ADAPTER_T
    z_bot = -P.CAP_AXIS_Z + CAP_BOTTOM_Z                     # gripper bottom face in the tool frame
    tongue_t = 5
    x_holes = P.CAP_ORIGIN_X + CAP_HOLES[0][1]
    plate = box(xa0, xa1, -32.5, 32.5, z_bot - tongue_t, 18)
    plate += box(xa0, x_holes + 9, -32.5, 32.5, z_bot - tongue_t, z_bot)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        plate -= cyl_x(7 * math.cos(a), 7 * math.sin(a), xa0 - 1, xa1 + 1, 2.8)
    for cx, cy in CAP_HOLES:                                  # CAP (x, y) -> tool (CAP_ORIGIN_X + y, -x)
        plate -= cyl_z(P.CAP_ORIGIN_X + cy, -cx, z_bot - tongue_t - 1, z_bot + 1, 3.4)
        plate -= cyl_z(P.CAP_ORIGIN_X + cy, -cx, z_bot - tongue_t - 1, z_bot - tongue_t + 3, 6.2)
    plate -= cyl_x(0, 0, xa0 - 1, xa1 + 1, 6)                                   # cable pass (XL330 + iPhone USB)
    out.append(("cap_adapter_plate", plate, PRINT_A, True))
    # the CAP robot gripper: body with its XL330 inside, phone holder, iPhone (landscape, camera near the centreline)
    out.append(("cap_gripper_body", cap_box(-32.5, 32.5, -64.3, 48.7, -5.3, 33.0), CAP_WHITE, False))
    out.append(("cap_phone_holder", cap_box(-26.0, 16.0, -64.3, -30.0, 33.0, 142.0), CAP_WHITE, False))
    out.append(("iphone", cap_box(-31.8, 114.9, -41.0, -32.5, 43.6, 114.1), PHONE, False))
    out.append(("iphone_camera", cap_box(-8.0, 8.0, -32.5, -29.5, 92.0, 108.0), (0.05, 0.05, 0.05), False))
    return out


def link_finger(side: int):
    """CAP finger (angular jaw in reality) as a parallel stand-in, in the tool frame at grip = 0 (closed).
    Pads cover CAP y 40..135 mm, i.e. tool x = PAD_X +/- 47 mm; inner face at +/-12 mm."""
    s = side
    ya, yb = sorted((s * 12, s * 24))
    x0 = P.CAP_ORIGIN_X + 24
    x1 = P.CAP_ORIGIN_X + 131
    finger = box(x0, x1, ya, yb, -15, 15)
    return [(f"cap_finger_{'l' if s > 0 else 'r'}", finger, CAP_BLUE, False)]


# ------------------------------------------------------------------ assembly
ARMS = ("a", "b")
DEFAULT_Q = dict(x=0, y=0, yaw=0, lift=400, drawer=0,
                 j1=0, j2=0, j3=0, j4=0, j5=0, grip=20,
                 j1b=0, j2b=0, j3b=0, j4b=0, j5b=0, gripb=20)


def arm_key(name, arm):
    """Joint name for an arm: arm a keeps the v0 names (j1 .. j5, grip), arm b adds a 'b'."""
    return name if arm == "a" else name + "b"


def link_transforms(q):
    q = {**DEFAULT_Q, **q}
    T_base = Location((q["x"], q["y"], 0), (0, 0, q["yaw"]))
    T_car = T_base * Location((0, 0, q["lift"]))
    T = dict(base=T_base, carriage=T_car, drawer=T_car * Location((q["drawer"], 0, 0)))
    for arm, ys in zip(ARMS, P.ARM_Y):
        k = lambda n: q[arm_key(n, arm)]
        T_ua = T_car * Location((P.SHOULDER_X, ys, P.L1_ABOVE_TRAY)) * Location((0, 0, 0), (0, 0, k("j1")))
        T_fa = T_ua * Location((P.L1, 0, P.L2_ABOVE_TRAY - P.L1_ABOVE_TRAY)) * Location((0, 0, 0), (0, 0, k("j2")))
        T_wr = T_fa * Location((P.L2, 0, 0)) * Location((0, 0, 0), (0, 0, k("j3")))
        T_hand = T_wr * Location((0, 0, -P.PITCH_DROP)) * Location((0, 0, 0), (0, k("j4"), 0))
        T_tool = T_hand * Location((0, 0, 0), (k("j5"), 0, 0))
        T.update({f"upper_arm_{arm}": T_ua, f"forearm_{arm}": T_fa, f"wrist_{arm}": T_wr, f"hand_{arm}": T_hand,
                  f"tool_{arm}": T_tool, f"finger_l_{arm}": T_tool * Location((0, k("grip"), 0)),
                  f"finger_r_{arm}": T_tool * Location((0, -k("grip"), 0))})
    return T


@lru_cache(maxsize=1)
def links():
    out = dict(base=link_base(), carriage=link_carriage(), drawer=link_drawer())
    per_arm = dict(upper_arm=link_upper_arm(), forearm=link_forearm(), wrist=link_wrist(), hand=link_hand(),
                   tool=link_tool(), finger_l=link_finger(1), finger_r=link_finger(-1))
    for arm in ARMS:
        for name, items in per_arm.items():
            out[f"{name}_{arm}"] = items
    return out


def assembly(q=None):
    T = link_transforms(q or {})
    parts = []
    for name, items in links().items():
        for label, shape, rgb, printable in items:
            s = T[name] * shape
            s.label = label
            s.color = Color(*rgb)
            parts.append(s)
    return parts


def fingertip(q, arm="a"):
    """Grasp centre of an arm (CAP: 140 mm in front of the gripper origin)."""
    T = link_transforms(q)[f"tool_{arm}"]
    return (T * Location((P.WRIST_TO_TIP, 0, 0))).position
