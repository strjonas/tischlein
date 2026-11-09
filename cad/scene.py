"""Household props and the poses used for renders and the MuJoCo demo."""
from __future__ import annotations

import math

from build123d import Color, Location

import robot as R
from robot import P, box, cyl_x, cyl_y, cyl_z

WHITE = (0.92, 0.92, 0.90)
GLASS = (0.75, 0.88, 0.92)
FABRIC = (0.55, 0.62, 0.70)
WOOD_D = (0.55, 0.40, 0.28)
FOOD = (0.95, 0.85, 0.35)
BOX = (0.85, 0.30, 0.25)


def colored(shape, rgb, label):
    shape.color = Color(*rgb)
    shape.label = label
    return shape


# ------------------------------------------------------------------ IK helpers
def arm_ik(target_xy, elbow=1, hand_yaw=0.0, arm="a"):
    """Wrist-axis target in the carriage frame -> (j1, j2, j3) in degrees.
    hand_yaw: desired hand heading in the carriage frame (deg)."""
    x = target_xy[0] - P.SHOULDER_X
    y = target_xy[1] - P.ARM_Y[0 if arm == "a" else 1]
    r2 = x * x + y * y
    c2 = (r2 - P.L1**2 - P.L2**2) / (2 * P.L1 * P.L2)
    c2 = max(-1.0, min(1.0, c2))
    q2 = elbow * math.acos(c2)
    q1 = math.atan2(y, x) - math.atan2(P.L2 * math.sin(q2), P.L1 + P.L2 * math.cos(q2))
    j1, j2 = math.degrees(q1), math.degrees(q2)
    return j1, j2, hand_yaw - j1 - j2


def pose_for_tip(tip_x, tip_y, tip_z, pitch=0.0, elbow=1, hand_yaw=0.0, grip=20, base=(0, 0, 0), reach_to=None):
    """Grasp-centre target for arm a in the robot frame -> full joint dict (arm b parked). pitch 0 = forward, 90 = down.
    reach_to: distance pitch axis -> target point (default: the grasp centre)."""
    L = reach_to if reach_to is not None else P.WRIST_TO_TIP
    th = math.radians(hand_yaw)
    reach = L * math.cos(math.radians(pitch))
    wx = tip_x - reach * math.cos(th)
    wy = tip_y - reach * math.sin(th)
    pitch_axis_z = tip_z + L * math.sin(math.radians(pitch))
    lift = pitch_axis_z - (P.L2_ABOVE_TRAY - P.PITCH_DROP)
    j1, j2, j3 = arm_ik((wx, wy), elbow, hand_yaw)
    return dict(x=base[0], y=base[1], yaw=base[2], lift=lift, drawer=0, j1=j1, j2=j2, j3=j3, j4=pitch, j5=0, grip=grip,
                **PARK_B)


def park(arm):
    """Arm folded back beside the carriage, hand hanging down outboard of the tray arms and behind the tray front,
    so it clears both the tray load and a bed edge. Needs the tray >= ~0.3 m high (fingertips 0.24 m below it)."""
    ys = P.ARM_Y[0 if arm == "a" else 1]
    sgn = 1 if ys > 0 else -1
    j1, j2, j3 = arm_ik((P.SHOULDER_X - 30, ys + sgn * 110), elbow=sgn, hand_yaw=0.0, arm=arm)
    k = "" if arm == "a" else "b"
    return {f"j1{k}": j1, f"j2{k}": j2, f"j3{k}": j3, f"j4{k}": 90, f"j5{k}": 0, f"grip{k}": 4}


PARK_A = park("a")
PARK_B = park("b")
PARK = {**PARK_A, **PARK_B}


def tuck(arm):
    """Idle arm out of the other arm's way (26 Oct, sim/fridge_door.py): upper arm pointing back along its side of
    the carriage, forearm forward, hand hanging at x ~0.19 m. Both arms move in the same horizontal planes, so the
    park pose above (elbow 35 cm ahead of the shoulder) blocks the other arm's work space in front. The hand's heading
    keeps the iPhone (11.5 cm to the hand's right) outboard: forward for the right arm, backward for the left."""
    k = "" if arm == "a" else "b"
    sgn = -1 if arm == "a" else 1
    return {f"j1{k}": sgn * 150.0, f"j2{k}": -sgn * 150.0, f"j3{k}": 0.0 if arm == "a" else 179.0,
            f"j4{k}": 90, f"j5{k}": 0, f"grip{k}": 4}


TUCK_A = tuck("a")
TUCK_B = tuck("b")


# ------------------------------------------------------------------ props
def floor(x0=-1500, x1=2500, y0=-1800, y1=1800):
    return [colored(box(x0, x1, y0, y1, -10, 0), (0.80, 0.78, 0.74), "floor")]


def fridge(front_x=500, door_deg=-150, shelves=(950, 1150, 1350, 1550)):
    """Fridge-freezer, 600 x 650 x 1850, opening toward -x, hinge on the +y side."""
    w, d, h = 600, 650, 1850
    out = []
    shell = box(front_x, front_x + d, -w / 2, w / 2, 0, h)
    shell -= box(front_x - 1, front_x + d - 40, -w / 2 + 35, w / 2 - 35, 880, h - 40)
    out.append(colored(shell, WHITE, "fridge_shell"))
    out.append(colored(box(front_x - 70, front_x, -w / 2, w / 2, 20, 840), WHITE, "freezer_door"))
    for z in shelves:
        out.append(colored(box(front_x + 40, front_x + d - 45, -w / 2 + 35, w / 2 - 35, z - 5, z), GLASS, "glass_shelf"))
        out.append(colored(box(front_x + 36, front_x + 46, -w / 2 + 35, w / 2 - 35, z - 9, z + 6), WHITE, "shelf_trim"))
    door = box(-70, 0, -w, 0, 880, h - 20)
    door -= box(-60, -1, -w + 25, -25, 900, h - 40)
    for z in (1000, 1250, 1500):
        door += box(-65, -10, -w + 30, -30, z, z + 4)
        door += box(-65, -55, -w + 30, -30, z, z + 60)
    door += box(-90, -70, -w + 20, -w + 40, 1050, 1300)   # vertical handle bar
    hinge = Location((front_x, w / 2, 0), (0, 0, door_deg))
    out.append(colored(hinge * door, WHITE, "fridge_door"))
    return out


MEAL_KNOB = 18   # knob depth in front of the box


def meal_box(x, y, z, label="meal_box"):
    """180 x 120 x 70 food box with a 30 mm grip knob on its front face (the one convention we ask for)."""
    b = box(x, x + 180, y - 60, y + 60, z, z + 70) - box(x + 4, x + 176, y - 56, y + 56, z + 4, z + 72)
    lid = box(x - 2, x + 182, y - 62, y + 62, z + 66, z + 72)
    knob = cyl_x(y, z + 35, x - MEAL_KNOB, x, 30) + cyl_x(y, z + 35, x - 4, x, 40)
    return [colored(b + knob, BOX, label), colored(lid, (0.95, 0.95, 0.95), label + "_lid"),
            colored(box(x + 10, x + 170, y - 50, y + 50, z + 4, z + 40), FOOD, label + "_food")]


def bottle(x, y, z, label="bottle"):
    b = cyl_z(x, y, z, z + 190, 66) + cyl_z(x, y, z + 190, z + 225, 30) + cyl_z(x, y, z + 225, z + 240, 32)
    return [colored(b, (0.35, 0.70, 0.95), label)]


def phone(x, y):
    return [colored(box(x - 37, x + 37, y - 78, y + 78, 0, 9), (0.05, 0.05, 0.06), "phone")]


def bed(edge_x=330, y0=-1100, y1=900, top=550):
    out = [colored(box(edge_x, edge_x + 950, y0, y1, 280, top - 180), WOOD_D, "bed_frame")]
    for lx in (edge_x + 20, edge_x + 900):
        for ly in (y0 + 20, y1 - 50):
            out.append(colored(box(lx, lx + 30, ly, ly + 30, 0, 280), WOOD_D, "bed_leg"))
    out.append(colored(box(edge_x, edge_x + 950, y0, y1, top - 180, top), (0.93, 0.93, 0.95), "mattress"))
    out.append(colored(box(edge_x + 20, edge_x + 900, y0 + 20, y1 - 380, top, top + 40), FABRIC, "duvet"))
    out.append(colored(box(edge_x + 200, edge_x + 750, y1 - 330, y1 - 40, top, top + 120), (0.97, 0.97, 0.97), "pillow"))
    return out


def bedside_table(x0, y0, h=580):
    t = box(x0, x0 + 400, y0, y0 + 420, h - 25, h)
    t += box(x0, x0 + 400, y0, y0 + 420, 0, 60)
    t += box(x0 + 370, x0 + 400, y0, y0 + 420, 60, h - 25)
    return [colored(t, WOOD_D, "bedside_table")]


# ------------------------------------------------------------------ scenes
FRIDGE_FRONT = 720 - P.AXLE_SHIFT   # robot-frame x of the fridge front when the drawer is out
SHELF = 950


def scene_hero():
    q = dict(lift=520, drawer=0, **PARK)
    props = floor() + bottle(P.TRAY_X0 + 200, -80, 520) + meal_box(P.TRAY_X0 + 30, 10, 520)
    return q, props


def scene_fridge():
    """Mid-transfer: drawer out at shelf height, the right hand pulls the box across by its knob."""
    bx = 640 - P.AXLE_SHIFT                     # box front face during the pull
    q = pose_for_tip(bx - MEAL_KNOB / 2, 0, SHELF + 38, pitch=0, elbow=-1, grip=3, reach_to=P.PAD_X)
    q["drawer"] = P.DRAWER_STROKE
    props = floor() + fridge(front_x=FRIDGE_FRONT, door_deg=-150)
    props += meal_box(bx, 0, SHELF)
    props += bottle(FRIDGE_FRONT + 160, 190, SHELF)
    props += bottle(FRIDGE_FRONT + 300, -170, SHELF)
    return q, props


def scene_bed():
    lift = 640
    edge = 360 - P.AXLE_SHIFT
    q = dict(lift=lift, drawer=P.DRAWER_STROKE, **PARK)
    props = floor() + bed(edge_x=edge) + bedside_table(edge - 430, 620)
    props += meal_box(P.TRAY_X0 + P.DRAWER_STROKE + 50, -20, lift)
    props += bottle(P.TRAY_X0 + P.DRAWER_STROKE + 185, 85, lift)
    return q, props


def scene_floor():
    px = 646 - P.AXLE_SHIFT
    q = pose_for_tip(px, -40, 9 + 30, pitch=90, elbow=1, grip=44, reach_to=P.PAD_X)
    q["drawer"] = 0
    props = floor() + phone(px, -40)
    return q, props


SCENES = dict(hero=scene_hero, fridge=scene_fridge, bed=scene_bed, floor=scene_floor)
