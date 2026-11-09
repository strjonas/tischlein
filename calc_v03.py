"""Hand checks for the v0.3 proposal's three configurations: does each one stand on its own? -> out/calc_v03.txt

uv run python calc_v03.py

Static tipping about the front feet (forward) and about the feet on one side (sideways), per configuration. Static moments
don't depend on the height; braking and fast arm moves do (calc.py has the braking cases). Masses from
params.M, changed where v0.3 differs: the hand carries the infrared camera instead of an iPhone, the left arm is a
stump, and configurations 1-2 stand on fixed feet with optional ballast instead of wheels and battery.
Safety factor = restoring moment / overturning moment; >= 1.5 OK, 1.1-1.5 marginal, < 1.1 tips.
"""
from __future__ import annotations

from pathlib import Path

import params as P

ROOT = Path(__file__).resolve().parent
A = P.AXLE_SHIFT
M = P.M
HAND = M["arm_shoulder"] + M["arm_L1"] + M["arm_L2"] + M["wrist_gripper"] - 0.20 + 0.06   # iPhone 13 Pro -> camera
STUMP = M["arm_shoulder"] + M["arm_L1"] + M["arm_L2"] + 0.10                                # yaw servo + printed head
FRONT_X = P.FRONT_CASTER_X           # the stand's front feet sit where the front casters go
SIDE_Y = min(P.TRACK / 2 - 20, P.FRONT_CASTER_Y)   # narrowest foot line sideways (rear feet on the frame rails)


def stage(order, ballast=0.0):
    """Fixed masses (kg, x, y, z in mm) for a configuration, without the carriage group."""
    out = [("base frame + deck", M["base_frame"], -40 - A, 0, 80),
           ("mast", 1.2 if order == 1 else M["mast"], P.MAST_X, 0, 930)]
    if order >= 2:
        out.append(("lift motor + belts", 0.5, P.MAST_X - 40, -75, 200))
    if order >= 3:
        out += [("hub motors", 2 * M["hub_motor"], 0, 0, P.WHEEL_D / 2), ("LiFePO4", 2.6, -150 - A, 0, 70),
                ("electronics", M["electronics"], -170 - A, 0, 120), ("lidar", M["lidar"], -120 - A, 0, 150)]
    else:
        out.append(("bench electronics", 0.4, -170 - A, 0, 120))
    if ballast:
        out.append((f"{ballast:.0f} kg ballast", ballast, -250 - A, 0, 60))
    return out


def group(order, tray_z, drawer=0.0, hand_x=None, hand_y=None, payload=0.0, pay_x=None, pay_y=0.0, stump_x=None):
    """Carriage, tray, hand (centre of mass at hand_x), stump (tucked unless placed), payload."""
    hx = P.SHOULDER_X + 100 if hand_x is None else hand_x
    hy = P.ARM_Y[0] if hand_y is None else hand_y
    out = [("carriage", M["carriage"], P.TRAY_X0 + 40, 0, tray_z + 80),
           ("tray", M["tray"], P.TRAY_X0 + P.TRAY_DEPTH / 2 + drawer, 0, tray_z),
           ("hand arm", HAND, hx, hy, tray_z + P.ARM_ABOVE_TRAY)]
    if order >= 2:
        out.append(("stump", STUMP, P.SHOULDER_X if stump_x is None else stump_x, P.ARM_Y[1], tray_z + P.ARM_ABOVE_TRAY))
    if payload:                       # pay_x: in the hand (world x), or on the tray (tray x, moves out with the drawer)
        out.append(("payload", payload, pay_x if pay_x is not None else hx, pay_y, tray_z + 60))
    return out


def sf_forward(masses):
    rest = -sum(m * (x - FRONT_X) for _, m, x, y, z in masses if x < FRONT_X)
    over = sum(m * (x - FRONT_X) for _, m, x, y, z in masses if x >= FRONT_X)
    return rest / over if over > 0 else float("inf")


def sf_side(masses):
    """Tipping over the right-hand feet (y = -SIDE_Y)."""
    rest = sum(m * (y + SIDE_Y) for _, m, x, y, z in masses if y > -SIDE_Y)
    over = -sum(m * (y + SIDE_Y) for _, m, x, y, z in masses if y <= -SIDE_Y)
    return rest / over if over > 0 else float("inf")


def verdict(sf):
    return "OK" if sf >= 1.5 else ("MARGINAL" if sf >= 1.1 else "TIPS")


# the hand's centre of mass for "arm full forward": ~60 % of the way from shoulder to fingertips
REACH = P.L1 + P.L2 + P.PAD_X
FWD = P.SHOULDER_X + 0.6 * REACH
SIDE = P.ARM_Y[0] - 0.6 * REACH

CASES = [
    # (label, order, ballast, group kwargs, which)
    ("configuration 1, table height 0.65 m: hand full forward, 1 kg in it", 1, 0, dict(tray_z=650, hand_x=FWD, payload=1.0,
     pay_x=P.SHOULDER_X + REACH), "fwd"),
    ("configuration 1, same, 5 kg ballast", 1, 5, dict(tray_z=650, hand_x=FWD, payload=1.0, pay_x=P.SHOULDER_X + REACH), "fwd"),
    ("configuration 1, drawer out, 2 kg bottle on its front (tongue test)", 1, 0, dict(tray_z=650, drawer=P.DRAWER_STROKE,
     payload=2.0, pay_x=P.TRAY_X0 + P.TRAY_DEPTH - 30 + P.DRAWER_STROKE), "fwd"),
    ("configuration 1, same, 5 kg ballast", 1, 5, dict(tray_z=650, drawer=P.DRAWER_STROKE, payload=2.0,
     pay_x=P.TRAY_X0 + P.TRAY_DEPTH - 30 + P.DRAWER_STROKE), "fwd"),
    ("configuration 1, hand straight out to the right, 1 kg", 1, 0, dict(tray_z=650, hand_x=P.SHOULDER_X, hand_y=SIDE,
     payload=1.0, pay_x=P.SHOULDER_X, pay_y=P.ARM_Y[0] - REACH), "side"),
    ("configuration 1, same, 5 kg ballast", 1, 5, dict(tray_z=650, hand_x=P.SHOULDER_X, hand_y=SIDE, payload=1.0,
     pay_x=P.SHOULDER_X, pay_y=P.ARM_Y[0] - REACH), "side"),
    ("configuration 2, floor: hand pitched down 0.4 m ahead, stump forward too", 2, 5, dict(tray_z=210, hand_x=FWD,
     payload=0.3, pay_x=P.SHOULDER_X + 420, stump_x=FWD), "fwd"),
    ("configuration 2, drawer out at bed height, 5 kg parcel", 2, 5, dict(tray_z=640, drawer=P.DRAWER_STROKE, payload=5.0,
     pay_x=P.TRAY_X0 + P.TRAY_DEPTH / 2 + P.DRAWER_STROKE), "fwd"),
    ("configuration 2, same, no ballast", 2, 0, dict(tray_z=640, drawer=P.DRAWER_STROKE, payload=5.0,
     pay_x=P.TRAY_X0 + P.TRAY_DEPTH / 2 + P.DRAWER_STROKE), "fwd"),
    ("configuration 3, hand full forward, 1 kg in it", 3, 0, dict(tray_z=650, hand_x=FWD, payload=1.0,
     pay_x=P.SHOULDER_X + REACH), "fwd"),
    ("configuration 3, drawer out at bed height, 5 kg parcel", 3, 0, dict(tray_z=640, drawer=P.DRAWER_STROKE, payload=5.0,
     pay_x=P.TRAY_X0 + P.TRAY_DEPTH / 2 + P.DRAWER_STROKE), "fwd"),
    ("configuration 3, hand straight out to the right, 1 kg", 3, 0, dict(tray_z=650, hand_x=P.SHOULDER_X, hand_y=SIDE,
     payload=1.0, pay_x=P.SHOULDER_X, pay_y=P.ARM_Y[0] - REACH), "side"),
]

if __name__ == "__main__":
    lines = ["Does each configuration stand on its own? (calc_v03.py; static, no braking; SF >= 1.5 OK)", ""]
    for o in (1, 2, 3):
        st = stage(o)
        g = group(o, 650)
        lines.append(f"configuration {o}: {sum(m for _, m, *_ in st + g):.1f} kg without payload or ballast")
    lines.append(f"hand arm {HAND:.2f} kg (v0.2 with iPhone {HAND + 0.14:.2f}), stump {STUMP:.2f} kg")
    lines.append("")
    for label, order, ballast, kw, which in CASES:
        ms = stage(order, ballast) + group(order, **kw)
        sf = sf_forward(ms) if which == "fwd" else sf_side(ms)
        lines.append(f"  {label:64s} {'forward ' if which == 'fwd' else 'sideways'} SF {sf:4.1f}  {verdict(sf)}")
    txt = "\n".join(lines)
    (ROOT / "out" / "calc_v03.txt").write_text(txt + "\n")
    print(txt)
