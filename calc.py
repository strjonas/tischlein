"""Back-of-envelope engineering checks for Tischlein v0.2 (two arms, CAP gripper, 1.8 m mast).

Run: uv run python calc.py   -> prints a report and writes figures to out/
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import params as P

G = 9.81
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def section(t):
    print(f"\n## {t}")


# ---------------------------------------------------------------- arm statics
def scara_ik(x, y):
    """Planar IK, elbow-left. Returns (q1, q2) or None."""
    r2 = x * x + y * y
    c2 = (r2 - P.L1**2 - P.L2**2) / (2 * P.L1 * P.L2)
    if abs(c2) > 1:
        return None
    q2 = math.acos(c2)
    q1 = math.atan2(y, x) - math.atan2(P.L2 * math.sin(q2), P.L1 + P.L2 * math.cos(q2))
    return q1, q2


def jacobian(q1, q2):
    l1, l2 = P.L1 / 1000, P.L2 / 1000
    return np.array([
        [-l1 * math.sin(q1) - l2 * math.sin(q1 + q2), -l2 * math.sin(q1 + q2)],
        [l1 * math.cos(q1) + l2 * math.cos(q1 + q2), l2 * math.cos(q1 + q2)],
    ])


section("Arm: horizontal pulling (sliding a box onto the tray)")
# Pull straight back toward the shoulder, the realistic 'slide onto tray' case.
mu_cases = {"PP box on glass shelf (mu~0.3)": 0.3, "sticky/wet worst case (mu~0.6)": 0.6}
cap_j1 = P.STS3215_RATED_NM * P.J1_RATIO
cap_j2 = P.STS3215_RATED_NM * P.J2_RATIO
print(f"J1 rated {cap_j1:.1f} N·m (peak {P.STS3215_PEAK_NM*P.J1_RATIO:.1f}), "
      f"J2 rated {cap_j2:.1f} N·m (peak {P.STS3215_PEAK_NM*P.J2_RATIO:.1f})")
reaches = np.linspace(150, P.L1 + P.L2 - 5, 60)
worst = {}
for name, mu in mu_cases.items():
    for mass in (1.0, 2.0, 3.0):
        F = mu * mass * G
        ok_until = None
        for r in reaches:
            q = scara_ik(r, 0)
            tau = jacobian(*q).T @ np.array([-F, 0.0])   # pull toward -x
            if abs(tau[0]) <= cap_j1 and abs(tau[1]) <= cap_j2:
                ok_until = r
        worst[(name, mass)] = ok_until
        print(f"  {name}, {mass:.0f} kg: pull force {F:4.1f} N, "
              f"within rated torque out to r = {ok_until:.0f} mm" if ok_until else
              f"  {name}, {mass:.0f} kg: exceeds rated torque everywhere")

# Same pull, but sideways (e.g. arm swung 90 deg to sweep an item off a side shelf)
section("Arm: worst case, tangential pull at full reach")
for F in (5, 10, 20):
    print(f"  {F:2d} N tangential at {P.L1+P.L2} mm -> J1 needs {F*(P.L1+P.L2)/1000:.1f} N·m "
          f"({'ok' if F*(P.L1+P.L2)/1000 <= cap_j1 else 'over rated'})")

section("Arm: lifting with the gripper (wrist pitch STS3250, direct drive)")
tip = P.WRIST_TO_TIP / 1000
for m in (0.3, 0.5, 1.0):
    # roll servo ~45 mm, CAP gripper + its servo ~0.2 kg at ~0.16 m, iPhone ~0.19 kg at ~0.09 m, payload at the grasp centre
    tau = (m * tip + 0.055 * 0.045 + 0.20 * 0.16 + 0.19 * 0.09) * G
    print(f"  {m:.1f} kg held forward at {tip*1000:.0f} mm: wrist pitch {tau:.2f} N·m vs rated {P.STS3250_RATED_NM:.2f} "
          f"(peak {P.STS3250_PEAK_NM:.1f}) -> {'ok' if tau <= P.STS3250_RATED_NM else 'only short peaks'}")

section("Arm: vertical bending at the shoulder bearings (the servo does NOT carry this)")
arm_moment = G * (P.M["arm_L1"] * 0.14 + P.M["arm_L2"] * 0.36 + P.M["wrist_gripper"] * 0.55 + 0.5 * 0.70)
spacing = 0.040
print(f"  arm + 0.5 kg at full reach: {arm_moment:.1f} N·m -> {arm_moment/spacing:.0f} N per bearing "
      f"at {spacing*1000:.0f} mm spacing (6001: C0 ~ 2.4 kN, fine)")

section("Fridge door: pull through a straight (singular) arm")
for F in (20, 40, 60):
    q = scara_ik(P.L1 + P.L2 - 2, 0)
    tau = jacobian(*q).T @ np.array([-F, 0.0])
    print(f"  {F} N straight back, arm almost straight: J1 {abs(tau[0]):.2f} N·m, J2 {abs(tau[1]):.2f} N·m "
          f"(load goes into bearings, not servos)")
print("  -> the base, not the arm, provides the pulling force; arm stays near-straight and compliant.")

# ---------------------------------------------------------------- lift
section("Lift: NEMA17 + 3:1 GT2 reduction + GT2 belt on the mast, constant-force spring balancer")
m_arm = P.M["arm_shoulder"] + P.M["arm_L1"] + P.M["arm_L2"] + P.M["wrist_gripper"]
m_moving = P.M["carriage"] + P.M["tray"] + P.N_ARMS * m_arm
balancer = m_moving                      # a 3-5 kg tool balancer set to the carriage's own weight
print(f"  moving mass with {P.N_ARMS} arms: {m_moving:.2f} kg -> balancer range must include {balancer:.1f} kg (a 3-5 kg tool balancer)")
r_p = P.LIFT_PULLEY_T * 2 / (2 * math.pi) / 1000
for payload in (0, 1, 3, 5):
    for bal in (0.0, balancer):
        F = ((m_moving - bal + payload) * G) * 1.25 + 3.0   # 25% for friction/acceleration, 3 N rail drag
        tau_motor = F * r_p / P.LIFT_RATIO / 0.9
        print(f"  {'balanced' if bal else 'no balancer'}: +{payload} kg: {F:5.1f} N, motor {tau_motor:.3f} N·m "
              f"(at 600 rpm ~{P.NEMA17_NM_AT_600RPM_24V} available, margin {P.NEMA17_NM_AT_600RPM_24V/tau_motor:.1f}x)")
v = 600 / 60 / P.LIFT_RATIO * 2 * math.pi * r_p * 1000
print(f"  speed at 600 rpm motor: {v:.0f} mm/s -> full stroke {(P.TRAY_Z_MAX-P.TRAY_Z_MIN)/v:.0f} s "
      f"(run at ~60 mm/s for quiet: {(P.TRAY_Z_MAX-P.TRAY_Z_MIN)/60:.0f} s)")
hold = P.NEMA17_HOLD_NM * P.LIFT_RATIO / r_p
print(f"  powered holding force {hold:.0f} N.")

section("Lift on power loss (26 Oct review: the balancer alone does not hold a payload)")
# The balancer is set to the carriage's own weight, so any payload hangs on the unpowered motor.
# Shorted coils give back-EMF damping only (no static force): the tray creeps down at the speed where the
# damping equals the payload's weight. 17HS4401 class: Kt ~0.25 N*m/A (0.42 N*m at 1.7 A), R ~1.5 ohm/phase.
kt, r_ph = 0.25, 1.5
c_damp = kt * kt / r_ph                      # N*m per rad/s at the motor, one phase pair shorted (rough)
for payload in (1, 3, 5):
    tau = payload * G * r_p / P.LIFT_RATIO
    omega = tau / c_damp
    v = omega * r_p / P.LIFT_RATIO
    print(f"  {payload} kg on the tray, power off, coils shorted: creeps down at ~{v*1000:.0f} mm/s "
          f"({(P.TRAY_Z_MAX - P.TRAY_Z_MIN) / 1000 / max(v, 1e-9) / 60:.0f} min for the full stroke)")
print("  -> not a holding mechanism. Fix: a NEMA17 with a 24 V power-off (spring-applied) brake.")
for brake_nm in (0.2, 0.4):
    f_hold = brake_nm * P.LIFT_RATIO / r_p
    need = 5 * G + 0.25 * m_moving * G       # 5 kg parcel + 25% of the carriage if the balancer is mis-set
    print(f"  brake {brake_nm} N*m at the motor -> holds {f_hold:.0f} N at the carriage "
          f"(5 kg parcel + a mis-set balancer needs {need:.0f} N: {'ok' if f_hold > 1.5 * need else 'too weak'})")
print("  Belt break: the balancer still carries the carriage; only a payload would come down (slowly, balancer friction).")
print("  Bench test: 5 kg on the tray at 1.0 m, cut power, drift < 2 mm in 10 min; repeat with the brake wire pulled.")

# ---------------------------------------------------------------- base
section("Base: thresholds")
for d, name in ((P.WHEEL_D, "drive wheel"), (P.CASTER_D, "caster 75 mm"), (100, "caster 100 mm"), (50, "caster 50 mm")):
    r = d / 2
    for h in (10, 15, 20):
        if h >= r:
            continue
        ratio = math.sqrt(2 * r * h - h * h) / (r - h)
        print(f"  {name:13s} step {h} mm: push force = {ratio:4.2f} x wheel load")
print("  -> 75 mm casters handle ~10 mm, 100 mm casters ~15-20 mm. Thresholds are a real limit.")

section("Base: masses and static tipping")
A = P.AXLE_SHIFT
parts = [  # (name, mass, x, z); frame-mounted parts moved back with the axle shift
    ("hub motors", 2 * P.M["hub_motor"], 0, P.WHEEL_D / 2),
    ("battery", P.M["battery"], -150 - A, 70),
    ("base frame", P.M["base_frame"], -40 - A, 80),
    ("electronics", P.M["electronics"], -170 - A, 120),
    ("lidar", P.M["lidar"], -120 - A, 150),
    ("mast", P.M["mast"], P.MAST_X, 930),
]
total_static = sum(p[1] for p in parts)
print(f"  base + mast: {total_static:.1f} kg; moving lift group {m_moving:.2f} kg; total robot "
      f"{total_static + m_moving:.1f} kg without payload")


def tipping_report(tray_z, payload, payload_x, arm_x, decel=0.0, label="", drawer=0.0):
    """Moment about the front caster line (x = FRONT_CASTER_X), kg·mm. Positive = tipping forward."""
    fx = P.FRONT_CASTER_X
    fixed = m_moving - P.N_ARMS * (P.M["arm_L2"] + P.M["wrist_gripper"]) - P.M["tray"]
    group = [(fixed, P.TRAY_X0 + 40, tray_z + 80),
             (P.M["tray"], P.TRAY_X0 + P.TRAY_DEPTH / 2 + drawer, tray_z),
             (P.N_ARMS * (P.M["arm_L2"] + P.M["wrist_gripper"]), arm_x, tray_z + P.ARM_ABOVE_TRAY),
             (payload, payload_x + drawer, tray_z + 60)]
    allp = [(m, x, z) for _, m, x, z in parts] + group
    grav = sum(m * (x - fx) for m, x, z in allp)              # >0 tips forward
    inert = sum(m * decel / G * z for m, x, z in allp)        # braking while driving forward
    M_tip = grav + inert
    # restoring capacity is the negative part; report safety factor
    restoring = -sum(m * (x - fx) for m, x, z in allp if x < fx)
    overturning = sum(m * (x - fx) for m, x, z in allp if x >= fx) + inert
    sf = restoring / overturning if overturning > 0 else float("inf")
    print(f"  {label:48s} SF {sf:4.1f} {'OK' if sf > 1.5 else ('MARGINAL' if sf > 1.1 else 'TIPS')}")
    return sf


T = P.TRAY_Z_MAX
tipping_report(400, 3.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, label="3 kg on tray, tray at 0.4 m, standing")
tipping_report(T, 3.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, label="3 kg on tray, tray at top, standing")
tipping_report(T, 1.0, P.SHOULDER_X + 690, P.SHOULDER_X + 560, label="0.5 kg in each gripper, arms full forward, top")
tipping_report(700, 3.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, decel=1.0, label="3 kg, tray at 0.7 m, braking 1 m/s²")
tipping_report(T, 3.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, decel=1.0, label="3 kg, tray at TOP, braking 1 m/s² (don't)")
tipping_report(950, 1.0, P.TRAY_X0 + 180, P.SHOULDER_X + 300, label="drawer out 240, 1 kg box at its front, 0.95 m", drawer=P.DRAWER_STROKE)
tipping_report(950, 2.0, P.TRAY_X0 + 180, P.SHOULDER_X + 300, label="drawer out 240, 2 kg box at its front, 0.95 m", drawer=P.DRAWER_STROKE)
tipping_report(950, 3.0, P.TRAY_X0 + 180, P.SHOULDER_X + 300, label="drawer out 240, 3 kg box at its front, 0.95 m", drawer=P.DRAWER_STROKE)
tipping_report(640, 1.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, label="drawer out over the bed, 1 kg meal", drawer=P.DRAWER_STROKE)
tipping_report(640, 5.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, label="drawer out over the bed, 5 kg parcel", drawer=P.DRAWER_STROKE)
tipping_report(300, 5.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, decel=1.5, label="5 kg parcel, tray at 0.3 m, hard brake 1.5 m/s²")
print("  (v0.2 numbers: axle 12 cm forward, two arms with CAP hands and phones, 1.8 m mast with lamp and balancer.")
print("   The axle shift alone took the 5 kg-over-the-bed case from ~2.3 to ~1.9; the docs quoted 2.3 until 26 Oct.)")

section("Tipping with the home battery (LiFePO4 2.6 kg instead of the 1.4 kg hoverboard pack, same rear spot)")
_bat = [p for p in parts if p[0] == "battery"][0]
parts[parts.index(_bat)] = ("battery", 2.6, _bat[2], _bat[3])
tipping_report(640, 5.0, P.TRAY_X0 + 125, P.SHOULDER_X + 100, label="drawer out over the bed, 5 kg parcel (LiFePO4)", drawer=P.DRAWER_STROKE)
tipping_report(950, 3.0, P.TRAY_X0 + 180, P.SHOULDER_X + 300, label="drawer out 240, 3 kg box at its front (LiFePO4)", drawer=P.DRAWER_STROKE)
parts[parts.index(("battery", 2.6, _bat[2], _bat[3]))] = _bat

section("Bench stand: the real base frame, no hub motors, no battery")
# The bench stand is the robot's own 2020 frame with its straddle legs, standing on fixed feet where the wheels
# and casters will go. Without the 5.6 kg of hub motors the mast tips forward much sooner: add ballast at the back.
saved = list(parts)
bench = [p for p in parts if p[0] not in ("hub motors", "battery", "electronics", "lidar")]
bench.append(("electronics", 0.4, -170 - A, 120))
for ballast in (0.0, 5.0, 8.0):
    parts[:] = bench + ([("ballast", ballast, -250 - A, 60)] if ballast else [])
    tipping_report(P.TRAY_Z_MAX, 1.0, P.SHOULDER_X + 690, P.SHOULDER_X + 560,
                   label=f"bench, {ballast:.0f} kg ballast: 0.5 kg per hand, arms forward, top")
    tipping_report(950, 3.0, P.TRAY_X0 + 180, P.SHOULDER_X + 300,
                   label=f"bench, {ballast:.0f} kg ballast: drawer out, 3 kg box at its front", drawer=P.DRAWER_STROKE)
parts[:] = saved
print("  -> on the bench, put >= 5 kg (a sandbag or a dumbbell plate, later the battery) on the rear of the deck,")
print("     and stand the frame on four fixed rubber feet instead of casters, so it can't roll during tests.")

section("Base: sideways tipping when the arm swings out (roll about the wheel line)")
half_track = P.TRACK / 2
for reach_y, m in ((P.ARM_Y[1] + P.L1 + P.L2 + P.WRIST_TO_TIP, 0.5), (P.ARM_Y[1] + P.L1 + P.L2 + P.WRIST_TO_TIP, 1.0)):
    # one arm stretched straight out sideways, the other parked over the tray
    over = (m + P.M["arm_L2"] + P.M["wrist_gripper"]) * (reach_y - P.WRIST_TO_TIP / 2 - half_track) + P.M["arm_L1"] * (P.ARM_Y[1] + P.L1 / 2 - half_track)
    restore = (total_static + m_moving - P.M["arm_L2"] - P.M["wrist_gripper"] - P.M["arm_L1"]) * half_track
    print(f"  {m} kg held {reach_y} mm to the side: SF {restore/max(over,1e-6):.1f}")

section("Battery")
wh = 36 * 4.4
for name, w in (("idle (Pi, cams, lidar)", 9), ("mixed use", 22), ("busy, two arms", 45)):
    print(f"  {name:24s} {w:3d} W -> {wh/w:4.1f} h from a healthy {wh:.0f} Wh pack")
wh_lfp = 25.6 * 10
print(f"  LiFePO4 8S 10 Ah: {wh_lfp:.0f} Wh -> {wh_lfp/22:.0f} h mixed use; 3 A dock charger: {wh_lfp/(29.2*3)*1.15:.1f} h for 0-100%")

section("Servo power on the bench (26 Oct review: the 12 V wiring was underspecified)")
stall = dict(STS3215=2.7, STS3250=4.2)          # A at 12 V, retailer datasheets
n = dict(STS3215=5, STS3250=1)                  # arm A: J1, J2, yaw, roll, drawer + pitch
worst = sum(stall[k] * n[k] for k in n)
moving = 2 * stall["STS3215"] + 0.5 * stall["STS3250"] + 3 * 0.3
print(f"  every servo stalled at once: {worst:.1f} A; realistic peak (two joints near stall, others moving): ~{moving:.0f} A")
print("  -> the Waveshare adapter is a signal converter: route servo power around it, not through it.")
print("     12 V 15 A supply -> 15 A main fuse -> latching e-stop -> fuse block: J1+J2 5 A, wrist (yaw, pitch, roll) 7.5 A,")
print("     drawer 3 A; 1.0 mm2 wire to each branch; 2200 uF at the block; data + GND daisy-chained through the adapter.")
print("     In every servo's EEPROM: protection current ~2 A (STS3215) / 3 A (STS3250), torque limit 80%, max temp 65 C.")
print("     XL330 + U2D2 stay on their own 5 V supply; lift stepper on 24 V; common ground.")

section("Door opening: can the base pull hard enough?")
# share of the weight on the drive wheels: moments about the axle; the sprung rear caster takes the rest
x_cg = sum(m * x for _, m, x, z in parts) / total_static
drive_share = 0.75
print(f"  CG of base + mast at x = {x_cg:.0f} mm (axle at 0, rear caster at {P.REAR_CASTER_X:.0f}); "
      f"drive wheels carry ~{drive_share:.0%} with a light rear-caster spring")
drive_load = (total_static + m_moving) * G * drive_share
for mu in (0.4, 0.7):
    print(f"  floor mu {mu}: max pull before wheel slip ~{mu*drive_load:.0f} N "
          f"(fridge doors are typically tens of N at breakaway; measure yours)")

# ---------------------------------------------------------------- figures (height coverage lives in figs.py)
# top view workspace
fig, ax = plt.subplots(figsize=(7, 7))
th = np.linspace(-math.radians(135), math.radians(135), 200)
for q2 in np.linspace(-math.radians(150), math.radians(150), 25):
    xs = P.SHOULDER_X + P.L1 * np.cos(th) + P.L2 * np.cos(th + q2)
    ys = P.L1 * np.sin(th) + P.L2 * np.sin(th + q2)
    ax.plot(xs, ys, color="tab:blue", alpha=0.08)
ax.add_patch(plt.Rectangle((P.TRAY_X0, -P.TRAY_WIDTH / 2), P.TRAY_DEPTH, P.TRAY_WIDTH, color="tab:orange", alpha=0.35, label="tray"))
ax.add_patch(plt.Rectangle((P.BASE_X_REAR, -P.TRACK / 2), P.BASE_X_FRONT - P.BASE_X_REAR, P.TRACK, fill=False, lw=2, label="base"))
for s in (-1, 1):
    ax.add_patch(plt.Rectangle((-P.WHEEL_D / 2, s * P.TRACK / 2 - P.WHEEL_W / 2), P.WHEEL_D, P.WHEEL_W, color="0.2"))
ax.plot([P.SHOULDER_X], [0], "ko")
ax.set_aspect("equal"); ax.set_xlim(-350, 900); ax.set_ylim(-600, 600)
ax.set_title("Top view: wrist reach (blue), tray (orange)")
ax.legend(fontsize=8)
fig.savefig(OUT / "reach_top.png", dpi=130, bbox_inches="tight")
print(f"\nfigures -> {OUT}")
