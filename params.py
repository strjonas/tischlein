"""Single source of truth for Tischlein v0.2 geometry and masses.

Units: mm, kg, N, s. Frame: x forward, y left, z up.
Origin: floor, under the midpoint of the drive axle.
Everything here is a design target, not a measured value.
"""

# --- drive base (mid-wheel drive, like a power wheelchair) ---
WHEEL_D = 165            # hoverboard 6.5" hub motor
WHEEL_W = 55
TRACK = 390              # wheel centre to centre -> outer width ~445
AXLE_SHIFT = 120         # v0.2: drive axle 120 mm further forward in the frame (frame x values below include it)
BASE_X_REAR = -260 - AXLE_SHIFT       # rear edge of base frame
BASE_X_FRONT = 175 - AXLE_SHIFT       # front edge of base frame
FRONT_CASTER_X = 320 - AXLE_SHIFT     # front casters on two straddle legs; the tray drops between them
FRONT_CASTER_Y = 200
REAR_CASTER_X = -225 - AXLE_SHIFT     # rear caster, sprung
CASTER_D = 100           # 100 mm: rolls over socks and ~15 mm thresholds
FRAME_Z0 = CASTER_D + 28    # bottom of the 2020 frame (above caster plates)
DECK_Z = FRAME_Z0 + 20 + 6  # top of the 6 mm deck plate

# --- mast (vertical axis) ---
MAST_X = 195 - AXLE_SHIFT  # centre of the 20x40 extrusion (40 along x)
MAST_W_X = 40
MAST_W_Y = 20
MAST_Z0 = 20             # bottom of extrusion
MAST_LEN = 1800          # v0.2: 1.5 -> 1.8 m, fingertips up to ~1.58 m
RAIL_H = 8               # MGN12 rail height
BLOCK_H = 13 - RAIL_H    # MGN12H block top above rail top (13 total from rail base)
LIFT_BLOCK_SPACING = 150

# --- carriage, tray, arm (all ride together on the lift) ---
CARRIAGE_X0 = MAST_X + MAST_W_X / 2 + RAIL_H + BLOCK_H   # back face of carriage plate
CARRIAGE_T = 8
YOKE_X0 = CARRIAGE_X0 + CARRIAGE_T                       # 2020 cross bar behind the drawer
TRAY_X0 = YOKE_X0 + 20                                   # back edge of tray (drawer retracted)
TRAY_DEPTH = 250
TRAY_WIDTH = 280
TRAY_T = 6
TRAY_LIP = 15            # low lips: the hand (bottom at +24) must pass over them
ARM_RAIL_Y = TRAY_WIDTH / 2 + 13 + 10                    # centre of the fixed 2020 side arms
DRAWER_STROKE = 240      # 250 mm full-extension slides
TRAY_Z_MIN = 75          # side arms, slides and drawer servo hang below the tray surface
TRAY_Z_MAX = MAST_Z0 + MAST_LEN - 280   # 1540: carriage top stays below the top block and balancer

SHOULDER_X = YOKE_X0 + 40
ARM_Y = (-120, 120)      # v0.2: two arms, shoulders at y = -120 (arm "a", right) and +120 (arm "b", left)
L1_ABOVE_TRAY = 155      # upper arm plane above tray surface
L2_ABOVE_TRAY = 115      # forearm plane
ARM_ABOVE_TRAY = L2_ABOVE_TRAY
L1 = 220                 # shoulder -> elbow
L2 = 220                 # elbow -> wrist yaw axis
PITCH_DROP = 75          # wrist pitch axis below forearm plane
# v0.2 hand: wrist pitch -> roll stage -> CAP gripper (NYU robot gripper + iPhone)
ROLL_BRACKET = 25        # pitch axis -> roll servo body
ROLL_HORN_X = 66         # pitch axis -> roll horn face (STS3215 body ~38 mm deep along its axis)
CAP_ADAPTER_T = 6        # printed plate between roll horn and the CAP gripper
CAP_BACK = 64.3          # CAP gripper frame origin sits this far in front of its back face (EgoGym model)
CAP_GC = 140             # CAP gripper origin -> grasp centre
CAP_AXIS_Z = 17          # finger mid-plane above the CAP gripper origin; the roll axis runs through it
CAP_ORIGIN_X = ROLL_HORN_X + CAP_ADAPTER_T + CAP_BACK     # 136.3 in the roll frame
WRIST_TO_TIP = round(CAP_ORIGIN_X + CAP_GC)               # 276: pitch axis -> grasp centre
WRIST_DROP = PITCH_DROP + WRIST_TO_TIP - (L2_ABOVE_TRAY - PITCH_DROP)  # used by calc figure
CAP_OPEN = 90            # fingertip gap, fully open (angular jaw)
PAD_X = round(CAP_ORIGIN_X + 100)                         # 236: pitch axis -> centre of the finger contact pads
PITCH_RANGE = (-15, 100)  # deg; nose-up is limited by the phone holder meeting the wrist block

# face (the old phone) on the carriage, and the lamp on the mast top
FACE_W, FACE_H, FACE_T = 75, 150, 10
FACE_Z_ABOVE_TRAY = 260  # screen centre
LAMP_D = 230             # shade, bottom diameter

# --- masses (estimates) ---
N_ARMS = 2
M = dict(
    hub_motor=2.8,           # each, with tyre
    battery=1.4,             # 10S hoverboard pack (LiFePO4 later: ~2.6 kg)
    base_frame=2.2,          # 2020 frame + 6 mm plywood deck + caster hardware
    electronics=0.8,         # Pi, drivers, DC-DCs, wiring
    lidar=0.2,
    mast=1.55,               # 1.8 m 2040 + MGN12 rail + top block + spring balancer + lamp
    carriage=0.65,           # plate, yoke, two shoulder housings, face phone + mount
    tray=0.45,
    arm_shoulder=0.30,       # per arm: bearings, housing, 60T pulley, servo
    arm_L1=0.35,             # per arm: link + elbow block + elbow servo
    arm_L2=0.22,             # per arm
    wrist_gripper=0.62,      # per arm: yaw + STS3250 pitch + roll servos, CAP gripper + its servo, iPhone
)

# --- actuators ---
STS3215_RATED_NM = 0.98      # 10 kg.cm rated at 12 V (Feetech listing)
STS3215_PEAK_NM = 2.94       # 30 kg.cm stall at 12 V
STS3215_NOLOAD_RPM = 45      # 0.222 s / 60 deg
STS3250_RATED_NM = 1.6       # wrist pitch in v0.2; 50 kg.cm stall at 12 V, rated ~1/3 (estimate)
STS3250_PEAK_NM = 4.9
J1_RATIO = 3.0               # GT2 20T -> 60T
J2_RATIO = 2.0               # GT2 20T -> 40T
NEMA17_HOLD_NM = 0.42
NEMA17_NM_AT_600RPM_24V = 0.22   # typical 17HS4401-class curve, rough
LIFT_RATIO = 3.0
LIFT_PULLEY_T = 20           # GT2, 2 mm pitch

# --- GT2 pulley geometry ---
def gt2_pd(teeth):
    import math
    return teeth * 2 / math.pi
