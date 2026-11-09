"""Tischlein v0.3 proposal on top of the v0.2 CAD, in three configurations.

uv run python cad/v03.py            # -> out/v03_*.glb (with a floor, for Blender), out/v03_views.json
uv run python cad/v03.py --viewer   # -> out/v03_*_viewer.glb (no floor plate, for web viewers)

What changes against v0.2 (cad/robot.py stays the v0.2 model that the simulations use):
  - arm b is a stump: shoulder, elbow and the wrist yaw servo turn a printed head (hook, pad, scoop loop) at the
    height where the hand's pads would be, as in sim/fridge_door.py (`stumpyaw`); no pitch, roll, gripper or phone
  - arm a's iPhone is replaced by the infrared camera in its phone-shaped insert (cad/ir_camera.py)
  - the drawer gets a spring-steel tongue
  - CAP's gripper is drawn from its own printed STLs where we have them (top plate, phone holder, bottom plate)

Configurations:
  1  the base frame as a stand on feet with ballast, the mast with the carriage bolted at a fixed height (two printed
     blocks in place of the rail carriages), arm a with CAP gripper and infrared camera, the tongue drawer
  2  + lift (rail, braked motor, belts, balancer), + the stump
  3  + drive (hoverboard wheels, casters), battery, electronics, lidar, lamp, face phone: the robot
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from build123d import Location, Pos, Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_camera as IRC  # noqa: E402
import robot as R  # noqa: E402
import scene as S  # noqa: E402

P = R.P
OUT = R.ROOT / "out"
SPRING_STEEL = (0.62, 0.64, 0.68)
CONCEPT = (0.50, 0.36, 0.78)          # drawn as a concept only: not designed in detail, not simulated on the robot

# ------------------------------------------------------------------ arm b: the stump head (wrist frame, mm)
HOOK_Z = -P.PITCH_DROP                 # hook at the pitch-axis height, where the hand's pads would be
SHANK_Y, END_Y = 25.0, -30.0           # sim/fridge_door.py HOOK_SHANK_Y, HOOK_END_Y
TIP_X = P.PAD_X + 25.5                 # tip centre behind the hook point (PAD_X): HOOK_TIP_X


def link_stump_head():
    """Printed head on the wrist yaw horn (horn face at z = -48): a post down to the hook plane, a shank beside
    the handle bar, the hook tip behind it, a TPU pad on the shank's outer face and a loop for the scoop plate."""
    zh = -48
    head = R.cyl_z(0, 0, zh - 6, zh, 26)                                   # horn disc, 4x M2.5 on r = 7 mm
    head += R.box(-10, 10, -10, 10, HOOK_Z - 10, zh - 6)                   # post
    head += R.box(-10, 100, -8, SHANK_Y + 8, HOOK_Z - 8, HOOK_Z + 8)       # root of the shank
    head += R.box(100, TIP_X + 5, SHANK_Y - 8, SHANK_Y + 8, HOOK_Z - 8, HOOK_Z + 8)   # shank beside the bar
    head += R.box(TIP_X - 5, TIP_X + 5, END_Y, SHANK_Y, HOOK_Z - 14, HOOK_Z + 14)    # tip behind the bar
    head += R.box(140, 170, SHANK_Y + 8, SHANK_Y + 14, HOOK_Z + 8, HOOK_Z + 26) - \
        R.cyl_y(155, HOOK_Z + 19, SHANK_Y + 7, SHANK_Y + 15, 9)            # scoop loop
    for k in range(4):
        a = math.radians(45 + 90 * k)
        head -= R.cyl_z(7 * math.cos(a), 7 * math.sin(a), zh - 7, zh + 1, 2.8)
    pad = R.box(185, TIP_X - 8, SHANK_Y + 8, SHANK_Y + 14, HOOK_Z - 12, HOOK_Z + 12)
    return [("stump_head", head, R.PRINT_A, True), ("stump_pad_tpu", pad, R.TPU, True)]


# ------------------------------------------------------------------ arm a: infrared camera instead of the iPhone
def ir_items():
    """The insert, board, lens and LEDs in the tool frame (phone frame -> CAP frame -> tool frame)."""
    T = R.CAP_LOC * Location((0, 0, 0), (IRC.TILT, 0, 0))
    z_pcb, _ = IRC.pcb_z()
    cx, cy = IRC.LENS[0], IRC.LENS[1]
    board = R.box(cx - IRC.PCB_W / 2, cx + IRC.PCB_W / 2, cy - IRC.PCB_W / 2, cy + IRC.PCB_W / 2,
                  z_pcb, z_pcb + IRC.PCB_T)
    lens = R.cyl_z(cx, cy, IRC.LENS[2] - 3, z_pcb, 14) + R.box(cx - 8, cx + 8, cy - 8, cy + 8, z_pcb - 11, z_pcb)
    leds = None
    for x, y in IRC.LED_XY:
        led = R.cyl_z(x, y, IRC.PHONE["z_back"] - 6, IRC.PHONE["z_back"] + 4, 5)
        leds = led if leds is None else leds + led
    return [("ir_cam_insert", T * IRC.insert(), R.PRINT_B, True),
            ("ir_camera_board", T * board, R.PCB, False),
            ("ir_camera_lens", T * lens, (0.05, 0.05, 0.06), False),
            ("ir_leds_940nm", T * leds, (0.35, 0.10, 0.12), False)]


def link_tool_v03():
    keep = [it for it in R.link_tool() if it[0] not in ("iphone", "iphone_camera", "cap_gripper_body",
                                                         "cap_phone_holder")]
    return keep + ir_items()


# ------------------------------------------------------------------ drawer: the tongue
def link_drawer_v03():
    tx1 = P.TRAY_X0 + P.TRAY_DEPTH
    tongue = R.box(tx1 - 30, tx1 + 45, -P.TRAY_WIDTH / 2 + 12, P.TRAY_WIDTH / 2 - 12, -P.TRAY_T - 1.0, -P.TRAY_T)
    return R.link_drawer() + [("tongue_spring_steel", tongue, SPRING_STEEL, False)]


# ------------------------------------------------------------------ stand for configurations 1 and 2
LIFT_DRIVE = {"rail_mgn12", "nema17", "nema17_mount", "lift_pulley_20T", "lift_reduction_60T", "motor_pulley_20T",
              "lift_lower_bearing_block", "lift_top_idler", "lift_top_block", "lift_belt", "balancer_hanger",
              "spring_balancer_3_5kg"}
DRIVE = {"hub_motor", "hub_motor_hub", "axle", "axle_clamp_adapter", "caster_wheel", "caster_fork",
         "front_caster_pad_tpu", "rear_caster_spring_mount", "battery_hoverboard_36v", "electronics_box",
         "pi_and_drivers", "lidar_post", "lidar_rplidar_c1", "lamp_holder", "lamp_led_disk", "lamp_shade"}
FACE = {"face_mount", "face_phone", "face_eye"}
ARM_B_CARRIAGE = {"servo_J1", "bearing_6000", "pulley_20T_J1", "belt_J1_200"}   # mirrored, same labels as arm a's


def stand_items():
    """Feet that put the frame at its caster height (printed posts or wooden blocks) and a 5 kg plate as ballast."""
    out = []
    z0 = P.FRAME_Z0
    rail_y = P.TRACK / 2 - 20
    for x in (P.BASE_X_REAR + 20, P.BASE_X_FRONT - 20):
        for y in (-rail_y, rail_y):
            out.append(("stand_foot", R.box(x - 20, x + 20, y - 20, y + 20, 8, z0), R.PRINT_B, True))
            out.append(("rubber_foot", R.cyl_z(x, y, 0, 8, 36), R.RUBBER, False))
    for y in (-P.FRONT_CASTER_Y, P.FRONT_CASTER_Y):
        x = P.FRONT_CASTER_X
        out.append(("stand_foot", R.box(x - 20, x + 20, y - 20, y + 20, 8, z0), R.PRINT_B, True))
        out.append(("rubber_foot", R.cyl_z(x, y, 0, 8, 36), R.RUBBER, False))
    zt = z0 + 20 + 6
    out.append(("ballast_5kg_plate", R.cyl_z(P.BASE_X_REAR + 110, 0, zt, zt + 28, 200), (0.2, 0.2, 0.22), False))
    return out


def carriage_items(order):
    items = R.link_carriage()
    out = []
    for lab, shp, rgb, pr in items:
        if order < 3 and lab in FACE:
            continue
        left = shp.bounding_box().center().Y > 1.0
        if order == 1 and (lab.endswith("_b") or (lab in ARM_B_CARRIAGE and left)):
            continue
        if order == 1 and lab == "mgn12h_block":            # no rail yet: a printed block bolts the carriage on
            out.append(("mast_fixed_block", shp, R.PRINT_B, True))
            continue
        out.append((lab, shp, rgb, pr))
    return out


def scoop_tool(drop):
    """Concept: a long-handled scoop the stump carries by its loop (wrist frame). The handle slides in the loop, so the
    plate rests on the floor under its own weight. drop: loop height above the floor (mm)."""
    lx, ly, lz = 155.0, SHANK_Y + 11, HOOK_Z + 19
    zf = lz - drop
    handle = R.cyl_z(lx, ly, zf + 2, lz + 25, 8)
    plate = R.box(lx - 10, lx + 110, ly - 120, ly + 10, zf, zf + 1.5)
    edge = Pos(lx + 110, ly - 120, zf) * (Rot(0, 0, 0) * R.box(0, 8, 0, 130, 0, 1.0))
    return [("scoop_tool_concept", handle + plate, CONCEPT, False), ("scoop_feather_edge", edge, CONCEPT, False)]


SCOOP_DROP = {"value": None}     # set per view: draw the scoop tool on the stump


def links_v03(order=3):
    base = [it for it in R.link_base() if not (order < 3 and it[0] in DRIVE) and not (order < 2 and it[0] in LIFT_DRIVE)]
    if order < 3:
        base += stand_items()
    out = dict(base=base, carriage=carriage_items(order), drawer=link_drawer_v03())
    arm_a = dict(upper_arm=R.link_upper_arm(), forearm=R.link_forearm(), wrist=R.link_wrist(), hand=R.link_hand(),
                 tool=link_tool_v03(), finger_l=R.link_finger(1), finger_r=R.link_finger(-1))
    for name, items in arm_a.items():
        out[f"{name}_a"] = items
    if order >= 2:
        out["upper_arm_b"] = R.link_upper_arm()
        out["forearm_b"] = R.link_forearm()
        out["wrist_b"] = link_stump_head() + (scoop_tool(SCOOP_DROP["value"]) if SCOOP_DROP["value"] else [])
    return out


# ------------------------------------------------------------------ CAP's own printed parts (meshes)
def cap_meshes(T_tool):
    """CAP's printed STLs placed in the robot: STL -> CAP frame (cad/ir_camera.py CAP_PARTS, bottom plate by the
    same ICP) -> tool frame (robot.CAP_LOC) -> world."""
    import trimesh
    T_cap = loc_matrix(T_tool * R.CAP_LOC)
    out = []
    for nm, T in {**IRC.CAP_PARTS, **CAP_BOTTOM}.items():
        m = trimesh.load(IRC.CAP_STL / f"{nm}.stl")
        m.apply_transform(T_cap @ np.array(T))
        out.append((f"cap_{nm}", m, R.CAP_WHITE))
    return out


# bottom plate, placed by the same ICP as CAP_PARTS (cad/ir_camera.py --align; median 0.93 mm, p90 4.3 mm: visual only)
CAP_BOTTOM = {"bottom-plate": [[0.99966, -0.02602, 0.00102, -129.0483], [0.02604, 0.99955, -0.01521, 93.16177],
                               [-0.00063, 0.01523, 0.99988, 25.29738], [0, 0, 0, 1]]}


def loc_matrix(loc):
    """build123d Location -> 4x4 numpy matrix (mm)."""
    t = loc.wrapped.Transformation()
    M = np.eye(4)
    for i in range(3):
        for j in range(4):
            M[i, j] = t.Value(i + 1, j + 1)
    return M


def paint(m, rgb):
    """One PBR material per mesh. glTF colours are linear, ours sRGB."""
    import trimesh
    lin = [float(c) ** 2.2 for c in rgb[:3]]
    m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=lin + [1.0], metallicFactor=0.0, roughnessFactor=0.6))


def to_glb(q, order, props, path, tol=0.4, floor=True):
    """Tessellate every part, add CAP's meshes, write one glTF binary (metres, y up). floor=False leaves the floor
    plate out (web viewers frame the model by its bounding box and draw their own ground)."""
    import trimesh
    T = R.link_transforms(q)
    scene = trimesh.Scene()
    total = {}
    for name, items in links_v03(order).items():
        for lab, shp, rgb, pr in items:
            if name in ("tool_a",) and lab.startswith("cap_") and lab != "cap_adapter_plate":
                continue
            s = T[name] * shp
            v, f = s.tessellate(tol, 0.3)
            if not f:
                continue
            m = trimesh.Trimesh(np.array([(p.X, p.Y, p.Z) for p in v]), np.array(f), process=False)
            paint(m, rgb)
            scene.add_geometry(m, node_name=f"{name}/{lab}/{len(scene.geometry)}")
            total[lab] = total.get(lab, 0) + 1
    for lab, m, rgb in cap_meshes(T["tool_a"]):
        paint(m, rgb)
        scene.add_geometry(m, node_name=f"tool_a/{lab}")
    for shp in props:
        if not floor and shp.label == "floor":
            continue
        v, f = shp.tessellate(tol, 0.3)
        if not f:
            continue
        m = trimesh.Trimesh(np.array([(p.X, p.Y, p.Z) for p in v]), np.array(f), process=False)
        rgb = tuple(shp.color)[:3] if shp.color is not None else (0.8, 0.8, 0.8)
        paint(m, rgb)
        scene.add_geometry(m, node_name=f"props/{shp.label or 'prop'}/{len(scene.geometry)}")
    # mm, z up -> m, y up
    M = np.diag([0.001, 0.001, 0.001, 1.0]) @ np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    scene.apply_transform(M)
    scene.export(path)
    return total


def views():
    """(file, configuration, q, props, scoop drop, title). Poses from cad/scene.py where they fit."""
    hero_q, _ = S.scene_hero()
    fr_q, fr_props = S.scene_fridge()
    bed_q, bed_props = S.scene_bed()
    tuck_b = {k: v for k, v in S.TUCK_B.items() if k in ("j1b", "j2b")} | {"j3b": 90.0}
    # configuration 1: the bench stand. Carriage bolted at table height, drawer out, tongue on the table edge, hand at a bottle
    lift1 = 650
    table_top = lift1 - P.TRAY_T - 1
    tray_front = P.TRAY_X0 + P.TRAY_DEPTH + P.DRAWER_STROKE
    bench = S.pose_for_tip(tray_front + 165, -60, lift1 + 40, pitch=0, grip=40, reach_to=P.PAD_X)
    bench.update(lift=lift1, drawer=P.DRAWER_STROKE)
    table = S.colored(R.box(tray_front + 15, tray_front + 615, -400, 400, table_top - 25, table_top), (0.72, 0.6, 0.45),
                      "table")
    legs = [S.colored(R.box(x, x + 30, y, y + 30, 0, table_top - 25), (0.72, 0.6, 0.45), "table_leg")
            for x in (tray_front + 40, tray_front + 560) for y in (-370, 340)]
    props1 = S.floor() + [table] + legs + S.bottle(tray_front + 165, -60, table_top) + \
        S.meal_box(tray_front + 40, 40, table_top)
    # configuration 2: floor. The hand's fingertips right of a phone, the stump's scoop plate left of it: sweep it on
    px = 420
    fl = S.pose_for_tip(px, -50, 10, pitch=90, grip=30, reach_to=P.PAD_X)
    wb = S.arm_ik((px - 35, 305), elbow=-1, hand_yaw=-90.0, arm="b")
    fl.update(j1b=wb[0], j2b=wb[1], j3b=wb[2])
    drop = fl["lift"] + P.L2_ABOVE_TRAY + HOOK_Z + 19
    return [
        ("v03_config1.glb", 1, bench, props1, None,
         "Configuration 1: arm with CAP gripper and IR camera, tongue drawer, on the stand. Carriage bolted at table height"),
        ("v03_config2.glb", 2, fl, S.floor() + S.phone(px, 0), drop,
         "Configuration 2: + lift and stump. Floor: the stump holds a scoop plate, the hand sweeps (scoop drawn as a concept)"),
        ("v03_config3.glb", 3, dict(hero_q, **tuck_b), S.floor(), None,
         "Configuration 3: + drive, battery, lidar, lamp, face phone: the robot"),
        ("v03_fridge.glb", 3, dict(fr_q, **tuck_b), fr_props, None, "At the fridge: the hand pulls a box onto the tray"),
        ("v03_bed.glb", 3, dict(bed_q, **tuck_b), bed_props, None, "At the bed: tray out at mattress height"),
    ]


if __name__ == "__main__":
    info = []
    viewer = "--viewer" in sys.argv               # out/v03_*_viewer.glb, no floor plate
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for fn, order, q, props, drop, title in views():
        if only and fn not in only:
            continue
        SCOOP_DROP["value"] = drop
        if viewer:
            fn = fn.replace(".glb", "_viewer.glb")
        counts = to_glb(q, order, props, OUT / fn, floor=not viewer)
        sz = (OUT / fn).stat().st_size / 1e6
        info.append(dict(file=fn, order=order, title=title, parts=sum(counts.values()), mb=round(sz, 1)))
        print(f"{fn}: {sum(counts.values())} parts, {sz:.1f} MB")
    if not viewer:
        (OUT / "v03_views.json").write_text(json.dumps(info, indent=1))
