"""Render the same robot in different "skins": bare, industrial, cute, minimal, furniture.

/Applications/Blender.app/Contents/MacOS/Blender -b -P render/variants.py -- out/scene_hero.glb out/variant_<skin>.png <skin>

Every skin keeps the mechanics exactly as they are: the shells only cover parts that do not move relative
to each other (base frame, mast back and sides, arm links), and bellows cover the mast above and below
the carriage. The phone that is the head camera anyway becomes a face.
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out_png, skin = argv[0], argv[1], argv[2]
LIFT = float(argv[3]) if len(argv) > 3 else 0.52    # tray height in the scene (m)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
mx = max(max(abs(v) for v in o.dimensions) for o in objs)
if mx > 50:
    for o in bpy.context.scene.objects:
        if o.parent is None:
            o.scale = tuple(s * 0.001 for s in o.scale)
            o.location = o.location * 0.001


# ---------------------------------------------------------------- materials
def mat(name, rgb, rough=0.5, metal=0.0, emit=None, emit_strength=0.0, sheen=0.0, coat=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    return m


def knit(name, rgb):
    """Fabric look: high roughness, sheen, fine noise bump."""
    m = mat(name, rgb, rough=0.95, sheen=0.15)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexWave")
    tex.inputs["Scale"].default_value = 160
    tex.inputs["Distortion"].default_value = 2.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def wood(name, light=(0.62, 0.43, 0.26), dark=(0.42, 0.26, 0.14)):
    m = mat(name, light, rough=0.45, coat=0.2)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexWave")
    tex.wave_type = "RINGS"
    tex.inputs["Scale"].default_value = 3.0
    tex.inputs["Distortion"].default_value = 6.0
    tex.inputs["Detail"].default_value = 3.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].color = (*light, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def group(name):
    n = name.lower()
    if n.startswith(("lamp_shade", "lamp_led")):
        return "lamp"
    if n.startswith(("face_phone", "face_eye")):
        return "face"
    if n.startswith("spring_balancer"):
        return "balancer"
    if n.startswith(("cap_gripper", "cap_phone", "cap_finger", "iphone")):
        return "cap"
    if n.startswith(("bed_", "mattress", "duvet", "pillow", "bedside")):
        return "scene"
    if n.startswith(("bottle", "meal_box")):
        return "prop"
    if n.startswith("floor"):
        return "floor"
    if n.startswith(("mast", "rail_mgn", "lift_belt", "lift_top", "mgn12h")):
        return "mast"
    if n.startswith(("frame_", "deck", "straddle", "battery", "pi_and", "electronics", "estop", "lidar", "mast_foot",
                     "nema17", "lift_lower", "lift_pulley", "lift_reduction", "motor_pulley", "rear_caster")):
        return "base"
    if n.startswith(("hub_motor", "axle")):
        return "wheel"
    if n.startswith(("caster", "front_caster")):
        return "caster"
    if n.startswith(("tray_half", "tray_seam", "tray_belt")):
        return "tray"
    if n.startswith(("servo", "nema")):
        return "servo"
    if n.startswith(("l1_2020", "l2_2020", "tray_arm", "yoke_2020", "slide_250", "rod_4mm", "shaft_", "bearing")):
        return "alu"
    if n.startswith(("finger_finray", "front_caster_pad", "suction")):
        return "soft"
    if n.startswith("head_camera"):
        return "head"
    return "print"


for o in objs:
    o["grp"] = group(o.name)
    for p in o.data.polygons:
        p.use_smooth = False


def paint(grp, m):
    for o in objs:
        if o["grp"] == grp:
            o.data.materials.clear()
            o.data.materials.append(m)


def hide(*grps):
    for o in objs:
        if o["grp"] in grps:
            o.hide_render = True


# ---------------------------------------------------------------- shapes
def rounded_box(name, size, loc, bevel, m, segs=8):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    mod = o.modifiers.new("bevel", "BEVEL")
    mod.width = bevel
    mod.segments = segs
    mod.limit_method = "NONE"
    o.data.materials.append(m)
    bpy.ops.object.shade_smooth()
    return o


def cyl(name, r, h, loc, m, verts=48, rot=(0, 0, 0), scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=loc, vertices=verts, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.data.materials.append(m)
    bpy.ops.object.shade_smooth()
    return o


def bellows(name, z0, z1, r, m, pitch=0.018, x=0.195):
    """Accordion cover around the mast between z0 and z1."""
    n = max(2, int((z1 - z0) / pitch))
    parts = []
    for i in range(n):
        z = z0 + (i + 0.5) * (z1 - z0) / n
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=(z1 - z0) / n * 0.55,
                                         location=(x, 0, z), major_segments=32, minor_segments=8)
        t = bpy.context.object
        t.scale = (1.0, 0.75, 1.0)
        parts.append(t)
    cyl(name + "_core", r * 0.92, z1 - z0, (x, 0, (z0 + z1) / 2), m, scale=(1, 0.75, 1))
    for t in parts:
        t.data.materials.append(m)
        bpy.ops.object.select_all(action="DESELECT")
    return parts


def face(name, loc, w, h, screen_m, eye_m, style="round", yaw=0.0, frame_m=None, depth=0.012):
    """A phone as a face: dark screen plus two eyes, looking along +x."""
    x, y, z = loc
    if frame_m is not None:
        rounded_box(name + "_shell", (depth + 0.012, w + 0.016, h + 0.016), (x - 0.006, y, z), 0.012, frame_m)
    rounded_box(name + "_screen", (0.004, w, h), (x + depth / 2, y, z), 0.006, screen_m)
    ex = x + depth / 2 + 0.003
    for s in (-1, 1):
        if style == "round":
            cyl(name + f"_eye{s}", h * 0.13, 0.002, (ex, y + s * w * 0.21, z + h * 0.05), eye_m, rot=(0, math.pi / 2, 0),
                scale=(1.0, 1.0, 1.35))
        elif style == "pill":
            cyl(name + f"_eye{s}", h * 0.07, 0.002, (ex, y + s * w * 0.2, z), eye_m, rot=(0, math.pi / 2, 0), scale=(1.0, 0.6, 1.0))
        elif style == "dot":
            cyl(name + f"_eye{s}", h * 0.05, 0.002, (ex, y + s * w * 0.16, z + h * 0.04), eye_m, rot=(0, math.pi / 2, 0))


MAST_X, MAST_TOP = 0.195, 1.52
TRAY_Z = LIFT
CAR_Z0, CAR_Z1 = TRAY_Z - 0.06, TRAY_Z + 0.20      # carriage + arm stack on the mast

# ---------------------------------------------------------------- skins
bg_rgb = (0.93, 0.94, 0.96)
if skin == "bare":
    pass

elif skin == "industrial":
    black = mat("anod_black", (0.03, 0.03, 0.035), rough=0.35, metal=0.8)
    orange = mat("safety_orange", (0.95, 0.30, 0.02), rough=0.45)
    yellow = mat("hazard", (0.95, 0.75, 0.05), rough=0.5)
    steel = mat("steel", (0.6, 0.6, 0.62), rough=0.3, metal=1.0)
    for g in ("mast", "alu"):
        paint(g, black)
    paint("print", orange)
    paint("tray", orange)
    paint("base", black)
    # chequer-plate deck and hazard stripe bumper
    rounded_box("bumper", (0.03, 0.46, 0.06), (0.19, 0, 0.17), 0.008, yellow)
    # work-light head: phone in a black cage, round amber eyes
    face("head", (MAST_X + 0.035, 0, MAST_TOP + 0.02), 0.075, 0.15, mat("scr", (0.01, 0.01, 0.012), 0.2),
         mat("amber", (1, 0.55, 0.1), emit=(1, 0.55, 0.1), emit_strength=6), style="round", frame_m=black)
    hide("head")

elif skin == "cute":
    cream = mat("cream", (0.93, 0.89, 0.80), rough=0.55, coat=0.3)
    sage = knit("sage_knit", (0.55, 0.65, 0.52))
    terracotta = mat("terracotta", (0.80, 0.42, 0.30), rough=0.6)
    oak = wood("oak")
    paint("tray", oak)
    paint("print", cream)
    paint("alu", cream)
    # pebble base shell over the frame (wheels stay free underneath)
    rounded_box("base_shell", (0.50, 0.50, 0.20), (-0.04, 0, 0.21), 0.09, cream, segs=10)
    rounded_box("bumper", (0.06, 0.40, 0.05), (0.21, 0, 0.17), 0.024, terracotta)
    for s in (-1, 1):
        cyl(f"leg_sleeve{s}", 0.022, 0.16, (0.27, s * 0.2, 0.15), sage, rot=(0, math.pi / 2, 0))
    # knit bellows on the mast above and below the carriage
    bellows("bel_lo", 0.31, CAR_Z0, 0.034, sage, pitch=0.03)
    bellows("bel_hi", CAR_Z1, MAST_TOP - 0.06, 0.034, sage, pitch=0.03)
    cyl("neck", 0.03, 0.08, (MAST_X, 0, MAST_TOP - 0.03), cream)
    # round head with big eyes and a little cap
    cyl("head_ball", 0.095, 0.11, (MAST_X + 0.0, 0, MAST_TOP + 0.03), cream, rot=(0, math.pi / 2, 0))
    face("head", (MAST_X + 0.05, 0, MAST_TOP + 0.03), 0.10, 0.10, mat("scr", (0.02, 0.025, 0.03), 0.25),
         mat("eyes", (0.95, 0.97, 1.0), emit=(0.95, 0.97, 1.0), emit_strength=4), style="round", depth=0.004)
    cap = cyl("cap", 0.085, 0.05, (MAST_X - 0.01, 0, MAST_TOP + 0.115), terracotta)
    cyl("cap_brim", 0.06, 0.008, (MAST_X + 0.07, 0, MAST_TOP + 0.092), terracotta, scale=(1.2, 1.0, 1.0))
    hide("head", "mast")
    bg_rgb = (0.96, 0.94, 0.90)

elif skin == "minimal":
    white = mat("matte_white", (0.86, 0.86, 0.85), rough=0.6)
    graphite = mat("graphite", (0.08, 0.085, 0.09), rough=0.5)
    fabric = knit("grey_knit", (0.36, 0.37, 0.38))
    light = mat("strip", (0.8, 0.9, 1.0), emit=(0.8, 0.9, 1.0), emit_strength=8)
    paint("print", white)
    paint("alu", graphite)
    paint("tray", white)
    paint("servo", graphite)
    rounded_box("base_shell", (0.48, 0.47, 0.19), (-0.045, 0, 0.205), 0.05, white, segs=12)
    rounded_box("base_band", (0.485, 0.475, 0.012), (-0.045, 0, 0.135), 0.006, graphite)
    rounded_box("light_strip", (0.004, 0.30, 0.008), (0.196, 0, 0.255), 0.002, light)
    # one slim knit sleeve over the back and sides of the mast (front slot for the carriage)
    rounded_box("mast_sleeve_lo", (0.06, 0.05, CAR_Z0 - 0.30), (MAST_X - 0.012, 0, (0.30 + CAR_Z0) / 2), 0.02, fabric)
    rounded_box("mast_sleeve_hi", (0.06, 0.05, MAST_TOP - 0.08 - CAR_Z1), (MAST_X - 0.012, 0, (CAR_Z1 + MAST_TOP - 0.08) / 2), 0.02, fabric)
    # head: dark glossy visor pill with two faint light dots
    rounded_box("head_shell", (0.11, 0.15, 0.13), (MAST_X + 0.005, 0, MAST_TOP + 0.02), 0.05, white, segs=12)
    rounded_box("visor", (0.02, 0.13, 0.075), (MAST_X + 0.055, 0, MAST_TOP + 0.025), 0.03,
                mat("visor", (0.01, 0.01, 0.012), rough=0.08, coat=1.0), segs=10)
    face("head", (MAST_X + 0.058, 0, MAST_TOP + 0.025), 0.10, 0.06, mat("noscreen", (0.01, 0.01, 0.012), 0.08),
         mat("dots", (0.7, 0.85, 1.0), emit=(0.7, 0.85, 1.0), emit_strength=5), style="pill", depth=0.002)
    hide("head", "mast")
    bg_rgb = (0.90, 0.90, 0.90)

elif skin == "minimal_lamp":
    # v0.2 target look: minimal shells + the lamp on top + oak tray + a small face on the carriage.
    # Positions come from params.py, so this skin follows the CAD.
    import os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(bpy.data.filepath or __file__), "..")))
    import params as PP
    mm = 0.001
    white = mat("matte_white", (0.86, 0.86, 0.85), rough=0.6)
    graphite = mat("graphite", (0.08, 0.085, 0.09), rough=0.5)
    fabric = knit("grey_knit", (0.42, 0.43, 0.44))
    light = mat("strip", (1.0, 0.85, 0.6), emit=(1.0, 0.85, 0.6), emit_strength=6)
    oak = wood("oak")
    paint("print", white)
    paint("alu", graphite)
    paint("servo", graphite)
    paint("tray", oak)
    paint("lamp", mat("shade_warm", (0.96, 0.92, 0.84), rough=0.8, emit=(1.0, 0.8, 0.55), emit_strength=2.0))
    paint("face", mat("face_glass", (0.02, 0.02, 0.025), rough=0.1, coat=1.0))
    hide("balancer", "mast")
    for o in objs:
        if o.name.lower().startswith("face_eye"):
            o.data.materials.clear()
            o.data.materials.append(mat("eye_glow", (1.0, 0.85, 0.6), emit=(1.0, 0.85, 0.6), emit_strength=4))
    x0, x1 = PP.BASE_X_REAR * mm - 0.012, PP.BASE_X_FRONT * mm + 0.012
    rounded_box("base_shell", (x1 - x0, 0.47, 0.19), ((x0 + x1) / 2, 0, 0.215), 0.05, white, segs=12)
    rounded_box("base_band", (x1 - x0 + 0.004, 0.474, 0.012), ((x0 + x1) / 2, 0, 0.14), 0.006, graphite)
    rounded_box("light_strip", (0.004, 0.30, 0.008), (x1 + 0.001, 0, 0.26), 0.002, light)
    mx = PP.MAST_X * mm
    top = (PP.MAST_Z0 + PP.MAST_LEN) * mm
    z_lo1, z_lo2 = 0.31, TRAY_Z - 0.06
    z_hi1, z_hi2 = TRAY_Z + (PP.FACE_Z_ABOVE_TRAY + PP.FACE_H / 2 + 20) * mm, top - 0.13
    # one slim knit sleeve over the mast's back and sides (front slot for the carriage), a cover over the balancer
    rounded_box("mast_sleeve_lo", (0.07, 0.05, z_lo2 - z_lo1), (mx - 0.012, 0, (z_lo1 + z_lo2) / 2), 0.022, fabric)
    rounded_box("mast_sleeve_hi", (0.07, 0.05, z_hi2 - z_hi1), (mx - 0.012, 0, (z_hi1 + z_hi2) / 2), 0.022, fabric)
    rounded_box("balancer_cover", (0.13, 0.11, 0.16), (mx - 0.06, 0, top - 0.06), 0.04, white, segs=10)
    lamp = bpy.data.lights.new("lamp", "POINT")
    lamp.energy = 30
    lamp.color = (1.0, 0.8, 0.55)
    lo = bpy.data.objects.new("lamp", lamp)
    lo.location = (mx, 0, top + 0.09)
    bpy.context.scene.collection.objects.link(lo)
    bg_rgb = (0.93, 0.92, 0.90)

elif skin == "furniture":
    oak = wood("oak")
    walnut = wood("walnut", light=(0.36, 0.22, 0.13), dark=(0.20, 0.11, 0.06))
    linen = knit("linen", (0.80, 0.76, 0.68))
    felt = knit("felt", (0.28, 0.30, 0.33))
    brass = mat("brass", (0.80, 0.62, 0.32), rough=0.3, metal=1.0)
    shade_m = mat("shade", (0.95, 0.90, 0.80), rough=0.8, emit=(1.0, 0.78, 0.5), emit_strength=2.5)
    paint("tray", oak)
    paint("print", mat("warmgrey", (0.55, 0.52, 0.48), 0.6))
    paint("alu", brass)
    # felted plinth base
    rounded_box("plinth", (0.49, 0.48, 0.18), (-0.045, 0, 0.20), 0.04, felt, segs=6)
    rounded_box("plinth_top", (0.50, 0.49, 0.02), (-0.045, 0, 0.30), 0.008, mat("walnut_flat", (0.30, 0.18, 0.10), rough=0.45, coat=0.3))
    # linen sleeve on the mast, a lamp shade on top (reading light + night light + camera light)
    rounded_box("mast_sleeve_lo", (0.06, 0.05, CAR_Z0 - 0.31), (MAST_X - 0.012, 0, (0.31 + CAR_Z0) / 2), 0.022, linen)
    rounded_box("mast_sleeve_hi", (0.06, 0.05, MAST_TOP - 0.12 - CAR_Z1), (MAST_X - 0.012, 0, (CAR_Z1 + MAST_TOP - 0.12) / 2), 0.022, linen)
    bpy.ops.mesh.primitive_cone_add(radius1=0.15, radius2=0.09, depth=0.17, location=(MAST_X - 0.01, 0, MAST_TOP + 0.03), vertices=64)
    sh = bpy.context.object
    sh.data.materials.append(shade_m)
    bpy.ops.object.shade_smooth()
    lamp = bpy.data.lights.new("lamp", "POINT")
    lamp.energy = 25
    lamp.color = (1.0, 0.8, 0.55)
    lo = bpy.data.objects.new("lamp", lamp)
    lo.location = (MAST_X - 0.01, 0, MAST_TOP - 0.02)
    bpy.context.scene.collection.objects.link(lo)
    # a small, quiet face on the carriage, at the eye level of someone in bed
    face("face", (0.24, 0, TRAY_Z + 0.27), 0.07, 0.05, mat("scr", (0.03, 0.03, 0.03), 0.3),
         mat("eyes", (1.0, 0.85, 0.6), emit=(1.0, 0.85, 0.6), emit_strength=3), style="dot", frame_m=walnut)
    hide("head", "mast")
    bg_rgb = (0.95, 0.93, 0.89)

import os
hide("floor")
if skin == "bare" and os.environ.get("HIDE_PROPS"):
    hide("prop")
if skin != "bare" and not os.environ.get("KEEP_PROPS"):
    hide("prop")

# ---------------------------------------------------------------- scene
scene = bpy.context.scene
world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (*bg_rgb, 1)
bg.inputs[1].default_value = 0.9
bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -0.0005))
floor = bpy.context.object
floor.data.materials.append(mat("floor", tuple(c * 0.92 for c in bg_rgb), rough=0.7))

sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 3.0
sun.angle = math.radians(10)
so = bpy.data.objects.new("sun", sun)
so.rotation_euler = (math.radians(50), math.radians(10), math.radians(35))
scene.collection.objects.link(so)
area = bpy.data.lights.new("fill", "AREA")
area.energy = 300
area.size = 3
ao = bpy.data.objects.new("fill", area)
ao.location = (1.6, -0.3, 2.6)
ao.rotation_euler = (math.radians(30), 0, 0)
scene.collection.objects.link(ao)

env = os.environ
cam = Vector(tuple(float(v) for v in env.get("CAM", "1.75,-1.15,1.15").split(",")))
look = Vector(tuple(float(v) for v in env.get("LOOK", "0.10,0.0,0.80").split(",")))
camd = bpy.data.cameras.new("cam")
camd.lens = float(env.get("LENS", "38"))
camo = bpy.data.objects.new("cam", camd)
camo.location = cam
camo.rotation_euler = (look - cam).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(camo)
scene.camera = camo

scene.render.engine = "CYCLES"
scene.cycles.samples = int(__import__("os").environ.get("SAMPLES", "48"))
scene.cycles.use_denoising = True
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception:
    pass
scene.view_settings.view_transform = "AgX"
rx, ry = (int(v) for v in env.get("RES", "900x1200").split("x"))
scene.render.resolution_x = rx
scene.render.resolution_y = ry
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print("wrote", out_png)
