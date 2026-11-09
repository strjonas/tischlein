"""Render a scene GLB with Blender (headless).

/Applications/Blender.app/Contents/MacOS/Blender -b -P render/blender_render.py -- \
    out/scene_hero.glb out/render_hero.png  cam_x cam_y cam_z  look_x look_y look_z  [lens_mm]
Coordinates in metres, robot frame (x forward, y left, z up).
"""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out_png = argv[0], argv[1]
cam = Vector(tuple(float(v) for v in argv[2:5]))
look = Vector(tuple(float(v) for v in argv[5:8]))
lens = float(argv[8]) if len(argv) > 8 else 35.0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)

# glTF is y-up; the importer converts to Blender's z-up, matching our robot frame.
# build123d writes millimetres as metres*1000? Normalise by bounding size.
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
mx = max(max(abs(v) for v in o.dimensions) for o in objs)
scale = 0.001 if mx > 50 else 1.0
if scale != 1.0:
    for o in bpy.context.scene.objects:
        if o.parent is None:
            o.scale = (o.scale[0] * scale, o.scale[1] * scale, o.scale[2] * scale)
            o.location = o.location * scale

# soften materials: slightly rough plastic look
for m in bpy.data.materials:
    if m.use_nodes:
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Roughness"].default_value = 0.55
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.35

for o in objs:
    for p in o.data.polygons:
        p.use_smooth = False

scene = bpy.context.scene
world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.93, 0.94, 0.96, 1)
bg.inputs[1].default_value = 0.9

sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 3.2
sun.angle = math.radians(8)
so = bpy.data.objects.new("sun", sun)
so.rotation_euler = (math.radians(50), math.radians(10), math.radians(35))
scene.collection.objects.link(so)
area = bpy.data.lights.new("fill", "AREA")
area.energy = 350
area.size = 3
ao = bpy.data.objects.new("fill", area)
ao.location = (cam.x * 0.6, cam.y * 0.6 + 1.0, 2.6)
ao.rotation_euler = (math.radians(30), 0, 0)
scene.collection.objects.link(ao)

camd = bpy.data.cameras.new("cam")
camd.lens = lens
camo = bpy.data.objects.new("cam", camd)
camo.location = cam
direction = look - cam
camo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(camo)
scene.camera = camo

scene.render.engine = "CYCLES"
scene.cycles.samples = int(__import__("os").environ.get("SAMPLES", "64"))
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
scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
scene.render.resolution_x = 1600
scene.render.resolution_y = 1150
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print("wrote", out_png)
