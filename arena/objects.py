"""The benchmark objects. Real sizes and masses of shop items, no lugs, tabs or robot grips.

The list follows Choi, Deyle, Kemp 2009 (what people with ALS ask a robot to retrieve, arXiv 0902.2186), plus what the
value model counts: a drink, a lidded meal box, a mug, a bowl, a parcel, a sealed bag. Masses are typical; each trial
draws +-15 %. Rounded things are built from capsules and cylinders, thin things are as thin as they are: a thing thinner
than contact_floor() is flagged, because the physics cannot judge it (the old "coin" result came from contact softness).

Each object lists its grasps: where a parallel gripper can close on it, as (point in the object frame, jaw axis,
width between the jaws, allowed approach directions). A robot's own hand then decides which of them it can use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Grasp:
    p: tuple                  # object frame
    jaw: tuple                # closing axis, object frame
    width: float              # distance across the object here
    approach: tuple = ()      # allowed approach directions in the object frame (along the fingers, into the object)
    name: str = ""


@dataclass
class Obj:
    name: str
    mass: float
    geoms: list               # (type, size, pos[, fromto])
    material: str = "plastic"
    grasps: list = field(default_factory=list)
    size: tuple = (0.1, 0.1, 0.1)     # bounding size, m
    rest_z: float = 0.0               # object origin height above its support
    rgba: str = "0.8 0.6 0.3 1"
    liquid: dict = None               # bowl-like: fill level and rim, for the spill check
    soft: bool = False


def _sides(sx, sy, z):
    """Side grasps of an upright thing with footprint sx x sy at height z: jaws across each horizontal axis, approach
    along the other horizontal axis or from above."""
    return [Grasp((0, 0, z), (0, 1, 0), sy, ((1, 0, 0), (-1, 0, 0), (0, 0, -1)), "across y"),
            Grasp((0, 0, z), (1, 0, 0), sx, ((0, 1, 0), (0, -1, 0), (0, 0, -1)), "across x")]


def _round(r, z):
    """A round thing: jaws across it from any side; eight approaches stand for the circle."""
    a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    out = []
    for t in a:
        c, s = float(np.cos(t)), float(np.sin(t))
        out.append(Grasp((0, 0, z), (-s, c, 0), 2 * r, ((c, s, 0),), f"round {int(np.degrees(t))}"))
    return out + [Grasp((0, 0, z), (0, 1, 0), 2 * r, ((0, 0, -1),), "round top")]


def box(sx, sy, sz, c=(0, 0, 0)):
    return ("box", (sx / 2, sy / 2, sz / 2), c)


def cyl(r, h, c=(0, 0, 0)):
    return ("cylinder", (r, h / 2), c)


def caps(a, b, r):
    return ("capsule", (r,), None, (*a, *b))


CATALOG = {
    # ----- what falls off the bed (ALS list, top of the ranking)
    "phone": Obj("phone", 0.19, [box(0.074, 0.150, 0.008, (0, 0, 0.004))], "glass", _sides(0.074, 0.150, 0.004)[:1] + [
        Grasp((0, 0, 0.004), (1, 0, 0), 0.074, ((0, 1, 0), (0, -1, 0), (0, 0, -1)), "across x")], (0.074, 0.15, 0.008), 0.0, "0.10 0.10 0.12 1"),
    "remote": Obj("remote", 0.12, [box(0.046, 0.170, 0.022, (0, 0, 0.011))], "plastic", _sides(0.046, 0.170, 0.011), (0.046, 0.17, 0.022), 0.0, "0.15 0.15 0.18 1"),
    "pill_bottle": Obj("pill_bottle", 0.06, [cyl(0.021, 0.070, (0, 0, 0.035))], "plastic", _round(0.021, 0.04), (0.042, 0.042, 0.07), 0.0, "0.90 0.55 0.20 1"),
    "glasses": Obj("glasses", 0.03, [box(0.135, 0.016, 0.040, (0, 0, 0.020)), caps((-0.06, 0.012, 0.03), (-0.06, 0.055, 0.03), 0.003),
                                     caps((0.06, 0.012, 0.03), (0.06, 0.055, 0.03), 0.003)], "plastic",
                    [Grasp((0, 0, 0.022), (0, 1, 0), 0.016, ((1, 0, 0), (-1, 0, 0), (0, 0, -1)), "frame")], (0.135, 0.07, 0.04), 0.0, "0.25 0.20 0.20 1"),
    "spoon": Obj("spoon", 0.03, [box(0.012, 0.150, 0.003, (0, 0, 0.0015)), box(0.038, 0.050, 0.004, (0, 0.095, 0.002))], "steel",
                 [Grasp((0, -0.02, 0.0015), (1, 0, 0), 0.012, ((0, 1, 0), (0, -1, 0), (0, 0, -1)), "stem")], (0.038, 0.20, 0.004), 0.0, "0.75 0.76 0.78 1"),
    "toothbrush": Obj("toothbrush", 0.02, [box(0.012, 0.190, 0.012, (0, 0, 0.006))], "plastic",
                      [Grasp((0, 0, 0.006), (1, 0, 0), 0.012, ((0, 1, 0), (0, -1, 0), (0, 0, -1)), "shaft")], (0.012, 0.19, 0.012), 0.0, "0.30 0.70 0.90 1"),
    "tablet_box": Obj("tablet_box", 0.04, [box(0.055, 0.105, 0.020, (0, 0, 0.010))], "paper", _sides(0.055, 0.105, 0.010), (0.055, 0.105, 0.02), 0.0, "0.95 0.95 0.92 1"),
    "tissue_pack": Obj("tissue_pack", 0.03, [box(0.075, 0.110, 0.035, (0, 0, 0.0175))], "paper", _sides(0.075, 0.11, 0.0175), (0.075, 0.11, 0.035), 0.0, "0.80 0.90 0.95 1"),
    # ----- drinks and food
    "bottle": Obj("bottle", 0.54, [cyl(0.0325, 0.150, (0, 0, 0.075)), cyl(0.014, 0.055, (0, 0, 0.1775))], "pet", _round(0.0325, 0.09), (0.065, 0.065, 0.205), 0.0, "0.55 0.80 0.95 0.9"),
    "can": Obj("can", 0.36, [cyl(0.033, 0.115, (0, 0, 0.0575))], "steel", _round(0.033, 0.06), (0.066, 0.066, 0.115), 0.0, "0.80 0.20 0.25 1"),
    "meal_box": Obj("meal_box", 0.55, [box(0.190, 0.135, 0.065, (0, 0, 0.0325))], "plastic",
                    [Grasp((0, 0, 0.035), (1, 0, 0), 0.190, ((0, 1, 0), (0, -1, 0)), "long"), Grasp((0, 0, 0.035), (0, 1, 0), 0.135, ((1, 0, 0), (-1, 0, 0)), "short"),
                     Grasp((0.095, 0, 0.056), (0, 0, 1), 0.016, ((-1, 0, 0),), "rim end"), Grasp((-0.095, 0, 0.056), (0, 0, 1), 0.016, ((1, 0, 0),), "rim end"),
                     Grasp((0, 0.0675, 0.056), (0, 0, 1), 0.016, ((0, -1, 0),), "rim end"), Grasp((0, -0.0675, 0.056), (0, 0, 1), 0.016, ((0, 1, 0),), "rim end")], (0.19, 0.135, 0.065), 0.0, "0.85 0.90 0.80 0.95"),
    "mug": Obj("mug", 0.33, [cyl(0.041, 0.095, (0, 0, 0.0475)), box(0.012, 0.012, 0.070, (0.056, 0, 0.0475)), box(0.016, 0.012, 0.012, (0.046, 0, 0.078)),
                             box(0.016, 0.012, 0.012, (0.046, 0, 0.017))], "ceramic",
                _round(0.041, 0.055)[1:7] + [Grasp((0.056, 0, 0.0475), (0, 1, 0), 0.012, ((1, 0, 0), (0, 0, -1)), "handle")], (0.10, 0.082, 0.095), 0.0, "0.90 0.88 0.82 1",
                liquid=dict(level=0.065, rim=0.095, r=0.036)),
    "bowl": Obj("bowl", 0.32, [], "ceramic", [Grasp((0, 0, 0.035), (0, 1, 0), 0.170, ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)), "across"),
                                              Grasp((0.083, 0, 0.066), (1, 0, 0), 0.006, ((0, 0, -1),), "rim pinch")], (0.17, 0.17, 0.07), 0.0, "0.93 0.93 0.90 1",
                 liquid=dict(level=0.045, rim=0.07, r=0.080)),
    # ----- carried, not gripped
    "book": Obj("book", 0.35, [box(0.150, 0.025, 0.210, (0, 0, 0.105))], "paper", [Grasp((0, 0, 0.13), (0, 1, 0), 0.025, ((1, 0, 0), (-1, 0, 0)), "spine"),
                                                                                  Grasp((0, 0, 0.19), (0, 1, 0), 0.025, ((0, 0, -1),), "top edge")], (0.15, 0.025, 0.21), 0.0, "0.75 0.30 0.25 1"),
    "parcel": Obj("parcel", 3.0, [box(0.30, 0.22, 0.14, (0, 0, 0.07))], "paper", [Grasp((0, 0, 0.09), (1, 0, 0), 0.30, ((0, 1, 0), (0, -1, 0)), "long"),
                                                                                 Grasp((0, 0, 0.09), (0, 1, 0), 0.22, ((1, 0, 0), (-1, 0, 0)), "short")], (0.30, 0.22, 0.14), 0.0, "0.72 0.56 0.38 1"),
    "bin_bag": Obj("bin_bag", 2.0, [box(0.24, 0.18, 0.26, (0, 0, 0.13)), box(0.20, 0.006, 0.06, (0, 0, 0.29))], "plastic",
                   [Grasp((0, 0, 0.29), (0, 1, 0), 0.006, ((0, 0, -1), (1, 0, 0), (-1, 0, 0)), "knot")], (0.24, 0.18, 0.32), 0.0, "0.25 0.25 0.28 1"),
}

FLOOR_SET = ("phone", "remote", "pill_bottle", "glasses", "spoon", "toothbrush", "tablet_box", "tissue_pack")
FRAGILE = ("mug", "bowl", "glasses")


def contact_floor():
    """Thinnest thing the contact model can judge: twice the penetration under a typical 5 N contact (see tests)."""
    return 0.002


def mjcf(o: Obj, name=None, pos=(0, 0, -3)):
    """Body with a free joint, placed out of sight; scenarios put it where it belongs."""
    nm = name or o.name
    lines = [f'<body name="{nm}" pos="{pos[0]} {pos[1]} {pos[2]}">', f'  <freejoint name="{nm}"/>']
    geoms = o.geoms
    if o.name == "bowl":
        geoms = _bowl()
    for i, g in enumerate(geoms):
        t = g[0]
        s = " ".join(f"{x:.5f}" for x in g[1])
        pp = ""
        if len(g) > 3 and g[3] is not None:
            pp = ' fromto="' + " ".join(f"{x:.5f}" for x in g[3]) + '"'
        elif g[2] is not None:
            pp = ' pos="' + " ".join(f"{x:.5f}" for x in g[2]) + '"'
        if len(g) > 4:
            pp += ' euler="' + " ".join(f"{x:.5f}" for x in g[4]) + '"'
        lines.append(f'  <geom name="obj_{nm}_{i}" class="obj" type="{t}" size="{s}"{pp} rgba="{o.rgba}" mass="{o.mass / len(geoms):.5f}"/>')
    lines.append(f'  <site name="site_{nm}" size="0.004" rgba="1 0 0 0"/>')
    lines.append("</body>")
    return "\n".join(lines)


def _bowl():
    """Hollow bowl, 17 cm across, 7 cm high: floor plus 16 wall segments. A real bowl has no flat to pinch."""
    out = [("cylinder", (0.055, 0.003), (0, 0, 0.003))]
    n, rw, t, h = 16, 0.082, 0.006, 0.07
    for k in range(n):
        a = 2 * np.pi * k / n
        c = (rw * np.cos(a), rw * np.sin(a), h / 2 + 0.003)
        out.append(("box", (t / 2, np.pi * rw / n + 0.002, h / 2 - 0.003), c, None, (0, 0, a)))
    for k in range(n):                      # the shallow cone between floor and wall
        a = 2 * np.pi * k / n
        c = (0.066 * np.cos(a), 0.066 * np.sin(a), 0.010)
        out.append(("box", (0.016, np.pi * 0.066 / n + 0.002, 0.003), c, None, (0, -0.35, a)))
    return out


def surface_points(o: Obj, n=8):
    """Points on the object, object frame, for the visibility rays of the camera model."""
    sx, sy, sz = o.size
    c = np.array([0, 0, sz / 2])
    pts = [c]
    for dx in (-0.4, 0.4):
        for dy in (-0.4, 0.4):
            for dz in (-0.4, 0.4):
                pts.append(c + np.array([dx * sx, dy * sy, dz * sz]))
    return np.array(pts[:n + 1])
