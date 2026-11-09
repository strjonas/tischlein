"""The plug-in format: a robot is data. A base, a chain of powered axes, a gripper, cameras, a BOM.

A new design idea is a new Robot(...) in zoo.py, not a new simulation. Frames: robot origin on the floor under the axle
midpoint, X forward, Y left, Z up. Tool frame at the end of the chain: +X along the fingers, jaws close along Y, the site
"ee" sits between the pads.

Everything cheap hardware gets wrong is a field here, because that is where the money is: a design that tolerates
10 mm of sag or 1 deg of backlash can use cheaper parts, and the arena can show whether it does:

    Axis.force / peak / speed    continuous and peak force, no-load speed: the torque-speed line
    Axis.friction                Coulomb friction at the output (a worm or lead screw also self-locks: selflock)
    Axis.backlash                total lost motion, rad or m
    Axis.flex                    bending stiffness of the structure at this axis's root, N m/rad
    Gripper.pad                  pad material, sets friction against every object (see assemble.PAD_MU)
    Cam                          mount, field of view, range, kind; visibility is ray-cast, nothing is assumed seen
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geometry

COL = dict(rob="0.24 0.47 0.85 1", robd="0.15 0.17 0.20 1", arm="0.90 0.55 0.15 1", jaw="0.85 0.25 0.20 1", pad="0.12 0.12 0.12 1")


@dataclass
class Axis:
    name: str
    kind: str                       # slide | hinge
    axis: tuple
    lo: float
    hi: float
    force: float                    # continuous rating at the output, N or N m
    speed: float                    # no-load speed, m/s or rad/s
    pos: tuple = (0, 0, 0)          # joint frame in the parent frame
    euler: tuple = None             # fixed rotation of this body relative to its parent, deg
    peak: float = None              # peak force, default 1.5 x continuous
    mass: float = 0.3
    com: tuple = (0, 0, 0)
    geoms: list = field(default_factory=list)
    stages: int = 1                 # coupled telescope stages (the last is the tip)
    friction: float = None          # Coulomb friction at the output
    armature: float = None          # reflected rotor inertia
    selflock: bool = False          # holds its load with power off
    backlash: float = 0.0
    flex: float = 0.0               # root bending stiffness, N m/rad; 0 = rigid
    flex_at: tuple = (0, 0, 0)      # where the bend happens, in this body's frame
    home: float = None              # stowed value
    telescope_shells: bool = True   # False for a moving solid tray/fork rather than nested tubes

    def __post_init__(self):
        if self.peak is None:
            self.peak = 1.5 * self.force
        if self.friction is None:
            self.friction = 0.02 * self.force
        if self.armature is None:
            self.armature = 0.3 if self.kind == "slide" else 0.01
        if self.home is None:
            self.home = min(max(0.0, self.lo), self.hi)


@dataclass
class Gripper:
    stroke: float = 0.10            # full opening between the pads
    jaw_len: float = 0.06
    jaw_h: float = 0.035            # pad height, across the closing direction
    jaw_t: float = 0.008
    palm: tuple = (0.07, 0.10, 0.05)      # length, width, height of the hand body behind the jaws
    force: float = 40.0             # squeeze
    speed: float = 0.04             # jaw opening speed
    pad: str = "tpu"
    selflock: bool = False          # worm: keeps its grip with power off
    mass: float = 0.25


@dataclass
class Cam:
    name: str
    parent: str                     # body: "robot", "ee", or an axis name
    pos: tuple
    look: tuple                     # direction in the parent frame
    fov: tuple = (69.0, 42.0)       # horizontal, vertical, deg
    near: float = 0.10
    far: float = 3.0
    kind: str = "rgbd"              # rgbd | rgb | ir


@dataclass
class Base:
    x: tuple = (-0.25, 0.30)
    y: tuple = (-0.22, 0.22)
    z: tuple = (0.035, 0.27)
    mass: float = 20.0
    com: tuple = (0.03, 0.0, 0.14)
    parts: list = field(default_factory=list)       # extra fixed masses: (kg, centre, size)
    wheel_r: float = 0.0825
    wheel_w: float = 0.05
    wheel_y: float = 0.185
    wheel_x: float = 0.0
    wheel_m: float = 0.8
    torque: float = 10.0            # continuous, per wheel
    vmax: float = 0.40              # top speed in the home, m/s
    wmax: float = math.radians(60)
    casters: list = field(default_factory=lambda: [(-0.21, 0.15), (-0.21, -0.15), (0.26, 0.15), (0.26, -0.15)])
    caster_r: float = 0.0375
    caster_w: float = 0.02          # width of the caster wheel
    travel: float = 0.012           # spring travel of each caster leg, +- m. See reach.Tier1.climbs
    trail: float = 0.015
    drive_share: float = 0.6
    extra: list = field(default_factory=list)       # fixed collision geoms, ("box", lo, hi)
    lidar: tuple = (0.20, 0.0, 0.28)
    lidar_kind: str = "c1"          # c1: 2D, +-3 cm, 0.72 deg


@dataclass
class Deck:
    x: tuple
    y: tuple
    z: float
    fence: float = 0.04
    parent: str = "robot"           # or an axis name: a tray that rides on the lift


@dataclass
class Robot:
    name: str
    label: str
    base: Base
    chain: list
    grip: Gripper
    cams: list
    deck: Deck = None
    kits: tuple = ()
    extra_mass: float = 0.0
    motors: int = 0
    hand_pos: tuple = (0, 0, 0)     # hand origin in the last axis's frame
    note: str = ""
    stow_valid: bool = True

    @property
    def axes(self):
        return {a.name: a for a in self.chain}

    def footprint(self):
        """Rectangle (x0, x1, y0, y1) of everything fixed on the base, and of the stowed arm within reason."""
        x0, x1 = self.base.x
        y0, y1 = self.base.y
        y0, y1 = min(y0, -self.base.wheel_y - self.base.wheel_w / 2), max(y1, self.base.wheel_y + self.base.wheel_w / 2)
        for g in self.base.extra:
            if g[0] == "box":
                x0, x1 = min(x0, g[1][0]), max(x1, g[2][0])
                y0, y1 = min(y0, g[1][1]), max(y1, g[2][1])
        if self.deck and self.deck.parent == "robot":
            x0, x1 = min(x0, self.deck.x[0]), max(x1, self.deck.x[1])
            y0, y1 = min(y0, self.deck.y[0]), max(y1, self.deck.y[1])
        return x0, x1, y0, y1


def _f(v):
    return " ".join(f"{x:.5f}" for x in v)


def _box_I(m, s):
    return (m * (s[1] ** 2 + s[2] ** 2) / 12, m * (s[0] ** 2 + s[2] ** 2) / 12, m * (s[0] ** 2 + s[1] ** 2) / 12)


def _combine(parts):
    M = sum(p[0] for p in parts)
    c = sum(p[0] * np.asarray(p[1], float) for p in parts) / M
    I = np.zeros((3, 3))
    for m, p, s in parts:
        r = np.asarray(p, float) - c
        I += np.diag(_box_I(m, s)) + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return M, c, I


def _inertial(M, c, I, ind):
    fi = f"{I[0, 0]:.6f} {I[1, 1]:.6f} {I[2, 2]:.6f} {I[0, 1]:.6f} {I[0, 2]:.6f} {I[1, 2]:.6f}"
    return f'{ind}<inertial pos="{_f(c)}" mass="{M:.4f}" fullinertia="{fi}"/>'


def _geoms(gs, prefix, cls, rgba, ind):
    out = []
    for i, g in enumerate(gs):
        if g[0] == "box":
            lo, hi = np.array(g[1], float), np.array(g[2], float)
            c = (lo + hi) / 2
            s = abs(hi - lo) / 2
            r = g[3] if len(g) > 3 else rgba
            out.append(f'{ind}<geom name="{prefix}_{i}" class="{cls}" type="box" pos="{_f(c)}" size="{_f(s)}" rgba="{r}"/>')
        elif g[0] == "cyl":
            r = g[4] if len(g) > 4 else rgba
            out.append(f'{ind}<geom name="{prefix}_{i}" class="{cls}" type="cylinder" fromto="{_f(g[1])} {_f(g[2])}" size="{g[3]:.4f}" rgba="{r}"/>')
    return out


def robot_mjcf(r: Robot):
    b = r.base
    L = []
    L.append('<body name="robot" pos="0 0 0">')
    L.append('  <freejoint name="base"/>')
    (x0, x1), (y0, y1), (z0, z1) = b.x, b.y, b.z
    packs = geometry.packages(b)
    chassis = geometry.chassis(b)
    parts = geometry.inertia_parts(b.mass-2*b.wheel_m+r.extra_mass-sum(p[1] for p in packs), chassis)
    parts += list(b.parts) + [(m, c, s) for _, m, c, s in packs]
    L.append(_inertial(*_combine(parts), "  "))
    L += _geoms(chassis, "rob_base", "rob", COL["rob"], "  ")
    L += _geoms(b.extra, "rob_x", "rob", COL["robd"], "  ")
    for name, _, center, size in packs:
        c, s = np.array(center), np.array(size)/2
        L += _geoms([("box", tuple(c-s), tuple(c+s))], f"rob_package_{name}", "rob", COL["robd"], "  ")
    if r.deck and r.deck.parent == "robot":
        L += _deck(r.deck, "  ")
    for nm, (px, py, pz) in dict(lidar=b.lidar).items():
        L.append(f'  <site name="lidar" pos="{px} {py} {pz}" size="0.01" rgba="0 0 0 0"/>')
        lidar_case = [("cyl", (px, py, pz-0.02), (px, py, pz+0.02), 0.025)]
        if pz-.02 > b.z[1]+.003:
            lidar_case.append(("cyl", (px, py, b.z[1]), (px, py, pz-0.02), .009))
        L += _geoms(lidar_case, "rob_lidar_case", "rob", COL["robd"], "  ")
    for c in r.cams:
        if c.parent == "robot":
            L.append(f'  <site name="cam_{c.name}" pos="{_f(c.pos)}" size="0.008" rgba="0 0 0 0"/>')
            L += _geoms(geometry.camera_geoms(c), f"rob_camera_{c.name}", "rob", COL["robd"], "  ")
    # drive wheels: 5 cm compliant tyres as ellipsoids (a cylinder's single contact jumps between its rims)
    for side, s in (("l", 1), ("r", -1)):
        L.append(f'  <body name="wheel_{side}" pos="{b.wheel_x} {s * b.wheel_y} {b.wheel_r}">')
        L.append(f'    <joint name="wheel_{side}" type="hinge" axis="0 1 0" damping="0.05" armature="0.02"/>')
        L.append(f'    <inertial pos="0 0 0" mass="{b.wheel_m}" diaginertia="0.003 0.005 0.003"/>')
        L.append(f'    <geom name="rob_tyre_{side}" class="tyre" type="ellipsoid" size="{b.wheel_r} {b.wheel_w / 2} {b.wheel_r}" rgba="0.15 0.17 0.2 1"/>')
        L.append('  </body>')
    # casters: a spring leg (3 mm, then a hard stop, so the drive wheels carry drive_share of the weight), a swivel
    # with trail and a rolling wheel. Frictionless balls hide what makes real bases drift: caster flop on reversing,
    # and the climb over a threshold, which small casters lose
    for i, (cx, cy) in enumerate(b.casters):
        L.append(f'  <body name="caster{i}" pos="{cx} {cy} {b.caster_r}">')
        L.append(f'    <joint name="caster{i}_z" type="slide" axis="0 0 1" range="{-b.travel} {b.travel}"/>')
        L.append('    <inertial pos="0 0 0" mass="0.06" diaginertia="4e-5 4e-5 4e-5"/>')
        L.append(f'    <body name="caster{i}_s" pos="0 0 0">')
        L.append(f'      <joint name="caster{i}_s" type="hinge" axis="0 0 1" damping="0.002" armature="1e-4"/>')
        L.append('      <inertial pos="0 0 0" mass="0.03" diaginertia="2e-5 2e-5 2e-5"/>')
        L.append(f'      <body name="caster{i}_w" pos="{-b.trail} 0 0">')
        L.append(f'        <joint name="caster{i}_w" type="hinge" axis="0 1 0" damping="0.0005" armature="1e-5"/>')
        L.append(f'        <inertial pos="0 0 0" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>')
        L.append(f'        <geom name="rob_cast{i}" class="caster" type="ellipsoid" size="{b.caster_r} {b.caster_w / 2} {b.caster_r}" rgba="0.1 0.1 0.1 1"/>')
        L.append('      </body>\n    </body>\n  </body>')
    # chain
    ind = "  "
    open_bodies = 0
    for ai, a in enumerate(r.chain):
        d = np.array(a.axis, float)
        eu = f' euler="{_f(np.radians(a.euler))}"' if a.euler else ""
        telescopic = a.name == "reach" and a.kind == "slide" and a.telescope_shells
        if telescopic:
            L.append(f'{ind}<body name="housing_{a.name}" pos="{_f(a.pos)}"{eu}>')
            if a.flex:
                for n, ax in (("p", (0, 1, 0)), ("y", (0, 0, 1))):
                    L.append(f'{ind}  <joint name="flex_{a.name}_{n}" type="hinge" axis="{_f(ax)}" pos="{_f(a.flex_at)}" '
                             f'stiffness="{a.flex:.1f}" damping="{a.flex * 0.01:.2f}" armature="0.01" range="-0.05 0.05"/>')
            fixed = geometry.telescope_shell(a, -1)
            L.append(_inertial(*_combine(geometry.inertia_parts(a.mass*0.25, fixed)), ind+"  "))
            L += _geoms(fixed, f"arm_housing_{a.name}", "arm", COL["arm"], ind+"  ")
            ind += "  "; open_bodies += 1
        for k in range(a.stages):
            nm = a.name if k == 0 else f"{a.name}{k + 1}"
            pos = a.pos if k == 0 and not telescopic else (0, 0, 0)
            e = eu if k == 0 and not telescopic else ""
            L.append(f'{ind}<body name="{nm}" pos="{_f(pos)}"{e}>')
            if k == 0 and a.flex and not telescopic:
                for n, ax in (("p", (0, 1, 0)), ("y", (0, 0, 1))):
                    L.append(f'{ind}  <joint name="flex_{a.name}_{n}" type="hinge" axis="{_f(ax)}" pos="{_f(a.flex_at)}" '
                             f'stiffness="{a.flex:.1f}" damping="{a.flex * 0.01:.2f}" armature="0.01" range="-0.05 0.05"/>')
            lo = a.lo / a.stages
            hi = a.hi / a.stages
            L.append(f'{ind}  <joint name="{nm}" type="{a.kind}" axis="{_f(a.axis)}" range="{lo:.5f} {hi:.5f}" '
                     f'damping="{5.0 if a.kind == "slide" else 0.2}" frictionloss="{a.friction / (a.stages if k == 0 else 1e3):.4f}" armature="{a.armature:.4f}"/>')
            if a.backlash and k == 0:
                L.append(f'{ind}  <joint name="bl_{a.name}" type="{a.kind}" axis="{_f(a.axis)}" range="{-a.backlash / 2:.6f} {a.backlash / 2:.6f}" '
                         f'damping="{0.5 if a.kind == "slide" else 0.02}" armature="{a.armature * 0.2:.4f}"/>')
            m = a.mass * (0.75 if telescopic else 1.0) / a.stages
            gs = geometry.telescope_shell(a, k) if telescopic else geometry.axis_geoms(a, r.chain[ai+1] if ai+1 < len(r.chain) else None)
            L.append(_inertial(*_combine(geometry.inertia_parts(m, gs)), ind+"  "))
            L += _geoms(gs, f"arm_{nm}", "arm", COL["arm"], ind + "  ")
            for c in r.cams:
                if c.parent == a.name and k == a.stages - 1:
                    L.append(f'{ind}  <site name="cam_{c.name}" pos="{_f(c.pos)}" size="0.008" rgba="0 0 0 0"/>')
                    L += _geoms(geometry.camera_geoms(c), f"arm_camera_{c.name}", "arm", COL["robd"], ind+"  ")
                    if c.name == "head" and a.name == "turn":
                        L += _geoms([("cyl", (0.05, c.pos[1], 0.02), (0.05, c.pos[1], c.pos[2]), 0.006),
                                     ("cyl", (0.05, c.pos[1], c.pos[2]), (c.pos[0]-0.03, c.pos[1], c.pos[2]), 0.006)],
                                    "arm_camera_support", "arm", COL["robd"], ind+"  ")
            if r.deck and r.deck.parent == a.name and k == a.stages - 1:
                L += _deck(r.deck, ind + "  ")
            ind += "  "
            open_bodies += 1
    # gripper at the end of the chain
    g = r.grip
    pl, pw, ph = g.palm
    L.append(f'{ind}<body name="hand" pos="{_f(r.hand_pos)}">')
    L.append(f'{ind}  <inertial pos="{pl / 2} 0 0" mass="{g.mass}" diaginertia="{_f(_box_I(g.mass, g.palm))}"/>')
    L.append(f'{ind}  <geom name="arm_palm" class="arm" type="box" pos="{pl / 2} 0 0" size="{pl / 2} {pw / 2} {ph / 2}" rgba="{COL["jaw"]}"/>')
    jm = g.jaw_len / 2 + pl
    for nm, s in (("a", 1), ("b", -1)):
        L.append(f'{ind}  <body name="jaw_{nm}" pos="0 0 0">')
        L.append(f'{ind}    <joint name="jaw_{nm}" type="slide" axis="0 {s} 0" range="0 {g.stroke / 2:.4f}" damping="5" armature="0.05" solreflimit="0.004 1"/>')
        L.append(f'{ind}    <inertial pos="{jm} {s * 0.01} 0" mass="0.04" diaginertia="2e-5 2e-5 2e-5"/>')
        # the jaw body sits outside the pad; the pad is what touches things
        L.append(f'{ind}    <geom name="arm_jaw_{nm}" class="arm" type="box" pos="{jm} {s * (g.jaw_t / 2 + 0.003)} 0" size="{g.jaw_len / 2} {g.jaw_t / 2} {g.jaw_h / 2}" rgba="{COL["jaw"]}"/>')
        L.append(f'{ind}    <geom name="pad_{nm}" class="pad" type="box" pos="{jm} {s * 0.0015} 0" size="{g.jaw_len / 2} 0.0015 {g.jaw_h / 2}" rgba="{COL["pad"]}"/>')
        L.append(f'{ind}  </body>')
    L.append(f'{ind}  <site name="ee" pos="{jm} 0 0" size="0.006" rgba="1 0 0 0.6"/>')
    for c in r.cams:
        if c.parent == "ee":
            L.append(f'{ind}  <site name="cam_{c.name}" pos="{_f(c.pos)}" size="0.008" rgba="0 0 0 0"/>')
            L += _geoms(geometry.camera_geoms(c), f"arm_camera_{c.name}", "arm", COL["robd"], ind+"  ")
    L.append(f'{ind}</body>')
    for _ in range(open_bodies):
        ind = ind[:-2]
        L.append(f'{ind}</body>')
    L.append('</body>')
    return "\n".join(L)


def _deck(d: Deck, ind):
    x0, x1 = d.x
    y0, y1 = d.y
    z = d.z
    out = [f'{ind}<geom name="rob_deck" class="rob" type="box" pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {z - 0.005}" size="{(x1 - x0) / 2} {(y1 - y0) / 2} 0.005" rgba="0.30 0.32 0.36 1"/>']
    if d.fence:
        t = 0.004
        for n, lo, hi in (("f", (x1 - t, y0, z), (x1, y1, z + d.fence)), ("r", (x0, y0, z), (x0 + t, y1, z + d.fence)),
                          ("s", (x0, y0, z), (x1, y0 + t, z + d.fence)), ("n", (x0, y1 - t, z), (x1, y1, z + d.fence))):
            c = [(a + e) / 2 for a, e in zip(lo, hi)]
            s = [(e - a) / 2 for a, e in zip(lo, hi)]
            out.append(f'{ind}<geom name="rob_deck_{n}" class="rob" type="box" pos="{_f(c)}" size="{_f(s)}" rgba="0.15 0.17 0.20 1"/>')
    out.append(f'{ind}<site name="deck" pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {z}" size="0.005" rgba="0 0 0 0"/>')
    return out
