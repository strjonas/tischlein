"""The fixed world: an ordinary rented flat with a person in bed. Nothing in it is shaped for a robot.

The floor plan is fixed and versioned (see lock.py), so later edits to a robot entry cannot move the goalposts.

- The kitchen is stock: a tall fridge with a handle, a microwave on the worktop, a dishwasher under it. Aids that a robot entry
  brings along (fridge opener, microwave button pusher, threshold ramps) are kits listed on the robot (KITS below).
- The world comes as a set of homes (HOMES), not one flat. They span: doors 0.70-0.90 m,
  mattress 0.45-0.72 m, thresholds up to 20 mm, tile, laminate, carpet, fridges hinged by a wall or free, and seals of
  25-55 N. Homes E-H are the held-out test set: nothing may be tuned on them.
- Thresholds are real steps, a rug has a real edge.
- A person lies in the bed. Contact with them is judged.

Coordinates: x east, y north, z up, floor z = 0, metres. Doors open with negative hinge angles as in scene.py.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

COL = dict(
    wall="0.93 0.92 0.89 1", floor="0.80 0.76 0.70 1", tile="0.78 0.80 0.82 1", carpet="0.55 0.58 0.62 1",
    wood="0.72 0.58 0.42 1", furn="0.86 0.84 0.80 1", bed="0.80 0.84 0.90 1", steel="0.62 0.64 0.67 1",
    white="0.96 0.96 0.95 1", dark="0.20 0.21 0.23 1", glass="0.55 0.70 0.80 0.45", skin="0.85 0.68 0.56 1",
    rug="0.45 0.35 0.40 1", act="0.95 0.35 0.65 1", fixture="0.55 0.40 0.75 1", kit="0.30 0.75 0.60 1",
)
FLOOR_MU = dict(laminate=0.5, tile=0.45, carpet=0.8)     # rubber tread on floor; estimates, swept in DR
WALL_H, T_WALL = 2.5, 0.10


@dataclass(frozen=True)
class Home:
    name: str = "A"
    door_w: float = 0.80           # clear width of the two interior doors on the route
    bed_h: float = 0.58            # mattress top
    threshold: float = 0.015       # step at the bedroom and kitchen doors
    floor: str = "laminate"
    rug: bool = True               # 8 mm rug beside the bed
    fridge: str = "tall"           # tall: door from 4 cm up; combi: fridge door from 0.75 m up
    fridge_x: float = 4.30         # west edge of the fridge: 4.30 stands against the wall
    hinge: str = "west"            # west/east: which side of the fridge the door hangs on
    seal_N: float = 35.0           # pull at the handle to break the seal
    closes_below: float = 25.0     # deg; below this angle the door falls shut, above it stays
    aisle: float = 1.40            # free floor in front of the fridge
    counter_h: float = 0.90
    dark: bool = False

    def key(self):
        return asdict(self)


# dev homes A-D may be used while designing; test homes E-H are for scoring only
HOMES = {
    "A": Home("A"),
    "B": Home("B", door_w=0.70, bed_h=0.45, threshold=0.020, floor="tile", rug=False, fridge="combi", seal_N=45.0, aisle=1.10),
    "C": Home("C", door_w=0.90, bed_h=0.70, threshold=0.0, floor="laminate", fridge_x=5.20, hinge="east", seal_N=25.0, closes_below=35.0),
    "D": Home("D", door_w=0.76, bed_h=0.50, threshold=0.012, floor="carpet", rug=False, fridge_x=5.20, seal_N=55.0, aisle=0.95, counter_h=0.86),
    "E": Home("E", door_w=0.73, bed_h=0.55, threshold=0.018, floor="laminate", fridge="combi", fridge_x=5.20, hinge="east", seal_N=40.0, aisle=1.20),
    "F": Home("F", door_w=0.85, bed_h=0.62, threshold=0.010, floor="tile", rug=False, seal_N=30.0, closes_below=45.0, aisle=1.00),
    "G": Home("G", door_w=0.80, bed_h=0.72, threshold=0.005, floor="carpet", fridge="combi", seal_N=50.0, counter_h=0.92),
    "H": Home("H", door_w=0.70, bed_h=0.48, threshold=0.015, floor="laminate", rug=False, fridge_x=5.20, seal_N=35.0, aisle=0.90, dark=True),
}
DEV, TEST = "ABCD", "EFGH"

# Things a robot entry may install in a home.
#   name: what it does
KITS = {
    "fridge_opener": "linear actuator on the fridge door: opens and closes on command, holds it open",
    "strap": "strap from the fridge handle down to 0.40 m: pull low instead of at the handle",
    "hold_open": "stay that holds the fridge door past 60 deg",
    "mw_pusher": "button pusher on the microwave's release and start buttons, on command",
    "ramps": "rubber ramps over the thresholds",
    "tags": "printed fiducial tags on appliances, fridge and bed",
    "shelf_riser": "raises the robot-zone shelf items to the front edge (fixed bins)",
}


class B:
    """Tiny MJCF writer, as in scene.py: axis-aligned boxes from corners, fromto cylinders."""

    def __init__(self):
        self.lines: list[str] = []
        self.names: set[str] = set()
        self.ports: dict[str, tuple] = {}
        self.regions: dict[str, tuple] = {}     # name: ((x0, y0, z0), (x1, y1, z1)), goals are stated on these
        self.doors: dict[str, dict] = {}

    def _name(self, n):
        assert n not in self.names, n
        self.names.add(n)
        return n

    def box(self, name, lo, hi, cls="world", rgba=None, mat=None, ind=""):
        c = [(a + b) / 2 for a, b in zip(lo, hi)]
        s = [abs(b - a) / 2 for a, b in zip(lo, hi)]
        assert min(s) > 0, (name, lo, hi)
        rg = f' rgba="{rgba}"' if rgba else ""
        self.lines.append(f'{ind}<geom name="{self._name(name)}" class="{cls}" type="box" pos="{_f(c)}" size="{_f(s)}"{rg}/>')

    def cyl(self, name, a, b, r, cls="world", rgba=None, ind="", kind="cylinder"):
        rg = f' rgba="{rgba}"' if rgba else ""
        self.lines.append(f'{ind}<geom name="{self._name(name)}" class="{cls}" type="{kind}" fromto="{_f(a)} {_f(b)}" size="{r:.4f}"{rg}/>')

    def sph(self, name, c, r, cls="world", rgba=None, ind=""):
        rg = f' rgba="{rgba}"' if rgba else ""
        self.lines.append(f'{ind}<geom name="{self._name(name)}" class="{cls}" type="sphere" pos="{_f(c)}" size="{r:.4f}"{rg}/>')

    def raw(self, s):
        self.lines.append(s)


def _f(v):
    return " ".join(f"{x:.4f}" for x in v)


# ---------------------------------------------------------------- the flat ---
def build_shell(b: B, h: Home, kits=()):
    W = COL["wall"]
    fl = dict(laminate=COL["floor"], tile=COL["tile"], carpet=COL["carpet"])[h.floor]
    # a plane, not a slab: something pressed into a thin floor tunnels through it and the trial explodes
    b.lines.append(f'<geom name="floor" class="world" type="plane" pos="5 4.35 0" size="5.3 4.3 0.1" rgba="{fl}"/>')

    def wall(n, lo, hi):
        b.box("wall_" + n, lo, hi, rgba=W)

    wall("w", (-0.12, 0.28, 0), (0.0, 8.42, WALL_H))
    wall("n", (-0.12, 8.30, 0), (7.72, 8.42, WALL_H))
    wall("s", (-0.12, 0.28, 0), (7.72, 0.40, WALL_H))
    wall("e_s", (7.60, 0.28, 0), (7.72, 3.15, WALL_H))
    wall("e_n", (7.60, 4.03, 0), (7.72, 8.42, WALL_H))
    # doors on the route: centred at x 3.60 (bedroom) and 4.90 (kitchen), clear width door_w
    d = h.door_w / 2
    du, dk = 3.60, 4.90
    wall("hu_1", (0.0, 4.20, 0), (du - d, 4.30, WALL_H))
    wall("hu_2", (du + d, 4.20, 0), (dk - d, 4.30, WALL_H))
    wall("hu_3", (dk + d, 4.20, 0), (7.60, 4.30, WALL_H))
    wall("uk", (4.20, 4.30, 0), (4.30, 8.30, WALL_H))
    wall("hb_1", (0.0, 2.90, 0), (0.55, 3.00, WALL_H))
    wall("hb_2", (1.25, 2.90, 0), (7.60, 3.00, WALL_H))
    wall("bo", (2.60, 0.40, 0), (2.70, 2.90, WALL_H))
    for n, lo, hi in [("lu", (du - d, 4.20, 2.05), (du + d, 4.30, WALL_H)), ("lk", (dk - d, 4.20, 2.05), (dk + d, 4.30, WALL_H)),
                      ("lb", (0.55, 2.90, 2.05), (1.25, 3.00, WALL_H)), ("lf", (7.60, 3.15, 2.05), (7.72, 4.03, WALL_H))]:
        wall(n, lo, hi)
    # door leaves parked open against the walls: 4 cm of leaf plus a 2 cm stop narrow the opening
    b.box("door_u_leaf", (du + d, 4.30, 0.01), (du + d + 0.04, 4.30 + h.door_w, 2.0), rgba=COL["furn"])
    b.box("door_k_leaf", (dk - d - 0.04, 4.30, 0.01), (dk - d, 4.30 + h.door_w, 2.0), rgba=COL["furn"])
    b.box("door_b_slide", (1.25, 2.86, 0.01), (2.00, 2.90, 2.0), rgba=COL["furn"])
    # thresholds: steps with a 4 mm chamfer are the common German case; modelled as sharp steps of the given height
    if h.threshold > 0:
        for n, c in (("u", du), ("k", dk)):
            b.box(f"thr_{n}", (c - d, 4.20, 0.0), (c + d, 4.30, h.threshold), rgba=COL["wood"])
            if "ramps" in kits:                          # rubber wedges, 1:10, both sides
                L = 10 * h.threshold
                th = math.atan2(h.threshold, L)
                t = 0.005
                for s_, y in ((1, 4.30 + L / 2), (-1, 4.20 - L / 2)):
                    cz = h.threshold / 2 - t * math.cos(th)
                    b.lines.append(f'<geom name="ramp_{n}{s_ > 0}" class="world" type="box" pos="{c:.4f} {y:.4f} {cz:.4f}" size="{d:.4f} {L / 2 / math.cos(th):.4f} {t}" euler="{-s_ * th:.5f} 0 0" rgba="{COL["dark"]}"/>')
    b.ports["door_u"] = (du, 4.25, 0.0)
    b.ports["door_k"] = (dk, 4.25, 0.0)
    # hall furniture: shoe cabinet narrows the hall to 0.95 m in one place
    b.box("shoe_cab", (2.0, 3.0, 0.0), (2.8, 3.25, 0.90), rgba=COL["furn"])


def build_bedroom(b: B, h: Home):
    z = h.bed_h
    fr = z - 0.16
    b.box("bed_frame", (0.05, 6.25, 0.14), (1.45, 8.25, fr), rgba=COL["wood"])    # 14 cm of air under the frame
    for i, (x, y) in enumerate([(0.08, 6.28), (1.37, 6.28), (0.08, 8.15), (1.37, 8.15)]):
        b.box(f"bed_leg{i}", (x, y, 0.0), (x + 0.05, y + 0.05, 0.14), rgba=COL["wood"])
    b.box("bed_mattress", (0.08, 6.28, fr), (1.42, 8.22, z), rgba=COL["bed"])
    b.box("bed_head", (0.05, 8.20, 0.0), (1.45, 8.28, z + 0.42), rgba=COL["wood"])
    b.box("bed_pillow", (0.25, 7.75, z), (1.25, 8.15, z + 0.10), rgba=COL["white"])
    # over-bed table on a column at the bed's east side
    tz = z + 0.22
    b.box("obt_top", (0.55, 6.95, tz - 0.02), (1.55, 7.40, tz), rgba=COL["wood"])
    b.box("obt_col", (1.47, 7.10, 0.03), (1.55, 7.25, tz - 0.02), rgba=COL["steel"])
    b.box("obt_foot", (1.40, 6.95, 0.0), (1.58, 7.40, 0.03), rgba=COL["steel"])
    b.regions["table"] = ((0.60, 6.98, tz), (1.48, 7.37, tz + 0.40))
    b.regions["reach_person"] = ((0.55, 6.95, tz), (1.25, 7.40, tz + 0.40))
    b.box("bedside", (1.55, 7.85, 0.0), (1.98, 8.28, 0.60), rgba=COL["wood"])
    b.regions["bedside"] = ((1.57, 7.87, 0.60), (1.96, 8.26, 0.95))
    b.box("desk", (2.40, 7.65, 0.72), (3.90, 8.28, 0.75), rgba=COL["wood"])
    b.box("desk_leg1", (2.42, 7.67, 0), (2.47, 8.26, 0.72), rgba=COL["wood"])
    b.box("desk_leg2", (3.83, 7.67, 0), (3.88, 8.26, 0.72), rgba=COL["wood"])
    b.box("chair", (2.90, 7.10, 0.0), (3.35, 7.55, 0.46), rgba=COL["dark"])
    b.box("wardrobe", (0.05, 4.32, 0.0), (1.85, 4.85, 2.10), rgba=COL["furn"])
    # bookshelf on the east wall, boards at 0.40 / 0.80 / 1.20 / 1.60
    sx0, sx1, sy0, sy1 = 3.85, 4.20, 5.80, 7.00
    for i, zz in enumerate((0.40, 0.80, 1.20, 1.60)):
        b.box(f"fshelf_b{i}", (sx0, sy0, zz - 0.02), (sx1, sy1, zz), rgba=COL["wood"])
    b.box("fshelf_side_s", (sx0, sy0 - 0.02, 0.0), (sx1, sy0, 1.80), rgba=COL["wood"])
    b.box("fshelf_side_n", (sx0, sy1, 0.0), (sx1, sy1 + 0.02, 1.80), rgba=COL["wood"])
    b.box("fshelf_back", (4.18, sy0, 0.0), (4.20, sy1, 1.80), rgba=COL["wood"])
    b.regions["shelf_080"] = ((3.87, 5.82, 0.80), (4.18, 6.98, 1.18))
    b.regions["shelf_120"] = ((3.87, 5.82, 1.20), (4.18, 6.98, 1.58))
    # charging dock on the south wall of the bedroom, robot backs on
    b.box("dock_plate", (2.36, 4.30, 0.0), (2.62, 4.37, 0.30), rgba=COL["dark"])
    b.ports["dock"] = (2.53, 4.70, math.pi / 2)
    if h.rug:
        b.box("rug", (1.60, 5.60, 0.0), (3.00, 7.40, 0.008), rgba=COL["rug"])
    # the person: on their back, head on the pillow. Static capsules; contact is judged, not rendered soft
    px, sk = 0.75, COL["skin"]
    zp = z + 0.10
    b.raw('<body name="person" pos="0 0 0">')
    b.sph("per_head", (px, 7.92, zp + 0.08), 0.10, cls="person", rgba=sk, ind="  ")
    b.cyl("per_torso", (px, 7.68, zp), (px, 7.18, zp), 0.17, cls="person", rgba="0.55 0.65 0.75 1", ind="  ", kind="capsule")
    b.cyl("per_legs_l", (px - 0.10, 7.05, zp - 0.04), (px - 0.10, 6.40, zp - 0.05), 0.08, cls="person", rgba="0.55 0.65 0.75 1", ind="  ", kind="capsule")
    b.cyl("per_legs_r", (px + 0.10, 7.05, zp - 0.04), (px + 0.10, 6.40, zp - 0.05), 0.08, cls="person", rgba="0.55 0.65 0.75 1", ind="  ", kind="capsule")
    b.cyl("per_arm_r", (px + 0.21, 7.70, zp), (px + 0.28, 7.32, zp - 0.02), 0.045, cls="person", rgba=sk, ind="  ", kind="capsule")
    b.cyl("per_arm_l", (px - 0.21, 7.70, zp), (px - 0.28, 7.32, zp - 0.02), 0.045, cls="person", rgba=sk, ind="  ", kind="capsule")
    b.raw('</body>')


def build_fridge(b: B, h: Home, kits):
    x0, x1 = h.fridge_x, h.fridge_x + 0.60
    yf, yb, ztop = 7.65, 8.30, 1.85
    fr = COL["white"]
    z0 = 0.40 if h.fridge == "tall" else 0.78                     # cavity floor
    b.box("fr_bottom", (x0, yf, 0.0), (x1, yb, z0), rgba=fr)
    b.box("fr_top", (x0, yf, ztop - 0.05), (x1, yb, ztop), rgba=fr)
    b.box("fr_back", (x0, yb - 0.06, z0), (x1, yb, ztop - 0.05), rgba=fr)
    b.box("fr_side_w", (x0, yf, z0), (x0 + 0.04, yb - 0.06, ztop - 0.05), rgba=fr)
    b.box("fr_side_e", (x1 - 0.04, yf, z0), (x1, yb - 0.06, ztop - 0.05), rgba=fr)
    # glass shelves with a 6 mm front lip; heights differ between the two fridge types
    shelves = (0.62, 0.84, 1.10, 1.36) if h.fridge == "tall" else (1.00, 1.22, 1.46)
    for i, z in enumerate(shelves):
        b.box(f"fr_shelf{i}", (x0 + 0.04, yf + 0.05, z - 0.008), (x1 - 0.04, yb - 0.06, z), rgba=COL["glass"])
        b.box(f"fr_lip{i}", (x0 + 0.04, yf + 0.05, z), (x1 - 0.04, yf + 0.058, z + 0.006), rgba=COL["glass"])
        b.regions[f"fridge_shelf{i}"] = ((x0 + 0.05, yf + 0.06, z), (x1 - 0.05, yb - 0.07, z + 0.2))
    # the shelf the benchmark uses: the lowest at or above 0.80 m ("robot zone")
    use = next(i for i, z in enumerate(shelves) if z >= 0.80)
    b.ports["fridge_item"] = ((x0 + x1) / 2, yf + 0.14, shelves[use])
    b.ports["fridge_item2"] = ((x0 + x1) / 2 + 0.12, yf + 0.14, shelves[use])
    b.regions["fridge"] = ((x0 + 0.04, yf, z0), (x1 - 0.04, yb, ztop))
    zd0, zd1 = (0.04, 1.83) if h.fridge == "tall" else (0.76, 1.83)
    ax = x0 if h.hinge == "west" else x1
    sgn = 1 if h.hinge == "west" else -1                          # direction along the door from hinge to free edge
    b.raw(f'<body name="fr_door" pos="{ax:.4f} {yf:.4f} 0">')
    rng = "-1.75 0" if h.hinge == "west" else "0 1.75"
    b.raw(f'  <joint name="fr_door" type="hinge" axis="0 0 1" range="{rng}"/>')
    a, c = (0.0, 0.60) if sgn > 0 else (-0.60, 0.0)
    b.box("fr_door_slab", (a, -0.06, zd0), (c, 0.0, zd1), rgba=fr, ind="  ")
    # door bins on the inside, 6 cm rails
    for i, z in enumerate((zd0 + 0.40, zd0 + 0.75)):
        b.box(f"fr_bin{i}_f", (a + 0.05, 0.075, z), (c - 0.05, 0.08, z + 0.06), rgba=COL["glass"], ind="  ")
        b.box(f"fr_bin{i}_b", (a + 0.05, 0.0, z - 0.008), (c - 0.05, 0.08, z), rgba=COL["glass"], ind="  ")
    # vertical bar handle at the free edge
    hx = (0.60 - 0.05) * sgn
    zh0, zh1 = 1.00, 1.30
    b.cyl("fr_handle", (hx, -0.105, zh0), (hx, -0.105, zh1), 0.012, rgba=COL["steel"], ind="  ")
    for n, zz in (("a", zh0 + 0.01), ("b", zh1 - 0.01)):
        b.box(f"fr_handle_{n}", (hx - 0.008, -0.105, zz - 0.008), (hx + 0.008, -0.06, zz + 0.008), rgba=COL["steel"], ind="  ")
    if "strap" in kits:
        b.cyl("fr_strap", (hx, -0.105, 0.40), (hx, -0.105, 1.0), 0.004, cls="vis", rgba=COL["kit"], ind="  ")
        b.box("fr_strap_ring", (hx - 0.03, -0.125, 0.385), (hx + 0.03, -0.085, 0.415), rgba=COL["kit"], ind="  ")
    b.raw('</body>')
    b.doors["fr_door"] = dict(hinge=(ax, yf), sign=-sgn, handle_r=0.55, handle=(hx, -0.105, (zh0 + zh1) / 2), axis="z",
                              leaf=(1.0 * sgn, 0.0), seal_N=h.seal_N, closes_below=h.closes_below)
    b.ports["fridge_handle"] = (ax + hx, yf - 0.105, (zh0 + zh1) / 2)


def build_kitchen(b: B, h: Home, kits):
    c = h.counter_h
    # worktop along the north wall from the fridge to the east wall
    fx1 = h.fridge_x + 0.60
    segs = [(fx1, 7.60)] if h.fridge_x <= 4.31 else [(4.30, h.fridge_x), (fx1, 7.60)]
    for i, (a, e) in enumerate(segs):
        if a < 5.70 < e:                                           # dishwasher slot at x 5.70..6.30
            b.box(f"cab_{i}a", (a, 7.70, 0.0), (5.70, 8.30, c - 0.04), rgba=COL["furn"])
            b.box(f"cab_{i}b", (6.30, 7.70, 0.0), (e, 8.30, c - 0.04), rgba=COL["furn"])
        else:
            b.box(f"cab_{i}", (a, 7.70, 0.0), (e, 8.30, c - 0.04), rgba=COL["furn"])
        b.box(f"top_{i}", (a, 7.68, c - 0.04), (e, 8.30, c), rgba=COL["wood"])
    b.box("upper_cab", (5.3, 7.95, c + 0.55), (7.60, 8.30, c + 1.25), rgba=COL["furn"])
    b.regions["worktop"] = ((6.85, 7.70, c), (7.55, 8.28, c + 0.40))
    b.ports["worktop_item"] = (7.10, 7.92, c)
    # dishwasher under the worktop: door drops down toward the room, bottom rack inside
    dx0, dx1, dy, z0, z1 = 5.70, 6.30, 7.70, 0.10, c - 0.06
    b.box("dw_plinth", (dx0, dy, 0.0), (dx1, 8.30, z0), rgba=COL["furn"])
    b.box("dw_back", (dx0, 8.24, z0), (dx1, 8.30, z1), rgba=COL["steel"])
    b.box("dw_side_w", (dx0, dy, z0), (dx0 + 0.02, 8.24, z1), rgba=COL["steel"])
    b.box("dw_side_e", (dx1 - 0.02, dy, z0), (dx1, 8.24, z1), rgba=COL["steel"])
    b.box("dw_top", (dx0, dy, z1 - 0.02), (dx1, 8.24, z1), rgba=COL["steel"])
    b.box("dw_rack", (dx0 + 0.04, dy + 0.05, 0.26), (dx1 - 0.04, 8.20, 0.27), rgba=COL["steel"])
    b.regions["dw_rack"] = ((dx0 + 0.05, dy + 0.06, 0.27), (dx1 - 0.05, 8.19, 0.60))
    b.raw(f'<body name="dw_door" pos="0 {dy:.4f} {z0:.4f}">')
    b.raw('  <joint name="dw_door" type="hinge" axis="1 0 0" range="0 1.57"/>')
    b.box("dw_door_slab", (dx0 + 0.005, -0.05, 0.0), (dx1 - 0.005, 0.0, z1 - z0), rgba=COL["furn"], ind="  ")
    hz = z1 - z0 - 0.06
    b.cyl("dw_handle", ((dx0 + dx1) / 2 - 0.15, -0.09, hz), ((dx0 + dx1) / 2 + 0.15, -0.09, hz), 0.010, rgba=COL["steel"], ind="  ")
    for s in (-1, 1):
        xc = (dx0 + dx1) / 2 + s * 0.14
        b.box(f"dw_handle_{'ab'[s > 0]}", (xc - 0.008, -0.09, hz - 0.008), (xc + 0.008, -0.05, hz + 0.008), rgba=COL["steel"], ind="  ")
    b.raw('</body>')
    b.doors["dw_door"] = dict(hinge=(0.0, dy, z0), axis="x", sign=1, handle=((dx0 + dx1) / 2, dy - 0.09, z0 + hz), mass=6.0, height=z1 - z0)
    b.ports["dw_handle"] = ((dx0 + dx1) / 2, dy - 0.09, z0 + hz)
    # countertop microwave on the worktop (46 x 34 x 28 cm); door hinged on the west, release button on the east panel
    mx0, mx1, my0, my1 = 6.10, 6.56, 7.72, 8.06
    z0 = c
    b.box("mw_bottom", (mx0, my0, z0), (mx1, my1, z0 + 0.03), rgba=COL["white"])
    b.box("mw_top", (mx0, my0, z0 + 0.25), (mx1, my1, z0 + 0.28), rgba=COL["white"])
    b.box("mw_back", (mx0, my1 - 0.02, z0 + 0.03), (mx1, my1, z0 + 0.25), rgba=COL["white"])
    b.box("mw_side_w", (mx0, my0, z0 + 0.03), (mx0 + 0.02, my1, z0 + 0.25), rgba=COL["white"])
    b.box("mw_panel", (mx1 - 0.11, my0, z0 + 0.03), (mx1, my1, z0 + 0.25), rgba=COL["dark"])
    b.regions["mw_cavity"] = ((mx0 + 0.03, my0 + 0.03, z0 + 0.03), (mx1 - 0.12, my1 - 0.03, z0 + 0.24))
    b.ports["mw_item"] = ((mx0 + mx1 - 0.11) / 2 + 0.005, my0 + 0.16, z0 + 0.03)
    b.ports["mw_btn_open"] = (mx1 - 0.055, my0 - 0.003, z0 + 0.07)
    b.ports["mw_btn_start"] = (mx1 - 0.055, my0 - 0.003, z0 + 0.17)
    b.raw(f'<body name="mw_door" pos="{mx0:.4f} {my0:.4f} 0">')
    b.raw('  <joint name="mw_door" type="hinge" axis="0 0 1" range="-1.75 0"/>')
    b.box("mw_door_slab", (0.0, -0.035, z0 + 0.005), (mx1 - mx0 - 0.11, 0.0, z0 + 0.275), rgba="0.25 0.27 0.30 1", ind="  ")
    b.box("mw_door_win", (0.05, -0.036, z0 + 0.05), (0.28, -0.035, z0 + 0.23), cls="vis", rgba=COL["glass"], ind="  ")
    b.raw('</body>')
    b.doors["mw_door"] = dict(hinge=(mx0, my0), axis="z", sign=-1, width=mx1 - mx0 - 0.11)
    for n, off in (("open", 0.07), ("start", 0.17)):
        p = (mx1 - 0.055, my0, z0 + off)
        b.raw(f'<body name="btn_{n}" pos="{_f(p)}">')
        b.raw(f'  <joint name="btn_{n}" type="slide" axis="0 1 0" range="0 0.003"/>')
        b.box(f"btn_{n}_cap", (-0.025, -0.006, -0.012), (0.025, 0.0, 0.012), rgba=COL["act"] if n == "open" else "0.30 0.80 0.40 1", ind="  ")
        b.raw('</body>')
    if "mw_pusher" in kits:
        b.box("mw_pusher_box", (mx1 - 0.08, my0 - 0.03, z0 + 0.04), (mx1 - 0.03, my0 - 0.008, z0 + 0.20), cls="vis", rgba=COL["kit"])
    # kitchen table narrows the aisle in front of the fridge to home.aisle
    ty1 = 7.59 - h.aisle
    ty0 = ty1 - 0.80
    tx0, tx1 = (h.fridge_x + 0.02, h.fridge_x + 0.92)
    if ty0 > 4.5:
        tz = 0.74
        b.box("ktable_top", (tx0, ty0, tz - 0.03), (tx1, ty1, tz), rgba=COL["wood"])
        for i, (x, y) in enumerate([(tx0 + 0.02, ty0 + 0.02), (tx1 - 0.08, ty0 + 0.02), (tx0 + 0.02, ty1 - 0.08), (tx1 - 0.08, ty1 - 0.08)]):
            b.box(f"ktable_leg{i}", (x, y, 0), (x + 0.06, y + 0.06, tz - 0.03), rgba=COL["wood"])
        b.regions["ktable"] = ((tx0 + 0.05, ty0 + 0.05, tz), (tx1 - 0.05, ty1 - 0.05, tz + 0.30))


def build_rest(b: B, h: Home):
    # bathroom and washer-dryer (WNC254A40 class), simplified but with a real porthole and drum
    b.box("tub", (0.0, 0.40, 0.0), (1.70, 1.15, 0.58), rgba=COL["white"])
    b.box("toilet", (0.0, 1.20, 0.0), (0.65, 1.58, 0.42), rgba=COL["white"])
    b.box("sink_b", (0.0, 1.95, 0.80), (0.45, 2.45, 0.88), rgba=COL["white"])
    x0, x1, y0, y1, hh = 2.00, 2.60, 2.25, 2.848, 0.848
    cy, cz, r_open, r_bez, r_drum = (y0 + y1) / 2, 0.50, 0.16, 0.245, 0.25
    drum = (2.14, 2.50)
    b.ports["porthole"] = (x0, cy, cz)
    b.ports["drum_c"] = (sum(drum) / 2, cy, cz)
    w = COL["white"]
    b.box("wm_top", (x0, y0, hh - 0.14), (x1, y1, hh), rgba=w)
    b.box("wm_back", (x1 - 0.02, y0, 0), (x1, y1, hh - 0.14), rgba=w)
    b.box("wm_side_s", (x0, y0, 0), (x1, y0 + 0.02, hh - 0.14), rgba=w)
    b.box("wm_side_n", (x0, y1 - 0.02, 0), (x1, y1, hh - 0.14), rgba=w)
    b.box("wm_base", (x0, y0, 0), (x1, y1, cz - r_drum - 0.02), rgba=w)
    b.box("wm_front_s", (x0, y0, cz - r_drum - 0.02), (x0 + 0.02, cy - r_bez, hh - 0.14), rgba=w)
    b.box("wm_front_n", (x0, cy + r_bez, cz - r_drum - 0.02), (x0 + 0.02, y1, hh - 0.14), rgba=w)
    b.box("wm_front_t", (x0, cy - r_bez, cz + r_bez), (x0 + 0.02, cy + r_bez, hh - 0.14), rgba=w)
    b.box("wm_front_b", (x0, cy - r_bez, cz - r_drum - 0.02), (x0 + 0.02, cy + r_bez, cz - r_bez), rgba=w)
    n = 16
    for i in range(n):
        a = 2 * math.pi * i / n
        rm = (r_open + r_bez) / 2
        c = (x0 + 0.07, cy + rm * math.cos(a), cz + rm * math.sin(a))
        half = (0.07, (r_bez - r_open) / 2, math.pi * rm / n + 0.006)
        b.lines.append(f'<geom name="wm_gasket{i}" class="world" type="box" pos="{_f(c)}" size="{_f(half)}" euler="{_f((a, 0, 0))}" rgba="0.30 0.32 0.35 1"/>')
        rd = r_drum + 0.01
        c = (sum(drum) / 2, cy + rd * math.cos(a), cz + rd * math.sin(a))
        half = ((drum[1] - drum[0]) / 2, 0.01, math.pi * rd / n + 0.006)
        b.lines.append(f'<geom name="wm_drum{i}" class="world" type="box" pos="{_f(c)}" size="{_f(half)}" euler="{_f((a, 0, 0))}" rgba="0.70 0.72 0.75 1"/>')
    b.box("wm_drum_back", (drum[1], cy - r_drum, cz - r_drum), (drum[1] + 0.02, cy + r_drum, cz + r_drum), rgba="0.70 0.72 0.75 1")
    # hall, front door (closed: the robot stays inside the flat), mat
    b.box("frontdoor", (7.60, 3.15, 0.0), (7.66, 4.03, 2.05), rgba=COL["wood"])
    b.box("doormat", (6.75, 3.25, 0.0), (7.50, 3.95, 0.010), rgba=COL["dark"])
    b.regions["doormat"] = ((6.78, 3.28, 0.010), (7.48, 3.92, 0.60))
    b.ports["doormat_item"] = (7.10, 3.60, 0.010)
    b.box("hall_bin", (5.40, 3.02, 0.0), (5.75, 3.32, 0.45), rgba=COL["dark"])
    b.ports["bin_top"] = (5.575, 3.17, 0.45)


def build(h: Home, kits=()):
    b = B()
    build_shell(b, h, kits)
    build_bedroom(b, h)
    build_fridge(b, h, kits)
    build_kitchen(b, h, kits)
    build_rest(b, h)
    return b
