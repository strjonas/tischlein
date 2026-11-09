"""Everything a home demands of a body, as a flat list of physical interactions. Robot-agnostic: the hand and the
kits decide which candidates apply. Each class is a kind of household need (dropped item, drink, snack, hot meal, clearing
away, ...), so what a body can reach becomes its envelope over those needs.
"""
from __future__ import annotations

import math

import numpy as np

from .homes import B, Home
from .objects import CATALOG, FLOOR_SET
from .reach import Contact, rz

DOWN = (0, 0, -1)


def world_cands(cat, pos, yaw, block=None):
    """Grasp candidates of an object at rest: world point, approach, jaw axis, width. block: drop approaches that
    run through something, a function of the approach direction."""
    o = CATALOG[cat]
    R = rz(yaw)
    out = []
    for g in o.grasps:
        for a in g.approach:
            aw = R @ np.array(a, float)
            if block and block(aw):
                continue
            p = np.asarray(pos, float) + R @ np.array(g.p, float)
            depth = float(np.abs(np.array(a, float)) @ np.array(o.size)) if g.name != "rim end" else 0.03
            out.append((p, aw, R @ np.array(g.jaw, float), g.width, depth))
    return out


def _from_south(a):
    return a[1] < 0.3


def floor_spots(h: Home):
    """Where things land. Open floor, by furniture, a hand's width under the bed, under the desk."""
    return [("open floor", 2.45, 6.10), ("beside the bed", 1.60, 6.70), ("by the table foot", 1.68, 7.15), ("foot of the bed", 0.80, 6.08),
            ("under the bed edge", 1.36, 6.80), ("under the desk", 3.20, 7.72), ("by the chair", 2.80, 7.30)]


def rug_z(h: Home, x, y):
    return 0.008 if h.rug and 1.60 <= x <= 3.00 and 5.60 <= y <= 7.40 else 0.0


def all_contacts(h: Home, w: B, kits=()):
    C = []
    # ----- dropped items: every object at every spot, two orientations
    for cat in FLOOR_SET:
        for nm, x, y in floor_spots(h):
            for yaw in (0.3, 1.8):
                z = rug_z(h, x, y)
                p = (x, y, z)
                C.append(Contact(f"pick {cat}, {nm}", "dropped item", "grasp", world_cands(cat, p, yaw), cat, (p, yaw), CATALOG[cat].mass))
    # ----- place on the over-bed table, in the person's reach
    t = w.regions["table"][0][2]
    for cat in ("phone", "bottle", "meal_box", "mug", "tablet_box", "bowl"):
        for x, y in ((1.22, 7.32), (1.22, 7.04), (1.10, 7.20), (0.90, 7.20)):
            p = (x, y, t)
            C.append(Contact(f"place {cat} on table", "handover", "place", world_cands(cat, p, 1.57, lambda a: a[2] > 0.5), cat, (p, 1.57), CATALOG[cat].mass, group=f"table {cat}"))
        p = (1.30, 7.62, h.bed_h)                                                    # or on the mattress beside the arm
        C.append(Contact(f"set {cat} on the mattress", "handover", "place", world_cands(cat, p, 1.57, lambda a: a[2] > 0.5), cat, (p, 1.57), CATALOG[cat].mass, group=f"table {cat}"))
    # ----- fridge
    fr_open = -1.6 if h.hinge == "west" else 1.6
    pf = w.ports["fridge_item"]
    for cat, cls, off in (("bottle", "drink", 0.0), ("can", "drink", 0.0), ("meal_box", "snack, cold food", 0.0), ("bottle", "drink", 0.14), ("meal_box", "snack, cold food", 0.14)):
        p = (pf[0], pf[1] + off, pf[2])
        cs = world_cands(cat, p, 0.0, lambda a: a[1] < 0.5 or a[2] < -0.5)          # shelf above: from the front only
        C.append(Contact(f"pick {cat} from fridge shelf +{int(off * 100)} cm", cls, "grasp", cs, cat, (p, 0.0), CATALOG[cat].mass, doors={"fr_door": fr_open}))
    # door bin: the bin rides on the open door, the item stands in it with a 6 cm rail
    hinge = w.doors["fr_door"]["hinge"]
    sgn = 1 if h.hinge == "west" else -1
    d = 0.30
    if h.hinge == "west":
        p = (hinge[0] - 0.05, hinge[1] - d, w.ports["fridge_handle"][2] - 0.35)
    else:
        p = (hinge[0] + 0.05, hinge[1] - d, w.ports["fridge_handle"][2] - 0.35)
    C.append(Contact("pick bottle from door bin", "drink", "grasp", world_cands("bottle", p, 0.0, lambda a: a[2] > -0.5), "bottle", (p, 0.0), 0.54, doors={"fr_door": fr_open}, tol_deg=25))
    # fridge door by hand, for bodies without the opener: pull at the handle against the seal, then round the arc
    fh = np.array(w.ports["fridge_handle"])
    if "fridge_opener" not in kits:
        C.append(Contact("break fridge seal at the handle", "drink", "pull", [(fh, (0, 1, 0), (1, 0, 0), 0.024)], "fr_handle", force=(0, h.seal_N, 0)))
        hv = np.array([hinge[0], hinge[1], fh[2]])
        r = fh - hv
        for deg in (20, 45, 70, 90):
            th = math.radians(deg) * (-sgn)
            pa = hv + rz(th) @ r
            a = rz(th) @ np.array([0, 1.0, 0])
            C.append(Contact(f"swing fridge door to {deg} deg", "drink", "pull", [(pa, a, rz(th) @ np.array([1.0, 0, 0]), 0.024)], "fr_handle", force=tuple(a * 12.0), doors={"fr_door": th}, tol_deg=30))
    # ----- hot meal: into and out of the microwave
    pm = w.ports["mw_item"]
    for cat in ("meal_box", "mug"):
        cs = world_cands(cat, pm, 0.0, lambda a: a[1] < 0.5)
        C.append(Contact(f"{cat} into microwave", "hot meal", "place", cs, cat, (pm, 0.0), CATALOG[cat].mass, doors={"mw_door": -1.6}))
        C.append(Contact(f"{cat} out of microwave", "hot meal", "grasp", cs, cat, (pm, 0.0), CATALOG[cat].mass, doors={"mw_door": -1.6}))
    if "mw_pusher" not in kits:
        for nm, port in (("release", "mw_btn_open"), ("start", "mw_btn_start")):
            C.append(Contact(f"press microwave {nm}", "hot meal", "press", [(w.ports[port], (0, 1, 0), None, None)], "btn_", force=(0, -8.0, 0)))
        hg = w.doors["mw_door"]["hinge"]
        edge = np.array([hg[0] + 0.30, hg[1] - 0.05, w.ports["mw_item"][2] + 0.08])
        C.append(Contact("close microwave door by push", "hot meal", "press", [(edge, (0, 1, 0), None, None)], "mw_door", force=(0, -12.0, 0), doors={"mw_door": -0.5}))
    pw = w.ports["worktop_item"]
    C.append(Contact("pick meal box from worktop", "hot meal", "grasp", world_cands("meal_box", pw, 0.0, lambda a: a[1] < 0.3), "meal_box", (pw, 0.0), 0.55))
    # ----- long-tail fetch: bookshelf at three heights, bedside table
    for zname, z in (("0.80", 0.80), ("1.20", 1.20), ("1.60", 1.60)):
        for cat in ("tablet_box", "book", "tissue_pack"):
            p = (3.95, 6.40, z)                                                      # near the front, spine to the room
            for yaw in (0.0, 1.57):                                                  # either way round
                cs = world_cands(cat, p, yaw, lambda a: a[0] < 0.5)                  # the shelf opens to the west
                C.append(Contact(f"pick {cat} from shelf at {zname}", "long-tail fetch", "grasp", cs, cat, (p, yaw), CATALOG[cat].mass, group=f"{cat} shelf {zname}"))
    for cat in ("tablet_box", "pill_bottle"):
        p = (1.75, 8.05, 0.60)
        C.append(Contact(f"pick {cat} from bedside table", "long-tail fetch", "grasp", world_cands(cat, p, 0.0, lambda a: a[1] > 0.5), cat, (p, 0.0), CATALOG[cat].mass))
    # ----- clear away
    for cat in ("mug", "bowl"):
        p = (1.05, 7.20, t)
        C.append(Contact(f"pick used {cat} from table", "clear away", "grasp", world_cands(cat, p, 1.57), cat, (p, 1.57), CATALOG[cat].mass))
        C.append(Contact(f"set {cat} on worktop", "clear away", "place", world_cands(cat, pw, 0.0, _from_south), cat, (pw, 0.0), CATALOG[cat].mass))
    # ----- deliveries and rubbish
    pd = w.ports["doormat_item"]
    C.append(Contact("pick parcel from door mat", "delivery inward", "grasp", world_cands("parcel", pd, 0.0), "parcel", (pd, 0.0), 3.0, group="parcel"))
    C.append(Contact("courier sets parcel on the tray", "delivery inward", "tray", [], "parcel", group="parcel"))
    pb = w.ports["bin_top"]
    p = (pb[0], pb[1], pb[2] - 0.30)
    C.append(Contact("lift bin bag out of the bin", "clear away", "grasp", world_cands("bin_bag", p, 0.0), "bin_bag", (p, 0.0), 2.0))
    # ----- household extras: dishwasher and washer-dryer
    rack = w.regions["dw_rack"][0]
    p = ((rack[0] + w.regions["dw_rack"][1][0]) / 2, rack[1] + 0.14, rack[2])
    for cat in ("bowl", "mug"):
        C.append(Contact(f"{cat} into dishwasher rack", "household", "place", world_cands(cat, p, 0.0, lambda a: a[2] > -0.5), cat, (p, 0.0), CATALOG[cat].mass, doors={"dw_door": 1.5}, tol_deg=25))
    dc = np.array(w.ports["drum_c"])
    for nm, dp in (("front", -0.10), ("middle", 0.05), ("back", 0.15)):
        p = (dc[0] + dp, dc[1], dc[2] - 0.20)
        C.append(Contact(f"garment from drum, {nm}", "household", "grasp", [(p, (0.5, 0, -0.866), (0, 1, 0), 0.03), (p, (1, 0, 0), (0, 1, 0), 0.03), (p, (0.5, 0, -0.866), (0, 0, 1), 0.03)], None, mass=0.4))
    return C
