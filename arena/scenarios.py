"""The tests. A scenario sets up the world, tells the robot what is wanted in the words a user would use, and states the
goal on the world state. It says nothing about how: no joint angles, no motion script, no named grasp.

Goal functions read the truth and belong to the judge's side. task functions are the reference controller's plan and see only
what skills.Controller sees; a robot entry may replace them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import objects, appliance_judge
from .contacts import floor_spots, rug_z, world_cands
from .reach import Contact
from .sim import World


@dataclass
class Scenario:
    name: str
    cls: str                       # need class of the value model
    objs: tuple                    # objects the trial needs
    setup: object                  # setup(world, rng) -> context dict
    task: object                   # task(ctx) -> task(controller) generator factory
    goal: object                   # goal(world, ctx) -> (ok, why)
    limit: float = 240.0           # s of robot time
    place: object = None           # place(world, tier1, ctx, rng): put the robot at its own best pose (skill tests)


def _in(region, p, margin=0.0):
    return all(lo - margin <= x <= hi + margin for x, lo, hi in zip(p, region[0], region[1]))


def _resting(w: World, name, tol=0.03):
    v = w.obj[name]["v"]
    return float(np.linalg.norm(w.d.qvel[v:v + 3])) < tol and float(np.linalg.norm(w.d.qvel[v+3:v+6])) < .3


def _released_on(w, name, supports):
    """Contact with an accepted support, and no remaining contact with the robot."""
    body = w.obj[name]["b"]
    supported = False
    for con in w.d.contact:
        a, b = int(con.geom1), int(con.geom2)
        if w.m.geom_bodyid[a] != body and w.m.geom_bodyid[b] != body:
            continue
        other = b if w.m.geom_bodyid[a] == body else a
        if w.g_rob[other] and con.dist < .001:
            return False
        if w.gname[other] in supports:
            supported = True
    return supported


def _upright(w: World, name, deg=20):
    return w.obj_mat(name)[2, 2] > math.cos(math.radians(deg))


def deliver_goal(name, upright=True):
    def goal(w: World, ctx):
        reg = ctx["zone"]
        p = w.obj_pos(name)
        if not _in(reg, p, 0.02):
            return False, f"{name} not in reach of the person (at {np.round(p, 2).tolist()})"
        if not _resting(w, name):
            return False, f"{name} still moving"
        if upright and not _upright(w, name):
            return False, f"{name} fell over"
        if not _released_on(w, name, {"obt_top"}):
            return False, f"{name} not released onto the table"
        return True, ""
    return goal


def table_spot(w: World):
    z = w.b.world.regions["table"][0][2]
    return np.array([1.15, 7.22, z])


def _deliver_zone(w: World):
    r = w.b.world.regions["reach_person"]
    return r


# ------------------------------------------------------------- floor pickup ---
def floor_scenario(cat):
    def setup(w: World, rng):
        h = w.home
        spots = floor_spots(h)
        nm, x, y = spots[rng.integers(len(spots))]
        x += rng.uniform(-0.05, 0.05)
        y += rng.uniform(-0.05, 0.05)
        yaw = rng.uniform(-math.pi, math.pi)
        z = rug_z(h, x, y)
        w.place(cat, (x, y, z + 0.001), yaw)
        return dict(obj=cat, where=nm, hint=(x + rng.normal(0, 0.25), y + rng.normal(0, 0.25)), zone=_deliver_zone(w), spot=table_spot(w))

    def task(ctx):
        def run(c):
            hx, hy = ctx["hint"]
            c.hint = (hx, hy)
            yield from c.look_near(hx, hy)
            ok = yield from c.grasp_obj(cat, cat, lambda pos, yaw: _pick(c, cat, pos, yaw), lambda p: rug_z(c.B.home, p[0], p[1]))
            if not ok:
                return False
            return (yield from c.place_obj(cat, cat, ctx["spot"], 1.57))
        return run
    return Scenario(f"floor_{cat}", "dropped item", (cat,), setup, task, deliver_goal(cat, upright=False))


def _pick(c, cat, pos, yaw, block=None, doors=None):
    cs = world_cands(cat, pos, yaw, block)
    return Contact("pick", "x", "grasp", cs, cat, (tuple(pos), yaw), objects.CATALOG[cat].mass, doors=doors or {})


# ------------------------------------------------------------- put down here ---
def carry_scenario(cat):
    """Pick from a known spot on the bedside table and put it in reach: the shortest fetch, a baseline for the rest."""
    def setup(w: World, rng):
        x, y = 1.75 + rng.uniform(-0.05, 0.05), 8.02 + rng.uniform(-0.05, 0.05)
        yaw = rng.uniform(-math.pi, math.pi)
        w.place(cat, (x, y, 0.602), yaw)
        return dict(obj=cat, zone=_deliver_zone(w), spot=table_spot(w), z=0.60, hint=(x + rng.normal(0, 0.05), y + rng.normal(0, 0.05)))

    def task(ctx):
        def run(c):
            c.hint = ctx["hint"]
            yield from c.look_near(*ctx["hint"], z=ctx["z"])
            ok = yield from c.grasp_obj(cat, cat, lambda pos, yaw: _pick(c, cat, pos, yaw, lambda a: a[1] > 0.5), lambda p: ctx["z"])
            if not ok:
                return False
            return (yield from c.place_obj(cat, cat, ctx["spot"], 1.57))
        return run
    return Scenario(f"bedside_{cat}", "long-tail fetch", (cat,), setup, task, deliver_goal(cat))


# -------------------------------------------------------------------- fridge ---
def fridge_scenario(cat, cls):
    def setup(w: World, rng):
        p = np.array(w.b.world.ports["fridge_item"], float)
        p[0] += rng.uniform(-0.05, 0.05)
        p[1] += rng.uniform(-0.03, 0.03)
        yaw = rng.uniform(-0.3, 0.3)
        w.place(cat, (p[0], p[1], p[2] + 0.001), yaw)
        return dict(obj=cat, zone=_deliver_zone(w), spot=table_spot(w), z=p[2], front=w.b.world.ports["fridge_handle"], hint=(p[0], p[1]),
                    doors={"fr_door": -1.6 if w.home.hinge == "west" else 1.6})

    def task(ctx):
        def run(c):
            ok = yield from c.fridge_open()
            if not ok:
                return False
            c.hint = ctx["hint"]
            yield from c.look_near(*ctx["hint"], radius=1.0, z=ctx["z"] + 0.05)
            ok = yield from c.grasp_obj(cat, cat, lambda pos, yaw: _pick(c, cat, pos, yaw, lambda a: a[1] < 0.5 or a[2] < -0.5, doors=ctx["doors"]), lambda p: ctx["z"], kind="fridge")
            if not ok:
                return False
            yield from c.fridge_close()
            return (yield from c.place_obj(cat, cat, ctx["spot"], 1.57))
        return run

    def goal(w, ctx):
        ok, why = deliver_goal(cat)(w, ctx)
        if not ok:
            return ok, why
        if w.door_angle("fr_door") > math.radians(3):
            return False, "fridge left open"
        return True, ""
    return Scenario(f"fridge_{cat}", cls, (cat,), setup, task, goal, limit=420.0)


# ---------------------------------------------------------------- navigation ---
def nav_scenario(name, gx, gy, gyaw, tol=0.10):
    def setup(w: World, rng):
        return dict(goal=(gx, gy, gyaw))

    def task(ctx):
        def run(c):
            return (yield from c.goto(gx, gy, gyaw, tol=0.03))
        return run

    def goal(w, ctx):
        x, y, yaw = w.base_pose()
        e = math.hypot(x - gx, y - gy)
        ctx["dock_err"] = e
        return e < tol, f"{e * 100:.0f} cm off"
    return Scenario(name, "navigation", (), setup, task, goal, limit=180.0)


# ---------------------------------------------------------------- skill tests ---
# One physical skill at a time, robot already at its best pose with the object in view. What is left to judge is the
# hardware: reach, stroke, friction, force, precision, what its cameras see. A job's reliability is the chain of these.
SURFACES = dict(floor="dropped item", bedside="long-tail fetch", shelf="long-tail fetch", fridge="drink", worktop="hot meal", table="clear away")


def _spot(w: World, where, cat, rng):
    """World position, yaw, approach filter, doors, support height."""
    W = w.b.world
    o = objects.CATALOG[cat]
    if where == "floor":
        nm, x, y = floor_spots(w.home)[rng.integers(len(floor_spots(w.home)))]
        x, y = x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05)
        z = rug_z(w.home, x, y)
        return (x, y, z), rng.uniform(-math.pi, math.pi), None, {}, nm
    if where == "bedside":
        return (1.75 + rng.uniform(-0.05, 0.05), 8.02 + rng.uniform(-0.05, 0.05), 0.60), rng.uniform(-math.pi, math.pi), (lambda a: a[1] > 0.5), {}, "bedside table"
    if where == "shelf":
        return (3.95 + rng.uniform(-0.02, 0.03), 6.40 + rng.uniform(-0.3, 0.3), 0.80), rng.choice([0.0, 1.57]) + rng.uniform(-0.15, 0.15), (lambda a: a[0] < 0.5), {}, "shelf"
    if where == "worktop":
        p = W.ports["worktop_item"]
        return (p[0] + rng.uniform(-0.1, 0.1), p[1] + rng.uniform(-0.05, 0.05), p[2]), rng.uniform(-math.pi, math.pi), (lambda a: a[1] > 0.5), {}, "worktop"
    if where == "table":
        z = W.regions["table"][0][2]
        return (1.15 + rng.uniform(-0.08, 0.08), 7.20 + rng.uniform(-0.12, 0.12), z), rng.uniform(-math.pi, math.pi), None, {}, "over-bed table"
    if where == "fridge":
        p = W.ports["fridge_item"]
        ang = -1.6 if w.home.hinge == "west" else 1.6
        return (p[0] + rng.uniform(-0.05, 0.05), p[1] + rng.uniform(-0.03, 0.03), p[2]), rng.uniform(-0.3, 0.3), (lambda a: a[1] < 0.5 or a[2] < -0.5), {"fr_door": ang}, "fridge shelf"
    raise KeyError(where)


def pick_scenario(cat, where):
    cls = SURFACES[where] if cat != "meal_box" or where != "fridge" else "snack, cold food"

    def setup(w: World, rng):
        p, yaw, block, doors, nm = _spot(w, where, cat, rng)
        w.place(cat, (p[0], p[1], p[2] + 0.0015), yaw)
        return dict(pos=np.array(p, float), yaw=yaw, block=block, doors=doors, where=nm, z=p[2], open=doors)

    def place(w: World, T, ctx, rng):
        cs = world_cands(cat, ctx["pos"], ctx["yaw"], ctx["block"])
        c = Contact("pick", "x", "grasp", cs, cat, (tuple(ctx["pos"]), ctx["yaw"]), objects.CATALOG[cat].mass, doors=ctx["doors"])
        sol = T.solve(c, seeds=40, rng=rng)
        ctx["sol"] = sol
        ctx["prior"] = rng.normal(0, 0.02, 2)
        if not sol["ok"]:
            ctx["infeasible"] = f"{sol['why']}: {sol['detail']}"[:70]
            return False
        x, y, yaw = sol["base"]
        ctx["nominal_pose"] = (x, y, yaw)
        q = np.array(sol["q"])
        T._set(x, y, yaw, q)
        pg = T.d.site_xpos[T.ee].copy()
        R = T.d.site_xmat[T.ee].reshape(3, 3)
        look = pg - R[:, 0] * 0.09 + np.array([0, 0, 0.10])
        u = T._ik(np.concatenate([[x, y, yaw], q]), look, R[:, 0], R[:, 1], fix_base=True)
        qs = u[0][3:] if u[0] is not None else q
        n = rng.normal(0, 1, 3) * np.array([0.008, 0.008, math.radians(1.0)])        # it arrived by itself: docking error
        w.place_robot(x + n[0], y + n[1], yaw + n[2], q=qs)
        return True

    def task(ctx):
        def run(c):
            p = ctx["pos"] + np.r_[ctx["prior"], 0]                                 # where it thinks it saw it, from afar
            c.mem[cat] = (p, -1.0, ctx["yaw"], "prior", 1.0, -1.0)
            c.q_cmd = np.array(c.obs.q)
            c.hint = tuple(p[:2])
            ok = yield from c.grasp_obj(cat, cat, lambda pos, yaw: _pick(c, cat, pos, yaw, ctx["block"], ctx["doors"]), lambda q: ctx["z"], attempts=2)
            if not ok:
                return False
            yield from c._hold(2.0)
            return True
        return run

    def goal(w: World, ctx):
        p = w.obj_pos(cat)
        ee = w.d.site_xpos[w.ee]
        if p[2] < ctx["z"] + 0.025:
            return False, "not lifted"
        if float(np.linalg.norm(p - ee)) > 0.18:
            return False, "not in the hand"
        return True, ""
    # a skill test judges whether it works; how long it takes is measured, not a pass mark: a 3 cm/s lift is slow, not incapable
    return Scenario(f"pick_{cat}_{where}", cls, (cat,), setup, task, goal, limit=180.0, place=place)


PICKS = dict(floor=("phone", "remote", "pill_bottle", "tablet_box", "tissue_pack", "spoon", "glasses", "toothbrush"),
             bedside=("tablet_box", "pill_bottle", "mug"), shelf=("tablet_box", "book", "tissue_pack"), fridge=("bottle", "can", "meal_box"),
             worktop=("meal_box", "mug", "bowl"), table=("mug", "bowl", "bottle"))

SCENARIOS = {}
for wh, cats in PICKS.items():
    for c_ in cats:
        s_ = pick_scenario(c_, wh)
        SCENARIOS[s_.name] = s_
for s in [floor_scenario(c) for c in ("phone", "remote", "pill_bottle", "tablet_box", "tissue_pack", "spoon", "glasses", "toothbrush")]:
    SCENARIOS[s.name] = s
for s in [carry_scenario(c) for c in ("tablet_box", "pill_bottle")]:
    SCENARIOS[s.name] = s
for s in [fridge_scenario("bottle", "drink"), fridge_scenario("can", "drink"), fridge_scenario("meal_box", "snack, cold food")]:
    SCENARIOS[s.name] = s
for s in [nav_scenario("nav_kitchen", 5.00, 6.90, 1.57), nav_scenario("nav_bedside", 2.05, 6.60, 1.57), nav_scenario("nav_hall", 6.50, 3.60, 0.0)]:
    SCENARIOS[s.name] = s


def surface_delivery(cat, where):
    def setup(w, rng):
        p, yaw, block, doors, label = _spot(w, where, cat, rng)
        w.place(cat, np.array(p)+(0, 0, .0015), yaw)
        # A region hint, rather than the randomized true object pose.
        hint = w.b.world.ports["worktop_item"][:2] if where == "worktop" else (3.95, 6.40)
        return dict(hint=hint, z=p[2], block=block, doors=doors, where=label,
                    zone=_deliver_zone(w), spot=table_spot(w))
    def task(ctx):
        def run(c):
            c.hint = ctx["hint"]
            c.mark("search_transit")
            if not (yield from c.look_near(*ctx["hint"], z=ctx["z"])):
                c.fail("search_navigation", "could not reach search region")
                return False
            if not (yield from c.grasp_obj(cat, cat, lambda p, yaw: _pick(c, cat, p, yaw, ctx["block"], ctx["doors"]), lambda p: ctx["z"])):
                return False
            return (yield from c.place_obj(cat, cat, ctx["spot"], 1.57))
        return run
    return Scenario(f"{where}_{cat}", "prepared meal delivery" if where == "worktop" else "long-tail fetch",
                    (cat,), setup, task, deliver_goal(cat), limit=600)


def clear_scenario(cat):
    def setup(w, rng):
        z = w.b.world.regions["table"][0][2]
        w.place(cat, (1.12+rng.uniform(-.06, .06), 7.20+rng.uniform(-.08, .08), z+.0015), rng.uniform(-3.14, 3.14))
        # The dirty cup is empty; this is not a request to carry spilled liquid.
        w.obj[cat]["empty"] = True
        return dict(z=z, hint=(1.12, 7.20), destination=w.b.world.ports["worktop_item"])
    def task(ctx):
        def run(c):
            c.hint = ctx["hint"]
            c.mark("search_transit")
            if not (yield from c.look_near(*ctx["hint"], z=ctx["z"])):
                c.fail("search_navigation", "could not reach table search region")
                return False
            if not (yield from c.grasp_obj(cat, cat, lambda p, y: _pick(c, cat, p, y), lambda p: ctx["z"])):
                return False
            return (yield from c.place_obj(cat, cat, ctx["destination"], 0))
        return run
    def goal(w, ctx):
        if not _in(w.b.world.regions["worktop"], w.obj_pos(cat), .02):
            return False, "not on the kitchen worktop"
        return bool(_resting(w, cat) and _released_on(w, cat, {f"top_{i}" for i in range(3)})), "not released at rest on worktop"
    return Scenario(f"clear_{cat}", "clear away", (cat,), setup, task, goal, limit=600)


def microwave_scenario():
    cat = "meal_box"
    def setup(w, rng):
        p = np.array(w.b.world.ports["worktop_item"])
        w.place(cat, p+(rng.uniform(-.05, .05), rng.uniform(-.04, .04), .0015), rng.uniform(-.2, .2))
        return dict(hint=tuple(p[:2]), z=p[2], zone=_deliver_zone(w), spot=table_spot(w))
    def task(ctx):
        def run(c):
            W = c.B.world
            c.hint = ctx["hint"]
            if not (yield from c.look_near(*ctx["hint"], z=ctx["z"])):
                c.fail("search_navigation", "could not reach meal search region")
                return False
            if not (yield from c.grasp_obj(cat, cat, lambda p, y: _pick(c, cat, p, y, lambda a: a[1] < .5), lambda p: ctx["z"])):
                return False
            # Set the meal down while the single hand opens the door.
            staging = np.array(W.ports["worktop_item"])+(.15, 0, 0)
            if not (yield from c.place_obj(cat, cat, staging, 0)):
                return False
            if not (yield from c.microwave_door(True)):
                return False
            c.hint = tuple(staging[:2])
            if not (yield from c.grasp_obj(cat, cat, lambda p, y: _pick(c, cat, p, y), lambda p: ctx["z"])):
                return False
            if not (yield from c.place_obj(cat, cat, W.ports["mw_item"], 0)):
                return False
            if not (yield from c.microwave_door(False)):
                return False
            c.mark("microwave_heat")
            if "mw_pusher" not in c.r.kits:
                c.fail("missing_button_skill", "microwave requires the button pusher")
                return False
            yield dict(q=c.q_cmd, kit="mw_start")
            yield from c.wait(12)
            if not (yield from c.microwave_door(True)):
                return False
            p = W.ports["mw_item"]
            c.hint = tuple(p[:2])
            if not (yield from c.grasp_obj(cat, cat, lambda p_, y: _pick(c, cat, p_, y, lambda a: a[1] < .5, {"mw_door": -1.66}), lambda p_: p[2])):
                return False
            # Deliver first: the occupied hand cannot close the door reliably.
            if not (yield from c.place_obj(cat, cat, ctx["spot"], 1.57)):
                return False
            return (yield from c.microwave_door(False))
        return run
    def goal(w, ctx):
        if not appliance_judge.heated(w, cat):
            return False, "meal did not spend 10 s inside the closed, running microwave"
        if w.door_angle("mw_door") > .052:
            return False, "microwave left open"
        return deliver_goal(cat)(w, ctx)
    return Scenario("microwave_meal", "hot meal", (cat,), setup, task, goal, limit=1200)


for wh in ("worktop", "shelf"):
    for cat in PICKS[wh]:
        s = surface_delivery(cat, wh)
        SCENARIOS[s.name] = s
for cat in ("mug", "bowl"):
    s = clear_scenario(cat)
    SCENARIOS[s.name] = s
s = microwave_scenario()
SCENARIOS[s.name] = s
FULL_JOBS = [k for k in SCENARIOS if not k.startswith(("pick_", "nav_"))]
