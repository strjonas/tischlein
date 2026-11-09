"""One trial: world, robot, scenario, seed -> a scored record. judge.py aggregates these."""
from __future__ import annotations

import math
import time

import numpy as np

from . import assemble, homes, objects, reach, skills, zoo, judge, provenance
from .diagnostics import Trace
from .contact_audit import clear
from .scenarios import SCENARIOS, Scenario
from .sim import CTRL_DT, World

_cache = {}


def _model(robot_name, home_key, objs, phys=None, variant=None):
    r = zoo.ROBOTS[robot_name]()
    if variant:
        variant(r)
    return assemble.build(r, homes.HOMES[home_key], objs=[(o, o) for o in objs], phys=phys)


def run(robot_name, scenario, home_key="A", seed=0, dr=True, noise=1.0, record=False, variant=None, phys=None, verbose=False, tune=None, policy="improved"):
    """Runs to the goal, the controller's verdict, or the time limit. Fresh models each time: DR edits them in place."""
    s: Scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    t0 = time.time()
    b = _model(robot_name, home_key, s.objs, phys, variant)
    belief = _model(robot_name, home_key, list(objects.CATALOG), phys, variant)
    ident = provenance.identity(b.robot, s.name, home_key, seed, policy, dr=dr, noise=noise, phys=phys, tune=tune)
    ident.update(model_hash=b.digest)
    if not b.robot.stow_valid:
        return dict(valid=False, ok=False, robot=robot_name, scenario=s.name, cls=s.cls, home=home_key, seed=seed,
                    policy=policy, why="MODEL INVALID: no collision-free travelling pose found", provenance=ident)
    w = World(b, seed=seed, dr=dr, noise=noise)
    rng = np.random.default_rng(seed + 7919)
    T = reach.Tier1(belief)
    initial_pose = T.dock_pose()
    # A mapped dock has finite repeatability; the estimator starts at its nominal pose.
    pose_rng = np.random.default_rng(seed+7950)
    pose_error = pose_rng.normal(0, 1, 3)*np.array([.008, .008, math.radians(1)]) if noise else np.zeros(3)
    w.place_robot(*(np.array(initial_pose)+pose_error))
    ctx = s.setup(w, rng)
    if s.place and not s.place(w, T, ctx, rng):
        return dict(valid=True, robot=robot_name, scenario=s.name, cls=s.cls, home=home_key, seed=seed, ok=False, why="tier 1: " + ctx["infeasible"], tier1=True, t=0.0, log=[], wall=round(time.time() - t0, 1), provenance=ident)
    w.start(doors=ctx.get("open"))
    if record:
        w.record = []
    c = skills.Controller(None, belief, s.task(ctx), T=T, initial_pose=ctx.get("nominal_pose", initial_pose), policy=policy)
    c.tune.update(tune or {})
    ob = w.step()
    limit = s.limit
    done_at = None
    trace = Trace(w, s)
    while w.t < limit:
        cmd = c.act(ob)
        w.command(**cmd)
        ob = w.step()
        trace.step(w, c)
        if verbose and w.tick % 500 == 0:
            x, y, yaw = w.base_pose()
            print(f"   t {w.t:6.1f}  true ({x:.2f} {y:.2f} {math.degrees(yaw):.0f})  est ({c.loc.x:.2f} {c.loc.y:.2f} {math.degrees(c.loc.th):.0f})  q {np.round(ob.q, 2)} grip {ob.grip:.3f} {c.log[-1:] }", flush=True)
        if c.result is not None:
            done_at = w.t
            for _ in range(int(1.5 / CTRL_DT)):             # let things settle before judging
                w.command(q=c.q_cmd)
                ob = w.step()
            break
        if judge.unsafe(w) or not np.all(np.isfinite(w.d.qpos)):
            break
    verdict = judge.evaluate(s, w, ctx)
    mon = w.mon
    x, y, yaw = w.base_pose()
    loc_err = math.hypot(x - c.loc.x, y - c.loc.y)
    rec = dict(valid=verdict["valid"], robot=robot_name, scenario=s.name, cls=s.cls, home=home_key, seed=seed, ok=verdict["ok"], why=verdict["why"], safety_failures=verdict["failures"],
               claimed=c.result, t=round(done_at if done_at is not None else w.t, 1), timed_out=done_at is None and not judge.unsafe(w), log=c.log, bumps=mon["bumps"], max_bump=round(mon["max_bump"], 1), person_N=round(mon["max_person"], 1),
               tilt=round(mon["max_tilt"], 2), heat=round(mon["peak_heat"], 2), energy_Wh=round(w.energy / 3600, 3), loc_err_cm=round(loc_err * 100, 1), wall=round(time.time() - t0, 1), where=ctx.get("where"), dock_err=ctx.get("dock_err"),
               provenance=ident, diagnostics=trace.result(w, c), policy=policy)
    rec["collision_free"] = clear(rec)
    if record:
        rec["_frames"] = (b, w.record)
    return rec
