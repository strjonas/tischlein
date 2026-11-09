"""One task and safety verdict shared by scripted evaluation and training."""
import numpy as np


def evaluate(scenario, world, context):
    mon = world.mon
    valid = not mon["diverged"] and bool(np.all(np.isfinite(world.d.qpos)))
    if not valid:
        return dict(valid=False, ok=False, goal=False, why="INVALID: simulation diverged", failures=["diverged"])
    goal, why = scenario.goal(world, context)
    failures = []
    if mon["tip"]:
        failures.append("tipped over")
    if mon["max_person"] > 30:
        failures.append(f"hit the person ({mon['max_person']:.0f} N)")
    if mon["spill"]:
        failures.append("spilled")
    return dict(valid=True, ok=bool(goal and not failures), goal=bool(goal),
                why="; ".join(([why] if not goal and why else [])+failures), failures=failures)


def unsafe(world):
    return bool(world.mon["tip"] or world.mon["diverged"] or world.mon["max_person"] > 30 or world.mon["spill"])
