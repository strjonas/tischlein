"""Contents-aware appliance credit; this remains a timer proxy, not heat transfer."""
import numpy as np

from .navigation import aabb


def accumulate(world, dt):
    """Count only time an entire object is inside the closed, running oven."""
    if not world.state["mw_running"] or not world.d.eq_active[world.eq["mw_door"]]:
        return
    low, high = map(np.asarray, world.b.world.regions["mw_cavity"])
    times = world.state.setdefault("mw_contents_seconds", {})
    for name, obj in world.obj.items():
        geoms = np.nonzero(world.m.geom_bodyid == obj["b"])[0]
        boxes = [aabb(world.m, world.d, int(g)) for g in geoms]
        if boxes and all(np.all(a >= low-.003) and np.all(b <= high+.003) for a,b in boxes):
            times[name] = times.get(name, 0.0)+dt


def heated(world, name, minimum_seconds=10.0):
    return world.state.get("mw_contents_seconds", {}).get(name, 0.0) >= minimum_seconds
