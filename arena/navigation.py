"""Height-aware configuration-space maps, including swept in-place turns.

Each horizontal height band gets the robot material that actually occupies it.
This avoids treating an entire tall robot as a solid extruded footprint. The
bands and raster still approximate geometry; dynamic collision remains active.
"""
import math
import mujoco
import numpy as np
from scipy import ndimage, signal

EDGES = (0, .12, .20, .40, .60, .80, 1.0, 1.2, 1.4, 1.6, 2.05)


def aabb(m, d, g):
    size = m.geom_size[g].copy()
    kind = m.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        size = np.array([size[0], size[0], size[0]+size[1]])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        size = np.array([size[0], size[0], size[1]])
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        size[:] = size[0]
    radius = np.abs(d.geom_xmat[g].reshape(3, 3))@size
    return d.geom_xpos[g]-radius, d.geom_xpos[g]+radius


def maps(T):
    m, d = T.m, T.d
    from .reach import CELL, HEADINGS
    T._set(0, 0, 0, T.home_q)
    # Include payload material when constructing a travelling map.
    robot = T.g_rob.copy()
    if getattr(T, "held", None):
        name = T.held[0]
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        robot |= m.geom_bodyid == body
    rob_boxes = [aabb(m, d, g) for g in np.nonzero(robot)[0]]
    shape = T.low.shape
    layers = []
    for z0, z1 in zip(EDGES[:-1], EDGES[1:]):
        parts = [(lo[:2], hi[:2]) for lo, hi in rob_boxes if lo[2] < z1-1e-4 and hi[2] > z0+1e-4]
        if not parts:
            continue
        occupied = np.zeros(shape, bool)
        for g, name in enumerate(T.names):
            if robot[g] or T.g_obj[g] or not m.geom_contype[g] or name == "floor":
                continue
            if name.startswith(("thr_", "ramp_")) and (T.climbs() or "ramps" in T.r.kits):
                continue
            if name.startswith(("rug", "doormat")):
                continue
            low, high = aabb(m, d, g)
            if low[2] >= z1-1e-4 or high[2] <= z0+1e-4:
                continue
            a = np.floor((low[:2]+(.2, 0))/CELL).astype(int)
            b = np.ceil((high[:2]+(.2, 0))/CELL).astype(int)
            occupied[max(0,a[0]):min(shape[0],b[0]), max(0,a[1]):min(shape[1],b[1])] = True
        if occupied.any():
            layers.append((occupied, parts))
    radius = int(math.ceil(max(np.max(np.abs(np.r_[lo[:2], hi[:2]])) for lo, hi in rob_boxes)/CELL))+2
    gx, gy = np.mgrid[-radius:radius+1, -radius:radius+1]*CELL
    def footprint(parts, theta):
        c, s = math.cos(theta), math.sin(theta)
        x, y = c*gx+s*gy, -s*gx+c*gy
        mask = np.zeros(gx.shape, bool)
        for low, high in parts:
            mask |= (x >= low[0]-.01)&(x <= high[0]+.01)&(y >= low[1]-.01)&(y <= high[1]+.01)
        # Cross-correlation of occupied world points and robot material.
        return mask[::-1, ::-1]
    def clear(theta):
        out = np.ones(shape, bool)
        for occupied, parts in layers:
            kernel = footprint(parts, theta)
            hit = signal.fftconvolve(occupied, kernel, mode="same") > .5
            out &= ~hit
        return out
    free, turning = [], []
    step = 2*math.pi/HEADINGS
    for h in range(HEADINGS):
        maps_ = [clear((h+t)*step) for t in (0, .25, .5, .75, 1)]
        free.append(maps_[0])
        turning.append(np.logical_and.reduce(maps_))
    return np.array(free), np.array(turning)
