"""Collision envelopes for components omitted by the initial arena.

Dimensions are explicit engineering assumptions, not manufacturer CAD. Nested
telescope shells have clearance and overlap; no whole-arm collision exclusions.
Component mass is allocated from the existing axis/base budgets, not added twice.
"""
from __future__ import annotations

import numpy as np


def bounds(gs):
    lo, hi = [], []
    for g in gs:
        if g[0] == "box":
            lo.append(g[1]); hi.append(g[2])
        elif g[0] == "cyl":
            lo.append(np.minimum(g[1], g[2])-g[3])
            hi.append(np.maximum(g[1], g[2])+g[3])
    return np.min(lo, axis=0), np.max(hi, axis=0)


def inertia_parts(mass, gs):
    """Uniform envelope-based inertia; retains actual length and parallel-axis terms."""
    pieces, volumes = [], []
    for g in gs:
        low, high = bounds([g])
        size = np.maximum(high-low, 0.002)
        pieces.append(((low+high)/2, size))
        volumes.append(np.prod(size))
    total = sum(volumes)
    return [(mass*v/total, c, s) for (c, s), v in zip(pieces, volumes)]


def telescope_shell(axis, stage):
    """stage=-1: outer fixed sleeve; stage=0..N-1: nested moving tubes."""
    k = int(np.argmax(np.abs(axis.axis)))
    sign = np.sign(axis.axis[k])
    lo, hi = bounds(axis.geoms[:1])
    original = hi[k]-lo[k]
    length = max(original, (axis.hi-axis.lo)/axis.stages+0.06)
    # Keep the hand-side edge unchanged: added overlap extends toward the root.
    if sign > 0:
        lo[k] = hi[k]-length
    else:
        hi[k] = lo[k]+length
    transverse = [j for j in range(3) if j != k]
    for j in transverse:
        extra = 0.003*(axis.stages-stage)
        lo[j] -= extra; hi[j] += extra
    thickness = 0.002
    out = []
    for j in transverse:
        for side in (0, 1):
            a, b = lo.copy(), hi.copy()
            if side:
                a[j] = b[j]-thickness
            else:
                b[j] = a[j]+thickness
            out.append(("box", tuple(a), tuple(b)))
    return out


def axis_geoms(axis, following=None):
    gs = list(axis.geoms)
    if not gs:
        # Servo/turntable casing, allocated within axis.mass. Deliberately visible
        # to both environment and nonadjacent robot links.
        size = 0.026 if axis.force >= 5 else 0.019
        if axis.name == "turn":
            # Annular carriage bearing around the mast, rather than a solid
            # block occupying the same material as that mast.
            for j in (0, 1):
                for sign in (-1, 1):
                    low, high = np.array([-.065, -.065, -.022]), np.array([.065, .065, .022])
                    low[j], high[j] = (0.048, .065) if sign > 0 else (-.065, -.048)
                    gs.append(("box", tuple(low), tuple(high)))
        else:
            gs.append(("box", (-size, -size, -size), (size, size, size)))
    if following is not None and np.linalg.norm(following.pos) > 0.005:
        p = np.array(following.pos, float)
        # Avoid inventing a second beam where a specified beam already exists.
        low, high = bounds(gs)
        if np.any(p < low-0.005) or np.any(p > high+0.005):
            start = p/np.linalg.norm(p)*.050 if axis.name == "turn" else np.zeros(3)
            gs.append(("cyl", tuple(start), tuple(p), 0.010))
    return gs


def camera_geoms(camera):
    p = np.array(camera.pos, float)
    look = np.array(camera.look, float)
    look /= np.linalg.norm(look)
    # Lens is in front of the case, so visibility rays do not start inside it.
    return [("cyl", tuple(p-0.036*look), tuple(p-0.004*look), 0.015)]


def packages(base):
    """A battery, compute/controller and power electronics inside the chassis.

    The battery requires 45 mm height; on a thinner platform its housing extends
    above the deck and is included in collisions. Wiring/connector/service space
    and a fabrication drawing still need a physical packaging review.
    """
    x0, x1 = base.x; y0, y1 = base.y; z0, z1 = base.z
    z = max((z0+z1)/2, z0+0.0225)
    return [
        ("battery", 2.0, ((x0+x1)/2-0.045, 0.0, z), (0.15, 0.12, 0.045)),
        ("compute", 0.35, ((x0+x1)/2+0.075, -0.095, z), (0.10, 0.065, 0.030)),
        ("power", 0.30, ((x0+x1)/2+0.075, 0.095, z), (0.10, 0.065, 0.025)),
    ]


def chassis(base):
    """Frame envelope with actual wheel/caster wells, not a box through tyres."""
    rects = [(base.x[0], base.x[1], base.y[0], base.y[1])]
    cuts = [(base.wheel_x-base.wheel_r-.005, base.wheel_x+base.wheel_r+.005,
             sign*base.wheel_y-base.wheel_w/2-.004, sign*base.wheel_y+base.wheel_w/2+.004) for sign in (-1, 1)]
    cuts += [(x-base.trail-base.caster_r-.005, x-base.trail+base.caster_r+.005,
              y-base.caster_w/2-.006, y+base.caster_w/2+.006) for x,y in base.casters]
    for a,b,c,d in cuts:
        remain = []
        for x0,x1,y0,y1 in rects:
            ix0,ix1,iy0,iy1 = max(x0,a),min(x1,b),max(y0,c),min(y1,d)
            if ix0 >= ix1 or iy0 >= iy1:
                remain.append((x0,x1,y0,y1)); continue
            remain += [(x0,ix0,y0,y1),(ix1,x1,y0,y1),(ix0,ix1,y0,iy0),(ix0,ix1,iy1,y1)]
        rects = [r for r in remain if r[1]-r[0] > .001 and r[3]-r[2] > .001]
    return [("box", (x0,y0,base.z[0]), (x1,y1,base.z[1])) for x0,x1,y0,y1 in rects]
