"""Pictures of the world and of runs. Looking at frames catches most setup bugs before they become numbers."""
from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def camera(look, dist, azim, elev):
    c = mujoco.MjvCamera()
    c.lookat[:] = look
    c.distance, c.azimuth, c.elevation = dist, azim, elev
    return c


def shot(m, d, cam, w=800, h=500, hide_ceiling=True):
    with mujoco.Renderer(m, h, w) as r:
        r.update_scene(d, cam)
        return r.render().copy()


def sheet(frames, cols=3, labels=None, path=None):
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * h), "white")
    for i, f in enumerate(frames):
        im = Image.fromarray(f)
        if labels:
            ImageDraw.Draw(im).text((8, 6), labels[i], fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    if path:
        out.save(path)
    return out
