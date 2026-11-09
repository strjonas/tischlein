"""Top view of the door-opening run: how much floor the robot needs next to the fridge.

uv run python sim/door_fig.py -> out/door_space.png
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
import params as P  # noqa: E402
import door as D  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ROBOT, DOOR = "#2a78d6", "#eb6834"

at, T, marks = D.plan(follow=True)
ts = np.linspace(0, T - 0.01, 200)
path = np.array([(at(t)["base_x"], at(t)["base_y"]) for t in ts])

fig, ax = plt.subplots(figsize=(6.4, 6.0), facecolor=SURF)
ax.set_facecolor(SURF)
fx = D.FX
ax.add_patch(plt.Rectangle((fx, -0.30), 0.65, 0.60, color="#d9d8d3", zorder=1))
ax.text(fx + 0.33, 0.0, "fridge", ha="center", va="center", fontsize=9, color=INK2)
for deg, a in ((0, 0.35), (45, 0.55), (100, 1.0)):
    th = math.radians(deg)
    hx, hy = D.HINGE
    end = (hx - 0.6 * math.sin(th), hy - 0.6 * math.cos(th))
    ax.plot([hx, end[0]], [hy, end[1]], color=DOOR, lw=3, alpha=a, solid_capstyle="round", zorder=3)
    ax.text(end[0] - 0.02, end[1] + (0.03 if deg else -0.05), f"{deg}°", fontsize=8, color=INK2, ha="right")


def footprint(x, y, alpha, label=None):
    w = P.TRACK / 2000 + P.WHEEL_W / 2000
    ax.add_patch(plt.Rectangle((x + P.BASE_X_REAR / 1000, y - w), (P.FRONT_CASTER_X + 50 - P.BASE_X_REAR) / 1000, 2 * w,
                               fill=False, ec=ROBOT, lw=1.5, alpha=alpha, zorder=4))
    if label:
        ax.text(x - 0.05, y - w - 0.05, label, fontsize=8, color=INK2, ha="center", va="top")


footprint(*path[0], 0.5, "start")
footprint(*path[-1], 1.0, "end")
ax.plot(path[:, 0], path[:, 1], color=ROBOT, lw=2, zorder=5, label="robot centre path")
dx = path[-1] - path[0]
ax.annotate("", xy=path[-1], xytext=path[0], arrowprops=dict(arrowstyle="-", color=ROBOT, lw=0))
ax.text(-0.95, 0.95, f"the base travels {abs(dx[0])*100:.0f} cm back\nand {dx[1]*100:.0f} cm toward the hinge side",
        fontsize=9, color=INK, va="top")
ax.set_xlim(-1.5, 1.45)
ax.set_ylim(-0.75, 1.2)
ax.set_aspect("equal")
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(colors=INK2, labelsize=8)
ax.set_xlabel("x [m]  (fridge front at 0.72)", color=INK2, fontsize=8)
ax.set_title("Opening a fridge door with one arm: floor it needs (top view)", loc="left", fontsize=10, color=INK)
ax.legend(loc="lower left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "out" / "door_space.png", dpi=150, facecolor=SURF)
print("-> out/door_space.png")
