"""Height-coverage figure: what Tischlein v0.2 can reach vs typical household heights.

uv run python figs.py -> out/heights.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import params as P

OUT = Path(__file__).parent / "out"
ROBOT = "#2a78d6"
HOME = "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

tray_lo, tray_hi = P.TRAY_Z_MIN / 1000, P.TRAY_Z_MAX / 1000
tip_hi = (P.TRAY_Z_MAX + P.L2_ABOVE_TRAY - P.PITCH_DROP) / 1000
rows = [
    ("Tray surface (lift range)", tray_lo, tray_hi, ROBOT),
    ("Fingertips (arm, forward or down)", 0.0, tip_hi, ROBOT),
    ("Floor: dropped phone, remote", 0.0, 0.012, HOME),
    ("Gap under bed / sofa", 0.0, 0.15, HOME),
    ("Mattress top", 0.45, 0.65, HOME),
    ("Bedside table", 0.50, 0.75, HOME),
    ("Kitchen counter", 0.88, 0.94, HOME),
    ("Elevator buttons", 0.90, 1.25, HOME),
    ("Under-counter fridge shelves", 0.10, 0.80, HOME),
    ("Fridge-freezer shelves", 0.90, 1.65, HOME),
]

fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURF)
ax.set_facecolor(SURF)
for i, (label, lo, hi, c) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, max(hi - lo, 0.012), left=lo, height=0.56, color=c, edgecolor=SURF, linewidth=2)
    ax.text(-0.03, y, label, ha="right", va="center", fontsize=9, color=INK)
ax.axvline(tip_hi, color=INK2, lw=1, ls=(0, (3, 3)))
ax.text(tip_hi - 0.02, len(rows) - 0.25, f"highest fingertip ≈ {tip_hi:.2f} m", fontsize=8, color=INK2, va="bottom", ha="right")
ax.text(tip_hi - 0.02, 0.42, "top shelf of tall fridge-freezers: just out of reach", fontsize=8, color=INK2, va="bottom", ha="right")
ax.set_xlim(0, 1.75)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_yticks([])
ax.set_xlabel("height above floor [m]", color=INK2, fontsize=9)
ax.tick_params(axis="x", colors=INK2, labelsize=8)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ROBOT, label="Tischlein v0.2 (design values)"), Patch(color=HOME, label="typical household heights")],
          loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8)
ax.set_title("What heights it can serve", loc="left", fontsize=11, color=INK, pad=22)
fig.subplots_adjust(left=0.33, right=0.98, top=0.86, bottom=0.12)
fig.savefig(OUT / "heights.png", dpi=150, facecolor=SURF)
print("->", OUT / "heights.png")
