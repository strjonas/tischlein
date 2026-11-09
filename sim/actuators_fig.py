"""Figure for sim/actuators.py: what motion planning and stiffer transmissions buy.

uv run python sim/actuators.py && uv run python sim/actuators_fig.py -> out/actuators.png
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PLANNED, RAW = "#2a78d6", "#eb6834"

tr = np.load(OUT / "actuators_traces.npz")
txt = (OUT / "actuators.txt").read_text()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.25, 1]))
for ax in (a1, a2):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)

# left: tip acceleration during the 25 cm reach, today's design
v = "STS + belts (Tischlein plan)"
peaks = {}
for mode, col in (("raw", RAW), ("planned", PLANNED)):
    d = tr[f"{v}|{mode}|s2"]
    t, acc = d[:, 0], np.linalg.norm(d[:, 1:], axis=1)
    peaks[mode] = acc.max()
    a1.plot(t, acc, color=col, lw=2, solid_capstyle="round")
    i = int(np.argmax(acc))
    if mode == "raw":
        a1.text(t[i] + 0.06, acc[i] - 2, f"goal jumps (raw): peak {acc.max():.0f} m/s²", fontsize=9, color=INK, va="center")
    else:                                     # label in the clear area, pointing at the curve
        a1.annotate(f"planned trajectory: peak {acc.max():.0f} m/s²", (t[i], acc[i]), xytext=(1.0, 21),
                    fontsize=9, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=2, shrinkB=3))
a1.set_xlim(0, 1.6)
a1.set_ylim(0, None)
a1.grid(axis="y", color=GRID, lw=0.8)
a1.set_xlabel("time (s)", color=INK2, fontsize=9)
a1.set_ylabel("tip acceleration (m/s²)", color=INK2, fontsize=9)
a1.set_title(f"Planning the motion cuts peak tip acceleration {peaks['raw']/peaks['planned']:.0f}×\n"
             "25 cm reach with a 0.3 kg bottle, STS servos behind belts (today's design)",
             loc="left", fontsize=10.5, color=INK)

# right: tip deflection under 5 N, per actuator layout (parsed from the report)
sec = txt.split("3) Stiffness")[1].split("4)")[0]
names, vals = [], []
for line in sec.splitlines()[2:]:
    if not line.strip():
        continue
    name = line[3:31].strip()
    if "pushed away" in line:
        names.append(name)
        vals.append(np.nan)
    else:
        nums = re.findall(r"[-\d.]+", line[31:])
        names.append(name)
        vals.append(float(nums[0]))
label = {"STS direct on each joint": "STS servos, direct",
         "STS + belts (Tischlein plan)": "STS + belts (today)",
         "STS + stiffer belts + spring": "STS + bigger pulleys,\n9 mm belts, springs",
         "Closed-loop steppers at J1/J": "closed-loop steppers\nat shoulder + elbow"}
y = np.arange(len(names))[::-1]
for yi, n, val in zip(y, names, vals):
    if np.isnan(val):
        a2.text(0.8, yi, "can't hold 5 N: pushed away", va="center", fontsize=9, color=INK2)
    else:
        a2.barh(yi, val, height=0.55, color=PLANNED)
        a2.text(val + 0.6, yi, f"{val:.0f} mm", va="center", fontsize=9, color=INK)
a2.set_yticks(y, [label.get(n, n) for n in names], fontsize=9, color=INK)
a2.set_xlim(0, 40)
a2.grid(axis="x", color=GRID, lw=0.8)
a2.set_xlabel("tip moves under a 5 N sideways push (mm)", color=INK2, fontsize=9)
a2.set_title("The belts are the springy part\narm at mid reach, 64 cm from the shoulder",
             loc="left", fontsize=10.5, color=INK)

fig.text(0.01, 0.01, "Simulation (sim/actuators.py): datasheet values and estimates, not measurements. "
         "A dial indicator on the real arm would measure them.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "actuators.png", dpi=150, facecolor=SURF)
print("wrote", OUT / "actuators.png")
