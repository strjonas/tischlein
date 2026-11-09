"""Summarise results/*.json into a markdown table and out/cap_zero_shot.png.

    third_party/EgoGym/.venv/bin/python experiments/cap_zero_shot/summarize.py [--tag v2]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ROBOTS = ["cap", "tisch5", "tisch6", "tisch6sel", "tisch6fixed", "capnear", "tisch6near", "benchlift", "benchnolift"]
LABEL = {"cap": "floating gripper (ideal)", "tisch5": "Tischlein, today's wrist", "tisch6": "+ roll, always followed",
         "tisch6sel": "+ roll, used selectively",
         "tisch6fixed": "+ roll, base parked", "capnear": "ideal gripper, starting close",
         "tisch6near": "full robot, starting close", "benchlift": "bench: arm + lift, no base",
         "benchnolift": "bench: arm only, lift fixed"}
COLOR = {"cap": "#8a8984", "tisch5": "#2a78d6", "tisch6": "#eb6834", "tisch6sel": "#1baf7a",
         "tisch6fixed": "#eda100", "capnear": "#8a8984", "tisch6near": "#2a78d6", "benchlift": "#1baf7a",
         "benchnolift": "#eda100"}
TASKS = ["pick", "open", "close"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def load(tag):
    out = {}
    for f in glob.glob(os.path.join(HERE, "results", f"*_{tag}.json")):
        d = json.load(open(f))
        out[(d["task"], d["robot"])] = d
    return out


def table(res):
    rows = ["| Task | Embodiment | Success | 95% CI | Episodes | Mean IK position error | Episodes with IK error > 2 cm | Mean IK angle error |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in TASKS:
        for r in ROBOTS:
            d = res.get((t, r))
            if not d:
                continue
            ep = d["episodes_detail"]
            k, n = sum(e["success"] for e in ep), len(ep)
            p, lo, hi = wilson(k, n)
            if "ik_pos_mean" in ep[0]:
                pos = f"{np.mean([e['ik_pos_mean'] for e in ep]) * 100:.1f} cm"
                big = f"{np.mean([e['ik_pos_max'] > 0.02 for e in ep]) * 100:.0f}%"
                ang = f"{np.mean([e['ik_ang_mean'] for e in ep]):.1f}°"
            else:
                pos = big = ang = "—"
            rows.append(f"| {t} | {LABEL[r]} | **{p * 100:.0f}%** | {lo * 100:.0f}–{hi * 100:.0f}% | {n} | {pos} | {big} | {ang} |")
    return "\n".join(rows)


def chart(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    present = [r for r in ROBOTS if any((t, r) in res for t in TASKS)]
    w = 0.8 / max(1, len(present))
    for i, t in enumerate(TASKS):
        for j, r in enumerate(present):
            d = res.get((t, r))
            if not d:
                continue
            ep = d["episodes_detail"]
            p, lo, hi = wilson(sum(e["success"] for e in ep), len(ep))
            x = i - 0.4 + w * (j + 0.5)
            ax.bar(x, p * 100, width=w - 0.02, color=COLOR[r], edgecolor=surface, linewidth=2, zorder=2)
            ax.plot([x, x], [lo * 100, hi * 100], color=ink2, linewidth=1.2, zorder=3)
            ax.text(x, hi * 100 + 2, f"{p * 100:.0f}", ha="center", va="bottom", fontsize=8, color=ink)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([t.capitalize() for t in TASKS], color=ink, fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_ylabel("success, % of episodes", color=ink2, fontsize=9)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[r]) for r in present]
    ax.legend(handles, [LABEL[r] for r in present], frameon=False, fontsize=8, labelcolor=ink,
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_title("Pretrained CAP skills, zero-shot in EgoGym: ideal gripper vs Tischlein's kinematics",
                 color=ink, fontsize=10, loc="left")
    fig.text(0.01, 0.005, "whiskers: 95% Wilson interval", color=ink2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


ANCHOR = [("v3_frozen", "exact point (perfect depth)"), ("v3_iso05", "0.5 cm error"),
          ("v3_iso1", "1 cm error: SE, tap tracked over 4 wrist moves*"), ("v3_plane", "SE, no depth: tap ray meets the table plane"),
          ("v3_ray8", "SE, monocular depth net (8% of distance)"), ("v3_iso2", "2 cm error"), ("v3_iso4", "4 cm error")]


def anchor_chart(path):
    """Pick success on the bench rig vs how the contact point was obtained (docs/cameras.md)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid, bar = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
    rows = []
    for tag, label in ANCHOR:
        f = os.path.join(HERE, "results", f"pick_benchlift_{tag}.json")
        if os.path.exists(f):
            ep = json.load(open(f))["episodes_detail"]
            rows.append((label, *wilson(sum(e["success"] for e in ep), len(ep))))
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for i, (label, pr, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, pr * 100, height=0.62, color=bar, edgecolor=surface, linewidth=2, zorder=2)
        ax.plot([lo * 100, hi * 100], [y, y], color=ink2, linewidth=1.2, zorder=3)
        ax.text(hi * 100 + 1.5, y, f"{pr * 100:.0f}%", va="center", fontsize=8, color=ink)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8, color=ink)
    ax.set_xlim(0, 100)
    ax.set_xlabel("pick success, % of 150 episodes (bench rig: arm + lift, no base)", color=ink2, fontsize=8)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.xaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Pretrained pick vs how accurately the tapped contact point is known", color=ink, fontsize=10, loc="left")
    fig.text(0.01, 0.01, "whiskers: 95% Wilson interval.  * median when the point is tracked (22 of 40 scenes; "
             "triangulate.py); otherwise the table-plane fallback", color=ink2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()
    res = load(a.tag)
    md = table(res)
    print(md)
    chart(res, os.path.join(ROOT, "out", "cap_zero_shot.png"))
    print("chart -> out/cap_zero_shot.png")
    anchor_chart(os.path.join(ROOT, "out", "cap_anchor_error.png"))
    print("chart -> out/cap_anchor_error.png")
