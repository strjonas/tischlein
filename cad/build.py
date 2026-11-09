"""Export print files, a STEP assembly and per-scene GLBs for rendering.

uv run python cad/build.py            # everything
uv run python cad/build.py scenes     # only scene GLBs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from build123d import Color, Compound, Location, export_gltf, export_step, export_stl

import robot as R
import scene as S

OUT = R.ROOT / "out"
PRINT_DIR = R.ROOT / "print"


def export_prints():
    """One STL per printable part, in its link frame. Orientation for printing is noted in print/README."""
    PRINT_DIR.mkdir(exist_ok=True)
    seen = {}
    report = []
    for link, items in R.links().items():
        for label, shape, rgb, printable in items:
            if not printable:
                continue
            n = seen.get(label, 0)
            seen[label] = n + 1
            if n:                          # identical copies: print the count, export once
                continue
            bb = shape.bounding_box()
            centred = Location((-bb.center().X, -bb.center().Y, -bb.min.Z)) * shape
            export_stl(centred, str(PRINT_DIR / f"{label}.stl"), tolerance=0.02, angular_tolerance=0.15)
            report.append(dict(part=label, link=link, size_mm=[round(bb.size.X), round(bb.size.Y), round(bb.size.Z)],
                               volume_cm3=round(shape.volume / 1000, 1),
                               fits_250=bool(max(bb.size.X, bb.size.Y, bb.size.Z) <= 250)))
    for r in report:
        r["qty"] = seen[r["part"]]
    (PRINT_DIR / "parts.json").write_text(json.dumps(report, indent=1))
    return report


def export_assembly_step(q=None):
    parts = R.assembly(q or S.PARK | dict(lift=520))
    export_step(Compound(children=parts), str(OUT / "tischlein_v0_2.step"))


def export_scene(name):
    q, props = S.SCENES[name]()
    parts = R.assembly(q) + props
    comp = Compound(children=parts)
    path = OUT / f"scene_{name}.glb"
    export_gltf(comp, str(path), binary=True, linear_deflection=0.5, angular_deflection=0.3)
    tip = R.fingertip(q)
    return dict(scene=name, q={k: round(v, 1) for k, v in q.items()}, fingertip=[round(tip.X), round(tip.Y), round(tip.Z)])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    what = sys.argv[1:] or ["prints", "step", "scenes"]
    if "prints" in what:
        rep = export_prints()
        vol = sum(r["volume_cm3"] * r["qty"] for r in rep)
        print(f"{len(rep)} printable part types, {vol:.0f} cm3 total (~{vol*1.27/1000*1.15:.2f} kg PETG at 100%; real prints are lighter)")
        for r in rep:
            flag = "" if r["fits_250"] else "  <-- larger than the 250 mm bed"
            print(f"  {r['qty']}x {r['part']:28s} {r['size_mm']} {r['volume_cm3']:6.1f} cm3{flag}")
    if "step" in what:
        export_assembly_step()
        print("STEP ->", OUT / "tischlein_v0_2.step")
    if "scenes" in what:
        info = [export_scene(n) for n in S.SCENES]
        (OUT / "scenes.json").write_text(json.dumps(info, indent=1))
        for i in info:
            print(i)
