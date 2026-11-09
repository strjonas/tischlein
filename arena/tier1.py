"""Run tier 1: every robot, every contact, every home. One row per contact.

    .venv/bin/python -m arena.tier1 [--robots a b] [--homes ABCD] [--workers 9]   # -> arena/results/tier1.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import assemble, contacts, homes, objects, reach, zoo

OUT = Path(__file__).resolve().parent / "results" / "v2"
OBJS = [(k, k) for k in objects.CATALOG]


def evaluate(robot, home, seed=0, seeds=24):
    """Every contact of one home for one robot object: {key: result}."""
    b = assemble.build(robot, home, objs=OBJS)
    T = reach.Tier1(b)
    C = contacts.all_contacts(home, b.world, b.robot.kits)
    rng = np.random.default_rng(seed)
    out = {}
    for i, c in enumerate(C):
        r = T.solve(c, seeds=seeds, rng=rng)
        out[f"{i}|{c.cls}|{c.name}"] = dict(cls=c.cls, kind=c.kind, group=c.group or c.name, ok=bool(r["ok"]), why=r.get("why", ""), detail=r.get("detail", ""), load=r.get("load"), margin=r.get("margin"))
    return out, T.covered()


def merge(a, b):
    """A test counts as passed if any tool of the robot passes it."""
    if a is None:
        return b
    for k, v in b.items():
        if k in a and not a[k]["ok"] and v["ok"]:
            a[k] = v
    return a


def run_one(job):
    name, hk = job
    t0 = time.time()
    home = homes.HOMES[hk]
    robot = zoo.ROBOTS[name]()
    res, area = evaluate(robot, home)
    for f in zoo.VARIANTS.get(name, []):
        res = merge(res, evaluate(f(), home)[0])
    from . import provenance
    return dict(robot=name, home=hk, area=area, res=res, wall=round(time.time() - t0, 1), provenance=provenance.identity(robot,"tier1",hk,0))


def score(res):
    """Group alternatives, then share of tests passed per need class."""
    g = defaultdict(lambda: [False, "", ""])
    for k, v in res.items():
        key = (v["cls"], v["group"])
        if v["ok"]:
            g[key][0] = True
        elif not g[key][1]:
            g[key][1], g[key][2] = v["why"], v["detail"]
    cls = defaultdict(lambda: [0, 0])
    why = Counter()
    for (c, _), (ok, w, _) in g.items():
        cls[c][1] += 1
        cls[c][0] += ok
        if not ok:
            why[w] += 1
    return {c: v[0] / v[1] for c, v in cls.items()}, {c: v for c, v in cls.items()}, why


def reasons(R, robot, top=6):
    """The most common blocked tests of a robot, for reading what the design lacks."""
    seen = Counter()
    for r in R:
        if r["robot"] != robot:
            continue
        g = defaultdict(list)
        for k, v in r["res"].items():
            g[(v["cls"], v["group"])].append(v)
        for key, vs in g.items():
            if not any(v["ok"] for v in vs):
                seen[(key[0], key[1].split(",")[0][:30], vs[0]["why"], vs[0]["detail"][:34])] += 1
    return seen.most_common(top)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="*", default=list(zoo.ROBOTS))
    ap.add_argument("--homes", default="ABCD")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--merge", action="store_true", help="replace only these robots in tier1.json")
    a = ap.parse_args()
    jobs = [(r, h) for r in a.robots for h in a.homes]
    t0 = time.time()
    with Pool(a.workers) as p:
        R = p.map(run_one, jobs, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.merge and (OUT / "tier1.json").exists():
        R = [r for r in json.loads((OUT / "tier1.json").read_text()) if r["robot"] not in a.robots] + R
    (OUT / "tier1.json").write_text(json.dumps(R))
    print(f"{len(R)} robot-home pairs in {time.time() - t0:.0f} s")
    by = defaultdict(list)
    for r in R:
        by[r["robot"]].append(r)
    classes = ["dropped item", "handover", "drink", "snack, cold food", "hot meal", "long-tail fetch", "clear away", "delivery inward", "household"]
    print(f"{'':16s}" + " ".join(f"{c[:8]:>8s}" for c in classes) + "   why it fails")
    for name, rs in by.items():
        tot = defaultdict(lambda: [0, 0])
        why = Counter()
        for r in rs:
            _, c, w = score(r["res"])
            why += w
            for k, v in c.items():
                tot[k][0] += v[0]
                tot[k][1] += v[1]
        print(f"{name:16s}" + " ".join(f"{tot[c][0] / max(tot[c][1], 1):8.0%}" for c in classes) + "   " + ", ".join(f"{k} {v}" for k, v in why.most_common(5)))
    if a.detail:
        for name in by:
            print(name)
            for k, v in reasons(R, name, 10):
                print("   ", v, k)
