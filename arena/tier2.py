"""Run tier 2: robots x scenarios x seeds, in parallel, appending to a log so a run can stop and resume.

    .venv/bin/python -m arena.tier2 --robots stretch8 revolution_low --scenarios floor_phone --homes ABCD --seeds 8
    .venv/bin/python -m arena.tier2 --report

Every home in a trial list is cycled through with the seeds. A trial's record is in results/tier2.jsonl. Invalid trials
(numerical divergence) are logged and excluded from the rates; their share is reported, because a high share means the
number means little.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from . import trial, zoo
from .scenarios import SCENARIOS

OUT = Path(__file__).resolve().parent / "results" / "v2"
LOG = OUT / "skills.jsonl"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def one(job):
    try:
        return trial.run(job["robot"], job["scenario"], job["home"], job["seed"])
    except Exception as e:
        return dict(robot=job["robot"], scenario=job["scenario"], home=job["home"], seed=job["seed"], valid=False, ok=False, why=f"CRASH {e!r}"[:200], wall=0)


def run(robots, scenarios, homes, seeds, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    from . import provenance
    from dataclasses import asdict
    current_source = provenance.source()
    robot_hashes = {r: provenance.digest(asdict(zoo.ROBOTS[r]())) for r in robots}
    if LOG.exists():
        for l in LOG.read_text().splitlines():
            r = json.loads(l)
            if (r.get("provenance", {}).get("source_hash") == current_source
                    and r.get("provenance", {}).get("robot_hash") == robot_hashes.get(r["robot"])):
                done.add((r["robot"], r["scenario"], r["home"], r["seed"]))
    jobs = [dict(robot=r, scenario=s, home=homes[k % len(homes)], seed=k) for r in robots for s in scenarios for k in range(seeds)]
    jobs = [j for j in jobs if (j["robot"], j["scenario"], j["home"], j["seed"]) not in done]
    print(f"{len(jobs)} trials to run", flush=True)
    t0 = time.time()
    with Pool(workers, maxtasksperchild=8) as p, LOG.open("a") as fh:
        for i, r in enumerate(p.imap_unordered(one, jobs), 1):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)} {time.time() - t0:5.0f} s]", flush=True)


def load(path=None):
    f = path or LOG
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def table(R):
    by = defaultdict(lambda: defaultdict(list))
    for r in R:
        by[r["robot"]][r["scenario"]].append(r)
    return by


def report():
    R = load()
    by = table(R)
    scen = sorted({r["scenario"] for r in R})
    print(f"{'':16s}" + " ".join(f"{s.replace('floor_', 'f_')[:9]:>9s}" for s in scen))
    for name in zoo.ROBOTS:
        if name not in by:
            continue
        row = []
        for s in scen:
            rs = [r for r in by[name].get(s, []) if r.get("valid")]
            row.append(f"{sum(r['ok'] for r in rs):3d}/{len(rs):<3d}   " if rs else "   -     ")
        inv = sum(not r.get("valid") for ss in by[name].values() for r in ss)
        print(f"{name:16s}" + " ".join(f"{x:>9s}" for x in row) + (f"   invalid {inv}" if inv else ""))
    why = Counter()
    for r in R:
        if r.get("valid") and not r["ok"]:
            k = (r.get("log") or [r["why"]])[-1].split("(")[0].split(":")[-1].strip()[:40]
            why[(r["robot"], k)] += 1
    print("\nwhy trials fail:")
    for (rb, k), v in why.most_common(24):
        print(f"  {v:4d}  {rb:16s} {k}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="*", default=list(zoo.ROBOTS))
    ap.add_argument("--scenarios", nargs="*", default=[k for k in SCENARIOS if k.startswith("pick_")] + ["nav_kitchen", "nav_bedside", "nav_hall"])
    ap.add_argument("--homes", default="ABCD")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--log", default=None, help="results file, e.g. tier2_test.jsonl for the held-out homes")
    a = ap.parse_args()
    if a.log:
        LOG = OUT / a.log
    if a.reset and LOG.exists():
        LOG.unlink()
    if not a.report:
        run(a.robots, a.scenarios, a.homes, a.seeds, a.workers)
    report()
