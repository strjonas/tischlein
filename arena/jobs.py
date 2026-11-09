"""Complete-job competition, with resumable logs tied to code, robot and policy.

    python -m arena.jobs --robots revolution_low stretch8 orthodox_v03 --homes ABCD --seeds 2
    python -m arena.jobs --robots all --homes EFGH --seeds 1

Every seed runs in every requested home. Unreachable starts count as failures;
there is no multiplication of pickup rates into unobserved complete-job scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from . import provenance, trial, zoo
from .scenarios import FULL_JOBS
from .tier2 import wilson

OUT = Path(__file__).resolve().parent/"results"/"v2"
CORE = ["floor_tablet_box", "floor_remote", "bedside_pill_bottle", "fridge_bottle", "worktop_meal_box", "clear_mug", "microwave_meal"]


def one(job):
    try:
        result = trial.run(**job["args"])
    except Exception as exc:
        result = dict(robot=job["args"]["robot_name"], scenario=job["args"]["scenario"], home=job["args"]["home_key"],
                      seed=job["args"]["seed"], policy=job["args"]["policy"], valid=False, ok=False,
                      why=f"CRASH: {type(exc).__name__}: {exc}")
    result["run_id"] = job["run_id"]
    result["requested_source_hash"] = job["source"]
    if result.get("provenance", {}).get("source_hash", job["source"]) != job["source"] or provenance.source() != job["source"]:
        result.update(valid=False, ok=False, why="source changed during experiment; rerun this batch")
    return result


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def report(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load(path)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["robot"], row.get("policy", "improved"), row.get("requested_source_hash"))].append(row)
    lines = ["| Robot | Policy | Complete jobs | Without >40 N unintended contact | 95% interval | Invalid |", "|---|---|---:|---:|---:|---:|"]
    for (robot, policy, source), rs in sorted(groups.items()):
        # Invalid simulations do not become successes; retain the planned denominator.
        k, n = sum(r["ok"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        clean = sum(r["ok"] and r.get("collision_free", False) for r in rs)
        lines.append(f"| {robot} | {policy} | {k}/{n} ({k/n:.0%}) | {clean}/{n} | {lo:.0%}–{hi:.0%} | {sum(not r['valid'] for r in rs)} |")
        print(robot, policy, f"{k}/{n}", "failures", dict(Counter(failure(r) for r in rs if not r["ok"])), flush=True)
    path.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)
    return rows


def failure(row):
    if not row.get("valid", True):
        return "invalid simulation"
    if row.get("safety_failures"):
        return " + ".join(row["safety_failures"])
    d = row.get("diagnostics", {})
    return d.get("controller_failure") or ("timeout: "+d.get("terminal_stage", "unknown") if row.get("timed_out") else d.get("terminal_stage", row.get("why", "unknown")))


def run(robots, scenarios, homes, seeds, policies, path, workers=6, seed_offset=0):
    source = provenance.source()
    robot_hashes = {r: provenance.digest(__import__("dataclasses").asdict(zoo.ROBOTS[r]())) for r in robots}
    done = {row.get("run_id") for row in load(path) if row.get("valid")}
    jobs = []
    for robot in robots:
        for scenario in scenarios:
            for home in homes:
                for k in range(seeds):
                    for policy in policies:
                        args = dict(robot_name=robot, scenario=scenario, home_key=home, seed=seed_offset+k, policy=policy)
                        run_id = provenance.digest(dict(source=source, robot=robot_hashes[robot], **args))
                        if run_id not in done:
                            jobs.append(dict(args=args, run_id=run_id, source=source))
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} complete jobs, source {source[:12]}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as file:
        # map preserves paired order in the log; workers simulate independently.
        for i, result in enumerate(pool.map(one, jobs, chunksize=1), 1):
            file.write(json.dumps(result, default=str)+"\n"); file.flush()
            if i % 12 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} in {time.time()-t0:.0f}s", flush=True)
    return report(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=["revolution_low", "stretch8", "orthodox_v03"])
    ap.add_argument("--scenarios", nargs="+", default=CORE)
    ap.add_argument("--homes", default="ABCD")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=100)
    ap.add_argument("--policies", nargs="+", default=["improved"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log", default="full_jobs.jsonl")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    path = OUT/a.log
    if a.report:
        report(path)
    else:
        run(list(zoo.ROBOTS) if a.robots == ["all"] else a.robots,
            FULL_JOBS if a.scenarios == ["all"] else a.scenarios, a.homes, a.seeds, a.policies, path, a.workers, a.seed_offset)


if __name__ == "__main__":
    main()
