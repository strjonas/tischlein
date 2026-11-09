"""Training in the arena, end to end: a learner improves a skill on the dev homes and is checked on homes it never saw.

    .venv/bin/python -m arena.learn --robot stretch8 --workers 9    # -> arena/results/learn.json

The learner is the cross-entropy method on a few numbers of the grasp skill, which is what fits a CPU: each candidate is
judged on whole trials, with domain randomization on. A neural policy uses arena.env the same way and needs more samples
than this machine gives in an evening; the point here is that the loop closes, under fair rules: training on homes A-D only,
the score from held-out homes E-H, seeds the learner never used.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import trial
from .homes import DEV, TEST
from .tier2 import wilson

OUT = Path(__file__).resolve().parent / "results"
SKILLS = ("pick_tablet_box_floor", "pick_phone_floor", "pick_remote_floor", "pick_bottle_table", "pick_mug_table", "pick_tablet_box_bedside")
P = dict(pre=(0.05, 0.16), look_up=(0.04, 0.24), in_speed=(0.015, 0.09), in_guard=(0.25, 1.5), drop_extra=(-0.006, 0.012))
KEYS = list(P)


def one(j):
    robot, sk, hk, seed, tune = j
    r = trial.run(robot, sk, hk, seed, tune=tune)
    return bool(r["ok"]), bool(r["valid"])


def score(pool, robot, tune, homes, seeds):
    jobs = [(robot, sk, homes[i % len(homes)], s, tune) for i, s in enumerate(seeds) for sk in SKILLS]
    R = pool.map(one, jobs, chunksize=1)
    return sum(a for a, _ in R), sum(b for _, b in R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="stretch8")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args()
    lo = np.array([P[k][0] for k in KEYS])
    hi = np.array([P[k][1] for k in KEYS])
    from .skills import Controller
    default = dict(pre=0.09, look_up=0.12, in_speed=0.04, in_guard=0.6, drop_extra=0.0)
    mu = np.array([default[k] for k in KEYS])
    sig = (hi - lo) * 0.3
    rng = np.random.default_rng(0)
    hist = []
    with Pool(a.workers, maxtasksperchild=16) as pool:
        best, best_v = dict(default), -1
        for g in range(a.gens):
            seeds = [1000 + g * 100 + k for k in range(a.episodes)]           # new seeds each generation: no memorising
            pop = np.clip(rng.normal(mu, sig, (a.pop, len(KEYS))), lo, hi)
            pop[0] = mu
            vals = []
            for x in pop:
                t = dict(zip(KEYS, map(float, x)))
                k, n = score(pool, a.robot, t, DEV, seeds)
                vals.append(k / max(n, 1))
            vals = np.array(vals)
            elite = pop[np.argsort(-vals)[:max(3, a.pop // 4)]]
            mu, sig = elite.mean(0), elite.std(0) + (hi - lo) * 0.03
            hist.append(dict(gen=g, mean=float(vals.mean()), best=float(vals.max()), mu=dict(zip(KEYS, map(float, mu)))))
            print(f"gen {g}: pop mean {vals.mean():.0%}  best {vals.max():.0%}  mean of elite {dict(zip(KEYS, np.round(mu, 3)))}", flush=True)
        tuned = dict(zip(KEYS, map(float, mu)))
        test_seeds = list(range(5000, 5012))
        out = {}
        for name, t, hs in (("reference, dev homes", default, DEV), ("learned, dev homes", tuned, DEV), ("reference, held-out homes", default, TEST), ("learned, held-out homes", tuned, TEST)):
            k, n = score(pool, a.robot, t, hs, test_seeds)
            lo_, hi_ = wilson(k, n)
            out[name] = dict(ok=k, n=n, lo=lo_, hi=hi_)
            print(f"{name:28s} {k}/{n} = {k / max(n, 1):.0%} ({lo_:.0%}-{hi_:.0%})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "learn.json").write_text(json.dumps(dict(robot=a.robot, hist=hist, tuned=tuned, default=default, result=out), indent=1))


if __name__ == "__main__":
    main()
