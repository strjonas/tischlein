"""The arena is a fixed yardstick only if it cannot drift unnoticed. This pins it.

    .venv/bin/python -m arena.lock            # check against LOCK.json
    .venv/bin/python -m arena.lock --write    # after a deliberate change: new version, new baseline

The lock hashes what defines the test: homes, objects, physics settings, scenarios, judge, sensors and the reference
controller. A result only compares with another under the same hash. Robot entries and results are free to change: they are what
the arena measures. A deliberate change means a new arena version; old results stay valid under their own hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco

from . import __version__, assemble, homes

ROOT = Path(__file__).resolve().parent
FIXED = ("homes.py", "objects.py", "robot.py", "geometry.py", "assemble.py", "contacts.py", "scenarios.py", "sim.py", "skills.py", "trial.py", "reach.py", "judge.py", "env.py", "stow.py", "diagnostics.py", "jobs.py", "navigation.py", "motion_planner.py", "rotation_error.py", "provenance.py", "appliance_judge.py", "contact_audit.py")


def digest():
    out = {}
    for f in FIXED:
        out[f] = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16]
    for k, h in homes.HOMES.items():
        b = assemble.build(None, h, objs=[])
        out[f"home_{k}"] = b.digest
    out["PHYS"] = hashlib.sha256(repr(sorted(assemble.PHYS.items())).encode()).hexdigest()[:16]
    out["mujoco"] = mujoco.__version__
    out["version"] = __version__
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    p = ROOT / "LOCK.json"
    d = digest()
    if a.write:
        p.write_text(json.dumps(d, indent=1) + "\n")
        print("written", p)
    else:
        old = json.loads(p.read_text())
        diff = [k for k in sorted(set(d) | set(old)) if d.get(k) != old.get(k)]
        print("arena unchanged" if not diff else f"CHANGED since the lock: {diff}")
        sys.exit(1 if diff else 0)
