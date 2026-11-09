"""Content-addressed identities: changed code/configuration cannot resume old results."""
from dataclasses import asdict
import hashlib
import json
import importlib.metadata
from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parent
FILES = ("__init__.py", "provenance.py", "homes.py", "objects.py", "robot.py", "geometry.py", "assemble.py", "contacts.py", "reach.py",
         "sim.py", "skills.py", "scenarios.py", "trial.py", "judge.py", "env.py", "stow.py", "diagnostics.py", "jobs.py", "motion_planner.py", "navigation.py", "rotation_error.py", "appliance_judge.py", "contact_audit.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source():
    return digest({f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES})


def identity(robot, scenario, home, seed, policy="improved", **options):
    return dict(source_hash=source(), robot_hash=digest(asdict(robot)), mujoco=mujoco.__version__,
                libraries={k: importlib.metadata.version(k) for k in ("numpy", "scipy")},
                scenario=scenario, home=home, seed=seed, policy=policy, options=options)
