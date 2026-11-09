# Arena: a simulated flat for complete household jobs

An open MuJoCo benchmark for mobile manipulators that fetch and carry for a person in bed. One fixed world, a fixed judge, and robots as plug-in data entries. A robot starts at its dock and has to find an object, grasp it, carry it, and release it on the right support; the physics decides whether the job is done.

## What it tests

- **The world.** Eight parameterised homes (`homes.py`): doors 0.70–0.90 m wide, mattress heights 0.45–0.72 m, thresholds up to 20 mm, tile, laminate and carpet floors, fridges with 25–55 N door seals and different hinge sides, a microwave and a dishwasher. A person lies in the bed. Homes A–D are for development, E–H are held out.
- **The jobs.** Seven standard complete jobs (fetch from the floor, bedside, fridge or worktop; clear a mug; heat a meal in the microwave) and a catalogue of 22. See [FULL_JOBS.md](FULL_JOBS.md) for the goal definitions.
- **The robot.** A `Robot` entry in [zoo.py](zoo.py) is a base, a kinematic chain with motor limits, backlash and flex, a hand, cameras and optional appliance aids. Self-collision, component envelopes, per-axis force and speed limits, backlash, flex, and sensor noise and latency are on. Four reference entries ship with the repository.
- **The controller.** A scripted controller sees only sensors: a mapped floor plan, camera detections, lidar, wheel odometry and gripper state. [judge.py](judge.py) and the physics see the truth. Every trial logs why it ended.
- **The physics.** [tests/test_physics.py](tests/test_physics.py) checks the contact settings against textbook answers (friction slip angle, tipping force, pinch holding force, wheel step climbing, penetration) at three timesteps. 27 of 27 pass; the table is written to `results/v2/physics_validation.json`.

## Run it

Python 3.13 and [uv](https://docs.astral.sh/uv/), from the repository root:

```bash
uv sync --locked
uv run python -m arena.tests.test_v2              # behavioural regressions
uv run python -m arena.tests.test_contact_audit
uv run python -m arena.tests.test_appliance_judge
uv run python -m arena.tests.test_physics         # 27 physics checks
uv run python -m arena.lock                       # the pinned world is unchanged
uv run python -m arena.jobs --robots revolution_low stretch8 orthodox_v03 \
    --homes ABCD --seeds 1 --seed-offset 0 --workers 6
uv run python -m arena.jobs --report              # tabulate the log again
```

`arena.jobs` appends to `results/v2/full_jobs.jsonl`, resumes interrupted runs, and prints completion counts with 95 % Wilson intervals and failure reasons. Use `--robots all` for every registered entry, `--scenarios all` for all 22 jobs, `--policies reference improved` to compare controller versions, and `--homes EFGH` for the held-out homes.

Diagnostics: `arena.tier1` checks whether a body can reach and hold the load at every contact in every home (milliseconds per test); `arena.tier2` runs single picks. `arena.env` is a reset/step interface over the same world and judge for training, `arena.learn` a small training loop on it, and `arena.laundry` a separate cloth-in-drum scene.

## Add a robot

A new design is a `Robot` entry in [zoo.py](zoo.py), not a new world. Helper constructors for lifts, telescopes and hinge axes are at the top of the file. Specialist skills such as tool changes or second-arm assistance need their own dynamic skills in [skills.py](skills.py); an entry gets no credit for them by listing them.

## Layout

| File | Role |
|---|---|
| `homes.py`, `objects.py` | the world and the object catalogue |
| `robot.py`, `geometry.py`, `zoo.py`, `stow.py` | robot entries, collision envelopes, folded stow poses |
| `assemble.py`, `sim.py` | MJCF assembly and the simulation world with sensors |
| `reach.py`, `contacts.py`, `navigation.py`, `motion_planner.py` | reachability, grasp candidates, base maps, joint-space planning |
| `skills.py`, `scenarios.py` | the controller and the job definitions |
| `judge.py`, `appliance_judge.py`, `contact_audit.py` | goal, safety and contact judging |
| `trial.py`, `jobs.py`, `provenance.py`, `lock.py` | running trials, hash-tagged resumable logs, the pinned arena |
| `render.py` | pictures of the world and of runs |

## Scope

This is an engineering-envelope simulation, not manufacturer CAD: fasteners, cable routing, tolerances and fatigue are outside it. Vision returns abstract detections rather than processing images. Microwave heating is a timer and state proxy; food, liquids, steam and deformable bags are not modelled, and the person is stationary. Force and tilt thresholds are engineering proxies, not safety criteria. The bundled scripted controller is a baseline, so a failed job says as much about the controller as about the robot. Real-world use needs measured friction, stiffness, backlash, camera error and appliance seal forces, which the model exposes as parameters.
