"""Tischlein arena: one fixed world and fixed tests in which every robot design is run.

Layers, each usable without the ones below it:
  homes.py objects.py   the world, frozen and versioned; no robot-specific anything
  robot.py zoo.py       the plug-in: a robot is a parts list, a kinematic chain and a hand
  assemble.py           world + robot + physics settings -> a MuJoCo model
  reach.py tier1.py     tier 1: can the body get there and hold the load? (milliseconds per test)
  sim.py skills.py      tier 2: real contacts, motors, sensors, a controller that sees only sensors
  scenarios.py          the tests, as goals on the world state
  trial.py jobs.py      run a trial, run complete jobs with resumable, hash-tagged logs
  env.py learn.py       reset/step interface for training, and a small training loop
  laundry.py            real cloth in a real drum, a separate small scene
  lock.py               pins the arena: results compare only under the same hash
"""
__version__ = "2.0.1"
