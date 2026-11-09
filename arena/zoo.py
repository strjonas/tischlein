"""Reference robots for the arena: each one is a data entry (base, kinematic chain, hand, cameras, appliance aids).

stretch8          telescoping reacher on a rear mast, 3-axis wrist
revolution_600    straight reacher on a turning carriage, wrist yaw + pitch
revolution_low    revolution_600 with a low deck and a wide hand
orthodox_v03      SCARA arm on a 1.8 m mast with a CAP-style hand (the repository's original mobile design)

An appliance aid a robot needs is listed in kits= (see homes.KITS); it is not hidden in the world.
"""
from __future__ import annotations

import math
import copy

from . import stow
from .robot import Axis, Base, Cam, Deck, Gripper, Robot

D = math.radians
WRIST_CAM = Cam("wrist", "ee", (0.03, 0.0, 0.05), (0.92, 0.0, -0.39), fov=(87, 58), near=0.07, far=0.60)   # depth-camera class, on top of the palm

def _lift(pos, lo, hi, force, speed, mass=1.0, selflock=True, friction=None, geoms=None, flex=0.0, flex_at=(0, 0, 0), home=None):
    """Stowed at 0.60 m: above the deck and the bed frame, hand held in close."""
    return Axis("lift", "slide", (0, 0, 1), lo, hi, force, speed, pos=pos, mass=mass, selflock=selflock, friction=friction,
                geoms=geoms or [("box", (-0.04, -0.05, -0.05), (0.04, 0.05, 0.05))], armature=0.5, flex=flex, flex_at=flex_at, home=min(max(0.60, lo), hi) if home is None else home)


def _sag_k(sag_mm, L, kg=1.0):
    """Root stiffness that gives sag_mm at the tip for kg at lever L."""
    return 9.81 * kg * L * L / (sag_mm / 1000)


def stretch8(sag_mm=5.0, backlash=0.0, travel=0.012):
    """Telescoping reacher: mast at the rear corner, telescope to the left, 3-axis wrist, 8 motors.
    travel: caster-leg travel over a threshold; every entry here carries 12 mm sprung legs."""
    b = Base(travel=travel, extra=[("box", (-0.33, -0.22, 0.03), (-0.27, -0.16, 1.42)), ("box", (-0.34, -0.23, 1.42), (-0.23, -0.12, 1.49))],
             mass=20.0, parts=[(1.5, (0.03, 0, 0.30), (0.55, 0.44, 0.02)), (2.4, (-0.30, -0.19, 0.72), (0.06, 0.06, 1.4)), (0.4, (-0.30, -0.19, 1.45), (0.1, 0.1, 0.07))])
    chain = [
        _lift((-0.29, 0, 0), 0.03, 1.30, 200, 0.10, mass=1.4, friction=3.0,
              geoms=[("box", (-0.052, -0.16, -0.04), (0.052, -0.12, 0.10)), ("box", (-0.0275, -0.12, -0.0275), (0.0275, 0.02, 0.0275))]),
        Axis("reach", "slide", (0, 1, 0), 0.0, 0.60, 60, 0.15, stages=4, mass=1.1, com=(0, -0.05, 0), friction=2.0, armature=0.15,
             geoms=[("box", (-0.0245, -0.11, -0.0245), (0.0245, 0.11, 0.0245))], flex=_sag_k(sag_mm, 0.75), flex_at=(0, -0.12, 0), home=0.0),
        Axis("yaw", "hinge", (0, 0, 1), -1.75, 1.75, 1.5, D(90), pos=(0, 0.135, 0), euler=(0, 0, 90), mass=0.30, friction=0.02, backlash=backlash, home=0.0,
             geoms=[("box", (-0.02, -0.0225, -0.0225), (0.02, 0.0225, 0.0225))]),
        Axis("pitch", "hinge", (0, 1, 0), -1.75, 1.75, 3.0, D(60), pos=(0.045, 0, 0), mass=0.25, friction=0.02, backlash=backlash, home=0.0,
             geoms=[("box", (-0.02, -0.025, -0.02), (0.02, 0.025, 0.02))]),
        Axis("roll", "hinge", (1, 0, 0), -3.2, 3.2, 1.5, D(90), pos=(0.03, 0, 0), mass=0.20, friction=0.02, backlash=backlash, home=0.0),
    ]
    return Robot("stretch8", "Telescoping reacher on a rear mast (8 motors)", b, chain, Gripper(stroke=0.10, jaw_len=0.06, force=80, speed=0.04, selflock=True, mass=0.30),
                 [WRIST_CAM, Cam("head", "robot", (-0.25, -0.19, 1.47), (0.8, 0.0, -0.6), fov=(69, 42), near=0.4, far=4.0)],
                 Deck((-0.20, 0.28), (-0.20, 0.20), 0.30, parent="robot"), kits=("fridge_opener", "mw_pusher", "ramps"), motors=8)


def revolution_600():
    """Straight reacher on a turning carriage, wrist yaw + pitch. The carriage stays above the base; this entry has no floor tool."""
    b = Base(x=(-0.22, 0.22), y=(-0.20, 0.20), z=(0.04, 0.22), mass=13.0, com=(-0.04, 0, 0.12), wheel_y=0.19, torque=8.0, vmax=0.30, casters=[(0.19, 0.0), (-0.19, 0.0)],
             caster_r=0.05, drive_share=0.7, extra=[("box", (-0.03, -0.03, 0.04), (0.03, 0.03, 1.55))], parts=[(1.5, (0, 0, 0.8), (0.06, 0.06, 1.5))])
    chain = [_lift((0, 0, 0), 0.27, 1.40, 150, 0.04, mass=1.0, friction=8.0, geoms=[("box", (-0.05, -0.05, -0.03), (0.05, 0.05, 0.05))]),
             Axis("turn", "hinge", (0, 0, 1), -3.0, 3.0, 6.0, D(60), mass=0.3, friction=0.2, home=0.0, armature=0.05),
             Axis("reach", "slide", (1, 0, 0), 0.0, 0.50, 30, 0.12, stages=3, pos=(0.09, 0, 0), mass=0.6, com=(0.12, 0, 0), friction=1.5, armature=0.1,
                  geoms=[("box", (0.0, -0.02, -0.02), (0.25, 0.02, 0.02))], flex=_sag_k(8.0, 0.55), flex_at=(0, 0, 0), home=0.0),
             Axis("wyaw", "hinge", (0, 0, 1), -1.6, 1.6, 2.0, D(90), pos=(0.28, 0, 0), mass=0.12, friction=0.02, home=0.0),
             Axis("wpitch", "hinge", (0, 1, 0), -1.6, 1.6, 2.0, D(60), pos=(0.03, 0, 0), mass=0.10, friction=0.02, home=0.0)]
    return Robot("revolution_600", "Revolution-600 (reacher on turning carriage)", b, chain, Gripper(stroke=0.09, jaw_len=0.06, force=40, speed=0.04, mass=0.20),
                 [Cam("wrist", "ee", (0.03, 0.0, 0.05), (0.92, 0.0, -0.39), fov=(70, 55), near=0.10, far=0.8, kind="rgb"),
                  Cam("head", "turn", (0.09, -0.07, 0.20), (0.9, 0.0, -0.44), fov=(70, 55), near=0.3, far=3.0, kind="rgb")],
                 Deck((0.09, 0.26), (-0.14, 0.14), 0.24, parent="robot"), kits=("fridge_opener", "mw_pusher"), motors=8)


def revolution_low():
    """Revolution-600 with a low deck so that the carriage clears the base at 12.5 cm and wrist pitch reaches the floor,
    a 20 cm hand for meal boxes, and ramps for the thresholds. It folds over the base."""
    r = revolution_600()
    r.name, r.label = "revolution_low", "Revolution low: low deck, 20 cm hand"
    r.base.z = (0.04, 0.095)                         # a hoverboard deck is about this high
    r.chain[0].lo = 0.125
    r.kits = r.kits + ("ramps",)
    r.grip = Gripper(stroke=0.20, jaw_len=0.09, jaw_h=0.04, force=40, speed=0.05, palm=(0.05, 0.26, 0.04), mass=0.28)
    r.deck = Deck((0.09, 0.26), (-0.14, 0.14), 0.115, parent="robot")
    return r


def orthodox_v03():
    """Repo root v0.3: SCARA arms on a 1.8 m mast, CAP-style hand. One working arm here; the stump is a second-hand aid."""
    b = Base(x=(-0.38, 0.055), y=(-0.20, 0.20), z=(0.13, 0.17), mass=9.0, com=(-0.14, 0, 0.13), wheel_y=0.195, torque=8.0, vmax=0.30, casters=[(0.20, 0.20), (0.20, -0.20), (-0.345, 0.0)],
             caster_r=0.05, drive_share=0.75, extra=[("box", (0.055, -0.02, 0.02), (0.095, 0.02, 1.82))], parts=[(1.55, (0.075, 0, 0.9), (0.04, 0.04, 1.8))])
    chain = [_lift((0.11, 0, 0), 0.075, 1.54, 120, 0.06, mass=1.1, selflock=False, friction=2.0, geoms=[("box", (0.0, -0.14, -0.02), (0.25, 0.14, 0.0))]),
             Axis("shoulder", "hinge", (0, 0, 1), -2.2, 2.2, 3.0, D(60), pos=(0.05, 0.0, 0.155), mass=0.65, com=(0.11, 0, 0), friction=0.15, armature=0.03, home=-1.2,
                  geoms=[("box", (0.0, -0.02, -0.015), (0.22, 0.02, 0.015))], flex=_sag_k(10.0, 0.55)),
             Axis("elbow", "hinge", (0, 0, 1), -2.5, 2.5, 2.0, D(90), pos=(0.22, 0, -0.04), mass=0.22, com=(0.11, 0, 0), friction=0.1, armature=0.02, home=2.2,
                  geoms=[("box", (0.0, -0.018, -0.012), (0.22, 0.018, 0.012))]),
             Axis("wyaw", "hinge", (0, 0, 1), -1.8, 1.8, 1.0, D(150), pos=(0.22, 0, -0.03), mass=0.15, friction=0.05, home=-1.0),
             Axis("wpitch", "hinge", (0, 1, 0), -0.26, 1.75, 1.6, D(120), pos=(0.03, 0, -0.03), mass=0.15, friction=0.05, home=0.0),
             Axis("wroll", "hinge", (1, 0, 0), -2.9, 2.9, 1.0, D(150), pos=(0.03, 0, 0), mass=0.12, friction=0.05, home=0.0)]
    return Robot("orthodox_v03", "Orthodox v0.3 (SCARA arm on 1.8 m mast)", b, chain, Gripper(stroke=0.09, jaw_len=0.07, force=25, speed=0.06, palm=(0.11, 0.08, 0.05), mass=0.30),
                 [Cam("wrist", "ee", (0.03, 0.0, 0.05), (0.92, 0.0, -0.39), fov=(90, 60), near=0.10, far=0.8, kind="rgb")],
                 Deck((0.11, 0.32), (-0.14, 0.14), 0.0, parent="lift"), kits=("fridge_opener", "mw_pusher"), motors=11)


VARIANTS = {}       # other tool configurations of one robot (tier 1 passes if any passes)


def _stowed(f):
    def make(*a, **k):
        return stow.fit(copy.deepcopy(f(*a, **k)))
    return make


ROBOTS = {f.__name__: _stowed(f) for f in (stretch8, revolution_600, orthodox_v03)}
ROBOTS.update(revolution_low=_stowed(revolution_low))
