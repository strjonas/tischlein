"""Behavioural regressions for the validity of the arena's physics, judge and sensor boundary."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace

import mujoco
import numpy as np

from arena import assemble, geometry, homes, judge, provenance, reach, zoo
from arena.env import ArenaEnv
from arena.robot import Axis, Base, Gripper, Robot
from arena.skills import Controller
from arena.rotation_error import tool_error


class Validity(unittest.TestCase):
    def test_orientation_error_handles_opposite_fingers_and_wrong_jaw_axis(self):
        self.assertAlmostEqual(np.linalg.norm(tool_error(np.eye(3), (-1,0,0))), np.pi)
        self.assertGreater(np.linalg.norm(tool_error(np.eye(3), (1,0,0), (0,0,1))), 1.5)
        self.assertLess(np.linalg.norm(tool_error(np.eye(3), (1,0,0), (0,-1,0))), 1e-9)
    def test_nonadjacent_arm_collision_really_generates_contacts(self):
        # A distal beam folds through the shoulder beam. Parent-child filtering
        # is legitimate; the nonadjacent shoulder/distal pair must still collide.
        axes = [Axis("a", "hinge", (0, 0, 1), -3.2, 3.2, 10, 1, pos=(0, 0, .8),
                     geoms=[("box", (0, -.03, -.03), (1, .03, .03))]),
                Axis("b", "hinge", (0, 0, 1), -3.2, 3.2, 10, 1, pos=(1, 0, 0), home=np.pi),
                Axis("c", "hinge", (0, 0, 1), -3.2, 3.2, 10, 1, pos=(.5, 0, 0),
                     geoms=[("box", (0, -.03, -.03), (.8, .03, .03))])]
        robot = Robot("collision_test", "test", Base(), axes, Gripper(), [])
        b = assemble.build(robot, homes.HOMES["A"])
        d = mujoco.MjData(b.m)
        assemble.ground_robot(b, d, 2.4, 6.1, 0)
        mujoco.mj_forward(b.m, d)
        names = assemble.geom_owner(b.m)
        hits = [c for c in d.contact if {names[c.geom1], names[c.geom2]} == {"arm_a_0", "arm_c_0"} and c.dist < -.01]
        self.assertTrue(hits, "folded nonadjacent beams passed through each other")

    def test_telescope_overlap_at_maximum_extension(self):
        for name in ("revolution_low", "stretch8"):
            r = zoo.ROBOTS[name]()
            a = r.axes["reach"]
            axis = int(np.argmax(np.abs(a.axis)))
            intervals = []
            for k in range(-1, a.stages):
                low, high = geometry.bounds(geometry.telescope_shell(a, k))
                shift = (k+1)*a.hi/a.stages
                intervals.append((low[axis]+shift, high[axis]+shift))
            for first, second in zip(intervals, intervals[1:]):
                self.assertGreaterEqual(min(first[1], second[1])-max(first[0], second[0]), .04)

    def test_no_simulator_pose_read_in_controller(self):
        class Poison:
            def base_pose(self):
                raise AssertionError("controller accessed ground truth")
        b = assemble.build(zoo.ROBOTS["revolution_low"](), homes.HOMES["A"])
        T = reach.Tier1(b)
        pose = (2.0, 5.1, .2)
        c = Controller(Poison(), b, lambda c: iter(()), T=T, initial_pose=pose)
        self.assertEqual(c.pose(), pose)

    def test_unsafe_satisfied_goal_is_rejected_by_training_and_judge(self):
        e = ArenaEnv("revolution_low", "pick_can_fridge", home="A")
        e.reset(seed=0)
        e.s = replace(e.s, goal=lambda w, ctx: (True, "satisfied"))
        for flag, value in (("tip", True), ("spill", True), ("max_person", 100.0)):
            e.w.mon.update(tip=False, spill=False, max_person=0)
            e.w.mon[flag] = value
            self.assertFalse(judge.evaluate(e.s, e.w, e.ctx)["ok"])
            _, reward, ended, _, info = e.step(np.zeros(e.action_size))
            self.assertFalse(info["success"])
            self.assertTrue(ended)
            self.assertLess(reward, 0)

    def test_camera_variants_do_not_mutate_other_robots(self):
        a = zoo.ROBOTS["stretch8"]()
        a.cams[0].far = .08
        b = zoo.ROBOTS["stretch8"]()
        self.assertGreater(b.cams[0].far, .5)

    def test_identity_changes_when_a_robot_changes(self):
        r = zoo.ROBOTS["revolution_low"]()
        a = provenance.identity(r, "fridge_bottle", "A", 1)
        r.grip.force *= .5
        b = provenance.identity(r, "fridge_bottle", "A", 1)
        self.assertEqual(a["source_hash"], b["source_hash"])
        self.assertNotEqual(a["robot_hash"], b["robot_hash"])

    def test_high_arm_can_pass_over_furniture_but_low_arm_cannot(self):
        base = Base(x=(-.12,.12), y=(-.12,.12), z=(.035,.095), mass=10,
                    wheel_r=.05, wheel_y=.09, wheel_w=.02, casters=[(.08,0),(-.08,0)])
        arm = Axis("lift", "slide", (0,0,1), .1, 1.3, 100, .1, home=1.2,
                   geoms=[("box", (0,-.02,-.02), (.6,.02,.02))])
        robot = Robot("height_test", "test", base, [arm], Gripper(), [])
        T = reach.Tier1(assemble.build(robot, homes.HOMES["A"]))
        i,j = int((2.2+.2)/reach.CELL), int(8.0/reach.CELL)
        self.assertTrue(T.free[4,i,j], "planner extruded the arm footprint down through the bedside cabinet")
        T.home_q = np.array([.59])
        T._grid()
        self.assertFalse(T.free[4,i,j], "planner missed an arm physically intersecting the bedside cabinet")

    def test_open_command_recovers_from_a_blocked_squeeze(self):
        e = ArenaEnv("revolution_low", "pick_can_fridge", home="A")
        e.reset(seed=0)
        w = e.w
        for q in w.q_jaw:
            w.d.qpos[q] = .065
        w.grip_cmd = -.03
        w.command(grip=.17)
        self.assertGreater(w.grip_cmd, .12, "opening had to unwind an invisible 16 cm position error")

    def test_full_fridge_job_starts_with_closed_door(self):
        e = ArenaEnv("revolution_low", "fridge_bottle", home="A")
        e.reset(seed=0)
        self.assertLess(e.w.door_angle("fr_door"), .01)
        self.assertTrue(e.w.d.eq_active[e.w.eq["fr_door"]])

    def test_microwave_release_actually_pops_the_door(self):
        e = ArenaEnv("revolution_low", "pick_can_fridge", home="A")
        e.reset(seed=0)
        e.w.command(kit="mw_release")
        for _ in range(100):
            e.w.step()
        self.assertGreater(e.w.door_angle("mw_door"), .08)


if __name__ == "__main__":
    unittest.main()
