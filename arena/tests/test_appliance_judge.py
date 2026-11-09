"""An empty microwave timer must not count as heating a delivered meal."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from arena import appliance_judge, assemble, homes, scenarios, zoo
from arena.sim import World


class Contents(unittest.TestCase):
    def setUp(self):
        b = assemble.build(None, homes.HOMES["A"], objs=[("meal_box", "meal_box")])
        m, d = b.m, mujoco.MjData(b.m)
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "meal_box")
        eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "mw_door_latch")
        self.assertGreaterEqual(eq, 0)
        self.w = SimpleNamespace(b=b, m=m, d=d, obj={"meal_box":dict(b=body)},
                                 eq={"mw_door":eq}, state=dict(mw_running=True, mw_heated=100.0))
        d.eq_active[eq] = 1
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "meal_box")
        self.adr = m.jnt_qposadr[joint]

    def place(self, xyz):
        self.w.d.qpos[self.adr:self.adr+3] = xyz
        self.w.d.qpos[self.adr+3:self.adr+7] = [1,0,0,0]
        mujoco.mj_forward(self.w.m, self.w.d)

    def test_timer_outside_oven_earns_no_meal_credit(self):
        self.place(self.w.b.world.ports["worktop_item"])
        appliance_judge.accumulate(self.w, 12)
        self.assertFalse(appliance_judge.heated(self.w, "meal_box"))

    def test_entire_meal_inside_closed_running_oven_earns_credit(self):
        self.place(self.w.b.world.ports["mw_item"])
        appliance_judge.accumulate(self.w, 12)
        self.assertTrue(appliance_judge.heated(self.w, "meal_box"))

    def test_center_inside_but_meal_protruding_earns_no_credit(self):
        low, high = np.array(self.w.b.world.regions["mw_cavity"])
        p = (low+high)/2
        p[0] = low[0]+.01
        self.place(p)
        appliance_judge.accumulate(self.w, 12)
        self.assertFalse(appliance_judge.heated(self.w, "meal_box"))

    def test_open_door_earns_no_credit(self):
        self.place(self.w.b.world.ports["mw_item"])
        self.w.d.eq_active[self.w.eq["mw_door"]] = 0
        appliance_judge.accumulate(self.w, 12)
        self.assertFalse(appliance_judge.heated(self.w, "meal_box"))

    def test_full_job_goal_rejects_an_empty_timer_even_if_delivery_is_satisfied(self):
        self.w.door_angle = lambda _: 0
        with patch("arena.scenarios.deliver_goal", return_value=lambda w,c: (True, "")):
            ok, _ = scenarios.SCENARIOS["microwave_meal"].goal(self.w, {})
        self.assertFalse(ok)

    def test_full_job_goal_accepts_contents_credit_with_delivery_satisfied(self):
        self.place(self.w.b.world.ports["mw_item"])
        appliance_judge.accumulate(self.w, 12)
        self.w.door_angle = lambda _: 0
        with patch("arena.scenarios.deliver_goal", return_value=lambda w,c: (True, "")):
            ok, _ = scenarios.SCENARIOS["microwave_meal"].goal(self.w, {})
        self.assertTrue(ok)

    def test_runtime_tracks_the_meal_in_the_cavity(self):
        b = assemble.build(zoo.ROBOTS["revolution_low"](), homes.HOMES["A"], objs=[("meal_box", "meal_box")])
        w = World(b, dr=False, noise=0)
        w.place_robot(2.4, 6.1, 0)
        w.place("meal_box", b.world.ports["mw_item"])
        w.start()
        w.command(kit="mw_start")
        for _ in range(600):
            w.step()
        self.assertTrue(appliance_judge.heated(w, "meal_box"))


if __name__ == "__main__":
    unittest.main()
