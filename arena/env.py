"""The training interface: reset / step over the same world, robot, sensors and judge as the competition.

    env = ArenaEnv("stretch8", "pick_can_fridge", home="A")
    obs, info = env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(action)

Same five-tuple as Gymnasium, without the dependency. An action is, per 20 ms tick: base speed, base turn rate, an offset on
every axis position, and a gripper opening; the motor limits stay in force. Domain randomization is on by default and
draws every uncertain number per episode, so a policy cannot learn the simulator's nominal values. Observations are what
the robot's sensors give; the truth is only in info["truth"], for reward shaping and asymmetric critics, and a policy that
reads it at run time has left the arena's rules.

Rewards are the judge's: +1 when the goal holds at the end, and shaping from the distance of the hand to the object. Home E-H stay
out of training: they are the held-out test.

Speed on this machine is reported by `python -m arena.env --bench`. For GPU-scale training the same MJCF loads in MuJoCo Warp or
Newton on an NVIDIA machine; the arena's Python appliance logic is then the part to port.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from . import objects, reach, trial, judge
from .homes import DEV
from .scenarios import SCENARIOS
from .sim import CTRL_DT, World


class ArenaEnv:
    def __init__(self, robot="stretch8", scenario="pick_can_fridge", home=None, dr=True, noise=1.0, phys=None):
        self.robot, self.home, self.dr, self.noise, self.phys = robot, home, dr, noise, phys
        self.s = SCENARIOS[scenario]
        self.belief = {}
        self.w = None
        self.na = None
        self.episode = 0

    @property
    def action_size(self):
        return 2 + self.w.na + 1

    def reset(self, seed=None):
        seed = self.episode if seed is None else seed
        self.episode += 1
        hk = self.home or DEV[seed % len(DEV)]
        b = trial._model(self.robot, hk, self.s.objs, self.phys)
        if hk not in self.belief:
            self.belief[hk] = reach.Tier1(trial._model(self.robot, hk, list(objects.CATALOG), self.phys))
        T = self.belief[hk]
        self.w = World(b, seed=seed, dr=self.dr, noise=self.noise)
        self.w.place_robot(*T.dock_pose())
        rng = np.random.default_rng(seed + 7919)
        self.ctx = self.s.setup(self.w, rng)
        feasible = not self.s.place or self.s.place(self.w, T, self.ctx, rng)
        self.w.start(doors=self.ctx.get("open"))
        self.q = self.w.true_q()
        self.grip = self.w.true_grip()
        self.t0 = self.w.t
        self.ob = self.w.step()
        return self._vec(self.ob), dict(truth=self._truth(), feasible=feasible, home=hk)

    def _vec(self, o):
        """Flat vector: axis positions, speeds, forces, opening, squeeze, tilt, then the latest object detection in the
        camera frame (zeros when nothing is seen). Rendered images are available through arena.render if a policy wants pixels."""
        targets = [det for det in o.dets if det.name in self.s.objs]
        det = targets[-1].pos if targets else np.zeros(3)
        return np.concatenate([o.q, o.qd, o.tau, [o.grip, o.grip_f, o.tilt], det, [float(bool(targets))]]).astype(np.float32)

    def _truth(self):
        w = self.w
        return dict(base=w.base_pose(), ee=w.d.site_xpos[w.ee].copy(), obj={k: w.obj_pos(k) for k in w.obj})

    def step(self, action):
        a = np.asarray(action, float)
        w = self.w
        n = w.na
        lim = np.array([x["a"].speed * 0.05 for x in w.ax])      # at most 5 % of top speed per tick... of a second
        self.q = self.q + np.clip(a[2:2 + n], -1, 1) * lim
        self.grip = float(np.clip(self.grip + np.clip(a[2 + n], -1, 1) * 0.002, 0.0, w.r.grip.stroke))
        w.command(q=self.q, v=float(np.clip(a[0], -1, 1)) * 0.2, w=float(np.clip(a[1], -1, 1)) * 0.6, grip=self.grip, squeeze=self.grip < 0.005)
        self.ob = w.step()
        terminated = judge.unsafe(w)
        truncated = w.t - self.t0 >= self.s.limit
        ok = False
        if truncated or terminated:
            ok = judge.evaluate(self.s, w, self.ctx)["ok"]
        tr = self._truth()
        name = self.s.objs[0] if self.s.objs else None
        shaping = -float(np.linalg.norm(tr["ee"] - tr["obj"][name])) * 0.01 if name else 0.0
        return self._vec(self.ob), (1.0 if ok else -1.0 if terminated else 0.0) + shaping, terminated, truncated, dict(truth=tr, success=ok)


def bench(steps=2000):
    out = {}
    for sc in ("pick_can_fridge", "pick_phone_floor"):
        e = ArenaEnv("stretch8", sc, home="A")
        t0 = time.time()
        obs, _ = e.reset(seed=0)
        t_reset = time.time() - t0
        A = np.zeros(e.action_size)
        t0 = time.time()
        for i in range(steps):
            r = e.step(A)
            if r[2] or r[3]:
                break
        dt = time.time() - t0
        out[sc] = dict(reset_s=round(t_reset, 2), steps_per_s=round((i + 1) / dt), realtime=round((i + 1) * CTRL_DT / dt, 1), nv=e.w.m.nv, ngeom=e.w.m.ngeom)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    print(bench())
