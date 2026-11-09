"""A learned door-opening policy: behaviour cloning from demonstrations, one arm vs two arms.

The question: does a *learned* policy keep the two-arm advantage the scripted runs show (sim/fridge_door.py)?
Each episode draws a random fridge and a random parking error, and the policy only sees what the robot can
measure: the door angle (camera, noisy), where the hinge is relative to the robot (fridge tag, noisy), and its
own joint positions. It outputs a 1 s chunk of joint targets (10 steps at 10 Hz), ACT-style, from a small MLP.

  uv run python sim/door_policy.py collect one 120     # expert demos (scripted planner) -> out/policy/demos_one.npz
  uv run python sim/door_policy.py collect two 120
  third_party/EgoGym/.venv/bin/python sim/door_policy.py train one   # torch MLP -> out/policy/mlp_one.npz
  third_party/EgoGym/.venv/bin/python sim/door_policy.py train two
  uv run python sim/door_policy.py eval 60             # both policies + both experts on the same 60 new fridges

Random per episode: door type (fridge-freezer upper door / tall door), self-closing (closes below 25 deg, below
60 deg, always), seal 10-45 N, floor mu 0.5-0.85, door mass +-25 %, the robot parked +-3 cm / +-3 cm / +-4 deg
off its plan. Demos are kept only when the expert succeeded (standard for behaviour cloning).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "policy"
HZ = 10                  # policy rate
CHUNK = 10               # predicted steps (1 s)
JKEYS = ["j1", "j2", "j3", "j4", "j5", "grip_l", "j1b", "j2b", "j3b", "j4b", "j5b", "gripb"]
CLOSINGS = ("cam25", "cam60", "always")


def sample_case(rng):
    return dict(kind=str(rng.choice(["upper", "full"])), closing=str(rng.choice(CLOSINGS)),
                seal=float(rng.uniform(10, 45)), mu=float(rng.uniform(0.5, 0.85)),
                mass_scale=float(rng.uniform(0.75, 1.25)), dx0=float(rng.uniform(-0.03, 0.03)),
                dby=float(rng.uniform(-0.03, 0.03)), yaw0=float(np.radians(rng.uniform(-4, 4))),
                seed=int(rng.integers(1 << 30)))


# ------------------------------------------------------------------ simulation side (main venv: mujoco, no torch)
def _setup(strategy, case):
    sys.path.insert(0, str(ROOT / "sim"))
    import mujoco
    import fridge_door as F
    _, info = F.best_plan(strategy, case["kind"], case["closing"])
    T, _ = F.plan(strategy, case["kind"], case["closing"], by0=info["by0"] + case["dby"], dx0=case["dx0"],
                  yaw0=case["yaw0"])
    traction = case["mu"] * F.DRIVE_SHARE * F.ROBOT_MASS * F.G
    m = F.build_model(case["kind"], traction, arms=1 if strategy == "one" else 2,
                      door_mass=F.DOORS[case["kind"]]["mass"] * case["mass_scale"], tag=f"_{mp.current_process().pid}")
    return F, mujoco, m, T


def observe(F, m, d, jq, aid, rng, door_q):
    """What the robot can measure, in its own frame."""
    bx, by, yaw = d.qpos[jq["base_x"]], d.qpos[jq["base_y"]], d.qpos[jq["base_yaw"]]
    phi = -d.qpos[door_q] + rng.normal(0, math.radians(1.5))
    h = F.to_robot((bx, by, yaw), np.array([F.FX, F.HY])) + rng.normal(0, 0.005, 2)
    rel_yaw = -yaw + rng.normal(0, math.radians(1.0))
    joints = [d.qpos[jq[n]] for n in ("lift", "drawer", "j1", "j2", "j3", "j4", "j5", "grip", "j1b", "j2b", "j3b",
                                      "j4b", "j5b", "gripb")]
    return np.array([math.sin(phi), math.cos(phi), phi, h[0], h[1], math.sin(rel_yaw), math.cos(rel_yaw), *joints])


def state_to_action(F, st, base):
    """Target joint state (deg/m, world base) -> action vector (base target in the current robot frame)."""
    bt = np.array([st["base_x"], st["base_y"]])
    rel = F.to_robot(base, bt)
    dyaw = math.radians(st["base_yaw"]) - base[2]
    arm = [math.radians(st[k]) if k in F.ANGLES else st[k] for k in JKEYS]
    return np.array([rel[0], rel[1], dyaw, st["lift"], st["drawer"], *arm])


def action_to_state(F, a, base):
    c, s = math.cos(base[2]), math.sin(base[2])
    st = dict(base_x=base[0] + c * a[0] - s * a[1], base_y=base[1] + s * a[0] + c * a[1],
              base_yaw=math.degrees(base[2] + a[2]), lift=a[3], drawer=a[4])
    for k, v in zip(JKEYS, a[5:]):
        st[k] = math.degrees(v) if k in F.ANGLES else v
    return st


def episode(args):
    """One rollout. controller: 'expert' (scripted planner) or an MLP weight dict. Returns (obs, act, ok, case)."""
    strategy, case, controller = args
    F, mujoco, m, T = _setup(strategy, case)
    rng = np.random.default_rng(case["seed"])
    d = mujoco.MjData(m)
    aid = {nm: m.actuator(nm).id for nm in F.ACT}
    jq = {nm: m.jnt_qposadr[m.actuator_trnid[aid[nm]][0]] for nm in F.ACT}
    door_q = m.jnt_qposadr[m.joint("door").id]
    door_v = m.jnt_dofadr[m.joint("door").id]

    def vec(st):
        v = np.zeros(m.nu)
        for nm, key in F.ACT.items():
            v[aid[nm]] = math.radians(st[key]) if key in F.ANGLES else st[key]
        return v
    vmax = np.zeros(m.nu)
    for nm, key in F.ACT.items():
        vmax[aid[nm]] = math.radians(F.VM[key]) if key in F.ANGLES else F.VM[key]
    q0 = vec(T.at(0.0))
    for nm in F.ACT:
        d.qpos[jq[nm]] = q0[aid[nm]]
    d.qpos[m.jnt_qposadr[m.joint("grip_r").id]] = q0[aid["grip"]]
    d.qpos[m.jnt_qposadr[m.joint("grip_rb").id]] = q0[aid["gripb"]]
    d.ctrl[:] = q0
    mujoco.mj_forward(m, d)
    cmd = q0.copy()
    dt = m.opt.timestep
    n_sub = int(round(1.0 / HZ / dt))
    T_end = T.t + (3.0 if controller != "expert" else 0.0)
    obs_log, act_log = [], []
    ens = {}                      # temporal ensembling: step index -> list of predicted target states
    k = 0
    tgt_prev = q0.copy()
    while d.time < T_end:
        base = (d.qpos[jq["base_x"]], d.qpos[jq["base_y"]], d.qpos[jq["base_yaw"]])
        o = observe(F, m, d, jq, aid, rng, door_q)
        F.MEAS["phi"] = o[2]
        if controller == "expert":
            chunk = [state_to_action(F, T.at(d.time + (i + 1) / HZ), base) for i in range(CHUNK)]
            obs_log.append(o)
            act_log.append(np.concatenate(chunk))
            st = T.at(d.time + 1.0 / HZ)
        else:
            pred = mlp_forward(controller, o).reshape(CHUNK, -1)
            for i in range(CHUNK):
                ens.setdefault(k + 1 + i, []).append(action_to_state(F, pred[i], base))
            got = ens.pop(k + 1)
            w = np.exp(-0.3 * np.arange(len(got)))[::-1]        # newest prediction weighs most
            st = {key: float(np.dot(w, [g[key] for g in got]) / w.sum()) for key in got[0]}
        tgt = vec(st)
        for j in range(n_sub):           # linear between policy steps, then the joint speed limits
            u = (j + 1) / n_sub
            goal = tgt_prev + (tgt - tgt_prev) * u
            cmd = cmd + np.clip(goal - cmd, -vmax * dt * 1.05, vmax * dt * 1.05)
            d.ctrl[:] = cmd
            phi = -d.qpos[door_q]
            d.qfrc_applied[door_v] = F.door_torque(phi, case["seal"], case["closing"])
            mujoco.mj_step(m, d)
        tgt_prev = tgt
        k += 1
        if not np.all(np.isfinite(d.qpos)):
            break
    final = math.degrees(-d.qpos[door_q])
    ok = bool(final > 88 and np.all(np.isfinite(d.qpos)))
    return (np.array(obs_log), np.array(act_log), ok, final, case)


def mlp_forward(W, o):
    x = (o - W["o_mu"]) / W["o_sd"]
    for i in range(W["n"]):
        x = x @ W[f"w{i}"] + W[f"b{i}"]
        if i < W["n"] - 1:
            x = np.maximum(x, 0.0)
    return x * W["a_sd"] + W["a_mu"]


def collect(strategy, n, workers=6, seed=1):
    rng = np.random.default_rng(seed + (0 if strategy == "one" else 1000))
    cases = [sample_case(rng) for _ in range(n)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(episode, [(strategy, c, "expert") for c in cases])
    keep = [r for r in res if r[2]]
    print(f"{strategy}: expert succeeded in {len(keep)}/{n} demos")
    for cl in CLOSINGS:
        sel = [r for r in res if r[4]["closing"] == cl]
        print(f"   {cl:>6}: {sum(r[2] for r in sel)}/{len(sel)}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"demos_{strategy}.npz", obs=np.concatenate([r[0] for r in keep]),
                        act=np.concatenate([r[1] for r in keep]),
                        ep=np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(keep)]))


# ------------------------------------------------------------------ training (EgoGym venv: torch)
def train(strategy, epochs=400, hidden=512, layers=4):
    import torch
    D = np.load(OUT / f"demos_{strategy}.npz")
    o, a, ep = D["obs"].astype(np.float32), D["act"].astype(np.float32), D["ep"]
    o_mu, o_sd = o.mean(0), o.std(0) + 1e-3
    a_mu, a_sd = a.mean(0), a.std(0) + 1e-3
    eps = np.unique(ep)
    val = np.isin(ep, eps[:: 10])                 # every 10th demo held out
    X = torch.tensor((o - o_mu) / o_sd)
    Y = torch.tensor((a - a_mu) / a_sd)
    dims = [o.shape[1]] + [hidden] * (layers - 1) + [a.shape[1]]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*mods)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = np.where(~val)[0]
    g = torch.Generator().manual_seed(0)
    for e in range(epochs):
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            xb = X[b] + 0.05 * torch.randn(len(b), X.shape[1], generator=g)   # input noise: robustness to drift
            loss = torch.nn.functional.smooth_l1_loss(net(xb), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if e % 50 == 0 or e == epochs - 1:
            with torch.no_grad():
                vl = torch.nn.functional.smooth_l1_loss(net(X[val]), Y[val]).item()
            print(f"{strategy} epoch {e:4d}  train {loss.item():.4f}  val {vl:.4f}", flush=True)
    W = dict(n=len(dims) - 1, o_mu=o_mu, o_sd=o_sd, a_mu=a_mu, a_sd=a_sd)
    lin = [mm for mm in net if isinstance(mm, torch.nn.Linear)]
    for i, l in enumerate(lin):
        W[f"w{i}"] = l.weight.detach().numpy().T.copy()
        W[f"b{i}"] = l.bias.detach().numpy().copy()
    np.savez(OUT / f"mlp_{strategy}.npz", **W)
    print(f"{strategy}: {len(eps)} demos, {len(o)} samples -> {OUT / f'mlp_{strategy}.npz'}")


def evaluate(n, workers=6, seed=99):
    rng = np.random.default_rng(seed)
    cases = [sample_case(rng) for _ in range(n)]
    rows = {}
    for strategy in ("one", "two"):
        Wf = np.load(OUT / f"mlp_{strategy}.npz")
        W = {k: Wf[k] for k in Wf.files}
        W["n"] = int(W["n"])
        for ctrl, name in (("expert", "scripted"), (W, "learned")):
            with mp.get_context("spawn").Pool(workers) as pool:
                res = pool.map(episode, [(strategy, c, ctrl) for c in cases])
            rows[(strategy, name)] = res
    lines = [f"{n} new random fridges, same for every row. Success = door >= 88 deg at the end (held, for 'always').",
             f"{'arms':>4} {'controller':>10} {'all':>8} " + " ".join(f"{c:>8}" for c in CLOSINGS) +
             f" {'upper':>8} {'full':>8}"]
    for (strategy, name), res in rows.items():
        def rate(sel):
            return f"{sum(r[2] for r in sel)}/{len(sel)}"
        lines.append(f"{strategy:>4} {name:>10} {rate(res):>8} " +
                     " ".join(f"{rate([r for r in res if r[4]['closing'] == c]):>8}" for c in CLOSINGS) +
                     f" {rate([r for r in res if r[4]['kind'] == 'upper']):>8} {rate([r for r in res if r[4]['kind'] == 'full']):>8}")
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "out" / "door_policy_eval.txt").write_text(txt + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect(sys.argv[2], int(sys.argv[3]))
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "eval":
        evaluate(int(sys.argv[2]))
