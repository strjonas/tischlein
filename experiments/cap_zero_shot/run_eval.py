"""Run a pretrained CAP policy zero-shot in EgoGym on one embodiment, in parallel worker processes.

    third_party/EgoGym/.venv/bin/python experiments/cap_zero_shot/run_eval.py --task pick --robot tisch6 --episodes 100

Robots: cap (ideal floating gripper), tisch5 (today's wrist), tisch6 (+ wrist roll), tisch6fixed (base parked).
Writes experiments/cap_zero_shot/results/<task>_<robot>_<tag>.json with one record per episode.

--anchor (v3) replaces EgoGym's contact anchor, which follows the object's true pose every step, with what a
real robot has: a world point fixed when the episode starts (deprojected once, then tracked by the robot's own
kinematics, as in the CAP paper), plus an error from the depth source:
    frozen          fixed world point, no error
    iso:0.02        + Gaussian error, 2 cm per axis (any depth source, generic)
    ray:0.10        + error along the camera ray, 10% of the distance (monocular depth estimate)
    plane:0.02      the click's ray intersected with a plane 2 cm above the table top (no depth at all)
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EGO = os.path.abspath(os.path.join(HERE, "..", "..", "third_party", "EgoGym"))
THRESH = {"pick": 0.05, "open": 0.5, "close": 0.09}
OBJECTS = {"pick": "all_pick", "open": "all_open", "close": "all_close"}


def make_anchor(mode, obs, key, rng, top_z=0.76):
    """World anchor (4x4) for the episode, from the true object pose at reset and an error model."""
    T = obs[key].reshape(4, 4).astype(np.float64).copy()
    p = T[:3, 3].copy()
    cam = obs["camera_pose"].reshape(4, 4)[:3, 3].astype(np.float64)
    kind, _, val = mode.partition(":")
    v = float(val) if val else 0.0
    if kind == "iso":
        p = p + rng.normal(0.0, v, 3)
    elif kind == "ray":
        u = (p - cam) / np.linalg.norm(p - cam)
        p = p + u * np.linalg.norm(p - cam) * rng.normal(0.0, v)
    elif kind == "plane":
        u = (p - cam) / np.linalg.norm(p - cam)
        if u[2] < -1e-3:
            t = (top_z + v - cam[2]) / u[2]
            p = cam + u * t
    T[:3, 3] = p
    return T.reshape(16).astype(np.float32), float(np.linalg.norm(p - obs[key].reshape(4, 4)[:3, 3]))


def degrade(obs, mode, rng):
    """What the wrist camera would see at night (27 Oct): gray = a mono camera; red = crude stand-in for a mono
    camera under 850 nm light (red channel only); dark = 15 % of the light with sensor noise, then auto-gain;
    darkgray = both. Only the policy's input (rgb_ego) changes."""
    if not mode:
        return obs
    img = obs["rgb_ego"].astype(np.float32) / 255.0
    if mode in ("gray", "darkgray"):
        img = np.repeat((img @ np.array([0.299, 0.587, 0.114], np.float32))[..., None], 3, axis=-1)
    elif mode == "red":
        img = np.repeat(img[..., :1], 3, axis=-1)
    if mode in ("dark", "darkgray"):
        x = img * 0.15
        x = x + rng.normal(0.0, 1.0, x.shape).astype(np.float32) * np.sqrt(x * 2.4e-4 + 1.5e-3 ** 2)
        img = x / 0.15
    out = dict(obs)
    out["rgb_ego"] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return out


def set_camera(env, spec):
    """Stand-in for a different wrist camera (27 Oct): "fov:<deg>" sets the vertical field of view (iPhone-like
    default 53.1), "shift:<m>" moves the lens sideways in the camera frame, "up:<m>" along the image's up axis,
    "back:<m>" back along the viewing axis (27 Oct: a camera board that can't sit where the iPhone's lens was).
    Applied after every reset, because EgoGym rebuilds the model each episode. Returns fresh observations."""
    if not spec:
        return None
    u = env.unwrapped
    cid = u.model.camera("egocentric").id
    for part in spec.split(","):
        k, v = part.split(":")
        if k == "fov":
            u.model.cam_fovy[cid] = float(v)
        elif k in ("shift", "up", "back"):      # camera frame: x right, y up, z backward (it looks along -z)
            ax = {"shift": 0, "up": 1, "back": 2}[k]
            u.model.cam_pos[cid] = u.model.cam_pos[cid] + u.model.cam_mat0[cid].reshape(3, 3)[:, ax] * float(v)
    import mujoco
    mujoco.mj_forward(u.model, u.data)
    return u.get_obs()


def worker(args):
    wid, task, robot, n, seed, num_objs, max_steps, record_dir, anchor, image, cam = args
    key = "object_pose" if task == "pick" else "handle_pose"
    os.chdir(EGO)
    sys.path[:0] = [EGO, HERE]
    import torch
    torch.set_num_threads(1)
    import gymnasium as gym
    import tischlein_cap  # noqa: F401  registers tisch* robots and context-aware tasks
    import egogym.assets.constants as C
    from baselines.cap_policy import CAPPolicy, get_config

    cfg = get_config()
    cfg.update(model_path=f"checkpoints/{task}.pt", device="cpu")
    if os.environ.get("GRASP_STYLE"):
        cfg["grasping_style"] = os.environ["GRASP_STYLE"]
    policy = CAPPolicy(config=cfg)
    objects = getattr(C, f"{OBJECTS[task]}_objects_set")
    env = gym.make(f"Egogym-{task.capitalize()}-v0", robot=robot, action_space="delta",
                   render_size=(299, 224), num_objs=num_objs, seed=seed + 1000 * wid, objects_set=objects)
    out = []
    arng = np.random.default_rng(seed + 1000 * wid + 77)
    for ep in range(n):
        obs, info = env.reset()
        obs = set_camera(env, cam) or obs
        policy.reset()
        anchor_T, anchor_e = (make_anchor(anchor, obs, key, arng) if anchor else (None, 0.0))
        if anchor_T is not None:
            obs[key] = anchor_T
        frames = []
        success, steps, reward = False, 0, 0.0
        for steps in range(1, max_steps + 1):
            action = policy.get_action(degrade(obs, image, arng))
            obs, reward, term, trunc, info = env.step(action)
            if anchor_T is not None:
                obs[key] = anchor_T
            if record_dir and wid == 0 and ep < 2:
                frames.append(np.concatenate([obs["rgb_ego"], obs["rgb_exo"]], axis=1))
            if reward > THRESH[task]:
                success = True
                break
            if term:
                break
        rob = env.unwrapped.robot
        rec = dict(worker=wid, ep=ep, success=bool(success), steps=steps, reward=float(reward),
                   object=info.get("object_name", ""), anchor_err=anchor_e)
        obj = getattr(env.unwrapped, "object", None)
        if obj is not None and getattr(obj, "last_set_pose", None) is not None:
            rec["target_xyz"] = [float(v) for v in obj.last_set_pose[:3, 3]]
        log = getattr(rob, "ik_log", None)
        if log:
            pos = np.array([e["pos"] for e in log])
            app = np.array([e["app"] for e in log])
            ang = np.array([e["ang"] for e in log])
            rec.update(ik_pos_max=float(pos.max()), ik_pos_mean=float(pos.mean()),
                       ik_app_max=float(app.max()), ik_ang_max=float(ang.max()), ik_ang_mean=float(ang.mean()),
                       ik_start_pos=float(pos[0]), ik_start_ang=float(ang[0]),
                       base_travel=float(abs(rob.q[0])), lift_final=float(rob.q[2]))
        out.append(rec)
        if frames:
            import imageio
            imageio.mimsave(os.path.join(record_dir, f"{task}_{robot}_w{wid}_e{ep}.mp4"), frames, fps=4)
        print(f"[w{wid}] {task}/{robot} ep {ep}: {'OK ' if success else 'miss'} {rec['object']} steps={steps}",
              flush=True)
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick")
    ap.add_argument("--robot", default="cap")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_objs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--anchor", default="", help="frozen | iso:<m> | ray:<rel> | plane:<m above top>")
    ap.add_argument("--image", default="", help="gray | red | dark | darkgray (night-camera stand-ins)")
    ap.add_argument("--cam", default="", help="fov:<deg>,shift:<m>,up:<m>,back:<m> (a different wrist camera)")
    a = ap.parse_args()
    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    rec_dir = os.path.join(HERE, "videos") if a.record else None
    if rec_dir:
        os.makedirs(rec_dir, exist_ok=True)
    per = [a.episodes // a.workers + (1 if i < a.episodes % a.workers else 0) for i in range(a.workers)]
    jobs = [(i, a.task, a.robot, per[i], a.seed, a.num_objs, a.max_steps, rec_dir, a.anchor, a.image, a.cam)
            for i in range(a.workers) if per[i]]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        results = [r for chunk in pool.map(worker, jobs) for r in chunk]
    sr = np.mean([r["success"] for r in results])
    path = os.path.join(res_dir, f"{a.task}_{a.robot}_{a.tag}.json")
    with open(path, "w") as f:
        json.dump(dict(task=a.task, robot=a.robot, episodes=len(results), success_rate=float(sr),
                       seed=a.seed, num_objs=a.num_objs, max_steps=a.max_steps, anchor=a.anchor, image=a.image, cam=a.cam,
                       minutes=(time.time() - t0) / 60, episodes_detail=results), f, indent=1)
    print(f"{a.task}/{a.robot}: {sr * 100:.1f}% of {len(results)}  ({(time.time() - t0) / 60:.1f} min) -> {path}")


if __name__ == "__main__":
    main()
