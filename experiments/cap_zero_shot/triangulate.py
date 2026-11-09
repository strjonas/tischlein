"""How accurately can the iPhone SE (no depth) locate a tapped point? Two wrist views, triangulated.

    third_party/EgoGym/.venv/bin/python experiments/cap_zero_shot/triangulate.py [episodes]

For each EgoGym pick scene (random objects, textures, tables):
  1. the tap: the visible pixel of the target object nearest to its centre, in the first wrist image;
  2. the truth: the 3D surface point under that pixel, from the renderer's depth buffer;
  0. the wrist first moves to ~35 cm from the object (the bench protocol's start);
  3. the wrist moves 8 cm sideways (the robot knows how far from its joint angles; we add the error of cheap
     servos to that relative pose: 1.5 mm and 0.3 deg per axis), a second image is taken;
  4. the tapped patch is found again in the second image by template matching along the epipolar line (sub-pixel),
     and the two rays are intersected (midpoint method).
Rendered at 960 x 720 with the CAP camera's 53.1 deg vertical field of view (the SE streams 1920 x 1440).
Rendered images are cleaner than a real phone's (no blur, no noise), so the matching part is optimistic.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EGO = os.path.abspath(os.path.join(HERE, "..", "..", "third_party", "EgoGym"))
os.chdir(EGO)
sys.path[:0] = [EGO]
import gymnasium as gym  # noqa: E402
import egogym  # noqa: E402,F401
import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

W, H, FOVY = 960, 720, 53.1
F_PX = (H / 2) / np.tan(np.radians(FOVY) / 2)
K = np.array([[F_PX, 0, W / 2], [0, F_PX, H / 2], [0, 0, 1.0]])
BASELINE = float(os.environ.get("TRI_BASE", 0.08))
T_SD = float(os.environ.get("TRI_T_SD", 0.0015))      # per-view pose error: translation (m) and rotation (deg)
R_SD = float(os.environ.get("TRI_R_SD", 0.3))         # 1.5 mm / 0.3 deg ~ cheap servos; ARKit tracking ~0.5 mm / 0.1 deg
DIST = float(os.environ.get("TRI_DIST", 0.35))
MODE = os.environ.get("TRI_MODE", "track")    # "track": KLT through N_STEPS small moves; "pair": one move + template match
N_STEPS = int(os.environ.get("TRI_STEPS", 4))   # camera-to-object distance for the first view (bench start ~0.35 m)


def project(T_wc, p):
    """World point -> pixel (MuJoCo camera: looks along -z, x right, y up; image v grows downward)."""
    q = np.linalg.inv(T_wc) @ np.append(p, 1.0)
    x, y, z = q[:3]
    return np.array([W / 2 + F_PX * x / -z, H / 2 - F_PX * y / -z]), -z


def ray(T_wc, uv):
    d_cam = np.array([(uv[0] - W / 2) / F_PX, -(uv[1] - H / 2) / F_PX, -1.0])
    d = T_wc[:3, :3] @ d_cam
    return T_wc[:3, 3], d / np.linalg.norm(d)


def midpoint(o1, d1, o2, d2):
    w0 = o1 - o2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w0, d2 @ w0
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return (o1 + s * d1 + o2 + t * d2) / 2


def match(img1, img2, uv1, T1, T2, half=20):
    """Find uv1's patch in img2 near the epipolar line, sub-pixel. Returns uv2 or None."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    u, v = int(round(uv1[0])), int(round(uv1[1]))
    if not (half <= u < W - half and half <= v < H - half):
        return None
    tpl = g1[v - half:v + half + 1, u - half:u + half + 1]
    # epipolar line: project points along the first ray at 0.15 .. 1.5 m into image 2
    o, d = ray(T1, uv1)
    pts = [project(T2, o + d * z)[0] for z in np.linspace(0.15, 1.5, 200)]
    best, best_uv = -2.0, None
    res = cv2.matchTemplate(g2, tpl, cv2.TM_CCOEFF_NORMED)
    for p in pts:
        cu, cv_ = int(round(p[0])) - half, int(round(p[1])) - half
        for dv in range(-3, 4):
            vv, uu = cv_ + dv, cu
            if 0 <= vv < res.shape[0] and 0 <= uu < res.shape[1] and res[vv, uu] > best:
                best, best_uv = res[vv, uu], (uu, vv)
    if best_uv is None or best < 0.6:
        return None
    uu, vv = best_uv
    # sub-pixel: parabola through the neighbours
    def sub(a, b0, c):
        den = a - 2 * b0 + c
        return 0.0 if abs(den) < 1e-9 else 0.5 * (a - c) / den
    du = sub(res[vv, uu - 1], res[vv, uu], res[vv, uu + 1]) if 0 < uu < res.shape[1] - 1 else 0.0
    dv = sub(res[vv - 1, uu], res[vv, uu], res[vv + 1, uu]) if 0 < vv < res.shape[0] - 1 else 0.0
    return np.array([uu + half + du, vv + half + dv])


def triangulate_n(rays):
    """Least-squares point closest to all rays (o, d)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        Pm = np.eye(3) - np.outer(d, d)
        A += Pm
        b += Pm @ o
    return np.linalg.solve(A, b)


def perturb(T, rng, t_sd=None, r_sd_deg=None):
    t_sd = T_SD if t_sd is None else t_sd
    r_sd_deg = R_SD if r_sd_deg is None else r_sd_deg
    P = T.copy()
    P[:3, 3] += rng.normal(0, t_sd, 3)
    P[:3, :3] = R.from_rotvec(np.radians(rng.normal(0, r_sd_deg, 3))).as_matrix() @ P[:3, :3]
    return P


def main(n=100, seed=3):
    env = gym.make("Egogym-Pick-v0", robot="cap", action_space="delta", render_size=(W, H), num_objs=1, seed=seed)
    rng = np.random.default_rng(seed)
    depth_r = seg_r = None
    errs, misses, dists, occluded = [], 0, [], 0
    for ep in range(n):
        obs, _ = env.reset()
        u = env.unwrapped
        m, d = u.model, u.data
        if depth_r is None:
            depth_r = mujoco.Renderer(m, height=H, width=W)
            depth_r.enable_depth_rendering()
            seg_r = mujoco.Renderer(m, height=H, width=W)
            seg_r.enable_segmentation_rendering()
        # bench protocol: first drive the camera along its view ray to ~DIST from the object
        T0 = obs["camera_pose"].reshape(4, 4).astype(np.float64)
        p0 = obs["object_pose"].reshape(4, 4)[:3, 3].astype(np.float64)
        q = np.linalg.inv(T0) @ np.append(p0, 1.0)
        if np.linalg.norm(q[:3]) > DIST:
            step = np.eye(4)
            step[:3, 3] = q[:3] * (1 - DIST / np.linalg.norm(q[:3]))
            obs, *_ = env.step(np.append(step.reshape(-1), 0.0))
        T1 = obs["camera_pose"].reshape(4, 4).astype(np.float64)
        p_obj = obs["object_pose"].reshape(4, 4)[:3, 3].astype(np.float64)
        uv_c, _ = project(T1, p_obj)
        # the tap: the visible pixel of the target object closest to its centre's projection (the gripper's own
        # fingers often hide the centre at this range; a user would tap what is visible)
        seg_r.update_scene(d, "egocentric")
        seg = seg_r.render()
        obj_geoms = {g for g in range(m.ngeom)
                     if f"{u.object.name}_object" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")}
        mask = np.isin(seg[:, :, 0], list(obj_geoms)) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        mask = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0       # not on the silhouette edge
        if not mask.any():
            occluded += 1
            continue
        vs, us = np.nonzero(mask)
        k = np.argmin((us - uv_c[0]) ** 2 + (vs - uv_c[1]) ** 2)
        uv1 = np.array([us[k], vs[k]], dtype=float)
        depth_r.update_scene(d, "egocentric")
        z = depth_r.render()[int(uv1[1]), int(uv1[0])]
        o1, d1 = ray(T1, uv1)
        cos = -(np.linalg.inv(T1)[:3, :3] @ d1)[2]              # depth buffer is along the optical axis
        truth = o1 + d1 * z / cos

        img1 = obs["rgb_ego"].copy()
        if MODE == "pair":
            step = np.eye(4)
            step[0, 3] = BASELINE                               # one move, 8 cm to the camera's right
            obs2, *_ = env.step(np.append(step.reshape(-1), 0.0))
            T2 = obs2["camera_pose"].reshape(4, 4).astype(np.float64)
            uv2 = match(img1, obs2["rgb_ego"], uv1, T1, T2)
            if uv2 is None:
                misses += 1
                continue
            views = [(T1, uv1), (T2, uv2)]
        else:
            # track the tapped point through N small moves (pyramidal Lucas-Kanade), triangulate over all views
            views, prev, pt, lost = [(T1, uv1)], cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), uv1.astype(np.float32), False
            for _ in range(N_STEPS):
                step = np.eye(4)
                step[0, 3] = BASELINE / N_STEPS
                obs2, *_ = env.step(np.append(step.reshape(-1), 0.0))
                cur = cv2.cvtColor(obs2["rgb_ego"], cv2.COLOR_RGB2GRAY)
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pt.reshape(1, 1, 2), None, winSize=(31, 31), maxLevel=3)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, winSize=(31, 31), maxLevel=3)
                if not (st[0, 0] and st2[0, 0]) or np.linalg.norm(back.reshape(2) - pt) > 0.5:   # forward-backward check
                    lost = True
                    break
                pt, prev = nxt.reshape(2), cur
                views.append((obs2["camera_pose"].reshape(4, 4).astype(np.float64), pt.astype(float)))
            if lost:
                misses += 1
                continue
        # what the robot believes about the camera poses: kinematics (or phone tracking) with an error per view
        B = [(perturb(T, rng), uv) for T, uv in views]
        B1 = B[0][0]
        est = triangulate_n([ray(T, uv) for T, uv in B])
        # the robot only needs the point relative to its own (believed) first pose: compare in that frame
        e = np.linalg.norm((np.linalg.inv(B1) @ np.append(est, 1))[:3] - (np.linalg.inv(T1) @ np.append(truth, 1))[:3])
        errs.append(e)
        dists.append(z / cos)
        print(f"ep {ep:3d}  dist {z / cos:5.2f} m  error {e * 100:5.2f} cm  {info_name(u)}", flush=True)
    errs = np.array(errs)
    txt = (f"{len(errs)} matched of {n} scenes ({misses} not found again, {occluded} objects hidden behind the gripper), "
           f"distance {np.median(dists):.2f} m median, baseline {BASELINE * 100:.0f} cm, mode {MODE}\n"
           f"3D error of the triangulated tap: median {np.median(errs) * 100:.2f} cm, mean {errs.mean() * 100:.2f} cm, "
           f"90th percentile {np.percentile(errs, 90) * 100:.2f} cm, worst {errs.max() * 100:.2f} cm\n")
    print(txt)
    tag = os.environ.get("TRI_TAG", "")
    with open(os.path.join(HERE, "results", f"triangulate{tag}.txt"), "w") as f:
        f.write(__doc__ + f"\npose error per view {T_SD * 1000:.1f} mm / {R_SD:.2f} deg\n" + txt)


def info_name(u):
    o = getattr(u, "object", None)
    return getattr(o, "name", "") if o is not None else ""


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
