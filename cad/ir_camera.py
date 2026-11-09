"""Infrared wrist camera in place of the iPhone: where its lens must sit, and a printed insert that puts it there.

uv run python cad/ir_camera.py          # -> print/ir_cam_insert.stl, out/ir_camera_report.txt, out/ir_camera_mount.png
uv run python cad/ir_camera.py --align  # re-place CAP's printed STLs in EgoGym's gripper frame (prints CAP_PARTS)

Why: CAP's pick skill tolerates a different lens (40-90 deg vertical field of view: 72-84 % in sim) but not a lens in a
different place (2 cm to the side: 42 %; experiments/cap_zero_shot/run_camswap.sh). So the insert copies the iPhone
13 Pro's outline where CAP's printed holder grips the phone, and the camera's lens sits where the phone's main lens
was. CAP's own holder and top plate stay unchanged.

Frames (mm):
  gripper  EgoGym's CAP model ("base" body): x right (seen from behind), y forward (grasp centre at y = 140), z up.
           The iPhone's camera ("egocentric"): pos (0, -36, 100), rotated 75 deg about x, so it looks 15 deg below
           the gripper's forward axis. fovy 53.1 deg, 960 x 720.
  phone    the gripper frame rotated -75 deg about x, so the phone lies flat: x along its length (lens at x = 0),
           y across it (top edge y = 100.27, bottom 27.64), z through it (camera side 57.86 incl. the camera bump,
           body 62.57 ... 70.22 = 7.65 mm). The lens point is (0, 87.28, 60.66). All geometry below is built here.

Camera: Arducam B0332 (OV9281, 1 MP mono global shutter, USB UVC, no IR filter, 38 x 38 mm board, M12 lens). The
values marked MEASURE come from listings, not from a board in hand.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from build123d import Align, Box, Cylinder, Location, Pos, export_stl

ROOT = Path(__file__).resolve().parents[1]
EGO_MESH = ROOT / "third_party/EgoGym/egogym/assets/embodiments/cap/meshes"
MIN, CEN = Align.MIN, Align.CENTER

# ------------------------------------------------------------------ the target: the iPhone's lens (EgoGym model)
TILT = 75.0                                     # deg about x, camera_body euler 1.309 rad
CAM_GRIPPER = np.array([0.0, -36.0, 100.0])     # lens point in the gripper frame
PHONE = dict(x0=-31.77, x1=114.90, y0=27.64, y1=100.27, z_cam=57.86, z_back=62.57, z_screen=70.22)
LENS = np.array([0.0, 87.28, 60.66])            # the same point in the phone frame
FOVY, ASPECT = 53.1, 4 / 3                      # CAP's camera

# CAP's printed robot-gripper parts (third_party/cap_gripper, from nyu-gripper.pages.dev), placed in the gripper frame
# by ICP against EgoGym's gripper mesh (median error 0.5 mm top plate, 0.9 mm phone holder; --align recomputes).
# The phone leans on the holder's strut (x -13..17, y 42..50) and the top plate's post (x 15..37, y 20..86) with its
# camera side, and stands in the plate's cradle (x 17..45 at the bottom edge). Behind the lens nothing is in the way.
# (EgoGym's own gripper mesh has an extra clip at the phone's top edge right of the lens, x 3..21; the printed robot
# gripper doesn't. EgoGym's phone also sinks ~1-2 mm into the printed holder: its model is approximate. So the insert
# copies the phone's outline and the lens offset from the phone's edges, and the printed holder seats it like a phone.)
CAP_PARTS = {
    "top-plate": [[1.0, 0.00018, 0.00022, -91.58766], [-0.00018, 1.0, -0.00022, -87.68909],
                  [-0.00022, 0.00022, 1.0, 24.03208], [0, 0, 0, 1]],
    "top-phone-holder": [[-0.99964, -0.02161, 0.01576, 0.81382], [-0.02144, 0.99972, 0.0105, -74.05555],
                         [-0.01598, 0.01015, -0.99982, 61.01395], [0, 0, 0, 1]],
}
CAP_STL = ROOT / "third_party/cap_gripper"
POST_WINDOW = (25.0, 80.5)      # small window in the top plate's post: a cable tie through it holds the insert
CRADLE_LIP = ((27.4, 63.6), (34.8, 70.8))   # (y, z) at both ends of the lip's inner face, x 17.6 .. 45.5

# ------------------------------------------------------------------ the camera board
PCB_W, PCB_T = 38.0, 1.6
HOLES = 34.0                    # M2 pattern (the board also has a 28 mm one)
HOLDER = 16.5                   # square M12 lens-holder base + clearance (MEASURE)
LENS_BARREL = 14.5              # hole for the M12 lens barrel
PUPIL_ABOVE_PCB = 9.0           # PCB front face -> the lens's entrance pupil (MEASURE: depends on the lens)
LED_D = 5.2                     # 5 mm 940 nm LEDs (e.g. Vishay TSAL6400), pressed in
LED_XY = [(-24.0, LENS[1] - 11.0), (-24.0, LENS[1] + 9.0)]   # left of the lens, clear of the post on its right
X_END = 50.0                    # the insert ends just past the cradle (x 45.5); the phone went on to x 114.9


def pcb_z():
    """PCB front face (phone frame z) that puts the lens pupil where the iPhone's lens was."""
    want = LENS[2] + PUPIL_ABOVE_PCB
    return want, want


def insert():
    """The printed part, phone frame."""
    zb, zs = PHONE["z_back"], PHONE["z_screen"]
    z_pcb, _ = pcb_z()
    slab = Pos(PHONE["x0"], PHONE["y0"], zb) * Box(X_END - PHONE["x0"], PHONE["y1"] - PHONE["y0"], zs - zb,
                                                     align=(MIN, MIN, MIN))
    part = slab
    cx, cy = LENS[0], LENS[1]
    # the board overhangs the phone's top edge by ~6 mm: the slab grows a brow there, full thickness
    top = cy + PCB_W / 2 + 2.0
    part += Pos(cx - PCB_W / 2 - 2.0, PHONE["y1"] - 1, zb) * Box(PCB_W + 4, top - PHONE["y1"] + 1, zs - zb,
                                                                   align=(MIN, MIN, MIN))
    # The board sits at the depth that puts its lens pupil on the iPhone's: on four bosses in a pocket when that is
    # inside the slab (the default lens), on bosses above the screen face with a fence when it is behind it.
    BOSS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
    floor = z_pcb - 1.5                               # 1.5 mm under the board for its front-side parts
    if floor < zs:
        part -= Pos(cx, cy, floor) * Box(PCB_W + 0.8, PCB_W + 0.8, zs - floor + 1, align=(CEN, CEN, MIN))
    for sx, sy in BOSS:
        b0 = min(floor, zs - 3)
        part += Pos(cx + sx * HOLES / 2, cy + sy * HOLES / 2, b0) * Cylinder(3.0, z_pcb - b0, align=(CEN, CEN, MIN))
    if z_pcb + PCB_T > zs:
        h = z_pcb + PCB_T - zs
        part += Pos(cx - PCB_W / 2 - 2.0, cy - PCB_W / 2, zs) * Box(2.0, PCB_W, h, align=(MIN, MIN, MIN))
        part += Pos(cx - PCB_W / 2, cy - PCB_W / 2 - 2.0, zs) * Box(PCB_W, 2.0, h, align=(MIN, MIN, MIN))
    # holes: lens holder + barrel through the slab, M2 pilot holes, LEDs, a cable slot to the left
    part -= Pos(cx, cy, zb - 1) * Box(HOLDER, HOLDER, zs - zb + 2, align=(CEN, CEN, MIN))
    part -= Pos(cx, cy, zb - 1) * Cylinder(LENS_BARREL / 2, zs - zb + 2, align=(CEN, CEN, MIN))
    for sx, sy in BOSS:
        part -= Pos(cx + sx * HOLES / 2, cy + sy * HOLES / 2, zs - 4) * Cylinder(0.8, z_pcb - zs + 5, align=(CEN, CEN, MIN))
    for x, y in LED_XY:
        part -= Pos(x, y, zb - 1) * Cylinder(LED_D / 2, zs - zb + 2, align=(CEN, CEN, MIN))
    # LED and USB wires leave to the left, in a groove on the screen side
    part -= Pos(PHONE["x0"] - 1, cy - 4, zs - 2.5) * Box(cx - PHONE["x0"] - PCB_W / 2 + 1, 8, 3, align=(MIN, MIN, MIN))
    for dx in (-5.0, 5.0):          # cable-tie holes either side of the post's window
        part -= Pos(POST_WINDOW[0] + dx, POST_WINDOW[1], zb - 1) * Box(2.5, 4.5, zs - zb + 2, align=(CEN, CEN, MIN))
    # the cradle's lip leans over the bottom edge on the screen side (y 27.6 at z 63.8 -> y 34.6 at z 70.6):
    # chamfer that corner, so the insert seats on the camera-side corner as the phone does
    (y0, z0), (y1, z1) = CRADLE_LIP
    th = math.atan2(-(y1 - y0), z1 - z0)                 # plane normal, pointing into the slab
    n = np.array([math.cos(th), math.sin(th)])
    c = np.array([y0, z0]) - 10 * n
    part -= Location((33.0, c[0], c[1]), (math.degrees(th), 0, 0)) * Box(36, 20, 40)
    return part


# ------------------------------------------------------------------ checks against CAP's own model
def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def to_phone(p_gripper):
    return np.asarray(p_gripper) @ rot_x(-TILT).T


def to_gripper(p_phone):
    return np.asarray(p_phone) @ rot_x(TILT).T


def cap_parts_phone_frame(n=150000):
    """Surface points of CAP's printed top plate and phone holder, phone frame."""
    import trimesh
    pts = []
    for nm, T in CAP_PARTS.items():
        m = trimesh.load(CAP_STL / f"{nm}.stl")
        m.apply_transform(np.array(T))
        pts.append(to_phone(m.sample(n, seed=4)))
    return np.vstack(pts)


def align(names=tuple(CAP_PARTS)):
    """Place CAP's printed STLs in EgoGym's gripper frame: point-to-point ICP against its gripper mesh from the 24
    axis-aligned orientations and a few offsets, the best by median distance. ~1 min per part. Prints CAP_PARTS."""
    import itertools
    import trimesh
    from scipy.spatial import cKDTree
    G = trimesh.load(EGO_MESH / "gripper_body.stl").sample(60000, seed=1)
    tree = cKDTree(G)
    rots = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, k in enumerate(perm):
                R[i, k] = sg[i]
            if np.linalg.det(R) > 0:
                rots.append(R)

    def icp(P, R, t, it):
        for _ in range(it):
            dist, idx = tree.query(P @ R.T + t)
            keep = dist < np.percentile(dist, 90)
            A, B = P[keep], G[idx[keep]]
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            t = cb - ca @ R.T
        return R, t, float(np.median(tree.query(P @ R.T + t)[0]))
    out = {}
    for nm in names:
        P = trimesh.load(CAP_STL / f"{nm}.stl").sample(4000, seed=2)
        offs = [np.zeros(3)] + [np.array(o) * 25.0 for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
        best = min((icp(P, R0, G.mean(0) - P.mean(0) @ R0.T + o, 12) for R0 in rots for o in offs), key=lambda r: r[2])
        R, t, med = icp(P, best[0], best[1], 80)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        out[nm] = T.round(5).tolist()
        print(f"{nm}: median {med:.2f} mm\n  {out[nm]}")
    return out


def interference(stl):
    """Points of CAP's printed top plate and phone holder that lie inside the printed insert."""
    import trimesh
    G = cap_parts_phone_frame()
    ins = trimesh.load(stl)
    lo, hi = ins.bounds
    near = G[np.all((G > lo - 0.5) & (G < hi + 0.5), axis=1)]
    if not len(near):
        return 0, None
    # inside test by ray parity along +z, only for the points in the insert's bounding box (no rtree needed)
    tri = ins.triangles
    inside = []
    for p in near:
        a, b, c = tri[:, 0, :2] - p[:2], tri[:, 1, :2] - p[:2], tri[:, 2, :2] - p[:2]
        cr = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        w0, w1, w2 = cr(b, c), cr(c, a), cr(a, b)
        hitm = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        s = w0 + w1 + w2
        ok = hitm & (np.abs(s) > 1e-12)
        z = (w0[ok] * tri[ok, 0, 2] + w1[ok] * tri[ok, 1, 2] + w2[ok] * tri[ok, 2, 2]) / s[ok]
        if (z > p[2]).sum() % 2 == 1:
            inside.append(p)
    inside = np.array(inside)
    return len(inside), (inside.min(0), inside.max(0)) if len(inside) else None


def view_occlusion(back):
    """Share of the camera's 224 x 224 policy view covered by CAP's own parts, with the lens moved back."""
    import trimesh
    gb = trimesh.load(EGO_MESH / "gripper_body.stl")
    P = gb.sample(200000, seed=5)
    R = rot_x(TILT)                                     # camera axes in the gripper frame (MuJoCo: looks along -z)
    right, up, fwd = R[:, 0], R[:, 1], -R[:, 2]
    c = CAM_GRIPPER - fwd * back
    d = P - c
    zc = d @ fwd
    ok = zc > 1.0
    f = 360 / math.tan(math.radians(FOVY / 2))          # pixels at 960 x 720
    u = 480 + f * (d[ok] @ right) / zc[ok]
    v = 360 - f * (d[ok] @ up) / zc[ok]
    # CAP feeds a centre square, resized to 224: the middle 720 x 720
    m = (u > 120) & (u < 840) & (v > 0) & (v < 720)
    grid = np.zeros((72, 72), bool)
    grid[(v[m] // 10).astype(int), ((u[m] - 120) // 10).astype(int)] = True
    return grid.mean()


def finger_parallax(shift=3.0):
    """Image shift (px at 960 x 720) of the near and far ends of the fingers when the lens moves sideways."""
    f = 360 / math.tan(math.radians(FOVY / 2))
    near, far = np.array([25, 40, 15]), np.array([45, 130, 15])
    dn, df = np.linalg.norm(near - CAM_GRIPPER), np.linalg.norm(far - CAM_GRIPPER)
    return f * shift / dn, f * shift / df, dn, df


def lens_choice():
    """OV9281: 1280 x 800, 3 um pixels -> 3.84 x 2.40 mm. Longest focal length that still covers CAP's view."""
    w, h = 3.84, 2.40
    f_v = (h / 2) / math.tan(math.radians(FOVY / 2))
    hfov = 2 * math.degrees(math.atan(math.tan(math.radians(FOVY / 2)) * ASPECT))
    f_h = (w / 2) / math.tan(math.radians(hfov / 2))
    stock_v = 2 * math.degrees(math.atan((h / 2) / ((w / 2) / math.tan(math.radians(35)))))
    return f_v, f_h, hfov, stock_v


def illumination(d_m, n=2, ie_mw_sr=40.0):
    """On-axis irradiance (W/m2) from n LEDs of ie mW/sr each (TSAL6400 typical at 100 mA)."""
    return n * ie_mw_sr / 1000 / d_m ** 2


def report():
    z_pcb, want = pcb_z()
    back = z_pcb - want
    f_v, f_h, hfov, stock_v = lens_choice()
    pn, pf, dn, df = finger_parallax()
    lines = [
        "IR wrist camera in place of the iPhone (cad/ir_camera.py)",
        "",
        "Target (EgoGym's CAP model, gripper frame, mm):",
        f"  lens point {CAM_GRIPPER.tolist()}, looking 15 deg below the gripper's forward axis, fovy {FOVY} deg, 4:3",
        f"  on the 13 Pro: {LENS[0] - PHONE['x0']:.1f} mm from the phone's end, {PHONE['y1'] - LENS[1]:.1f} mm below its top edge,",
        f"  {LENS[2] - PHONE['z_cam']:.1f} mm behind the front of the camera bump",
        "",
        "Board placement:",
        f"  PCB front at z = {z_pcb:.1f} (screen face {PHONE['z_screen']}: " +
        ("in a pocket in the insert" if z_pcb < PHONE["z_screen"] else "on bosses behind it") + "), so the lens pupil",
        f"  ({PUPIL_ABOVE_PCB} mm above the PCB: MEASURE on the lens used, then change PUPIL_ABOVE_PCB) is on the iPhone's",
        "  CAP's printed parts leave the space behind the lens free; tolerances: experiments/cap_zero_shot/run_cammount.sh",
        "",
        "Lens (OV9281 1/4\", 1280 x 800):",
        f"  to cover CAP's view: f <= {min(f_v, f_h):.2f} mm (vertical {f_v:.2f}, horizontal {f_h:.2f} for {hfov:.1f} deg)",
        f"  stock 70 deg (H) lens: {stock_v:.0f} deg vertical (sim: 40 deg vertical still 81 %); a ~2.1 mm M12 lens covers",
        "  the full view, then undistort and crop to a virtual 53.1 deg pinhole camera in software",
        "",
        "Alignment check on the real camera:",
        f"  fingers are {dn:.0f}-{df:.0f} mm from the lens; 3 mm of lens offset moves them {pf:.0f}-{pn:.0f} px at 960 x 720",
        f"  (parallax {pn - pf:.0f} px between near and far ends, which a tilt can't fake). Fit the tilt in software,",
        "  then the residual finger offset against EgoGym's rendering should stay under ~10 px",
        "",
        "Infrared light (2x 940 nm LEDs, ~40 mW/sr each at 100 mA, +-25 deg):",
        *[f"  {d:.1f} m: {illumination(d):.2f} W/m2" for d in (0.2, 0.3, 0.5, 1.0)],
        "  (office light of ~300 lux carries roughly 1 W/m2 of visible light; the OV9281 is less sensitive at 940 nm",
        "  than at 850 nm, check its curve. 940 nm is invisible; 850 nm LEDs glow faintly red)",
        "",
        "CAP's own parts in the policy's view (share of the 224 x 224 input):",
        *[f"  lens {b:4.1f} mm back: {100 * view_occlusion(b):4.1f} %" for b in (0.0, 3.0, 5.0, 10.0)],
    ]
    return "\n".join(lines)


def figure(stl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    ph = trimesh.load(EGO_MESH / "iphone.stl")
    ins = trimesh.load(stl)
    G, Ph, I = to_gripper(cap_parts_phone_frame(15000)), ph.sample(8000, seed=1), to_gripper(ins.sample(12000, seed=1))
    R = rot_x(TILT)
    fwd = -R[:, 2]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, (i, j, t) in zip(ax, ((1, 2, "side (y forward, z up)"), (0, 2, "from behind (x right, z up)"))):
        a.scatter(G[:, i], G[:, j], s=0.2, c="0.6", label="CAP top plate + phone holder (printed)")
        a.scatter(Ph[:, i], Ph[:, j], s=0.2, c="0.2", alpha=0.25, label="iPhone 13 Pro (removed)")
        a.scatter(I[:, i], I[:, j], s=0.3, c="tab:orange", label="printed insert")
        a.plot(CAM_GRIPPER[i], CAM_GRIPPER[j], "m*", ms=14)
        a.annotate("iPhone lens = target", (CAM_GRIPPER[i], CAM_GRIPPER[j]), (8, 8), textcoords="offset points",
                   color="m", fontsize=9)
        a.plot([CAM_GRIPPER[i], CAM_GRIPPER[i] + 80 * fwd[i]], [CAM_GRIPPER[j], CAM_GRIPPER[j] + 80 * fwd[j]], "m-")
        a.set_title(t)
        a.set_aspect("equal")
        a.grid(alpha=0.3)
    ax[0].legend(markerscale=15, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=90)


if __name__ == "__main__":
    if "--align" in sys.argv:
        align()
        sys.exit()
    part = insert()
    stl = ROOT / "print" / "ir_cam_insert.stl"
    # print camera side down (z_back on the bed), bosses up: no supports
    export_stl(Location((0, 0, -PHONE["z_back"])) * part, str(stl), tolerance=0.02, angular_tolerance=0.15)
    import trimesh
    m = trimesh.load(stl)
    m.apply_translation((0, 0, PHONE["z_back"]))
    tmp = ROOT / "out" / "_ir_insert_phoneframe.stl"
    m.export(tmp)
    n, box = interference(tmp)
    txt = report() + "\n\nInterference with CAP's printed top plate and holder (their surface points inside the insert): " + \
        (f"{n}" + (f", region {box[0].round(1).tolist()} .. {box[1].round(1).tolist()}" if box else ""))
    txt += f"\nInsert: {m.extents.round(1).tolist()} mm, {m.volume / 1000:.1f} cm3 (~{m.volume / 1000 * 1.27:.0f} g PETG solid)"
    figure(tmp, ROOT / "out" / "ir_camera_mount.png")
    tmp.unlink()
    (ROOT / "out" / "ir_camera_report.txt").write_text(txt + "\n")
    print(txt)
