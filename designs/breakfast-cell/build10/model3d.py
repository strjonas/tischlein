"""Detailed 3D model of build10: every bought or printed part at its catalogue size,
placed where it would be bolted, in the large (90 cm) or small fridge cell.

Used for the geometry checks of the whole morning (sequence.py) and of the shelf
coverage (coverage in sequence.py), and as the base of build11's model.

Sizes: STS3215 45.2 x 24.7 x 35 mm, shaft about 11 mm from one end; 2040 V-slot;
V-wheels 24 mm; 608 bearings; GT2 20T (r 6.4 mm) and printed 60T (r 19.1 mm) and
48T (r 15.3 mm) pulleys; 20 x 20 aluminium tube; Weck 580 ml tapered jar. Printed
parts are blocks of the right size, not CAD. Not a dynamic model.

Coordinates: fridge right edge x=0, front face y=0, depth +y,
board top (the fridge's floor) z=0. Metres.
"""
import math
import numpy as np

# ------------------------------------------------------------------ robot layout
S = np.array([.06, -.02])          # shoulder axis (the mast spot)
L1 = L2 = .35
MAST = dict(x=.06, y=.0365, half=.02, top=1.50)      # two 2040 face to face = 40 x 40
UPPER_DZ = .03                     # upper arm runs 30 mm above the forearm (the link plane)
HAND_DZ = -.008                    # wrist axes 8 mm below the link plane
LIFT = (.110, 1.39)                # link-plane range set by the foot, the top idler and the counterweight's travel
CW_C = 1.530                       # counterweight centre height = CW_C - lift (it rides the belt's back run)
# Tool points in the yaw frame (pitch = roll = 0): wrist yaw axis at the origin, x = heading.
YAW_TO_ROLL, ROLL_TO_PITCH = .059, .0185
JAR_IN_PITCH = .1115               # jar axis ahead of the pitch axis (palm face + jar radius + 4 mm)
TOOL = dict(jar=np.array([YAW_TO_ROLL + ROLL_TO_PITCH + JAR_IN_PITCH, 0, HAND_DZ]),
            bar=np.array([YAW_TO_ROLL + ROLL_TO_PITCH + .150, .0375, HAND_DZ]),     # handle bar, at the fingertips
            tip=np.array([YAW_TO_ROLL + ROLL_TO_PITCH + JAR_IN_PITCH + .058, .0535, HAND_DZ]))   # fixed fingertip
GRIP = dict(jar=0.0, bar=.63, open=-.45, tip=.63)   # jaw angle: 0 closed on the jar, + closes further
JAR = dict(r_bot=.043, r_top=.050, h=.11, grip=.055, m=.75)
JOINTS = ['lift', 'shoulder', 'elbow', 'yaw', 'roll', 'pitch', 'grip']
LIMITS = dict(lift=LIFT, shoulder=(-math.radians(175), math.radians(175)), elbow=(-math.radians(152), math.radians(152)),
              yaw=(-math.radians(170), math.radians(170)), roll=(-math.radians(150), math.radians(150)),
              pitch=(-math.radians(95), math.radians(95)), grip=(-.70, .75))

# ------------------------------------------------------------------ parts (BOM ids from bom.csv)
PARTS = {
    'servo': ('E01', 'Feetech STS3215-C018 servo'),
    'adapter': ('E02', 'Waveshare bus servo adapter'),
    'psu': ('R01', 'Mean Well LRS-150-12 supply, printed terminal cover'),
    'bearing': ('A01', '608-2RS bearing'),
    'bolt': ('A02', 'M8 x 70 bolt (joint axle)'),
    'belt_joint': ('A03', 'GT2 200 mm closed belt, glass fibre (3:1 joint drive)'),
    'pulley20': ('A04', 'GT2 20T pulley'),
    'estop': ('A06', 'Latching e-stop on the 12 V servo supply'),
    'camera': ('A07', 'Wrist camera, 32 x 32 mm, 130 deg'),
    'scene_cam': ('A07', 'Scene camera (second InnoMaker camera) in a printed case'),
    'print': ('A08', 'Printed PETG part'),
    'tube': ('H01', 'Aluminium tube 20 x 20 x 1.5'),
    'vslot': ('A11', 'V-slot 2040, 1500 mm'),
    'gantry': ('A12', 'Gantry plate with POM V-wheels'),
    'belt_lift': ('A04', 'GT2 6 mm belt from the pulley kit, lift loop'),
    'nuts': ('A14', 'M5 hammer nuts'),
    'vhb': ('A17', '3M VHB tape (aids, mast brace)'),
    'board': ('H02', 'Spruce board 800 x 600 x 18 under the fridge'),
    'bracket': ('H03', 'Steel angle bracket 60 x 60 x 45 (mast foot)'),
    'cw': ('A15', 'Cast-iron weight plate 1.25 kg, counterweight on the lift belt'),
    'jar': ('kitchen', 'Weck 580 ml tapered jar, glass lid'),
    'fridge': ('owned', 'Fridge 60 x 60 x 90 cm'),
    'microwave': ('owned', 'Microwave 44.5 x 36 x 26 cm'),
}

MAT = {
    'alu': ('0.78 0.80 0.83 1', .6, .5), 'vslot': ('0.72 0.74 0.77 1', .5, .4), 'slot': ('0.25 0.26 0.28 1', .1, .1),
    'petg': ('0.93 0.45 0.12 1', .3, .2), 'petg2': ('0.20 0.55 0.62 1', .3, .2),
    'servo': ('0.10 0.10 0.11 1', .3, .2), 'belt': ('0.05 0.05 0.05 1', .1, .1), 'steel': ('0.55 0.57 0.60 1', .8, .6),
    'wheel': ('0.12 0.12 0.13 1', .2, .1), 'bearing': ('0.70 0.72 0.75 1', .8, .6), 'wood': ('0.86 0.70 0.48 1', .1, .1),
    'fridge': ('0.95 0.95 0.96 1', .3, .2), 'fridge_in': ('0.93 0.95 0.97 1', .2, .1), 'glass_shelf': ('0.75 0.88 0.95 0.35', .9, .3),
    'rack': ('0.88 0.92 0.96 0.85', .3, .2), 'mw': ('0.22 0.23 0.25 1', .4, .3), 'mw_in': ('0.80 0.80 0.78 1', .2, .1),
    'mw_glass': ('0.08 0.08 0.10 0.85', .9, .5), 'button': ('0.85 0.85 0.85 1', .3, .2), 'red': ('0.85 0.10 0.10 1', .3, .2),
    'yellow': ('0.95 0.80 0.10 1', .3, .2), 'jar_glass': ('0.80 0.92 0.95 0.35', .9, .5), 'food': ('0.85 0.55 0.20 1', .1, .1),
    'plate': ('0.85 0.92 0.95 0.6', .9, .4), 'wall': ('0.94 0.93 0.90 1', .05, .05), 'floor': ('0.70 0.64 0.56 1', .1, .1),
    'pcb': ('0.10 0.45 0.20 1', .2, .2), 'psu': ('0.70 0.72 0.74 1', .7, .4), 'cable': ('0.08 0.08 0.08 1', .1, .1),
    'bottle': ('0.30 0.60 0.35 0.9', .4, .3), 'milk': ('0.97 0.97 0.95 1', .2, .1), 'veg': ('0.45 0.70 0.25 1', .1, .1),
    'cheese': ('0.95 0.85 0.40 1', .1, .1),
}

# Collision classes: (contype, conaffinity). env never touches env; fingers and the jar
# ignore handles and each other (they hold them); the robot checks itself for non-adjacent links.
CLS = dict(env=(1, 6), robot=(2, 7), finger=(8, 1), handle=(16, 2), jar=(4, 3), vis=(0, 0))
ROBOT_MARGIN = .01


def _f(v):
    return ' '.join(f'{x:.5f}' for x in np.atleast_1d(v))


def quat_z(a):
    return np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])


AX = dict(x=np.array([.70710678, 0, .70710678, 0]), y=np.array([.70710678, -.70710678, 0, 0]), z=np.array([1., 0, 0, 0]))


class Body:
    def __init__(self, name, pos=(0, 0, 0), quat=None, joint=None, free=False):
        self.name, self.pos, self.quat, self.joint, self.free = name, np.asarray(pos, float), quat, joint, free
        self.geoms, self.children = [], []

    def add(self, kind, name, pos, size, mat, part, cls='robot', quat=None, group=0):
        self.geoms.append(dict(type=kind, name=name, pos=np.asarray(pos, float), size=np.atleast_1d(np.asarray(size, float)),
                               mat=mat, part=part, cls=cls, quat=quat, group=group))
        return self

    def box(self, name, lo, hi, mat, part, cls='robot', group=0):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return self.add('box', name, (lo + hi) / 2, np.maximum((hi - lo) / 2, 5e-4), mat, part, cls, group=group)

    def obox(self, name, centre, half, angle, mat, part, cls='robot', group=0):
        return self.add('box', name, centre, half, mat, part, cls, quat=quat_z(angle), group=group)

    def cyl(self, name, centre, r, half, mat, part, cls='robot', axis='z', group=0):
        return self.add('cylinder', name, centre, (r, half), mat, part, cls, quat=AX[axis], group=group)

    def child(self, body):
        self.children.append(body)
        return body

    def xml(self, ind=1):
        p = '  ' * ind
        q = f' quat="{_f(self.quat)}"' if self.quat is not None else ''
        out = [f'{p}<body name="{self.name}" pos="{_f(self.pos)}"{q}>']
        if self.free:
            out.append(f'{p}  <freejoint name="{self.name}"/>')
        if self.joint:
            j = self.joint
            rng = f' range="{j["range"][0]:.4f} {j["range"][1]:.4f}" limited="true"' if 'range' in j else ''
            out.append(f'{p}  <joint name="{j["name"]}" type="{j["type"]}" axis="{_f(j["axis"])}" pos="{_f(j.get("pos", (0, 0, 0)))}"{rng}/>')
        for g in self.geoms:
            ct, ca = CLS[g['cls']]
            q = f' quat="{_f(g["quat"])}"' if g['quat'] is not None else ''
            mg = f' margin="{ROBOT_MARGIN}"' if g['cls'] in ('robot', 'finger', 'jar') else ''
            out.append(f'{p}  <geom name="{g["name"]}" type="{g["type"]}" pos="{_f(g["pos"])}" size="{_f(g["size"])}"{q} '
                       f'material="{g["mat"]}" contype="{ct}" conaffinity="{ca}" group="{g["group"]}"{mg}/>')
        for c in self.children:
            out.append(c.xml(ind + 1))
        out.append(f'{p}</body>')
        return '\n'.join(out)


def belt(body, name, c1, r1, c2, r2, z, half_h=.003, part='belt_joint'):
    """Open-belt runs between two pulleys (the wraps are hidden by the pulleys)."""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    d = c2 - c1
    L = np.linalg.norm(d)
    e = d / L
    phi = math.asin((r1 - r2) / L)
    ang = math.atan2(e[1], e[0])
    for k, s in enumerate((1, -1)):
        a = ang + s * (math.pi / 2 + phi)
        n = np.array([math.cos(a), math.sin(a)])
        p1, p2 = c1 + r1 * n, c2 + r2 * n
        mid, seg = (p1 + p2) / 2, p2 - p1
        body.obox(f'{name}_{k}', [mid[0], mid[1], z], [np.linalg.norm(seg) / 2, .00075, half_h],
                  math.atan2(seg[1], seg[0]), 'belt', part, 'vis')


def servo_block(body, name, centre, half, cls='robot'):
    """STS3215 body with its label band."""
    c, h = np.asarray(centre, float), np.asarray(half, float)
    body.box(name, c - h, c + h, 'servo', 'servo', cls)


# ------------------------------------------------------------------ fixtures
def fixture_dims(kind):
    if kind == 'small':
        return dict(W=.47, D=.47, H=.50, shelves=[.05, .21], top_block=(.36, .45), handle_z=.30, crisper=False)
    return dict(W=.60, D=.60, H=.90, shelves=[.20, .37, .54], top_block=(.69, .85), handle_z=.55, crisper=True)


def build_fixture(world, kind):
    f = fixture_dims(kind)
    W, D, H, t, y0 = f['W'], f['D'], f['H'], .04, .08
    E = 'env'
    # board, floor, wall
    world.box('floor', [-1.6, -1.4, -.028], [1.2, 1.0, -.018], 'floor', None, 'vis')
    world.box('board', [.18 - .80, 0, -.018], [.18, .60, 0], 'wood', 'board', E)          # 800 x 600, uncut
    world.box('wall_behind', [-W - .5, D + .03, -.018], [.7, D + .05, 1.8], 'wall', None, E)
    # cabinet
    fr = Body('fridge')
    world.child(fr)
    fr.box('f_left', [-W, 0, 0], [-W + t, D, H], 'fridge', 'fridge', E)
    fr.box('f_right', [-t, 0, 0], [0, D, H], 'fridge', 'fridge', E)
    fr.box('f_back', [-W, D - .05, 0], [0, D, H], 'fridge', 'fridge', E)
    fr.box('f_bottom', [-W, 0, 0], [0, D, .05], 'fridge', 'fridge', E)
    fr.box('f_roof', [-W, 0, H - .05], [0, D, H], 'fridge', 'fridge', E)
    fr.box('f_top_block', [-W + t, 0, f['top_block'][0]], [-t, D - .05, f['top_block'][1]], 'fridge_in', 'fridge', E)
    if f['crisper']:
        fr.box('f_crisper', [-W + t, y0, .05], [-t, D - .05, .195], 'rack', 'fridge', E)
        fr.box('veg1', [-.45, .25, .05], [-.33, .40, .12], 'veg', None, 'vis')
    for z in (f['shelves'][1:] if kind == 'small' else f['shelves']):
        fr.box(f'f_shelf_{z:.2f}', [-W + t, y0, z - .006], [-t, D - .05, z], 'glass_shelf', 'fridge', E)
    # some food on the shelves, away from the stocked bay (visual only)
    zs = f['shelves'][1]
    fr.box('cheese', [-.50, .40, zs], [-.38, .50, zs + .05], 'cheese', None, 'vis')
    fr.cyl('pot', [-.45, .20, zs + .04], .045, .04, 'milk', None, 'vis')
    # door, hinged on the left front corner, opens to the left (axis -z, so q = opening angle)
    door = world.child(Body('f_door', pos=(-W, 0, 0), joint=dict(name='fridge_door', type='hinge', axis=(0, 0, -1),
                                                                    range=(0, math.radians(120)))))
    door.box('f_door_slab', [0, -.05, .01], [W, 0, H], 'fridge', 'fridge', E)
    door.box('f_door_racks', [.04, 0, .06], [W - .04, .07, H - .06], 'rack', 'fridge', E, group=3)
    for k, z in enumerate((.12, .40, .66) if kind == 'large' else (.08, .26)):
        door.box(f'rack{k}', [.04, .0, z], [W - .04, .07, z + .01], 'rack', None, 'vis')
        door.box(f'rack{k}_lip', [.04, .065, z], [W - .04, .07, z + .06], 'rack', None, 'vis')
    for k, x in enumerate((.10, .17)):
        door.cyl(f'bottle{k}', [x, .035, .14 + .11 if kind == 'large' else .10], .03, .11 if kind == 'large' else .06,
                 'bottle', None, 'vis')
    # printed handle pin (VHB pad on the door face near the free edge, short arm, pin hanging
    # down): the hand rises onto it with the lift and lets go by lowering, at any door angle
    hz = f['handle_z']
    door.box('f_pin_pad', [W - .07, -.053, hz + .02], [W - .03, -.05, hz + .065], 'petg2', 'print', 'handle')
    door.box('f_pin_arm', [W - .062, -.108, hz + .03], [W - .038, -.05, hz + .045], 'petg2', 'print', 'handle')
    door.cyl('f_bar', [W - .05, -.095, hz - .0025], .011, .0325, 'petg2', 'print', 'handle')
    # microwave on two battens, right edges aligned with the fridge
    z0 = H + .01
    mw = world.child(Body('microwave'))
    mw.box('batten_l', [-.43, .02, H], [-.40, .34, z0], 'wood', None, 'vis')
    mw.box('batten_r', [-.045, .02, H], [-.015, .34, z0], 'wood', None, 'vis')
    mw.box('m_bottom', [-.445, 0, z0 - .0], [0, .36, z0 + .035], 'mw', 'microwave', E)
    mw.box('m_top', [-.445, 0, z0 + .225], [0, .36, z0 + .26], 'mw', 'microwave', E)
    mw.box('m_left', [-.445, 0, z0], [-.425, .36, z0 + .26], 'mw', 'microwave', E)
    mw.box('m_panel', [-.125, 0, z0], [0, .36, z0 + .26], 'mw', 'microwave', E)
    mw.box('m_back', [-.445, .33, z0], [0, .36, z0 + .26], 'mw', 'microwave', E)
    mw.box('m_cavity', [-.425, .02, z0 + .035], [-.125, .33, z0 + .036], 'mw_in', None, 'vis')
    mw.box('m_panel_face', [-.118, -.002, z0 + .02], [-.007, 0, z0 + .24], 'mw_glass', None, 'vis')
    for k, z in enumerate((.20, .17, .14, .11)):
        mw.box(f'm_key{k}', [-.10, -.006, z0 + z], [-.07, 0, z0 + z + .018], 'button', None, 'vis')
    # door-release key (bottom) and start key, pressed by the fingertip
    mw.box('m_release', [-.105, -.008, z0 + .035], [-.055, 0, z0 + .06], 'button', 'microwave', 'handle')
    mw.box('m_start', [-.055, -.008, z0 + .11], [-.025, 0, z0 + .128], 'yellow', 'microwave', 'handle')
    tt = mw.child(Body('turntable', pos=(-.275, .175, z0 + .035), joint=dict(name='turntable', type='hinge', axis=(0, 0, 1))))
    tt.cyl('m_plate', [0, 0, .008], .125, .007, 'plate', 'microwave', E)
    mdoor = world.child(Body('m_door', pos=(-.445, 0, z0), joint=dict(name='mw_door', type='hinge', axis=(0, 0, -1),
                                                                     range=(0, math.radians(120)))))
    mdoor.box('m_door_slab', [0, -.04, 0], [.32, 0, .26], 'mw', 'microwave', E)
    mdoor.box('m_door_window', [.03, -.0415, .04], [.29, -.04, .22], 'mw_glass', None, 'vis')
    mdoor.box('m_pin_pad', [.2675, -.043, .145], [.3075, -.04, .19], 'petg2', 'print', 'handle')
    mdoor.box('m_pin_arm', [.2755, -.098, .155], [.2995, -.04, .17], 'petg2', 'print', 'handle')
    mdoor.cyl('m_bar', [.2875, -.085, .1225], .011, .0325, 'petg2', 'print', 'handle')
    f.update(z0=z0, mw_centre=np.array([-.275, .175]), plate_z=z0 + .05, mw_top=z0 + .26, t=t, y0=y0)
    return f


# ------------------------------------------------------------------ the robot
def build_robot(world, f):
    E = 'env'
    mx, my, mh, top = MAST['x'], MAST['y'], MAST['half'], MAST['top']
    mast = world.child(Body('mast'))
    # two 2040 rails bolted face to face (front rail y < my, back rail y > my)
    mast.box('rail_front', [mx - mh, my - mh, 0], [mx + mh, my, top], 'vslot', 'vslot', E)
    mast.box('rail_back', [mx - mh, my, 0], [mx + mh, my + mh, top], 'vslot', 'vslot', E)
    for k, (x, y, sx, sy) in enumerate([(mx - .01, my - mh, .003, .0005), (mx + .01, my - mh, .003, .0005),
                                         (mx - mh, my - .01, .0005, .003), (mx - mh, my + .01, .0005, .003),
                                         (mx + mh, my - .01, .0005, .003), (mx + mh, my + .01, .0005, .003),
                                         (mx - .01, my + mh, .003, .0005), (mx + .01, my + mh, .003, .0005)]):
        mast.box(f'slot{k}', [x - sx, y - sy, .002], [x + sx, y + sy, top - .002], 'slot', None, 'vis')
    # foot: a steel angle bracket on the back face and the printed lift-servo mount on the right
    # face, both screwed into the board (no bracket in front, so the board stays flush with the fridge)
    mast.box('bracket_h', [mx - .0225, my + mh, 0], [mx + .0225, my + mh + .06, .003], 'steel', 'bracket', E)
    mast.box('bracket_v', [mx - .0225, my + mh, 0], [mx + .0225, my + mh + .003, .06], 'steel', 'bracket', E)
    # brace: printed block on the back slot of the left face, VHB-taped to the fridge's side wall
    mast.box('brace', [0, my + .001, .78], [mx - mh, my + mh, .84], 'petg', 'print', E)
    mast.box('brace_vhb', [-.0005, my + .001, .78], [.0015, my + mh, .84], 'slot', 'vhb', 'vis')
    # lift belt loop on the right side (plane x = .11): 48T drive pulley at the foot, 48T idler at the top
    bx, rp = .11, 48 * .002 / (2 * math.pi)
    z_lo, z_hi = .040, top - .02
    mast.cyl('lift_pulley', [bx, my, z_lo], rp, .0045, 'petg', 'print', E, axis='x')
    mast.cyl('top_idler', [bx, my, z_hi], rp, .0045, 'petg', 'print', E, axis='x')
    mast.box('top_block', [mx - mh, my - mh - .004, top], [bx + .012, my + mh + .004, top + .025], 'petg', 'print', E)
    mast.box('top_idler_cheek', [bx + .006, my - .02, z_hi - .02], [bx + .012, my + .02, top], 'petg', 'print', E)
    for k, yy in enumerate((my - rp, my + rp)):
        mast.box(f'lift_belt{k}', [bx - .003, yy - .00075, z_lo], [bx + .003, yy + .00075, z_hi], 'belt', 'belt_lift', 'vis')
    # lift servo: shaft along -x into the pulley, body to the right, on a printed foot bracket
    mast.box('lift_servo', [bx + .006, my - .0225, z_lo - .0124], [bx + .041, my + .0226, z_lo + .0124], 'servo', 'servo', E)
    mast.box('lift_servo_mount', [mx + mh, my - .03, 0], [bx + .045, my + .03, z_lo - .0124], 'petg', 'print', E)
    # top: e-stop and scene camera
    mast.box('estop_box', [mx - .034, my - .034, top + .025], [mx + .034, my + .034, top + .093], 'yellow', 'estop', E)
    mast.cyl('estop_knob', [mx, my, top + .103], .02, .01, 'red', 'estop', E)
    mast.box('cam_arm', [mx - .006, my - .14, top + .06], [mx + .006, my - .034, top + .07], 'petg', 'print', E)
    mast.box('scene_cam', [mx - .03, my - .17, top + .045], [mx + .03, my - .14, top + .075], 'servo', 'scene_cam', E)
    # electronics on the board beside the fridge
    mast.box('psu', [.072, .20, 0], [.169, .359, .030], 'psu', 'psu', E)
    mast.box('psu_cover', [.072, .19, 0], [.169, .20, .032], 'petg', 'print', E)
    mast.box('adapter', [.09, .38, 0], [.13, .41, .012], 'pcb', 'adapter', E)
    mast.box('cable_up', [mx + mh + .001, my + .006, .05], [mx + mh + .009, my + .014, .70], 'cable', None, 'vis')

    # counterweight: a 1.25 kg plate clamped on the belt's back run, behind the mast, moving
    # opposite to the carriage (its slide value is set from the lift: CW_C - lift)
    cw = world.child(Body('counterweight', joint=dict(name='cw', type='slide', axis=(0, 0, 1), range=(0, 1.6))))
    cw.cyl('cw_plate', [bx, my + mh + .0115, 0], .075, .0075, 'steel', 'cw', E, axis='y')
    cw.box('cw_clamp', [bx - .012, my + rp - .004, -.02], [bx + .012, my + mh + .004, .02], 'petg', 'print', E)

    # carriage (link-plane height = slide value)
    car = world.child(Body('carriage', pos=(S[0], S[1], 0), joint=dict(name='lift', type='slide', axis=(0, 0, 1), range=LIFT)))
    py0 = (my - .0265) - S[1]                    # plate back face behind the shoulder axis
    car.box('gantry_plate', [-.05, py0, -.05], [.05, py0 + .004, .10], 'alu', 'gantry')
    wy = (my - .01) - S[1]
    for k, (sx, z) in enumerate([(s, z) for s in (-1, 1) for z in (-.035, .02, .075)]):
        x = sx * (mh + .0085)
        car.cyl(f'wheel{k}', [x, wy, z], .012, .005, 'wheel', 'gantry', 'vis', axis='y')
        car.cyl(f'wheel_axle{k}', [x, (py0 + wy) / 2 + .002, z], .0035, (wy - py0) / 2 - .003, 'steel', 'gantry', 'vis', axis='y')
    car.box('belt_clamp', [.044, py0, -.05], [.058, (my - rp + .004) - S[1], -.022], 'petg', 'print')
    # shoulder: printed bracket and bearing housing (two 608) above the upper arm hub
    car.box('shoulder_bracket', [-.022, -.006, .052], [.022, py0, .10], 'petg', 'print')
    car.cyl('shoulder_housing', [0, 0, .076], .02, .024, 'petg', 'print')
    car.cyl('shoulder_bearing_top', [0, 0, .097], .011, .0035, 'bearing', 'bearing', 'vis')
    car.box('shoulder_servo_mount', [.015, -.016, .0886], [.095, .016, .10], 'petg', 'print')
    servo_block(car, 'shoulder_servo', [.0586 + .0116, 0, .0711], [.0226, .0124, .0175])
    car.cyl('shoulder_pulley20', [.0586, 0, .0480], .0064, .0035, 'steel', 'pulley20', 'vis')
    belt(car, 'shoulder_belt', [0, 0], .0191, [.0586, 0], .0064, .0480)

    # upper arm (plane = link plane + 30 mm)
    # shoulder zero points forward (-y), so the joint's end stops face the mast (+y)
    ua = car.child(Body('upper_arm', pos=(0, 0, UPPER_DZ), quat=quat_z(-math.pi / 2),
                        joint=dict(name='shoulder', type='hinge', axis=(0, 0, 1), range=LIMITS['shoulder'])))
    ua.cyl('ua_hub', [0, 0, 0], .021, .014, 'petg', 'print')
    ua.cyl('ua_pulley60', [0, 0, .018], .0191, .0035, 'petg', 'print')
    ua.cyl('shoulder_bolt', [0, 0, .045], .004, .05, 'steel', 'bolt', 'vis')
    ua.box('ua_clamp', [0, -.016, -.014], [.045, .016, .014], 'petg', 'print')
    ua.box('ua_tube', [.045, -.01, -.01], [L1 - .03, .01, .01], 'alu', 'tube')
    ua.box('elbow_block', [L1 - .035, -.017, -.012], [L1, .017, .028], 'petg', 'print')
    ua.cyl('elbow_housing', [L1, 0, .008], .02, .02, 'petg', 'print')
    ua.cyl('elbow_bearing_top', [L1, 0, .029], .011, .0035, 'bearing', 'bearing', 'vis')
    # elbow servo hangs under the upper arm, shaft down, 58.6 mm behind the elbow (200 mm belt)
    es = L1 - .0586
    servo_block(ua, 'elbow_servo', [es - .0116, 0, -.0275], [.0226, .0124, .0175])
    ua.box('elbow_servo_saddle', [es - .04, -.014, -.012], [es + .016, .014, -.010], 'petg', 'print')
    ua.cyl('elbow_pulley20', [es, 0, -.0505], .0064, .0035, 'steel', 'pulley20', 'vis')
    belt(ua, 'elbow_belt', [es, 0], .0064, [L1, 0], .0191, -.0505)

    # forearm (the link plane)
    fa = ua.child(Body('forearm', pos=(L1, 0, -UPPER_DZ), joint=dict(name='elbow', type='hinge', axis=(0, 0, 1),
                                                                    range=LIMITS['elbow'])))
    fa.cyl('fa_hub', [0, 0, -.002], .021, .014, 'petg', 'print')
    fa.cyl('fa_pulley60', [0, 0, -.0205], .0191, .0035, 'petg', 'print')
    fa.cyl('elbow_bolt', [0, 0, .02], .004, .045, 'steel', 'bolt', 'vis')
    fa.box('fa_clamp', [0, -.016, -.014], [.045, .016, .012], 'petg', 'print')
    fa.box('fa_tube', [.045, -.01, -.01], [L2 - .045, .01, .01], 'alu', 'tube')
    fa.box('yaw_mount', [L2 - .05, -.016, -.016], [L2 - .033, .016, .019], 'petg', 'print')
    servo_block(fa, 'yaw_servo', [L2 - .0106, 0, 0], [.0226, .0124, .0175])

    # wrist yaw (vertical axis, shaft down), then roll (along the hand), then pitch (lateral axis)
    yw = fa.child(Body('wrist_yaw', pos=(L2, 0, 0), joint=dict(name='yaw', type='hinge', axis=(0, 0, 1), range=LIMITS['yaw'])))
    yw.cyl('yaw_horn', [0, 0, -.019], .011, .0015, 'steel', 'servo', 'vis')
    yw.box('yaw_bracket', [-.014, -.02, -.0245], [.019, .02, -.0205], 'petg', 'print')
    yw.box('yaw_bracket_front', [.015, -.02, -.0245], [.019, .02, .027], 'petg', 'print')
    servo_block(yw, 'roll_servo', [.037, 0, HAND_DZ + .0115], [.0175, .0124, .0226])
    rl = yw.child(Body('wrist_roll', pos=(YAW_TO_ROLL, 0, HAND_DZ), joint=dict(name='roll', type='hinge', axis=(1, 0, 0),
                                                                             range=LIMITS['roll'])))
    rl.cyl('roll_horn', [-.002, 0, 0], .011, .002, 'steel', 'servo', 'vis')
    rl.box('roll_plate', [0, -.021, -.016], [.004, .021, .036], 'petg', 'print')
    servo_block(rl, 'pitch_servo', [ROLL_TO_PITCH, 0, .0115], [.01235, .0175, .0226])
    pt = rl.child(Body('wrist_pitch', pos=(ROLL_TO_PITCH, 0, 0), joint=dict(name='pitch', type='hinge', axis=(0, -1, 0),
                                                                          range=LIMITS['pitch'])))
    for k, s in enumerate((-1, 1)):
        pt.box(f'pitch_cheek{k}', [-.012, s * .0185 - .0015, -.014], [.03, s * .0185 + .0015, .014], 'petg', 'print')
    pt.box('gripper_frame', [.016, -.035, -.016], [.032, .035, .016], 'petg', 'print')
    servo_block(pt, 'gripper_servo', [.0447, -.0075, 0], [.01235, .0226, .0175])
    pt.box('palm', [.057, -.012, -.015], [.061, .045, .015], 'petg', 'print')
    jc = JAR_IN_PITCH                        # jar axis in this frame: (jc, 0, 0)
    r_mid = JAR['r_bot'] + (JAR['r_top'] - JAR['r_bot']) * JAR['grip'] / JAR['h']
    pt.box('fixed_finger', [.045, r_mid + .002, -.0125], [jc + .058, r_mid + .012, .0125], 'petg', 'print', 'finger')
    pt.obox('fixed_v1', [jc - .030, r_mid - .006, 0], [.012, .004, .0125], math.radians(-35), 'petg2', 'print', 'finger')
    pt.obox('fixed_v2', [jc + .030, r_mid - .006, 0], [.012, .004, .0125], math.radians(35), 'petg2', 'print', 'finger')
    pt.box('wrist_cam_mount', [.022, -.008, .016], [.032, .008, .026], 'petg', 'print')
    pt.box('wrist_cam', [.020, -.016, .026], [.034, .016, .036], 'pcb', 'camera')
    # moving jaw on the gripper servo's shaft (vertical when roll = 0)
    px, py = .0447, -.019
    jw = pt.child(Body('jaw', pos=(px, py, 0), joint=dict(name='grip', type='hinge', axis=(0, 0, 1), range=LIMITS['grip'])))
    jw.cyl('jaw_hub', [0, 0, 0], .009, .0125, 'petg', 'print', 'finger')
    yb = -(r_mid + .012) - py
    jw.box('jaw_link', [0, yb, -.0125], [.02, 0, .0125], 'petg', 'print', 'finger')
    jw.box('jaw_finger', [.015, yb, -.0125], [jc - px + .058, yb + .010, .0125], 'petg', 'print', 'finger')
    jw.box('jaw_pad', [jc - px - .02, yb + .010, -.0125], [jc - px + .02, yb + .012, .0125], 'petg2', 'print', 'finger')
    return dict(mast=mast, carriage=car)


def build_jar(world, pos=(0, 0, 0)):
    jar = world.child(Body('jar', pos=pos, free=True))
    n = 5
    for k in range(n):
        z0, z1 = JAR['h'] * k / n, JAR['h'] * (k + 1) / n
        r = JAR['r_bot'] + (JAR['r_top'] - JAR['r_bot']) * (k + .5) / n
        jar.cyl(f'jar_glass{k}', [0, 0, (z0 + z1) / 2], r, (z1 - z0) / 2, 'jar_glass', 'jar', 'jar')
        if z1 <= .086:
            jar.cyl(f'jar_food{k}', [0, 0, (z0 + z1) / 2], r - .004, (z1 - z0) / 2 - .0005, 'food', None, 'vis')
    jar.cyl('jar_lid', [0, 0, JAR['h'] + .003], JAR['r_top'] + .004, .003, 'jar_glass', 'jar', 'jar')
    return jar


def mjcf(kind='large', offscreen=(1600, 1200)):
    world = Body('world')
    f = build_fixture(world, kind)
    build_robot(world, f)
    build_jar(world, (-.16, .15, f['shelves'][1] + .0005))
    mats = '\n'.join(f'    <material name="{k}" rgba="{v[0]}" specular="{v[1]}" shininess="{v[2]}"/>' for k, v in MAT.items())
    bodies = '\n'.join(c.xml(1) for c in world.children)
    wg = []
    for g in world.geoms:
        ct, ca = CLS[g['cls']]
        wg.append(f'    <geom name="{g["name"]}" type="{g["type"]}" pos="{_f(g["pos"])}" size="{_f(g["size"])}" '
                  f'material="{g["mat"]}" contype="{ct}" conaffinity="{ca}" group="{g["group"]}"/>')
    xml = f"""<mujoco model="build10_{kind}">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 0"/>
  <visual><global offwidth="{offscreen[0]}" offheight="{offscreen[1]}"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/><map znear="0.01"/></visual>
  <asset>
{mats}
  </asset>
  <worldbody>
    <light pos="0.8 -1.6 2.6" dir="-0.35 0.6 -1" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <light pos="-1.4 -1.2 2.2" dir="0.6 0.5 -1" diffuse="0.25 0.25 0.25" castshadow="false"/>
{chr(10).join(wg)}
{bodies}
  </worldbody>
  <contact>
    <exclude body1="carriage" body2="mast"/>
    <exclude body1="counterweight" body2="mast"/>
    <exclude body1="wrist_yaw" body2="wrist_pitch"/>
    <exclude body1="forearm" body2="wrist_roll"/>
    <exclude body1="wrist_roll" body2="jaw"/>
    <exclude body1="wrist_yaw" body2="jaw"/>
  </contact>
</mujoco>"""
    meta = dict(kind=kind, fixture={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in f.items()})
    return xml, meta, f, world


# ------------------------------------------------------------------ kinematics
def planar_ik(Y, L1=L1, L2=L2):
    d = np.asarray(Y[:2]) - S
    c = (d @ d - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(c) > 1:
        return []
    out = []
    for sign in (-1, 1):
        b = sign * math.acos(c)
        a = math.atan2(d[1], d[0]) - math.atan2(L2 * math.sin(b), L1 + L2 * math.cos(b))
        out.append((a, b))
    return out


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ik(g, psi, tool='jar'):
    """All arm solutions (lift, shoulder, elbow, yaw) with the tool point at g and the
    hand heading psi (pitch = roll = 0)."""
    t = TOOL[tool]
    c, s = math.cos(psi), math.sin(psi)
    Y = np.array([g[0] - (c * t[0] - s * t[1]), g[1] - (s * t[0] + c * t[1])])
    zl = g[2] - t[2]
    sols = []
    for a, b in planar_ik(Y):
        yaw = wrap(psi - a - b)
        sols.append(np.array([zl, wrap(a + math.pi / 2), b, yaw]))     # shoulder joint zero = -y
    return sols


def within_limits(q):
    names = ['lift', 'shoulder', 'elbow', 'yaw']
    return all(LIMITS[n][0] - 1e-9 <= q[i] <= LIMITS[n][1] + 1e-9 for i, n in enumerate(names))
