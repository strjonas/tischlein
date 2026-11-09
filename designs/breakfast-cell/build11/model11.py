"""Build11: build10's robot and cell with a new hand, bearings, door pins, freezer, rails and ribs.

Changes to build10 (model3d.py, which stays the record of build10):
- One finger set for everything: the fixed finger is a flat printed plate (146 x 44 x 8 mm)
  that is the side of a jar grip and, with the hand rolled -90 deg, a fork under boxes and
  trays; the moving jaw carries the V that centres the jar. Two 8 mm nubs at the plate's
  tip press the microwave keys and hook the freezer flap.
- 608 support bearings: a yoke under the wrist yaw, a bearing on the pitch joint's free side.
- Door pins 9 cm long with domed ends and 2.5 cm further off the door (the long plate, and the
  jaw while it opens, stay clear of the door), and a second, inner pin on
  the fridge door rack, 25 cm from the hinge (door to 100 deg).
- Cell: the 90 cm fridge's top block is a 4-star freezer compartment with a bottom-hinged
  flap and a printed loop tab; printed rails (20 mm, 80 mm apart) under boxes and trays;
  two silicone ribs on the turntable and on the serving spot.
- The mast foot is a bolt-on plate (drawn on the board).

The same Body tree gives a kinematic model (gravity off, 10 mm clearance margins, as
build10's sequence.py uses) and a dynamic one (gravity, rated servos as position actuators
with stall-torque caps, friction on every contact pair that matters): mjcf11(..., dynamic=True).
"""
from pathlib import Path
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'build10'))
import model3d as M  # noqa: E402

PX, PY = .0447, -.019                 # jaw hinge in the pitch frame (model3d.py)
PALM = .061
PLATE = dict(face=.0465, t=.008, hw=.022, x0=.045, x1=.191)
NUB = .008
RAILS = dict(gap=.080, w=.015, h=.020)   # printed rails under each box spot in the fridge
RIBS = dict(channel=.064, w=.016)        # silicone ribs on the turntable and the serving spot: the fork's channel between them
FREEZER = dict(floor=.69, roof=.84, front=.08, flap_h=.13)
PIN2_X = .25
BAR_A = .55          # jaw angle on a door pin: the pin sits 1 cm back from the plate's tip, held in the narrowing V
PIN_OUT = .025       # door pins stand this much further off the door than build10's

OBJECTS = {
    'jar': dict(mode='side', m=.75, r_bot=.043, r_top=.050, h=.11, grip=.055, label='Weck 580 ml jar, 0.75 kg'),
    'box': dict(mode='fork', m=1.87, L=.15, W=.21, H=.075, foot=(.13, .19), label='IKEA 365+ 1 L glass box, full, 1.87 kg'),
    'tray': dict(mode='fork', m=.43, L=.19, W=.14, H=.045, foot=(.165, .11), label='Frozen lasagna, 400 g tray'),
}


# ------------------------------------------------------------------ hand geometry
def jaw_parts(jc_v=.1115):
    """Jaw geoms in the jaw body frame (hinge at the origin): link, finger and a V of two
    pads mirroring build10's fixed V (so the jar is held exactly as in build10)."""
    yb = -(PLATE['face'] + .012) - PY
    r_v = PLATE['face'] - .006
    parts = [('box', 'jaw_link', [0, yb, -.0125], [.02, 0, .0125]),
             ('box', 'jaw_finger', [.015, yb, -.0125], [PLATE['x1'] - PX, yb + .010, .0125])]
    for k, (dx, ang) in enumerate(((-.030, 35), (.030, -35))):
        parts.append(('obox', f'jaw_v{k}', [jc_v + dx - PX, -r_v - PY, 0], [.012, .004, .02], math.radians(ang)))
    return parts


def jaw_points(a, jc_v=.1115, n=40):
    """Inner contact points of the jaw (pitch frame) at jaw angle a: along the finger's
    inner face and the V pads' inner corners."""
    yb = -(PLATE['face'] + .012) - PY
    pts = [(x, yb + .010) for x in np.linspace(.015, PLATE['x1'] - PX, n)]
    r_v = PLATE['face'] - .006
    for dx, ang in ((-.030, 35), (.030, -35)):
        c = np.array([jc_v + dx - PX, -r_v - PY]); th = math.radians(ang)
        u, v = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        pts += [tuple(c + s * .012 * u + .004 * v) for s in np.linspace(-1, 1, 7)]
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[PX + x * ca - y * sa, PY + x * sa + y * ca] for x, y in pts])


def jaw_angle_to(y_target, x_range, lo=-.6, hi=1.2):
    """Smallest jaw angle at which any jaw point within x_range reaches y >= y_target."""
    for a in np.arange(lo, hi, .002):
        p = jaw_points(a)
        sel = p[(p[:, 0] >= x_range[0]) & (p[:, 0] <= x_range[1])]
        if len(sel) and sel[:, 1].max() >= y_target:
            return float(a)
    return None


def bar_point(a=.55):
    """Where a 22 mm pin sits when the jaw at angle a clamps it against the plate."""
    face = PLATE['face']
    best = None
    for x in np.arange(.12, PLATE['x1'] - .01, .001):
        c = np.array([x, face - .011])
        p = jaw_points(a, n=200)
        d = np.min(np.linalg.norm(p - c, axis=1))
        if best is None or abs(d - .011) < best[0]:
            best = (abs(d - .011), x)
    return best[1]


def configure(obj):
    """Set build10's module-level tool points and grip angles for this hand and object, so
    sequence.py's planner (which reads them) plans with build11's hand."""
    o = OBJECTS[obj]
    yr = M.YAW_TO_ROLL + M.ROLL_TO_PITCH
    if o['mode'] == 'side':
        jc = PALM + o['r_bot'] + (o['r_top'] - o['r_bot']) * o['grip'] / o['h'] + .004
        M.JAR = dict(r_bot=o['r_bot'], r_top=o['r_top'], h=o['h'], grip=o['grip'], m=o['m'])
        held = 0.0
    else:
        jc = PALM + o['L'] / 2 + .004
        M.JAR = dict(r_bot=PLATE['face'], r_top=PLATE['face'], h=o['H'], grip=PLATE['face'], m=o['m'])
        held = jaw_angle_to(PLATE['face'] - o['H'], (PALM, PALM + o['L']))      # V pads on the top
    M.JAR_IN_PITCH = jc
    M.TOOL['jar'] = np.array([yr + jc, 0, M.HAND_DZ])
    xb = bar_point(BAR_A)
    M.TOOL['bar'] = np.array([yr + xb, PLATE['face'] - .011, M.HAND_DZ])
    M.TOOL['tip'] = np.array([yr + PLATE['x1'] + NUB, PLATE['face'] + PLATE['t'] / 2, M.HAND_DZ + PLATE['hw'] - .005])
    M.TOOL['tip_low'] = np.array([yr + PLATE['x1'] + NUB, PLATE['face'] + PLATE['t'] / 2, M.HAND_DZ - PLATE['hw'] + .005])
    M.GRIP.update(jar=held, bar=BAR_A, tip=.55, open=-.45)
    return dict(jc=jc, held=held, roll=-math.pi / 2 if o['mode'] == 'fork' else 0.0,
                tool_h=o['grip'] if o['mode'] == 'side' else PLATE['face'], mode=o['mode'])


def rebuild_hand(car, jc_v=.1115):
    pt = _find(car, 'wrist_pitch')
    face, t, hw = PLATE['face'], PLATE['t'], PLATE['hw']
    pt.geoms = [g for g in pt.geoms if g['name'] not in ('fixed_finger', 'fixed_v1', 'fixed_v2')]
    pt.box('plate', [PLATE['x0'], face, -hw], [PLATE['x1'], face + t, hw], 'petg', 'print', 'finger')
    pt.box('nub_top', [PLATE['x1'], face + .001, hw - .009], [PLATE['x1'] + NUB, face + t - .001, hw - .001], 'petg2', 'print', 'finger')
    pt.box('nub_low', [PLATE['x1'], face + .001, -hw + .001], [PLATE['x1'] + NUB, face + t - .001, -hw + .009], 'petg2', 'print', 'finger')
    pt.cyl('pitch_608', [0, -.0235, 0], .014, .004, 'bearing', 'bearing', 'robot', axis='y')
    jw = _find(pt, 'jaw')
    jw.geoms = [g for g in jw.geoms if g['name'] == 'jaw_hub']
    for p in jaw_parts(jc_v):
        if p[0] == 'box':
            jw.box(p[1], p[2], p[3], 'petg', 'print', 'finger')
        else:
            jw.obox(p[1], p[2], p[3], p[4], 'petg2', 'print', 'finger')
    fa = _find(car, 'forearm')
    L2 = M.L2
    fa.box('yaw_yoke', [L2 - .032, -.016, -.037], [L2 + .016, .016, -.031], 'petg', 'print')
    fa.box('yaw_yoke_strut', [L2 - .032, -.016, -.031], [L2 - .024, .016, -.016], 'petg', 'print')
    fa.cyl('yaw_608', [L2, 0, -.034], .011, .0035, 'bearing', 'bearing', 'vis')


def _find(body, name):
    if body.name == name:
        return body
    for c in body.children:
        r = _find(c, name)
        if r is not None:
            return r
    return None


# ------------------------------------------------------------------ cell
def fixture(world, kind, spots):
    """build10's cell plus build11's aids. spots: dict of rail sets to place, name -> (x, y, z, psi)."""
    f = M.build_fixture(world, kind)
    W, D, t = f['W'], f['D'], f['t']
    door = next(c for c in world.children if c.name == 'f_door')
    # longer outer pin: 9 cm, hanging from the same arm
    hz = f['handle_z']
    for g in door.geoms:
        if g['name'] == 'f_bar':
            g['pos'][2], g['size'][1] = hz + .03 - .045, .045
            g['pos'][1] -= PIN_OUT
        if g['name'] == 'f_pin_arm':
            g['pos'][1] -= PIN_OUT / 2; g['size'][1] += PIN_OUT / 2
    # inner pin on a clip at the door rack edge, 25 cm from the hinge
    z2 = .45 if kind == 'large' else .30
    door.box('f_pin2_pad', [PIN2_X - .02, .07, z2 + .03], [PIN2_X + .02, .073, z2 + .075], 'petg2', 'print', 'handle')
    door.box('f_pin2_arm', [PIN2_X - .012, .07, z2 + .045], [PIN2_X + .012, .12, z2 + .06], 'petg2', 'print', 'handle')
    door.cyl('f_bar2', [PIN2_X, .107, z2], .011, .045, 'petg2', 'print', 'handle')
    mdoor = next(c for c in world.children if c.name == 'm_door')
    for g in mdoor.geoms:
        if g['name'] == 'm_bar':
            g['pos'][2], g['size'][1] = .1225 + .0325 - .045, .045
            g['pos'][1] -= PIN_OUT
        if g['name'] == 'm_pin_arm':
            g['pos'][1] -= PIN_OUT / 2; g['size'][1] += PIN_OUT / 2
    for body in (door, mdoor):                         # printed pins have domed ends: a plate that meets an end slides off it
        for g in body.geoms:
            if g['name'] in ('f_bar', 'f_bar2', 'm_bar'):
                g['type'] = 'capsule'
                g['size'] = [g['size'][0], g['size'][1] - g['size'][0]]
    fr = next(c for c in world.children if c.name == 'fridge')
    if kind == 'large':                                 # 4-star compartment with a bottom-hinged flap
        fr.geoms = [g for g in fr.geoms if g['name'] != 'f_top_block']
        Z0, Z1, Y0 = FREEZER['floor'], FREEZER['roof'], FREEZER['front']
        fr.box('f_shelf_ice', [-W + t, Y0, Z0], [-t, D - .05, Z0 + .01], 'fridge_in', 'fridge', 'env')
        fr.box('f_ice_roof', [-W + t, Y0, Z1], [-t, D - .05, Z1 + .01], 'fridge_in', 'fridge', 'env')
        fr.box('f_ice_sill', [-W + t, Y0, Z0], [-t, Y0 + .01, Z0 + .015], 'fridge_in', 'fridge', 'env')
        flap = world.child(M.Body('flap', pos=(-W / 2, Y0, Z0 - .002),     # open, its top is flush with the floor
                                  joint=dict(name='flap', type='hinge', axis=(1, 0, 0), range=(0, math.radians(95)))))
        # closed: standing in the compartment's opening; opening turns its top edge forward and down
        # to a ledge level with the compartment floor. A printed loop on the top edge: a bar 10 mm
        # in front of the flap, open at the top, that the plate's nub hooks from above.
        h = FREEZER['flap_h']
        flap.box('flap_slab', [-W / 2 + t + .01, 0, 0], [W / 2 - t - .01, .012, h], 'mw', 'fridge', 'env')
        zt = h - .06                                     # low enough for the plate to come from above under the roof
        flap.box('flap_tab', [-.02, -.014, zt - .012], [.02, -.010, zt + .012], 'petg2', 'print', 'handle')
        flap.box('flap_tab_floor', [-.02, -.014, zt - .012], [.02, 0, zt - .008], 'petg2', 'print', 'handle')
        f.update(flap_tab=np.array([0, -.007, zt]))
        f.update(flap_hinge=np.array([-W / 2, Y0, Z0 - .002]))
    r = world.child(M.Body('rails', free=True))            # moved per test by the planner
    for k, s in enumerate((-1, 1)):
        r.box(f'f_shelf_rail{k}', [s * RAILS['gap'] / 2 - RAILS['w'] / 2, -.085, 0],
              [s * RAILS['gap'] / 2 + RAILS['w'] / 2, .085, RAILS['h']], 'petg2', 'print', 'env')
    for name, (x, y, z, psi) in spots.items():
        r = world.child(M.Body(f'rails_{name}', pos=(x, y, z), quat=M.quat_z(psi - math.pi / 2)))
        for k, s in enumerate((-1, 1)):
            r.box(f'f_shelf_rail_{name}{k}', [s * RAILS['gap'] / 2 - RAILS['w'] / 2, -.085, 0],
                  [s * RAILS['gap'] / 2 + RAILS['w'] / 2, .085, RAILS['h']], 'petg2', 'print', 'env')
    # silicone ribs on the turntable (they turn with it) and on the serving spot
    tt = next(c for c in _all(world) if c.name == 'turntable')
    a, b = RIBS['channel'] / 2, RIBS['channel'] / 2 + RIBS['w']
    for k, s in enumerate((-1, 1)):
        x0, x1 = sorted((s * a, s * b))
        tt.box(f'm_plate_rib{k}', [x0, -.085, .015], [x1, .085, .015 + RAILS['h']], 'cable', 'ribs', 'env')
        world.box(f'm_top_rib{k}', [SERVE[0] + x0, SERVE[1] - .085, f['mw_top']],
                  [SERVE[0] + x1, SERVE[1] + .085, f['mw_top'] + RAILS['h']], 'cable', 'ribs', 'env')
    world.box('foot_plate', [.02, .0, 0], [.20, .09, .002], 'petg2', 'print', 'vis')
    return f


SERVE = (-.20, .14)


def _all(body):
    out = [body]
    for c in body.children:
        out += _all(c)
    return out


def build_object(world, obj, pos=(0, 0, -3)):
    o = OBJECTS[obj]
    b = world.child(M.Body('jar', pos=pos, free=True))
    if o['mode'] == 'side':
        n = 5
        for k in range(n):
            z0, z1 = o['h'] * k / n, o['h'] * (k + 1) / n
            r = o['r_bot'] + (o['r_top'] - o['r_bot']) * (k + .5) / n
            b.cyl(f'jar_glass{k}', [0, 0, (z0 + z1) / 2], r, (z1 - z0) / 2, 'jar_glass', 'jar', 'jar')
            if z1 <= .086:
                b.cyl(f'jar_food{k}', [0, 0, (z0 + z1) / 2], r - .004, (z1 - z0) / 2 - .0005, 'food', None, 'vis')
        b.cyl('jar_lid', [0, 0, o['h'] + .003], o['r_top'] + .004, .003, 'jar_glass', 'jar', 'jar')
    else:
        fx, fy = o['foot'][1] / 2, o['foot'][0] / 2        # across, along the hand
        b.box('obj_base', [-fx, -fy, 0], [fx, fy, o['H'] - .012], 'jar_glass' if obj == 'box' else 'tray', 'jar', 'jar')
        b.box('obj_rim', [-o['W'] / 2, -o['L'] / 2, o['H'] - .012], [o['W'] / 2, o['L'] / 2, o['H']],
              'jar_glass' if obj == 'box' else 'tray', 'jar', 'jar')
        b.box('obj_food', [-fx + .005, -fy + .005, .004], [fx - .005, fy - .005, o['H'] * .75], 'food', None, 'vis')
        if obj == 'box':
            b.box('obj_lid', [-o['W'] / 2 + .005, -o['L'] / 2 + .005, o['H']], [o['W'] / 2 - .005, o['L'] / 2 - .005, o['H'] + .004],
                  'lid', None, 'vis')
    return b


# ------------------------------------------------------------------ XML
MAT_EXTRA = {'tray': ('0.80 0.55 0.30 1', .3, .2), 'lid': ('0.55 0.75 0.65 0.95', .2, .1)}
# build10's fingers were short and never collided with the door pins, keys or the robot's own
# links (finger class (8, 1)). Build11's 146 mm plate must: the fingers now also meet handles
# and robot parts, and the planner allows touching only the pin or key in use (plan11.py).
CLS_KIN = dict(M.CLS, finger=(8, 19))
CLS_DYN = dict(env=(1, 0), robot=(2, 25), finger=(4, 27), jar=(8, 7), handle=(16, 6), vis=(0, 0))

# servo drives: kp, kv, force cap (stall at the joint), armature, damping, frictionloss
DRIVES = dict(lift=(2.0e4, 900., 192., 2.0, 40., 3.0), shoulder=(300., 12., 7.06, .10, .5, .05), elbow=(300., 12., 7.06, .10, .5, .05),
              yaw=(60., 1.0, 2.94, .012, .05, .02), roll=(60., 1.0, 2.94, .012, .05, .02), pitch=(60., 1.0, 2.94, .012, .05, .02),
              grip=(30., .3, .98, .012, .02, .01))
DRIVES_STOCK = dict(DRIVES)
APPL = ('fridge_door', 'mw_door', 'turntable', 'flap')


def mjcf11(kind='large', obj='jar', spots=None, dynamic=False, offscreen=(1600, 1200)):
    world = M.Body('world')
    f = fixture(world, kind, spots or {})
    parts = M.build_robot(world, f)
    rebuild_hand(parts['carriage'])
    build_object(world, obj)
    mats = dict(M.MAT, **MAT_EXTRA)
    mat_xml = '\n'.join(f'    <material name="{k}" rgba="{v[0]}" specular="{v[1]}" shininess="{v[2]}"/>' for k, v in mats.items())
    cls_saved, margin_saved = M.CLS, M.ROBOT_MARGIN
    M.CLS = CLS_DYN if dynamic else CLS_KIN
    M.ROBOT_MARGIN = 0.0 if dynamic else margin_saved
    try:
        bodies = '\n'.join(c.xml(1) for c in world.children)
        wg = []
        for g in world.geoms:
            ct, ca = M.CLS[g['cls']]
            wg.append(f'    <geom name="{g["name"]}" type="{g["type"]}" pos="{M._f(g["pos"])}" size="{M._f(g["size"])}" '
                      f'material="{g["mat"]}" contype="{ct}" conaffinity="{ca}" group="{g["group"]}"/>')
    finally:
        M.CLS, M.ROBOT_MARGIN = cls_saved, margin_saved
    if dynamic:
        option = ('<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="20" '
                  'solver="Newton" iterations="80" ls_iterations="40"><flag multiccd="enable"/></option>')
        acts = []
        for j, (kp, kv, cap, arm, damp, fl) in DRIVES.items():
            lo, hi = M.LIMITS[j]
            acts.append(f'    <position name="a_{j}" joint="{j}" kp="{kp}" kv="{kv}" forcerange="{-cap} {cap}" ctrlrange="{lo} {hi}"/>')
        for j in APPL:
            if j == 'flap' and kind != 'large':
                continue                              # only the 90 cm fridge has the freezer flap
            acts.append(f'    <position name="a_{j}" joint="{j}" kp="3000" kv="60" forcerange="-400 400"/>')
        actuators = '  <actuator>\n' + '\n'.join(acts) + '\n  </actuator>'
        default = '  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" friction="0.4 0.005 0.0001" condim="4"/></default>'
    else:
        option = '<option gravity="0 0 0"/>'
        actuators, default = '', ''
    xml = f"""<mujoco model="build11_{kind}_{obj}">
  <compiler angle="radian" autolimits="true"/>
  {option}
{default}
  <visual><global offwidth="{offscreen[0]}" offheight="{offscreen[1]}"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/><map znear="0.01"/></visual>
  <asset>
{mat_xml}
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
    <exclude body1="forearm" body2="wrist_yaw"/>
    <exclude body1="wrist_roll" body2="jaw"/>
    <exclude body1="wrist_yaw" body2="jaw"/>
  </contact>
{actuators}
</mujoco>"""
    if dynamic:                      # the rails are fixed in the world: a mocap body placed per trial
        import re
        xml = re.sub(r'(<body name="rails"[^>]*)>\s*<freejoint name="rails"/>', r'\1 mocap="true">', xml)
    return xml, f, world


def joint_tuning(m):
    """Armature, damping and friction of the robot joints in the dynamic model."""
    for j, (kp, kv, cap, arm, damp, fl) in DRIVES.items():
        jid = m.joint(j).id
        dof = m.jnt_dofadr[jid]
        m.dof_armature[dof], m.dof_damping[dof], m.dof_frictionloss[dof] = arm, damp, fl
