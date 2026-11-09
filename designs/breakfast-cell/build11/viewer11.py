"""Builds viewer.html: the build11 cell in three.js, the planned mornings (jar and 1 L box,
90 and 50 cm fridge), one whole morning per menu recorded in physics, clickable parts with
their BOM lines, and the checks.

Run plan11.py, doors11.py and dyn11.py first.

    uv run python designs/breakfast-cell/build11/viewer11.py
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import doors11 as DD  # noqa: E402
import model11 as X  # noqa: E402
M = X.M

MENUS = ('jar_large', 'box_large', 'jar_small', 'box_small')
PHYS_FPS = 7.5           # every second frame of the 15 fps physics recording
NAMES = {
    'shoulder_servo': 'Shoulder servo', 'elbow_servo': 'Elbow servo', 'yaw_servo': 'Wrist yaw servo',
    'roll_servo': 'Wrist roll servo', 'pitch_servo': 'Wrist pitch servo', 'gripper_servo': 'Gripper servo',
    'lift_servo': 'Lift servo', 'ua_tube': 'Upper arm, 350 mm', 'fa_tube': 'Forearm, 350 mm', 'gantry_plate': 'Carriage plate',
    'rail_front': 'Mast: 2040 V-slot (front)', 'rail_back': 'Mast: 2040 V-slot (back)', 'brace': 'Mast brace, taped to the fridge side',
    'cw_plate': 'Counterweight 1.25 kg', 'wrist_cam': 'Wrist camera', 'scene_cam': 'Scene camera',
    'plate': 'Fixed finger: the flat plate (side of the jar grip, fork under boxes)', 'nub_top': 'Key nub (top)', 'nub_low': 'Key nub (low)',
    'jaw_finger': 'Moving jaw', 'jaw_v0': 'Jaw V-pad', 'jaw_v1': 'Jaw V-pad', 'jaw_link': 'Moving jaw', 'palm': 'Palm',
    'pitch_608': 'Support bearing, wrist pitch (608)', 'yaw_608': 'Support bearing, wrist yaw (608)',
    'yaw_yoke': 'Yaw bearing yoke (printed)', 'yaw_yoke_strut': 'Yaw bearing yoke (printed)',
    'f_bar': 'Fridge door pin (outer)', 'f_pin_arm': 'Fridge door pin arm', 'f_bar2': 'Fridge door pin (inner, on the rack)',
    'f_pin2_arm': 'Inner pin clip', 'f_pin2_pad': 'Inner pin clip', 'm_bar': 'Microwave door pin', 'm_pin_arm': 'Microwave door pin arm',
    'm_release': 'Microwave door key', 'm_start': 'Microwave start key', 'm_plate_rib0': 'Silicone rib on the turntable',
    'm_plate_rib1': 'Silicone rib on the turntable', 'm_top_rib0': 'Silicone rib, serving spot', 'm_top_rib1': 'Silicone rib, serving spot',
    'f_shelf_rail0': 'Printed rail under the box', 'f_shelf_rail1': 'Printed rail under the box', 'foot_plate': 'Bolt-on mast foot plate',
}
EXTRA_PARTS = {      # build11 parts that are not in build10's model; geom name prefix -> parts-list line
    'm_plate_rib': 'X01', 'm_top_rib': 'X01', 'f_shelf_rail': 'A08', 'obj_lid': 'X02', 'yaw_servo': 'X03',
}
METAL = {'alu', 'vslot', 'steel', 'bearing', 'psu'}


def geom_parts(world):
    out = {}

    def walk(b):
        for g in b.geoms:
            out[g['name']] = g['part']
        for c in b.children:
            walk(c)
    walk(world)
    return out


def scene(kind, obj):
    X.configure(obj)
    xml, f, world = X.mjcf11(kind, obj)
    m = mujoco.MjModel.from_xml_string(xml)
    parts = geom_parts(world)
    bodies = []
    for b in range(m.nbody):
        j = None
        if m.body_jntnum[b]:
            ji = m.body_jntadr[b]
            j = dict(type=int(m.jnt_type[ji]), axis=[round(float(v), 5) for v in m.jnt_axis[ji]], adr=int(m.jnt_qposadr[ji]))
        bodies.append(dict(name=m.body(b).name, parent=int(m.body_parentid[b]),
                           pos=[round(float(v), 5) for v in m.body_pos[b]], quat=[round(float(v), 6) for v in m.body_quat[b]], joint=j))
    geoms = []
    for g in range(m.ngeom):
        if m.geom_group[g] == 3:
            continue
        name = m.geom(g).name
        t = {int(mujoco.mjtGeom.mjGEOM_BOX): 'box', int(mujoco.mjtGeom.mjGEOM_CAPSULE): 'cap'}.get(int(m.geom_type[g]), 'cyl')
        geoms.append(dict(n=name, b=int(m.geom_bodyid[g]), t=t,
                          s=[round(float(v), 5) for v in m.geom_size[g]], p=[round(float(v), 5) for v in m.geom_pos[g]],
                          q=[round(float(v), 6) for v in m.geom_quat[g]], m=m.mat(m.geom_matid[g]).name,
                          part=next((v for k, v in EXTRA_PARTS.items() if name.startswith(k)), parts.get(name))))
    jadr = {m.joint(i).name: int(m.jnt_qposadr[i]) for i in range(m.njnt)}
    return dict(bodies=bodies, geoms=geoms), jadr, int(m.nq)


def physics_frames(key, seq, jadr, nq):
    """One nominal whole morning in physics, run closed-loop (the robot senses and plans on as it
    goes), mapped onto the viewer's (kinematic) joint layout."""
    p = DD.draw(np.random.default_rng(7), True)
    rec = []
    r = DD.run(seq, seq['obj'], seq['kind'], p, record=rec, live=True)
    D = DD.Y.Dyn(seq['obj'], seq['kind'])
    dadr = D.jadr
    rails = seq['frames'][0][jadr['rails']:jadr['rails'] + 7]
    frames, labels, keep = [], [], int(round(DD.P11.FPS / PHYS_FPS))
    for i, (t, lab, qp) in enumerate(rec):
        if i % keep:
            continue
        q = [0.0] * nq
        for name, a in jadr.items():
            if name == 'rails':
                q[a:a + 7] = rails
            elif name in dadr:
                n = 7 if name == 'jar' else 1
                q[a:a + n] = qp[dadr[name]:dadr[name] + n]
        frames.append([round(v, 4) for v in q])
        labels.append(lab)
    steps, last = [], None
    texts = {s['label']: s['text'] for s in seq['steps']}
    texts.update(take_hot='Seen again after heating: the hot take is planned from what the camera sees',
                 heat='Heat, shortened to 3½ turns at the turntable\'s real 5 rpm; the Shelly cuts the power with it at 0 or 180 deg')
    labels = [labels[next((i for i, x in enumerate(labels) if x != 'start'), 0)] if lab == 'start' else lab for lab in labels]
    for k, lab in enumerate(labels):
        if lab != last:
            steps.append(dict(start=round(k / PHYS_FPS, 2), label=lab, text=texts.get(lab, lab.replace('_', ' '))))
            last = lab
    r['heat_s'] = round(sum(1 for lab in labels if lab == 'heat') / PHYS_FPS, 1)
    return dict(fps=PHYS_FPS, frames=frames, steps=steps), r


def cause(d):
    """Why a random morning failed, by its first real cause."""
    if d['ok']:
        return ''
    rep, dev = d['replan_error'] or '', d['step_end_error_deg']
    if rep.startswith('fridge_open') or dev.get('fridge_open', 0) < -20:
        return 'seal not broken'
    first = next((k for k, v in dev.items() if abs(v) > 5), None)
    if first:
        return f'{first.replace("_", " ")}: door ended off target'
    if d['meal'] and not d['meal']['ok']:
        return 'meal ' + d['meal']['why'].split(' ')[0]
    if rep:
        return 'could not plan ' + rep.split(':')[0].replace('_', ' ')
    stalled = {j: v for j, v in d['stalled_s'].items() if v > .2}
    if stalled:
        return 'servo at its limit over 0.2 s (' + ', '.join(stalled) + ')'
    return ', '.join(f'{k.replace("_", " ")} not shut' for k, v in d['final_deg'].items() if abs(v) >= 1) or 'other'


def summary(r):
    """A random-morning batch for the page: totals, each trial's seal, catch and verdict, and why it failed."""
    if not r:
        return None
    rows = [dict(seal=d['params']['seal_N'], catch=d['params']['mw_catch_N'], ok=d['ok'], why=cause(d)) for d in r['detail'][1:]]
    return dict(runs=r['runs'], passed=r['passed'], nominal_ok=r['nominal_ok'], seal=[(x['seal'], x['catch'], x['ok']) for x in rows],
                why=[x['why'] for x in rows if not x['ok']])


def main():
    DD.use_yaw('sts3250')             # the wrist yaw is an STS3250
    plan = json.loads((HERE / 'plan11.json').read_text())
    pres = json.loads((HERE / 'plan11-results.json').read_text())
    doors = json.loads((HERE / 'doors11-results.json').read_text()) if (HERE / 'doors11-results.json').exists() else {}
    dyn = json.loads((HERE / 'dyn11-results.json').read_text()) if (HERE / 'dyn11-results.json').exists() else {}
    bom = list(csv.DictReader(open(HERE / 'bom.csv')))     # the wrist yaw is the STS3250 (X03); six STS3215 for the other joints
    byid = {r['id']: r for r in bom}
    parts = {}
    for key, (bid, label) in M.PARTS.items():
        r = byid.get(bid)
        parts[key] = dict(id=bid, label=label, item=r['item'] if r else None, qty=r['qty'] if r else None, use=r['use'] if r else None)
    for bid in set(EXTRA_PARTS.values()):
        r = byid[bid]
        parts[bid] = dict(id=bid, label=r['item'], item=r['item'], qty=r['qty'], use=r['use'])
    mats = {k: dict(c=[float(x) for x in v[0].split()], metal=k in METAL) for k, v in dict(M.MAT, **X.MAT_EXTRA).items()}
    menus = {}
    for key in MENUS:
        seq = plan['sequences'][key]
        sc, jadr, nq = scene(seq['kind'], seq['obj'])
        phys, r = physics_frames(key, seq, jadr, nq)
        res = {k: v for k, v in pres[key].items() if k not in ('penetrations',)}
        menus[key] = dict(scene=sc, plan=dict(fps=plan['fps'], frames=seq['frames'], steps=seq['steps']), phys=phys,
                          results=res, phys_run=dict(ok=r['ok'], meal=r['meal'], duration_s=r['duration_s'], heat_s=r['heat_s'], stalled_s=r['stalled_s'],
                                                     step_end_error_deg=r['step_end_error_deg'], off_path_mm=r['off_path_mm']),
                          doors=summary(doors.get(key)), doors_sts=summary(doors.get(f'{key}+sts3250')),
                          doors_cal=summary(doors.get(f'{key}+sts3250+cal')),
                          dyn={k: v for k, v in dyn[key].items() if k != 'detail'} if key in dyn else None)
        print(key, 'physics', 'OK' if r['ok'] else 'FAIL', len(phys['frames']), 'frames', flush=True)
    sweep = doors.get('sweep')
    data = dict(menus=menus, names=NAMES, parts=parts, mats=mats, bom=bom, sweep=sweep,
                tray=pres.get('tray_large'))
    blob = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    html = (HERE / 'viewer_template.html').read_text().replace('/*DATA*/null', blob)
    (HERE / 'viewer.html').write_text(html)
    print(f'viewer.html {len(html) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
