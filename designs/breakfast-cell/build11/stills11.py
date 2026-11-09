"""Still renders of build11 (MuJoCo's renderer) from the planned mornings in plan11.json:
key moments of the jar and the 1 L box morning on the 90 cm fridge -> render-mornings.png.
Run plan11.py first.

    uv run python designs/breakfast-cell/build11/stills11.py
"""
from pathlib import Path
import json
import sys
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model11 as X  # noqa: E402


def frame_at(seq, label, where=.5):
    idx = [i for i, lab in enumerate(seq['labels']) if lab == label] or [len(seq['frames']) - 1]
    return seq['frames'][idx[min(len(idx) - 1, int(where * len(idx)))]]


def shot(m, d, r, lookat, dist, az, el):
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    r.update_scene(d, cam, scene_option=opt)
    return r.render()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seqs = json.loads((HERE / 'plan11.json').read_text())['sequences']
    picks = [
        ('jar_large', 'fridge_open', .5, 'Outer pin: the plate takes the seal pull', ((-.3, -.25, .55), 1.5, 120, -18)),
        ('jar_large', 'fridge_wide', .7, 'Inner pin on the door rack: door to 100°', ((-.45, -.1, .5), 1.6, 110, -20)),
        ('jar_large', 'take', .7, 'Jar: plate beside it, jaw V centres it', ((-.16, .05, .45), .9, 140, -20)),
        ('box_large', 'take', .75, '1 L box: hand rolled, plate as a fork under it', ((-.4, .05, .45), 1.0, 140, -18)),
        ('box_large', 'to_microwave', .8, 'Box onto the silicone ribs on the turntable', ((-.27, .0, 1.0), 1.0, 130, -15)),
        ('box_large', 'serve', .9, 'Served on the ribs on top of the microwave', ((-.2, .0, 1.2), 1.2, 125, -18)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(27, 16))
    models = {}
    for ax, (key, label, w, title, view) in zip(axes.flat, picks):
        seq = seqs[key]
        if key not in models:
            X.configure(seq['obj'])
            xml, _, _ = X.mjcf11(seq['kind'], seq['obj'], offscreen=(1000, 800))
            m = mujoco.MjModel.from_xml_string(xml)
            models[key] = (m, mujoco.MjData(m))
        m, d = models[key]
        d.qpos[:] = frame_at(seq, label, w)
        mujoco.mj_forward(m, d)
        with mujoco.Renderer(m, 800, 1000) as r:
            img = shot(m, d, r, *view)
        ax.imshow(img); ax.axis('off'); ax.set_title(title, fontsize=19)
    fig.tight_layout()
    fig.savefig(HERE / 'render-mornings.png', dpi=60)
    print('render-mornings.png')


if __name__ == '__main__':
    main()
