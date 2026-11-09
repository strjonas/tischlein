# Print notes (v0.2)

All parts fit a 250 × 250 × 250 mm print bed. `parts.json` lists size, volume and quantity. Each STL is exported in the part's link frame, centred, with its lowest face on z = 0. Rotate it in the slicer so the biggest flat face sits on the bed.

Parts ending in `_b` are the left arm's mirror images; the per-arm parts (root clamps, elbow, wrist, roll bracket, adapter plate) are identical for both arms, so print two.

## The CAP gripper is printed from its published files, not from here

Download the "Robot Gripper" parts from [nyu-gripper.pages.dev](https://nyu-gripper.pages.dev/robot-gripper) and print them unchanged: bottom plate, top plate, top phone holder, finger ×2, string-finger connector, gripper motor horn, motor attachment. They drive a Dynamixel XL330 by a Kevlar string.

- Unchanged matters: the pretrained skills learned this hand and this phone position.
- Their Stretch quick-release mount is replaced by our `cap_adapter_plate`, an L-bracket. Its upright bolts to the roll servo's horn (4× M2.5 on r = 7 mm); its tongue runs under their bottom plate.
  - Two M3 screws go up from below into the bottom plate's own hex-nut traps.
  - The pattern was measured from their STL on 26 Oct: 52.1 mm apart, 27.0 mm in front of the plate's back face (`CAP_HOLES` in `cad/robot.py`).
  - The back face itself has no holes. Print the adapter flat on its tongue.
- For the **iPhone SE 2020** (smaller than the 13 Pro their holder was made for), the holder needs a spacer, so the SE's lens sits where the 13 Pro's main lens was (±3 mm). Not drawn yet:
  1. With calipers, measure each phone's lens centre from its top edge and its side edge.
  2. Then it's one function in `link_tool()`: a sleeve with the 13 Pro's outline and an SE-shaped pocket, offset so the lenses coincide.
- `third_party/cap_gripper/` (gitignored) holds a downloaded copy for measuring.

## First print gauges, not parts

The fits come from datasheets and the SO-101 servo STEP, not from anything printed on your machine. Check these first:

1. **Servo pocket.** Print just the bottom 15 mm of `wrist_block` and slide an STS3215 in. The STS3250 has the same case.
2. **Bearing seat.** One 6000 seat (26.1 mm bore). Press-fit or glue: decide after the test.
3. **2020 socket.** The `l1_root_clamp` socket (20.3 mm). It should slide on with light force.

Change `SERVO_POCKET`, `BEARING_6000` or the socket numbers in `cad/robot.py`, then run `uv run python cad/build.py prints` again.

## Material

- PETG for everything structural: 4 walls, 30–40% gyroid infill. The arm blocks and the shoulder housings take bending loads, so print them with the bearing bore vertical.
- TPU for the caster pads.
- PLA is fine for gauges and covers.

## Not printed

Bought parts: GT2 pulleys, belts, bearings, shafts, rails, slides, extrusions, and the 2040 shoulder beam (two shoulders 240 mm apart don't fit on a printed plate).

## Still approximate

- The roll bracket's cable path for the XL330 and the iPhone's USB cable. The roll range will be about ±180° with a cable loop.
- The electronics box, lidar post and caster mounts. Measure your actual boards and casters first.
- The face mount: sized for a 75 × 150 mm phone. Measure yours.
