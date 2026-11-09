# Simulation notes

Modelling pitfalls found in the original mobile-design experiments. Paths are relative to the repository root.

- **Closures in the timeline bind late.** `plan()` builds lambdas that read local variables; reusing a name later in `plan()` (`phis`, `pre`) silently changed earlier segments' targets. Give every waypoint its own name.
- **The two arms share their horizontal planes: they can't cross.** CAD's `park` pose (forearm back from an elbow 35 cm ahead of the shoulder) blocks the other arm. Use `scene.TUCK_A/B` (upper arm back along the carriage, forearm forward, hand down at x ≈ 0.19 m) whenever the other arm works in front.
- **The iPhone sticks out 11.5 cm to the hand's right** (CAP frame −x … +115 mm). A hanging hand must point it outboard (right arm heading 0°, left arm 180°), or it lands on the tray and the arm fights it at 40–150 N.
- **L1 = L2, so a wrist path through the shoulder is singular:** the IK flips 100°+ in one step. Swing the wrist round the front of the shoulder.
- **Speed limits:** `Timeline.need()` must use the *fastest* sub-interval of a path, not the average, or the rate limiter lags behind closed-loop targets and the hand hits things.
- **`robustness.py` hard-coded a v0 world coordinate** (`FINAL_FACE_X`), so on v0.2 it pulled 12 cm short: 8/100. Grep for absolute x values when the frame moves again.
- **The CAP mounting holes are in the bottom plate's underside** (2× M3, nut traps, 52.1 mm apart, 27 mm from the back face), not its back face; the adapter is an L-bracket now (`CAP_HOLES` in `cad/robot.py`).
- **The long CAP hand changes every scripted motion** (pitch axis → grasp centre 276 mm, v0 was 110):
  - After pulling a box onto the drawer, the wrist sits beside its own shoulder. The hand has to leave on the far side (left for arm a), not toward the shoulder.
  - Unfolding next to the fridge sweeps the hand through its side wall. Unfold before driving up.
  - Parked hands must hang outboard and behind the tray front, or they hit the tray load and a bed edge (`scene.park`).
- **Angled grasps of the front knob don't work:** the fingers close across the knob's axis only when the hand points straight at it.
- **EgoGym "cap" robot welds the mocap to the *camera* body**, not the gripper. `tischlein_cap.py` converts camera pose ↔ gripper pose with the model's own offsets (read in `setup_mj`).
- **A roll joint that chases every roll request costs pick success** (67% vs 79% in sim), with or without realistic step limits.
  - At full stretch, at the table edge, the flat arm is at its straight-elbow singularity, and the IK spins the roll joint.
  - The fix is a prior that keeps the roll near 0 unless a clear turn is requested: `roll_prior=0.35` rad in `Tischlein6Selective` → 77%.
  - The real controller needs the same rule.
- **Door sims:** the robot base has no collision shapes in MuJoCo, so robot-vs-door clearance is checked geometrically (`door_clearance`). The fridge door here is a fridge-freezer upper door (bottom edge 0.885 m). The tray is lowered just below it during the swing.
- **Angles are degrees.** MJCF uses `compiler angle="degree"`, so joint ranges are in degrees (a `range="-2.0 0"` once locked the door at 2°). `qpos` is still radians.
- **Thin collision geoms explode** under heavy contact. The tray and shelves are now 15–24 mm thick boxes. Finger pads are 12 mm thick, inner face at ±12 mm + grip.
- **Boxes above ~2.5 kg** still make the sim numerically unstable at the moment the box drops from shelf to tray. Treat results above 2 kg as unknown, not as failure.
- **Grip depth on the knob matters.** Pads are 60 mm long and the knob is 18 mm deep. Target the pad centre at `box_face − 33 mm` so the pad tips overlap the knob by ~15 mm. Targeting the knob centre gives only 6 mm overlap and caused the 33% result.
- **The hand can't lift heavy things**, and the arm plane sits at tray +115/+155 mm. The hand (z +24…+56) cannot pass over items on the tray, and the tray lips must stay ≤ ~20 mm so the hand can exit sideways. That is why the drawer exists: a fixed tray cannot receive a box pulled by a hand that ends up behind it.
- **The sim base is holonomic** (slide x, slide y, hinge yaw) and kinematic with force caps. A real two-wheel base can't slide sideways; the door result depends on that.
- **Seal model** in `door.py`: closing torque `F_b · r_h` fading to 0 over 8 mm of handle gap. Self-closing 1 N·m below 85°, hold-open beyond 92°. The door panel is kept 2 mm off the cabinet and below the freezer top, otherwise friction holds it.
- **Servo frame.** `vendor/so-arm100/STS3215_03a.step` has the output axis at (12.5, 0) and the horn face at z = 18.7. `robot.servo()` re-origins it to axis = z, horn face = 0, body toward −x/−z. Horn holes: 4× on r = 7 mm at 45°.
- **`box()` in robot.py sorts its corners**: Box fails on negative sizes otherwise.
- **MuJoCo free-camera azimuth.** Around 25–35° looks at the scene from the robot's back-right (−x, −y). ~200° looks from behind the fridge.
- **Paths:** frames go to `out/*_frames/` (gitignored). `sim/_door.xml` is regenerated by `door.py` from `tischlein.xml`.
