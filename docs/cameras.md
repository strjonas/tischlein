# Cameras: how accurate must the contact point be, and what does a phone without depth achieve?

## Short answer

- **The pretrained CAP skills don't need a depth camera.** They look at plain RGB plus one 3D point, the "contact anchor": where to grab. Depth is used once, at the start, to turn a tap on the screen into that 3D point. After that the robot tracks the point with its own joint angles ([CAP paper](https://arxiv.org/html/2602.09017): "We track the anchor in the camera frame using the robot's forward kinematics").
- **How accurate does that point have to be? About 1 cm.** Simulated on the bench rig (arm + lift, no base), 150 pick episodes per row ([chart](../out/cap_anchor_error.png)):

| Contact point error | Pick success |
|---|---:|
| exact (perfect depth) | **81%** (ideal floating gripper: 80%) |
| 0.5 cm | 78% |
| 1 cm | **75%** |
| 2 cm | 60% |
| 4 cm | 37% |
| SE without depth: the tap's ray meets the known table plane | **71%** |
| SE with a monocular depth network (~8% of the distance) | 67% |

(`experiments/cap_zero_shot/results/pick_benchlift_v3_*.json`. The contact point is a fixed world point here, as on the real robot; EgoGym's default follows the object's true pose every step, which gave 79%.)

- **Can the SE get to ~1 cm without a depth sensor? Partly.** `experiments/cap_zero_shot/triangulate.py` measures it on EgoGym's rendered wrist images (960 × 720, the CAP camera's field of view), 35 cm from the object, 40 random scenes per row. The robot moves the wrist 8 cm sideways and works out the tapped point from both views:

| Method | Point found again | Error, median / 90th percentile |
|---|---:|---:|
| One move, patch matching along the epipolar line | 29 of 40 | 1.3 / 4.6 cm (exact poses), 1.7 / 5.5 cm (servo pose error) |
| **Track the point through 4 small moves (optical flow)** | **22 of 40** | **0.9 / 1.7 cm (exact poses), 1.1 / 2.0 cm (servo pose error)** |

  - So **~1 cm when the point can be tracked** (worth ~75% pick). On plain, untextured objects it is lost about half the time; then fall back to the table plane (71%).
  - **Expect ~72–75% for the SE vs ~81% with perfect depth** (sim). A LiDAR iPhone (judgement: ~1 cm on small objects) wouldn't do much better than the tracked SE.
  - The pose error barely matters: phone tracking (ARKit-like, 0.5 mm / 0.1°) and cheap-servo kinematics (1.5 mm / 0.3°) give nearly the same result. The image matching is the limit.
  - Rendered images are cleaner than a phone's; real labels and packaging have more texture than many of EgoGym's objects. Both effects are unmeasured.
- **A phone without depth is enough for a bench test.** Plane first (trivial), tracked triangulation second (~100 lines, in the sim script already).
- **A depth camera on the wrist is not needed.** The skills were trained on iPhone 13 Pro images; another camera breaks that match, and better depth is worth ~5 points at most here.
- **The head needs no depth camera.** The lidar does navigation and works in the dark; a phone is the face and the overview camera.

## What still has to be checked on a real iPhone SE

The simulation can't answer these:

1. **Can it stream to the host computer at all?** CAP's robot setup streams the phone with Record3D over USB ([CAP xArm guide](https://cap-policy.gitbook.io/guide/xarm.md)). Record3D needs a LiDAR or Face ID (TrueDepth) iPhone ([record3d](https://pypi.org/project/record3d)); the SE 2020 has neither, so **Record3D will not work on the SE.** Try, in this order:
   - **AnySense** (NYU's data app, iOS 17+, USB streaming via `anysense-streaming`; [App Store](https://apps.apple.com/mt/app/anysense/id6742254654)). Install it on the SE and check whether it streams RGB over USB. Its listing doesn't say whether it needs LiDAR.
   - **Continuity Camera** on a Mac: the SE appears as a webcam (QuickTime → New Movie Recording → camera menu). Check that the picture is the full uncropped wide camera.
   - Fallback: a small own iOS app (AVFoundation → USB), an afternoon of work.
2. **Field of view.** The SE's camera is ~28 mm equivalent, the 13 Pro's main camera 26 mm: the SE sees ~7% less. Fine for the 224 × 224 policy input; check that both fingertips are in the picture.
3. **Where the lens sits.** The CAP phone holder places the 13 Pro so its camera block touches the holder. The SE is smaller and its lens is elsewhere. A printed spacer must put the SE's lens where the 13 Pro's main lens was (±3 mm), because the policy expects the fingers at a fixed spot in the image. Measure both lens centres with calipers from the phone's top and side edges; the spacer is then one CAD function (not drawn yet). [ir-camera.md](ir-camera.md) measures how much lens offset the pick skill tolerates: ±5 mm costs at most 2 points.
4. **Low light and heat.** A 20-minute USB stream in a dim room: is the picture usable with the lamp on, and does the phone throttle?

## Phone with LiDAR

A phone with LiDAR (iPhone 12 Pro or 13 Pro; the 13 Pro is CAP's own phone) matches the image exactly, runs Record3D and has depth built in. Dedicated depth cameras (Luxonis OAK-D SR, Orbbec Gemini 335, Intel RealSense D405) would add depth next to the phone but break the image match if they replace it.
