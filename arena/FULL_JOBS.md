# Complete-job protocol

The arena scores a robot on whole jobs, not isolated pickups. A job starts at the charging dock and ends only when the goal state holds in the physics world. Isolated picks (`arena.tier2`) and reach checks (`arena.tier1`) remain as diagnostics; their rates are never multiplied into a job score.

## What a complete job requires

The standard suite has seven jobs (`arena.jobs.CORE`):

| Job | Start state | Goal |
|---|---|---|
| `floor_tablet_box` | a box on the floor at one of the home's floor spots | on the person's table, upright, at rest, within reach |
| `floor_remote` | a remote on the floor | the same |
| `bedside_pill_bottle` | a pill bottle on the bedside table | the same |
| `fridge_bottle` | a bottle in the closed fridge | the same, with the fridge door closed again |
| `worktop_meal_box` | a prepared meal on the kitchen worktop | the same |
| `clear_mug` | an empty mug on the person's table | on the kitchen worktop, released and at rest |
| `microwave_meal` | the meal on the worktop | staged while one hand opens the microwave door, loaded, door closed, started, waited for, retrieved, delivered; door closed again |

`--scenarios all` expands the run to the full catalogue of 22 jobs.

Each trial starts at the dock with a finite docking error and a nominal initial pose estimate. A goal holds only if the object has been physically released onto the right support, is at rest, and, where it applies, is upright and within the person's reach. Tipping, spilling and person contact above 30 N reject a completion. Microwave heating is a timer and state proxy, not a thermal model; the whole object must spend ten seconds inside the closed, running oven.

Each seed runs in every requested home. Docking noise is drawn separately from object placement, so a sensor ablation does not change which object pose is tested. Homes A to D are the development set; E to H are held out and share the basic floor plan with different doors, mattress heights, thresholds, floors, fridge seals and hinges.

## Identity and resumption

Every result carries hashes of the source, the world model, the robot entry, the policy, the library versions and the seed. A log resumes only entries whose identity matches, so a change to code or robot cannot silently continue an old run. `python -m arena.lock` pins the files that define the test (homes, objects, physics settings, scenarios, judge, sensors and the reference controller) against `LOCK.json`.

## Collision geometry

- Nonadjacent arm links collide. Only same-body and direct parent/child pairs are filtered.
- Telescopes have fixed outer sleeves and overlapping hollow moving sections. Link inertia comes from the geometric envelope.
- Wrist drives, cameras and the lidar have cases. A battery, computer and power-electronics envelope shares the robot's mass budget. The chassis has driven-wheel and caster wells.
- The rotating bearing of the turning-carriage entries clears the mast at every angle.

These are engineering envelopes, not manufacturer CAD. Fasteners, cable routing, tolerances and fatigue are outside the model.

## Controller and judge

The controller sees sensors only: the mapped floor plan, its own kinematics at nominal values, camera detections, lidar, wheel odometry, jaw state and squeeze. Ground truth is available to the observer and to the judge, never to the controller (`tests/test_v2.py` checks the boundary).

Two policies share one judge:

- `reference`: two-stage motion planning with the corrected gripper firmware (jaw lag during blocked commands, release waits for measured opening).
- `improved`: adds local arm planning before base shuffles, capture-time camera transforms, upright carry, placement planned from the measured grasp transform, and collision-checked joint-space paths. Grasps require squeeze feedback and, when a fresh detection exists, visible lift.

Training (`arena.env`) uses the same judge: a flat observation and action vector over the same world, robot and sensors, with domain randomisation on by default. The flat vector omits lidar, wheel odometry and the appliance-aid commands that the scripted controllers use.

## Failure evidence

Each trial logs controller messages, phase transitions, motor fractions, tracking error, the controller's failure reason, judge-side snapshots of base, hand and object poses, jaw opening and force, and the peak contact pairs. The observer records whether the target was seen, lifted and transported without feeding those milestones back to the controller. `arena.contact_audit` also checks hand contact with furniture and the robot's own links, and reports completions without unintended contact above 40 N as a separate quality count.

Contact thresholds are engineering proxies, not safety criteria.
