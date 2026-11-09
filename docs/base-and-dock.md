# Base, bench stand and docking station

## The bench stand: use the robot's own base frame, not plywood

A plywood foot for the mast is a part thrown away later. Better: **build the real base frame first and leave the drive parts out.**

- **What it is:** the 2020 aluminium frame with its straddle legs, the mast foot and gussets, and the deck. No hub motors, battery or drive electronics yet.
- **Feet:** four fixed rubber feet where the wheels and front casters will go, so it can't roll during tests.
- **Gain:** the mast, carriage and arm are tested on their real foot, at their real height, with the real flex. Later the hub motors, battery and electronics bolt into the frame that already exists.
- **Ballast is required.** Without the 5.6 kg of hub motors and the battery, the stand tips forward with the arms stretched out at the top (safety factor 0.8, `calc.py`). With ≥5 kg at the back of the deck (a sandbag, a dumbbell plate, later the battery) it's 2.5–2.6; with 8 kg, 3.5–3.8.
- **Alternative:** clamp the mast foot to a heavy table.

## What lives in the base

The base is where the weight should be: low and toward the back.

| Part | Where | Why there |
|---|---|---|
| Hub motors (2 × 2.8 kg) | on the axle, 12 cm forward of the frame's old axle line | Turning circle 0.88 m at tray height |
| Battery | rear, as low as possible | It's the counterweight: LiFePO4 (2.6 kg) instead of the hoverboard pack (1.4 kg) lifts the "5 kg parcel over the bed" tipping factor from 1.9 to 2.2 |
| Electronics box: hoverboard board or 2× ZS-X11H, ESP32, bucks, fuse block, e-stop | rear, above the battery | Short wires to the battery and motors |
| Lidar (RPLIDAR C1) | on a post at the rear, above the parked hands | 360° view over the base |
| Charging contacts | rear face, ~7 cm high | The robot reverses into the dock and faces the bed while charging |
| Later: Pi 5 | electronics box | |

**Deck material.** The frame carries the loads; the deck only holds things down and covers them.
- 6 mm birch plywood is fine and the easiest to drill.
- Alternatively 3 mm aluminium composite (Dibond), which doesn't swell or splinter.
- Either way the shell covers it later.

## Docking station: the robot charges itself

The user should not have to put the robot on a charger. The robot drives onto a dock, the way robot vacuums do. Of the options, the dock is the most reliable, because it needs only driving, no manipulation:

| Option | How | Verdict |
|---|---|---|
| **Contact dock** | Two spring contacts on the dock, two brass pads on the robot's rear. The robot finds the dock with the lidar (a V-notch target at lidar height) and reverses the last 30 cm straight. A printed funnel corrects ±3 cm | **Yes.** Driving only; robot vacuums show ~99% docking |
| Magnetic connector in a funnel | Same, but a 2-pin magnetic pogo connector (10 A class) instead of open contacts; the magnet does the last few mm | Good variant: no exposed contacts. Check the connector's real current rating first; vendor ratings vary from 2 to 25 A |
| Robot plugs in a cable with its hand | The hand picks up a magnetic plug and puts it on a socket on its own base | No: a manipulation skill that must work every single night. Keep it as a manual fallback (a person can plug in the same plug) |

**The dock, concretely:**
- **Where:** beside the bed, against the wall. The robot reverses in and its front, with the tray and the drawer, faces the bed. So while charging it is the bedside table with the lamp on, exactly where it is needed.
- **Contacts:** live only when the robot is there. A reed switch in the dock sees a magnet on the robot and switches the charger output through a relay, with a 5 A fuse. The charger voltage (29.2 V for LiFePO4 8S) is safe to touch; the interlock is against shorts from keys or coins.
- **Charging:** a 3 A LiFePO4 charger, 0–100% in ~3.5 h. The robot's BMS decides; the ESP32 reports the state.

**What else happens at the dock (the "reset"):**
- The lift homes on its lower limit switch: this catches a skipped belt tooth.
- The arms tuck, and each servo's position is compared with the tuck pose.
- The dock is the map origin: the robot relocalises there.
- A calibration tag next to the dock: each wrist camera looks at it after docking, and the result is compared with the kinematics. This catches a bent phone mount or a slipped belt.
- At night, while charging: copy the day's recordings to storage, install software updates, restart.

**Before the lidar exists:** dock under teleop, or plug the same magnetic plug in by hand.

**CAD:** not drawn yet. Contact block, funnel, robot-side pad holder and lidar V-target: next CAD session, after the real base frame and battery are measured.
