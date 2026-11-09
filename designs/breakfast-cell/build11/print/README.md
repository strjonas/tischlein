# Build11: first prints

A fit-test kit and the door pins. Made by [fit_kit.py](fit_kit.py); [preview.png](preview.png) shows them as they lie on the bed.

| File | What it is | Print time, roughly |
|---|---|---|
| `fit_gauge.stl` | Bearing seats, holes and nut pockets, 4 sizes each | 1.5–2 h |
| `servo_frames.stl` | 3 rings that slide over an STS3215 | 30 min |
| `slot_keys.stl` | 3 keys for the V-slot opening | 10 min |
| `pin_fridge.stl` | Outer fridge door pin (pin 77.5 mm below the arm) | 2 h |
| `pin_microwave.stl` | Microwave door pin (pin 65 mm below the arm) | 2 h |

The rest of build11's printed parts are **not print-ready yet**: joint housings, pulleys, wrist and hand. The 3D model draws them at catalogue size for the physics, without screw holes, bearing seats or clearances. Those clearances should come from your printer, which is what the gauge measures.

## Settings

- **Material:** PETG if you have it, the same material the real parts will use. PLA is fine for the gauge, but its clearances differ slightly from PETG.
- **Gauge, frames, keys:** 0.2 mm layers, 3 walls, 15% infill.
- **Pins:** 0.2 mm layers, 4 walls, 40% gyroid infill. They lie on a flat side and need no supports. If the first millimetre under the round pin curls, turn on "support on build plate only".

## Reading the gauge

On each row, sizes increase from left to right. The notched corner is front left.

| Row (back to front) | Sizes, mm | You want |
|---|---|---|
| 608 bearing seats | 22.0, 22.1, 22.2, 22.3 | The bearing presses in with your thumb and doesn't fall out |
| M8 holes | 8.0, 8.2, 8.4, 8.6 | The joint bolt slides through without wobbling |
| M5 holes | 5.0, 5.2, 5.4, 5.6 | Clearance for the M5 screws |
| M3 holes | 3.0, 3.2, 3.4, 3.6 | Clearance for the M3 screws |
| M5 nut pockets | 8.0, 8.2, 8.4, 8.6 across flats | The nut presses in and stays |
| M3 nut pockets | 5.5, 5.7, 5.9, 6.1 across flats | The nut presses in and stays |

- **Servo frames:** 1, 2 or 3 bumps mean +0.2, +0.3 or +0.4 mm. You want the ring that slides onto the servo body with light friction.
- **Slot keys:** 5.8, 6.0 and 6.2 mm. You want the one that slides into the V-slot without play.

The bearing seats and holes need only a 608 bearing and M8, M5 and M3 bolts and nuts.

Write down one number per row. Those numbers set the clearances in the real parts.

## Door pins

- **Geometry:** these follow `model11.py`: a 22 mm pin, its axis 70 mm from the door face, hanging from a 15 mm arm, with a domed end so the hand slides off it.
- **Changed from the model:** the pad is 70 mm tall instead of 45. That gives more tape against the pull's turning force. The extra height sits above the arm, where the hand never goes. A rib above the arm stiffens the joint between arm and pad.
- **Fitting:** the wide 40 × 70 mm wall is the face that takes the VHB tape. Clean the door with alcohol first, and press for 30 s.
- **Not in the kit:** the inner fridge pin clips onto the door rack, so its shape depends on the fridge.

```bash
uv run python designs/breakfast-cell/build11/print/fit_kit.py
```
