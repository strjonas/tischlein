# Touchdown sensing for the put-down

Build11 lets go of the meal where the plan says the support is. The robot has no sensor that tells it the support has taken the weight, and in physics letting go where the plan says has about 3 mm of margin with the box. This study asks whether feeling the touchdown, using the servos' own load readings, fixes that without a new sensor. It is a separate placement study; the results in the [build11 README](README.md) use the plan's release rule.

**Result.** Lowering until the touch is felt placed every meal gently. The STS3215s already report their load: for the jar the wrist pitch servo feels the touch clearly, for the box the lift servo does, if its reading is no noisier than about 2% of stall. A load cell between wrist and hand works for both and is the fallback.

## What feeling the touchdown changes

[senses11.py](senses11.py) uses dyn11's physics: the same servos, friction and random errors. The meal is put down into the microwave with the plate higher or lower than the robot believes, by e = −10 to +10 mm. Causes would be a plate sitting high on its roller ring, belt stretch or a lift count that drifted. Seven values of e × 10 random trials = 70 put-downs per row, all rows on the same trials.

| How it decides to let go | Jar: placed | worst drop | worst push on the plate | Box: placed | worst drop | worst push |
|---|---:|---:|---:|---:|---:|---:|
| **At the planned height** (40 mm/s) | 70/70 | 7.2 mm | 125 N | **50/70** | 4.3 mm | **365 N** |
| Same, slower (10 mm/s) | 70/70 | 6.9 mm | 74 N | 50/70 | 4.2 mm | 346 N |
| Lift servo reading, noise 0.5% of stall | 70/70 | 0.6 mm | 23 N | 70/70 | 0 | 26 N |
| Lift servo reading, noise 2% | 68/70 | 9.9 mm | 93 N | 70/70 | 0 | 29 N |
| Lift servo reading, noise 5% | 67/70 | 16.4 mm | 139 N | 60/70 | 4.2 mm | 30 N |
| **Wrist pitch servo reading**, noise 0.5 / 2 / 5% | 70/70 each | 0.7 mm | 26 N | (box: nothing to feel) | | |
| Load cell in the wrist (0.1 N noise) | 70/70 | 0.6 mm | 26 N | 70/70 | 0 | 26 N |

"Feeling" here means: lower at 10 mm/s and read the servos at 50 Hz. Stop when the reading has dropped by 60% of the meal's weight for three readings in a row, then let go. Then finish the plan shifted by the height where the touch was felt. If nothing is felt 15 mm below the plan, stop and keep holding. This never happened.

**What the plan does at the planned height:**

- **Box.** After letting go, the fork drops 3 mm below the box and slides out. If the plate is 5 mm or more lower than the robot thinks, the fork never gets out from under the box: it pulls the box about 15 cm back out of the microwave. That was all 20 failures. If the plate is 10 mm higher, the fork pushes the box into the glass plate with up to 365 N.
- **Jar.** The jar sits 3–6 mm lower in the V-fingers than the planner assumes. So even with no error it touches down early, presses the plate with about 25 N and slides up in the fingers. With ±10 mm of error it drops 7 mm or presses with 75 N. It still ends up upright every time.

**With feeling,** neither menu dropped more than 0.7 mm or pressed harder than about 29 N, which is the meal's own weight plus a few newtons. It costs about 1.5 s per put-down, roughly 3 s a morning.

**Which reading to use:**

| Menu | Use | Why |
|---|---|---|
| Jar | Wrist pitch servo | The jar is 29% of the pitch servo's stall torque. It works even at 5% noise. Through the lift it is only 4% of stall and needs ≤ 0.5% noise. |
| Box | Lift servo | The box sits over the pitch axis, so the pitch feels nothing. It is 9% of the lift's stall: fine at 2% noise, failed 10/70 at 5%. |

The wrist load cell works for both because it does not see the lift's friction. It lengthens the hand slightly, and the pitch servo already holds the jar at 89% of its rating.

**Does it hold up?** Three trials at ±5 mm were rerun with softer contacts (solref 0.008) and a finer timestep (1 ms) ([senses11-sweep.json](senses11-sweep.json)):
- The box failure at −5 mm appears in every setting.
- Every felt put-down succeeds in every setting.
- The push forces vary with contact stiffness: the planned box at +5 mm pressed with 57–79 N. Read the forces as a range, not a prediction.
- MuJoCo clamps the stiffest setting tried (0.002) to the default at a 2 ms timestep, so that row repeats the default.

## Bench check

| Check | Pass |
|---|---|
| Lower the jar and the box onto the microwave plate at 10 mm/s, 20 times each, with 5 mm shims under the plate and without. Log the pitch and lift load at 50 Hz. | Lift reading noise ≤ 2% of stall while lowering (box); pitch ≤ 5% (jar). 20/20 felt put-downs: no audible drop, the box never dragged. |

## What this does not show

- **The real STS3215 reading.** Its "load" register is the motor's drive effort (PWM duty), not a measured torque, and its current register has 6.5 mA steps. The model uses the true actuator force plus Gaussian noise. Gear friction and deadband may make the real reading worse.
- **Other put-downs.** Only the put-down into the microwave was tested. The serving put-down and the take from the fridge use the same mechanism but were not run.
- **Breakage.** Glass breaking is not modelled. The push and impact forces depend on contact stiffness.

## Reproduce

```bash
uv run python designs/breakfast-cell/build11/senses11.py 10
```
```bash
uv run python designs/breakfast-cell/build11/senses11.py --sweep
```

The first command writes `senses11-results.json` in about 90 s. The second writes `senses11-sweep.json`.
