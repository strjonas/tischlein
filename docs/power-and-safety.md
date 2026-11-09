# Power, the lift on power loss, and wiring

Two points about v0.2 needed fixing: the lift didn't really hold a payload without power, and the bench wiring was underspecified. Numbers are in `out/calc_report.txt` (`calc.py`).

## The lift when the power goes

**The question:** can a constant-force spring balancer keep the lift in place on power loss, with a relay that shorts the stepper coils as a second safety?

**What holds:**
- The balancer is set to the carriage's own weight (two arms, tray, face: ~4.1 kg). It carries exactly that. **Anything on the tray is not balanced:** a 5 kg parcel pulls 49 N down on an unpowered motor.
- Shorted stepper coils are a damper, not a brake. They turn motion into heat in the windings, so the faster the tray falls the harder they resist, but at standstill they hold nothing. The stepper's detent torque is negligible.
- Result (rough, `calc.py`): with power off and coils shorted, the tray **creeps down at ~1 mm/s with 1 kg, ~5 mm/s with 5 kg**: the full stroke in 5 minutes with a parcel. Slow, not dangerous, but not "stays put", and it ends on whatever is below the tray.

**Fix: a power-off brake on the lift motor.**
- A NEMA17 with a 24 V spring-applied brake: the brake grips when power is off and releases when 24 V is on. That is how every vertical machine axis is held. Typical brake torque 0.2–0.5 N·m (e.g. [igus drylin E](https://www.igus.co.uk/product/MOT-AN-S-060-005-042-M-D-AAAD) 0.5 N·m; JVL MAB17 0.5–0.7 N·m as an add-on).
- Through the 3:1 belt and the 20T pulley, 0.2 N·m holds ~94 N at the carriage and 0.4 N·m ~188 N. A 5 kg parcel plus a badly set balancer needs ~59 N. **0.4 N·m gives 3× margin.**
- The brake coil hangs on the 24 V lift supply behind the e-stop, so the e-stop also applies the brake.
- **Keep the balancer.** It no longer has a safety job. It halves the motor's work, which is what makes 5 kg parcels liftable (motor margin 0.8× without it, 1.5× with it at full speed), and it keeps the carriage from dropping if the belt ever breaks.
- **Drop the coil-shorting relay.** It added nothing once there's a brake.
- **Bench test:** 5 kg on the tray at 1.0 m, pull the plug, measure the drift after 10 minutes: < 2 mm passes. Then the same with the brake wire disconnected, to know the failure case (expect the creep above).

## Bench wiring

**The issue:**
- On the 12 V bus there are six Feetech servos (5× STS3215 at 2.7 A stall each, 1× STS3250 at 4.2 A). If all stalled at once that's ~18 A; a realistic peak (two joints near stall, the rest moving) is ~8 A.
- Feeding everything from one 12 V 10 A supply through the Waveshare bus adapter is outside what the adapter is specified for. That board is a signal converter with a barrel jack. Waveshare publishes no current rating for it ([wiki](https://docs.waveshare.com/Bus_Servo_Adapter_A)), and the servos' own 3-pin daisy-chain cables are good for ~2–3 A.
- A bigger supply doesn't make the adapter or the thin cables carry more.

**The wiring:**

```
12 V 15 A supply ── 15 A fuse ── e-stop (latching, 2-pole: 12 V + 24 V) ── fuse block ─┬─ 5 A  ─ J1 + J2 (shoulder, elbow)
                                                                    + 2200 µF          ├─ 7.5 A ─ wrist: yaw, pitch (STS3250), roll
                                                                                       └─ 3 A  ─ drawer servo
Data + GND: host ─ USB ─ Waveshare adapter ─ daisy-chained to every servo (the adapter powers nothing)
5 V 3 A ─ U2D2 power hub ─ XL330 (CAP gripper)          24 V 3 A ─ e-stop ─ TMC2209 + brake coil (lift)
All grounds joined at one point.
```

- **Power injection per branch:** each branch gets its 12 V straight from the fuse block, on 1.0 mm² wire. Only data and ground run through the servo-to-servo chain.
- **In each servo's EEPROM** (Feetech registers, set once):
  - protection current ~2 A (STS3215) / ~3 A (STS3250);
  - torque limit 80%;
  - max temperature 65 °C;
  - overload protection on.
  
  These are also the "stop on contact" behaviour near people.
- **Parts:** a 12 V 15 A supply instead of 10 A, a 6-way blade fuse block with fuses, a 2-pole latching e-stop, WAGO connectors, wire, a capacitor, 3-pin extension leads.

## On the robot

- **Battery:**
  - First, the hoverboard's own 36 V pack.
  - For use in a home, a LiFePO4 8S 25.6 V 10 Ah pack with BMS: 256 Wh, ~12 h mixed use. No used Li-ion in a bedroom.
- **Fuses, e-stop and supplies:**
  - A fuse at the battery.
  - The latching e-stop cuts the motor drivers, the servo bus and the lift (so the brake grips).
  - Bucks: 12 V for the servos, 24 V for the lift, 5 V for logic.
- **Charging:** at the dock, see [base-and-dock.md](base-and-dock.md).
