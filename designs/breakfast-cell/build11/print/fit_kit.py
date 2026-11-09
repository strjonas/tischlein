"""Build11's first prints: a fit-test kit and the door pins.

The rest of build11's printed parts (joint housings, pulleys, wrist, hand) are not print-ready
yet: model11.py draws them at catalogue size for the physics, without screw holes, bearing seats
or clearances. Those clearances should come from this printer, so the kit below measures them.

Writes into this folder:
  fit_gauge.stl       608 bearing seats, M8/M5/M3 holes, M5/M3 nut pockets, each in 4 sizes
  servo_frames.stl    3 rings that slide over an STS3215 body, +0.2/+0.3/+0.4 mm clearance
  slot_keys.stl       3 keys for the 2040 V-slot opening, 5.8/6.0/6.2 mm
  pin_fridge.stl      outer fridge door pin: 22 mm pin, 77.5 mm below the arm, domed end
  pin_microwave.stl   microwave door pin: the same with 65 mm below the arm
  preview.png         all parts, for a look before slicing

Door pins follow model11.py (build10's pin, 25 mm further off the door, 9 cm fridge pin):
pin axis 70 mm from the door face, arm 15 mm thick, pin top inside the arm. Two changes:
the pad is 70 mm tall instead of 45 (more tape against the pull's moment; the extra height
is above the arm, where the hand never goes), and a rib above the arm stiffens the root.
Each pin is drawn lying on one flat side so it prints without supports.

    uv run python designs/breakfast-cell/build11/print/fit_kit.py
"""
from pathlib import Path

from build123d import Axis, Box, Cylinder, Plane, Polygon, Pos, RegularPolygon, Rot, export_stl, extrude, fillet

HERE = Path(__file__).resolve().parent

COLS = (16, 44, 72, 100)                       # gauge columns, smallest size on the left (notched corner)
SEAT = (22.0, 22.1, 22.2, 22.3)                # 608: 22 mm outside, 7 mm wide
M8 = (8.0, 8.2, 8.4, 8.6)
M5 = (5.0, 5.2, 5.4, 5.6)
M3 = (3.0, 3.2, 3.4, 3.6)
HEX_M5 = (8.0, 8.2, 8.4, 8.6)                  # across flats; an M5 nut is 8 mm
HEX_M3 = (5.5, 5.7, 5.9, 6.1)                  # an M3 nut is 5.5 mm
STS3215 = (45.2, 24.7)                         # catalogue body, mm
SERVO_CLEAR = (0.2, 0.3, 0.4)
SLOT = (5.8, 6.0, 6.2)                         # 2040 V-slot opening is about 6.2 mm


def fit_gauge():
    t = 9.0
    g = Pos(58, 58, t / 2) * Box(116, 116, t)
    g -= Pos(0, 0, t / 2) * Rot(0, 0, 45) * Box(9, 9, t + 2)              # orientation notch, front left
    for x, d in zip(COLS, SEAT):                                         # bearing seats, 7.2 deep, push-out hole
        g -= Pos(x, 102, t - 3.6 + .01) * Cylinder(d / 2, 7.2)
        g -= Pos(x, 102, t / 2) * Cylinder(5, t + 2)
    for y, sizes in ((80, M8), (64, M5), (50, M3)):                      # through holes
        for x, d in zip(COLS, sizes):
            g -= Pos(x, y, t / 2) * Cylinder(d / 2, t + 2)
    for y, sizes in ((34, HEX_M5), (18, HEX_M3)):                        # nut pockets, 4 deep, with a 3 mm hole through
        for x, af in zip(COLS, sizes):
            g -= Pos(x, y, t - 4) * extrude(RegularPolygon(af / 2, 6, major_radius=False), 4.01)
            g -= Pos(x, y, t / 2) * Cylinder(1.5, t + 2)
    return g


def servo_frames():
    wall, h = 2.4, 6.0
    out = None
    for i, c in enumerate(SERVO_CLEAR):
        L, W = STS3215[0] + c, STS3215[1] + c
        ring = Pos(0, i * (W + 2 * wall + 6), h / 2) * (Box(L + 2 * wall, W + 2 * wall, h) - Box(L, W, h + 2))
        for k in range(i + 1):                                           # 1, 2 or 3 bumps say which ring it is
            ring += Pos(-L / 2 + 6 + 5 * k, i * (W + 2 * wall + 6) - W / 2 - wall / 2, h + .5) * Cylinder(.9, 1)
        out = ring if out is None else out + ring
    return out


def slot_keys():
    k = Pos(15, 12, 1.5) * Box(30, 24, 3)
    for i, w in enumerate(SLOT):
        k += Pos(5 + 10 * i, 12, 3 + 1) * Box(w, 18, 2)
    return k


def door_pin(below_arm):
    """Printed lying on the side x = 0. y: out from the door face; z: up the door when installed.
    Pin axis at x = 10 so the bed cuts a 9 mm flat along it."""
    r, axis_y = 11.0, 70.0
    pad = Pos(20, 2, 35) * Box(40, 4, 70)                                # VHB face at y = 0
    arm = Pos(10, 42, 17.5) * Box(24, 82, 15)                            # y 1..83 (overlaps the pad), z 10..25
    rib = extrude(Plane.YZ * Polygon((3, 24), (34, 24), (3, 55), align=None), 12)  # 12 mm rib above the arm,
                                                                         # on the bed side so it needs no support
    top, bottom = 22.5, 10.0 - below_arm
    pin = Pos(10, axis_y, (top + bottom) / 2) * Cylinder(r, top - bottom)
    pin = fillet(pin.edges().group_by(Axis.Z)[0], radius=10)            # domed end, 2 mm flat at the tip
    part = pad + arm + rib + pin
    part -= Pos(-50, 50, 0) * Box(100, 300, 400)                         # flat side on the bed: keep x >= 0
    return Rot(0, -90, 0) * part                                         # exported lying on that side (x -> z)


def main():
    parts = dict(fit_gauge=fit_gauge(), servo_frames=servo_frames(), slot_keys=slot_keys(),
                 pin_fridge=door_pin(77.5), pin_microwave=door_pin(65.0))
    for name, p in parts.items():
        export_stl(p, str(HERE / f'{name}.stl'))
        bb = p.bounding_box()
        print(f'{name:14s} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:6.1f} mm  {p.volume / 1000:6.1f} cm3')


if __name__ == '__main__':
    main()
