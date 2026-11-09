#!/bin/zsh
# v3 (26 Oct): diff-drive base integrated along its current heading (v2 let it slide sideways after turning),
# arm links checked against the table top / cabinet front, and the anchor-error sweep for the iPhone SE question.
cd "$(dirname "$0")/../.."
PY=third_party/EgoGym/.venv/bin/python
run() {  # task robot episodes [anchor] [tag]
  $PY experiments/cap_zero_shot/run_eval.py --task $1 --robot $2 --episodes $3 --workers ${W:-9} \
    --anchor "${4:-}" --tag ${5:-v3} 2>&1 | grep -E "% of|Error|Traceback"
}
for robot in tisch5 tisch6sel tisch6near benchlift benchnolift; do run pick $robot 150; done
for robot in tisch5 tisch6sel; do run close $robot 100; done
# depth source for the contact anchor, on the bench rig (arm + lift, no base)
run pick capnear 150 frozen v3_frozen
run pick benchlift 150 frozen v3_frozen
run pick benchlift 150 iso:0.01 v3_iso1
run pick benchlift 150 iso:0.02 v3_iso2
run pick benchlift 150 iso:0.04 v3_iso4
run pick benchlift 150 ray:0.08 v3_ray8
run pick benchlift 150 plane:0.02 v3_plane
