#!/bin/zsh
# 27 Oct: could a cheap mono infrared camera replace the iPhone? Same rig and episodes as pick_capnear_v3_frozen_w5
# (colour, iPhone-like 53.1 deg vertical field of view: 80.7 %). Grey images plus a different lens or mounting.
cd "$(dirname "$0")/../.."
PY=third_party/EgoGym/.venv/bin/python
for cam in fov:40 fov:70 fov:90 shift:0.02 shift:0.04; do
  $PY experiments/cap_zero_shot/run_eval.py --task pick --robot capnear --episodes 150 --workers ${W:-5} \
    --anchor frozen --image gray --cam $cam --tag v3_frozen_gray_${cam/:/} 2>&1 | grep -E "% of|Error|Traceback"
done
