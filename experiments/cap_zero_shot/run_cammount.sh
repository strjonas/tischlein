#!/bin/zsh
# 27 Oct: how exactly must a printed holder place an infrared camera's lens? Same rig and episodes as
# run_camswap.sh (grey images, 150 episodes, contact point frozen). Lateral, vertical and axial offsets.
cd "$(dirname "$0")/../.."
PY=third_party/EgoGym/.venv/bin/python
for cam in shift:0.005 shift:0.01 up:0.01 back:0.005 back:0.01 back:0.02; do
  $PY experiments/cap_zero_shot/run_eval.py --task pick --robot capnear --episodes 150 --workers ${W:-5} \
    --anchor frozen --image gray --cam $cam --tag v3_frozen_gray_${cam/:/} 2>&1 | grep -E "% of|Error|Traceback"
done
