#!/bin/zsh
# 27 Oct: can CAP's pick skill work at night? Same rig and seed as pick_capnear_v3_frozen (80 %), only the wrist
# camera image changes: mono, a crude near-infrared stand-in, 15 % light with sensor noise, and both.
cd "$(dirname "$0")/../.."
PY=third_party/EgoGym/.venv/bin/python
for img in gray red dark darkgray; do
  $PY experiments/cap_zero_shot/run_eval.py --task pick --robot capnear --episodes 150 --workers ${W:-6} \
    --anchor frozen --image $img --tag v3_frozen_$img 2>&1 | grep -E "% of|Error|Traceback"
done
