#!/bin/zsh
# The runs behind the v2 table (9 worker processes; ~4-9 min per pick run, ~2-3 min per close run).
cd "$(dirname "$0")/../.."
PY=third_party/EgoGym/.venv/bin/python
run() {  # task robot episodes
  $PY experiments/cap_zero_shot/run_eval.py --task $1 --robot $2 --episodes $3 --workers ${W:-9} --tag v2 2>&1 | grep -E "% of|Error|Traceback"
}
for robot in cap tisch5 tisch6 tisch6sel tisch6fixed capnear tisch6near benchlift benchnolift; do run pick $robot 150; done
for robot in cap tisch5 tisch6 tisch6sel; do run close $robot 100; done
for robot in cap tisch5 tisch6; do run open $robot 100; done
