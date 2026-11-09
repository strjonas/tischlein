#!/bin/zsh
# v3 reruns of the embodiments whose base moves (or link check matters) and that run_v3.sh didn't cover.
cd "$(dirname "$0")/../.."
PY=third_party/EgoGym/.venv/bin/python
run() { $PY experiments/cap_zero_shot/run_eval.py --task $1 --robot $2 --episodes $3 --workers ${W:-9} --tag v3 2>&1 | grep -E "% of|Error|Traceback"; }
run pick tisch6 150; run pick tisch6fixed 150; run close tisch6 100; run open tisch5 100; run open tisch6 100
