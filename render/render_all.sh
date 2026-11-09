#!/bin/zsh
# Render every scene. Camera positions in metres (robot frame).
cd "$(dirname "$0")/.."
B=/Applications/Blender.app/Contents/MacOS/Blender
r() { $B -b -P render/blender_render.py -- out/scene_$1.glb out/render_$2.png ${@:3} 2>&1 | grep -E "^wrote|Error" }
r hero   hero        2.10 -1.45 1.35   0.18 0.0 0.70  30
r hero   arm_closeup 0.95 -0.80 1.10   0.34 0.05 0.62 42
r fridge fridge      -0.85 -1.25 1.85  0.62 0.05 0.95 30
r bed    bed         -0.95 -1.75 1.60  0.45 0.0 0.60  30
r floor  floor       1.55 -1.25 1.00   0.40 0.0 0.25  33
