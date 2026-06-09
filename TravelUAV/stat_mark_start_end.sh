#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/data/TravelUAV/Dataset/newdata/Carla_Town01"
OUT="/mnt/data/TravelUAV/Dataset/newdata/Carla_Town01_mark_start_end.csv"

echo "task_name,start_x,start_y,start_z,end_x,end_y,end_z,target_x,target_y,target_z" > "$OUT"

find "$BASE" -mindepth 2 -maxdepth 2 -name "mark.json" | sort | while read -r mark; do
    task_dir="$(dirname "$mark")"
    task_name="$(basename "$task_dir")"

    python3 - "$mark" "$task_name" >> "$OUT" <<'PY'
import json
import sys

mark_path = sys.argv[1]
task_name = sys.argv[2]

try:
    with open(mark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start = data.get("start", [None, None, None])
    end = data.get("end", [None, None, None])
    target = data.get("target", {}).get("position", [None, None, None])

    print(
        f"{task_name},"
        f"{start[0]},{start[1]},{start[2]},"
        f"{end[0]},{end[1]},{end[2]},"
        f"{target[0]},{target[1]},{target[2]}"
    )

except Exception as e:
    print(f"{task_name},ERROR,{e},,,,,,,")
PY

done

echo "Done."
echo "Saved to: $OUT"
