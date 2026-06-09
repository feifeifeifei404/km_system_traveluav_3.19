#!/usr/bin/env bash
set -euo pipefail

result_base="/mnt/data/TravelUAV/result/eval_closeloop/eval_town05"
timing_base="/mnt/data/TravelUAV/result/timing/eval_town05"
threshold=5
mode="${1:-dry-run}"

if [ ! -d "$result_base" ]; then
  echo "ERROR: result_base path not found: $result_base"
  exit 1
fi

echo "Result base: $result_base"
echo "Timing base: $timing_base"
echo "Delete condition: downcamera image count <= $threshold"
echo "Mode: $mode"
echo

for d in "$result_base"/*; do
  [ -d "$d" ] || continue

  task_name="$(basename "$d")"
  down="$d/downcamera"
  timing_dir="$timing_base/$task_name"

  [ -d "$down" ] || continue

  n=$(find "$down" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)

  if [ "$n" -le "$threshold" ]; then
    echo "$task_name  downcamera_images=$n"
    echo "  result_dir=$d"
    echo "  timing_dir=$timing_dir"

    if [ "$mode" = "delete" ]; then
      rm -rf -- "$d"
      echo "  deleted result_dir"

      if [ -d "$timing_dir" ]; then
        rm -rf -- "$timing_dir"
        echo "  deleted timing_dir"
      else
        echo "  timing_dir not found, skipped"
      fi
    else
      echo "  dry-run only"
    fi

    echo
  fi
done
