#!/usr/bin/env bash
set -euo pipefail

timing_base="/mnt/data/TravelUAV/result/timing/eval_town01"
mode="${1:-dry-run}"

tasks=(
08302fc9-8117-434e-a77f-46cabecf6571
13bb9524-b181-4ff5-83de-49b5405c17d0
13e34434-fedc-4653-a38b-6011e10ee205
1c2cc77c-cf12-41b9-8f9a-22116124504d
277ed81c-926e-4919-8812-ccf2a095f87d
2c4b0138-21f9-442a-a0a1-8720e2129c6a
2d082a0b-a7e6-44b1-b495-3d3baef4bc75
2f4d6b24-ded5-47a6-a957-229343d86d1f
3161e044-e3c8-41e5-aeaa-c01bb199bf78
375b8348-9c9b-4144-9fd0-b7db583e4ba3
37d7800b-ff9c-43ad-aecf-cc699e215e66
3cc234ed-e39c-45a2-9ff4-4f21af6da63f
3f7a028c-179f-491e-bf21-3f7309be5a23
4afa6177-e031-42ac-af19-a226706ec1f0
509cb681-b257-4cf9-9fe0-90126dcdd64f
6e52ed13-8996-4f26-a334-7a04f01b6e86
73b974d5-6067-45fd-8742-169be6c28740
75688799-d886-4b3a-915d-e36d5d20a18f
7ac04726-4c4e-4f93-81b4-b684d54bcf75
7ba26cc2-c676-40fe-83d5-0f8b3404146a
a3931b68-e769-4787-805c-b7f8af49282b
a43cfb70-9ca7-4584-8cfd-43b109084aa2
bb038237-392e-4f51-9948-ea903dadc99b
bdf52e38-938d-48f5-88f4-d4914f405df5
c30260ba-1b02-41a8-9b62-da1c459a33ed
c99dfc69-f19d-4d4e-bc2c-85d715791b72
cd5ea717-a5ca-4c89-b814-7e637911515d
df24d535-4158-4e9f-9226-4535e928336a
dfb01511-fdfe-4769-b38f-25b3a7e0f79c
eab95d31-01ca-4edc-ae70-95f464c28199
)

echo "Timing base: $timing_base"
echo "Mode: $mode"
echo

for task in "${tasks[@]}"; do
  d="$timing_base/$task"

  if [ -d "$d" ]; then
    echo "$task"
    echo "  found: $d"

    if [ "$mode" = "delete" ]; then
      rm -rf -- "$d"
      echo "  deleted"
    else
      echo "  dry-run only"
    fi
  else
    echo "$task"
    echo "  not found, skipped"
  fi

  echo
done
