#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/coco/val2014" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
image_folder="$1"
mkdir -p outputs

python eval/chair_generate.py \
  --model-size 7b --image-folder "$image_folder" \
  --gamma 0 --output outputs/llava_next_7b_vanilla.jsonl
python eval/chair_generate.py \
  --model-size 7b --image-folder "$image_folder" \
  --gamma 2 --output outputs/llava_next_7b_acg.jsonl

python eval/chair_generate.py \
  --model-size 13b --image-folder "$image_folder" \
  --gamma 0 --output outputs/llava_next_13b_vanilla.jsonl
python eval/chair_generate.py \
  --model-size 13b --image-folder "$image_folder" \
  --gamma 2 --output outputs/llava_next_13b_acg.jsonl
