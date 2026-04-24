#!/usr/bin/env bash
# End-to-end Stage-2 pipeline driver for one dataset.
#
# Chains:
#   1. legacy .mat  -> canonical .mat       (scripts/legacy_mat_to_canonical.py)
#   2. canonical .mat + teacher checkpoint -> canonical TeacherExportRecord
#      parquet (train / validation / final_test) + DataManifest + validated
#      TeacherBaselineReport                 (scripts/generate_teacher_exports.py)
#   3. Stage-2 training loop over TRAIN population with PEFT LoRA + cls_head
#                                             (scripts/run_stage2_train.py)
#   4. Head-only inference on final_test     (scripts/run_stage2_inference.py)
#   5. Formal head-only eval (validation -> threshold freeze -> final_test)
#                                             (scripts/run_formal_head_only_eval.py)
#
# This is a gated-namespace driver.  Formal eval (step 5) writes to
# outputs/formal/head_only/<dataset>/.  Everything else stays under
# outputs/gated/.

set -euo pipefail

dataset=""
qwen_path=""
max_steps=40
batch_size=2
max_train_samples=1024
validation_subset=128
final_test_subset=256
gpu_index=0
run_id=""
skip_regenerate_exports=0
output_root="."

usage() {
  cat <<'EOF'
Usage: scripts/run_full_stage2_pipeline.sh --dataset <amazon|yelpchi> --qwen-path PATH [options]

Required:
  --dataset <amazon|yelpchi>
  --qwen-path PATH         local HF checkpoint (e.g. /data1/mq/models/Qwen3-4B-Instruct-2507)

Optional:
  --max-steps N            training steps (default: 40)
  --batch-size N           per-step samples (default: 2)
  --max-train-samples N    cap on TRAIN rows used (default: 1024)
  --validation-subset N    eval validation rows (default: 128)
  --final-test-subset N    eval final_test rows (default: 256)
  --gpu-index N            GPU index (default: 0)
  --run-id RUN             identifier suffix (default: <dataset>_run_v1)
  --skip-teacher-exports   reuse existing outputs/gated/teacher_exports/... + manifests/...
  --output-root DIR        where outputs/ and manifests/ live (default: .)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --qwen-path) qwen_path="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --max-train-samples) max_train_samples="$2"; shift 2 ;;
    --validation-subset) validation_subset="$2"; shift 2 ;;
    --final-test-subset) final_test_subset="$2"; shift 2 ;;
    --gpu-index) gpu_index="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --skip-teacher-exports) skip_regenerate_exports=1; shift ;;
    --output-root) output_root="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$dataset" || -z "$qwen_path" ]]; then
  echo "--dataset and --qwen-path are required" >&2
  usage >&2
  exit 1
fi
if [[ "$dataset" != "amazon" && "$dataset" != "yelpchi" ]]; then
  echo "--dataset must be amazon or yelpchi" >&2
  exit 1
fi
if [[ -z "$run_id" ]]; then
  run_id="${dataset}_run_v1"
fi

commit="$(git -C "$output_root" rev-parse HEAD 2>/dev/null || git rev-parse HEAD)"
export PYTHONPATH="${output_root}:${output_root}/priorf_gnn:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_index}"

train_parquet="$output_root/outputs/gated/teacher_exports/$dataset/train/teacher_export.parquet"
val_parquet="$output_root/outputs/gated/teacher_exports/$dataset/validation/teacher_export.parquet"
test_parquet="$output_root/outputs/gated/teacher_exports/$dataset/final_test/teacher_export.parquet"
data_manifest="$output_root/manifests/$dataset/data_manifest.json"
stage2_dir="$output_root/outputs/gated/stage2/$dataset/$run_id"
eval_dir="$output_root/outputs/gated/eval/$dataset"
formal_dir="$output_root/outputs/formal/head_only/$dataset"

if [[ $skip_regenerate_exports -eq 0 ]]; then
  case "$dataset" in
    amazon)  canonical_mat="$output_root/assets/data/Amazon_canonical.mat" ;;
    yelpchi) canonical_mat="$output_root/assets/data/YelpChi_canonical.mat" ;;
  esac
  if [[ ! -f "$canonical_mat" ]]; then
    echo "[step 1/5] legacy .mat -> canonical .mat ($dataset)"
    python "$output_root/scripts/legacy_mat_to_canonical.py" --dataset "$dataset"
  else
    echo "[step 1/5] canonical .mat already present ($canonical_mat)"
  fi

  if [[ ! -f "$train_parquet" || ! -f "$val_parquet" || ! -f "$test_parquet" || ! -f "$data_manifest" ]]; then
    echo "[step 2/5] running teacher inference -> canonical TeacherExportRecord parquet ($dataset)"
    python "$output_root/scripts/generate_teacher_exports.py" --dataset "$dataset" --output-dir "$output_root"
  else
    echo "[step 2/5] canonical teacher exports already present for $dataset"
  fi
else
  echo "[step 1-2/5] skipped teacher export regeneration (--skip-teacher-exports)"
fi

echo "[step 3/5] Stage-2 training ($dataset, max_steps=$max_steps, batch_size=$batch_size)"
python "$output_root/scripts/run_stage2_train.py" \
  --dataset "$dataset" \
  --qwen-path "$qwen_path" \
  --teacher-export-train "$train_parquet" \
  --teacher-export-validation "$val_parquet" \
  --data-manifest "$data_manifest" \
  --output-dir "$stage2_dir" \
  --max-steps "$max_steps" \
  --batch-size "$batch_size" \
  --max-train-samples "$max_train_samples" \
  --validation-subset "$validation_subset" \
  --gpu-index 0

config_fp="$(python -c "import json; print(json.load(open('$stage2_dir/run_record.json'))['_runtime_provenance']['config_fingerprint_sha256'])")"

echo "[step 4/5] Head-only inference on final_test ($dataset)"
python "$output_root/scripts/run_stage2_inference.py" \
  --dataset "$dataset" \
  --qwen-path "$qwen_path" \
  --peft-adapter "$stage2_dir/peft_adapter" \
  --cls-head "$stage2_dir/cls_head.pt" \
  --teacher-export "$test_parquet" \
  --data-manifest "$data_manifest" \
  --population final_test \
  --output-dir "$eval_dir" \
  --subset-size "$final_test_subset" \
  --step "$max_steps" \
  --gpu-index 0

echo "[step 5/5] Formal head-only eval ($dataset) -> outputs/formal/head_only/$dataset/"
python "$output_root/scripts/run_formal_head_only_eval.py" \
  --dataset "$dataset" \
  --qwen-path "$qwen_path" \
  --peft-adapter "$stage2_dir/peft_adapter" \
  --cls-head "$stage2_dir/cls_head.pt" \
  --teacher-export-validation "$val_parquet" \
  --teacher-export-final-test "$test_parquet" \
  --data-manifest "$data_manifest" \
  --output-dir "$formal_dir" \
  --validation-subset "$validation_subset" \
  --final-test-subset "$final_test_subset" \
  --run-id "$run_id" \
  --commit "$commit" \
  --config-fingerprint "$config_fp" \
  --step "$max_steps" \
  --include-oracle-diagnostics \
  --gpu-index 0

echo
echo "=== FULL STAGE-2 PIPELINE OK ($dataset) ==="
echo "  stage2 run   : $stage2_dir"
echo "  eval report  : $eval_dir/scorer_report_${dataset}_final_test.json"
echo "  formal report: $formal_dir/formal_head_only_report_${dataset}.json"
