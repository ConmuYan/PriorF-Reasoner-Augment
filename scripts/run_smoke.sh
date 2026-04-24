#!/usr/bin/env bash
# Fail-closed smoke launcher.
#
# Semantics:
#   * runs in `diagnostic` namespace; never enters the formal namespace.
#   * `set -euo pipefail` + explicit stage chaining: if any stage fails,
#     the launcher exits non-zero immediately and does NOT continue to
#     later metric checks.
#   * does NOT accept --mode; smoke is always diagnostic. Callers that
#     want a gated or formal pipeline must use run_full_pipeline.sh
#     directly.
#
# Smoke is a diagnostic sanity check on the canonical plumbing.  A green
# smoke run is explicitly NOT evidence that formal metrics are valid --
# see README.md "What does not count as success".

set -euo pipefail

output_root="outputs"
subset_size="8"

usage() {
  cat <<'EOF'
Usage: scripts/run_smoke.sh [--output-root DIR] [--subset-size N]

Runs a minimal diagnostic smoke sweep:
  1. data_validation_pass  (graph_data.validators on a tiny subset)
  2. evidence_card_smoke   (build_evidence_card round-trip)
  3. prompt_builder_smoke  (prompt_builder.build_prompt round-trip)
  4. head_scoring_smoke    (score_head on mocked torch tensors if available)

Any stage failure aborts the pipeline.  No training happens.  No formal
artifacts are produced.  Output always goes under outputs/diagnostic/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --subset-size)
      subset_size="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "run_smoke: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

namespace_dir="$output_root/diagnostic/smoke"
mkdir -p "$namespace_dir"
export PRIORF_OUTPUT_NAMESPACE="diagnostic"
export PRIORF_OUTPUT_DIR="$namespace_dir"
export PRIORF_SMOKE_SUBSET_SIZE="$subset_size"

echo "run_smoke: namespace=diagnostic output_dir=$namespace_dir subset_size=$subset_size"

# Any failure in pytest short-circuits before the next step.
python -m pytest tests/test_smoke_pipeline.py -q

echo "run_smoke: OK (diagnostic)"
