#!/usr/bin/env bash
# Stage 1 launcher: structured generation SFT (gated namespace).
#
# Stage 1 is TRL SFTTrainer on the canonical Evidence Card -> strict JSON
# prompt/completion format.  It is gated (writes under outputs/gated/),
# not formal: a green Stage 1 does not by itself allow formal reporting;
# Stage 2 canonical joint training + gate manifest are required first.

set -euo pipefail

output_root="outputs"
gate_manifest=""
extra_args=()

usage() {
  cat <<'EOF'
Usage: scripts/run_stage1.sh [--output-root DIR] [--gate-manifest PATH] [-- command ...]

  --output-root DIR      Output root directory (default: outputs).
  --gate-manifest PATH   Optional.  If present, must pass gate_check; mode
                         becomes `formal`.  Otherwise mode is `gated`.
  -- command ...         Explicit Stage 1 entrypoint to execute (usually a
                         TRL SFTTrainer launcher).  If omitted, only the
                         namespace directory is printed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --gate-manifest)
      gate_manifest="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      echo "run_stage1: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$gate_manifest" ]]; then
  exec scripts/run_full_pipeline.sh \
    --mode formal \
    --gate-manifest "$gate_manifest" \
    --output-root "$output_root" \
    -- "${extra_args[@]}"
else
  exec scripts/run_full_pipeline.sh \
    --mode gated \
    --output-root "$output_root" \
    -- "${extra_args[@]}"
fi
