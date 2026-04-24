#!/usr/bin/env bash
# Stage 2 launcher: canonical joint trainer (gated by default, formal on request).
#
# Stage 2 runs train.train_stage2_canonical (Accelerate + L_gen + L_cls +
# L_distill) in the gated namespace.  Formal mode is opt-in: only with
# --gate-manifest and a passing gate_check.

set -euo pipefail

output_root="outputs"
gate_manifest=""
extra_args=()

usage() {
  cat <<'EOF'
Usage: scripts/run_stage2.sh [--output-root DIR] [--gate-manifest PATH] [-- command ...]

  --output-root DIR      Output root directory (default: outputs).
  --gate-manifest PATH   Optional.  Present => formal mode via
                         run_full_pipeline.sh; absent => gated mode.
  -- command ...         Stage 2 entrypoint (typically
                         `python -m train.train_stage2_canonical ...`).
                         If omitted, only the namespace directory is
                         printed for the caller to inspect.
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
      echo "run_stage2: unknown argument: $1" >&2
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
