#!/usr/bin/env bash
# Formal evaluation launcher.
#
# Only accepts formal mode, and only with a valid gate manifest.  Any
# attempt to invoke formal evaluation without a passing gate_check exits
# non-zero without creating any output directory.  Diagnostic / gated
# evaluation callers must use run_full_pipeline.sh directly.

set -euo pipefail

output_root="outputs"
gate_manifest=""
extra_args=()

usage() {
  cat <<'EOF'
Usage: scripts/run_eval.sh --gate-manifest PATH [--output-root DIR] [-- command ...]

Runs formal evaluation.  A valid gate manifest is required; the launcher
invokes scripts/gate_check.py before any evaluation command.

  --gate-manifest PATH   (required) path to gate_manifest.json
  --output-root DIR      output root directory (default: outputs)
  -- command ...         optional eval entrypoint (e.g.
                         `python -m eval.eval_head_only ...`).

On missing gate manifest, failing gate_check, or any failure in the
provided command, the launcher exits non-zero.
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
      echo "run_eval: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$gate_manifest" ]]; then
  echo "run_eval: --gate-manifest is required for formal evaluation" >&2
  exit 1
fi

exec scripts/run_full_pipeline.sh \
  --mode formal \
  --gate-manifest "$gate_manifest" \
  --output-root "$output_root" \
  -- "${extra_args[@]}"
