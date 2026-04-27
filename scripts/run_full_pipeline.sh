#!/usr/bin/env bash
set -euo pipefail

mode=""
gate_manifest=""
output_root="outputs"
python_bin="${PYTHON:-}"

usage() {
  cat <<'EOF'
Usage: scripts/run_full_pipeline.sh --mode <formal|gated|diagnostic> [--gate-manifest PATH] [--output-root DIR] [-- command ...]

formal      Requires a valid gate manifest and runs gate_check before any command.
gated       Does not enter the formal namespace.
diagnostic  Does not enter the formal namespace.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --gate-manifest)
      gate_manifest="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "run_full_pipeline: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$mode" ]]; then
  echo "run_full_pipeline: --mode is required" >&2
  exit 1
fi

if [[ -z "$python_bin" ]]; then
  if [[ $# -gt 0 && "$(basename "$1")" == python* ]]; then
    python_bin="$1"
  elif command -v python >/dev/null 2>&1; then
    python_bin="python"
  else
    python_bin="python3"
  fi
fi

case "$mode" in
  formal)
    if [[ -z "$gate_manifest" ]]; then
      echo "run_full_pipeline: formal mode requires --gate-manifest" >&2
      exit 1
    fi
    "$python_bin" scripts/gate_check.py --manifest-path "$gate_manifest"
    namespace_dir="$output_root/formal"
    ;;
  gated)
    namespace_dir="$output_root/gated"
    ;;
  diagnostic)
    namespace_dir="$output_root/diagnostic"
    ;;
  *)
    echo "run_full_pipeline: unsupported mode: $mode" >&2
    exit 1
    ;;
esac

mkdir -p "$namespace_dir"
export PRIORF_OUTPUT_NAMESPACE="$mode"
export PRIORF_OUTPUT_DIR="$namespace_dir"

if [[ $# -gt 0 ]]; then
  "$@"
else
  printf '%s\n' "$namespace_dir"
fi
