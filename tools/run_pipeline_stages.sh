#!/usr/bin/env bash
set -euo pipefail

# This helper script runs MyDailyNews in:
# 1) full mode, or
# 2) stage-by-stage debug mode (one run per stop checkpoint).
#
# It is intentionally verbose so you can see exactly which command is being run.
# `--debug` is forwarded for every run by default.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/config.json}"
BRIEF_MODE="${BRIEF_MODE:-general}"  # general | detailed | both
DEBUG_ENABLED="${DEBUG_ENABLED:-1}"  # 1 = pass --debug, 0 = disable

# Where stage checkpoint JSON artifacts will be written.
# You can override this per run:
#   STAGE_ARTIFACT_DIR=/tmp/mydailynews-stages ./tools/run_pipeline_stages.sh step headline_select
STAGE_ARTIFACT_DIR="${STAGE_ARTIFACT_DIR:-$PROJECT_ROOT/output/diagnostics/stages/manual}"
# Note: when --stop-after-stage is used, the Python CLI now enables intermediate
# payload saving by default, so each step run preserves full stage objects.

# Ordered list of supported stop checkpoints.
STAGES=(
  prior_reports
  snapshot
  shared_headline_scoring
  candidate_prepare
  headline_limit
  headline_decisions
  headline_select
  article_fetch
  enrichment
  evidence_distillation
  delta_extraction
  final_brief
  write_output
)

usage() {
  cat <<'EOF'
Usage:
  ./tools/run_pipeline_stages.sh full
  ./tools/run_pipeline_stages.sh step <stage_name>
  ./tools/run_pipeline_stages.sh all-steps

Examples:
  ./tools/run_pipeline_stages.sh full
  BRIEF_MODE=detailed ./tools/run_pipeline_stages.sh step article_fetch
  BRIEF_MODE=general DEBUG_ENABLED=1 ./tools/run_pipeline_stages.sh all-steps

Environment variables:
  PYTHON_BIN         Python executable (default: python)
  CONFIG_PATH        Config path (default: <repo>/config.json)
  BRIEF_MODE         general|detailed|both (default: general)
  DEBUG_ENABLED      1/0 (default: 1)
  STAGE_ARTIFACT_DIR Artifact output directory
EOF
}

has_stage() {
  local needle="$1"
  for stage in "${STAGES[@]}"; do
    if [[ "$stage" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

debug_flag=()
if [[ "$DEBUG_ENABLED" == "1" ]]; then
  debug_flag+=(--debug)
fi

run_full() {
  echo "==> Running full pipeline (brief=${BRIEF_MODE})"
  "$PYTHON_BIN" "$PROJECT_ROOT/main.py" \
    --config "$CONFIG_PATH" \
    --brief "$BRIEF_MODE" \
    "${debug_flag[@]}"
}

run_step() {
  local stage="$1"
  if ! has_stage "$stage"; then
    echo "Unsupported stage: $stage" >&2
    echo "Supported: ${STAGES[*]}" >&2
    exit 1
  fi

  echo "==> Running until stage: $stage (brief=${BRIEF_MODE})"
  "$PYTHON_BIN" "$PROJECT_ROOT/main.py" \
    --config "$CONFIG_PATH" \
    --brief "$BRIEF_MODE" \
    --stop-after-stage "$stage" \
    --dump-stage-artifacts \
    --stage-artifact-dir "$STAGE_ARTIFACT_DIR" \
    "${debug_flag[@]}"
}

run_all_steps() {
  echo "==> Running one debug pass per stage (brief=${BRIEF_MODE})"
  echo "    Artifacts: $STAGE_ARTIFACT_DIR"
  for stage in "${STAGES[@]}"; do
    run_step "$stage"
    echo ""
  done
}

main() {
  local mode="${1:-}"
  case "$mode" in
    full)
      run_full
      ;;
    step)
      if [[ $# -lt 2 ]]; then
        echo "Missing stage name for 'step' mode." >&2
        usage
        exit 1
      fi
      run_step "$2"
      ;;
    all-steps)
      run_all_steps
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
