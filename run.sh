#!/usr/bin/env bash
# Diacritize Common Voice Arabic validated clips, build an IPA lexicon, then
# train and run Montreal Forced Aligner.
#
# Not installed by this repository:
#   - the DiaCTC environment (torch, k2, transformers, librosa)
#   - mfa on PATH (the VoxCommunis notebook uses the conda env named aligner)
#   - MDC_API_KEY, a Mozilla Data Collective token, for the download stage
#
# Usage:
#   export MDC_API_KEY=...
#   ./run.sh
#   ./run.sh --from-stage diacritize
#   ./run.sh --stage select --max-hours 2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CV_DIR="${CV_DIR:-$ROOT/data/cv-corpus-27.0-2026-09-11/ar}"
export DIACTC_ROOT="${DIACTC_ROOT:-$ROOT/third_party/DiaCTC}"
export WORK_DIR="${WORK_DIR:-$ROOT/work/ar}"

PYTHON="${PYTHON:-python3}"
cd "$ROOT"
exec "$PYTHON" "$ROOT/run_cv_arabic.py" "$@"
