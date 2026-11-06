#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-smartisp-t4}"
GPU="${2:-T4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
REPO_URL="${SMARTISP_REPO_URL:-${REPO_URL:-https://github.com/thushar-chandra/SmartISP.git}}"

cd "$REPO_ROOT"
mkdir -p outputs

if ! command -v colab >/dev/null 2>&1; then
    echo "error: colab CLI is not on PATH" >&2
    echo "install it with: uv tool install google-colab-cli" >&2
    exit 1
fi

colab new -s "$SESSION" --gpu "$GPU"
trap 'colab stop -s "$SESSION" >/dev/null 2>&1 || true' EXIT

colab install -s "$SESSION" -r requirements/colab.txt
colab exec -s "$SESSION" --env "SMARTISP_REPO_URL=$REPO_URL" -f scripts/colab_smoke_test.py
colab log -s "$SESSION" -o outputs/colab-smoke.md
colab stop -s "$SESSION"
trap - EXIT
