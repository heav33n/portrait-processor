#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and run this script again."
  exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "Install Rust from https://rustup.rs/ and run this script again."
  exit 1
fi
uv sync --locked --dev
.venv/bin/python tools/install_launcher.py
echo "Ready. Launch with ./run.sh (AI models download on first use)."

