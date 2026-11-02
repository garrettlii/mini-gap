#!/usr/bin/env bash
# Phase 0 setup for Apple Silicon macOS. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

say "Checking platform"
[[ "$(uname -s)" == "Darwin" ]] || { echo "This script targets macOS."; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || echo "Warning: not Apple Silicon; MLX steps will be skipped."
echo "RAM: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"

say "Xcode command line tools"
xcode-select -p >/dev/null 2>&1 || { xcode-select --install; echo "Re-run this script after the installer finishes."; exit 0; }

say "Homebrew packages (uv, git, ffmpeg)"
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
for pkg in uv git ffmpeg; do brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"; done

say "Python environment"
uv python install 3.11
uv sync --all-extras

say "MuJoCo Menagerie Panda"
MEN="${MENAGERIE_DIR:-$HOME/menagerie}"
[[ -d "$MEN/.git" ]] || git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$MEN"
# Copy without overwriting our workcell.xml.
rsync -a --ignore-existing "$MEN/franka_emika_panda/" assets/franka_emika_panda/

say ".env"
[[ -f .env ]] || { cp .env.example .env; echo "Created .env: add your API key."; }

say "Git"
[[ -d .git ]] || { git init -q && git add -A && git commit -qm "chore: initial mini-gap scaffold"; echo "Initialized repo."; }

say "Sanity checks"
uv run python scripts/sanity_check.py
uv run pytest -q
echo
echo "Next: uv run mjpython scripts/view_workcell.py   (interactive viewer)"
