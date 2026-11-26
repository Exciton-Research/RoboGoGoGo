#!/usr/bin/env bash
# Install from a checkout without changing system Python or global npm packages.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [--dev] [--headless]

  --dev       Include pytest, Ruff and API test dependencies.
  --headless  Install the Python engine only; skip Node.js and the Studio build.
  --help      Show this help.

Requires Python 3.11+; Studio also needs Node.js 22.6+ and npm.
Set PYTHON to choose an interpreter and ROBOGOGOGO_VENV to choose the virtualenv.
EOF
}

fail() { echo "Setup: $*" >&2; exit 1; }
dev=false
headless=false
for argument in "$@"; do
  case "$argument" in
    --dev) dev=true ;;
    --headless) headless=true ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; fail "Unknown option: $argument" ;;
  esac
done

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON:-python3}"
venv_dir="${ROBOGOGOGO_VENV:-$repo_dir/.venv}"
command -v "$python_bin" >/dev/null || fail "Install Python 3.11+; see docs/installation.md."
"$python_bin" - <<'PY'
import platform
import sys

if sys.version_info < (3, 11):
    sys.exit("Setup: Python 3.11+ is required. Set PYTHON=python3.12 to select it.")
system, machine = platform.system(), platform.machine().lower()
if (system, machine) not in {
    ("Darwin", "arm64"), ("Linux", "x86_64"), ("Linux", "aarch64"), ("Linux", "arm64")
}:
    sys.exit(
        f"Setup: no standard JAX CPU wheel path for {system}/{machine}. "
        "See docs/installation.md for platform limitations."
    )
print(f"Python {platform.python_version()} · {system}/{machine}")
PY

if ! "$headless"; then
  command -v node >/dev/null || fail "Install Node.js 22.6+; see docs/installation.md."
  command -v npm >/dev/null || fail "npm must be installed alongside Node.js."
  node -e 'const [major, minor] = process.versions.node.split(".").map(Number);
    if (major < 22 || (major === 22 && minor < 6)) {
      console.error("Setup: Node.js 22.6+ is required; use an LTS release."); process.exit(1);
    }'
fi

if [[ ! -e "$venv_dir" ]]; then
  "$python_bin" -m venv "$venv_dir" || fail "Cannot create a virtualenv. On Ubuntu, install python3-venv."
fi
[[ -x "$venv_dir/bin/python" ]] || fail "$venv_dir exists but is not a Python virtualenv. Choose ROBOGOGOGO_VENV."
"$venv_dir/bin/python" -m pip install --upgrade pip
extras=""
if ! "$headless"; then extras="studio,perception"; fi
if "$dev"; then extras="${extras:+$extras,}dev"; fi
requirement="."
if [[ -n "$extras" ]]; then requirement=".[$extras]"; fi
"$venv_dir/bin/python" -m pip install -e "$requirement"

if ! "$headless"; then
  npm ci --prefix studio
  npm run build --prefix studio
fi

echo "Checking one guarded simulation step (first JAX compilation can take a while)…"
"$venv_dir/bin/python" -m robogogogo --headless --steps 1
if "$headless"; then
  echo "Installed. Run: $venv_dir/bin/python -m robogogogo --headless --steps 100"
else
  echo "Installed. Run: $venv_dir/bin/robogogogo-studio"
  echo "Then open http://127.0.0.1:8765. See docs/getting-started.md."
fi
