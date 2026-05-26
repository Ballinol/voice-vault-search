#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== Voice Vault Search: setup ==="
echo "Plugin folder: $(pwd)"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Install Python 3.10+ first."
  exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[info] Python $PYVER"
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" || {
  echo "[ERROR] Python 3.10+ required. Got $PYVER"
  exit 1
}

if [ -d ".venv" ]; then
  echo "[info] .venv exists, reusing"
else
  echo "[step 1/3] Creating venv..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "[step 2/3] Upgrading pip..."
python -m pip install --upgrade pip --quiet

echo "[step 3/3] Installing deps (~2 GB, 5-10 min first time)..."
python -m pip install -r requirements.txt

echo
echo "=== Setup complete ==="
echo
echo "Reload Obsidian, then open the Voice Vault Search view via the ribbon icon."
echo
echo "For NVIDIA GPU acceleration (Linux only — macOS uses MPS or CPU):"
echo "  pip uninstall torch -y"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
