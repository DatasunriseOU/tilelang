#!/usr/bin/env bash
# Install the locally-built tvm_ffi (from 3rdparty/tvm/3rdparty/tvm-ffi/)
# into .venv313 and lock it against accidental pip reinstalls.
#
# Why: the PyPI `apache-tvm-ffi==0.1.7` wheel ships a stale ABI that lacks
# `MapGetMissingObject` — `import tvm` crashes at startup. The local source
# (apache-tvm-ffi 0.1.11) matches the libtvm_ffi.dylib we built in `build/`.
#
# This script is idempotent. Run it after a fresh venv create or after
# `pip uninstall apache-tvm-ffi`.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=$REPO/.venv313
SRC=$REPO/3rdparty/tvm/3rdparty/tvm-ffi

if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv313 not found at $VENV"
    echo "Create it first: python3.13 -m venv .venv313"
    exit 1
fi

if [ ! -f "$SRC/pyproject.toml" ]; then
    echo "ERROR: bundled tvm-ffi source not found at $SRC"
    echo "Did 3rdparty/tvm submodule fail to init?"
    exit 1
fi

echo "[1/3] Uninstalling any existing apache-tvm-ffi..."
"$VENV/bin/pip" uninstall -y apache-tvm-ffi 2>/dev/null || true

echo "[2/3] Installing local editable apache-tvm-ffi from $SRC..."
"$VENV/bin/pip" install -e "$SRC"

echo "[3/3] Locking against pip reinstalls..."
mkdir -p "$VENV/etc/pip"
cat > "$VENV/etc/pip/constraints.txt" << EOF
# CPPMEGA LOCK: apache-tvm-ffi must be the locally-built editable install.
# pip will refuse to install any other version.
# To break this lock: delete this file AND uninstall AND reinstall via
#   bash $0
apache-tvm-ffi==0.1.11
EOF

cat > "$VENV/pip.conf" << EOF
[global]
constraint = $VENV/etc/pip/constraints.txt
EOF

# Layer 3: import-time path guard
cat > "$SRC/python/tvm_ffi/_cppmega_lock.py" << 'PYEOF'
"""CPPMEGA local-build lock for tvm_ffi.

This file marks ``apache-tvm-ffi`` as a locally-built editable install. The
local build is symbol-matched against
``$REPO/build/lib/libtvm_ffi.dylib``; PyPI's apache-tvm-ffi==0.1.7 has
a stale ABI (missing ``MapGetMissingObject``) and crashes ``import tvm``.

If you see the breach warning below, run:
  bash $REPO/scripts/setup_local_tvm_ffi.sh
"""
import os, sys

_THIS = os.path.dirname(os.path.abspath(__file__))
# Marker check: editable install lives under 3rdparty/tvm/3rdparty/tvm-ffi/
if "tvm-ffi/python/tvm_ffi" not in os.path.realpath(_THIS).replace(os.sep, "/"):
    sys.stderr.write(
        "\n*** CPPMEGA LOCK BREACH ***\n"
        f"tvm_ffi imported from: {_THIS}\n"
        "Expected: <repo>/3rdparty/tvm/3rdparty/tvm-ffi/python/tvm_ffi\n"
        "Someone replaced the editable install with a PyPI wheel.\n"
        "Reinstall: bash <repo>/scripts/setup_local_tvm_ffi.sh\n"
        "*** end ***\n\n"
    )
PYEOF

# Inject the guard into __init__.py if not already present
INIT="$SRC/python/tvm_ffi/__init__.py"
if ! grep -q "_cppmega_lock" "$INIT"; then
    python3 - << PYEOF
init_path = "$INIT"
with open(init_path) as f:
    src = f.read()
lines = src.split("\n")
inject_at = 0
in_docstring = False
for i, ln in enumerate(lines):
    s = ln.strip()
    if s.startswith('"""') and s.count('"""') == 1:
        in_docstring = not in_docstring
        continue
    if in_docstring or s.startswith("#") or not s:
        continue
    inject_at = i
    break
lines.insert(inject_at, "from . import _cppmega_lock as _  # noqa: F401  CPPMEGA local-build lock")
with open(init_path, "w") as f:
    f.write("\n".join(lines))
print(f"injected lock at {init_path}:{inject_at}")
PYEOF
fi

echo
echo "Done. Verifying:"
PYTHONPATH=$REPO/3rdparty/tvm/python:. \
  DYLD_LIBRARY_PATH=$REPO/build/lib:/opt/homebrew/lib \
  "$VENV/bin/python" -c "
import tvm_ffi; print(f'tvm_ffi: {tvm_ffi.__file__} v{tvm_ffi.__version__}')
import tvm; print(f'tvm: {tvm.__version__}')
" 2>&1 | grep -vE "WARNING|UserWarning|Field|policy|register_object" | tail -3
