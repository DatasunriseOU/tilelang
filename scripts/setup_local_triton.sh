#!/usr/bin/env bash
# Install Triton (with Apple backend) from the locally-built source at
# /Volumes/external/sources/triton-pr9701 into .venv313 so we can capture
# TTIR from @triton.jit kernels on this Mac.
#
# Companion to scripts/setup_local_tvm_ffi.sh. Run AFTER tvm_ffi is locked.
#
# Constraints honored:
#   * .venv313/etc/pip/constraints.txt is passed via --constraint to pip,
#     so apache-tvm-ffi==0.1.11 cannot be replaced.
#   * apache-tvm-ffi, tvm, tilelang installs are NOT touched.
#
# Strategy:
#   1. Try editable install of the local Triton source (gives us Apple backend).
#   2. On editable failure, fall back to PyPI `triton` (CPython 3.13 wheel
#      may not exist on macOS — that case is reported, not silently skipped).
#   3. After install, append the resolved triton version to constraints.txt
#      (idempotent: replaces any existing triton== line).
#
# Idempotent: re-running is safe.

set -euo pipefail

REPO=/private/tmp/tl_apache_tvm_swap
VENV=$REPO/.venv313
TRITON_SRC=/Volumes/external/sources/triton-pr9701
CONSTRAINTS=$VENV/etc/pip/constraints.txt
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
# Triton LLVM tarball that the Apple backend was tested against
# (per third_party/apple/SETUP.md). The hash matches the cmake cache that
# was used when build-direct/ was last configured.
LLVM_SYSPATH=${LLVM_SYSPATH:-$HOME/.triton/llvm/llvm-27d654c4-macos-arm64}
# Reuse a dedicated cmake build dir so editable re-installs don't recompile.
TRITON_BUILD_DIR=${TRITON_BUILD_DIR:-$TRITON_SRC/build/cmake.macosx-cpython-3.13}

if [ ! -x "$PIP" ]; then
    echo "ERROR: pip not found at $PIP. Create the venv first." >&2
    exit 1
fi

if [ ! -f "$CONSTRAINTS" ]; then
    echo "ERROR: constraints lock missing at $CONSTRAINTS." >&2
    echo "Run scripts/setup_local_tvm_ffi.sh first." >&2
    exit 1
fi

echo "[1/4] Checking current triton state..."
if "$PY" -c "import triton, sys; sys.stdout.write(triton.__version__+'\n')" 2>/dev/null; then
    echo "      Triton already importable: $($PY -c 'import triton;print(triton.__version__)')"
fi

echo "[2/4] Attempting editable install from $TRITON_SRC ..."
INSTALL_PATH=""
TRITON_VERSION=""
EDITABLE_LOG=$(mktemp)
if [ -f "$TRITON_SRC/setup.py" ] || [ -f "$TRITON_SRC/pyproject.toml" ]; then
    # NOTE: per third_party/apple/SETUP.md the Apple-backend build needs:
    #   * LLVM_SYSPATH pointing at the prebuilt Triton LLVM tarball
    #   * apple,nvidia,amd in codegen-backends (TritonGPUTransforms hard-
    #     includes nvidia NVWS dialect headers + the apple SETUP patch lists
    #     all three). No CUDA SDK is required; we only need tablegen headers.
    if [ ! -d "$LLVM_SYSPATH" ]; then
        echo "      WARNING: LLVM_SYSPATH=$LLVM_SYSPATH does not exist." >&2
        echo "      Apple backend build needs the Triton LLVM tarball." >&2
    fi
    if LLVM_SYSPATH="$LLVM_SYSPATH" \
        TRITON_BUILD_DIR="$TRITON_BUILD_DIR" \
        TRITON_CODEGEN_BACKENDS=apple,nvidia,amd \
        TRITON_BUILD_PROTON=OFF \
        "$PIP" install --constraint "$CONSTRAINTS" \
        --no-build-isolation -e "$TRITON_SRC" \
        > "$EDITABLE_LOG" 2>&1; then
        INSTALL_PATH="editable:$TRITON_SRC"
    else
        echo "      Editable install FAILED. Last 40 lines of stderr:" >&2
        tail -40 "$EDITABLE_LOG" >&2
        echo "      Full log: $EDITABLE_LOG" >&2
    fi
else
    echo "      No setup.py / pyproject.toml at $TRITON_SRC. Skipping editable." >&2
fi

if [ -z "$INSTALL_PATH" ]; then
    echo "[2b/4] Falling back to PyPI triton wheel..."
    PYPI_LOG=$(mktemp)
    if "$PIP" install --constraint "$CONSTRAINTS" triton > "$PYPI_LOG" 2>&1; then
        INSTALL_PATH="pypi"
    else
        echo "      PyPI install ALSO FAILED. stderr:" >&2
        cat "$PYPI_LOG" >&2
        exit 2
    fi
fi

echo "[3/4] Verifying triton import + co-existence with tvm_ffi/tvm/tilelang..."
TRITON_VERSION=$("$PY" -c "import triton;print(triton.__version__)")
echo "      triton version: $TRITON_VERSION"
echo "      install path:   $INSTALL_PATH"

# Sanity-check the four imports that MUST coexist.
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
PYTHONPATH=$REPO:$REPO/3rdparty/tvm/python \
"$PY" - <<'PYEOF'
import sys
errs = []
for mod in ("tvm_ffi", "tvm", "tilelang", "triton"):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {mod}: OK ({v})")
    except Exception as e:
        errs.append((mod, repr(e)))
        print(f"  {mod}: FAIL {e!r}")
if errs:
    sys.exit(f"Coexistence check failed: {errs}")
PYEOF

echo "[4/4] Pinning triton version in $CONSTRAINTS ..."
# Strip any prior triton== line and append the freshly-installed one.
TMPC=$(mktemp)
grep -v -E '^triton(==| @ )' "$CONSTRAINTS" > "$TMPC" || true
echo "triton==$TRITON_VERSION" >> "$TMPC"
mv "$TMPC" "$CONSTRAINTS"
echo "      Appended: triton==$TRITON_VERSION"

echo ""
echo "DONE. Triton $TRITON_VERSION installed via $INSTALL_PATH."
echo "      To capture TTIR run:"
echo "        DYLD_LIBRARY_PATH=/opt/homebrew/lib \\"
echo "        PYTHONPATH=$REPO:$REPO/3rdparty/tvm/python \\"
echo "        $PY <your_ttir_script>.py"
