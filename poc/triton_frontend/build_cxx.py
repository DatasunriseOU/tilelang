"""Locate (and optionally build) the ``_triton_frontend_cxx`` extension.

The C++ shim under ``poc/triton_frontend/_cxx/`` is *not* installed as part of
the regular Python package. It is expected to be built out-of-tree (cmake +
ninja) and the resulting ``.so``/``.dylib`` to be made importable via
``sys.path``. This module is the one place that knows how to do that:

- :func:`ensure_built` is a process-cached helper that
  (a) first tries plain ``import _triton_frontend_cxx``;
  (b) failing that, looks for an already-built extension under
      ``_cxx/build/`` and adds it to ``sys.path`` so the next ``import`` works;
  (c) optionally invokes cmake + ninja to produce the extension when called
      with ``build=True``.

Run as a CLI:

    python -m poc.triton_frontend.build_cxx --check
    python -m poc.triton_frontend.build_cxx --build

``--check`` exits 0 if importable, 1 otherwise (no side effects). ``--build``
runs the cmake + ninja flow with environment auto-detection (brew LLVM/MLIR
on macOS, system ``/usr/lib/llvm-<N>`` on Linux) and then re-checks.

The build always falls into the ``TRITON_FRONTEND_STUB_BUILD`` mode unless
the caller has set ``TRITON_INSTALL_DIR`` (an upstream OpenAI Triton install
prefix). Stub mode produces a *loadable* extension whose ``run_rewrite``
returns the documented internal-error code; that is the state the Python
facade in :mod:`poc.triton_frontend.ptr_analysis` falls back from to the
MVP scalar lowering path.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

SHIM_MODULE_NAME = "_triton_frontend_cxx"

_THIS_DIR = Path(__file__).resolve().parent
_CXX_DIR = _THIS_DIR / "_cxx"
_BUILD_DIR = _CXX_DIR / "build"

# Where we stage a "fake" Triton install with real native archives harvested
# from a local Triton source build. The CMakeLists.txt reads
# `TRITON_INSTALL_DIR` and links against `lib/lib{TritonIR,TritonAnalysis,
# TritonGPUIR}.a` there. The vendored fallback under `_cxx/../vendored/triton`
# only contains stub archives without a real architecture, so on a host with
# Triton compiled in-tree we synthesize valid Mach-O / ELF archives here and
# point CMake at this staging dir instead.
_TRITON_STAGE_DIR = _BUILD_DIR / "_triton_stage"

# Process-level latch so repeated ensure_built() calls don't keep poking
# sys.path or shelling out to cmake. The value caches the last result.
_ENSURE_RESULT: Optional[bool] = None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _candidate_extensions() -> Iterable[Path]:
    """Yield possible filenames the shim might have been built under.

    Both the standard CPython suffix (e.g. ``.cpython-3xx-darwin.so``) and the
    plain ``.so``/``.dylib`` fallbacks are accepted -- ``setup.py``-style
    installs vs. a hand-rolled cmake build place them slightly differently.
    """
    if not _BUILD_DIR.exists():
        return
    for entry in _BUILD_DIR.iterdir():
        name = entry.name
        if not name.startswith(SHIM_MODULE_NAME):
            continue
        if name.endswith(".so") or name.endswith(".dylib") or name.endswith(".pyd"):
            yield entry


def _add_build_dir_to_syspath() -> bool:
    """Prepend the cmake build dir to ``sys.path`` if it contains the shim."""
    if any(_candidate_extensions()):
        path_str = str(_BUILD_DIR)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        # Invalidate any cached negative lookup of the module name.
        importlib.invalidate_caches()
        return True
    return False


# ---------------------------------------------------------------------------
# Environment auto-detection
# ---------------------------------------------------------------------------


def _detect_mlir_dirs() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(MLIR_DIR, LLVM_DIR)`` for cmake's ``find_package`` lookup.

    Honours pre-set environment variables first, then probes:
      - macOS:   ``brew --prefix llvm`` -> ``$prefix/lib/cmake/{mlir,llvm}``
      - Linux:   ``/usr/lib/llvm-<N>/lib/cmake/{mlir,llvm}`` for N in 18..14
                 (preferring the largest version that has both directories).

    Returns ``(None, None)`` if nothing is found; the caller then surfaces a
    clear error from cmake itself rather than silently guessing.
    """
    mlir = os.environ.get("MLIR_DIR")
    llvm = os.environ.get("LLVM_DIR")
    if mlir and llvm:
        return mlir, llvm

    if sys.platform == "darwin":
        brew = shutil.which("brew")
        if brew:
            try:
                prefix = subprocess.check_output(
                    [brew, "--prefix", "llvm"], text=True
                ).strip()
            except subprocess.CalledProcessError:
                prefix = ""
            if prefix:
                mlir = mlir or f"{prefix}/lib/cmake/mlir"
                llvm = llvm or f"{prefix}/lib/cmake/llvm"
                return mlir, llvm

    if sys.platform.startswith("linux"):
        # Iterate newest to oldest. GB10 dev images today ship LLVM 18; older
        # CI runners may only have 16/15. Stop at the first prefix where both
        # `mlir` and `llvm` cmake dirs exist.
        for ver in (20, 19, 18, 17, 16, 15, 14):
            base = Path(f"/usr/lib/llvm-{ver}/lib/cmake")
            mlir_dir = base / "mlir"
            llvm_dir = base / "llvm"
            if mlir_dir.is_dir() and llvm_dir.is_dir():
                return str(mlir_dir), str(llvm_dir)
        # Some distros put the cmake helpers directly under
        # /usr/lib/cmake/{mlir,llvm} (e.g. `apt install libmlir-dev`).
        mlir_dir = Path("/usr/lib/cmake/mlir")
        llvm_dir = Path("/usr/lib/cmake/llvm")
        if mlir_dir.is_dir() and llvm_dir.is_dir():
            return str(mlir_dir), str(llvm_dir)

    return mlir, llvm


# ---------------------------------------------------------------------------
# Build driver
# ---------------------------------------------------------------------------


def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Wrapper around ``subprocess.run`` that always streams output."""
    print(f"[build_cxx] $ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=False, **kwargs)


def _detect_triton_source_build() -> Optional[Path]:
    """Locate a Triton source-build dir whose objs we can re-archive.

    Returns the path to a directory that looks like
    ``<triton_src>/build/cmake.<plat>-cpython-<ver>/`` if one is found,
    else None. The directory is expected to contain
    ``lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/*.o``.

    The probe order is:
      1. ``$TRITON_SRC_BUILD_DIR`` env var (explicit override)
      2. The installed ``triton`` package's ``__file__`` -> walk up to
         find a sibling ``build/cmake.*`` directory.
    """
    explicit = os.environ.get("TRITON_SRC_BUILD_DIR")
    if explicit:
        p = Path(explicit)
        if (p / "lib" / "Dialect" / "Triton" / "IR" / "CMakeFiles" /
                "TritonIR.dir").is_dir():
            return p
    try:
        import triton  # type: ignore
    except Exception:
        return None
    tri_file = Path(getattr(triton, "__file__", "") or "")
    if not tri_file.exists():
        return None
    # triton/__init__.py -> triton/ -> python/ -> <src>/
    src_root = tri_file.parent.parent.parent
    build_root = src_root / "build"
    if not build_root.is_dir():
        return None
    # Newest cmake.*-cpython-* dir wins.
    candidates = sorted(
        (p for p in build_root.iterdir()
         if p.is_dir() and p.name.startswith("cmake.")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        if (cand / "lib" / "Dialect" / "Triton" / "IR" / "CMakeFiles" /
                "TritonIR.dir").is_dir():
            return cand
    return None


def _stage_triton_install(verbose: bool = True) -> Optional[Path]:
    """Materialize a synthetic Triton install dir for the cmake link step.

    The vendored ``_cxx/../vendored/triton/lib/*.a`` archives committed to
    the repo are stubs without a real architecture (they parse as ``current
    ar archive`` but contain only a symbol table and no member objects),
    so the macOS / Linux linker refuses them with "invalid control bits".
    This helper finds a local Triton source build (see
    :func:`_detect_triton_source_build`) and uses ``libtool`` / ``ar`` to
    re-archive its ``.o`` files into ``_TRITON_STAGE_DIR/lib/``. The
    ``include/`` symlink reuses the (header-only) vendored tree.

    Returns the staging dir on success (suitable for
    ``-DTRITON_INSTALL_DIR=...``), or None if we couldn't find a usable
    Triton source build -- in which case the cmake step will fall back to
    the vendored stub path and fail at link time with a clear message.
    """
    build = _detect_triton_source_build()
    if build is None:
        if verbose:
            print(
                "[build_cxx] no Triton source build found; cmake will fall "
                "back to the vendored stub archives (link will likely fail). "
                "Set TRITON_SRC_BUILD_DIR=<path>/build/cmake.<plat>-cpython-<v>",
                file=sys.stderr,
            )
        return None

    # Map archive name -> directory of .o files relative to the triton build.
    components = {
        "libTritonIR": build / "lib" / "Dialect" / "Triton" / "IR" /
                       "CMakeFiles" / "TritonIR.dir",
        "libTritonAnalysis": build / "lib" / "Analysis" / "CMakeFiles" /
                             "TritonAnalysis.dir",
        "libTritonGPUIR": build / "lib" / "Dialect" / "TritonGPU" / "IR" /
                          "CMakeFiles" / "TritonGPUIR.dir",
    }

    stage_lib = _TRITON_STAGE_DIR / "lib"
    stage_lib.mkdir(parents=True, exist_ok=True)

    # Pick the archive tool. macOS prefers `libtool -static`; on Linux we
    # fall back to `ar rcs`. Both produce the standard `ar`-format archives
    # that `find_library` + the linker accept.
    use_libtool = sys.platform == "darwin" and shutil.which("libtool") is not None
    ar_tool = shutil.which("ar")
    if not use_libtool and not ar_tool:
        if verbose:
            print(
                "[build_cxx] neither libtool nor ar found; cannot stage Triton "
                "archives. Install Xcode CLT or binutils.",
                file=sys.stderr,
            )
        return None

    for name, obj_dir in components.items():
        out = stage_lib / f"{name}.a"
        # Cache: skip if archive newer than every .o file.
        objs = sorted(obj_dir.glob("*.o"))
        if not objs:
            if verbose:
                print(f"[build_cxx] no .o files in {obj_dir}; skipping {name}",
                      file=sys.stderr)
            continue
        if out.exists():
            out_mtime = out.stat().st_mtime
            if all(o.stat().st_mtime <= out_mtime for o in objs):
                continue
            out.unlink()
        if use_libtool:
            cmd = ["libtool", "-static", "-o", str(out)] + [str(o) for o in objs]
        else:
            cmd = [ar_tool, "rcs", str(out)] + [str(o) for o in objs]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            if verbose:
                print(
                    f"[build_cxx] failed to archive {name}: {res.stderr.strip()}",
                    file=sys.stderr,
                )
            return None

    # Header tree: symlink to the vendored copy (header-only, OK to share).
    stage_inc = _TRITON_STAGE_DIR / "include"
    vendored_inc = _CXX_DIR.parent / "vendored" / "triton" / "include"
    if not stage_inc.exists() and vendored_inc.is_dir():
        try:
            stage_inc.symlink_to(vendored_inc)
        except OSError:
            # Fall back to copying if symlinks aren't allowed.
            shutil.copytree(vendored_inc, stage_inc)

    if verbose:
        print(
            f"[build_cxx] staged Triton install at {_TRITON_STAGE_DIR} "
            f"(from {build})",
            file=sys.stderr,
        )
    return _TRITON_STAGE_DIR


def _build(verbose: bool = True) -> Tuple[bool, str]:
    """Run cmake + ninja. Returns ``(ok, stderr_summary)``.

    Stderr summary is the last ~30 lines of output on failure (empty on
    success). The function never raises -- callers branch on ``ok``.
    """
    if not _CXX_DIR.is_dir():
        return False, f"cmake source dir missing: {_CXX_DIR}"

    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    if not cmake:
        return False, "cmake not found on PATH"
    if not ninja:
        return False, "ninja not found on PATH (`brew install ninja` or `apt install ninja-build`)"

    mlir_dir, llvm_dir = _detect_mlir_dirs()
    if not mlir_dir or not llvm_dir:
        return False, (
            "could not auto-detect MLIR_DIR / LLVM_DIR. Set them explicitly:\n"
            "  export MLIR_DIR=/path/to/llvm/lib/cmake/mlir\n"
            "  export LLVM_DIR=/path/to/llvm/lib/cmake/llvm"
        )

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        cmake, "-S", str(_CXX_DIR), "-B", str(_BUILD_DIR), "-GNinja",
        f"-DMLIR_DIR={mlir_dir}",
        f"-DLLVM_DIR={llvm_dir}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    triton_install = os.environ.get("TRITON_INSTALL_DIR")
    if not triton_install:
        # Try to stage a real Triton install from a local source build before
        # cmake configures (so the find_library(TRITON_IR_LIB REQUIRED) step
        # succeeds against valid archives instead of the vendored stubs).
        staged = _stage_triton_install(verbose=verbose)
        if staged is not None:
            triton_install = str(staged)
    if triton_install:
        configure_cmd.append(f"-DTRITON_INSTALL_DIR={triton_install}")
    if os.environ.get("TRITON_FRONTEND_USE_NLOHMANN_JSON"):
        configure_cmd.append("-DTRITON_FRONTEND_USE_NLOHMANN_JSON=ON")
    if os.environ.get("TRITON_FRONTEND_STUB_BUILD"):
        configure_cmd.append("-DTRITON_FRONTEND_STUB_BUILD=ON")

    capture = not verbose
    res = _run(
        configure_cmd,
        capture_output=capture,
        text=True,
    )
    if res.returncode != 0:
        tail = (res.stderr or "")[-2000:] if capture else "(see stderr above)"
        return False, f"cmake configure failed (rc={res.returncode}):\n{tail}"

    res = _run(
        [ninja, "-C", str(_BUILD_DIR)],
        capture_output=capture,
        text=True,
    )
    if res.returncode != 0:
        tail = (res.stderr or "")[-2000:] if capture else "(see stderr above)"
        return False, f"ninja build failed (rc={res.returncode}):\n{tail}"

    if not _add_build_dir_to_syspath():
        return False, (
            f"build reported success but no {SHIM_MODULE_NAME}*.so was "
            f"produced under {_BUILD_DIR}"
        )
    return True, ""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def shim_importable() -> bool:
    """Return True iff ``_triton_frontend_cxx`` is importable right now."""
    return importlib.util.find_spec(SHIM_MODULE_NAME) is not None


def ensure_built(build: bool = False, verbose: bool = True) -> bool:
    """Make the shim importable; optionally build it first.

    Steps:
      1. Already in ``sys.modules`` / on ``sys.path``? -> True.
      2. Pre-built ``.so`` in ``_cxx/build/``?         -> add to sys.path, True.
      3. ``build=True``?                                -> run cmake + ninja,
                                                           then retry step 2.
      4. Otherwise                                      -> False (no exception).

    The result is cached for the lifetime of the process so repeated calls
    after a successful build don't re-stat the build dir.
    """
    global _ENSURE_RESULT
    if _ENSURE_RESULT is True:
        return True

    if shim_importable():
        _ENSURE_RESULT = True
        return True

    if _add_build_dir_to_syspath() and shim_importable():
        _ENSURE_RESULT = True
        return True

    if not build:
        return False

    ok, err = _build(verbose=verbose)
    if not ok:
        if verbose:
            print(f"[build_cxx] BUILD FAILED: {err}", file=sys.stderr)
        return False

    if shim_importable():
        _ENSURE_RESULT = True
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m poc.triton_frontend.build_cxx",
        description=(
            "Locate or build the _triton_frontend_cxx pybind11 extension. "
            "Used by poc.triton_frontend.ptr_analysis to enable multi-element "
            "tile loads via the vendored microsoft/triton-shared PtrAnalysis."
        ),
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--check", action="store_true",
        help="Exit 0 if the shim is importable, 1 otherwise. No side effects.",
    )
    grp.add_argument(
        "--build", action="store_true",
        help="Run cmake + ninja to produce the shim (auto-detects MLIR_DIR/LLVM_DIR).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress cmake/ninja output (capture into the failure summary).",
    )
    args = parser.parse_args(argv)

    if args.check:
        # Probe both the regular import path and the cmake build dir, but do
        # not invoke cmake. This is what CI / test fixtures want.
        ok = ensure_built(build=False, verbose=not args.quiet)
        if ok:
            print(f"[build_cxx] OK: {SHIM_MODULE_NAME} is importable")
            return 0
        print(
            f"[build_cxx] MISSING: {SHIM_MODULE_NAME} not importable. "
            f"Run `python -m poc.triton_frontend.build_cxx --build` to produce it.",
            file=sys.stderr,
        )
        return 1

    # --build
    ok = ensure_built(build=True, verbose=not args.quiet)
    if ok:
        print(f"[build_cxx] OK: {SHIM_MODULE_NAME} built and importable")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
