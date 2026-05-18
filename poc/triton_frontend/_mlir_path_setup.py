"""Probe the host for an MLIR Python binding and wire it as ``mlir.ir``.

Imported (eagerly, with no side-effects on failure) by the package
``__init__.py`` *before* :mod:`mlir_walker`. The walker's
``try_import_mlir`` already honours ``MLIR_PYTHON_PACKAGE_PREFIX`` and
``LLVM_INSTALL_DIR``; this module adds three additional auto-probes
in priority order so a freshly-cloned checkout on macOS / Linux
finds the bindings without any environment pre-step:

1. **Triton's vendored LLVM** at ``~/.triton/llvm/llvm-*/python_packages/mlir_core``
   -- exact LLVM revision the C++ shim is linked against; preferred when
   present because there is zero ABI drift versus
   ``_triton_frontend_cxx.so``. As of LLVM 18+ Triton's tarball does not
   ship Python bindings, but the probe is cheap (one ``isdir`` check).

2. **Brew LLVM** at ``$(brew --prefix llvm)/python_packages/mlir_core``.
   On macOS Homebrew the bindings are not built by default; this branch
   is here for the day they are.

3. **IREE compiler** -- ``iree-base-compiler`` ships a fully-built
   ``mlir.ir``-compatible API under ``iree.compiler.ir`` plus the native
   ``_mlir.so`` C-extension. We register an ``mlir`` package alias in
   ``sys.modules`` so downstream ``from mlir import ir`` succeeds. This
   is the path that works on this Mac (Python 3.12 venv at
   ``/tmp/triton-build-venv12``); install via::

       python3.12 -m venv /tmp/triton-build-venv12
       /tmp/triton-build-venv12/bin/pip install 'iree-base-compiler==3.10.0'

   The 3.11 release of ``iree-base-compiler`` has a known
   ``_site_initialize`` bug (``ir._Context = ir.Context`` raises
   ``AttributeError`` because the C++ extension renamed ``Context`` to
   ``_BaseContext``); 3.10 is the highest version that imports cleanly.
   See README.md "Python MLIR bindings" for full context.

Hard constraints (per task spec):

- Don't silently downgrade. If every probe misses we leave ``sys.path``
  untouched so the existing one-shot UserWarning in ``mlir_walker``
  still fires.
- Don't break existing imports. ``from poc.triton_frontend import
  mlir_walker`` must work with or without ``mlir.ir``.
- Don't pip-install anything at runtime. We only adjust ``sys.path``
  and ``sys.modules`` based on what's already on disk.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import List, Optional

__all__ = [
    "probe_and_wire_mlir",
    "bootstrap_jaxlib_alias",
    "SELECTED_PATH",
    "SELECTED_SOURCE",
]


# Set after a successful probe so README / debug tools can show which
# branch fired. ``None`` means every probe missed and the fallback
# warning will fire when ``mlir_walker`` calls ``try_import_mlir``.
SELECTED_PATH: Optional[str] = None
SELECTED_SOURCE: Optional[str] = None  # one of {"triton", "brew", "iree", "env"}


def _candidate_triton_paths() -> List[str]:
    """Triton vendored LLVM tarballs land under ~/.triton/llvm/llvm-<rev>-<plat>."""
    home = os.path.expanduser("~")
    pattern = os.path.join(home, ".triton", "llvm", "llvm-*-*", "python_packages", "mlir_core")
    return sorted(glob.glob(pattern))


def _candidate_brew_paths() -> List[str]:
    """Homebrew LLVM ships at /opt/homebrew/opt/llvm (arm64) or /usr/local/opt/llvm (x86)."""
    return [
        "/opt/homebrew/opt/llvm/python_packages/mlir_core",
        "/usr/local/opt/llvm/python_packages/mlir_core",
    ]


def _try_iree_alias() -> Optional[str]:
    """Register ``iree.compiler.ir`` as ``mlir.ir`` if importable.

    IREE bundles a near-stock LLVM MLIR Python API (modulo the
    ``Context`` constructor wrapper for dialect registration). We:

    1. Try ``import iree.compiler.ir`` (this triggers the IREE site-init
       which registers all IREE-relevant dialects on the context).
    2. On success, build an ``mlir`` package object whose ``ir`` /
       ``passmanager`` / ``dialects`` submodules forward to the IREE
       counterparts and stash it in ``sys.modules``. From the caller's
       point of view ``import mlir.ir`` is now indistinguishable from a
       direct ``mlir_core`` install.

    Returns the IREE site-packages path on success, ``None`` on failure.
    """
    try:
        import iree.compiler as _iree_compiler  # type: ignore  # noqa: WPS433
        from iree.compiler import ir as _ir  # type: ignore  # noqa: WPS433
    except Exception:
        return None
    # Build a synthetic ``mlir`` package that delegates to ``iree.compiler``.
    # We only need ``mlir.ir`` for our walker; ``passmanager`` / ``dialects``
    # are forwarded for completeness so future emitters that touch them
    # don't need to know the alias trick.
    import types
    mlir_pkg = types.ModuleType("mlir")
    mlir_pkg.__path__ = []  # mark as package so ``from mlir import ir`` works
    mlir_pkg.ir = _ir
    sys.modules.setdefault("mlir", mlir_pkg)
    sys.modules.setdefault("mlir.ir", _ir)
    try:
        from iree.compiler import passmanager as _pm  # type: ignore  # noqa: WPS433
        mlir_pkg.passmanager = _pm
        sys.modules.setdefault("mlir.passmanager", _pm)
    except Exception:
        pass
    try:
        from iree.compiler import dialects as _dialects  # type: ignore  # noqa: WPS433
        mlir_pkg.dialects = _dialects
        sys.modules.setdefault("mlir.dialects", _dialects)
    except Exception:
        pass
    return os.path.dirname(_iree_compiler.__file__)


def _set_env_prefix(path: str) -> None:
    """Mirror the discovered prefix into ``MLIR_PYTHON_PACKAGE_PREFIX``.

    ``mlir_walker._augment_sys_path_from_env`` already reads this var,
    and external tooling (mlir-opt, downstream scripts) may want it too.
    Don't clobber a user-set value -- they're explicitly overriding us.
    """
    if not os.environ.get("MLIR_PYTHON_PACKAGE_PREFIX"):
        os.environ["MLIR_PYTHON_PACKAGE_PREFIX"] = path


def bootstrap_jaxlib_alias() -> bool:
    """Alias ``jaxlib.mlir`` into ``sys.modules['mlir']`` so ``import mlir.ir`` works.

    This is the CHEAPEST way to get mlir Python bindings on a Mac without
    building LLVM Python packages from source (jaxlib bundles them).
    Falls back gracefully if jaxlib isn't installed; returns ``True`` iff
    the alias was created (or was already in place).

    Idempotency: if ``mlir`` / ``mlir.ir`` are already on ``sys.modules``
    (whether placed there by an earlier call to this function, by an IREE
    alias from :func:`probe_and_wire_mlir`, or by a real ``mlir_core``
    import), we return ``True`` without re-aliasing.

    NB: we deliberately do NOT prepend ``<jaxlib_path>`` to ``sys.path``.
    jaxlib's wheel ships a ``triton`` sub-package which would shadow the
    real Triton install (causing a circular-import failure as
    ``triton.backends`` is partially initialised). Aliasing via
    ``sys.modules`` sidesteps that shadowing.
    """
    # Already aliased / installed? No-op.
    if "mlir.ir" in sys.modules:
        return True
    try:
        # Cheap probe: does ``mlir.ir`` import via the normal mechanism
        # (real mlir_core install, IREE alias, etc.)?
        import importlib

        importlib.import_module("mlir.ir")
        return True
    except Exception:
        pass

    try:
        import importlib

        jaxlib_mlir = importlib.import_module("jaxlib.mlir")
        jaxlib_mlir_ir = importlib.import_module("jaxlib.mlir.ir")
    except Exception:
        return False

    sys.modules.setdefault("mlir", jaxlib_mlir)
    sys.modules.setdefault("mlir.ir", jaxlib_mlir_ir)

    # Forward optional sub-modules if jaxlib exposes them (best-effort).
    # We deliberately do NOT pre-load dialect C-extensions
    # (``jaxlib.mlir.dialects.func`` etc.) here: each generated dialect
    # module is a distinct shared object that links its own LLVM, and on
    # macOS LLVM's CommandLine option registry is process-global, so
    # pre-loading the dialect extensions can race with later
    # ``triton._C`` / C++ shim loads and surface as
    # ``LLVM ERROR: Option 'basic' already exists!``. The walker only
    # needs ``mlir.ir`` to parse TTIR + dispatch via OP_TABLE; dialect-
    # specific introspection (test_op_emitters_memory) is handled lazily
    # via ``pytest.importorskip("jaxlib.mlir.ir")`` in those tests.
    for sub in ("passmanager", "dialects"):
        try:
            import importlib

            mod = importlib.import_module(f"jaxlib.mlir.{sub}")
            sys.modules.setdefault(f"mlir.{sub}", mod)
        except Exception:
            pass

    return True


def probe_and_wire_mlir() -> Optional[str]:
    """Run the probe order and update ``sys.path`` / ``sys.modules``.

    Returns the path of the chosen bindings (string) or ``None`` if
    every probe missed. The caller is the package ``__init__`` and it
    treats both outcomes as success -- the walker's own one-shot
    UserWarning communicates the miss to the user.
    """
    global SELECTED_PATH, SELECTED_SOURCE

    # If the user already exported MLIR_PYTHON_PACKAGE_PREFIX, respect it
    # outright -- nothing to do here, mlir_walker will pick it up.
    env_prefix = os.environ.get("MLIR_PYTHON_PACKAGE_PREFIX")
    if env_prefix and os.path.isdir(env_prefix):
        if env_prefix not in sys.path:
            sys.path.insert(0, env_prefix)
        SELECTED_PATH = env_prefix
        SELECTED_SOURCE = "env"
        return env_prefix

    # Probe 1: Triton's vendored LLVM (matches our C++ shim's LLVM rev).
    for cand in _candidate_triton_paths():
        if os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            _set_env_prefix(cand)
            SELECTED_PATH = cand
            SELECTED_SOURCE = "triton"
            return cand

    # Probe 2: brew LLVM (may exist in the future; not on this Mac today).
    for cand in _candidate_brew_paths():
        if os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            _set_env_prefix(cand)
            SELECTED_PATH = cand
            SELECTED_SOURCE = "brew"
            return cand

    # Probe 3: IREE's bundled MLIR Python API. This is the path that
    # works on this Mac as of 2026-05.
    iree_path = _try_iree_alias()
    if iree_path is not None:
        SELECTED_PATH = iree_path
        SELECTED_SOURCE = "iree"
        return iree_path

    # Probe 4: jaxlib's bundled mlir bindings (jaxlib.mlir.*). Do not do
    # this before Triton is loaded: pre-aliasing jaxlib's MLIR dialect
    # extension and then importing Triton's native bindings can trigger
    # nanobind duplicate-registration aborts. Numeric harnesses load Triton
    # first, then call ``bootstrap_jaxlib_alias`` just before MLIR parsing.
    triton_already_loaded = any(
        name == "triton" or name.startswith("triton.")
        for name in sys.modules
    )
    eager_jaxlib = os.environ.get("TRITON_FRONTEND_EAGER_JAXLIB_MLIR") == "1"
    if (triton_already_loaded or eager_jaxlib) and bootstrap_jaxlib_alias():
        try:
            import jaxlib.mlir  # type: ignore  # noqa: WPS433
            jaxlib_path = os.path.dirname(jaxlib.mlir.__file__)
        except Exception:
            jaxlib_path = "<jaxlib alias>"
        SELECTED_PATH = jaxlib_path
        SELECTED_SOURCE = "jaxlib"
        return jaxlib_path

    return None


# Run the probe at import time. The package ``__init__`` imports this
# module *before* ``mlir_walker``, so by the time the walker calls
# ``try_import_mlir`` either ``mlir.ir`` is on ``sys.modules`` (IREE
# alias path) or ``sys.path`` has the right ``mlir_core`` directory
# (brew / Triton path). If nothing fires the walker emits its
# DEGRADED_WARNING_MESSAGE exactly once, unchanged.
probe_and_wire_mlir()
