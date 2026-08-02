import contextlib
import ctypes
import logging
import os
import sys
import warnings
from pathlib import Path


def _compute_version() -> str:
    """Return the package version without being polluted by unrelated installs.

    Preference order:
    1) If running from a source checkout (VERSION file present at repo root),
       use the dynamic version from version_provider (falls back to plain VERSION).
    2) Otherwise, use importlib.metadata for the installed distribution.
    3) As a last resort, return a dev sentinel.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent
        version_file = repo_root / "VERSION"
        if version_file.is_file():
            try:
                from version_provider import dynamic_metadata  # type: ignore

                return dynamic_metadata("version")
            except Exception:
                # Fall back to the raw VERSION file if provider isn't available.
                return version_file.read_text().strip()
    except Exception:
        # If any of the above fails, fall through to installed metadata.
        pass

    try:
        from importlib.metadata import version as _dist_version  # py3.8+

        return _dist_version("tilelang")
    except Exception as exc:
        warnings.warn(
            f"tilelang version metadata unavailable ({exc!r}); using development version.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "0.0.dev0"


__version__ = _compute_version()
del _compute_version


logger = logging.getLogger(__name__)


def _import_optional_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return None
        raise
    return torch


_TORCH = _import_optional_torch()


def set_log_level(level):
    """Set the logging level for the module's logger.

    Args:
        level (str or int): Can be the string name of the level (e.g., 'INFO') or the actual level (e.g., logging.INFO).
        OPTIONS: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)


def _init_logger():
    """Initialize the logger specific for this module with custom settings and a Tqdm-based handler."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    class TqdmLoggingHandler(logging.Handler):
        """Custom logging handler that directs log output to tqdm progress bar to avoid interference."""

        def __init__(self, level=logging.NOTSET):
            """Initialize the handler with an optional log level."""
            super().__init__(level)

        def emit(self, record):
            """Emit a log record. Messages are written to tqdm to ensure output in progress bars isn't corrupted."""
            try:
                msg = self.format(record)
                if tqdm is not None:
                    tqdm.write(msg)
            except Exception:
                self.handleError(record)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s  [TileLang:%(name)s:%(levelname)s] (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level("INFO")


from .env import env as env  # noqa: F401

# Skip logger initialization in light import mode
if not env.is_light_import():
    _init_logger()

del _init_logger


def _guard_rocm_tvm_ffi_torch_c_dlpack(torch_module):
    if getattr(torch_module.version, "hip", None) is None:
        return

    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")


@contextlib.contextmanager
def _lazy_extension_imports():
    if _TORCH is not None:
        _guard_rocm_tvm_ffi_torch_c_dlpack(_TORCH)

    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | os.RTLD_LAZY)
    try:
        yield
    finally:
        sys.setdlopenflags(old_flags)


# Skip heavy imports in light import mode
if not env.is_light_import():
    with _lazy_extension_imports():
        from .env import enable_cache, disable_cache, is_cache_enabled  # noqa: F401

        import tvm
        import tvm.base  # noqa: F401
        from tvm import DataType  # noqa: F401

        # Setup tvm search path before importing tvm
        from . import libinfo

        def _load_tile_lang_lib():
            """Load Tile Lang lib"""
            if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
                for path in libinfo.get_dll_directories():
                    os.add_dll_directory(path)
            lib_path = libinfo.find_lib_path("tilelang")
            return ctypes.CDLL(lib_path, mode=ctypes.DEFAULT_MODE | os.RTLD_LAZY), lib_path

        # only load once here
        if env.SKIP_LOADING_TILELANG_SO == "0":
            _LIB, _LIB_PATH = _load_tile_lang_lib()

    if _TORCH is not None:
        from .jit import jit, JITKernel, compile, par_compile  # noqa: F401
        from .profiler import Profiler  # noqa: F401
        from .cache import clear_cache  # noqa: F401
        from .utils import (
            TensorSupplyType,  # noqa: F401
            deprecated,  # noqa: F401
            build_date,  # noqa: F401
        )
    else:
        TensorSupplyType = None  # type: ignore

        def deprecated(method_name, new_method_name, phaseout_version=None):
            import functools

            def _deprecate(func):
                @functools.wraps(func)
                def _wrapper(*args, **kwargs):
                    warnings.warn(
                        f"{method_name} is deprecated, use {new_method_name} instead"
                        + (f" and will be removed in {phaseout_version}" if phaseout_version else ""),
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return func(*args, **kwargs)

                return _wrapper

            return _deprecate

        def build_date(version_str=None):
            import re

            match = re.search(r"\.d(\d{8})\.", version_str or __version__)
            return int(match.group(1)) if match else None

    from .layout import (
        Layout,  # noqa: F401
        Fragment,  # noqa: F401
    )
    from . import (
        analysis,  # noqa: F401
        transform,  # noqa: F401
        language,  # noqa: F401
        tools,  # noqa: F401
    )
    from .language import dtypes  # noqa: F401
    from .transform import PassConfigKey  # noqa: F401

    if _TORCH is not None:
        from .autotuner import autotune  # noqa: F401
        from . import engine  # noqa: F401
        from .engine import (  # noqa: F401
            FusionAutogradPlan,
            FusionBlockDescriptor,
            FusionBlockRegistry,
            FusionOptimizer,
            FusionRegionBuilder,
            FusionScheduleRegistry,
            build_fusion_region,
            build_fusion_region_from_blocks,
            build_fusion_regions_from_blocks,
            compile_fusion_region,
            lower,
            register_c_postproc,
            register_cuda_postproc,
            register_hip_postproc,
        )
    else:

        def _requires_torch(*_args, **_kwargs):
            raise ModuleNotFoundError("torch is required for TileLang JIT, profiler, autotuner, and engine lowering APIs")

        jit = compile = par_compile = autotune = lower = _requires_torch
        JITKernel = Profiler = None  # type: ignore

        def clear_cache():
            cache_dir = env.TILELANG_CACHE_DIR
            raise RuntimeError(
                "tilelang.clear_cache() is disabled because deleting the cache directory "
                "is dangerous. If you accept the risk, remove it manually with "
                f"`rm -rf '{cache_dir}'`."
            )

    from .math import *  # noqa: F403
    from . import ir  # noqa: F401
    from . import tileop  # noqa: F401

    # Promote production language frontends (Triton TTIR, ...). The
    # subpackage registers ``tilelang.frontends.triton`` for both
    # direct callers and the ``compile()`` TTIR-dispatch helper.
    try:
        from . import frontends  # noqa: F401
        from .frontends.triton import (  # noqa: F401
            compile_ttir,
            from_ttir,
            from_triton_kernel,
        )
    except Exception:
        # Triton frontend is optional; tolerate environments without
        # the (large) implementation tree.
        pass
    for _backend_name in ("cpu", "cuda", "rocm"):
        try:
            globals()[_backend_name] = __import__(
                f"{__name__}.{_backend_name}",
                fromlist=[_backend_name],
            )
        except Exception:
            # Backend packages are optional in partially merged local
            # checkouts; importing tilelang.engine must keep working for
            # non-backend-specific lowering.
            pass
    # Fork-specific: ``tilelang.backend`` hosts the reduction-lowerer dispatch
    # and the standalone Metal backend (gemm + reduction). Importing it here
    # registers those implementations (upstream #2165 moved only the
    # CUDA/CPU/ROCm GEMM packages out to ``tilelang.{cpu,cuda,rocm}``).
    try:
        from . import backend as _backend  # noqa: F401
    except Exception:
        pass


del _lazy_extension_imports
del _import_optional_torch
try:
    del _backend_name
except NameError:
    pass
