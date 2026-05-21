"""Standalone numeric check that drives Triton -> TileLang end-to-end.

Importing ``triton`` (and therefore ``triton._C.libtriton``) at module
scope conflicts with ``_triton_frontend_cxx`` -- both statically link
their own LLVM. We therefore guard the triton import behind a module-
level skip that fires when the shim is already loaded in this process,
and lazy-import triton inside the test body. Running this file on its
own (no shim loaded yet) behaves exactly like before.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from poc.triton_frontend._test_harness.native_import_guard import (
    triton_import_block_reason,
)


# If the C++ shim / TileLang / TVM side has already been imported in this
# Python process, loading Triton's libtriton too can register duplicate LLVM
# cl::opts and SIGABRT the interpreter. Skip the whole module cleanly in that
# case; operators can rerun this file in a fresh process to exercise it.
_triton_block_reason = triton_import_block_reason()
if _triton_block_reason is not None:
    pytest.skip(
        "test_standalone.py skipped: "
        f"{_triton_block_reason} Re-run "
        "`pytest poc/triton_frontend/tests/test_standalone.py` in a fresh "
        "Python process to exercise it.",
        allow_module_level=True,
    )

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from poc.triton_frontend import from_triton_kernel  # noqa: E402


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_run():
    BLOCK = 128
    func = from_triton_kernel(
        _vector_add_kernel,
        constexprs={"BLOCK": BLOCK},
        target="metal",
    )
    text = str(func)
    assert "_vector_add_kernel" in text
    assert "arg0" in text
    assert "arg1" in text
    assert "arg2" in text

    result = _run_vector_add_numeric_smoke_in_fresh_process()
    assert result["verdict"] == "NUMERIC_PASS", result
    assert result["max_abs_err"] is not None


def _run_vector_add_numeric_smoke_in_fresh_process() -> dict[str, object]:
    sentinel = "__TRITON_STANDALONE_VECTOR_ADD__"
    script = f"""
import dataclasses
import json
from poc.triton_frontend._test_harness import numeric_smoke

result = numeric_smoke.run_one("vector_add", numeric_smoke._probe_deps())
print({sentinel!r} + json.dumps(dataclasses.asdict(result), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "standalone vector_add numeric smoke failed with "
            f"exit={completed.returncode}\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(sentinel):
            result = json.loads(line[len(sentinel) :])
            if result["verdict"] == "SKIP":
                pytest.skip(result.get("detail") or "<no detail>")
            return result
    pytest.fail(
        "standalone vector_add numeric smoke did not emit result sentinel\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )


if __name__ == "__main__":
    test_run()
