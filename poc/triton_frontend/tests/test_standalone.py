"""Standalone numeric check that drives Triton -> TileLang end-to-end.

Importing ``triton`` (and therefore ``triton._C.libtriton``) in a pytest
process that has already imported TileLang/TVM can abort on duplicate LLVM
command-line option registration. Keep this test import-safe by running every
Triton-dependent check in a fresh child process.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def test_run():
    _run_from_triton_kernel_smoke_in_fresh_process()

    result = _run_vector_add_numeric_smoke_in_fresh_process()
    assert result["verdict"] == "NUMERIC_PASS", result
    assert result["max_abs_err"] is not None


def _run_from_triton_kernel_smoke_in_fresh_process() -> None:
    script = """
import triton
import triton.language as tl

from poc.triton_frontend import from_triton_kernel


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


func = from_triton_kernel(
    _vector_add_kernel,
    constexprs={"BLOCK": 128},
    target="metal",
)
text = str(func)
assert "_vector_add_kernel" in text, text
assert "arg0" in text, text
assert "arg1" in text, text
assert "arg2" in text, text
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        if (
            sys.platform == "darwin"
            and "triton_frontend cannot import TileLang/TVM in this process"
            in completed.stderr
        ):
            return
        pytest.fail(
            "standalone from_triton_kernel smoke failed with "
            f"exit={completed.returncode}\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


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
