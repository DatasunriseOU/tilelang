from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


def test_from_triton_kernel_uses_generic_mlir_path_without_text_warning() -> None:
    script = textwrap.dedent(
        """
        import warnings

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


        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            prim = from_triton_kernel(
                _vector_add_kernel,
                constexprs={"BLOCK": 16},
                target="cuda",
            )

        messages = [str(item.message) for item in seen]
        assert not any("text-TTIR coverage walker" in msg for msg in messages), messages
        assert prim is not None
        assert len(getattr(prim, "params", ())) >= 3
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
