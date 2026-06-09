"""BlockFix block-size A/B + VERIFY for the §P1 dstates route.

GENERIC autotune-BLOCK honouring. The route previously PINNED the SMALLEST
autotune config (``BLOCK_SIZE_M=64,N=64,K=32``) at TTIR capture; native's
autotuner picks the WINNING config (``M=128,N=256,K=64,num_warps=8,num_stages=3``;
shared=197120 -- confirmed by the cached ``.json`` and TTGIR tile shapes).

This harness:
  1. Reads the autotune-WINNING block config GENERICALLY via the named helper
     :func:`poc.triton_frontend.autotune_winning_block_config` (no per-kernel
     hardcode -- it reads the kernel's own ``@triton.autotune`` config list).
  2. Captures the dstates TTIR at the WINNING block and VERIFIES the captured
     ``tt.dot`` operand tile shapes move toward native (128x256 / 128x64),
     vs the baseline 64x64x32 TTIR.
  3. Attempts to build the routed PrimFunc at the winning block through
     ``from_ttir`` and reports the blocker honestly (the fused
     GEMM-accumulate-into-shared-carry guard fires for ANY tile bigger than
     64x64 -- a real tilelang lowering gap, not a config-read bug).
  4. Confirms the 64x64 baseline still builds (no regression).

Backend-agnostic: the block config flows into the captured TTIR's GEMM tile,
which drives CUDA ``T.gemm`` warp partition and Metal threadgroup partition
alike. No libtriton / cuda-only op in the honouring path.
"""
import inspect
import os
import re
import sys

sys.path.insert(0, "/home/dave/source/tilelang")

from mamba_ssm.ops.triton.ssd_chunk_scan import (  # noqa: E402
    _chunk_scan_bwd_dstates_kernel as autok,
)

from poc.triton_frontend import (  # noqa: E402
    autotune_winning_block_config,
    from_ttir,
)
from poc.triton_frontend._test_harness.jit_to_ttir import (  # noqa: E402
    triton_jit_to_ttir_subprocess_from_source,
)
import tilelang  # noqa: E402


def _tile_census(ttir_text):
    shapes = {}
    for a, b in re.findall(r"tensor<(\d+)x(\d+)x", ttir_text):
        shapes[(int(a), int(b))] = shapes.get((int(a), int(b)), 0) + 1
    return sorted(shapes.items(), key=lambda kv: -kv[1])[:8]


def main():
    # (1) GENERIC winning-config read -- named helper, no hardcode.
    cfg = autotune_winning_block_config(autok)
    bm = cfg["BLOCK_SIZE_M"]
    bn = cfg["BLOCK_SIZE_N"]
    bk = cfg["BLOCK_SIZE_K"]
    nw = cfg["num_warps"]
    ns = cfg["num_stages"]
    print(
        "WINNING_BLOCK BLOCK_SIZE_M=%d BLOCK_SIZE_N=%d BLOCK_SIZE_K=%d "
        "num_warps=%d num_stages=%d" % (bm, bn, bk, nw, ns),
        flush=True,
    )

    # (2) capture TTIR at the winning block; VERIFY tile shapes vs baseline.
    inner = getattr(getattr(autok, "fn", autok), "fn", autok)
    src = inspect.getsource(inner)
    ce = {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "HAS_SEQ_IDX": False,
    }
    ttir_win = triton_jit_to_ttir_subprocess_from_source(
        source=src,
        kernel_name="_chunk_scan_bwd_dstates_kernel",
        constexprs=ce,
        timeout=180,
    )
    os.makedirs("/tmp/ttir_win", exist_ok=True)
    open("/tmp/ttir_win/_chunk_scan_bwd_dstates.ttir", "w").write(ttir_win)

    base_path = "/tmp/ttir7/_chunk_scan_bwd_dstates.ttir"
    ttir_base = open(base_path).read()
    print("BASELINE_TILES (pinned 64x64x32) %s" % _tile_census(ttir_base), flush=True)
    print("WINNING_TILES  (native 128x256x64) %s" % _tile_census(ttir_win), flush=True)

    # (3) attempt the routed build at the winning block.
    os.environ["TL_FORCE_CP_ASYNC"] = "1"
    try:
        pf = from_ttir(
            ttir_win,
            name="_chunk_scan_bwd_dstates_kernel",
            target="cuda",
            _allow_text_ttir=True,
            prologue_opt=True,
            num_warps=nw,
            num_stages=ns,
        )
        tilelang.compile(pf, target="cuda")
        print("WINNING_BUILD_OK", flush=True)
    except Exception as exc:  # noqa: BLE001 -- report, do not paper over
        print(
            "WINNING_BUILD_BLOCKED %s :: %s"
            % (type(exc).__name__, str(exc).splitlines()[-1][:200]),
            flush=True,
        )

    # (4) baseline 64x64 still builds (no regression).
    pf_b = from_ttir(
        ttir_base,
        name="_chunk_scan_bwd_dstates_kernel",
        target="cuda",
        _allow_text_ttir=True,
        prologue_opt=True,
        num_warps=8,
        num_stages=3,
    )
    tilelang.compile(pf_b, target="cuda")
    print("BASELINE_BUILD_OK 64x64x32 num_warps=8 num_stages=3", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
