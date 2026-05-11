import re

with open('src/transform/pipeline_planning.cc', 'r') as f:
    content = f.read()

pattern = r"""\s*if\s*\(op->op\.same_as\(builtin::call_extern\(\)\)\)\s*\{.*?func_name\s*==\s*"tl::tcgen5mma_gemm_ts"\s*\|\|\s*func_name\s*==\s*"tl::tcgen5mma_gemm_ss"\)\s*\{.*?\}\s*// TODO \(lei\) Link wgmma to buffers and tl\.wait_wgmma\s*\}"""
# Let's just find the exact block and replace it using python
