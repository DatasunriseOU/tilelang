with open("examples/dequantize_gemm/example_dequant_gemm_fine_grained.py", "r") as f:
    text = f.read()

import re

# We will replace the # TODO block with T.copy(B[...], B_shared) WITHOUT coalesced_width
replacement = """
                # TODO(lei): Layout Inference Pass is not efficient to handle the four dims int8 load
                # Replacing with T.copy
                T.copy(B[bx * (block_N // micro_size_y) : bx * (block_N // micro_size_y) + block_N // micro_size_y,
                         ko * (block_K // micro_size_k) : ko * (block_K // micro_size_k) + block_K // micro_size_k,
                         0 : micro_size_y,
                         0 : micro_size_k // num_elems_per_byte],
                       B_shared)
"""

pattern = re.compile(r"\s*# TODO\(lei\): Layout Inference Pass is not efficient.*?(?=\s*for ki in T\.serial)", re.DOTALL)
new_text = pattern.sub(replacement, text)

with open("examples/dequantize_gemm/example_dequant_gemm_fine_grained_test2.py", "w") as f:
    f.write(new_text)
