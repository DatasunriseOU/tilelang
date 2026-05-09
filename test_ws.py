import tilelang as tl
import sys
sys.path.append(".")
from examples.flash_decoding.example_gqa_decode import flashattn
kernel = flashattn(batch=2, heads=4, groups=1, seqlen_kv=128, dim=64, block_N=128, block_H=64, num_split=1, num_stages=2, threads=128)
print("kernel target attr:", kernel.attrs.get("target", "NOT_FOUND"))
