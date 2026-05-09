import tilelang
import tilelang.language as T
from examples.flash_decoding.example_gqa_decode import flashattn_gqa_decode_split
print("Compiling...")
try:
    kernel = tilelang.compile(flashattn_gqa_decode_split)
except Exception as e:
    import traceback
    traceback.print_exc()
