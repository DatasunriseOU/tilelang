import re

with open("src/transform/lower_tma_to_ptr_arith.cc", "r") as f:
    content = f.read()

# Actually, it's easier to use replace tool for pipeline_planning.cc.
