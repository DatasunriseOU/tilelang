import re

# 1. Update test_dot_reduce_atomic.py
with open('poc/triton_frontend/tests/test_dot_reduce_atomic.py', 'r') as f:
    t1 = f.read()

# remove imports of map_tt_dot and map_tt_reduce from op_mapping
t1 = re.sub(r'\s*map_tt_dot,\n', '\n', t1)
t1 = re.sub(r'\s*map_tt_reduce,\n', '\n', t1)

# add import from poc.triton_frontend.op_emitters.reduction
t1 = re.sub(
    r'(from poc.triton_frontend.op_mapping import \([^\)]+\))',
    r'\1\nfrom poc.triton_frontend.op_emitters.reduction import map_tt_dot, map_tt_reduce',
    t1, count=1
)

with open('poc/triton_frontend/tests/test_dot_reduce_atomic.py', 'w') as f:
    f.write(t1)


# 2. Update test_op_mapping.py
with open('poc/triton_frontend/tests/test_op_mapping.py', 'r') as f:
    t2 = f.read()

# remove map_tt_make_range import
t2 = re.sub(r'\s*map_tt_make_range,\n', '\n', t2)

# import emit_tt_make_range from op_emitters.memory to test the same logic?
# Actually, the test is specifically testing the old op_mapping.py logic.
# Wait, let's just delete the tests that are testing the dead stubs we removed!
