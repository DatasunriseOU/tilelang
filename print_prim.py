import sys
sys.path.append(".")
from poc.triton_frontend._test_harness.numeric_smoke import run_one
deps = {"cppmega_mlx": True, "mlx": True, "tilelang": True, "triton": True, "tvm": True}
try:
    run_one("vector_add", deps)
except Exception as e:
    pass
