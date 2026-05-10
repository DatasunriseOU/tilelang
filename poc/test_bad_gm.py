import sys
sys.path.insert(0, "/private/tmp/tl_apache_tvm_swap")
from poc.torch_dynamo.fx_to_tilelang import FXToTileLang
class _BadGM:
    class graph:
        nodes = []
    meta = {}
try:
    lowerer = FXToTileLang(_BadGM(), [])
    print(lowerer.run())
except Exception as e:
    print("Exception:", type(e), e)
