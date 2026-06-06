import sys
sys.path.insert(0, "/home/dave/source/tilelang")
import tilelang, tvm
from poc.triton_frontend import from_ttir
name = sys.argv[1]
ttir = open("/tmp/ttir7/%s.ttir" % name).read()
pf = from_ttir(ttir, name=name+"_kernel", target="cuda", _allow_text_ttir=True)
js = tvm.ir.save_json(pf)
open("/tmp/pf_%s.json" % name, "w").write(js)
print("WROTE json len=%d" % len(js))
# verify round-trip in-process
pf2 = tvm.ir.load_json(js)
print("ROUNDTRIP_OK type=%s nparams=%d" % (type(pf2).__name__, len(pf2.params)))
