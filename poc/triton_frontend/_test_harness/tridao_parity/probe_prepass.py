import sys
sys.path.insert(0, "/home/dave/source/tilelang")
from poc.triton_frontend.pipeline import run_ptr_analysis_pre_pass_subprocess, run_ptr_analysis_pre_pass, _libtriton_loaded
from poc.triton_frontend.ptr_analysis import shim_available, shim_subprocess_available, dialects_available

print("libtriton_loaded", _libtriton_loaded())
print("shim_available", shim_available())
print("shim_subprocess_available", shim_subprocess_available())
print("dialects_available", dialects_available())

name = sys.argv[1] if len(sys.argv) > 1 else "_chunk_scan_bwd_dstates"
ttir = open("/tmp/ttir7/%s.ttir" % name).read()
print("INPUT tt.load=%d tt.addptr=%d tt.make_range/splat" % (ttir.count("tt.load"), ttir.count("tt.addptr")))

# Try subprocess path
try:
    if _libtriton_loaded():
        out, state = run_ptr_analysis_pre_pass_subprocess(ttir)
        which = "subprocess"
    else:
        out, state = run_ptr_analysis_pre_pass(ttir)
        which = "in-process"
    print("PREPASS(%s) OK len_out=%d nstates=%d" % (which, len(out), len(state)))
    print("  make_tptr=%d tts.=%d tt.load=%d tt.addptr=%d" % (
        out.count("make_tptr"), out.count("tts."), out.count("tt.load"), out.count("tt.addptr")))
    # show a few state entries
    for i, (k, v) in enumerate(list(state.items())[:6]):
        print("  STATE[%s] = %r" % (k, v))
    open("/tmp/prepass_%s.out" % name, "w").write(out)
    print("  wrote /tmp/prepass_%s.out" % name)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("PREPASS RAISED", repr(e))
