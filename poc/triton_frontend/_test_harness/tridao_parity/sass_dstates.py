import sys, re, os
sys.path.insert(0, "/home/dave/source/tilelang")
import tilelang, tvm
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()

def dump(label, prologue_opt):
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=prologue_opt)
    k = tilelang.compile(pf, target="cuda")
    path = "/tmp/sass_dstates_%s.sass" % label
    try:
        k.export_sass(path)
    except Exception as e:
        s = k._get_sass()
        open(path, "w").write(s)
    sass = open(path).read()
    def cnt(pat):
        return len(re.findall(pat, sass))
    utmaldg = cnt(r'\bUTMALDG')
    ldg = cnt(r'\bLDG\b')
    ldgsts = cnt(r'\bLDGSTS')
    stl = cnt(r'\bSTL\b')
    ldl = cnt(r'\bLDL\b')
    hmma = cnt(r'\bHMMA')
    ffma = cnt(r'\bFFMA\b')
    cpasync = cnt(r'cp\.async') + ldgsts
    print("=== SASS %s (prologue_opt=%s) ===" % (label, prologue_opt))
    print("UTMALDG=%d LDG=%d LDGSTS=%d STL=%d LDL=%d spill(STL+LDL)=%d HMMA=%d FFMA=%d" % (
        utmaldg, ldg, ldgsts, stl, ldl, stl+ldl, hmma, ffma))
    print("path=%s len=%d" % (path, len(sass)))
    return dict(utmaldg=utmaldg, ldg=ldg, ldgsts=ldgsts, spill=stl+ldl, hmma=hmma, ffma=ffma)

off = dump("OFF", False)
opt = dump("OPT", True)
print("=== DELTA OFF->OPT ===")
for kk in off:
    print("%s: %d -> %d" % (kk, off[kk], opt[kk]))
