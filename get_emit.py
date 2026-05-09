with open("poc/triton_frontend/op_emitters/control.py") as f:
    text = f.read()
import re
m = re.search(r'def _emit_region', text)
if m:
    print(text[m.start():m.start()+1000])
