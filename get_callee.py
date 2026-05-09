with open("poc/triton_frontend/op_emitters/control.py") as f:
    text = f.read()

import re
m = re.search(r'def _parse_callee_attr', text)
if m:
    print(text[m.start():m.start()+800])
