with open("poc/triton_frontend/__init__.py") as f:
    text = f.read()

import re
m = re.search(r'def _prepass', text)
if m:
    print(text[m.start():m.start()+800])
