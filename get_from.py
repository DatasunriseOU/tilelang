with open("poc/triton_frontend/__init__.py") as f:
    text = f.read()

import re
m = re.search(r'def from_ttir', text)
if m:
    print(text[m.start():m.start()+1500])
