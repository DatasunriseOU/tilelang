with open("poc/triton_frontend/__init__.py") as f:
    text = f.read()

import re
m = re.search(r'def _walk_text_ttir', text)
if m:
    print(text[m.start():m.start()+2000])
