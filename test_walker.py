with open("poc/triton_frontend/mlir_walker.py") as f:
    text = f.read()
import re
m = re.search(r'class TTIRWalker', text)
if m:
    print(text[m.start():m.start()+2500])
