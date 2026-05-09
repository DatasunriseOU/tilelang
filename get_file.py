with open("examples/flash_decoding/example_gqa_decode.py") as f:
    text = f.read()

import re
m = re.search(r'for k in T.Pipelined', text)
if m:
    print(text[m.start():m.start()+2000])
