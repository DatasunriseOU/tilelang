with open("src/transform/producer_consumer_ws.cc") as f:
    text = f.read()

import re
m = re.search(r'// \s*3\.\s*Identify\s*Producer\s*vs\s*Consumer\s*Statements', text)
if m:
    print(text[m.start()-50:m.start()+2000])
