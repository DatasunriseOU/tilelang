with open("src/transform/producer_consumer_ws.cc", "r") as f:
    text = f.read()

import re
m = re.search(r'class ConsumerSyncRewriter', text)
if m:
    print(text[m.start():m.start()+1000])
