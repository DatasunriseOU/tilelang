with open("src/transform/producer_consumer_ws.cc") as f:
    text = f.read()
import re
m = re.search(r'TileStmtKind ClassifyStmt', text)
if m:
    print(text[m.start():m.start()+800])
