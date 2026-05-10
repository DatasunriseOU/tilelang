import ast
import re

file_path = "poc/triton_frontend/op_mapping.py"
with open(file_path, "r") as f:
    source = f.read()

funcs_to_delete = [
    "map_tt_load",
    "map_tt_store",
    "map_tt_dot",
    "map_tt_reduce",
    "map_tt_broadcast",
    "map_tt_splat",
    "map_tt_expand_dims",
    "map_tt_reshape",
    "map_tt_make_range",
]

# We also want to delete the comments preceding them, like "DEAD-BUT-LOADED"
# Let's find the start and end of each function using AST
tree = ast.parse(source)
lines = source.split("\n")
lines_to_delete = set()

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in funcs_to_delete:
        start = node.lineno - 1
        # Look backwards for comments immediately preceding the function
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        end = node.end_lineno
        for i in range(start, end):
            lines_to_delete.add(i)

new_lines = []
for i, line in enumerate(lines):
    if i not in lines_to_delete:
        new_lines.append(line)

new_source = "\n".join(new_lines)

# Remove the entries from OP_TABLE
for func in funcs_to_delete:
    # regex to remove `"name": func,` line
    pattern = r'\s*"[^"]+":\s*' + func + r',\n'
    new_source = re.sub(pattern, '', new_source)

# Also remove them from __all__
for func in funcs_to_delete:
    pattern = r'\s*"' + func + r'",\n'
    new_source = re.sub(pattern, '', new_source)

with open(file_path, "w") as f:
    f.write(new_source)

print("Done")
