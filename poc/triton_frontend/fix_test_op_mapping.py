import ast
import re

file_path = "poc/triton_frontend/tests/test_op_mapping.py"
with open(file_path, "r") as f:
    source = f.read()

funcs_to_delete = [
    "test_map_tt_make_range",
    "test_map_tt_dead_stubs_are_marked",
    "test_op_mapping_coverage",
]

tree = ast.parse(source)
lines = source.split("\n")
lines_to_delete = set()

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in funcs_to_delete:
        start = node.lineno - 1
        end = node.end_lineno
        # also remove decorators like @pytest.mark.parametrize
        while start > 0 and lines[start - 1].strip().startswith("@"):
            start -= 1
        for i in range(start, end):
            lines_to_delete.add(i)

new_lines = []
for i, line in enumerate(lines):
    if i not in lines_to_delete:
        new_lines.append(line)

new_source = "\n".join(new_lines)

# Remove imports
imports_to_remove = [
    "map_tt_make_range",
    "map_tt_broadcast",
    "map_tt_dot",
    "map_tt_load",
    "map_tt_where",
]

for imp in imports_to_remove:
    new_source = re.sub(r'\s*' + imp + r',\n', '\n', new_source)

with open(file_path, "w") as f:
    f.write(new_source)

print("Done")
