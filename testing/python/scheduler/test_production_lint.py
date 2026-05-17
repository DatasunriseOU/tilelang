from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _attribute_chain(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


class _ProductionMonkeypatchVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "pytest" or alias.name.startswith("unittest.mock"):
                self.findings.append(f"{self.path}:{node.lineno}: import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"pytest", "mock", "unittest.mock"}:
            self.findings.append(f"{self.path}:{node.lineno}: import from {node.module}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_args(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_args(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attribute_chain(node.func)
        if chain is not None and (
            chain[0] == "monkeypatch" or chain in {("mock", "patch"), ("unittest", "mock", "patch")}
        ):
            self.findings.append(f"{self.path}:{node.lineno}: call {'.'.join(chain)}")
        self.generic_visit(node)

    def _check_args(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            args.append(node.args.vararg)
        if node.args.kwarg is not None:
            args.append(node.args.kwarg)
        if any(arg.arg == "monkeypatch" for arg in args):
            self.findings.append(f"{self.path}:{node.lineno}: monkeypatch argument")


def test_lint_tilelang_production_has_no_monkeypatch_or_mock_patch() -> None:
    findings: list[str] = []
    for path in sorted((ROOT / "tilelang").rglob("*.py")):
        relative_path = path.relative_to(ROOT)
        if relative_path.parts[:2] == ("tilelang", "testing"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _ProductionMonkeypatchVisitor(relative_path)
        visitor.visit(tree)
        findings.extend(visitor.findings)

    assert findings == []
