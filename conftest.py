"""Worktree-mode conftest for the z3-idea-10 branch.

The cppmega.mlx ``.venv`` ships with a scikit-build "editable" tilelang
install whose meta-path finder hard-codes ``/tmp/tl_apache_tvm_swap`` as the
source root. When this branch is checked out into a separate worktree
(``/tmp/tl_idea10_worktree``) the in-tree edits are invisible until we
rewrite the redirecting finder's mapping. We do that here at session start
so ``pytest`` resolves ``tilelang.*`` to the worktree's files.

This file is a no-op when the editable redirecting finder is not present
(e.g. on systems where tilelang is wheel-installed or simply on PYTHONPATH).
"""

from __future__ import annotations

import os
import sys


_WORKTREE_ROOT = os.path.abspath(os.path.dirname(__file__))


def _retarget_editable_finder(target_root: str) -> None:
    """Point the scikit-build redirecting finder at ``target_root`` for tilelang.*."""
    for finder in list(sys.meta_path):
        if "ScikitBuildRedirectingFinder" not in type(finder).__name__:
            continue
        for attr in dir(finder):
            if attr.startswith("_"):
                continue
            v = getattr(finder, attr, None)
            if not isinstance(v, dict):
                continue
            if not any(isinstance(k, str) and k.startswith("tilelang") for k in v):
                continue
            for key in list(v.keys()):
                if not (isinstance(key, str) and key.startswith("tilelang")):
                    continue
                old = v[key]
                if not isinstance(old, str):
                    continue
                # Only rewrite if currently pointing somewhere else under
                # /tmp/tl_apache_tvm_swap or any non-target root.
                if old.startswith(target_root):
                    continue
                # Find a mirror path in the worktree.
                #   .../tilelang/foo.py  ->  ${target_root}/tilelang/foo.py
                idx = old.rfind("/tilelang/")
                if idx == -1:
                    if old.endswith("/tilelang"):
                        v[key] = os.path.join(target_root, "tilelang")
                    continue
                rel = old[idx + 1:]
                v[key] = os.path.join(target_root, rel)


_retarget_editable_finder(_WORKTREE_ROOT)
