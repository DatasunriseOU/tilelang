# Vendored helper for triton-shared integration with the unified fused-kernel compiler.
# Copyright (c) 2026 Project Contributors.
# Original triton-shared sources Copyright (c) Microsoft Corporation and Meta Platforms, Inc.
# Licensed under the MIT License.
"""Vendor drift detector for the microsoft/triton-shared snapshot.

Compares the on-disk file set under
``poc/triton_frontend/vendored/triton_shared/`` against the committed
``.vendor-manifest.sha256`` and reports any of:

  * missing manifest entries (files dropped or renamed),
  * unexpected files in the tracked sub-trees (forgotten to add to the
    manifest after a re-vendor),
  * sha256 mismatches (file edited locally without bumping the manifest).

Tracked sub-trees: ``include/``, ``lib/``, plus the four top-level files
``LICENSE``, ``RegisterTritonStructured.{h,cc}``, ``CMakeLists.txt`` is
deliberately excluded because it is *our* build glue, not vendored upstream
content.

Run from the repo root:

    python -m poc.triton_frontend.vendored.triton_shared.check_vendor_drift

Exit code 0 = no drift, 1 = drift detected. Pass ``--refresh`` to overwrite
the manifest with the current on-disk hashes (use after a deliberate
re-vendoring, then commit the manifest change alongside the source bump).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

VENDOR_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = VENDOR_DIR / ".vendor-manifest.sha256"
TRACKED_SUBTREES: tuple[str, ...] = ("include", "lib")
TRACKED_TOP_LEVEL: tuple[str, ...] = (
    "LICENSE",
    "RegisterTritonStructured.h",
    "RegisterTritonStructured.cc",
)


def _iter_tracked_files() -> Iterable[Path]:
    for sub in TRACKED_SUBTREES:
        root = VENDOR_DIR / sub
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path
    for name in TRACKED_TOP_LEVEL:
        path = VENDOR_DIR / name
        if path.is_file():
            yield path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    if not path.is_file():
        return expected
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `shasum -a 256` output: "<hex>  <relpath>"
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, rel = parts[0], parts[1].lstrip("*")
        expected[rel] = digest
    return expected


def _compute_actual() -> dict[str, str]:
    actual: dict[str, str] = {}
    for path in _iter_tracked_files():
        rel = str(path.relative_to(VENDOR_DIR))
        actual[rel] = _sha256(path)
    return actual


def check_drift() -> tuple[bool, list[str]]:
    """Return (is_clean, problems). Empty problem list iff is_clean."""
    expected = _read_manifest(MANIFEST_PATH)
    actual = _compute_actual()
    problems: list[str] = []

    if not expected:
        try:
            display = str(MANIFEST_PATH.relative_to(VENDOR_DIR))
        except ValueError:
            display = str(MANIFEST_PATH)
        problems.append(f"manifest missing or empty: {display}")
        return False, problems

    expected_set = set(expected)
    actual_set = set(actual)

    for rel in sorted(expected_set - actual_set):
        problems.append(f"missing on disk: {rel}")
    for rel in sorted(actual_set - expected_set):
        problems.append(f"untracked (add to manifest or remove): {rel}")
    for rel in sorted(expected_set & actual_set):
        if expected[rel] != actual[rel]:
            problems.append(
                f"sha256 mismatch: {rel}\n"
                f"  expected: {expected[rel]}\n"
                f"  actual:   {actual[rel]}"
            )
    return not problems, problems


def _refresh_manifest() -> None:
    actual = _compute_actual()
    lines = [
        "# sha256 manifest for the vendored microsoft/triton-shared snapshot.",
        "# Upstream commit: 08684f92ad30696362dce1760a83be889639a3e4",
        "# Refresh by running: python -m poc.triton_frontend.vendored.triton_shared.check_vendor_drift --refresh",
        "# (one space + one space separator to match `shasum -a 256`).",
    ]
    for rel in sorted(actual):
        lines.append(f"{actual[rel]}  {rel}")
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite the manifest with the current on-disk hashes.",
    )
    args = parser.parse_args(argv)

    if args.refresh:
        _refresh_manifest()
        print(f"manifest refreshed: {MANIFEST_PATH}")
        return 0

    clean, problems = check_drift()
    if clean:
        print(f"OK: no vendor drift ({len(_compute_actual())} tracked files)")
        return 0
    sys.stderr.write("vendor drift detected:\n")
    for line in problems:
        sys.stderr.write(f"  - {line}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
