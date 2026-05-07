"""Regression test: vendored triton-shared sources match the committed manifest.

If this fails, either:
  * a vendored file was edited (revert the edit, or re-vendor and refresh
    the manifest with ``check_vendor_drift.py --refresh``), or
  * a new file was added under ``include/`` or ``lib/`` without bumping
    the manifest.
"""

from __future__ import annotations

from poc.triton_frontend.vendored.triton_shared import check_vendor_drift


def test_no_vendor_drift() -> None:
    clean, problems = check_vendor_drift.check_drift()
    assert clean, "vendor drift detected:\n  - " + "\n  - ".join(problems)


def test_check_drift_returns_problems_when_manifest_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(check_vendor_drift, "MANIFEST_PATH", tmp_path / "missing.sha256")
    clean, problems = check_vendor_drift.check_drift()
    assert not clean
    assert any("manifest missing" in p for p in problems)
