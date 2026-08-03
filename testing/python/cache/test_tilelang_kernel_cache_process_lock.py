from __future__ import annotations

import multiprocessing
from pathlib import Path
import sys

import pytest

from tilelang.cache.kernel_cache import _exclusive_file_lock


def _hold_lock(
    path: str,
    attempted: multiprocessing.synchronize.Event,
    entered: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    attempted.set()
    with _exclusive_file_lock(path):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release cache-key lock")


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl cache lock is POSIX-only")
def test_cache_key_lock_serializes_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    lock_path = str(tmp_path / "same-key.lock")
    first_attempted = context.Event()
    first_entered = context.Event()
    first_release = context.Event()
    second_attempted = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    first = context.Process(
        target=_hold_lock,
        args=(lock_path, first_attempted, first_entered, first_release),
    )
    second = context.Process(
        target=_hold_lock,
        args=(lock_path, second_attempted, second_entered, second_release),
    )

    first.start()
    assert first_attempted.wait(10)
    assert first_entered.wait(10)
    second.start()
    assert second_attempted.wait(10)
    assert not second_entered.wait(0.5)

    first_release.set()
    assert second_entered.wait(10)
    second_release.set()
    first.join(10)
    second.join(10)
    assert first.exitcode == 0
    assert second.exitcode == 0
