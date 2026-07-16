from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from replication_runtime.noaa_gsod_v1.contract import NoaaGsodError
from replication_runtime.noaa_gsod_v1.development_implementation import (
    FIXED_RELATIVE_PATH_SET_HASH,
    IMPLEMENTATION_RELATIVE_PATHS,
    build_development_implementation_set,
    verify_development_implementation_set,
)


PROJECT = Path(__file__).resolve().parents[1]


def _copy_fixed_tree(destination: Path) -> None:
    for relative_path in IMPLEMENTATION_RELATIVE_PATHS:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative_path, target)


def test_implementation_set_is_stable_exact_and_live_verifiable() -> None:
    first = build_development_implementation_set()
    second = build_development_implementation_set()
    assert first == second
    assert first["fixed_relative_path_set_hash"] == FIXED_RELATIVE_PATH_SET_HASH
    assert [row["relative_path"] for row in first["files"]] == list(
        IMPLEMENTATION_RELATIVE_PATHS
    )
    assert verify_development_implementation_set(first) == first
    assert verify_development_implementation_set(
        first,
        repository_root=PROJECT,
        verify_live_files=True,
    ) == first


def test_tampering_any_fixed_file_changes_the_set_hash(tmp_path: Path) -> None:
    repository = tmp_path / "copied-repository"
    _copy_fixed_tree(repository)
    baseline = build_development_implementation_set(repository_root=repository)
    for relative_path in IMPLEMENTATION_RELATIVE_PATHS:
        target = repository / relative_path
        original = target.read_bytes()
        target.write_bytes(original + b"\n# implementation-set tamper\n")
        changed = build_development_implementation_set(repository_root=repository)
        assert changed["implementation_set_hash"] != baseline[
            "implementation_set_hash"
        ]
        target.write_bytes(original)
        restored = build_development_implementation_set(repository_root=repository)
        assert restored == baseline


def test_missing_or_symlinked_fixed_dependency_fails_closed(tmp_path: Path) -> None:
    repository = tmp_path / "copied-repository"
    _copy_fixed_tree(repository)
    relative_path = IMPLEMENTATION_RELATIVE_PATHS[0]
    target = repository / relative_path
    original = target.read_bytes()
    target.unlink()
    with pytest.raises(NoaaGsodError, match="missing"):
        build_development_implementation_set(repository_root=repository)

    external = tmp_path / "external.py"
    external.write_bytes(original)
    target.symlink_to(external)
    with pytest.raises(NoaaGsodError, match="symbolic link"):
        build_development_implementation_set(repository_root=repository)


def test_committed_cli_entrypoints_import_from_a_direct_script_launch() -> None:
    scripts = (
        "scripts/export_noaa_gsod_development_source_v1.py",
        "scripts/prepare_noaa_gsod_development_freeze_v1.py",
        "scripts/run_noaa_gsod_formal_development_v1.py",
    )
    for relative_path in scripts:
        completed = subprocess.run(
            [sys.executable, str(PROJECT / relative_path), "--help"],
            cwd=PROJECT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
