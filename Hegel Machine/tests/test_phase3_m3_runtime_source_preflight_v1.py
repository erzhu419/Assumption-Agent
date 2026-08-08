from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile

import pytest

from hegel_machine import phase3_m3_runtime_source_preflight_v1 as preflight


RUNTIME_PATHS = (
    "runtime/b.py",
    "runtime/a.py",
)

VENDORED_TOMLI_SHA256 = {
    "Hegel Machine/src/hegel_machine/_vendor/tomli/LICENSE": (
        "b80816b0d530b8accb4c2211783790984a6e3b61922c2b5ee92f3372ab2742fe"
    ),
    "Hegel Machine/src/hegel_machine/_vendor/tomli/__init__.py": (
        "9eb042d7c0db5d14c2168ec4946e410de5a91c9cce86892f5e4db5e4633c6762"
    ),
    "Hegel Machine/src/hegel_machine/_vendor/tomli/_parser.py": (
        "a412234c86bf710b361e0943276961f0e25fa6d7c36ba7a0e7eec87a3e018c7b"
    ),
    "Hegel Machine/src/hegel_machine/_vendor/tomli/_re.py": (
        "a12359fe294523a72112e434d58452a14c9d050affa2417f9927474e4166bfdd"
    ),
    "Hegel Machine/src/hegel_machine/_vendor/tomli/_types.py": (
        "f864c6d9552a929c7032ace654ee05ef26ca75d21b027b801d77e65907138b74"
    ),
}


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed.stdout


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "--all")
    _git(
        repository,
        "-c",
        "user.name=M3 Runtime Test",
        "-c",
        "user.email=m3-runtime-test@local.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD").decode("ascii").strip()


def test_vendored_tomli_is_exact_and_inside_runtime_source_closure() -> None:
    runtime_paths = set(preflight.DEFAULT_M3_RUNTIME_SOURCE_PATHS)
    assert len(runtime_paths) == len(preflight.DEFAULT_M3_RUNTIME_SOURCE_PATHS)
    assert len(runtime_paths) <= preflight.MAX_RUNTIME_SOURCE_FILES
    assert "Hegel Machine/src/hegel_machine/_vendor/__init__.py" in runtime_paths
    assert "Hegel Machine/src/hegel_machine/phase2b_adapter.py" in runtime_paths
    assert set(VENDORED_TOMLI_SHA256) <= runtime_paths

    for repository_path, expected_sha256 in VENDORED_TOMLI_SHA256.items():
        payload = (preflight.REPOSITORY_ROOT / repository_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256

    assert not any(
        "__pycache__" in Path(path).parts
        or Path(path).suffix in {".pyc", ".pyo"}
        for path in runtime_paths
    )


@pytest.fixture
def runtime_repository() -> tuple[Path, str]:
    with tempfile.TemporaryDirectory(
        prefix="hegel-m3-runtime-source-test-", dir="/tmp"
    ) as raw:
        repository = Path(raw).resolve() / "runtime-repository"
        repository.mkdir(mode=0o700)
        _git(repository, "init", "--quiet")
        source = repository / "runtime"
        source.mkdir(mode=0o755)
        (source / "a.py").write_bytes(b"A = 1\n")
        (source / "b.py").write_bytes(b"B = 2\n")
        (source / "a.py").chmod(0o644)
        (source / "b.py").chmod(0o644)
        yield repository, _commit(repository, "runtime source basis")


def test_success_resolves_full_commit_and_builds_deterministic_crosslinked_receipt(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, commit = runtime_repository

    first = preflight.build_runtime_source_preflight_v1(
        "HEAD",
        repository_root=repository,
        runtime_paths=RUNTIME_PATHS,
    )
    second = preflight.build_runtime_source_preflight_v1(
        commit[:12],
        repository_root=repository,
        runtime_paths=tuple(reversed(RUNTIME_PATHS)),
    )

    assert first.expected_runtime_commit == commit
    assert dict(first.manifest_fields) == dict(second.manifest_fields)
    assert dict(first.receipt_fields) == dict(second.receipt_fields)
    rows = first.manifest_fields["runtime_source_files"]
    assert [row["repository_path"] for row in rows] == [
        "runtime/a.py",
        "runtime/b.py",
    ]
    assert first.receipt_fields["git_index_matches_commit"] is True
    assert first.receipt_fields["working_tree_matches_commit"] is True
    assert first.receipt_fields["docker_invoked"] is False
    assert first.receipt_fields["state_changed"] is False
    preflight.validate_runtime_source_preflight_v1(
        first.manifest_fields,
        first.receipt_fields,
    )


def test_dirty_runtime_bytes_are_rejected(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, commit = runtime_repository
    (repository / "runtime/a.py").write_bytes(b"A = 999\n")

    with pytest.raises(preflight.M3RuntimeSourcePreflightError) as caught:
        preflight.build_runtime_source_preflight_v1(
            commit,
            repository_root=repository,
            runtime_paths=RUNTIME_PATHS,
        )

    assert caught.value.code == preflight.FAIL_BYTES


def test_missing_runtime_file_is_rejected(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, commit = runtime_repository
    (repository / "runtime/a.py").unlink()

    with pytest.raises(preflight.M3RuntimeSourcePreflightError) as caught:
        preflight.build_runtime_source_preflight_v1(
            commit,
            repository_root=repository,
            runtime_paths=RUNTIME_PATHS,
        )

    assert caught.value.code == preflight.FAIL_WORKTREE


def test_runtime_symlink_is_rejected_even_when_target_bytes_match(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, commit = runtime_repository
    source = repository / "runtime/a.py"
    replacement = repository / "same-bytes.py"
    replacement.write_bytes(source.read_bytes())
    replacement.chmod(0o644)
    source.unlink()
    source.symlink_to(replacement)

    with pytest.raises(preflight.M3RuntimeSourcePreflightError) as caught:
        preflight.build_runtime_source_preflight_v1(
            commit,
            repository_root=repository,
            runtime_paths=RUNTIME_PATHS,
        )

    assert caught.value.code == preflight.FAIL_SYMLINK


def test_posix_mode_mismatch_is_rejected_even_when_git_executable_bit_is_same(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, commit = runtime_repository
    (repository / "runtime/a.py").chmod(0o600)

    with pytest.raises(preflight.M3RuntimeSourcePreflightError) as caught:
        preflight.build_runtime_source_preflight_v1(
            commit,
            repository_root=repository,
            runtime_paths=RUNTIME_PATHS,
        )

    assert caught.value.code == preflight.FAIL_MODE


def test_staged_index_substitution_is_rejected_when_worktree_bytes_match_commit(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, commit = runtime_repository
    source = repository / "runtime/a.py"
    committed = source.read_bytes()
    source.write_bytes(b"A = 'staged-substitution'\n")
    _git(repository, "add", "--", "runtime/a.py")
    source.write_bytes(committed)
    source.chmod(0o644)

    with pytest.raises(preflight.M3RuntimeSourcePreflightError) as caught:
        preflight.build_runtime_source_preflight_v1(
            commit,
            repository_root=repository,
            runtime_paths=RUNTIME_PATHS,
        )

    assert caught.value.code == preflight.FAIL_INDEX


def test_expected_commit_must_be_checked_out_head(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, first_commit = runtime_repository
    (repository / "unrelated.txt").write_text("new commit\n", encoding="utf-8")
    _commit(repository, "unrelated second commit")

    with pytest.raises(preflight.M3RuntimeSourcePreflightError) as caught:
        preflight.build_runtime_source_preflight_v1(
            first_commit,
            repository_root=repository,
            runtime_paths=RUNTIME_PATHS,
        )

    assert caught.value.code == preflight.FAIL_COMMIT


def test_unrelated_dirty_file_does_not_taint_path_scoped_runtime_identity(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, _commit_id = runtime_repository
    unrelated = repository / "notes.txt"
    unrelated.write_text("committed\n", encoding="utf-8")
    commit = _commit(repository, "add unrelated file")
    unrelated.write_text("dirty but outside runtime closure\n", encoding="utf-8")

    result = preflight.build_runtime_source_preflight_v1(
        commit,
        repository_root=repository,
        runtime_paths=RUNTIME_PATHS,
    )

    assert result.expected_runtime_commit == commit
    assert result.receipt_fields["path_scoped_status_clean"] is True


def test_manifest_or_receipt_tamper_is_rejected(
    runtime_repository: tuple[Path, str],
) -> None:
    repository, commit = runtime_repository
    result = preflight.build_runtime_source_preflight_v1(
        commit,
        repository_root=repository,
        runtime_paths=RUNTIME_PATHS,
    )
    receipt = dict(result.receipt_fields)
    receipt["working_tree_matches_commit"] = False

    with pytest.raises(preflight.M3RuntimeSourcePreflightError) as caught:
        preflight.validate_runtime_source_preflight_v1(
            result.manifest_fields,
            receipt,
        )

    assert caught.value.code == preflight.FAIL_RECEIPT
