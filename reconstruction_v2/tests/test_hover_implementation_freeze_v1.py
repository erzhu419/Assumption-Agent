from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import hover_implementation_freeze_v1 as freeze


ROLE_PATHS = {
    "python_role": "roles/example.py",
    "asset_role": "roles/asset.json",
}


def _git(project: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _manifest(project: Path, bindings: dict[str, dict[str, str]]) -> dict[str, object]:
    body: dict[str, object] = {
        "bindings": bindings,
        "schema": freeze.SCHEMA,
        "version": "v1",
    }
    return {**body, freeze.HASH_FIELD: freeze.stable_hash(body)}


def _project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "project"
    (project / "roles").mkdir(parents=True)
    (project / "manifests").mkdir()
    (project / "roles" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "roles" / "asset.json").write_text('{"asset":true}\n', encoding="utf-8")
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "synthetic@example.invalid")
    _git(project, "config", "user.name", "Synthetic Test")
    _git(project, "add", "roles")
    _git(project, "commit", "-q", "-m", "roles")
    bindings: dict[str, dict[str, str]] = {}
    for role, relative in ROLE_PATHS.items():
        raw = (project / relative).read_bytes()
        bindings[role] = {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": freeze.git_blob_sha1(raw),
            "relative_path": relative,
        }
    path = project / freeze.MANIFEST_RELATIVE_PATH
    path.write_text(
        json.dumps(_manifest(project, bindings), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(project, "add", freeze.MANIFEST_RELATIVE_PATH)
    _git(project, "commit", "-q", "-m", "freeze")
    return project, ROLE_PATHS


def test_verifies_manifest_roles_working_bytes_head_and_loaded_origin(tmp_path: Path) -> None:
    project, expected = _project(tmp_path)
    receipt = freeze.verify_committed_implementation_freeze(
        project, expected_role_paths=expected
    )
    assert receipt[freeze.HASH_FIELD]
    assert receipt["required_role_count"] == 2
    assert receipt["all_bindings_byte_match_committed_HEAD"] is True
    assert receipt["git_HEAD_stable_during_verification"] is True
    module_path = (project / expected["python_role"]).resolve()
    module = SimpleNamespace(
        __file__=str(module_path),
        __spec__=SimpleNamespace(origin=str(module_path)),
    )
    assert freeze.verify_loaded_module_origins(
        project=project,
        implementation_receipt=receipt,
        loaded_modules_by_role={"python_role": module},
        expected_roles=("python_role",),
    ) == {"python_role": str(module_path)}
    assert freeze.verify_expected_git_head(
        project=project, expected_git_head=receipt["verified_git_head"]
    ) == receipt["verified_git_head"]


def test_rejects_dirty_role_even_when_manifest_remains_committed(tmp_path: Path) -> None:
    project, expected = _project(tmp_path)
    (project / expected["python_role"]).write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="working bytes"):
        freeze.verify_committed_implementation_freeze(
            project, expected_role_paths=expected
        )


def test_rejects_manifest_self_hash_and_exact_role_drift(tmp_path: Path) -> None:
    project, expected = _project(tmp_path)
    path = project / freeze.MANIFEST_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[freeze.HASH_FIELD] = "0" * 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _git(project, "add", freeze.MANIFEST_RELATIVE_PATH)
    _git(project, "commit", "-q", "-m", "bad self hash")
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="self-hash"):
        freeze.verify_committed_implementation_freeze(
            project, expected_role_paths=expected
        )

    good_project, expected = _project(tmp_path / "second")
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="role set"):
        freeze.verify_committed_implementation_freeze(
            good_project,
            expected_role_paths={**expected, "missing": "roles/missing.py"},
        )


def test_rejects_unsafe_or_overlapping_expected_paths(tmp_path: Path) -> None:
    project, _expected = _project(tmp_path)
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="unsafe"):
        freeze.verify_committed_implementation_freeze(
            project, expected_role_paths={"python_role": "../escape.py"}
        )
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="registry"):
        freeze.verify_committed_implementation_freeze(
            project,
            expected_role_paths={"python_role": "roles/example.py", "other": "roles/example.py"},
        )


def test_rejects_head_drift_and_wrong_loaded_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, expected = _project(tmp_path)
    real_head = freeze._git_head(project)
    calls = iter((real_head, "f" * 40))
    monkeypatch.setattr(freeze, "_git_head", lambda _repository: next(calls))
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="HEAD drifted"):
        freeze.verify_committed_implementation_freeze(
            project, expected_role_paths=expected
        )

    monkeypatch.undo()
    receipt = freeze.verify_committed_implementation_freeze(
        project, expected_role_paths=expected
    )
    wrong = (project / expected["asset_role"]).resolve()
    module = SimpleNamespace(
        __file__=str(wrong), __spec__=SimpleNamespace(origin=str(wrong))
    )
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="origin drifted"):
        freeze.verify_loaded_module_origins(
            project=project,
            implementation_receipt=receipt,
            loaded_modules_by_role={"python_role": module},
            expected_roles=("python_role",),
        )
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="role set"):
        freeze.verify_loaded_module_origins(
            project=project,
            implementation_receipt=receipt,
            loaded_modules_by_role={},
            expected_roles=("python_role",),
        )


def test_rejects_unfrozen_loaded_project_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, expected = _project(tmp_path)
    receipt = freeze.verify_committed_implementation_freeze(
        project, expected_role_paths=expected
    )
    unfrozen = project / "roles" / "unfrozen.py"
    unfrozen.write_text("VALUE = 9\n", encoding="utf-8")
    module = SimpleNamespace(
        __file__=str(unfrozen),
        __spec__=SimpleNamespace(origin=str(unfrozen)),
    )
    monkeypatch.setitem(sys.modules, "synthetic_unfrozen_project_module", module)
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="unfrozen"):
        freeze.verify_no_unfrozen_project_modules(
            project=project,
            implementation_receipt=receipt,
        )


def test_rejects_executable_git_mode_and_symlinked_parent(tmp_path: Path) -> None:
    project, expected = _project(tmp_path)
    role = project / expected["python_role"]
    role.chmod(0o755)
    _git(project, "add", expected["python_role"])
    _git(project, "commit", "-q", "-m", "executable role")
    with pytest.raises(freeze.HoVerImplementationFreezeError, match="unsafe"):
        freeze.verify_committed_implementation_freeze(
            project, expected_role_paths=expected
        )

    second, _second_expected = _project(tmp_path / "second")
    (second / "alias").symlink_to(second / "roles", target_is_directory=True)
    raw = (second / "roles" / "example.py").read_bytes()
    bindings = {
        "python_role": {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": freeze.git_blob_sha1(raw),
            "relative_path": "alias/example.py",
        }
    }
    manifest = second / freeze.MANIFEST_RELATIVE_PATH
    manifest.write_text(
        json.dumps(_manifest(second, bindings), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(second, "add", "alias", freeze.MANIFEST_RELATIVE_PATH)
    _git(second, "commit", "-q", "-m", "symlink parent")
    with pytest.raises(
        freeze.HoVerImplementationFreezeError, match="symlink component"
    ):
        freeze.verify_committed_implementation_freeze(
            second,
            expected_role_paths={"python_role": "alias/example.py"},
        )
