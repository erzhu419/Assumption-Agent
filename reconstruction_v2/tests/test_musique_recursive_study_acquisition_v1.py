from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator
import zipfile

import pytest

from assumption_agent.benchmarks import musique_recursive_study_acquisition_v1 as study
from assumption_agent.benchmarks.musique_official_core_comparison_v1 import (
    generate_selection_secret,
)
from assumption_agent.models import stable_hash


@pytest.fixture
def private_tmp_path() -> Iterator[Path]:
    parent = Path(__file__).resolve().parents[1] / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recursive-acquisition-", dir=parent) as root:
        yield Path(root)


def _row(index: int) -> dict:
    token = f"signal{index:03d}"
    return {
        "id": f"official-dev-{index:03d}",
        "question": f"Which signal belongs to record {index}?",
        "answer": token,
        "answer_aliases": [f"the {token}", f"signal {index:03d}"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "root",
                "paragraph_text": f"Root evidence for {index}.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "leaf",
                "paragraph_text": f"Leaf evidence contains {token}.",
                "is_supporting": True,
            },
            *[
                {
                    "idx": position,
                    "title": f"distractor-{position}",
                    "paragraph_text": f"Distractor {position} for {index}.",
                    "is_supporting": False,
                }
                for position in range(2, 5)
            ],
        ],
    }


def _initialize_project(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (path / "impl.py").write_text("VERSION = 1\n", encoding="utf-8")


def test_fresh_pack_forms_all_eight_blocks_without_public_content(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = private_tmp_path / "project"
    _initialize_project(project)
    artifacts = project / "artifacts"
    artifacts.mkdir()
    secret_path = artifacts / "selection.key"
    generate_selection_secret(project=project, output=secret_path)
    archive_path = artifacts / "musique.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            f"data/{study.OFFICIAL_DEV_MEMBER_BASENAME}",
            "\n".join(json.dumps(_row(index)) for index in range(120)) + "\n",
        )
    monkeypatch.setattr(study, "IMPLEMENTATION_RELATIVE_FILES", ("impl.py",))
    monkeypatch.setattr(study, "OFFICIAL_ARCHIVE_SHA256", study._sha256_file(archive_path))
    monkeypatch.setattr(
        study,
        "_runtime_attestation_binding",
        lambda _project: {
            "relative_path": study.RUNTIME_ATTESTATION_RELATIVE_PATH,
            "file_sha256": "3" * 64,
            "receipt_sha256": "4" * 64,
            "implementation_set_sha256": "5" * 64,
            "formal_entry_policy_sha256": "6" * 64,
            "runtime_filesystem_binding_sha256": "7" * 64,
        },
    )
    official_repository = project / "official-repository"
    official_repository.mkdir()
    monkeypatch.setattr(
        study,
        "official_source_receipt",
        lambda _path: {
            "repository": "official",
            "commit": study.OFFICIAL_SOURCE_COMMIT,
            "license": {"spdx": "CC-BY-4.0"},
        },
    )
    prereg = study.build_preregistration(
        project=project,
        official_repository=official_repository,
        selection_secret_path=secret_path,
    )
    prereg_path = project / "prereg.json"
    study._write_json_exclusive(
        prereg_path,
        prereg,
        hash_field="preregistration_sha256",
    )
    private_root = artifacts / "fresh-pack"
    private_locator = artifacts / "custody" / "locator.json"
    receipt = study.acquire_private_pack(
        project=project,
        preregistration_path=prereg_path,
        official_repository=official_repository,
        source_archive=archive_path,
        private_root=private_root,
        private_locator_path=private_locator,
        selection_secret_path=secret_path,
    )
    assert receipt["counts"]["selected_rows"] == 96
    assert receipt["counts"]["blocks"] == study.BLOCK_COUNTS
    assert sorted(path.stem for path in private_root.glob("*.jsonl")) == sorted(
        study.BLOCK_ORDER
    )
    assert private_locator.stat().st_mode & 0o077 == 0
    locator = json.loads(private_locator.read_text(encoding="utf-8"))
    assert locator["selection_secret_included"] is False
    serialized = json.dumps(receipt, sort_keys=True)
    assert "official-dev-" not in serialized
    assert "Which signal" not in serialized
    assert str(private_root) not in serialized
    assert receipt["safety"]["scores_computed"] == 0


def test_public_writer_recomputes_one_canonical_self_hash(
    private_tmp_path: Path,
) -> None:
    output = private_tmp_path / "manifest.json"
    study._write_json_exclusive(
        output,
        {"value": 1, "receipt_sha256": "stale"},
        hash_field="receipt_sha256",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    declared = payload.pop("receipt_sha256")
    assert declared == stable_hash(payload)
