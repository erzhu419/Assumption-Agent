from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator
import zipfile

import pytest

from assumption_agent.benchmarks.musique_official_core_comparison_v1 import (
    _normalize_source_row,
    _oracle_conformance_receipt,
    _selection_key,
    acquire_private_pack,
    evaluate_aliases_primary,
    evaluate_aliases_secondary,
    evaluate_support_primary,
    evaluate_support_secondary,
    generate_selection_secret,
    normalize_answer_primary,
    normalize_answer_secondary,
)


@pytest.fixture
def private_tmp_path() -> Iterator[Path]:
    native_private_parent = Path(__file__).resolve().parents[1] / "artifacts"
    native_private_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="musique-test-", dir=native_private_parent) as directory:
        yield Path(directory)


def _row(item_id: str) -> dict:
    token = item_id.replace("-", "")
    return {
        "id": item_id,
        "question": f"Which signal belongs to {item_id}?",
        "answer": f"Signal-{token}",
        "answer_aliases": [f"signal {token}", f"The Signal {token}"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "root",
                "paragraph_text": "Synthetic root paragraph.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "leaf",
                "paragraph_text": f"Synthetic leaf contains Signal {token}.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "distractor-1",
                "paragraph_text": "Synthetic distractor paragraph one.",
                "is_supporting": False,
            },
            {
                "idx": 3,
                "title": "distractor-2",
                "paragraph_text": "Synthetic distractor paragraph two.",
                "is_supporting": False,
            },
            {
                "idx": 4,
                "title": "distractor-3",
                "paragraph_text": "Synthetic distractor paragraph three.",
                "is_supporting": False,
            },
        ],
    }


def test_independent_answer_oracles_agree() -> None:
    cases = [
        ("The Alpha", ["alpha", "the alpha"]),
        ("A-B", ["a b", "ab"]),
        ("two two", ["two", "two two"]),
        ("", ["", "none"]),
    ]
    for prediction, aliases in cases:
        assert normalize_answer_primary(prediction) == normalize_answer_secondary(prediction)
        assert evaluate_aliases_primary(prediction, aliases) == evaluate_aliases_secondary(
            prediction, aliases
        )
    receipt = _oracle_conformance_receipt()
    assert receipt["oracle_disagreement_count"] == 0
    assert len(receipt["conformance_sha256"]) == 64


def test_independent_support_oracles_agree() -> None:
    assert evaluate_support_primary([0, 2], [0, 1, 2]) == Fraction(2, 3)
    assert evaluate_support_secondary([0, 2], [0, 1, 2]) == Fraction(2, 3)
    assert evaluate_support_primary([], []) == evaluate_support_secondary([], [])


def test_source_row_requires_local_support_and_distinct_aliases() -> None:
    accepted = _normalize_source_row(_row("item-1"))
    assert accepted is not None
    assert len(accepted["corpus"]) == 5
    assert [paragraph["idx"] for paragraph in accepted["corpus"]] == list(range(5))
    assert len(accepted["normalized_answers"]) == 2

    no_alias = _row("item-2")
    no_alias["answer_aliases"] = [no_alias["answer"]]
    assert _normalize_source_row(no_alias) is None

    one_support = _row("item-3")
    one_support["paragraphs"][1]["is_supporting"] = False
    assert _normalize_source_row(one_support) is None

    empty_alias = _row("item-4")
    empty_alias["answer_aliases"] = [empty_alias["answer"], "the"]
    assert _normalize_source_row(empty_alias) is None

    non_contiguous = _row("item-5")
    non_contiguous["paragraphs"][-1]["idx"] = 7
    assert _normalize_source_row(non_contiguous) is None


def test_selection_key_is_fixed_and_content_independent() -> None:
    secret_a = b"a" * 32
    secret_b = b"b" * 32
    assert _selection_key("item-1", secret_a) == _selection_key("item-1", secret_a)
    assert _selection_key("item-1", secret_a) != _selection_key("item-2", secret_a)
    assert _selection_key("item-1", secret_a) != _selection_key("item-1", secret_b)
    assert len(_selection_key("item-1", secret_a)) == 64


def _initialize_ignored_test_project(project: Path) -> None:
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / ".gitignore").write_text("artifacts/\n", encoding="utf-8")


def test_private_pack_formation_uses_12_6_6_and_public_receipt_has_no_ids(
    private_tmp_path: Path,
) -> None:
    tmp_path = private_tmp_path
    project = tmp_path / "project"
    _initialize_ignored_test_project(project)
    module = project / "assumption_agent/benchmarks/musique_official_core_comparison_v1.py"
    test_file = project / "tests/test_musique_official_core_comparison_v1.py"
    module.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    module.write_text("module", encoding="utf-8")
    test_file.write_text("test", encoding="utf-8")
    artifacts = project / "artifacts" / "qualification"
    artifacts.mkdir(parents=True)
    secret_path = artifacts / "selection.key"
    secret_commitment = generate_selection_secret(project=project, output=secret_path)
    prereg = {
        "schema": "musique-official-core-comparison-v1-preregistration",
        "source": {
            "repository": "https://github.com/StonyBrookNLP/musique.git",
            "commit": "922ac98f19a201998dbdae6d7f2887a5258dbdeb",
            "license": {"spdx": "CC-BY-4.0"},
        },
        "selection": {"selection_secret_commitment_sha256": secret_commitment},
        "oracles": {"synthetic_conformance": _oracle_conformance_receipt()},
    }
    from assumption_agent.benchmarks import musique_official_core_comparison_v1 as module_under_test

    prereg["implementation"] = module_under_test._implementation_binding(project)
    prereg["preregistration_sha256"] = module_under_test.stable_hash(prereg)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    archive_path = artifacts / "source.zip"
    source_lines = "\n".join(json.dumps(_row(f"item-{index:02d}")) for index in range(30))
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/musique_ans_v1.0_train.jsonl", source_lines + "\n")
    deliberately_old = prereg_path.stat().st_mtime_ns - 1_000_000_000
    os.utime(archive_path, ns=(deliberately_old, deliberately_old))

    private_root = artifacts / "private-pack"
    receipt = acquire_private_pack(
        project=project,
        preregistration_path=prereg_path,
        source_archive=archive_path,
        private_root=private_root,
        selection_secret_path=secret_path,
    )
    assert receipt["counts"]["splits"] == {
        "train": 12,
        "development": 6,
        "residual_sealed": 6,
    }
    assert receipt["counts"]["selected_rows"] == 24
    serialized = json.dumps(receipt, sort_keys=True)
    assert "item-" not in serialized
    assert "Which signal" not in serialized
    assert str(artifacts) not in serialized
    assert receipt["source"]["claim_scope"] == (
        "multi_alias_eligible_subset_of_official_train_not_full_musique"
    )
    assert receipt["ordering"]["evidence_scope"] == (
        "local_filesystem_only_not_source_provenance"
    )
    assert receipt["ordering"]["archive_acquisition_order_claimed_from_mtime"] is False
    assert len(list(private_root.glob("*.jsonl"))) == 3


def test_private_inputs_fail_closed_outside_ignored_boundary(private_tmp_path: Path) -> None:
    tmp_path = private_tmp_path
    project = tmp_path / "project"
    _initialize_ignored_test_project(project)
    public_secret = project / "selection.key"
    with pytest.raises(PermissionError, match="artifacts"):
        generate_selection_secret(project=project, output=public_secret)
    traversal_secret = project / "artifacts" / ".." / "selection.key"
    with pytest.raises(PermissionError, match="resolves outside"):
        generate_selection_secret(project=project, output=traversal_secret)


def test_selection_secret_creation_is_exclusive_and_private(private_tmp_path: Path) -> None:
    tmp_path = private_tmp_path
    project = tmp_path / "project"
    _initialize_ignored_test_project(project)
    secret_path = project / "artifacts" / "selection.key"
    commitment = generate_selection_secret(project=project, output=secret_path)
    assert len(commitment) == 64
    assert len(secret_path.read_bytes()) == 32
    assert secret_path.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        generate_selection_secret(project=project, output=secret_path)
