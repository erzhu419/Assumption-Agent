from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from assumption_agent.benchmarks import hotpot_recursive_acquisition_v1 as acquisition
from assumption_agent.models import stable_hash
from tests.test_hotpot_family_out_v1 import _source_row


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Hotpot Recursive Test",
            "-c",
            "user.email=hotpot-recursive@example.invalid",
            *arguments,
        ],
        check=True,
        capture_output=True,
    )


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (project / "impl.py").write_text("VERSION = 1\n", encoding="utf-8")
    _git(project, "add", ".gitignore", "impl.py")
    _git(project, "commit", "-q", "-m", "initial")
    return project


def _write_hashed(
    path: Path, payload: Mapping[str, Any], hash_field: str, mode: int = 0o644
) -> None:
    acquisition._write_json_exclusive(
        path, payload, hash_field=hash_field, mode=mode
    )


def _prior_pack_and_receipt(
    project: Path,
    source_rows: list[dict[str, Any]],
) -> tuple[Path, Path, frozenset[str]]:
    prior_rows = [
        acquisition.prior._normalize_source_row(row) for row in source_rows[:12]
    ]
    assert all(row is not None for row in prior_rows)
    raw = b"".join(acquisition._canonical_bytes(row) + b"\n" for row in prior_rows)
    pack = project / "artifacts" / "prior" / "pack.jsonl"
    pack.parent.mkdir(parents=True)
    pack.write_bytes(raw)
    os.chmod(pack, 0o600)
    item_set = stable_hash([stable_hash(row) for row in prior_rows])
    receipt_body = {
        "schema": acquisition.prior.ACQUISITION_SCHEMA,
        "decision": "fresh_family_out_pack_formed_measurement_not_authorized",
        "preregistration_sha256": "1" * 64,
        "preregistration_custody": {
            "preregistration_file_sha256": "2" * 64,
            "preregistration_head_blob_sha256": "2" * 64,
            "repository_commit": "3" * 40,
        },
        "source": {
            "file_sha256": acquisition.SOURCE_SHA256,
            "file_size": acquisition.SOURCE_SIZE,
            "hf_repository_commit": acquisition.HF_REPOSITORY_COMMIT,
            "original_CMU_JSON_equivalence_claim": False,
            "row_count": acquisition.SOURCE_ROW_COUNT,
        },
        "counts": {
            "source_rows": acquisition.SOURCE_ROW_COUNT,
            "structurally_valid_rows": len(source_rows),
            "eligible_unique_id_rows": len(source_rows),
            "selected_rows": 12,
        },
        "acquisition_runtime": acquisition.prior.acquisition_runtime_binding(),
        "prospective_ordering": {
            "preregistration_committed_before_source_row_open": True,
            "acquisition_consumed_before_source_row_open": True,
            "source_rows_opened_before_consumption": 0,
            "acquisition_consumption_file_sha256": "4" * 64,
            "acquisition_consumption_sha256": "5" * 64,
            "retry_replay_resample_authorized": False,
        },
        "commitments": {
            "private_pack_file_sha256": acquisition._sha256_bytes(raw),
            "item_commitment_set_sha256": item_set,
            "selection_secret_commitment_sha256": "6" * 64,
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "safety": {
            "model_calls": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
            "measurement_executed": False,
        },
    }
    receipt = {
        **receipt_body,
        "acquisition_sha256": stable_hash(receipt_body),
    }
    receipt_path = project / "prior-acquisition.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", "utf-8")
    return pack, receipt_path, frozenset(row["item_id"] for row in prior_rows)


def _prepared_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, list[dict[str, Any]], frozenset[str]]:
    project = _project(tmp_path)
    rows = [_source_row(index) for index in range(180)]
    source = project / "artifacts" / "source.parquet"
    source.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), source)
    monkeypatch.setattr(acquisition, "SOURCE_SIZE", source.stat().st_size)
    monkeypatch.setattr(acquisition, "SOURCE_SHA256", acquisition._sha256_file(source))
    monkeypatch.setattr(acquisition, "SOURCE_ROW_COUNT", len(rows))
    monkeypatch.setattr(acquisition, "IMPLEMENTATION_RELATIVE_FILES", ("impl.py",))
    monkeypatch.setattr(
        acquisition,
        "RETAINED_P_LINEAGE_RELATIVE_FILES",
        (
            ("P_formation_receipt", "impl.py"),
            ("P_frozen_program", "impl.py"),
            ("M1_pre_run_freeze", "impl.py"),
            ("M1_positive_promotion_report", "impl.py"),
        ),
    )
    pack, receipt, prior_ids = _prior_pack_and_receipt(project, rows)
    return project, source, pack, rows, prior_ids


def test_constants_schemas_and_future_implementation_closure_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert acquisition.BLOCK_COUNTS == {
        "F_Q": 36,
        "M_L4": 24,
        "A_form": 24,
        "A_hold": 24,
        "F_search": 24,
        "M_search": 24,
    }
    assert acquisition.SELECTED_COUNT == 156
    assert acquisition.PRIVATE_BLOCK_ROW_KEYS == {
        "block",
        "item_id",
        "question",
        "corpus",
        "support_indices",
        "source_row_sha256",
    }
    assert "assumption_agent/benchmarks/hotpot_recursive_l4_v1.py" in (
        acquisition.IMPLEMENTATION_RELATIVE_FILES
    )
    assert "assumption_agent/benchmarks/hotpot_evaluator_coevolution_v2.py" in (
        acquisition.IMPLEMENTATION_RELATIVE_FILES
    )

    project = _project(tmp_path)
    monkeypatch.setattr(
        acquisition,
        "IMPLEMENTATION_RELATIVE_FILES",
        ("impl.py", "hotpot_recursive_l4_v1.py"),
    )
    with pytest.raises(acquisition.HotpotRecursiveAcquisitionError, match="missing"):
        acquisition.implementation_binding(project)


def test_preregistration_is_zero_row_access_and_has_no_cmu_equivalence_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _source, _prior_pack, rows, _prior_ids = _prepared_project(
        tmp_path, monkeypatch
    )
    del rows
    prior_receipt = project / "prior-acquisition.json"
    secret = project / "artifacts" / "selection.key"
    acquisition.generate_selection_secret(project=project, output=secret)

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("private rows must not open during preregistration")

    monkeypatch.setattr(acquisition, "_read_prior_private_ids_after_marker", forbidden)
    payload = acquisition.build_preregistration(
        project=project,
        selection_secret_path=secret,
        prior_acquisition_receipt_path=prior_receipt,
    )
    assert payload["safety"]["source_rows_read"] == 0
    assert payload["safety"]["prior_private_pack_rows_read"] == 0
    assert payload["source"]["label_provenance"] == (
        "source_provided_supporting_facts_from_fixed_HF_parquet"
    )
    assert payload["source"][
        "original_CMU_JSON_byte_or_row_equivalence_claim"
    ] is False
    assert payload["source"]["original_CMU_JSON_label_equivalence_claim"] is False
    assert payload["selection"]["block_counts"] == acquisition.BLOCK_COUNTS
    public = json.dumps(payload, sort_keys=True)
    for forbidden_text in (
        "hotpot-000",
        "Which records",
        "Sentence 0-0",
        str(_prior_pack),
        '"item_id"',
        '"question"',
        '"corpus"',
        '"support_indices"',
    ):
        assert forbidden_text not in public


def test_one_shot_excludes_exact_prior_pack_and_forms_six_private_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, source, prior_pack, _rows, prior_ids = _prepared_project(
        tmp_path, monkeypatch
    )
    prior_receipt = project / "prior-acquisition.json"
    secret = project / "artifacts" / "selection.key"
    acquisition.generate_selection_secret(project=project, output=secret)
    preregistration_path = project / "hotpot-recursive.prereg.json"
    preregistration = acquisition.build_preregistration(
        project=project,
        selection_secret_path=secret,
        prior_acquisition_receipt_path=prior_receipt,
    )
    _write_hashed(
        preregistration_path, preregistration, "preregistration_sha256"
    )

    # A self-hashed but uncommitted preregistration cannot consume authority.
    with pytest.raises(Exception, match="clean tracked HEAD blob"):
        acquisition.acquire_private_blocks(
            project=project,
            preregistration_path=preregistration_path,
            selection_secret_path=secret,
            prior_acquisition_receipt_path=prior_receipt,
            prior_private_pack_path=prior_pack,
            source_parquet_path=source,
            private_root=project / "artifacts" / "new-pack",
            private_locator_path=project / "artifacts" / "new-locator.json",
        )
    marker = project / acquisition.ACQUISITION_CONSUMPTION_RELATIVE
    assert not marker.exists()
    _git(project, "add", "hotpot-recursive.prereg.json")
    _git(project, "commit", "-q", "-m", "freeze recursive preregistration")

    events: list[str] = []
    original_prior_reader = acquisition._read_prior_private_ids_after_marker
    original_parquet = pq.ParquetFile

    def prior_reader(**kwargs: Any):
        assert marker.is_file()
        events.append("prior-pack")
        return original_prior_reader(**kwargs)

    def parquet_after_prior(path: Path):
        assert marker.is_file()
        assert events == ["prior-pack"]
        events.append("source")
        return original_parquet(path)

    monkeypatch.setattr(acquisition, "_read_prior_private_ids_after_marker", prior_reader)
    monkeypatch.setattr(pq, "ParquetFile", parquet_after_prior)
    private_root = project / "artifacts" / "new-pack"
    locator = project / "artifacts" / "new-locator.json"
    receipt = acquisition.acquire_private_blocks(
        project=project,
        preregistration_path=preregistration_path,
        selection_secret_path=secret,
        prior_acquisition_receipt_path=prior_receipt,
        prior_private_pack_path=prior_pack,
        source_parquet_path=source,
        private_root=private_root,
        private_locator_path=locator,
    )
    assert events == ["prior-pack", "source"]
    assert receipt["counts"]["selected_rows"] == 156
    assert receipt["counts"]["selected_prior_id_overlap"] == 0
    assert receipt["prior_exclusion"]["excluded_prior_item_count"] == 12
    assert receipt["prospective_ordering"][
        "marker_persisted_before_prior_private_pack_open"
    ] is True
    assert receipt["prospective_ordering"][
        "marker_persisted_before_source_row_open"
    ] is True

    selected_ids: set[str] = set()
    for block, count in acquisition.BLOCK_COUNTS.items():
        rows = [
            json.loads(line)
            for line in (private_root / f"{block}.jsonl").read_text("utf-8").splitlines()
        ]
        assert len(rows) == count
        assert all(set(row) == acquisition.PRIVATE_BLOCK_ROW_KEYS for row in rows)
        assert all(row["block"] == block for row in rows)
        selected_ids.update(row["item_id"] for row in rows)
    assert len(selected_ids) == 156
    assert selected_ids.isdisjoint(prior_ids)

    serialized = json.dumps(receipt, sort_keys=True)
    for forbidden_text in (
        "hotpot-",
        "Which records",
        "Sentence",
        str(private_root),
        str(prior_pack),
        '"item_id"',
        '"question"',
        '"corpus"',
        '"support_indices"',
    ):
        assert forbidden_text not in serialized

    public_receipt = project / "hotpot-recursive.acquisition.json"
    _write_hashed(public_receipt, receipt, "acquisition_sha256")
    loaded, blocks = acquisition.load_acquisition_binding(public_receipt)
    assert loaded["counts"]["selected_rows"] == 156
    assert [row.block for row in blocks] == list(acquisition.BLOCK_ORDER)
    assert [row.count for row in blocks] == [
        acquisition.BLOCK_COUNTS[name] for name in acquisition.BLOCK_ORDER
    ]

    calls = list(events)
    with pytest.raises(FileExistsError, match="already consumed"):
        acquisition.acquire_private_blocks(
            project=project,
            preregistration_path=preregistration_path,
            selection_secret_path=secret,
            prior_acquisition_receipt_path=prior_receipt,
            prior_private_pack_path=prior_pack,
            source_parquet_path=source,
            private_root=project / "artifacts" / "alternate-pack",
            private_locator_path=project / "artifacts" / "alternate-locator.json",
        )
    assert events == calls

    tampered = json.loads(public_receipt.read_text("utf-8"))
    tampered["commitments"]["block_files"][0]["count"] = 35
    body = dict(tampered)
    body.pop("acquisition_sha256")
    tampered["acquisition_sha256"] = stable_hash(body)
    public_receipt.write_text(json.dumps(tampered, sort_keys=True) + "\n", "utf-8")
    with pytest.raises(acquisition.HotpotRecursiveAcquisitionError, match="block"):
        acquisition.load_acquisition_binding(public_receipt)
