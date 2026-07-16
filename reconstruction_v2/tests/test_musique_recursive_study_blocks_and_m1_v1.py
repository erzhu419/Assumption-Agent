from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import threading
from typing import Any

import pytest

from assumption_agent.benchmarks import musique_m1_retrieval_runner_v1 as m1
from assumption_agent.benchmarks import musique_recursive_study_acquisition_v1 as acquisition
from assumption_agent.benchmarks import musique_recursive_study_blocks_v1 as blocks
from assumption_agent.benchmarks.musique_official_core_comparison_v1 import (
    normalize_answer_primary,
)
from assumption_agent.models import stable_hash


PROJECT = Path(__file__).resolve().parents[1]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _row(block: str, index: int) -> dict[str, Any]:
    token = f"signal{index:03d}"
    answers = [token, f"the {token}", f"signal {index:03d}"]
    normalized = list(
        dict.fromkeys(
            value
            for answer in answers
            if (value := normalize_answer_primary(answer))
        )
    )
    return {
        "schema": acquisition.PRIVATE_ROW_SCHEMA,
        "block": block,
        "item_id": f"private-{block}-{index:03d}",
        "question": f"Which private record contains {token}?",
        "corpus": [
            *[
                {
                    "idx": position,
                    "title": f"distractor-{position}",
                    "text": f"Unrelated evidence {position} for row {index}.",
                    "is_supporting": False,
                }
                for position in range(5)
            ],
            {
                "idx": 5,
                "title": "root evidence",
                "text": f"The root for {token} points onward.",
                "is_supporting": True,
            },
            {
                "idx": 6,
                "title": "leaf evidence",
                "text": f"The leaf contains {token}.",
                "is_supporting": True,
            },
        ],
        "answers": answers,
        "normalized_answers": normalized,
        "support_indices": [5, 6],
        "source_row_sha256": stable_hash(
            {"official_source_row": f"{block}-{index}"}
        ),
    }


def _block_bytes(block: str, offset: int) -> bytes:
    return b"".join(
        _canonical(_row(block, offset + index)) + b"\n"
        for index in range(acquisition.BLOCK_COUNT)
    )


def _block_rows(raw: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.splitlines()]


def _study_fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    raw_by_block = {
        block: _block_bytes(block, ordinal * 100)
        for ordinal, block in enumerate(acquisition.BLOCK_ORDER)
    }
    block_files = []
    for block in acquisition.BLOCK_ORDER:
        rows = _block_rows(raw_by_block[block])
        block_files.append(
            {
                "block": block,
                "count": len(rows),
                "file_sha256": m1._sha256_bytes(raw_by_block[block]),
                "item_commitment_set_sha256": stable_hash(
                    [stable_hash(row) for row in rows]
                ),
            }
        )
    receipt_body = {
        "schema": acquisition.ACQUISITION_SCHEMA,
        "decision": "fresh_private_pack_formed_no_formation_or_measurement_authority",
        "source": {
            "repository": "synthetic-official",
            "commit": acquisition.OFFICIAL_SOURCE_COMMIT,
            "dataset": "MuSiQue-Answerable v1.0",
            "source_split": "official_dev",
            "archive_sha256": acquisition.OFFICIAL_ARCHIVE_SHA256,
            "official_dev_member_sha256": "7" * 64,
            "split_disjoint_from_prior_official_train_cohort": True,
        },
        "counts": {
            "source_rows": 120,
            "eligible_rows": 120,
            "selected_rows": acquisition.SELECTED_COUNT,
            "blocks": acquisition.BLOCK_COUNTS,
            "oracle_disagreements": 0,
        },
        "commitments": {
            "private_pack_sha256": stable_hash(block_files),
            "selection_secret_commitment_sha256": "1" * 64,
            "block_files": block_files,
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "ordering": {
            "preregistration_sha256": "2" * 64,
            "all_eight_blocks_formed_together": True,
            "formation_or_measurement_before_pack_complete": False,
            "preregistration_preceded_block_files_local_mtime": True,
            "ordering_evidence_scope": "local_filesystem_only",
        },
        "private_boundary": {
            "source_archive_git_ignored": True,
            "selection_secret_git_ignored": True,
            "private_pack_git_ignored": True,
            "private_locator_git_ignored": True,
            "secret_free_private_locator_formed": True,
            "private_locator_path_persisted_publicly": False,
        },
        "safety": {
            "model_calls": 0,
            "network_calls_during_acquisition": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
            "prior_closed_cohort_accessed": False,
            "measurement_blocks_scored": 0,
        },
    }
    receipt = {
        **receipt_body,
        "acquisition_sha256": stable_hash(receipt_body),
    }
    receipt_path = tmp_path / "acquisition.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt_path, raw_by_block


def _write_block(tmp_path: Path, block: str, raw: bytes) -> Path:
    root = tmp_path / "private-pack"
    root.mkdir(exist_ok=True)
    path = root / f"{block}.jsonl"
    path.write_bytes(raw)
    os.chmod(path, 0o600)
    return path


def _form_f1(
    tmp_path: Path, receipt_path: Path, raw_by_block: dict[str, bytes]
):
    f1 = _write_block(tmp_path, "F1", raw_by_block["F1"])
    output = tmp_path / "f1-formation"
    result = blocks.form_study_typed_retriever(
        block_path=f1,
        acquisition_receipt_path=receipt_path,
        expected_block="F1",
        output_dir=output,
    )
    return result, output


def _runtime_fixture(tmp_path: Path) -> dict[str, Path]:
    runtime = tmp_path / "venv" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"synthetic-python")
    os.chmod(runtime, 0o755)
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    llm.mkdir()
    embedding.mkdir()
    (llm / "asset.bin").write_bytes(b"llm")
    (embedding / "asset.bin").write_bytes(b"embedding")
    base = tmp_path / "base-binding.json"
    attestation = tmp_path / "attestation-v2.json"
    base.write_text('{"synthetic":"base"}\n', encoding="utf-8")
    attestation.write_text('{"synthetic":"attestation"}\n', encoding="utf-8")
    return {
        "runtime": runtime,
        "llm": llm,
        "embedding": embedding,
        "base": base,
        "attestation": attestation,
    }


def _attested() -> dict[str, Any]:
    return {
        "attestation_receipt_sha256": "3" * 64,
        "base_binding_receipt_sha256": "4" * 64,
        "formal_entry_executable_identity_probe_calls": 0,
        "implementation_set_sha256": "5" * 64,
        "runtime_filesystem_binding_sha256": "6" * 64,
        "binding_sha256": "7" * 64,
    }


class _PreparedRuntime:
    @property
    def safe_binding(self) -> dict[str, Any]:
        return _attested()

    def fresh_reverify(self) -> dict[str, Any]:
        return _attested()

    def retrieve(self, **_kwargs: Any) -> tuple[int, ...]:
        raise AssertionError("synthetic test must replace the private official arm")


def _build_freeze(
    *,
    tmp_path: Path,
    receipt_path: Path,
    formation_root: Path,
    runtime: dict[str, Path],
    execution_root: Path,
    freeze_name: str = "m1.freeze.json",
) -> Path:
    freeze_path = tmp_path / freeze_name
    m1.build_m1_pre_run_freeze(
        project_root=PROJECT,
        acquisition_receipt_path=receipt_path,
        formation_receipt_path=formation_root / "formation.receipt.json",
        frozen_program_path=formation_root / "frozen_program.json",
        runtime_python=runtime["runtime"],
        local_llm_model=runtime["llm"],
        local_embedding_model=runtime["embedding"],
        base_binding_receipt_path=runtime["base"],
        attestation_receipt_path=runtime["attestation"],
        execution_root=execution_root,
        authorization_hash=stable_hash(
            {"authorization": freeze_name}
        ),
        output_path=freeze_path,
    )
    return freeze_path


def _execute(
    *,
    receipt_path: Path,
    formation_root: Path,
    runtime: dict[str, Path],
    freeze_path: Path,
    m1_path: Path,
    execution_root: Path,
) -> dict[str, Any]:
    return m1.execute_m1_retrieval_formal(
        project_root=PROJECT,
        pre_run_freeze_path=freeze_path,
        m1_block_path=m1_path,
        acquisition_receipt_path=receipt_path,
        formation_receipt_path=formation_root / "formation.receipt.json",
        frozen_program_path=formation_root / "frozen_program.json",
        runtime_python=runtime["runtime"],
        local_llm_model=runtime["llm"],
        local_embedding_model=runtime["embedding"],
        base_binding_receipt_path=runtime["base"],
        attestation_receipt_path=runtime["attestation"],
        execution_root=execution_root,
    )


def test_exact_block_parser_and_formation_access_are_partitioned(
    tmp_path: Path,
) -> None:
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    f1 = _write_block(tmp_path, "F1", raw_by_block["F1"])
    loaded = blocks.load_formation_block(
        block_path=f1,
        acquisition_receipt_path=receipt_path,
        expected_block="F1",
    )
    assert len(loaded.items) == 12
    assert loaded.block == "F1"
    assert loaded.safe_payload()["raw_content_persisted"] is False

    m1_path = tmp_path / "private-pack" / "M1.jsonl"
    m1_path.write_bytes(raw_by_block["M1"])
    with pytest.raises(blocks.MuSiQueStudyBlockError, match="formation access"):
        blocks.load_formation_block(
            block_path=m1_path,
            acquisition_receipt_path=receipt_path,
            expected_block="M1",
        )
    a_form_path = _write_block(tmp_path, "A_form", raw_by_block["A_form"])
    anchor_formation = blocks.load_formation_block(
        block_path=a_form_path,
        acquisition_receipt_path=receipt_path,
        expected_block="A_form",
    )
    assert len(anchor_formation.items) == 12
    with pytest.raises(blocks.MuSiQueStudyBlockError, match="F1, F2, or F3"):
        blocks.form_study_typed_retriever(
            block_path=a_form_path,
            acquisition_receipt_path=receipt_path,
            expected_block="A_form",
        )


def test_parser_rejects_hash_schema_and_canonical_byte_drift(
    tmp_path: Path,
) -> None:
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    f2 = _write_block(tmp_path, "F2", raw_by_block["F2"])
    raw = f2.read_bytes()
    f2.write_bytes(raw.replace(b'"block":"F2"', b'"block": "F2"', 1))
    with pytest.raises(blocks.MuSiQueStudyBlockError, match="file hash"):
        blocks.load_formation_block(
            block_path=f2,
            acquisition_receipt_path=receipt_path,
            expected_block="F2",
        )

    f2.write_bytes(raw)
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt["counts"]["selected_rows"] = 95
    receipt["acquisition_sha256"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "acquisition_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(blocks.MuSiQueStudyBlockError, match="count|ordering"):
        blocks.load_formation_block(
            block_path=f2,
            acquisition_receipt_path=receipt_path,
            expected_block="F2",
        )


def test_F1_typed_formation_reuses_finite_DSL_and_is_public_safe(
    tmp_path: Path,
) -> None:
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    result, output = _form_f1(tmp_path, receipt_path, raw_by_block)
    assert not result.program.type_issues()
    assert result.receipt["offline_contract"] == {
        "model_calls": 0,
        "network_calls": 0,
        "generator_calls": 0,
        "online_evaluator_calls": 0,
        "measurement_block_accessed": False,
    }
    loaded, receipt, envelope = blocks.load_study_frozen_program(
        frozen_program_path=output / "frozen_program.json",
        formation_receipt_path=output / "formation.receipt.json",
        verify_live=True,
        implementation_root=PROJECT,
    )
    assert loaded.program_hash == result.program.program_hash
    assert receipt["receipt_hash"] == result.receipt["receipt_hash"]
    assert envelope["program_hash"] == loaded.program_hash
    public = (output / "formation.receipt.json").read_text("utf-8") + (
        output / "frozen_program.json"
    ).read_text("utf-8")
    for private_text in (
        "private-F1-",
        "Which private record",
        "root evidence",
        "support_indices",
        "normalized_answers",
    ):
        assert private_text not in public


def test_m1_synthetic_core_is_36_way_gold_free_and_hash_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    _formation, formation_root = _form_f1(tmp_path, receipt_path, raw_by_block)
    runtime = _runtime_fixture(tmp_path)
    monkeypatch.setattr(
        m1, "prepare_formal_runtime_v2", lambda **_kwargs: _PreparedRuntime()
    )
    monkeypatch.setattr(m1, "_CLEAN_MODULE_CLI_ACTIVE", True)
    execution_root = tmp_path / "m1-formal-root"

    # M1 deliberately does not exist while the complete self-hashed freeze is
    # built; the builder has no M1 path parameter.
    freeze_path = _build_freeze(
        tmp_path=tmp_path,
        receipt_path=receipt_path,
        formation_root=formation_root,
        runtime=runtime,
        execution_root=execution_root,
    )
    assert not (tmp_path / "private-pack" / "M1.jsonl").exists()
    freeze = json.loads(freeze_path.read_text("utf-8"))
    declared = freeze.pop("freeze_hash")
    assert stable_hash(freeze) == declared
    assert freeze["ordering"]["measurement_block_rows_read_while_freezing"] == 0

    frozen_bytes = freeze_path.read_bytes()
    tampered = json.loads(frozen_bytes)
    tampered["execution_contract"]["maximum_concurrency"] = 35
    freeze_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(m1.MuSiQueM1RunnerError, match="freeze drifted"):
        _execute(
            receipt_path=receipt_path,
            formation_root=formation_root,
            runtime=runtime,
            freeze_path=freeze_path,
            m1_path=tmp_path / "private-pack" / "M1.jsonl",
            execution_root=execution_root,
        )
    assert execution_root.exists() is False
    freeze_path.write_bytes(frozen_bytes)

    m1_path = _write_block(tmp_path, "M1", raw_by_block["M1"])
    barrier = threading.Barrier(m1.WORK_UNIT_COUNT)
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def record(arm: str, item: blocks.RetrievalStudyItem) -> None:
        assert not hasattr(item, "support_indices")
        assert not hasattr(item, "answers")
        with lock:
            calls.append((arm, item.item_id))
        barrier.wait(timeout=15)

    def raw(item: blocks.RetrievalStudyItem):
        record("raw", item)
        return (0, 1, 2, 3, 4)

    def typed(_program, item: blocks.RetrievalStudyItem):
        record("typed", item)
        return (5, 6, 0, 1, 2)

    def hippo(item: blocks.RetrievalStudyItem, _runtime, _root):
        record("hippo", item)
        return (5, 0, 1, 2, 3)

    monkeypatch.setattr(m1, "_canonical_raw_retrieve", raw)
    monkeypatch.setattr(m1, "_typed_program_retrieve", typed)
    monkeypatch.setattr(m1, "_official_retrieve", hippo)
    report = _execute(
        receipt_path=receipt_path,
        formation_root=formation_root,
        runtime=runtime,
        freeze_path=freeze_path,
        m1_path=m1_path,
        execution_root=execution_root,
    )
    assert len(calls) == m1.WORK_UNIT_COUNT == 36
    assert report["valid"] is True
    assert report["execution"]["configured_maximum_concurrency"] == 36
    assert report["execution"]["retrieval_call_count"] == 36
    assert report["execution"][
        "all_terminals_joined_before_support_scoring"
    ] is True
    arms = report["measurement"]["arm_metrics"]
    assert arms["canonical_RAW"]["support_hit_count"] == 0
    assert arms["frozen_P"]["support_hit_count"] == 24
    assert arms["official_HippoRAG"]["support_hit_count"] == 12
    disposition = report["measurement"]["promotion_disposition"]
    assert disposition["policy"] == m1.PROMOTION_POLICY
    assert disposition["positive_net_support"] is True
    assert disposition["disposition"] == "promote_P_to_retained_generation_one"
    assert disposition["archive_mutated_by_runner"] is False
    assert report["execution"]["generator_calls"] == 0
    assert report["execution"]["external_network_calls"] == 0
    assert report["execution"]["online_evaluator_calls"] == 0
    assert report["execution"]["retries"] == 0
    assert report["execution"]["replays"] == 0
    assert report["execution"]["resamples"] == 0

    persisted = "\n".join(
        path.read_text("utf-8")
        for path in (freeze_path, *(sorted(execution_root.iterdir())))
    )
    for private_text in (
        "private-M1-",
        "Which private record",
        "root evidence",
        "support_indices",
        "signal100",
    ):
        assert private_text not in persisted

    call_count = len(calls)
    with pytest.raises(m1.MuSiQueM1RunnerError, match="replay is forbidden"):
        _execute(
            receipt_path=receipt_path,
            formation_root=formation_root,
            runtime=runtime,
            freeze_path=freeze_path,
            m1_path=m1_path,
            execution_root=execution_root,
        )
    assert len(calls) == call_count


def test_m1_failure_is_terminal_and_formal_signature_has_no_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    _formation, formation_root = _form_f1(tmp_path, receipt_path, raw_by_block)
    runtime = _runtime_fixture(tmp_path)
    monkeypatch.setattr(
        m1, "prepare_formal_runtime_v2", lambda **_kwargs: _PreparedRuntime()
    )
    monkeypatch.setattr(m1, "_CLEAN_MODULE_CLI_ACTIVE", True)
    m1_path = _write_block(tmp_path, "M1", raw_by_block["M1"])
    execution_root = tmp_path / "m1-failed-root"
    freeze_path = _build_freeze(
        tmp_path=tmp_path,
        receipt_path=receipt_path,
        formation_root=formation_root,
        runtime=runtime,
        execution_root=execution_root,
        freeze_name="m1.failed.freeze.json",
    )
    calls = 0
    lock = threading.Lock()

    def fail(_item, _runtime, _root):
        nonlocal calls
        with lock:
            calls += 1
        raise RuntimeError("private official failure text")

    monkeypatch.setattr(m1, "_official_retrieve", fail)
    with pytest.raises(m1.MuSiQueM1RunnerError, match="cannot be replayed"):
        _execute(
            receipt_path=receipt_path,
            formation_root=formation_root,
            runtime=runtime,
            freeze_path=freeze_path,
            m1_path=m1_path,
            execution_root=execution_root,
        )
    failure_path = execution_root / m1.FAILURE_FILENAME
    failure = json.loads(failure_path.read_text("utf-8"))
    declared = failure.pop("failure_hash")
    assert stable_hash(failure) == declared
    assert failure["authorization_consumed"] is True
    assert failure["retries"] == failure["replays"] == failure["resamples"] == 0
    assert failure["replay_authorized"] is False
    text = failure_path.read_text("utf-8")
    assert "private official failure text" not in text
    assert "private-M1" not in text

    prior = calls
    with pytest.raises(m1.MuSiQueM1RunnerError, match="replay is forbidden"):
        _execute(
            receipt_path=receipt_path,
            formation_root=formation_root,
            runtime=runtime,
            freeze_path=freeze_path,
            m1_path=m1_path,
            execution_root=execution_root,
        )
    assert calls == prior

    parameters = inspect.signature(m1.execute_m1_retrieval_formal).parameters
    forbidden = {
        "operator_factory",
        "raw_results",
        "typed_results",
        "hipporag_results",
        "result_injection",
        "retriever",
        "runner",
    }
    assert forbidden.isdisjoint(parameters)
