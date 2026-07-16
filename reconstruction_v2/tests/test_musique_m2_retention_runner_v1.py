from __future__ import annotations

import inspect
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from assumption_agent.benchmarks import musique_m1_retrieval_runner_v1 as m1
from assumption_agent.benchmarks import musique_m2_retention_runner_v1 as m2
from assumption_agent.benchmarks import musique_recursive_study_blocks_v1 as blocks
from assumption_agent.models import stable_hash
from tests.test_musique_recursive_study_blocks_and_m1_v1 import (
    _PreparedRuntime as _M1PreparedRuntime,
    _build_freeze as _build_m1_freeze,
    _form_f1,
    _runtime_fixture,
    _study_fixture,
    _write_block,
)


PROJECT = Path(__file__).resolve().parents[1]


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
    def __init__(self, events: list[str], *, fail_postflight: bool = False):
        self.events = events
        self.fail_postflight = fail_postflight

    @property
    def safe_binding(self) -> dict[str, Any]:
        return _attested()

    def fresh_reverify(self) -> dict[str, Any]:
        self.events.append("postflight")
        if self.fail_postflight:
            raise RuntimeError("private postflight failure")
        return _attested()

    def retrieve(self, **_kwargs: Any) -> tuple[int, ...]:
        raise AssertionError("synthetic test replaces the official arm")


def _form_f2(
    tmp_path: Path, receipt_path: Path, raw_by_block: dict[str, bytes]
) -> Path:
    f2 = _write_block(tmp_path, "F2", raw_by_block["F2"])
    output = tmp_path / "f2-formation"
    blocks.form_study_typed_retriever(
        block_path=f2,
        acquisition_receipt_path=receipt_path,
        expected_block="F2",
        output_dir=output,
    )
    return output


def _write_positive_m1_report(
    *,
    tmp_path: Path,
    receipt_path: Path,
    m1_freeze_path: Path,
) -> Path:
    acquisition = blocks.load_study_acquisition_binding(receipt_path)
    commitment = acquisition.commitment_for("M1")
    freeze = json.loads(m1_freeze_path.read_text("utf-8"))
    measurement = {
        "primary_metric": "official_support_recall_at_5",
        "arm_metrics": {
            "canonical_RAW": {
                "support_hit_count": 0,
                "support_total": 24,
            },
            "frozen_P": {
                "support_hit_count": 1,
                "support_total": 24,
            },
            "official_HippoRAG": {
                "support_hit_count": 0,
                "support_total": 24,
            },
        },
        "paired_P_minus_RAW": {
            "net_support_hit_count": 1,
            "support_recall_delta": 1 / 24,
            "gain_item_count": 1,
            "harm_item_count": 0,
            "tie_item_count": 11,
        },
        "promotion_disposition": {
            "policy": m2.M1_PROMOTION_POLICY,
            "positive_net_support": True,
            "disposition": m2.M1_POSITIVE_DISPOSITION,
            "archive_mutated_by_runner": False,
        },
        "score_closure_hash": stable_hash({"synthetic": "M1"}),
        "item_level_rows_persisted": False,
        "raw_content_persisted": False,
    }
    body = {
        "schema": m2.M1_REPORT_SCHEMA,
        "valid": True,
        "freeze_hash": freeze["freeze_hash"],
        "freeze_file_sha256": m2._sha256_file(m1_freeze_path),
        "measurement_block_id_hash": stable_hash({"block": "M1"}),
        "measurement_block_file_sha256": commitment.file_sha256,
        "measurement": measurement,
        "execution": {
            "arm_ids": list(m1.ARM_IDS),
            "item_count": m1.M1_ITEM_COUNT,
            "work_unit_count": m1.WORK_UNIT_COUNT,
            "retrieval_call_count": m1.WORK_UNIT_COUNT,
            "retrieval_terminal_count": m1.WORK_UNIT_COUNT,
            "configured_maximum_concurrency": m1.MAXIMUM_CONCURRENCY,
            "all_terminals_joined_before_support_scoring": True,
            "ranking_receipt_set_hash": stable_hash({"synthetic": "rankings"}),
            "generator_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "runtime": {
            "attestation_receipt_sha256": "3" * 64,
            "formal_entry_executable_identity_probe_calls": 0,
            "official_arm_terminal_count": m1.M1_ITEM_COUNT,
            "worker_process_count_inferred_from_arm_count": False,
            "postflight_fresh_filesystem_attestation": True,
            "postflight_binding_sha256": "7" * 64,
        },
        "sealed_or_test_content_accessed": False,
        "raw_content_persisted": False,
    }
    report = {**body, "report_hash": stable_hash(body)}
    path = tmp_path / "m1.aggregate.report.json"
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_postflight: bool = False,
) -> dict[str, Any]:
    receipt_path, raw_by_block = _study_fixture(tmp_path)
    _p_formation, p_root = _form_f1(tmp_path, receipt_path, raw_by_block)
    q_root = _form_f2(tmp_path, receipt_path, raw_by_block)
    runtime = _runtime_fixture(tmp_path)
    monkeypatch.setattr(
        m1, "prepare_formal_runtime_v2", lambda **_kwargs: _M1PreparedRuntime()
    )
    m1_freeze_path = _build_m1_freeze(
        tmp_path=tmp_path,
        receipt_path=receipt_path,
        formation_root=p_root,
        runtime=runtime,
        execution_root=tmp_path / "unused-m1-root",
    )
    m1_report_path = _write_positive_m1_report(
        tmp_path=tmp_path,
        receipt_path=receipt_path,
        m1_freeze_path=m1_freeze_path,
    )
    events: list[str] = []

    def prepare(**_kwargs: Any) -> _PreparedRuntime:
        events.append("preflight")
        return _PreparedRuntime(events, fail_postflight=fail_postflight)

    monkeypatch.setattr(m2, "prepare_formal_runtime_v2", prepare)
    return {
        "receipt": receipt_path,
        "raw": raw_by_block,
        "p_root": p_root,
        "q_root": q_root,
        "runtime": runtime,
        "m1_freeze": m1_freeze_path,
        "m1_report": m1_report_path,
        "events": events,
    }


def _common(
    *, fixture: dict[str, Any], execution_root: Path
) -> dict[str, Any]:
    runtime = fixture["runtime"]
    return {
        "project_root": PROJECT,
        "acquisition_receipt_path": fixture["receipt"],
        "p_formation_receipt_path": fixture["p_root"]
        / "formation.receipt.json",
        "p_frozen_program_path": fixture["p_root"] / "frozen_program.json",
        "q_formation_receipt_path": fixture["q_root"]
        / "formation.receipt.json",
        "q_frozen_program_path": fixture["q_root"] / "frozen_program.json",
        "m1_pre_run_freeze_path": fixture["m1_freeze"],
        "m1_promotion_report_path": fixture["m1_report"],
        "runtime_python": runtime["runtime"],
        "local_llm_model": runtime["llm"],
        "local_embedding_model": runtime["embedding"],
        "base_binding_receipt_path": runtime["base"],
        "attestation_receipt_path": runtime["attestation"],
        "execution_root": execution_root,
    }


def _build_m2_freeze(
    *,
    tmp_path: Path,
    fixture: dict[str, Any],
    execution_root: Path,
    name: str = "m2.freeze.json",
) -> Path:
    path = tmp_path / name
    m2.build_m2_pre_run_freeze(
        **_common(fixture=fixture, execution_root=execution_root),
        authorization_hash=stable_hash({"authorization": name}),
        output_path=path,
    )
    return path


def _execute(
    *,
    fixture: dict[str, Any],
    execution_root: Path,
    freeze_path: Path,
    m2_path: Path,
) -> dict[str, Any]:
    return m2.execute_m2_retention_formal(
        **_common(fixture=fixture, execution_root=execution_root),
        pre_run_freeze_path=freeze_path,
        m2_block_path=m2_path,
    )


def test_m2_is_36_way_gold_free_postflight_then_l4_scored_and_no_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path, monkeypatch)
    execution_root = tmp_path / "m2-formal-root"
    freeze_path = _build_m2_freeze(
        tmp_path=tmp_path,
        fixture=fixture,
        execution_root=execution_root,
    )

    # Freeze has no M2 input parameter and M2 does not exist during freeze.
    assert "m2_block_path" not in inspect.signature(
        m2.build_m2_pre_run_freeze
    ).parameters
    assert not (tmp_path / "private-pack" / "M2.jsonl").exists()
    freeze = json.loads(freeze_path.read_text("utf-8"))
    declared = freeze.pop("freeze_hash")
    assert stable_hash(freeze) == declared
    assert freeze["ordering"] == {
        "measurement_block_rows_read_while_freezing": 0,
        "measurement_support_labels_read_while_freezing": 0,
        "pre_run_freeze_complete_before_measurement_open": True,
    }
    assert freeze["positive_m1_promotion_binding"]["disposition"] == (
        m2.M1_POSITIVE_DISPOSITION
    )

    frozen_bytes = freeze_path.read_bytes()
    tampered = json.loads(frozen_bytes)
    tampered["execution_contract"]["maximum_concurrency"] = 35
    freeze_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(m2, "_CLEAN_MODULE_CLI_ACTIVE", True)
    with pytest.raises(m2.MuSiQueM2RunnerError, match="self-hash|drifted"):
        _execute(
            fixture=fixture,
            execution_root=execution_root,
            freeze_path=freeze_path,
            m2_path=tmp_path / "private-pack" / "M2.jsonl",
        )
    assert execution_root.exists() is False
    freeze_path.write_bytes(frozen_bytes)

    m2_path = _write_block(tmp_path, "M2", fixture["raw"]["M2"])
    barrier = threading.Barrier(m2.WORK_UNIT_COUNT)
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def record(component: str, item: blocks.RetrievalStudyItem) -> None:
        assert not hasattr(item, "support_indices")
        assert not hasattr(item, "answers")
        assert not hasattr(item, "normalized_answers")
        with lock:
            calls.append((component, item.item_id))
            fixture["events"].append("retrieve")
        barrier.wait(timeout=15)

    def p(_program: Any, item: blocks.RetrievalStudyItem) -> tuple[int, ...]:
        record("P", item)
        return (0, 1, 2, 5, 6)

    def q(_program: Any, item: blocks.RetrievalStudyItem) -> tuple[int, ...]:
        record("Q", item)
        return (0, 1, 2, 6, 5)

    def hippo(
        item: blocks.RetrievalStudyItem, _runtime: Any, _root: Path
    ) -> tuple[int, ...]:
        record("official", item)
        return (5, 0, 1, 2, 3)

    monkeypatch.setattr(m2, "_p_retrieve", p)
    monkeypatch.setattr(m2, "_q_retrieve", q)
    monkeypatch.setattr(m2, "_official_retrieve", hippo)
    score = m2._score_measurement

    def score_after_postflight(**kwargs: Any) -> dict[str, Any]:
        assert fixture["events"].count("retrieve") == m2.WORK_UNIT_COUNT
        assert fixture["events"][-1] == "postflight"
        fixture["events"].append("score")
        return score(**kwargs)

    monkeypatch.setattr(m2, "_score_measurement", score_after_postflight)
    report = _execute(
        fixture=fixture,
        execution_root=execution_root,
        freeze_path=freeze_path,
        m2_path=m2_path,
    )
    assert len(calls) == m2.WORK_UNIT_COUNT == 36
    assert report["execution"]["retrieval_call_count"] == 36
    assert report["execution"]["retrieval_terminal_count"] == 36
    assert report["execution"]["canonical_raw_derivation_count"] == 12
    assert report["execution"]["configured_maximum_concurrency"] == 36
    assert report["runtime"]["postflight_fresh_filesystem_attestation"] is True
    assert fixture["events"][-2:] == ["postflight", "score"]

    measurement = report["measurement"]
    l4 = measurement["l4_recursive_retention"]
    assert tuple(l4["arm_metrics"]) == m2.L4_ARM_IDS
    assert l4["arm_metrics"]["empty"]["support_hit_count"] == 0
    assert l4["arm_metrics"]["P"]["support_hit_count"] == 12
    assert l4["arm_metrics"]["Q"]["support_hit_count"] == 12
    assert l4["arm_metrics"]["P_plus_Q"]["support_hit_count"] == 24
    assert l4["retention"]["delta"] == 0.5
    assert l4["novelty"]["net_delta"] == 0.5
    assert l4["forgetting"]["forgotten_support_count"] == 0
    direct = measurement["homologous_direct_top5_comparison"]["arm_metrics"]
    assert direct["canonical_RAW"]["support_hit_count"] == 0
    assert direct["recursive_typed_retrieval"]["support_hit_count"] == 24
    assert direct["official_HippoRAG"]["support_hit_count"] == 12
    assert direct["recursive_typed_retrieval"]["support_hit_count"] == (
        l4["arm_metrics"]["P_plus_Q"]["support_hit_count"]
    )
    assert measurement["homologous_direct_top5_comparison"][
        "recursive_typed_retrieval_is_exact_l4_P_plus_Q_ranking"
    ] is True

    persisted = "\n".join(
        path.read_text("utf-8")
        for path in (freeze_path, *sorted(execution_root.iterdir()))
    )
    for private_text in (
        "private-M2-",
        "Which private record",
        "root evidence",
        "support_indices",
        "signal300",
    ):
        assert private_text not in persisted

    prior_calls = len(calls)
    with pytest.raises(m2.MuSiQueM2RunnerError, match="replay is forbidden"):
        _execute(
            fixture=fixture,
            execution_root=execution_root,
            freeze_path=freeze_path,
            m2_path=m2_path,
        )
    assert len(calls) == prior_calls

    forbidden = {
        "operator_factory",
        "raw_results",
        "p_results",
        "q_results",
        "hipporag_results",
        "result_injection",
        "retriever",
        "runner",
        "callback",
    }
    assert forbidden.isdisjoint(
        inspect.signature(m2.execute_m2_retention_formal).parameters
    )


def test_m2_postflight_failure_is_terminal_unscored_and_not_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path, monkeypatch, fail_postflight=True)
    execution_root = tmp_path / "m2-postflight-failure-root"
    freeze_path = _build_m2_freeze(
        tmp_path=tmp_path,
        fixture=fixture,
        execution_root=execution_root,
        name="m2.postflight.failure.freeze.json",
    )
    m2_path = _write_block(tmp_path, "M2", fixture["raw"]["M2"])
    calls = 0
    lock = threading.Lock()

    def retrieve(
        _program: Any, _item: blocks.RetrievalStudyItem
    ) -> tuple[int, ...]:
        nonlocal calls
        with lock:
            calls += 1
        return (0, 1, 2, 3, 4)

    def official(
        _item: blocks.RetrievalStudyItem, _runtime: Any, _root: Path
    ) -> tuple[int, ...]:
        nonlocal calls
        with lock:
            calls += 1
        return (0, 1, 2, 3, 4)

    monkeypatch.setattr(m2, "_p_retrieve", retrieve)
    monkeypatch.setattr(m2, "_q_retrieve", retrieve)
    monkeypatch.setattr(m2, "_official_retrieve", official)
    monkeypatch.setattr(m2, "_CLEAN_MODULE_CLI_ACTIVE", True)
    monkeypatch.setattr(
        m2,
        "_score_measurement",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("support scoring must not run after failed postflight")
        ),
    )
    with pytest.raises(m2.MuSiQueM2RunnerError, match="cannot be replayed"):
        _execute(
            fixture=fixture,
            execution_root=execution_root,
            freeze_path=freeze_path,
            m2_path=m2_path,
        )
    assert calls == m2.WORK_UNIT_COUNT
    failure = json.loads(
        (execution_root / m2.FAILURE_FILENAME).read_text("utf-8")
    )
    declared = failure.pop("failure_hash")
    assert stable_hash(failure) == declared
    assert failure["failure_stage"] == "fresh_runtime_postflight_before_scoring"
    assert failure["authorization_consumed"] is True
    assert failure["retrieval_attempt_count"] == m2.WORK_UNIT_COUNT
    assert failure["retrieval_terminal_count"] == m2.WORK_UNIT_COUNT
    assert failure["retries"] == failure["replays"] == failure["resamples"] == 0
    assert failure["replay_authorized"] is False
    assert "private postflight failure" not in (
        execution_root / m2.FAILURE_FILENAME
    ).read_text("utf-8")

    prior = calls
    with pytest.raises(m2.MuSiQueM2RunnerError, match="replay is forbidden"):
        _execute(
            fixture=fixture,
            execution_root=execution_root,
            freeze_path=freeze_path,
            m2_path=m2_path,
        )
    assert calls == prior
