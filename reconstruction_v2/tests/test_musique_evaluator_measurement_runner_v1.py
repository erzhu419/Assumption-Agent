from __future__ import annotations

import concurrent.futures
import inspect
import json
from pathlib import Path
import threading

import pytest

from assumption_agent.benchmarks import musique_evaluator_measurement_runner_v1 as runner
from assumption_agent.benchmarks import musique_evaluator_stage_formation_v1 as formation
from assumption_agent.models import stable_hash
from tests.test_musique_evaluator_stage_formation_v1 import _form_a_and_f3
from tests.test_musique_recursive_study_blocks_and_m1_v1 import _write_block


PROJECT = Path(__file__).resolve().parents[1]


def _freeze_a_hold(formed: dict[str, object], tmp_path: Path) -> tuple[Path, Path]:
    execution_root = tmp_path / "a-hold-formal-root"
    freeze_path = tmp_path / "a-hold.freeze.json"
    runner.build_a_hold_pre_run_freeze(
        project_root=PROJECT,
        acquisition_receipt_path=formed["acquisition"],
        a_form_private_evidence_path=formed["a_private"],
        a_form_public_receipt_path=formed["a_public"],
        execution_root=execution_root,
        authorization_hash=stable_hash({"authorization": "A_hold"}),
        output_path=freeze_path,
    )
    return freeze_path, execution_root


def _a_hold_cli_arguments(
    formed: dict[str, object],
    *,
    freeze_path: Path,
    block_path: Path,
    execution_root: Path,
) -> list[str]:
    return [
        "execute-a-hold",
        "--project-root",
        str(PROJECT),
        "--acquisition-receipt",
        str(formed["acquisition"]),
        "--a-form-private-evidence",
        str(formed["a_private"]),
        "--a-form-public-receipt",
        str(formed["a_public"]),
        "--execution-root",
        str(execution_root),
        "--a-hold-block",
        str(block_path),
        "--pre-run-freeze",
        str(freeze_path),
    ]


def test_a_hold_freeze_precedes_open_and_clean_cli_is_gold_free_max_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formed = _form_a_and_f3(tmp_path)
    freeze_path, execution_root = _freeze_a_hold(formed, tmp_path)
    a_hold_path = tmp_path / "private-pack" / "A_hold.jsonl"
    assert not a_hold_path.exists()

    freeze = json.loads(freeze_path.read_text("utf-8"))
    assert freeze["ordering"] == {
        "measurement_block_rows_read_while_freezing": 0,
        "measurement_support_labels_read_while_freezing": 0,
        "pre_run_freeze_complete_before_measurement_open": True,
    }
    work_units = formation.candidate_set_binding()["candidate_count"] * 12
    assert freeze["execution_contract"]["maximum_local_concurrency"] == work_units

    a_hold_path = _write_block(tmp_path, "A_hold", formed["raw"]["A_hold"])
    with pytest.raises(runner.MuSiQueEvaluatorMeasurementError, match="clean module CLI"):
        runner.execute_a_hold_formal(
            project_root=PROJECT,
            pre_run_freeze_path=freeze_path,
            a_hold_block_path=a_hold_path,
            acquisition_receipt_path=formed["acquisition"],
            a_form_private_evidence_path=formed["a_private"],
            a_form_public_receipt_path=formed["a_public"],
            execution_root=execution_root,
        )
    assert not execution_root.exists()

    original_retrieve = formation.typed_retrieve
    real_executor = concurrent.futures.ThreadPoolExecutor
    calls = 0
    lock = threading.Lock()
    configured_workers: list[int] = []

    def gold_free_retrieve(program, question, corpus):
        nonlocal calls
        assert isinstance(question, str)
        assert all(not hasattr(paragraph, "is_supporting") for paragraph in corpus)
        assert all(not hasattr(paragraph, "support_indices") for paragraph in corpus)
        with lock:
            calls += 1
        return original_retrieve(program, question, corpus)

    def recording_executor(*args, **kwargs):
        configured_workers.append(int(kwargs.get("max_workers", args[0] if args else 0)))
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(formation, "typed_retrieve", gold_free_retrieve)
    monkeypatch.setattr(
        formation.concurrent.futures, "ThreadPoolExecutor", recording_executor
    )
    assert runner.main(
        _a_hold_cli_arguments(
            formed,
            freeze_path=freeze_path,
            block_path=a_hold_path,
            execution_root=execution_root,
        )
    ) == 0

    report_path = execution_root / runner.REPORT_FILENAME["A_hold"]
    report = json.loads(report_path.read_text("utf-8"))
    assert calls == work_units
    assert configured_workers == [work_units]
    assert report["valid"] is True
    assert report["execution"]["retrieval_call_count"] == work_units
    assert report["execution"]["retrieval_terminal_count"] == work_units
    assert report["execution"]["all_terminals_joined_before_support_scoring"] is True
    assert report["transition_verification"] == {
        "confidence": runner.ANCHOR_CONFIDENCE,
        "policy": "strict_wilson_lower_bound_improvement_v1",
        "recomputed_from_exact_A_hold_evidence": True,
        "official_support_objective_replaced": False,
    }
    public_text = freeze_path.read_text("utf-8") + report_path.read_text("utf-8")
    for private_text in (
        "private-A_hold-",
        "Which private record",
        "root evidence",
        '"support_indices"',
        '"items"',
    ):
        assert private_text not in public_text


def test_anchor_reverification_rejects_rehashed_execution_or_policy_tamper(
    tmp_path: Path,
) -> None:
    formed = _form_a_and_f3(tmp_path)
    freeze_path, execution_root = _freeze_a_hold(formed, tmp_path)
    a_hold_path = _write_block(tmp_path, "A_hold", formed["raw"]["A_hold"])
    assert runner.main(
        _a_hold_cli_arguments(
            formed,
            freeze_path=freeze_path,
            block_path=a_hold_path,
            execution_root=execution_root,
        )
    ) == 0
    private_path = execution_root / runner.PRIVATE_EVIDENCE_FILENAME["A_hold"]
    report_path = execution_root / runner.REPORT_FILENAME["A_hold"]
    original = report_path.read_bytes()

    for mutation in ("execution", "transition"):
        report = json.loads(original)
        if mutation == "execution":
            report["execution"]["network_calls"] = 1
        else:
            report["transition_verification"][
                "official_support_objective_replaced"
            ] = True
        body = dict(report)
        body.pop("report_hash")
        report["report_hash"] = stable_hash(body)
        report_path.write_text(json.dumps(report, sort_keys=True) + "\n", "utf-8")
        with pytest.raises(
            runner.MuSiQueEvaluatorMeasurementError,
            match="exact anchor evidence|Wilson transition",
        ):
            runner.load_and_reverify_a_hold_artifacts(
                private_evidence_path=private_path,
                report_path=report_path,
                a_form_private_evidence_path=formed["a_private"],
                a_form_public_receipt_path=formed["a_public"],
                project_root=PROJECT,
            )
        report_path.write_bytes(original)


def test_m3_reverifies_anchor_measures_prospective_utility_and_cannot_replay(
    tmp_path: Path,
) -> None:
    formed = _form_a_and_f3(tmp_path)
    a_freeze, a_root = _freeze_a_hold(formed, tmp_path)
    a_block = _write_block(tmp_path, "A_hold", formed["raw"]["A_hold"])
    assert runner.main(
        _a_hold_cli_arguments(
            formed,
            freeze_path=a_freeze,
            block_path=a_block,
            execution_root=a_root,
        )
    ) == 0

    m3_root = tmp_path / "m3-formal-root"
    m3_freeze = tmp_path / "m3.freeze.json"
    m3_block = tmp_path / "private-pack" / "M3.jsonl"
    assert not m3_block.exists()
    runner.build_m3_pre_run_freeze(
        project_root=PROJECT,
        acquisition_receipt_path=formed["acquisition"],
        a_form_private_evidence_path=formed["a_private"],
        a_form_public_receipt_path=formed["a_public"],
        f3_private_evidence_path=formed["f3_private"],
        f3_public_receipt_path=formed["f3_public"],
        a_hold_private_evidence_path=(
            a_root / runner.PRIVATE_EVIDENCE_FILENAME["A_hold"]
        ),
        a_hold_report_path=a_root / runner.REPORT_FILENAME["A_hold"],
        execution_root=m3_root,
        authorization_hash=stable_hash({"authorization": "M3"}),
        output_path=m3_freeze,
    )
    assert not m3_block.exists()
    m3_block = _write_block(tmp_path, "M3", formed["raw"]["M3"])
    arguments = [
        "execute-m3",
        "--project-root",
        str(PROJECT),
        "--acquisition-receipt",
        str(formed["acquisition"]),
        "--a-form-private-evidence",
        str(formed["a_private"]),
        "--a-form-public-receipt",
        str(formed["a_public"]),
        "--execution-root",
        str(m3_root),
        "--f3-private-evidence",
        str(formed["f3_private"]),
        "--f3-public-receipt",
        str(formed["f3_public"]),
        "--a-hold-private-evidence",
        str(a_root / runner.PRIVATE_EVIDENCE_FILENAME["A_hold"]),
        "--a-hold-report",
        str(a_root / runner.REPORT_FILENAME["A_hold"]),
        "--m3-block",
        str(m3_block),
        "--pre-run-freeze",
        str(m3_freeze),
    ]
    assert runner.main(arguments) == 0
    report = json.loads(
        (m3_root / runner.REPORT_FILENAME["M3"]).read_text("utf-8")
    )
    assert report["valid"] is True
    assert report["anchor_reverification"][
        "strict_wilson_transition_recomputed_from_exact_A_hold_evidence"
    ] is True
    assert report["anchor_reverification"][
        "completed_before_M3_utility_evaluation"
    ] is True
    assert report["core_result"]["measurement_partition"] == "M3"
    assert report["core_result"]["model_calls"] == 0
    assert report["core_result"]["online_evaluator_calls"] == 0

    with pytest.raises(runner.MuSiQueEvaluatorMeasurementError, match="replay is forbidden"):
        runner.main(arguments)


def test_formal_surfaces_have_no_result_injection_and_freezes_cannot_open_blocks(
) -> None:
    assert runner.formal_signatures_have_no_injection_surface() is True
    assert "a_hold_block_path" not in inspect.signature(
        runner.build_a_hold_pre_run_freeze
    ).parameters
    assert "m3_block_path" not in inspect.signature(
        runner.build_m3_pre_run_freeze
    ).parameters
    forbidden = {
        "candidate_programs",
        "evidence",
        "operator",
        "operator_factory",
        "result_injection",
        "results",
        "retriever",
        "runner",
    }
    for function in (runner.execute_a_hold_formal, runner.execute_m3_formal):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
    with pytest.raises(
        runner.MuSiQueEvaluatorMeasurementError,
        match="external or git-ignored",
    ):
        runner._require_private_execution_boundary(
            PROJECT / "formal-evaluator-root-must-be-ignored-test"
        )
