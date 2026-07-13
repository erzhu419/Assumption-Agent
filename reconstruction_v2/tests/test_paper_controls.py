from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from assumption_agent.benchmarks.paper_controls import (
    ControlSource,
    PaperControlRunner,
    PaperRecordStore,
    SkillLearnBackendPool,
    control_config_hash,
    controls_from_freeze_receipt,
    finalize_sealed_journal,
    open_sealed_journal,
    validate_freeze_receipt,
)
from assumption_agent.benchmarks import paper_controls as paper_controls_module
from assumption_agent.benchmarks import paper_freeze, skilllearn_experiment
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.paper_report import PaperTrialRecord
from assumption_agent.benchmarks.skilllearn_compiler import (
    SKILL_ACTION_LOWERING_VERSION,
    SKILL_FALLBACK_SEMANTICS_VERSION,
    SKILL_ROUTING_VERSION,
    skilllearn_program_treatment_hash,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import SkillLearnTrialObservation
from assumption_agent.benchmarks.skilllearnbench import SkillLearnBenchAdapter
from assumption_agent.models import SplitName, stable_hash
from assumption_agent.models import HypothesisProgram
from assumption_agent.splits import SplitAccessGuard, SplitManifest


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = (
    ROOT
    / "reference"
    / "self_evo_continual_20260707"
    / "repos"
    / "SkillLearnBench"
)
PROTOCOL_PATH = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json"
)
V36_PROTOCOL_PATH = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_6_ruoli_gpt54mini.json"
)
V37_PROTOCOL_PATH = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_7_ruoli_gpt54mini.json"
)
V38_PROTOCOL_PATH = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_8_ruoli_gpt54mini.json"
)
V39_PROTOCOL_PATH = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_9_ruoli_gpt54mini.json"
)
V310_PROTOCOL_PATH = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_10_ruoli_gpt54mini.json"
)
V311_PROTOCOL_PATH = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_11_ruoli_gpt54mini.json"
)
MANIFEST_PATH = (
    ROOT / "manifests" / "skilllearnbench_instance_holdout_offline_ready_v1.json"
)
PERFORMANCE_CLAIM_BLOCKER = "proposal_model_failure_evidence_present"
INVALID_COUNTERFACTUAL_CLAIM_BLOCKER = (
    "invalid_counterfactual_evidence_present"
)


class FakePaperBackend:
    agent_id = "codex"
    model = "gpt-5.3-codex-spark"
    max_steps = 100

    def __init__(
        self,
        *,
        invalidate_once: tuple[str, str] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.invalidate_once = invalidate_once
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[str, str, str]] = []
        self._invalidated = False

    def run(self, request, *, skill_source_dir, trace_id):
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        control_id = request.program_id or "raw_no_skill"
        self.calls.append((request.item_id, control_id, request.variant.value))
        invalid = (
            not self._invalidated
            and self.invalidate_once == (request.item_id, control_id)
        )
        if invalid:
            self._invalidated = True
        success = skill_source_dir is not None and not invalid
        return SkillLearnTrialObservation(
            request=request,
            success=success,
            score=float(success),
            metrics={"task_success": float(success)},
            total_tokens=100,
            steps=10,
            duration_seconds=0.1,
            provider_fingerprint="provider-fixed",
            fairness_fingerprint="budget-fixed",
            error_type="endpoint_error" if invalid else None,
            upstream_result_hash=stable_hash({"trace": trace_id, "invalid": invalid}),
        )


def test_paper_controls_resume_without_repeating_valid_trials(tmp_path: Path) -> None:
    runner, backend, manifest = _runner(tmp_path)
    item_ids = manifest.validation_ids[:2]

    first = runner.run(item_ids, split=SplitName.VALIDATION, repeats=1)
    second = runner.run(item_ids, split=SplitName.VALIDATION, repeats=1)

    assert len(first) == 12
    assert len(second) == 12
    assert len(backend.calls) == 12
    assert all(row.protocol_hash == runner.protocol.protocol_hash for row in first)
    assert all(row.manifest_hash == manifest.manifest_hash for row in first)
    for item_id in item_ids:
        item_hash = stable_hash({"item_id": item_id})
        pair_ids = {
            row.pair_id for row in first if row.item_id_hash == item_hash
        }
        assert len(pair_ids) == 1


def test_paper_controls_reject_stale_records_after_control_content_changes(
    tmp_path: Path,
) -> None:
    runner, _, manifest = _runner(tmp_path)
    item_id = manifest.validation_ids[0]
    runner.run((item_id,), split=SplitName.VALIDATION, repeats=1)
    changed = next(row for row in runner.controls if row.root is not None)
    assert changed.root is not None
    skill = next(changed.root.rglob("SKILL.md"))
    skill.write_text("# changed control treatment\n", encoding="utf-8")
    backend = FakePaperBackend()
    resumed = PaperControlRunner(
        adapter=SkillLearnBenchAdapter(BENCH_ROOT),
        manifest=manifest,
        guard=SplitAccessGuard(manifest),
        backend=backend,
        protocol=runner.protocol,
        controls=runner.controls,
        record_store=PaperRecordStore(tmp_path / "records.jsonl"),
        evaluator_epoch="skilllearn-eval-test",
    )

    with pytest.raises(PermissionError, match="control configuration changed"):
        resumed.run((item_id,), split=SplitName.VALIDATION, repeats=1)
    assert backend.calls == []


def test_paper_controls_validate_lock_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("\n", encoding="utf-8")
    called: list[bool] = []

    def reject_before_model(*args, **kwargs):
        called.append(True)
        raise PermissionError("lock rejected before model work")

    monkeypatch.setattr(
        paper_controls_module,
        "validate_protocol_lock_for_execution",
        reject_before_model,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper_controls",
            "--project-root",
            str(ROOT),
            "--benchmark-root",
            str(BENCH_ROOT),
            "--manifest",
            str(MANIFEST_PATH),
            "--protocol",
            str(PROTOCOL_PATH),
            "--protocol-lock",
            str(lock_path),
            "--env-file",
            str(env_path),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--records",
            str(tmp_path / "records.jsonl"),
            "--trials-dir",
            str(tmp_path / "trials"),
            "--split",
            "validation",
        ],
    )

    with pytest.raises(PermissionError, match="before model work"):
        paper_controls_module.main()
    assert called == [True]


def test_paper_controls_retry_only_invalid_same_key(tmp_path: Path) -> None:
    manifest = SplitManifest.read(MANIFEST_PATH)
    target_item = manifest.validation_ids[0]
    runner, backend, _ = _runner(
        tmp_path,
        invalidate_once=(target_item, "promoted_v2"),
    )

    first = runner.run((target_item,), split=SplitName.VALIDATION, repeats=1)
    second = runner.run((target_item,), split=SplitName.VALIDATION, repeats=1)

    assert len(backend.calls) == 7
    first_target = next(row for row in first if row.control_id == "promoted_v2")
    second_target = next(row for row in second if row.control_id == "promoted_v2")
    assert first_target.valid is False
    assert first_target.attempt == 1
    assert second_target.valid is True
    assert second_target.attempt == 2
    assert first_target.pair_id == second_target.pair_id


def test_paper_controls_use_bounded_backend_pool(tmp_path: Path) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    item_id = manifest.validation_ids[0]
    family = manifest.family_by_id[item_id]
    backends = (
        FakePaperBackend(delay_seconds=0.02),
        FakePaperBackend(delay_seconds=0.02),
    )
    runner = PaperControlRunner(
        adapter=SkillLearnBenchAdapter(BENCH_ROOT),
        manifest=manifest,
        guard=SplitAccessGuard(manifest),
        backend=SkillLearnBackendPool(backends),
        protocol=protocol,
        controls=_controls(tmp_path, protocol, families={family}),
        record_store=PaperRecordStore(tmp_path / "parallel.records.jsonl"),
        evaluator_epoch="skilllearn-eval-test",
    )

    records = runner.run(
        (item_id,),
        split=SplitName.VALIDATION,
        repeats=1,
        parallel_workers=2,
    )

    assert len(records) == 6
    assert all(backend.calls for backend in backends)


def test_missing_family_skill_is_a_logged_policy_abstention(tmp_path: Path) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    item_id = manifest.validation_ids[0]
    family = manifest.family_by_id[item_id]
    controls = _controls(tmp_path, protocol, families={family})
    promoted = next(row for row in controls if row.id == "promoted_v2")
    assert promoted.root is not None
    family_dir = promoted.root / family
    (family_dir / "SKILL.md").unlink()
    family_dir.rmdir()
    backend = FakePaperBackend()
    runner = PaperControlRunner(
        adapter=SkillLearnBenchAdapter(BENCH_ROOT),
        manifest=manifest,
        guard=SplitAccessGuard(manifest),
        backend=backend,
        protocol=protocol,
        controls=controls,
        record_store=PaperRecordStore(tmp_path / "records.jsonl"),
        evaluator_epoch="skilllearn-eval-test",
    )

    records = runner.run((item_id,), split=SplitName.VALIDATION, repeats=1)

    record = next(row for row in records if row.control_id == "promoted_v2")
    call = next(row for row in backend.calls if row[1] == "promoted_v2")
    assert call[2] == "policy_off"
    assert record.valid is True


def test_compiled_control_uses_per_item_routes(tmp_path: Path) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    item_ids = manifest.validation_ids[:2]
    families = {manifest.family_by_id[item_id] for item_id in item_ids}
    controls = _controls(tmp_path, protocol, families=families)
    promoted = next(row for row in controls if row.id == "promoted_v2")
    assert promoted.root is not None
    route = promoted.root / "items" / stable_hash({"item_id": item_ids[0]})
    skill = route / "routed-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# routed\n", encoding="utf-8")
    (promoted.root / "compile_manifest.json").write_text(
        json.dumps(
                {
                    "routing_version": SKILL_ROUTING_VERSION,
                    "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
                    "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
                    "external_verifier_exposed_to_agent": False,
                "item_routes": {
                    stable_hash({"item_id": item_ids[0]}): str(
                        route.relative_to(promoted.root)
                    ),
                    stable_hash({"item_id": item_ids[1]}): None,
                },
            }
        ),
        encoding="utf-8",
    )
    backend = FakePaperBackend()
    runner = PaperControlRunner(
        adapter=SkillLearnBenchAdapter(BENCH_ROOT),
        manifest=manifest,
        guard=SplitAccessGuard(manifest),
        backend=backend,
        protocol=protocol,
        controls=controls,
        record_store=PaperRecordStore(tmp_path / "per-item.records.jsonl"),
        evaluator_epoch="skilllearn-eval-test",
    )

    runner.run(item_ids, split=SplitName.VALIDATION, repeats=1, parallel_workers=2)

    promoted_calls = [row for row in backend.calls if row[1] == "promoted_v2"]
    assert (item_ids[0], "promoted_v2", "policy_on") in promoted_calls
    assert (item_ids[1], "promoted_v2", "policy_off") in promoted_calls


def test_sealed_receipt_and_journal_are_content_bound(tmp_path: Path) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    lock = {
        "lock_hash": stable_hash({"lock": 1}),
        "code_fingerprint": {"tree_hash": "locked-code"},
        "git": {"commit": "locked-commit"},
    }
    receipt = {
        "frozen": True,
        "protocol_hash": protocol.protocol_hash,
        "protocol_lock_hash": lock["lock_hash"],
        "manifest_hash": manifest.manifest_hash,
        "evaluator_epoch": "skilllearn-eval-test",
        "code_fingerprint": lock["code_fingerprint"],
        "git_commit": lock["git"]["commit"],
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    assert validate_freeze_receipt(
        receipt,
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
    ) == "skilllearn-eval-test"
    drifted = {**receipt, "git_commit": "other-commit"}
    drifted["receipt_hash"] = stable_hash(
        {key: value for key, value in drifted.items() if key != "receipt_hash"}
    )
    with pytest.raises(PermissionError, match="git commit mismatch"):
        validate_freeze_receipt(
            drifted,
            protocol=protocol,
            protocol_lock=lock,
            manifest=manifest,
        )
    controls = _controls(tmp_path, protocol, families={manifest.family_by_id[manifest.test_ids[0]]})
    journal_path = tmp_path / "sealed.json"
    record_path = tmp_path / "records.jsonl"
    first = open_sealed_journal(
        journal_path,
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
        controls=controls,
        record_path=record_path,
    )
    assert first["status"] == "in_progress"
    source = next(row.root for row in controls if row.root is not None)
    assert source is not None
    family_dir = next(path for path in source.iterdir() if path.is_dir())
    (family_dir / "SKILL.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="control_config_hash"):
        open_sealed_journal(
            journal_path,
            protocol=protocol,
            protocol_lock=lock,
            manifest=manifest,
            controls=controls,
            record_path=record_path,
        )


def test_freeze_compiles_content_bound_validation_and_test_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    evaluator_epoch = f"skilllearn-eval-{manifest.manifest_hash[:12]}"
    recursive_archive_path = tmp_path / "recursive.archive.json"
    no_recursive_archive_path = tmp_path / "no-recursive.archive.json"
    recursive_archive = _archive_payload("recursive-policy", evaluator_epoch)
    no_recursive_archive = _archive_payload("no-recursive-policy", evaluator_epoch)
    protocol_lock_hash = stable_hash({"lock": "paper"})
    recursive_archive_path.write_text(
        json.dumps(recursive_archive),
        encoding="utf-8",
    )
    no_recursive_archive_path.write_text(
        json.dumps(no_recursive_archive),
        encoding="utf-8",
    )
    recursive_report_path = tmp_path / "recursive.report.json"
    no_recursive_report_path = tmp_path / "no-recursive.report.json"
    recursive_report_path.write_text(
        json.dumps(
            _development_report(
                protocol,
                manifest,
                archive_hash=recursive_archive["archive_hash"],
                recursive=True,
                hypothesis_id="recursive-policy",
                protocol_lock_hash=protocol_lock_hash,
            )
        ),
        encoding="utf-8",
    )
    no_recursive_report_path.write_text(
        json.dumps(
            _development_report(
                protocol,
                manifest,
                archive_hash=no_recursive_archive["archive_hash"],
                recursive=False,
                hypothesis_id="no-recursive-policy",
                protocol_lock_hash=protocol_lock_hash,
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_freeze, "_validate_protocol_lock", lambda *args: None)
    lock = {
        "lock_hash": protocol_lock_hash,
        "code_fingerprint": {"tree_hash": "frozen"},
        "git": {"commit": "commit"},
        "primary_manifest_hash": manifest.manifest_hash,
        "secondary_manifest_hash": "unused",
    }

    receipt = paper_freeze.freeze_paper_workspace(
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
        benchmark_root=BENCH_ROOT,
        project_root=ROOT,
        recursive_report_path=recursive_report_path,
        recursive_archive_path=recursive_archive_path,
        no_recursive_report_path=no_recursive_report_path,
        no_recursive_archive_path=no_recursive_archive_path,
        controls_output_root=tmp_path / "frozen-controls",
    )

    assert receipt["frozen"] is True
    assert receipt["selected_candidate_available"] is True
    assert receipt["receipt_hash"] == stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    for split in ("validation", "test"):
        controls = controls_from_freeze_receipt(receipt, split=split)
        assert {row.id for row in controls} == {
            str(row["id"]) for row in protocol.payload["controls"]
        }
        assert receipt["control_sets"][split]["config_hash"] == control_config_hash(
            controls
        )


def test_execution_report_binds_proposal_failure_claim_fields(
    tmp_path: Path,
) -> None:
    class Generation:
        def to_dict(self) -> dict[str, object]:
            return {"proposal_model_failure_count": 1}

    class Archive:
        def to_dict(self) -> dict[str, object]:
            return {"archive_hash": "archive-hash"}

    class Guard:
        test_accessed = False

    report = skilllearn_experiment._execution_report(
        plan={
            "counterfactual_invalid_evidence_policy": (
                "generation_terminal_non_claim_v1"
            )
        },
        preflight={},
        generations=(Generation(),),
        stop_reason="proposal_model_failure",
        archive=Archive(),
        archive_path=tmp_path / "archive.json",
        guard=Guard(),
    )

    assert report["proposal_model_failure_count"] == 1
    assert report["proposal_model_failures_present"] is True
    assert report["invalid_counterfactual_pair_count"] == 0
    assert report["invalid_counterfactual_pairs_present"] is False
    assert report["counterfactual_provider_mismatch_count"] == 0
    assert report["counterfactual_budget_mismatch_count"] == 0
    assert report["performance_claim_eligible"] is False
    assert report["performance_claim_blockers"] == [
        PERFORMANCE_CLAIM_BLOCKER
    ]


def test_freeze_rejects_honest_proposal_model_failure_report() -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    failed_generation = {
        "promoted": False,
        "recursive_depth": 0,
        "accepted_hypothesis_id": None,
        "evaluated_candidate_treatment_hash": None,
        "promotion_decision": None,
        "proposal_model_failure_count": 1,
    }
    report["generation"] = dict(failed_generation)
    report["generations"] = [failed_generation]
    report["proposal_model_failure_count"] = 1
    report["proposal_model_failures_present"] = True
    report["invalid_counterfactual_pair_count"] = 0
    report["invalid_counterfactual_pairs_present"] = False
    report["counterfactual_provider_mismatch_count"] = 0
    report["counterfactual_budget_mismatch_count"] = 0
    report["performance_claim_eligible"] = False
    report["performance_claim_blockers"] = [PERFORMANCE_CLAIM_BLOCKER]

    with pytest.raises(
        ValueError,
        match="development report contains proposal model failures",
    ):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_freeze_rejects_generation_failure_hidden_by_top_level_zero() -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    report["generations"][0]["proposal_model_failure_count"] = 1

    with pytest.raises(
        ValueError,
        match="development proposal model failure count mismatch",
    ):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_execution_report_binds_invalid_counterfactual_claim_fields(
    tmp_path: Path,
) -> None:
    summary = {
        "invalid_pair_count": 1,
        "provider_mismatch_count": 0,
        "budget_mismatch_count": 0,
    }

    class Generation:
        def to_dict(self) -> dict[str, object]:
            return {
                "proposal_model_failure_count": 0,
                "promotion_summary": summary,
                "promotion_decision": {"summary": summary},
            }

    class Archive:
        def to_dict(self) -> dict[str, object]:
            return {"archive_hash": "archive-hash"}

    class Guard:
        test_accessed = False

    report = skilllearn_experiment._execution_report(
        plan={
            "counterfactual_invalid_evidence_policy": (
                "generation_terminal_non_claim_v1"
            )
        },
        preflight={},
        generations=(Generation(),),
        stop_reason="invalid_counterfactual_evidence",
        archive=Archive(),
        archive_path=tmp_path / "archive.json",
        guard=Guard(),
    )

    assert report["invalid_counterfactual_pair_count"] == 1
    assert report["invalid_counterfactual_pairs_present"] is True
    assert report["performance_claim_eligible"] is False
    assert report["performance_claim_blockers"] == [
        INVALID_COUNTERFACTUAL_CLAIM_BLOCKER
    ]


def test_execution_report_preserves_legacy_promotion_summary_schema(
    tmp_path: Path,
) -> None:
    summary = {
        "invalid_pair_count": 0,
        "provider_mismatch_count": 0,
        "budget_mismatch_count": 0,
        "valid_activation_count": 1,
        "activated_gain_count": 1,
        "activated_harm_count": 0,
        "abstention_count": 0,
        "activation_precision": 1.0,
        "activation_precision_defined": True,
        "activated_harm_rate": 0.0,
        "activated_harm_rate_defined": True,
        "abstention_rate": 0.0,
    }

    class Generation:
        def to_dict(self) -> dict[str, object]:
            return {
                "proposal_model_failure_count": 0,
                "promotion_summary": dict(summary),
                "promotion_decision": {"summary": dict(summary)},
            }

    class Archive:
        def to_dict(self) -> dict[str, object]:
            return {"archive_hash": "archive-hash"}

    class Guard:
        test_accessed = False

    report = skilllearn_experiment._execution_report(
        plan={},
        preflight={},
        generations=(Generation(),),
        stop_reason="consecutive_non_promotion_limit",
        archive=Archive(),
        archive_path=tmp_path / "archive.json",
        guard=Guard(),
    )

    for container in (
        report["generation"]["promotion_summary"],
        report["generation"]["promotion_decision"]["summary"],
    ):
        assert not (
            set(container)
            & skilllearn_experiment._PROMOTION_SUMMARY_DIAGNOSTIC_KEYS
        )


@pytest.mark.parametrize(
    "protocol_path",
    (
        V36_PROTOCOL_PATH,
        V37_PROTOCOL_PATH,
        V38_PROTOCOL_PATH,
        V39_PROTOCOL_PATH,
        V310_PROTOCOL_PATH,
        V311_PROTOCOL_PATH,
    ),
)
def test_freeze_accepts_clean_contrastive_report(protocol_path: Path) -> None:
    protocol = PaperProtocol.read(protocol_path)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )

    paper_freeze._validate_development_report(
        report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=True,
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        (
            "contrastive_training_evidence_policy",
            None,
            "development generation contrastive evidence policy mismatch",
        ),
        (
            "success_control_count",
            0,
            "development generation contrastive evidence counts are inconsistent",
        ),
    ),
)
@pytest.mark.parametrize(
    "protocol_path",
    (
        V36_PROTOCOL_PATH,
        V37_PROTOCOL_PATH,
        V38_PROTOCOL_PATH,
        V39_PROTOCOL_PATH,
        V310_PROTOCOL_PATH,
        V311_PROTOCOL_PATH,
    ),
)
def test_freeze_rejects_contrastive_generation_evidence_drift(
    protocol_path: Path,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    protocol = PaperProtocol.read(protocol_path)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    report["generations"][0][field] = value
    report["generation"] = dict(report["generations"][0])

    with pytest.raises(ValueError, match=expected_error):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_freeze_rejects_honest_invalid_counterfactual_report() -> None:
    protocol = PaperProtocol.read(V36_PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    generation = report["generations"][0]
    decision = generation["promotion_decision"]
    decision["summary"]["invalid_pair_count"] = 1
    decision["allowed"] = False
    decision["blockers"] = ["invalid_counterfactual_pairs"]
    generation["promotion_summary"] = dict(decision["summary"])
    generation["promoted"] = False
    report["generation"] = dict(generation)
    report["invalid_counterfactual_pair_count"] = 1
    report["invalid_counterfactual_pairs_present"] = True
    report["performance_claim_eligible"] = False
    report["performance_claim_blockers"] = [
        INVALID_COUNTERFACTUAL_CLAIM_BLOCKER
    ]

    with pytest.raises(
        ValueError,
        match="development report contains invalid counterfactual evidence",
    ):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_freeze_rejects_invalid_counterfactual_evidence_in_legacy_report() -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    generation = report["generations"][0]
    decision = generation["promotion_decision"]
    decision["summary"]["invalid_pair_count"] = 1
    decision["allowed"] = False
    decision["blockers"] = ["invalid_counterfactual_pairs"]
    generation["promotion_summary"] = dict(decision["summary"])
    generation["promoted"] = False
    report["generation"] = dict(generation)

    with pytest.raises(
        ValueError,
        match="development report contains invalid counterfactual evidence",
    ):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_freeze_rejects_generation_invalid_count_hidden_at_top_level() -> None:
    protocol = PaperProtocol.read(V36_PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    report["generations"][0]["promotion_decision"]["summary"][
        "invalid_pair_count"
    ] = 1
    report["generations"][0]["promotion_summary"]["invalid_pair_count"] = 1

    with pytest.raises(
        ValueError,
        match="development invalid counterfactual pair count mismatch",
    ):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        (
            "proposal_model_failures_present",
            True,
            "development proposal model failure presence mismatch",
        ),
        (
            "performance_claim_eligible",
            False,
            "development performance claim eligibility mismatch",
        ),
        (
            "performance_claim_blockers",
            [PERFORMANCE_CLAIM_BLOCKER],
            "development performance claim blockers mismatch",
        ),
    ),
)
def test_freeze_rejects_tampered_performance_claim_binding(
    field: str,
    value: object,
    expected_error: str,
) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    report[field] = value

    with pytest.raises(ValueError, match=expected_error):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_freeze_accepts_clean_performance_claim_binding() -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )

    paper_freeze._validate_development_report(
        report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=True,
    )


def test_freeze_rejects_promotion_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    report["plan"]["promotion_contract"]["maximum_harm_rate"] = 1.0

    with pytest.raises(
        ValueError,
        match="development report plan mismatch: promotion_contract",
    ):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )

    lock = {
        "promotion": {
            **protocol.promotion_gate_spec.to_dict(),
            "maximum_harm_rate": 1.0,
        }
    }

    def reject_promotion_drift(*args, **kwargs):
        assert lock["promotion"] != protocol.promotion_gate_spec.to_dict()
        raise PermissionError("execution promotion contract lock mismatch")

    monkeypatch.setattr(
        paper_freeze,
        "validate_protocol_lock_for_execution",
        reject_promotion_drift,
    )
    with pytest.raises(
        PermissionError,
        match="promotion contract lock mismatch",
    ):
        paper_freeze._validate_protocol_lock(
            protocol,
            lock,
            manifest,
            ROOT,
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("extra_key", "promotion decision schema mismatch"),
        ("decision_contract", "promotion decision contract mismatch"),
        ("allowed", "promotion decision status mismatch"),
        ("candidate", "effective promotion thresholds mismatch"),
        ("effective", "effective promotion thresholds mismatch"),
        ("accepted", "promotion decision has no accepted hypothesis"),
    ),
)
def test_freeze_rejects_tampered_generation_promotion_decision(
    mutation: str,
    expected_error: str,
) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    row = report["generations"][0]
    decision = row["promotion_decision"]
    if mutation == "extra_key":
        decision["unexpected"] = True
    elif mutation == "decision_contract":
        decision["promotion_contract"]["maximum_harm_rate"] = 1.0
    elif mutation == "allowed":
        decision["allowed"] = False
    elif mutation == "candidate":
        decision["candidate_thresholds"]["minimum_effect_lower_bound"] = 0.5
    elif mutation == "effective":
        decision["effective_thresholds"]["maximum_harm_rate"] = 1.0
    else:
        row["accepted_hypothesis_id"] = None

    with pytest.raises(ValueError, match=expected_error):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_freeze_recomputes_blockers_from_promotion_summary() -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    report = _development_report(
        protocol,
        manifest,
        archive_hash="unused",
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    decision = report["generations"][0]["promotion_decision"]
    decision["summary"].update(
        {
            "pair_count": 10,
            "baseline_success_count": 10,
            "candidate_success_count": 0,
            "gain_count": 0,
            "harm_count": 10,
            "tie_count": 0,
            "activation_count": 10,
            "selection_change_count": 10,
            "baseline_preserved_count": 0,
            "invalid_pair_count": 0,
            "provider_mismatch_count": 0,
            "budget_mismatch_count": 0,
            "baseline_mean_cost": 1.0,
            "candidate_mean_cost": 1.0,
            "cost_ratio": 1.0,
            "mean_effect": -1.0,
            "effect_standard_error": 0.0,
            "effect_lower_bound": -1.0,
            "harm_rate": 1.0,
            "activation_rate": 1.0,
        }
    )
    report["generations"][0]["promotion_summary"] = dict(
        decision["summary"]
    )
    decision["effect_lower_bound"] = -1.0

    with pytest.raises(
        ValueError,
        match="development promotion summary blockers mismatch",
    ):
        paper_freeze._validate_development_report(
            report,
            protocol=protocol,
            manifest=manifest,
            recursive_validation_enabled=True,
        )


def test_frozen_archive_rejects_candidate_treatment_substitution(
    tmp_path: Path,
) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    evaluator_epoch = f"skilllearn-eval-{manifest.manifest_hash[:12]}"
    hypothesis_id = "recursive-policy"
    archive = _archive_payload(hypothesis_id, evaluator_epoch)
    original = HypothesisProgram.from_dict(archive["hypotheses"][hypothesis_id])
    archive["hypotheses"][hypothesis_id]["action_graph"][0]["value"] = (
        "A materially different lowered directive that was never evaluated."
    )
    substituted = HypothesisProgram.from_dict(archive["hypotheses"][hypothesis_id])
    assert skilllearn_program_treatment_hash(original) != (
        skilllearn_program_treatment_hash(substituted)
    )
    incumbent_id = str(archive["incumbent_id"])
    archive["archive_hash"] = stable_hash(
        {
            "hypotheses": {hypothesis_id: substituted.payload_hash},
            "nodes": {
                incumbent_id: stable_hash(archive["nodes"][incumbent_id]),
            },
            "scores": {},
            "incumbent_id": incumbent_id,
        }
    )
    report = _development_report(
        protocol,
        manifest,
        archive_hash=archive["archive_hash"],
        recursive=True,
        hypothesis_id=hypothesis_id,
    )
    paper_freeze._validate_development_report(
        report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=True,
    )
    archive_path = tmp_path / "substituted-archive.json"
    archive_path.write_text(json.dumps(archive), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="archive candidate treatment and development evidence differ",
    ):
        paper_freeze.read_frozen_archive(
            archive_path,
            expected_evaluator_epoch=evaluator_epoch,
            expected_report=report,
            promotion_spec=protocol.promotion_gate_spec,
        )


def test_frozen_archive_cross_checks_candidate_decision_thresholds(
    tmp_path: Path,
) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    evaluator_epoch = f"skilllearn-eval-{manifest.manifest_hash[:12]}"
    archive = _archive_payload("recursive-policy", evaluator_epoch)
    archive_path = tmp_path / "archive.json"
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    report = _development_report(
        protocol,
        manifest,
        archive_hash=archive["archive_hash"],
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    decision = report["generations"][0]["promotion_decision"]
    decision["candidate_thresholds"]["minimum_effect_lower_bound"] = 0.2
    decision["effective_thresholds"]["minimum_effect_lower_bound"] = 0.2
    paper_freeze._validate_development_report(
        report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=True,
    )

    with pytest.raises(
        ValueError,
        match="archive candidate thresholds and decision differ",
    ):
        paper_freeze.read_frozen_archive(
            archive_path,
            expected_evaluator_epoch=evaluator_epoch,
            expected_report=report,
            promotion_spec=protocol.promotion_gate_spec,
        )

    unknown_report = _development_report(
        protocol,
        manifest,
        archive_hash=archive["archive_hash"],
        recursive=True,
        hypothesis_id="recursive-policy",
    )
    unknown_report["generations"][0]["accepted_hypothesis_id"] = "unknown-policy"
    with pytest.raises(
        ValueError,
        match="development decision references an unknown hypothesis",
    ):
        paper_freeze.read_frozen_archive(
            archive_path,
            expected_evaluator_epoch=evaluator_epoch,
            expected_report=unknown_report,
            promotion_spec=protocol.promotion_gate_spec,
        )

    malformed_archive = json.loads(json.dumps(archive))
    malformed_archive["hypotheses"]["recursive-policy"]["expected_effect"][
        "maximum_cost_ratio"
    ] = "not-a-number"
    malformed_path = tmp_path / "malformed-archive.json"
    malformed_path.write_text(json.dumps(malformed_archive), encoding="utf-8")
    with pytest.raises(ValueError, match="archive hypothesis payload is malformed"):
        paper_freeze.read_frozen_archive(
            malformed_path,
            expected_evaluator_epoch=evaluator_epoch,
            expected_report=unknown_report,
            promotion_spec=protocol.promotion_gate_spec,
        )
def test_completed_sealed_journal_rejects_missing_or_changed_records(tmp_path: Path) -> None:
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    lock = {"lock_hash": stable_hash({"lock": "journal"})}
    family = manifest.family_by_id[manifest.test_ids[0]]
    controls = _controls(tmp_path, protocol, families={family})
    journal_path = tmp_path / "sealed.json"
    record_path = tmp_path / "records.jsonl"
    open_sealed_journal(
        journal_path,
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
        controls=controls,
        record_path=record_path,
    )
    record = PaperTrialRecord(
        item_id_hash=stable_hash({"item_id": manifest.test_ids[0]}),
        family_hash=stable_hash({"family": family}),
        split="test",
        control_id="raw_no_skill",
        protocol_hash=protocol.protocol_hash,
        manifest_hash=manifest.manifest_hash,
        evaluator_epoch="skilllearn-eval-test",
        pair_id=stable_hash({"pair": 1})[:20],
        repeat=1,
        success=False,
        score=0.0,
        valid=True,
        provider_fingerprint="provider-fixed",
        fairness_fingerprint="budget-fixed",
        total_tokens=10,
        steps=1,
        duration_seconds=0.1,
    )
    store = PaperRecordStore(record_path)
    store.append(record)
    completed = finalize_sealed_journal(journal_path, records=(record,))
    assert completed["status"] == "complete"
    resumed = open_sealed_journal(
        journal_path,
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
        controls=controls,
        record_path=record_path,
    )
    assert resumed["access_invocation_count"] == 2
    finalize_sealed_journal(journal_path, records=(record,))
    record_path.unlink()
    with pytest.raises(PermissionError, match="record file is missing"):
        open_sealed_journal(
            journal_path,
            protocol=protocol,
            protocol_lock=lock,
            manifest=manifest,
            controls=controls,
            record_path=record_path,
        )


def _archive_payload(hypothesis_id: str, evaluator_epoch: str) -> dict[str, object]:
    payload = json.loads((ROOT / "baselines" / "static_generic_program.json").read_text())
    payload["id"] = hypothesis_id
    payload["evaluator_epoch"] = evaluator_epoch
    payload["status"] = "promoted"
    program = HypothesisProgram.from_dict(payload)
    node_id = f"node-{hypothesis_id}"
    node = {
        "id": node_id,
        "parent_id": None,
        "active_hypothesis_ids": [hypothesis_id],
        "evaluator_epoch_id": evaluator_epoch,
        "runtime_version": "test-runtime",
        "generation": 0,
        "status": "incumbent",
    }
    archive_hash = stable_hash(
        {
            "hypotheses": {hypothesis_id: program.payload_hash},
            "nodes": {node_id: stable_hash(node)},
            "scores": {},
            "incumbent_id": node_id,
        }
    )
    return {
        "hypotheses": {hypothesis_id: program.to_dict()},
        "nodes": {node_id: node},
        "score_records": {},
        "incumbent_id": node_id,
        "archive_hash": archive_hash,
        "raw_content_persisted": False,
    }


def _development_report(
    protocol: PaperProtocol,
    manifest: SplitManifest,
    *,
    archive_hash: object,
    recursive: bool,
    hypothesis_id: str,
    protocol_lock_hash: str | None = None,
) -> dict[str, object]:
    phase = protocol.payload["phases"]["development"]
    evaluator_epoch = f"skilllearn-eval-{manifest.manifest_hash[:12]}"
    payload = json.loads(
        (ROOT / "baselines" / "static_generic_program.json").read_text()
    )
    payload["id"] = hypothesis_id
    payload["evaluator_epoch"] = evaluator_epoch
    payload["status"] = "promoted"
    program = HypothesisProgram.from_dict(payload)
    generation = {
        "promoted": True,
        "recursive_depth": 0,
        "proposal_model_failure_count": 0,
        "accepted_hypothesis_id": hypothesis_id,
        "evaluated_candidate_treatment_hash": (
            skilllearn_program_treatment_hash(program)
        ),
        "promotion_decision": _promotion_decision(
            protocol,
            program,
            evaluator_epoch=evaluator_epoch,
        ),
    }
    if protocol.payload["protocol_version"] in {
        "3.6.0",
        "3.7.0",
        "3.8.0",
        "3.9.0",
        "3.10.0",
        "3.11.0",
    }:
        train_count = int(phase["train_count"])
        generation.update(
            {
                "train_observation_count": train_count,
                "valid_train_observation_count": train_count,
                "training_residual_count": 1,
                "success_control_count": train_count - 1,
                "example_count": train_count,
                "contrastive_training_evidence_policy": protocol.payload[
                    "execution"
                ]["contrastive_training_evidence_policy"],
            }
        )
    generation["promotion_summary"] = dict(
        generation["promotion_decision"]["summary"]
    )
    return {
        "mode": "execute",
        "executed": True,
        "test_content_accessed": False,
        "preflight": {"blockers": []},
        "plan": {
            "paper_protocol_id": protocol.id,
            "paper_protocol_hash": protocol.protocol_hash,
            "promotion_contract": protocol.promotion_gate_spec.to_dict(),
            "protocol_lock_hash": protocol_lock_hash,
            "manifest_hash": manifest.manifest_hash,
            "experiment_phase": "development",
            "train_count": phase["train_count"],
            "validation_count": phase["validation_count"],
            "agent_id": protocol.payload["agent_id"],
            "model": protocol.payload["model"],
            "trial_provider_mode": protocol.payload["trial_provider_mode"],
            "max_steps": protocol.payload["max_steps"],
            "parallel_workers": phase["parallel_workers"],
            "minimum_trigger_support": protocol.payload["evolution"][
                "minimum_trigger_support"
            ],
            **(
                {
                    "codex_agent_execution_policy": (
                        protocol.codex_agent_execution_policy.to_dict()
                    ),
                    "codex_agent_execution_policy_hash": (
                        protocol.codex_agent_execution_policy.policy_hash
                    ),
                }
                if protocol.payload["protocol_version"]
                not in {"3.1.0", "3.2.0"}
                else {}
            ),
            "runner_agent_registry_isolation": protocol.payload["execution"][
                "runner_agent_registry_isolation"
            ],
            "trial_timeout_policy": protocol.payload["execution"][
                "trial_timeout_policy"
            ],
            "provider_failure_policy": protocol.payload["execution"][
                "provider_failure_policy"
            ],
            "container_egress_policy": protocol.payload["execution"][
                "container_egress_policy"
            ],
            "dependency_cache_policy": protocol.payload["execution"][
                "dependency_cache_policy"
            ],
            "provider_dns_policy": protocol.payload["execution"][
                "provider_dns_policy"
            ],
            **{
                field: protocol.payload["execution"][field]
                for field in (
                    "provider_route_policy",
                    "counterfactual_replay_policy",
                    "baseline_arm_evidence_replay_policy",
                    "root_proposal_replay_policy",
                    "training_evidence_replay_policy",
                    "invalid_trial_retry_policy",
                    "invalid_trial_max_attempts",
                    "invalid_trial_retry_backoff_seconds",
                    "invalid_trial_retry_workers",
                    "local_evidence_transport",
                    "network_scope_audit",
                    "proposal_failure_isolation_policy",
                    "openai_compatible_codex_config",
                    "codex_network_minimization",
                    "model_only_tool_policy",
                    "verifier_execution_receipt_policy",
                    "offline_verifier_policy",
                    "trial_network_budget_policy",
                    "trial_network_byte_limit",
                    "skill_routing",
                    "skill_action_lowering",
                    "skill_fallback_semantics",
                    "proposal_candidate_selection",
                )
            },
            **{
                field: protocol.payload["execution"][field]
                for field in (
                    "contrastive_training_evidence_policy",
                    "counterfactual_invalid_evidence_policy",
                    "model_inference_concurrency_policy",
                    "model_inference_slots",
                    "proposal_diversity_policy",
                    "proposal_response_max_tokens",
                )
                if field in protocol.payload["execution"]
            },
            "training_evidence_policy": protocol.payload["execution"][
                "training_evidence_policy"
            ],
            "development_prewarm_version": protocol.payload["execution"][
                "development_prewarm"
            ],
            "prewarm_passed": True,
            "prewarm_receipt_hash": stable_hash({"prewarm": manifest.manifest_hash}),
            "recursive_validation_enabled": recursive,
            "max_generations": protocol.payload["evolution"]["max_generations"],
            "max_consecutive_non_promotions": protocol.payload["evolution"][
                "max_consecutive_non_promotions"
            ],
            "proposal_candidates_per_generation": protocol.payload["evolution"][
                "proposal_candidates_per_generation"
            ],
            "test_content_accessed": False,
        },
        "generation": dict(generation),
        "generations": [generation],
        "generation_count": 1,
        "proposal_model_failure_count": 0,
        "proposal_model_failures_present": False,
        "invalid_counterfactual_pair_count": 0,
        "invalid_counterfactual_pairs_present": False,
        "counterfactual_provider_mismatch_count": 0,
        "counterfactual_budget_mismatch_count": 0,
        "performance_claim_eligible": True,
        "performance_claim_blockers": [],
        "archive_hash": archive_hash,
    }


def _promotion_decision(
    protocol: PaperProtocol,
    program: HypothesisProgram,
    *,
    evaluator_epoch: str,
) -> dict[str, object]:
    spec = protocol.promotion_gate_spec
    candidate = {
        "minimum_effect_lower_bound": program.expected_effect.minimum_delta,
        "maximum_harm_rate": program.expected_effect.maximum_harm_rate,
        "maximum_cost_ratio": program.expected_effect.maximum_cost_ratio,
    }
    summary = {
        "pair_count": 10,
        "baseline_success_count": 0,
        "candidate_success_count": 10,
        "gain_count": 10,
        "harm_count": 0,
        "tie_count": 0,
        "activation_count": 10,
        "selection_change_count": 10,
        "baseline_preserved_count": 0,
        "invalid_pair_count": 0,
        "provider_mismatch_count": 0,
        "budget_mismatch_count": 0,
        "baseline_mean_cost": 1.0,
        "candidate_mean_cost": 1.0,
        "cost_ratio": 1.0,
        "mean_effect": 1.0,
        "effect_standard_error": 0.0,
        "effect_lower_bound": 1.0,
        "harm_rate": 0.0,
        "activation_rate": 1.0,
    }
    if protocol.payload["protocol_version"] in {
        "3.6.0",
        "3.7.0",
        "3.8.0",
        "3.9.0",
        "3.10.0",
        "3.11.0",
    }:
        summary.update(
            {
                "valid_activation_count": 10,
                "activated_gain_count": 10,
                "activated_harm_count": 0,
                "abstention_count": 0,
                "activation_precision": 1.0,
                "activation_precision_defined": True,
                "activated_harm_rate": 0.0,
                "activated_harm_rate_defined": True,
                "abstention_rate": 0.0,
            }
        )
    return {
        "allowed": True,
        "blockers": [],
        "summary": summary,
        "effect_lower_bound": 1.0,
        "evaluator_epoch": evaluator_epoch,
        "promotion_contract": spec.to_dict(),
        "candidate_metric": program.expected_effect.metric,
        "candidate_thresholds": candidate,
        "effective_thresholds": spec.effective_thresholds(program),
        "policy": "evaluator_owned_paired_validation_v2",
    }


def _runner(
    tmp_path: Path,
    *,
    invalidate_once: tuple[str, str] | None = None,
):
    protocol = PaperProtocol.read(PROTOCOL_PATH)
    manifest = SplitManifest.read(MANIFEST_PATH)
    families = {manifest.family_by_id[item_id] for item_id in manifest.validation_ids[:2]}
    backend = FakePaperBackend(invalidate_once=invalidate_once)
    runner = PaperControlRunner(
        adapter=SkillLearnBenchAdapter(BENCH_ROOT),
        manifest=manifest,
        guard=SplitAccessGuard(manifest),
        backend=backend,
        protocol=protocol,
        controls=_controls(tmp_path, protocol, families=families),
        record_store=PaperRecordStore(tmp_path / "records.jsonl"),
        evaluator_epoch="skilllearn-eval-test",
    )
    return runner, backend, manifest


def _controls(
    tmp_path: Path,
    protocol: PaperProtocol,
    *,
    families: set[str],
) -> tuple[ControlSource, ...]:
    controls: list[ControlSource] = []
    for row in protocol.payload["controls"]:
        control_id = str(row["id"])
        if control_id == "raw_no_skill":
            controls.append(ControlSource(control_id, None))
            continue
        root = tmp_path / "controls" / control_id
        for family in families:
            family_dir = root / family
            family_dir.mkdir(parents=True, exist_ok=True)
            (family_dir / "SKILL.md").write_text(
                f"# {control_id}\n",
                encoding="utf-8",
            )
        controls.append(ControlSource(control_id, root))
    return tuple(controls)
