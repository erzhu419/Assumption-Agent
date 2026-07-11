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
from assumption_agent.benchmarks import paper_freeze
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.paper_report import PaperTrialRecord
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
PROTOCOL_PATH = ROOT / "manifests" / "skilllearn_paper_protocol_v3_ruoli_gpt54mini.json"
MANIFEST_PATH = (
    ROOT / "manifests" / "skilllearnbench_instance_holdout_credential_independent_v1.json"
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
                "routing_version": "per_item_trigger_routing_v1",
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
    lock = {"lock_hash": stable_hash({"lock": 1})}
    receipt = {
        "frozen": True,
        "protocol_hash": protocol.protocol_hash,
        "protocol_lock_hash": lock["lock_hash"],
        "manifest_hash": manifest.manifest_hash,
        "evaluator_epoch": "skilllearn-eval-test",
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    assert validate_freeze_receipt(
        receipt,
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
    ) == "skilllearn-eval-test"
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
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paper_freeze, "_validate_protocol_lock", lambda *args: None)
    lock = {
        "lock_hash": stable_hash({"lock": "paper"}),
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
) -> dict[str, object]:
    phase = protocol.payload["phases"]["development"]
    return {
        "mode": "execute",
        "executed": True,
        "test_content_accessed": False,
        "preflight": {"blockers": []},
        "plan": {
            "manifest_hash": manifest.manifest_hash,
            "train_count": phase["train_count"],
            "validation_count": phase["validation_count"],
            "model": protocol.payload["model"],
            "trial_provider_mode": protocol.payload["trial_provider_mode"],
            "max_steps": protocol.payload["max_steps"],
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
                    "trial_network_budget_policy",
                    "trial_network_byte_limit",
                )
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
        "generation": {"promoted": True, "recursive_depth": 0},
        "generations": [{"promoted": True, "recursive_depth": 0}],
        "generation_count": 1,
        "archive_hash": archive_hash,
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
