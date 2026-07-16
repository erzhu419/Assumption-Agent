from __future__ import annotations

import csv
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any
import zipfile

import pytest

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from replication_runtime.financial_semantic_v2 import pack as period_pack
from replication_runtime.financial_semantic_v2.pack import payload_hash
from replication_runtime.financial_semantic_v2.plan import FixedPeriodOutTreatmentV2
from replication_runtime.financial_sec13f_contract_v2 import sealed_access
from replication_runtime.financial_sec13f_contract_v2 import sealed_freeze
from replication_runtime.financial_sec13f_contract_v2 import sealed_materialize
from replication_runtime.financial_sec13f_contract_v2 import sealed_plan
from replication_runtime.financial_sec13f_contract_v2 import sealed_prepare
from replication_runtime.financial_sec13f_contract_v2 import sealed_prewarm
from replication_runtime.financial_sec13f_contract_v2 import sealed_runner
from replication_runtime.financial_sec13f_contract_v2.hygienic_materialize import (
    CURRENT_ALIAS,
    PREVIOUS_ALIAS,
)
from replication_runtime.financial_sec13f_contract_v2.treatment import (
    load_fixed_contract_candidate_v2,
)


PROJECT = Path(__file__).resolve().parents[1]
MANAGER_COUNT = 32
ISSUER_COUNT = 20


def _write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_period(root: Path, *, current: bool) -> None:
    date = "31-MAR-2026" if current else "31-DEC-2025"
    prefix = "C" if current else "P"
    cover: list[list[object]] = []
    info: list[list[object]] = []
    for manager_index in range(MANAGER_COUNT):
        accession = f"{prefix}{manager_index:05d}"
        manager = f"Synthetic Fund {manager_index:02d} LLC"
        cover.append([accession, date, "13F HOLDINGS REPORT", manager])
        for issuer_index in range(ISSUER_COUNT):
            baseline = 10_000 + manager_index * 100 + issuer_index
            value = baseline + (issuer_index + 1) * 1_000 + manager_index if current else baseline
            info.append(
                [
                    accession,
                    f"Synthetic Issuer {issuer_index:02d}",
                    "COM",
                    f"{100_000_000 + issuer_index:09d}",
                    value,
                ]
            )
        info.append([accession, "Synthetic Note", "PUT", f"{900_000_000 + manager_index:09d}", 777])
    _write_tsv(
        root / "COVERPAGE.tsv",
        ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "REPORTTYPE", "FILINGMANAGER_NAME"],
        cover,
    )
    _write_tsv(
        root / "INFOTABLE.tsv",
        ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE"],
        info,
    )


def _zip_period(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.iterdir()):
            archive.write(path, arcname=f"official/{path.name}")


def _authorization(
    *,
    study_id: str,
    pack_hash: str,
    pack_file_sha: str,
    view_hash: str,
    candidate_id: str,
) -> dict[str, Any]:
    prereq = {
        name: {"manifest_hash": payload_hash({"name": name})}
        for name in (
            "controls_disposition",
            "family_out_disposition",
            "promotion_decision",
            "sealed_preregistration",
        )
    }
    body = {
        "manifest_version": sealed_access.AUTHORIZATION_VERSION,
        "study_id": study_id,
        "authorization": {
            "private_pack_content_access_authorized": True,
            "sealed_evaluation_authorized": True,
            "sealed_preparation_authorized": True,
            "sealed_scoring_authorized": True,
            "sealed_item_count_authorized": 4,
            "online_judge_authorized": False,
            "candidate_mutation_authorized": False,
        },
        "decision": {
            "sealed_authorization_decision": "authorize_exact_preregistered_replication_c_sealed_evaluation",
            "post_controls": True,
            "controls_disposition_accepted": True,
            "family_out_disposition_accepted": True,
            "incident_explicitly_adjudicated": True,
            "sealed_item_set_changed": False,
        },
        "prerequisite_bindings": prereq,
        "sealed_cohort_binding": {
            "private_pack_hash": pack_hash,
            "measurement_view_hash": view_hash,
            "item_count": 4,
            "item_replacement_authorized": False,
            "precommitted_private_pack_file_sha256": pack_file_sha,
        },
        "candidate_and_provider_binding": {
            "candidate_id": candidate_id,
            "provider_label": "plus",
            "pro_fallback_authorized": False,
            "candidate_unchanged_after_promotion_and_controls": True,
        },
        "incident_adjudication": {
            "accepted_for_current_pack_continuation": True,
            "semantic_holdout_blindness_preserved": True,
            "strict_zero_byte_policy_satisfied": False,
            "original_zero_byte_pre_authorization_claim_waived": True,
        },
        "sequencing": {
            "access_journal_must_open_before_next_private_pack_byte_read": True,
            "next_private_pack_access_must_be_recorded": True,
        },
    }
    return {**body, "manifest_hash": payload_hash(body)}


@pytest.fixture()
def synthetic_sealed(tmp_path: Path) -> SimpleNamespace:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    _write_period(previous, current=False)
    _write_period(current, current=True)
    previous_zip = tmp_path / "previous.zip"
    current_zip = tmp_path / "current.zip"
    _zip_period(previous, previous_zip)
    _zip_period(current, current_zip)
    pack = period_pack.build_public_pack(
        previous_source=previous_zip,
        current_source=current_zip,
        previous_period_label="2025 Q4",
        current_period_label="2026 Q1",
        preregistration_seed="sealed-synthetic-test",
        previous_container_root=PREVIOUS_ALIAS,
        current_container_root=CURRENT_ALIAS,
    )
    view = period_pack.build_measurement_view(pack)
    private_pack = tmp_path / "private.pack.json"
    period_pack.write_json(private_pack, pack)
    candidate = load_fixed_contract_candidate_v2(PROJECT)
    auth = _authorization(
        study_id=sealed_freeze.STUDY_ID,
        pack_hash=pack["pack_hash"],
        pack_file_sha=period_pack.sha256_file(private_pack),
        view_hash=view["measurement_view_hash"],
        candidate_id=candidate.candidate_id,
    )
    upstream = tmp_path / "upstream"
    (upstream / "core").mkdir(parents=True)
    (upstream / "agents").mkdir()
    (upstream / "core" / "eval_runner.py").write_text("# synthetic\n", encoding="utf-8")
    (upstream / "agents" / "__init__.py").write_text("# synthetic\n", encoding="utf-8")
    return SimpleNamespace(
        previous=previous,
        current=current,
        previous_zip=previous_zip,
        current_zip=current_zip,
        pack=pack,
        view=view,
        private_pack=private_pack,
        candidate=candidate,
        authorization=auth,
        upstream=upstream,
    )


def test_authorized_prepare_claims_before_read_and_materializes_exact_four(
    synthetic_sealed: SimpleNamespace,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal"
    preparation_root = tmp_path / "preparation"
    receipt = sealed_prepare.prepare_sealed_partition_v1(
        private_pack_path=synthetic_sealed.private_pack,
        measurement_view=synthetic_sealed.view,
        authorization=synthetic_sealed.authorization,
        journal_root=journal,
        previous_source=synthetic_sealed.previous,
        current_source=synthetic_sealed.current,
        output_root=preparation_root,
        study_id=sealed_freeze.STUDY_ID,
        candidate_id=synthetic_sealed.candidate.candidate_id,
    )
    assert (journal / sealed_access.CLAIM_FILENAME).is_file()
    assert (journal / sealed_access.COMPLETION_FILENAME).is_file()
    assert not (journal / sealed_access.FAILURE_FILENAME).exists()
    completion = period_pack.read_json(journal / sealed_access.COMPLETION_FILENAME)
    assert completion["raw_file_sha256_matches_precommit"] is True
    assert completion["verified_public_pack_hash_matches_commitment"] is True
    assert receipt["oracle_call_count"] == 2
    assert receipt["cross_oracle_agreement"] is True
    assert receipt["access_journal"]["claim_hash"] == period_pack.read_json(
        journal / sealed_access.CLAIM_FILENAME
    )["claim_hash"]
    assert receipt["access_journal"]["completion_hash"] == completion["completion_hash"]

    payload = period_pack.read_json(preparation_root / sealed_prepare.SEALED_PAYLOAD_FILENAME)
    gold = period_pack.read_json(preparation_root / sealed_prepare.SEALED_GOLD_FILENAME)
    benchmark = tmp_path / "sealed-benchmark"
    report = sealed_materialize.materialize_sealed_benchmark_v1(
        upstream_benchmark_root=synthetic_sealed.upstream,
        measurement_view=synthetic_sealed.view,
        sealed_payload=payload,
        sealed_gold=gold,
        previous_archive=synthetic_sealed.previous_zip,
        current_archive=synthetic_sealed.current_zip,
        output_root=benchmark,
    )
    assert report["item_count"] == 4
    assert report["sealed_task_count_materialized"] == 4
    assert report["measurement_task_count_materialized"] == 0
    assert report["sealed_content_persisted_in_report"] is False
    task_dirs = list((benchmark / "tasks" / "financial-analysis").iterdir())
    assert len(task_dirs) == 4
    assert all(
        (task / "tests" / "expected_output.json").stat().st_mode & 0o777
        == 0o644
        for task in task_dirs
    )
    formal_rows = [
        {"item_id_hash": payload_hash({"item_id": item["item_id"]})}
        for item in payload["sealed_items"]
    ]
    prewarm_body = {
        "prewarm_version": sealed_prewarm.PREWARM_VERSION,
        "sealed_payload_hash": payload["sealed_payload_hash"],
        "benchmark_tree_hash": report["benchmark_tree_hash"],
        "formal_execution_cache_only": True,
        "formal_image_cache_only": True,
        "formal_offline_verifier_cache_only": True,
        "formal_verifier_network": "none",
        "model_calls": 0,
        "online_judge_calls": 0,
        "formal_cache_rows": formal_rows,
    }
    prewarm = {**prewarm_body, "prewarm_hash": payload_hash(prewarm_body)}
    freeze = {
        "materialization_hash": report["materialization_hash"],
        "prewarm_hash": prewarm["prewarm_hash"],
        "sealed_gold_hash": "0" * 64,
    }
    with pytest.raises(sealed_runner.SealedRunnerError):
        sealed_runner._validate_inputs(
            benchmark=benchmark,
            payload=payload,
            prewarm=prewarm,
            freeze=freeze,
        )
    freeze["sealed_gold_hash"] = report["sealed_gold_hash"]
    assert len(
        sealed_runner._validate_inputs(
            benchmark=benchmark,
            payload=payload,
            prewarm=prewarm,
            freeze=freeze,
        )
    ) == 4


def test_access_hash_mismatch_consumes_claim_without_completion(
    synthetic_sealed: SimpleNamespace,
    tmp_path: Path,
) -> None:
    auth = json.loads(json.dumps(synthetic_sealed.authorization))
    body = dict(auth)
    body.pop("manifest_hash")
    body["sealed_cohort_binding"]["precommitted_private_pack_file_sha256"] = "0" * 64
    auth = {**body, "manifest_hash": payload_hash(body)}
    journal = tmp_path / "failed-journal"
    with pytest.raises(sealed_access.SealedAccessError):
        sealed_access.read_authorized_private_pack_v1(
            unresolved_private_pack_path=synthetic_sealed.private_pack,
            journal_root=journal,
            authorization=auth,
            expected_study_id=sealed_freeze.STUDY_ID,
            expected_private_pack_hash=synthetic_sealed.pack["pack_hash"],
            expected_measurement_view_hash=synthetic_sealed.view["measurement_view_hash"],
            expected_candidate_id=synthetic_sealed.candidate.candidate_id,
        )
    assert (journal / sealed_access.CLAIM_FILENAME).is_file()
    assert (journal / sealed_access.FAILURE_FILENAME).is_file()
    assert not (journal / sealed_access.COMPLETION_FILENAME).exists()
    with pytest.raises(sealed_access.SealedAccessError):
        sealed_access.read_authorized_private_pack_v1(
            unresolved_private_pack_path=synthetic_sealed.private_pack,
            journal_root=journal,
            authorization=auth,
            expected_study_id=sealed_freeze.STUDY_ID,
            expected_private_pack_hash=synthetic_sealed.pack["pack_hash"],
            expected_measurement_view_hash=synthetic_sealed.view["measurement_view_hash"],
            expected_candidate_id=synthetic_sealed.candidate.candidate_id,
        )


def _treatment(candidate: Any) -> FixedPeriodOutTreatmentV2:
    return FixedPeriodOutTreatmentV2(
        recipe_id=candidate.recipe_id,
        program_set_hash=candidate.program_set_hash,
        period_out_treatment_id="e" * 64,
        external_skill_source_receipt_hash=candidate.external_skill_source_receipt_hash,
        candidate_skill_source=candidate.candidate_skill_source,
    )


def test_sealed_plan_submits_eight_backends_at_one_barrier(
    synthetic_sealed: SimpleNamespace,
) -> None:
    plan = sealed_plan.build_sealed_plan_v1(
        targets=tuple(sealed_plan.SealedTargetV1(f"sealed-{index}", index) for index in range(4)),
        manifest_hash="a" * 64,
        evaluator_epoch="sealed-test",
        treatment=_treatment(synthetic_sealed.candidate),
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        codex_agent_execution_policy_hash="b" * 64,
    )
    inner = threading.Barrier(8)

    class Backend:
        def run(self, request: Any, *, skill_source_dir: Path | None, trace_id: str) -> object:
            inner.wait()
            return object()

    execution = sealed_plan.execute_sealed_plan_v1(
        plan=plan,
        backend_factory=lambda _work: Backend(),
    )
    assert execution.safe_payload()["physical_execution_count"] == 8
    assert execution.maximum_active_backend_calls == 8
    assert plan.safe_payload()["model_inference_slots"] == 8
    assert plan.safe_payload()["retry_count"] == 0
    assert plan.safe_payload()["split"] == "test"


def test_all_eight_containers_disconnect_before_any_verifier_materializes(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    disconnected: set[str] = set()
    materialized: list[str] = []

    class Host:
        def run(self, command: list[str], **_kwargs: Any) -> SimpleNamespace:
            if command[:3] == ["docker", "network", "disconnect"]:
                with lock:
                    disconnected.add(command[-1])
                return SimpleNamespace(returncode=0, stdout="")
            if command[:2] == ["docker", "inspect"]:
                return SimpleNamespace(returncode=0, stdout="{}\n")
            raise AssertionError(command)

    class DockerProxy:
        def __init__(self, host: Host) -> None:
            self.delegate = host
            self._verifier_sources = {}
            self.egress_policy = SimpleNamespace(network_name="sealed-network")

        def run(self, command: list[str], *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            with lock:
                assert len(disconnected) == 8
                materialized.append(command[2])
            return SimpleNamespace(returncode=0, stdout="")

    class OperatorProxy:
        def __init__(self, delegate: DockerProxy) -> None:
            self.delegate = delegate

        def run(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("sealed proxy must bypass the arm proxy after checkpoint")

    class Backend:
        def __init__(self, index: int) -> None:
            self.durable_arm = "raw" if index < 4 else "candidate"
            self.durable_work_unit_hash = payload_hash({"work": index})
            self.durable_request_hash = payload_hash({"request": index})
            self.durable_state_root = tmp_path / f"state-{index}"
            self.durable_state_root.mkdir()
            self._sealed_network_isolation_receipt_hash = None

        def _checkpoint_raw_before_verifier_v2(self) -> None:
            assert self.durable_arm == "raw"

        def _execute_contract_plan_before_verifier_v2(
            self, *, delegate: Any, container_name: str
        ) -> None:
            assert self.durable_arm == "candidate"
            assert isinstance(delegate, DockerProxy)
            assert container_name.startswith("container-")

    host = Host()
    docker = DockerProxy(host)
    coordinator = sealed_runner._SealedVerifierIsolationCoordinatorV1()
    proxies = []
    backends = []
    for index in range(8):
        backend = Backend(index)
        backends.append(backend)
        chain = docker if index < 4 else OperatorProxy(docker)
        proxies.append(
            sealed_runner._SealedOfflineVerifierIsolationProxyV1(
                chain,
                backend=backend,
                network_name="sealed-network",
                coordinator=coordinator,
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = tuple(
            executor.submit(
                proxy.run,
                ["docker", "exec", f"container-{index}", "/tests/test.sh"],
            )
            for index, proxy in enumerate(proxies)
        )
        for future in futures:
            assert future.result().returncode == 0
    assert coordinator.complete is True
    assert len(disconnected) == len(materialized) == 8
    receipts = [
        period_pack.read_json(
            backend.durable_state_root / sealed_runner.NETWORK_ISOLATION_FILENAME
        )
        for backend in backends
    ]
    assert len({row["receipt_hash"] for row in receipts}) == 8
    assert all(row["before_attached_network_count"] == 0 for row in receipts)
    assert all(row["after_attached_network_count"] == 0 for row in receipts)
    assert all(row["tests_materialized_before_disconnect"] is False for row in receipts)


def test_runner_binds_materialized_gold_to_execution_freeze(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = {
        "sealed_payload_hash": "a" * 64,
        "sealed_items": [
            {"item_id": f"sealed-{index}"} for index in range(4)
        ],
    }
    materialization_body = {
        "sealed_payload_hash": payload["sealed_payload_hash"],
        "sealed_gold_hash": "b" * 64,
        "benchmark_tree_hash": "c" * 64,
    }
    materialization = {
        **materialization_body,
        "materialization_hash": payload_hash(materialization_body),
    }
    formal_rows = [
        {"item_id_hash": payload_hash({"item_id": item["item_id"]})}
        for item in payload["sealed_items"]
    ]
    prewarm_body = {
        "prewarm_version": sealed_prewarm.PREWARM_VERSION,
        "sealed_payload_hash": payload["sealed_payload_hash"],
        "benchmark_tree_hash": materialization["benchmark_tree_hash"],
        "formal_execution_cache_only": True,
        "formal_image_cache_only": True,
        "formal_offline_verifier_cache_only": True,
        "formal_verifier_network": "none",
        "model_calls": 0,
        "online_judge_calls": 0,
        "formal_cache_rows": formal_rows,
    }
    prewarm = {**prewarm_body, "prewarm_hash": payload_hash(prewarm_body)}
    freeze = {
        "materialization_hash": materialization["materialization_hash"],
        "prewarm_hash": prewarm["prewarm_hash"],
        "sealed_gold_hash": "0" * 64,
    }
    monkeypatch.setattr(sealed_runner, "read_json", lambda _path: materialization)
    monkeypatch.setattr(
        sealed_runner,
        "sealed_benchmark_tree_receipt_v1",
        lambda _root: {"tree_hash": materialization["benchmark_tree_hash"]},
    )
    with pytest.raises(sealed_runner.SealedRunnerError):
        sealed_runner._validate_inputs(
            benchmark=tmp_path,
            payload=payload,
            prewarm=prewarm,
            freeze=freeze,
        )
    freeze["sealed_gold_hash"] = materialization["sealed_gold_hash"]
    assert len(
        sealed_runner._validate_inputs(
            benchmark=tmp_path,
            payload=payload,
            prewarm=prewarm,
            freeze=freeze,
        )
    ) == 4


@pytest.mark.parametrize(
    ("disposition", "expected_returncode"),
    (("executed_complete", 0), ("executed_incomplete_no_retry", 3)),
)
def test_runner_main_returns_zero_only_for_executed_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    disposition: str,
    expected_returncode: int,
) -> None:
    args = SimpleNamespace(
        project_root=tmp_path,
        benchmark_root=tmp_path / "benchmark",
        measurement_view=tmp_path / "view.json",
        sealed_payload=tmp_path / "payload.json",
        prewarm=tmp_path / "prewarm.json",
        execution_freeze=tmp_path / "freeze.json",
        env_file=tmp_path / "provider.env",
        output_root=tmp_path / "output",
    )
    monkeypatch.setattr(
        sealed_runner,
        "_parser",
        lambda: SimpleNamespace(parse_args=lambda _argv: args),
    )
    monkeypatch.setattr(
        sealed_runner,
        "load_fixed_contract_candidate_v2",
        lambda _project: object(),
    )
    monkeypatch.setattr(sealed_runner, "read_json", lambda _path: {})
    monkeypatch.setattr(
        sealed_runner, "validate_sealed_execution_freeze_v1", lambda *_args, **_kwargs: "f" * 64
    )
    monkeypatch.setattr(
        sealed_runner,
        "run_sealed_v1",
        lambda **_kwargs: {
            "report_hash": "e" * 64,
            "evidence_valid": disposition == "executed_complete",
            "disposition": disposition,
        },
    )
    assert sealed_runner.main([]) == expected_returncode


def test_invalid_pair_is_not_counted_as_tie_gain_or_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class Observation:
        def __init__(self, *, valid: bool, success: bool, name: str) -> None:
            self.valid = valid
            self.success = success
            self.observation_hash = payload_hash({"name": name})

    monkeypatch.setattr(sealed_runner, "SkillLearnTrialObservation", Observation)
    target = sealed_plan.SealedTargetV1("sealed-0", 0)
    pair = SimpleNamespace(
        target=target,
        pair_id="pair",
        raw_observation=Observation(valid=False, success=True, name="raw"),
        candidate_observation=Observation(valid=True, success=True, name="candidate"),
    )
    result = sealed_runner._sealed_descriptive_results(SimpleNamespace(pair_results=(pair,)))
    assert result["invalid_pair_count"] == 1
    assert result["valid_pair_count"] == 0
    assert result["raw_successes"] == result["candidate_successes"] == 0
    assert result["gain_count"] == result["harm_count"] == result["tie_count"] == 0
    assert result["pairs"][0]["relation"] == "invalid"
    assert result["pairs"][0]["delta"] is None
    assert (
        sealed_runner._sealed_terminal_disposition_v1(result)
        == "executed_incomplete_no_retry"
    )
    assert (
        sealed_runner._sealed_terminal_disposition_v1(
            {"valid_pair_count": 4, "invalid_pair_count": 0}
        )
        == "executed_complete"
    )


def test_freeze_binds_four_pairs_plus_only_and_one_shot_launcher(
    synthetic_sealed: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_root = tmp_path / "prep"
    preparation = sealed_prepare.prepare_sealed_partition_v1(
        private_pack_path=synthetic_sealed.private_pack,
        measurement_view=synthetic_sealed.view,
        authorization=synthetic_sealed.authorization,
        journal_root=tmp_path / "journal",
        previous_source=synthetic_sealed.previous,
        current_source=synthetic_sealed.current,
        output_root=preparation_root,
        study_id=sealed_freeze.STUDY_ID,
        candidate_id=synthetic_sealed.candidate.candidate_id,
    )
    payload = period_pack.read_json(preparation_root / sealed_prepare.SEALED_PAYLOAD_FILENAME)
    gold = period_pack.read_json(preparation_root / sealed_prepare.SEALED_GOLD_FILENAME)
    benchmark = tmp_path / "benchmark"
    materialization = sealed_materialize.materialize_sealed_benchmark_v1(
        upstream_benchmark_root=synthetic_sealed.upstream,
        measurement_view=synthetic_sealed.view,
        sealed_payload=payload,
        sealed_gold=gold,
        previous_archive=synthetic_sealed.previous_zip,
        current_archive=synthetic_sealed.current_zip,
        output_root=benchmark,
    )
    prewarm_body = {
        "prewarm_version": sealed_prewarm.PREWARM_VERSION,
        "sealed_payload_hash": payload["sealed_payload_hash"],
        "materialization_hash": materialization["materialization_hash"],
        "benchmark_tree_hash": materialization["benchmark_tree_hash"],
        "item_count": 4,
        "formal_execution_cache_only": True,
    }
    prewarm = {**prewarm_body, "prewarm_hash": payload_hash(prewarm_body)}
    protocol = PaperProtocol.read(PROJECT / V320_PROTOCOL_RELATIVE_PATH)
    provider = {
        "provider_label": "plus",
        "binding_hash": "d" * 64,
        "model": protocol.payload["model"],
    }
    launcher_source = PROJECT / "scripts" / "launch_tmux_detached_sealed_once.py"
    supplemental = [
        {
            "relative_path": "scripts/launch_tmux_detached_sealed_once.py",
            "file_sha256": period_pack.sha256_file(launcher_source),
        }
    ]
    closure_body = {
        "closure_version": sealed_freeze.SOURCE_CLOSURE_VERSION,
        "scope_policy": "entire_runtime_closure_plus_committed_launchers_v1",
        "runtime_source_closure": {
            "source_commit": "c" * 40,
            "closure_hash": "b" * 64,
        },
        "supplemental_files": supplemental,
        "supplemental_file_count": 1,
        "supplemental_file_set_hash": payload_hash(supplemental),
        "source_commit": "c" * 40,
    }
    synthetic_closure = {
        **closure_body,
        "closure_hash": payload_hash(closure_body),
    }
    monkeypatch.setattr(
        sealed_freeze,
        "build_sealed_source_closure_v1",
        lambda _project: synthetic_closure,
    )
    freeze = sealed_freeze.build_sealed_execution_freeze_v1(
        project_root=PROJECT,
        measurement_view=synthetic_sealed.view,
        authorization=synthetic_sealed.authorization,
        preparation=preparation,
        sealed_payload=payload,
        materialization=materialization,
        prewarm=prewarm,
        provider_binding=provider,
        candidate=synthetic_sealed.candidate,
    )
    assert freeze["execution_policy"]["physical_model_calls"] == 8
    assert freeze["execution_policy"]["outer_workers"] == 8
    assert freeze["execution_policy"]["provider_label"] == "plus"
    assert freeze["execution_policy"]["failure_disposition"] == "executed_incomplete_no_retry"
    assert freeze["sealed_content_persisted_in_freeze"] is False
    assert freeze["sealed_access"] == preparation["access_journal"]

    launcher_path = PROJECT / "scripts" / "launch_tmux_detached_sealed_once.py"
    spec = importlib.util.spec_from_file_location("sealed_launcher_test", launcher_path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT / "scripts"))
    try:
        spec.loader.exec_module(launcher)
    finally:
        sys.path.remove(str(PROJECT / "scripts"))
    output = tmp_path / "formal-output"
    mini_repo = tmp_path / "committed-freeze-repo"
    mini_repo.mkdir()
    freeze_path = mini_repo / "freeze.json"
    period_pack.write_json(freeze_path, freeze)
    subprocess.run(["git", "init", "-q"], cwd=mini_repo, check=True)
    subprocess.run(["git", "add", "freeze.json"], cwd=mini_repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=sealed-test",
            "-c",
            "user.email=sealed-test@example.invalid",
            "commit",
            "-qm",
            "freeze",
        ],
        cwd=mini_repo,
        check=True,
    )
    command = [
        sys.executable,
        "-B",
        "-m",
        launcher.RUNNER_MODULE,
        "--project-root", str(PROJECT),
        "--benchmark-root", str(benchmark),
        "--measurement-view", str(tmp_path / "view.json"),
        "--sealed-payload", str(preparation_root / sealed_prepare.SEALED_PAYLOAD_FILENAME),
        "--prewarm", str(tmp_path / "prewarm.json"),
        "--execution-freeze", str(freeze_path),
        "--env-file", str(tmp_path / "provider.env"),
        "--output-root", str(output),
    ]
    assert launcher._safe_command(command, working_directory=PROJECT, output_root=output.resolve())[1:4] == ["-B", "-m", launcher.RUNNER_MODULE]
    identity = launcher._freeze_identity(command, mini_repo)
    assert identity["sealed_execution_freeze_hash"] == freeze["manifest_hash"]
    assert len(identity["sealed_execution_freeze_committed_at_git_commit"]) == 40
    with pytest.raises(Exception):
        launcher._safe_command(command + ["--retry"], working_directory=PROJECT, output_root=output.resolve())
