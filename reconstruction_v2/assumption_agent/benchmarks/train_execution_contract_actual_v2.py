from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..events import JsonlEventSink, MemoryEventSink
from ..models import stable_hash
from ..secure_env import (
    configured_api_origin,
    configured_model,
    configured_skilllearn_provider_mode,
)
from ..splits import SplitManifest
from .docker_egress import (
    DockerEgressPolicy,
    configured_trial_network_byte_limit,
)
from .execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
    ExecutionContractSubprocessBackendV2,
)
from .offline_verifier import SkillLearnOfflineVerifierRuntimeCache
from .paper_protocol import PaperProtocol
from .prewarm import (
    FrozenTaskInputPrebuiltImageCache,
    validate_development_prewarm_receipt,
)
from .skilllearn_lifecycle import (
    SkillLearnModelInferenceLimiter,
    SkillLearnProviderCircuit,
    SkillLearnSubprocessBackend,
)
from .task_input_freeze import (
    FrozenTaskInputClosure,
    expected_prewarm_closure_rows,
    load_frozen_task_input_closure,
)
from .train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
    V320_MANIFEST_RELATIVE_PATH,
    TrainExecutionContractIntegrationV2,
    compile_v320_train_execution_contract_candidates_v2,
)
from .train_outcome_production_runner_v2 import (
    ProductionTrainCandidateRunnerV2,
)
from .train_outcome_ranker_v2 import (
    FrozenRawTrainBaselineSetV2,
    TrainCandidateWorkUnitV2,
    TrainOutcomeRankerV2,
    TrainOutcomeRankingResultV2,
)
from .v320_train_candidate_material_v2 import (
    V320_EVALUATOR_EPOCH,
    V320_MANIFEST_HASH,
    V320_MODEL,
    V320_SOURCE_RELATIVE_ROOT,
)


TRAIN_EXECUTION_CONTRACT_ACTUAL_VERSION = (
    "v320_train_execution_contract_actual_offline_ranking_v2"
)
V320_PROTOCOL_RELATIVE_PATH = (
    "manifests/skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json"
)
V320_PREWARM_RELATIVE_PATH = (
    f"{V320_SOURCE_RELATIVE_ROOT}/development_prewarm.json"
)
ACTUAL_REPORT_FILENAME = "ranking.report.json"
FAILURE_REPORT_FILENAME = "ranking.failure.json"
ASSET_PREFLIGHT_FILENAME = "asset_preflight.report.json"
EXECUTION_EVENTS_FILENAME = "execution.events.jsonl"
OUTER_WORKERS = 56
MODEL_INFERENCE_SLOTS = 48


class TrainExecutionContractActualError(PermissionError):
    """The actual TRAIN ranking crossed a frozen execution boundary."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrainExecutionContractActualError(
            f"{label} is not a regular file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainExecutionContractActualError(
            f"{label} is not readable JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TrainExecutionContractActualError(
            f"{label} is not an object"
        )
    return value


def _configure_environment(protocol: PaperProtocol) -> None:
    payload = protocol.payload
    execution = payload["execution"]
    assert isinstance(execution, Mapping)
    expected_origin = str(payload["provider_endpoint_origin"])
    configured_origin = configured_api_origin()
    if configured_origin and configured_origin != expected_origin:
        raise TrainExecutionContractActualError(
            "configured provider origin differs from v3.20"
        )
    if not os.environ.get("ASSUMPTION_V2_API_KEY", "").strip():
        raise TrainExecutionContractActualError(
            "TRAIN ranking provider key is absent"
        )
    os.environ["ASSUMPTION_V2_API_BASE"] = expected_origin
    os.environ["ASSUMPTION_V2_MODEL"] = str(payload["model"])
    os.environ["ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE"] = str(
        payload["trial_provider_mode"]
    )
    os.environ["ASSUMPTION_V2_API_ALLOWED_IPV4S"] = ",".join(
        str(value) for value in payload["provider_endpoint_ipv4s"]
    )
    os.environ["ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY"] = "1"
    os.environ["ASSUMPTION_V2_TRIAL_NETWORK_BYTE_LIMIT"] = str(
        execution["trial_network_byte_limit"]
    )
    if (
        configured_model() != V320_MODEL
        or configured_skilllearn_provider_mode()
        != payload["trial_provider_mode"]
        or configured_api_origin() != expected_origin
        or configured_trial_network_byte_limit()
        != execution["trial_network_byte_limit"]
    ):
        raise TrainExecutionContractActualError(
            "TRAIN ranking provider contract drifted"
        )
    egress = DockerEgressPolicy.from_env()
    if (
        egress.endpoint_origin != expected_origin
        or tuple(egress.allowed_ipv4s)
        != tuple(payload["provider_endpoint_ipv4s"])
    ):
        raise TrainExecutionContractActualError(
            "TRAIN ranking egress authority drifted"
        )


def _verify_canary(
    path: Path,
    *,
    provider_label: str,
) -> dict[str, Any]:
    payload = _read_json(path, "provider canary report")
    if (
        provider_label not in {"plus", "pro"}
        or payload.get("canary_version") != "proposal_canary_v1"
        or payload.get("model") != V320_MODEL
        or payload.get("provider_chain") != ["openai_compatible"]
        or payload.get("accepted") is not True
        or payload.get("api_key_present") is not True
        or payload.get("secret_value_persisted") is not False
        or payload.get("raw_content_persisted") is not False
    ):
        raise TrainExecutionContractActualError(
            "provider canary did not authorize this process start"
        )
    return {
        "provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "canary_file_sha256": _sha256_file(path),
        "canary_payload_hash": stable_hash(payload),
        "canary_accepted": True,
        "secret_value_persisted": False,
    }


def _scoped_frozen_inputs(
    frozen: FrozenTaskInputClosure,
    *,
    active_item_hashes: set[str],
) -> FrozenTaskInputClosure:
    ledger = {
        key: value
        for key, value in frozen.ledger_by_item_hash.items()
        if key in active_item_hashes
    }
    return FrozenTaskInputClosure(
        source=frozen.source,
        receipt=frozen.receipt,
        receipt_path=frozen.receipt_path,
        ledger_by_item_hash=ledger,
    )


@dataclass(frozen=True)
class _RuntimeAssets:
    prebuilt_cache: FrozenTaskInputPrebuiltImageCache = field(
        compare=False,
        repr=False,
    )
    offline_cache: SkillLearnOfflineVerifierRuntimeCache = field(
        compare=False,
        repr=False,
    )
    provider_circuit: SkillLearnProviderCircuit = field(
        compare=False,
        repr=False,
    )
    model_limiter: SkillLearnModelInferenceLimiter = field(
        compare=False,
        repr=False,
    )
    preflight_report: Mapping[str, Any]


def _prepare_scoped_runtime_assets(
    *,
    project_root: Path,
    destination: Path,
    protocol: PaperProtocol,
    manifest: SplitManifest,
    baseline_set: FrozenRawTrainBaselineSetV2,
    active_item_hashes: set[str],
    expected_active_item_count: int,
    preflight_policy: str,
    event_sink: JsonlEventSink,
    task_input_cache_root: Path | None,
) -> _RuntimeAssets:
    benchmark_root = (
        project_root / SKILLLEARN_BENCHMARK_RELATIVE_ROOT
    ).resolve(strict=True)
    prewarm_path = (project_root / V320_PREWARM_RELATIVE_PATH).resolve(
        strict=True
    )
    prewarm = _read_json(prewarm_path, "v3.20 prewarm receipt")
    frozen = load_frozen_task_input_closure(
        protocol.payload,
        project_root=project_root,
    )
    if frozen is None:
        raise TrainExecutionContractActualError(
            "v3.20 frozen task-input closure is unavailable"
        )
    execution = protocol.payload["execution"]
    assert isinstance(execution, Mapping)
    validate_development_prewarm_receipt(
        prewarm,
        manifest=manifest,
        expected_version=str(execution["development_prewarm"]),
        frozen_task_inputs=frozen,
    )

    active_hashes = set(active_item_hashes)
    baseline_by_hash = {
        row.item_id_hash: row
        for row in baseline_set.rows
    }
    prewarm_by_hash = {
        str(row["item_id_hash"]): row
        for row in prewarm.get("items", ())
        if isinstance(row, Mapping)
    }
    if (
        not isinstance(expected_active_item_count, int)
        or isinstance(expected_active_item_count, bool)
        or expected_active_item_count <= 0
        or not isinstance(preflight_policy, str)
        or not preflight_policy
        or len(active_hashes) != expected_active_item_count
        or not active_hashes <= set(baseline_by_hash)
        or not active_hashes <= set(prewarm_by_hash)
    ):
        raise TrainExecutionContractActualError(
            "active TRAIN local-asset coverage drifted"
        )
    scoped_frozen = _scoped_frozen_inputs(
        frozen,
        active_item_hashes=active_hashes,
    )
    prebuilt_cache = FrozenTaskInputPrebuiltImageCache(
        benchmark_root,
        frozen_task_inputs=scoped_frozen,
        expected_prewarm_rows=expected_prewarm_closure_rows(prewarm),
        cache_only=True,
        event_sink=event_sink,
        task_input_cache_root=task_input_cache_root,
    )
    offline_cache = SkillLearnOfflineVerifierRuntimeCache(
        event_sink=event_sink
    )
    provider_circuit = SkillLearnProviderCircuit()
    model_limiter = SkillLearnModelInferenceLimiter(MODEL_INFERENCE_SLOTS)
    preflight_sink = MemoryEventSink()

    def resolve(item_hash: str) -> dict[str, Any]:
        baseline = baseline_by_hash[item_hash]
        observation = baseline.observation
        prewarm_row = prewarm_by_hash[item_hash]
        backend = SkillLearnSubprocessBackend(
            benchmark_root,
            agent_id=observation.request.agent_id,
            model=observation.request.model,
            max_steps=observation.request.max_steps,
            provider_mode=str(protocol.payload["trial_provider_mode"]),
            record_upstream=False,
            prebuilt_cache=prebuilt_cache,
            offline_verifier_cache=offline_cache,
            provider_circuit=provider_circuit,
            model_inference_limiter=model_limiter,
            train_action_design_policy=str(
                execution["train_action_design_policy"]
            ),
            codex_agent_execution_policy=protocol.codex_agent_execution_policy,
            event_sink=preflight_sink,
        )
        image, runtime = backend.prewarm_trial_environment(
            family=baseline.family,
            item_id=baseline.item_id,
            trace_id=f"v320-contract-asset-preflight:{item_hash[:20]}",
        )
        if (
            image.reused is not True
            or image.image_id != observation.prebuilt_image_id
            or image.cache_key != observation.prebuilt_image_key
            or image.image_id != prewarm_row.get("prebuilt_image_id")
            or image.cache_key != prewarm_row.get("prebuilt_image_key")
            or runtime is None
            or runtime.profile.profile_id
            != observation.offline_verifier_profile_id
            or runtime.runtime_key
            != observation.offline_verifier_runtime_key
            or runtime.profile.profile_id
            != prewarm_row.get("offline_verifier_profile_id")
            or runtime.runtime_key
            != prewarm_row.get("offline_verifier_runtime_key")
        ):
            raise TrainExecutionContractActualError(
                "active TRAIN local asset differs from frozen RAW"
            )
        return {
            "item_id_hash": item_hash,
            "family_hash": baseline.family_hash,
            "prebuilt_image_key_hash": stable_hash(
                {"prebuilt_image_key": image.cache_key}
            ),
            "prebuilt_image_id_hash": stable_hash(
                {"prebuilt_image_id": image.image_id}
            ),
            "offline_verifier_profile_id_hash": stable_hash(
                {"offline_verifier_profile_id": runtime.profile.profile_id}
            ),
            "offline_verifier_runtime_key_hash": stable_hash(
                {"offline_verifier_runtime_key": runtime.runtime_key}
            ),
            "prebuilt_cache_reused": True,
            "model_calls": 0,
            "raw_content_persisted": False,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        rows = tuple(executor.map(resolve, sorted(active_hashes)))
    if model_limiter.maximum_active != 0 or provider_circuit.error_type is not None:
        raise TrainExecutionContractActualError(
            "local asset preflight crossed the no-model boundary"
        )
    report_without_hash = {
        "preflight_policy": preflight_policy,
        "passed": True,
        "active_item_count": len(rows),
        "rows": list(rows),
        "row_set_hash": stable_hash({"rows": list(rows)}),
        "model_calls": 0,
        "evaluator_calls": 0,
        "online_judge_calls": 0,
        "validation_accessed": False,
        "test_accessed": False,
        "raw_content_persisted": False,
    }
    report = {
        **report_without_hash,
        "report_hash": stable_hash(report_without_hash),
    }
    (destination / ASSET_PREFLIGHT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _RuntimeAssets(
        prebuilt_cache=prebuilt_cache,
        offline_cache=offline_cache,
        provider_circuit=provider_circuit,
        model_limiter=model_limiter,
        preflight_report=report,
    )


def _prepare_runtime_assets(
    *,
    project_root: Path,
    destination: Path,
    protocol: PaperProtocol,
    manifest: SplitManifest,
    integration: TrainExecutionContractIntegrationV2,
    event_sink: JsonlEventSink,
    task_input_cache_root: Path | None,
) -> _RuntimeAssets:
    active_hashes = {
        route.item_id_hash
        for candidate in integration.candidate_specs
        for route in candidate.item_routes
    }
    return _prepare_scoped_runtime_assets(
        project_root=project_root,
        destination=destination,
        protocol=protocol,
        manifest=manifest,
        baseline_set=integration.raw_projection.baseline_set,
        active_item_hashes=active_hashes,
        expected_active_item_count=7,
        preflight_policy=(
            "v320_active_train_local_asset_exact_reuse_preflight_v2"
        ),
        event_sink=event_sink,
        task_input_cache_root=task_input_cache_root,
    )


@dataclass(frozen=True)
class TrainExecutionContractActualV2:
    output_root: Path = field(compare=False)
    integration: TrainExecutionContractIntegrationV2 = field(
        compare=False,
        repr=False,
    )
    ranking: TrainOutcomeRankingResultV2 = field(
        compare=False,
        repr=False,
    )
    report: Mapping[str, Any]

    @property
    def report_path(self) -> Path:
        return self.output_root / ACTUAL_REPORT_FILENAME

    def verify(self) -> None:
        self.integration.verify()
        self.ranking.verify()
        path = self.report_path
        persisted = _read_json(path, "TRAIN ranking report")
        without_hash = dict(persisted)
        report_hash = without_hash.pop("report_hash", None)
        if (
            persisted != dict(self.report)
            or report_hash != stable_hash(without_hash)
            or persisted.get("execution_completed") is not True
            or persisted.get("ranking_hash") != self.ranking.ranking_hash
            or persisted.get("validation_accessed") is not False
            or persisted.get("test_accessed") is not False
            or persisted.get("online_judge_calls") != 0
        ):
            raise TrainExecutionContractActualError(
                "TRAIN ranking report drifted"
            )


def run_v320_train_execution_contract_actual_v2(
    *,
    project_root: Path,
    output_root: Path,
    canary_report_path: Path,
    provider_label: str,
    task_input_cache_root: Path | None = None,
) -> TrainExecutionContractActualV2:
    """Execute all 56 active routes in one flat, bounded parallel pool."""

    project = project_root.resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("TRAIN ranking output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
        if (
            protocol.payload.get("protocol_version") != "3.20.0"
            or protocol.payload.get("model") != V320_MODEL
            or protocol.payload.get("max_steps") != 100
        ):
            raise TrainExecutionContractActualError(
                "v3.20 execution protocol drifted"
            )
        _configure_environment(protocol)
        canary = _verify_canary(
            canary_report_path.resolve(strict=True),
            provider_label=provider_label,
        )
        manifest = SplitManifest.read(project / V320_MANIFEST_RELATIVE_PATH)
        if manifest.manifest_hash != V320_MANIFEST_HASH:
            raise TrainExecutionContractActualError(
                "v3.20 execution manifest drifted"
            )
        integration = compile_v320_train_execution_contract_candidates_v2(
            project_root=project,
            output_root=destination / "compile_integration",
        )
        event_sink = JsonlEventSink(destination / EXECUTION_EVENTS_FILENAME)
        assets = _prepare_runtime_assets(
            project_root=project,
            destination=destination,
            protocol=protocol,
            manifest=manifest,
            integration=integration,
            event_sink=event_sink,
            task_input_cache_root=task_input_cache_root,
        )
        benchmark_root = (
            project / SKILLLEARN_BENCHMARK_RELATIVE_ROOT
        ).resolve(strict=True)
        execution = protocol.payload["execution"]
        assert isinstance(execution, Mapping)
        trials_root = destination / "worker_state"

        def backend_factory(
            work: TrainCandidateWorkUnitV2,
            bundle: ExecutionContractCompileBundleV2,
        ) -> ExecutionContractSubprocessBackendV2:
            baseline_request = work.baseline.observation.request
            return ExecutionContractSubprocessBackendV2(
                benchmark_root,
                agent_id=baseline_request.agent_id,
                model=baseline_request.model,
                max_steps=baseline_request.max_steps,
                provider_mode=str(protocol.payload["trial_provider_mode"]),
                trials_dir=trials_root / work.work_unit_hash,
                record_upstream=False,
                prebuilt_cache=assets.prebuilt_cache,
                offline_verifier_cache=assets.offline_cache,
                provider_circuit=assets.provider_circuit,
                model_inference_limiter=assets.model_limiter,
                train_action_design_policy=str(
                    execution["train_action_design_policy"]
                ),
                codex_agent_execution_policy=(
                    protocol.codex_agent_execution_policy
                ),
                event_sink=event_sink,
                execution_contract_bundle=bundle,
            )

        production_runner = ProductionTrainCandidateRunnerV2(
            baseline_set=integration.raw_projection.baseline_set,
            candidate_bundles=integration.candidate_bundles_by_hash,
            backend_factory=backend_factory,
            trace_prefix="v320-train-contract-actual02",
        )
        ranking = TrainOutcomeRankerV2(max_workers=OUTER_WORKERS).rank(
            baseline_set=integration.raw_projection.baseline_set,
            candidates=integration.candidate_specs,
            runner=production_runner,
        )
        ranking.verify()
        if (
            len(ranking.run_results) != 56
            or len(ranking.replay_receipts) != 476
            or production_runner.retained_backend_count != 56
            or len(production_runner.backend_instance_hashes) != 56
            or assets.model_limiter.maximum_active <= 0
            or assets.model_limiter.maximum_active > MODEL_INFERENCE_SLOTS
            or assets.provider_circuit.error_type is not None
        ):
            raise TrainExecutionContractActualError(
                "actual TRAIN execution grid or concurrency drifted"
            )

        candidate_rows = [
            {
                **candidate.safe_payload(),
                "historical_candidate_subset_hash": compiled.subset.subset_hash,
                "historical_canonical_set_hash": (
                    compiled.subset.canonical_set_hash
                ),
                "generation": compiled.subset.generation,
            }
            for candidate, compiled in zip(
                ranking.candidates,
                sorted(
                    integration.candidates,
                    key=lambda row: row.spec.candidate_hash,
                ),
                strict=True,
            )
        ]
        report_without_hash: dict[str, Any] = {
            "execution_policy": TRAIN_EXECUTION_CONTRACT_ACTUAL_VERSION,
            "execution_completed": True,
            "provider_canary": canary,
            "integration_report_hash": integration.report["report_hash"],
            "asset_preflight_report_hash": assets.preflight_report[
                "report_hash"
            ],
            "manifest_hash": manifest.manifest_hash,
            "evaluator_epoch": V320_EVALUATOR_EPOCH,
            "model_hash": stable_hash({"model": V320_MODEL}),
            "candidate_rows": candidate_rows,
            "candidate_row_set_hash": stable_hash(
                {"candidate_rows": candidate_rows}
            ),
            "ranking": ranking.to_dict(),
            "ranking_hash": ranking.ranking_hash,
            "outcomes": [row.safe_payload() for row in ranking.outcomes],
            "outcome_set_hash": ranking.outcome_set_hash,
            "run_receipts": [
                row.safe_payload() for row in ranking.run_results
            ],
            "replay_receipts": [
                row.safe_payload() for row in ranking.replay_receipts
            ],
            "outer_worker_limit": OUTER_WORKERS,
            "effective_outer_workers": ranking.effective_worker_count,
            "maximum_concurrent_runner_calls": (
                ranking.maximum_concurrent_runner_calls
            ),
            "model_inference_slot_limit": MODEL_INFERENCE_SLOTS,
            "maximum_concurrent_model_calls": (
                assets.model_limiter.maximum_active
            ),
            "distinct_backend_instance_count": (
                production_runner.retained_backend_count
            ),
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "validation_accessed": False,
            "test_accessed": False,
            "promotion_gate_applied": False,
            "promotion_authorized": False,
            "fresh_development_claim_authorized": False,
            "raw_candidate_trial_artifacts_persisted": False,
            "secret_value_persisted": False,
        }
        report = {
            **report_without_hash,
            "report_hash": stable_hash(report_without_hash),
        }
        (destination / ACTUAL_REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = TrainExecutionContractActualV2(
            output_root=destination,
            integration=integration,
            ranking=ranking,
            report=report,
        )
        result.verify()
        return result
    except Exception as exc:
        failure_without_hash = {
            "execution_policy": TRAIN_EXECUTION_CONTRACT_ACTUAL_VERSION,
            "execution_completed": False,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "raw_error_persisted": False,
            "secret_value_persisted": False,
        }
        failure = {
            **failure_without_hash,
            "report_hash": stable_hash(failure_without_hash),
        }
        (destination / FAILURE_REPORT_FILENAME).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all active v3.20 TRAIN execution-contract routes with "
            "offline evaluation and a flat parallel pool."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument(
        "--provider-label",
        choices=("plus", "pro"),
        required=True,
    )
    parser.add_argument("--task-input-cache-root", type=Path)
    args = parser.parse_args()
    result = run_v320_train_execution_contract_actual_v2(
        project_root=args.project_root,
        output_root=args.output_root,
        canary_report_path=args.canary_report,
        provider_label=args.provider_label,
        task_input_cache_root=args.task_input_cache_root,
    )
    top = next(
        row
        for row in result.ranking.aggregates
        if row.candidate_hash == result.ranking.top_candidate_hash
    )
    print(
        json.dumps(
            {
                "execution_completed": True,
                "ranking_hash": result.ranking.ranking_hash,
                "top_candidate_hash": result.ranking.top_candidate_hash,
                "top_invalid_count": top.invalid_count,
                "top_regression_count": top.regression_count,
                "top_recovery_count": top.recovery_count,
                "top_score_delta_units": top.score_delta_units,
                "active_execution_count": len(result.ranking.run_results),
                "inactive_replay_count": len(
                    result.ranking.replay_receipts
                ),
                "maximum_concurrent_runner_calls": (
                    result.ranking.maximum_concurrent_runner_calls
                ),
                "maximum_concurrent_model_calls": result.report[
                    "maximum_concurrent_model_calls"
                ],
                "online_judge_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
