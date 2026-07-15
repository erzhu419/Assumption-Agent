from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from ..events import JsonlEventSink
from ..models import HypothesisProgram, HypothesisStatus, ResidualExample, stable_hash
from ..splits import SplitManifest
from ..typed_execution_contract import (
    TypedExecutionContract,
    TypedExecutionContractRegistry,
    derive_train_execution_contract,
)
from ..typed_operator_grammar import (
    MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
    FamilyCapabilityGraph,
    TypedProgramBindingRegistry,
    TypedRecipeSelectionSnapshot,
    TypedSelectionSnapshotLedger,
    WorkflowKind,
    build_family_capability_graph,
    canonical_typed_recipe_selection_request,
    canonical_typed_recipe_selection_response,
    freeze_typed_recipe_selection_snapshot,
    freeze_typed_selection_snapshot_ledger,
    materialize_recipe_selection,
)
from .execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
    ExecutionContractSubprocessBackendV2,
    build_execution_contract_compile_bundle_v2,
)
from .historical_raw_train_projection_v2 import HistoricalRawTrainProjectionV2
from .paper_protocol import PaperProtocol
from .skilllearn_compiler import SkillLearnProgramCompiler
from .train_execution_contract_actual_v2 import (
    MODEL_INFERENCE_SLOTS,
    V320_PROTOCOL_RELATIVE_PATH,
    _configure_environment,
    _prepare_scoped_runtime_assets,
    _verify_canary,
)
from .train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
    V320_MANIFEST_RELATIVE_PATH,
    _load_raw_projection,
    _mine_v320_residuals,
)
from .train_outcome_production_runner_v2 import ProductionTrainCandidateRunnerV2
from .train_outcome_ranker_v2 import (
    TrainCandidateSpecV2,
    TrainCandidateWorkUnitV2,
    TrainOutcomeRankingResultV2,
    TrainOutcomeRankerV2,
)
from .typed_portable_integration import _project_portable_graph
from .typed_task_capability import PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
from .v320_train_candidate_material_v2 import (
    V320_EVALUATOR_EPOCH,
    V320_MANIFEST_HASH,
    V320_MODEL,
    V320_SOURCE_RELATIVE_ROOT,
    V320TrainCandidateMaterialV2,
    load_v320_train_candidate_material_v2,
)


TRAIN_ITEM_OUT_CROSSFIT_POLICY = (
    "train_item_out_execution_contract_crossfit_v1"
)
CROSSFIT_COMPILE_POLICY = (
    "v320_train_item_out_execution_contract_compile_v2"
)
CROSSFIT_ACTUAL_POLICY = (
    "v320_train_item_out_execution_contract_actual_offline_v2"
)
TARGET_FAMILY = "organize-messy-files"
HELDOUT_ITEM_ID = "organize-messy-files-2"
GRAPH_SOURCE_ITEM_IDS = (
    "organize-messy-files-5",
    "organize-messy-files-6",
)
SELECTED_WORKFLOW = WorkflowKind.ORGANIZE_COLLECTION
SELECTED_RECIPE_ID = "recipe_0443d11a27ce50690356"
EXPECTED_FULL_GRAPH_HASH = (
    "8df3ac39d152dd4eb8288e5cd9cbcd37ef03f85db69526f2f5c9ea013aa873c2"
)
EXPECTED_PROJECTED_GRAPH_HASH = (
    "dfa967562901e54b62bf3e44a26486640e3b1791761d33a55b05e019dfbc96e5"
)
EXPECTED_SNAPSHOT_HASH = (
    "f0658025004be64cb64bee3358cbb9c37363bf70cd74f7a86c8e0d4a9532f5df"
)
EXPECTED_CONTRACT_HASH = (
    "d166655e3499539025552c2eab5fd50703ef542088ab5d400213a7f636ea1b9f"
)
EXPECTED_FOLD_RECEIPT_HASH = (
    "6819b23a3a6b84417d7be049cf6c885d6011a47d97220ae0ae97613059f27a4e"
)
EXPECTED_SNAPSHOT_LEDGER_HASH = (
    "630c12c4393b5fc5a0b8d20bca527fe76025b74c4f8f87341a4a26adcf5c4127"
)
EXPECTED_CANDIDATE_HASH = (
    "a34f06d0e3f9082380f4a29f0596f1f4a3c867c4ab7cd1f6e550ae38dc052634"
)
EXPECTED_WORK_UNIT_HASH = (
    "17463272acb75c16e8f6a9aa3f0035345672744fa6d3ef01b1b19c6ebc84fcb4"
)
SOURCE_TRAIN_RANKING_HASH = (
    "2ec01860409bfcba4a43c239810481450c0e5d126b3378234d437f234b14db33"
)
SOURCE_SEMANTIC_RANKING_HASH = (
    "58b35d342d458cea6682c066001ff620d8ef25bf63baef09e0bfda47494e82e0"
)
SOURCE_TOP_CANDIDATE_HASH = (
    "72c5ea9ef7407ee5df571df5f7e691d2f6a7657dc765a014b3e71aa9e0fcd295"
)
SOURCE_RANKING_REPORT_HASH = (
    "03184ce2224dd48d6e680df013d3ec20ea11267e11b04158580f0a1dd0773525"
)
SOURCE_RANKING_REPORT_RELATIVE_PATH = (
    "artifacts/train_execution_contract_development_v2_v320_train_"
    "resume_pro_actual01/ranking.report.json"
)
COMPILE_REPORT_FILENAME = "crossfit.compile.report.json"
ACTUAL_REPORT_FILENAME = "crossfit.report.json"
FAILURE_REPORT_FILENAME = "crossfit.failure.json"
EXECUTION_EVENTS_FILENAME = "crossfit.execution.events.jsonl"
ASSET_PREFLIGHT_POLICY = (
    "v320_train_item_out_local_asset_exact_reuse_preflight_v2"
)


class TrainExecutionContractCrossfitError(PermissionError):
    """A TRAIN item-out diagnostic crossed its registered fold."""


def _verify_source_ranking_report(project_root: Path) -> dict[str, Any]:
    path = project_root / SOURCE_RANKING_REPORT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise TrainExecutionContractCrossfitError(
            "source TRAIN ranking report is unavailable"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainExecutionContractCrossfitError(
            "source TRAIN ranking report is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise TrainExecutionContractCrossfitError(
            "source TRAIN ranking report is malformed"
        )
    without_hash = dict(payload)
    report_hash = without_hash.pop("report_hash", None)
    ranking = payload.get("ranking")
    if not isinstance(ranking, Mapping):
        raise TrainExecutionContractCrossfitError(
            "source TRAIN ranking receipt is malformed"
        )
    top_aggregates = [
        row
        for row in ranking.get("aggregates", ())
        if isinstance(row, Mapping)
        and row.get("candidate_hash") == SOURCE_TOP_CANDIDATE_HASH
    ]
    recovery_outcomes = [
        row
        for row in payload.get("outcomes", ())
        if isinstance(row, Mapping)
        and row.get("candidate_hash") == SOURCE_TOP_CANDIDATE_HASH
        and row.get("item_id_hash")
        == stable_hash({"item_id": HELDOUT_ITEM_ID})
        and row.get("recovery") is True
    ]
    if (
        report_hash != SOURCE_RANKING_REPORT_HASH
        or report_hash != stable_hash(without_hash)
        or payload.get("execution_completed") is not True
        or payload.get("ranking_hash") != SOURCE_TRAIN_RANKING_HASH
        or payload.get("semantic_ranking_hash")
        != SOURCE_SEMANTIC_RANKING_HASH
        or ranking.get("top_candidate_hash")
        != SOURCE_TOP_CANDIDATE_HASH
        or len(top_aggregates) != 1
        or top_aggregates[0].get("recovery_count") != 1
        or top_aggregates[0].get("regression_count") != 0
        or len(recovery_outcomes) != 1
        or payload.get("online_judge_calls") != 0
        or payload.get("validation_accessed") is not False
        or payload.get("test_accessed") is not False
    ):
        raise TrainExecutionContractCrossfitError(
            "source TRAIN ranking report drifted"
        )
    return {
        "source_report_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_report_hash": report_hash,
        "source_ranking_hash": SOURCE_TRAIN_RANKING_HASH,
        "source_semantic_ranking_hash": SOURCE_SEMANTIC_RANKING_HASH,
        "source_top_candidate_hash": SOURCE_TOP_CANDIDATE_HASH,
        "source_recovery_item_id_hash": stable_hash(
            {"item_id": HELDOUT_ITEM_ID}
        ),
        "source_top_recovery_count": 1,
        "source_top_regression_count": 0,
        "raw_source_content_embedded": False,
    }


def _support_task_id_hashes(
    contract: TypedExecutionContract,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                support.task_id_hash
                for invariant in contract.invariants
                for support in invariant.supports
            }
        )
    )


@dataclass(frozen=True)
class TrainItemOutCompileV2:
    output_root: Path = field(compare=False)
    report: Mapping[str, Any]
    candidate_material: V320TrainCandidateMaterialV2 = field(
        compare=False,
        repr=False,
    )
    raw_projection: HistoricalRawTrainProjectionV2 = field(
        compare=False,
        repr=False,
    )
    graph: FamilyCapabilityGraph = field(compare=False, repr=False)
    snapshot: TypedRecipeSelectionSnapshot = field(compare=False, repr=False)
    snapshot_ledger: TypedSelectionSnapshotLedger = field(
        compare=False,
        repr=False,
    )
    program: HypothesisProgram = field(compare=False, repr=False)
    typed_program_registry: TypedProgramBindingRegistry = field(
        compare=False,
        repr=False,
    )
    contract: TypedExecutionContract = field(compare=False, repr=False)
    bundle: ExecutionContractCompileBundleV2 = field(
        compare=False,
        repr=False,
    )
    candidate: TrainCandidateSpecV2
    work: TrainCandidateWorkUnitV2

    @property
    def report_path(self) -> Path:
        return self.output_root / COMPILE_REPORT_FILENAME

    def verify(self) -> None:
        self.candidate_material.verify()
        self.raw_projection.verify()
        if self.graph.validate() or self.snapshot.validate():
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out typed graph drifted"
            )
        if self.snapshot_ledger.validate():
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out snapshot ledger drifted"
            )
        if (
            self.program.validate()
            or self.program.status is not HypothesisStatus.CANDIDATE
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out program drifted"
            )
        bound = self.typed_program_registry.require_bound_recipe(self.program)
        if (
            bound.snapshot.snapshot_hash != self.snapshot.snapshot_hash
            or bound.recipe.recipe_id != SELECTED_RECIPE_ID
            or self.contract.validate(self.graph)
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out binding or contract drifted"
            )
        self.bundle.verify()
        self.candidate.verify()
        reconstructed = TrainCandidateSpecV2.from_verified_bundle(
            candidate_id=(
                "v320-train-loo-organize-2-organize-collection"
            ),
            bundle=self.bundle,
            static_complexity=5,
        )
        if reconstructed.safe_payload() != self.candidate.safe_payload():
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out candidate bundle drifted"
            )
        if (
            self.graph.graph_hash != EXPECTED_PROJECTED_GRAPH_HASH
            or self.snapshot.snapshot_hash != EXPECTED_SNAPSHOT_HASH
            or self.snapshot_ledger.ledger_hash
            != EXPECTED_SNAPSHOT_LEDGER_HASH
            or self.contract.contract_hash != EXPECTED_CONTRACT_HASH
            or self.candidate.candidate_hash != EXPECTED_CANDIDATE_HASH
            or self.work.work_unit_hash != EXPECTED_WORK_UNIT_HASH
            or not self.work.candidate_active
            or self.work.baseline.item_id != HELDOUT_ITEM_ID
            or _support_task_id_hashes(self.contract)
            != tuple(
                sorted(
                    stable_hash({"task_id": value})
                    for value in GRAPH_SOURCE_ITEM_IDS
                )
            )
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out leakage boundary drifted"
            )
        try:
            persisted = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out compile report is unreadable"
            ) from exc
        without_hash = dict(persisted)
        report_hash = without_hash.pop("report_hash", None)
        if (
            persisted != dict(self.report)
            or report_hash != stable_hash(without_hash)
            or persisted.get("compile_passed") is not True
            or persisted.get("heldout_excluded_from_graph") is not True
            or persisted.get("heldout_excluded_from_contract") is not True
            or persisted.get(
                "fold_and_workflow_selected_post_source_ranking"
            )
            is not True
            or persisted.get("targeted_item_out_refit_falsification")
            is not True
            or persisted.get("unbiased_crossfit") is not False
            or persisted.get("workflow_reselected_without_heldout")
            is not False
            or persisted.get("validation_or_test_content_accessed") is not False
            or persisted.get("model_calls") != 0
            or persisted.get("evaluator_calls") != 0
            or persisted.get("online_judge_calls") != 0
            or persisted.get("freeze_or_promotion_authorized") is not False
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out compile report drifted"
            )


def compile_v320_train_item_out_crossfit_v2(
    *,
    project_root: Path,
    output_root: Path,
) -> TrainItemOutCompileV2:
    """Compile one organize-2 route from organize-5/-6 TRAIN evidence."""

    project = project_root.resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("TRAIN item-out compile output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        source_ranking_receipt = _verify_source_ranking_report(project)
        source_root = (project / V320_SOURCE_RELATIVE_ROOT).resolve(strict=True)
        manifest_path = (project / V320_MANIFEST_RELATIVE_PATH).resolve(
            strict=True
        )
        manifest = SplitManifest.read(manifest_path)
        if manifest.manifest_hash != V320_MANIFEST_HASH:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out manifest drifted"
            )
        material = load_v320_train_candidate_material_v2(
            project_root=project,
            source_root=source_root,
        )
        raw_projection = _load_raw_projection(
            project_root=project,
            source_root=source_root,
            manifest_path=manifest_path,
        )
        v320_residuals, items = _mine_v320_residuals(
            project_root=project,
            manifest=manifest,
            raw_projection=raw_projection,
        )
        heldout_items = tuple(row for row in items if row.id == HELDOUT_ITEM_ID)
        if len(heldout_items) != 1:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out public item binding drifted"
            )
        heldout_item = heldout_items[0]
        baseline_rows = tuple(
            row
            for row in raw_projection.baseline_set.rows
            if row.item_id == HELDOUT_ITEM_ID
        )
        if len(baseline_rows) != 1 or baseline_rows[0].success:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out RAW baseline drifted"
            )
        baseline = baseline_rows[0]

        evidence = material.typed_source.ledger.evidence
        graph_residuals = tuple(
            sorted(
                (
                    row
                    for row in evidence.residuals
                    if row.task_id in GRAPH_SOURCE_ITEM_IDS
                    and row.family == TARGET_FAMILY
                    and not row.baseline_success
                ),
                key=lambda row: row.task_id,
            )
        )
        if tuple(row.task_id for row in graph_residuals) != GRAPH_SOURCE_ITEM_IDS:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out graph source cohort drifted"
            )
        profile_hashes = tuple(
            sorted(
                str(row.context.get("action_context_profile_hash") or "")
                for row in graph_residuals
            )
        )
        try:
            action_profiles = {
                value: evidence.action_profiles[value]
                for value in profile_hashes
            }
        except KeyError as exc:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out action profile is missing"
            ) from exc
        all_trials = {
            row.trial_id_hash: row
            for row in material.typed_source.ledger.trials
        }
        graph_trial_hashes = tuple(
            sorted(
                stable_hash({"item_id": value})
                for value in GRAPH_SOURCE_ITEM_IDS
            )
        )
        try:
            graph_trials = {
                value: all_trials[value] for value in graph_trial_hashes
            }
        except KeyError as exc:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out trial evidence is missing"
            ) from exc
        heldout_item_hash = stable_hash({"item_id": HELDOUT_ITEM_ID})
        heldout_task_hash = stable_hash({"task_id": HELDOUT_ITEM_ID})
        if (
            heldout_item.id_hash != heldout_item_hash
            or baseline.item_id_hash != heldout_item_hash
            or heldout_item_hash in graph_trials
            or any(row.task_id == HELDOUT_ITEM_ID for row in graph_residuals)
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out graph contains held-out evidence"
            )

        full_graph = build_family_capability_graph(
            target_family=TARGET_FAMILY,
            failures=graph_residuals,
            action_profiles=action_profiles,
            trial_evidence=graph_trials,
        )
        graph = _project_portable_graph(full_graph)
        snapshot = freeze_typed_recipe_selection_snapshot(graph)
        recipes = tuple(
            row for row in graph.recipes if row.workflow is SELECTED_WORKFLOW
        )
        if (
            MAX_REGISTERED_ARTIFACTS_PER_FAMILY != 6
            or full_graph.graph_hash != EXPECTED_FULL_GRAPH_HASH
            or graph.graph_hash != EXPECTED_PROJECTED_GRAPH_HASH
            or snapshot.snapshot_hash != EXPECTED_SNAPSHOT_HASH
            or len(recipes) != 1
            or recipes[0].recipe_id != SELECTED_RECIPE_ID
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out projected recipe drifted"
            )

        contract_residuals = tuple(
            row for row in v320_residuals if row.task_id != HELDOUT_ITEM_ID
        )
        contract = derive_train_execution_contract(
            graph=graph,
            recipe_id=SELECTED_RECIPE_ID,
            residuals=contract_residuals,
        )
        contract_source_task_hashes = _support_task_id_hashes(contract)
        expected_source_task_hashes = tuple(
            sorted(
                stable_hash({"task_id": value})
                for value in GRAPH_SOURCE_ITEM_IDS
            )
        )
        if (
            contract.contract_hash != EXPECTED_CONTRACT_HASH
            or contract_source_task_hashes != expected_source_task_hashes
            or heldout_task_hash in contract_source_task_hashes
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out contract contains held-out evidence"
            )

        fold_receipt = {
            "policy": TRAIN_ITEM_OUT_CROSSFIT_POLICY,
            "upstream_typed_source_receipt_hash": (
                material.typed_source.receipt_hash
            ),
            "upstream_source_train_receipt_hash": (
                evidence.source_train_receipt_hash
            ),
            "target_family_hash": graph.target_family_hash,
            "heldout_item_id_hash": heldout_item_hash,
            "heldout_task_id_hash": heldout_task_hash,
            "graph_source_item_id_hashes": list(graph_trial_hashes),
            "graph_source_transition_id_hashes": sorted(
                stable_hash({"transition_id": row.transition_id})
                for row in graph_residuals
            ),
            "graph_source_action_profile_hashes": list(profile_hashes),
            "graph_source_trial_evidence_hashes": sorted(
                row.evidence_hash for row in graph_trials.values()
            ),
            "maximum_registered_artifacts_per_family": (
                MAX_REGISTERED_ARTIFACTS_PER_FAMILY
            ),
            "full_graph_hash": full_graph.graph_hash,
            "projected_graph_hash": graph.graph_hash,
            "projected_snapshot_hash": snapshot.snapshot_hash,
            "contract_source_task_id_hashes": list(
                contract_source_task_hashes
            ),
            "heldout_excluded_from_graph": True,
            "heldout_excluded_from_contract": True,
            "graph_source_count": len(graph_residuals),
            "minimum_independent_support": 2,
            "diagnostic_only": True,
            "freeze_or_promotion_authorized": False,
            "validation_or_test_content_accessed": False,
            "model_calls": 0,
            "evaluator_calls": 0,
            "raw_content_persisted": False,
        }
        fold_receipt_hash = stable_hash(fold_receipt)
        if fold_receipt_hash != EXPECTED_FOLD_RECEIPT_HASH:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out fold receipt drifted"
            )

        graph_set_hash = stable_hash(
            {
                "outcomes": [
                    {
                        "target_family_hash": graph.target_family_hash,
                        "graph_hash": graph.graph_hash,
                        "availability_error_hash": None,
                    }
                ]
            }
        )
        model_catalog_set_hash = stable_hash(
            {"catalog_hashes": [snapshot.expected_model_catalog_hash]}
        )
        upstream_ledger = (
            material.typed_source.ledger.production_snapshot_ledger
        )
        snapshot_ledger = freeze_typed_selection_snapshot_ledger(
            (snapshot,),
            feasibility_preregistration_hash=(
                upstream_ledger.feasibility_preregistration_hash
            ),
            feasibility_result_receipt_sha256=(
                upstream_ledger.feasibility_result_receipt_sha256
            ),
            feasibility_decision_hash=(
                upstream_ledger.feasibility_decision_hash
            ),
            feasibility_report_hash=(
                upstream_ledger.feasibility_report_hash
            ),
            manifest_hash=manifest.manifest_hash,
            source_train_receipt_hash=fold_receipt_hash,
            expected_graph_set_hash=graph_set_hash,
            expected_model_catalog_set_hash=model_catalog_set_hash,
            expected_target_family_hashes=(graph.target_family_hash,),
        )
        if snapshot_ledger.ledger_hash != EXPECTED_SNAPSHOT_LEDGER_HASH:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out diagnostic ledger drifted"
            )

        program = materialize_recipe_selection(
            {"recipe_id": SELECTED_RECIPE_ID},
            graph=graph,
            evaluator_epoch=V320_EVALUATOR_EPOCH,
            expected_graph_hash=snapshot.expected_graph_hash,
            expected_model_catalog_hash=snapshot.expected_model_catalog_hash,
        )
        request = canonical_typed_recipe_selection_request(
            snapshot=snapshot,
            snapshot_ledger=snapshot_ledger,
            evaluator_epoch=V320_EVALUATOR_EPOCH,
            selection_round=1,
        )
        response = canonical_typed_recipe_selection_response(
            SELECTED_RECIPE_ID
        )
        registry = TypedProgramBindingRegistry(
            snapshot_ledger=snapshot_ledger
        )
        binding = registry.register(
            program,
            snapshot=snapshot,
            recipe_id=SELECTED_RECIPE_ID,
            request_kind="select_typed_root_recipe",
            request_hash=stable_hash(request),
            response_hash=stable_hash(response),
            selection_round=1,
        )
        static_complexity = len(program.action_graph) + 1
        if static_complexity != 5:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out program complexity drifted"
            )

        compiler = SkillLearnProgramCompiler(
            typed_program_registry=registry,
            require_typed_bindings=True,
            portable_capability_compiler_mode=(
                PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
            ),
        )
        compiled = compiler.compile(
            programs=(program,),
            items=(heldout_item,),
            split_manifest=manifest,
            output_root=destination / "base_compile",
            method_name="candidate",
            allowed_statuses={HypothesisStatus.CANDIDATE},
            target_item_ids=(HELDOUT_ITEM_ID,),
            target_split="train",
            trace_id="v320-train-item-out-organize-2-compile",
        )
        contract_registry = TypedExecutionContractRegistry()
        contract_registry.register(contract, graph=graph)
        bundle = build_execution_contract_compile_bundle_v2(
            base_compile_result=compiled,
            programs=(program,),
            items=(heldout_item,),
            typed_program_registry=registry,
            execution_contract_registry=contract_registry,
            output_root=destination / "contract_bundle",
        )
        candidate = TrainCandidateSpecV2.from_verified_bundle(
            candidate_id=(
                "v320-train-loo-organize-2-organize-collection"
            ),
            bundle=bundle,
            static_complexity=static_complexity,
        )
        work = TrainCandidateWorkUnitV2(
            candidate=candidate,
            baseline=baseline,
        )
        if (
            candidate.candidate_hash != EXPECTED_CANDIDATE_HASH
            or work.work_unit_hash != EXPECTED_WORK_UNIT_HASH
            or len(candidate.item_routes) != 1
            or candidate.item_routes[0].item_id_hash != heldout_item_hash
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out compiled route drifted"
            )

        report_without_hash: dict[str, Any] = {
            "compile_policy": CROSSFIT_COMPILE_POLICY,
            "compile_passed": True,
            "source_train_ranking_receipt": source_ranking_receipt,
            "source_train_ranking_hash": SOURCE_TRAIN_RANKING_HASH,
            "source_semantic_ranking_hash": (
                SOURCE_SEMANTIC_RANKING_HASH
            ),
            "source_top_candidate_hash": SOURCE_TOP_CANDIDATE_HASH,
            "fold_receipt": fold_receipt,
            "fold_receipt_hash": fold_receipt_hash,
            "manifest_hash": manifest.manifest_hash,
            "full_graph_hash": full_graph.graph_hash,
            "projected_graph_hash": graph.graph_hash,
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_ledger_hash": snapshot_ledger.ledger_hash,
            "recipe_id": SELECTED_RECIPE_ID,
            "workflow": SELECTED_WORKFLOW.value,
            "program_id_hash": stable_hash({"program_id": program.id}),
            "program_payload_hash": program.payload_hash,
            "typed_binding_hash": binding.binding_hash,
            "execution_contract_hash": contract.contract_hash,
            "contract_source_task_id_hashes": list(
                contract_source_task_hashes
            ),
            "base_compile_manifest_hash": compiled.manifest_hash,
            "compile_bundle_manifest_hash": bundle.manifest_hash,
            "candidate_hash": candidate.candidate_hash,
            "candidate_behavior_hash": candidate.candidate_behavior_hash,
            "work_unit_hash": work.work_unit_hash,
            "heldout_item_id_hash": heldout_item_hash,
            "heldout_baseline_evidence_hash": (
                baseline.baseline_evidence_hash
            ),
            "heldout_baseline_success": baseline.success,
            "heldout_excluded_from_graph": True,
            "heldout_excluded_from_contract": True,
            "fold_and_workflow_selected_post_source_ranking": True,
            "targeted_item_out_refit_falsification": True,
            "unbiased_crossfit": False,
            "workflow_reselected_without_heldout": False,
            "expected_active_execution_count": 1,
            "expected_inactive_raw_replay_count": 37,
            "compile_is_non_scoring_diagnostic": True,
            "freeze_or_promotion_authorized": False,
            "validation_or_test_content_accessed": False,
            "model_calls": 0,
            "evaluator_calls": 0,
            "online_judge_calls": 0,
            "network_calls": 0,
            "raw_program_task_or_evaluator_content_persisted": False,
        }
        report = {
            **report_without_hash,
            "report_hash": stable_hash(report_without_hash),
        }
        (destination / COMPILE_REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = TrainItemOutCompileV2(
            output_root=destination,
            report=report,
            candidate_material=material,
            raw_projection=raw_projection,
            graph=graph,
            snapshot=snapshot,
            snapshot_ledger=snapshot_ledger,
            program=program,
            typed_program_registry=registry,
            contract=contract,
            bundle=bundle,
            candidate=candidate,
            work=work,
        )
        result.verify()
        return result
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise


@dataclass(frozen=True)
class TrainItemOutActualV2:
    output_root: Path = field(compare=False)
    compilation: TrainItemOutCompileV2 = field(compare=False, repr=False)
    ranking: TrainOutcomeRankingResultV2 = field(compare=False, repr=False)
    report: Mapping[str, Any]

    @property
    def report_path(self) -> Path:
        return self.output_root / ACTUAL_REPORT_FILENAME

    def verify(self) -> None:
        self.compilation.verify()
        self.ranking.verify()
        try:
            persisted = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out actual report is unreadable"
            ) from exc
        without_hash = dict(persisted)
        report_hash = without_hash.pop("report_hash", None)
        if (
            persisted != dict(self.report)
            or report_hash != stable_hash(without_hash)
            or persisted.get("execution_completed") is not True
            or persisted.get("active_execution_count") != 1
            or persisted.get("inactive_raw_replay_count") != 37
            or persisted.get("offline_evaluation_only") is not True
            or persisted.get("online_judge_calls") != 0
            or persisted.get("validation_accessed") is not False
            or persisted.get("test_accessed") is not False
            or persisted.get("promotion_authorized") is not False
            or persisted.get(
                "fold_and_workflow_selected_post_source_ranking"
            )
            is not True
            or persisted.get("targeted_item_out_refit_falsification")
            is not True
            or persisted.get("unbiased_crossfit") is not False
            or persisted.get("workflow_reselected_without_heldout")
            is not False
            or persisted.get("raw_candidate_trial_artifacts_persisted")
            is not True
            or persisted.get(
                "raw_candidate_trial_artifacts_embedded_in_report"
            )
            is not False
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out actual report drifted"
            )


def run_v320_train_item_out_crossfit_actual_v2(
    *,
    project_root: Path,
    output_root: Path,
    canary_report_path: Path,
    provider_label: str,
    task_input_cache_root: Path | None = None,
) -> TrainItemOutActualV2:
    """Run the single registered organize-2 item-out route offline."""

    project = project_root.resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("TRAIN item-out actual output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
        if (
            protocol.payload.get("protocol_version") != "3.20.0"
            or protocol.payload.get("model") != V320_MODEL
            or protocol.payload.get("max_steps") != 100
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out execution protocol drifted"
            )
        _configure_environment(protocol)
        canary = _verify_canary(
            canary_report_path.resolve(strict=True),
            provider_label=provider_label,
        )
        manifest = SplitManifest.read(project / V320_MANIFEST_RELATIVE_PATH)
        if manifest.manifest_hash != V320_MANIFEST_HASH:
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out execution manifest drifted"
            )
        compilation = compile_v320_train_item_out_crossfit_v2(
            project_root=project,
            output_root=destination / "compile_diagnostic",
        )
        event_sink = JsonlEventSink(destination / EXECUTION_EVENTS_FILENAME)
        active_hash = compilation.work.baseline.item_id_hash
        assets = _prepare_scoped_runtime_assets(
            project_root=project,
            destination=destination,
            protocol=protocol,
            manifest=manifest,
            baseline_set=compilation.raw_projection.baseline_set,
            active_item_hashes={active_hash},
            expected_active_item_count=1,
            preflight_policy=ASSET_PREFLIGHT_POLICY,
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
            baseline_set=compilation.raw_projection.baseline_set,
            candidate_bundles={
                compilation.candidate.candidate_hash: compilation.bundle
            },
            backend_factory=backend_factory,
            trace_prefix="v320-train-item-out-organize-2-actual",
        )
        ranking = TrainOutcomeRankerV2(max_workers=1).rank(
            baseline_set=compilation.raw_projection.baseline_set,
            candidates=(compilation.candidate,),
            runner=production_runner,
        )
        ranking.verify()
        if (
            len(ranking.run_results) != 1
            or len(ranking.replay_receipts) != 37
            or production_runner.retained_backend_count != 1
            or len(production_runner.backend_instance_hashes) != 1
            or assets.model_limiter.maximum_active <= 0
            or assets.model_limiter.maximum_active > MODEL_INFERENCE_SLOTS
            or assets.provider_circuit.error_type is not None
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out execution boundary drifted"
            )
        run_result = ranking.run_results[0]
        if (
            run_result.work_unit_hash != EXPECTED_WORK_UNIT_HASH
            or not run_result.offline_evaluation.evaluation_valid
        ):
            raise TrainExecutionContractCrossfitError(
                "TRAIN item-out evaluation is not valid"
            )
        outcome = next(
            row
            for row in ranking.outcomes
            if row.work_unit_hash == EXPECTED_WORK_UNIT_HASH
        )
        heldout_recovery = outcome.recovery
        report_without_hash: dict[str, Any] = {
            "execution_policy": CROSSFIT_ACTUAL_POLICY,
            "execution_completed": True,
            "provider_canary": canary,
            "compile_report_hash": compilation.report["report_hash"],
            "asset_preflight_report_hash": assets.preflight_report[
                "report_hash"
            ],
            "manifest_hash": manifest.manifest_hash,
            "evaluator_epoch": V320_EVALUATOR_EPOCH,
            "model_hash": stable_hash({"model": V320_MODEL}),
            "source_train_ranking_hash": SOURCE_TRAIN_RANKING_HASH,
            "source_semantic_ranking_hash": (
                SOURCE_SEMANTIC_RANKING_HASH
            ),
            "source_top_candidate_hash": SOURCE_TOP_CANDIDATE_HASH,
            "candidate_hash": compilation.candidate.candidate_hash,
            "work_unit_hash": EXPECTED_WORK_UNIT_HASH,
            "heldout_item_id_hash": active_hash,
            "heldout_excluded_from_graph": True,
            "heldout_excluded_from_contract": True,
            "fold_and_workflow_selected_post_source_ranking": True,
            "targeted_item_out_refit_falsification": True,
            "unbiased_crossfit": False,
            "workflow_reselected_without_heldout": False,
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
            "active_execution_count": 1,
            "inactive_raw_replay_count": 37,
            "maximum_concurrent_runner_calls": (
                ranking.maximum_concurrent_runner_calls
            ),
            "maximum_concurrent_model_calls": (
                assets.model_limiter.maximum_active
            ),
            "heldout_evaluation_valid": True,
            "heldout_baseline_success": outcome.baseline_success,
            "heldout_candidate_success": outcome.candidate_success,
            "heldout_recovery_observed": heldout_recovery,
            "in_sample_recovery_survived_item_out": heldout_recovery,
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "validation_accessed": False,
            "test_accessed": False,
            "promotion_gate_applied": False,
            "promotion_authorized": False,
            "fresh_development_claim_authorized": False,
            "single_fold_incumbent_authorized": False,
            "raw_candidate_trial_artifacts_persisted": True,
            "raw_candidate_trial_artifacts_embedded_in_report": False,
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
        result = TrainItemOutActualV2(
            output_root=destination,
            compilation=compilation,
            ranking=ranking,
            report=report,
        )
        result.verify()
        return result
    except Exception as exc:
        failure_without_hash = {
            "execution_policy": CROSSFIT_ACTUAL_POLICY,
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
            "Compile and run the single v3.20 organize-2 TRAIN item-out "
            "execution-contract diagnostic."
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
    result = run_v320_train_item_out_crossfit_actual_v2(
        project_root=args.project_root,
        output_root=args.output_root,
        canary_report_path=args.canary_report,
        provider_label=args.provider_label,
        task_input_cache_root=args.task_input_cache_root,
    )
    print(
        json.dumps(
            {
                "execution_completed": True,
                "report_hash": result.report["report_hash"],
                "ranking_hash": result.ranking.ranking_hash,
                "candidate_hash": result.compilation.candidate.candidate_hash,
                "work_unit_hash": result.compilation.work.work_unit_hash,
                "heldout_candidate_success": result.report[
                    "heldout_candidate_success"
                ],
                "heldout_recovery_observed": result.report[
                    "heldout_recovery_observed"
                ],
                "active_execution_count": 1,
                "inactive_raw_replay_count": 37,
                "online_judge_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
