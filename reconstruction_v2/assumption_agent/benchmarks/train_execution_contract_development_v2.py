from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from ..models import HypothesisStatus, ResidualExample, stable_hash
from ..proposer import TRAIN_ACTION_DESIGN_POLICY_VERSION
from ..splits import SplitAccessGuard, SplitManifest
from ..typed_execution_contract import (
    TypedExecutionContractRegistry,
    derive_train_execution_contract,
)
from .execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
    build_execution_contract_compile_bundle_v2,
)
from .historical_raw_train_projection_v2 import (
    HistoricalRawTrainProjectionV2,
    load_historical_raw_train_projection_v2,
)
from .skilllearn_compiler import SkillLearnProgramCompiler
from .skilllearn_lifecycle import (
    ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    SkillLearnResidualMiner,
)
from .skilllearnbench import SkillLearnBenchAdapter
from .train_outcome_ranker_v2 import TrainCandidateSpecV2
from .train_proposal_diagnostic import _public_train_items
from .typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
)
from .v320_train_candidate_material_v2 import (
    V320_EVALUATOR_EPOCH,
    V320_MANIFEST_HASH,
    V320_PROTOCOL_LOCK_SHA256,
    V320_RECURSIVE_EVENTS_SHA256,
    V320_SOURCE_RELATIVE_ROOT,
    V320TrainCandidateMaterialV2,
    V320TrainCandidateSubsetV2,
    load_v320_train_candidate_material_v2,
)


TRAIN_EXECUTION_CONTRACT_INTEGRATION_VERSION = (
    "v320_train_execution_contract_non_scoring_integration_v2"
)
V320_SOURCE_TRACE_ID = (
    "skilllearn-paired-9c7eb39a5b51-g1:shared-train"
)
V320_SOURCE_OBSERVATION_SET_HASH = (
    "9e57099459ccf2f905cb5ee44a9238fb51768a6b6b6401969856b8762c385c85"
)
V320_MANIFEST_FILE_SHA256 = (
    "795757be4c0eeb42331042fcbaa3eeba63e451619ee80c1a124c23eefef0dbb1"
)
V320_MANIFEST_RELATIVE_PATH = (
    "manifests/skilllearnbench_instance_holdout_offline_ready_v1.json"
)
SKILLLEARN_BENCHMARK_RELATIVE_ROOT = (
    "reference/self_evo_continual_20260707/repos/SkillLearnBench"
)
INTEGRATION_REPORT_FILENAME = "integration.report.json"


class TrainExecutionContractIntegrationError(PermissionError):
    """The non-scoring TRAIN execution-contract integration failed closed."""


@dataclass(frozen=True)
class CompiledTrainCandidateV2:
    subset: V320TrainCandidateSubsetV2
    bundle: ExecutionContractCompileBundleV2 = field(
        compare=False,
        repr=False,
    )
    spec: TrainCandidateSpecV2

    def verify(self) -> None:
        self.subset.verify()
        self.bundle.verify()
        self.spec.verify()
        reconstructed = TrainCandidateSpecV2.from_verified_bundle(
            candidate_id=self.subset.candidate_id,
            bundle=self.bundle,
            static_complexity=self.subset.static_complexity,
        )
        if (
            reconstructed.safe_payload() != self.spec.safe_payload()
            or len(self.spec.item_routes)
            != self.subset.expected_active_item_count
        ):
            raise TrainExecutionContractIntegrationError(
                "compiled TRAIN candidate binding drifted"
            )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "candidate_subset_hash": self.subset.subset_hash,
            "candidate_hash": self.spec.candidate_hash,
            "candidate_id_hash": self.spec.candidate_id_hash,
            "candidate_behavior_hash": self.spec.candidate_behavior_hash,
            "generation": self.subset.generation,
            "historical_canonical_set_hash": (
                self.subset.canonical_set_hash
            ),
            "program_count": len(self.subset.program_ids),
            "static_complexity": self.spec.static_complexity,
            "active_item_count": len(self.spec.item_routes),
            "base_compile_manifest_hash": (
                self.spec.base_compile_manifest_hash
            ),
            "compile_bundle_manifest_hash": (
                self.spec.compile_bundle_manifest_hash
            ),
            "program_set_hash": self.spec.program_set_hash,
            "typed_binding_set_hash": self.spec.typed_binding_set_hash,
            "typed_snapshot_ledger_hash": (
                self.spec.typed_snapshot_ledger_hash
            ),
            "execution_contract_set_hash": (
                self.spec.execution_contract_set_hash
            ),
            "item_route_set_hash": stable_hash(
                {
                    "item_routes": [
                        row.safe_payload() for row in self.spec.item_routes
                    ]
                }
            ),
            "validation_or_test_content_accessed": False,
            "model_calls": 0,
            "evaluator_calls": 0,
            "raw_program_or_task_content_persisted": False,
        }


@dataclass(frozen=True)
class TrainExecutionContractIntegrationV2:
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
    residuals: tuple[ResidualExample, ...] = field(
        compare=False,
        repr=False,
    )
    candidates: tuple[CompiledTrainCandidateV2, ...]

    @property
    def report_path(self) -> Path:
        return self.output_root / INTEGRATION_REPORT_FILENAME

    @property
    def candidate_specs(self) -> tuple[TrainCandidateSpecV2, ...]:
        return tuple(row.spec for row in self.candidates)

    @property
    def candidate_bundles_by_hash(
        self,
    ) -> dict[str, ExecutionContractCompileBundleV2]:
        return {row.spec.candidate_hash: row.bundle for row in self.candidates}

    def verify(self) -> None:
        self.candidate_material.verify()
        self.raw_projection.verify()
        if (
            len(self.residuals) != 38
            or sum(row.baseline_success for row in self.residuals) != 9
            or sum(not row.baseline_success for row in self.residuals) != 29
            or len(self.candidates) != 14
            or tuple(
                sorted(self.candidates, key=lambda row: row.spec.candidate_hash)
            )
            != self.candidates
            or len({row.spec.candidate_hash for row in self.candidates}) != 14
            or sum(len(row.spec.item_routes) for row in self.candidates) != 56
        ):
            raise TrainExecutionContractIntegrationError(
                "TRAIN execution-contract integration evidence drifted"
            )
        for candidate in self.candidates:
            candidate.verify()
        path = self.report_path
        if path.is_symlink() or not path.is_file():
            raise TrainExecutionContractIntegrationError(
                "TRAIN execution-contract integration report is missing"
            )
        try:
            persisted = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainExecutionContractIntegrationError(
                "TRAIN execution-contract integration report is unreadable"
            ) from exc
        without_hash = dict(persisted)
        report_hash = without_hash.pop("report_hash", None)
        if (
            persisted != dict(self.report)
            or report_hash != stable_hash(without_hash)
            or persisted.get("integration_passed") is not True
            or persisted.get("validation_or_test_content_accessed") is not False
            or persisted.get("model_calls") != 0
            or persisted.get("evaluator_calls") != 0
            or persisted.get("online_judge_calls") != 0
        ):
            raise TrainExecutionContractIntegrationError(
                "TRAIN execution-contract integration report drifted"
            )


def _load_raw_projection(
    *,
    project_root: Path,
    source_root: Path,
    manifest_path: Path,
) -> HistoricalRawTrainProjectionV2:
    projection = load_historical_raw_train_projection_v2(
        source_root=source_root,
        manifest_path=manifest_path,
        source_trace_id=V320_SOURCE_TRACE_ID,
        evaluator_epoch=V320_EVALUATOR_EPOCH,
        expected_source_observation_set_hash=(
            V320_SOURCE_OBSERVATION_SET_HASH
        ),
        expected_protocol_lock_file_sha256=V320_PROTOCOL_LOCK_SHA256,
        expected_event_ledger_file_sha256=V320_RECURSIVE_EVENTS_SHA256,
        expected_manifest_file_sha256=V320_MANIFEST_FILE_SHA256,
    )
    projection.verify()
    if (
        projection.baseline_set.manifest_hash != V320_MANIFEST_HASH
        or projection.baseline_set.evaluator_epoch != V320_EVALUATOR_EPOCH
        or len(projection.baseline_set.rows) != 38
        or sum(row.success for row in projection.baseline_set.rows) != 9
    ):
        raise TrainExecutionContractIntegrationError(
            "historical v3.20 RAW baseline cohort drifted"
        )
    return projection


def _mine_v320_residuals(
    *,
    project_root: Path,
    manifest: SplitManifest,
    raw_projection: HistoricalRawTrainProjectionV2,
) -> tuple[tuple[ResidualExample, ...], tuple[Any, ...]]:
    benchmark_root = (
        project_root / SKILLLEARN_BENCHMARK_RELATIVE_ROOT
    ).resolve(strict=True)
    train_items = _public_train_items(benchmark_root, manifest)
    if set(train_items) != set(manifest.train_ids):
        raise TrainExecutionContractIntegrationError(
            "public TRAIN item inventory drifted"
        )
    adapter = SkillLearnBenchAdapter(benchmark_root)
    # Deliberately bypass the adapter's 100-item discovery.  The preloaded
    # inventory contains only manifest-owned TRAIN public metadata.
    adapter._items = dict(train_items)  # type: ignore[attr-defined]
    adapter._required_env_by_item = {  # type: ignore[attr-defined]
        item_id: () for item_id in train_items
    }
    residuals = SkillLearnResidualMiner(
        adapter=adapter,
        manifest=manifest,
        guard=SplitAccessGuard(manifest),
        contrastive_training_evidence_policy=(
            ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
        train_action_design_policy=TRAIN_ACTION_DESIGN_POLICY_VERSION,
    ).mine(
        tuple(
            row.observation for row in raw_projection.baseline_set.rows
        ),
        trace_id="v320-train-contract-residual-replay",
    )
    if (
        len(residuals) != 38
        or sum(row.baseline_success for row in residuals) != 9
        or sum(not row.baseline_success for row in residuals) != 29
        or {row.task_id for row in residuals} != set(manifest.train_ids)
    ):
        raise TrainExecutionContractIntegrationError(
            "v3.20 TRAIN residual replay drifted"
        )
    return residuals, tuple(train_items.values())


def _validate_subset_routes(
    material: V320TrainCandidateMaterialV2,
    *,
    residuals: tuple[ResidualExample, ...],
) -> None:
    for subset in material.subsets:
        programs = material.program_set_for(subset)
        active = tuple(
            row
            for row in residuals
            if any(program.matches(row.features) for program in programs)
        )
        if (
            len(active) != subset.expected_active_item_count
            or any(row.baseline_success for row in active)
            or len({row.task_id for row in active}) != len(active)
        ):
            raise TrainExecutionContractIntegrationError(
                "v3.20 candidate TRAIN route replay drifted"
            )


def _build_contract_registry(
    material: V320TrainCandidateMaterialV2,
    *,
    residuals: tuple[ResidualExample, ...],
) -> TypedExecutionContractRegistry:
    registry = TypedExecutionContractRegistry()
    for program in material.programs:
        bound = material.typed_program_registry.require_bound_recipe(program)
        contract = derive_train_execution_contract(
            graph=bound.snapshot.graph,
            recipe_id=bound.recipe.recipe_id,
            residuals=residuals,
        )
        registry.register(contract, graph=bound.snapshot.graph)
    return registry


def _compile_one_candidate(
    *,
    subset: V320TrainCandidateSubsetV2,
    material: V320TrainCandidateMaterialV2,
    items: tuple[Any, ...],
    manifest: SplitManifest,
    contract_registry: TypedExecutionContractRegistry,
    candidates_root: Path,
) -> CompiledTrainCandidateV2:
    programs = material.program_set_for(subset)
    candidate_root = candidates_root / subset.subset_hash
    compiler = SkillLearnProgramCompiler(
        typed_program_registry=material.typed_program_registry,
        require_typed_bindings=True,
        portable_capability_compiler_mode=(
            PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
    )
    compiled = compiler.compile(
        programs=programs,
        items=items,
        split_manifest=manifest,
        output_root=candidate_root / "base_compile",
        method_name="candidate",
        allowed_statuses={HypothesisStatus.SHADOW},
        target_item_ids=manifest.train_ids,
        target_split="train",
        trace_id=f"v320-contract-compile-{subset.subset_hash[:20]}",
    )
    bundle = build_execution_contract_compile_bundle_v2(
        base_compile_result=compiled,
        programs=programs,
        items=items,
        typed_program_registry=material.typed_program_registry,
        execution_contract_registry=contract_registry,
        output_root=candidate_root / "contract_bundle",
    )
    spec = TrainCandidateSpecV2.from_verified_bundle(
        candidate_id=subset.candidate_id,
        bundle=bundle,
        static_complexity=subset.static_complexity,
    )
    result = CompiledTrainCandidateV2(
        subset=subset,
        bundle=bundle,
        spec=spec,
    )
    result.verify()
    return result


def compile_v320_train_execution_contract_candidates_v2(
    *,
    project_root: Path,
    output_root: Path,
) -> TrainExecutionContractIntegrationV2:
    """Compile all 14 historical subsets without scoring or model calls."""

    project = project_root.resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "TRAIN execution-contract integration output already exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        source_root = (project / V320_SOURCE_RELATIVE_ROOT).resolve(
            strict=True
        )
        manifest_path = (project / V320_MANIFEST_RELATIVE_PATH).resolve(
            strict=True
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
        manifest = SplitManifest.read(manifest_path)
        if manifest.manifest_hash != V320_MANIFEST_HASH:
            raise TrainExecutionContractIntegrationError(
                "v3.20 TRAIN manifest hash drifted"
            )
        residuals, items = _mine_v320_residuals(
            project_root=project,
            manifest=manifest,
            raw_projection=raw_projection,
        )
        _validate_subset_routes(material, residuals=residuals)
        contract_registry = _build_contract_registry(
            material,
            residuals=residuals,
        )
        candidates_root = destination / "candidates"
        candidates_root.mkdir()

        def compile_subset(
            subset: V320TrainCandidateSubsetV2,
        ) -> CompiledTrainCandidateV2:
            return _compile_one_candidate(
                subset=subset,
                material=material,
                items=items,
                manifest=manifest,
                contract_registry=contract_registry,
                candidates_root=candidates_root,
            )

        with ThreadPoolExecutor(max_workers=14) as executor:
            compiled_rows = tuple(
                executor.map(compile_subset, material.subsets)
            )
        candidates = tuple(
            sorted(compiled_rows, key=lambda row: row.spec.candidate_hash)
        )
        if (
            len({row.spec.candidate_hash for row in candidates}) != 14
            or sum(len(row.spec.item_routes) for row in candidates) != 56
        ):
            raise TrainExecutionContractIntegrationError(
                "compiled candidate route grid drifted"
            )

        candidate_rows = [row.safe_payload() for row in candidates]
        report_without_hash: dict[str, Any] = {
            "integration_policy": (
                TRAIN_EXECUTION_CONTRACT_INTEGRATION_VERSION
            ),
            "integration_passed": True,
            "candidate_material_receipt_hash": material.receipt.receipt_hash,
            "historical_raw_projection_receipt_hash": (
                raw_projection.receipt.receipt_hash
            ),
            "manifest_hash": manifest.manifest_hash,
            "typed_snapshot_ledger_hash": (
                material.typed_source.projected_snapshot_ledger_hash
            ),
            "source_train_item_count": 38,
            "source_train_success_count": 9,
            "source_train_failure_count": 29,
            "program_count": 6,
            "candidate_count": len(candidates),
            "candidate_rows": candidate_rows,
            "candidate_row_set_hash": stable_hash(
                {"candidate_rows": candidate_rows}
            ),
            "full_outcome_count": len(candidates) * 38,
            "active_execution_count": sum(
                len(row.spec.item_routes) for row in candidates
            ),
            "inactive_raw_replay_count": (
                len(candidates) * 38
                - sum(len(row.spec.item_routes) for row in candidates)
            ),
            "compile_workers": 14,
            "scoring_performed": False,
            "freeze_or_promotion_authorized": False,
            "validation_or_test_content_accessed": False,
            "model_calls": 0,
            "evaluator_calls": 0,
            "online_judge_calls": 0,
            "network_calls": 0,
            "raw_program_or_task_content_persisted": False,
        }
        report = {
            **report_without_hash,
            "report_hash": stable_hash(report_without_hash),
        }
        report_path = destination / INTEGRATION_REPORT_FILENAME
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = TrainExecutionContractIntegrationV2(
            output_root=destination,
            report=report,
            candidate_material=material,
            raw_projection=raw_projection,
            residuals=residuals,
            candidates=candidates,
        )
        result.verify()
        return result
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the 14 v3.20 TRAIN execution-contract candidates "
            "without scoring or model calls."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = compile_v320_train_execution_contract_candidates_v2(
        project_root=args.project_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "integration_passed": True,
                "report_hash": result.report["report_hash"],
                "candidate_count": len(result.candidates),
                "active_execution_count": result.report[
                    "active_execution_count"
                ],
                "model_calls": 0,
                "evaluator_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
