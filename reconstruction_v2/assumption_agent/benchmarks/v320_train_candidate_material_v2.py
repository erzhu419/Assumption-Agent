from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..models import HypothesisProgram, HypothesisStatus, stable_hash
from ..typed_operator_grammar import (
    TypedProgramBindingRegistry,
    validate_typed_selection_history_payloads,
)
from .historical_typed_selection_projection_v2 import (
    HISTORICAL_PROJECTED_LEDGER_HASH,
    HistoricalTypedSelectionSourceReceiptV2,
    load_historical_portable_typed_selection_projection_v2,
)


V320_TRAIN_CANDIDATE_MATERIAL_VERSION = (
    "v320_historical_train_candidate_material_v2"
)
V320_SOURCE_RELATIVE_ROOT = (
    "artifacts/"
    "paper_primary_v3_20_offline86_ruoli_gpt54mini_"
    "outer38_model48_portable01"
)
V320_SOURCE_COMMIT = "7da8a3d0a40653013ea8e253d81982d00e5d3c37"
V320_PROTOCOL_LOCK_SHA256 = (
    "7cdc2f41db273de14afaf87bab75408f038927951282de2e41769ff22cb169fb"
)
V320_RECURSIVE_EVENTS_SHA256 = (
    "beb933673942662a3cf69d503f5bb13f00ec21af8a276cfedf6807e1b86e83e6"
)
V320_RECURSIVE_ARCHIVE_SHA256 = (
    "dcf1d80bd13f64819f1482714a33e016cccc676cd3b390efd5b3d863bab59b48"
)
V320_RECURSIVE_REPORT_SHA256 = (
    "98664f6baceb1bf6f5291e9c3637a54b14491152facae24cbce930c4870cdf59"
)
V320_RECURSIVE_ARCHIVE_HASH = (
    "58d55253582b8b8d5933e967ed03d7b319fe970ae0325c18e4e42a84a05edd04"
)
V320_MANIFEST_HASH = (
    "9c7eb39a5b514b68fd87de5e60329a8fbaf626db6150d18b26dc59f17d80b652"
)
V320_EVALUATOR_EPOCH = "skilllearn-eval-9c7eb39a5b51"
V320_MODEL = "gpt-5.4-mini"
V320_MAX_STEPS = 100

_CANDIDATE_SELECTION_EVENT = (
    "hypothesis_training_candidate_selection_completed"
)
_CANDIDATE_BUNDLE_POLICY = (
    "train_only_union_program_set_single_paired_validation_"
    "conservative_thresholds_v1"
)
_REQUIRED_TRACE_PAIRS = (
    (
        1,
        "skilllearn-paired-9c7eb39a5b51-g1:recursive",
        "skilllearn-paired-9c7eb39a5b51-g1:no-recursive",
    ),
    (
        2,
        "skilllearn-paired-9c7eb39a5b51-recursive-g2",
        "skilllearn-paired-9c7eb39a5b51-no-recursive-g2",
    ),
)
_ARCHIVE_FIELDS = frozenset(
    {
        "archive_hash",
        "hypotheses",
        "incumbent_id",
        "nodes",
        "raw_content_persisted",
        "score_records",
        "typed_bindings",
        "typed_selection_history",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "event",
        "stage",
        "trace_id",
        "payload",
        "payload_hash",
        "event_id",
        "raw_content_persisted",
    }
)
_SUBSET_FIELDS = frozenset(
    {
        "accepted_behavior_hashes",
        "accepted_hypothesis_hashes",
        "accepted_hypothesis_ids",
        "canonical_set_hash",
        "ranking_priority",
        "root_hypothesis_hashes",
        "root_hypothesis_ids",
        "selected",
        "selection_uses_validation",
        "union_training_metrics",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class V320TrainCandidateMaterialError(PermissionError):
    """The historical v3.20 TRAIN candidate source failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(root: Path, name: str, expected_sha256: str) -> Path:
    if not _SHA256.fullmatch(expected_sha256):
        raise V320TrainCandidateMaterialError(
            "v3.20 expected file hash is malformed"
        )
    candidate = root / name
    if candidate.is_symlink():
        raise V320TrainCandidateMaterialError(
            "v3.20 candidate source contains a symlink"
        )
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise V320TrainCandidateMaterialError(
            "v3.20 candidate source escaped its frozen root"
        ) from exc
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise V320TrainCandidateMaterialError(
            f"v3.20 {name} file hash drifted"
        )
    return path


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V320TrainCandidateMaterialError(
            f"v3.20 {label} is not readable JSON"
        ) from exc
    if not isinstance(value, dict):
        raise V320TrainCandidateMaterialError(
            f"v3.20 {label} is not an object"
        )
    return value


def _load_events(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise V320TrainCandidateMaterialError(
            "v3.20 event ledger is unreadable"
        ) from exc
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise V320TrainCandidateMaterialError(
                f"v3.20 event {line_number} is malformed"
            ) from exc
        if not isinstance(row, dict) or set(row) != _EVENT_FIELDS:
            raise V320TrainCandidateMaterialError(
                f"v3.20 event {line_number} envelope drifted"
            )
        payload = row.get("payload")
        event = row.get("event")
        stage = row.get("stage")
        trace_id = row.get("trace_id")
        if (
            not isinstance(payload, dict)
            or not isinstance(event, str)
            or not event
            or not isinstance(stage, str)
            or not stage
            or not isinstance(trace_id, str)
            or not trace_id
            or row.get("payload_hash") != stable_hash(payload)
            or row.get("event_id")
            != stable_hash(
                {
                    "event": event,
                    "stage": stage,
                    "trace_id": trace_id,
                    "payload": payload,
                }
            )[:24]
            or row.get("raw_content_persisted") is not False
        ):
            raise V320TrainCandidateMaterialError(
                f"v3.20 event {line_number} envelope failed"
            )
        events.append(row)
    if not events:
        raise V320TrainCandidateMaterialError(
            "v3.20 event ledger is empty"
        )
    return tuple(events)


def _behavior_hash(program: HypothesisProgram) -> str:
    payload = program.to_dict()
    for key in (
        "id",
        "status",
        "parent_id",
        "lineage",
        "created_from_transition_ids",
    ):
        payload.pop(key, None)
    return stable_hash(payload)


def _archive_hash(
    archive: Mapping[str, Any],
    programs: Mapping[str, HypothesisProgram],
) -> str:
    nodes = archive.get("nodes")
    scores = archive.get("score_records")
    typed_bindings = archive.get("typed_bindings")
    selection_history = archive.get("typed_selection_history")
    if not all(
        isinstance(value, Mapping)
        for value in (nodes, scores, typed_bindings, selection_history)
    ):
        raise V320TrainCandidateMaterialError(
            "v3.20 archive ledgers are malformed"
        )
    return stable_hash(
        {
            "hypotheses": {
                key: value.payload_hash
                for key, value in sorted(programs.items())
            },
            "nodes": {
                str(key): stable_hash(dict(value))
                for key, value in sorted(nodes.items())
                if isinstance(value, Mapping)
            },
            "scores": {
                str(key): dict(value)
                for key, value in sorted(scores.items())
                if isinstance(value, Mapping)
            },
            "incumbent_id": archive.get("incumbent_id"),
            "typed_bindings": {
                str(key): dict(value)
                for key, value in sorted(typed_bindings.items())
                if isinstance(value, Mapping)
            },
            "typed_selection_history": {
                str(key): dict(value)
                for key, value in sorted(selection_history.items())
                if isinstance(value, Mapping)
            },
        }
    )


def _strict_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V320TrainCandidateMaterialError(
            f"v3.20 {label} is not a positive integer"
        )
    return value


@dataclass(frozen=True)
class V320TrainCandidateSubsetV2:
    generation: int
    source_trace_id_hash: str
    canonical_set_hash: str
    program_ids: tuple[str, ...] = field(compare=False, repr=False)
    accepted_hypothesis_hashes: tuple[str, ...]
    accepted_behavior_hashes: tuple[str, ...]
    static_complexity: int
    expected_active_item_count: int
    selected: bool

    @property
    def candidate_id(self) -> str:
        return (
            f"v320-g{self.generation}-"
            f"{self.canonical_set_hash[:20]}"
        )

    @property
    def subset_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "candidate_material_policy": (
                V320_TRAIN_CANDIDATE_MATERIAL_VERSION
            ),
            "generation": self.generation,
            "source_trace_id_hash": self.source_trace_id_hash,
            "canonical_set_hash": self.canonical_set_hash,
            "program_id_hashes": [
                stable_hash({"program_id": value})
                for value in self.program_ids
            ],
            "accepted_hypothesis_hashes": list(
                self.accepted_hypothesis_hashes
            ),
            "accepted_behavior_hashes": list(
                self.accepted_behavior_hashes
            ),
            "program_count": len(self.program_ids),
            "static_complexity": self.static_complexity,
            "expected_active_item_count": (
                self.expected_active_item_count
            ),
            "selected_in_historical_outer_loop": self.selected,
            "validation_or_test_content_accessed": False,
            "model_calls": 0,
            "evaluator_calls": 0,
            "raw_program_or_task_content_persisted": False,
        }

    def verify(self) -> None:
        hashes = (
            self.source_trace_id_hash,
            self.canonical_set_hash,
            *self.accepted_hypothesis_hashes,
            *self.accepted_behavior_hashes,
        )
        if (
            self.generation not in {1, 2}
            or not self.program_ids
            or self.program_ids != tuple(sorted(set(self.program_ids)))
            or len(self.program_ids) != len(self.accepted_hypothesis_hashes)
            or len(self.program_ids) != len(self.accepted_behavior_hashes)
            or tuple(sorted(set(self.accepted_hypothesis_hashes)))
            != self.accepted_hypothesis_hashes
            or tuple(sorted(set(self.accepted_behavior_hashes)))
            != self.accepted_behavior_hashes
            or self.static_complexity <= 0
            or self.expected_active_item_count <= 0
            or any(not _SHA256.fullmatch(value) for value in hashes)
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate subset receipt drifted"
            )


@dataclass(frozen=True)
class V320TrainCandidateSourceReceiptV2:
    typed_source_receipt_hash: str
    candidate_subset_set_hash: str
    expected_active_route_count: int

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "candidate_material_policy": (
                V320_TRAIN_CANDIDATE_MATERIAL_VERSION
            ),
            "source_commit": V320_SOURCE_COMMIT,
            "protocol_lock_file_sha256": V320_PROTOCOL_LOCK_SHA256,
            "recursive_events_file_sha256": V320_RECURSIVE_EVENTS_SHA256,
            "recursive_archive_file_sha256": (
                V320_RECURSIVE_ARCHIVE_SHA256
            ),
            "recursive_report_file_sha256": (
                V320_RECURSIVE_REPORT_SHA256
            ),
            "recursive_archive_hash": V320_RECURSIVE_ARCHIVE_HASH,
            "manifest_hash": V320_MANIFEST_HASH,
            "evaluator_epoch": V320_EVALUATOR_EPOCH,
            "model": V320_MODEL,
            "max_steps": V320_MAX_STEPS,
            "typed_snapshot_ledger_hash": (
                HISTORICAL_PROJECTED_LEDGER_HASH
            ),
            "typed_source_receipt_hash": self.typed_source_receipt_hash,
            "candidate_subset_set_hash": self.candidate_subset_set_hash,
            "generation_count": 2,
            "program_count": 6,
            "candidate_subset_count": 14,
            "expected_full_outcome_count": 14 * 38,
            "expected_active_route_count": self.expected_active_route_count,
            "expected_inactive_replay_count": (
                14 * 38 - self.expected_active_route_count
            ),
            "source_recursive_and_no_recursive_payloads_identical": True,
            "validation_or_test_content_accessed": False,
            "model_calls": 0,
            "evaluator_calls": 0,
            "raw_program_or_task_content_persisted": False,
        }

    def verify(self) -> None:
        if (
            not _SHA256.fullmatch(self.typed_source_receipt_hash)
            or not _SHA256.fullmatch(self.candidate_subset_set_hash)
            or self.expected_active_route_count != 56
            or self.safe_payload()["expected_inactive_replay_count"] != 476
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate source receipt drifted"
            )


@dataclass(frozen=True)
class V320TrainCandidateMaterialV2:
    receipt: V320TrainCandidateSourceReceiptV2
    typed_source: HistoricalTypedSelectionSourceReceiptV2 = field(
        compare=False,
        repr=False,
    )
    programs: tuple[HypothesisProgram, ...] = field(
        compare=False,
        repr=False,
    )
    typed_program_registry: TypedProgramBindingRegistry = field(
        compare=False,
        repr=False,
    )
    subsets: tuple[V320TrainCandidateSubsetV2, ...]

    def program_set_for(
        self,
        subset: V320TrainCandidateSubsetV2,
    ) -> tuple[HypothesisProgram, ...]:
        by_id = {row.id: row for row in self.programs}
        try:
            return tuple(by_id[value] for value in subset.program_ids)
        except KeyError as exc:
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate references an unknown program"
            ) from exc

    def verify(self) -> None:
        self.receipt.verify()
        self.typed_source.verify()
        if (
            self.typed_source.receipt_hash
            != self.receipt.typed_source_receipt_hash
            or self.typed_source.projected_snapshot_ledger_hash
            != HISTORICAL_PROJECTED_LEDGER_HASH
            or len(self.programs) != 6
            or len(self.subsets) != 14
            or tuple(sorted(self.subsets, key=lambda row: row.subset_hash))
            != self.subsets
            or len({row.canonical_set_hash for row in self.subsets}) != 14
            or sum(row.expected_active_item_count for row in self.subsets)
            != self.receipt.expected_active_route_count
            or stable_hash(
                {"candidate_subsets": [row.safe_payload() for row in self.subsets]}
            )
            != self.receipt.candidate_subset_set_hash
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate material receipt drifted"
            )
        program_ids = {row.id for row in self.programs}
        subset_program_ids = {
            value for subset in self.subsets for value in subset.program_ids
        }
        if program_ids != subset_program_ids:
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate material program coverage drifted"
            )
        for program in self.programs:
            if program.validate() or program.status is not HypothesisStatus.SHADOW:
                raise V320TrainCandidateMaterialError(
                    "v3.20 candidate material program is invalid"
                )
            binding = self.typed_program_registry.require(program)
            if binding.program_executable_hash != _behavior_hash(program):
                raise V320TrainCandidateMaterialError(
                    "v3.20 candidate executable binding drifted"
                )
        for subset in self.subsets:
            subset.verify()
            programs = self.program_set_for(subset)
            if (
                tuple(sorted(row.payload_hash for row in programs))
                == subset.accepted_hypothesis_hashes
            ):
                raise V320TrainCandidateMaterialError(
                    "v3.20 candidate receipt retained post-decision hashes"
                )
            if tuple(
                sorted(
                    replace(
                        row,
                        status=HypothesisStatus.CANDIDATE,
                    ).payload_hash
                    for row in programs
                )
            ) != subset.accepted_hypothesis_hashes or tuple(
                sorted(_behavior_hash(row) for row in programs)
            ) != subset.accepted_behavior_hashes:
                raise V320TrainCandidateMaterialError(
                    "v3.20 candidate program evidence drifted"
                )


def _programs_and_registry(
    archive: Mapping[str, Any],
    *,
    typed_source: HistoricalTypedSelectionSourceReceiptV2,
) -> tuple[tuple[HypothesisProgram, ...], TypedProgramBindingRegistry]:
    raw_programs = archive.get("hypotheses")
    raw_bindings = archive.get("typed_bindings")
    raw_history = archive.get("typed_selection_history")
    if (
        not isinstance(raw_programs, Mapping)
        or not isinstance(raw_bindings, Mapping)
        or not isinstance(raw_history, Mapping)
        or set(raw_programs) != set(raw_bindings)
    ):
        raise V320TrainCandidateMaterialError(
            "v3.20 archive typed program coverage drifted"
        )
    programs: dict[str, HypothesisProgram] = {}
    for key, raw_program in raw_programs.items():
        if not isinstance(key, str) or not isinstance(raw_program, Mapping):
            raise V320TrainCandidateMaterialError(
                "v3.20 archive hypothesis row is malformed"
            )
        try:
            program = HypothesisProgram.from_dict(raw_program)
        except (KeyError, TypeError, ValueError) as exc:
            raise V320TrainCandidateMaterialError(
                "v3.20 archive hypothesis cannot be restored"
            ) from exc
        if (
            program.id != key
            or program.validate()
            or program.status is not HypothesisStatus.SHADOW
            or program.evaluator_epoch != V320_EVALUATOR_EPOCH
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 archive hypothesis provenance drifted"
            )
        programs[key] = program

    registry = TypedProgramBindingRegistry(
        snapshot_ledger=typed_source.ledger.production_snapshot_ledger
    )
    sortable: list[tuple[int, str, Mapping[str, Any]]] = []
    for key, raw_binding in raw_bindings.items():
        if not isinstance(raw_binding, Mapping):
            raise V320TrainCandidateMaterialError(
                "v3.20 archive typed binding is malformed"
            )
        lineage = raw_binding.get("lineage_program_ids")
        if not isinstance(lineage, list) or not all(
            isinstance(value, str) for value in lineage
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 archive binding lineage is malformed"
            )
        sortable.append((len(lineage), str(key), raw_binding))
    binding_hashes: set[str] = set()
    for _, key, raw_binding in sorted(sortable):
        try:
            restored = registry.restore_safe_payload(
                programs[key],
                raw_binding,
            )
        except (KeyError, PermissionError, TypeError, ValueError) as exc:
            raise V320TrainCandidateMaterialError(
                "v3.20 archive typed binding cannot be restored"
            ) from exc
        if restored.program_executable_hash != _behavior_hash(programs[key]):
            raise V320TrainCandidateMaterialError(
                "v3.20 archive behavior binding drifted"
            )
        binding_hashes.add(restored.binding_hash)
    try:
        history = validate_typed_selection_history_payloads(
            raw_history,
            snapshot_ledger=typed_source.ledger.production_snapshot_ledger,
        )
    except PermissionError as exc:
        raise V320TrainCandidateMaterialError(
            "v3.20 archive typed selection history drifted"
        ) from exc
    if binding_hashes != {str(row["binding_hash"]) for row in history}:
        raise V320TrainCandidateMaterialError(
            "v3.20 archive typed history coverage drifted"
        )
    return tuple(programs[key] for key in sorted(programs)), registry


def _validate_candidate_payload(
    payload: Mapping[str, Any],
    *,
    generation: int,
    trace_id: str,
    programs_by_id: Mapping[str, HypothesisProgram],
) -> tuple[V320TrainCandidateSubsetV2, ...]:
    selected_ids = payload.get("selected_candidate_hypothesis_ids")
    raw_subsets = payload.get("candidate_subsets")
    candidates = payload.get("candidates")
    if (
        payload.get("candidate_bundle_policy") != _CANDIDATE_BUNDLE_POLICY
        or payload.get("selection_uses_validation") is not False
        or payload.get("selection_uses_validation_outcomes") is not False
        or payload.get("proposal_candidate_count") != 3
        or payload.get("static_accepted_candidate_count") != 3
        or payload.get("repaired_candidate_count") != 0
        or payload.get("repair_model_failure_count") != 0
        or not isinstance(selected_ids, list)
        or len(selected_ids) != 3
        or selected_ids != sorted(set(selected_ids))
        or not isinstance(raw_subsets, list)
        or len(raw_subsets) != 7
        or not isinstance(candidates, list)
        or len(candidates) != 3
    ):
        raise V320TrainCandidateMaterialError(
            "v3.20 candidate selection payload drifted"
        )
    if any(value not in programs_by_id for value in selected_ids):
        raise V320TrainCandidateMaterialError(
            "v3.20 candidate selection references an unknown program"
        )
    candidate_rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, Mapping):
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate audit row is malformed"
            )
        accepted_id = row.get("accepted_id")
        if not isinstance(accepted_id, str) or accepted_id in candidate_rows_by_id:
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate audit identity drifted"
            )
        program = programs_by_id.get(accepted_id)
        if program is None:
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate audit references an unknown program"
            )
        candidate_hash = replace(
            program,
            status=HypothesisStatus.CANDIDATE,
        ).payload_hash
        if (
            row.get("root_id") != accepted_id
            or row.get("accepted_hash") != candidate_hash
            or row.get("root_hash") != candidate_hash
            or row.get("selected") is not True
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate audit program hash drifted"
            )
        candidate_rows_by_id[accepted_id] = row
    if set(candidate_rows_by_id) != set(selected_ids):
        raise V320TrainCandidateMaterialError(
            "v3.20 selected candidate audit coverage drifted"
        )

    expected_power_set = {
        tuple(values)
        for size in range(1, len(selected_ids) + 1)
        for values in itertools.combinations(selected_ids, size)
    }
    seen_subsets: set[tuple[str, ...]] = set()
    subsets: list[V320TrainCandidateSubsetV2] = []
    for raw_subset in raw_subsets:
        if not isinstance(raw_subset, Mapping) or set(raw_subset) != _SUBSET_FIELDS:
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate subset schema drifted"
            )
        ids = raw_subset.get("accepted_hypothesis_ids")
        metrics = raw_subset.get("union_training_metrics")
        if (
            not isinstance(ids, list)
            or not ids
            or ids != sorted(set(ids))
            or not isinstance(metrics, Mapping)
            or raw_subset.get("root_hypothesis_ids") != ids
            or raw_subset.get("selection_uses_validation") is not False
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate subset identity drifted"
            )
        key = tuple(ids)
        if key not in expected_power_set or key in seen_subsets:
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate subsets are not the exact power set"
            )
        seen_subsets.add(key)
        programs = tuple(programs_by_id[value] for value in key)
        accepted_hashes = tuple(
            sorted(
                replace(
                    program,
                    status=HypothesisStatus.CANDIDATE,
                ).payload_hash
                for program in programs
            )
        )
        behavior_hashes = tuple(
            sorted(_behavior_hash(program) for program in programs)
        )
        canonical_set_hash = stable_hash(
            {
                "accepted_hypothesis_ids": list(key),
                "accepted_behavior_hashes": list(behavior_hashes),
            }
        )
        selected = raw_subset.get("selected")
        if (
            raw_subset.get("accepted_hypothesis_hashes")
            != list(accepted_hashes)
            or raw_subset.get("root_hypothesis_hashes")
            != list(accepted_hashes)
            or raw_subset.get("accepted_behavior_hashes")
            != list(behavior_hashes)
            or raw_subset.get("canonical_set_hash") != canonical_set_hash
            or not isinstance(selected, bool)
            or metrics.get("bundle_size") != len(key)
            or metrics.get("activation_precision_numerator")
            != metrics.get("failure_activation_count")
            or metrics.get("activation_precision_denominator")
            != metrics.get("failure_activation_count")
            or metrics.get("success_false_positive_activation_count") != 0
        ):
            raise V320TrainCandidateMaterialError(
                "v3.20 candidate subset evidence drifted"
            )
        subsets.append(
            V320TrainCandidateSubsetV2(
                generation=generation,
                source_trace_id_hash=stable_hash({"trace_id": trace_id}),
                canonical_set_hash=canonical_set_hash,
                program_ids=key,
                accepted_hypothesis_hashes=accepted_hashes,
                accepted_behavior_hashes=behavior_hashes,
                static_complexity=_strict_positive_int(
                    metrics.get("complexity"),
                    "candidate complexity",
                ),
                expected_active_item_count=_strict_positive_int(
                    metrics.get("failure_activation_count"),
                    "candidate active item count",
                ),
                selected=selected,
            )
        )
    if seen_subsets != expected_power_set:
        raise V320TrainCandidateMaterialError(
            "v3.20 candidate power set is incomplete"
        )
    selected_subsets = [row for row in subsets if row.selected]
    if (
        len(selected_subsets) != 1
        or selected_subsets[0].program_ids != tuple(selected_ids)
        or payload.get("selected_candidate_set_hash")
        != selected_subsets[0].canonical_set_hash
    ):
        raise V320TrainCandidateMaterialError(
            "v3.20 historical outer-loop selection drifted"
        )
    return tuple(subsets)


def load_v320_train_candidate_material_v2(
    *,
    project_root: Path,
    source_root: Path | None = None,
) -> V320TrainCandidateMaterialV2:
    """Restore the 14 unique v3.20 TRAIN-only candidate subsets.

    This is a read-only compatibility adapter.  It authenticates the exact
    historical protocol, report, archive, and event ledger, restores the six
    opaque typed bindings under the current projected ledger, and proves that
    recursive/no-recursive copies contain the same two exact power sets.  It
    performs no model, evaluator, validation, or test access.
    """

    project = project_root.resolve(strict=True)
    source = (
        source_root.resolve(strict=True)
        if source_root is not None
        else (project / V320_SOURCE_RELATIVE_ROOT).resolve(strict=True)
    )
    try:
        source.relative_to(project)
    except ValueError as exc:
        raise V320TrainCandidateMaterialError(
            "v3.20 source root escaped the project"
        ) from exc
    if not source.is_dir() or source.is_symlink():
        raise V320TrainCandidateMaterialError(
            "v3.20 source root is not a regular directory"
        )

    protocol_path = _verified_file(
        source,
        "protocol_lock.json",
        V320_PROTOCOL_LOCK_SHA256,
    )
    events_path = _verified_file(
        source,
        "development_recursive.events.jsonl",
        V320_RECURSIVE_EVENTS_SHA256,
    )
    archive_path = _verified_file(
        source,
        "development_recursive.archive.json",
        V320_RECURSIVE_ARCHIVE_SHA256,
    )
    report_path = _verified_file(
        source,
        "development_recursive.report.json",
        V320_RECURSIVE_REPORT_SHA256,
    )
    protocol = _json_object(protocol_path, "protocol lock")
    archive = _json_object(archive_path, "recursive archive")
    report = _json_object(report_path, "recursive report")
    events = _load_events(events_path)

    protocol_without_hash = dict(protocol)
    lock_hash = protocol_without_hash.pop("lock_hash", None)
    execution = protocol.get("execution")
    if (
        lock_hash != stable_hash(protocol_without_hash)
        or protocol.get("git", {}).get("commit") != V320_SOURCE_COMMIT
        or protocol.get("git", {}).get("scoped_dirty") is not False
        or protocol.get("primary_manifest_hash") != V320_MANIFEST_HASH
        or protocol.get("model") != V320_MODEL
        or protocol.get("max_steps") != V320_MAX_STEPS
        or protocol.get("sealed_test_content_accessed") is not False
        or protocol.get("sealed_test_scoring_performed") is not False
        or not isinstance(execution, Mapping)
        or execution.get("typed_selection_snapshot_source", {}).get(
            "snapshot_ledger_hash"
        )
        != HISTORICAL_PROJECTED_LEDGER_HASH
        or execution.get("codex_agent_execution_policy", {}).get(
            "web_search_mode"
        )
        != "disabled"
    ):
        raise V320TrainCandidateMaterialError(
            "v3.20 protocol authority drifted"
        )
    if (
        set(archive) != _ARCHIVE_FIELDS
        or archive.get("archive_hash") != V320_RECURSIVE_ARCHIVE_HASH
        or archive.get("incumbent_id") is not None
        or archive.get("raw_content_persisted") is not False
        or report.get("archive_hash") != V320_RECURSIVE_ARCHIVE_HASH
        or report.get("mode") != "execute"
        or report.get("executed") is not True
        or report.get("test_content_accessed") is not False
    ):
        raise V320TrainCandidateMaterialError(
            "v3.20 recursive development authority drifted"
        )

    typed_source = load_historical_portable_typed_selection_projection_v2(
        project_root=project
    )
    programs, registry = _programs_and_registry(
        archive,
        typed_source=typed_source,
    )
    programs_by_id = {row.id: row for row in programs}
    if _archive_hash(archive, programs_by_id) != V320_RECURSIVE_ARCHIVE_HASH:
        raise V320TrainCandidateMaterialError(
            "v3.20 recursive archive content hash drifted"
        )

    selection_events = {
        str(row["trace_id"]): row
        for row in events
        if row["event"] == _CANDIDATE_SELECTION_EVENT
    }
    required_traces = {
        trace_id
        for _, recursive_trace, no_recursive_trace in _REQUIRED_TRACE_PAIRS
        for trace_id in (recursive_trace, no_recursive_trace)
    }
    if set(selection_events) != required_traces:
        raise V320TrainCandidateMaterialError(
            "v3.20 candidate selection event coverage drifted"
        )
    all_subsets: list[V320TrainCandidateSubsetV2] = []
    generation_program_ids: list[set[str]] = []
    for generation, recursive_trace, no_recursive_trace in _REQUIRED_TRACE_PAIRS:
        recursive_payload = selection_events[recursive_trace]["payload"]
        no_recursive_payload = selection_events[no_recursive_trace]["payload"]
        if recursive_payload != no_recursive_payload:
            raise V320TrainCandidateMaterialError(
                "v3.20 paired candidate selection payloads differ"
            )
        subsets = _validate_candidate_payload(
            recursive_payload,
            generation=generation,
            trace_id=recursive_trace,
            programs_by_id=programs_by_id,
        )
        all_subsets.extend(subsets)
        generation_program_ids.append(
            {
                value for subset in subsets for value in subset.program_ids
            }
        )
    if (
        generation_program_ids[0] & generation_program_ids[1]
        or generation_program_ids[0] | generation_program_ids[1]
        != set(programs_by_id)
    ):
        raise V320TrainCandidateMaterialError(
            "v3.20 candidate generations are not disjoint and complete"
        )

    canonical_subsets = tuple(sorted(all_subsets, key=lambda row: row.subset_hash))
    subset_set_hash = stable_hash(
        {"candidate_subsets": [row.safe_payload() for row in canonical_subsets]}
    )
    receipt = V320TrainCandidateSourceReceiptV2(
        typed_source_receipt_hash=typed_source.receipt_hash,
        candidate_subset_set_hash=subset_set_hash,
        expected_active_route_count=sum(
            row.expected_active_item_count for row in canonical_subsets
        ),
    )
    material = V320TrainCandidateMaterialV2(
        receipt=receipt,
        typed_source=typed_source,
        programs=programs,
        typed_program_registry=registry,
        subsets=canonical_subsets,
    )
    material.verify()
    return material
