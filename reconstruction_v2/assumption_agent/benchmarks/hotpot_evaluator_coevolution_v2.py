"""Prospective offline Hotpot evaluator transition and search-utility study.

The module implements one finite evaluator epoch transition.  A fixed policy
DSL selects typed Q components from the same frozen 84-program grammar.  The
incumbent maximizes direct Q support; challengers score the derived ``P+Q``
arm.  ``A_form`` alone forms one action-distinct challenger and
``F_search`` freezes both future selections before ``A_hold`` may open.

The anchor executes only retained P and the two distinct A-form Q actions.
All 72 retrievals join before source-provided labels are consulted.  A
one-sided exact paired sign-flip p-value at 0.10 is the sole promotion rule.
Only a promoted transition can authorize ``M_search``.  Its four physical
arms (RAW, incumbent combined, active combined, official HippoRAG) execute as
one 96-party barrier, followed by fresh runtime postflight and offline score.
No formal command accepts injected evidence, programs, retrievers, or results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence

from ..archive import EvaluatorEpoch, EvaluatorSpec, PolicyArchive
from ..models import stable_hash
from . import hotpot_recursive_acquisition_v1 as acquisition
from . import hotpot_recursive_l4_v1 as l4
from .hotpot_family_out_runner_v1 import (
    BWRAP_SHA256,
    _probe_bubblewrap,
    verify_capability_receipt,
)
from .l4_retention_protocol_v1 import deterministic_rrf
from .musique_formal_runtime_binding_v2 import PreparedFormalRuntimeV2
from .musique_typed_retriever_formation_v1 import (
    TypedRetrievalProgram,
    enumerate_programs,
)


VERSION = "hotpot_evaluator_coevolution_v2"
EVIDENCE_CACHE_SCHEMA = f"{VERSION}_private_evidence_cache"
A_FORM_SCHEMA = f"{VERSION}_A_form_receipt"
F_SEARCH_SCHEMA = f"{VERSION}_F_search_receipt"
ANCHOR_FREEZE_SCHEMA = f"{VERSION}_A_hold_pre_run_freeze"
ANCHOR_REPORT_SCHEMA = f"{VERSION}_A_hold_aggregate_report"
SEARCH_FREEZE_SCHEMA = f"{VERSION}_M_search_pre_run_freeze"
SEARCH_REPORT_SCHEMA = f"{VERSION}_M_search_aggregate_report"
FAILURE_SCHEMA = f"{VERSION}_failure"
CONSUMPTION_SCHEMA = f"{VERSION}_authorization_consumption"
FORMATION_CONSUMPTION_SCHEMA = f"{VERSION}_formation_consumption"
IMPLEMENTATION_SCHEMA = f"{VERSION}_implementation"

TOP_K = acquisition.TOP_K
FOLD_COUNT = 4
A_ITEM_COUNT = acquisition.BLOCK_COUNTS["A_form"]
F_SEARCH_ITEM_COUNT = acquisition.BLOCK_COUNTS["F_search"]
A_HOLD_ITEM_COUNT = acquisition.BLOCK_COUNTS["A_hold"]
M_SEARCH_ITEM_COUNT = acquisition.BLOCK_COUNTS["M_search"]
CANDIDATE_COUNT = len(tuple(enumerate_programs()))
ANCHOR_COMPONENT_IDS = ("retained_P", "incumbent_Q", "challenger_Q")
SEARCH_ARM_IDS = (
    "canonical_RAW",
    "incumbent_combined",
    "active_combined",
    "official_HippoRAG",
)
ANCHOR_WORK_UNIT_COUNT = len(ANCHOR_COMPONENT_IDS) * A_HOLD_ITEM_COUNT
SEARCH_WORK_UNIT_COUNT = len(SEARCH_ARM_IDS) * M_SEARCH_ITEM_COUNT
ANCHOR_MAXIMUM_CONCURRENCY = ANCHOR_WORK_UNIT_COUNT
SEARCH_MAXIMUM_CONCURRENCY = SEARCH_WORK_UNIT_COUNT
PROMOTION_ALPHA = Fraction(1, 10)

ANCHOR_CONSUMPTION_FILENAME = "a_hold.authorization.consumed.json"
ANCHOR_PRIVATE_EVIDENCE_FILENAME = "a_hold.private.evidence.json"
ANCHOR_REPORT_FILENAME = "a_hold.aggregate.report.json"
ANCHOR_FAILURE_FILENAME = "a_hold.failure.json"
SEARCH_CONSUMPTION_FILENAME = "m_search.authorization.consumed.json"
SEARCH_REPORT_FILENAME = "m_search.aggregate.report.json"
SEARCH_FAILURE_FILENAME = "m_search.failure.json"
A_FORM_CONSUMPTION_RELATIVE = (
    "artifacts/hotpot_recursive_evaluator_v2/a_form.authorization.consumed.json"
)
F_SEARCH_CONSUMPTION_RELATIVE = (
    "artifacts/hotpot_recursive_evaluator_v2/f_search.authorization.consumed.json"
)

IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/archive.py",
    "assumption_agent/evaluation.py",
    "assumption_agent/events.py",
    "assumption_agent/models.py",
    "assumption_agent/benchmarks/hotpot_family_out_acquisition_v1.py",
    "assumption_agent/benchmarks/hotpot_recursive_acquisition_v1.py",
    "assumption_agent/benchmarks/hotpot_recursive_l4_v1.py",
    "assumption_agent/benchmarks/hotpot_evaluator_coevolution_v2.py",
    "assumption_agent/benchmarks/hotpot_family_out_runner_v1.py",
    "assumption_agent/benchmarks/l4_retention_protocol_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_m1_retrieval_runner_v1.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class HotpotEvaluatorCoevolutionError(RuntimeError):
    """The prospective evaluator, custody, or execution contract drifted."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HotpotEvaluatorCoevolutionError("required file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HotpotEvaluatorCoevolutionError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise HotpotEvaluatorCoevolutionError(
                f"{field_name} path contains a symlink"
            )
    if not candidate.is_file():
        raise HotpotEvaluatorCoevolutionError(f"{field_name} is unavailable")
    raw = candidate.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotEvaluatorCoevolutionError(f"{field_name} is invalid") from exc
    if not isinstance(payload, dict):
        raise HotpotEvaluatorCoevolutionError(f"{field_name} must be one object")
    return payload, raw


def _write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, mode: int = 0o600
) -> None:
    raw = json.dumps(
        payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answer"',
        '"corpus"',
        '"item_id"',
        '"paragraph_text"',
        '"private_block_path"',
        '"private_evidence_path"',
        '"question"',
        '"support_indices"',
        '"text"',
        "/home/",
        "/tmp/",
    )
    if any(value in raw for value in forbidden):
        raise HotpotEvaluatorCoevolutionError(
            "public evaluator artifact contains private content or path"
        )


def current_implementation_binding(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotEvaluatorCoevolutionError(
                f"implementation file missing or symlinked: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


@dataclass(frozen=True)
class ItemProgramEvidence:
    item_commitment_sha256: str
    q_direct_hits: int
    combined_hits: int
    support_total: int
    combined_complete: int
    combined_coverage: int
    retained_added: int
    novelty_added: int
    retained_displaced: int
    invalid: bool
    action_sha256: str

    def validate(self) -> "ItemProgramEvidence":
        _require_sha256(self.item_commitment_sha256, "item commitment")
        _require_sha256(self.action_sha256, "retrieval action")
        integer_fields = (
            self.q_direct_hits,
            self.combined_hits,
            self.support_total,
            self.combined_complete,
            self.combined_coverage,
            self.retained_added,
            self.novelty_added,
            self.retained_displaced,
        )
        if (
            any(type(value) is not int or value < 0 for value in integer_fields)
            or self.support_total <= 0
            or self.q_direct_hits > self.support_total
            or self.combined_hits > self.support_total
            or self.combined_complete not in {0, 1}
            or self.combined_coverage not in {0, 1}
            or type(self.invalid) is not bool
            or (self.invalid and any(value != 0 for value in integer_fields[:2]))
        ):
            raise HotpotEvaluatorCoevolutionError("item evidence is malformed")
        return self


@dataclass(frozen=True)
class ProgramEvidence:
    program_sha256: str
    program_length: int
    rows: tuple[ItemProgramEvidence, ...]

    def validate(self) -> "ProgramEvidence":
        _require_sha256(self.program_sha256, "program hash")
        if type(self.program_length) is not int or self.program_length <= 0:
            raise HotpotEvaluatorCoevolutionError("program length is invalid")
        if len(self.rows) < FOLD_COUNT:
            raise HotpotEvaluatorCoevolutionError("too few evaluator evidence rows")
        commitments = [row.validate().item_commitment_sha256 for row in self.rows]
        if len(set(commitments)) != len(commitments):
            raise HotpotEvaluatorCoevolutionError("duplicate evidence item")
        return self


@dataclass(frozen=True)
class EvaluatorPolicy:
    id: str
    objective: str
    complexity: int

    @property
    def policy_sha256(self) -> str:
        return stable_hash(asdict(self))


INCUMBENT_POLICY_ID = "q_direct_micro_v2"


def evaluator_policies() -> tuple[EvaluatorPolicy, ...]:
    """Return the complete bounded policy DSL in deterministic order."""

    return (
        EvaluatorPolicy(INCUMBENT_POLICY_ID, "q_direct_micro", 1),
        EvaluatorPolicy("combined_micro_v2", "combined_micro", 1),
        EvaluatorPolicy("combined_complete_v2", "combined_complete", 2),
        EvaluatorPolicy("combined_coverage_v2", "combined_coverage", 2),
        EvaluatorPolicy("combined_retention_v2", "combined_retention", 3),
        EvaluatorPolicy(
            "combined_novelty_balanced_v2", "combined_novelty_balanced", 4
        ),
    )


def fixed_programs() -> tuple[TypedRetrievalProgram, ...]:
    programs = tuple(enumerate_programs())
    if (
        len(programs) != 84
        or len({program.program_hash for program in programs}) != len(programs)
        or any(program.type_issues() for program in programs)
    ):
        raise HotpotEvaluatorCoevolutionError("fixed typed program pool drifted")
    return programs


def candidate_set_binding() -> dict[str, Any]:
    programs = fixed_programs()
    return {
        "candidate_count": len(programs),
        "program_order_sha256": stable_hash(
            [program.program_hash for program in programs]
        ),
        "program_payload_set_sha256": stable_hash(
            [program.to_dict() for program in programs]
        ),
        "selectable_pool_excludes_retained_P": True,
        "all_candidates_type_valid": True,
        "top_k": TOP_K,
    }


def _validate_evidence_set(
    evidence: Sequence[ProgramEvidence],
) -> tuple[ProgramEvidence, ...]:
    rows = tuple(value.validate() for value in evidence)
    if not rows:
        raise HotpotEvaluatorCoevolutionError("evaluator evidence set is empty")
    hashes = [value.program_sha256 for value in rows]
    if len(set(hashes)) != len(hashes):
        raise HotpotEvaluatorCoevolutionError("duplicate program evidence")
    item_order = tuple(row.item_commitment_sha256 for row in rows[0].rows)
    if any(
        tuple(item.item_commitment_sha256 for item in value.rows) != item_order
        for value in rows[1:]
    ):
        raise HotpotEvaluatorCoevolutionError("program evidence item grids differ")
    return rows


def evidence_set_sha256(evidence: Sequence[ProgramEvidence]) -> str:
    rows = _validate_evidence_set(evidence)
    return stable_hash(
        [
            {
                "program_sha256": value.program_sha256,
                "program_length": value.program_length,
                "rows": [asdict(row) for row in value.rows],
            }
            for value in rows
        ]
    )


def _program_action_sha256(value: ProgramEvidence) -> str:
    value.validate()
    return stable_hash([row.action_sha256 for row in value.rows])


def _totals(
    value: ProgramEvidence, positions: Sequence[int]
) -> dict[str, int]:
    if not positions or any(
        type(position) is not int or not 0 <= position < len(value.rows)
        for position in positions
    ):
        raise HotpotEvaluatorCoevolutionError("evaluator positions are invalid")
    rows = [value.rows[position] for position in positions]
    return {
        "q_direct_hits": sum(row.q_direct_hits for row in rows),
        "combined_hits": sum(row.combined_hits for row in rows),
        "support_total": sum(row.support_total for row in rows),
        "combined_complete": sum(row.combined_complete for row in rows),
        "combined_coverage": sum(row.combined_coverage for row in rows),
        "retained_added": sum(row.retained_added for row in rows),
        "novelty_added": sum(row.novelty_added for row in rows),
        "retained_displaced": sum(row.retained_displaced for row in rows),
        "invalid_count": sum(row.invalid for row in rows),
    }


def _policy_values(
    policy: EvaluatorPolicy, totals: Mapping[str, int]
) -> tuple[int, ...]:
    q_hits = totals["q_direct_hits"]
    combo = totals["combined_hits"]
    retained = totals["retained_added"]
    novelty = totals["novelty_added"]
    displaced = totals["retained_displaced"]
    if policy.objective == "q_direct_micro":
        return (q_hits, combo)
    if policy.objective == "combined_micro":
        return (combo, q_hits)
    if policy.objective == "combined_complete":
        return (totals["combined_complete"], combo, q_hits)
    if policy.objective == "combined_coverage":
        return (totals["combined_coverage"], combo, q_hits)
    if policy.objective == "combined_retention":
        return (retained - displaced, combo, retained, -displaced, q_hits)
    if policy.objective == "combined_novelty_balanced":
        return (
            min(retained, novelty),
            combo,
            retained + novelty,
            -displaced,
            q_hits,
        )
    raise HotpotEvaluatorCoevolutionError("unknown evaluator policy")


def select_program(
    policy: EvaluatorPolicy,
    evidence: Sequence[ProgramEvidence],
    *,
    retained_p_program_sha256: str,
    positions: Sequence[int] | None = None,
) -> ProgramEvidence:
    rows = _validate_evidence_set(evidence)
    _require_sha256(retained_p_program_sha256, "retained P program hash")
    selected_positions = (
        tuple(range(len(rows[0].rows))) if positions is None else tuple(positions)
    )
    eligible = [row for row in rows if row.program_sha256 != retained_p_program_sha256]
    if not eligible:
        raise HotpotEvaluatorCoevolutionError("no selectable Q program remains")
    return min(
        eligible,
        key=lambda row: (
            _totals(row, selected_positions)["invalid_count"],
            *(-value for value in _policy_values(
                policy, _totals(row, selected_positions)
            )),
            row.program_length,
            row.program_sha256,
        ),
    )


def _formation_core_body(
    evidence: Sequence[ProgramEvidence], *, retained_p_program_sha256: str
) -> dict[str, Any]:
    rows = _validate_evidence_set(evidence)
    item_count = len(rows[0].rows)
    incumbent = evaluator_policies()[0]
    incumbent_selected = select_program(
        incumbent,
        rows,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    crossfit: list[dict[str, Any]] = []
    full_selected: dict[str, str] = {}
    for policy in evaluator_policies():
        full = select_program(
            policy,
            rows,
            retained_p_program_sha256=retained_p_program_sha256,
        )
        full_selected[policy.id] = full.program_sha256
        held_hits = held_total = 0
        folds: list[dict[str, Any]] = []
        for fold in range(FOLD_COUNT):
            held_positions = tuple(
                position for position in range(item_count)
                if position % FOLD_COUNT == fold
            )
            held_set = set(held_positions)
            fit_positions = tuple(
                position for position in range(item_count)
                if position not in held_set
            )
            selected = select_program(
                policy,
                rows,
                retained_p_program_sha256=retained_p_program_sha256,
                positions=fit_positions,
            )
            totals = _totals(selected, held_positions)
            held_hits += totals["combined_hits"]
            held_total += totals["support_total"]
            folds.append(
                {
                    "fold": fold,
                    "fit_item_count": len(fit_positions),
                    "held_item_count": len(held_positions),
                    "selected_program_sha256": selected.program_sha256,
                    "held_combined_support_hits": totals["combined_hits"],
                    "held_support_total": totals["support_total"],
                }
            )
        crossfit.append(
            {
                "policy": asdict(policy),
                "policy_sha256": policy.policy_sha256,
                "full_selected_program_sha256": full.program_sha256,
                "full_selected_action_sha256": _program_action_sha256(full),
                "held_combined_support_hits": held_hits,
                "held_support_total": held_total,
                "folds": folds,
            }
        )
    distinct = [
        row for row in crossfit
        if row["policy"]["id"] != INCUMBENT_POLICY_ID
        and row["full_selected_program_sha256"]
        != incumbent_selected.program_sha256
        and row["full_selected_action_sha256"]
        != _program_action_sha256(incumbent_selected)
    ]
    if not distinct:
        raise HotpotEvaluatorCoevolutionError(
            "no action-distinct evaluator challenger formed on A_form"
        )
    chosen = min(
        distinct,
        key=lambda row: (
            -row["held_combined_support_hits"],
            row["policy"]["complexity"],
            row["policy_sha256"],
        ),
    )
    challenger = next(
        policy for policy in evaluator_policies()
        if policy.id == chosen["policy"]["id"]
    )
    return {
        "schema": f"{VERSION}_formation_core",
        "partition": "A_form",
        "candidate_program_count": len(rows),
        "item_count": item_count,
        "fold_policy": "private_block_position_modulo_4_v1",
        "fold_count": FOLD_COUNT,
        "formation_objective": "held_P_Q_combined_source_support_hits",
        "evidence_set_sha256": evidence_set_sha256(rows),
        "retained_P_program_sha256": retained_p_program_sha256,
        "incumbent_policy": asdict(incumbent),
        "challenger_policy": asdict(challenger),
        "incumbent_selected_program_sha256": incumbent_selected.program_sha256,
        "incumbent_selected_action_sha256": _program_action_sha256(
            incumbent_selected
        ),
        "challenger_selected_program_sha256": chosen[
            "full_selected_program_sha256"
        ],
        "challenger_selected_action_sha256": chosen[
            "full_selected_action_sha256"
        ],
        "action_identical_policy_excluded_from_challenger": True,
        "crossfit": crossfit,
        "anchor_accessed": False,
        "search_measurement_accessed": False,
        "model_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }


def form_challenger_from_evidence(
    evidence: Sequence[ProgramEvidence], *, retained_p_program_sha256: str
) -> dict[str, Any]:
    """Pure formation core used by the formal A-form stage and unit tests."""

    body = _formation_core_body(
        evidence, retained_p_program_sha256=retained_p_program_sha256
    )
    return {**body, "formation_sha256": stable_hash(body)}


def _verify_formation_core(
    receipt: Mapping[str, Any],
    evidence: Sequence[ProgramEvidence],
    *,
    retained_p_program_sha256: str,
) -> tuple[EvaluatorPolicy, EvaluatorPolicy]:
    expected = form_challenger_from_evidence(
        evidence, retained_p_program_sha256=retained_p_program_sha256
    )
    if dict(receipt) != expected:
        raise HotpotEvaluatorCoevolutionError("A-form core receipt drifted")
    policies = {policy.id: policy for policy in evaluator_policies()}
    try:
        incumbent = policies[str(receipt["incumbent_policy"]["id"])]
        challenger = policies[str(receipt["challenger_policy"]["id"])]
    except (KeyError, TypeError) as exc:
        raise HotpotEvaluatorCoevolutionError("evaluator policy identity drifted") from exc
    return incumbent, challenger


def freeze_search_choices_from_evidence(
    *,
    search_evidence: Sequence[ProgramEvidence],
    a_form_evidence: Sequence[ProgramEvidence],
    formation_receipt: Mapping[str, Any],
    retained_p_program_sha256: str,
) -> dict[str, Any]:
    """Freeze both future selections on F_search before A_hold opens."""

    incumbent, challenger = _verify_formation_core(
        formation_receipt,
        a_form_evidence,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    rows = _validate_evidence_set(search_evidence)
    old = select_program(
        incumbent,
        rows,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    new = select_program(
        challenger,
        rows,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    old_action_sha256 = _program_action_sha256(old)
    new_action_sha256 = _program_action_sha256(new)
    body: dict[str, Any] = {
        "schema": f"{VERSION}_search_formation_core",
        "partition": "F_search",
        "formation_sha256": formation_receipt["formation_sha256"],
        "evidence_set_sha256": evidence_set_sha256(rows),
        "candidate_program_count": len(rows),
        "item_count": len(rows[0].rows),
        "incumbent_policy_id": incumbent.id,
        "challenger_policy_id": challenger.id,
        "incumbent_selected_program_sha256": old.program_sha256,
        "incumbent_selected_action_sha256": old_action_sha256,
        "challenger_selected_program_sha256": new.program_sha256,
        "challenger_selected_action_sha256": new_action_sha256,
        "measurable_contrast": (
            old.program_sha256 != new.program_sha256
            and old_action_sha256 != new_action_sha256
        ),
        "behavior_distinct_required": True,
        "identical_action_has_no_fallback": True,
        "anchor_accessed": False,
        "search_measurement_accessed": False,
        "model_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    return {**body, "search_formation_sha256": stable_hash(body)}


def exact_paired_sign_flip(deltas: Sequence[int]) -> dict[str, Any]:
    """Return a deterministic one-sided exact randomization p-value."""

    if not deltas or any(type(value) is not int for value in deltas):
        raise HotpotEvaluatorCoevolutionError("paired deltas are malformed")
    observed = sum(deltas)
    magnitudes = [abs(value) for value in deltas if value != 0]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        next_distribution: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            next_distribution[subtotal + magnitude] += count
            next_distribution[subtotal - magnitude] += count
        distribution = next_distribution
    denominator = 1 << len(magnitudes)
    numerator = sum(
        count for subtotal, count in distribution.items() if subtotal >= observed
    )
    p_value = Fraction(numerator, denominator)
    return {
        "test": "one_sided_exact_paired_sign_flip_v1",
        "observed_net_support_hits": observed,
        "nonzero_pair_count": len(magnitudes),
        "p_value_numerator": p_value.numerator,
        "p_value_denominator": p_value.denominator,
        "p_value": float(p_value),
        "alpha_numerator": PROMOTION_ALPHA.numerator,
        "alpha_denominator": PROMOTION_ALPHA.denominator,
        "promoted": p_value <= PROMOTION_ALPHA,
        "sole_promotion_criterion": True,
    }


def _archive_transition(
    *,
    incumbent_policy: EvaluatorPolicy,
    challenger_policy: EvaluatorPolicy,
    anchor_manifest_sha256: str,
    incumbent_hits: int,
    challenger_hits: int,
    support_total: int,
    item_count: int,
    promoted: bool,
) -> dict[str, Any]:
    """Apply an exact-decision epoch transition and selective invalidation."""

    _require_sha256(anchor_manifest_sha256, "anchor manifest")
    incumbent_spec = EvaluatorSpec(
        id=incumbent_policy.id,
        version=VERSION,
        implementation_hash=stable_hash({"module": VERSION}),
        criteria_hash=incumbent_policy.policy_sha256,
        anchor_manifest_hash=anchor_manifest_sha256,
    )
    challenger_spec = EvaluatorSpec(
        id=challenger_policy.id,
        version=VERSION,
        implementation_hash=stable_hash({"module": VERSION}),
        criteria_hash=challenger_policy.policy_sha256,
        anchor_manifest_hash=anchor_manifest_sha256,
    )
    incumbent_epoch = EvaluatorEpoch(
        id=f"eval_epoch_0_{stable_hash(asdict(incumbent_spec))[:10]}",
        index=0,
        evaluator=incumbent_spec,
    )
    archive = PolicyArchive()
    node = archive.create_node(
        active_hypothesis_ids=(),
        evaluator_epoch_id=incumbent_epoch.id,
        runtime_version=VERSION,
    )
    safe_item_ids = tuple(
        stable_hash({"anchor": anchor_manifest_sha256, "ordinal": ordinal})
        for ordinal in range(item_count)
    )
    dependent = archive.record_score(
        archive_node_id=node.id,
        split="A_hold",
        evaluator_epoch_id=incumbent_epoch.id,
        metric="evaluator_dependent_program_selection_support",
        successes=incumbent_hits,
        total=support_total,
        item_ids=safe_item_ids,
    )
    independent = archive.record_score(
        archive_node_id=node.id,
        split="A_hold",
        evaluator_epoch_id="fixed_source_support_v1",
        metric="independent_source_support",
        successes=challenger_hits,
        total=support_total,
        item_ids=safe_item_ids,
    )
    if promoted:
        invalidated = archive.invalidate_evaluator_epoch(incumbent_epoch.id)
        next_epoch = EvaluatorEpoch(
            id=(
                "eval_epoch_1_"
                + stable_hash(
                    {
                        "challenger": asdict(challenger_spec),
                        "parent": incumbent_epoch.id,
                    }
                )[:10]
            ),
            index=1,
            evaluator=challenger_spec,
            parent_epoch_id=incumbent_epoch.id,
        )
    else:
        invalidated = ()
        next_epoch = incumbent_epoch
    payload = {
        "promoted": promoted,
        "incumbent_epoch_id": incumbent_epoch.id,
        "next_epoch_id": next_epoch.id,
        "next_evaluator_id": next_epoch.evaluator.id,
        "parent_epoch_id": next_epoch.parent_epoch_id,
        "invalidated_score_record_ids": list(invalidated),
        "dependent_score_record_id": dependent.id,
        "dependent_score_valid_after_transition": archive.score_records[
            dependent.id
        ].valid,
        "independent_source_score_record_id": independent.id,
        "independent_source_score_valid_after_transition": archive.score_records[
            independent.id
        ].valid,
        "selective_invalidation_performed": promoted,
        "independent_source_record_retained": True,
        "archive_snapshot_sha256": stable_hash(archive.to_dict()),
    }
    payload["transition_sha256"] = stable_hash(payload)
    return payload


def _source_binding(
    receipt: Mapping[str, Any], receipt_raw: bytes, block: str
) -> dict[str, Any]:
    commitment = l4._commitment(receipt, block)
    return {
        "acquisition_file_sha256": _sha256_bytes(receipt_raw),
        "acquisition_sha256": receipt["acquisition_sha256"],
        "block_id_sha256": stable_hash({"block": block}),
        "block_file_sha256": commitment.file_sha256,
        "item_commitment_set_sha256": commitment.item_commitment_set_sha256,
        "item_count": commitment.count,
    }


def _load_acquisition_live(
    *, project: Path, path: str | Path
) -> tuple[dict[str, Any], bytes]:
    try:
        return l4._load_acquisition(project=project, path=path)
    except l4.HotpotRecursiveL4Error as exc:
        raise HotpotEvaluatorCoevolutionError(
            "acquisition implementation or retained-P binding drifted"
        ) from exc


def _output_path_binding(path: str | Path) -> str:
    return stable_hash({"absolute_output_path": str(Path(path).absolute())})


def _formation_marker_path(project: Path, stage: str) -> Path:
    relative = {
        "A_form": A_FORM_CONSUMPTION_RELATIVE,
        "F_search": F_SEARCH_CONSUMPTION_RELATIVE,
    }.get(stage)
    if relative is None:
        raise HotpotEvaluatorCoevolutionError("unknown formation marker stage")
    path = project / relative
    try:
        l4._assert_git_ignored_private_path(
            project=project, path=path, require_file=None
        )
    except Exception as exc:
        raise HotpotEvaluatorCoevolutionError(
            "fixed formation marker is outside ignored artifacts"
        ) from exc
    return path


def _preflight_stage_outputs(
    *,
    project: Path,
    private_cache_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> tuple[Path, Path]:
    private = Path(os.path.abspath(os.fspath(private_cache_output_path)))
    public = Path(os.path.abspath(os.fspath(public_receipt_output_path)))
    if private == public or private.exists() or public.exists():
        raise HotpotEvaluatorCoevolutionError(
            "evaluator formation outputs must be distinct fresh files"
        )
    try:
        l4._assert_git_ignored_private_path(
            project=project, path=private, require_file=None
        )
    except Exception as exc:
        raise HotpotEvaluatorCoevolutionError(
            "private evidence output must be below ignored artifacts"
        ) from exc
    if public.is_symlink() or not public.parent.is_dir():
        raise HotpotEvaluatorCoevolutionError(
            "public receipt output parent is unavailable or symlinked"
        )
    return private, public


def _consume_formation_once(
    *,
    project: Path,
    stage: str,
    source_binding: Mapping[str, Any],
    lineage_binding: Mapping[str, Any],
    implementation: Mapping[str, Any],
    private_cache_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> tuple[dict[str, Any], bytes]:
    marker_path = _formation_marker_path(project, stage)
    body: dict[str, Any] = {
        "schema": FORMATION_CONSUMPTION_SCHEMA,
        "stage": stage,
        "source_binding": dict(source_binding),
        "lineage_binding": dict(lineage_binding),
        "implementation": dict(implementation),
        "output_bindings": {
            "private_evidence_output_sha256": _output_path_binding(
                private_cache_output_path
            ),
            "public_receipt_output_sha256": _output_path_binding(
                public_receipt_output_path
            ),
        },
        "private_block_rows_opened_before_consumption": 0,
        "retry_replay_resample_authorized": False,
        "raw_content_persisted": False,
    }
    marker = {**body, "consumption_sha256": stable_hash(body)}
    _write_json_exclusive(marker_path, marker, mode=0o600)
    return marker, marker_path.read_bytes()


def _formation_consumption_binding(
    marker: Mapping[str, Any], marker_raw: bytes
) -> dict[str, Any]:
    return {
        "marker_file_sha256": _sha256_bytes(marker_raw),
        "marker_sha256": marker["consumption_sha256"],
        "marker_written_before_private_block_open": True,
        "private_block_rows_opened_before_marker": 0,
        "retry_replay_resample_authorized": False,
        "private_path_persisted_publicly": False,
    }


def _load_and_verify_formation_marker(
    *,
    project: Path,
    stage: str,
    expected_source_binding: Mapping[str, Any],
    expected_lineage_binding: Mapping[str, Any],
    expected_implementation: Mapping[str, Any],
    private_cache_path: str | Path,
    public_receipt_path: str | Path,
) -> tuple[dict[str, Any], bytes]:
    marker_path = _formation_marker_path(project, stage)
    marker, raw = _read_json(marker_path, f"{stage} formation marker")
    body = dict(marker)
    declared = _require_sha256(
        body.pop("consumption_sha256", None), "formation marker hash"
    )
    if (
        marker.get("schema") != FORMATION_CONSUMPTION_SCHEMA
        or marker.get("stage") != stage
        or stable_hash(body) != declared
        or marker.get("source_binding") != dict(expected_source_binding)
        or marker.get("lineage_binding") != dict(expected_lineage_binding)
        or marker.get("implementation") != dict(expected_implementation)
        or marker.get("output_bindings")
        != {
            "private_evidence_output_sha256": _output_path_binding(
                private_cache_path
            ),
            "public_receipt_output_sha256": _output_path_binding(
                public_receipt_path
            ),
        }
        or marker.get("private_block_rows_opened_before_consumption") != 0
        or marker.get("retry_replay_resample_authorized") is not False
        or marker.get("raw_content_persisted") is not False
    ):
        raise HotpotEvaluatorCoevolutionError(
            f"{stage} one-shot formation marker drifted"
        )
    return marker, raw


def _load_lineage(
    *,
    project: Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
) -> tuple[
    TypedRetrievalProgram,
    TypedRetrievalProgram,
    dict[str, Any],
]:
    p_program, p_binding = l4._load_p(
        project=project,
        formation_path=p_formation_receipt_path,
        program_path=p_frozen_program_path,
    )
    q_program, q_binding = l4.load_q(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        formation_receipt_path=q_formation_receipt_path,
        frozen_program_path=q_frozen_program_path,
    )
    if p_program.program_hash == q_program.program_hash:
        raise HotpotEvaluatorCoevolutionError(
            "generation-two Q is identical to retained P"
        )
    lineage = {
        "retained_P": dict(p_binding),
        "formed_generation_two_Q": dict(q_binding),
        "evaluator_selectable_pool_not_limited_to_formed_Q": True,
        "retained_P_excluded_from_selectable_pool": True,
    }
    return p_program, q_program, lineage


def _program_by_hash(program_sha256: str) -> TypedRetrievalProgram:
    _require_sha256(program_sha256, "selected program hash")
    row = next(
        (program for program in fixed_programs() if program.program_hash == program_sha256),
        None,
    )
    if row is None:
        raise HotpotEvaluatorCoevolutionError(
            "selected program is outside the frozen candidate pool"
        )
    return row


def _evaluate_program_grid(
    *,
    p_program: TypedRetrievalProgram,
    items: Sequence[l4.RecursiveItem],
) -> tuple[tuple[ProgramEvidence, ...], dict[str, Any]]:
    """Run the frozen 84 x item grid gold-free, then build local evidence."""

    programs = fixed_programs()
    if len(items) < FOLD_COUNT:
        raise HotpotEvaluatorCoevolutionError("formation block is too small")
    p_rankings: dict[int, tuple[int, ...]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(items), thread_name_prefix="hotpot-eval-p-shared"
    ) as executor:
        futures = {
            ordinal: executor.submit(l4._ranking, p_program, item.retrieval_view())
            for ordinal, item in enumerate(items)
        }
        for ordinal, future in futures.items():
            p_rankings[ordinal] = future.result()

    attempted = completed = 0
    lock = threading.Lock()

    def run_one(
        program_ordinal: int, item_ordinal: int
    ) -> tuple[tuple[int, int], tuple[int, ...] | None, bool]:
        nonlocal attempted, completed
        with lock:
            attempted += 1
        try:
            ranking = l4._ranking(
                programs[program_ordinal], items[item_ordinal].retrieval_view()
            )
            invalid = False
        except l4.HotpotRecursiveL4Error:
            ranking = None
            invalid = True
        with lock:
            completed += 1
        return (program_ordinal, item_ordinal), ranking, invalid

    work_unit_count = len(programs) * len(items)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=work_unit_count,
        thread_name_prefix="hotpot-evaluator-grid",
    ) as executor:
        futures = [
            executor.submit(run_one, program_ordinal, item_ordinal)
            for program_ordinal in range(len(programs))
            for item_ordinal in range(len(items))
        ]
        terminal_rows = [future.result() for future in futures]
    if attempted != work_unit_count or completed != work_unit_count:
        raise HotpotEvaluatorCoevolutionError(
            "evaluator formation retrieval closure is incomplete"
        )
    terminals = {
        key: (ranking, invalid) for key, ranking, invalid in terminal_rows
    }
    if len(terminals) != work_unit_count:
        raise HotpotEvaluatorCoevolutionError(
            "evaluator formation terminal keys are not one-to-one"
        )

    evidence: list[ProgramEvidence] = []
    invalid_count = 0
    for program_ordinal, program in enumerate(programs):
        item_rows: list[ItemProgramEvidence] = []
        for item_ordinal, item in enumerate(items):
            q_ranking, invalid = terminals[(program_ordinal, item_ordinal)]
            invalid_count += int(invalid)
            if q_ranking is None:
                item_rows.append(
                    ItemProgramEvidence(
                        item_commitment_sha256=item.row_commitment_sha256,
                        q_direct_hits=0,
                        combined_hits=0,
                        support_total=len(item.support_indices),
                        combined_complete=0,
                        combined_coverage=0,
                        retained_added=0,
                        novelty_added=0,
                        retained_displaced=0,
                        invalid=True,
                        action_sha256=stable_hash(
                            {
                                "item_commitment_sha256": item.row_commitment_sha256,
                                "invalid": True,
                                "q_ranking": None,
                                "combined_ranking": None,
                            }
                        ),
                    ).validate()
                )
                continue
            p_ranking = p_rankings[item_ordinal]
            combined = l4._indices(
                deterministic_rrf(
                    (l4._doc_ids(p_ranking), l4._doc_ids(q_ranking))
                )
            )
            supports = frozenset(item.support_indices)
            q_hits = supports.intersection(q_ranking)
            p_hits = supports.intersection(p_ranking)
            combined_hits = supports.intersection(combined)
            item_rows.append(
                ItemProgramEvidence(
                    item_commitment_sha256=item.row_commitment_sha256,
                    q_direct_hits=len(q_hits),
                    combined_hits=len(combined_hits),
                    support_total=len(supports),
                    combined_complete=int(len(combined_hits) == len(supports)),
                    combined_coverage=int(bool(combined_hits)),
                    retained_added=len(combined_hits - q_hits),
                    novelty_added=len(combined_hits - p_hits),
                    retained_displaced=len(p_hits - combined_hits),
                    invalid=False,
                    action_sha256=stable_hash(
                        {
                            "item_commitment_sha256": item.row_commitment_sha256,
                            "invalid": False,
                            "q_ranking": list(q_ranking),
                            "combined_ranking": list(combined),
                        }
                    ),
                ).validate()
            )
        evidence.append(
            ProgramEvidence(
                program_sha256=program.program_hash,
                program_length=program.program_length,
                rows=tuple(item_rows),
            ).validate()
        )
    execution = {
        "candidate_program_count": len(programs),
        "item_count": len(items),
        "shared_retained_P_retrieval_call_count": len(items),
        "candidate_retrieval_work_unit_count": work_unit_count,
        "candidate_retrieval_attempt_count": attempted,
        "candidate_retrieval_terminal_count": completed,
        "configured_candidate_maximum_concurrency": work_unit_count,
        "all_candidate_terminals_joined_before_support_scoring": True,
        "invalid_terminal_count": invalid_count,
        "model_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }
    return tuple(evidence), execution


def _serialize_evidence(evidence: Sequence[ProgramEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "program_sha256": value.program_sha256,
            "program_length": value.program_length,
            "rows": [asdict(row) for row in value.rows],
        }
        for value in _validate_evidence_set(evidence)
    ]


def _deserialize_evidence(value: object) -> tuple[ProgramEvidence, ...]:
    if not isinstance(value, list):
        raise HotpotEvaluatorCoevolutionError("private evidence list is malformed")
    result: list[ProgramEvidence] = []
    try:
        for program in value:
            if not isinstance(program, Mapping) or set(program) != {
                "program_sha256",
                "program_length",
                "rows",
            }:
                raise HotpotEvaluatorCoevolutionError(
                    "private program evidence schema drifted"
                )
            raw_rows = program["rows"]
            if not isinstance(raw_rows, list):
                raise HotpotEvaluatorCoevolutionError(
                    "private item evidence is malformed"
                )
            result.append(
                ProgramEvidence(
                    program_sha256=str(program["program_sha256"]),
                    program_length=int(program["program_length"]),
                    rows=tuple(ItemProgramEvidence(**dict(row)) for row in raw_rows),
                ).validate()
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise HotpotEvaluatorCoevolutionError(
            "private evidence payload is invalid"
        ) from exc
    return _validate_evidence_set(result)


def _cache_payload(
    *,
    stage: str,
    source_binding: Mapping[str, Any],
    lineage_binding: Mapping[str, Any],
    evidence: Sequence[ProgramEvidence],
    execution: Mapping[str, Any],
    implementation: Mapping[str, Any],
    formation_consumption_binding: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": EVIDENCE_CACHE_SCHEMA,
        "stage": stage,
        "implementation": dict(implementation),
        "source_binding": dict(source_binding),
        "lineage_binding": dict(lineage_binding),
        "candidate_set_binding": candidate_set_binding(),
        "evidence_set_sha256": evidence_set_sha256(evidence),
        "evidence": _serialize_evidence(evidence),
        "execution": dict(execution),
        "formation_consumption_binding": dict(formation_consumption_binding),
        "raw_question_or_corpus_persisted": False,
    }
    return {**body, "cache_sha256": stable_hash(body)}


def _load_cache(
    path: str | Path, *, expected_stage: str, verify_live_project: Path | None = None
) -> tuple[tuple[ProgramEvidence, ...], dict[str, Any], bytes]:
    if verify_live_project is not None:
        try:
            l4._assert_git_ignored_private_path(
                project=verify_live_project,
                path=Path(path),
                require_file=True,
            )
        except Exception as exc:
            raise HotpotEvaluatorCoevolutionError(
                f"{expected_stage} evidence is not in the ignored private boundary"
            ) from exc
    payload, raw = _read_json(path, f"{expected_stage} private evidence cache")
    body = dict(payload)
    declared = _require_sha256(body.pop("cache_sha256", None), "cache hash")
    expected_keys = {
        "cache_sha256",
        "candidate_set_binding",
        "evidence",
        "evidence_set_sha256",
        "execution",
        "formation_consumption_binding",
        "implementation",
        "lineage_binding",
        "raw_question_or_corpus_persisted",
        "schema",
        "source_binding",
        "stage",
    }
    evidence = _deserialize_evidence(payload.get("evidence"))
    fixed = fixed_programs()
    expected_item_count = {
        "A_form": A_ITEM_COUNT,
        "F_search": F_SEARCH_ITEM_COUNT,
    }.get(expected_stage)
    if expected_item_count is None:
        raise HotpotEvaluatorCoevolutionError("unknown evidence-cache stage")
    source = payload.get("source_binding")
    expected_source_keys = {
        "acquisition_file_sha256",
        "acquisition_sha256",
        "block_file_sha256",
        "block_id_sha256",
        "item_commitment_set_sha256",
        "item_count",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != EVIDENCE_CACHE_SCHEMA
        or payload.get("stage") != expected_stage
        or stable_hash(body) != declared
        or payload.get("candidate_set_binding") != candidate_set_binding()
        or payload.get("evidence_set_sha256") != evidence_set_sha256(evidence)
        or tuple(row.program_sha256 for row in evidence)
        != tuple(program.program_hash for program in fixed)
        or tuple(row.program_length for row in evidence)
        != tuple(program.program_length for program in fixed)
        or any(len(row.rows) != expected_item_count for row in evidence)
        or not isinstance(source, Mapping)
        or set(source) != expected_source_keys
        or source.get("block_id_sha256") != stable_hash({"block": expected_stage})
        or source.get("item_count") != expected_item_count
        or stable_hash(
            [row.item_commitment_sha256 for row in evidence[0].rows]
        )
        != source.get("item_commitment_set_sha256")
        or payload.get("raw_question_or_corpus_persisted") is not False
        or (
            verify_live_project is not None
            and payload.get("implementation")
            != current_implementation_binding(verify_live_project)
        )
    ):
        raise HotpotEvaluatorCoevolutionError(
            f"{expected_stage} private evidence cache drifted"
        )
    for field in (
        "acquisition_file_sha256",
        "acquisition_sha256",
        "block_file_sha256",
        "block_id_sha256",
        "item_commitment_set_sha256",
    ):
        _require_sha256(source.get(field), f"{expected_stage} {field}")
    execution = payload.get("execution")
    expected_work_units = CANDIDATE_COUNT * expected_item_count
    expected_execution_keys = {
        "all_candidate_terminals_joined_before_support_scoring",
        "candidate_program_count",
        "candidate_retrieval_attempt_count",
        "candidate_retrieval_terminal_count",
        "candidate_retrieval_work_unit_count",
        "configured_candidate_maximum_concurrency",
        "external_network_calls",
        "invalid_terminal_count",
        "item_count",
        "model_calls",
        "online_evaluator_calls",
        "replays",
        "resamples",
        "retries",
        "shared_retained_P_retrieval_call_count",
    }
    if (
        not isinstance(execution, Mapping)
        or set(execution) != expected_execution_keys
        or execution.get("candidate_program_count") != CANDIDATE_COUNT
        or execution.get("item_count") != expected_item_count
        or execution.get("shared_retained_P_retrieval_call_count")
        != expected_item_count
        or execution.get("candidate_retrieval_work_unit_count")
        != expected_work_units
        or execution.get("candidate_retrieval_attempt_count") != expected_work_units
        or execution.get("candidate_retrieval_terminal_count") != expected_work_units
        or execution.get("configured_candidate_maximum_concurrency")
        != expected_work_units
        or execution.get("all_candidate_terminals_joined_before_support_scoring")
        is not True
        or execution.get("invalid_terminal_count") != 0
        or execution.get("model_calls") != 0
        or execution.get("external_network_calls") != 0
        or execution.get("online_evaluator_calls") != 0
        or execution.get("retries") != 0
        or execution.get("replays") != 0
        or execution.get("resamples") != 0
    ):
        raise HotpotEvaluatorCoevolutionError(
            f"{expected_stage} formation execution contract drifted"
        )
    return evidence, payload, raw


def _load_public_receipt(
    path: str | Path,
    *,
    expected_schema: str,
    hash_field: str,
    verify_live_project: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_json(path, "public evaluator receipt")
    body = dict(payload)
    declared = _require_sha256(body.pop(hash_field, None), hash_field)
    if (
        payload.get("schema") != expected_schema
        or stable_hash(body) != declared
        or payload.get("raw_content_persisted") is not False
        or (
            verify_live_project is not None
            and payload.get("implementation")
            != current_implementation_binding(verify_live_project)
        )
    ):
        raise HotpotEvaluatorCoevolutionError("public evaluator receipt drifted")
    _assert_public_safe(payload)
    return payload, raw


def _public_evidence_binding(
    *, cache: Mapping[str, Any], cache_raw: bytes
) -> dict[str, Any]:
    return {
        "private_evidence_file_sha256": _sha256_bytes(cache_raw),
        "private_evidence_cache_sha256": cache["cache_sha256"],
        "evidence_set_sha256": cache["evidence_set_sha256"],
        "private_path_persisted_publicly": False,
        "item_level_evidence_persisted_publicly": False,
    }


def _write_stage_outputs(
    *,
    project: Path,
    private_cache_output_path: str | Path,
    public_receipt_output_path: str | Path,
    cache: Mapping[str, Any],
    public: Mapping[str, Any],
) -> None:
    private = Path(os.path.abspath(os.fspath(private_cache_output_path)))
    public_path = Path(os.path.abspath(os.fspath(public_receipt_output_path)))
    if private == public_path or private.exists() or public_path.exists():
        raise HotpotEvaluatorCoevolutionError(
            "evaluator formation outputs must be distinct fresh files"
        )
    try:
        l4._assert_git_ignored_private_path(
            project=project, path=private, require_file=None
        )
    except Exception as exc:
        raise HotpotEvaluatorCoevolutionError(
            "private evidence output must be below ignored artifacts"
        ) from exc
    _write_json_exclusive(private, cache, mode=0o600)
    try:
        _write_json_exclusive(public_path, public, mode=0o644)
    except Exception:
        # Do not silently reuse a private cache whose public receipt failed.
        raise


def form_a_form_stage(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    private_evidence_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    """Open only A_form, form one action-distinct evaluator challenger."""

    project = Path(project_root).resolve(strict=True)
    receipt, receipt_raw = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    commitment = l4._commitment(receipt, "A_form")
    p_program, _q_program, lineage = _load_lineage(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
    )
    implementation = current_implementation_binding(project)
    source = _source_binding(receipt, receipt_raw, "A_form")
    _preflight_stage_outputs(
        project=project,
        private_cache_output_path=private_evidence_output_path,
        public_receipt_output_path=public_receipt_output_path,
    )
    marker, marker_raw = _consume_formation_once(
        project=project,
        stage="A_form",
        source_binding=source,
        lineage_binding=lineage,
        implementation=implementation,
        private_cache_output_path=private_evidence_output_path,
        public_receipt_output_path=public_receipt_output_path,
    )
    consumption_binding = _formation_consumption_binding(marker, marker_raw)
    items = l4._load_private_block(
        project=project, path=a_form_block_path, commitment=commitment
    )
    evidence, execution = _evaluate_program_grid(
        p_program=p_program, items=items
    )
    core = form_challenger_from_evidence(
        evidence, retained_p_program_sha256=p_program.program_hash
    )
    cache = _cache_payload(
        stage="A_form",
        source_binding=source,
        lineage_binding=lineage,
        evidence=evidence,
        execution=execution,
        implementation=implementation,
        formation_consumption_binding=consumption_binding,
    )
    cache_raw = (
        json.dumps(cache, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    body: dict[str, Any] = {
        "schema": A_FORM_SCHEMA,
        "stage": "A_form",
        "status": "challenger_formed_offline_on_exact_A_form",
        "implementation": implementation,
        "source_binding": source,
        "lineage_binding": lineage,
        "candidate_set_binding": candidate_set_binding(),
        "evidence_binding": _public_evidence_binding(
            cache=cache, cache_raw=cache_raw
        ),
        "core_receipt": core,
        "execution": execution,
        "prospective_ordering": consumption_binding,
        "offline_contract": {
            "model_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "measurement_blocks_accessed": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "raw_content_persisted": False,
    }
    public = {**body, "receipt_sha256": stable_hash(body)}
    _assert_public_safe(public)
    _write_stage_outputs(
        project=project,
        private_cache_output_path=private_evidence_output_path,
        public_receipt_output_path=public_receipt_output_path,
        cache=cache,
        public=public,
    )
    return public


def load_a_form_bundle(
    *,
    project: Path,
    private_evidence_path: str | Path,
    public_receipt_path: str | Path,
) -> tuple[tuple[ProgramEvidence, ...], dict[str, Any], dict[str, Any], bytes, bytes]:
    evidence, cache, cache_raw = _load_cache(
        private_evidence_path,
        expected_stage="A_form",
        verify_live_project=project,
    )
    public, public_raw = _load_public_receipt(
        public_receipt_path,
        expected_schema=A_FORM_SCHEMA,
        hash_field="receipt_sha256",
        verify_live_project=project,
    )
    expected_keys = {
        "candidate_set_binding",
        "core_receipt",
        "evidence_binding",
        "execution",
        "implementation",
        "lineage_binding",
        "offline_contract",
        "prospective_ordering",
        "raw_content_persisted",
        "receipt_sha256",
        "schema",
        "source_binding",
        "stage",
        "status",
    }
    _verify_formation_core(
        public.get("core_receipt", {}),
        evidence,
        retained_p_program_sha256=cache["lineage_binding"]["retained_P"][
            "program_hash"
        ],
    )
    marker, marker_raw = _load_and_verify_formation_marker(
        project=project,
        stage="A_form",
        expected_source_binding=cache["source_binding"],
        expected_lineage_binding=cache["lineage_binding"],
        expected_implementation=cache["implementation"],
        private_cache_path=private_evidence_path,
        public_receipt_path=public_receipt_path,
    )
    consumption_binding = _formation_consumption_binding(marker, marker_raw)
    if (
        set(public) != expected_keys
        or public.get("stage") != "A_form"
        or public.get("status") != "challenger_formed_offline_on_exact_A_form"
        or public.get("offline_contract")
        != {
            "model_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "measurement_blocks_accessed": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        }
        or public.get("source_binding") != cache.get("source_binding")
        or public.get("lineage_binding") != cache.get("lineage_binding")
        or public.get("execution") != cache.get("execution")
        or public.get("candidate_set_binding") != candidate_set_binding()
        or public.get("evidence_binding")
        != _public_evidence_binding(cache=cache, cache_raw=cache_raw)
        or cache.get("formation_consumption_binding") != consumption_binding
        or public.get("prospective_ordering") != consumption_binding
    ):
        raise HotpotEvaluatorCoevolutionError("A-form bundle cross-binding drifted")
    return evidence, cache, public, cache_raw, public_raw


def form_f_search_stage(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    f_search_block_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    private_evidence_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    """Open only F_search and freeze both future evaluator selections."""

    project = Path(project_root).resolve(strict=True)
    a_evidence, a_cache, a_public, a_cache_raw, a_public_raw = load_a_form_bundle(
        project=project,
        private_evidence_path=a_form_private_evidence_path,
        public_receipt_path=a_form_public_receipt_path,
    )
    receipt, receipt_raw = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    commitment = l4._commitment(receipt, "F_search")
    p_program, _q_program, lineage = _load_lineage(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
    )
    if a_cache.get("lineage_binding") != lineage:
        raise HotpotEvaluatorCoevolutionError("A-form lineage differs from F-search")
    implementation = current_implementation_binding(project)
    source = _source_binding(receipt, receipt_raw, "F_search")
    _preflight_stage_outputs(
        project=project,
        private_cache_output_path=private_evidence_output_path,
        public_receipt_output_path=public_receipt_output_path,
    )
    marker, marker_raw = _consume_formation_once(
        project=project,
        stage="F_search",
        source_binding=source,
        lineage_binding=lineage,
        implementation=implementation,
        private_cache_output_path=private_evidence_output_path,
        public_receipt_output_path=public_receipt_output_path,
    )
    consumption_binding = _formation_consumption_binding(marker, marker_raw)
    items = l4._load_private_block(
        project=project, path=f_search_block_path, commitment=commitment
    )
    evidence, execution = _evaluate_program_grid(
        p_program=p_program, items=items
    )
    core = freeze_search_choices_from_evidence(
        search_evidence=evidence,
        a_form_evidence=a_evidence,
        formation_receipt=a_public["core_receipt"],
        retained_p_program_sha256=p_program.program_hash,
    )
    cache = _cache_payload(
        stage="F_search",
        source_binding=source,
        lineage_binding=lineage,
        evidence=evidence,
        execution=execution,
        implementation=implementation,
        formation_consumption_binding=consumption_binding,
    )
    cache_raw = (
        json.dumps(cache, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    a_binding = {
        "private_evidence_file_sha256": _sha256_bytes(a_cache_raw),
        "private_evidence_cache_sha256": a_cache["cache_sha256"],
        "public_receipt_file_sha256": _sha256_bytes(a_public_raw),
        "public_receipt_sha256": a_public["receipt_sha256"],
        "formation_sha256": a_public["core_receipt"]["formation_sha256"],
    }
    body: dict[str, Any] = {
        "schema": F_SEARCH_SCHEMA,
        "stage": "F_search",
        "status": (
            "future_search_contrast_frozen"
            if core["measurable_contrast"]
            else "closed_before_anchor_no_distinct_future_action"
        ),
        "implementation": implementation,
        "source_binding": source,
        "lineage_binding": lineage,
        "A_form_binding": a_binding,
        "candidate_set_binding": candidate_set_binding(),
        "evidence_binding": _public_evidence_binding(
            cache=cache, cache_raw=cache_raw
        ),
        "core_receipt": core,
        "execution": execution,
        "prospective_ordering": consumption_binding,
        "offline_contract": {
            "model_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "anchor_accessed": False,
            "measurement_blocks_accessed": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "raw_content_persisted": False,
    }
    public = {**body, "receipt_sha256": stable_hash(body)}
    _assert_public_safe(public)
    _write_stage_outputs(
        project=project,
        private_cache_output_path=private_evidence_output_path,
        public_receipt_output_path=public_receipt_output_path,
        cache=cache,
        public=public,
    )
    return public


def load_f_search_bundle(
    *,
    project: Path,
    private_evidence_path: str | Path,
    public_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
) -> tuple[tuple[ProgramEvidence, ...], dict[str, Any], dict[str, Any], bytes, bytes]:
    a_evidence, a_cache, a_public, a_cache_raw, a_public_raw = load_a_form_bundle(
        project=project,
        private_evidence_path=a_form_private_evidence_path,
        public_receipt_path=a_form_public_receipt_path,
    )
    evidence, cache, cache_raw = _load_cache(
        private_evidence_path,
        expected_stage="F_search",
        verify_live_project=project,
    )
    public, public_raw = _load_public_receipt(
        public_receipt_path,
        expected_schema=F_SEARCH_SCHEMA,
        hash_field="receipt_sha256",
        verify_live_project=project,
    )
    expected_core = freeze_search_choices_from_evidence(
        search_evidence=evidence,
        a_form_evidence=a_evidence,
        formation_receipt=a_public["core_receipt"],
        retained_p_program_sha256=cache["lineage_binding"]["retained_P"][
            "program_hash"
        ],
    )
    expected_a_binding = {
        "private_evidence_file_sha256": _sha256_bytes(a_cache_raw),
        "private_evidence_cache_sha256": a_cache["cache_sha256"],
        "public_receipt_file_sha256": _sha256_bytes(a_public_raw),
        "public_receipt_sha256": a_public["receipt_sha256"],
        "formation_sha256": a_public["core_receipt"]["formation_sha256"],
    }
    marker, marker_raw = _load_and_verify_formation_marker(
        project=project,
        stage="F_search",
        expected_source_binding=cache["source_binding"],
        expected_lineage_binding=cache["lineage_binding"],
        expected_implementation=cache["implementation"],
        private_cache_path=private_evidence_path,
        public_receipt_path=public_receipt_path,
    )
    consumption_binding = _formation_consumption_binding(marker, marker_raw)
    expected_public_keys = {
        "A_form_binding",
        "candidate_set_binding",
        "core_receipt",
        "evidence_binding",
        "execution",
        "implementation",
        "lineage_binding",
        "offline_contract",
        "prospective_ordering",
        "raw_content_persisted",
        "receipt_sha256",
        "schema",
        "source_binding",
        "stage",
        "status",
    }
    if (
        set(public) != expected_public_keys
        or public.get("stage") != "F_search"
        or public.get("status")
        != (
            "future_search_contrast_frozen"
            if expected_core["measurable_contrast"]
            else "closed_before_anchor_no_distinct_future_action"
        )
        or public.get("candidate_set_binding") != candidate_set_binding()
        or public.get("offline_contract")
        != {
            "model_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "anchor_accessed": False,
            "measurement_blocks_accessed": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        }
        or public.get("core_receipt") != expected_core
        or public.get("A_form_binding") != expected_a_binding
        or public.get("source_binding") != cache.get("source_binding")
        or public.get("lineage_binding") != cache.get("lineage_binding")
        or public.get("execution") != cache.get("execution")
        or public.get("evidence_binding")
        != _public_evidence_binding(cache=cache, cache_raw=cache_raw)
        or cache.get("formation_consumption_binding") != consumption_binding
        or public.get("prospective_ordering") != consumption_binding
    ):
        raise HotpotEvaluatorCoevolutionError("F-search bundle cross-binding drifted")
    return evidence, cache, public, cache_raw, public_raw


def _new_root(path: str | Path, project: Path) -> Path:
    try:
        return l4._new_root(path, project)
    except l4.HotpotRecursiveL4Error as exc:
        raise HotpotEvaluatorCoevolutionError(
            "execution root must be fresh, ignored, and private"
        ) from exc


def _root_hash(path: str | Path, project: Path) -> str:
    return stable_hash({"absolute_execution_root": str(_new_root(path, project))})


def _bundle_binding(
    *,
    cache: Mapping[str, Any],
    cache_raw: bytes,
    public: Mapping[str, Any],
    public_raw: bytes,
    semantic_hash: str,
) -> dict[str, Any]:
    return {
        "private_evidence_file_sha256": _sha256_bytes(cache_raw),
        "private_evidence_cache_sha256": cache["cache_sha256"],
        "public_receipt_file_sha256": _sha256_bytes(public_raw),
        "public_receipt_sha256": public["receipt_sha256"],
        "semantic_sha256": semantic_hash,
        "private_path_persisted_publicly": False,
    }


def _artifact_bundles(
    *,
    project: Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_evidence_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[
    tuple[ProgramEvidence, ...],
    dict[str, Any],
    dict[str, Any],
    tuple[ProgramEvidence, ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    a_evidence, a_cache, a_public, a_cache_raw, a_public_raw = load_a_form_bundle(
        project=project,
        private_evidence_path=a_form_private_evidence_path,
        public_receipt_path=a_form_public_receipt_path,
    )
    f_evidence, f_cache, f_public, f_cache_raw, f_public_raw = load_f_search_bundle(
        project=project,
        private_evidence_path=f_search_private_evidence_path,
        public_receipt_path=f_search_public_receipt_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
    )
    a_binding = _bundle_binding(
        cache=a_cache,
        cache_raw=a_cache_raw,
        public=a_public,
        public_raw=a_public_raw,
        semantic_hash=a_public["core_receipt"]["formation_sha256"],
    )
    f_binding = _bundle_binding(
        cache=f_cache,
        cache_raw=f_cache_raw,
        public=f_public,
        public_raw=f_public_raw,
        semantic_hash=f_public["core_receipt"]["search_formation_sha256"],
    )
    return (
        a_evidence,
        a_cache,
        a_public,
        f_evidence,
        f_cache,
        f_public,
        a_binding,
        f_binding,
    )


def _anchor_execution_contract() -> dict[str, Any]:
    return {
        "physical_component_ids": list(ANCHOR_COMPONENT_IDS),
        "item_count": A_HOLD_ITEM_COUNT,
        "physical_work_unit_count": ANCHOR_WORK_UNIT_COUNT,
        "maximum_concurrency": ANCHOR_MAXIMUM_CONCURRENCY,
        "single_start_barrier_party_count": ANCHOR_WORK_UNIT_COUNT,
        "derived_arms": ["incumbent_combined", "challenger_combined"],
        "all_terminals_join_before_offline_support_scoring": True,
        "promotion_test": "one_sided_exact_paired_sign_flip_v1",
        "promotion_alpha_numerator": PROMOTION_ALPHA.numerator,
        "promotion_alpha_denominator": PROMOTION_ALPHA.denominator,
        "sole_promotion_criterion": True,
        "model_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def build_a_hold_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_evidence_path: str | Path,
    f_search_public_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze A_hold without accepting or opening an A_hold block path."""

    project = Path(project_root).resolve(strict=True)
    receipt, receipt_raw = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    source = _source_binding(receipt, receipt_raw, "A_hold")
    (
        _a_evidence,
        a_cache,
        a_public,
        _f_evidence,
        f_cache,
        f_public,
        a_binding,
        f_binding,
    ) = _artifact_bundles(
        project=project,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    p_program, _formed_q, lineage = _load_lineage(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
    )
    if (
        a_cache.get("lineage_binding") != lineage
        or f_cache.get("lineage_binding") != lineage
    ):
        raise HotpotEvaluatorCoevolutionError("evaluator lineage binding drifted")
    a_core = a_public["core_receipt"]
    f_core = f_public["core_receipt"]
    if f_core.get("measurable_contrast") is not True:
        raise HotpotEvaluatorCoevolutionError(
            "F-search actions coincide; A_hold must remain unopened"
        )
    a_incumbent = _program_by_hash(a_core["incumbent_selected_program_sha256"])
    a_challenger = _program_by_hash(a_core["challenger_selected_program_sha256"])
    if (
        a_incumbent.program_hash == a_challenger.program_hash
        or p_program.program_hash
        in {a_incumbent.program_hash, a_challenger.program_hash}
    ):
        raise HotpotEvaluatorCoevolutionError(
            "A-form anchor actions are not P plus two distinct Q programs"
        )
    body: dict[str, Any] = {
        "schema": ANCHOR_FREEZE_SCHEMA,
        "decision": "authorize_exact_A_hold_once_after_distinct_F_search",
        "implementation": current_implementation_binding(project),
        "source_binding": source,
        "lineage_binding": lineage,
        "A_form_binding": a_binding,
        "F_search_binding": f_binding,
        "anchor_actions": {
            "retained_P_program_sha256": p_program.program_hash,
            "incumbent_Q_program_sha256": a_incumbent.program_hash,
            "challenger_Q_program_sha256": a_challenger.program_hash,
            "all_three_programs_distinct": True,
        },
        "prospective_search_actions": {
            "incumbent_Q_program_sha256": f_core[
                "incumbent_selected_program_sha256"
            ],
            "incumbent_Q_action_sha256": f_core[
                "incumbent_selected_action_sha256"
            ],
            "challenger_Q_program_sha256": f_core[
                "challenger_selected_program_sha256"
            ],
            "challenger_Q_action_sha256": f_core[
                "challenger_selected_action_sha256"
            ],
            "measurable_contrast": True,
            "behavior_distinct": True,
            "frozen_before_A_hold_open": True,
        },
        "candidate_set_binding": candidate_set_binding(),
        "execution_contract": _anchor_execution_contract(),
        "authorization_hash": _require_sha256(
            authorization_hash, "A_hold authorization"
        ),
        "execution_root_sha256": _root_hash(execution_root, project),
        "ordering": {
            "A_hold_rows_read_while_freezing": 0,
            "A_hold_labels_read_while_freezing": 0,
            "M_search_rows_read_while_freezing": 0,
            "freeze_complete_before_A_hold_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_sha256": stable_hash(body)}
    _assert_public_safe(freeze)
    destination = Path(output_path).absolute()
    if destination.exists():
        raise HotpotEvaluatorCoevolutionError("A_hold freeze output already exists")
    _write_json_exclusive(destination, freeze, mode=0o644)
    return freeze


def _load_anchor_freeze(
    path: str | Path, *, project: Path
) -> tuple[dict[str, Any], str]:
    freeze, raw = _read_json(path, "A_hold pre-run freeze")
    body = dict(freeze)
    declared = _require_sha256(body.pop("freeze_sha256", None), "A_hold freeze")
    if (
        freeze.get("schema") != ANCHOR_FREEZE_SCHEMA
        or stable_hash(body) != declared
        or freeze.get("implementation") != current_implementation_binding(project)
        or freeze.get("candidate_set_binding") != candidate_set_binding()
        or freeze.get("execution_contract") != _anchor_execution_contract()
        or freeze.get("raw_content_persisted") is not False
    ):
        raise HotpotEvaluatorCoevolutionError("A_hold freeze drifted")
    _assert_public_safe(freeze)
    return freeze, _sha256_bytes(raw)


def _verify_anchor_inputs(
    *,
    project: Path,
    freeze: Mapping[str, Any],
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_evidence_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[
    dict[str, Any],
    l4.BlockCommitment,
    TypedRetrievalProgram,
    TypedRetrievalProgram,
    TypedRetrievalProgram,
    EvaluatorPolicy,
    EvaluatorPolicy,
]:
    receipt, receipt_raw = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    commitment = l4._commitment(receipt, "A_hold")
    if freeze.get("source_binding") != _source_binding(
        receipt, receipt_raw, "A_hold"
    ):
        raise HotpotEvaluatorCoevolutionError("A_hold source binding drifted")
    (
        a_evidence,
        a_cache,
        a_public,
        _f_evidence,
        f_cache,
        f_public,
        a_binding,
        f_binding,
    ) = _artifact_bundles(
        project=project,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    if (
        freeze.get("A_form_binding") != a_binding
        or freeze.get("F_search_binding") != f_binding
        or f_public["core_receipt"].get("measurable_contrast") is not True
    ):
        raise HotpotEvaluatorCoevolutionError(
            "A_hold formation/search binding drifted"
        )
    f_core = f_public["core_receipt"]
    expected_future_actions = {
        "incumbent_Q_program_sha256": f_core[
            "incumbent_selected_program_sha256"
        ],
        "incumbent_Q_action_sha256": f_core[
            "incumbent_selected_action_sha256"
        ],
        "challenger_Q_program_sha256": f_core[
            "challenger_selected_program_sha256"
        ],
        "challenger_Q_action_sha256": f_core[
            "challenger_selected_action_sha256"
        ],
        "measurable_contrast": True,
        "behavior_distinct": True,
        "frozen_before_A_hold_open": True,
    }
    if freeze.get("prospective_search_actions") != expected_future_actions:
        raise HotpotEvaluatorCoevolutionError(
            "A_hold prospective F_search actions drifted"
        )
    p_program, _formed_q, lineage = _load_lineage(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
    )
    if (
        freeze.get("lineage_binding") != lineage
        or a_cache.get("lineage_binding") != lineage
        or f_cache.get("lineage_binding") != lineage
    ):
        raise HotpotEvaluatorCoevolutionError("A_hold lineage drifted")
    core = a_public["core_receipt"]
    incumbent_policy, challenger_policy = _verify_formation_core(
        core,
        a_evidence,
        retained_p_program_sha256=p_program.program_hash,
    )
    incumbent_q = _program_by_hash(core["incumbent_selected_program_sha256"])
    challenger_q = _program_by_hash(core["challenger_selected_program_sha256"])
    expected_actions = {
        "retained_P_program_sha256": p_program.program_hash,
        "incumbent_Q_program_sha256": incumbent_q.program_hash,
        "challenger_Q_program_sha256": challenger_q.program_hash,
        "all_three_programs_distinct": True,
    }
    if freeze.get("anchor_actions") != expected_actions:
        raise HotpotEvaluatorCoevolutionError("A_hold selected actions drifted")
    return (
        receipt,
        commitment,
        p_program,
        incumbent_q,
        challenger_q,
        incumbent_policy,
        challenger_policy,
    )


def _aggregate_arm(
    *,
    arm_id: str,
    items: Sequence[l4.RecursiveItem],
    rankings: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(items) != len(rankings):
        raise HotpotEvaluatorCoevolutionError("arm ranking count drifted")
    hits = [
        len(frozenset(item.support_indices).intersection(ranking))
        for item, ranking in zip(items, rankings)
    ]
    total = sum(len(item.support_indices) for item in items)
    return {
        "arm_id": arm_id,
        "support_hit_count": sum(hits),
        "support_total": total,
        "support_recall_at_5": sum(hits) / total,
        "complete_item_count": sum(
            hit == len(item.support_indices) for hit, item in zip(hits, items)
        ),
        "covered_item_count": sum(hit > 0 for hit in hits),
    }


def _paired_arm(
    *,
    left: str,
    right: str,
    items: Sequence[l4.RecursiveItem],
    arms: Mapping[str, Sequence[Sequence[int]]],
    sole_promotion_criterion: bool = False,
) -> dict[str, Any]:
    deltas: list[int] = []
    for ordinal, item in enumerate(items):
        supports = frozenset(item.support_indices)
        left_hits = len(supports.intersection(arms[left][ordinal]))
        right_hits = len(supports.intersection(arms[right][ordinal]))
        deltas.append(left_hits - right_hits)
    paired_test = exact_paired_sign_flip(deltas)
    paired_test["sole_promotion_criterion"] = sole_promotion_criterion
    return {
        "left_arm_id": left,
        "right_arm_id": right,
        "net_support_hit_count": sum(deltas),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
        "paired_test": paired_test,
    }


def execute_a_hold_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path,
    a_hold_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_evidence_path: str | Path,
    f_search_public_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    """Execute exact A_hold once; callable only through this module's CLI."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise HotpotEvaluatorCoevolutionError(
            "formal A_hold is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_anchor_freeze(
        pre_run_freeze_path, project=project
    )
    root = _new_root(execution_root, project)
    if freeze.get("execution_root_sha256") != _root_hash(root, project):
        raise HotpotEvaluatorCoevolutionError("A_hold execution-root binding drifted")
    if root.exists():
        raise HotpotEvaluatorCoevolutionError("fresh A_hold root exists; replay forbidden")
    (
        _receipt,
        commitment,
        p_program,
        incumbent_q,
        challenger_q,
        incumbent_policy,
        challenger_policy,
    ) = _verify_anchor_inputs(
        project=project,
        freeze=freeze,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise HotpotEvaluatorCoevolutionError(
            "fresh A_hold root exists; replay forbidden"
        ) from exc
    attempted = completed = 0
    lock = threading.Lock()
    barrier = threading.Barrier(ANCHOR_WORK_UNIT_COUNT)
    stage = "authorization_consumption"
    try:
        consumption_body = {
            "schema": CONSUMPTION_SCHEMA,
            "stage": "A_hold",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_sha256": freeze["execution_root_sha256"],
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {
            **consumption_body,
            "consumption_sha256": stable_hash(consumption_body),
        }
        _write_json_exclusive(root / ANCHOR_CONSUMPTION_FILENAME, consumption)
        stage = "exact_A_hold_open_after_consumption"
        items = l4._load_private_block(
            project=project, path=a_hold_block_path, commitment=commitment
        )
        if len(items) != A_HOLD_ITEM_COUNT:
            raise HotpotEvaluatorCoevolutionError("A_hold item count drifted")
        work_units = tuple(
            (ordinal, component, item.retrieval_view())
            for ordinal, item in enumerate(items)
            for component in ANCHOR_COMPONENT_IDS
        )
        if len(work_units) != ANCHOR_WORK_UNIT_COUNT:
            raise HotpotEvaluatorCoevolutionError("A_hold work-unit grid drifted")
        stage = "maximum_width_gold_free_retrieval"

        def run_one(
            unit: tuple[int, str, l4.RetrievalItem]
        ) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            ordinal, component, item = unit
            with lock:
                attempted += 1
            try:
                barrier.wait(timeout=120)
            except threading.BrokenBarrierError as exc:
                raise HotpotEvaluatorCoevolutionError(
                    "A_hold maximum-width barrier did not close"
                ) from exc
            if component == "retained_P":
                ranking = l4._ranking(p_program, item)
            elif component == "incumbent_Q":
                ranking = l4._ranking(incumbent_q, item)
            elif component == "challenger_Q":
                ranking = l4._ranking(challenger_q, item)
            else:  # pragma: no cover
                raise HotpotEvaluatorCoevolutionError("unknown A_hold component")
            with lock:
                completed += 1
            return (ordinal, component), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=ANCHOR_MAXIMUM_CONCURRENCY,
            thread_name_prefix="hotpot-evaluator-anchor",
        ) as executor:
            futures = [executor.submit(run_one, unit) for unit in work_units]
            terminal_rows = [future.result() for future in futures]
        if attempted != ANCHOR_WORK_UNIT_COUNT or completed != ANCHOR_WORK_UNIT_COUNT:
            raise HotpotEvaluatorCoevolutionError("A_hold terminal closure incomplete")
        direct = dict(terminal_rows)
        if len(direct) != ANCHOR_WORK_UNIT_COUNT:
            raise HotpotEvaluatorCoevolutionError("A_hold terminal keys drifted")
        stage = "offline_source_support_scoring_after_join"
        arms: dict[str, list[tuple[int, ...]]] = {
            "incumbent_combined": [],
            "challenger_combined": [],
        }
        ranking_receipts: list[dict[str, Any]] = []
        for ordinal, item in enumerate(items):
            p = direct[(ordinal, "retained_P")]
            incumbent = direct[(ordinal, "incumbent_Q")]
            challenger = direct[(ordinal, "challenger_Q")]
            arms["incumbent_combined"].append(
                l4._indices(
                    deterministic_rrf(
                        (l4._doc_ids(p), l4._doc_ids(incumbent))
                    )
                )
            )
            arms["challenger_combined"].append(
                l4._indices(
                    deterministic_rrf(
                        (l4._doc_ids(p), l4._doc_ids(challenger))
                    )
                )
            )
        for (ordinal, component), ranking in sorted(direct.items()):
            ranking_receipts.append(
                {
                    "ordinal_sha256": stable_hash({"ordinal": ordinal}),
                    "component_id": component,
                    "ranking_sha256": stable_hash(
                        {"retrieved_indices": list(ranking)}
                    ),
                }
            )
        metrics = {
            arm: _aggregate_arm(arm_id=arm, items=items, rankings=rankings)
            for arm, rankings in arms.items()
        }
        paired = _paired_arm(
            left="challenger_combined",
            right="incumbent_combined",
            items=items,
            arms=arms,
            sole_promotion_criterion=True,
        )
        transition = _archive_transition(
            incumbent_policy=incumbent_policy,
            challenger_policy=challenger_policy,
            anchor_manifest_sha256=commitment.item_commitment_set_sha256,
            incumbent_hits=metrics["incumbent_combined"]["support_hit_count"],
            challenger_hits=metrics["challenger_combined"]["support_hit_count"],
            support_total=metrics["incumbent_combined"]["support_total"],
            item_count=len(items),
            promoted=paired["paired_test"]["promoted"],
        )
        private_body: dict[str, Any] = {
            "schema": f"{VERSION}_A_hold_private_evidence",
            "freeze_sha256": freeze["freeze_sha256"],
            "source_binding": freeze["source_binding"],
            "item_rows": [
                {
                    "item_commitment_sha256": item.row_commitment_sha256,
                    "support_indices": list(item.support_indices),
                    "incumbent_combined_ranking": list(
                        arms["incumbent_combined"][ordinal]
                    ),
                    "challenger_combined_ranking": list(
                        arms["challenger_combined"][ordinal]
                    ),
                }
                for ordinal, item in enumerate(items)
            ],
            "raw_question_or_corpus_persisted": False,
        }
        private_evidence = {
            **private_body,
            "evidence_sha256": stable_hash(private_body),
        }
        private_path = root / ANCHOR_PRIVATE_EVIDENCE_FILENAME
        _write_json_exclusive(private_path, private_evidence, mode=0o600)
        private_raw = private_path.read_bytes()
        report_body: dict[str, Any] = {
            "schema": ANCHOR_REPORT_SCHEMA,
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "source_binding": freeze["source_binding"],
            "A_form_binding": freeze["A_form_binding"],
            "F_search_binding": freeze["F_search_binding"],
            "anchor_actions": freeze["anchor_actions"],
            "prospective_search_actions": freeze["prospective_search_actions"],
            "arm_metrics": metrics,
            "challenger_minus_incumbent": paired,
            "evaluator_epoch_transition": transition,
            "private_evidence_binding": {
                "file_sha256": _sha256_bytes(private_raw),
                "evidence_sha256": private_evidence["evidence_sha256"],
                "private_path_persisted_publicly": False,
                "item_level_evidence_persisted_publicly": False,
            },
            "execution": {
                "physical_component_ids": list(ANCHOR_COMPONENT_IDS),
                "item_count": len(items),
                "physical_work_unit_count": ANCHOR_WORK_UNIT_COUNT,
                "retrieval_attempt_count": attempted,
                "retrieval_terminal_count": completed,
                "configured_maximum_concurrency": ANCHOR_MAXIMUM_CONCURRENCY,
                "observed_start_barrier_party_count": barrier.parties,
                "all_work_units_released_from_single_start_barrier": True,
                "all_terminals_joined_before_support_scoring": True,
                "ranking_receipt_set_sha256": stable_hash(ranking_receipts),
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
            },
            "M_search_opened": False,
            "M_search_authorized": transition["promoted"],
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_sha256": stable_hash(report_body)}
        _assert_public_safe(report)
        stage = "aggregate_report_persistence"
        path = root / ANCHOR_REPORT_FILENAME
        _write_json_exclusive(path, report)
        persisted, _raw = _read_json(path, "persisted A_hold report")
        if persisted != report:
            raise HotpotEvaluatorCoevolutionError("persisted A_hold report drifted")
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "stage": "A_hold",
            "valid": False,
            "freeze_sha256": freeze["freeze_sha256"],
            "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (
                root / ANCHOR_CONSUMPTION_FILENAME
            ).is_file(),
            "physical_work_unit_count": ANCHOR_WORK_UNIT_COUNT,
            "retrieval_attempt_count": attempted,
            "retrieval_terminal_count": completed,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        failure = {**failure_body, "failure_sha256": stable_hash(failure_body)}
        try:
            _write_json_exclusive(root / ANCHOR_FAILURE_FILENAME, failure)
        except Exception:
            pass
        raise HotpotEvaluatorCoevolutionError(
            "formal A_hold failed and cannot be replayed"
        ) from exc


def load_and_reverify_a_hold(
    *,
    project: Path,
    pre_run_freeze_path: str | Path,
    private_evidence_path: str | Path,
    report_path: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_evidence_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the transition from exact private anchor evidence."""

    freeze, freeze_file_hash = _load_anchor_freeze(
        pre_run_freeze_path, project=project
    )
    (
        _receipt,
        commitment,
        _p_program,
        _incumbent_q,
        _challenger_q,
        incumbent_policy,
        challenger_policy,
    ) = _verify_anchor_inputs(
        project=project,
        freeze=freeze,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    try:
        l4._assert_git_ignored_private_path(
            project=project,
            path=Path(private_evidence_path),
            require_file=True,
        )
    except Exception as exc:
        raise HotpotEvaluatorCoevolutionError(
            "A_hold private evidence is outside the ignored boundary"
        ) from exc
    private, private_raw = _read_json(private_evidence_path, "A_hold private evidence")
    private_body = dict(private)
    private_hash = _require_sha256(
        private_body.pop("evidence_sha256", None), "A_hold evidence hash"
    )
    rows = private.get("item_rows")
    if (
        private.get("schema") != f"{VERSION}_A_hold_private_evidence"
        or stable_hash(private_body) != private_hash
        or private.get("freeze_sha256") != freeze["freeze_sha256"]
        or private.get("source_binding") != freeze["source_binding"]
        or private.get("raw_question_or_corpus_persisted") is not False
        or not isinstance(rows, list)
        or len(rows) != A_HOLD_ITEM_COUNT
    ):
        raise HotpotEvaluatorCoevolutionError("A_hold private evidence drifted")
    commitments: list[str] = []
    incumbent_hits: list[int] = []
    challenger_hits: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "challenger_combined_ranking",
            "incumbent_combined_ranking",
            "item_commitment_sha256",
            "support_indices",
        }:
            raise HotpotEvaluatorCoevolutionError(
                "A_hold private item evidence schema drifted"
            )
        commitments.append(
            _require_sha256(row.get("item_commitment_sha256"), "anchor item")
        )
        supports = row.get("support_indices")
        incumbent = row.get("incumbent_combined_ranking")
        challenger = row.get("challenger_combined_ranking")
        if (
            not isinstance(supports, list)
            or len(supports) != 2
            or len(set(supports)) != 2
            or any(type(value) is not int or value < 0 for value in supports)
            or not isinstance(incumbent, list)
            or not isinstance(challenger, list)
            or len(incumbent) != TOP_K
            or len(challenger) != TOP_K
            or len(set(incumbent)) != TOP_K
            or len(set(challenger)) != TOP_K
            or any(type(value) is not int or value < 0 for value in incumbent + challenger)
        ):
            raise HotpotEvaluatorCoevolutionError(
                "A_hold private support/ranking evidence drifted"
            )
        support_set = frozenset(supports)
        incumbent_hits.append(len(support_set.intersection(incumbent)))
        challenger_hits.append(len(support_set.intersection(challenger)))
    if stable_hash(commitments) != commitment.item_commitment_set_sha256:
        raise HotpotEvaluatorCoevolutionError("A_hold item commitment set drifted")
    total = 2 * len(rows)
    metrics = {
        "incumbent_combined": {
            "arm_id": "incumbent_combined",
            "support_hit_count": sum(incumbent_hits),
            "support_total": total,
            "support_recall_at_5": sum(incumbent_hits) / total,
            "complete_item_count": sum(value == 2 for value in incumbent_hits),
            "covered_item_count": sum(value > 0 for value in incumbent_hits),
        },
        "challenger_combined": {
            "arm_id": "challenger_combined",
            "support_hit_count": sum(challenger_hits),
            "support_total": total,
            "support_recall_at_5": sum(challenger_hits) / total,
            "complete_item_count": sum(value == 2 for value in challenger_hits),
            "covered_item_count": sum(value > 0 for value in challenger_hits),
        },
    }
    deltas = [
        challenger - incumbent
        for incumbent, challenger in zip(incumbent_hits, challenger_hits)
    ]
    paired = {
        "left_arm_id": "challenger_combined",
        "right_arm_id": "incumbent_combined",
        "net_support_hit_count": sum(deltas),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
        "paired_test": exact_paired_sign_flip(deltas),
    }
    transition = _archive_transition(
        incumbent_policy=incumbent_policy,
        challenger_policy=challenger_policy,
        anchor_manifest_sha256=commitment.item_commitment_set_sha256,
        incumbent_hits=sum(incumbent_hits),
        challenger_hits=sum(challenger_hits),
        support_total=total,
        item_count=len(rows),
        promoted=paired["paired_test"]["promoted"],
    )
    report, report_raw = _read_json(report_path, "A_hold aggregate report")
    report_body = dict(report)
    report_hash = _require_sha256(
        report_body.pop("report_sha256", None), "A_hold report hash"
    )
    expected_private_binding = {
        "file_sha256": _sha256_bytes(private_raw),
        "evidence_sha256": private_hash,
        "private_path_persisted_publicly": False,
        "item_level_evidence_persisted_publicly": False,
    }
    expected_report_keys = {
        "A_form_binding",
        "F_search_binding",
        "M_search_authorized",
        "M_search_opened",
        "anchor_actions",
        "arm_metrics",
        "challenger_minus_incumbent",
        "evaluator_epoch_transition",
        "execution",
        "freeze_file_sha256",
        "freeze_sha256",
        "private_evidence_binding",
        "prospective_search_actions",
        "raw_content_persisted",
        "report_sha256",
        "schema",
        "source_binding",
        "valid",
    }
    execution = report.get("execution")
    expected_execution_keys = {
        "all_terminals_joined_before_support_scoring",
        "all_work_units_released_from_single_start_barrier",
        "configured_maximum_concurrency",
        "external_network_calls",
        "item_count",
        "observed_start_barrier_party_count",
        "online_evaluator_calls",
        "physical_component_ids",
        "physical_work_unit_count",
        "ranking_receipt_set_sha256",
        "replays",
        "resamples",
        "retrieval_attempt_count",
        "retrieval_terminal_count",
        "retries",
    }
    if (
        set(report) != expected_report_keys
        or report.get("schema") != ANCHOR_REPORT_SCHEMA
        or report.get("valid") is not True
        or stable_hash(report_body) != report_hash
        or report.get("freeze_sha256") != freeze["freeze_sha256"]
        or report.get("freeze_file_sha256") != freeze_file_hash
        or report.get("source_binding") != freeze["source_binding"]
        or report.get("A_form_binding") != freeze["A_form_binding"]
        or report.get("F_search_binding") != freeze["F_search_binding"]
        or report.get("anchor_actions") != freeze["anchor_actions"]
        or report.get("prospective_search_actions")
        != freeze["prospective_search_actions"]
        or report.get("private_evidence_binding") != expected_private_binding
        or report.get("arm_metrics") != metrics
        or report.get("challenger_minus_incumbent") != paired
        or report.get("evaluator_epoch_transition") != transition
        or report.get("M_search_opened") is not False
        or report.get("M_search_authorized") is not transition["promoted"]
        or not isinstance(execution, Mapping)
        or set(execution) != expected_execution_keys
        or execution.get("physical_component_ids") != list(ANCHOR_COMPONENT_IDS)
        or execution.get("item_count") != A_HOLD_ITEM_COUNT
        or execution.get("physical_work_unit_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("retrieval_attempt_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("retrieval_terminal_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("configured_maximum_concurrency")
        != ANCHOR_MAXIMUM_CONCURRENCY
        or execution.get("observed_start_barrier_party_count")
        != ANCHOR_WORK_UNIT_COUNT
        or execution.get("all_work_units_released_from_single_start_barrier")
        is not True
        or execution.get("all_terminals_joined_before_support_scoring") is not True
        or execution.get("external_network_calls") != 0
        or execution.get("online_evaluator_calls") != 0
        or execution.get("retries") != 0
        or execution.get("replays") != 0
        or execution.get("resamples") != 0
        or report.get("raw_content_persisted") is not False
    ):
        raise HotpotEvaluatorCoevolutionError(
            "A_hold report differs from exact private evidence"
        )
    _require_sha256(
        execution.get("ranking_receipt_set_sha256"),
        "A_hold ranking receipt set",
    )
    _assert_public_safe(report)
    binding = {
        "private_evidence_file_sha256": _sha256_bytes(private_raw),
        "private_evidence_sha256": private_hash,
        "public_report_file_sha256": _sha256_bytes(report_raw),
        "public_report_sha256": report_hash,
        "anchor_transition_sha256": transition["transition_sha256"],
        "challenger_promoted": transition["promoted"],
        "recomputed_from_exact_private_evidence": True,
        "private_path_persisted_publicly": False,
    }
    return report, binding


def _search_execution_contract() -> dict[str, Any]:
    return {
        "physical_arm_ids": list(SEARCH_ARM_IDS),
        "item_count": M_SEARCH_ITEM_COUNT,
        "physical_work_unit_count": SEARCH_WORK_UNIT_COUNT,
        "maximum_concurrency": SEARCH_MAXIMUM_CONCURRENCY,
        "single_start_barrier_party_count": SEARCH_WORK_UNIT_COUNT,
        "primary_comparison": "active_combined_minus_incumbent_combined",
        "secondary_comparisons": [
            "active_combined_minus_official_HippoRAG",
            "active_combined_minus_canonical_RAW",
        ],
        "all_terminals_join_before_runtime_postflight_and_offline_scoring": True,
        "model_calls_outside_official_arm": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def build_m_search_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_evidence_path: str | Path,
    f_search_public_receipt_path: str | Path,
    a_hold_pre_run_freeze_path: str | Path,
    a_hold_private_evidence_path: str | Path,
    a_hold_report_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze M_search only after an exact promoted A_hold transition."""

    project = Path(project_root).resolve(strict=True)
    anchor_report, anchor_binding = load_and_reverify_a_hold(
        project=project,
        pre_run_freeze_path=a_hold_pre_run_freeze_path,
        private_evidence_path=a_hold_private_evidence_path,
        report_path=a_hold_report_path,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    transition = anchor_report["evaluator_epoch_transition"]
    if (
        transition.get("promoted") is not True
        or transition.get("selective_invalidation_performed") is not True
        or transition.get("independent_source_record_retained") is not True
    ):
        raise HotpotEvaluatorCoevolutionError(
            "challenger was not promoted; M_search must remain unopened"
        )
    receipt, receipt_raw = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    source = _source_binding(receipt, receipt_raw, "M_search")
    (
        _a_evidence,
        a_cache,
        a_public,
        _f_evidence,
        f_cache,
        f_public,
        a_binding,
        f_binding,
    ) = _artifact_bundles(
        project=project,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    p_program, _formed_q, lineage = _load_lineage(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
    )
    if (
        a_cache.get("lineage_binding") != lineage
        or f_cache.get("lineage_binding") != lineage
    ):
        raise HotpotEvaluatorCoevolutionError("M_search lineage drifted")
    f_core = f_public["core_receipt"]
    if f_core.get("measurable_contrast") is not True:
        raise HotpotEvaluatorCoevolutionError(
            "F_search lacks a distinct prospective contrast"
        )
    incumbent_q = _program_by_hash(f_core["incumbent_selected_program_sha256"])
    active_q = _program_by_hash(f_core["challenger_selected_program_sha256"])
    if (
        incumbent_q.program_hash == active_q.program_hash
        or p_program.program_hash in {incumbent_q.program_hash, active_q.program_hash}
    ):
        raise HotpotEvaluatorCoevolutionError(
            "M_search actions are not P plus two distinct Q programs"
        )
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    paths = l4._runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = l4._prepare_runtime(project, paths)
    body: dict[str, Any] = {
        "schema": SEARCH_FREEZE_SCHEMA,
        "decision": "authorize_exact_promoted_evaluator_M_search_once",
        "implementation": current_implementation_binding(project),
        "source_binding": source,
        "lineage_binding": lineage,
        "A_form_binding": a_binding,
        "F_search_binding": f_binding,
        "A_hold_binding": anchor_binding,
        "evaluator_epoch_transition": {
            "transition_sha256": transition["transition_sha256"],
            "incumbent_epoch_id": transition["incumbent_epoch_id"],
            "active_epoch_id": transition["next_epoch_id"],
            "active_evaluator_id": transition["next_evaluator_id"],
            "promoted": True,
        },
        "search_actions": {
            "retained_P_program_sha256": p_program.program_hash,
            "incumbent_Q_program_sha256": incumbent_q.program_hash,
            "incumbent_Q_action_sha256": f_core[
                "incumbent_selected_action_sha256"
            ],
            "active_Q_program_sha256": active_q.program_hash,
            "active_Q_action_sha256": f_core[
                "challenger_selected_action_sha256"
            ],
            "all_three_programs_distinct": True,
            "incumbent_and_active_behavior_distinct": True,
            "frozen_on_F_search_before_A_hold": True,
        },
        "candidate_set_binding": candidate_set_binding(),
        "capability_binding": {
            "file_sha256": _sha256_bytes(capability_raw),
            "receipt_sha256": capability["receipt_sha256"],
            "bwrap_file_sha256": capability["bwrap_file_sha256"],
            "probe_contract_sha256": capability["probe_contract_sha256"],
            "fresh_probe_required_before_authorization": True,
        },
        "runtime_binding": l4._runtime_binding(prepared, paths),
        "execution_contract": _search_execution_contract(),
        "authorization_hash": _require_sha256(
            authorization_hash, "M_search authorization"
        ),
        "execution_root_sha256": _root_hash(execution_root, project),
        "ordering": {
            "M_search_rows_read_while_freezing": 0,
            "M_search_labels_read_while_freezing": 0,
            "A_hold_transition_reverified_before_freeze": True,
            "freeze_complete_before_M_search_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_sha256": stable_hash(body)}
    _assert_public_safe(freeze)
    destination = Path(output_path).absolute()
    if destination.exists():
        raise HotpotEvaluatorCoevolutionError("M_search freeze output exists")
    _write_json_exclusive(destination, freeze, mode=0o644)
    return freeze


def _load_search_freeze(
    path: str | Path, *, project: Path
) -> tuple[dict[str, Any], str]:
    freeze, raw = _read_json(path, "M_search pre-run freeze")
    body = dict(freeze)
    declared = _require_sha256(body.pop("freeze_sha256", None), "M_search freeze")
    if (
        freeze.get("schema") != SEARCH_FREEZE_SCHEMA
        or stable_hash(body) != declared
        or freeze.get("implementation") != current_implementation_binding(project)
        or freeze.get("candidate_set_binding") != candidate_set_binding()
        or freeze.get("execution_contract") != _search_execution_contract()
        or freeze.get("raw_content_persisted") is not False
    ):
        raise HotpotEvaluatorCoevolutionError("M_search freeze drifted")
    _assert_public_safe(freeze)
    return freeze, _sha256_bytes(raw)


def execute_m_search_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path,
    m_search_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_evidence_path: str | Path,
    f_search_public_receipt_path: str | Path,
    a_hold_pre_run_freeze_path: str | Path,
    a_hold_private_evidence_path: str | Path,
    a_hold_report_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    """Execute exact M_search once after a promoted evaluator transition."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise HotpotEvaluatorCoevolutionError(
            "formal M_search is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_search_freeze(
        pre_run_freeze_path, project=project
    )
    root = _new_root(execution_root, project)
    if freeze.get("execution_root_sha256") != _root_hash(root, project):
        raise HotpotEvaluatorCoevolutionError("M_search root binding drifted")
    if root.exists():
        raise HotpotEvaluatorCoevolutionError("fresh M_search root exists; replay forbidden")
    anchor_report, anchor_binding = load_and_reverify_a_hold(
        project=project,
        pre_run_freeze_path=a_hold_pre_run_freeze_path,
        private_evidence_path=a_hold_private_evidence_path,
        report_path=a_hold_report_path,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    if (
        freeze.get("A_hold_binding") != anchor_binding
        or anchor_report["evaluator_epoch_transition"].get("promoted") is not True
    ):
        raise HotpotEvaluatorCoevolutionError("promoted anchor binding drifted")
    receipt, receipt_raw = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    commitment = l4._commitment(receipt, "M_search")
    if freeze.get("source_binding") != _source_binding(
        receipt, receipt_raw, "M_search"
    ):
        raise HotpotEvaluatorCoevolutionError("M_search source binding drifted")
    (
        _a_evidence,
        a_cache,
        _a_public,
        _f_evidence,
        f_cache,
        f_public,
        a_binding,
        f_binding,
    ) = _artifact_bundles(
        project=project,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_evidence_path=f_search_private_evidence_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    if (
        freeze.get("A_form_binding") != a_binding
        or freeze.get("F_search_binding") != f_binding
    ):
        raise HotpotEvaluatorCoevolutionError("M_search formation binding drifted")
    p_program, _formed_q, lineage = _load_lineage(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        q_formation_receipt_path=q_formation_receipt_path,
        q_frozen_program_path=q_frozen_program_path,
    )
    if (
        freeze.get("lineage_binding") != lineage
        or a_cache.get("lineage_binding") != lineage
        or f_cache.get("lineage_binding") != lineage
    ):
        raise HotpotEvaluatorCoevolutionError("M_search lineage binding drifted")
    f_core = f_public["core_receipt"]
    incumbent_q = _program_by_hash(f_core["incumbent_selected_program_sha256"])
    active_q = _program_by_hash(f_core["challenger_selected_program_sha256"])
    expected_actions = {
        "retained_P_program_sha256": p_program.program_hash,
        "incumbent_Q_program_sha256": incumbent_q.program_hash,
        "incumbent_Q_action_sha256": f_core[
            "incumbent_selected_action_sha256"
        ],
        "active_Q_program_sha256": active_q.program_hash,
        "active_Q_action_sha256": f_core[
            "challenger_selected_action_sha256"
        ],
        "all_three_programs_distinct": True,
        "incumbent_and_active_behavior_distinct": True,
        "frozen_on_F_search_before_A_hold": True,
    }
    if freeze.get("search_actions") != expected_actions:
        raise HotpotEvaluatorCoevolutionError("M_search selected actions drifted")
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    expected_capability = {
        "file_sha256": _sha256_bytes(capability_raw),
        "receipt_sha256": capability["receipt_sha256"],
        "bwrap_file_sha256": capability["bwrap_file_sha256"],
        "probe_contract_sha256": capability["probe_contract_sha256"],
        "fresh_probe_required_before_authorization": True,
    }
    if freeze.get("capability_binding") != expected_capability:
        raise HotpotEvaluatorCoevolutionError("M_search capability binding drifted")
    live_probe = _probe_bubblewrap()
    if (
        live_probe.get("bwrap_file_sha256") != BWRAP_SHA256
        or live_probe.get("probe_contract_sha256")
        != capability["probe_contract_sha256"]
        or live_probe.get("probe_returncode") != 0
    ):
        raise HotpotEvaluatorCoevolutionError("fresh bwrap preflight drifted")
    paths = l4._runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared: PreparedFormalRuntimeV2 = l4._prepare_runtime(project, paths)
    safe_runtime = prepared.safe_binding
    if freeze.get("runtime_binding") != l4._runtime_binding(prepared, paths):
        raise HotpotEvaluatorCoevolutionError("M_search runtime binding drifted")
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise HotpotEvaluatorCoevolutionError(
            "fresh M_search root exists; replay forbidden"
        ) from exc
    attempted = completed = 0
    lock = threading.Lock()
    barrier = threading.Barrier(SEARCH_WORK_UNIT_COUNT)
    stage = "authorization_consumption"
    try:
        consumption_body = {
            "schema": CONSUMPTION_SCHEMA,
            "stage": "M_search",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_sha256": freeze["execution_root_sha256"],
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {
            **consumption_body,
            "consumption_sha256": stable_hash(consumption_body),
        }
        _write_json_exclusive(root / SEARCH_CONSUMPTION_FILENAME, consumption)
        stage = "exact_M_search_open_after_consumption"
        items = l4._load_private_block(
            project=project, path=m_search_block_path, commitment=commitment
        )
        if len(items) != M_SEARCH_ITEM_COUNT:
            raise HotpotEvaluatorCoevolutionError("M_search item count drifted")
        work_units = tuple(
            (ordinal, arm, item.retrieval_view())
            for ordinal, item in enumerate(items)
            for arm in SEARCH_ARM_IDS
        )
        if len(work_units) != SEARCH_WORK_UNIT_COUNT:
            raise HotpotEvaluatorCoevolutionError("M_search work-unit grid drifted")
        stage = "maximum_width_four_arm_retrieval"

        def run_one(
            unit: tuple[int, str, l4.RetrievalItem]
        ) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            ordinal, arm, item = unit
            with lock:
                attempted += 1
            try:
                barrier.wait(timeout=120)
            except threading.BrokenBarrierError as exc:
                raise HotpotEvaluatorCoevolutionError(
                    "M_search maximum-width barrier did not close"
                ) from exc
            raw = tuple(paragraph.idx for paragraph in item.corpus[:TOP_K])
            if arm == "canonical_RAW":
                ranking = raw
            elif arm == "incumbent_combined":
                p = l4._ranking(p_program, item)
                q = l4._ranking(incumbent_q, item)
                ranking = l4._indices(
                    deterministic_rrf(
                        (l4._doc_ids(p), l4._doc_ids(q))
                    )
                )
            elif arm == "active_combined":
                p = l4._ranking(p_program, item)
                q = l4._ranking(active_q, item)
                ranking = l4._indices(
                    deterministic_rrf(
                        (l4._doc_ids(p), l4._doc_ids(q))
                    )
                )
            elif arm == "official_HippoRAG":
                ranking = l4._official(
                    prepared, item, root / f"official_item_{ordinal:02d}"
                )
            else:  # pragma: no cover
                raise HotpotEvaluatorCoevolutionError("unknown M_search arm")
            ranking = l4._validate_direct_ranking(ranking, item)
            with lock:
                completed += 1
            return (ordinal, arm), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=SEARCH_MAXIMUM_CONCURRENCY,
            thread_name_prefix="hotpot-evaluator-search",
        ) as executor:
            futures = [executor.submit(run_one, unit) for unit in work_units]
            terminal_rows = [future.result() for future in futures]
        if attempted != SEARCH_WORK_UNIT_COUNT or completed != SEARCH_WORK_UNIT_COUNT:
            raise HotpotEvaluatorCoevolutionError("M_search terminal closure incomplete")
        direct = dict(terminal_rows)
        if len(direct) != SEARCH_WORK_UNIT_COUNT:
            raise HotpotEvaluatorCoevolutionError("M_search terminal keys drifted")
        stage = "fresh_runtime_postflight_before_scoring"
        postflight = prepared.fresh_reverify()
        if postflight != safe_runtime:
            raise HotpotEvaluatorCoevolutionError("M_search runtime postflight drifted")
        stage = "offline_source_support_scoring_after_join"
        arms = {
            arm: [direct[(ordinal, arm)] for ordinal in range(len(items))]
            for arm in SEARCH_ARM_IDS
        }
        metrics = {
            arm: _aggregate_arm(arm_id=arm, items=items, rankings=rankings)
            for arm, rankings in arms.items()
        }
        primary = _paired_arm(
            left="active_combined",
            right="incumbent_combined",
            items=items,
            arms=arms,
        )
        versus_official = _paired_arm(
            left="active_combined",
            right="official_HippoRAG",
            items=items,
            arms=arms,
        )
        versus_raw = _paired_arm(
            left="active_combined",
            right="canonical_RAW",
            items=items,
            arms=arms,
        )
        ranking_receipts = [
            {
                "ordinal_sha256": stable_hash({"ordinal": ordinal}),
                "arm_id": arm,
                "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)}),
            }
            for (ordinal, arm), ranking in sorted(direct.items())
        ]
        report_body: dict[str, Any] = {
            "schema": SEARCH_REPORT_SCHEMA,
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "source_binding": freeze["source_binding"],
            "A_hold_binding": freeze["A_hold_binding"],
            "evaluator_epoch_transition": freeze["evaluator_epoch_transition"],
            "search_actions": freeze["search_actions"],
            "arm_metrics": metrics,
            "primary_active_minus_incumbent": primary,
            "secondary_active_minus_official_HippoRAG": versus_official,
            "secondary_active_minus_RAW": versus_raw,
            "disposition": {
                "evaluator_transition_had_positive_search_utility": (
                    primary["net_support_hit_count"] > 0
                ),
                "positive_net_on_fixed_cohort_active_minus_official_HippoRAG": (
                    versus_official["net_support_hit_count"] > 0
                ),
                "positive_net_on_fixed_cohort_active_minus_RAW": (
                    versus_raw["net_support_hit_count"] > 0
                ),
                "search_measurement_used_for_evaluator_promotion": False,
                "followup_gate_authorized": False,
                "statistical_superiority_claim": False,
                "family_out_claim": False,
                "compute_budget_equivalence_claim": False,
            },
            "execution": {
                "physical_arm_ids": list(SEARCH_ARM_IDS),
                "item_count": len(items),
                "physical_work_unit_count": SEARCH_WORK_UNIT_COUNT,
                "retrieval_attempt_count": attempted,
                "retrieval_terminal_count": completed,
                "configured_maximum_concurrency": SEARCH_MAXIMUM_CONCURRENCY,
                "observed_start_barrier_party_count": barrier.parties,
                "all_work_units_released_from_single_start_barrier": True,
                "all_terminals_joined_before_postflight_and_support_scoring": True,
                "ranking_receipt_set_sha256": stable_hash(ranking_receipts),
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
            },
            "runtime": {
                "capability_receipt_sha256": capability["receipt_sha256"],
                "fresh_bwrap_preflight_before_authorization": True,
                "official_arm_terminal_count": M_SEARCH_ITEM_COUNT,
                "official_arm_uses_frozen_local_LLM_OpenIE": True,
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight["binding_sha256"],
            },
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_sha256": stable_hash(report_body)}
        _assert_public_safe(report)
        stage = "aggregate_report_persistence"
        path = root / SEARCH_REPORT_FILENAME
        _write_json_exclusive(path, report)
        persisted, _raw = _read_json(path, "persisted M_search report")
        if persisted != report:
            raise HotpotEvaluatorCoevolutionError("persisted M_search report drifted")
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "stage": "M_search",
            "valid": False,
            "freeze_sha256": freeze["freeze_sha256"],
            "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (
                root / SEARCH_CONSUMPTION_FILENAME
            ).is_file(),
            "physical_work_unit_count": SEARCH_WORK_UNIT_COUNT,
            "retrieval_attempt_count": attempted,
            "retrieval_terminal_count": completed,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        failure = {**failure_body, "failure_sha256": stable_hash(failure_body)}
        try:
            _write_json_exclusive(root / SEARCH_FAILURE_FILENAME, failure)
        except Exception:
            pass
        raise HotpotEvaluatorCoevolutionError(
            "formal M_search failed and cannot be replayed"
        ) from exc


def formal_signatures_have_no_injection_surface() -> bool:
    """Audit the formal public surfaces for result/callback injection."""

    import inspect

    forbidden = {
        "candidate_programs",
        "evidence",
        "operator",
        "operator_factory",
        "result",
        "results",
        "retriever",
        "runner",
        "callback",
    }
    functions = (
        form_a_form_stage,
        form_f_search_stage,
        build_a_hold_pre_run_freeze,
        execute_a_hold_formal,
        build_m_search_pre_run_freeze,
        execute_m_search_formal,
    )
    return all(
        forbidden.isdisjoint(inspect.signature(function).parameters)
        for function in functions
    )


def _add_project_acquisition_lineage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--p-formation-receipt", type=Path, required=True)
    parser.add_argument("--p-frozen-program", type=Path, required=True)
    parser.add_argument("--q-formation-receipt", type=Path, required=True)
    parser.add_argument("--q-frozen-program", type=Path, required=True)


def _add_a_f_artifacts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a-form-private-evidence", type=Path, required=True)
    parser.add_argument("--a-form-public-receipt", type=Path, required=True)
    parser.add_argument("--f-search-private-evidence", type=Path, required=True)
    parser.add_argument("--f-search-public-receipt", type=Path, required=True)


def _add_anchor_artifacts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a-hold-pre-run-freeze", type=Path, required=True)
    parser.add_argument("--a-hold-private-evidence", type=Path, required=True)
    parser.add_argument("--a-hold-report", type=Path, required=True)


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capability-receipt", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--local-llm-model", type=Path, required=True)
    parser.add_argument("--local-embedding-model", type=Path, required=True)
    parser.add_argument("--base-binding-receipt", type=Path, required=True)
    parser.add_argument("--attestation-receipt", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    form_a = commands.add_parser("form-a")
    form_search = commands.add_parser("form-search")
    freeze_a = commands.add_parser("freeze-a-hold")
    execute_a = commands.add_parser("execute-a-hold")
    freeze_m = commands.add_parser("freeze-m-search")
    execute_m = commands.add_parser("execute-m-search")
    for command in (form_a, form_search, freeze_a, execute_a, freeze_m, execute_m):
        _add_project_acquisition_lineage(command)
    form_a.add_argument("--a-form-block", type=Path, required=True)
    form_a.add_argument("--private-evidence-output", type=Path, required=True)
    form_a.add_argument("--public-receipt-output", type=Path, required=True)
    form_search.add_argument("--f-search-block", type=Path, required=True)
    form_search.add_argument("--a-form-private-evidence", type=Path, required=True)
    form_search.add_argument("--a-form-public-receipt", type=Path, required=True)
    form_search.add_argument("--private-evidence-output", type=Path, required=True)
    form_search.add_argument("--public-receipt-output", type=Path, required=True)
    for command in (freeze_a, execute_a, freeze_m, execute_m):
        _add_a_f_artifacts(command)
        command.add_argument("--execution-root", type=Path, required=True)
    freeze_a.add_argument("--authorization-hash", required=True)
    freeze_a.add_argument("--output", type=Path, required=True)
    execute_a.add_argument("--pre-run-freeze", type=Path, required=True)
    execute_a.add_argument("--a-hold-block", type=Path, required=True)
    for command in (freeze_m, execute_m):
        _add_anchor_artifacts(command)
        _add_runtime(command)
    freeze_m.add_argument("--authorization-hash", required=True)
    freeze_m.add_argument("--output", type=Path, required=True)
    execute_m.add_argument("--pre-run-freeze", type=Path, required=True)
    execute_m.add_argument("--m-search-block", type=Path, required=True)
    args = parser.parse_args(argv)
    lineage = {
        "project_root": args.project_root,
        "acquisition_receipt_path": args.acquisition_receipt,
        "p_formation_receipt_path": args.p_formation_receipt,
        "p_frozen_program_path": args.p_frozen_program,
        "q_formation_receipt_path": args.q_formation_receipt,
        "q_frozen_program_path": args.q_frozen_program,
    }
    if args.command == "form-a":
        form_a_form_stage(
            **lineage,
            a_form_block_path=args.a_form_block,
            private_evidence_output_path=args.private_evidence_output,
            public_receipt_output_path=args.public_receipt_output,
        )
        return 0
    if args.command == "form-search":
        form_f_search_stage(
            **lineage,
            f_search_block_path=args.f_search_block,
            a_form_private_evidence_path=args.a_form_private_evidence,
            a_form_public_receipt_path=args.a_form_public_receipt,
            private_evidence_output_path=args.private_evidence_output,
            public_receipt_output_path=args.public_receipt_output,
        )
        return 0
    formations = {
        "a_form_private_evidence_path": args.a_form_private_evidence,
        "a_form_public_receipt_path": args.a_form_public_receipt,
        "f_search_private_evidence_path": args.f_search_private_evidence,
        "f_search_public_receipt_path": args.f_search_public_receipt,
    }
    if args.command == "freeze-a-hold":
        build_a_hold_pre_run_freeze(
            **lineage,
            **formations,
            execution_root=args.execution_root,
            authorization_hash=args.authorization_hash,
            output_path=args.output,
        )
        return 0
    if args.command == "execute-a-hold":
        global _CLEAN_MODULE_CLI_ACTIVE
        _CLEAN_MODULE_CLI_ACTIVE = True
        try:
            execute_a_hold_formal(
                **lineage,
                **formations,
                pre_run_freeze_path=args.pre_run_freeze,
                a_hold_block_path=args.a_hold_block,
                execution_root=args.execution_root,
            )
        finally:
            _CLEAN_MODULE_CLI_ACTIVE = False
        return 0
    anchor = {
        "a_hold_pre_run_freeze_path": args.a_hold_pre_run_freeze,
        "a_hold_private_evidence_path": args.a_hold_private_evidence,
        "a_hold_report_path": args.a_hold_report,
    }
    runtime = {
        "capability_receipt_path": args.capability_receipt,
        "runtime_python": args.runtime_python,
        "local_llm_model": args.local_llm_model,
        "local_embedding_model": args.local_embedding_model,
        "base_binding_receipt_path": args.base_binding_receipt,
        "attestation_receipt_path": args.attestation_receipt,
    }
    if args.command == "freeze-m-search":
        build_m_search_pre_run_freeze(
            **lineage,
            **formations,
            **anchor,
            **runtime,
            execution_root=args.execution_root,
            authorization_hash=args.authorization_hash,
            output_path=args.output,
        )
        return 0
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_m_search_formal(
            **lineage,
            **formations,
            **anchor,
            **runtime,
            pre_run_freeze_path=args.pre_run_freeze,
            m_search_block_path=args.m_search_block,
            execution_root=args.execution_root,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
