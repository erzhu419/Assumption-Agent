"""Prospective offline Hotpot evaluator co-evolution with equal-compute portfolios.

This module defines one finite evaluator transition in which *both* policies
select an unordered pair of typed lexical retrievers.  Every compared derived
arm is exactly ``RRF(P, Q1, Q2)[:5]``.  The incumbent maximizes distribution-
robust direct-Q retrieval; the challenger maximizes the worst fixed
environment/fold marginal contribution beyond retained P.  Consequently a
challenger cannot obtain extra retrieval calls merely by changing evaluator.

The formal formation stages use two independently acquired 24-item
environments and four acquisition-order folds per environment.  The full
``P + 84 Q`` grid is completed before source supports are scored.  A_hold uses
six arm-qualified physical calls per item (288 terminals for 48 items), and a
promoted evaluator alone can authorize the eight-call M_search grid (192
terminals for 24 items, including RAW and frozen local official HippoRAG).

No formal API accepts injected evidence, programs, retrievers, results, or a
measurement-block path while constructing a freeze.  Private formation caches
may retain integer rankings and source-support indices; public artifacts are
aggregate/hash-only.  This file intentionally does not read any real study
data at import time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence

from ..archive import EvaluatorEpoch, EvaluatorSpec, PolicyArchive
from ..models import stable_hash
from . import hotpot_evaluator_coevolution_v2 as v2
from . import hotpot_recursive_l4_v1 as l4
from .hotpot_family_out_runner_v1 import (
    _probe_bubblewrap,
    verify_capability_receipt,
)
from .l4_retention_protocol_v1 import deterministic_rrf
from .musique_formal_runtime_binding_v2 import PreparedFormalRuntimeV2
from .musique_typed_retriever_formation_v1 import (
    TypedRetrievalProgram,
    enumerate_programs,
)


VERSION = "hotpot_evaluator_portfolio_coevolution_v1"
IMPLEMENTATION_SCHEMA = f"{VERSION}_implementation"
PRIVATE_CACHE_SCHEMA = f"{VERSION}_private_formation_cache"
FORMATION_RECEIPT_SCHEMA = f"{VERSION}_formation_receipt"
ANCHOR_FREEZE_SCHEMA = f"{VERSION}_A_hold_pre_run_freeze"
ANCHOR_REPORT_SCHEMA = f"{VERSION}_A_hold_aggregate_report"
SEARCH_FREEZE_SCHEMA = f"{VERSION}_M_search_pre_run_freeze"
SEARCH_REPORT_SCHEMA = f"{VERSION}_M_search_aggregate_report"
CONSUMPTION_SCHEMA = f"{VERSION}_authorization_consumption"
FAILURE_SCHEMA = f"{VERSION}_failure"

TOP_K = 5
FOLD_COUNT = 4
FORMATION_ENV_COUNT = 2
FORMATION_ENV_ITEM_COUNT = 24
FORMATION_ITEM_COUNT = FORMATION_ENV_COUNT * FORMATION_ENV_ITEM_COUNT
A_HOLD_ITEM_COUNT = 48
M_SEARCH_ITEM_COUNT = 24
CANDIDATE_COUNT = 84
PROMOTION_ALPHA = Fraction(1, 10)

A_FORM_ENVIRONMENTS = ("A_form_0", "A_form_1")
F_SEARCH_ENVIRONMENTS = ("F_search_0", "F_search_1")
FORMATION_BLOCKS = A_FORM_ENVIRONMENTS + F_SEARCH_ENVIRONMENTS
ANCHOR_BLOCK = "A_hold"
SEARCH_BLOCK = "M_search"

CAPABILITY_FAMILIES = tuple(
    (seed, expansion)
    for seed in ("bm25", "tfidf")
    for expansion in ("none", "token_one_hop", "entity_token_one_hop")
)

INCUMBENT_POLICY_ID = "q_direct_two_environment_maximin_pair_v1"
CHALLENGER_POLICY_ID = "marginal_eight_cell_maximin_pair_v1"

ANCHOR_COMPONENT_IDS = (
    "incumbent_P",
    "incumbent_Q1",
    "incumbent_Q2",
    "challenger_P",
    "challenger_Q1",
    "challenger_Q2",
)
SEARCH_COMPONENT_IDS = (
    "canonical_RAW",
    "incumbent_P",
    "incumbent_Q1",
    "incumbent_Q2",
    "active_P",
    "active_Q1",
    "active_Q2",
    "official_HippoRAG",
)
ANCHOR_WORK_UNIT_COUNT = len(ANCHOR_COMPONENT_IDS) * A_HOLD_ITEM_COUNT
SEARCH_WORK_UNIT_COUNT = len(SEARCH_COMPONENT_IDS) * M_SEARCH_ITEM_COUNT
ANCHOR_MAXIMUM_CONCURRENCY = ANCHOR_WORK_UNIT_COUNT
SEARCH_MAXIMUM_CONCURRENCY = SEARCH_WORK_UNIT_COUNT

A_FORM_CONSUMPTION_RELATIVE = (
    "artifacts/hotpot_evaluator_portfolio_v1/"
    "a_form.authorization.consumed.json"
)
F_SEARCH_CONSUMPTION_RELATIVE = (
    "artifacts/hotpot_evaluator_portfolio_v1/"
    "f_search.authorization.consumed.json"
)

OLD_FINAL_DISPOSITION_FILE_SHA256 = (
    "631c80917688fd38762579b7bf9f65546c70d213a46e9edd95a05b56610a2949"
)
OLD_FINAL_DISPOSITION_SHA256 = (
    "487831a0ec75d796e7c1a28e22f498fb7b65151c5b164b577bcbb6b960941aef"
)

# The acquisition implementation is added prospectively in the same closure.
# Import is deliberately lazy so the pure selector and its tests remain usable
# while that independently audited module is being constructed.
ACQUISITION_MODULE_RELATIVE = (
    "assumption_agent/benchmarks/hotpot_evaluator_robust_acquisition_v1.py"
)

IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/archive.py",
    "assumption_agent/models.py",
    "assumption_agent/benchmarks/hotpot_recursive_l4_v1.py",
    "assumption_agent/benchmarks/hotpot_evaluator_coevolution_v2.py",
    ACQUISITION_MODULE_RELATIVE,
    "assumption_agent/benchmarks/hotpot_evaluator_portfolio_coevolution_v1.py",
    "assumption_agent/benchmarks/l4_retention_protocol_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class HotpotEvaluatorPortfolioError(RuntimeError):
    """The portfolio grammar, custody, or formal execution drifted."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HotpotEvaluatorPortfolioError("required file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HotpotEvaluatorPortfolioError(f"{field} must be lowercase sha256")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: str | Path, field: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise HotpotEvaluatorPortfolioError(f"{field} path contains symlink")
    if not candidate.is_file():
        raise HotpotEvaluatorPortfolioError(f"{field} is unavailable")
    raw = candidate.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotEvaluatorPortfolioError(f"{field} is invalid") from exc
    if not isinstance(value, dict):
        raise HotpotEvaluatorPortfolioError(f"{field} must be one object")
    return value, raw


def _write_json_exclusive(
    path: Path, value: Mapping[str, Any], *, mode: int = 0o600
) -> None:
    raw = json.dumps(
        value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False
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


def _assert_public_safe(value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answer"',
        '"corpus"',
        '"item_id"',
        '"paragraph_text"',
        '"private_block_path"',
        '"private_evidence_path"',
        '"question"',
        '"q_rankings"',
        '"p_rankings"',
        '"support_indices"',
        '"text"',
        "/home/",
        "/tmp/",
    )
    if any(token in raw for token in forbidden):
        raise HotpotEvaluatorPortfolioError(
            "public artifact contains private content or path"
        )


def current_implementation_binding(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotEvaluatorPortfolioError(
                f"implementation file missing or symlinked: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def fixed_programs() -> tuple[TypedRetrievalProgram, ...]:
    programs = tuple(enumerate_programs())
    if (
        len(programs) != CANDIDATE_COUNT
        or len({program.program_hash for program in programs}) != CANDIDATE_COUNT
        or any(program.type_issues() for program in programs)
    ):
        raise HotpotEvaluatorPortfolioError("fixed typed program pool drifted")
    return programs


def capability_family(program: TypedRetrievalProgram) -> tuple[str, str]:
    family = (program.seed_algorithm, program.expansion_mode)
    if family not in CAPABILITY_FAMILIES:
        raise HotpotEvaluatorPortfolioError("typed capability family drifted")
    return family


def candidate_set_binding() -> dict[str, Any]:
    programs = fixed_programs()
    family_counts = Counter(capability_family(program) for program in programs)
    return {
        "candidate_count": len(programs),
        "program_order_sha256": stable_hash(
            [program.program_hash for program in programs]
        ),
        "program_payload_set_sha256": stable_hash(
            [program.to_dict() for program in programs]
        ),
        "capability_family_count": len(CAPABILITY_FAMILIES),
        "capability_family_membership_sha256": stable_hash(
            [
                {
                    "seed_algorithm": seed,
                    "expansion_mode": expansion,
                    "candidate_count": family_counts[(seed, expansion)],
                }
                for seed, expansion in CAPABILITY_FAMILIES
            ]
        ),
        "shortlist_limit_per_policy_per_family": 1,
        "portfolio_width": 2,
        "derived_arm_width_including_retained_P": 3,
        "unordered_portfolios": True,
        "retained_P_excluded_from_portfolios": True,
        "retained_P_behavior_aliases_excluded_from_portfolios": True,
        "all_candidates_type_valid": True,
        "top_k": TOP_K,
    }


def _validate_ranking(value: Sequence[int], field: str) -> tuple[int, ...]:
    result = tuple(value)
    if (
        len(result) != TOP_K
        or any(type(index) is not int or index < 0 for index in result)
        or len(set(result)) != len(result)
    ):
        raise HotpotEvaluatorPortfolioError(f"{field} is malformed")
    return result


@dataclass(frozen=True)
class GridItemEvidence:
    item_commitment_sha256: str
    p_ranking: tuple[int, ...]
    support_indices: tuple[int, ...]

    def validate(self) -> "GridItemEvidence":
        _require_sha256(self.item_commitment_sha256, "item commitment")
        _validate_ranking(self.p_ranking, "retained-P ranking")
        if (
            not self.support_indices
            or any(type(index) is not int or index < 0 for index in self.support_indices)
            or len(set(self.support_indices)) != len(self.support_indices)
        ):
            raise HotpotEvaluatorPortfolioError("source supports are malformed")
        return self


@dataclass(frozen=True)
class ProgramGridEvidence:
    program_sha256: str
    program_length: int
    seed_algorithm: str
    expansion_mode: str
    q_rankings: tuple[tuple[tuple[int, ...] | None, ...], ...]

    @property
    def family(self) -> tuple[str, str]:
        return (self.seed_algorithm, self.expansion_mode)

    def validate(self, environment_count: int, item_count: int) -> "ProgramGridEvidence":
        _require_sha256(self.program_sha256, "program hash")
        if (
            type(self.program_length) is not int
            or self.program_length <= 0
            or self.family not in CAPABILITY_FAMILIES
            or len(self.q_rankings) != environment_count
            or any(len(rows) != item_count for rows in self.q_rankings)
        ):
            raise HotpotEvaluatorPortfolioError("program grid is malformed")
        for environment in self.q_rankings:
            for ranking in environment:
                if ranking is not None:
                    _validate_ranking(ranking, "Q ranking")
        return self


@dataclass(frozen=True)
class FormationGridEvidence:
    environment_ids: tuple[str, ...]
    items: tuple[tuple[GridItemEvidence, ...], ...]
    programs: tuple[ProgramGridEvidence, ...]

    def validate(
        self, expected_environment_ids: Sequence[str] | None = None
    ) -> "FormationGridEvidence":
        if (
            len(self.environment_ids) != FORMATION_ENV_COUNT
            or len(set(self.environment_ids)) != FORMATION_ENV_COUNT
            or len(self.items) != FORMATION_ENV_COUNT
            or any(len(rows) != FORMATION_ENV_ITEM_COUNT for rows in self.items)
        ):
            raise HotpotEvaluatorPortfolioError("formation environment grid drifted")
        if (
            expected_environment_ids is not None
            and self.environment_ids != tuple(expected_environment_ids)
        ):
            raise HotpotEvaluatorPortfolioError("formation environment ids drifted")
        commitments: list[str] = []
        for rows in self.items:
            for item in rows:
                commitments.append(item.validate().item_commitment_sha256)
        if len(set(commitments)) != FORMATION_ITEM_COUNT:
            raise HotpotEvaluatorPortfolioError("formation items are not disjoint")
        if len(self.programs) != CANDIDATE_COUNT:
            raise HotpotEvaluatorPortfolioError("candidate grid is not complete")
        hashes = [
            program.validate(
                FORMATION_ENV_COUNT, FORMATION_ENV_ITEM_COUNT
            ).program_sha256
            for program in self.programs
        ]
        if len(set(hashes)) != CANDIDATE_COUNT:
            raise HotpotEvaluatorPortfolioError("candidate program hashes duplicate")
        return self


@dataclass(frozen=True)
class PortfolioAssessment:
    program_sha256s: tuple[str, str]
    program_lengths: tuple[int, int]
    families: tuple[tuple[str, str], tuple[str, str]]
    pair_sha256: str
    action_sha256: str
    invalid_count: int
    environment_q_hits: tuple[int, int]
    environment_q_complete: tuple[int, int]
    environment_q_coverage: tuple[int, int]
    cell_net: tuple[int, ...]
    cell_gain: tuple[int, ...]
    cell_harm: tuple[int, ...]
    cell_combined_hits: tuple[int, ...]

    @property
    def total_program_length(self) -> int:
        return sum(self.program_lengths)

    @property
    def public_summary(self) -> dict[str, Any]:
        return {
            "program_sha256s": list(self.program_sha256s),
            "pair_sha256": self.pair_sha256,
            "action_sha256": self.action_sha256,
            "capability_family_sha256s": [
                stable_hash({"seed_algorithm": family[0], "expansion_mode": family[1]})
                for family in self.families
            ],
            "invalid_count": self.invalid_count,
            "environment_q_hits": list(self.environment_q_hits),
            "environment_q_complete": list(self.environment_q_complete),
            "environment_q_coverage": list(self.environment_q_coverage),
            "cell_net": list(self.cell_net),
            "cell_gain": list(self.cell_gain),
            "cell_harm": list(self.cell_harm),
            "cell_combined_hits": list(self.cell_combined_hits),
            "total_program_length": self.total_program_length,
        }


def _doc_ids(ranking: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"doc_{index}" for index in ranking)


def _indices(ranking: Sequence[str]) -> tuple[int, ...]:
    result: list[int] = []
    for value in ranking:
        if not isinstance(value, str) or not value.startswith("doc_"):
            raise HotpotEvaluatorPortfolioError("RRF document id drifted")
        result.append(int(value[4:]))
    return _validate_ranking(result, "RRF ranking")


def fuse_rankings(*rankings: Sequence[int]) -> tuple[int, ...]:
    if len(rankings) not in {2, 3}:
        raise HotpotEvaluatorPortfolioError("portfolio RRF width drifted")
    validated = tuple(_validate_ranking(row, "RRF input") for row in rankings)
    return _indices(deterministic_rrf(tuple(_doc_ids(row) for row in validated)))


def _program_behavior_sha256(
    grid: FormationGridEvidence, program: ProgramGridEvidence
) -> str:
    return stable_hash(
        [
            {
                "environment_id": environment_id,
                "rows": [
                    {
                        "item_commitment_sha256": item.item_commitment_sha256,
                        "invalid": ranking is None,
                        "ranking": None if ranking is None else list(ranking),
                    }
                    for item, ranking in zip(
                        grid.items[environment_ordinal],
                        program.q_rankings[environment_ordinal],
                    )
                ],
            }
            for environment_ordinal, environment_id in enumerate(
                grid.environment_ids
            )
        ]
    )


def _retained_p_behavior_sha256(grid: FormationGridEvidence) -> str:
    """Hash retained-P rankings over both environments without gold labels."""

    validated = grid.validate()
    return stable_hash(
        [
            {
                "environment_id": environment_id,
                "rows": [
                    {
                        "item_commitment_sha256": item.item_commitment_sha256,
                        "invalid": False,
                        "ranking": list(item.p_ranking),
                    }
                    for item in validated.items[environment_ordinal]
                ],
            }
            for environment_ordinal, environment_id in enumerate(
                validated.environment_ids
            )
        ]
    )


def canonical_behavior_programs(
    grid: FormationGridEvidence,
    *,
    retained_p_program_sha256: str,
) -> tuple[ProgramGridEvidence, ...]:
    """Exclude retained P, then collapse Q actions gold-free."""

    validated = grid.validate()
    _require_sha256(retained_p_program_sha256, "retained P program")
    retained_behavior = _retained_p_behavior_sha256(validated)
    classes: dict[str, list[ProgramGridEvidence]] = {}
    for program in validated.programs:
        behavior = _program_behavior_sha256(validated, program)
        if (
            program.program_sha256 == retained_p_program_sha256
            or behavior == retained_behavior
        ):
            continue
        classes.setdefault(behavior, []).append(program)
    if len(classes) < 2:
        raise HotpotEvaluatorPortfolioError("fewer than two Q behaviors remain")
    representatives = [
        min(rows, key=lambda row: (row.program_length, row.program_sha256))
        for rows in classes.values()
    ]
    return tuple(sorted(representatives, key=lambda row: row.program_sha256))


def _single_program_as_pair(program: ProgramGridEvidence) -> tuple[ProgramGridEvidence, ...]:
    return (program,)


def _portfolio_assessment(
    grid: FormationGridEvidence,
    programs: Sequence[ProgramGridEvidence],
) -> PortfolioAssessment:
    """Assess a single-Q shortlist candidate or a two-Q formal portfolio."""

    validated = grid.validate()
    selected = tuple(programs)
    if len(selected) not in {1, 2}:
        raise HotpotEvaluatorPortfolioError("portfolio width is invalid")
    if len({row.program_sha256 for row in selected}) != len(selected):
        raise HotpotEvaluatorPortfolioError("portfolio repeats one program")
    ordered = tuple(sorted(selected, key=lambda row: row.program_sha256))
    if len(ordered) == 1:
        # The single-program assessment is an internal shortlist device.  Its
        # pair identity is domain separated from an executable two-Q portfolio.
        hashes = (ordered[0].program_sha256, ordered[0].program_sha256)
        lengths = (ordered[0].program_length, ordered[0].program_length)
        families = (ordered[0].family, ordered[0].family)
    else:
        hashes = (ordered[0].program_sha256, ordered[1].program_sha256)
        lengths = (ordered[0].program_length, ordered[1].program_length)
        families = (ordered[0].family, ordered[1].family)

    invalid_count = 0
    environment_q_hits: list[int] = []
    environment_q_complete: list[int] = []
    environment_q_coverage: list[int] = []
    cell_net: list[int] = []
    cell_gain: list[int] = []
    cell_harm: list[int] = []
    cell_combined_hits: list[int] = []
    action_rows: list[dict[str, Any]] = []

    for environment_ordinal, environment_id in enumerate(validated.environment_ids):
        q_hits_total = q_complete_total = q_coverage_total = 0
        cell_rows: list[list[tuple[int, int, int]]] = [
            [] for _ in range(FOLD_COUNT)
        ]
        for item_ordinal, item in enumerate(validated.items[environment_ordinal]):
            rankings = tuple(
                program.q_rankings[environment_ordinal][item_ordinal]
                for program in ordered
            )
            if any(ranking is None for ranking in rankings):
                invalid_count += 1
                action_rows.append(
                    {
                        "environment_id": environment_id,
                        "item_commitment_sha256": item.item_commitment_sha256,
                        "invalid": True,
                    }
                )
                cell_rows[item_ordinal % FOLD_COUNT].append((0, 0, 0))
                continue
            concrete = tuple(ranking for ranking in rankings if ranking is not None)
            q_pair = (
                concrete[0]
                if len(concrete) == 1
                else fuse_rankings(concrete[0], concrete[1])
            )
            combined = fuse_rankings(item.p_ranking, *concrete)
            supports = frozenset(item.support_indices)
            p_hits = len(supports.intersection(item.p_ranking))
            direct_hits = len(supports.intersection(q_pair))
            combined_hits = len(supports.intersection(combined))
            delta = combined_hits - p_hits
            q_hits_total += direct_hits
            q_complete_total += int(direct_hits == len(supports))
            q_coverage_total += int(direct_hits > 0)
            cell_rows[item_ordinal % FOLD_COUNT].append(
                (delta, int(delta > 0), int(delta < 0))
            )
            action_rows.append(
                {
                    "environment_id": environment_id,
                    "item_commitment_sha256": item.item_commitment_sha256,
                    "invalid": False,
                    "combined_ranking": list(combined),
                }
            )
        environment_q_hits.append(q_hits_total)
        environment_q_complete.append(q_complete_total)
        environment_q_coverage.append(q_coverage_total)
        for fold_ordinal, rows in enumerate(cell_rows):
            cell_net.append(sum(row[0] for row in rows))
            cell_gain.append(sum(row[1] for row in rows))
            cell_harm.append(sum(row[2] for row in rows))
            # combined = P + marginal; compute directly again would duplicate
            # rankings.  The cell total below is reconstructed exactly from P.
            cell_positions = [
                position
                for position in range(FORMATION_ENV_ITEM_COUNT)
                if position % FOLD_COUNT == fold_ordinal
            ]
            p_cell_hits = sum(
                len(
                    frozenset(validated.items[environment_ordinal][position].support_indices)
                    .intersection(validated.items[environment_ordinal][position].p_ranking)
                )
                for position in cell_positions
            )
            cell_combined_hits.append(p_cell_hits + cell_net[-1])

    return PortfolioAssessment(
        program_sha256s=hashes,
        program_lengths=lengths,
        families=families,
        pair_sha256=stable_hash(
            {
                "kind": "single_shortlist" if len(ordered) == 1 else "unordered_pair",
                "program_sha256s": list(hashes),
            }
        ),
        action_sha256=stable_hash(action_rows),
        invalid_count=invalid_count,
        environment_q_hits=tuple(environment_q_hits),  # type: ignore[arg-type]
        environment_q_complete=tuple(environment_q_complete),  # type: ignore[arg-type]
        environment_q_coverage=tuple(environment_q_coverage),  # type: ignore[arg-type]
        cell_net=tuple(cell_net),
        cell_gain=tuple(cell_gain),
        cell_harm=tuple(cell_harm),
        cell_combined_hits=tuple(cell_combined_hits),
    )


def incumbent_key(value: PortfolioAssessment) -> tuple[Any, ...]:
    """Distribution-robust direct-Q objective, then pooled evidence."""

    return (
        value.invalid_count,
        -min(value.environment_q_hits),
        -min(value.environment_q_complete),
        -min(value.environment_q_coverage),
        -sum(value.environment_q_hits),
        -sum(value.environment_q_complete),
        -sum(value.environment_q_coverage),
        value.total_program_length,
        value.pair_sha256,
    )


def challenger_key(value: PortfolioAssessment) -> tuple[Any, ...]:
    """Worst-cell marginal contribution objective over 2 x 4 fixed cells."""

    if not (
        len(value.cell_net)
        == len(value.cell_gain)
        == len(value.cell_harm)
        == len(value.cell_combined_hits)
        == FORMATION_ENV_COUNT * FOLD_COUNT
    ):
        raise HotpotEvaluatorPortfolioError("challenger cell grid drifted")
    return (
        value.invalid_count,
        -min(value.cell_net),
        max(value.cell_harm),
        -min(value.cell_gain),
        -sum(value.cell_net),
        sum(value.cell_harm),
        -sum(value.cell_gain),
        -min(value.cell_combined_hits),
        -sum(value.cell_combined_hits),
        value.total_program_length,
        value.pair_sha256,
    )


def _policy_shortlist(
    grid: FormationGridEvidence,
    *,
    policy_id: str,
    retained_p_program_sha256: str,
) -> tuple[ProgramGridEvidence, ...]:
    canonical = canonical_behavior_programs(
        grid, retained_p_program_sha256=retained_p_program_sha256
    )
    key = incumbent_key if policy_id == INCUMBENT_POLICY_ID else challenger_key
    rows: list[ProgramGridEvidence] = []
    for family in CAPABILITY_FAMILIES:
        eligible = [program for program in canonical if program.family == family]
        if eligible:
            rows.append(
                min(
                    eligible,
                    key=lambda program: key(
                        _portfolio_assessment(grid, _single_program_as_pair(program))
                    ),
                )
            )
    if len(rows) < 2:
        raise HotpotEvaluatorPortfolioError("fewer than two capability families remain")
    return tuple(rows)


def select_portfolio(
    grid: FormationGridEvidence,
    *,
    policy_id: str,
    retained_p_program_sha256: str,
) -> tuple[PortfolioAssessment, tuple[ProgramGridEvidence, ...]]:
    """Select one unordered, action-deduplicated, cross-family portfolio."""

    if policy_id not in {INCUMBENT_POLICY_ID, CHALLENGER_POLICY_ID}:
        raise HotpotEvaluatorPortfolioError("unknown portfolio evaluator policy")
    shortlist = _policy_shortlist(
        grid,
        policy_id=policy_id,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    candidates: list[PortfolioAssessment] = []
    for left in range(len(shortlist)):
        for right in range(left + 1, len(shortlist)):
            if shortlist[left].family == shortlist[right].family:
                continue
            candidates.append(
                _portfolio_assessment(grid, (shortlist[left], shortlist[right]))
            )
    if not candidates:
        raise HotpotEvaluatorPortfolioError("no two-family portfolios remain")

    # Dedupe on the executable P+Q1+Q2 action, before looking at policy score.
    action_classes: dict[str, list[PortfolioAssessment]] = {}
    for candidate in candidates:
        action_classes.setdefault(candidate.action_sha256, []).append(candidate)
    canonical = [
        min(
            rows,
            key=lambda row: (row.total_program_length, row.pair_sha256),
        )
        for rows in action_classes.values()
    ]
    key = incumbent_key if policy_id == INCUMBENT_POLICY_ID else challenger_key
    return min(canonical, key=key), shortlist


def form_portfolio_policies_from_evidence(
    grid: FormationGridEvidence,
    *,
    expected_environment_ids: Sequence[str],
    retained_p_program_sha256: str,
) -> dict[str, Any]:
    """Pure formation core.  It never substitutes a runner-up for contrast."""

    validated = grid.validate(expected_environment_ids)
    incumbent, incumbent_shortlist = select_portfolio(
        validated,
        policy_id=INCUMBENT_POLICY_ID,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    challenger, challenger_shortlist = select_portfolio(
        validated,
        policy_id=CHALLENGER_POLICY_ID,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    measurable = (
        incumbent.program_sha256s != challenger.program_sha256s
        and incumbent.action_sha256 != challenger.action_sha256
    )
    body: dict[str, Any] = {
        "schema": f"{VERSION}_formation_core",
        "environment_ids": list(validated.environment_ids),
        "environment_count": FORMATION_ENV_COUNT,
        "environment_item_count": FORMATION_ENV_ITEM_COUNT,
        "item_count": FORMATION_ITEM_COUNT,
        "fold_count_per_environment": FOLD_COUNT,
        "fold_policy": "acquisition_ordinal_modulo_4_within_environment_v1",
        "fixed_cell_count": FORMATION_ENV_COUNT * FOLD_COUNT,
        "candidate_program_count": len(validated.programs),
        "gold_free_behavior_class_count": len(
            canonical_behavior_programs(
                validated,
                retained_p_program_sha256=retained_p_program_sha256,
            )
        ),
        "retained_P_program_sha256": _require_sha256(
            retained_p_program_sha256, "retained P program"
        ),
        "retained_P_behavior_sha256": _retained_p_behavior_sha256(validated),
        "retained_P_excluded_from_portfolios": True,
        "retained_P_hash_and_behavior_class_excluded": True,
        "incumbent_policy_id": INCUMBENT_POLICY_ID,
        "challenger_policy_id": CHALLENGER_POLICY_ID,
        "incumbent_shortlist_program_sha256s": [
            row.program_sha256 for row in incumbent_shortlist
        ],
        "challenger_shortlist_program_sha256s": [
            row.program_sha256 for row in challenger_shortlist
        ],
        "incumbent": incumbent.public_summary,
        "challenger": challenger.public_summary,
        "measurable_contrast": measurable,
        "behavior_distinct_required": True,
        "identical_action_has_no_runner_up_or_fallback": True,
        "both_policies_used_identical_48_rows": True,
        "portfolio_action": "deterministic_RRF_retained_P_Q1_Q2_at_5",
        "logical_retrieval_calls_per_compared_arm": 3,
        "model_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    return {**body, "formation_sha256": stable_hash(body)}


def _acquisition_module() -> Any:
    try:
        return importlib.import_module(
            ".hotpot_evaluator_robust_acquisition_v1", package=__package__
        )
    except (ImportError, AttributeError) as exc:
        raise HotpotEvaluatorPortfolioError(
            "portfolio acquisition implementation is unavailable"
        ) from exc


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    count: int
    file_sha256: str
    item_commitment_set_sha256: str


def _load_acquisition_live(
    *, project: Path, path: str | Path
) -> tuple[dict[str, Any], bytes, dict[str, BlockCommitment]]:
    acquisition = _acquisition_module()
    try:
        receipt, rows = acquisition.load_acquisition_binding(path)
    except Exception as exc:
        raise HotpotEvaluatorPortfolioError("acquisition receipt is invalid") from exc
    raw = Path(path).read_bytes()
    expected_counts = {
        "A_form_0": 24,
        "A_form_1": 24,
        "F_search_0": 24,
        "F_search_1": 24,
        "A_hold": 48,
        "M_search": 24,
    }
    if (
        getattr(acquisition, "BLOCK_COUNTS", None) != expected_counts
        or tuple(getattr(acquisition, "BLOCK_ORDER", ()))
        != tuple(expected_counts)
        or receipt.get("implementation")
        != acquisition.implementation_binding(project)
        or receipt.get("retained_P_lineage")
        != acquisition.retained_p_lineage_binding(project)
        or receipt.get("portfolio_design_binding")
        != acquisition.portfolio_design_binding(project)
    ):
        raise HotpotEvaluatorPortfolioError(
            "acquisition implementation, blocks, lineage, or design drifted"
        )
    commitments: dict[str, BlockCommitment] = {}
    for row in rows:
        commitment = BlockCommitment(
            block=str(row.block),
            count=int(row.count),
            file_sha256=_require_sha256(row.file_sha256, "block file"),
            item_commitment_set_sha256=_require_sha256(
                row.item_commitment_set_sha256, "block item set"
            ),
        )
        commitments[commitment.block] = commitment
    if (
        set(commitments) != set(expected_counts)
        or any(
            commitments[block].count != count
            for block, count in expected_counts.items()
        )
    ):
        raise HotpotEvaluatorPortfolioError("acquisition commitments drifted")
    _assert_public_safe(receipt)
    return receipt, raw, commitments


def _source_binding(
    receipt: Mapping[str, Any], receipt_raw: bytes, commitment: BlockCommitment
) -> dict[str, Any]:
    return {
        "acquisition_file_sha256": _sha256_bytes(receipt_raw),
        "acquisition_sha256": _require_sha256(
            receipt.get("acquisition_sha256"), "acquisition semantic hash"
        ),
        "portfolio_design_binding": dict(
            receipt.get("portfolio_design_binding", {})
        ),
        "block_id_sha256": stable_hash({"block": commitment.block}),
        "block_file_sha256": commitment.file_sha256,
        "item_commitment_set_sha256": commitment.item_commitment_set_sha256,
        "item_count": commitment.count,
    }


def _l4_commitment(value: BlockCommitment) -> l4.BlockCommitment:
    return l4.BlockCommitment(
        block=value.block,
        count=value.count,
        file_sha256=value.file_sha256,
        item_commitment_set_sha256=value.item_commitment_set_sha256,
    )


def _load_old_final_disposition(path: str | Path) -> dict[str, Any]:
    value, raw = _read_json(path, "old terminal disposition")
    body = dict(value)
    declared = _require_sha256(
        body.pop("disposition_sha256", None), "old disposition"
    )
    l5 = value.get("L5")
    terminal = value.get("terminal_policy")
    if (
        _sha256_bytes(raw) != OLD_FINAL_DISPOSITION_FILE_SHA256
        or declared != OLD_FINAL_DISPOSITION_SHA256
        or stable_hash(body) != declared
        or value.get("schema") != "hotpot_recursive_study_v1_final_disposition"
        or value.get("status")
        != "L4_narrow_positive_L5_no_promotion_terminal"
        or not isinstance(l5, Mapping)
        or l5.get("challenger_promoted") is not False
        or l5.get("M_search_opened") is not False
        or not isinstance(terminal, Mapping)
        or terminal.get("future_L5_requires_new_mechanism_and_new_cohort")
        is not True
        or terminal.get("same_anchor_retry_replay_resample") is not False
    ):
        raise HotpotEvaluatorPortfolioError("old terminal disposition drifted")
    _assert_public_safe(value)
    return {
        "file_sha256": _sha256_bytes(raw),
        "disposition_sha256": declared,
        "old_M_search_opened": False,
        "requires_new_mechanism_and_new_cohort": True,
    }


def _load_p_lineage(
    *,
    project: Path,
    acquisition_receipt: Mapping[str, Any],
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
) -> tuple[TypedRetrievalProgram, dict[str, Any]]:
    try:
        program, p_binding = l4._load_p(
            project=project,
            formation_path=p_formation_receipt_path,
            program_path=p_frozen_program_path,
        )
        positive = l4._load_positive_p_lineage(
            m1_freeze_path=m1_freeze_path,
            m1_report_path=m1_report_path,
            p_binding=p_binding,
        )
    except l4.HotpotRecursiveL4Error as exc:
        raise HotpotEvaluatorPortfolioError("retained-P lineage drifted") from exc
    acquisition_lineage = acquisition_receipt.get("retained_P_lineage")
    files = (
        acquisition_lineage.get("files")
        if isinstance(acquisition_lineage, Mapping)
        else None
    )
    if not isinstance(files, list):
        raise HotpotEvaluatorPortfolioError("acquisition P lineage is unavailable")
    by_role = {
        str(row.get("role")): str(row.get("sha256"))
        for row in files
        if isinstance(row, Mapping)
    }
    expected = {
        "P_formation_receipt": p_binding["formation_receipt_file_sha256"],
        "P_frozen_program": p_binding["frozen_program_file_sha256"],
        "M1_pre_run_freeze": positive["m1_freeze_file_sha256"],
        "M1_positive_promotion_report": positive["m1_report_file_sha256"],
    }
    if any(by_role.get(role) != digest for role, digest in expected.items()):
        raise HotpotEvaluatorPortfolioError(
            "retained P differs from acquisition preregistration"
        )
    return program, {"retained_P": p_binding, "positive_M1": positive}


def _grid_to_dict(grid: FormationGridEvidence) -> dict[str, Any]:
    validated = grid.validate()
    return {
        "environment_ids": list(validated.environment_ids),
        "items": [
            [
                {
                    "item_commitment_sha256": row.item_commitment_sha256,
                    "p_ranking": list(row.p_ranking),
                    "support_indices": list(row.support_indices),
                }
                for row in environment
            ]
            for environment in validated.items
        ],
        "programs": [
            {
                "program_sha256": program.program_sha256,
                "program_length": program.program_length,
                "seed_algorithm": program.seed_algorithm,
                "expansion_mode": program.expansion_mode,
                "q_rankings": [
                    [None if row is None else list(row) for row in environment]
                    for environment in program.q_rankings
                ],
            }
            for program in validated.programs
        ],
    }


def _grid_from_dict(value: object) -> FormationGridEvidence:
    if not isinstance(value, Mapping):
        raise HotpotEvaluatorPortfolioError("private formation grid is malformed")
    try:
        items = tuple(
            tuple(
                GridItemEvidence(
                    item_commitment_sha256=str(row["item_commitment_sha256"]),
                    p_ranking=tuple(row["p_ranking"]),
                    support_indices=tuple(row["support_indices"]),
                )
                for row in environment
            )
            for environment in value["items"]
        )
        programs = tuple(
            ProgramGridEvidence(
                program_sha256=str(row["program_sha256"]),
                program_length=int(row["program_length"]),
                seed_algorithm=str(row["seed_algorithm"]),
                expansion_mode=str(row["expansion_mode"]),
                q_rankings=tuple(
                    tuple(None if ranking is None else tuple(ranking) for ranking in environment)
                    for environment in row["q_rankings"]
                ),
            )
            for row in value["programs"]
        )
        grid = FormationGridEvidence(
            environment_ids=tuple(value["environment_ids"]),
            items=items,
            programs=programs,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HotpotEvaluatorPortfolioError(
            "private formation grid is malformed"
        ) from exc
    return grid.validate()


def formation_evidence_sha256(grid: FormationGridEvidence) -> str:
    return stable_hash(_grid_to_dict(grid))


FORMATION_WORK_UNIT_COUNT = (
    CANDIDATE_COUNT + 1
) * FORMATION_ITEM_COUNT
FORMATION_ENV_WORK_UNIT_COUNT = (CANDIDATE_COUNT + 1) * FORMATION_ENV_ITEM_COUNT
FORMATION_MAXIMUM_CONCURRENCY = FORMATION_ENV_WORK_UNIT_COUNT


def _evaluate_formation_grid(
    *,
    p_program: TypedRetrievalProgram,
    environment_ids: Sequence[str],
    environments: Sequence[Sequence[l4.RecursiveItem]],
) -> tuple[FormationGridEvidence, dict[str, Any]]:
    """Complete one 4,080-terminal grid before reading support labels."""

    if (
        tuple(environment_ids) not in {A_FORM_ENVIRONMENTS, F_SEARCH_ENVIRONMENTS}
        or len(environments) != FORMATION_ENV_COUNT
        or any(len(rows) != FORMATION_ENV_ITEM_COUNT for rows in environments)
    ):
        raise HotpotEvaluatorPortfolioError("formation input environments drifted")
    programs = fixed_programs()
    work_units: list[tuple[int, int, int, l4.RetrievalItem]] = []
    for environment_ordinal, rows in enumerate(environments):
        for item_ordinal, item in enumerate(rows):
            view = item.retrieval_view()
            work_units.append((environment_ordinal, item_ordinal, -1, view))
            work_units.extend(
                (environment_ordinal, item_ordinal, program_ordinal, view)
                for program_ordinal in range(CANDIDATE_COUNT)
            )
    if len(work_units) != FORMATION_WORK_UNIT_COUNT:
        raise HotpotEvaluatorPortfolioError("formation work grid drifted")
    attempted = completed = barrier_count = 0
    lock = threading.Lock()

    def run_one(
        unit: tuple[int, int, int, l4.RetrievalItem], barrier: threading.Barrier
    ) -> tuple[tuple[int, int, int], tuple[int, ...] | None]:
        nonlocal attempted, completed
        environment_ordinal, item_ordinal, program_ordinal, item = unit
        with lock:
            attempted += 1
        try:
            barrier.wait(timeout=180)
        except threading.BrokenBarrierError as exc:
            raise HotpotEvaluatorPortfolioError(
                "formation maximum-width barrier did not close"
            ) from exc
        try:
            ranking = l4._ranking(
                p_program if program_ordinal == -1 else programs[program_ordinal],
                item,
            )
        except l4.HotpotRecursiveL4Error:
            if program_ordinal == -1:
                raise
            ranking = None
        with lock:
            completed += 1
        return (environment_ordinal, item_ordinal, program_ordinal), ranking

    terminals: list[tuple[tuple[int, int, int], tuple[int, ...] | None]] = []
    for environment_ordinal in range(FORMATION_ENV_COUNT):
        environment_units = [
            unit for unit in work_units if unit[0] == environment_ordinal
        ]
        if len(environment_units) != FORMATION_ENV_WORK_UNIT_COUNT:
            raise HotpotEvaluatorPortfolioError("formation environment grid drifted")
        barrier = threading.Barrier(FORMATION_ENV_WORK_UNIT_COUNT)
        barrier_count += 1
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=FORMATION_MAXIMUM_CONCURRENCY,
            thread_name_prefix=f"hotpot-portfolio-env{environment_ordinal}",
        ) as executor:
            futures = [
                executor.submit(run_one, unit, barrier)
                for unit in environment_units
            ]
            terminals.extend(future.result() for future in futures)
    direct = dict(terminals)
    if (
        attempted != FORMATION_WORK_UNIT_COUNT
        or completed != FORMATION_WORK_UNIT_COUNT
        or len(direct) != FORMATION_WORK_UNIT_COUNT
    ):
        raise HotpotEvaluatorPortfolioError("formation terminal closure incomplete")

    # Source labels are consulted only after every retrieval terminal joined.
    item_grid: list[tuple[GridItemEvidence, ...]] = []
    for environment_ordinal, rows in enumerate(environments):
        item_grid.append(
            tuple(
                GridItemEvidence(
                    item_commitment_sha256=item.row_commitment_sha256,
                    p_ranking=direct[(environment_ordinal, item_ordinal, -1)],  # type: ignore[arg-type]
                    support_indices=item.support_indices,
                ).validate()
                for item_ordinal, item in enumerate(rows)
            )
        )
    program_grid = tuple(
        ProgramGridEvidence(
            program_sha256=program.program_hash,
            program_length=program.program_length,
            seed_algorithm=program.seed_algorithm,
            expansion_mode=program.expansion_mode,
            q_rankings=tuple(
                tuple(
                    direct[(environment_ordinal, item_ordinal, program_ordinal)]
                    for item_ordinal in range(FORMATION_ENV_ITEM_COUNT)
                )
                for environment_ordinal in range(FORMATION_ENV_COUNT)
            ),
        ).validate(FORMATION_ENV_COUNT, FORMATION_ENV_ITEM_COUNT)
        for program_ordinal, program in enumerate(programs)
    )
    grid = FormationGridEvidence(
        environment_ids=tuple(environment_ids),
        items=tuple(item_grid),
        programs=program_grid,
    ).validate(environment_ids)
    return grid, {
        "candidate_program_count": CANDIDATE_COUNT,
        "environment_count": FORMATION_ENV_COUNT,
        "environment_item_count": FORMATION_ENV_ITEM_COUNT,
        "item_count": FORMATION_ITEM_COUNT,
        "physical_work_unit_count": FORMATION_WORK_UNIT_COUNT,
        "retrieval_attempt_count": attempted,
        "retrieval_terminal_count": completed,
        "configured_maximum_concurrency": FORMATION_MAXIMUM_CONCURRENCY,
        "environment_barrier_count": barrier_count,
        "environment_barrier_party_count": FORMATION_ENV_WORK_UNIT_COUNT,
        "all_terminals_joined_before_support_scoring": True,
        "model_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def _formation_marker_path(project: Path, stage: str) -> Path:
    mapping = {
        "A_form": A_FORM_CONSUMPTION_RELATIVE,
        "F_search": F_SEARCH_CONSUMPTION_RELATIVE,
    }
    try:
        return project / mapping[stage]
    except KeyError as exc:
        raise HotpotEvaluatorPortfolioError("unknown formation stage") from exc


def _write_formation_marker(
    *,
    project: Path,
    stage: str,
    acquisition_file_sha256: str,
    output_cache_path: str | Path,
    output_receipt_path: str | Path,
) -> dict[str, Any]:
    marker_path = _formation_marker_path(project, stage)
    if marker_path.exists():
        raise HotpotEvaluatorPortfolioError(
            f"{stage} authorization is already consumed; replay forbidden"
        )
    body = {
        "schema": f"{VERSION}_formation_consumption",
        "stage": stage,
        "acquisition_file_sha256": _require_sha256(
            acquisition_file_sha256, "acquisition file"
        ),
        "private_cache_output_sha256": stable_hash(
            {"absolute_output": str(Path(output_cache_path).absolute())}
        ),
        "public_receipt_output_sha256": stable_hash(
            {"absolute_output": str(Path(output_receipt_path).absolute())}
        ),
        "private_block_rows_opened_before_marker": 0,
        "retry_replay_resample_authorized": False,
        "raw_content_persisted": False,
    }
    marker = {**body, "consumption_sha256": stable_hash(body)}
    _write_json_exclusive(marker_path, marker)
    return {
        "marker_file_sha256": _sha256_file(marker_path),
        "marker_sha256": marker["consumption_sha256"],
        "marker_written_before_both_private_environment_blocks_open": True,
        "private_block_rows_opened_before_marker": 0,
        "retry_replay_resample_authorized": False,
        "private_path_persisted_publicly": False,
    }


def _formation_source_binding(
    *,
    receipt: Mapping[str, Any],
    receipt_raw: bytes,
    commitments: Mapping[str, BlockCommitment],
    environment_ids: Sequence[str],
) -> dict[str, Any]:
    """Bind one formation stage to its exact ordered pair of private blocks."""

    environments = tuple(environment_ids)
    if environments not in {A_FORM_ENVIRONMENTS, F_SEARCH_ENVIRONMENTS}:
        raise HotpotEvaluatorPortfolioError("formation environment ids drifted")
    if set(commitments) != set(_acquisition_module().BLOCK_ORDER):
        raise HotpotEvaluatorPortfolioError("acquisition commitment set drifted")
    blocks = [
        _source_binding(receipt, receipt_raw, commitments[environment])
        for environment in environments
    ]
    if any(row["item_count"] != FORMATION_ENV_ITEM_COUNT for row in blocks):
        raise HotpotEvaluatorPortfolioError("formation environment count drifted")
    return {
        "acquisition_file_sha256": _sha256_bytes(receipt_raw),
        "acquisition_sha256": _require_sha256(
            receipt.get("acquisition_sha256"), "acquisition semantic hash"
        ),
        "portfolio_design_binding": dict(
            receipt.get("portfolio_design_binding", {})
        ),
        "environment_ids": list(environments),
        "environment_count": FORMATION_ENV_COUNT,
        "environment_item_count": FORMATION_ENV_ITEM_COUNT,
        "item_count": FORMATION_ITEM_COUNT,
        "blocks": blocks,
        "ordered_block_binding_sha256": stable_hash(blocks),
    }


def _formation_receipt_binding(
    *, private_cache_path: str | Path, public_receipt_path: str | Path
) -> dict[str, Any]:
    """Return a path-free binding to an exact private/public formation bundle."""

    cache, cache_raw = _read_json(private_cache_path, "private formation cache")
    public, public_raw = _read_json(public_receipt_path, "public formation receipt")
    cache_body = dict(cache)
    public_body = dict(public)
    cache_hash = _require_sha256(
        cache_body.pop("cache_sha256", None), "private formation cache"
    )
    receipt_hash = _require_sha256(
        public_body.pop("receipt_sha256", None), "public formation receipt"
    )
    formation_hash = _require_sha256(
        public.get("formation_sha256"), "formation semantic hash"
    )
    if stable_hash(cache_body) != cache_hash or stable_hash(public_body) != receipt_hash:
        raise HotpotEvaluatorPortfolioError("formation bundle self-hash drifted")
    return {
        "private_cache_file_sha256": _sha256_bytes(cache_raw),
        "private_cache_sha256": cache_hash,
        "public_receipt_file_sha256": _sha256_bytes(public_raw),
        "public_receipt_sha256": receipt_hash,
        "formation_sha256": formation_hash,
        "private_path_persisted_publicly": False,
    }


def _load_formation_bundle(
    *,
    project: Path,
    private_cache_path: str | Path,
    public_receipt_path: str | Path,
    expected_stage: str,
    acquisition_receipt: Mapping[str, Any],
    acquisition_raw: bytes,
    commitments: Mapping[str, BlockCommitment],
    p_lineage: Mapping[str, Any],
    old_disposition: Mapping[str, Any],
) -> tuple[FormationGridEvidence, dict[str, Any], dict[str, Any]]:
    """Strictly rederive a formation grid, selector decision, and receipt."""

    environment_ids = {
        "A_form": A_FORM_ENVIRONMENTS,
        "F_search": F_SEARCH_ENVIRONMENTS,
    }.get(expected_stage)
    if environment_ids is None:
        raise HotpotEvaluatorPortfolioError("unknown formation stage")
    cache, cache_raw = _read_json(private_cache_path, "private formation cache")
    public, _public_raw = _read_json(public_receipt_path, "public formation receipt")
    cache_body = dict(cache)
    public_body = dict(public)
    cache_hash = _require_sha256(
        cache_body.pop("cache_sha256", None), "private formation cache"
    )
    receipt_hash = _require_sha256(
        public_body.pop("receipt_sha256", None), "public formation receipt"
    )
    if (
        stable_hash(cache_body) != cache_hash
        or stable_hash(public_body) != receipt_hash
        or cache.get("schema") != PRIVATE_CACHE_SCHEMA
        or public.get("schema") != FORMATION_RECEIPT_SCHEMA
        or cache.get("stage") != expected_stage
        or public.get("stage") != expected_stage
    ):
        raise HotpotEvaluatorPortfolioError("formation bundle envelope drifted")
    grid = _grid_from_dict(cache.get("grid"))
    grid.validate(environment_ids)
    core = form_portfolio_policies_from_evidence(
        grid,
        expected_environment_ids=environment_ids,
        retained_p_program_sha256=p_lineage["retained_P"]["program_hash"],
    )
    source = _formation_source_binding(
        receipt=acquisition_receipt,
        receipt_raw=acquisition_raw,
        commitments=commitments,
        environment_ids=environment_ids,
    )
    implementation = current_implementation_binding(project)
    grid_hash = formation_evidence_sha256(grid)
    ordering = _load_formation_marker(
        project=project,
        stage=expected_stage,
        acquisition_file_sha256=_sha256_bytes(acquisition_raw),
        private_cache_path=private_cache_path,
        public_receipt_path=public_receipt_path,
    )
    offline_contract = {
        "model_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "measurement_blocks_accessed": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }
    design = acquisition_receipt.get("portfolio_design_binding")
    if (
        cache.get("formation_evidence_sha256") != grid_hash
        or cache.get("source_binding") != source
        or cache.get("portfolio_design_binding") != design
        or cache.get("lineage_binding") != p_lineage
        or cache.get("old_terminal_disposition_binding") != old_disposition
        or cache.get("formation_core") != core
        or public.get("formation_sha256") != core["formation_sha256"]
        or public.get("formation_core") != core
        or public.get("source_binding") != source
        or public.get("portfolio_design_binding") != design
        or public.get("lineage_binding") != p_lineage
        or public.get("old_terminal_disposition_binding") != old_disposition
        or public.get("candidate_set_binding") != candidate_set_binding()
        or cache.get("formation_consumption_binding") != ordering
        or public.get("prospective_ordering") != ordering
        or public.get("execution") != cache.get("execution")
        or public.get("offline_contract") != offline_contract
        or public.get("status")
        != (
            "behavior_distinct_portfolio_frozen"
            if core["measurable_contrast"]
            else "terminal_no_behavior_distinct_portfolio"
        )
        or public.get("implementation") != implementation
        or public.get("private_evidence_binding")
        != {
            "private_cache_file_sha256": _sha256_bytes(cache_raw),
            "private_cache_sha256": cache_hash,
            "formation_evidence_sha256": grid_hash,
            "private_path_persisted_publicly": False,
        }
    ):
        raise HotpotEvaluatorPortfolioError("formation bundle derivation drifted")
    _assert_public_safe(public)
    return grid, public, _formation_receipt_binding(
        private_cache_path=private_cache_path,
        public_receipt_path=public_receipt_path,
    )


ANCHOR_CONSUMPTION_FILENAME = "a_hold.authorization.consumed.json"
ANCHOR_PRIVATE_EVIDENCE_FILENAME = "a_hold.private.evidence.json"
ANCHOR_REPORT_FILENAME = "a_hold.aggregate.report.json"
ANCHOR_FAILURE_FILENAME = "a_hold.failure.json"
SEARCH_CONSUMPTION_FILENAME = "m_search.authorization.consumed.json"
SEARCH_REPORT_FILENAME = "m_search.aggregate.report.json"
SEARCH_FAILURE_FILENAME = "m_search.failure.json"


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _new_root(path: str | Path, project: Path) -> Path:
    try:
        return l4._new_root(path, project)
    except l4.HotpotRecursiveL4Error as exc:
        raise HotpotEvaluatorPortfolioError(
            "execution root must be fresh, ignored, and private"
        ) from exc


def _root_hash(path: str | Path, project: Path) -> str:
    return stable_hash({"absolute_execution_root": str(_new_root(path, project))})


def _program_by_hash(program_sha256: str) -> TypedRetrievalProgram:
    digest = _require_sha256(program_sha256, "typed program")
    rows = [program for program in fixed_programs() if program.program_hash == digest]
    if len(rows) != 1:
        raise HotpotEvaluatorPortfolioError("selected typed program is unavailable")
    return rows[0]


def _program_pair(core: Mapping[str, Any], role: str) -> tuple[TypedRetrievalProgram, ...]:
    summary = core.get(role)
    hashes = summary.get("program_sha256s") if isinstance(summary, Mapping) else None
    if (
        not isinstance(hashes, list)
        or len(hashes) != 2
        or len(set(hashes)) != 2
        or any(not isinstance(value, str) for value in hashes)
    ):
        raise HotpotEvaluatorPortfolioError("selected portfolio pair drifted")
    programs = tuple(_program_by_hash(value) for value in hashes)
    if tuple(sorted(hashes)) != tuple(hashes):
        raise HotpotEvaluatorPortfolioError("portfolio pair order drifted")
    return programs


def exact_paired_sign_flip(deltas: Sequence[int]) -> dict[str, Any]:
    """Magnitude-preserving exact one-sided sign-flip test at alpha 0.10."""

    if not deltas or any(type(value) is not int for value in deltas):
        raise HotpotEvaluatorPortfolioError("paired deltas are malformed")
    observed = sum(deltas)
    magnitudes = [abs(value) for value in deltas if value]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
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


def _policy_criteria_hash(policy_id: str) -> str:
    if policy_id == INCUMBENT_POLICY_ID:
        return stable_hash({"policy": policy_id, "key": "incumbent_key"})
    if policy_id == CHALLENGER_POLICY_ID:
        return stable_hash({"policy": policy_id, "key": "challenger_key"})
    raise HotpotEvaluatorPortfolioError("unknown evaluator policy")


def _archive_transition(
    *,
    anchor_manifest_sha256: str,
    incumbent_hits: int,
    challenger_hits: int,
    support_total: int,
    item_count: int,
    promoted: bool,
) -> dict[str, Any]:
    """Apply the exact evaluator transition and selective invalidation."""

    anchor = _require_sha256(anchor_manifest_sha256, "anchor manifest")
    incumbent_spec = EvaluatorSpec(
        id=INCUMBENT_POLICY_ID,
        version=VERSION,
        implementation_hash=stable_hash({"module": VERSION}),
        criteria_hash=_policy_criteria_hash(INCUMBENT_POLICY_ID),
        anchor_manifest_hash=anchor,
    )
    challenger_spec = EvaluatorSpec(
        id=CHALLENGER_POLICY_ID,
        version=VERSION,
        implementation_hash=stable_hash({"module": VERSION}),
        criteria_hash=_policy_criteria_hash(CHALLENGER_POLICY_ID),
        anchor_manifest_hash=anchor,
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
    safe_items = tuple(
        stable_hash({"anchor": anchor, "ordinal": ordinal})
        for ordinal in range(item_count)
    )
    dependent = archive.record_score(
        archive_node_id=node.id,
        split="A_hold",
        evaluator_epoch_id=incumbent_epoch.id,
        metric="evaluator_dependent_portfolio_support",
        successes=incumbent_hits,
        total=support_total,
        item_ids=safe_items,
    )
    independent = archive.record_score(
        archive_node_id=node.id,
        split="A_hold",
        evaluator_epoch_id="fixed_source_support_v1",
        metric="independent_source_support",
        successes=challenger_hits,
        total=support_total,
        item_ids=safe_items,
    )
    if promoted:
        invalidated = archive.invalidate_evaluator_epoch(incumbent_epoch.id)
        next_epoch = EvaluatorEpoch(
            id=(
                "eval_epoch_1_"
                + stable_hash(
                    {"challenger": asdict(challenger_spec), "parent": incumbent_epoch.id}
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


def _load_formation_marker(
    *,
    project: Path,
    stage: str,
    acquisition_file_sha256: str,
    private_cache_path: str | Path,
    public_receipt_path: str | Path,
) -> dict[str, Any]:
    marker, marker_raw = _read_json(
        _formation_marker_path(project, stage), "formation consumption marker"
    )
    body = dict(marker)
    declared = _require_sha256(
        body.pop("consumption_sha256", None), "formation consumption"
    )
    if (
        marker.get("schema") != f"{VERSION}_formation_consumption"
        or marker.get("stage") != stage
        or stable_hash(body) != declared
        or marker.get("acquisition_file_sha256") != acquisition_file_sha256
        or marker.get("private_cache_output_sha256")
        != stable_hash({"absolute_output": str(Path(private_cache_path).absolute())})
        or marker.get("public_receipt_output_sha256")
        != stable_hash({"absolute_output": str(Path(public_receipt_path).absolute())})
        or marker.get("private_block_rows_opened_before_marker") != 0
        or marker.get("retry_replay_resample_authorized") is not False
    ):
        raise HotpotEvaluatorPortfolioError("formation consumption marker drifted")
    return {
        "marker_file_sha256": _sha256_bytes(marker_raw),
        "marker_sha256": declared,
        "marker_written_before_both_private_environment_blocks_open": True,
        "private_block_rows_opened_before_marker": 0,
        "retry_replay_resample_authorized": False,
        "private_path_persisted_publicly": False,
    }


def _form_stage(
    *,
    project_root: str | Path,
    stage: str,
    acquisition_receipt_path: str | Path,
    environment_0_block_path: str | Path,
    environment_1_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    private_cache_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    environments = {
        "A_form": A_FORM_ENVIRONMENTS,
        "F_search": F_SEARCH_ENVIRONMENTS,
    }.get(stage)
    if environments is None:
        raise HotpotEvaluatorPortfolioError("unknown formation stage")
    receipt, receipt_raw, commitments = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    p_program, lineage = _load_p_lineage(
        project=project,
        acquisition_receipt=receipt,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
    )
    old = _load_old_final_disposition(old_final_disposition_path)
    source = _formation_source_binding(
        receipt=receipt,
        receipt_raw=receipt_raw,
        commitments=commitments,
        environment_ids=environments,
    )
    private_output = Path(private_cache_output_path).absolute()
    public_output = Path(public_receipt_output_path).absolute()
    if private_output.exists() or public_output.exists():
        raise HotpotEvaluatorPortfolioError("formation output already exists")
    l4._assert_git_ignored_private_path(
        project=project, path=private_output, require_file=None
    )
    ordering = _write_formation_marker(
        project=project,
        stage=stage,
        acquisition_file_sha256=_sha256_bytes(receipt_raw),
        output_cache_path=private_output,
        output_receipt_path=public_output,
    )
    try:
        block_rows = tuple(
            l4._load_private_block(
                project=project,
                path=path,
                commitment=_l4_commitment(commitments[environment]),
            )
            for environment, path in zip(
                environments, (environment_0_block_path, environment_1_block_path)
            )
        )
    except Exception as exc:
        raise HotpotEvaluatorPortfolioError(
            "formation private environment failed after authorization consumption"
        ) from exc
    grid, execution = _evaluate_formation_grid(
        p_program=p_program,
        environment_ids=environments,
        environments=block_rows,
    )
    core = form_portfolio_policies_from_evidence(
        grid,
        expected_environment_ids=environments,
        retained_p_program_sha256=p_program.program_hash,
    )
    implementation = current_implementation_binding(project)
    evidence_hash = formation_evidence_sha256(grid)
    cache_body: dict[str, Any] = {
        "schema": PRIVATE_CACHE_SCHEMA,
        "stage": stage,
        "source_binding": source,
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "lineage_binding": lineage,
        "old_terminal_disposition_binding": old,
        "formation_evidence_sha256": evidence_hash,
        "formation_core": core,
        "formation_consumption_binding": ordering,
        "execution": execution,
        "grid": _grid_to_dict(grid),
        "raw_question_or_corpus_persisted": False,
    }
    cache = {**cache_body, "cache_sha256": stable_hash(cache_body)}
    cache_raw = _json_bytes(cache)
    public_body: dict[str, Any] = {
        "schema": FORMATION_RECEIPT_SCHEMA,
        "stage": stage,
        "status": (
            "behavior_distinct_portfolio_frozen"
            if core["measurable_contrast"]
            else "terminal_no_behavior_distinct_portfolio"
        ),
        "formation_sha256": core["formation_sha256"],
        "formation_core": core,
        "implementation": implementation,
        "source_binding": source,
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "lineage_binding": lineage,
        "old_terminal_disposition_binding": old,
        "candidate_set_binding": candidate_set_binding(),
        "private_evidence_binding": {
            "private_cache_file_sha256": _sha256_bytes(cache_raw),
            "private_cache_sha256": cache["cache_sha256"],
            "formation_evidence_sha256": evidence_hash,
            "private_path_persisted_publicly": False,
        },
        "execution": execution,
        "prospective_ordering": ordering,
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
    public = {**public_body, "receipt_sha256": stable_hash(public_body)}
    _assert_public_safe(public)
    _write_json_exclusive(private_output, cache, mode=0o600)
    _write_json_exclusive(public_output, public, mode=0o644)
    if private_output.read_bytes() != cache_raw:
        raise HotpotEvaluatorPortfolioError("private formation persistence drifted")
    return public


def form_a_form_stage(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_0_block_path: str | Path,
    a_form_1_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    private_cache_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    return _form_stage(
        project_root=project_root,
        stage="A_form",
        acquisition_receipt_path=acquisition_receipt_path,
        environment_0_block_path=a_form_0_block_path,
        environment_1_block_path=a_form_1_block_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        private_cache_output_path=private_cache_output_path,
        public_receipt_output_path=public_receipt_output_path,
    )


def form_f_search_stage(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    f_search_0_block_path: str | Path,
    f_search_1_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    private_cache_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    return _form_stage(
        project_root=project_root,
        stage="F_search",
        acquisition_receipt_path=acquisition_receipt_path,
        environment_0_block_path=f_search_0_block_path,
        environment_1_block_path=f_search_1_block_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        private_cache_output_path=private_cache_output_path,
        public_receipt_output_path=public_receipt_output_path,
    )


def _artifact_bundles(
    *,
    project: Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[
    dict[str, Any], bytes, dict[str, BlockCommitment], TypedRetrievalProgram,
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    receipt, receipt_raw, commitments = _load_acquisition_live(
        project=project, path=acquisition_receipt_path
    )
    p_program, lineage = _load_p_lineage(
        project=project,
        acquisition_receipt=receipt,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
    )
    old = _load_old_final_disposition(old_final_disposition_path)
    _a_grid, a_public, a_binding = _load_formation_bundle(
        project=project,
        private_cache_path=a_form_private_cache_path,
        public_receipt_path=a_form_public_receipt_path,
        expected_stage="A_form",
        acquisition_receipt=receipt,
        acquisition_raw=receipt_raw,
        commitments=commitments,
        p_lineage=lineage,
        old_disposition=old,
    )
    _f_grid, f_public, f_binding = _load_formation_bundle(
        project=project,
        private_cache_path=f_search_private_cache_path,
        public_receipt_path=f_search_public_receipt_path,
        expected_stage="F_search",
        acquisition_receipt=receipt,
        acquisition_raw=receipt_raw,
        commitments=commitments,
        p_lineage=lineage,
        old_disposition=old,
    )
    return (
        receipt, receipt_raw, commitments, p_program, lineage,
        a_public, f_public, a_binding, f_binding,
    )


def _anchor_execution_contract() -> dict[str, Any]:
    return {
        "physical_component_ids": list(ANCHOR_COMPONENT_IDS),
        "item_count": A_HOLD_ITEM_COUNT,
        "physical_work_unit_count": ANCHOR_WORK_UNIT_COUNT,
        "maximum_concurrency": ANCHOR_MAXIMUM_CONCURRENCY,
        "single_start_barrier_party_count": ANCHOR_WORK_UNIT_COUNT,
        "derived_arms": ["incumbent_portfolio", "challenger_portfolio"],
        "logical_retrieval_calls_per_compared_arm_item": 3,
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


def _aggregate_arm(
    *, arm_id: str, items: Sequence[l4.RecursiveItem], rankings: Sequence[Sequence[int]]
) -> dict[str, Any]:
    try:
        return v2._aggregate_arm(arm_id=arm_id, items=items, rankings=rankings)
    except v2.HotpotEvaluatorCoevolutionError as exc:
        raise HotpotEvaluatorPortfolioError("portfolio aggregate drifted") from exc


def _paired_arm(
    *,
    left: str,
    right: str,
    items: Sequence[l4.RecursiveItem],
    arms: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, Any]:
    deltas: list[int] = []
    for ordinal, item in enumerate(items):
        supports = frozenset(item.support_indices)
        deltas.append(
            len(supports.intersection(arms[left][ordinal]))
            - len(supports.intersection(arms[right][ordinal]))
        )
    return {
        "left_arm_id": left,
        "right_arm_id": right,
        "net_support_hit_count": sum(deltas),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
        "paired_test": exact_paired_sign_flip(deltas),
    }


def build_a_hold_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze A_hold without accepting or opening an A_hold block path."""

    project = Path(project_root).resolve(strict=True)
    (
        receipt, receipt_raw, commitments, p_program, lineage,
        a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    a_core = a_public["formation_core"]
    f_core = f_public["formation_core"]
    if a_core.get("measurable_contrast") is not True or f_core.get(
        "measurable_contrast"
    ) is not True:
        raise HotpotEvaluatorPortfolioError(
            "formation action coincides; A_hold must remain unopened"
        )
    a_inc = _program_pair(a_core, "incumbent")
    a_chal = _program_pair(a_core, "challenger")
    if p_program.program_hash in {
        *(program.program_hash for program in a_inc),
        *(program.program_hash for program in a_chal),
    }:
        raise HotpotEvaluatorPortfolioError("retained P entered anchor portfolio")
    source = _source_binding(receipt, receipt_raw, commitments[ANCHOR_BLOCK])
    body: dict[str, Any] = {
        "schema": ANCHOR_FREEZE_SCHEMA,
        "decision": "authorize_exact_A_hold_once_after_two_distinct_portfolios",
        "implementation": current_implementation_binding(project),
        "source_binding": source,
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "lineage_binding": lineage,
        "A_form_binding": a_binding,
        "F_search_binding": f_binding,
        "anchor_actions": {
            "retained_P_program_sha256": p_program.program_hash,
            "incumbent": a_core["incumbent"],
            "challenger": a_core["challenger"],
            "equal_three_call_portfolio_grammar": True,
            "behavior_distinct": True,
        },
        "prospective_search_actions": {
            "incumbent": f_core["incumbent"],
            "challenger": f_core["challenger"],
            "measurable_contrast": True,
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
        raise HotpotEvaluatorPortfolioError("A_hold freeze output already exists")
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
        raise HotpotEvaluatorPortfolioError("A_hold freeze drifted")
    _assert_public_safe(freeze)
    return freeze, _sha256_bytes(raw)


def _verify_anchor_inputs(
    *,
    project: Path,
    freeze: Mapping[str, Any],
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[
    dict[str, Any],
    BlockCommitment,
    TypedRetrievalProgram,
    tuple[TypedRetrievalProgram, ...],
    tuple[TypedRetrievalProgram, ...],
]:
    (
        receipt, receipt_raw, commitments, p_program, lineage,
        a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    commitment = commitments[ANCHOR_BLOCK]
    if (
        freeze.get("source_binding")
        != _source_binding(receipt, receipt_raw, commitment)
        or freeze.get("portfolio_design_binding")
        != receipt.get("portfolio_design_binding")
        or freeze.get("lineage_binding") != lineage
        or freeze.get("A_form_binding") != a_binding
        or freeze.get("F_search_binding") != f_binding
    ):
        raise HotpotEvaluatorPortfolioError("A_hold source or lineage drifted")
    a_core = a_public["formation_core"]
    f_core = f_public["formation_core"]
    if (
        a_core.get("measurable_contrast") is not True
        or f_core.get("measurable_contrast") is not True
        or freeze.get("anchor_actions")
        != {
            "retained_P_program_sha256": p_program.program_hash,
            "incumbent": a_core["incumbent"],
            "challenger": a_core["challenger"],
            "equal_three_call_portfolio_grammar": True,
            "behavior_distinct": True,
        }
        or freeze.get("prospective_search_actions")
        != {
            "incumbent": f_core["incumbent"],
            "challenger": f_core["challenger"],
            "measurable_contrast": True,
            "frozen_before_A_hold_open": True,
        }
    ):
        raise HotpotEvaluatorPortfolioError("A_hold portfolio action drifted")
    incumbent = _program_pair(a_core, "incumbent")
    challenger = _program_pair(a_core, "challenger")
    if p_program.program_hash in {
        *(program.program_hash for program in incumbent),
        *(program.program_hash for program in challenger),
    }:
        raise HotpotEvaluatorPortfolioError("retained P entered anchor portfolio")
    return receipt, commitment, p_program, incumbent, challenger


def execute_a_hold_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path,
    a_hold_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    """Execute exact A_hold once through the clean module CLI only."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise HotpotEvaluatorPortfolioError(
            "formal A_hold is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_anchor_freeze(
        pre_run_freeze_path, project=project
    )
    root = _new_root(execution_root, project)
    if freeze.get("execution_root_sha256") != _root_hash(root, project):
        raise HotpotEvaluatorPortfolioError("A_hold execution root drifted")
    if root.exists():
        raise HotpotEvaluatorPortfolioError("fresh A_hold root exists; replay forbidden")
    (
        _receipt, commitment, p_program, incumbent, challenger
    ) = _verify_anchor_inputs(
        project=project,
        freeze=freeze,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    os.mkdir(root, 0o700)
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
            project=project,
            path=a_hold_block_path,
            commitment=_l4_commitment(commitment),
        )
        if len(items) != A_HOLD_ITEM_COUNT:
            raise HotpotEvaluatorPortfolioError("A_hold item count drifted")
        work_units = tuple(
            (ordinal, component, item.retrieval_view())
            for ordinal, item in enumerate(items)
            for component in ANCHOR_COMPONENT_IDS
        )
        if len(work_units) != ANCHOR_WORK_UNIT_COUNT:
            raise HotpotEvaluatorPortfolioError("A_hold work-unit grid drifted")
        programs = {
            "incumbent_P": p_program,
            "incumbent_Q1": incumbent[0],
            "incumbent_Q2": incumbent[1],
            "challenger_P": p_program,
            "challenger_Q1": challenger[0],
            "challenger_Q2": challenger[1],
        }
        stage = "maximum_width_six_component_retrieval"

        def run_one(
            unit: tuple[int, str, l4.RetrievalItem]
        ) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            ordinal, component, item = unit
            with lock:
                attempted += 1
            try:
                barrier.wait(timeout=180)
            except threading.BrokenBarrierError as exc:
                raise HotpotEvaluatorPortfolioError(
                    "A_hold maximum-width barrier did not close"
                ) from exc
            ranking = l4._validate_direct_ranking(
                l4._ranking(programs[component], item), item
            )
            with lock:
                completed += 1
            return (ordinal, component), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=ANCHOR_MAXIMUM_CONCURRENCY,
            thread_name_prefix="hotpot-portfolio-anchor",
        ) as executor:
            terminal_rows = [
                future.result()
                for future in (executor.submit(run_one, unit) for unit in work_units)
            ]
        direct = dict(terminal_rows)
        if (
            attempted != ANCHOR_WORK_UNIT_COUNT
            or completed != ANCHOR_WORK_UNIT_COUNT
            or len(direct) != ANCHOR_WORK_UNIT_COUNT
        ):
            raise HotpotEvaluatorPortfolioError("A_hold terminal closure incomplete")
        stage = "offline_source_support_scoring_after_join"
        arms: dict[str, list[tuple[int, ...]]] = {
            "incumbent_portfolio": [],
            "challenger_portfolio": [],
        }
        for ordinal in range(len(items)):
            arms["incumbent_portfolio"].append(
                fuse_rankings(
                    direct[(ordinal, "incumbent_P")],
                    direct[(ordinal, "incumbent_Q1")],
                    direct[(ordinal, "incumbent_Q2")],
                )
            )
            arms["challenger_portfolio"].append(
                fuse_rankings(
                    direct[(ordinal, "challenger_P")],
                    direct[(ordinal, "challenger_Q1")],
                    direct[(ordinal, "challenger_Q2")],
                )
            )
        metrics = {
            arm: _aggregate_arm(arm_id=arm, items=items, rankings=rankings)
            for arm, rankings in arms.items()
        }
        paired = _paired_arm(
            left="challenger_portfolio",
            right="incumbent_portfolio",
            items=items,
            arms=arms,
        )
        transition = _archive_transition(
            anchor_manifest_sha256=commitment.item_commitment_set_sha256,
            incumbent_hits=metrics["incumbent_portfolio"]["support_hit_count"],
            challenger_hits=metrics["challenger_portfolio"]["support_hit_count"],
            support_total=metrics["incumbent_portfolio"]["support_total"],
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
                    "incumbent_portfolio_ranking": list(
                        arms["incumbent_portfolio"][ordinal]
                    ),
                    "challenger_portfolio_ranking": list(
                        arms["challenger_portfolio"][ordinal]
                    ),
                }
                for ordinal, item in enumerate(items)
            ],
            "raw_question_or_corpus_persisted": False,
        }
        private = {**private_body, "evidence_sha256": stable_hash(private_body)}
        private_path = root / ANCHOR_PRIVATE_EVIDENCE_FILENAME
        _write_json_exclusive(private_path, private, mode=0o600)
        private_raw = private_path.read_bytes()
        ranking_receipts = [
            {
                "ordinal_sha256": stable_hash({"ordinal": ordinal}),
                "component_id": component,
                "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)}),
            }
            for (ordinal, component), ranking in sorted(direct.items())
        ]
        report_body: dict[str, Any] = {
            "schema": ANCHOR_REPORT_SCHEMA,
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "source_binding": freeze["source_binding"],
            "portfolio_design_binding": freeze["portfolio_design_binding"],
            "A_form_binding": freeze["A_form_binding"],
            "F_search_binding": freeze["F_search_binding"],
            "anchor_actions": freeze["anchor_actions"],
            "prospective_search_actions": freeze["prospective_search_actions"],
            "arm_metrics": metrics,
            "challenger_minus_incumbent": paired,
            "evaluator_epoch_transition": transition,
            "private_evidence_binding": {
                "file_sha256": _sha256_bytes(private_raw),
                "evidence_sha256": private["evidence_sha256"],
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
        _write_json_exclusive(root / ANCHOR_REPORT_FILENAME, report, mode=0o600)
        persisted, _raw = _read_json(
            root / ANCHOR_REPORT_FILENAME, "persisted A_hold report"
        )
        if persisted != report:
            raise HotpotEvaluatorPortfolioError("persisted A_hold report drifted")
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
        raise HotpotEvaluatorPortfolioError(
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
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the evaluator transition from exact private anchor evidence."""

    freeze, freeze_file_hash = _load_anchor_freeze(
        pre_run_freeze_path, project=project
    )
    _receipt, commitment, _p, _inc, _chal = _verify_anchor_inputs(
        project=project,
        freeze=freeze,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    l4._assert_git_ignored_private_path(
        project=project, path=Path(private_evidence_path), require_file=True
    )
    private, private_raw = _read_json(private_evidence_path, "A_hold private evidence")
    private_body = dict(private)
    evidence_hash = _require_sha256(
        private_body.pop("evidence_sha256", None), "A_hold evidence"
    )
    rows = private.get("item_rows")
    if (
        private.get("schema") != f"{VERSION}_A_hold_private_evidence"
        or stable_hash(private_body) != evidence_hash
        or private.get("freeze_sha256") != freeze["freeze_sha256"]
        or private.get("source_binding") != freeze["source_binding"]
        or private.get("raw_question_or_corpus_persisted") is not False
        or not isinstance(rows, list)
        or len(rows) != A_HOLD_ITEM_COUNT
    ):
        raise HotpotEvaluatorPortfolioError("A_hold private evidence drifted")
    commitments: list[str] = []
    incumbent_hits: list[int] = []
    challenger_hits: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "challenger_portfolio_ranking",
            "incumbent_portfolio_ranking",
            "item_commitment_sha256",
            "support_indices",
        }:
            raise HotpotEvaluatorPortfolioError("A_hold private row drifted")
        commitments.append(
            _require_sha256(row.get("item_commitment_sha256"), "anchor item")
        )
        supports = row.get("support_indices")
        inc = row.get("incumbent_portfolio_ranking")
        chal = row.get("challenger_portfolio_ranking")
        if (
            not isinstance(supports, list)
            or len(supports) != 2
            or not isinstance(inc, list)
            or not isinstance(chal, list)
        ):
            raise HotpotEvaluatorPortfolioError("A_hold support evidence drifted")
        _validate_ranking(inc, "incumbent portfolio")
        _validate_ranking(chal, "challenger portfolio")
        support_set = frozenset(supports)
        incumbent_hits.append(len(support_set.intersection(inc)))
        challenger_hits.append(len(support_set.intersection(chal)))
    if stable_hash(commitments) != commitment.item_commitment_set_sha256:
        raise HotpotEvaluatorPortfolioError("A_hold item commitment set drifted")
    total = 2 * len(rows)
    metrics = {
        "incumbent_portfolio": {
            "arm_id": "incumbent_portfolio",
            "support_hit_count": sum(incumbent_hits),
            "support_total": total,
            "support_recall_at_5": sum(incumbent_hits) / total,
            "complete_item_count": sum(value == 2 for value in incumbent_hits),
            "covered_item_count": sum(value > 0 for value in incumbent_hits),
        },
        "challenger_portfolio": {
            "arm_id": "challenger_portfolio",
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
        "left_arm_id": "challenger_portfolio",
        "right_arm_id": "incumbent_portfolio",
        "net_support_hit_count": sum(deltas),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
        "paired_test": exact_paired_sign_flip(deltas),
    }
    transition = _archive_transition(
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
        report_body.pop("report_sha256", None), "A_hold report"
    )
    expected_private = {
        "file_sha256": _sha256_bytes(private_raw),
        "evidence_sha256": evidence_hash,
        "private_path_persisted_publicly": False,
        "item_level_evidence_persisted_publicly": False,
    }
    execution = report.get("execution")
    if (
        report.get("schema") != ANCHOR_REPORT_SCHEMA
        or report.get("valid") is not True
        or stable_hash(report_body) != report_hash
        or report.get("freeze_sha256") != freeze["freeze_sha256"]
        or report.get("freeze_file_sha256") != freeze_file_hash
        or report.get("source_binding") != freeze["source_binding"]
        or report.get("portfolio_design_binding")
        != freeze["portfolio_design_binding"]
        or report.get("A_form_binding") != freeze["A_form_binding"]
        or report.get("F_search_binding") != freeze["F_search_binding"]
        or report.get("anchor_actions") != freeze["anchor_actions"]
        or report.get("prospective_search_actions")
        != freeze["prospective_search_actions"]
        or report.get("private_evidence_binding") != expected_private
        or report.get("arm_metrics") != metrics
        or report.get("challenger_minus_incumbent") != paired
        or report.get("evaluator_epoch_transition") != transition
        or report.get("M_search_opened") is not False
        or report.get("M_search_authorized") is not transition["promoted"]
        or not isinstance(execution, Mapping)
        or execution.get("physical_component_ids") != list(ANCHOR_COMPONENT_IDS)
        or execution.get("physical_work_unit_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("retrieval_attempt_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("retrieval_terminal_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("all_terminals_joined_before_support_scoring") is not True
        or report.get("raw_content_persisted") is not False
    ):
        raise HotpotEvaluatorPortfolioError(
            "A_hold report differs from exact private evidence"
        )
    _assert_public_safe(report)
    return report, {
        "private_evidence_file_sha256": _sha256_bytes(private_raw),
        "private_evidence_sha256": evidence_hash,
        "public_report_file_sha256": _sha256_bytes(report_raw),
        "public_report_sha256": report_hash,
        "anchor_transition_sha256": transition["transition_sha256"],
        "challenger_promoted": transition["promoted"],
        "recomputed_from_exact_private_evidence": True,
        "private_path_persisted_publicly": False,
    }


def _search_execution_contract() -> dict[str, Any]:
    return {
        "physical_component_ids": list(SEARCH_COMPONENT_IDS),
        "item_count": M_SEARCH_ITEM_COUNT,
        "physical_work_unit_count": SEARCH_WORK_UNIT_COUNT,
        "maximum_concurrency": SEARCH_MAXIMUM_CONCURRENCY,
        "single_start_barrier_party_count": SEARCH_WORK_UNIT_COUNT,
        "derived_arms": [
            "canonical_RAW",
            "retained_P",
            "incumbent_portfolio",
            "active_portfolio",
            "official_HippoRAG",
        ],
        "primary_comparison": "active_portfolio_minus_incumbent_portfolio",
        "secondary_comparisons": [
            "active_portfolio_minus_official_HippoRAG",
            "active_portfolio_minus_canonical_RAW",
            "active_portfolio_minus_retained_P",
        ],
        "logical_retrieval_calls_per_primary_arm_item": 3,
        "all_terminals_join_before_runtime_postflight_and_offline_scoring": True,
        "model_calls_outside_official_arm": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def _promoted_transition_binding(transition: Mapping[str, Any]) -> dict[str, Any]:
    """Project an exact promoted anchor transition into the search freeze."""

    if (
        transition.get("promoted") is not True
        or transition.get("selective_invalidation_performed") is not True
        or transition.get("independent_source_record_retained") is not True
    ):
        raise HotpotEvaluatorPortfolioError(
            "challenger was not promoted; M_search must remain unopened"
        )
    for field in (
        "transition_sha256",
        "incumbent_epoch_id",
        "next_epoch_id",
        "next_evaluator_id",
    ):
        if not isinstance(transition.get(field), str) or not transition[field]:
            raise HotpotEvaluatorPortfolioError("anchor transition binding drifted")
    _require_sha256(transition["transition_sha256"], "anchor transition")
    return {
        "transition_sha256": transition["transition_sha256"],
        "incumbent_epoch_id": transition["incumbent_epoch_id"],
        "active_epoch_id": transition["next_epoch_id"],
        "active_evaluator_id": transition["next_evaluator_id"],
        "promoted": True,
    }


def _assert_search_transition_binding(
    frozen: object, anchor_transition: Mapping[str, Any]
) -> None:
    if frozen != _promoted_transition_binding(anchor_transition):
        raise HotpotEvaluatorPortfolioError("M_search evaluator transition drifted")


def build_m_search_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
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
    """Freeze M_search only after exact promoted anchor re-verification."""

    project = Path(project_root).resolve(strict=True)
    anchor_report, anchor_binding = load_and_reverify_a_hold(
        project=project,
        pre_run_freeze_path=a_hold_pre_run_freeze_path,
        private_evidence_path=a_hold_private_evidence_path,
        report_path=a_hold_report_path,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    transition = anchor_report["evaluator_epoch_transition"]
    transition_binding = _promoted_transition_binding(transition)
    (
        receipt, receipt_raw, commitments, p_program, lineage,
        a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    del a_public
    f_core = f_public["formation_core"]
    if f_core.get("measurable_contrast") is not True:
        raise HotpotEvaluatorPortfolioError("F_search contrast is unavailable")
    incumbent = _program_pair(f_core, "incumbent")
    active = _program_pair(f_core, "challenger")
    if p_program.program_hash in {
        *(program.program_hash for program in incumbent),
        *(program.program_hash for program in active),
    }:
        raise HotpotEvaluatorPortfolioError("retained P entered search portfolio")
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    paths = l4._runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = l4._prepare_runtime(project, paths)
    source = _source_binding(receipt, receipt_raw, commitments[SEARCH_BLOCK])
    body: dict[str, Any] = {
        "schema": SEARCH_FREEZE_SCHEMA,
        "decision": "authorize_exact_promoted_portfolio_M_search_once",
        "implementation": current_implementation_binding(project),
        "source_binding": source,
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "lineage_binding": lineage,
        "A_form_binding": a_binding,
        "F_search_binding": f_binding,
        "A_hold_binding": anchor_binding,
        "evaluator_epoch_transition": transition_binding,
        "search_actions": {
            "retained_P_program_sha256": p_program.program_hash,
            "incumbent": f_core["incumbent"],
            "active": f_core["challenger"],
            "equal_three_call_portfolio_grammar": True,
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
        raise HotpotEvaluatorPortfolioError("M_search freeze output already exists")
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
        raise HotpotEvaluatorPortfolioError("M_search freeze drifted")
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
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    old_final_disposition_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
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
    """Execute exact M_search once after a promoted portfolio transition."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise HotpotEvaluatorPortfolioError(
            "formal M_search is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_search_freeze(
        pre_run_freeze_path, project=project
    )
    root = _new_root(execution_root, project)
    if freeze.get("execution_root_sha256") != _root_hash(root, project):
        raise HotpotEvaluatorPortfolioError("M_search execution root drifted")
    if root.exists():
        raise HotpotEvaluatorPortfolioError("fresh M_search root exists; replay forbidden")
    anchor_report, anchor_binding = load_and_reverify_a_hold(
        project=project,
        pre_run_freeze_path=a_hold_pre_run_freeze_path,
        private_evidence_path=a_hold_private_evidence_path,
        report_path=a_hold_report_path,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    if (
        freeze.get("A_hold_binding") != anchor_binding
        or anchor_report["evaluator_epoch_transition"].get("promoted") is not True
    ):
        raise HotpotEvaluatorPortfolioError("promoted anchor binding drifted")
    _assert_search_transition_binding(
        freeze.get("evaluator_epoch_transition"),
        anchor_report["evaluator_epoch_transition"],
    )
    (
        receipt, receipt_raw, commitments, p_program, lineage,
        _a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        old_final_disposition_path=old_final_disposition_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    if (
        freeze.get("source_binding")
        != _source_binding(receipt, receipt_raw, commitments[SEARCH_BLOCK])
        or freeze.get("portfolio_design_binding")
        != receipt.get("portfolio_design_binding")
        or freeze.get("lineage_binding") != lineage
        or freeze.get("A_form_binding") != a_binding
        or freeze.get("F_search_binding") != f_binding
    ):
        raise HotpotEvaluatorPortfolioError("M_search source or lineage drifted")
    f_core = f_public["formation_core"]
    incumbent = _program_pair(f_core, "incumbent")
    active = _program_pair(f_core, "challenger")
    expected_actions = {
        "retained_P_program_sha256": p_program.program_hash,
        "incumbent": f_core["incumbent"],
        "active": f_core["challenger"],
        "equal_three_call_portfolio_grammar": True,
        "incumbent_and_active_behavior_distinct": True,
        "frozen_on_F_search_before_A_hold": True,
    }
    if freeze.get("search_actions") != expected_actions:
        raise HotpotEvaluatorPortfolioError("M_search selected actions drifted")
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    expected_capability = {
        "file_sha256": _sha256_bytes(capability_raw),
        "receipt_sha256": capability["receipt_sha256"],
        "bwrap_file_sha256": capability["bwrap_file_sha256"],
        "probe_contract_sha256": capability["probe_contract_sha256"],
        "fresh_probe_required_before_authorization": True,
    }
    if freeze.get("capability_binding") != expected_capability:
        raise HotpotEvaluatorPortfolioError("M_search capability drifted")
    live_probe = _probe_bubblewrap()
    if (
        live_probe.get("bwrap_file_sha256") != capability["bwrap_file_sha256"]
        or live_probe.get("probe_contract_sha256")
        != capability["probe_contract_sha256"]
        or live_probe.get("probe_returncode") != 0
    ):
        raise HotpotEvaluatorPortfolioError("fresh bwrap preflight drifted")
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
        raise HotpotEvaluatorPortfolioError("M_search runtime binding drifted")
    os.mkdir(root, 0o700)
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
            project=project,
            path=m_search_block_path,
            commitment=_l4_commitment(commitments[SEARCH_BLOCK]),
        )
        if len(items) != M_SEARCH_ITEM_COUNT:
            raise HotpotEvaluatorPortfolioError("M_search item count drifted")
        work_units = tuple(
            (ordinal, component, item.retrieval_view())
            for ordinal, item in enumerate(items)
            for component in SEARCH_COMPONENT_IDS
        )
        if len(work_units) != SEARCH_WORK_UNIT_COUNT:
            raise HotpotEvaluatorPortfolioError("M_search work grid drifted")
        local_programs = {
            "incumbent_P": p_program,
            "incumbent_Q1": incumbent[0],
            "incumbent_Q2": incumbent[1],
            "active_P": p_program,
            "active_Q1": active[0],
            "active_Q2": active[1],
        }
        stage = "maximum_width_eight_component_retrieval"

        def run_one(
            unit: tuple[int, str, l4.RetrievalItem]
        ) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            ordinal, component, item = unit
            with lock:
                attempted += 1
            try:
                barrier.wait(timeout=180)
            except threading.BrokenBarrierError as exc:
                raise HotpotEvaluatorPortfolioError(
                    "M_search maximum-width barrier did not close"
                ) from exc
            if component == "canonical_RAW":
                ranking = tuple(paragraph.idx for paragraph in item.corpus[:TOP_K])
            elif component == "official_HippoRAG":
                ranking = l4._official(
                    prepared, item, root / f"official_item_{ordinal:02d}"
                )
            else:
                ranking = l4._ranking(local_programs[component], item)
            ranking = l4._validate_direct_ranking(ranking, item)
            with lock:
                completed += 1
            return (ordinal, component), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=SEARCH_MAXIMUM_CONCURRENCY,
            thread_name_prefix="hotpot-portfolio-search",
        ) as executor:
            terminal_rows = [
                future.result()
                for future in (executor.submit(run_one, unit) for unit in work_units)
            ]
        direct = dict(terminal_rows)
        if (
            attempted != SEARCH_WORK_UNIT_COUNT
            or completed != SEARCH_WORK_UNIT_COUNT
            or len(direct) != SEARCH_WORK_UNIT_COUNT
        ):
            raise HotpotEvaluatorPortfolioError("M_search terminal closure incomplete")
        stage = "fresh_runtime_postflight_before_scoring"
        postflight = prepared.fresh_reverify()
        if postflight != safe_runtime:
            raise HotpotEvaluatorPortfolioError("M_search runtime postflight drifted")
        stage = "offline_source_support_scoring_after_join"
        arms: dict[str, list[tuple[int, ...]]] = {
            "canonical_RAW": [],
            "retained_P": [],
            "incumbent_portfolio": [],
            "active_portfolio": [],
            "official_HippoRAG": [],
        }
        for ordinal in range(len(items)):
            arms["canonical_RAW"].append(direct[(ordinal, "canonical_RAW")])
            arms["retained_P"].append(direct[(ordinal, "active_P")])
            arms["official_HippoRAG"].append(
                direct[(ordinal, "official_HippoRAG")]
            )
            arms["incumbent_portfolio"].append(
                fuse_rankings(
                    direct[(ordinal, "incumbent_P")],
                    direct[(ordinal, "incumbent_Q1")],
                    direct[(ordinal, "incumbent_Q2")],
                )
            )
            arms["active_portfolio"].append(
                fuse_rankings(
                    direct[(ordinal, "active_P")],
                    direct[(ordinal, "active_Q1")],
                    direct[(ordinal, "active_Q2")],
                )
            )
        metrics = {
            arm: _aggregate_arm(arm_id=arm, items=items, rankings=rankings)
            for arm, rankings in arms.items()
        }
        primary = _paired_arm(
            left="active_portfolio",
            right="incumbent_portfolio",
            items=items,
            arms=arms,
        )
        versus_official = _paired_arm(
            left="active_portfolio",
            right="official_HippoRAG",
            items=items,
            arms=arms,
        )
        versus_raw = _paired_arm(
            left="active_portfolio",
            right="canonical_RAW",
            items=items,
            arms=arms,
        )
        versus_p = _paired_arm(
            left="active_portfolio",
            right="retained_P",
            items=items,
            arms=arms,
        )
        ranking_receipts = [
            {
                "ordinal_sha256": stable_hash({"ordinal": ordinal}),
                "component_id": component,
                "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)}),
            }
            for (ordinal, component), ranking in sorted(direct.items())
        ]
        report_body: dict[str, Any] = {
            "schema": SEARCH_REPORT_SCHEMA,
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "source_binding": freeze["source_binding"],
            "portfolio_design_binding": freeze["portfolio_design_binding"],
            "A_hold_binding": freeze["A_hold_binding"],
            "evaluator_epoch_transition": freeze["evaluator_epoch_transition"],
            "search_actions": freeze["search_actions"],
            "arm_metrics": metrics,
            "primary_active_minus_incumbent": primary,
            "secondary_active_minus_official_HippoRAG": versus_official,
            "secondary_active_minus_RAW": versus_raw,
            "secondary_active_minus_retained_P": versus_p,
            "disposition": {
                "evaluator_transition_had_positive_search_utility": (
                    primary["net_support_hit_count"] > 0
                ),
                "search_measurement_used_for_evaluator_promotion": False,
                "followup_same_source_attempt_authorized": False,
                "statistical_superiority_claim": False,
                "family_out_claim": False,
                "compute_equivalence_claim_against_RAW_or_HippoRAG": False,
                "primary_evaluator_arms_equal_typed_retrieval_calls": True,
                "positive_net_on_fixed_cohort_active_minus_retained_P": (
                    versus_p["net_support_hit_count"] > 0
                ),
            },
            "execution": {
                "physical_component_ids": list(SEARCH_COMPONENT_IDS),
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
        _write_json_exclusive(root / SEARCH_REPORT_FILENAME, report, mode=0o600)
        persisted, _raw = _read_json(
            root / SEARCH_REPORT_FILENAME, "persisted M_search report"
        )
        if persisted != report:
            raise HotpotEvaluatorPortfolioError("persisted M_search report drifted")
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
        raise HotpotEvaluatorPortfolioError(
            "formal M_search failed and cannot be replayed"
        ) from exc


def formal_signatures_have_no_injection_surface() -> bool:
    forbidden = {
        "candidate_programs", "evidence", "operator", "operator_factory",
        "result", "results", "retriever", "runner", "callback",
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


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--p-formation-receipt", type=Path, required=True)
    parser.add_argument("--p-frozen-program", type=Path, required=True)
    parser.add_argument("--m1-freeze", type=Path, required=True)
    parser.add_argument("--m1-report", type=Path, required=True)
    parser.add_argument("--old-final-disposition", type=Path, required=True)


def _add_formations(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a-form-private-cache", type=Path, required=True)
    parser.add_argument("--a-form-public-receipt", type=Path, required=True)
    parser.add_argument("--f-search-private-cache", type=Path, required=True)
    parser.add_argument("--f-search-public-receipt", type=Path, required=True)


def _add_anchor(parser: argparse.ArgumentParser) -> None:
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
    form_f = commands.add_parser("form-search")
    freeze_a = commands.add_parser("freeze-a-hold")
    execute_a = commands.add_parser("execute-a-hold")
    freeze_m = commands.add_parser("freeze-m-search")
    execute_m = commands.add_parser("execute-m-search")
    for command in (form_a, form_f, freeze_a, execute_a, freeze_m, execute_m):
        _add_common(command)
    for command in (freeze_a, execute_a, freeze_m, execute_m):
        _add_formations(command)
        command.add_argument("--execution-root", type=Path, required=True)
    for prefix, command in (("a-form", form_a), ("f-search", form_f)):
        command.add_argument(f"--{prefix}-0-block", type=Path, required=True)
        command.add_argument(f"--{prefix}-1-block", type=Path, required=True)
        command.add_argument("--private-cache-output", type=Path, required=True)
        command.add_argument("--public-receipt-output", type=Path, required=True)
    freeze_a.add_argument("--authorization-hash", required=True)
    freeze_a.add_argument("--output", type=Path, required=True)
    execute_a.add_argument("--pre-run-freeze", type=Path, required=True)
    execute_a.add_argument("--a-hold-block", type=Path, required=True)
    for command in (freeze_m, execute_m):
        _add_anchor(command)
        _add_runtime(command)
    freeze_m.add_argument("--authorization-hash", required=True)
    freeze_m.add_argument("--output", type=Path, required=True)
    execute_m.add_argument("--pre-run-freeze", type=Path, required=True)
    execute_m.add_argument("--m-search-block", type=Path, required=True)
    args = parser.parse_args(argv)
    common = {
        "project_root": args.project_root,
        "acquisition_receipt_path": args.acquisition_receipt,
        "p_formation_receipt_path": args.p_formation_receipt,
        "p_frozen_program_path": args.p_frozen_program,
        "m1_freeze_path": args.m1_freeze,
        "m1_report_path": args.m1_report,
        "old_final_disposition_path": args.old_final_disposition,
    }
    if args.command == "form-a":
        form_a_form_stage(
            **common,
            a_form_0_block_path=args.a_form_0_block,
            a_form_1_block_path=args.a_form_1_block,
            private_cache_output_path=args.private_cache_output,
            public_receipt_output_path=args.public_receipt_output,
        )
        return 0
    if args.command == "form-search":
        form_f_search_stage(
            **common,
            f_search_0_block_path=args.f_search_0_block,
            f_search_1_block_path=args.f_search_1_block,
            private_cache_output_path=args.private_cache_output,
            public_receipt_output_path=args.public_receipt_output,
        )
        return 0
    formations = {
        "a_form_private_cache_path": args.a_form_private_cache,
        "a_form_public_receipt_path": args.a_form_public_receipt,
        "f_search_private_cache_path": args.f_search_private_cache,
        "f_search_public_receipt_path": args.f_search_public_receipt,
    }
    if args.command == "freeze-a-hold":
        build_a_hold_pre_run_freeze(
            **common, **formations,
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
                **common, **formations,
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
            **common, **formations, **anchor, **runtime,
            execution_root=args.execution_root,
            authorization_hash=args.authorization_hash,
            output_path=args.output,
        )
        return 0
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_m_search_formal(
            **common, **formations, **anchor, **runtime,
            pre_run_freeze_path=args.pre_run_freeze,
            m_search_block_path=args.m_search_block,
            execution_root=args.execution_root,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
