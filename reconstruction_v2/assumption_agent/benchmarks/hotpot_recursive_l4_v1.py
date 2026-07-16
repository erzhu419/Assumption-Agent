"""Fresh Hotpot generation-two formation and one-shot L4 retention study.

The retained program P is the exact MuSiQue-F1 program already promoted on
MuSiQue M1 and transferred without adaptation to the first Hotpot family-out
cohort.  A novel program Q is formed only on the new ``F_Q`` block.  Candidate
selection maximizes the frozen P+Q RRF retrieval on formation data while
excluding P-equivalent behavior; four-fold diagnostics are descriptive and
never become an extra gate.

The formal ``M_L4`` command opens no measurement row until a complete pre-run
freeze has been committed and one authorization has been consumed.  RAW, P,
Q, and official HippoRAG are four physical retrieval work units per item,
released by one maximum-width barrier.  P+Q is then derived deterministically
from the direct P/Q rankings after all terminals and a fresh runtime postflight.
Only local source-provided support labels are used for scoring.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterable, Mapping, Sequence

from ..models import stable_hash
from . import hotpot_recursive_acquisition_v1 as acquisition
from .hotpot_family_out_runner_v1 import (
    BWRAP_SHA256,
    _probe_bubblewrap,
    verify_capability_receipt,
)
from .l4_retention_protocol_v1 import deterministic_rrf
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .musique_official_core_comparison_v1 import _assert_git_ignored_private_path
from .musique_recursive_study_blocks_v1 import load_study_frozen_program
from .musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    TypedRetrievalProgram,
    enumerate_programs,
    retrieve as typed_retrieve,
)


VERSION = "hotpot_recursive_l4_v1"
Q_FORMATION_SCHEMA = f"{VERSION}_Q_formation_receipt"
Q_PROGRAM_SCHEMA = f"{VERSION}_Q_frozen_program"
FREEZE_SCHEMA = f"{VERSION}_pre_run_freeze"
REPORT_SCHEMA = f"{VERSION}_aggregate_report"
FAILURE_SCHEMA = f"{VERSION}_failure"
CONSUMPTION_SCHEMA = f"{VERSION}_authorization_consumption"
IMPLEMENTATION_SCHEMA = f"{VERSION}_implementation"
TOP_K = 5
FOLD_COUNT = 4
DIRECT_COMPONENT_IDS = (
    "canonical_RAW",
    "retained_P",
    "novel_Q",
    "official_HippoRAG",
)
DERIVED_ARM_IDS = (
    "canonical_RAW",
    "retained_P",
    "novel_Q",
    "P_plus_Q_RRF",
    "official_HippoRAG",
)
M_ITEM_COUNT = acquisition.BLOCK_COUNTS["M_L4"]
WORK_UNIT_COUNT = len(DIRECT_COMPONENT_IDS) * M_ITEM_COUNT
MAXIMUM_CONCURRENCY = WORK_UNIT_COUNT
CONSUMPTION_FILENAME = "l4.authorization.consumed.json"
REPORT_FILENAME = "l4.aggregate.report.json"
FAILURE_FILENAME = "l4.failure.json"
Q_FORMATION_CONSUMPTION_RELATIVE = (
    "artifacts/hotpot_recursive_acquisition_v1/"
    "q_formation.authorization.consumed.json"
)
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/hotpot_recursive_acquisition_v1.py",
    "assumption_agent/benchmarks/hotpot_recursive_l4_v1.py",
    "assumption_agent/benchmarks/hotpot_evaluator_coevolution_v2.py",
    "assumption_agent/benchmarks/hotpot_family_out_runner_v1.py",
    "assumption_agent/benchmarks/l4_retention_protocol_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/benchmarks/musique_m1_retrieval_runner_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class HotpotRecursiveL4Error(RuntimeError):
    """The fresh formation or formal retention contract drifted."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HotpotRecursiveL4Error("required file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HotpotRecursiveL4Error(f"{field_name} must be lowercase SHA-256")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], mode: int = 0o600) -> None:
    raw = json.dumps(
        payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise HotpotRecursiveL4Error(f"{field_name} contains a symlink")
    if not candidate.is_file():
        raise HotpotRecursiveL4Error(f"{field_name} is unavailable")
    raw = candidate.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotRecursiveL4Error(f"{field_name} is invalid") from exc
    if not isinstance(payload, dict):
        raise HotpotRecursiveL4Error(f"{field_name} must be one object")
    return payload, raw


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answer"',
        '"corpus"',
        '"item_id"',
        '"paragraph_text"',
        '"private_block_path"',
        '"question"',
        '"support_indices"',
        '"text"',
        "/home/",
        "/tmp/",
    )
    if any(value in raw for value in forbidden):
        raise HotpotRecursiveL4Error("public artifact contains private content or path")


def current_implementation_binding(project: Path) -> dict[str, Any]:
    project = project.resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotRecursiveL4Error(f"implementation file missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


@dataclass(frozen=True)
class RecursiveItem:
    item_id: str = field(repr=False)
    question: str = field(repr=False)
    corpus: tuple[RetrievalParagraph, ...] = field(repr=False)
    support_indices: tuple[int, ...] = field(repr=False)
    row_commitment_sha256: str

    @property
    def item_id_hash(self) -> str:
        return stable_hash({"item_id": self.item_id})

    def retrieval_view(self) -> "RetrievalItem":
        return RetrievalItem(question=self.question, corpus=self.corpus)


@dataclass(frozen=True)
class RetrievalItem:
    question: str = field(repr=False)
    corpus: tuple[RetrievalParagraph, ...] = field(repr=False)

    def hipporag_paragraphs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "idx": paragraph.idx,
                "title": paragraph.title,
                "paragraph_text": paragraph.text,
            }
            for paragraph in self.corpus
        )


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    count: int
    file_sha256: str
    item_commitment_set_sha256: str


def _load_acquisition(
    *, project: Path, path: str | Path
) -> tuple[dict[str, Any], bytes]:
    payload, _blocks = acquisition.load_acquisition_binding(path)
    raw = Path(path).read_bytes()
    if (
        payload.get("implementation")
        != acquisition.implementation_binding(project)
        or payload.get("retained_P_lineage")
        != acquisition.retained_p_lineage_binding(project)
    ):
        raise HotpotRecursiveL4Error(
            "acquisition implementation or retained-P preregistration drifted"
        )
    _assert_public_safe(payload)
    return payload, raw


def _retained_p_file_hashes(receipt: Mapping[str, Any]) -> dict[str, str]:
    lineage = receipt.get("retained_P_lineage")
    rows = lineage.get("files") if isinstance(lineage, Mapping) else None
    if not isinstance(rows, list):
        raise HotpotRecursiveL4Error("retained-P preregistration is unavailable")
    result = {
        str(row.get("role")): str(row.get("sha256"))
        for row in rows
        if isinstance(row, Mapping)
    }
    expected = {role for role, _relative in acquisition.RETAINED_P_LINEAGE_RELATIVE_FILES}
    if set(result) != expected:
        raise HotpotRecursiveL4Error("retained-P preregistration roles drifted")
    return result


def _assert_p_matches_preregistration(
    *, receipt: Mapping[str, Any], p_binding: Mapping[str, str]
) -> None:
    hashes = _retained_p_file_hashes(receipt)
    if (
        hashes["P_formation_receipt"]
        != p_binding.get("formation_receipt_file_sha256")
        or hashes["P_frozen_program"]
        != p_binding.get("frozen_program_file_sha256")
    ):
        raise HotpotRecursiveL4Error("retained P differs from preregistration")


def _assert_positive_lineage_matches_preregistration(
    *, receipt: Mapping[str, Any], lineage: Mapping[str, str]
) -> None:
    hashes = _retained_p_file_hashes(receipt)
    if (
        hashes["M1_pre_run_freeze"]
        != lineage.get("m1_freeze_file_sha256")
        or hashes["M1_positive_promotion_report"]
        != lineage.get("m1_report_file_sha256")
    ):
        raise HotpotRecursiveL4Error(
            "retained-P positive lineage differs from preregistration"
        )


def _commitment(receipt: Mapping[str, Any], block: str) -> BlockCommitment:
    if block not in acquisition.BLOCK_COUNTS:
        raise HotpotRecursiveL4Error("unknown recursive block")
    rows = receipt.get("commitments", {}).get("block_files", [])
    row = next(
        (value for value in rows if isinstance(value, Mapping) and value.get("block") == block),
        None,
    )
    if not isinstance(row, Mapping):
        raise HotpotRecursiveL4Error("recursive block commitment is unavailable")
    return BlockCommitment(
        block=block,
        count=row["count"],
        file_sha256=row["file_sha256"],
        item_commitment_set_sha256=row["item_commitment_set_sha256"],
    )


def _load_private_block(
    *,
    project: Path,
    path: str | Path,
    commitment: BlockCommitment,
) -> tuple[RecursiveItem, ...]:
    private = _assert_git_ignored_private_path(
        project=project, path=Path(path), require_file=True
    )
    if _sha256_file(private) != commitment.file_sha256:
        raise HotpotRecursiveL4Error("private recursive block file drifted")
    items: list[RecursiveItem] = []
    row_commitments: list[str] = []
    for raw_line in private.read_bytes().splitlines():
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HotpotRecursiveL4Error("private recursive row is invalid") from exc
        expected = set(acquisition.PRIVATE_BLOCK_ROW_KEYS)
        if (
            not isinstance(row, dict)
            or set(row) != expected
            or row.get("block") != commitment.block
            or _canonical_bytes(row) != raw_line
        ):
            raise HotpotRecursiveL4Error("private recursive row schema drifted")
        corpus_raw = row.get("corpus")
        supports = row.get("support_indices")
        if not isinstance(corpus_raw, list) or not isinstance(supports, list):
            raise HotpotRecursiveL4Error("private recursive labels are malformed")
        corpus: list[RetrievalParagraph] = []
        observed: list[int] = []
        for ordinal, paragraph in enumerate(corpus_raw):
            if (
                not isinstance(paragraph, Mapping)
                or set(paragraph) != {"idx", "is_supporting", "text", "title"}
                or paragraph.get("idx") != ordinal
                or type(paragraph.get("is_supporting")) is not bool
                or not isinstance(paragraph.get("title"), str)
                or not isinstance(paragraph.get("text"), str)
            ):
                raise HotpotRecursiveL4Error("private recursive corpus drifted")
            corpus.append(
                RetrievalParagraph(ordinal, paragraph["title"], paragraph["text"])
            )
            if paragraph["is_supporting"]:
                observed.append(ordinal)
        if (
            len(corpus) < TOP_K
            or supports != observed
            or len(observed) != 2
            or not isinstance(row.get("item_id"), str)
            or not isinstance(row.get("question"), str)
        ):
            raise HotpotRecursiveL4Error("private recursive item contract drifted")
        row_hash = stable_hash(row)
        row_commitments.append(row_hash)
        items.append(
            RecursiveItem(
                item_id=row["item_id"],
                question=row["question"],
                corpus=tuple(corpus),
                support_indices=tuple(observed),
                row_commitment_sha256=row_hash,
            )
        )
    if (
        len(items) != commitment.count
        or len({item.item_id_hash for item in items}) != len(items)
        or stable_hash(row_commitments) != commitment.item_commitment_set_sha256
    ):
        raise HotpotRecursiveL4Error("private recursive block closure drifted")
    return tuple(items)


def _load_p(
    *, project: Path, formation_path: str | Path, program_path: str | Path
) -> tuple[TypedRetrievalProgram, dict[str, str]]:
    formation = Path(formation_path).resolve(strict=True)
    frozen = Path(program_path).resolve(strict=True)
    program, receipt, envelope = load_study_frozen_program(
        frozen_program_path=frozen,
        formation_receipt_path=formation,
        verify_live=True,
        implementation_root=project,
    )
    expected = stable_hash({"block": "F1"})
    if receipt.get("formation_block_id_hash") != expected:
        raise HotpotRecursiveL4Error("retained P was not formed on exact F1")
    return program, {
        "formation_receipt_file_sha256": _sha256_file(formation),
        "formation_receipt_hash": receipt["receipt_hash"],
        "frozen_program_file_sha256": _sha256_file(frozen),
        "frozen_program_envelope_hash": envelope["envelope_hash"],
        "program_hash": program.program_hash,
        "formed_on_block_id_hash": expected,
    }


def _ranking(program: TypedRetrievalProgram, item: RecursiveItem | RetrievalItem) -> tuple[int, ...]:
    try:
        value = tuple(typed_retrieve(program, item.question, item.corpus))
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise HotpotRecursiveL4Error("typed retrieval failed") from exc
    if (
        len(value) != TOP_K
        or len(set(value)) != TOP_K
        or any(type(index) is not int or not 0 <= index < len(item.corpus) for index in value)
    ):
        raise HotpotRecursiveL4Error("typed retrieval violates top-five contract")
    return value


def _doc_ids(ranking: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"document_{index:03d}" for index in ranking)


def _indices(ranking: Sequence[str]) -> tuple[int, ...]:
    return tuple(int(value.removeprefix("document_")) for value in ranking)


@dataclass(frozen=True)
class CandidateScore:
    program: TypedRetrievalProgram
    behavior_sha256: str
    invalid_count: int
    combined_hits: int
    q_direct_hits: int
    novelty_added: int
    retained_added: int
    p_support_forgotten: int
    retained_displaced: int

    @property
    def rank(self) -> tuple[Any, ...]:
        return (
            self.invalid_count,
            -self.combined_hits,
            self.p_support_forgotten,
            -self.novelty_added,
            -self.retained_added,
            self.retained_displaced,
            -self.q_direct_hits,
            self.program.program_length,
            self.program.program_hash,
        )


def _assess_candidate(
    *,
    program: TypedRetrievalProgram,
    p_program: TypedRetrievalProgram,
    items: Sequence[RecursiveItem],
) -> CandidateScore:
    combined_hits = q_hits = novelty = retained = forgotten = displaced = invalid = 0
    behavior: list[dict[str, Any]] = []
    for item in items:
        raw = tuple(paragraph.idx for paragraph in item.corpus[:TOP_K])
        try:
            p = _ranking(p_program, item)
            q = _ranking(program, item)
            p_arm = p
            q_arm = q
            combined = _indices(deterministic_rrf((_doc_ids(p), _doc_ids(q))))
        except HotpotRecursiveL4Error:
            invalid += 1
            behavior.append({"invalid": True})
            continue
        supports = frozenset(item.support_indices)
        p_set = supports.intersection(p_arm)
        q_set = supports.intersection(q_arm)
        combined_set = supports.intersection(combined)
        combined_hits += len(combined_set)
        q_hits += len(supports.intersection(q))
        novelty += len(combined_set - p_set)
        retained += len(combined_set - q_set)
        forgotten += len(p_set - combined_set)
        displaced += len(q_set - combined_set)
        behavior.append({"q_ranking": list(q), "combined_ranking": list(combined)})
    return CandidateScore(
        program=program,
        behavior_sha256=stable_hash(behavior),
        invalid_count=invalid,
        combined_hits=combined_hits,
        q_direct_hits=q_hits,
        novelty_added=novelty,
        retained_added=retained,
        p_support_forgotten=forgotten,
        retained_displaced=displaced,
    )


def _select_q(
    *, p_program: TypedRetrievalProgram, items: Sequence[RecursiveItem]
) -> tuple[CandidateScore, tuple[CandidateScore, ...], str]:
    p_behavior = stable_hash(
        [{"q_ranking": list(_ranking(p_program, item))} for item in items]
    )
    candidates: list[CandidateScore] = []
    for program in enumerate_programs():
        if program.program_hash == p_program.program_hash:
            continue
        score = _assess_candidate(program=program, p_program=p_program, items=items)
        direct_behavior = stable_hash(
            [{"q_ranking": list(_ranking(program, item))} for item in items]
        )
        if direct_behavior == p_behavior:
            continue
        candidates.append(score)
    if not candidates:
        raise HotpotRecursiveL4Error("no behavior-novel Q candidate exists")
    candidates.sort(key=lambda row: row.rank)
    return candidates[0], tuple(candidates), p_behavior


def form_q(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    f_q_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project = Path(project_root).resolve(strict=True)
    receipt, receipt_raw = _load_acquisition(
        project=project, path=acquisition_receipt_path
    )
    commitment = _commitment(receipt, "F_Q")
    p_program, p_binding = _load_p(
        project=project,
        formation_path=p_formation_receipt_path,
        program_path=p_frozen_program_path,
    )
    _assert_p_matches_preregistration(receipt=receipt, p_binding=p_binding)
    implementation = current_implementation_binding(project)
    output = Path(output_dir).absolute()
    marker = _assert_git_ignored_private_path(
        project=project,
        path=project / Q_FORMATION_CONSUMPTION_RELATIVE,
        require_file=None,
    )
    if output.exists():
        raise HotpotRecursiveL4Error("Q formation output already exists")
    if marker.exists():
        raise HotpotRecursiveL4Error("Q formation was already consumed")
    marker_body: dict[str, Any] = {
        "schema": f"{VERSION}_Q_formation_consumption",
        "acquisition_sha256": receipt["acquisition_sha256"],
        "formation_block_id_hash": stable_hash({"block": "F_Q"}),
        "formation_block_file_sha256": commitment.file_sha256,
        "formation_item_commitment_set_sha256": (
            commitment.item_commitment_set_sha256
        ),
        "retained_P_program_hash": p_program.program_hash,
        "implementation_set_sha256": implementation["set_sha256"],
        "output_path_sha256": stable_hash(
            {"absolute_Q_formation_output": str(output)}
        ),
        "formation_rows_opened_before_consumption": 0,
        "retry_replay_resample_authorized": False,
    }
    marker_payload = {
        **marker_body,
        "consumption_sha256": stable_hash(marker_body),
    }
    _write_json_exclusive(marker, marker_payload)
    marker_raw = marker.read_bytes()
    items = _load_private_block(
        project=project, path=f_q_block_path, commitment=commitment
    )
    winner, candidates, p_behavior = _select_q(
        p_program=p_program, items=items
    )
    folds: list[dict[str, Any]] = []
    fold_programs: list[str] = []
    fold_behaviors: list[str] = []
    for fold in range(FOLD_COUNT):
        held_positions = tuple(
            position for position in range(len(items)) if position % FOLD_COUNT == fold
        )
        held_set = set(held_positions)
        fit = tuple(
            item for position, item in enumerate(items) if position not in held_set
        )
        held = tuple(items[position] for position in held_positions)
        fold_winner, fit_candidates, _ = _select_q(
            p_program=p_program, items=fit
        )
        held_score = _assess_candidate(
            program=fold_winner.program,
            p_program=p_program,
            items=held,
        )
        full_score = _assess_candidate(
            program=fold_winner.program,
            p_program=p_program,
            items=items,
        )
        fold_programs.append(fold_winner.program.program_hash)
        fold_behaviors.append(full_score.behavior_sha256)
        folds.append(
            {
                "fold": fold,
                "fit_item_count": len(fit),
                "held_item_count": len(held),
                "fit_candidate_count": len(fit_candidates),
                "selected_program_hash": fold_winner.program.program_hash,
                "selected_full_behavior_sha256": full_score.behavior_sha256,
                "held_combined_support_hits": held_score.combined_hits,
                "held_q_direct_support_hits": held_score.q_direct_hits,
                "held_novelty_added": held_score.novelty_added,
                "held_retained_added": held_score.retained_added,
                "held_P_support_forgotten": held_score.p_support_forgotten,
                "held_retained_displaced": held_score.retained_displaced,
                "held_invalid_count": held_score.invalid_count,
            }
        )
    formation_body: dict[str, Any] = {
        "schema": Q_FORMATION_SCHEMA,
        "status": "Q_formed_offline_on_exact_F_Q",
        "implementation": implementation,
        "source_binding": {
            "acquisition_file_sha256": _sha256_bytes(receipt_raw),
            "acquisition_sha256": receipt["acquisition_sha256"],
            "formation_block_id_hash": stable_hash({"block": "F_Q"}),
            "formation_block_file_sha256": commitment.file_sha256,
            "formation_item_commitment_set_sha256": (
                commitment.item_commitment_set_sha256
            ),
            "formation_item_count": commitment.count,
        },
        "retained_P_binding": p_binding,
        "prospective_ordering": {
            "formation_consumption_file_sha256": _sha256_bytes(marker_raw),
            "formation_consumption_sha256": marker_payload[
                "consumption_sha256"
            ],
            "formation_rows_opened_before_consumption": 0,
            "formation_consumed_before_F_Q_open": True,
            "retry_replay_resample_authorized": False,
        },
        "search": {
            "candidate_grammar": "frozen_84_program_typed_lexical_DSL_v1",
            "enumerated_candidate_count": len(tuple(enumerate_programs())),
            "eligible_behavior_novel_candidate_count": len(candidates),
            "P_equivalent_program_or_F_Q_behavior_excluded": True,
            "selection_objective": (
                "min_invalid_then_max_P_Q_RRF_support_then_min_P_forgetting_"
                "then_max_novelty_then_max_retained_then_min_Q_displaced_"
                "then_max_Q_direct_then_min_complexity_then_hash_v2"
            ),
            "RRF_constant": 60,
            "top_k": TOP_K,
            "performance_gate": False,
        },
        "selection": {
            "selected_program_hash": winner.program.program_hash,
            "selected_behavior_sha256": winner.behavior_sha256,
            "P_direct_behavior_sha256": p_behavior,
            "combined_support_hits": winner.combined_hits,
            "support_total": sum(len(item.support_indices) for item in items),
            "q_direct_support_hits": winner.q_direct_hits,
            "novelty_added": winner.novelty_added,
            "retained_added": winner.retained_added,
            "P_support_forgotten": winner.p_support_forgotten,
            "retained_displaced": winner.retained_displaced,
            "invalid_count": winner.invalid_count,
            "program_length": winner.program.program_length,
        },
        "crossfit": {
            "fold_assignment": "private_HMAC_selected_block_position_modulo_4_v1",
            "fold_count": FOLD_COUNT,
            "folds": folds,
            "selected_program_stable": len(set(fold_programs)) == 1
            and fold_programs[0] == winner.program.program_hash,
            "selected_behavior_stable": len(set(fold_behaviors)) == 1
            and fold_behaviors[0] == winner.behavior_sha256,
            "used_as_gate": False,
        },
        "offline_contract": {
            "model_calls": 0,
            "external_network_calls": 0,
            "study_level_answer_generator_calls": 0,
            "online_evaluator_calls": 0,
            "measurement_blocks_accessed": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "raw_content_persisted": False,
    }
    formation = {
        **formation_body,
        "formation_sha256": stable_hash(formation_body),
    }
    program_body: dict[str, Any] = {
        "schema": Q_PROGRAM_SCHEMA,
        "implementation": implementation,
        "program": winner.program.to_dict(),
        "program_hash": winner.program.program_hash,
        "formation_sha256": formation["formation_sha256"],
        "formation_block_id_hash": stable_hash({"block": "F_Q"}),
        "retained_P_program_hash": p_program.program_hash,
        "raw_content_persisted": False,
    }
    frozen = {**program_body, "envelope_sha256": stable_hash(program_body)}
    _assert_public_safe(formation)
    _assert_public_safe(frozen)
    output.mkdir(parents=True, mode=0o755)
    _write_json_exclusive(output / "formation.receipt.json", formation, mode=0o644)
    _write_json_exclusive(output / "frozen_program.json", frozen, mode=0o644)
    return formation, frozen


def load_q(
    *,
    project: Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
) -> tuple[TypedRetrievalProgram, dict[str, str]]:
    acquisition_receipt, acquisition_raw = _load_acquisition(
        project=project, path=acquisition_receipt_path
    )
    formation_commitment = _commitment(acquisition_receipt, "F_Q")
    _p_program, expected_p_binding = _load_p(
        project=project,
        formation_path=p_formation_receipt_path,
        program_path=p_frozen_program_path,
    )
    _assert_p_matches_preregistration(
        receipt=acquisition_receipt, p_binding=expected_p_binding
    )
    formation_path = Path(os.path.abspath(os.fspath(formation_receipt_path)))
    frozen_path = Path(os.path.abspath(os.fspath(frozen_program_path)))
    if (
        formation_path.name != "formation.receipt.json"
        or frozen_path.name != "frozen_program.json"
        or formation_path.parent != frozen_path.parent
    ):
        raise HotpotRecursiveL4Error("Q artifact layout drifted")
    marker_path = _assert_git_ignored_private_path(
        project=project,
        path=project / Q_FORMATION_CONSUMPTION_RELATIVE,
        require_file=True,
    )
    marker, marker_raw = _read_json(marker_path, "Q formation consumption")
    marker_body = dict(marker)
    marker_hash = _require_sha256(
        marker_body.pop("consumption_sha256", None), "Q formation consumption"
    )
    expected_marker_body = {
        "schema": f"{VERSION}_Q_formation_consumption",
        "acquisition_sha256": acquisition_receipt["acquisition_sha256"],
        "formation_block_id_hash": stable_hash({"block": "F_Q"}),
        "formation_block_file_sha256": formation_commitment.file_sha256,
        "formation_item_commitment_set_sha256": (
            formation_commitment.item_commitment_set_sha256
        ),
        "retained_P_program_hash": expected_p_binding["program_hash"],
        "implementation_set_sha256": current_implementation_binding(project)[
            "set_sha256"
        ],
        "output_path_sha256": stable_hash(
            {"absolute_Q_formation_output": str(formation_path.parent)}
        ),
        "formation_rows_opened_before_consumption": 0,
        "retry_replay_resample_authorized": False,
    }
    if marker_body != expected_marker_body or stable_hash(marker_body) != marker_hash:
        raise HotpotRecursiveL4Error("Q formation consumption drifted")
    formation, formation_raw = _read_json(
        formation_path, "Q formation receipt"
    )
    body = dict(formation)
    declared = _require_sha256(body.pop("formation_sha256", None), "Q formation hash")
    frozen, frozen_raw = _read_json(frozen_path, "Q frozen program")
    frozen_body = dict(frozen)
    envelope_hash = _require_sha256(
        frozen_body.pop("envelope_sha256", None), "Q program envelope"
    )
    if (
        formation.get("schema") != Q_FORMATION_SCHEMA
        or stable_hash(body) != declared
        or formation.get("status") != "Q_formed_offline_on_exact_F_Q"
        or frozen.get("schema") != Q_PROGRAM_SCHEMA
        or stable_hash(frozen_body) != envelope_hash
        or frozen.get("formation_sha256") != declared
        or frozen.get("raw_content_persisted") is not False
        or formation.get("raw_content_persisted") is not False
        or frozen.get("implementation") != current_implementation_binding(project)
        or formation.get("implementation") != current_implementation_binding(project)
        or formation.get("source_binding")
        != {
            "acquisition_file_sha256": _sha256_bytes(acquisition_raw),
            "acquisition_sha256": acquisition_receipt["acquisition_sha256"],
            "formation_block_id_hash": stable_hash({"block": "F_Q"}),
            "formation_block_file_sha256": formation_commitment.file_sha256,
            "formation_item_commitment_set_sha256": (
                formation_commitment.item_commitment_set_sha256
            ),
            "formation_item_count": formation_commitment.count,
        }
        or formation.get("retained_P_binding") != expected_p_binding
        or frozen.get("formation_block_id_hash")
        != stable_hash({"block": "F_Q"})
        or frozen.get("retained_P_program_hash")
        != expected_p_binding["program_hash"]
        or formation.get("prospective_ordering")
        != {
            "formation_consumption_file_sha256": _sha256_bytes(marker_raw),
            "formation_consumption_sha256": marker_hash,
            "formation_rows_opened_before_consumption": 0,
            "formation_consumed_before_F_Q_open": True,
            "retry_replay_resample_authorized": False,
        }
    ):
        raise HotpotRecursiveL4Error("Q formation/program drifted")
    try:
        program = TypedRetrievalProgram.from_dict(dict(frozen["program"]))
    except (KeyError, TypeError) as exc:
        raise HotpotRecursiveL4Error("Q program payload is invalid") from exc
    if (
        program.type_issues()
        or frozen.get("program_hash") != program.program_hash
        or frozen.get("program") != program.to_dict()
    ):
        raise HotpotRecursiveL4Error("Q program type/hash drifted")
    if formation.get("selection", {}).get("selected_program_hash") != program.program_hash:
        raise HotpotRecursiveL4Error("Q selection differs from frozen program")
    _assert_public_safe(formation)
    _assert_public_safe(frozen)
    return program, {
        "formation_receipt_file_sha256": _sha256_bytes(formation_raw),
        "formation_sha256": declared,
        "frozen_program_file_sha256": _sha256_bytes(frozen_raw),
        "frozen_program_envelope_sha256": envelope_hash,
        "program_hash": program.program_hash,
        "formed_on_block_id_hash": stable_hash({"block": "F_Q"}),
    }


def _load_positive_p_lineage(
    *,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    p_binding: Mapping[str, str],
) -> dict[str, str]:
    freeze, freeze_raw = _read_json(m1_freeze_path, "M1 pre-run freeze")
    report, report_raw = _read_json(m1_report_path, "M1 promotion report")
    freeze_body = dict(freeze)
    freeze_hash = _require_sha256(freeze_body.pop("freeze_hash", None), "M1 freeze")
    report_body = dict(report)
    report_hash = _require_sha256(report_body.pop("report_hash", None), "M1 report")
    measurement = report.get("measurement")
    metrics = measurement.get("arm_metrics") if isinstance(measurement, Mapping) else None
    disposition = (
        measurement.get("promotion_disposition")
        if isinstance(measurement, Mapping)
        else None
    )
    if (
        stable_hash(freeze_body) != freeze_hash
        or stable_hash(report_body) != report_hash
        or report.get("valid") is not True
        or report.get("freeze_hash") != freeze_hash
        or not isinstance(metrics, Mapping)
        or metrics.get("frozen_P", {}).get("support_hit_count", -1)
        <= metrics.get("canonical_RAW", {}).get("support_hit_count", -1)
        or not isinstance(disposition, Mapping)
        or disposition.get("disposition") != "promote_P_to_retained_generation_one"
        or freeze.get("p_operator_binding", {}).get("program_hash")
        != p_binding.get("program_hash")
    ):
        raise HotpotRecursiveL4Error("positive retained-P lineage drifted")
    return {
        "m1_freeze_file_sha256": _sha256_bytes(freeze_raw),
        "m1_freeze_sha256": freeze_hash,
        "m1_report_file_sha256": _sha256_bytes(report_raw),
        "m1_report_sha256": report_hash,
        "disposition": "promote_P_to_retained_generation_one",
        "program_hash": p_binding["program_hash"],
    }


@dataclass(frozen=True)
class RuntimePaths:
    runtime_python: Path = field(repr=False)
    local_llm_model: Path = field(repr=False)
    local_embedding_model: Path = field(repr=False)
    base_binding_receipt: Path = field(repr=False)
    attestation_receipt: Path = field(repr=False)


def _runtime_paths(
    *,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> RuntimePaths:
    value = RuntimePaths(
        runtime_python=Path(runtime_python).absolute(),
        local_llm_model=Path(local_llm_model).resolve(strict=True),
        local_embedding_model=Path(local_embedding_model).resolve(strict=True),
        base_binding_receipt=Path(base_binding_receipt_path).resolve(strict=True),
        attestation_receipt=Path(attestation_receipt_path).resolve(strict=True),
    )
    if not value.runtime_python.is_file():
        raise HotpotRecursiveL4Error("runtime Python is unavailable")
    return value


def _prepare_runtime(project: Path, paths: RuntimePaths) -> PreparedFormalRuntimeV2:
    return prepare_formal_runtime_v2(
        project_root=project,
        attestation_receipt_path=paths.attestation_receipt,
        base_binding_receipt_path=paths.base_binding_receipt,
        runtime_python=paths.runtime_python,
        local_llm_model=paths.local_llm_model,
        local_embedding_model=paths.local_embedding_model,
    )


def _runtime_binding(prepared: PreparedFormalRuntimeV2, paths: RuntimePaths) -> dict[str, Any]:
    return {
        "prepared_safe_binding": prepared.safe_binding,
        "base_binding_file_sha256": _sha256_file(paths.base_binding_receipt),
        "attestation_file_sha256": _sha256_file(paths.attestation_receipt),
        "fresh_preflight_before_authorization": True,
        "fresh_postflight_before_scoring": True,
    }


def _new_root(path: str | Path, project: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(root.parents), root):
        if component.is_symlink():
            raise HotpotRecursiveL4Error("execution root contains a symlink")
    if not root.parent.is_dir():
        raise HotpotRecursiveL4Error("execution root parent is unavailable")
    try:
        _assert_git_ignored_private_path(project=project, path=root, require_file=False)
    except Exception as exc:
        raise HotpotRecursiveL4Error("execution root must be ignored and private") from exc
    return root


def _root_hash(path: str | Path, project: Path) -> str:
    return stable_hash({"absolute_execution_root": str(_new_root(path, project))})


def build_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    m1_pre_run_freeze_path: str | Path,
    m1_promotion_report_path: str | Path,
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
    project = Path(project_root).resolve(strict=True)
    receipt, receipt_raw = _load_acquisition(
        project=project, path=acquisition_receipt_path
    )
    measurement = _commitment(receipt, "M_L4")
    p_program, p_binding = _load_p(
        project=project,
        formation_path=p_formation_receipt_path,
        program_path=p_frozen_program_path,
    )
    _assert_p_matches_preregistration(receipt=receipt, p_binding=p_binding)
    q_program, q_binding = load_q(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        formation_receipt_path=q_formation_receipt_path,
        frozen_program_path=q_frozen_program_path,
    )
    if p_program.program_hash == q_program.program_hash:
        raise HotpotRecursiveL4Error("P and Q programs are not distinct")
    lineage = _load_positive_p_lineage(
        m1_freeze_path=m1_pre_run_freeze_path,
        m1_report_path=m1_promotion_report_path,
        p_binding=p_binding,
    )
    _assert_positive_lineage_matches_preregistration(
        receipt=receipt, lineage=lineage
    )
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    paths = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = _prepare_runtime(project, paths)
    body: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "decision": "authorize_exact_fresh_Hotpot_L4_once",
        "implementation": current_implementation_binding(project),
        "authorization_hash": _require_sha256(
            authorization_hash, "execution authorization"
        ),
        "execution_root_hash": _root_hash(execution_root, project),
        "source_binding": {
            "acquisition_file_sha256": _sha256_bytes(receipt_raw),
            "acquisition_sha256": receipt["acquisition_sha256"],
            "measurement_block_id_hash": stable_hash({"block": "M_L4"}),
            "measurement_block_file_sha256": measurement.file_sha256,
            "measurement_item_commitment_set_sha256": (
                measurement.item_commitment_set_sha256
            ),
            "measurement_item_count": measurement.count,
        },
        "retained_P_binding": p_binding,
        "novel_Q_binding": q_binding,
        "positive_P_lineage": lineage,
        "capability_binding": {
            "file_sha256": _sha256_bytes(capability_raw),
            "receipt_sha256": capability["receipt_sha256"],
            "bwrap_file_sha256": capability["bwrap_file_sha256"],
            "probe_contract_sha256": capability["probe_contract_sha256"],
            "fresh_probe_required_before_authorization": True,
        },
        "runtime_binding": _runtime_binding(prepared, paths),
        "execution_contract": {
            "direct_components": list(DIRECT_COMPONENT_IDS),
            "derived_arms": list(DERIVED_ARM_IDS),
            "item_count": M_ITEM_COUNT,
            "top_k": TOP_K,
            "physical_work_unit_count": WORK_UNIT_COUNT,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "single_start_barrier": True,
            "P_plus_Q_is_deterministic_RRF_not_additional_call": True,
            "RRF_constant": 60,
            "all_terminals_joined_before_scoring": True,
            "fresh_runtime_postflight_before_scoring": True,
            "primary_estimand": "Y(P_plus_Q_RRF)-Y(novel_Q)",
            "secondary_official_comparison": (
                "Y(P_plus_Q_RRF)-Y(official_HippoRAG)"
            ),
            "study_level_answer_generator_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "performance_gate": False,
        },
        "ordering": {
            "measurement_rows_read_while_freezing": 0,
            "support_labels_read_while_freezing": 0,
            "freeze_complete_before_M_L4_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_sha256": stable_hash(body)}
    _assert_public_safe(freeze)
    output = Path(output_path).absolute()
    if output.exists():
        raise HotpotRecursiveL4Error("L4 freeze output already exists")
    _write_json_exclusive(output, freeze, mode=0o644)
    return freeze


def _load_freeze(path: str | Path) -> tuple[dict[str, Any], str]:
    payload, raw = _read_json(path, "L4 pre-run freeze")
    body = dict(payload)
    declared = _require_sha256(body.pop("freeze_sha256", None), "L4 freeze")
    expected_contract = {
        "direct_components": list(DIRECT_COMPONENT_IDS),
        "derived_arms": list(DERIVED_ARM_IDS),
        "item_count": M_ITEM_COUNT,
        "top_k": TOP_K,
        "physical_work_unit_count": WORK_UNIT_COUNT,
        "maximum_concurrency": MAXIMUM_CONCURRENCY,
        "single_start_barrier": True,
        "P_plus_Q_is_deterministic_RRF_not_additional_call": True,
        "RRF_constant": 60,
        "all_terminals_joined_before_scoring": True,
        "fresh_runtime_postflight_before_scoring": True,
        "primary_estimand": "Y(P_plus_Q_RRF)-Y(novel_Q)",
        "secondary_official_comparison": (
            "Y(P_plus_Q_RRF)-Y(official_HippoRAG)"
        ),
        "study_level_answer_generator_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
        "performance_gate": False,
    }
    if (
        payload.get("schema") != FREEZE_SCHEMA
        or payload.get("decision") != "authorize_exact_fresh_Hotpot_L4_once"
        or stable_hash(body) != declared
        or payload.get("execution_contract") != expected_contract
        or payload.get("ordering")
        != {
            "measurement_rows_read_while_freezing": 0,
            "support_labels_read_while_freezing": 0,
            "freeze_complete_before_M_L4_open": True,
        }
        or payload.get("raw_content_persisted") is not False
    ):
        raise HotpotRecursiveL4Error("L4 pre-run freeze drifted")
    _require_sha256(payload.get("authorization_hash"), "authorization hash")
    _require_sha256(payload.get("execution_root_hash"), "execution root hash")
    _assert_public_safe(payload)
    return payload, _sha256_bytes(raw)


def _validate_direct_ranking(
    value: Sequence[int], item: RetrievalItem
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise HotpotRecursiveL4Error("retrieval output is scalar")
    ranking = tuple(value)
    if (
        len(ranking) != TOP_K
        or len(set(ranking)) != TOP_K
        or any(type(index) is not int or not 0 <= index < len(item.corpus) for index in ranking)
    ):
        raise HotpotRecursiveL4Error("retrieval output violates top-five contract")
    return ranking


def _official(
    runtime: PreparedFormalRuntimeV2,
    item: RetrievalItem,
    work_root: Path,
) -> tuple[int, ...]:
    return runtime.retrieve(
        question=item.question,
        paragraphs=item.hipporag_paragraphs(),
        work_root=work_root,
    )


def _aggregate(
    *,
    arm_id: str,
    items: Sequence[RecursiveItem],
    rankings: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item, ranking in zip(items, rankings):
        hits = len(frozenset(item.support_indices).intersection(ranking))
        rows.append(
            {
                "item_id_hash": item.item_id_hash,
                "support_hits": hits,
                "support_total": len(item.support_indices),
                "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)}),
            }
        )
    total = sum(row["support_total"] for row in rows)
    hits = sum(row["support_hits"] for row in rows)
    return {
        "arm_id": arm_id,
        "item_count": len(rows),
        "support_hit_count": hits,
        "support_total": total,
        "support_recall_at_5": float(Fraction(hits, total)),
        "items_with_any_support_hit": sum(row["support_hits"] > 0 for row in rows),
        "ranking_score_closure_sha256": stable_hash(rows),
    }


def _paired(
    *,
    left: str,
    right: str,
    items: Sequence[RecursiveItem],
    arm_rankings: Mapping[str, Sequence[tuple[int, ...]]],
) -> dict[str, Any]:
    deltas: list[int] = []
    for ordinal, item in enumerate(items):
        supports = frozenset(item.support_indices)
        deltas.append(
            len(supports.intersection(arm_rankings[left][ordinal]))
            - len(supports.intersection(arm_rankings[right][ordinal]))
        )
    support_total = sum(len(item.support_indices) for item in items)
    net = sum(deltas)
    return {
        "left_arm": left,
        "right_arm": right,
        "net_support_hit_count": net,
        "support_recall_delta": float(Fraction(net, support_total)),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
    }


def _score(
    *,
    items: Sequence[RecursiveItem],
    direct: Mapping[tuple[int, str], tuple[int, ...]],
) -> dict[str, Any]:
    arms: dict[str, list[tuple[int, ...]]] = {arm: [] for arm in DERIVED_ARM_IDS}
    for ordinal, _item in enumerate(items):
        raw = direct[(ordinal, "canonical_RAW")]
        p = direct[(ordinal, "retained_P")]
        q = direct[(ordinal, "novel_Q")]
        official = direct[(ordinal, "official_HippoRAG")]
        arms["canonical_RAW"].append(raw)
        arms["retained_P"].append(p)
        arms["novel_Q"].append(q)
        arms["P_plus_Q_RRF"].append(
            _indices(deterministic_rrf((_doc_ids(p), _doc_ids(q))))
        )
        arms["official_HippoRAG"].append(official)
    metrics = {
        arm: _aggregate(arm_id=arm, items=items, rankings=rankings)
        for arm, rankings in arms.items()
    }
    retention = _paired(
        left="P_plus_Q_RRF",
        right="novel_Q",
        items=items,
        arm_rankings=arms,
    )
    novelty = _paired(
        left="P_plus_Q_RRF",
        right="retained_P",
        items=items,
        arm_rankings=arms,
    )
    versus_official = _paired(
        left="P_plus_Q_RRF",
        right="official_HippoRAG",
        items=items,
        arm_rankings=arms,
    )
    forgotten = 0
    forgetting_items = 0
    for ordinal, item in enumerate(items):
        supports = frozenset(item.support_indices)
        p_hits = supports.intersection(arms["retained_P"][ordinal])
        pq_hits = supports.intersection(arms["P_plus_Q_RRF"][ordinal])
        lost = len(p_hits - pq_hits)
        forgotten += lost
        forgetting_items += int(lost > 0)
    return {
        "primary_metric": "offline_micro_source_support_recall_at_5",
        "primary_estimand": "Y(P_plus_Q_RRF)-Y(novel_Q)",
        "arm_metrics": metrics,
        "retained_P_contribution": retention,
        "novel_Q_contribution": novelty,
        "P_plus_Q_minus_official_HippoRAG": versus_official,
        "forgetting": {
            "support_hit_count_P_lost_in_P_plus_Q": forgotten,
            "item_count_with_forgetting": forgetting_items,
            "support_total": metrics["P_plus_Q_RRF"]["support_total"],
        },
        "disposition": {
            "retained_improvement_observed": retention["net_support_hit_count"] > 0,
            "novel_improvement_observed": novelty["net_support_hit_count"] > 0,
            "positive_net_on_fixed_cohort_vs_official_HippoRAG": (
                versus_official["net_support_hit_count"] > 0
            ),
            "statistical_superiority_claim": False,
            "family_out_claim_for_P_plus_Q": False,
            "compute_budget_equivalence_claim": False,
            "performance_gate": False,
            "retry_or_followup_gate_authorized": False,
        },
        "P_adapted_after_prior_Hotpot_outcome": False,
        "Q_formed_only_on_disjoint_F_Q": True,
        "M_L4_used_for_selection": False,
        "raw_content_persisted": False,
    }


def execute_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path,
    m_l4_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    m1_pre_run_freeze_path: str | Path,
    m1_promotion_report_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise HotpotRecursiveL4Error(
            "formal L4 execution is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_freeze(pre_run_freeze_path)
    if freeze.get("implementation") != current_implementation_binding(project):
        raise HotpotRecursiveL4Error("live L4 implementation drifted")
    root = _new_root(execution_root, project)
    if freeze.get("execution_root_hash") != _root_hash(root, project):
        raise HotpotRecursiveL4Error("L4 execution root binding drifted")
    if root.exists():
        raise HotpotRecursiveL4Error("fresh L4 root exists; replay is forbidden")
    receipt, receipt_raw = _load_acquisition(
        project=project, path=acquisition_receipt_path
    )
    commitment = _commitment(receipt, "M_L4")
    expected_source = {
        "acquisition_file_sha256": _sha256_bytes(receipt_raw),
        "acquisition_sha256": receipt["acquisition_sha256"],
        "measurement_block_id_hash": stable_hash({"block": "M_L4"}),
        "measurement_block_file_sha256": commitment.file_sha256,
        "measurement_item_commitment_set_sha256": (
            commitment.item_commitment_set_sha256
        ),
        "measurement_item_count": commitment.count,
    }
    if freeze.get("source_binding") != expected_source:
        raise HotpotRecursiveL4Error("L4 source binding drifted")
    p_program, p_binding = _load_p(
        project=project,
        formation_path=p_formation_receipt_path,
        program_path=p_frozen_program_path,
    )
    _assert_p_matches_preregistration(receipt=receipt, p_binding=p_binding)
    q_program, q_binding = load_q(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        formation_receipt_path=q_formation_receipt_path,
        frozen_program_path=q_frozen_program_path,
    )
    if freeze.get("retained_P_binding") != p_binding or freeze.get(
        "novel_Q_binding"
    ) != q_binding:
        raise HotpotRecursiveL4Error("frozen P/Q bindings drifted")
    lineage = _load_positive_p_lineage(
        m1_freeze_path=m1_pre_run_freeze_path,
        m1_report_path=m1_promotion_report_path,
        p_binding=p_binding,
    )
    _assert_positive_lineage_matches_preregistration(
        receipt=receipt, lineage=lineage
    )
    if freeze.get("positive_P_lineage") != lineage:
        raise HotpotRecursiveL4Error("retained-P lineage drifted")
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    if freeze.get("capability_binding") != {
        "file_sha256": _sha256_bytes(capability_raw),
        "receipt_sha256": capability["receipt_sha256"],
        "bwrap_file_sha256": capability["bwrap_file_sha256"],
        "probe_contract_sha256": capability["probe_contract_sha256"],
        "fresh_probe_required_before_authorization": True,
    }:
        raise HotpotRecursiveL4Error("L4 capability binding drifted")
    live_probe = _probe_bubblewrap()
    if (
        live_probe.get("bwrap_file_sha256") != BWRAP_SHA256
        or live_probe.get("probe_contract_sha256")
        != capability["probe_contract_sha256"]
        or live_probe.get("probe_returncode") != 0
    ):
        raise HotpotRecursiveL4Error("fresh bwrap preflight drifted")
    paths = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = _prepare_runtime(project, paths)
    safe_runtime = prepared.safe_binding
    if freeze.get("runtime_binding") != _runtime_binding(prepared, paths):
        raise HotpotRecursiveL4Error("formal runtime binding drifted")
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise HotpotRecursiveL4Error("fresh L4 root exists; replay is forbidden") from exc
    stage = "authorization_consumption"
    attempted = completed = 0
    lock = threading.Lock()
    barrier = threading.Barrier(WORK_UNIT_COUNT)
    try:
        consumption_body: dict[str, Any] = {
            "schema": CONSUMPTION_SCHEMA,
            "authorization_hash": freeze["authorization_hash"],
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_hash": freeze["execution_root_hash"],
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {
            **consumption_body,
            "consumption_sha256": stable_hash(consumption_body),
        }
        _write_json_exclusive(root / CONSUMPTION_FILENAME, consumption)
        stage = "exact_M_L4_open_after_freeze"
        items = _load_private_block(
            project=project,
            path=m_l4_block_path,
            commitment=commitment,
        )
        if len(items) != M_ITEM_COUNT:
            raise HotpotRecursiveL4Error("M_L4 item count drifted")
        work_units = tuple(
            (ordinal, component, item.retrieval_view())
            for ordinal, item in enumerate(items)
            for component in DIRECT_COMPONENT_IDS
        )
        if len(work_units) != WORK_UNIT_COUNT:
            raise HotpotRecursiveL4Error("L4 work-unit grid drifted")
        stage = "retrieval_execution"

        def run_one(
            unit: tuple[int, str, RetrievalItem]
        ) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            ordinal, component, item = unit
            with lock:
                attempted += 1
            try:
                barrier.wait(timeout=120)
            except threading.BrokenBarrierError as exc:
                raise HotpotRecursiveL4Error(
                    "L4 maximum-width start barrier did not close"
                ) from exc
            if component == "canonical_RAW":
                value = tuple(paragraph.idx for paragraph in item.corpus[:TOP_K])
            elif component == "retained_P":
                value = _ranking(p_program, item)
            elif component == "novel_Q":
                value = _ranking(q_program, item)
            elif component == "official_HippoRAG":
                value = _official(
                    prepared,
                    item,
                    root / f"official_item_{ordinal:02d}",
                )
            else:  # pragma: no cover
                raise HotpotRecursiveL4Error("unknown L4 direct component")
            ranking = _validate_direct_ranking(value, item)
            with lock:
                completed += 1
            return (ordinal, component), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAXIMUM_CONCURRENCY,
            thread_name_prefix="hotpot-recursive-l4",
        ) as executor:
            futures = [executor.submit(run_one, unit) for unit in work_units]
            terminal_rows = [future.result() for future in futures]
        if attempted != WORK_UNIT_COUNT or completed != WORK_UNIT_COUNT:
            raise HotpotRecursiveL4Error("L4 terminal closure is incomplete")
        direct = dict(terminal_rows)
        if len(direct) != WORK_UNIT_COUNT:
            raise HotpotRecursiveL4Error("L4 terminal keys are not one-to-one")
        stage = "fresh_runtime_postflight_before_scoring"
        postflight = prepared.fresh_reverify()
        if postflight != safe_runtime:
            raise HotpotRecursiveL4Error("L4 runtime postflight drifted")
        stage = "offline_source_support_scoring_after_join"
        measurement = _score(items=items, direct=direct)
        ranking_receipts = [
            {
                "ordinal_sha256": stable_hash({"ordinal": ordinal}),
                "component_id": component,
                "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)}),
            }
            for (ordinal, component), ranking in sorted(direct.items())
        ]
        report_body: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "measurement": measurement,
            "execution": {
                "direct_components": list(DIRECT_COMPONENT_IDS),
                "derived_arms": list(DERIVED_ARM_IDS),
                "item_count": len(items),
                "physical_work_unit_count": WORK_UNIT_COUNT,
                "retrieval_call_count": attempted,
                "retrieval_terminal_count": completed,
                "configured_maximum_concurrency": MAXIMUM_CONCURRENCY,
                "observed_start_barrier_party_count": barrier.parties,
                "all_work_units_released_from_single_start_barrier": True,
                "all_terminals_joined_before_support_scoring": True,
                "ranking_receipt_set_sha256": stable_hash(ranking_receipts),
                "study_level_answer_generator_calls": 0,
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
            },
            "runtime": {
                "capability_receipt_sha256": capability["receipt_sha256"],
                "fresh_bwrap_preflight_before_authorization": True,
                "official_arm_terminal_count": M_ITEM_COUNT,
                "official_arm_uses_frozen_local_LLM_OpenIE": True,
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight["binding_sha256"],
            },
            "sealed_or_test_content_accessed": False,
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_sha256": stable_hash(report_body)}
        _assert_public_safe(report)
        stage = "aggregate_report_persistence"
        path = root / REPORT_FILENAME
        _write_json_exclusive(path, report)
        persisted, _ = _read_json(path, "persisted L4 report")
        persisted_body = dict(persisted)
        persisted_hash = persisted_body.pop("report_sha256", None)
        if persisted != report or stable_hash(persisted_body) != persisted_hash:
            raise HotpotRecursiveL4Error("persisted L4 report drifted")
        return persisted
    except Exception as exc:
        failure_body: dict[str, Any] = {
            "schema": FAILURE_SCHEMA,
            "valid": False,
            "freeze_sha256": freeze["freeze_sha256"],
            "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (root / CONSUMPTION_FILENAME).is_file(),
            "physical_work_unit_count": WORK_UNIT_COUNT,
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
            _write_json_exclusive(root / FAILURE_FILENAME, failure)
        except Exception:
            pass
        raise HotpotRecursiveL4Error(
            "formal L4 run failed and cannot be replayed"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    form = sub.add_parser("form-q")
    freeze = sub.add_parser("freeze")
    execute = sub.add_parser("execute")
    for command in (form, freeze, execute):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--acquisition-receipt", type=Path, required=True)
        command.add_argument("--p-formation-receipt", type=Path, required=True)
        command.add_argument("--p-frozen-program", type=Path, required=True)
    form.add_argument("--f-q-block", type=Path, required=True)
    form.add_argument("--output-dir", type=Path, required=True)
    for command in (freeze, execute):
        command.add_argument("--q-formation-receipt", type=Path, required=True)
        command.add_argument("--q-frozen-program", type=Path, required=True)
        command.add_argument("--m1-pre-run-freeze", type=Path, required=True)
        command.add_argument("--m1-promotion-report", type=Path, required=True)
        command.add_argument("--capability-receipt", type=Path, required=True)
        command.add_argument("--runtime-python", type=Path, required=True)
        command.add_argument("--local-llm-model", type=Path, required=True)
        command.add_argument("--local-embedding-model", type=Path, required=True)
        command.add_argument("--base-binding-receipt", type=Path, required=True)
        command.add_argument("--attestation-receipt", type=Path, required=True)
        command.add_argument("--execution-root", type=Path, required=True)
    freeze.add_argument("--authorization-hash", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute.add_argument("--pre-run-freeze", type=Path, required=True)
    execute.add_argument("--m-l4-block", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "form-q":
        form_q(
            project_root=args.project_root,
            acquisition_receipt_path=args.acquisition_receipt,
            f_q_block_path=args.f_q_block,
            p_formation_receipt_path=args.p_formation_receipt,
            p_frozen_program_path=args.p_frozen_program,
            output_dir=args.output_dir,
        )
        return 0
    common = {
        "project_root": args.project_root,
        "acquisition_receipt_path": args.acquisition_receipt,
        "p_formation_receipt_path": args.p_formation_receipt,
        "p_frozen_program_path": args.p_frozen_program,
        "q_formation_receipt_path": args.q_formation_receipt,
        "q_frozen_program_path": args.q_frozen_program,
        "m1_pre_run_freeze_path": args.m1_pre_run_freeze,
        "m1_promotion_report_path": args.m1_promotion_report,
        "capability_receipt_path": args.capability_receipt,
        "runtime_python": args.runtime_python,
        "local_llm_model": args.local_llm_model,
        "local_embedding_model": args.local_embedding_model,
        "base_binding_receipt_path": args.base_binding_receipt,
        "attestation_receipt_path": args.attestation_receipt,
        "execution_root": args.execution_root,
    }
    if args.command == "freeze":
        build_pre_run_freeze(
            **common,
            authorization_hash=args.authorization_hash,
            output_path=args.output,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_formal(
            **common,
            pre_run_freeze_path=args.pre_run_freeze,
            m_l4_block_path=args.m_l4_block,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
