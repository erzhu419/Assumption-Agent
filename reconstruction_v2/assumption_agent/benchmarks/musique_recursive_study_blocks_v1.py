"""Exact private-block parser and offline formation for the MuSiQue study.

Formation access is limited to ``F1``, ``F2``, ``F3``, and ``A_form``.  A
separate measurement loader requires a pre-run freeze binding and is intended
for formal runners; it prevents a formation caller from casually opening M1,
M2, M3, or A_hold.  Every block is checked against the self-hashed acquisition
receipt, exact file bytes, canonical JSONL, exact row schema, and the public
item-commitment set before any row is returned.

Typed retriever formation reuses the finite DSL and selection implementation
from :mod:`musique_typed_retriever_formation_v1`.  It performs no model,
network, generator, or online-evaluator call and emits only aggregate/hash
evidence plus the frozen typed program grammar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from . import musique_recursive_study_acquisition_v1 as acquisition
from .musique_official_core_comparison_v1 import (
    normalize_answer_primary,
    normalize_answer_secondary,
)
from .musique_typed_retriever_formation_v1 import (
    DSL_VERSION,
    FOLD_POLICY,
    MAX_CANDIDATES,
    MAX_SEMANTIC_NODES,
    OPERATOR_VERSION,
    SELECTION_POLICY,
    TOKENIZER_VERSION,
    TOP_K,
    RetrievalParagraph,
    TypedRetrievalProgram,
    _TrainItem,
    _assess,
    _select,
    current_implementation_binding as current_typed_implementation_binding,
    enumerate_programs,
)


PARSER_VERSION = "musique_recursive_study_exact_block_parser_v1"
FORMATION_VERSION = "musique_recursive_study_typed_formation_v1"
FORMATION_RECEIPT_SCHEMA = f"{FORMATION_VERSION}_receipt"
FROZEN_PROGRAM_SCHEMA = f"{FORMATION_VERSION}_frozen_program"
FORMATION_BLOCKS = ("F1", "F2", "F3", "A_form")
RETRIEVER_FORMATION_BLOCKS = ("F1", "F2", "F3")
MEASUREMENT_BLOCKS = ("M1", "M2", "M3", "A_hold")
EXPECTED_ROW_KEYS = frozenset(
    {
        "answers",
        "block",
        "corpus",
        "item_id",
        "normalized_answers",
        "question",
        "schema",
        "source_row_sha256",
        "support_indices",
    }
)
EXPECTED_ACQUISITION_KEYS = frozenset(
    {
        "acquisition_sha256",
        "commitments",
        "counts",
        "decision",
        "ordering",
        "private_boundary",
        "safety",
        "schema",
        "source",
    }
)
IMPLEMENTATION_SCHEMA = "musique_recursive_study_formation_implementation_v1"
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/musique_recursive_study_acquisition_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MuSiQueStudyBlockError(RuntimeError):
    """A private study block or formation artifact failed closed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueStudyBlockError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _absolute_no_symlink(path: str | Path, field_name: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueStudyBlockError(
                f"{field_name} contains a symlink component"
            )
    return candidate


def _read_json_object(path: Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    path = _absolute_no_symlink(path, field_name)
    if not path.is_file():
        raise MuSiQueStudyBlockError(f"{field_name} is unavailable")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueStudyBlockError(f"{field_name} is invalid") from exc
    if not isinstance(value, dict):
        raise MuSiQueStudyBlockError(f"{field_name} must contain one object")
    return value, raw


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answers"',
        '"corpus"',
        '"item_id"',
        '"normalized_answers"',
        '"paragraph_text"',
        '"private_root"',
        '"question"',
        '"source_row_sha256"',
        '"support_indices"',
    )
    if any(token in serialized for token in forbidden):
        raise MuSiQueStudyBlockError(
            "public study artifact contains private content or locator keys"
        )


def current_study_formation_implementation_binding(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(__file__).resolve(strict=True).parents[2]
        if project_root is None
        else Path(project_root).resolve(strict=True)
    )
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueStudyBlockError(
                f"study formation implementation is missing: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def _validate_implementation_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "files",
        "set_sha256",
    }:
        raise MuSiQueStudyBlockError(
            "study formation implementation binding is malformed"
        )
    if value.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueStudyBlockError(
            "study formation implementation schema drifted"
        )
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(
        IMPLEMENTATION_RELATIVE_FILES
    ):
        raise MuSiQueStudyBlockError(
            "study formation implementation set drifted"
        )
    rows = []
    for expected, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row.get("path") != expected
        ):
            raise MuSiQueStudyBlockError(
                "study formation implementation row drifted"
            )
        rows.append(
            {
                "path": expected,
                "sha256": _require_sha256(
                    row.get("sha256"), "implementation file hash"
                ),
            }
        )
    set_hash = _require_sha256(
        value.get("set_sha256"), "implementation set hash"
    )
    if stable_hash(rows) != set_hash:
        raise MuSiQueStudyBlockError(
            "study formation implementation set hash drifted"
        )
    return {"schema": IMPLEMENTATION_SCHEMA, "files": rows, "set_sha256": set_hash}


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    count: int
    file_sha256: str
    item_commitment_set_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyAcquisitionBinding:
    acquisition_sha256: str
    acquisition_file_sha256: str
    private_pack_sha256: str
    blocks: tuple[BlockCommitment, ...]

    def commitment_for(self, block: str) -> BlockCommitment:
        matches = tuple(row for row in self.blocks if row.block == block)
        if len(matches) != 1:
            raise MuSiQueStudyBlockError(
                "requested block has no unique public commitment"
            )
        return matches[0]

    def safe_payload(self) -> dict[str, Any]:
        return {
            "acquisition_sha256": self.acquisition_sha256,
            "acquisition_file_sha256": self.acquisition_file_sha256,
            "private_pack_sha256": self.private_pack_sha256,
            "block_commitment_set_sha256": stable_hash(
                [row.to_dict() for row in self.blocks]
            ),
            "block_count": len(self.blocks),
            "raw_content_or_private_paths_persisted": False,
        }


def load_study_acquisition_binding(
    acquisition_receipt_path: str | Path,
) -> StudyAcquisitionBinding:
    payload, raw = _read_json_object(
        Path(acquisition_receipt_path), "study acquisition receipt"
    )
    if set(payload) != EXPECTED_ACQUISITION_KEYS:
        raise MuSiQueStudyBlockError(
            "study acquisition receipt key set drifted"
        )
    body = dict(payload)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "acquisition receipt hash"
    )
    if (
        payload.get("schema") != acquisition.ACQUISITION_SCHEMA
        or payload.get("decision")
        != "fresh_private_pack_formed_no_formation_or_measurement_authority"
        or stable_hash(body) != declared
    ):
        raise MuSiQueStudyBlockError(
            "study acquisition receipt self-hash or decision drifted"
        )
    counts = payload.get("counts")
    commitments = payload.get("commitments")
    ordering = payload.get("ordering")
    source = payload.get("source")
    private_boundary = payload.get("private_boundary")
    safety = payload.get("safety")
    if not all(
        isinstance(value, Mapping)
        for value in (
            counts,
            commitments,
            ordering,
            source,
            private_boundary,
            safety,
        )
    ):
        raise MuSiQueStudyBlockError(
            "study acquisition receipt sections are malformed"
        )
    if set(source) != {
        "archive_sha256",
        "commit",
        "dataset",
        "official_dev_member_sha256",
        "repository",
        "source_split",
        "split_disjoint_from_prior_official_train_cohort",
    } or (
        source.get("commit") != acquisition.OFFICIAL_SOURCE_COMMIT
        or source.get("dataset") != "MuSiQue-Answerable v1.0"
        or source.get("source_split") != "official_dev"
        or source.get("archive_sha256") != acquisition.OFFICIAL_ARCHIVE_SHA256
        or source.get("split_disjoint_from_prior_official_train_cohort")
        is not True
    ):
        raise MuSiQueStudyBlockError(
            "study acquisition source schema or identity drifted"
        )
    _require_sha256(
        source.get("official_dev_member_sha256"),
        "official DEV member hash",
    )
    if set(counts) != {
        "blocks",
        "eligible_rows",
        "oracle_disagreements",
        "selected_rows",
        "source_rows",
    } or (
        type(counts.get("source_rows")) is not int
        or type(counts.get("eligible_rows")) is not int
        or counts.get("source_rows", 0) < counts.get("eligible_rows", 0)
        or counts.get("eligible_rows", 0) < acquisition.SELECTED_COUNT
        or counts.get("oracle_disagreements") != 0
    ):
        raise MuSiQueStudyBlockError(
            "study acquisition count schema drifted"
        )
    if set(commitments) != {
        "block_files",
        "item_ids_persisted_publicly",
        "private_pack_sha256",
        "private_paths_persisted_publicly",
        "selection_secret_commitment_sha256",
    } or (
        commitments.get("item_ids_persisted_publicly") is not False
        or commitments.get("private_paths_persisted_publicly") is not False
    ):
        raise MuSiQueStudyBlockError(
            "study acquisition commitment schema drifted"
        )
    _require_sha256(
        commitments.get("selection_secret_commitment_sha256"),
        "selection secret commitment",
    )
    if set(ordering) != {
        "all_eight_blocks_formed_together",
        "formation_or_measurement_before_pack_complete",
        "ordering_evidence_scope",
        "preregistration_preceded_block_files_local_mtime",
        "preregistration_sha256",
    }:
        raise MuSiQueStudyBlockError(
            "study acquisition ordering schema drifted"
        )
    _require_sha256(
        ordering.get("preregistration_sha256"), "preregistration hash"
    )
    if set(private_boundary) != {
        "private_locator_path_persisted_publicly",
        "private_locator_git_ignored",
        "private_pack_git_ignored",
        "secret_free_private_locator_formed",
        "selection_secret_git_ignored",
        "source_archive_git_ignored",
    } or any(
        private_boundary.get(field) is not True
        for field in (
            "private_locator_git_ignored",
            "private_pack_git_ignored",
            "secret_free_private_locator_formed",
            "selection_secret_git_ignored",
            "source_archive_git_ignored",
        )
    ) or private_boundary.get(
        "private_locator_path_persisted_publicly"
    ) is not False:
        raise MuSiQueStudyBlockError(
            "study acquisition private-boundary schema drifted"
        )
    if set(safety) != {
        "measurement_blocks_scored",
        "model_calls",
        "network_calls_during_acquisition",
        "online_evaluator_calls",
        "prior_closed_cohort_accessed",
        "scores_computed",
    } or any(
        safety.get(field) != 0
        for field in (
            "measurement_blocks_scored",
            "model_calls",
            "network_calls_during_acquisition",
            "online_evaluator_calls",
            "scores_computed",
        )
    ) or safety.get("prior_closed_cohort_accessed") is not False:
        raise MuSiQueStudyBlockError(
            "study acquisition safety schema drifted"
        )
    if (
        counts.get("selected_rows") != acquisition.SELECTED_COUNT
        or counts.get("blocks") != acquisition.BLOCK_COUNTS
        or ordering.get("all_eight_blocks_formed_together") is not True
        or ordering.get("formation_or_measurement_before_pack_complete")
        is not False
        or ordering.get("preregistration_preceded_block_files_local_mtime")
        is not True
        or ordering.get("ordering_evidence_scope")
        != "local_filesystem_only"
    ):
        raise MuSiQueStudyBlockError(
            "study acquisition count or ordering contract drifted"
        )
    block_files = commitments.get("block_files")
    if not isinstance(block_files, list) or len(block_files) != len(
        acquisition.BLOCK_ORDER
    ):
        raise MuSiQueStudyBlockError(
            "study acquisition block commitment set is malformed"
        )
    blocks: list[BlockCommitment] = []
    for expected_block, row in zip(acquisition.BLOCK_ORDER, block_files):
        if not isinstance(row, Mapping) or set(row) != {
            "block",
            "count",
            "file_sha256",
            "item_commitment_set_sha256",
        }:
            raise MuSiQueStudyBlockError(
                "study block commitment row is malformed"
            )
        if (
            row.get("block") != expected_block
            or row.get("count") != acquisition.BLOCK_COUNTS[expected_block]
        ):
            raise MuSiQueStudyBlockError(
                "study block commitment order or count drifted"
            )
        blocks.append(
            BlockCommitment(
                block=expected_block,
                count=int(row["count"]),
                file_sha256=_require_sha256(
                    row.get("file_sha256"), "block file hash"
                ),
                item_commitment_set_sha256=_require_sha256(
                    row.get("item_commitment_set_sha256"),
                    "block item commitment set",
                ),
            )
        )
    private_pack_hash = _require_sha256(
        commitments.get("private_pack_sha256"), "private pack hash"
    )
    if stable_hash([row.to_dict() for row in blocks]) != private_pack_hash:
        raise MuSiQueStudyBlockError(
            "study private-pack commitment drifted"
        )
    binding = StudyAcquisitionBinding(
        acquisition_sha256=declared,
        acquisition_file_sha256=_sha256_bytes(raw),
        private_pack_sha256=private_pack_hash,
        blocks=tuple(blocks),
    )
    _assert_public_safe(payload)
    _assert_public_safe(binding.safe_payload())
    return binding


@dataclass(frozen=True)
class RetrievalStudyItem:
    """Gold-free view passed to retrieval operators during measurement."""

    item_id: str = field(repr=False)
    question: str = field(repr=False)
    corpus: tuple[RetrievalParagraph, ...] = field(repr=False)
    item_commitment_sha256: str

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
class StudyItem:
    item_id: str = field(repr=False)
    question: str = field(repr=False)
    corpus: tuple[RetrievalParagraph, ...] = field(repr=False)
    support_indices: tuple[int, ...] = field(repr=False)
    answers: tuple[str, ...] = field(repr=False)
    normalized_answers: tuple[str, ...] = field(repr=False)
    source_row_sha256: str = field(repr=False)
    item_commitment_sha256: str

    @property
    def item_id_hash(self) -> str:
        return stable_hash({"item_id": self.item_id})

    def retrieval_view(self) -> RetrievalStudyItem:
        return RetrievalStudyItem(
            item_id=self.item_id,
            question=self.question,
            corpus=self.corpus,
            item_commitment_sha256=self.item_commitment_sha256,
        )


@dataclass(frozen=True)
class LoadedStudyBlock:
    block: str
    items: tuple[StudyItem, ...] = field(repr=False)
    file_sha256: str
    item_commitment_set_sha256: str
    acquisition: StudyAcquisitionBinding

    def safe_payload(self) -> dict[str, Any]:
        body = {
            "block_id_hash": stable_hash({"block": self.block}),
            "item_count": len(self.items),
            "file_sha256": self.file_sha256,
            "item_commitment_set_sha256": self.item_commitment_set_sha256,
            "item_id_set_sha256": stable_hash(
                [item.item_id_hash for item in self.items]
            ),
            "total_support_count": sum(
                len(item.support_indices) for item in self.items
            ),
            "acquisition_sha256": self.acquisition.acquisition_sha256,
            "private_pack_sha256": self.acquisition.private_pack_sha256,
            "raw_content_persisted": False,
        }
        _assert_public_safe(body)
        return body


def _parse_exact_block(
    *,
    block_path: str | Path,
    expected_block: str,
    binding: StudyAcquisitionBinding,
) -> LoadedStudyBlock:
    if expected_block not in acquisition.BLOCK_ORDER:
        raise MuSiQueStudyBlockError("unknown study block")
    path = _absolute_no_symlink(block_path, "exact study block")
    if path.name != f"{expected_block}.jsonl" or not path.is_file():
        raise MuSiQueStudyBlockError(
            "study parser accepts only the exact named block file"
        )
    commitment = binding.commitment_for(expected_block)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "rb") as handle:
        raw = handle.read()
    if _sha256_bytes(raw) != commitment.file_sha256:
        raise MuSiQueStudyBlockError("exact study block file hash mismatch")
    if not raw or not raw.endswith(b"\n"):
        raise MuSiQueStudyBlockError("study block JSONL framing drifted")
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            if not line:
                raise MuSiQueStudyBlockError("study block contains a blank row")
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise MuSiQueStudyBlockError(
                    "study block row must be an object"
                )
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueStudyBlockError("study block JSONL is invalid") from exc
    if len(rows) != commitment.count:
        raise MuSiQueStudyBlockError("study block row count mismatch")
    canonical = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    if raw != canonical:
        raise MuSiQueStudyBlockError(
            "study block differs from canonical acquisition bytes"
        )
    if stable_hash([stable_hash(row) for row in rows]) != (
        commitment.item_commitment_set_sha256
    ):
        raise MuSiQueStudyBlockError(
            "study block item commitment set mismatch"
        )

    items: list[StudyItem] = []
    seen_ids: set[str] = set()
    for row in rows:
        if set(row) != EXPECTED_ROW_KEYS:
            raise MuSiQueStudyBlockError("study block row schema drifted")
        if (
            row.get("schema") != acquisition.PRIVATE_ROW_SCHEMA
            or row.get("block") != expected_block
        ):
            raise MuSiQueStudyBlockError(
                "study block row identity drifted"
            )
        item_id = row.get("item_id")
        question = row.get("question")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen_ids
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise MuSiQueStudyBlockError(
                "study item identity or question is malformed"
            )
        seen_ids.add(item_id)
        source_hash = _require_sha256(
            row.get("source_row_sha256"), "source row hash"
        )
        answers_raw = row.get("answers")
        normalized_raw = row.get("normalized_answers")
        if (
            not isinstance(answers_raw, list)
            or not answers_raw
            or any(not isinstance(value, str) for value in answers_raw)
            or not isinstance(normalized_raw, list)
            or len(normalized_raw) < 2
            or any(not isinstance(value, str) or not value for value in normalized_raw)
        ):
            raise MuSiQueStudyBlockError(
                "study answer envelope is malformed"
            )
        normalized_primary = tuple(
            dict.fromkeys(
                normalized
                for answer in answers_raw
                if (normalized := normalize_answer_primary(answer))
            )
        )
        normalized_secondary = tuple(
            dict.fromkeys(
                normalized
                for answer in answers_raw
                if (normalized := normalize_answer_secondary(answer))
            )
        )
        if (
            normalized_primary != normalized_secondary
            or tuple(normalized_raw) != normalized_primary
            or len(normalized_primary) < 2
        ):
            raise MuSiQueStudyBlockError(
                "study normalized-answer consensus drifted"
            )
        corpus_raw = row.get("corpus")
        support_raw = row.get("support_indices")
        if not isinstance(corpus_raw, list) or not isinstance(support_raw, list):
            raise MuSiQueStudyBlockError(
                "study retrieval envelope is malformed"
            )
        corpus: list[RetrievalParagraph] = []
        label_indices: list[int] = []
        for position, paragraph in enumerate(corpus_raw):
            if not isinstance(paragraph, Mapping) or set(paragraph) != {
                "idx",
                "is_supporting",
                "text",
                "title",
            }:
                raise MuSiQueStudyBlockError(
                    "study paragraph schema drifted"
                )
            index = paragraph.get("idx")
            title = paragraph.get("title")
            text = paragraph.get("text")
            supporting = paragraph.get("is_supporting")
            if (
                type(index) is not int
                or index != position
                or not isinstance(title, str)
                or not title.strip()
                or not isinstance(text, str)
                or not text.strip()
                or type(supporting) is not bool
            ):
                raise MuSiQueStudyBlockError(
                    "study paragraph value drifted"
                )
            corpus.append(RetrievalParagraph(index, title, text))
            if supporting:
                label_indices.append(index)
        if (
            not TOP_K <= len(corpus) <= 128
            or len(label_indices) < 2
            or len(corpus) - len(label_indices) < 1
            or any(type(value) is not int for value in support_raw)
            or tuple(support_raw) != tuple(label_indices)
        ):
            raise MuSiQueStudyBlockError(
                "study support-label or corpus eligibility drifted"
            )
        items.append(
            StudyItem(
                item_id=item_id,
                question=question,
                corpus=tuple(corpus),
                support_indices=tuple(label_indices),
                answers=tuple(answers_raw),
                normalized_answers=tuple(normalized_raw),
                source_row_sha256=source_hash,
                item_commitment_sha256=stable_hash(row),
            )
        )
    return LoadedStudyBlock(
        block=expected_block,
        items=tuple(items),
        file_sha256=commitment.file_sha256,
        item_commitment_set_sha256=commitment.item_commitment_set_sha256,
        acquisition=binding,
    )


def load_formation_block(
    *,
    block_path: str | Path,
    acquisition_receipt_path: str | Path,
    expected_block: str,
) -> LoadedStudyBlock:
    """Open one exact formation block; measurement blocks are inexpressible."""

    if expected_block not in FORMATION_BLOCKS:
        raise MuSiQueStudyBlockError(
            "formation access is limited to F1, F2, F3, and A_form"
        )
    binding = load_study_acquisition_binding(acquisition_receipt_path)
    return _parse_exact_block(
        block_path=block_path,
        expected_block=expected_block,
        binding=binding,
    )


def load_measurement_block_after_freeze(
    *,
    block_path: str | Path,
    acquisition_receipt_path: str | Path,
    measurement_freeze_path: str | Path,
    expected_block: str,
) -> LoadedStudyBlock:
    """Open an exact measurement block only under a persisted freeze artifact."""

    if expected_block not in MEASUREMENT_BLOCKS:
        raise MuSiQueStudyBlockError(
            "measurement loader accepts only M1, M2, M3, or A_hold"
        )
    freeze, _freeze_raw = _read_json_object(
        Path(measurement_freeze_path), "measurement pre-run freeze"
    )
    freeze_body = dict(freeze)
    declared_freeze = _require_sha256(
        freeze_body.pop("freeze_hash", None), "measurement freeze hash"
    )
    source = freeze.get("source_binding")
    ordering = freeze.get("ordering")
    if (
        stable_hash(freeze_body) != declared_freeze
        or not isinstance(source, Mapping)
        or not isinstance(ordering, Mapping)
        or source.get("measurement_block_id_hash")
        != stable_hash({"block": expected_block})
        or ordering.get("measurement_block_rows_read_while_freezing") != 0
        or ordering.get("measurement_support_labels_read_while_freezing") != 0
        or ordering.get("pre_run_freeze_complete_before_measurement_open")
        is not True
    ):
        raise MuSiQueStudyBlockError(
            "measurement pre-run freeze is invalid or for another block"
        )
    binding = load_study_acquisition_binding(acquisition_receipt_path)
    commitment = binding.commitment_for(expected_block)
    if (
        binding.acquisition_sha256
        != _require_sha256(
            source.get("acquisition_sha256"), "frozen acquisition hash"
        )
        or commitment.file_sha256
        != _require_sha256(
            source.get("measurement_block_file_sha256"),
            "frozen block file hash",
        )
    ):
        raise MuSiQueStudyBlockError(
            "measurement block differs from the pre-run freeze"
        )
    return _parse_exact_block(
        block_path=block_path,
        expected_block=expected_block,
        binding=binding,
    )


@dataclass(frozen=True)
class StudyFormationResult:
    program: TypedRetrievalProgram
    receipt: Mapping[str, Any]
    envelope: Mapping[str, Any]


def form_study_typed_retriever(
    *,
    block_path: str | Path,
    acquisition_receipt_path: str | Path,
    expected_block: str,
    output_dir: str | Path | None = None,
) -> StudyFormationResult:
    """Form one typed retriever on F1, F2, or F3 only."""

    if expected_block not in RETRIEVER_FORMATION_BLOCKS:
        raise MuSiQueStudyBlockError(
            "typed retriever formation is limited to F1, F2, or F3"
        )
    block = load_formation_block(
        block_path=block_path,
        acquisition_receipt_path=acquisition_receipt_path,
        expected_block=expected_block,
    )
    implementation = current_study_formation_implementation_binding()
    typed_implementation = current_typed_implementation_binding()
    items = tuple(
        _TrainItem(
            question=item.question,
            corpus=item.corpus,
            support_indices=item.support_indices,
        )
        for item in block.items
    )
    programs = tuple(enumerate_programs())
    if not programs or len(programs) > MAX_CANDIDATES:
        raise MuSiQueStudyBlockError("typed candidate budget drifted")
    if any(program.type_issues() for program in programs):
        raise MuSiQueStudyBlockError("typed grammar emitted an invalid program")
    winner, assessments, unique = _select(programs, items)
    aliases = tuple(
        row for row in assessments if row.behavior_hash == winner.behavior_hash
    )

    folds: list[dict[str, Any]] = []
    fold_program_hashes: list[str] = []
    fold_behavior_hashes: list[str] = []
    for fold_index in range(4):
        held_positions = tuple(
            position
            for position in range(len(items))
            if position % 4 == fold_index
        )
        held_set = set(held_positions)
        fit = tuple(
            item for position, item in enumerate(items) if position not in held_set
        )
        held = tuple(items[position] for position in held_positions)
        fold_winner, _all, fold_unique = _select(programs, fit)
        held_score = _assess(fold_winner.program, held)
        full_score = _assess(fold_winner.program, items)
        fold_program_hashes.append(fold_winner.program.program_hash)
        fold_behavior_hashes.append(full_score.behavior_hash)
        folds.append(
            {
                "fold_index": fold_index,
                "fit_item_count": len(fit),
                "held_item_count": len(held),
                "fit_behavior_unique_count": len(fold_unique),
                "selected_program_hash": fold_winner.program.program_hash,
                "selected_full_behavior_hash": full_score.behavior_hash,
                "held_support_hit_count": held_score.support_hit_count,
                "held_support_label_count": held_score.support_label_count,
                "held_invalid_count": held_score.invalid_count,
            }
        )
    receipt_body: dict[str, Any] = {
        "schema": FORMATION_RECEIPT_SCHEMA,
        "formation_version": FORMATION_VERSION,
        "status": "formed_offline_on_exact_formation_block",
        "formation_block_id_hash": stable_hash({"block": expected_block}),
        "implementation": implementation,
        "typed_dsl_implementation": typed_implementation,
        "source_binding": block.safe_payload(),
        "search_receipt": {
            "dsl_version": DSL_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "top_k": TOP_K,
            "candidate_count": len(programs),
            "candidate_budget": MAX_CANDIDATES,
            "type_valid_count": sum(
                not row.program.type_issues() for row in assessments
            ),
            "behavior_unique_count": len(unique),
            "behavior_alias_count": len(assessments) - len(unique),
            "selected_behavior_alias_class_size": len(aliases),
            "selection_policy": SELECTION_POLICY,
            "semantic_node_budget": MAX_SEMANTIC_NODES,
        },
        "selection_receipt": {
            "selected_program_hash": winner.program.program_hash,
            "selected_behavior_hash": winner.behavior_hash,
            "support_hit_count": winner.support_hit_count,
            "support_label_count": winner.support_label_count,
            "support_recall_at_5_numerator": winner.recall_at_5.numerator,
            "support_recall_at_5_denominator": winner.recall_at_5.denominator,
            "invalid_count": winner.invalid_count,
            "program_length": winner.program.program_length,
        },
        "crossfit_receipt": {
            "policy": FOLD_POLICY,
            "fold_count": 4,
            "folds": folds,
            "selected_program_stable": len(set(fold_program_hashes)) == 1
            and fold_program_hashes[0] == winner.program.program_hash,
            "selected_behavior_stable": len(set(fold_behavior_hashes)) == 1
            and fold_behavior_hashes[0] == winner.behavior_hash,
        },
        "offline_contract": {
            "model_calls": 0,
            "network_calls": 0,
            "generator_calls": 0,
            "online_evaluator_calls": 0,
            "measurement_block_accessed": False,
        },
        "performance_claim": False,
        "raw_content_persisted": False,
    }
    receipt = {**receipt_body, "receipt_hash": stable_hash(receipt_body)}
    envelope_body = {
        "schema": FROZEN_PROGRAM_SCHEMA,
        "operator_version": OPERATOR_VERSION,
        "implementation": implementation,
        "typed_dsl_implementation": typed_implementation,
        "program": winner.program.to_dict(),
        "program_hash": winner.program.program_hash,
        "formation_receipt_hash": receipt["receipt_hash"],
        "formation_block_id_hash": stable_hash({"block": expected_block}),
        "raw_content_persisted": False,
    }
    envelope = {**envelope_body, "envelope_hash": stable_hash(envelope_body)}
    _assert_public_safe(receipt)
    _assert_public_safe(envelope)

    if output_dir is not None:
        destination = _absolute_no_symlink(
            output_dir, "study formation output"
        )
        if destination.exists():
            raise MuSiQueStudyBlockError(
                "study formation output must be a fresh root"
            )
        destination.mkdir(parents=True, mode=0o700)
        _write_json_exclusive(destination / "formation.receipt.json", receipt)
        _write_json_exclusive(destination / "frozen_program.json", envelope)
    return StudyFormationResult(
        program=winner.program,
        receipt=receipt,
        envelope=envelope,
    )


def load_study_frozen_program(
    *,
    frozen_program_path: str | Path,
    formation_receipt_path: str | Path,
    verify_live: bool = True,
    implementation_root: str | Path | None = None,
) -> tuple[TypedRetrievalProgram, Mapping[str, Any], Mapping[str, Any]]:
    receipt, _receipt_raw = _read_json_object(
        Path(formation_receipt_path), "study formation receipt"
    )
    receipt_body = dict(receipt)
    declared_receipt = _require_sha256(
        receipt_body.pop("receipt_hash", None), "formation receipt hash"
    )
    if (
        receipt.get("schema") != FORMATION_RECEIPT_SCHEMA
        or stable_hash(receipt_body) != declared_receipt
        or receipt.get("status")
        != "formed_offline_on_exact_formation_block"
        or receipt.get("raw_content_persisted") is not False
        or receipt.get("offline_contract", {}).get("network_calls") != 0
        or receipt.get("offline_contract", {}).get(
            "measurement_block_accessed"
        )
        is not False
    ):
        raise MuSiQueStudyBlockError("study formation receipt drifted")
    implementation = _validate_implementation_binding(
        receipt.get("implementation")
    )
    envelope, _envelope_raw = _read_json_object(
        Path(frozen_program_path), "study frozen program"
    )
    envelope_body = dict(envelope)
    declared_envelope = _require_sha256(
        envelope_body.pop("envelope_hash", None), "frozen program hash"
    )
    expected_keys = {
        "envelope_hash",
        "formation_block_id_hash",
        "formation_receipt_hash",
        "implementation",
        "operator_version",
        "program",
        "program_hash",
        "raw_content_persisted",
        "schema",
        "typed_dsl_implementation",
    }
    if (
        set(envelope) != expected_keys
        or envelope.get("schema") != FROZEN_PROGRAM_SCHEMA
        or envelope.get("operator_version") != OPERATOR_VERSION
        or envelope.get("raw_content_persisted") is not False
        or stable_hash(envelope_body) != declared_envelope
        or envelope.get("formation_receipt_hash") != declared_receipt
        or envelope.get("formation_block_id_hash")
        != receipt.get("formation_block_id_hash")
        or envelope.get("implementation") != implementation
        or envelope.get("typed_dsl_implementation")
        != receipt.get("typed_dsl_implementation")
    ):
        raise MuSiQueStudyBlockError("study frozen program envelope drifted")
    payload = envelope.get("program")
    if not isinstance(payload, Mapping):
        raise MuSiQueStudyBlockError("study frozen program payload is malformed")
    try:
        program = TypedRetrievalProgram.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise MuSiQueStudyBlockError(
            "study frozen program payload is malformed"
        ) from exc
    if (
        dict(payload) != program.to_dict()
        or program.type_issues()
        or envelope.get("program_hash") != program.program_hash
        or receipt.get("selection_receipt", {}).get(
            "selected_program_hash"
        )
        != program.program_hash
    ):
        raise MuSiQueStudyBlockError("study frozen program grammar drifted")
    if verify_live:
        live = current_study_formation_implementation_binding(
            implementation_root
        )
        if live != implementation:
            raise MuSiQueStudyBlockError(
                "live study formation implementation drifted"
            )
        typed_live = current_typed_implementation_binding(
            implementation_root
        )
        if typed_live != receipt.get("typed_dsl_implementation"):
            raise MuSiQueStudyBlockError(
                "live typed DSL implementation drifted"
            )
    _assert_public_safe(receipt)
    _assert_public_safe(envelope)
    return program, receipt, envelope


__all__ = [
    "FORMATION_BLOCKS",
    "MEASUREMENT_BLOCKS",
    "MuSiQueStudyBlockError",
    "RetrievalStudyItem",
    "StudyItem",
    "form_study_typed_retriever",
    "load_formation_block",
    "load_measurement_block_after_freeze",
    "load_study_acquisition_binding",
    "load_study_frozen_program",
]
