"""Source-free, one-shot formal controller for the frozen BioASQ P1 study.

Only two public projections cross the action boundary: an opaque work ID plus
the exact question text, and an ordinal/text passage corpus.  Question family
and the non-empty set-valued qrels are held by the acquisition boundary until
the corresponding action archive has been sealed mode 0400.

The controller executes one immutable lifecycle:

* form and seal all five A_form recipe slates, then release qrels and fit E1
  exactly once;
* seal the unchanged E1 behavior on label-free F_search;
* seal E0/E1/RAW/official-HippoRAG A_hold actions, then release qrels;
* authorize M_search only after the preregistered A_hold E1-over-E0 test;
* if authorized, seal the same four M_search arms before its qrels and test L5.

All scoring is local integer arithmetic.  The public terminal contains safe
aggregates and commitments only; content-bearing archives remain private.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Protocol, Sequence

from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core


VERSION = "bioasq_p1_formal_controller_v1"
STUDY_ID = core.STUDY_ID
FAMILIES = ("yesno", "factoid", "list", "summary")
BLOCK_COUNTS = {
    "A_form": 96,
    "F_search": 32,
    "A_hold": 48,
    "M_search": 48,
}
FAMILY_COUNTS = {
    "A_form": 24,
    "F_search": 8,
    "A_hold": 12,
    "M_search": 12,
}
INITIAL_BLOCKS = ("A_form", "F_search", "A_hold")
SCORING_BLOCKS = ("A_hold", "M_search")
CORPUS_SIZE = 2_900
ALPHA = Fraction(1, 10)
REALITY_MINIMUM_STABLE_FAMILIES = 3

RECALL_SCALE = 300_000
RECIPROCAL_RANK_SCALE = 300_000
MAX_SET_UTILITY = 600_000

FORMAL_MARKER_FILENAME = "formal.marker.json"
FORMAL_TERMINAL_FILENAME = "formal_terminal.json"
PROMOTION_AUTHORIZATION_FILENAME = "promotion.authorization.json"
NO_CHANGE_COUNT_KEY = (
    "retry_replay_resample_model_provider_candidate_parser_family_quota_"
    "or_gate_change_count"
)

_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID_RE = re.compile(r"bioasq-work-v2-[0-9a-f]{64}\Z")


class BioasqP1FormalControllerError(RuntimeError):
    """The frozen one-shot formal lifecycle failed closed."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BioasqP1FormalControllerError(
            "formal value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value:
        raise BioasqP1FormalControllerError("self hash already exists")
    body = dict(value)
    body["self_sha256"] = stable_hash(body)
    return body


def _hex64(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise BioasqP1FormalControllerError(
            f"{field_name} is not a SHA-256 digest"
        )
    return value


def _work_id(value: object) -> str:
    if not isinstance(value, str) or _WORK_ID_RE.fullmatch(value) is None:
        raise BioasqP1FormalControllerError(
            "work ID is not the frozen opaque form"
        )
    return value


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


def _question_sha256(question_text: str) -> str:
    return hashlib.sha256(question_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AcquisitionClaim:
    """Safe commitments made before any formal public value is released."""

    source_identity_commitment: str
    corpus_selection_commitment: str
    block_disjointness_commitment: str
    source_qualification_commitment: str
    claim_sha256: str

    def __post_init__(self) -> None:
        body = {
            "block_disjointness_commitment": _hex64(
                self.block_disjointness_commitment,
                field_name="block disjointness commitment",
            ),
            "corpus_selection_commitment": _hex64(
                self.corpus_selection_commitment,
                field_name="corpus selection commitment",
            ),
            "schema": f"{VERSION}_acquisition_claim_v1",
            "source_identity_commitment": _hex64(
                self.source_identity_commitment,
                field_name="source identity commitment",
            ),
            "source_qualification_commitment": _hex64(
                self.source_qualification_commitment,
                field_name="source qualification commitment",
            ),
            "study_id": STUDY_ID,
        }
        if not hmac.compare_digest(
            _hex64(self.claim_sha256, field_name="acquisition claim"),
            stable_hash(body),
        ):
            raise BioasqP1FormalControllerError(
                "acquisition claim binding drifted"
            )

    @classmethod
    def create(
        cls,
        *,
        source_identity_commitment: str,
        corpus_selection_commitment: str,
        block_disjointness_commitment: str,
        source_qualification_commitment: str,
    ) -> "AcquisitionClaim":
        body = {
            "block_disjointness_commitment": (
                block_disjointness_commitment
            ),
            "corpus_selection_commitment": corpus_selection_commitment,
            "schema": f"{VERSION}_acquisition_claim_v1",
            "source_identity_commitment": source_identity_commitment,
            "source_qualification_commitment": (
                source_qualification_commitment
            ),
            "study_id": STUDY_ID,
        }
        return cls(
            source_identity_commitment=source_identity_commitment,
            corpus_selection_commitment=corpus_selection_commitment,
            block_disjointness_commitment=block_disjointness_commitment,
            source_qualification_commitment=(
                source_qualification_commitment
            ),
            claim_sha256=stable_hash(body),
        )


@dataclass(frozen=True, slots=True)
class FormalItemView:
    """The exact per-item projection visible to action-side components."""

    work_id: str
    question_text: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        try:
            canonical = core.validate_query_text(self.question_text)
        except core.BioasqP1TypedCoreError as exc:
            raise BioasqP1FormalControllerError(
                "public question violates the typed query contract"
            ) from exc
        if canonical != self.question_text:
            raise BioasqP1FormalControllerError(
                "public question is not the exact canonical text"
            )

    def private_payload(self) -> dict[str, object]:
        return {
            "question_text": self.question_text,
            "work_id": self.work_id,
        }


def _block_payload(
    block: str,
    items: Sequence[FormalItemView],
) -> dict[str, object]:
    return {
        "block": block,
        "items": [item.private_payload() for item in items],
    }


@dataclass(frozen=True, slots=True)
class BlockView:
    """A label-free block; family and qrels are absent by construction."""

    block: str
    items: tuple[FormalItemView, ...]
    view_sha256: str

    def __post_init__(self) -> None:
        if self.block not in BLOCK_COUNTS:
            raise BioasqP1FormalControllerError("block name drifted")
        if (
            not isinstance(self.items, tuple)
            or not self.items
            or any(not isinstance(item, FormalItemView) for item in self.items)
            or self.items
            != tuple(sorted(self.items, key=lambda item: item.work_id))
            or len({item.work_id for item in self.items}) != len(self.items)
        ):
            raise BioasqP1FormalControllerError(
                "block item registry drifted"
            )
        expected = stable_hash(_block_payload(self.block, self.items))
        if not hmac.compare_digest(
            _hex64(self.view_sha256, field_name="block view"),
            expected,
        ):
            raise BioasqP1FormalControllerError(
                "block view binding drifted"
            )

    @classmethod
    def create(
        cls,
        block: str,
        items: Sequence[FormalItemView],
    ) -> "BlockView":
        checked = tuple(sorted(tuple(items), key=lambda item: item.work_id))
        return cls(
            block=block,
            items=checked,
            view_sha256=stable_hash(_block_payload(block, checked)),
        )


def _corpus_payload(
    passages: Sequence[core.Passage],
) -> dict[str, object]:
    return {
        "passages": [
            core.passage_public_payload(passage) for passage in passages
        ]
    }


@dataclass(frozen=True, slots=True)
class CorpusView:
    """The exact ordinal/text corpus shared by all four measured arms."""

    passages: tuple[core.Passage, ...]
    view_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.passages, tuple)
            or len(self.passages) != CORPUS_SIZE
            or any(
                not isinstance(passage, core.Passage)
                for passage in self.passages
            )
            or tuple(passage.ordinal for passage in self.passages)
            != tuple(range(CORPUS_SIZE))
        ):
            raise BioasqP1FormalControllerError(
                "global public corpus drifted"
            )
        if not hmac.compare_digest(
            _hex64(self.view_sha256, field_name="corpus view"),
            stable_hash(_corpus_payload(self.passages)),
        ):
            raise BioasqP1FormalControllerError(
                "global public corpus binding drifted"
            )

    @classmethod
    def create(
        cls,
        passages: Sequence[core.Passage],
    ) -> "CorpusView":
        checked = tuple(passages)
        return cls(
            passages=checked,
            view_sha256=stable_hash(_corpus_payload(checked)),
        )

    @property
    def projection_sha256(self) -> str:
        return core.stable_hash(
            [
                core.passage_public_payload(passage)
                for passage in self.passages
            ]
        )


@dataclass(frozen=True, slots=True)
class CoordinateScoreRow:
    """The six frozen integer coordinate vectors for one public question."""

    work_id: str
    normalized_query_sha256: str
    corpus_projection_sha256: str
    score_vectors: Mapping[str, tuple[int, ...]]
    score_bundle_sha256: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        _hex64(self.normalized_query_sha256, field_name="coordinate query")
        _hex64(self.corpus_projection_sha256, field_name="coordinate corpus")
        if set(self.score_vectors) != set(core.SCORE_NAMES):
            raise BioasqP1FormalControllerError(
                "coordinate score registry drifted"
            )
        checked: dict[str, tuple[int, ...]] = {}
        for name in core.SCORE_NAMES:
            values = self.score_vectors[name]
            if (
                not isinstance(values, tuple)
                or len(values) != CORPUS_SIZE
                or any(
                    type(value) is not int
                    or abs(value) > core.MAX_SCORE_ABS
                    for value in values
                )
            ):
                raise BioasqP1FormalControllerError(
                    "coordinate score vector drifted"
                )
            checked[name] = values
        expected = core.stable_hash(
            {
                "ordinals": list(range(CORPUS_SIZE)),
                "scores": {
                    name: list(checked[name])
                    for name in core.SCORE_NAMES
                },
            }
        )
        if not hmac.compare_digest(
            _hex64(
                self.score_bundle_sha256,
                field_name="coordinate score bundle",
            ),
            expected,
        ):
            raise BioasqP1FormalControllerError(
                "coordinate score bundle drifted"
            )

    @classmethod
    def create(
        cls,
        *,
        item: FormalItemView,
        corpus: CorpusView,
        score_vectors: Mapping[str, Sequence[int]],
    ) -> "CoordinateScoreRow":
        checked = {
            name: tuple(score_vectors[name]) for name in core.SCORE_NAMES
        }
        score_hash = core.stable_hash(
            {
                "ordinals": list(range(CORPUS_SIZE)),
                "scores": {
                    name: list(checked[name])
                    for name in core.SCORE_NAMES
                },
            }
        )
        return cls(
            work_id=item.work_id,
            normalized_query_sha256=_question_sha256(item.question_text),
            corpus_projection_sha256=corpus.projection_sha256,
            score_vectors=checked,
            score_bundle_sha256=score_hash,
        )


@dataclass(frozen=True, slots=True)
class HippoResult:
    """One official-HippoRAG top-five result over the frozen corpus."""

    work_id: str
    normalized_query_sha256: str
    corpus_projection_sha256: str
    top5_ordinals: tuple[int, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        _hex64(self.normalized_query_sha256, field_name="HippoRAG query")
        _hex64(self.corpus_projection_sha256, field_name="HippoRAG corpus")
        _hex64(self.receipt_sha256, field_name="HippoRAG receipt")
        if (
            not isinstance(self.top5_ordinals, tuple)
            or len(self.top5_ordinals) != core.TOP_K
            or len(set(self.top5_ordinals)) != core.TOP_K
            or any(
                type(value) is not int
                or not 0 <= value < CORPUS_SIZE
                for value in self.top5_ordinals
            )
        ):
            raise BioasqP1FormalControllerError(
                "HippoRAG top-five drifted"
            )


@dataclass(frozen=True, slots=True)
class QrelRow:
    """One late, set-valued relevance row plus result-stratification family."""

    work_id: str
    family: str
    gold_ordinals: tuple[int, ...]
    corpus_projection_sha256: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        if self.family not in FAMILIES:
            raise BioasqP1FormalControllerError("qrel family drifted")
        if (
            not isinstance(self.gold_ordinals, tuple)
            or not self.gold_ordinals
            or self.gold_ordinals != tuple(sorted(self.gold_ordinals))
            or len(set(self.gold_ordinals)) != len(self.gold_ordinals)
            or any(
                type(value) is not int
                or not 0 <= value < CORPUS_SIZE
                for value in self.gold_ordinals
            )
        ):
            raise BioasqP1FormalControllerError(
                "set-valued qrels drifted"
            )
        _hex64(self.corpus_projection_sha256, field_name="qrel corpus")


def _qrel_pack_payload(
    block: str,
    action_archive_sha256: str,
    rows: Sequence[QrelRow],
) -> dict[str, object]:
    return {
        "action_archive_sha256": action_archive_sha256,
        "block": block,
        "rows": [
            {
                "corpus_projection_sha256": row.corpus_projection_sha256,
                "family": row.family,
                "gold_ordinals": list(row.gold_ordinals),
                "work_id": row.work_id,
            }
            for row in rows
        ],
    }


@dataclass(frozen=True, slots=True)
class QrelPack:
    """Late qrels cryptographically bound to the already sealed actions."""

    block: str
    action_archive_sha256: str
    rows: tuple[QrelRow, ...]
    pack_sha256: str

    def __post_init__(self) -> None:
        if self.block not in {"A_form", "A_hold", "M_search"}:
            raise BioasqP1FormalControllerError("qrel block drifted")
        archive = _hex64(
            self.action_archive_sha256,
            field_name="qrel action archive",
        )
        if (
            not isinstance(self.rows, tuple)
            or not self.rows
            or any(not isinstance(row, QrelRow) for row in self.rows)
            or self.rows
            != tuple(sorted(self.rows, key=lambda row: row.work_id))
            or len({row.work_id for row in self.rows}) != len(self.rows)
        ):
            raise BioasqP1FormalControllerError(
                "qrel pack row registry drifted"
            )
        expected = stable_hash(
            _qrel_pack_payload(self.block, archive, self.rows)
        )
        if not hmac.compare_digest(
            _hex64(self.pack_sha256, field_name="qrel pack"),
            expected,
        ):
            raise BioasqP1FormalControllerError(
                "qrel pack binding drifted"
            )

    @classmethod
    def create(
        cls,
        *,
        block: str,
        action_archive_sha256: str,
        rows: Sequence[QrelRow],
    ) -> "QrelPack":
        checked = tuple(sorted(tuple(rows), key=lambda row: row.work_id))
        return cls(
            block=block,
            action_archive_sha256=action_archive_sha256,
            rows=checked,
            pack_sha256=stable_hash(
                _qrel_pack_payload(
                    block,
                    action_archive_sha256,
                    checked,
                )
            ),
        )


class FormalAcquisitionBoundary(Protocol):
    """Trusted source holder; pre-action methods return only public types."""

    def claim_formal_attempt(
        self,
        formal_marker_sha256: str,
    ) -> AcquisitionClaim: ...

    def load_public_corpus(
        self,
        claim: AcquisitionClaim,
    ) -> CorpusView: ...

    def load_label_free_block(
        self,
        block: str,
        authorization: Mapping[str, object] | None = None,
    ) -> BlockView: ...

    def release_qrels_after_action_seal(
        self,
        block: str,
        custody_path: Path,
        sealed_action_archive: Mapping[str, object],
    ) -> QrelPack: ...


class GlobalCoordinateScorer(Protocol):
    """Frozen scorer receiving only public questions and ordinal/text corpus."""

    def score(
        self,
        corpus: CorpusView,
        items: Sequence[FormalItemView],
    ) -> Sequence[CoordinateScoreRow]: ...


class OfficialHippoRunner(Protocol):
    """Build-once official HippoRAG lane over the same public corpus."""

    def retrieve(
        self,
        corpus: CorpusView,
        items: Sequence[FormalItemView],
    ) -> Sequence[HippoResult]: ...


@dataclass(frozen=True, slots=True)
class ExactPairedComparison:
    item_count: int
    positive_count: int
    negative_count: int
    tie_count: int
    net_utility: int
    one_sided_exact_magnitude_preserving_tail: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.item_count) is not int
            or self.item_count <= 0
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.positive_count,
                    self.negative_count,
                    self.tie_count,
                )
            )
            or self.positive_count
            + self.negative_count
            + self.tie_count
            != self.item_count
            or type(self.net_utility) is not int
            or not isinstance(
                self.one_sided_exact_magnitude_preserving_tail,
                Fraction,
            )
            or not 0
            <= self.one_sided_exact_magnitude_preserving_tail
            <= 1
        ):
            raise BioasqP1FormalControllerError(
                "exact comparison drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "item_count": self.item_count,
            "negative_count": self.negative_count,
            "net_utility": self.net_utility,
            "one_sided_exact_magnitude_preserving_tail": (
                _fraction_payload(
                    self.one_sided_exact_magnitude_preserving_tail
                )
            ),
            "positive_count": self.positive_count,
            "tie_count": self.tie_count,
        }


def compare_paired_integer_utility(
    left: Sequence[int],
    right: Sequence[int],
) -> ExactPairedComparison:
    """Exact one-sided sign-flip test preserving every nonzero magnitude."""

    if (
        isinstance(left, (str, bytes))
        or isinstance(right, (str, bytes))
        or not left
        or len(left) != len(right)
        or any(type(value) is not int for value in (*left, *right))
    ):
        raise BioasqP1FormalControllerError(
            "paired integer utility vectors drifted"
        )
    differences = tuple(a - b for a, b in zip(left, right))
    magnitude_counts = Counter(abs(value) for value in differences if value)
    distribution: dict[int, int] = {0: 1}
    for magnitude, multiplicity in sorted(magnitude_counts.items()):
        signed_counts = {
            (2 * positives - multiplicity) * magnitude: math.comb(
                multiplicity, positives
            )
            for positives in range(multiplicity + 1)
        }
        updated: dict[int, int] = {}
        for total, total_count in distribution.items():
            for signed, signed_count in signed_counts.items():
                value = total + signed
                updated[value] = (
                    updated.get(value, 0) + total_count * signed_count
                )
        distribution = updated
    observed = sum(differences)
    tail_count = sum(
        count for total, count in distribution.items() if total >= observed
    )
    denominator = 1 << sum(magnitude_counts.values())
    return ExactPairedComparison(
        item_count=len(differences),
        positive_count=sum(value > 0 for value in differences),
        negative_count=sum(value < 0 for value in differences),
        tie_count=sum(value == 0 for value in differences),
        net_utility=observed,
        one_sided_exact_magnitude_preserving_tail=Fraction(
            tail_count,
            denominator,
        ),
    )


def set_recall_first_rr_utility(
    top5_ordinals: Sequence[int],
    gold_ordinals: Sequence[int],
) -> int:
    """Banker-rounded set Recall@5 plus first-relevant RR@5."""

    if (
        isinstance(top5_ordinals, (str, bytes))
        or isinstance(gold_ordinals, (str, bytes))
        or len(top5_ordinals) != core.TOP_K
        or len(set(top5_ordinals)) != core.TOP_K
        or not gold_ordinals
        or len(set(gold_ordinals)) != len(gold_ordinals)
        or any(
            type(value) is not int
            or not 0 <= value < CORPUS_SIZE
            for value in (*top5_ordinals, *gold_ordinals)
        )
    ):
        raise BioasqP1FormalControllerError(
            "set utility inputs drifted"
        )
    gold = frozenset(gold_ordinals)
    hit_count = len(gold.intersection(top5_ordinals))
    recall = round(Fraction(RECALL_SCALE * hit_count, len(gold)))
    first_rank = next(
        (
            rank
            for rank, ordinal in enumerate(top5_ordinals, start=1)
            if ordinal in gold
        ),
        None,
    )
    reciprocal_rank = (
        0
        if first_rank is None
        else round(Fraction(RECIPROCAL_RANK_SCALE, first_rank))
    )
    result = recall + reciprocal_rank
    if not 0 <= result <= MAX_SET_UTILITY:
        raise BioasqP1FormalControllerError(
            "set utility escaped its frozen range"
        )
    return result


@dataclass(frozen=True)
class _SealedFile:
    path: Path
    self_sha256: str
    file_sha256: str
    value: Mapping[str, object]


@dataclass(frozen=True)
class _ActionRow:
    item: FormalItemView
    coordinate: CoordinateScoreRow
    slate: core.ActionSlate
    raw: tuple[int, ...] | None = None
    e0: core.PolicyDecision | None = None
    e1: core.PolicyDecision | None = None
    hippo: HippoResult | None = None


@dataclass(frozen=True)
class _ScoredBlock:
    e1_e0: ExactPairedComparison
    e1_raw: ExactPairedComparison
    e1_hippo: ExactPairedComparison
    family_e1_raw: Mapping[str, ExactPairedComparison]
    family_e1_hippo: Mapping[str, ExactPairedComparison]
    arm_total_utility: Mapping[str, int]
    action_set_difference_count: Mapping[str, int]
    score_archive: _SealedFile


def _exclusive_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o400,
) -> None:
    if path.parent.is_symlink():
        raise BioasqP1FormalControllerError(
            "archive parent is a symlink"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, mode)
    except FileExistsError as exc:
        raise BioasqP1FormalControllerError(
            f"one-shot archive already exists: {path.name}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _seal_json(
    root: Path,
    filename: str,
    body: Mapping[str, object],
) -> _SealedFile:
    value = self_hashed(body)
    raw = canonical_bytes(value)
    path = root / filename
    _exclusive_bytes(path, raw)
    return _SealedFile(
        path=path,
        self_sha256=str(value["self_sha256"]),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        value=value,
    )


def _validate_block(block: BlockView, *, expected: str) -> None:
    if (
        not isinstance(block, BlockView)
        or block.block != expected
        or len(block.items) != BLOCK_COUNTS[expected]
    ):
        raise BioasqP1FormalControllerError(
            f"{expected} label-free block count drifted"
        )


def _validate_public_disjointness(
    blocks: Sequence[BlockView],
) -> None:
    work_ids: set[str] = set()
    question_hashes: set[str] = set()
    for block in blocks:
        for item in block.items:
            question_hash = _question_sha256(item.question_text)
            if item.work_id in work_ids or question_hash in question_hashes:
                raise BioasqP1FormalControllerError(
                    "formal work/question groups are not block-disjoint"
                )
            work_ids.add(item.work_id)
            question_hashes.add(question_hash)


def _validate_coordinate_rows(
    *,
    items: Sequence[FormalItemView],
    corpus: CorpusView,
    rows: Sequence[CoordinateScoreRow],
) -> Mapping[str, CoordinateScoreRow]:
    if (
        not isinstance(rows, Sequence)
        or len(rows) != len(items)
        or any(not isinstance(row, CoordinateScoreRow) for row in rows)
    ):
        raise BioasqP1FormalControllerError(
            "coordinate score coverage drifted"
        )
    by_work = {row.work_id: row for row in rows}
    if len(by_work) != len(rows) or set(by_work) != {
        item.work_id for item in items
    }:
        raise BioasqP1FormalControllerError(
            "coordinate score work IDs drifted"
        )
    for item in items:
        row = by_work[item.work_id]
        if (
            row.normalized_query_sha256
            != _question_sha256(item.question_text)
            or row.corpus_projection_sha256
            != corpus.projection_sha256
        ):
            raise BioasqP1FormalControllerError(
                "coordinate score binding drifted"
            )
    return by_work


def _compile_slate(
    *,
    item: FormalItemView,
    corpus: CorpusView,
    coordinate: CoordinateScoreRow,
) -> _ActionRow:
    vectors = coordinate.score_vectors
    try:
        slate = core.build_action_slate(
            item.question_text,
            corpus.passages,
            vectors["raw_ce"],
            vectors["focus_ce"],
            vectors["dense_base"],
            vectors["dense_support"],
            vectors["dense_contrast"],
            vectors["dense_coverage"],
        )
    except core.BioasqP1TypedCoreError as exc:
        raise BioasqP1FormalControllerError(
            "typed action slate formation failed"
        ) from exc
    if (
        slate.normalized_query_sha256
        != coordinate.normalized_query_sha256
        or slate.passage_projection_sha256
        != corpus.projection_sha256
        or slate.score_bundle_sha256
        != coordinate.score_bundle_sha256
    ):
        raise BioasqP1FormalControllerError(
            "typed action slate binding drifted"
        )
    return _ActionRow(
        item=item,
        coordinate=coordinate,
        slate=slate,
    )


def _compile_block(
    *,
    block: BlockView,
    corpus: CorpusView,
    coordinate_by_work: Mapping[str, CoordinateScoreRow],
) -> tuple[_ActionRow, ...]:
    return tuple(
        _compile_slate(
            item=item,
            corpus=corpus,
            coordinate=coordinate_by_work[item.work_id],
        )
        for item in block.items
    )


def _raw_top5(row: _ActionRow) -> tuple[int, ...]:
    scores = row.coordinate.score_vectors["raw_ce"]
    return tuple(
        sorted(
            range(CORPUS_SIZE),
            key=lambda ordinal: (-scores[ordinal], ordinal),
        )[: core.TOP_K]
    )


def _validate_hippo_rows(
    *,
    block: BlockView,
    corpus: CorpusView,
    rows: Sequence[HippoResult],
) -> Mapping[str, HippoResult]:
    if (
        not isinstance(rows, Sequence)
        or len(rows) != len(block.items)
        or any(not isinstance(row, HippoResult) for row in rows)
    ):
        raise BioasqP1FormalControllerError(
            "official HippoRAG coverage drifted"
        )
    by_work = {row.work_id: row for row in rows}
    if len(by_work) != len(rows) or set(by_work) != {
        item.work_id for item in block.items
    }:
        raise BioasqP1FormalControllerError(
            "official HippoRAG work IDs drifted"
        )
    for item in block.items:
        row = by_work[item.work_id]
        if (
            row.normalized_query_sha256
            != _question_sha256(item.question_text)
            or row.corpus_projection_sha256
            != corpus.projection_sha256
        ):
            raise BioasqP1FormalControllerError(
                "official HippoRAG binding drifted"
            )
    return by_work


def _apply_four_arms(
    *,
    block: BlockView,
    compiled: Sequence[_ActionRow],
    program: core.E1Program,
    hippo_rows: Sequence[HippoResult],
    corpus: CorpusView,
) -> tuple[_ActionRow, ...]:
    hippo_by_work = _validate_hippo_rows(
        block=block,
        corpus=corpus,
        rows=hippo_rows,
    )
    result: list[_ActionRow] = []
    for row in compiled:
        try:
            e0 = core.apply_e0(row.slate, stage=block.block)
            e1 = core.apply_e1(program, row.slate, stage=block.block)
        except core.BioasqP1TypedCoreError as exc:
            raise BioasqP1FormalControllerError(
                "typed policy application failed"
            ) from exc
        result.append(
            _ActionRow(
                item=row.item,
                coordinate=row.coordinate,
                slate=row.slate,
                raw=_raw_top5(row),
                e0=e0,
                e1=e1,
                hippo=hippo_by_work[row.item.work_id],
            )
        )
    return tuple(result)


def _aform_action_body(
    block: BlockView,
    rows: Sequence[_ActionRow],
) -> dict[str, object]:
    return {
        "all_five_recipe_slates_sealed_before_qrels": True,
        "block": block.block,
        "block_view_sha256": block.view_sha256,
        "label_bearing_action_input_count": 0,
        "recipe_ids": list(core.RECIPE_IDS),
        "rows": [
            {
                "slate": row.slate.audit_payload(),
                "work_id": row.item.work_id,
            }
            for row in rows
        ],
        "schema": f"{VERSION}_A_form_private_action_archive_v1",
        "study_id": STUDY_ID,
    }


def _four_arm_action_body(
    block: BlockView,
    rows: Sequence[_ActionRow],
    *,
    model_sha256: str,
) -> dict[str, object]:
    private_rows: list[dict[str, object]] = []
    for row in rows:
        if (
            row.raw is None
            or row.e0 is None
            or row.e1 is None
            or row.hippo is None
        ):
            raise BioasqP1FormalControllerError(
                "four-arm action row is incomplete"
            )
        private_rows.append(
            {
                "arms": {
                    "E0": list(row.e0.top5_ordinals),
                    "E1": list(row.e1.top5_ordinals),
                    "HippoRAG": list(row.hippo.top5_ordinals),
                    "RAW": list(row.raw),
                },
                "decision_binding": {
                    "E0": row.e0.decision_digest,
                    "E1": row.e1.decision_digest,
                    "HippoRAG": row.hippo.receipt_sha256,
                    "RAW": stable_hash(list(row.raw)),
                },
                "score_bundle_sha256": row.coordinate.score_bundle_sha256,
                "work_id": row.item.work_id,
            }
        )
    return {
        "E1_model_sha256": model_sha256,
        "block": block.block,
        "block_view_sha256": block.view_sha256,
        "four_arms_sealed_before_qrels": True,
        "label_bearing_action_input_count": 0,
        "rows": private_rows,
        "schema": (
            f"{VERSION}_{block.block}_private_four_arm_action_archive_v1"
        ),
        "study_id": STUDY_ID,
    }


def _validate_qrels(
    *,
    pack: QrelPack,
    block: BlockView,
    corpus: CorpusView,
    action_archive_sha256: str,
) -> Mapping[str, QrelRow]:
    if (
        not isinstance(pack, QrelPack)
        or pack.block != block.block
        or pack.action_archive_sha256 != action_archive_sha256
        or len(pack.rows) != len(block.items)
    ):
        raise BioasqP1FormalControllerError(
            "late qrel pack binding drifted"
        )
    by_work = {row.work_id: row for row in pack.rows}
    if len(by_work) != len(pack.rows) or set(by_work) != {
        item.work_id for item in block.items
    }:
        raise BioasqP1FormalControllerError(
            "late qrel work coverage drifted"
        )
    family_counts = Counter(row.family for row in pack.rows)
    if family_counts != Counter(
        {
            family: FAMILY_COUNTS[block.block]
            for family in FAMILIES
        }
    ):
        raise BioasqP1FormalControllerError(
            "late qrel family quota drifted"
        )
    if any(
        row.corpus_projection_sha256 != corpus.projection_sha256
        for row in pack.rows
    ):
        raise BioasqP1FormalControllerError(
            "late qrel corpus binding drifted"
        )
    return by_work


def _seal_qrel_pack(root: Path, pack: QrelPack) -> _SealedFile:
    return _seal_json(
        root,
        f"{pack.block}.qrels.private.json",
        {
            **_qrel_pack_payload(
                pack.block,
                pack.action_archive_sha256,
                pack.rows,
            ),
            "pack_sha256": pack.pack_sha256,
            "schema": f"{VERSION}_{pack.block}_private_qrel_archive_v1",
            "study_id": STUDY_ID,
        },
    )


def _fit_e1_once(
    rows: Sequence[_ActionRow],
    qrels: Mapping[str, QrelRow],
) -> core.E1Program:
    examples: list[core.AFormExample] = []
    for row in rows:
        gold = qrels[row.item.work_id].gold_ordinals
        utilities = tuple(
            set_recall_first_rr_utility(
                row.slate.action(recipe_id).top5_ordinals,
                gold,
            )
            for recipe_id in core.RECIPE_IDS
        )
        try:
            examples.append(core.make_aform_example(row.slate, utilities))
        except core.BioasqP1TypedCoreError as exc:
            raise BioasqP1FormalControllerError(
                "A_form example binding failed"
            ) from exc
    try:
        return core.fit_e1(tuple(examples))
    except core.BioasqP1TypedCoreError as exc:
        raise BioasqP1FormalControllerError(
            "E1 fitting failed"
        ) from exc


def _fsearch_behavior_body(
    block: BlockView,
    rows: Sequence[_ActionRow],
    program: core.E1Program,
    *,
    model_sha256: str,
) -> dict[str, object]:
    decisions: list[core.PolicyDecision] = []
    for row in rows:
        try:
            decisions.append(
                core.apply_e1(program, row.slate, stage="F_search")
            )
        except core.BioasqP1TypedCoreError as exc:
            raise BioasqP1FormalControllerError(
                "F_search unchanged-E1 application failed"
            ) from exc
    try:
        behavior = core.summarize_e1_behavior(
            program,
            tuple(decisions),
            stage="F_search",
        )
    except core.BioasqP1TypedCoreError as exc:
        raise BioasqP1FormalControllerError(
            "F_search behavior summary failed"
        ) from exc
    return {
        "E1_model_sha256": model_sha256,
        "behavior": behavior.payload(),
        "block": block.block,
        "block_view_sha256": block.view_sha256,
        "decisions": [
            {
                "decision_digest": decision.decision_digest,
                "selected_recipe_id": decision.selected_recipe_id,
                "top5_ordinals": list(decision.top5_ordinals),
                "work_id": row.item.work_id,
            }
            for row, decision in zip(rows, decisions)
        ],
        "label_or_qrel_open_count": 0,
        "model_fit_or_update_count": 0,
        "schema": f"{VERSION}_F_search_private_behavior_archive_v1",
        "study_id": STUDY_ID,
    }


def _score_four_arm_block(
    root: Path,
    block: BlockView,
    rows: Sequence[_ActionRow],
    qrels: Mapping[str, QrelRow],
) -> _ScoredBlock:
    utility: dict[str, list[int]] = {
        "E0": [],
        "E1": [],
        "HippoRAG": [],
        "RAW": [],
    }
    family_indices: dict[str, list[int]] = {
        family: [] for family in FAMILIES
    }
    action_difference = {
        "E1_vs_E0": 0,
        "E1_vs_HippoRAG": 0,
        "E1_vs_RAW": 0,
    }
    private_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if (
            row.e0 is None
            or row.e1 is None
            or row.hippo is None
            or row.raw is None
        ):
            raise BioasqP1FormalControllerError(
                "four-arm row is incomplete at score time"
            )
        qrel = qrels[row.item.work_id]
        arm_ordinals = {
            "E0": row.e0.top5_ordinals,
            "E1": row.e1.top5_ordinals,
            "HippoRAG": row.hippo.top5_ordinals,
            "RAW": row.raw,
        }
        row_utility = {
            arm: set_recall_first_rr_utility(
                ordinals,
                qrel.gold_ordinals,
            )
            for arm, ordinals in arm_ordinals.items()
        }
        for arm, value in row_utility.items():
            utility[arm].append(value)
        for other in ("E0", "HippoRAG", "RAW"):
            action_difference[f"E1_vs_{other}"] += int(
                frozenset(arm_ordinals["E1"])
                != frozenset(arm_ordinals[other])
            )
        family_indices[qrel.family].append(index)
        private_rows.append(
            {
                "family": qrel.family,
                "utilities": row_utility,
                "work_id": row.item.work_id,
            }
        )
    e1_e0 = compare_paired_integer_utility(
        utility["E1"], utility["E0"]
    )
    e1_raw = compare_paired_integer_utility(
        utility["E1"], utility["RAW"]
    )
    e1_hippo = compare_paired_integer_utility(
        utility["E1"], utility["HippoRAG"]
    )
    family_raw = {
        family: compare_paired_integer_utility(
            [utility["E1"][index] for index in indices],
            [utility["RAW"][index] for index in indices],
        )
        for family, indices in family_indices.items()
    }
    family_hippo = {
        family: compare_paired_integer_utility(
            [utility["E1"][index] for index in indices],
            [utility["HippoRAG"][index] for index in indices],
        )
        for family, indices in family_indices.items()
    }
    totals = {arm: sum(values) for arm, values in utility.items()}
    archive = _seal_json(
        root,
        f"{block.block}.scores.private.json",
        {
            "action_set_difference_count": action_difference,
            "aggregate": {
                "E1_minus_E0": e1_e0.payload(),
                "E1_minus_HippoRAG": e1_hippo.payload(),
                "E1_minus_RAW": e1_raw.payload(),
            },
            "arm_total_integer_set_Recall5_plus_first_RR5_utility": totals,
            "block": block.block,
            "family": {
                family: {
                    "E1_minus_HippoRAG": family_hippo[family].payload(),
                    "E1_minus_RAW": family_raw[family].payload(),
                }
                for family in FAMILIES
            },
            "rows": private_rows,
            "schema": (
                f"{VERSION}_{block.block}_private_score_archive_v1"
            ),
            "study_id": STUDY_ID,
            "utility_contract": {
                "maximum": MAX_SET_UTILITY,
                "recall_at_5_scale": RECALL_SCALE,
                "recall_rounding": "round_half_to_even",
                "reciprocal_rank_at_5_scale": RECIPROCAL_RANK_SCALE,
                "reciprocal_rank_rounding": "round_half_to_even",
                "set_relevance": True,
            },
        },
    )
    return _ScoredBlock(
        e1_e0=e1_e0,
        e1_raw=e1_raw,
        e1_hippo=e1_hippo,
        family_e1_raw=family_raw,
        family_e1_hippo=family_hippo,
        arm_total_utility=totals,
        action_set_difference_count=action_difference,
        score_archive=archive,
    )


def _comparison_pass(value: ExactPairedComparison) -> bool:
    return (
        value.net_utility > 0
        and value.one_sided_exact_magnitude_preserving_tail <= ALPHA
    )


def _stable_reality_families(
    value: _ScoredBlock,
) -> tuple[str, ...]:
    return tuple(
        family
        for family in FAMILIES
        if value.family_e1_raw[family].net_utility > 0
        and value.family_e1_hippo[family].net_utility > 0
    )


def _safe_block_result(value: _ScoredBlock) -> dict[str, object]:
    stable_families = _stable_reality_families(value)
    reality = (
        _comparison_pass(value.e1_raw)
        and _comparison_pass(value.e1_hippo)
        and len(stable_families) >= REALITY_MINIMUM_STABLE_FAMILIES
    )
    return {
        "action_set_difference_count": dict(
            value.action_set_difference_count
        ),
        "aggregate": {
            "E1_minus_E0": value.e1_e0.payload(),
            "E1_minus_HippoRAG": value.e1_hippo.payload(),
            "E1_minus_RAW": value.e1_raw.payload(),
        },
        "arm_total_integer_set_Recall5_plus_first_RR5_utility": dict(
            value.arm_total_utility
        ),
        "family": {
            family: {
                "E1_minus_HippoRAG": (
                    value.family_e1_hippo[family].payload()
                ),
                "E1_minus_RAW": value.family_e1_raw[family].payload(),
                "strictly_positive_against_both": (
                    family in stable_families
                ),
            }
            for family in FAMILIES
        },
        "promotion_passed": _comparison_pass(value.e1_e0),
        "reality_primary_passed": reality,
        "stable_family_count": len(stable_families),
        "stable_family_minimum": REALITY_MINIMUM_STABLE_FAMILIES,
    }


def _promotion_authorization(
    *,
    claim: AcquisitionClaim,
    comparison: ExactPairedComparison,
) -> dict[str, object]:
    return self_hashed(
        {
            "A_hold_E1_minus_E0": comparison.payload(),
            "block_disjointness_commitment": (
                claim.block_disjointness_commitment
            ),
            "comparison_net_strictly_positive": (
                comparison.net_utility > 0
            ),
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth": (
                comparison.one_sided_exact_magnitude_preserving_tail
                <= ALPHA
            ),
            "schema": f"{VERSION}_M_search_materialization_authorization_v1",
            "status": "A_hold_E1_promoted",
            "study_id": STUDY_ID,
        }
    )


def _write_failure_terminal(
    root: Path,
    *,
    execution_binding_sha256: str,
    marker_sha256: str,
    stage: str,
    exc: Exception,
) -> None:
    body = {
        "aggregate_only_public_terminal": True,
        "execution_binding_sha256": execution_binding_sha256,
        "failure_exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8", errors="replace")
        ).hexdigest(),
        "failure_exception_type_sha256": hashlib.sha256(
            type(exc).__name__.encode("ascii", errors="replace")
        ).hexdigest(),
        "failure_stage": stage,
        "formal_marker_sha256": marker_sha256,
        "item_query_document_qrel_action_or_per_item_score_values_published": (
            False
        ),
        "online_or_API_evaluator_calls": 0,
        NO_CHANGE_COUNT_KEY: 0,
        "schema": f"{VERSION}_safe_failure_terminal_v1",
        "status": "terminal_formal_failure_no_retry",
        "study_id": STUDY_ID,
    }
    try:
        _seal_json(root, FORMAL_TERMINAL_FILENAME, body)
    except Exception:
        pass


def _terminal_body(
    *,
    execution_binding_sha256: str,
    marker_sha256: str,
    claim: AcquisitionClaim,
    archives: Mapping[str, _SealedFile],
    ahold_score: _ScoredBlock,
    m_score: _ScoredBlock | None,
    protocol_call_counts: Mapping[str, int],
) -> dict[str, object]:
    promotion = _comparison_pass(ahold_score.e1_e0)
    return {
        "A_hold": _safe_block_result(ahold_score),
        "F_search": {
            "label_or_qrel_open_count": 0,
            "model_fit_or_update_count": 0,
            "private_behavior_archive_sha256": (
                archives["F_search_behavior"].self_sha256
            ),
        },
        "M_search": (
            {
                "L5_E1_minus_E0": m_score.e1_e0.payload(),
                "L5_passed": _comparison_pass(m_score.e1_e0),
                "opened_after_promotion": True,
            }
            if m_score is not None
            else {
                "L5_E1_minus_E0": None,
                "L5_passed": None,
                "opened_after_promotion": False,
            }
        ),
        "acquisition_claim_sha256": claim.claim_sha256,
        "aggregate_only_public_terminal": True,
        "archive_commitments": {
            name: archive.self_sha256
            for name, archive in sorted(archives.items())
        },
        "execution_binding_sha256": execution_binding_sha256,
        "formal_marker_sha256": marker_sha256,
        "formal_protocol_call_counts": dict(protocol_call_counts),
        "item_query_document_qrel_action_or_per_item_score_values_published": (
            False
        ),
        "online_or_API_evaluator_calls": 0,
        NO_CHANGE_COUNT_KEY: 0,
        "schema": f"{VERSION}_safe_terminal_v1",
        "status": (
            "terminal_complete_after_A_hold_promotion_and_M_search"
            if promotion
            else "terminal_A_hold_E1_not_promoted_M_search_unopened"
        ),
        "study_id": STUDY_ID,
    }


def run_formal_controller(
    *,
    work_root: Path,
    execution_binding_sha256: str,
    acquisition: FormalAcquisitionBoundary,
    coordinate_scorer: GlobalCoordinateScorer,
    hippo_runner: OfficialHippoRunner,
) -> Mapping[str, object]:
    """Execute the frozen formal attempt exactly once."""

    binding = _hex64(
        execution_binding_sha256,
        field_name="execution binding",
    )
    root = Path(work_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise BioasqP1FormalControllerError("formal work root is unsafe")
    os.chmod(root, 0o700)
    marker = self_hashed(
        {
            "execution_binding_sha256": binding,
            NO_CHANGE_COUNT_KEY: 0,
            "schema": f"{VERSION}_one_shot_marker_v1",
            "study_id": STUDY_ID,
        }
    )
    _exclusive_bytes(
        root / FORMAL_MARKER_FILENAME,
        canonical_bytes(marker),
    )
    marker_sha256 = str(marker["self_sha256"])
    call_counts: Counter[str] = Counter()
    stage = "claim_acquisition"
    try:
        claim = acquisition.claim_formal_attempt(marker_sha256)
        call_counts["acquisition_claim"] += 1
        if not isinstance(claim, AcquisitionClaim):
            raise BioasqP1FormalControllerError(
                "acquisition claim type drifted"
            )
        stage = "load_global_public_corpus"
        corpus = acquisition.load_public_corpus(claim)
        call_counts["public_corpus_load"] += 1
        if not isinstance(corpus, CorpusView):
            raise BioasqP1FormalControllerError(
                "public corpus type drifted"
            )

        initial_blocks: dict[str, BlockView] = {}
        for block_name in INITIAL_BLOCKS:
            stage = f"load_{block_name}_label_free"
            block = acquisition.load_label_free_block(block_name, None)
            call_counts[f"{block_name}_public_block_load"] += 1
            _validate_block(block, expected=block_name)
            initial_blocks[block_name] = block
        _validate_public_disjointness(
            tuple(initial_blocks[name] for name in INITIAL_BLOCKS)
        )

        initial_items = tuple(
            item
            for block_name in INITIAL_BLOCKS
            for item in initial_blocks[block_name].items
        )
        stage = "run_initial_coordinate_and_A_hold_Hippo_lanes"
        with ThreadPoolExecutor(max_workers=2) as pool:
            coordinate_future = pool.submit(
                coordinate_scorer.score,
                corpus,
                initial_items,
            )
            hippo_future = pool.submit(
                hippo_runner.retrieve,
                corpus,
                initial_blocks["A_hold"].items,
            )
            coordinate_rows = coordinate_future.result()
            hippo_ahold_rows = hippo_future.result()
        call_counts["coordinate_score_invocations"] += 1
        call_counts["official_HippoRAG_retrieve_invocations"] += 1
        coordinate_by_work = _validate_coordinate_rows(
            items=initial_items,
            corpus=corpus,
            rows=coordinate_rows,
        )

        compiled = {
            block_name: _compile_block(
                block=initial_blocks[block_name],
                corpus=corpus,
                coordinate_by_work=coordinate_by_work,
            )
            for block_name in INITIAL_BLOCKS
        }
        archives: dict[str, _SealedFile] = {}

        stage = "seal_A_form_all_five_recipe_slates"
        aform_action = _seal_json(
            root,
            "A_form.actions.private.json",
            _aform_action_body(
                initial_blocks["A_form"],
                compiled["A_form"],
            ),
        )
        archives["A_form_actions"] = aform_action

        stage = "release_A_form_qrels_after_action_seal"
        aform_pack = acquisition.release_qrels_after_action_seal(
            "A_form",
            aform_action.path,
            aform_action.value,
        )
        call_counts["qrel_release_invocations"] += 1
        aform_qrels = _validate_qrels(
            pack=aform_pack,
            block=initial_blocks["A_form"],
            corpus=corpus,
            action_archive_sha256=aform_action.self_sha256,
        )
        archives["A_form_qrels"] = _seal_qrel_pack(root, aform_pack)

        stage = "fit_and_seal_E1_exactly_once"
        program = _fit_e1_once(compiled["A_form"], aform_qrels)
        call_counts["E1_fit_invocations"] += 1
        model_archive = _seal_json(
            root,
            "E1.model.private.json",
            {
                "A_form_action_archive_sha256": aform_action.self_sha256,
                "A_form_qrel_pack_sha256": aform_pack.pack_sha256,
                "fit_count": 1,
                "model": program.payload(),
                "schema": f"{VERSION}_frozen_E1_model_v1",
                "study_id": STUDY_ID,
            },
        )
        archives["E1_model"] = model_archive

        stage = "seal_F_search_unchanged_E1_behavior_without_qrels"
        archives["F_search_behavior"] = _seal_json(
            root,
            "F_search.behavior.private.json",
            _fsearch_behavior_body(
                initial_blocks["F_search"],
                compiled["F_search"],
                program,
                model_sha256=model_archive.self_sha256,
            ),
        )

        stage = "seal_A_hold_four_arms"
        ahold_rows = _apply_four_arms(
            block=initial_blocks["A_hold"],
            compiled=compiled["A_hold"],
            program=program,
            hippo_rows=hippo_ahold_rows,
            corpus=corpus,
        )
        ahold_action = _seal_json(
            root,
            "A_hold.actions.private.json",
            _four_arm_action_body(
                initial_blocks["A_hold"],
                ahold_rows,
                model_sha256=model_archive.self_sha256,
            ),
        )
        archives["A_hold_actions"] = ahold_action

        stage = "release_A_hold_qrels_after_action_seal"
        ahold_pack = acquisition.release_qrels_after_action_seal(
            "A_hold",
            ahold_action.path,
            ahold_action.value,
        )
        call_counts["qrel_release_invocations"] += 1
        ahold_qrels = _validate_qrels(
            pack=ahold_pack,
            block=initial_blocks["A_hold"],
            corpus=corpus,
            action_archive_sha256=ahold_action.self_sha256,
        )
        archives["A_hold_qrels"] = _seal_qrel_pack(root, ahold_pack)

        stage = "score_A_hold_offline"
        ahold_score = _score_four_arm_block(
            root,
            initial_blocks["A_hold"],
            ahold_rows,
            ahold_qrels,
        )
        archives["A_hold_scores"] = ahold_score.score_archive

        m_score: _ScoredBlock | None = None
        if _comparison_pass(ahold_score.e1_e0):
            stage = "seal_M_search_authorization"
            authorization = _promotion_authorization(
                claim=claim,
                comparison=ahold_score.e1_e0,
            )
            _exclusive_bytes(
                root / PROMOTION_AUTHORIZATION_FILENAME,
                canonical_bytes(authorization),
            )

            stage = "materialize_M_search_after_promotion"
            m_block = acquisition.load_label_free_block(
                "M_search",
                authorization,
            )
            call_counts["M_search_public_block_load"] += 1
            _validate_block(m_block, expected="M_search")
            _validate_public_disjointness(
                (
                    *(
                        initial_blocks[name]
                        for name in INITIAL_BLOCKS
                    ),
                    m_block,
                )
            )

            stage = "run_conditional_M_search_lanes"
            with ThreadPoolExecutor(max_workers=2) as pool:
                coordinate_future = pool.submit(
                    coordinate_scorer.score,
                    corpus,
                    m_block.items,
                )
                hippo_future = pool.submit(
                    hippo_runner.retrieve,
                    corpus,
                    m_block.items,
                )
                m_coordinate_rows = coordinate_future.result()
                m_hippo_rows = hippo_future.result()
            call_counts["coordinate_score_invocations"] += 1
            call_counts["official_HippoRAG_retrieve_invocations"] += 1
            m_coordinate_by_work = _validate_coordinate_rows(
                items=m_block.items,
                corpus=corpus,
                rows=m_coordinate_rows,
            )
            m_compiled = _compile_block(
                block=m_block,
                corpus=corpus,
                coordinate_by_work=m_coordinate_by_work,
            )
            m_rows = _apply_four_arms(
                block=m_block,
                compiled=m_compiled,
                program=program,
                hippo_rows=m_hippo_rows,
                corpus=corpus,
            )

            stage = "seal_M_search_four_arms"
            m_action = _seal_json(
                root,
                "M_search.actions.private.json",
                _four_arm_action_body(
                    m_block,
                    m_rows,
                    model_sha256=model_archive.self_sha256,
                ),
            )
            archives["M_search_actions"] = m_action

            stage = "release_M_search_qrels_after_action_seal"
            m_pack = acquisition.release_qrels_after_action_seal(
                "M_search",
                m_action.path,
                m_action.value,
            )
            call_counts["qrel_release_invocations"] += 1
            m_qrels = _validate_qrels(
                pack=m_pack,
                block=m_block,
                corpus=corpus,
                action_archive_sha256=m_action.self_sha256,
            )
            archives["M_search_qrels"] = _seal_qrel_pack(root, m_pack)

            stage = "score_M_search_L5_offline"
            m_score = _score_four_arm_block(
                root,
                m_block,
                m_rows,
                m_qrels,
            )
            archives["M_search_scores"] = m_score.score_archive

        stage = "seal_safe_aggregate_terminal"
        terminal = self_hashed(
            _terminal_body(
                execution_binding_sha256=binding,
                marker_sha256=marker_sha256,
                claim=claim,
                archives=archives,
                ahold_score=ahold_score,
                m_score=m_score,
                protocol_call_counts={
                    "A_form_public_block_load": call_counts[
                        "A_form_public_block_load"
                    ],
                    "A_hold_public_block_load": call_counts[
                        "A_hold_public_block_load"
                    ],
                    "E1_fit_invocations": call_counts[
                        "E1_fit_invocations"
                    ],
                    "F_search_public_block_load": call_counts[
                        "F_search_public_block_load"
                    ],
                    "M_search_public_block_load": call_counts[
                        "M_search_public_block_load"
                    ],
                    "acquisition_claim": call_counts[
                        "acquisition_claim"
                    ],
                    "coordinate_score_invocations": call_counts[
                        "coordinate_score_invocations"
                    ],
                    "official_HippoRAG_retrieve_invocations": call_counts[
                        "official_HippoRAG_retrieve_invocations"
                    ],
                    "public_corpus_load": call_counts[
                        "public_corpus_load"
                    ],
                    "qrel_release_invocations": call_counts[
                        "qrel_release_invocations"
                    ],
                },
            )
        )
        _exclusive_bytes(
            root / FORMAL_TERMINAL_FILENAME,
            canonical_bytes(terminal),
        )
        return terminal
    except Exception as exc:
        _write_failure_terminal(
            root,
            execution_binding_sha256=binding,
            marker_sha256=marker_sha256,
            stage=stage,
            exc=exc,
        )
        if isinstance(exc, BioasqP1FormalControllerError):
            raise
        raise BioasqP1FormalControllerError(
            "formal controller failed closed"
        ) from exc


__all__ = [
    "ALPHA",
    "AcquisitionClaim",
    "BLOCK_COUNTS",
    "BioasqP1FormalControllerError",
    "BlockView",
    "CORPUS_SIZE",
    "CoordinateScoreRow",
    "CorpusView",
    "ExactPairedComparison",
    "FAMILIES",
    "FAMILY_COUNTS",
    "FORMAL_MARKER_FILENAME",
    "FORMAL_TERMINAL_FILENAME",
    "FormalAcquisitionBoundary",
    "FormalItemView",
    "GlobalCoordinateScorer",
    "HippoResult",
    "MAX_SET_UTILITY",
    "NO_CHANGE_COUNT_KEY",
    "OfficialHippoRunner",
    "PROMOTION_AUTHORIZATION_FILENAME",
    "QrelPack",
    "QrelRow",
    "REALITY_MINIMUM_STABLE_FAMILIES",
    "RECALL_SCALE",
    "RECIPROCAL_RANK_SCALE",
    "compare_paired_integer_utility",
    "run_formal_controller",
    "self_hashed",
    "set_recall_first_rr_utility",
    "stable_hash",
]
