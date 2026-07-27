"""One-shot late-qrel formal controller for the frozen DSTC9 P1 study.

The controller is deliberately source-free.  Its acquisition boundary can
release only a public 2,900-snippet corpus and block-scoped public dialogue
histories before an action archive is durably sealed.  Gold family and the
singleton relevant snippet ordinal have a separate type and release method.

The lifecycle is fixed:

* seal A_form slates, then open A_form qrels and fit E1 exactly once;
* apply the unchanged E1 to label-free F_search and seal behavior only;
* seal A_hold E0/E1/RAW/official-HippoRAG actions, then open qrels and score;
* materialize M_search only after the frozen A_hold E1-minus-E0 promotion;
* seal M_search actions before its qrels and measure unchanged-E1 L5.

Every content-bearing archive is private mode 0400.  The returned terminal is
aggregate-only and uses an exact integer singleton Recall@5 + RR@5 utility.
There is no online or API evaluator channel in this module.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Mapping, Protocol, Sequence

from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as core


VERSION = "dstc9_p1_formal_controller_v1"
STUDY_ID = core.STUDY_ID
FAMILIES = ("hotel", "restaurant", "taxi", "train")
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

# LCM(1, 2, 3, 4, 5) makes both terms exact integers.
RANK_SCALE = 60
MAX_SINGLETON_UTILITY = 2 * RANK_SCALE

FORMAL_MARKER_FILENAME = "formal.marker.json"
FORMAL_TERMINAL_FILENAME = "formal_terminal.json"
PROMOTION_AUTHORIZATION_FILENAME = "promotion.authorization.json"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID = re.compile(r"dstc9-work-v1-[0-9a-f]{64}\Z")


class Dstc9P1FormalControllerError(RuntimeError):
    """The one-shot formal lifecycle failed closed."""


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
        raise Dstc9P1FormalControllerError(
            "formal value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value:
        raise Dstc9P1FormalControllerError("self hash already exists")
    body = dict(value)
    body["self_sha256"] = stable_hash(body)
    return body


def _hex64(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Dstc9P1FormalControllerError(
            f"{field_name} is not a SHA-256 digest"
        )
    return value


def _work_id(value: object) -> str:
    if not isinstance(value, str) or _WORK_ID.fullmatch(value) is None:
        raise Dstc9P1FormalControllerError(
            "work ID is not the frozen opaque form"
        )
    return value


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


@dataclass(frozen=True, slots=True)
class AcquisitionClaim:
    """Safe commitments made before any formal public block is released."""

    source_identity_commitment: str
    corpus_selection_commitment: str
    block_disjointness_commitment: str
    query_only_predictor_commitment: str
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
            "query_only_predictor_commitment": _hex64(
                self.query_only_predictor_commitment,
                field_name="query-only predictor commitment",
            ),
            "schema": f"{VERSION}_acquisition_claim_v1",
            "source_identity_commitment": _hex64(
                self.source_identity_commitment,
                field_name="source identity commitment",
            ),
            "study_id": STUDY_ID,
        }
        if not hmac.compare_digest(
            _hex64(self.claim_sha256, field_name="claim"),
            stable_hash(body),
        ):
            raise Dstc9P1FormalControllerError(
                "acquisition claim binding drifted"
            )

    @classmethod
    def create(
        cls,
        *,
        source_identity_commitment: str,
        corpus_selection_commitment: str,
        block_disjointness_commitment: str,
        query_only_predictor_commitment: str,
    ) -> "AcquisitionClaim":
        body = {
            "block_disjointness_commitment": (
                block_disjointness_commitment
            ),
            "corpus_selection_commitment": corpus_selection_commitment,
            "query_only_predictor_commitment": (
                query_only_predictor_commitment
            ),
            "schema": f"{VERSION}_acquisition_claim_v1",
            "source_identity_commitment": source_identity_commitment,
            "study_id": STUDY_ID,
        }
        return cls(
            source_identity_commitment=source_identity_commitment,
            corpus_selection_commitment=corpus_selection_commitment,
            block_disjointness_commitment=(
                block_disjointness_commitment
            ),
            query_only_predictor_commitment=(
                query_only_predictor_commitment
            ),
            claim_sha256=stable_hash(body),
        )


@dataclass(frozen=True, slots=True)
class FormalItemView:
    """The complete per-item value visible to action-side components."""

    work_id: str
    block: str
    history: tuple[core.DialogueTurn, ...]

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        if self.block not in BLOCK_COUNTS:
            raise Dstc9P1FormalControllerError("item block drifted")
        if (
            not isinstance(self.history, tuple)
            or not self.history
            or any(
                not isinstance(turn, core.DialogueTurn)
                for turn in self.history
            )
        ):
            raise Dstc9P1FormalControllerError(
                "public history type drifted"
            )
        try:
            core.normalized_query_payload(self.history)
        except core.Dstc9P1TypedCoreError as exc:
            raise Dstc9P1FormalControllerError(
                "public history violates the typed query contract"
            ) from exc

    def private_payload(self) -> dict[str, object]:
        return {
            "block": self.block,
            "history": [
                core.turn_public_payload(turn) for turn in self.history
            ],
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
    """A complete block containing no family, qrel, split, or source ID."""

    block: str
    items: tuple[FormalItemView, ...]
    view_sha256: str

    def __post_init__(self) -> None:
        if self.block not in BLOCK_COUNTS:
            raise Dstc9P1FormalControllerError("block name drifted")
        if (
            not isinstance(self.items, tuple)
            or not self.items
            or any(
                not isinstance(item, FormalItemView)
                or item.block != self.block
                for item in self.items
            )
            or self.items
            != tuple(sorted(self.items, key=lambda item: item.work_id))
            or len({item.work_id for item in self.items})
            != len(self.items)
        ):
            raise Dstc9P1FormalControllerError(
                "block item registry drifted"
            )
        expected = stable_hash(_block_payload(self.block, self.items))
        if not hmac.compare_digest(
            _hex64(self.view_sha256, field_name="block view"),
            expected,
        ):
            raise Dstc9P1FormalControllerError(
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
    snippets: Sequence[core.KnowledgeSnippet],
) -> dict[str, object]:
    return {
        "snippets": [
            core.snippet_public_payload(snippet) for snippet in snippets
        ]
    }


@dataclass(frozen=True, slots=True)
class CorpusView:
    """The exact global public snippet projection shared by all arms."""

    snippets: tuple[core.KnowledgeSnippet, ...]
    view_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.snippets, tuple)
            or len(self.snippets) != CORPUS_SIZE
            or any(
                not isinstance(snippet, core.KnowledgeSnippet)
                for snippet in self.snippets
            )
            or tuple(snippet.ordinal for snippet in self.snippets)
            != tuple(range(CORPUS_SIZE))
        ):
            raise Dstc9P1FormalControllerError(
                "global public corpus drifted"
            )
        if not hmac.compare_digest(
            _hex64(self.view_sha256, field_name="corpus view"),
            stable_hash(_corpus_payload(self.snippets)),
        ):
            raise Dstc9P1FormalControllerError(
                "global public corpus binding drifted"
            )

    @classmethod
    def create(
        cls,
        snippets: Sequence[core.KnowledgeSnippet],
    ) -> "CorpusView":
        checked = tuple(snippets)
        return cls(
            snippets=checked,
            view_sha256=stable_hash(_corpus_payload(checked)),
        )

    @property
    def projection_sha256(self) -> str:
        return core.stable_hash(
            [
                core.snippet_public_payload(snippet)
                for snippet in self.snippets
            ]
        )


@dataclass(frozen=True, slots=True)
class BucketPrediction:
    """A query-only predicted bucket with explicit provenance."""

    work_id: str
    block: str
    predicted_bucket: int
    normalized_query_sha256: str
    predictor_commitment: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        if self.block not in BLOCK_COUNTS:
            raise Dstc9P1FormalControllerError(
                "bucket prediction block drifted"
            )
        if (
            type(self.predicted_bucket) is not int
            or self.predicted_bucket not in core.PREDICTED_BUCKETS
        ):
            raise Dstc9P1FormalControllerError(
                "predicted bucket drifted"
            )
        body = {
            "block": self.block,
            "normalized_query_sha256": _hex64(
                self.normalized_query_sha256,
                field_name="prediction query",
            ),
            "predicted_bucket": self.predicted_bucket,
            "predictor_commitment": _hex64(
                self.predictor_commitment,
                field_name="predictor commitment",
            ),
            "schema": f"{VERSION}_query_only_bucket_prediction_v1",
            "study_id": STUDY_ID,
            "work_id": self.work_id,
        }
        if not hmac.compare_digest(
            _hex64(
                self.provenance_sha256,
                field_name="prediction provenance",
            ),
            stable_hash(body),
        ):
            raise Dstc9P1FormalControllerError(
                "prediction provenance drifted"
            )

    @classmethod
    def create(
        cls,
        *,
        item: FormalItemView,
        predicted_bucket: int,
        predictor_commitment: str,
    ) -> "BucketPrediction":
        query = core.normalized_query_sha256(item.history)
        body = {
            "block": item.block,
            "normalized_query_sha256": query,
            "predicted_bucket": predicted_bucket,
            "predictor_commitment": predictor_commitment,
            "schema": f"{VERSION}_query_only_bucket_prediction_v1",
            "study_id": STUDY_ID,
            "work_id": item.work_id,
        }
        return cls(
            work_id=item.work_id,
            block=item.block,
            predicted_bucket=predicted_bucket,
            normalized_query_sha256=query,
            predictor_commitment=predictor_commitment,
            provenance_sha256=stable_hash(body),
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "block": self.block,
            "normalized_query_sha256": self.normalized_query_sha256,
            "predicted_bucket": self.predicted_bucket,
            "predictor_commitment": self.predictor_commitment,
            "provenance_sha256": self.provenance_sha256,
            "query_only_input_contract": True,
            "work_id": self.work_id,
        }


@dataclass(frozen=True, slots=True)
class CoordinateScoreRow:
    """Six global integer score vectors for one public query."""

    work_id: str
    block: str
    normalized_query_sha256: str
    corpus_projection_sha256: str
    score_vectors: Mapping[str, tuple[int, ...]]
    score_bundle_sha256: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        if self.block not in BLOCK_COUNTS:
            raise Dstc9P1FormalControllerError(
                "coordinate score block drifted"
            )
        _hex64(
            self.normalized_query_sha256,
            field_name="coordinate query",
        )
        _hex64(
            self.corpus_projection_sha256,
            field_name="coordinate corpus",
        )
        if set(self.score_vectors) != set(core.SCORE_NAMES):
            raise Dstc9P1FormalControllerError(
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
                raise Dstc9P1FormalControllerError(
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
            raise Dstc9P1FormalControllerError(
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
            block=item.block,
            normalized_query_sha256=core.normalized_query_sha256(
                item.history
            ),
            corpus_projection_sha256=corpus.projection_sha256,
            score_vectors=checked,
            score_bundle_sha256=score_hash,
        )


@dataclass(frozen=True, slots=True)
class HippoResult:
    """The complete official-HippoRAG result visible to the controller."""

    work_id: str
    block: str
    normalized_query_sha256: str
    corpus_projection_sha256: str
    top5_ordinals: tuple[int, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        if self.block not in SCORING_BLOCKS:
            raise Dstc9P1FormalControllerError(
                "HippoRAG block drifted"
            )
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
            raise Dstc9P1FormalControllerError(
                "HippoRAG top5 drifted"
            )


@dataclass(frozen=True, slots=True)
class QrelRow:
    """One late singleton gold ordinal and result-stratification family."""

    work_id: str
    family: str
    gold_ordinal: int
    corpus_projection_sha256: str

    def __post_init__(self) -> None:
        _work_id(self.work_id)
        if self.family not in FAMILIES:
            raise Dstc9P1FormalControllerError("qrel family drifted")
        if (
            type(self.gold_ordinal) is not int
            or not 0 <= self.gold_ordinal < CORPUS_SIZE
        ):
            raise Dstc9P1FormalControllerError(
                "singleton gold ordinal drifted"
            )
        _hex64(
            self.corpus_projection_sha256,
            field_name="qrel corpus",
        )


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
                "corpus_projection_sha256": (
                    row.corpus_projection_sha256
                ),
                "family": row.family,
                "gold_ordinal": row.gold_ordinal,
                "work_id": row.work_id,
            }
            for row in rows
        ],
    }


@dataclass(frozen=True, slots=True)
class QrelPack:
    """A late qrel pack cryptographically bound to an action archive."""

    block: str
    action_archive_sha256: str
    rows: tuple[QrelRow, ...]
    pack_sha256: str

    def __post_init__(self) -> None:
        if self.block not in {
            "A_form",
            "A_hold",
            "M_search",
        }:
            raise Dstc9P1FormalControllerError("qrel block drifted")
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
            or len({row.work_id for row in self.rows})
            != len(self.rows)
        ):
            raise Dstc9P1FormalControllerError(
                "qrel pack rows drifted"
            )
        expected = stable_hash(
            _qrel_pack_payload(self.block, archive, self.rows)
        )
        if not hmac.compare_digest(
            _hex64(self.pack_sha256, field_name="qrel pack"),
            expected,
        ):
            raise Dstc9P1FormalControllerError(
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
    """Trusted source holder; its prelabel outputs use only public types."""

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


class QueryOnlyBucketPredictor(Protocol):
    """Receives no source ledger, family, entity/doc ID, qrel, or score."""

    def predict(
        self,
        items: Sequence[FormalItemView],
    ) -> Sequence[BucketPrediction]: ...


class GlobalCoordinateScorer(Protocol):
    """One frozen scorer over the same public global corpus for every arm."""

    def score(
        self,
        corpus: CorpusView,
        items: Sequence[FormalItemView],
    ) -> Sequence[CoordinateScoreRow]: ...


class OfficialHippoRunner(Protocol):
    """One build-once official HippoRAG runner over the public corpus."""

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
            raise Dstc9P1FormalControllerError(
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
        raise Dstc9P1FormalControllerError(
            "paired integer utility vectors drifted"
        )
    differences = tuple(a - b for a, b in zip(left, right))
    magnitudes = tuple(abs(value) for value in differences if value)
    distribution: dict[int, int] = {0: 1}
    for magnitude in magnitudes:
        updated: dict[int, int] = {}
        for total, count in distribution.items():
            updated[total + magnitude] = (
                updated.get(total + magnitude, 0) + count
            )
            updated[total - magnitude] = (
                updated.get(total - magnitude, 0) + count
            )
        distribution = updated
    observed = sum(differences)
    tail_count = sum(
        count for total, count in distribution.items() if total >= observed
    )
    denominator = 1 << len(magnitudes)
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


def singleton_recall_rr_utility(
    top5_ordinals: Sequence[int],
    gold_ordinal: int,
) -> int:
    """Return exact integer Recall@5 + reciprocal-rank@5 utility."""

    if (
        isinstance(top5_ordinals, (str, bytes))
        or len(top5_ordinals) != core.TOP_K
        or len(set(top5_ordinals)) != core.TOP_K
        or any(
            type(value) is not int
            or not 0 <= value < CORPUS_SIZE
            for value in top5_ordinals
        )
        or type(gold_ordinal) is not int
        or not 0 <= gold_ordinal < CORPUS_SIZE
    ):
        raise Dstc9P1FormalControllerError(
            "singleton utility input drifted"
        )
    try:
        rank = tuple(top5_ordinals).index(gold_ordinal) + 1
    except ValueError:
        return 0
    return RANK_SCALE + RANK_SCALE // rank


@dataclass(frozen=True)
class _SealedFile:
    path: Path
    self_sha256: str
    file_sha256: str
    value: Mapping[str, object]


@dataclass(frozen=True)
class _ActionRow:
    item: FormalItemView
    prediction: BucketPrediction
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise Dstc9P1FormalControllerError(
            f"exclusive formal file is unsafe: {path.name}"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short formal write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
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
        raise Dstc9P1FormalControllerError(
            f"{expected} label-free block count drifted"
        )


def _validate_public_disjointness(
    blocks: Sequence[BlockView],
) -> None:
    work_ids: set[str] = set()
    query_hashes: set[str] = set()
    for block in blocks:
        for item in block.items:
            query_hash = core.normalized_query_sha256(item.history)
            if item.work_id in work_ids or query_hash in query_hashes:
                raise Dstc9P1FormalControllerError(
                    "formal work/query groups are not block-disjoint"
                )
            work_ids.add(item.work_id)
            query_hashes.add(query_hash)


def _validate_prediction_rows(
    *,
    block: BlockView,
    claim: AcquisitionClaim,
    rows: Sequence[BucketPrediction],
) -> Mapping[str, BucketPrediction]:
    if (
        not isinstance(rows, Sequence)
        or len(rows) != len(block.items)
        or any(not isinstance(row, BucketPrediction) for row in rows)
    ):
        raise Dstc9P1FormalControllerError(
            "query-only prediction coverage drifted"
        )
    by_work = {row.work_id: row for row in rows}
    if len(by_work) != len(rows) or set(by_work) != {
        item.work_id for item in block.items
    }:
        raise Dstc9P1FormalControllerError(
            "query-only prediction work IDs drifted"
        )
    for item in block.items:
        prediction = by_work[item.work_id]
        if (
            prediction.block != block.block
            or prediction.normalized_query_sha256
            != core.normalized_query_sha256(item.history)
            or prediction.predictor_commitment
            != claim.query_only_predictor_commitment
        ):
            raise Dstc9P1FormalControllerError(
                "query-only prediction binding drifted"
            )
    return by_work


def _validate_coordinate_rows(
    *,
    block: BlockView,
    corpus: CorpusView,
    rows: Sequence[CoordinateScoreRow],
) -> Mapping[str, CoordinateScoreRow]:
    if (
        not isinstance(rows, Sequence)
        or len(rows) != len(block.items)
        or any(not isinstance(row, CoordinateScoreRow) for row in rows)
    ):
        raise Dstc9P1FormalControllerError(
            "coordinate score coverage drifted"
        )
    by_work = {row.work_id: row for row in rows}
    if len(by_work) != len(rows) or set(by_work) != {
        item.work_id for item in block.items
    }:
        raise Dstc9P1FormalControllerError(
            "coordinate score work IDs drifted"
        )
    for item in block.items:
        row = by_work[item.work_id]
        if (
            row.block != block.block
            or row.normalized_query_sha256
            != core.normalized_query_sha256(item.history)
            or row.corpus_projection_sha256
            != corpus.projection_sha256
        ):
            raise Dstc9P1FormalControllerError(
                "coordinate score binding drifted"
            )
    return by_work


def _compile_slates(
    *,
    block: BlockView,
    corpus: CorpusView,
    claim: AcquisitionClaim,
    predictions: Sequence[BucketPrediction],
    coordinate_rows: Sequence[CoordinateScoreRow],
) -> tuple[_ActionRow, ...]:
    prediction_by_work = _validate_prediction_rows(
        block=block,
        claim=claim,
        rows=predictions,
    )
    score_by_work = _validate_coordinate_rows(
        block=block,
        corpus=corpus,
        rows=coordinate_rows,
    )
    result: list[_ActionRow] = []
    for item in block.items:
        prediction = prediction_by_work[item.work_id]
        coordinate = score_by_work[item.work_id]
        vectors = coordinate.score_vectors
        try:
            slate = core.build_action_slate(
                item.history,
                corpus.snippets,
                vectors["global_ce"],
                vectors["last_turn_ce"],
                vectors["minilm"],
                vectors["entity"],
                vectors["title"],
                vectors["body"],
                prediction.predicted_bucket,
            )
        except core.Dstc9P1TypedCoreError as exc:
            raise Dstc9P1FormalControllerError(
                "typed action slate formation failed"
            ) from exc
        if (
            slate.normalized_query_sha256
            != prediction.normalized_query_sha256
            or slate.snippet_projection_sha256
            != corpus.projection_sha256
            or slate.score_bundle_sha256
            != coordinate.score_bundle_sha256
        ):
            raise Dstc9P1FormalControllerError(
                "typed action slate binding drifted"
            )
        result.append(
            _ActionRow(
                item=item,
                prediction=prediction,
                coordinate=coordinate,
                slate=slate,
            )
        )
    return tuple(result)


def _form_label_free_slates(
    *,
    block: BlockView,
    corpus: CorpusView,
    claim: AcquisitionClaim,
    predictor: QueryOnlyBucketPredictor,
    coordinate_scorer: GlobalCoordinateScorer,
) -> tuple[_ActionRow, ...]:
    predictions = predictor.predict(block.items)
    coordinate_rows = coordinate_scorer.score(corpus, block.items)
    return _compile_slates(
        block=block,
        corpus=corpus,
        claim=claim,
        predictions=predictions,
        coordinate_rows=coordinate_rows,
    )


def _raw_top5(row: _ActionRow) -> tuple[int, ...]:
    scores = row.coordinate.score_vectors["global_ce"]
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
        raise Dstc9P1FormalControllerError(
            "official HippoRAG coverage drifted"
        )
    by_work = {row.work_id: row for row in rows}
    if len(by_work) != len(rows) or set(by_work) != {
        item.work_id for item in block.items
    }:
        raise Dstc9P1FormalControllerError(
            "official HippoRAG work IDs drifted"
        )
    for item in block.items:
        row = by_work[item.work_id]
        if (
            row.block != block.block
            or row.normalized_query_sha256
            != core.normalized_query_sha256(item.history)
            or row.corpus_projection_sha256
            != corpus.projection_sha256
        ):
            raise Dstc9P1FormalControllerError(
                "official HippoRAG binding drifted"
            )
    return by_work


def _form_four_arms(
    *,
    block: BlockView,
    corpus: CorpusView,
    claim: AcquisitionClaim,
    program: core.E1Program,
    predictor: QueryOnlyBucketPredictor,
    coordinate_scorer: GlobalCoordinateScorer,
    hippo_runner: OfficialHippoRunner,
) -> tuple[_ActionRow, ...]:
    predictions = predictor.predict(block.items)
    with ThreadPoolExecutor(max_workers=2) as pool:
        coordinate_future = pool.submit(
            coordinate_scorer.score,
            corpus,
            block.items,
        )
        hippo_future = pool.submit(
            hippo_runner.retrieve,
            corpus,
            block.items,
        )
        coordinate_rows = coordinate_future.result()
        hippo_rows = hippo_future.result()
    compiled = _compile_slates(
        block=block,
        corpus=corpus,
        claim=claim,
        predictions=predictions,
        coordinate_rows=coordinate_rows,
    )
    hippo_by_work = _validate_hippo_rows(
        block=block,
        corpus=corpus,
        rows=hippo_rows,
    )
    result: list[_ActionRow] = []
    for row in compiled:
        try:
            e0 = core.apply_e0(row.slate, stage=block.block)
            e1 = core.apply_e1(
                program,
                row.slate,
                stage=block.block,
            )
        except core.Dstc9P1TypedCoreError as exc:
            raise Dstc9P1FormalControllerError(
                "typed policy application failed"
            ) from exc
        result.append(
            _ActionRow(
                item=row.item,
                prediction=row.prediction,
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
        "block": block.block,
        "block_view_sha256": block.view_sha256,
        "label_bearing_input_count": 0,
        "rows": [
            {
                "prediction": row.prediction.audit_payload(),
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
            raise Dstc9P1FormalControllerError(
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
                "prediction": row.prediction.audit_payload(),
                "score_bundle_sha256": (
                    row.coordinate.score_bundle_sha256
                ),
                "work_id": row.item.work_id,
            }
        )
    return {
        "block": block.block,
        "block_view_sha256": block.view_sha256,
        "E1_model_sha256": model_sha256,
        "label_bearing_input_count": 0,
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
        raise Dstc9P1FormalControllerError(
            "late qrel pack binding drifted"
        )
    by_work = {row.work_id: row for row in pack.rows}
    if set(by_work) != {item.work_id for item in block.items}:
        raise Dstc9P1FormalControllerError(
            "late qrel work coverage drifted"
        )
    family_counts = Counter(row.family for row in pack.rows)
    if family_counts != Counter(
        {
            family: FAMILY_COUNTS[block.block]
            for family in FAMILIES
        }
    ):
        raise Dstc9P1FormalControllerError(
            "late qrel family quota drifted"
        )
    if any(
        row.corpus_projection_sha256 != corpus.projection_sha256
        for row in pack.rows
    ):
        raise Dstc9P1FormalControllerError(
            "late qrel corpus binding drifted"
        )
    return by_work


def _seal_qrel_pack(
    root: Path,
    pack: QrelPack,
) -> _SealedFile:
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
        gold = qrels[row.item.work_id].gold_ordinal
        utility = tuple(
            singleton_recall_rr_utility(
                row.slate.action(recipe_id).top5_ordinals,
                gold,
            )
            for recipe_id in core.RECIPE_IDS
        )
        try:
            examples.append(core.make_aform_example(row.slate, utility))
        except core.Dstc9P1TypedCoreError as exc:
            raise Dstc9P1FormalControllerError(
                "A_form example binding failed"
            ) from exc
    try:
        return core.fit_e1(tuple(examples))
    except core.Dstc9P1TypedCoreError as exc:
        raise Dstc9P1FormalControllerError(
            "E1 fitting failed"
        ) from exc


def _fsearch_behavior_body(
    block: BlockView,
    rows: Sequence[_ActionRow],
    program: core.E1Program,
    *,
    model_sha256: str,
) -> dict[str, object]:
    bucket_counts: Counter[int] = Counter()
    recipe_counts: Counter[str] = Counter()
    fallback_count = 0
    decisions: list[dict[str, object]] = []
    for row in rows:
        rule = program.rule(row.slate.predicted_bucket)
        action = row.slate.action(rule.selected_recipe_id)
        bucket_counts[row.slate.predicted_bucket] += 1
        recipe_counts[rule.selected_recipe_id] += 1
        fallback_count += int(rule.selected_recipe_id == core.E0_RECIPE_ID)
        decisions.append(
            {
                "action_behavior_digest": action.behavior_digest,
                "predicted_bucket": row.slate.predicted_bucket,
                "selected_recipe_id": rule.selected_recipe_id,
                "top5_ordinals": list(action.top5_ordinals),
                "work_id": row.item.work_id,
            }
        )
    return {
        "block": block.block,
        "block_view_sha256": block.view_sha256,
        "E1_model_sha256": model_sha256,
        "behavior": {
            "bucket_count": {
                str(bucket): bucket_counts[bucket]
                for bucket in core.PREDICTED_BUCKETS
            },
            "decision_set_sha256": stable_hash(decisions),
            "fallback_count": fallback_count,
            "item_count": len(rows),
            "recipe_count": {
                recipe_id: recipe_counts[recipe_id]
                for recipe_id in core.RECIPE_IDS
            },
        },
        "decisions": decisions,
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
    private_rows: list[dict[str, object]] = []
    action_difference = {
        "E1_vs_E0": 0,
        "E1_vs_HippoRAG": 0,
        "E1_vs_RAW": 0,
    }
    for index, row in enumerate(rows):
        if (
            row.e0 is None
            or row.e1 is None
            or row.hippo is None
            or row.raw is None
        ):
            raise Dstc9P1FormalControllerError(
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
            arm: singleton_recall_rr_utility(ordinals, qrel.gold_ordinal)
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
            "aggregate": {
                "E1_minus_E0": e1_e0.payload(),
                "E1_minus_HippoRAG": e1_hippo.payload(),
                "E1_minus_RAW": e1_raw.payload(),
            },
            "arm_total_integer_Recall5_plus_RR5_utility": totals,
            "block": block.block,
            "family": {
                family: {
                    "E1_minus_HippoRAG": (
                        family_hippo[family].payload()
                    ),
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
                "maximum": MAX_SINGLETON_UTILITY,
                "recall_at_5_scale": RANK_SCALE,
                "reciprocal_rank_at_5_scale": RANK_SCALE,
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


def _safe_block_result(value: _ScoredBlock) -> dict[str, object]:
    reality = (
        _comparison_pass(value.e1_raw)
        and _comparison_pass(value.e1_hippo)
        and all(
            value.family_e1_raw[family].net_utility > 0
            and value.family_e1_hippo[family].net_utility > 0
            for family in FAMILIES
        )
    )
    return {
        "aggregate": {
            "E1_minus_E0": value.e1_e0.payload(),
            "E1_minus_HippoRAG": value.e1_hippo.payload(),
            "E1_minus_RAW": value.e1_raw.payload(),
        },
        "arm_total_integer_Recall5_plus_RR5_utility": dict(
            value.arm_total_utility
        ),
        "family": {
            family: {
                "E1_minus_HippoRAG": (
                    value.family_e1_hippo[family].payload()
                ),
                "E1_minus_RAW": value.family_e1_raw[family].payload(),
            }
            for family in FAMILIES
        },
        "promotion_passed": _comparison_pass(value.e1_e0),
        "reality_primary_passed": reality,
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
        "item_query_document_qrel_or_per_item_score_values_published": False,
        "online_or_API_evaluator_calls": 0,
        "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
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
        "item_query_document_qrel_or_per_item_score_values_published": False,
        "online_or_API_evaluator_calls": 0,
        "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
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
    predictor: QueryOnlyBucketPredictor,
    coordinate_scorer: GlobalCoordinateScorer,
    hippo_runner: OfficialHippoRunner,
) -> Mapping[str, object]:
    """Execute the frozen one-shot lifecycle and return a safe terminal."""

    binding = _hex64(
        execution_binding_sha256,
        field_name="execution binding",
    )
    root = Path(work_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise Dstc9P1FormalControllerError("formal work root is unsafe")
    os.chmod(root, 0o700)
    marker = self_hashed(
        {
            "execution_binding_sha256": binding,
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": f"{VERSION}_one_shot_marker_v1",
            "study_id": STUDY_ID,
        }
    )
    _exclusive_bytes(
        root / FORMAL_MARKER_FILENAME,
        canonical_bytes(marker),
    )
    marker_sha256 = str(marker["self_sha256"])
    stage = "claim_acquisition"
    try:
        claim = acquisition.claim_formal_attempt(marker_sha256)
        if not isinstance(claim, AcquisitionClaim):
            raise Dstc9P1FormalControllerError(
                "acquisition claim type drifted"
            )
        stage = "load_global_public_corpus"
        corpus = acquisition.load_public_corpus(claim)
        if not isinstance(corpus, CorpusView):
            raise Dstc9P1FormalControllerError(
                "public corpus type drifted"
            )

        initial_blocks: dict[str, BlockView] = {}
        for block_name in INITIAL_BLOCKS:
            stage = f"load_{block_name}_label_free"
            block = acquisition.load_label_free_block(block_name, None)
            _validate_block(block, expected=block_name)
            initial_blocks[block_name] = block
        _validate_public_disjointness(
            tuple(initial_blocks[name] for name in INITIAL_BLOCKS)
        )

        archives: dict[str, _SealedFile] = {}
        stage = "form_and_seal_A_form_actions"
        aform_rows = _form_label_free_slates(
            block=initial_blocks["A_form"],
            corpus=corpus,
            claim=claim,
            predictor=predictor,
            coordinate_scorer=coordinate_scorer,
        )
        aform_action = _seal_json(
            root,
            "A_form.actions.private.json",
            _aform_action_body(initial_blocks["A_form"], aform_rows),
        )
        archives["A_form_actions"] = aform_action

        stage = "release_A_form_qrels_after_action_seal"
        aform_pack = acquisition.release_qrels_after_action_seal(
            "A_form",
            aform_action.path,
            aform_action.value,
        )
        aform_qrels = _validate_qrels(
            pack=aform_pack,
            block=initial_blocks["A_form"],
            corpus=corpus,
            action_archive_sha256=aform_action.self_sha256,
        )
        archives["A_form_qrels"] = _seal_qrel_pack(root, aform_pack)

        stage = "fit_and_seal_E1_once"
        program = _fit_e1_once(aform_rows, aform_qrels)
        model_archive = _seal_json(
            root,
            "E1.model.private.json",
            {
                "A_form_action_archive_sha256": (
                    aform_action.self_sha256
                ),
                "A_form_qrel_pack_sha256": aform_pack.pack_sha256,
                "fit_count": 1,
                "model": program.payload(),
                "schema": f"{VERSION}_frozen_E1_model_v1",
                "study_id": STUDY_ID,
            },
        )
        archives["E1_model"] = model_archive

        stage = "form_and_seal_F_search_behavior_without_qrels"
        f_rows = _form_label_free_slates(
            block=initial_blocks["F_search"],
            corpus=corpus,
            claim=claim,
            predictor=predictor,
            coordinate_scorer=coordinate_scorer,
        )
        archives["F_search_behavior"] = _seal_json(
            root,
            "F_search.behavior.private.json",
            _fsearch_behavior_body(
                initial_blocks["F_search"],
                f_rows,
                program,
                model_sha256=model_archive.self_sha256,
            ),
        )

        stage = "form_and_seal_A_hold_four_arms"
        ahold_rows = _form_four_arms(
            block=initial_blocks["A_hold"],
            corpus=corpus,
            claim=claim,
            program=program,
            predictor=predictor,
            coordinate_scorer=coordinate_scorer,
            hippo_runner=hippo_runner,
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
            stage = "materialize_M_search_label_free"
            m_block = acquisition.load_label_free_block(
                "M_search",
                authorization,
            )
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
            stage = "form_and_seal_M_search_four_arms"
            m_rows = _form_four_arms(
                block=m_block,
                corpus=corpus,
                claim=claim,
                program=program,
                predictor=predictor,
                coordinate_scorer=coordinate_scorer,
                hippo_runner=hippo_runner,
            )
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

        stage = "seal_safe_terminal"
        terminal = self_hashed(
            _terminal_body(
                execution_binding_sha256=binding,
                marker_sha256=marker_sha256,
                claim=claim,
                archives=archives,
                ahold_score=ahold_score,
                m_score=m_score,
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
        if isinstance(exc, Dstc9P1FormalControllerError):
            raise
        raise Dstc9P1FormalControllerError(
            "formal controller failed closed"
        ) from exc


__all__ = [
    "ALPHA",
    "AcquisitionClaim",
    "BLOCK_COUNTS",
    "BlockView",
    "BucketPrediction",
    "CORPUS_SIZE",
    "CoordinateScoreRow",
    "CorpusView",
    "Dstc9P1FormalControllerError",
    "ExactPairedComparison",
    "FAMILIES",
    "FAMILY_COUNTS",
    "FORMAL_MARKER_FILENAME",
    "FORMAL_TERMINAL_FILENAME",
    "FormalAcquisitionBoundary",
    "FormalItemView",
    "GlobalCoordinateScorer",
    "HippoResult",
    "OfficialHippoRunner",
    "PROMOTION_AUTHORIZATION_FILENAME",
    "QrelPack",
    "QrelRow",
    "QueryOnlyBucketPredictor",
    "compare_paired_integer_utility",
    "run_formal_controller",
    "self_hashed",
    "singleton_recall_rr_utility",
    "stable_hash",
]
