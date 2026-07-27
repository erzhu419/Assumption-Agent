"""Source-agnostic one-shot formal controller for TechQA P1.

The module deliberately has no file, archive-loader, network, model, API, or
online-evaluator entrypoint.  A trusted outer boundary may construct
``VerifiedSource`` only after the frozen public-source qualification succeeds
and may then call this controller with the three already-verified JSON
projections.

The scientific contract is fixed here:

* one HMAC secret selects question-ID/query-byte-disjoint TRAIN
  ``A_form``/``F_search`` and DEV ``A_hold``/``M_search`` cohorts,
  stratified by a preregistered operational query-intent classifier;
* the original ``DOC_IDS`` order is erased.  Every search cluster receives
  one canonical document-id-sorted corpus shared byte-for-byte by RAW, Agent,
  and the externally executed official HippoRAG arm;
* A_form labels six frozen typed recipes and fits E1 once.  ``F_search`` is
  audit-only.  A_hold forms RAW/E0/E1 and an official-Hippo request before
  scoring; the reality primary is E0 versus RAW and HippoRAG;
* only a strictly positive E1-minus-E0 result in every A_hold cluster
  authorizes M_search action materialization.  L5 is the unchanged E1 versus
  E0 on that untouched block.

All item/query/document/qrel/action values live only in the returned private
archive.  The safe terminal contains aggregate exact fractions and opaque
commitments.  There is no retry, replay, resampling, alternate candidate,
provider, parser, family, quota, or gate path.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import re
from typing import Mapping, Sequence

from . import techqa_p1_typed_core_v1 as core
from . import techqa_p1_official_hipporag_v1 as hippo_adapter


VERSION = "techqa_p1_formal_v1"
STUDY_ID = core.STUDY_ID

INFORMATION = "INFORMATION"
PROCEDURE = "PROCEDURE"
TROUBLESHOOT = "TROUBLESHOOT"
FAMILIES = (INFORMATION, PROCEDURE, TROUBLESHOOT)
FAMILY_IDS = FAMILIES

A_FORM = "A_form"
F_SEARCH = "F_search"
A_HOLD = "A_hold"
M_SEARCH = "M_search"
BLOCKS = (A_FORM, F_SEARCH, A_HOLD, M_SEARCH)
BLOCK_TO_SPLIT = {
    A_FORM: "TRAIN",
    F_SEARCH: "TRAIN",
    A_HOLD: "DEV",
    M_SEARCH: "DEV",
}
BLOCK_FAMILY_QUOTAS = {
    A_FORM: 36,
    F_SEARCH: 12,
    A_HOLD: 12,
    M_SEARCH: 12,
}
FAMILY_QUOTAS = BLOCK_FAMILY_QUOTAS
SOURCE_MINIMUM_FAMILY_COUNTS = {
    "TRAIN": (
        BLOCK_FAMILY_QUOTAS[A_FORM]
        + BLOCK_FAMILY_QUOTAS[F_SEARCH]
    ),
    "DEV": (
        BLOCK_FAMILY_QUOTAS[A_HOLD]
        + BLOCK_FAMILY_QUOTAS[M_SEARCH]
    ),
}
CLUSTER_COUNTS = {
    # A_form has the same frozen three-per-family cluster construction, with
    # twelve clusters because its preregistered quota is 36 per family.
    A_FORM: 12,
    F_SEARCH: 4,
    A_HOLD: 4,
    M_SEARCH: 4,
}
ITEMS_PER_FAMILY_PER_CLUSTER = 3
CANDIDATE_DOCUMENT_COUNT = 50
MAXIMUM_CLUSTER_DOCUMENT_COUNT = 450
TOP_K = core.TOP_K
HMAC_SECRET_BYTES = 32

TROUBLESHOOT_INDICATORS = (
    "why",
    "cause",
    "fix",
    "resolve",
    "error",
    "exception",
    "fail",
    "cannot",
    "unable",
    "not working",
    "crash",
    "hang",
    "problem",
    "issue",
)
PROCEDURE_INDICATORS = (
    "how to",
    "how do",
    "how can",
    "steps",
    "procedure",
    "instructions",
)

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"[^\x00-\x1f\x7f]{1,512}")


class TechqaP1FormalError(RuntimeError):
    """A frozen selection, byte binding, lifecycle, or scoring invariant failed."""


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
        raise TechqaP1FormalError("formal value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value:
        raise TechqaP1FormalError("self hash already exists")
    body = dict(value)
    body["self_sha256"] = stable_hash(body)
    return body


def _hex64(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise TechqaP1FormalError(f"{field} is not a SHA-256 digest")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise TechqaP1FormalError(f"{field} is invalid")
    return value


def _fraction_payload(value: Fraction) -> dict[str, int]:
    if not isinstance(value, Fraction):
        raise TechqaP1FormalError("aggregate is not an exact Fraction")
    return {
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


def _indicator_present(normalized: str, indicator: str) -> bool:
    words = indicator.split()
    expression = r"(?<!\w)" + r"\s+".join(
        re.escape(word) for word in words
    ) + r"(?!\w)"
    return re.search(expression, normalized) is not None


def operational_family(question_title: str, question_text: str) -> str:
    """Return the frozen operational intent stratum.

    TROUBLESHOOT has priority over PROCEDURE.  The normalized input is exactly
    the unchanged public title, one newline, and public question text.
    """

    normalized = core.normalize_text(
        core.serialize_query_text(question_title, question_text),
        field="normalized title+text",
    )
    if any(
        _indicator_present(normalized, value)
        for value in TROUBLESHOOT_INDICATORS
    ):
        return TROUBLESHOOT
    if any(
        _indicator_present(normalized, value)
        for value in PROCEDURE_INDICATORS
    ):
        return PROCEDURE
    return INFORMATION


@dataclass(frozen=True, slots=True)
class SourceCommitments:
    training_q_a_sha256: str
    dev_q_a_sha256: str
    training_dev_technotes_sha256: str
    qualification_receipt_sha256: str

    def __post_init__(self) -> None:
        for field, value in (
            ("training Q_A", self.training_q_a_sha256),
            ("dev Q_A", self.dev_q_a_sha256),
            ("training_dev_technotes", self.training_dev_technotes_sha256),
            ("qualification receipt", self.qualification_receipt_sha256),
        ):
            _hex64(value, field=field)

    def payload(self) -> dict[str, str]:
        return {
            "dev_q_a_sha256": self.dev_q_a_sha256,
            "qualification_receipt_sha256": (
                self.qualification_receipt_sha256
            ),
            "training_dev_technotes_sha256": (
                self.training_dev_technotes_sha256
            ),
            "training_q_a_sha256": self.training_q_a_sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedDocument:
    """One already-qualified TechQA document; never accepted from a path."""

    document_id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        _identifier(self.document_id, field="document ID")
        # Reuse the typed core's exact public-byte validation.
        core.Document(ordinal=0, title=self.title, text=self.text)

    def identity_payload(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "serialized_sha256": hashlib.sha256(
                (
                    self.title
                    + core.DOCUMENT_SERIALIZATION_SEPARATOR
                    + self.text
                ).encode("utf-8")
            ).hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class VerifiedQuestion:
    """One answerable, already-qualified TechQA question."""

    question_id: str
    question_title: str
    question_text: str
    document_ids: tuple[str, ...]
    gold_document_id: str

    def __post_init__(self) -> None:
        _identifier(self.question_id, field="question ID")
        core.serialize_query_text(self.question_title, self.question_text)
        _identifier(self.gold_document_id, field="gold document ID")
        if isinstance(self.document_ids, (str, bytes)):
            raise TechqaP1FormalError("DOC_IDS is not a document sequence")
        values = tuple(self.document_ids)
        if (
            len(values) != CANDIDATE_DOCUMENT_COUNT
            or len(set(values)) != CANDIDATE_DOCUMENT_COUNT
        ):
            raise TechqaP1FormalError(
                "DOC_IDS must be an unordered set of exactly 50 IDs"
            )
        for value in values:
            _identifier(value, field="candidate document ID")
        canonical = tuple(sorted(values))
        object.__setattr__(self, "document_ids", canonical)
        if self.gold_document_id not in canonical:
            raise TechqaP1FormalError(
                "singleton gold DOCUMENT is absent from DOC_IDS"
            )

    @property
    def normalized_query_sha256(self) -> str:
        normalized = core.normalize_text(
            core.serialize_query_text(
                self.question_title, self.question_text
            ),
            field="normalized query identity",
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @property
    def raw_query_bytes_sha256(self) -> str:
        return hashlib.sha256(
            core.serialize_query_text(
                self.question_title, self.question_text
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedSource:
    """Three qualified in-memory projections and their immutable commitments."""

    training_questions: tuple[VerifiedQuestion, ...]
    dev_questions: tuple[VerifiedQuestion, ...]
    documents: tuple[VerifiedDocument, ...]
    commitments: SourceCommitments

    def __post_init__(self) -> None:
        training = tuple(self.training_questions)
        dev = tuple(self.dev_questions)
        documents = tuple(self.documents)
        if (
            not training
            or not dev
            or not documents
            or any(not isinstance(row, VerifiedQuestion) for row in training)
            or any(not isinstance(row, VerifiedQuestion) for row in dev)
            or any(not isinstance(row, VerifiedDocument) for row in documents)
            or not isinstance(self.commitments, SourceCommitments)
        ):
            raise TechqaP1FormalError("verified source projection drifted")
        training = tuple(sorted(training, key=lambda row: row.question_id))
        dev = tuple(sorted(dev, key=lambda row: row.question_id))
        documents = tuple(sorted(documents, key=lambda row: row.document_id))
        object.__setattr__(self, "training_questions", training)
        object.__setattr__(self, "dev_questions", dev)
        object.__setattr__(self, "documents", documents)
        question_ids = [
            row.question_id for row in training + dev
        ]
        document_ids = [row.document_id for row in documents]
        if len(set(question_ids)) != len(question_ids):
            raise TechqaP1FormalError(
                "question IDs are not unique across verified splits"
            )
        if len(set(document_ids)) != len(document_ids):
            raise TechqaP1FormalError("document IDs are not unique")
        available = set(document_ids)
        for question in training + dev:
            if not set(question.document_ids) <= available:
                raise TechqaP1FormalError(
                    "DOC_IDS references a missing verified document"
                )

    @property
    def document_by_id(self) -> Mapping[str, VerifiedDocument]:
        return {row.document_id: row for row in self.documents}


@dataclass(frozen=True, slots=True)
class SelectedItem:
    block: str
    split: str
    family: str
    work_id: str
    selection_hmac_sha256: str
    question: VerifiedQuestion

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCKS
            or self.split != BLOCK_TO_SPLIT[self.block]
            or self.family not in FAMILIES
            or operational_family(
                self.question.question_title,
                self.question.question_text,
            )
            != self.family
        ):
            raise TechqaP1FormalError("selected item stratum drifted")
        _identifier(self.work_id, field="opaque work ID")
        _hex64(self.selection_hmac_sha256, field="selection HMAC")

    def private_payload(self) -> dict[str, object]:
        return {
            "block": self.block,
            "candidate_document_ids": list(self.question.document_ids),
            "family": self.family,
            "gold_document_id": self.question.gold_document_id,
            "normalized_query_sha256": (
                self.question.normalized_query_sha256
            ),
            "question_id": self.question.question_id,
            "selection_hmac_sha256": self.selection_hmac_sha256,
            "split": self.split,
            "work_id": self.work_id,
        }


@dataclass(frozen=True, slots=True)
class CohortBlock:
    block: str
    items: tuple[SelectedItem, ...]

    def __post_init__(self) -> None:
        if self.block not in BLOCKS:
            raise TechqaP1FormalError("cohort block name drifted")
        expected = BLOCK_FAMILY_QUOTAS[self.block]
        counts = {family: 0 for family in FAMILIES}
        for item in self.items:
            if not isinstance(item, SelectedItem) or item.block != self.block:
                raise TechqaP1FormalError("cohort block item drifted")
            counts[item.family] += 1
        if counts != {family: expected for family in FAMILIES}:
            raise TechqaP1FormalError("cohort family quota drifted")
        expected_order = tuple(
            sorted(
                self.items,
                key=lambda row: (
                    FAMILIES.index(row.family),
                    row.selection_hmac_sha256,
                    row.question.question_id,
                ),
            )
        )
        if self.items != expected_order:
            raise TechqaP1FormalError("cohort item order drifted")

    def private_payload(self) -> dict[str, object]:
        return {
            "block": self.block,
            "items": [row.private_payload() for row in self.items],
        }


@dataclass(frozen=True, slots=True)
class CohortSelection:
    blocks: tuple[CohortBlock, ...]
    secret_commitment_sha256: str
    selection_sha256: str

    def __post_init__(self) -> None:
        _hex64(self.secret_commitment_sha256, field="HMAC secret commitment")
        if tuple(row.block for row in self.blocks) != BLOCKS:
            raise TechqaP1FormalError("cohort block registry drifted")
        all_items = tuple(
            item for block in self.blocks for item in block.items
        )
        for values, label in (
            ([row.question.question_id for row in all_items], "question ID"),
            (
                [row.question.normalized_query_sha256 for row in all_items],
                "normalized query bytes",
            ),
            ([row.work_id for row in all_items], "work ID"),
        ):
            if len(set(values)) != len(values):
                raise TechqaP1FormalError(
                    f"selected {label} is not mutually disjoint across blocks"
                )
        expected = stable_hash(
            {
                "blocks": [
                    block.private_payload() for block in self.blocks
                ],
                "schema": f"{VERSION}_private_cohort_selection_v1",
                "secret_commitment_sha256": (
                    self.secret_commitment_sha256
                ),
                "study_id": STUDY_ID,
            }
        )
        if not hmac.compare_digest(
            _hex64(self.selection_sha256, field="cohort selection"),
            expected,
        ):
            raise TechqaP1FormalError("cohort selection hash drifted")

    def block(self, name: str) -> CohortBlock:
        if name not in BLOCKS:
            raise TechqaP1FormalError("requested cohort block drifted")
        return self.blocks[BLOCKS.index(name)]

    def private_payload(self) -> dict[str, object]:
        return {
            "blocks": [
                block.private_payload() for block in self.blocks
            ],
            "schema": f"{VERSION}_private_cohort_selection_v1",
            "secret_commitment_sha256": self.secret_commitment_sha256,
            "study_id": STUDY_ID,
        }


def _selection_digest(
    secret: bytes,
    *,
    split: str,
    family: str,
    question: VerifiedQuestion,
) -> str:
    return hmac.new(
        secret,
        canonical_bytes(
            {
                "family": family,
                "normalized_query_sha256": (
                    question.normalized_query_sha256
                ),
                "purpose": "single_frozen_cohort_order",
                "question_id": question.question_id,
                "split": split,
                "study_id": STUDY_ID,
            }
        ),
        hashlib.sha256,
    ).hexdigest()


def _work_id(secret: bytes, *, split: str, question_id: str) -> str:
    digest = hmac.new(
        secret,
        canonical_bytes(
            {
                "purpose": "opaque_work_id",
                "question_id": question_id,
                "split": split,
                "study_id": STUDY_ID,
            }
        ),
        hashlib.sha256,
    ).hexdigest()
    return f"techqa-work-v1-{digest}"


def select_question_cohorts(
    training_questions: Sequence[VerifiedQuestion],
    dev_questions: Sequence[VerifiedQuestion],
    *,
    hmac_secret: bytes,
) -> CohortSelection:
    """Select frozen cohorts from source-less verified question projections."""

    canonical_by_split: dict[str, tuple[VerifiedQuestion, ...]] = {}
    for split, questions in (
        ("TRAIN", training_questions),
        ("DEV", dev_questions),
    ):
        if (
            isinstance(questions, (str, bytes))
            or not isinstance(questions, Sequence)
        ):
            raise TechqaP1FormalError(
                f"{split} verified questions are not a sequence"
            )
        values = tuple(questions)
        if (
            not values
            or any(
                not isinstance(question, VerifiedQuestion)
                for question in values
            )
        ):
            raise TechqaP1FormalError(
                f"{split} verified question projection drifted"
            )
        canonical_by_split[split] = tuple(
            sorted(values, key=lambda row: row.question_id)
        )
    question_ids = [
        question.question_id
        for split in ("TRAIN", "DEV")
        for question in canonical_by_split[split]
    ]
    if len(set(question_ids)) != len(question_ids):
        raise TechqaP1FormalError(
            "question IDs are not unique across verified splits"
        )
    if (
        not isinstance(hmac_secret, bytes)
        or len(hmac_secret) != HMAC_SECRET_BYTES
    ):
        raise TechqaP1FormalError(
            "the single HMAC secret must be exactly 32 bytes"
        )
    secret_commitment = hashlib.sha256(hmac_secret).hexdigest()
    used_question_ids: set[str] = set()
    used_query_hashes: set[str] = set()
    selected: dict[str, list[SelectedItem]] = {
        block: [] for block in BLOCKS
    }
    for split, block_pair in (
        ("TRAIN", (A_FORM, F_SEARCH)),
        ("DEV", (A_HOLD, M_SEARCH)),
    ):
        for family in FAMILIES:
            ranked = sorted(
                (
                    (
                        _selection_digest(
                            hmac_secret,
                            split=split,
                            family=family,
                            question=question,
                        ),
                        question,
                    )
                    for question in canonical_by_split[split]
                    if operational_family(
                        question.question_title,
                        question.question_text,
                    )
                    == family
                ),
                key=lambda row: (row[0], row[1].question_id),
            )
            cursor = 0
            for block in block_pair:
                quota = BLOCK_FAMILY_QUOTAS[block]
                accepted = 0
                while cursor < len(ranked) and accepted < quota:
                    digest, question = ranked[cursor]
                    cursor += 1
                    if (
                        question.question_id in used_question_ids
                        or question.normalized_query_sha256
                        in used_query_hashes
                    ):
                        continue
                    item = SelectedItem(
                        block=block,
                        split=split,
                        family=family,
                        work_id=_work_id(
                            hmac_secret,
                            split=split,
                            question_id=question.question_id,
                        ),
                        selection_hmac_sha256=digest,
                        question=question,
                    )
                    selected[block].append(item)
                    used_question_ids.add(question.question_id)
                    used_query_hashes.add(
                        question.normalized_query_sha256
                    )
                    accepted += 1
                if accepted != quota:
                    raise TechqaP1FormalError(
                        f"{split}/{family}/{block} cannot satisfy the "
                        "frozen quota under ID/query-byte disjointness"
                    )
    blocks = tuple(
        CohortBlock(
            block=block,
            items=tuple(
                sorted(
                    selected[block],
                    key=lambda row: (
                        FAMILIES.index(row.family),
                        row.selection_hmac_sha256,
                        row.question.question_id,
                    ),
                )
            ),
        )
        for block in BLOCKS
    )
    private = {
        "blocks": [block.private_payload() for block in blocks],
        "schema": f"{VERSION}_private_cohort_selection_v1",
        "secret_commitment_sha256": secret_commitment,
        "study_id": STUDY_ID,
    }
    return CohortSelection(
        blocks=blocks,
        secret_commitment_sha256=secret_commitment,
        selection_sha256=stable_hash(private),
    )


def select_private_cohorts(
    source: VerifiedSource,
    *,
    hmac_secret: bytes,
) -> CohortSelection:
    """Delegate source-backed selection to the sole question-only selector."""

    if not isinstance(source, VerifiedSource):
        raise TechqaP1FormalError("verified source is absent")
    return select_question_cohorts(
        source.training_questions,
        source.dev_questions,
        hmac_secret=hmac_secret,
    )


@dataclass(frozen=True, slots=True)
class ClusterDocument:
    """Private document identity plus the sole public action projection."""

    document_id: str
    public_document: core.Document

    def __post_init__(self) -> None:
        _identifier(self.document_id, field="cluster document ID")
        if not isinstance(self.public_document, core.Document):
            raise TechqaP1FormalError(
                "cluster public document projection drifted"
            )

    def private_payload(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "ordinal": self.public_document.ordinal,
            "serialized_sha256": hashlib.sha256(
                core.serialize_document_bytes(self.public_document)
            ).hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class ClusterItem:
    selected: SelectedItem
    cluster_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.selected, SelectedItem)
            or type(self.cluster_index) is not int
            or not 0 <= self.cluster_index
            < CLUSTER_COUNTS[self.selected.block]
        ):
            raise TechqaP1FormalError("cluster item binding drifted")


@dataclass(frozen=True, slots=True)
class SearchCluster:
    block: str
    cluster_index: int
    items: tuple[ClusterItem, ...]
    documents: tuple[ClusterDocument, ...]
    corpus_sha256: str

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCKS
            or type(self.cluster_index) is not int
            or not 0 <= self.cluster_index < CLUSTER_COUNTS[self.block]
        ):
            raise TechqaP1FormalError("search cluster identity drifted")
        if len(self.items) != (
            ITEMS_PER_FAMILY_PER_CLUSTER * len(FAMILIES)
        ):
            raise TechqaP1FormalError("search cluster item count drifted")
        counts = {family: 0 for family in FAMILIES}
        for row in self.items:
            if (
                not isinstance(row, ClusterItem)
                or row.selected.block != self.block
                or row.cluster_index != self.cluster_index
            ):
                raise TechqaP1FormalError("cluster item lineage drifted")
            counts[row.selected.family] += 1
        if counts != {
            family: ITEMS_PER_FAMILY_PER_CLUSTER
            for family in FAMILIES
        }:
            raise TechqaP1FormalError("cluster family composition drifted")
        if (
            not self.documents
            or len(self.documents) > MAXIMUM_CLUSTER_DOCUMENT_COUNT
            or any(not isinstance(row, ClusterDocument) for row in self.documents)
        ):
            raise TechqaP1FormalError("cluster corpus size drifted")
        document_ids = [row.document_id for row in self.documents]
        ordinals = [row.public_document.ordinal for row in self.documents]
        serialized_hashes = [
            hashlib.sha256(
                core.serialize_document_bytes(row.public_document)
            ).hexdigest()
            for row in self.documents
        ]
        if (
            len(set(document_ids)) != len(document_ids)
            or ordinals != list(range(len(self.documents)))
            or serialized_hashes != sorted(serialized_hashes)
            or len(set(serialized_hashes)) != len(serialized_hashes)
        ):
            raise TechqaP1FormalError(
                "cluster corpus is not canonical unique public-byte order"
            )
        expected_ids = {
            document_id
            for item in self.items
            for document_id in item.selected.question.document_ids
        }
        if expected_ids != set(document_ids):
            raise TechqaP1FormalError(
                "cluster corpus is not the exact union of nine DOC_ID sets"
            )
        expected_hash = stable_hash(
            [row.private_payload() for row in self.documents]
        )
        if not hmac.compare_digest(
            _hex64(self.corpus_sha256, field="cluster corpus"),
            expected_hash,
        ):
            raise TechqaP1FormalError("cluster corpus hash drifted")

    @property
    def document_by_id(self) -> Mapping[str, ClusterDocument]:
        return {row.document_id: row for row in self.documents}

    def private_payload(self) -> dict[str, object]:
        return {
            "block": self.block,
            "cluster_index": self.cluster_index,
            "corpus_sha256": self.corpus_sha256,
            "documents": [
                row.private_payload() for row in self.documents
            ],
            "work_ids": [row.selected.work_id for row in self.items],
        }


def build_search_clusters(
    source: VerifiedSource,
    block: CohortBlock,
) -> tuple[SearchCluster, ...]:
    """Build the fixed three-items-per-family canonical search clusters."""

    if not isinstance(source, VerifiedSource) or not isinstance(
        block, CohortBlock
    ):
        raise TechqaP1FormalError("cluster construction input drifted")
    by_family = {
        family: tuple(
            row for row in block.items if row.family == family
        )
        for family in FAMILIES
    }
    document_by_id = source.document_by_id
    clusters: list[SearchCluster] = []
    for cluster_index in range(CLUSTER_COUNTS[block.block]):
        rows: list[ClusterItem] = []
        for family in FAMILIES:
            start = cluster_index * ITEMS_PER_FAMILY_PER_CLUSTER
            stop = start + ITEMS_PER_FAMILY_PER_CLUSTER
            family_rows = by_family[family][start:stop]
            if len(family_rows) != ITEMS_PER_FAMILY_PER_CLUSTER:
                raise TechqaP1FormalError(
                    "deterministic cluster slice is incomplete"
                )
            rows.extend(
                ClusterItem(
                    selected=row,
                    cluster_index=cluster_index,
                )
                for row in family_rows
            )
        # Family is used only to take the frozen three-per-family slice.  The
        # retrieval-facing query order is an opaque, label-independent HMAC
        # work order.
        rows.sort(key=lambda row: row.selected.work_id)
        document_ids = {
            document_id
            for row in rows
            for document_id in row.selected.question.document_ids
        }
        ordered_documents = sorted(
            (
                (
                    hashlib.sha256(
                        (
                            document_by_id[document_id].title
                            + core.DOCUMENT_SERIALIZATION_SEPARATOR
                            + document_by_id[document_id].text
                        ).encode("utf-8")
                    ).hexdigest(),
                    document_id,
                )
                for document_id in document_ids
            ),
            key=lambda row: (row[0], row[1]),
        )
        if len({row[0] for row in ordered_documents}) != len(
            ordered_documents
        ):
            raise TechqaP1FormalError(
                "cluster contains duplicate serialized document bytes"
            )
        documents = tuple(
            ClusterDocument(
                document_id=document_id,
                public_document=core.Document(
                    ordinal=ordinal,
                    title=document_by_id[document_id].title,
                    text=document_by_id[document_id].text,
                ),
            )
            for ordinal, (_serialized_hash, document_id) in enumerate(
                ordered_documents
            )
        )
        payload = [row.private_payload() for row in documents]
        clusters.append(
            SearchCluster(
                block=block.block,
                cluster_index=cluster_index,
                items=tuple(rows),
                documents=documents,
                corpus_sha256=stable_hash(payload),
            )
        )
    return tuple(clusters)


def public_action_projection(
    question: VerifiedQuestion,
    documents: Sequence[ClusterDocument],
) -> dict[str, object]:
    """Return the exact label-free projection accepted by the typed core."""

    if not isinstance(question, VerifiedQuestion):
        raise TechqaP1FormalError("public action question drifted")
    checked = tuple(documents)
    if not checked or any(
        not isinstance(row, ClusterDocument) for row in checked
    ):
        raise TechqaP1FormalError("public action documents drifted")
    return {
        "documents": [
            core.document_public_payload(row.public_document)
            for row in checked
        ],
        "question_text": question.question_text,
        "question_title": question.question_title,
    }


def _build_public_slate(
    question: VerifiedQuestion,
    documents: Sequence[ClusterDocument],
) -> core.ActionSlate:
    projection = public_action_projection(question, documents)
    if set(projection) != {
        "documents",
        "question_text",
        "question_title",
    }:
        raise TechqaP1FormalError("public action field set drifted")
    public_documents = tuple(
        core.document_from_public_fields(value)
        for value in projection["documents"]  # type: ignore[arg-type]
    )
    slate = core.build_action_slate(
        projection["question_title"],  # type: ignore[arg-type]
        projection["question_text"],  # type: ignore[arg-type]
        public_documents,
    )
    if (
        slate.raw_query_bytes_sha256
        != question.raw_query_bytes_sha256
        or slate.public_document_projection_sha256
        != stable_hash(
            [
                core.document_public_payload(row.public_document)
                for row in documents
            ]
        )
        or slate.serialized_document_set_sha256
        != stable_hash(
            [
                {
                    "ordinal": row.public_document.ordinal,
                    "serialized_sha256": hashlib.sha256(
                        core.serialize_document_bytes(
                            row.public_document
                        )
                    ).hexdigest(),
                }
                for row in documents
            ]
        )
    ):
        raise TechqaP1FormalError(
            "typed slate does not bind the exact shared query/document bytes"
        )
    return slate


@dataclass(frozen=True, slots=True)
class PrivateQrel:
    block: str
    cluster_index: int
    work_id: str
    family: str
    gold_document_id: str
    gold_document_ordinal: int
    corpus_sha256: str

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCKS
            or self.family not in FAMILIES
            or type(self.gold_document_ordinal) is not int
            or self.gold_document_ordinal < 0
        ):
            raise TechqaP1FormalError("private singleton qrel drifted")
        _identifier(self.work_id, field="qrel work ID")
        _identifier(self.gold_document_id, field="qrel gold document ID")
        _hex64(self.corpus_sha256, field="qrel corpus")

    def private_payload(self) -> dict[str, object]:
        return {
            "block": self.block,
            "cluster_index": self.cluster_index,
            "corpus_sha256": self.corpus_sha256,
            "family": self.family,
            "gold_document_id": self.gold_document_id,
            "gold_document_ordinal": self.gold_document_ordinal,
            "qrel_kind": "singleton_gold_DOCUMENT",
            "work_id": self.work_id,
        }


@dataclass(frozen=True, slots=True)
class HippoQueryBinding:
    """Private mapping from one public cluster ordinal to one opaque item."""

    query_ordinal: int
    work_id: str
    query_bytes_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.query_ordinal) is not int
            or not 0
            <= self.query_ordinal
            < hippo_adapter.EXPECTED_QUERY_COUNT
        ):
            raise TechqaP1FormalError("Hippo query ordinal drifted")
        _identifier(self.work_id, field="Hippo query work ID")
        _hex64(self.query_bytes_sha256, field="Hippo query bytes")

    def private_payload(self) -> dict[str, object]:
        return {
            "query_bytes_sha256": self.query_bytes_sha256,
            "query_ordinal": self.query_ordinal,
            "work_id": self.work_id,
        }


@dataclass(frozen=True, slots=True)
class HippoClusterRequest:
    """One exact public adapter input and its private nine-item mapping."""

    block: str
    cluster_index: int
    corpus_sha256: str
    adapter_input: Mapping[str, object]
    query_bindings: tuple[HippoQueryBinding, ...]
    request_sha256: str

    def __post_init__(self) -> None:
        if (
            self.block != A_HOLD
            or type(self.cluster_index) is not int
            or not 0 <= self.cluster_index < CLUSTER_COUNTS[A_HOLD]
        ):
            raise TechqaP1FormalError(
                "official Hippo cluster request stage drifted"
            )
        _hex64(self.corpus_sha256, field="Hippo cluster corpus")
        try:
            canonical_input = json.loads(
                hippo_adapter.canonical_bytes(
                    self.adapter_input
                ).decode("ascii")
            )
            cluster = hippo_adapter.validate_input(canonical_input)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            hippo_adapter.TechqaP1OfficialHippoRAGError,
        ) as exc:
            raise TechqaP1FormalError(
                "official Hippo adapter input drifted"
            ) from exc
        if (
            cluster.stage != self.block
            or cluster.cluster_ordinal != self.cluster_index
        ):
            raise TechqaP1FormalError(
                "official Hippo adapter stage/index binding drifted"
            )
        bindings = tuple(self.query_bindings)
        if (
            len(bindings) != hippo_adapter.EXPECTED_QUERY_COUNT
            or any(
                not isinstance(row, HippoQueryBinding)
                for row in bindings
            )
            or tuple(row.query_ordinal for row in bindings)
            != tuple(range(hippo_adapter.EXPECTED_QUERY_COUNT))
            or len({row.work_id for row in bindings}) != len(bindings)
        ):
            raise TechqaP1FormalError(
                "official Hippo query binding registry drifted"
            )
        for binding, query in zip(
            bindings, cluster.queries, strict=True
        ):
            expected_query_hash = hashlib.sha256(
                hippo_adapter.serialize_query(query).encode("utf-8")
            ).hexdigest()
            if not hmac.compare_digest(
                binding.query_bytes_sha256, expected_query_hash
            ):
                raise TechqaP1FormalError(
                    "official Hippo query byte binding drifted"
                )
        object.__setattr__(self, "adapter_input", canonical_input)
        object.__setattr__(self, "query_bindings", bindings)
        expected_request = stable_hash(
            self.private_payload(include_hash=False)
        )
        if not hmac.compare_digest(
            _hex64(
                self.request_sha256,
                field="Hippo cluster request",
            ),
            expected_request,
        ):
            raise TechqaP1FormalError(
                "official Hippo cluster request hash drifted"
            )

    def private_payload(
        self, *, include_hash: bool = True
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "adapter_input": dict(self.adapter_input),
            "block": self.block,
            "cluster_index": self.cluster_index,
            "corpus_sha256": self.corpus_sha256,
            "query_bindings": [
                row.private_payload() for row in self.query_bindings
            ],
            "schema": f"{VERSION}_private_Hippo_cluster_request_v1",
            "study_id": STUDY_ID,
        }
        if include_hash:
            body["self_sha256"] = self.request_sha256
        return body


def _hippo_cluster_request(
    cluster: SearchCluster,
) -> HippoClusterRequest:
    if cluster.block != A_HOLD:
        raise TechqaP1FormalError(
            "official Hippo request formed outside A_hold"
        )
    adapter_input = hippo_adapter.input_payload(
        stage=A_HOLD,
        cluster_ordinal=cluster.cluster_index,
        queries=[
            {
                "ordinal": ordinal,
                "question_text": item.selected.question.question_text,
                "question_title": item.selected.question.question_title,
            }
            for ordinal, item in enumerate(cluster.items)
        ],
        documents=[
            core.document_public_payload(row.public_document)
            for row in cluster.documents
        ],
    )
    bindings = tuple(
        HippoQueryBinding(
            query_ordinal=ordinal,
            work_id=item.selected.work_id,
            query_bytes_sha256=hashlib.sha256(
                core.serialize_query_text(
                    item.selected.question.question_title,
                    item.selected.question.question_text,
                ).encode("utf-8")
            ).hexdigest(),
        )
        for ordinal, item in enumerate(cluster.items)
    )
    provisional = {
        "adapter_input": adapter_input,
        "block": A_HOLD,
        "cluster_index": cluster.cluster_index,
        "corpus_sha256": cluster.corpus_sha256,
        "query_bindings": [
            row.private_payload() for row in bindings
        ],
        "schema": f"{VERSION}_private_Hippo_cluster_request_v1",
        "study_id": STUDY_ID,
    }
    return HippoClusterRequest(
        block=A_HOLD,
        cluster_index=cluster.cluster_index,
        corpus_sha256=cluster.corpus_sha256,
        adapter_input=adapter_input,
        query_bindings=bindings,
        request_sha256=stable_hash(provisional),
    )


@dataclass(frozen=True, slots=True)
class ActionRow:
    block: str
    cluster_index: int
    work_id: str
    corpus_sha256: str
    slate: core.ActionSlate
    raw_top5_ordinals: tuple[int, ...]
    e0: core.PolicyDecision
    e1: core.PolicyDecision | None

    def __post_init__(self) -> None:
        if self.block not in BLOCKS:
            raise TechqaP1FormalError("action row block drifted")
        _identifier(self.work_id, field="action work ID")
        _hex64(self.corpus_sha256, field="action corpus")
        if (
            not isinstance(self.slate, core.ActionSlate)
            or len(self.raw_top5_ordinals) != TOP_K
            or len(set(self.raw_top5_ordinals)) != TOP_K
            or not isinstance(self.e0, core.PolicyDecision)
            or self.e0.evaluator_id != "E0"
            or self.e0.stage
            != (F_SEARCH if self.block == A_FORM else self.block)
        ):
            raise TechqaP1FormalError("action row E0 binding drifted")
        expected_raw = self.slate.action(
            core.R0_RAW_BM25
        ).top5_document_ordinals
        if self.raw_top5_ordinals != expected_raw:
            raise TechqaP1FormalError("RAW is not frozen complete-query BM25")
        if self.block == A_FORM:
            if self.e1 is not None:
                raise TechqaP1FormalError(
                    "A_form cannot contain fitted E1"
                )
        else:
            if (
                not isinstance(self.e1, core.PolicyDecision)
                or self.e1.evaluator_id != "E1"
                or self.e1.stage != self.block
            ):
                raise TechqaP1FormalError("action row E1 binding drifted")

    def private_payload(self) -> dict[str, object]:
        arms: dict[str, object] = {
            "E0": _decision_payload(self.e0),
            "RAW": {
                "recipe_id": core.R0_RAW_BM25,
                "top5_document_ordinals": list(
                    self.raw_top5_ordinals
                ),
            },
        }
        if self.e1 is not None:
            arms["E1"] = _decision_payload(self.e1)
        return {
            "action_slate": self.slate.audit_payload(),
            "arms": arms,
            "block": self.block,
            "cluster_index": self.cluster_index,
            "corpus_sha256": self.corpus_sha256,
            "work_id": self.work_id,
        }


def _decision_payload(value: core.PolicyDecision) -> dict[str, object]:
    return {
        "conservative_minimum_delta": _fraction_payload(
            value.conservative_minimum_delta
        ),
        "e0_recipe_id": value.e0_recipe_id,
        "evaluator_id": value.evaluator_id,
        "fallback_to_e0": value.fallback_to_e0,
        "matched_signature": (
            list(value.matched_signature)
            if value.matched_signature is not None
            else None
        ),
        "selected_recipe_id": value.selected_recipe_id,
        "stage": value.stage,
        "top5_document_ordinals": list(value.top5_document_ordinals),
    }


@dataclass(frozen=True, slots=True)
class PreparedStage:
    block: str
    clusters: tuple[SearchCluster, ...]
    actions: tuple[ActionRow, ...]
    qrels: tuple[PrivateQrel, ...]
    hippo_cluster_requests: tuple[HippoClusterRequest, ...]
    action_archive_sha256: str
    qrel_archive_sha256: str

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCKS
            or len(self.clusters) != CLUSTER_COUNTS[self.block]
            or any(row.block != self.block for row in self.clusters)
            or any(row.block != self.block for row in self.actions)
            or any(row.block != self.block for row in self.qrels)
        ):
            raise TechqaP1FormalError("prepared stage binding drifted")
        expected_count = (
            BLOCK_FAMILY_QUOTAS[self.block] * len(FAMILIES)
        )
        if (
            len(self.actions) != expected_count
            or len(self.qrels) != expected_count
            or len({row.work_id for row in self.actions}) != expected_count
            or {row.work_id for row in self.actions}
            != {row.work_id for row in self.qrels}
        ):
            raise TechqaP1FormalError("prepared stage coverage drifted")
        requests = tuple(self.hippo_cluster_requests)
        if self.block == A_HOLD:
            if (
                len(requests) != CLUSTER_COUNTS[A_HOLD]
                or any(
                    not isinstance(row, HippoClusterRequest)
                    for row in requests
                )
                or tuple(row.cluster_index for row in requests)
                != tuple(range(CLUSTER_COUNTS[A_HOLD]))
            ):
                raise TechqaP1FormalError(
                    "official Hippo cluster request coverage drifted"
                )
            for request, cluster in zip(
                requests, self.clusters, strict=True
            ):
                if (
                    request.block != self.block
                    or request.cluster_index != cluster.cluster_index
                    or request.corpus_sha256 != cluster.corpus_sha256
                    or tuple(
                        row.work_id for row in request.query_bindings
                    )
                    != tuple(
                        row.selected.work_id for row in cluster.items
                    )
                ):
                    raise TechqaP1FormalError(
                        "official Hippo cluster request lineage drifted"
                    )
        elif requests:
            raise TechqaP1FormalError(
                "official Hippo request exists outside A_hold"
            )
        object.__setattr__(
            self, "hippo_cluster_requests", requests
        )
        action_body = self.private_action_payload(include_hash=False)
        qrel_body = self.private_qrel_payload(include_hash=False)
        if (
            not hmac.compare_digest(
                _hex64(
                    self.action_archive_sha256,
                    field="action archive",
                ),
                stable_hash(action_body),
            )
            or not hmac.compare_digest(
                _hex64(
                    self.qrel_archive_sha256,
                    field="qrel archive",
                ),
                stable_hash(qrel_body),
            )
        ):
            raise TechqaP1FormalError("prepared archive hash drifted")

    @property
    def action_by_work_id(self) -> Mapping[str, ActionRow]:
        return {row.work_id: row for row in self.actions}

    @property
    def qrel_by_work_id(self) -> Mapping[str, PrivateQrel]:
        return {row.work_id: row for row in self.qrels}

    @property
    def hippo_cluster_request_by_index(
        self,
    ) -> Mapping[int, HippoClusterRequest]:
        return {
            row.cluster_index: row
            for row in self.hippo_cluster_requests
        }

    def private_action_payload(
        self, *, include_hash: bool = True
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "actions": [row.private_payload() for row in self.actions],
            "block": self.block,
            "cluster_corpora": [
                row.private_payload() for row in self.clusters
            ],
            "official_HippoRAG_cluster_requests": [
                row.private_payload()
                for row in self.hippo_cluster_requests
            ],
            "qrel_values_present": False,
            "schema": f"{VERSION}_{self.block}_private_action_archive_v1",
            "study_id": STUDY_ID,
        }
        if include_hash:
            body["self_sha256"] = self.action_archive_sha256
        return body

    def private_qrel_payload(
        self, *, include_hash: bool = True
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "action_archive_sha256": self.action_archive_sha256,
            "block": self.block,
            "qrels": [row.private_payload() for row in self.qrels],
            "schema": f"{VERSION}_{self.block}_private_qrel_archive_v1",
            "study_id": STUDY_ID,
        }
        if include_hash:
            body["self_sha256"] = self.qrel_archive_sha256
        return body


def _form_stage(
    source: VerifiedSource,
    block: CohortBlock,
    *,
    e1_model: core.E1Model | None,
) -> PreparedStage:
    if block.block == A_FORM:
        if e1_model is not None:
            raise TechqaP1FormalError(
                "A_form cannot receive a fitted evaluator"
            )
    elif not isinstance(e1_model, core.E1Model):
        raise TechqaP1FormalError(
            f"{block.block} requires the one frozen E1 model"
        )
    clusters = build_search_clusters(source, block)
    hippo_cluster_requests = (
        tuple(_hippo_cluster_request(cluster) for cluster in clusters)
        if block.block == A_HOLD
        else ()
    )
    actions: list[ActionRow] = []
    qrels: list[PrivateQrel] = []
    for cluster in clusters:
        by_id = cluster.document_by_id
        for item in cluster.items:
            question = item.selected.question
            slate = _build_public_slate(question, cluster.documents)
            policy_stage = (
                F_SEARCH if block.block == A_FORM else block.block
            )
            e0 = core.apply_e0(slate, stage=policy_stage)
            e1 = (
                None
                if e1_model is None
                else core.apply_e1(
                    e1_model,
                    slate,
                    stage=block.block,
                )
            )
            actions.append(
                ActionRow(
                    block=block.block,
                    cluster_index=cluster.cluster_index,
                    work_id=item.selected.work_id,
                    corpus_sha256=cluster.corpus_sha256,
                    slate=slate,
                    raw_top5_ordinals=slate.action(
                        core.R0_RAW_BM25
                    ).top5_document_ordinals,
                    e0=e0,
                    e1=e1,
                )
            )
            gold = by_id[question.gold_document_id]
            qrels.append(
                PrivateQrel(
                    block=block.block,
                    cluster_index=cluster.cluster_index,
                    work_id=item.selected.work_id,
                    family=item.selected.family,
                    gold_document_id=question.gold_document_id,
                    gold_document_ordinal=gold.public_document.ordinal,
                    corpus_sha256=cluster.corpus_sha256,
                )
            )
    actions.sort(key=lambda row: row.work_id)
    qrels.sort(key=lambda row: row.work_id)
    action_body = {
        "actions": [row.private_payload() for row in actions],
        "block": block.block,
        "cluster_corpora": [
            row.private_payload() for row in clusters
        ],
        "official_HippoRAG_cluster_requests": [
            row.private_payload() for row in hippo_cluster_requests
        ],
        "qrel_values_present": False,
        "schema": f"{VERSION}_{block.block}_private_action_archive_v1",
        "study_id": STUDY_ID,
    }
    action_hash = stable_hash(action_body)
    qrel_body = {
        "action_archive_sha256": action_hash,
        "block": block.block,
        "qrels": [row.private_payload() for row in qrels],
        "schema": f"{VERSION}_{block.block}_private_qrel_archive_v1",
        "study_id": STUDY_ID,
    }
    return PreparedStage(
        block=block.block,
        clusters=clusters,
        actions=tuple(actions),
        qrels=tuple(qrels),
        hippo_cluster_requests=hippo_cluster_requests,
        action_archive_sha256=action_hash,
        qrel_archive_sha256=stable_hash(qrel_body),
    )


def _recall_at_five(
    top5_document_ordinals: Sequence[int],
    qrel: PrivateQrel,
) -> Fraction:
    values = tuple(top5_document_ordinals)
    if (
        len(values) != TOP_K
        or len(set(values)) != TOP_K
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise TechqaP1FormalError("top5 action is malformed")
    return Fraction(
        int(qrel.gold_document_ordinal in values),
        1,
    )


def _aform_examples(stage: PreparedStage) -> tuple[core.AFormExample, ...]:
    if stage.block != A_FORM:
        raise TechqaP1FormalError("E1 fit attempted outside A_form")
    qrels = stage.qrel_by_work_id
    examples: list[core.AFormExample] = []
    for row in stage.actions:
        qrel = qrels[row.work_id]
        utilities = {
            recipe_id: _recall_at_five(
                row.slate.action(recipe_id).top5_document_ordinals,
                qrel,
            )
            for recipe_id in core.RECIPE_IDS
        }
        examples.append(core.make_aform_example(row.slate, utilities))
    return tuple(examples)


@dataclass(frozen=True, slots=True)
class ExactComparison:
    """Exact paired document-recall comparison at aggregate/family/cluster."""

    left_arm: str
    right_arm: str
    pair_count: int
    left_total: Fraction
    right_total: Fraction
    mean_delta: Fraction
    family_mean_deltas: Mapping[str, Fraction]
    cluster_mean_deltas: Mapping[int, Fraction]
    positive_cluster_count: int
    nonpositive_cluster_count: int
    one_sided_cluster_sign_tail: Fraction

    def __post_init__(self) -> None:
        if (
            not self.left_arm
            or not self.right_arm
            or self.left_arm == self.right_arm
            or type(self.pair_count) is not int
            or self.pair_count <= 0
            or any(
                not isinstance(value, Fraction)
                for value in (
                    self.left_total,
                    self.right_total,
                    self.mean_delta,
                    self.one_sided_cluster_sign_tail,
                )
            )
            or set(self.family_mean_deltas) != set(FAMILIES)
            or any(
                not isinstance(value, Fraction)
                for value in self.family_mean_deltas.values()
            )
            or not self.cluster_mean_deltas
            or any(
                type(key) is not int or not isinstance(value, Fraction)
                for key, value in self.cluster_mean_deltas.items()
            )
            or self.positive_cluster_count
            + self.nonpositive_cluster_count
            != len(self.cluster_mean_deltas)
        ):
            raise TechqaP1FormalError("exact comparison drifted")
        expected_mean = (
            self.left_total - self.right_total
        ) / self.pair_count
        if self.mean_delta != expected_mean:
            raise TechqaP1FormalError("aggregate mean delta drifted")
        positive = sum(
            value > 0 for value in self.cluster_mean_deltas.values()
        )
        if (
            positive != self.positive_cluster_count
            or len(self.cluster_mean_deltas) - positive
            != self.nonpositive_cluster_count
        ):
            raise TechqaP1FormalError("cluster sign counts drifted")
        n = len(self.cluster_mean_deltas)
        expected_tail = Fraction(
            sum(
                math.comb(n, successes)
                for successes in range(positive, n + 1)
            ),
            2**n,
        )
        if self.one_sided_cluster_sign_tail != expected_tail:
            raise TechqaP1FormalError("exact cluster sign tail drifted")

    def aggregate_payload(self) -> dict[str, object]:
        return {
            "left_arm": self.left_arm,
            "left_total_exact_recall_at_5": _fraction_payload(
                self.left_total
            ),
            "mean_delta": _fraction_payload(self.mean_delta),
            "nonpositive_cluster_count": self.nonpositive_cluster_count,
            "one_sided_exact_cluster_sign_tail": _fraction_payload(
                self.one_sided_cluster_sign_tail
            ),
            "pair_count": self.pair_count,
            "positive_cluster_count": self.positive_cluster_count,
            "right_arm": self.right_arm,
            "right_total_exact_recall_at_5": _fraction_payload(
                self.right_total
            ),
        }

    def family_payload(self) -> dict[str, object]:
        return {
            family: _fraction_payload(self.family_mean_deltas[family])
            for family in FAMILIES
        }

    def cluster_payload(self) -> dict[str, object]:
        return {
            str(cluster): _fraction_payload(value)
            for cluster, value in sorted(
                self.cluster_mean_deltas.items()
            )
        }

    def private_payload(self) -> dict[str, object]:
        return {
            "aggregate": self.aggregate_payload(),
            "cluster_mean_deltas": self.cluster_payload(),
            "family_mean_deltas": self.family_payload(),
        }


def compare_exact_rows(
    *,
    left_arm: str,
    right_arm: str,
    rows: Sequence[
        tuple[str, int, Fraction, Fraction]
    ],
) -> ExactComparison:
    """Compare ``(family, cluster, left, right)`` exact utility rows."""

    checked = tuple(rows)
    if not checked:
        raise TechqaP1FormalError("paired comparison is empty")
    if any(
        family not in FAMILIES
        or type(cluster) is not int
        or not isinstance(left, Fraction)
        or not isinstance(right, Fraction)
        or left not in {Fraction(0, 1), Fraction(1, 1)}
        or right not in {Fraction(0, 1), Fraction(1, 1)}
        for family, cluster, left, right in checked
    ):
        raise TechqaP1FormalError("paired exact-recall row drifted")
    left_total = sum((row[2] for row in checked), Fraction(0, 1))
    right_total = sum((row[3] for row in checked), Fraction(0, 1))
    family_deltas: dict[str, Fraction] = {}
    for family in FAMILIES:
        values = [
            left - right
            for row_family, _cluster, left, right in checked
            if row_family == family
        ]
        if not values:
            raise TechqaP1FormalError(
                "paired comparison is missing an operational family"
            )
        family_deltas[family] = sum(
            values, Fraction(0, 1)
        ) / len(values)
    cluster_ids = sorted({row[1] for row in checked})
    cluster_deltas: dict[int, Fraction] = {}
    for cluster in cluster_ids:
        values = [
            left - right
            for _family, row_cluster, left, right in checked
            if row_cluster == cluster
        ]
        cluster_deltas[cluster] = sum(
            values, Fraction(0, 1)
        ) / len(values)
    positive = sum(value > 0 for value in cluster_deltas.values())
    tail = Fraction(
        sum(
            math.comb(len(cluster_deltas), successes)
            for successes in range(
                positive, len(cluster_deltas) + 1
            )
        ),
        2 ** len(cluster_deltas),
    )
    return ExactComparison(
        left_arm=left_arm,
        right_arm=right_arm,
        pair_count=len(checked),
        left_total=left_total,
        right_total=right_total,
        mean_delta=(left_total - right_total) / len(checked),
        family_mean_deltas=family_deltas,
        cluster_mean_deltas=cluster_deltas,
        positive_cluster_count=positive,
        nonpositive_cluster_count=len(cluster_deltas) - positive,
        one_sided_cluster_sign_tail=tail,
    )


def promotion_criterion(value: ExactComparison) -> bool:
    return (
        isinstance(value, ExactComparison)
        and value.left_arm == "E1"
        and value.right_arm == "E0"
        and len(value.cluster_mean_deltas) == 4
        and value.mean_delta > 0
        and all(delta > 0 for delta in value.cluster_mean_deltas.values())
        and value.one_sided_cluster_sign_tail == Fraction(1, 16)
    )


def l5_criterion(value: ExactComparison) -> bool:
    return promotion_criterion(value)


def reality_criterion(
    e0_vs_raw: ExactComparison,
    e0_vs_hipporag: ExactComparison,
) -> bool:
    for value, right in (
        (e0_vs_raw, "RAW"),
        (e0_vs_hipporag, "HippoRAG"),
    ):
        if not (
            isinstance(value, ExactComparison)
            and value.left_arm == "E0"
            and value.right_arm == right
            and len(value.cluster_mean_deltas) == 4
            and value.mean_delta > 0
            and all(
                delta > 0
                for delta in value.family_mean_deltas.values()
            )
            and all(
                delta > 0
                for delta in value.cluster_mean_deltas.values()
            )
            and value.one_sided_cluster_sign_tail
            == Fraction(1, 16)
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class PromotionAuthorization:
    comparison_sha256: str
    authorization_sha256: str

    def __post_init__(self) -> None:
        comparison = _hex64(
            self.comparison_sha256, field="promotion comparison"
        )
        expected = stable_hash(
            {
                "comparison_sha256": comparison,
                "criteria": {
                    "aggregate_mean_delta_strictly_positive": True,
                    "all_four_cluster_mean_deltas_strictly_positive": True,
                    "one_sided_exact_cluster_sign_tail": [1, 16],
                },
                "schema": f"{VERSION}_M_search_authorization_v1",
                "status": "A_hold_E1_promoted",
                "study_id": STUDY_ID,
            }
        )
        if not hmac.compare_digest(
            _hex64(
                self.authorization_sha256,
                field="promotion authorization",
            ),
            expected,
        ):
            raise TechqaP1FormalError(
                "promotion authorization binding drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "authorization_sha256": self.authorization_sha256,
            "comparison_sha256": self.comparison_sha256,
            "status": "A_hold_E1_promoted",
        }


def authorize_m_search(
    comparison: ExactComparison,
) -> PromotionAuthorization | None:
    if not promotion_criterion(comparison):
        return None
    comparison_hash = stable_hash(comparison.private_payload())
    body = {
        "comparison_sha256": comparison_hash,
        "criteria": {
            "aggregate_mean_delta_strictly_positive": True,
            "all_four_cluster_mean_deltas_strictly_positive": True,
            "one_sided_exact_cluster_sign_tail": [1, 16],
        },
        "schema": f"{VERSION}_M_search_authorization_v1",
        "status": "A_hold_E1_promoted",
        "study_id": STUDY_ID,
    }
    return PromotionAuthorization(
        comparison_sha256=comparison_hash,
        authorization_sha256=stable_hash(body),
    )


@dataclass(frozen=True, slots=True)
class PreparedStudy:
    source: VerifiedSource
    selection: CohortSelection
    a_form: PreparedStage
    f_search: PreparedStage
    a_hold: PreparedStage
    e1_model: core.E1Model
    e1_model_sha256: str
    prepromotion_archive_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, VerifiedSource)
            or not isinstance(self.selection, CohortSelection)
            or self.a_form.block != A_FORM
            or self.f_search.block != F_SEARCH
            or self.a_hold.block != A_HOLD
            or not isinstance(self.e1_model, core.E1Model)
        ):
            raise TechqaP1FormalError("prepared study lineage drifted")
        expected_model = stable_hash(self.e1_model.payload())
        if not hmac.compare_digest(
            _hex64(self.e1_model_sha256, field="E1 model"),
            expected_model,
        ):
            raise TechqaP1FormalError("E1 model hash drifted")
        expected_archive = stable_hash(
            self.prepromotion_private_payload(include_hash=False)
        )
        if not hmac.compare_digest(
            _hex64(
                self.prepromotion_archive_sha256,
                field="prepromotion private archive",
            ),
            expected_archive,
        ):
            raise TechqaP1FormalError(
                "prepromotion private archive hash drifted"
            )

    @property
    def hippo_cluster_requests(self) -> tuple[HippoClusterRequest, ...]:
        return self.a_hold.hippo_cluster_requests

    def prepromotion_private_payload(
        self, *, include_hash: bool = True
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "M_search_action_materialized": False,
            "cohort_selection": self.selection.private_payload(),
            "e1_model": self.e1_model.payload(),
            "source_commitments": self.source.commitments.payload(),
            "stages": {
                A_FORM: {
                    "action_archive": (
                        self.a_form.private_action_payload()
                    ),
                    "qrel_archive": self.a_form.private_qrel_payload(),
                },
                F_SEARCH: {
                    "action_archive": (
                        self.f_search.private_action_payload()
                    ),
                    "qrel_archive": (
                        self.f_search.private_qrel_payload()
                    ),
                },
                A_HOLD: {
                    "action_archive": (
                        self.a_hold.private_action_payload()
                    ),
                    "qrel_archive": self.a_hold.private_qrel_payload(),
                },
            },
            "schema": f"{VERSION}_prepromotion_private_archive_v1",
            "study_id": STUDY_ID,
        }
        if include_hash:
            body["self_sha256"] = self.prepromotion_archive_sha256
        return body


def prepare_formal_study(
    source: VerifiedSource,
    *,
    hmac_secret: bytes,
) -> PreparedStudy:
    """Select once, fit E1 once, and form all pre-promotion actions."""

    selection = select_private_cohorts(
        source, hmac_secret=hmac_secret
    )
    a_form = _form_stage(
        source,
        selection.block(A_FORM),
        e1_model=None,
    )
    e1_model = core.fit_e1(_aform_examples(a_form))
    f_search = _form_stage(
        source,
        selection.block(F_SEARCH),
        e1_model=e1_model,
    )
    a_hold = _form_stage(
        source,
        selection.block(A_HOLD),
        e1_model=e1_model,
    )
    model_hash = stable_hash(e1_model.payload())
    provisional = {
        "M_search_action_materialized": False,
        "cohort_selection": selection.private_payload(),
        "e1_model": e1_model.payload(),
        "source_commitments": source.commitments.payload(),
        "stages": {
            A_FORM: {
                "action_archive": a_form.private_action_payload(),
                "qrel_archive": a_form.private_qrel_payload(),
            },
            F_SEARCH: {
                "action_archive": f_search.private_action_payload(),
                "qrel_archive": f_search.private_qrel_payload(),
            },
            A_HOLD: {
                "action_archive": a_hold.private_action_payload(),
                "qrel_archive": a_hold.private_qrel_payload(),
            },
        },
        "schema": f"{VERSION}_prepromotion_private_archive_v1",
        "study_id": STUDY_ID,
    }
    return PreparedStudy(
        source=source,
        selection=selection,
        a_form=a_form,
        f_search=f_search,
        a_hold=a_hold,
        e1_model=e1_model,
        e1_model_sha256=model_hash,
        prepromotion_archive_sha256=stable_hash(provisional),
    )


@dataclass(frozen=True, slots=True)
class OfficialHippoClusterRun:
    """One canonical adapter input plus its validated safe cluster output."""

    adapter_input: Mapping[str, object]
    safe_output: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            canonical_input = json.loads(
                hippo_adapter.canonical_bytes(
                    self.adapter_input
                ).decode("ascii")
            )
            canonical_output = json.loads(
                hippo_adapter.canonical_bytes(
                    self.safe_output
                ).decode("ascii")
            )
            cluster = hippo_adapter.validate_input(canonical_input)
            checked_output = hippo_adapter.validate_output(
                canonical_output,
                expected_input=canonical_input,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            hippo_adapter.TechqaP1OfficialHippoRAGError,
        ) as exc:
            raise TechqaP1FormalError(
                "official Hippo cluster run validation failed"
            ) from exc
        if cluster.stage != A_HOLD:
            raise TechqaP1FormalError(
                "official Hippo cluster run is outside A_hold"
            )
        object.__setattr__(self, "adapter_input", canonical_input)
        object.__setattr__(self, "safe_output", checked_output)

    @property
    def cluster_index(self) -> int:
        return int(self.adapter_input["cluster_ordinal"])

    def private_receipt_payload(self) -> dict[str, object]:
        return {
            "adapter_input_self_sha256": self.adapter_input[
                "self_sha256"
            ],
            "adapter_output": dict(self.safe_output),
            "cluster_index": self.cluster_index,
            "schema": f"{VERSION}_verified_Hippo_cluster_run_v1",
            "study_id": STUDY_ID,
        }


@dataclass(frozen=True, slots=True)
class BoundHippoAction:
    work_id: str
    cluster_index: int
    query_ordinal: int
    query_bytes_sha256: str
    top5_ordinals: tuple[int, ...]
    top5_document_ids: tuple[str, ...]
    cluster_request_sha256: str
    adapter_input_self_sha256: str
    adapter_output_self_sha256: str
    adapter_outer_binding_sha256: str
    adapter_attempt_marker_self_sha256: str
    adapter_attempt_marker_file_sha256: str
    adapter_inner_receipt_sha256: str
    adapter_inner_output_sha256: str
    output_binding_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.work_id, field="bound Hippo work ID")
        if (
            type(self.cluster_index) is not int
            or not 0 <= self.cluster_index < CLUSTER_COUNTS[A_HOLD]
            or type(self.query_ordinal) is not int
            or not 0
            <= self.query_ordinal
            < hippo_adapter.EXPECTED_QUERY_COUNT
        ):
            raise TechqaP1FormalError(
                "bound Hippo cluster/query ordinal drifted"
            )
        for field, value in (
            ("bound Hippo query", self.query_bytes_sha256),
            ("bound Hippo cluster request", self.cluster_request_sha256),
            ("bound Hippo adapter input", self.adapter_input_self_sha256),
            ("bound Hippo adapter output", self.adapter_output_self_sha256),
            ("bound Hippo outer binding", self.adapter_outer_binding_sha256),
            (
                "bound Hippo attempt marker",
                self.adapter_attempt_marker_self_sha256,
            ),
            (
                "bound Hippo attempt marker file",
                self.adapter_attempt_marker_file_sha256,
            ),
            ("bound Hippo inner receipt", self.adapter_inner_receipt_sha256),
            ("bound Hippo inner output", self.adapter_inner_output_sha256),
            ("bound Hippo output", self.output_binding_sha256),
        ):
            _hex64(value, field=field)
        if (
            len(self.top5_ordinals) != TOP_K
            or len(set(self.top5_ordinals)) != TOP_K
            or any(
                type(value) is not int or value < 0
                for value in self.top5_ordinals
            )
            or len(self.top5_document_ids) != TOP_K
            or len(set(self.top5_document_ids)) != TOP_K
        ):
            raise TechqaP1FormalError("bound Hippo top5 drifted")
        for value in self.top5_document_ids:
            _identifier(value, field="bound Hippo document ID")

    def private_payload(self) -> dict[str, object]:
        return {
            "adapter_attempt_marker_file_sha256": (
                self.adapter_attempt_marker_file_sha256
            ),
            "adapter_attempt_marker_self_sha256": (
                self.adapter_attempt_marker_self_sha256
            ),
            "adapter_inner_output_sha256": (
                self.adapter_inner_output_sha256
            ),
            "adapter_inner_receipt_sha256": (
                self.adapter_inner_receipt_sha256
            ),
            "adapter_input_self_sha256": (
                self.adapter_input_self_sha256
            ),
            "adapter_outer_binding_sha256": (
                self.adapter_outer_binding_sha256
            ),
            "adapter_output_self_sha256": (
                self.adapter_output_self_sha256
            ),
            "cluster_index": self.cluster_index,
            "cluster_request_sha256": self.cluster_request_sha256,
            "output_binding_sha256": self.output_binding_sha256,
            "query_bytes_sha256": self.query_bytes_sha256,
            "query_ordinal": self.query_ordinal,
            "top5_document_ids": list(self.top5_document_ids),
            "top5_ordinals": list(self.top5_ordinals),
            "work_id": self.work_id,
        }


def bind_official_hippo_cluster_runs(
    stage: PreparedStage,
    runs: Sequence[OfficialHippoClusterRun],
) -> Mapping[str, BoundHippoAction]:
    """Validate four official cluster receipts, then bind all 36 actions."""

    if stage.block != A_HOLD:
        raise TechqaP1FormalError(
            "official Hippo binding is outside A_hold"
        )
    checked = tuple(runs)
    if (
        len(checked) != CLUSTER_COUNTS[A_HOLD]
        or any(
            not isinstance(row, OfficialHippoClusterRun)
            for row in checked
        )
        or len({row.cluster_index for row in checked}) != len(checked)
    ):
        raise TechqaP1FormalError(
            "official Hippo cluster run coverage is not exact"
        )
    by_cluster = {row.cluster_index: row for row in checked}
    expected_indices = set(range(CLUSTER_COUNTS[A_HOLD]))
    requests = stage.hippo_cluster_request_by_index
    clusters = {
        row.cluster_index: row for row in stage.clusters
    }
    if set(by_cluster) != expected_indices or set(requests) != expected_indices:
        raise TechqaP1FormalError(
            "official Hippo cluster index coverage drifted"
        )
    bound: dict[str, BoundHippoAction] = {}
    for cluster_index in sorted(expected_indices):
        run = by_cluster[cluster_index]
        request = requests[cluster_index]
        cluster = clusters[cluster_index]
        if hippo_adapter.canonical_bytes(
            run.adapter_input
        ) != hippo_adapter.canonical_bytes(request.adapter_input):
            raise TechqaP1FormalError(
                "official Hippo exact adapter input mismatch"
            )
        try:
            validated_cluster = hippo_adapter.validate_input(
                run.adapter_input
            )
            output = hippo_adapter.validate_output(
                run.safe_output,
                expected_input=run.adapter_input,
            )
        except hippo_adapter.TechqaP1OfficialHippoRAGError as exc:
            raise TechqaP1FormalError(
                "official Hippo cluster receipt validation failed"
            ) from exc
        if (
            validated_cluster.stage != A_HOLD
            or validated_cluster.cluster_ordinal != cluster_index
            or output["stage"] != A_HOLD
            or output["cluster_ordinal"] != cluster_index
            or output["outer_input_self_sha256"]
            != request.adapter_input["self_sha256"]
            or output["outer_binding_sha256"]
            != hippo_adapter.outer_binding(validated_cluster)
        ):
            raise TechqaP1FormalError(
                "official Hippo stage/hash mapping drifted"
            )
        rows = output["rows"]
        if not isinstance(rows, list):
            raise TechqaP1FormalError(
                "official Hippo cluster rows disappeared"
            )
        document_ids = tuple(
            row.document_id for row in cluster.documents
        )
        for binding, row in zip(
            request.query_bindings, rows, strict=True
        ):
            if (
                not isinstance(row, Mapping)
                or row.get("query_ordinal") != binding.query_ordinal
            ):
                raise TechqaP1FormalError(
                    "official Hippo query ordinal mapping drifted"
                )
            raw_ordinals = row.get("top5_document_ordinals")
            if not isinstance(raw_ordinals, list):
                raise TechqaP1FormalError(
                    "official Hippo top5 row disappeared"
                )
            ordinals = tuple(raw_ordinals)
            if any(value >= len(document_ids) for value in ordinals):
                raise TechqaP1FormalError(
                    "official Hippo output escaped the shared corpus"
                )
            resolved_ids = tuple(
                document_ids[value] for value in ordinals
            )
            output_body = {
                "adapter_attempt_marker_file_sha256": output[
                    "attempt_marker_file_sha256"
                ],
                "adapter_attempt_marker_self_sha256": output[
                    "attempt_marker_self_sha256"
                ],
                "adapter_inner_output_sha256": output[
                    "inner_output_sha256"
                ],
                "adapter_inner_receipt_sha256": output[
                    "inner_receipt_sha256"
                ],
                "adapter_input_self_sha256": request.adapter_input[
                    "self_sha256"
                ],
                "adapter_outer_binding_sha256": output[
                    "outer_binding_sha256"
                ],
                "adapter_output_self_sha256": output["self_sha256"],
                "cluster_index": cluster_index,
                "cluster_request_sha256": request.request_sha256,
                "query_bytes_sha256": binding.query_bytes_sha256,
                "query_ordinal": binding.query_ordinal,
                "top5_document_ids": list(resolved_ids),
                "top5_ordinals": list(ordinals),
                "work_id": binding.work_id,
            }
            bound[binding.work_id] = BoundHippoAction(
                work_id=binding.work_id,
                cluster_index=cluster_index,
                query_ordinal=binding.query_ordinal,
                query_bytes_sha256=binding.query_bytes_sha256,
                top5_ordinals=ordinals,
                top5_document_ids=resolved_ids,
                cluster_request_sha256=request.request_sha256,
                adapter_input_self_sha256=str(
                    request.adapter_input["self_sha256"]
                ),
                adapter_output_self_sha256=str(output["self_sha256"]),
                adapter_outer_binding_sha256=str(
                    output["outer_binding_sha256"]
                ),
                adapter_attempt_marker_self_sha256=str(
                    output["attempt_marker_self_sha256"]
                ),
                adapter_attempt_marker_file_sha256=str(
                    output["attempt_marker_file_sha256"]
                ),
                adapter_inner_receipt_sha256=str(
                    output["inner_receipt_sha256"]
                ),
                adapter_inner_output_sha256=str(
                    output["inner_output_sha256"]
                ),
                output_binding_sha256=stable_hash(output_body),
            )
    if set(bound) != {row.work_id for row in stage.actions}:
        raise TechqaP1FormalError(
            "official Hippo item binding coverage drifted"
        )
    return bound


@dataclass(frozen=True, slots=True)
class StageScore:
    block: str
    comparisons: Mapping[str, ExactComparison]
    private_rows: tuple[Mapping[str, object], ...]
    score_archive_sha256: str

    def __post_init__(self) -> None:
        if self.block not in {F_SEARCH, A_HOLD, M_SEARCH}:
            raise TechqaP1FormalError("score block drifted")
        if any(
            not isinstance(value, ExactComparison)
            for value in self.comparisons.values()
        ):
            raise TechqaP1FormalError("score comparison drifted")
        expected = stable_hash(self.private_payload(include_hash=False))
        if not hmac.compare_digest(
            _hex64(
                self.score_archive_sha256,
                field="private score archive",
            ),
            expected,
        ):
            raise TechqaP1FormalError("score archive hash drifted")

    def private_payload(
        self, *, include_hash: bool = True
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "block": self.block,
            "comparisons": {
                key: value.private_payload()
                for key, value in sorted(self.comparisons.items())
            },
            "primary_metric": "exact_singleton_DOCUMENT_recall_at_5",
            "rows": [dict(row) for row in self.private_rows],
            "schema": f"{VERSION}_{self.block}_private_score_archive_v1",
            "study_id": STUDY_ID,
        }
        if include_hash:
            body["self_sha256"] = self.score_archive_sha256
        return body


def _stage_arm_utilities(
    stage: PreparedStage,
    *,
    hippo: Mapping[str, BoundHippoAction] | None = None,
) -> tuple[dict[str, object], ...]:
    if stage.block == A_HOLD:
        if hippo is None or set(hippo) != {
            row.work_id for row in stage.actions
        }:
            raise TechqaP1FormalError(
                "A_hold official Hippo action population is incomplete"
            )
    elif hippo is not None:
        raise TechqaP1FormalError(
            "Hippo actions supplied outside A_hold"
        )
    qrels = stage.qrel_by_work_id
    rows: list[dict[str, object]] = []
    for action in stage.actions:
        if action.e1 is None:
            raise TechqaP1FormalError(
                "scored stage has no frozen E1 action"
            )
        qrel = qrels[action.work_id]
        arm_actions: dict[str, tuple[int, ...]] = {
            "E0": action.e0.top5_document_ordinals,
            "E1": action.e1.top5_document_ordinals,
            "RAW": action.raw_top5_ordinals,
        }
        if hippo is not None:
            arm_actions["HippoRAG"] = hippo[
                action.work_id
            ].top5_ordinals
        utilities = {
            arm: _recall_at_five(top5, qrel)
            for arm, top5 in arm_actions.items()
        }
        rows.append(
            {
                "cluster_index": action.cluster_index,
                "family": qrel.family,
                "utilities": utilities,
                "work_id": action.work_id,
            }
        )
    return tuple(rows)


def _score_stage(
    stage: PreparedStage,
    *,
    hippo: Mapping[str, BoundHippoAction] | None = None,
) -> StageScore:
    rows = _stage_arm_utilities(stage, hippo=hippo)
    pairs = [("E1", "E0")]
    if stage.block == A_HOLD:
        pairs.extend((("E0", "RAW"), ("E0", "HippoRAG")))
    comparisons: dict[str, ExactComparison] = {}
    for left, right in pairs:
        comparison = compare_exact_rows(
            left_arm=left,
            right_arm=right,
            rows=[
                (
                    str(row["family"]),
                    int(row["cluster_index"]),
                    row["utilities"][left],  # type: ignore[index]
                    row["utilities"][right],  # type: ignore[index]
                )
                for row in rows
            ],
        )
        comparisons[f"{left}_minus_{right}"] = comparison
    private_rows = tuple(
        {
            "cluster_index": row["cluster_index"],
            "family": row["family"],
            "utilities": {
                arm: _fraction_payload(value)
                for arm, value in row["utilities"].items()  # type: ignore[union-attr]
            },
            "work_id": row["work_id"],
        }
        for row in rows
    )
    provisional = {
        "block": stage.block,
        "comparisons": {
            key: value.private_payload()
            for key, value in sorted(comparisons.items())
        },
        "primary_metric": "exact_singleton_DOCUMENT_recall_at_5",
        "rows": [dict(row) for row in private_rows],
        "schema": f"{VERSION}_{stage.block}_private_score_archive_v1",
        "study_id": STUDY_ID,
    }
    return StageScore(
        block=stage.block,
        comparisons=comparisons,
        private_rows=private_rows,
        score_archive_sha256=stable_hash(provisional),
    )


def _materialize_m_search(
    prepared: PreparedStudy,
    authorization: PromotionAuthorization | None,
) -> PreparedStage:
    if authorization is None:
        raise TechqaP1FormalError(
            "M_search action materialization lacks promotion authorization"
        )
    if not isinstance(authorization, PromotionAuthorization):
        raise TechqaP1FormalError(
            "M_search promotion authorization type drifted"
        )
    return _form_stage(
        prepared.source,
        prepared.selection.block(M_SEARCH),
        e1_model=prepared.e1_model,
    )


@dataclass(frozen=True, slots=True)
class FormalResult:
    safe_terminal: Mapping[str, object]
    private_archive: Mapping[str, object]
    private_archive_sha256: str
    m_search: PreparedStage | None

    def __post_init__(self) -> None:
        expected_private = stable_hash(self.private_archive)
        if not hmac.compare_digest(
            _hex64(
                self.private_archive_sha256,
                field="final private archive",
            ),
            expected_private,
        ):
            raise TechqaP1FormalError("final private archive hash drifted")
        if self.safe_terminal.get("private_archive_sha256") != (
            self.private_archive_sha256
        ):
            raise TechqaP1FormalError(
                "safe terminal/private archive commitment drifted"
            )
        if (
            self.safe_terminal.get(
                "item_query_document_qrel_action_values_published"
            )
            is not False
            or self.safe_terminal.get(
                "cohort_question_and_normalized_query_disjoint"
            )
            is not True
            or self.safe_terminal.get(
                "cohort_gold_document_disjoint"
            )
            is not False
            or self.safe_terminal.get(
                "shared_corpus_and_gold_overlap_allowed"
            )
            is not True
            or self.safe_terminal.get("M_search_untouched_scope")
            != "query_and_action_not_document_disjoint"
            or self.safe_terminal.get(
                "online_or_API_evaluator_call_count"
            )
            != 0
            or self.safe_terminal.get(
                "retry_replay_resample_model_provider_candidate_parser_"
                "family_quota_or_gate_change_count"
            )
            != 0
        ):
            raise TechqaP1FormalError("safe terminal policy drifted")


def _safe_comparison(value: ExactComparison) -> dict[str, object]:
    return {
        "aggregate": value.aggregate_payload(),
        "cluster_mean_deltas": value.cluster_payload(),
        "family_mean_deltas": value.family_payload(),
    }


def _finalize_once(
    prepared: PreparedStudy,
    hippo_cluster_runs: Sequence[OfficialHippoClusterRun],
) -> FormalResult:
    bound_hippo = bind_official_hippo_cluster_runs(
        prepared.a_hold, hippo_cluster_runs
    )
    f_score = _score_stage(prepared.f_search)
    ahold_score = _score_stage(
        prepared.a_hold, hippo=bound_hippo
    )
    promotion_comparison = ahold_score.comparisons["E1_minus_E0"]
    authorization = authorize_m_search(promotion_comparison)
    reality = reality_criterion(
        ahold_score.comparisons["E0_minus_RAW"],
        ahold_score.comparisons["E0_minus_HippoRAG"],
    )
    m_stage: PreparedStage | None = None
    m_score: StageScore | None = None
    if authorization is not None:
        m_stage = _materialize_m_search(prepared, authorization)
        m_score = _score_stage(m_stage)
    hippo_private = {
        work_id: value.private_payload()
        for work_id, value in sorted(bound_hippo.items())
    }
    private_archive = {
        "M_search_action_materialized": m_stage is not None,
        "M_search_authorization": (
            authorization.payload()
            if authorization is not None
            else None
        ),
        "bound_official_HippoRAG_outputs": hippo_private,
        "verified_official_HippoRAG_cluster_runs": {
            str(run.cluster_index): run.private_receipt_payload()
            for run in sorted(
                hippo_cluster_runs,
                key=lambda row: row.cluster_index,
            )
        },
        "prepromotion_archive": (
            prepared.prepromotion_private_payload()
        ),
        "scores": {
            F_SEARCH: f_score.private_payload(),
            A_HOLD: ahold_score.private_payload(),
            M_SEARCH: (
                m_score.private_payload()
                if m_score is not None
                else None
            ),
        },
        "stage_M_search": (
            {
                "action_archive": m_stage.private_action_payload(),
                "qrel_archive": m_stage.private_qrel_payload(),
            }
            if m_stage is not None
            else None
        ),
        "schema": f"{VERSION}_final_private_archive_v1",
        "study_id": STUDY_ID,
    }
    private_hash = stable_hash(private_archive)
    promotion_passed = authorization is not None
    l5_comparison = (
        m_score.comparisons["E1_minus_E0"]
        if m_score is not None
        else None
    )
    safe_terminal = self_hashed(
        {
            "A_hold": {
                "E0_minus_HippoRAG": _safe_comparison(
                    ahold_score.comparisons["E0_minus_HippoRAG"]
                ),
                "E0_minus_RAW": _safe_comparison(
                    ahold_score.comparisons["E0_minus_RAW"]
                ),
                "E1_minus_E0_promotion": _safe_comparison(
                    promotion_comparison
                ),
                "promotion_passed": promotion_passed,
                "reality_primary_passed": reality,
            },
            "F_search": {
                "E1_minus_E0_audit_only": _safe_comparison(
                    f_score.comparisons["E1_minus_E0"]
                ),
                "changed_E0_or_E1": False,
            },
            "M_search": {
                "E1_minus_E0_L5": (
                    _safe_comparison(l5_comparison)
                    if l5_comparison is not None
                    else None
                ),
                "L5_passed": (
                    l5_criterion(l5_comparison)
                    if l5_comparison is not None
                    else None
                ),
                "actions_materialized_after_promotion": (
                    m_stage is not None
                ),
            },
            "aggregate_only_public_terminal": True,
            "candidate_original_order_used": False,
            "cohort_gold_document_disjoint": False,
            "cohort_question_and_normalized_query_disjoint": True,
            "cohort_selection_sha256": prepared.selection.selection_sha256,
            "e1_model_sha256": prepared.e1_model_sha256,
            "item_query_document_qrel_action_values_published": False,
            "M_search_untouched_scope": (
                "query_and_action_not_document_disjoint"
            ),
            "online_or_API_evaluator_call_count": 0,
            "operational_strata_are_native_gold_relation_families": False,
            "primary_metric": "exact_singleton_DOCUMENT_recall_at_5",
            "private_archive_sha256": private_hash,
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": f"{VERSION}_safe_terminal_v1",
            "shared_corpus_and_gold_overlap_allowed": True,
            "source_commitments": prepared.source.commitments.payload(),
            "status": (
                "terminal_complete_after_A_hold_promotion_and_M_search"
                if promotion_passed
                else "terminal_A_hold_E1_not_promoted_M_search_unmaterialized"
            ),
            "study_id": STUDY_ID,
        }
    )
    return FormalResult(
        safe_terminal=safe_terminal,
        private_archive=private_archive,
        private_archive_sha256=private_hash,
        m_search=m_stage,
    )


class OneShotFormalController:
    """In-memory one-use latch; the outer boundary owns durable attempt custody."""

    def __init__(self, prepared: PreparedStudy) -> None:
        if not isinstance(prepared, PreparedStudy):
            raise TechqaP1FormalError("prepared study is absent")
        self._prepared = prepared
        self._consumed = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    def finalize(
        self,
        official_hipporag_cluster_runs: Sequence[
            OfficialHippoClusterRun
        ],
    ) -> FormalResult:
        if self._consumed:
            raise TechqaP1FormalError(
                "formal controller is already consumed; replay is forbidden"
            )
        # Consume before validating external output: a malformed or partial
        # formal result fails closed and cannot be retried through this object.
        self._consumed = True
        return _finalize_once(
            self._prepared, official_hipporag_cluster_runs
        )


__all__ = [
    "A_FORM",
    "A_HOLD",
    "BLOCKS",
    "BLOCK_FAMILY_QUOTAS",
    "BLOCK_TO_SPLIT",
    "BoundHippoAction",
    "CANDIDATE_DOCUMENT_COUNT",
    "CLUSTER_COUNTS",
    "ClusterDocument",
    "ClusterItem",
    "CohortBlock",
    "CohortSelection",
    "ExactComparison",
    "FAMILIES",
    "FAMILY_IDS",
    "FAMILY_QUOTAS",
    "F_SEARCH",
    "FormalResult",
    "HMAC_SECRET_BYTES",
    "HippoClusterRequest",
    "HippoQueryBinding",
    "INFORMATION",
    "ITEMS_PER_FAMILY_PER_CLUSTER",
    "M_SEARCH",
    "MAXIMUM_CLUSTER_DOCUMENT_COUNT",
    "OfficialHippoClusterRun",
    "OneShotFormalController",
    "PROCEDURE",
    "PROCEDURE_INDICATORS",
    "PreparedStage",
    "PreparedStudy",
    "PrivateQrel",
    "PromotionAuthorization",
    "SearchCluster",
    "SelectedItem",
    "SourceCommitments",
    "SOURCE_MINIMUM_FAMILY_COUNTS",
    "STUDY_ID",
    "TOP_K",
    "TROUBLESHOOT",
    "TROUBLESHOOT_INDICATORS",
    "TechqaP1FormalError",
    "VERSION",
    "VerifiedDocument",
    "VerifiedQuestion",
    "VerifiedSource",
    "authorize_m_search",
    "bind_official_hippo_cluster_runs",
    "build_search_clusters",
    "canonical_bytes",
    "compare_exact_rows",
    "l5_criterion",
    "operational_family",
    "prepare_formal_study",
    "promotion_criterion",
    "public_action_projection",
    "reality_criterion",
    "select_private_cohorts",
    "select_question_cohorts",
    "self_hashed",
    "stable_hash",
]
