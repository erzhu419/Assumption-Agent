"""Outcome-blind source qualification for the official QASPER v0.3 release.

The public entry point deliberately runs the parser in an isolated child
process.  The child opens only the caller-supplied train/dev tarball and a
previously disclosed local reference file, and emits one aggregate-only JSON
receipt.  It never selects rows, samples rows, reads a selection secret, or
emits paper/question identifiers or dataset text.

Content nodes follow the final frozen QASPER graph contract: only non-empty
body paragraphs are deduplicated by exact Unicode text.  Title, abstract,
figure/table captions, and ``FLOAT SELECTED`` placeholders are not output
nodes.  Caption counts remain part of source qualification, while positional
paragraph occurrences remain distinct for section-membership and adjacency
accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


VERSION = "qasper_fresh_source_qualification_v03"
SCHEMA = VERSION
EXPECTED_MEMBERS = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
}
MIN_CONTENT_NODES = 5
MAX_CONTENT_NODES = 128
FLOAT_EVIDENCE_PREFIX = "FLOAT SELECTED"
TRAIN_SOURCE_INSERTION_EXCLUSION_COUNT = 16
PUBLIC_EXAMPLE_PAPER_ID_DENYLIST = frozenset(
    {
        "1705.00571",
        "1706.01875",
        "1707.03904",
        "1712.02555",
        "1802.08969",
        "1906.08593",
        "1909.09070",
        "2002.05829",
        "2004.04721",
    }
)
FORMAL_ARCHIVE_SHA256 = (
    "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
)
FORMAL_ARCHIVE_SIZE = 10_835_856
FORMAL_REFERENCE_SHA256 = (
    "880dc17ee85bccb59de79f374ba30dede1144db087bc8b0d6a39477dcdefa1ba"
)
FORMAL_CUSTODY_SHA256 = (
    "dcf6018d0fc508e538c6b1036ae036be34fa47b21615948464c94ea6deddc72a"
)
FORMAL_CUSTODY_COMMIT = "f1b4cb26"
FORMAL_MINIMUM_DISTINCT_ELIGIBLE_PAPERS = {"train": 192, "dev": 64}
NORMALIZATION = (
    "Unicode_NFKC_then_casefold_then_collapse_all_whitespace_to_single_"
    "ASCII_space_then_strip"
)

EXPECTED_PAPER_FIELDS = frozenset(
    {"title", "abstract", "full_text", "qas", "figures_and_tables"}
)
EXPECTED_SECTION_FIELDS = frozenset({"section_name", "paragraphs"})
EXPECTED_FLOAT_FIELDS = frozenset({"caption", "file"})
EXPECTED_QA_FIELDS = frozenset(
    {
        "question",
        "question_id",
        "nlp_background",
        "topic_background",
        "paper_read",
        "search_query",
        "answers",
        "highlighted_evidence",
    }
)
EXPECTED_ANNOTATION_FIELDS = frozenset(
    {"answer", "annotation_id", "worker_id", "highlighted_evidence"}
)
EXPECTED_ANSWER_FIELDS = frozenset(
    {
        "unanswerable",
        "extractive_spans",
        "yes_no",
        "free_form_answer",
        "evidence",
        "highlighted_evidence",
    }
)


class QasperQualificationError(RuntimeError):
    """Raised when the local source does not satisfy the frozen audit shape."""


@dataclass(frozen=True)
class ContentIndex:
    """Private in-memory exact-text node and occurrence index for one paper."""

    exact_text_occurrences: Mapping[str, int]
    exact_caption_occurrences: Mapping[str, int]
    body_paragraph_occurrences: int
    nonempty_body_paragraph_occurrences: int
    caption_occurrences: int
    nonempty_caption_occurrences: int
    section_membership_edge_occurrences: int
    paragraph_adjacency_edge_occurrences: int
    full_text_structure_content_sha256: str

    @property
    def unique_node_count(self) -> int:
        return len(self.exact_text_occurrences)

    @property
    def nonempty_occurrence_count(self) -> int:
        return sum(self.exact_text_occurrences.values())


@dataclass(frozen=True)
class QuestionRecord:
    split: str
    paper_id: str
    source_paper_ordinal: int
    normalized_title_sha256: str
    full_text_structure_content_sha256: str
    paper_label_free_commitment_sha256: str
    public_example_paper_denylisted: bool
    train_source_insertion_excluded: bool
    question_id: str
    normalized_question: str
    unique_node_count: int
    has_text_only_exact_reference_ge1: bool
    has_text_only_exact_reference_ge2: bool


@dataclass(frozen=True)
class PaperRecord:
    split: str
    source_paper_ordinal: int
    paper_id: str
    normalized_title_sha256: str
    full_text_structure_content_sha256: str
    paper_label_free_commitment_sha256: str
    public_example_paper_denylisted: bool
    train_source_insertion_excluded: bool

    @property
    def private_key(self) -> tuple[str, int]:
        return self.split, self.source_paper_ordinal


def normalize_text(value: str) -> str:
    """Return the sole frozen normalization used for collision/exposure audit."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json_no_duplicate_keys(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QasperQualificationError("dataset member is not strict UTF-8") from exc

    def hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QasperQualificationError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=hook)
    except QasperQualificationError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise QasperQualificationError("dataset member is not valid JSON") from exc


def _read_archive(archive: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_hash, archive_size = _sha256_path(archive)
    selected: dict[str, tuple[tarfile.TarInfo, bytes]] = {}
    regular_count = 0
    directory_count = 0
    other_count = 0
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle.getmembers():
                if member.isdir():
                    directory_count += 1
                    continue
                if not member.isfile():
                    other_count += 1
                    continue
                regular_count += 1
                for split, basename in EXPECTED_MEMBERS.items():
                    if Path(member.name).name != basename:
                        continue
                    if split in selected:
                        raise QasperQualificationError(
                            "archive contains duplicate official split member"
                        )
                    handle = bundle.extractfile(member)
                    if handle is None:
                        raise QasperQualificationError(
                            "official split member cannot be opened"
                        )
                    raw = handle.read()
                    if len(raw) != member.size:
                        raise QasperQualificationError(
                            "official split member size mismatch"
                        )
                    selected[split] = (member, raw)
    except (tarfile.TarError, OSError) as exc:
        raise QasperQualificationError("invalid QASPER train/dev tgz") from exc

    if set(selected) != set(EXPECTED_MEMBERS):
        raise QasperQualificationError("archive lacks an official QASPER v0.3 split")

    datasets: dict[str, Any] = {}
    member_receipts: dict[str, Any] = {}
    for split in ("train", "dev"):
        member, raw = selected[split]
        decoded = _json_no_duplicate_keys(raw)
        if not isinstance(decoded, Mapping):
            raise QasperQualificationError("split root must be a JSON object")
        datasets[split] = decoded
        member_receipts[split] = {
            "expected_basename": EXPECTED_MEMBERS[split],
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "file_size": len(raw),
            "tar_declared_size": member.size,
        }

    archive_receipt = {
        "file_sha256": archive_hash,
        "file_size": archive_size,
        "gzip_and_tar_integrity_passed": True,
        "regular_member_count": regular_count,
        "directory_member_count": directory_count,
        "nonregular_nondirectory_member_count": other_count,
        "official_members": member_receipts,
    }
    return datasets, archive_receipt


def _presence_template(expected: Iterable[str]) -> dict[str, Any]:
    return {
        "record_count": 0,
        "exact_expected_keyset_count": 0,
        "expected_field_presence_counts": {key: 0 for key in sorted(expected)},
        "unexpected_field_occurrence_count": 0,
    }


def _observe_schema(
    accumulator: dict[str, Any], record: Mapping[str, Any], expected: frozenset[str]
) -> None:
    accumulator["record_count"] += 1
    keys = set(record)
    if keys == set(expected):
        accumulator["exact_expected_keyset_count"] += 1
    for key in expected:
        if key in record:
            accumulator["expected_field_presence_counts"][key] += 1
    accumulator["unexpected_field_occurrence_count"] += len(keys - set(expected))


def _as_records(value: Any, *, expected: frozenset[str], field: str) -> list[Mapping[str, Any]]:
    """Accept raw list-of-records and columnar dict-of-lists QASPER encodings."""

    if isinstance(value, list):
        if not all(isinstance(row, Mapping) for row in value):
            raise QasperQualificationError(f"{field} must contain JSON objects")
        return list(value)
    if isinstance(value, Mapping):
        lengths = {
            len(column)
            for column in value.values()
            if isinstance(column, list)
        }
        if not lengths:
            if not value:
                return []
            raise QasperQualificationError(f"{field} columnar encoding is invalid")
        if len(lengths) != 1 or any(not isinstance(column, list) for column in value.values()):
            raise QasperQualificationError(f"{field} columns have unequal lengths")
        length = next(iter(lengths))
        return [
            {key: column[index] for key, column in value.items()}
            for index in range(length)
        ]
    raise QasperQualificationError(f"{field} must be a list or columnar object")


def _build_content_index(
    paper: Mapping[str, Any],
    schema: dict[str, Any],
) -> ContentIndex:
    exact_occurrences: Counter[str] = Counter()
    exact_caption_occurrences: Counter[str] = Counter()
    body_occurrences = 0
    nonempty_body_occurrences = 0
    caption_occurrences = 0
    nonempty_caption_occurrences = 0
    section_edges = 0
    adjacency_edges = 0
    canonical_sections: list[dict[str, Any]] = []

    sections = paper.get("full_text", [])
    if not isinstance(sections, list):
        raise QasperQualificationError("full_text must be a list")
    for section in sections:
        if not isinstance(section, Mapping):
            raise QasperQualificationError("full_text section must be an object")
        _observe_schema(schema["full_text_section"], section, EXPECTED_SECTION_FIELDS)
        section_name = section.get("section_name", "")
        if not isinstance(section_name, str):
            raise QasperQualificationError("full_text section name must be text")
        paragraphs = section.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            raise QasperQualificationError("section paragraphs must be a list")
        previous_was_nonempty = False
        canonical_paragraphs: list[str] = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, str):
                raise QasperQualificationError("body paragraph must be text")
            body_occurrences += 1
            canonical_paragraphs.append(paragraph)
            if paragraph:
                nonempty_body_occurrences += 1
                exact_occurrences[paragraph] += 1
                section_edges += 1
                if previous_was_nonempty:
                    adjacency_edges += 1
                previous_was_nonempty = True
            else:
                previous_was_nonempty = False
        canonical_sections.append(
            {"section_name": section_name, "paragraphs": canonical_paragraphs}
        )

    floats = paper.get("figures_and_tables", [])
    if not isinstance(floats, list):
        raise QasperQualificationError("figures_and_tables must be a list")
    for float_record in floats:
        if not isinstance(float_record, Mapping):
            raise QasperQualificationError("figure/table record must be an object")
        _observe_schema(schema["figure_or_table"], float_record, EXPECTED_FLOAT_FIELDS)
        caption = float_record.get("caption", "")
        if not isinstance(caption, str):
            raise QasperQualificationError("figure/table caption must be text")
        caption_occurrences += 1
        if caption:
            nonempty_caption_occurrences += 1
            exact_caption_occurrences[caption] += 1

    return ContentIndex(
        exact_text_occurrences=dict(exact_occurrences),
        exact_caption_occurrences=dict(exact_caption_occurrences),
        body_paragraph_occurrences=body_occurrences,
        nonempty_body_paragraph_occurrences=nonempty_body_occurrences,
        caption_occurrences=caption_occurrences,
        nonempty_caption_occurrences=nonempty_caption_occurrences,
        section_membership_edge_occurrences=section_edges,
        paragraph_adjacency_edge_occurrences=adjacency_edges,
        full_text_structure_content_sha256=_semantic_hash(
            {"full_text": canonical_sections}
        ),
    )


def _new_scoreability() -> dict[str, int]:
    return {
        "question_count": 0,
        "question_with_no_annotation_count": 0,
        "question_unanimously_answerable_count": 0,
        "question_unanimously_unanswerable_count": 0,
        "question_mixed_answerability_count": 0,
        "question_with_invalid_answerability_count": 0,
        "annotation_count": 0,
        "answerable_annotation_count": 0,
        "unanswerable_annotation_count": 0,
        "invalid_answerability_annotation_count": 0,
        "raw_evidence_group_count": 0,
        "raw_empty_evidence_group_count": 0,
        "raw_evidence_string_count": 0,
        "float_selected_evidence_string_count": 0,
        "group_with_float_selected_evidence_count": 0,
        "post_float_nonfloat_evidence_string_count": 0,
        "post_float_empty_gold_group_count": 0,
        "scoreable_nonempty_gold_group_count": 0,
        "unscoreable_nonempty_gold_group_count": 0,
        "exact_deduplicated_node_match_count": 0,
        "deduplicated_node_ambiguous_match_count": 0,
        "exact_missing_node_match_count": 0,
        "matched_node_single_occurrence_count": 0,
        "matched_node_multiple_occurrence_count": 0,
        "text_only_all_exact_reference_ge1_distinct_body_node_count": 0,
        "text_only_all_exact_reference_ge2_distinct_body_node_count": 0,
        "question_with_text_only_all_exact_reference_ge1_count": 0,
        "question_with_text_only_all_exact_reference_ge2_count": 0,
    }


def _audit_answers(
    qa: Mapping[str, Any],
    content: ContentIndex,
    scoreability: dict[str, int],
    schema: dict[str, Any],
) -> tuple[bool, bool]:
    scoreability["question_count"] += 1
    answers = qa.get("answers", [])
    if not isinstance(answers, list):
        raise QasperQualificationError("answers must be a list")
    if not answers:
        scoreability["question_with_no_annotation_count"] += 1
        return False, False

    flags: list[bool | None] = []
    question_reference_ge1 = False
    question_reference_ge2 = False
    for annotation in answers:
        if not isinstance(annotation, Mapping):
            raise QasperQualificationError("answer annotation must be an object")
        _observe_schema(schema["answer_annotation"], annotation, EXPECTED_ANNOTATION_FIELDS)
        if "highlighted_evidence" in annotation:
            schema["highlighted_evidence_presence"][
                "answer_annotation_field_presence_count"
            ] += 1
        payload = annotation.get("answer")
        if not isinstance(payload, Mapping):
            raise QasperQualificationError("answer annotation lacks answer object")
        _observe_schema(schema["answer_payload"], payload, EXPECTED_ANSWER_FIELDS)
        if "highlighted_evidence" in payload:
            schema["highlighted_evidence_presence"][
                "answer_payload_field_presence_count"
            ] += 1
        scoreability["annotation_count"] += 1

        unanswerable = payload.get("unanswerable")
        if type(unanswerable) is bool:
            flags.append(unanswerable)
            scoreability[
                "unanswerable_annotation_count"
                if unanswerable
                else "answerable_annotation_count"
            ] += 1
        else:
            flags.append(None)
            scoreability["invalid_answerability_annotation_count"] += 1

        evidence = payload.get("evidence", [])
        if not isinstance(evidence, list) or not all(
            isinstance(item, str) for item in evidence
        ):
            raise QasperQualificationError("answer evidence must be a list of text")
        scoreability["raw_evidence_group_count"] += 1
        scoreability["raw_evidence_string_count"] += len(evidence)
        if not evidence:
            scoreability["raw_empty_evidence_group_count"] += 1

        nonfloat: list[str] = []
        float_count = 0
        for evidence_string in evidence:
            if evidence_string.startswith(FLOAT_EVIDENCE_PREFIX):
                float_count += 1
            else:
                nonfloat.append(evidence_string)
        scoreability["float_selected_evidence_string_count"] += float_count
        if float_count:
            scoreability["group_with_float_selected_evidence_count"] += 1
        scoreability["post_float_nonfloat_evidence_string_count"] += len(nonfloat)

        if not nonfloat:
            scoreability["post_float_empty_gold_group_count"] += 1
            continue

        missing = False
        matched_distinct_nodes: set[str] = set()
        for evidence_string in nonfloat:
            occurrences = content.exact_text_occurrences.get(evidence_string, 0)
            if occurrences == 0:
                missing = True
                scoreability["exact_missing_node_match_count"] += 1
                continue
            # Exact-text deduplication makes the content-node target unique even
            # if the same node has more than one positional occurrence.
            scoreability["exact_deduplicated_node_match_count"] += 1
            matched_distinct_nodes.add(evidence_string)
            if occurrences == 1:
                scoreability["matched_node_single_occurrence_count"] += 1
            else:
                scoreability["matched_node_multiple_occurrence_count"] += 1
        scoreability[
            "unscoreable_nonempty_gold_group_count"
            if missing
            else "scoreable_nonempty_gold_group_count"
        ] += 1
        if not float_count and not missing:
            if len(matched_distinct_nodes) >= 1:
                scoreability[
                    "text_only_all_exact_reference_ge1_distinct_body_node_count"
                ] += 1
                question_reference_ge1 = True
            if len(matched_distinct_nodes) >= 2:
                scoreability[
                    "text_only_all_exact_reference_ge2_distinct_body_node_count"
                ] += 1
                question_reference_ge2 = True

    if any(flag is None for flag in flags):
        scoreability["question_with_invalid_answerability_count"] += 1
    elif all(flag is False for flag in flags):
        scoreability["question_unanimously_answerable_count"] += 1
    elif all(flag is True for flag in flags):
        scoreability["question_unanimously_unanswerable_count"] += 1
    else:
        scoreability["question_mixed_answerability_count"] += 1
    if question_reference_ge1:
        scoreability["question_with_text_only_all_exact_reference_ge1_count"] += 1
    if question_reference_ge2:
        scoreability["question_with_text_only_all_exact_reference_ge2_count"] += 1
    return question_reference_ge1, question_reference_ge2


def _duplicate_stats(values: Sequence[str]) -> dict[str, int]:
    counter = Counter(value for value in values if value)
    duplicate = {key: count for key, count in counter.items() if count > 1}
    return {
        "nonempty_value_count": sum(counter.values()),
        "unique_nonempty_value_count": len(counter),
        "duplicate_key_count": len(duplicate),
        "rows_in_duplicate_classes": sum(duplicate.values()),
    }


def _cross_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len({value for value in left if value} & {value for value in right if value})


def _merge_int_dicts(parts: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for part in parts:
        result.update(part)
    return dict(sorted(result.items()))


def _paper_duplicate_summary(
    papers: Sequence[PaperRecord], attribute: str
) -> tuple[dict[str, int], list[list[PaperRecord]]]:
    groups: dict[str, list[PaperRecord]] = {}
    for paper in papers:
        value = getattr(paper, attribute)
        if value:
            groups.setdefault(value, []).append(paper)
    duplicates = [members for members in groups.values() if len(members) > 1]
    train_classes = [
        members
        for members in duplicates
        if sum(member.split == "train" for member in members) > 1
    ]
    dev_classes = [
        members
        for members in duplicates
        if sum(member.split == "dev" for member in members) > 1
    ]
    cross_classes = [
        members
        for members in duplicates
        if {member.split for member in members} == {"train", "dev"}
    ]

    def paper_count(classes: Sequence[Sequence[PaperRecord]]) -> int:
        return len(
            {
                member.private_key
                for members in classes
                for member in members
            }
        )

    return (
        {
            "global_duplicate_class_count": len(duplicates),
            "global_paper_count_in_duplicate_classes": paper_count(duplicates),
            "within_train_duplicate_class_count": len(train_classes),
            "within_train_paper_count_in_duplicate_classes": len(
                {
                    member.private_key
                    for members in train_classes
                    for member in members
                    if member.split == "train"
                }
            ),
            "within_dev_duplicate_class_count": len(dev_classes),
            "within_dev_paper_count_in_duplicate_classes": len(
                {
                    member.private_key
                    for members in dev_classes
                    for member in members
                    if member.split == "dev"
                }
            ),
            "cross_split_duplicate_class_count": len(cross_classes),
            "cross_split_paper_count_in_duplicate_classes": paper_count(
                cross_classes
            ),
        },
        duplicates,
    )


def _paper_collision_clusters(
    papers: Sequence[PaperRecord],
) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    parent = {paper.private_key: paper.private_key for paper in papers}

    def find(key: tuple[str, int]) -> tuple[str, int]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    summaries: dict[str, dict[str, int]] = {}
    for public_name, private_attribute in (
        ("exact_paper_id", "paper_id"),
        ("normalized_title", "normalized_title_sha256"),
        (
            "canonical_label_free_full_text_structure_content",
            "full_text_structure_content_sha256",
        ),
    ):
        summary, groups = _paper_duplicate_summary(papers, private_attribute)
        summaries[public_name] = summary
        for members in groups:
            first = members[0].private_key
            for member in members[1:]:
                union(first, member.private_key)

    components: dict[tuple[str, int], list[PaperRecord]] = {}
    paper_by_key = {paper.private_key: paper for paper in papers}
    for key in parent:
        components.setdefault(find(key), []).append(paper_by_key[key])
    duplicate_components = [
        members for members in components.values() if len(members) > 1
    ]
    collided = {
        member.private_key
        for members in duplicate_components
        for member in members
    }
    cross_components = [
        members
        for members in duplicate_components
        if {member.split for member in members} == {"train", "dev"}
    ]
    within_only = {
        split: [
            members
            for members in duplicate_components
            if {member.split for member in members} == {split}
        ]
        for split in ("train", "dev")
    }
    cluster_summary = {
        "key_definitions": {
            "exact_paper_id": "exact_UTF8_map_key",
            "normalized_title": NORMALIZATION,
            "canonical_label_free_full_text_structure_content": (
                "sha256_over_canonical_JSON_ordered_sections_with_exact_section_"
                "names_and_exact_ordered_body_paragraph_text"
            ),
        },
        "duplicate_key_classes": summaries,
        "transitive_union_clusters": {
            "duplicate_cluster_count": len(duplicate_components),
            "paper_count_in_duplicate_clusters": len(collided),
            "cross_split_duplicate_cluster_count": len(cross_components),
            "cross_split_paper_count_in_duplicate_clusters": len(
                {
                    member.private_key
                    for members in cross_components
                    for member in members
                }
            ),
            "within_train_only_duplicate_cluster_count": len(
                within_only["train"]
            ),
            "within_dev_only_duplicate_cluster_count": len(within_only["dev"]),
        },
        "private_paper_commitments_emitted": 0,
    }
    return collided, cluster_summary


def build_qualification(
    archive_path: str | Path,
    reference_path: str | Path,
    *,
    enforce_formal_bindings: bool = False,
) -> dict[str, Any]:
    """Parse local inputs and return an aggregate-only qualification receipt."""

    archive = Path(archive_path).resolve(strict=True)
    reference = Path(reference_path).resolve(strict=True)
    if not archive.is_file() or not reference.is_file():
        raise QasperQualificationError("qualification inputs must be regular files")

    archive_hash_before_open, archive_size_before_open = _sha256_path(archive)
    reference_hash, reference_size = _sha256_path(reference)
    if enforce_formal_bindings and (
        archive_hash_before_open != FORMAL_ARCHIVE_SHA256
        or archive_size_before_open != FORMAL_ARCHIVE_SIZE
    ):
        raise QasperQualificationError("formal archive byte binding mismatch")
    if enforce_formal_bindings and reference_hash != FORMAL_REFERENCE_SHA256:
        raise QasperQualificationError("formal disclosed-reference binding mismatch")

    datasets, archive_receipt = _read_archive(archive)
    if (
        archive_receipt["file_sha256"] != archive_hash_before_open
        or archive_receipt["file_size"] != archive_size_before_open
    ):
        raise QasperQualificationError("archive changed during qualification")
    try:
        reference_text = reference.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise QasperQualificationError("reference file must be strict UTF-8") from exc
    normalized_reference = normalize_text(reference_text)

    schema = {
        "paper": _presence_template(EXPECTED_PAPER_FIELDS),
        "full_text_section": _presence_template(EXPECTED_SECTION_FIELDS),
        "figure_or_table": _presence_template(EXPECTED_FLOAT_FIELDS),
        "qa": _presence_template(EXPECTED_QA_FIELDS),
        "answer_annotation": _presence_template(EXPECTED_ANNOTATION_FIELDS),
        "answer_payload": _presence_template(EXPECTED_ANSWER_FIELDS),
        "highlighted_evidence_presence": {
            "qa_field_presence_count": 0,
            "answer_annotation_field_presence_count": 0,
            "answer_payload_field_presence_count": 0,
            "used_as_primary_gold_count": 0,
        },
    }
    split_counts: dict[str, dict[str, int]] = {}
    node_counts: dict[str, dict[str, int]] = {}
    scoreability: dict[str, dict[str, int]] = {}
    questions: list[QuestionRecord] = []
    paper_records: list[PaperRecord] = []
    paper_ids: dict[str, list[str]] = {"train": [], "dev": []}
    paper_titles: dict[str, list[str]] = {"train": [], "dev": []}

    for split in ("train", "dev"):
        papers = datasets[split]
        split_score = _new_scoreability()
        node = Counter(
            {
                "paper_count": 0,
                "body_paragraph_occurrence_count": 0,
                "nonempty_body_paragraph_occurrence_count": 0,
                "figure_or_table_caption_occurrence_count": 0,
                "nonempty_figure_or_table_caption_occurrence_count": 0,
                "nonempty_body_content_occurrence_count": 0,
                "unique_exact_nonempty_body_content_node_count": 0,
                "exact_duplicate_body_content_occurrence_count": 0,
                "unique_exact_nonempty_caption_text_count": 0,
                "section_membership_edge_occurrence_count": 0,
                "paragraph_adjacency_edge_occurrence_count": 0,
                "paper_with_5_to_128_unique_body_content_nodes_count": 0,
                "paper_below_5_unique_body_content_nodes_count": 0,
                "paper_above_128_unique_body_content_nodes_count": 0,
            }
        )
        min_nodes: int | None = None
        max_nodes: int | None = None
        question_count = 0
        for source_paper_ordinal, (raw_paper_id, paper) in enumerate(papers.items()):
            if not isinstance(raw_paper_id, str) or not isinstance(paper, Mapping):
                raise QasperQualificationError("paper map must bind text IDs to objects")
            _observe_schema(schema["paper"], paper, EXPECTED_PAPER_FIELDS)
            paper_ids[split].append(raw_paper_id)
            title = paper.get("title", "")
            if not isinstance(title, str):
                raise QasperQualificationError("paper title must be text")
            normalized_title = normalize_text(title)
            paper_titles[split].append(normalized_title)

            content = _build_content_index(paper, schema)
            normalized_title_sha256 = hashlib.sha256(
                normalized_title.encode("utf-8")
            ).hexdigest()
            paper_commitment = _semantic_hash(
                {
                    "split": split,
                    "source_paper_ordinal": source_paper_ordinal,
                    "paper_id_sha256": hashlib.sha256(
                        raw_paper_id.encode("utf-8")
                    ).hexdigest(),
                    "normalized_title_sha256": normalized_title_sha256,
                    "full_text_structure_content_sha256": (
                        content.full_text_structure_content_sha256
                    ),
                }
            )
            public_example_denylisted = (
                raw_paper_id in PUBLIC_EXAMPLE_PAPER_ID_DENYLIST
            )
            train_source_insertion_excluded = (
                split == "train"
                and source_paper_ordinal < TRAIN_SOURCE_INSERTION_EXCLUSION_COUNT
            )
            paper_records.append(
                PaperRecord(
                    split=split,
                    source_paper_ordinal=source_paper_ordinal,
                    paper_id=raw_paper_id,
                    normalized_title_sha256=normalized_title_sha256,
                    full_text_structure_content_sha256=(
                        content.full_text_structure_content_sha256
                    ),
                    paper_label_free_commitment_sha256=paper_commitment,
                    public_example_paper_denylisted=public_example_denylisted,
                    train_source_insertion_excluded=train_source_insertion_excluded,
                )
            )
            node["paper_count"] += 1
            node["body_paragraph_occurrence_count"] += content.body_paragraph_occurrences
            node["nonempty_body_paragraph_occurrence_count"] += (
                content.nonempty_body_paragraph_occurrences
            )
            node["figure_or_table_caption_occurrence_count"] += content.caption_occurrences
            node["nonempty_figure_or_table_caption_occurrence_count"] += (
                content.nonempty_caption_occurrences
            )
            node["nonempty_body_content_occurrence_count"] += (
                content.nonempty_occurrence_count
            )
            node["unique_exact_nonempty_body_content_node_count"] += content.unique_node_count
            node["exact_duplicate_body_content_occurrence_count"] += (
                content.nonempty_occurrence_count - content.unique_node_count
            )
            node["unique_exact_nonempty_caption_text_count"] += len(
                content.exact_caption_occurrences
            )
            node["section_membership_edge_occurrence_count"] += (
                content.section_membership_edge_occurrences
            )
            node["paragraph_adjacency_edge_occurrence_count"] += (
                content.paragraph_adjacency_edge_occurrences
            )
            min_nodes = (
                content.unique_node_count
                if min_nodes is None
                else min(min_nodes, content.unique_node_count)
            )
            max_nodes = (
                content.unique_node_count
                if max_nodes is None
                else max(max_nodes, content.unique_node_count)
            )
            if MIN_CONTENT_NODES <= content.unique_node_count <= MAX_CONTENT_NODES:
                node["paper_with_5_to_128_unique_body_content_nodes_count"] += 1
            elif content.unique_node_count < MIN_CONTENT_NODES:
                node["paper_below_5_unique_body_content_nodes_count"] += 1
            else:
                node["paper_above_128_unique_body_content_nodes_count"] += 1

            qas = _as_records(paper.get("qas", []), expected=EXPECTED_QA_FIELDS, field="qas")
            for qa in qas:
                _observe_schema(schema["qa"], qa, EXPECTED_QA_FIELDS)
                if "highlighted_evidence" in qa:
                    schema["highlighted_evidence_presence"][
                        "qa_field_presence_count"
                    ] += 1
                question_count += 1
                question = qa.get("question", "")
                question_id = qa.get("question_id", "")
                normalized_question = normalize_text(question) if isinstance(question, str) else ""
                safe_question_id = question_id if isinstance(question_id, str) else ""
                reference_ge1, reference_ge2 = _audit_answers(
                    qa, content, split_score, schema
                )
                questions.append(
                    QuestionRecord(
                        split=split,
                        paper_id=raw_paper_id,
                        source_paper_ordinal=source_paper_ordinal,
                        normalized_title_sha256=normalized_title_sha256,
                        full_text_structure_content_sha256=(
                            content.full_text_structure_content_sha256
                        ),
                        paper_label_free_commitment_sha256=paper_commitment,
                        public_example_paper_denylisted=(
                            public_example_denylisted
                        ),
                        train_source_insertion_excluded=(
                            train_source_insertion_excluded
                        ),
                        question_id=safe_question_id,
                        normalized_question=normalized_question,
                        unique_node_count=content.unique_node_count,
                        has_text_only_exact_reference_ge1=reference_ge1,
                        has_text_only_exact_reference_ge2=reference_ge2,
                    )
                )

        node_counts[split] = {
            **dict(node),
            "minimum_unique_body_content_nodes_per_paper": min_nodes or 0,
            "maximum_unique_body_content_nodes_per_paper": max_nodes or 0,
        }
        split_counts[split] = {
            "paper_count": len(papers),
            "question_count": question_count,
        }
        scoreability[split] = split_score

    by_split_questions = {
        split: [row for row in questions if row.split == split]
        for split in ("train", "dev")
    }
    paper_collision_keys, paper_collision_summary = _paper_collision_clusters(
        paper_records
    )
    collision = {
        "normalization": NORMALIZATION,
        "cross_split": {
            "exact_paper_id_overlap_count": _cross_overlap(
                paper_ids["train"], paper_ids["dev"]
            ),
            "normalized_paper_title_overlap_count": _cross_overlap(
                paper_titles["train"], paper_titles["dev"]
            ),
            "exact_question_id_overlap_count": _cross_overlap(
                (row.question_id for row in by_split_questions["train"]),
                (row.question_id for row in by_split_questions["dev"]),
            ),
            "normalized_question_overlap_count": _cross_overlap(
                (row.normalized_question for row in by_split_questions["train"]),
                (row.normalized_question for row in by_split_questions["dev"]),
            ),
        },
        "within_split": {
            split: {
                "paper_id": _duplicate_stats(paper_ids[split]),
                "normalized_paper_title": _duplicate_stats(paper_titles[split]),
                "question_id": _duplicate_stats(
                    [row.question_id for row in by_split_questions[split]]
                ),
                "normalized_question": _duplicate_stats(
                    [row.normalized_question for row in by_split_questions[split]]
                ),
            }
            for split in ("train", "dev")
        },
        "paper_duplicate_class_exclusion": paper_collision_summary,
    }

    question_id_counter = Counter(row.question_id for row in questions if row.question_id)
    question_text_counter = Counter(
        row.normalized_question for row in questions if row.normalized_question
    )

    source_custody_exclusions: dict[str, Any] = {
        "public_example_paper_id_denylist_declared_unique_count": len(
            PUBLIC_EXAMPLE_PAPER_ID_DENYLIST
        ),
        "train_source_insertion_prefix_declared_paper_count": (
            TRAIN_SOURCE_INSERTION_EXCLUSION_COUNT
        ),
        "split_aggregates": {},
        "paper_ids_emitted": 0,
    }
    for split in ("train", "dev"):
        split_papers = [paper for paper in paper_records if paper.split == split]
        source_custody_exclusions["split_aggregates"][split] = {
            "source_paper_count": len(split_papers),
            "public_example_denylist_matched_paper_count": sum(
                paper.public_example_paper_denylisted for paper in split_papers
            ),
            "train_source_insertion_prefix_excluded_paper_count": sum(
                paper.train_source_insertion_excluded for paper in split_papers
            ),
            "overlap_between_two_custody_exclusions_paper_count": sum(
                paper.public_example_paper_denylisted
                and paper.train_source_insertion_excluded
                for paper in split_papers
            ),
            "union_custody_excluded_paper_count": sum(
                paper.public_example_paper_denylisted
                or paper.train_source_insertion_excluded
                for paper in split_papers
            ),
        }

    eligibility: dict[str, dict[str, int]] = {}
    for split in ("train", "dev"):
        counts = Counter(
            {
                "all_question_count": 0,
                "nonempty_question_and_question_id_count": 0,
                "label_free_5_to_128_node_candidate_count": 0,
                "paper_or_question_collision_exclusion_count": 0,
                "collision_free_5_to_128_node_candidate_count": 0,
                "public_example_denylist_exclusion_question_count": 0,
                "train_source_insertion_prefix_exclusion_question_count": 0,
                "custody_exclusion_union_question_count": 0,
                "custody_clean_collision_free_candidate_count": 0,
                "exact_normalized_exposed_question_substring_exclusion_count": 0,
                "exposure_clean_label_free_candidate_count": 0,
                "structural_label_ge1_question_count": 0,
                "structural_label_ge2_question_count": 0,
            }
        )
        ge1_papers: set[str] = set()
        ge2_papers: set[str] = set()
        for row in by_split_questions[split]:
            counts["all_question_count"] += 1
            if not row.question_id or not row.normalized_question:
                continue
            counts["nonempty_question_and_question_id_count"] += 1
            if not MIN_CONTENT_NODES <= row.unique_node_count <= MAX_CONTENT_NODES:
                continue
            counts["label_free_5_to_128_node_candidate_count"] += 1
            collided = (
                (row.split, row.source_paper_ordinal) in paper_collision_keys
                or question_id_counter[row.question_id] != 1
                or question_text_counter[row.normalized_question] != 1
            )
            if collided:
                counts["paper_or_question_collision_exclusion_count"] += 1
                continue
            counts["collision_free_5_to_128_node_candidate_count"] += 1
            if row.public_example_paper_denylisted:
                counts["public_example_denylist_exclusion_question_count"] += 1
            if row.train_source_insertion_excluded:
                counts[
                    "train_source_insertion_prefix_exclusion_question_count"
                ] += 1
            if (
                row.public_example_paper_denylisted
                or row.train_source_insertion_excluded
            ):
                counts["custody_exclusion_union_question_count"] += 1
                continue
            counts["custody_clean_collision_free_candidate_count"] += 1
            exposed = bool(
                normalized_reference
                and row.normalized_question in normalized_reference
            )
            if exposed:
                counts[
                    "exact_normalized_exposed_question_substring_exclusion_count"
                ] += 1
                continue
            counts["exposure_clean_label_free_candidate_count"] += 1
            if row.has_text_only_exact_reference_ge1:
                counts["structural_label_ge1_question_count"] += 1
                ge1_papers.add(row.paper_id)
            if row.has_text_only_exact_reference_ge2:
                counts["structural_label_ge2_question_count"] += 1
                ge2_papers.add(row.paper_id)
        counts["structural_label_ge1_paper_count_one_question_cap"] = len(ge1_papers)
        counts["structural_label_ge2_paper_count_one_question_cap"] = len(ge2_papers)
        counts[
            "formal_eligible_question_count_before_one_question_per_paper_cap"
        ] = counts["structural_label_ge2_question_count"]
        counts["formal_eligible_paper_count_one_question_cap"] = len(ge2_papers)
        eligibility[split] = dict(counts)
    eligibility["combined"] = _merge_int_dicts(eligibility.values())

    capacity_by_split = {
        split: {
            "minimum_distinct_eligible_papers": (
                FORMAL_MINIMUM_DISTINCT_ELIGIBLE_PAPERS[split]
            ),
            "observed_distinct_eligible_papers": eligibility[split][
                "formal_eligible_paper_count_one_question_cap"
            ],
            "minimum_met": (
                eligibility[split]["formal_eligible_paper_count_one_question_cap"]
                >= FORMAL_MINIMUM_DISTINCT_ELIGIBLE_PAPERS[split]
            ),
        }
        for split in ("train", "dev")
    }
    formal_capacity_met = all(
        aggregate["minimum_met"] for aggregate in capacity_by_split.values()
    )
    if enforce_formal_bindings:
        qualification_status = (
            "source_qualified_for_frozen_block_capacity"
            if formal_capacity_met
            else "terminal_source_infeasible_for_frozen_block_capacity"
        )
    else:
        qualification_status = "synthetic_or_nonformal_aggregate_diagnostic"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "qualification_class": "outcome_blind_aggregate_only_source_qualification",
        "source_release": "QASPER_v0.3_official_train_dev",
        "selection_status": "not_performed",
        "selection_secret_opened_or_generated": False,
        "performance_or_retrieval_scoring_performed": False,
        "formal_binding_mode": enforce_formal_bindings,
        "formal_public_bindings": {
            "archive_sha256": FORMAL_ARCHIVE_SHA256,
            "archive_size": FORMAL_ARCHIVE_SIZE,
            "disclosed_reference_sha256": FORMAL_REFERENCE_SHA256,
            "custody_sha256": FORMAL_CUSTODY_SHA256,
            "custody_commit": FORMAL_CUSTODY_COMMIT,
            "all_input_byte_bindings_verified_before_row_parse": (
                enforce_formal_bindings
            ),
        },
        "archive": archive_receipt,
        "disclosed_reference": {
            "file_sha256": reference_hash,
            "file_size": reference_size,
            "normalization": NORMALIZATION,
            "only_used_for_exact_normalized_question_substring_exclusion": True,
        },
        "split_counts": split_counts,
        "field_schema": schema,
        "label_free_content_graph": {
            "node_definition": "unique_exact_nonempty_UTF8_body_paragraph_text",
            "title_is_a_content_node": False,
            "abstract_is_a_content_node": False,
            "body_paragraph_is_a_content_node": True,
            "figure_or_table_caption_is_a_content_node": False,
            "duplicate_text_occurrences_share_one_node": True,
            "all_positional_occurrences_retained_for_edges": True,
            "split_aggregates": node_counts,
        },
        "gold_evidence_scoreability": {
            "mapping_rule": "exact_Unicode_text_to_deduplicated_content_node",
            "float_filter_heuristic": "case_sensitive_startswith_FLOAT_SELECTED",
            "float_selected_removed_before_gold_mapping": True,
            "float_selected_caption_mapping_required": False,
            "deduplicated_node_ambiguity_is_zero_by_construction": True,
            "formal_reference_rule": (
                "one_annotation_reference_is_text_only_with_no_FLOAT_SELECTED_"
                "and_has_at_least_two_distinct_exact_mappable_body_nodes"
            ),
            "multi_reference_scoring_rule": "maximum_over_references_not_union",
            "highlighted_evidence_used_as_primary_gold": False,
            "split_aggregates": scoreability,
            "combined": _merge_int_dicts(scoreability.values()),
        },
        "collision_audit": collision,
        "custody_source_exclusions": source_custody_exclusions,
        "selection_eligibility": {
            "label_free_preconditions": True,
            "formal_eligibility_uses_structural_gold_only": True,
            "answer_type_used": False,
            "evidence_count_used_beyond_distinct_node_minimum": False,
            "structural_gold_hidden_from_label_free_action_view": True,
            "minimum_unique_content_nodes_inclusive": MIN_CONTENT_NODES,
            "maximum_unique_content_nodes_inclusive": MAX_CONTENT_NODES,
            "collision_exclusion": (
                "any_global_duplicate_class_or_transitive_cluster_over_exact_"
                "paper_id_normalized_title_or_canonical_label_free_full_text_"
                "structure_content_plus_any_global_exact_question_id_or_"
                "normalized_question_duplicate_class"
            ),
            "custody_exclusions_applied_before_question_reference_exposure": True,
            "exposed_question_rule": (
                "normalized_question_is_exact_substring_of_normalized_disclosed_"
                "reference"
            ),
            "formal_minimum_distinct_exact_body_nodes_in_one_text_only_reference": 2,
            "paper_cap": "at_most_one_question_selected_per_paper",
            "cross_block_constraint": "paper_disjoint",
            "split_aggregates": eligibility,
        },
        "formal_capacity_decision": {
            "distinct_eligible_paper_minimums": capacity_by_split,
            "all_minimums_met": formal_capacity_met,
            "capacity_shortfall_action": (
                "terminal_source_infeasible_without_smaller_blocks_relaxed_"
                "population_backup_or_same_source_redesign"
            ),
        },
        "qualification_operations": {
            "source_papers_programmatically_parsed": sum(
                part["paper_count"] for part in split_counts.values()
            ),
            "source_questions_programmatically_parsed": sum(
                part["question_count"] for part in split_counts.values()
            ),
            "answer_annotations_programmatically_parsed": sum(
                part["annotation_count"] for part in scoreability.values()
            ),
            "selection_or_sampling_operations": 0,
            "item_ids_or_text_emitted": 0,
            "answers_or_evidence_emitted": 0,
            "private_paper_commitments_emitted": 0,
        },
        "status": qualification_status,
    }
    report["qualification_sha256"] = _semantic_hash(report)
    return report


def _validate_child_receipt(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise QasperQualificationError("clean worker returned invalid receipt")
    declared = payload.get("qualification_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise QasperQualificationError("clean worker receipt lacks semantic hash")
    body = dict(payload)
    body.pop("qualification_sha256", None)
    if _semantic_hash(body) != declared:
        raise QasperQualificationError("clean worker receipt hash mismatch")
    if payload.get("formal_binding_mode") is True:
        bindings = payload.get("formal_public_bindings")
        archive = payload.get("archive")
        reference = payload.get("disclosed_reference")
        if (
            not isinstance(bindings, Mapping)
            or not isinstance(archive, Mapping)
            or not isinstance(reference, Mapping)
            or bindings.get("archive_sha256") != FORMAL_ARCHIVE_SHA256
            or bindings.get("archive_size") != FORMAL_ARCHIVE_SIZE
            or bindings.get("disclosed_reference_sha256")
            != FORMAL_REFERENCE_SHA256
            or bindings.get("custody_sha256") != FORMAL_CUSTODY_SHA256
            or bindings.get("custody_commit") != FORMAL_CUSTODY_COMMIT
            or bindings.get("all_input_byte_bindings_verified_before_row_parse")
            is not True
            or archive.get("file_sha256") != FORMAL_ARCHIVE_SHA256
            or archive.get("file_size") != FORMAL_ARCHIVE_SIZE
            or reference.get("file_sha256") != FORMAL_REFERENCE_SHA256
        ):
            raise QasperQualificationError("formal clean-worker binding drift")
    operations = payload.get("qualification_operations")
    if not isinstance(operations, Mapping) or any(
        operations.get(field) != 0
        for field in (
            "selection_or_sampling_operations",
            "item_ids_or_text_emitted",
            "answers_or_evidence_emitted",
            "private_paper_commitments_emitted",
        )
    ):
        raise QasperQualificationError("clean worker receipt violates redaction")
    return payload


def run_clean_qualification(
    archive_path: str | Path,
    reference_path: str | Path,
    *,
    enforce_formal_bindings: bool = False,
) -> dict[str, Any]:
    """Run qualification under ``python -I`` with a minimal environment."""

    archive = Path(archive_path).resolve(strict=True)
    reference = Path(reference_path).resolve(strict=True)
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--_aggregate-worker",
        "--archive",
        str(archive),
        "--reference",
        str(reference),
    ]
    if enforce_formal_bindings:
        command.append("--formal")
    environment = {
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C.UTF-8",
    }
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        close_fds=True,
        env=environment,
        cwd=str(Path(__file__).resolve().parent),
    )
    if completed.returncode != 0:
        # Do not forward worker stderr: malformed source data could otherwise
        # turn an exception representation into an unintended content channel.
        raise QasperQualificationError("clean aggregate worker failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QasperQualificationError("clean worker emitted non-JSON output") from exc
    return _validate_child_receipt(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--_aggregate-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments._aggregate_worker:
        receipt = build_qualification(
            arguments.archive,
            arguments.reference,
            enforce_formal_bindings=arguments.formal,
        )
    else:
        receipt = run_clean_qualification(
            arguments.archive,
            arguments.reference,
            enforce_formal_bindings=arguments.formal,
        )
    sys.stdout.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
