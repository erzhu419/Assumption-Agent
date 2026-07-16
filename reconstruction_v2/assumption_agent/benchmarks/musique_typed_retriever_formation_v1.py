"""Form a finite typed MuSiQue retriever from the exact TRAIN split only.

The public acquisition receipt is the sole authority for the TRAIN file hash,
row count, item-commitment set, and enclosing private-pack commitment.  This
module accepts the exact ``train.jsonl`` file rather than a pack directory and
never discovers sibling paths.  Formation is deterministic and fully local:
no model, network, or online evaluator is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from ..models import stable_hash


FORMATION_VERSION = "musique_train_only_typed_retriever_formation_v1"
DSL_VERSION = "musique_finite_typed_retrieval_dsl_v1"
OPERATOR_VERSION = "musique_typed_retriever_operator_v1"
ACQUISITION_SCHEMA = "musique-official-core-comparison-v1-acquisition"
PRIVATE_ROW_SCHEMA = "musique-official-core-comparison-v1-private-pack"
CLAIM_SCOPE = "multi_alias_eligible_subset_of_official_train_not_full_musique"
TOKENIZER_VERSION = "unicode_nfkc_casefold_alnum_v1"
FOLD_POLICY = "exact_train_position_modulo_4_v1"
SELECTION_POLICY = "negative_support_recall_at_5_then_invalid_then_program_length_then_hash_v1"
TOP_K = 5
EXPECTED_TRAIN_COUNT = 12
MAX_CANDIDATES = 128
MAX_SEMANTIC_NODES = 6
IMPLEMENTATION_BINDING_SCHEMA = "musique_typed_retriever_implementation_set_v1"
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class MuSiQueTypedFormationError(RuntimeError):
    """Raised when TRAIN formation cannot satisfy its frozen input contract."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MuSiQueTypedFormationError(f"{field} must be lowercase sha256")
    return value


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_existing_symlink_components(path: str | Path, field: str) -> Path:
    """Reject a target if its file or any existing ancestor is a symlink."""

    candidate = _absolute_lexical(path)
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueTypedFormationError(
                f"{field} may not contain an existing symlink component"
            )
    return candidate


def _containing_git_repository(path: Path) -> Path | None:
    anchor = path if path.is_dir() else path.parent
    while not anchor.exists():
        parent = anchor.parent
        if parent == anchor:
            return None
        anchor = parent
    if not anchor.is_dir():
        anchor = anchor.parent
    completed = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return None
    repository = Path(completed.stdout.strip()).resolve(strict=True)
    if not repository.is_dir():
        raise MuSiQueTypedFormationError("git repository root is invalid")
    return repository


def _require_ignored_untracked_if_in_repository(
    path: Path,
    field: str,
    *,
    directory: bool = False,
) -> None:
    repository = _containing_git_repository(path)
    if repository is None:
        return
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(repository).as_posix()
    except ValueError as exc:
        raise MuSiQueTypedFormationError(
            f"{field} repository containment is ambiguous"
        ) from exc
    ignore_path = f"{relative.rstrip('/')}/" if directory else relative
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            ignore_path,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ignored.returncode != 0:
        raise MuSiQueTypedFormationError(
            f"{field} must be git-ignored when it is inside a repository"
        )
    tracked = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--", relative],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if tracked.returncode != 0:
        raise MuSiQueTypedFormationError(f"could not audit {field} tracking state")
    if tracked.stdout.strip():
        raise MuSiQueTypedFormationError(
            f"{field} must be untracked even when an ignore rule also matches"
        )


def _validated_implementation_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "files",
        "set_sha256",
    }:
        raise MuSiQueTypedFormationError("implementation binding schema mismatch")
    if value.get("schema") != IMPLEMENTATION_BINDING_SCHEMA:
        raise MuSiQueTypedFormationError("implementation binding version mismatch")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(IMPLEMENTATION_RELATIVE_FILES):
        raise MuSiQueTypedFormationError("implementation file set mismatch")
    normalized_files: list[dict[str, str]] = []
    for expected_path, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise MuSiQueTypedFormationError("implementation file binding malformed")
        if row.get("path") != expected_path:
            raise MuSiQueTypedFormationError("implementation file order or path mismatch")
        normalized_files.append(
            {
                "path": expected_path,
                "sha256": _require_sha256(
                    row.get("sha256"), f"implementation sha256 for {expected_path}"
                ),
            }
        )
    set_sha256 = _require_sha256(
        value.get("set_sha256"), "implementation set sha256"
    )
    if stable_hash(normalized_files) != set_sha256:
        raise MuSiQueTypedFormationError("implementation set hash mismatch")
    return {
        "schema": IMPLEMENTATION_BINDING_SCHEMA,
        "files": normalized_files,
        "set_sha256": set_sha256,
    }


def current_implementation_binding(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the fixed source set that defines frozen retrieval semantics."""

    root = (
        Path(__file__).resolve(strict=True).parents[2]
        if project_root is None
        else Path(project_root).resolve(strict=True)
    )
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        _assert_no_existing_symlink_components(path, "implementation file")
        if not path.is_file():
            raise MuSiQueTypedFormationError(f"implementation file missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_BINDING_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def verify_live_implementation(
    expected: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Independently compare a frozen binding with the live implementation."""

    frozen = _validated_implementation_binding(expected)
    live = current_implementation_binding(project_root)
    if live != frozen:
        raise MuSiQueTypedFormationError("live implementation drifted from frozen binding")
    return live


def unicode_casefold_tokens(value: str) -> tuple[str, ...]:
    """Tokenize Unicode text after deterministic NFKC normalization/casefold."""

    if not isinstance(value, str):
        raise TypeError("tokenizer input must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(match.group(0) for match in _TOKEN.finditer(normalized))


@dataclass(frozen=True)
class RetrievalParagraph:
    idx: int
    title: str
    text: str


@dataclass(frozen=True)
class _TrainItem:
    question: str
    corpus: tuple[RetrievalParagraph, ...]
    support_indices: tuple[int, ...]


@dataclass(frozen=True)
class TypedRetrievalProgram:
    seed_algorithm: str
    title_weight: int
    text_weight: int
    expansion_mode: str
    expansion_weight: int
    top_k: int = TOP_K
    tokenizer_version: str = TOKENIZER_VERSION
    dsl_version: str = DSL_VERSION

    @property
    def semantic_nodes(self) -> tuple[Mapping[str, Any], ...]:
        nodes: list[Mapping[str, Any]] = [
            {"op": "unicode_tokenize", "version": self.tokenizer_version},
            {
                "op": "weighted_lexical_seed",
                "algorithm": self.seed_algorithm,
                "title_weight": self.title_weight,
                "text_weight": self.text_weight,
            },
        ]
        if self.expansion_mode != "none":
            nodes.extend(
                (
                    {
                        "op": "one_hop_graph_expand",
                        "mode": self.expansion_mode,
                        "seed_document_count": 1,
                    },
                    {
                        "op": "weighted_lexical_rerank",
                        "expansion_weight": self.expansion_weight,
                    },
                )
            )
        nodes.append({"op": "stable_rank", "tie_break": "ascending_paragraph_idx"})
        nodes.append({"op": "take", "k": self.top_k})
        return tuple(nodes)

    @property
    def program_length(self) -> int:
        return len(self.semantic_nodes)

    def type_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.dsl_version != DSL_VERSION:
            issues.append("dsl_version")
        if self.tokenizer_version != TOKENIZER_VERSION:
            issues.append("tokenizer_version")
        if self.seed_algorithm not in {"bm25", "tfidf"}:
            issues.append("seed_algorithm")
        if self.title_weight not in {1, 2, 4}:
            issues.append("title_weight")
        if self.text_weight not in {1, 2}:
            issues.append("text_weight")
        if self.expansion_mode not in {
            "none",
            "token_one_hop",
            "entity_token_one_hop",
        }:
            issues.append("expansion_mode")
        if self.expansion_mode == "none" and self.expansion_weight != 0:
            issues.append("expansion_weight")
        if self.expansion_mode != "none" and self.expansion_weight not in {1, 2, 4}:
            issues.append("expansion_weight")
        if self.top_k != TOP_K:
            issues.append("top_k")
        if len(self.semantic_nodes) > MAX_SEMANTIC_NODES:
            issues.append("semantic_node_budget")
        return tuple(sorted(set(issues)))

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "semantic_nodes": [dict(node) for node in self.semantic_nodes],
            "fixed_envelope": {
                "graph_hops": 0 if self.expansion_mode == "none" else 1,
                "seed_document_count": 0 if self.expansion_mode == "none" else 1,
                "stable_tie_break": "ascending_paragraph_idx",
            },
        }

    @property
    def program_hash(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TypedRetrievalProgram":
        return cls(
            seed_algorithm=str(value["seed_algorithm"]),
            title_weight=int(value["title_weight"]),
            text_weight=int(value["text_weight"]),
            expansion_mode=str(value["expansion_mode"]),
            expansion_weight=int(value["expansion_weight"]),
            top_k=int(value["top_k"]),
            tokenizer_version=str(value["tokenizer_version"]),
            dsl_version=str(value["dsl_version"]),
        )


@dataclass(frozen=True)
class CandidateAssessment:
    program: TypedRetrievalProgram
    support_hit_count: int
    support_label_count: int
    invalid_count: int
    behavior_hash: str

    @property
    def recall_at_5(self) -> Fraction:
        if self.support_label_count == 0:
            return Fraction(0, 1)
        return Fraction(self.support_hit_count, self.support_label_count)

    @property
    def rank(self) -> tuple[Fraction, int, int, str]:
        return (
            -self.recall_at_5,
            self.invalid_count,
            self.program.program_length,
            self.program.program_hash,
        )


@dataclass(frozen=True)
class FormationResult:
    program: TypedRetrievalProgram
    receipt: Mapping[str, Any]


def enumerate_programs() -> Iterable[TypedRetrievalProgram]:
    expansion_options = (
        ("none", 0),
        *(product for product in itertools.product(
            ("token_one_hop", "entity_token_one_hop"),
            (1, 2, 4),
        )),
    )
    for seed, title_weight, text_weight, expansion in itertools.product(
        ("bm25", "tfidf"),
        (1, 2, 4),
        (1, 2),
        expansion_options,
    ):
        yield TypedRetrievalProgram(
            seed_algorithm=seed,
            title_weight=title_weight,
            text_weight=text_weight,
            expansion_mode=expansion[0],
            expansion_weight=expansion[1],
        )


def _document_tokens(
    corpus: Sequence[RetrievalParagraph],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (unicode_casefold_tokens(paragraph.title), unicode_casefold_tokens(paragraph.text))
        for paragraph in corpus
    )


def _lexical_scores(
    program: TypedRetrievalProgram,
    query: Counter[str],
    documents: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
) -> tuple[float, ...]:
    combined = tuple(title + text for title, text in documents)
    document_count = len(combined)
    document_frequency = Counter(
        token for tokens in combined for token in set(tokens)
    )
    weighted_counts = tuple(
        Counter(
            {
                token: count * program.title_weight
                for token, count in Counter(title).items()
            }
        )
        + Counter(
            {
                token: count * program.text_weight
                for token, count in Counter(text).items()
            }
        )
        for title, text in documents
    )
    if program.seed_algorithm == "tfidf":
        return tuple(
            sum(
                query_frequency
                * counts.get(token, 0)
                * (math.log((document_count + 1) / (document_frequency[token] + 1)) + 1.0) ** 2
                for token, query_frequency in query.items()
            )
            for counts in weighted_counts
        )

    lengths = tuple(sum(counts.values()) for counts in weighted_counts)
    average_length = sum(lengths) / document_count
    k1 = 1.2
    b = 0.75
    scores: list[float] = []
    for counts, length in zip(weighted_counts, lengths):
        score = 0.0
        for token, query_frequency in query.items():
            frequency = counts.get(token, 0)
            if frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * length / average_length
            )
            score += query_frequency * inverse_document_frequency * (
                frequency * (k1 + 1.0) / denominator
            )
        scores.append(score)
    return tuple(scores)


def retrieve(
    program: TypedRetrievalProgram,
    question: str,
    corpus: Sequence[RetrievalParagraph],
) -> tuple[int, ...]:
    """Execute a frozen program without access to answers or support labels."""

    issues = program.type_issues()
    if issues:
        raise ValueError(f"ill-typed retrieval program: {issues}")
    if len(corpus) < TOP_K:
        raise ValueError("corpus is smaller than frozen top-k")
    indices = tuple(paragraph.idx for paragraph in corpus)
    if (
        any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("paragraph idx namespace is invalid")
    query_tokens = unicode_casefold_tokens(question)
    if not query_tokens:
        raise ValueError("question has no lexical tokens")
    documents = _document_tokens(corpus)
    seed_scores = _lexical_scores(program, Counter(query_tokens), documents)
    seed_order = sorted(
        range(len(corpus)),
        key=lambda position: (-seed_scores[position], corpus[position].idx),
    )
    final_scores = seed_scores
    if program.expansion_mode != "none":
        anchor_position = seed_order[0]
        anchor_tokens = set(documents[anchor_position][0] + documents[anchor_position][1])
        anchor_tokens.difference_update(query_tokens)
        if program.expansion_mode == "entity_token_one_hop":
            title_vocabulary = {
                token for title_tokens, _ in documents for token in title_tokens
            }
            anchor_tokens.intersection_update(title_vocabulary)
        expansion_scores = _lexical_scores(
            program,
            Counter(sorted(anchor_tokens)),
            documents,
        )
        final_scores = tuple(
            seed + program.expansion_weight * expansion
            for seed, expansion in zip(seed_scores, expansion_scores)
        )
    ranked = sorted(
        range(len(corpus)),
        key=lambda position: (-final_scores[position], corpus[position].idx),
    )
    return tuple(corpus[position].idx for position in ranked[:TOP_K])


def _verify_acquisition_receipt(path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MuSiQueTypedFormationError("acquisition receipt must be an object")
    declared = payload.get("acquisition_sha256")
    body = dict(payload)
    body.pop("acquisition_sha256", None)
    if payload.get("schema") != ACQUISITION_SCHEMA or stable_hash(body) != declared:
        raise MuSiQueTypedFormationError("acquisition receipt self-hash mismatch")
    if payload.get("decision") != "private_pack_formed_no_model_execution_authorized":
        raise MuSiQueTypedFormationError("acquisition decision mismatch")
    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("claim_scope") != CLAIM_SCOPE:
        raise MuSiQueTypedFormationError("MuSiQue claim scope mismatch")
    counts = payload.get("counts")
    commitments = payload.get("commitments")
    if not isinstance(counts, Mapping) or not isinstance(commitments, Mapping):
        raise MuSiQueTypedFormationError("acquisition counts or commitments malformed")
    split_counts = counts.get("splits")
    split_files = commitments.get("split_files")
    if not isinstance(split_counts, Mapping) or not isinstance(split_files, list):
        raise MuSiQueTypedFormationError("acquisition split commitments malformed")
    if split_counts.get("train") != EXPECTED_TRAIN_COUNT:
        raise MuSiQueTypedFormationError("exact TRAIN count mismatch")
    if counts.get("selected_rows") != sum(
        value for value in split_counts.values() if type(value) is int
    ):
        raise MuSiQueTypedFormationError("selected split count mismatch")
    seen_splits: set[str] = set()
    train_rows: list[Mapping[str, Any]] = []
    for row in split_files:
        if not isinstance(row, Mapping) or set(row) != {
            "split",
            "count",
            "file_sha256",
            "item_commitment_set_sha256",
        }:
            raise MuSiQueTypedFormationError("split file commitment schema mismatch")
        split = row.get("split")
        if not isinstance(split, str) or split in seen_splits:
            raise MuSiQueTypedFormationError("split file commitment is duplicated")
        seen_splits.add(split)
        if split_counts.get(split) != row.get("count"):
            raise MuSiQueTypedFormationError("split file count mismatch")
        _require_sha256(row.get("file_sha256"), "split file hash")
        _require_sha256(row.get("item_commitment_set_sha256"), "item set hash")
        if split == "train":
            train_rows.append(row)
    if set(split_counts) != seen_splits or len(train_rows) != 1:
        raise MuSiQueTypedFormationError("split commitment set mismatch")
    pack_hash = _require_sha256(
        commitments.get("private_pack_sha256"), "private pack hash"
    )
    if stable_hash(split_files) != pack_hash:
        raise MuSiQueTypedFormationError("private pack commitment mismatch")
    return payload, train_rows[0]


def _parse_exact_train(raw: bytes, train_commitment: Mapping[str, Any]) -> tuple[_TrainItem, ...]:
    if _sha256_bytes(raw) != train_commitment["file_sha256"]:
        raise MuSiQueTypedFormationError("exact TRAIN file hash mismatch")
    if not raw or not raw.endswith(b"\n"):
        raise MuSiQueTypedFormationError("exact TRAIN JSONL framing mismatch")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            raise MuSiQueTypedFormationError("blank TRAIN JSONL row")
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise MuSiQueTypedFormationError("TRAIN row must be an object")
        rows.append(value)
    if len(rows) != train_commitment["count"]:
        raise MuSiQueTypedFormationError("exact TRAIN row count mismatch")
    canonical = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    if raw != canonical:
        raise MuSiQueTypedFormationError("TRAIN JSONL is not canonical acquisition output")

    expected_keys = {
        "schema",
        "split",
        "item_id",
        "question",
        "corpus",
        "answers",
        "normalized_answers",
        "support_indices",
        "source_row_sha256",
    }
    item_commitments: list[str] = []
    items: list[_TrainItem] = []
    seen_private_ids: set[str] = set()
    for row in rows:
        if set(row) != expected_keys:
            raise MuSiQueTypedFormationError("TRAIN row schema mismatch")
        if row.get("schema") != PRIVATE_ROW_SCHEMA or row.get("split") != "train":
            raise MuSiQueTypedFormationError("non-TRAIN row supplied to formation")
        private_id = row.get("item_id")
        question = row.get("question")
        if (
            not isinstance(private_id, str)
            or not private_id
            or private_id in seen_private_ids
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise MuSiQueTypedFormationError("TRAIN identity or question malformed")
        seen_private_ids.add(private_id)
        _require_sha256(row.get("source_row_sha256"), "source row hash")
        answers = row.get("answers")
        normalized_answers = row.get("normalized_answers")
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(value, str) for value in answers)
            or not isinstance(normalized_answers, list)
            or len(normalized_answers) < 2
            or any(not isinstance(value, str) or not value for value in normalized_answers)
        ):
            raise MuSiQueTypedFormationError("TRAIN answer envelope malformed")
        corpus_raw = row.get("corpus")
        support_raw = row.get("support_indices")
        if not isinstance(corpus_raw, list) or not isinstance(support_raw, list):
            raise MuSiQueTypedFormationError("TRAIN retrieval envelope malformed")
        corpus: list[RetrievalParagraph] = []
        label_indices: list[int] = []
        for paragraph in corpus_raw:
            if not isinstance(paragraph, Mapping) or set(paragraph) != {
                "idx",
                "title",
                "text",
                "is_supporting",
            }:
                raise MuSiQueTypedFormationError("TRAIN paragraph schema mismatch")
            index = paragraph.get("idx")
            title = paragraph.get("title")
            text = paragraph.get("text")
            supporting = paragraph.get("is_supporting")
            if (
                type(index) is not int
                or not isinstance(title, str)
                or not title.strip()
                or not isinstance(text, str)
                or not text.strip()
                or type(supporting) is not bool
            ):
                raise MuSiQueTypedFormationError("TRAIN paragraph value malformed")
            corpus.append(RetrievalParagraph(index, title, text))
            if supporting:
                label_indices.append(index)
        if [paragraph.idx for paragraph in corpus] != list(range(len(corpus))):
            raise MuSiQueTypedFormationError("TRAIN paragraph idx drift")
        if len(corpus) < TOP_K or len(label_indices) < 2:
            raise MuSiQueTypedFormationError("TRAIN retrieval eligibility drift")
        if (
            any(type(value) is not int for value in support_raw)
            or tuple(support_raw) != tuple(label_indices)
        ):
            raise MuSiQueTypedFormationError("TRAIN supporting-label drift")
        item_commitments.append(stable_hash(row))
        # Answers, IDs, and paragraph labels are deliberately discarded here.
        items.append(
            _TrainItem(
                question=question,
                corpus=tuple(corpus),
                support_indices=tuple(label_indices),
            )
        )
    if stable_hash(item_commitments) != train_commitment["item_commitment_set_sha256"]:
        raise MuSiQueTypedFormationError("TRAIN item commitment set mismatch")
    return tuple(items)


def _load_inputs(
    train_jsonl_path: str | Path,
    acquisition_receipt_path: str | Path,
) -> tuple[tuple[_TrainItem, ...], dict[str, Any], Mapping[str, Any], Path, Path]:
    train_candidate = Path(train_jsonl_path)
    receipt_candidate = Path(acquisition_receipt_path)
    if train_candidate.name != "train.jsonl":
        raise MuSiQueTypedFormationError("formation accepts only the exact train.jsonl file")
    train_absolute = _assert_no_existing_symlink_components(
        train_candidate, "exact TRAIN file"
    )
    if receipt_candidate.is_symlink():
        raise MuSiQueTypedFormationError("public acquisition receipt may not be a symlink")
    if not train_absolute.is_file() or not receipt_candidate.is_file():
        raise MuSiQueTypedFormationError("formation input file is missing")
    _require_ignored_untracked_if_in_repository(train_absolute, "exact TRAIN file")
    train_path = train_absolute.resolve(strict=True)
    receipt_path = receipt_candidate.resolve(strict=True)
    if train_path == receipt_path or train_path.parent == receipt_path.parent:
        raise MuSiQueTypedFormationError(
            "exact TRAIN content and public acquisition receipt must be physically separated"
        )
    acquisition, train_commitment = _verify_acquisition_receipt(receipt_path)
    items = _parse_exact_train(train_path.read_bytes(), train_commitment)
    return items, acquisition, train_commitment, train_path, receipt_path


def _assess(
    program: TypedRetrievalProgram,
    items: Sequence[_TrainItem],
) -> CandidateAssessment:
    support_hits = 0
    support_labels = 0
    invalid = 0
    behavior: list[Mapping[str, Any]] = []
    for item in items:
        support_labels += len(item.support_indices)
        try:
            retrieved = retrieve(program, item.question, item.corpus)
        except (TypeError, ValueError, ArithmeticError):
            invalid += 1
            behavior.append({"invalid": True})
            continue
        support_hits += len(set(retrieved) & set(item.support_indices))
        behavior.append({"retrieved_idx": list(retrieved)})
    return CandidateAssessment(
        program=program,
        support_hit_count=support_hits,
        support_label_count=support_labels,
        invalid_count=invalid,
        behavior_hash=stable_hash(behavior),
    )


def _select(
    programs: Sequence[TypedRetrievalProgram],
    items: Sequence[_TrainItem],
) -> tuple[CandidateAssessment, tuple[CandidateAssessment, ...], tuple[CandidateAssessment, ...]]:
    assessments = tuple(_assess(program, items) for program in programs)
    by_behavior: dict[str, CandidateAssessment] = {}
    for assessment in sorted(assessments, key=lambda row: row.rank):
        by_behavior.setdefault(assessment.behavior_hash, assessment)
    unique = tuple(sorted(by_behavior.values(), key=lambda row: row.rank))
    return unique[0], assessments, unique


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_safe_public_payload(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden_keys = (
        '"answers"',
        '"corpus"',
        '"item_id"',
        '"normalized_answers"',
        '"paragraph"',
        '"question"',
        '"source_row_sha256"',
        '"support_indices"',
    )
    if any(key in serialized for key in forbidden_keys):
        raise MuSiQueTypedFormationError("unsafe content key in public formation payload")


def form_musique_typed_retriever(
    train_jsonl_path: str | Path,
    acquisition_receipt_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> FormationResult:
    """Select and optionally freeze one deterministic TRAIN-only retriever."""

    implementation = current_implementation_binding()
    items, acquisition, train_commitment, train_path, receipt_path = _load_inputs(
        train_jsonl_path,
        acquisition_receipt_path,
    )
    destination: Path | None = None
    if output_dir is not None:
        destination_absolute = _assert_no_existing_symlink_components(
            output_dir, "formation output"
        )
        if destination_absolute.exists() and not destination_absolute.is_dir():
            raise MuSiQueTypedFormationError("formation output must be a directory")
        _require_ignored_untracked_if_in_repository(
            destination_absolute, "formation output", directory=True
        )
        destination = destination_absolute.resolve(strict=False)
        train_parent = train_path.parent
        if (
            destination == train_parent
            or destination.is_relative_to(train_parent)
            or train_parent.is_relative_to(destination)
        ):
            raise MuSiQueTypedFormationError(
                "safe formation output must be disjoint from exact TRAIN content"
            )
    programs = tuple(enumerate_programs())
    if not programs or len(programs) > MAX_CANDIDATES:
        raise MuSiQueTypedFormationError("finite candidate budget exceeded")
    if any(program.type_issues() for program in programs):
        raise MuSiQueTypedFormationError("candidate grammar emitted an ill-typed program")
    winner, assessments, unique = _select(programs, items)
    winner_aliases = tuple(
        row for row in assessments if row.behavior_hash == winner.behavior_hash
    )

    fold_rows: list[dict[str, Any]] = []
    fold_program_hashes: list[str] = []
    fold_full_behavior_hashes: list[str] = []
    for fold_index in range(4):
        held_positions = tuple(
            position for position in range(len(items)) if position % 4 == fold_index
        )
        held_set = set(held_positions)
        fitting = tuple(item for position, item in enumerate(items) if position not in held_set)
        held = tuple(items[position] for position in held_positions)
        fold_winner, _, fold_unique = _select(programs, fitting)
        held_assessment = _assess(fold_winner.program, held)
        full_assessment = _assess(fold_winner.program, items)
        fold_program_hashes.append(fold_winner.program.program_hash)
        fold_full_behavior_hashes.append(full_assessment.behavior_hash)
        fold_rows.append(
            {
                "fold_index": fold_index,
                "fit_item_count": len(fitting),
                "held_out_item_count": len(held),
                "fit_behavior_unique_count": len(fold_unique),
                "selected_program_hash": fold_winner.program.program_hash,
                "selected_full_train_behavior_hash": full_assessment.behavior_hash,
                "held_out_support_hit_count": held_assessment.support_hit_count,
                "held_out_support_label_count": held_assessment.support_label_count,
                "held_out_invalid_count": held_assessment.invalid_count,
            }
        )

    acquisition_hash = _require_sha256(
        acquisition.get("acquisition_sha256"), "acquisition hash"
    )
    receipt_body: dict[str, Any] = {
        "schema": "musique_typed_retriever_formation_v1_receipt",
        "formation_version": FORMATION_VERSION,
        "status": "formed_train_only",
        "implementation": implementation,
        "source_binding": {
            "claim_scope": CLAIM_SCOPE,
            "acquisition_sha256": acquisition_hash,
            "acquisition_receipt_file_sha256": _sha256_file(receipt_path),
            "private_pack_sha256": acquisition["commitments"]["private_pack_sha256"],
            "train_file_sha256": train_commitment["file_sha256"],
            "train_item_commitment_set_sha256": train_commitment[
                "item_commitment_set_sha256"
            ],
            "train_item_count": len(items),
        },
        "search_receipt": {
            "dsl_version": DSL_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "top_k": TOP_K,
            "candidate_count": len(programs),
            "candidate_budget": MAX_CANDIDATES,
            "type_valid_count": sum(not row.program.type_issues() for row in assessments),
            "behavior_unique_count": len(unique),
            "behavior_alias_count": len(assessments) - len(unique),
            "behavior_alias_program_set_sha256": stable_hash(
                sorted(row.program.program_hash for row in winner_aliases)
            ),
            "behavior_deduplicated_before_selection": True,
            "selection_policy": SELECTION_POLICY,
            "maximum_semantic_nodes": max(
                len(row.program.semantic_nodes) for row in assessments
            ),
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
            "behavior_alias_class_size": len(winner_aliases),
        },
        "crossfit_receipt": {
            "policy": FOLD_POLICY,
            "fold_count": 4,
            "folds": fold_rows,
            "selected_program_stable": len(set(fold_program_hashes)) == 1
            and fold_program_hashes[0] == winner.program.program_hash,
            "selected_behavior_stable": len(set(fold_full_behavior_hashes)) == 1
            and fold_full_behavior_hashes[0] == winner.behavior_hash,
        },
        "offline_contract": {
            "partition": "train",
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "development_execution_authorized": False,
            "sealed_execution_authorized": False,
        },
        "claim_boundary": {
            "train_only_formation": True,
            "performance_claim": False,
            "claim_scope": CLAIM_SCOPE,
            "public_payload_contains_only_counts_hashes_and_program_grammar": True,
        },
        "raw_content_persisted": False,
    }
    receipt = {**receipt_body, "receipt_hash": stable_hash(receipt_body)}
    envelope = {
        "operator_version": OPERATOR_VERSION,
        "implementation": implementation,
        "program": winner.program.to_dict(),
        "program_hash": winner.program.program_hash,
        "formation_receipt_hash": receipt["receipt_hash"],
        "raw_content_persisted": False,
    }
    _assert_safe_public_payload(receipt)
    _assert_safe_public_payload(envelope)

    if destination is not None:
        _write_json_exclusive(destination / "formation.receipt.json", receipt)
        _write_json_exclusive(destination / "frozen_program.json", envelope)
    return FormationResult(program=winner.program, receipt=receipt)


def load_formation_receipt(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MuSiQueTypedFormationError("formation receipt must be an object")
    declared = payload.get("receipt_hash")
    body = dict(payload)
    body.pop("receipt_hash", None)
    if stable_hash(body) != declared:
        raise MuSiQueTypedFormationError("formation receipt hash mismatch")
    if (
        payload.get("schema") != "musique_typed_retriever_formation_v1_receipt"
        or payload.get("raw_content_persisted") is not False
        or payload.get("offline_contract", {}).get("online_evaluator_calls") != 0
        or payload.get("offline_contract", {}).get(
            "development_execution_authorized"
        )
        is not False
    ):
        raise MuSiQueTypedFormationError("unsafe formation receipt")
    _validated_implementation_binding(payload.get("implementation"))
    _assert_safe_public_payload(payload)
    return payload


def load_frozen_program(
    path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    verify_live: bool = False,
    implementation_root: str | Path | None = None,
) -> TypedRetrievalProgram:
    if receipt_path is None:
        raise MuSiQueTypedFormationError(
            "formation receipt is required to load a frozen program"
        )
    if type(verify_live) is not bool:
        raise MuSiQueTypedFormationError("verify_live must be boolean")
    receipt = load_formation_receipt(receipt_path)
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_keys = {
        "operator_version",
        "implementation",
        "program",
        "program_hash",
        "formation_receipt_hash",
        "raw_content_persisted",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected_keys:
        raise MuSiQueTypedFormationError("frozen program envelope schema mismatch")
    if (
        envelope.get("operator_version") != OPERATOR_VERSION
        or envelope.get("raw_content_persisted") is not False
    ):
        raise MuSiQueTypedFormationError("unsafe frozen program envelope")
    payload = envelope.get("program")
    if not isinstance(payload, Mapping):
        raise MuSiQueTypedFormationError("frozen program payload malformed")
    try:
        program = TypedRetrievalProgram.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise MuSiQueTypedFormationError("frozen program payload malformed") from exc
    if dict(payload) != program.to_dict() or program.type_issues():
        raise MuSiQueTypedFormationError("frozen program canonical payload mismatch")
    if envelope.get("program_hash") != program.program_hash:
        raise MuSiQueTypedFormationError("frozen program hash mismatch")
    _require_sha256(envelope.get("formation_receipt_hash"), "formation receipt binding")
    envelope_implementation = _validated_implementation_binding(
        envelope.get("implementation")
    )
    receipt_implementation = _validated_implementation_binding(
        receipt.get("implementation")
    )
    if envelope_implementation != receipt_implementation:
        raise MuSiQueTypedFormationError("frozen implementation binding mismatch")
    if receipt.get("receipt_hash") != envelope.get("formation_receipt_hash"):
        raise MuSiQueTypedFormationError("formation receipt binding mismatch")
    if receipt.get("selection_receipt", {}).get(
        "selected_program_hash"
    ) != program.program_hash:
        raise MuSiQueTypedFormationError("selected program binding mismatch")
    if verify_live:
        verify_live_implementation(
            envelope_implementation,
            project_root=implementation_root,
        )
    _assert_safe_public_payload(envelope)
    return program
