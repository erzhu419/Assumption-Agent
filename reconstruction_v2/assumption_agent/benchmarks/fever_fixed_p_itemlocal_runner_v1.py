"""One-shot FEVER item-local sentence retrieval with the exact frozen MuSiQue P.

The formal runner consumes a label-free 128-item action pack.  Each item has
one claim, 32 canonical sentence documents, and a frozen BM25 score/rank
vector used only by the RAW arm.  The other arms are the exact pre-existing
MuSiQue typed program P and the existing filesystem-attested official
HippoRAG runtime.

All 384 actions are joined and the official runtime is freshly re-attested
before a private action seal is durably written.  Only then is the late-label
callback invoked.  Public output is aggregate-only and is a terminal
measurement receipt: there is no p-value, promotion rule, gate, or later
stage.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..models import stable_hash
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .musique_recursive_study_blocks_v1 import load_study_frozen_program
from .musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    TypedRetrievalProgram,
    retrieve as typed_retrieve,
)


VERSION = "fever_fixed_p_itemlocal_runner_v1"
RESULT_SCHEMA = f"{VERSION}_aggregate_result"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure"
ACTION_PACK_SCHEMA = "fever_fixed_p_itemlocal_action_pack_v1"
ACTION_ITEM_SCHEMA = "fever_fixed_p_itemlocal_action_item_v1"
LABEL_PACK_SCHEMA = "fever_fixed_p_itemlocal_label_pack_v1"
LABEL_ITEM_SCHEMA = "fever_fixed_p_itemlocal_label_item_v1"
ACQUISITION_SCHEMA = "fever_official_fixed_transfer_acquisition_v1"
ACQUISITION_STATUS = "formal_itemlocal_pack_acquired"
IMPLEMENTATION_FREEZE_SCHEMA = (
    "fever_official_fixed_transfer_implementation_freeze_v1"
)

ITEM_COUNT = 128
DOCUMENTS_PER_ITEM = 32
TOP_K = 5
OFFICIAL_CONCURRENCY_CAP = 8
LOCAL_CONCURRENCY_CAP = 64
ARM_IDS = ("canonical_RAW", "exact_frozen_P", "official_HippoRAG")

FIXED_P_PROGRAM_SHA256 = (
    "0e9fea159e2dbcb302575f97954be8461c9921a91e11ef9b64a80ecab9640785"
)
FIXED_P_FROZEN_PROGRAM_FILE_SHA256 = (
    "3ea4362281fa6d86eec41506e7f017dd8794f8d09aecbac04fd2ce6309dda8a6"
)
FIXED_P_ENVELOPE_SHA256 = (
    "052cd52956d9196d78fa1bde77071433bddaa76d47b505619ad2d05142640474"
)
FIXED_P_FORMATION_RECEIPT_FILE_SHA256 = (
    "bbc4f1c737df7c908fc86160719789e20ee4e1c6cb4b77d25b6cb81188f15b6a"
)
FIXED_P_FORMATION_RECEIPT_SHA256 = (
    "5aacbf417ebd1fce30087c9fc4653aa563709c2302f57500cbcbaaa300fae145"
)
DESIGN_SHA256 = (
    "d000802fdc2a56aa8d91991abd013101a33aa147d89225d292b31b60b4d014aa"
)
DESIGN_FILE_SHA256 = (
    "b9b04f607b78a6b24d678de5dea1eff8c097c6a7f393100ceffd8dfe179c822b"
)

ACTION_PACK_RELATIVE_PATH = Path(
    "artifacts/fever_official_fixed_transfer_v1/action_pack.json"
)
LABEL_PACK_RELATIVE_PATH = Path(
    "artifacts/fever_official_fixed_transfer_v1/label_pack.json"
)
FORMAL_ROOT_RELATIVE_PATH = Path(
    "artifacts/fever_official_fixed_transfer_v1/formal_runner"
)
ATTEMPT_MARKER_RELATIVE_PATH = FORMAL_ROOT_RELATIVE_PATH / "formal.attempt.marker"
ACTION_SEAL_RELATIVE_PATH = FORMAL_ROOT_RELATIVE_PATH / "formal.action.seal.json"
WORK_ROOT_RELATIVE_PATH = FORMAL_ROOT_RELATIVE_PATH / "formal.work"
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/fever_official_fixed_transfer_implementation_freeze_v1.json"
)
ACQUISITION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/fever_official_fixed_transfer_acquisition_v1.json"
)
RESULT_RECEIPT_RELATIVE_PATH = Path(
    "manifests/fever_official_fixed_transfer_result_v1.json"
)
FAILURE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/fever_official_fixed_transfer_failure_v1.json"
)
P_FROZEN_PROGRAM_RELATIVE_PATH = Path(
    "manifests/musique_recursive_study_f1_formation_v1/frozen_program.json"
)
P_FORMATION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/musique_recursive_study_f1_formation_v1/formation.receipt.json"
)
OFFICIAL_BASE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
)
OFFICIAL_ATTESTATION_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_runtime_attestation_v2.json"
)

PRIVATE_MODE = 0o600
PUBLIC_MODE = 0o644
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_FORMAL_ENTRY_ACTIVE = False

_REQUIRED_FREEZE_PATHS = frozenset(
    {
        "assumption_agent/benchmarks/fever_fixed_p_itemlocal_runner_v1.py",
        "tests/test_fever_fixed_p_itemlocal_runner_v1.py",
        "assumption_agent/benchmarks/fever_fixed_p_itemlocal_acquisition_v1.py",
        "tests/test_fever_fixed_p_itemlocal_acquisition_v1.py",
        "manifests/fever_fixed_p_itemlocal_reranking_design_v1.json",
        str(P_FROZEN_PROGRAM_RELATIVE_PATH),
        str(P_FORMATION_RECEIPT_RELATIVE_PATH),
        "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
        "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
        "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
        "assumption_agent/models.py",
        str(OFFICIAL_BASE_RECEIPT_RELATIVE_PATH),
        str(OFFICIAL_ATTESTATION_RELATIVE_PATH),
        "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
        "replication_runtime/musique_official_hipporag_v1/contract.py",
        "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
        "replication_runtime/musique_official_hipporag_v1/worker.py",
    }
)


class FeverFixedPRunnerError(RuntimeError):
    """A frozen input, action, runtime, label, or custody invariant failed."""


class OfficialRuntimeProtocol(Protocol):
    @property
    def safe_binding(self) -> Mapping[str, Any]: ...

    def retrieve(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]: ...

    def fresh_reverify(self) -> Mapping[str, Any]: ...


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FeverFixedPRunnerError(f"{field_name} must be lowercase sha256")
    return value


def _assert_no_symlink_components(path: Path, field_name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise FeverFixedPRunnerError(
                f"{field_name} contains a symbolic-link component"
            )


def _require_regular_mode(path: Path, mode: int, field_name: str) -> None:
    _assert_no_symlink_components(path, field_name)
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != mode:
        raise FeverFixedPRunnerError(f"{field_name} mode or type drifted")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], mode: int) -> str:
    _assert_no_symlink_components(path.parent, "output parent")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = canonical_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return _sha256_bytes(raw)


def _self_hashed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    return {**dict(body), field_name: stable_hash(body)}


def _read_canonical_private_json(
    path: Path, *, field_name: str, expected_file_sha256: str
) -> tuple[dict[str, Any], bytes]:
    _require_regular_mode(path, PRIVATE_MODE, field_name)
    raw = path.read_bytes()
    if _sha256_bytes(raw) != _require_sha256(expected_file_sha256, field_name):
        raise FeverFixedPRunnerError(f"{field_name} file hash drifted")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverFixedPRunnerError(f"{field_name} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise FeverFixedPRunnerError(f"{field_name} is not canonical JSON")
    return value, raw


@dataclass(frozen=True)
class SentenceDocument:
    idx: int
    line_number: int
    title: str = field(repr=False)
    text: str = field(repr=False)

    def retrieval_paragraph(self) -> RetrievalParagraph:
        return RetrievalParagraph(self.idx, self.title, self.text)

    def official_paragraph(self) -> dict[str, object]:
        return {"idx": self.idx, "title": self.title, "paragraph_text": self.text}


@dataclass(frozen=True)
class ActionItem:
    ordinal: int
    item_id_hash: str
    claim: str = field(repr=False)
    documents: tuple[SentenceDocument, ...] = field(repr=False)
    bm25_scores: tuple[int, ...] = field(repr=False)
    bm25_rank: tuple[int, ...]
    action_item_sha256: str

    @property
    def raw_top5(self) -> tuple[int, int, int, int, int]:
        return tuple(self.bm25_rank[:TOP_K])  # type: ignore[return-value]

    @property
    def typed_corpus(self) -> tuple[RetrievalParagraph, ...]:
        return tuple(row.retrieval_paragraph() for row in self.documents)

    @property
    def official_corpus(self) -> tuple[dict[str, object], ...]:
        return tuple(row.official_paragraph() for row in self.documents)


@dataclass(frozen=True)
class ActionPack:
    items: tuple[ActionItem, ...]
    pack_sha256: str
    file_sha256: str
    item_commitment_set_sha256: str


@dataclass(frozen=True)
class LabelItem:
    ordinal: int
    item_id_hash: str
    action_item_sha256: str
    gold_indices: tuple[int, ...]
    source_label: str
    label_item_sha256: str


@dataclass(frozen=True)
class LabelPack:
    items: tuple[LabelItem, ...]
    pack_sha256: str
    file_sha256: str
    item_commitment_set_sha256: str


@dataclass(frozen=True)
class MeasurementOutcome:
    arm_metrics: Mapping[str, Mapping[str, Any]]
    paired_aggregates: Mapping[str, Mapping[str, int]]
    action_table_sha256: str
    action_seal_sha256: str
    action_seal_file_sha256: str
    runtime_binding_sha256: str
    action_pack_file_sha256: str
    action_pack_sha256: str
    action_item_commitment_set_sha256: str
    label_pack_file_sha256: str
    label_pack_sha256: str
    label_item_commitment_set_sha256: str


def _parse_document(value: object, expected_idx: int) -> SentenceDocument:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"doc_id", "line_number", "page_id", "sentence_text"}
        or value.get("doc_id") != expected_idx
        or type(value.get("line_number")) is not int
        or int(value["line_number"]) < 0
        or not isinstance(value.get("page_id"), str)
        or not str(value["page_id"]).strip()
        or not isinstance(value.get("sentence_text"), str)
        or not str(value["sentence_text"]).strip()
        or "\x00" in str(value["page_id"])
        or "\x00" in str(value["sentence_text"])
        or len(str(value["page_id"])) > 250_000
        or len(str(value["sentence_text"])) > 250_000
    ):
        raise FeverFixedPRunnerError("sentence document contract drifted")
    return SentenceDocument(
        expected_idx,
        int(value["line_number"]),
        str(value["page_id"]),
        str(value["sentence_text"]),
    )


def _parse_action_item(value: object, expected_ordinal: int) -> ActionItem:
    expected_keys = {
        "action_item_sha256",
        "bm25_rank",
        "bm25_scores",
        "claim",
        "documents",
        "item_id_hash",
        "ordinal",
        "schema",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema") != ACTION_ITEM_SCHEMA
        or value.get("ordinal") != expected_ordinal
        or not isinstance(value.get("claim"), str)
        or not str(value["claim"]).strip()
        or "\x00" in str(value["claim"])
        or len(str(value["claim"])) > 250_000
    ):
        raise FeverFixedPRunnerError("action item schema drifted")
    item_id_hash = _require_sha256(value.get("item_id_hash"), "item id hash")
    documents_raw = value.get("documents")
    scores_raw = value.get("bm25_scores")
    rank_raw = value.get("bm25_rank")
    if (
        not isinstance(documents_raw, list)
        or len(documents_raw) != DOCUMENTS_PER_ITEM
        or not isinstance(scores_raw, list)
        or len(scores_raw) != DOCUMENTS_PER_ITEM
        or not isinstance(rank_raw, list)
        or len(rank_raw) != DOCUMENTS_PER_ITEM
    ):
        raise FeverFixedPRunnerError("action item vector length drifted")
    documents = tuple(
        _parse_document(row, index) for index, row in enumerate(documents_raw)
    )
    if len({(row.title, row.line_number) for row in documents}) != DOCUMENTS_PER_ITEM:
        raise FeverFixedPRunnerError("sentence documents are not unique")
    scores: list[int] = []
    for score in scores_raw:
        if type(score) is not int:
            raise FeverFixedPRunnerError("BM25 score is not a quantized integer")
        scores.append(score)
    if (
        any(type(index) is not int for index in rank_raw)
        or sorted(rank_raw) != list(range(DOCUMENTS_PER_ITEM))
    ):
        raise FeverFixedPRunnerError("BM25 rank is not a full index permutation")
    expected_rank = sorted(
        range(DOCUMENTS_PER_ITEM), key=lambda index: (-scores[index], index)
    )
    if rank_raw != expected_rank:
        raise FeverFixedPRunnerError("BM25 score/rank consistency drifted")
    body = dict(value)
    declared = _require_sha256(
        body.pop("action_item_sha256", None), "action item commitment"
    )
    if stable_hash(body) != declared:
        raise FeverFixedPRunnerError("action item self-hash drifted")
    return ActionItem(
        ordinal=expected_ordinal,
        item_id_hash=item_id_hash,
        claim=str(value["claim"]),
        documents=documents,
        bm25_scores=tuple(scores),
        bm25_rank=tuple(rank_raw),
        action_item_sha256=declared,
    )


def load_action_pack(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_item_commitment_set_sha256: str,
) -> ActionPack:
    value, raw = _read_canonical_private_json(
        Path(path), field_name="action pack", expected_file_sha256=expected_file_sha256
    )
    expected_keys = {
        "document_count_per_item",
        "item_count",
        "items",
        "labels_included",
        "pack_sha256",
        "schema",
        "version",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != ACTION_PACK_SCHEMA
        or value.get("version") != "v1"
        or value.get("item_count") != ITEM_COUNT
        or value.get("document_count_per_item") != DOCUMENTS_PER_ITEM
        or value.get("labels_included") is not False
        or not isinstance(value.get("items"), list)
        or len(value["items"]) != ITEM_COUNT
    ):
        raise FeverFixedPRunnerError("action pack schema drifted")
    body = dict(value)
    pack_sha = _require_sha256(body.pop("pack_sha256", None), "action pack hash")
    if stable_hash(body) != pack_sha:
        raise FeverFixedPRunnerError("action pack self-hash drifted")
    items = tuple(
        _parse_action_item(row, ordinal)
        for ordinal, row in enumerate(value["items"])
    )
    if len({row.item_id_hash for row in items}) != ITEM_COUNT:
        raise FeverFixedPRunnerError("action pack item identities are not unique")
    set_hash = stable_hash([row.action_item_sha256 for row in items])
    if set_hash != _require_sha256(
        expected_item_commitment_set_sha256, "action item commitment set"
    ):
        raise FeverFixedPRunnerError("action item commitment set drifted")
    return ActionPack(items, pack_sha, _sha256_bytes(raw), set_hash)


def _parse_label_item(value: object, expected_ordinal: int) -> LabelItem:
    expected_keys = {
        "action_item_sha256",
        "gold_indices",
        "item_id_hash",
        "label_item_sha256",
        "ordinal",
        "schema",
        "source_label",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema") != LABEL_ITEM_SCHEMA
        or value.get("ordinal") != expected_ordinal
        or value.get("source_label") not in {"SUPPORTS", "REFUTES"}
    ):
        raise FeverFixedPRunnerError("label item schema drifted")
    item_id_hash = _require_sha256(value.get("item_id_hash"), "label item id hash")
    action_hash = _require_sha256(
        value.get("action_item_sha256"), "label action commitment"
    )
    gold_raw = value.get("gold_indices")
    if (
        not isinstance(gold_raw, list)
        or not gold_raw
        or len(gold_raw) > TOP_K
        or any(type(index) is not int for index in gold_raw)
        or len(set(gold_raw)) != len(gold_raw)
        or gold_raw != sorted(gold_raw)
        or any(not 0 <= index < DOCUMENTS_PER_ITEM for index in gold_raw)
    ):
        raise FeverFixedPRunnerError("single gold evidence set drifted")
    body = dict(value)
    declared = _require_sha256(
        body.pop("label_item_sha256", None), "label item commitment"
    )
    if stable_hash(body) != declared:
        raise FeverFixedPRunnerError("label item self-hash drifted")
    return LabelItem(
        expected_ordinal,
        item_id_hash,
        action_hash,
        tuple(gold_raw),
        str(value["source_label"]),
        declared,
    )


def load_label_pack(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_item_commitment_set_sha256: str,
) -> LabelPack:
    value, raw = _read_canonical_private_json(
        Path(path), field_name="label pack", expected_file_sha256=expected_file_sha256
    )
    expected_keys = {
        "gold_contract",
        "item_count",
        "items",
        "pack_sha256",
        "schema",
        "version",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != LABEL_PACK_SCHEMA
        or value.get("version") != "v1"
        or value.get("item_count") != ITEM_COUNT
        or value.get("gold_contract")
        != "one_preselected_gold_evidence_set_per_item"
        or not isinstance(value.get("items"), list)
        or len(value["items"]) != ITEM_COUNT
    ):
        raise FeverFixedPRunnerError("label pack schema drifted")
    body = dict(value)
    pack_sha = _require_sha256(body.pop("pack_sha256", None), "label pack hash")
    if stable_hash(body) != pack_sha:
        raise FeverFixedPRunnerError("label pack self-hash drifted")
    items = tuple(
        _parse_label_item(row, ordinal)
        for ordinal, row in enumerate(value["items"])
    )
    if len({row.item_id_hash for row in items}) != ITEM_COUNT:
        raise FeverFixedPRunnerError("label pack item identities are not unique")
    if {
        label: sum(row.source_label == label for row in items)
        for label in ("SUPPORTS", "REFUTES")
    } != {"SUPPORTS": 64, "REFUTES": 64}:
        raise FeverFixedPRunnerError("late label strata are not fixed 64/64")
    set_hash = stable_hash([row.label_item_sha256 for row in items])
    if set_hash != _require_sha256(
        expected_item_commitment_set_sha256, "label item commitment set"
    ):
        raise FeverFixedPRunnerError("label item commitment set drifted")
    return LabelPack(items, pack_sha, _sha256_bytes(raw), set_hash)


def exact_fixed_p() -> TypedRetrievalProgram:
    program = TypedRetrievalProgram(
        seed_algorithm="tfidf",
        title_weight=4,
        text_weight=1,
        expansion_mode="entity_token_one_hop",
        expansion_weight=2,
        top_k=5,
        tokenizer_version="unicode_nfkc_casefold_alnum_v1",
        dsl_version="musique_finite_typed_retrieval_dsl_v1",
    )
    if program.program_hash != FIXED_P_PROGRAM_SHA256 or program.type_issues():
        raise FeverFixedPRunnerError("exact fixed P materialization drifted")
    return program


def load_exact_frozen_p(project_root: str | Path) -> TypedRetrievalProgram:
    root = Path(project_root).resolve(strict=True)
    program_path = root / P_FROZEN_PROGRAM_RELATIVE_PATH
    receipt_path = root / P_FORMATION_RECEIPT_RELATIVE_PATH
    if (
        _sha256_file(program_path) != FIXED_P_FROZEN_PROGRAM_FILE_SHA256
        or _sha256_file(receipt_path) != FIXED_P_FORMATION_RECEIPT_FILE_SHA256
    ):
        raise FeverFixedPRunnerError("frozen P public files drifted")
    program, receipt, envelope = load_study_frozen_program(
        frozen_program_path=program_path,
        formation_receipt_path=receipt_path,
        verify_live=True,
        implementation_root=root,
    )
    if (
        program.program_hash != FIXED_P_PROGRAM_SHA256
        or envelope.get("envelope_hash") != FIXED_P_ENVELOPE_SHA256
        or receipt.get("receipt_hash") != FIXED_P_FORMATION_RECEIPT_SHA256
        or program != exact_fixed_p()
    ):
        raise FeverFixedPRunnerError("frozen P lineage drifted")
    return program


def _validate_ranking(value: Sequence[int], ordinal: int, arm: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise FeverFixedPRunnerError("retrieval ranking is not an index sequence")
    ranking = tuple(value)
    if (
        len(ranking) != TOP_K
        or len(set(ranking)) != TOP_K
        or any(type(index) is not int or not 0 <= index < DOCUMENTS_PER_ITEM for index in ranking)
    ):
        raise FeverFixedPRunnerError(
            f"{arm} ranking contract failed at ordinal {ordinal}"
        )
    return ranking


def _runtime_binding(runtime: OfficialRuntimeProtocol) -> tuple[dict[str, Any], str]:
    binding = dict(runtime.safe_binding)
    declared = binding.get("binding_sha256")
    if declared is None:
        digest = stable_hash(binding)
    else:
        digest = _require_sha256(declared, "official runtime binding")
    return binding, digest


def _run_action_barrier(
    *,
    action_pack: ActionPack,
    program: TypedRetrievalProgram,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
) -> tuple[dict[tuple[int, str], tuple[int, ...]], str, dict[str, Any]]:
    if program.program_hash != FIXED_P_PROGRAM_SHA256 or program != exact_fixed_p():
        raise FeverFixedPRunnerError("measurement did not receive exact frozen P")
    if work_root.exists() or work_root.is_symlink():
        raise FeverFixedPRunnerError("formal work root already exists")
    work_root.mkdir(mode=0o700, parents=True)
    preflight_binding, runtime_hash = _runtime_binding(runtime)
    results: dict[tuple[int, str], tuple[int, ...]] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def local_action(item: ActionItem, arm: str) -> tuple[int, ...]:
        if arm == "canonical_RAW":
            # BM25 scores/ranks terminate at RAW.
            return _validate_ranking(item.raw_top5, item.ordinal, arm)
        if arm != "exact_frozen_P":
            raise FeverFixedPRunnerError("unknown local arm")
        # Exact P sees only claim and sentence docs, never BM25 fields.
        p_value = typed_retrieve(program, item.claim, item.typed_corpus)
        return _validate_ranking(p_value, item.ordinal, arm)

    def official_action(item: ActionItem) -> tuple[int, ...]:
        # The official arm receives the same claim/docs, never BM25 fields or labels.
        value = runtime.retrieve(
            question=item.claim,
            paragraphs=item.official_corpus,
            work_root=work_root / f"official_item_{item.ordinal:03d}",
        )
        return _validate_ranking(value, item.ordinal, "official_HippoRAG")

    futures: dict[Future[Any], tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=LOCAL_CONCURRENCY_CAP) as local_pool, (
        ThreadPoolExecutor(max_workers=OFFICIAL_CONCURRENCY_CAP)
    ) as official_pool:
        for item in action_pack.items:
            for local_arm in ("canonical_RAW", "exact_frozen_P"):
                futures[local_pool.submit(local_action, item, local_arm)] = (
                    item.ordinal,
                    local_arm,
                )
            futures[official_pool.submit(official_action, item)] = (
                item.ordinal,
                "official_HippoRAG",
            )
        for future in as_completed(futures):
            ordinal, arm = futures[future]
            try:
                value = future.result()
                with lock:
                    results[(ordinal, arm)] = value
            except BaseException as exc:  # all submitted work still joins on pool exit
                errors.append(exc)

    try:
        postflight_binding = dict(runtime.fresh_reverify())
    except BaseException as exc:
        errors.append(exc)
        postflight_binding = {}
    if postflight_binding != preflight_binding:
        errors.append(FeverFixedPRunnerError("official runtime postflight drifted"))
    if errors:
        raise FeverFixedPRunnerError(
            "action barrier terminated with an action or postflight failure"
        ) from errors[0]
    if len(results) != ITEM_COUNT * len(ARM_IDS) or any(
        (ordinal, arm) not in results
        for ordinal in range(ITEM_COUNT)
        for arm in ARM_IDS
    ):
        raise FeverFixedPRunnerError("action barrier terminal count drifted")
    return results, runtime_hash, preflight_binding


def _persist_action_seal(
    *,
    path: Path,
    action_pack: ActionPack,
    rankings: Mapping[tuple[int, str], tuple[int, ...]],
    runtime_binding_sha256: str,
    acquisition_sha256: str | None,
) -> tuple[str, str, str]:
    action_rows = [
        {
            "ordinal": item.ordinal,
            "item_id_hash": item.item_id_hash,
            "action_item_sha256": item.action_item_sha256,
            "rankings": {
                arm: list(rankings[(item.ordinal, arm)]) for arm in ARM_IDS
            },
        }
        for item in action_pack.items
    ]
    action_table_hash = stable_hash(action_rows)
    body = {
        "schema": f"{VERSION}_label_free_action_seal",
        "version": VERSION,
        "status": "all_384_actions_terminal_postflight_verified_before_labels",
        "item_count": ITEM_COUNT,
        "work_unit_count": ITEM_COUNT * len(ARM_IDS),
        "arm_ids": list(ARM_IDS),
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "local_concurrency_cap": LOCAL_CONCURRENCY_CAP,
        "fixed_P_program_sha256": FIXED_P_PROGRAM_SHA256,
        "action_pack_file_sha256": action_pack.file_sha256,
        "action_pack_sha256": action_pack.pack_sha256,
        "action_item_commitment_set_sha256": action_pack.item_commitment_set_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "acquisition_sha256": acquisition_sha256,
        "action_table_sha256": action_table_hash,
        "action_rows": action_rows,
        "labels_opened_before_action_seal": False,
    }
    seal = _self_hashed(body, "action_seal_sha256")
    file_hash = _write_json_exclusive(path, seal, PRIVATE_MODE)
    return action_table_hash, str(seal["action_seal_sha256"]), file_hash


def _join_labels(action_pack: ActionPack, label_pack: LabelPack) -> None:
    if len(label_pack.items) != len(action_pack.items):
        raise FeverFixedPRunnerError("late label count drifted")
    for action, label in zip(action_pack.items, label_pack.items):
        if (
            action.ordinal != label.ordinal
            or action.item_id_hash != label.item_id_hash
            or action.action_item_sha256 != label.action_item_sha256
        ):
            raise FeverFixedPRunnerError("late label/action join drifted")


def _score_aggregate(
    *,
    action_pack: ActionPack,
    label_pack: LabelPack,
    rankings: Mapping[tuple[int, str], tuple[int, ...]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    _join_labels(action_pack, label_pack)
    per_arm: dict[str, list[tuple[int, int]]] = {arm: [] for arm in ARM_IDS}
    for label in label_pack.items:
        gold = frozenset(label.gold_indices)
        for arm in ARM_IDS:
            retrieved = frozenset(rankings[(label.ordinal, arm)])
            hits = len(gold & retrieved)
            complete = int(gold.issubset(retrieved))
            per_arm[arm].append((hits, complete))
    arm_metrics: dict[str, dict[str, Any]] = {}

    def metrics(arm: str, indices: Sequence[int]) -> dict[str, int | float]:
        support_total = sum(len(label_pack.items[index].gold_indices) for index in indices)
        hit_count = sum(per_arm[arm][index][0] for index in indices)
        return {
            "set_aware_support_hit_count_at_5": hit_count,
            "set_aware_support_total": support_total,
            "micro_set_aware_support_recall_at_5": hit_count / support_total,
            "complete_item_count": sum(per_arm[arm][index][1] for index in indices),
            "item_count": len(indices),
        }

    all_indices = tuple(range(ITEM_COUNT))
    stratum_indices = {
        source_label: tuple(
            index
            for index, label in enumerate(label_pack.items)
            if label.source_label == source_label
        )
        for source_label in ("SUPPORTS", "REFUTES")
    }
    for arm in ARM_IDS:
        arm_metrics[arm] = {
            "overall": metrics(arm, all_indices),
            "by_source_label": {
                source_label: metrics(arm, indices)
                for source_label, indices in stratum_indices.items()
            },
        }

    def paired(left: str, right: str) -> dict[str, int]:
        hit_deltas = [
            left_row[0] - right_row[0]
            for left_row, right_row in zip(per_arm[left], per_arm[right])
        ]
        complete_deltas = [
            left_row[1] - right_row[1]
            for left_row, right_row in zip(per_arm[left], per_arm[right])
        ]
        return {
            "item_count": ITEM_COUNT,
            "net_gold_sentence_hits_at_5": sum(hit_deltas),
            "gold_hit_gain_item_count": sum(delta > 0 for delta in hit_deltas),
            "gold_hit_harm_item_count": sum(delta < 0 for delta in hit_deltas),
            "gold_hit_tie_item_count": sum(delta == 0 for delta in hit_deltas),
            "net_complete_gold_sets_at_5": sum(complete_deltas),
            "complete_gain_item_count": sum(delta > 0 for delta in complete_deltas),
            "complete_harm_item_count": sum(delta < 0 for delta in complete_deltas),
            "complete_tie_item_count": sum(delta == 0 for delta in complete_deltas),
        }

    paired_aggregates = {
        "P_minus_HippoRAG": paired("exact_frozen_P", "official_HippoRAG"),
        "P_minus_RAW": paired("exact_frozen_P", "canonical_RAW"),
    }
    return arm_metrics, paired_aggregates


def run_measurement(
    action_pack: ActionPack,
    *,
    late_label_loader: Callable[[], LabelPack],
    program: TypedRetrievalProgram,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    action_seal_path: Path,
    acquisition_sha256: str | None = None,
) -> MeasurementOutcome:
    """Run all label-free actions, seal them, then open labels exactly once."""

    rankings, runtime_hash, _binding = _run_action_barrier(
        action_pack=action_pack,
        program=program,
        runtime=runtime,
        work_root=work_root,
    )
    action_table_hash, seal_hash, seal_file_hash = _persist_action_seal(
        path=action_seal_path,
        action_pack=action_pack,
        rankings=rankings,
        runtime_binding_sha256=runtime_hash,
        acquisition_sha256=acquisition_sha256,
    )
    # This callback is the first label access.  The action seal is already
    # closed, fsynced, and parent-directory fsynced at this exact point.
    label_pack = late_label_loader()
    arm_metrics, paired = _score_aggregate(
        action_pack=action_pack,
        label_pack=label_pack,
        rankings=rankings,
    )
    return MeasurementOutcome(
        arm_metrics=arm_metrics,
        paired_aggregates=paired,
        action_table_sha256=action_table_hash,
        action_seal_sha256=seal_hash,
        action_seal_file_sha256=seal_file_hash,
        runtime_binding_sha256=runtime_hash,
        action_pack_file_sha256=action_pack.file_sha256,
        action_pack_sha256=action_pack.pack_sha256,
        action_item_commitment_set_sha256=action_pack.item_commitment_set_sha256,
        label_pack_file_sha256=label_pack.file_sha256,
        label_pack_sha256=label_pack.pack_sha256,
        label_item_commitment_set_sha256=label_pack.item_commitment_set_sha256,
    )


def aggregate_result_body(outcome: MeasurementOutcome) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "version": VERSION,
        "status": "terminal_complete_fixed_candidate_itemlocal_measurement",
        "scope": "FEVER_item_local_sentence_retrieval_only",
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "fixed_P_program_sha256": FIXED_P_PROGRAM_SHA256,
        "execution": {
            "item_count": ITEM_COUNT,
            "sentence_documents_per_item": DOCUMENTS_PER_ITEM,
            "top_k": TOP_K,
            "arm_ids": list(ARM_IDS),
            "work_unit_count": ITEM_COUNT * len(ARM_IDS),
            "retrieval_terminal_count": ITEM_COUNT * len(ARM_IDS),
            "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
            "local_concurrency_cap": LOCAL_CONCURRENCY_CAP,
            "all_actions_joined_before_postflight_and_action_seal": True,
            "fresh_official_runtime_postflight_before_action_seal": True,
            "late_label_callback_after_durable_action_seal": True,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "retry_replay_resample": 0,
        },
        "bindings": {
            "action_pack_file_sha256": outcome.action_pack_file_sha256,
            "action_pack_sha256": outcome.action_pack_sha256,
            "action_item_commitment_set_sha256": outcome.action_item_commitment_set_sha256,
            "label_pack_file_sha256": outcome.label_pack_file_sha256,
            "label_pack_sha256": outcome.label_pack_sha256,
            "label_item_commitment_set_sha256": outcome.label_item_commitment_set_sha256,
            "action_table_sha256": outcome.action_table_sha256,
            "action_seal_sha256": outcome.action_seal_sha256,
            "action_seal_file_sha256": outcome.action_seal_file_sha256,
            "runtime_binding_sha256": outcome.runtime_binding_sha256,
        },
        "aggregate_arm_metrics": {
            arm: dict(outcome.arm_metrics[arm]) for arm in ARM_IDS
        },
        "aggregate_paired_differences": {
            key: dict(value) for key, value in sorted(outcome.paired_aggregates.items())
        },
        "single_gold_set_scoring": {
            "gold_sentence_hits_at_5": True,
            "complete_iff_every_sentence_in_the_prefixed_single_gold_set_is_in_top5": True,
        },
        "public_payload_contract": {
            "aggregate_only": True,
            "claims_documents_labels_rankings_scores_item_ids_or_item_commitments_included": False,
        },
    }


def _git(project_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ("git", "-C", str(project_root), *arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        raise FeverFixedPRunnerError("git custody command failed")
    return completed


def _git_layout(project_root: Path) -> tuple[Path, str]:
    repository = Path(
        _git(project_root, "rev-parse", "--show-toplevel").stdout.decode().strip()
    ).resolve(strict=True)
    relative = project_root.resolve(strict=True).relative_to(repository)
    prefix = "" if str(relative) == "." else relative.as_posix().rstrip("/") + "/"
    return repository, prefix


def _committed_current_bytes(project_root: Path, relative_path: Path) -> bytes:
    repository, prefix = _git_layout(project_root)
    live = project_root / relative_path
    _assert_no_symlink_components(live, str(relative_path))
    if not live.is_file():
        raise FeverFixedPRunnerError(f"committed public file is absent: {relative_path}")
    committed = _git(repository, "show", f"HEAD:{prefix}{relative_path.as_posix()}").stdout
    raw = live.read_bytes()
    if raw != committed:
        raise FeverFixedPRunnerError(f"public file differs from actual HEAD: {relative_path}")
    return raw


def _load_self_hashed_public(
    project_root: Path,
    relative_path: Path,
    *,
    schema: str,
    hash_field: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _committed_current_bytes(project_root, relative_path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverFixedPRunnerError(f"{relative_path} is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise FeverFixedPRunnerError(f"{relative_path} schema drifted")
    body = dict(value)
    declared = _require_sha256(body.pop(hash_field, None), hash_field)
    if stable_hash(body) != declared:
        raise FeverFixedPRunnerError(f"{relative_path} self-hash drifted")
    return value, raw


def verify_implementation_freeze(project_root: str | Path) -> tuple[dict[str, Any], str]:
    root = Path(project_root).resolve(strict=True)
    freeze, _raw = _load_self_hashed_public(
        root,
        IMPLEMENTATION_FREEZE_RELATIVE_PATH,
        schema=IMPLEMENTATION_FREEZE_SCHEMA,
        hash_field="implementation_freeze_sha256",
    )
    actual_head = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
    creation_head = freeze.get("creation_HEAD")
    bindings = freeze.get("bindings")
    if (
        _GIT_COMMIT_RE.fullmatch(actual_head) is None
        or not isinstance(creation_head, str)
        or _GIT_COMMIT_RE.fullmatch(creation_head) is None
        or freeze.get("fixed_P_program_sha256") != FIXED_P_PROGRAM_SHA256
        or not isinstance(bindings, list)
        or not bindings
    ):
        raise FeverFixedPRunnerError("implementation freeze contract drifted")
    if _git(root, "merge-base", "--is-ancestor", creation_head, actual_head, check=False).returncode != 0:
        raise FeverFixedPRunnerError("implementation freeze HEAD is not an ancestor")
    repository, prefix = _git_layout(root)
    observed_paths: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "file_sha256",
            "git_blob_sha1",
            "relative_path",
        }:
            raise FeverFixedPRunnerError("implementation freeze binding drifted")
        relative = binding.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in observed_paths
        ):
            raise FeverFixedPRunnerError("implementation freeze path drifted")
        observed_paths.add(relative)
        file_sha = _require_sha256(binding.get("file_sha256"), "bound file hash")
        blob_sha = binding.get("git_blob_sha1")
        if not isinstance(blob_sha, str) or re.fullmatch(r"[0-9a-f]{40}", blob_sha) is None:
            raise FeverFixedPRunnerError("bound git blob hash drifted")
        git_path = f"{prefix}{relative}"
        creation_raw = _git(repository, "show", f"{creation_head}:{git_path}").stdout
        head_raw = _git(repository, "show", f"HEAD:{git_path}").stdout
        observed_blob = _git(repository, "rev-parse", f"{creation_head}:{git_path}").stdout.decode().strip()
        live_path = root / relative
        _assert_no_symlink_components(live_path, relative)
        if (
            not live_path.is_file()
            or creation_raw != head_raw
            or live_path.read_bytes() != head_raw
            or _sha256_bytes(head_raw) != file_sha
            or observed_blob != blob_sha
        ):
            raise FeverFixedPRunnerError(f"implementation binding drifted: {relative}")
    if not _REQUIRED_FREEZE_PATHS.issubset(observed_paths):
        raise FeverFixedPRunnerError("implementation freeze omits required bindings")
    return freeze, actual_head


def load_acquisition_receipt(
    project_root: str | Path, *, expected_freeze_sha256: str
) -> tuple[dict[str, Any], str]:
    root = Path(project_root).resolve(strict=True)
    receipt, raw = _load_self_hashed_public(
        root,
        ACQUISITION_RECEIPT_RELATIVE_PATH,
        schema=ACQUISITION_SCHEMA,
        hash_field="acquisition_sha256",
    )
    counts = receipt.get("counts")
    commitments = receipt.get("commitments")
    if (
        receipt.get("version") != "v1"
        or receipt.get("status") != ACQUISITION_STATUS
        or receipt.get("implementation_freeze_sha256")
        != _require_sha256(expected_freeze_sha256, "implementation freeze hash")
        or receipt.get("source_labels_and_evidence_read_only_by_acquisition") is not True
        or receipt.get("label_pack_opened_by_action_runner_before_action_barrier") is not False
        or counts
        != {"document_count_per_item": DOCUMENTS_PER_ITEM, "item_count": ITEM_COUNT}
        or not isinstance(commitments, Mapping)
        or set(commitments)
        != {
            "action_item_commitment_set_sha256",
            "action_pack_file_sha256",
            "label_item_commitment_set_sha256",
            "label_pack_file_sha256",
        }
    ):
        raise FeverFixedPRunnerError("acquisition receipt contract drifted")
    for key, value in commitments.items():
        _require_sha256(value, key)
    return receipt, _sha256_bytes(raw)


def _assert_private_canonical_paths(project_root: Path) -> None:
    for relative in (ACTION_PACK_RELATIVE_PATH, LABEL_PACK_RELATIVE_PATH):
        path = project_root / relative
        _require_regular_mode(path, PRIVATE_MODE, str(relative))
        ignored = _git(project_root, "check-ignore", "-q", relative.as_posix(), check=False)
        if ignored.returncode != 0:
            raise FeverFixedPRunnerError(f"private pack is not git-ignored: {relative}")


def _formal_paths(project_root: Path) -> dict[str, Path]:
    return {
        "marker": project_root / ATTEMPT_MARKER_RELATIVE_PATH,
        "seal": project_root / ACTION_SEAL_RELATIVE_PATH,
        "work": project_root / WORK_ROOT_RELATIVE_PATH,
        "result": project_root / RESULT_RECEIPT_RELATIVE_PATH,
        "failure": project_root / FAILURE_RECEIPT_RELATIVE_PATH,
    }


def _consume_formal_marker(
    *,
    project_root: Path,
    actual_head: str,
    freeze_sha256: str,
    freeze_file_sha256: str,
    acquisition_sha256: str,
    acquisition_file_sha256: str,
    runtime_binding_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], str]:
    paths = _formal_paths(project_root)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FeverFixedPRunnerError("canonical formal output already exists")
    marker_body = {
        "schema": f"{VERSION}_formal_attempt_marker",
        "version": VERSION,
        "status": "sole_formal_attempt_consumed",
        "attempt_count": 1,
        "actual_HEAD": actual_head,
        "implementation_freeze_sha256": freeze_sha256,
        "implementation_freeze_file_sha256": freeze_file_sha256,
        "acquisition_sha256": acquisition_sha256,
        "acquisition_file_sha256": acquisition_file_sha256,
        "fixed_P_program_sha256": FIXED_P_PROGRAM_SHA256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "action_pack_or_label_pack_rows_opened_before_marker": 0,
    }
    marker = _self_hashed(marker_body, "marker_sha256")
    file_hash = _write_json_exclusive(paths["marker"], marker, PRIVATE_MODE)
    return marker, paths, file_hash


def _persist_failure(
    *,
    path: Path,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    exc: BaseException,
) -> None:
    body = {
        "schema": FAILURE_SCHEMA,
        "version": VERSION,
        "status": "terminal_infrastructure_or_implementation_invalid_no_replay",
        "actual_HEAD": marker["actual_HEAD"],
        "marker_sha256": marker["marker_sha256"],
        "marker_file_sha256": marker_file_sha256,
        "implementation_freeze_sha256": marker["implementation_freeze_sha256"],
        "acquisition_sha256": marker["acquisition_sha256"],
        "fixed_P_program_sha256": FIXED_P_PROGRAM_SHA256,
        "failure_class": type(exc).__name__,
        "private_claim_document_label_ranking_or_item_identity_included": False,
        "retry_replay_resample_or_backup_attempt_authorized": False,
    }
    failure = _self_hashed(body, "failure_sha256")
    if not path.exists() and not path.is_symlink():
        _write_json_exclusive(path, failure, PUBLIC_MODE)


def _assert_aggregate_only(value: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "action_item_sha256",
        "bm25_rank",
        "bm25_scores",
        "claim",
        "documents",
        "gold_indices",
        "item_id",
        "item_id_hash",
        "label_item_sha256",
        "rankings",
        "scores",
    }

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if forbidden_keys.intersection(node):
                raise FeverFixedPRunnerError("public result is not aggregate-only")
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
    for forbidden_term in ('"p_value"', '"promotion"', '"gate"', '"stage"'):
        if forbidden_term in serialized:
            raise FeverFixedPRunnerError("public result contains an undeclared decision field")


def run_canonical_formal(
    *,
    project_root: Path,
    runtime: PreparedFormalRuntimeV2,
    program: TypedRetrievalProgram,
) -> dict[str, Any]:
    if _FORMAL_ENTRY_ACTIVE is not True:
        raise FeverFixedPRunnerError("canonical formal run is CLI-only")
    if not isinstance(runtime, PreparedFormalRuntimeV2):
        raise FeverFixedPRunnerError("formal runtime type is not attested")
    if program.program_hash != FIXED_P_PROGRAM_SHA256 or program != exact_fixed_p():
        raise FeverFixedPRunnerError("formal program is not exact frozen P")
    root = project_root.resolve(strict=True)
    freeze, actual_head = verify_implementation_freeze(root)
    freeze_sha = str(freeze["implementation_freeze_sha256"])
    freeze_file_sha = _sha256_file(root / IMPLEMENTATION_FREEZE_RELATIVE_PATH)
    acquisition, acquisition_file_sha = load_acquisition_receipt(
        root, expected_freeze_sha256=freeze_sha
    )
    acquisition_sha = str(acquisition["acquisition_sha256"])
    runtime_binding, runtime_binding_sha = _runtime_binding(runtime)
    _assert_private_canonical_paths(root)
    marker, paths, marker_file_hash = _consume_formal_marker(
        project_root=root,
        actual_head=actual_head,
        freeze_sha256=freeze_sha,
        freeze_file_sha256=freeze_file_sha,
        acquisition_sha256=acquisition_sha,
        acquisition_file_sha256=acquisition_file_sha,
        runtime_binding_sha256=runtime_binding_sha,
    )
    commitments = acquisition["commitments"]
    try:
        action_pack = load_action_pack(
            root / ACTION_PACK_RELATIVE_PATH,
            expected_file_sha256=commitments["action_pack_file_sha256"],
            expected_item_commitment_set_sha256=commitments[
                "action_item_commitment_set_sha256"
            ],
        )
        outcome = run_measurement(
            action_pack,
            late_label_loader=lambda: load_label_pack(
                root / LABEL_PACK_RELATIVE_PATH,
                expected_file_sha256=commitments["label_pack_file_sha256"],
                expected_item_commitment_set_sha256=commitments[
                    "label_item_commitment_set_sha256"
                ],
            ),
            program=program,
            runtime=runtime,
            work_root=paths["work"],
            action_seal_path=paths["seal"],
            acquisition_sha256=acquisition_sha,
        )
        body = aggregate_result_body(outcome)
        body.update(
            {
                "actual_HEAD": actual_head,
                "implementation_freeze_sha256": freeze_sha,
                "implementation_freeze_file_sha256": freeze_file_sha,
                "acquisition_sha256": acquisition_sha,
                "acquisition_file_sha256": acquisition_file_sha,
                "marker_sha256": marker["marker_sha256"],
                "marker_file_sha256": marker_file_hash,
                "fixed_P_lineage": {
                    "program_sha256": FIXED_P_PROGRAM_SHA256,
                    "frozen_program_file_sha256": FIXED_P_FROZEN_PROGRAM_FILE_SHA256,
                    "frozen_program_envelope_sha256": FIXED_P_ENVELOPE_SHA256,
                    "formation_receipt_file_sha256": FIXED_P_FORMATION_RECEIPT_FILE_SHA256,
                    "formation_receipt_sha256": FIXED_P_FORMATION_RECEIPT_SHA256,
                },
                "runtime_binding": {
                    "binding_sha256": runtime_binding_sha,
                    "path_free_binding_sha256": stable_hash(runtime_binding),
                },
            }
        )
        result = _self_hashed(body, "result_sha256")
        _assert_aggregate_only(result)
        _write_json_exclusive(paths["result"], result, PUBLIC_MODE)
        return result
    except BaseException as exc:
        _persist_failure(
            path=paths["failure"],
            marker=marker,
            marker_file_sha256=marker_file_hash,
            exc=exc,
        )
        raise


def _prepare_formal_runtime(
    *,
    project_root: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> PreparedFormalRuntimeV2:
    root = project_root.resolve(strict=True)
    return prepare_formal_runtime_v2(
        project_root=root,
        attestation_receipt_path=root / OFFICIAL_ATTESTATION_RELATIVE_PATH,
        base_binding_receipt_path=root / OFFICIAL_BASE_RECEIPT_RELATIVE_PATH,
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--local-llm-model", required=True, type=Path)
    parser.add_argument("--local-embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.project_root.resolve(strict=True)
    program = load_exact_frozen_p(root)
    runtime = _prepare_formal_runtime(
        project_root=root,
        runtime_python=arguments.runtime_python,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    global _FORMAL_ENTRY_ACTIVE
    if _FORMAL_ENTRY_ACTIVE:
        raise FeverFixedPRunnerError("formal entry is already active")
    _FORMAL_ENTRY_ACTIVE = True
    try:
        result = run_canonical_formal(
            project_root=root,
            runtime=runtime,
            program=program,
        )
    finally:
        _FORMAL_ENTRY_ACTIVE = False
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_sha256": result["result_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "ACTION_ITEM_SCHEMA",
    "ACTION_PACK_SCHEMA",
    "ARM_IDS",
    "ActionItem",
    "ActionPack",
    "DOCUMENTS_PER_ITEM",
    "FIXED_P_PROGRAM_SHA256",
    "FeverFixedPRunnerError",
    "ITEM_COUNT",
    "LABEL_ITEM_SCHEMA",
    "LABEL_PACK_SCHEMA",
    "LabelItem",
    "LabelPack",
    "MeasurementOutcome",
    "OFFICIAL_CONCURRENCY_CAP",
    "OfficialRuntimeProtocol",
    "SentenceDocument",
    "TOP_K",
    "VERSION",
    "aggregate_result_body",
    "canonical_bytes",
    "exact_fixed_p",
    "load_action_pack",
    "load_exact_frozen_p",
    "load_label_pack",
    "run_measurement",
]


if __name__ == "__main__":  # pragma: no cover - formal CLI
    raise SystemExit(main())
