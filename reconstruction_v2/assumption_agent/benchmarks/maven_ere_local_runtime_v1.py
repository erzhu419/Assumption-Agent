"""Bound offline MiniLM, NLI, and official HippoRAG runtime for MAVEN-ERE."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks.maven_ere_nli_runtime_v1 import (
    MavenEreNLIWorkerPool,
    verify_maven_design,
)
from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    run_item_local_official_hipporag_v1,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasc_nli_v1 import binding as nli_binding
from replication_runtime.qasc_nli_v1.contract import MAXIMUM_PAIRS_PER_REQUEST, NLIPair
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


VERSION = "maven_ere_local_runtime_v1"
PREFLIGHT_SCHEMA = "maven_ere_local_runtime_preflight_v1"
BLOCK_ORDER = ("G_form", "A_form", "F_search", "A_hold", "M_search")
HIPPORAG_PHYSICAL_CAP = 2
LOCAL_TASK_PHYSICAL_CAP = 16
FORMAL_ROOT_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1")
HIPPORAG_STAGE_RELATIVE = FORMAL_ROOT_RELATIVE / "official_hipporag_item_work"

FAMILY_HYPOTHESES: Mapping[str, tuple[str, ...]] = {
    "CAUSAL": (
        "The event {A} caused the event {B}.",
        "The event {A} was a precondition for the event {B}.",
    ),
    "SUBEVENT": (
        "The event {B} was a subevent of the event {A}.",
        "The event {B} was part of the event {A}.",
    ),
    "TEMPORAL": (
        "The event {A} happened before the event {B}.",
        "The events {A} and {B} overlapped in time.",
        "The event {A} contained the event {B} in time.",
        "The events {A} and {B} happened simultaneously.",
        "The event {A} ended when the event {B} began.",
        "The event {A} began when the event {B} began.",
    ),
}
HYPOTHESIS_ROWS = tuple(
    (family, template)
    for family in core.FAMILY_ORDER
    for template in FAMILY_HYPOTHESES[family]
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MavenEreLocalRuntimeError(RuntimeError):
    """A private pack, local model, worker, or official adapter drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MavenEreLocalRuntimeError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class EventView:
    event_id: int
    event_type: str
    mentions: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ItemView:
    item_id: str
    common_query: str
    sentences: tuple[str, ...]
    events: tuple[EventView, ...]
    head_event: int
    tail_event: int
    generic_relations: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PreparedItem:
    view: ItemView
    item: core.ValidatedActionItem
    semantic_receipt_sha256: str


@dataclass(frozen=True)
class PreparedBlock:
    block: str
    items: tuple[PreparedItem, ...]
    preparation_sha256: str


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MavenEreLocalRuntimeError(f"{field} must be nonempty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MavenEreLocalRuntimeError(f"{field} is invalid Unicode") from exc
    return value


def _read_pack(path: str | Path, *, schema: str, block: str) -> Mapping[str, Any]:
    absolute = Path(path).absolute()
    if absolute.is_symlink() or not absolute.is_file() or absolute.stat().st_size > 512 * 1024 * 1024:
        raise MavenEreLocalRuntimeError("private pack is unavailable")
    try:
        raw = absolute.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreLocalRuntimeError("private pack is invalid") from exc
    if not isinstance(value, dict):
        raise MavenEreLocalRuntimeError("private pack root must be an object")
    body = dict(value)
    declared = body.pop("pack_sha256", None)
    if (
        value.get("schema") != schema
        or value.get("version") != "v1"
        or value.get("block") != block
        or declared != stable_hash(body)
    ):
        raise MavenEreLocalRuntimeError("private pack binding drifted")
    items = value.get("items")
    if not isinstance(items, list) or len(items) != value.get("item_count"):
        raise MavenEreLocalRuntimeError("private pack item count drifted")
    return value


def load_view_pack(path: str | Path, *, block: str) -> tuple[ItemView, ...]:
    if block not in BLOCK_ORDER:
        raise MavenEreLocalRuntimeError("block is invalid")
    pack = _read_pack(
        path,
        schema="maven_ere_g8_e1_action_view_pack_v1",
        block=block,
    )
    result: list[ItemView] = []
    seen: set[str] = set()
    exact_keys = {
        "common_query",
        "events",
        "generic_relations",
        "head_event",
        "item_id",
        "sentences",
        "tail_event",
    }
    for raw_item in pack["items"]:
        if not isinstance(raw_item, Mapping) or set(raw_item) != exact_keys:
            raise MavenEreLocalRuntimeError("action view item shape drifted")
        item_id = raw_item.get("item_id")
        if not isinstance(item_id, str) or not _SHA256.fullmatch(item_id) or item_id in seen:
            raise MavenEreLocalRuntimeError("private item ID is invalid")
        seen.add(item_id)
        raw_sentences = raw_item.get("sentences")
        if not isinstance(raw_sentences, list) or len(raw_sentences) < 6:
            raise MavenEreLocalRuntimeError("sentence list is invalid")
        sentences = tuple(
            _required_text(value, f"sentence {index}")
            for index, value in enumerate(raw_sentences)
        )
        raw_events = raw_item.get("events")
        if not isinstance(raw_events, list) or len(raw_events) < 2:
            raise MavenEreLocalRuntimeError("event sidecar is invalid")
        events: list[EventView] = []
        for event_index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, Mapping) or set(raw_event) != {
                "event_id",
                "event_type",
                "mentions",
            }:
                raise MavenEreLocalRuntimeError("event view shape drifted")
            if raw_event.get("event_id") != event_index:
                raise MavenEreLocalRuntimeError("event IDs are not contiguous")
            mentions_raw = raw_event.get("mentions")
            if not isinstance(mentions_raw, list) or not mentions_raw:
                raise MavenEreLocalRuntimeError("event mentions are invalid")
            mentions: list[tuple[str, int]] = []
            for raw_mention in mentions_raw:
                if not isinstance(raw_mention, Mapping) or set(raw_mention) != {
                    "sentence_ordinal",
                    "surface",
                }:
                    raise MavenEreLocalRuntimeError("mention view shape drifted")
                ordinal = raw_mention.get("sentence_ordinal")
                if (
                    isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or not 0 <= ordinal < len(sentences)
                ):
                    raise MavenEreLocalRuntimeError("mention ordinal is invalid")
                mentions.append(
                    (_required_text(raw_mention.get("surface"), "mention surface"), ordinal)
                )
            events.append(
                EventView(
                    event_index,
                    _required_text(raw_event.get("event_type"), "event type"),
                    tuple(mentions),
                )
            )
        head = raw_item.get("head_event")
        tail = raw_item.get("tail_event")
        if (
            isinstance(head, bool)
            or not isinstance(head, int)
            or isinstance(tail, bool)
            or not isinstance(tail, int)
            or head == tail
            or not 0 <= head < len(events)
            or not 0 <= tail < len(events)
        ):
            raise MavenEreLocalRuntimeError("query endpoints are invalid")
        relation_rows: list[tuple[int, int]] = []
        raw_relations = raw_item.get("generic_relations")
        if not isinstance(raw_relations, list):
            raise MavenEreLocalRuntimeError("generic relations are invalid")
        for raw_relation in raw_relations:
            if not isinstance(raw_relation, list) or len(raw_relation) != 2:
                raise MavenEreLocalRuntimeError("generic relation row is invalid")
            left, right = raw_relation
            if (
                isinstance(left, bool)
                or not isinstance(left, int)
                or isinstance(right, bool)
                or not isinstance(right, int)
                or not 0 <= left < right < len(events)
            ):
                raise MavenEreLocalRuntimeError("generic relation endpoints are invalid")
            relation_rows.append((left, right))
        if relation_rows != sorted(set(relation_rows)):
            raise MavenEreLocalRuntimeError("generic relations are not canonical")
        query = _required_text(raw_item.get("common_query"), "common query")
        result.append(
            ItemView(
                item_id=item_id,
                common_query=query,
                sentences=sentences,
                events=tuple(events),
                head_event=head,
                tail_event=tail,
                generic_relations=tuple(relation_rows),
            )
        )
    return tuple(result)


def _endpoint_alias(view: ItemView, event_id: int) -> str:
    aliases = core.canonical_aliases(
        tuple(surface for surface, _ordinal in view.events[event_id].mentions)
    )
    return aliases[0]


def fixed_hypotheses(view: ItemView) -> tuple[tuple[str, str], ...]:
    a = _endpoint_alias(view, view.head_event)
    b = _endpoint_alias(view, view.tail_event)
    return tuple((family, template.format(A=a, B=b)) for family, template in HYPOTHESIS_ROWS)


def nli_pairs(view: ItemView) -> tuple[NLIPair, ...]:
    hypotheses = fixed_hypotheses(view)
    return tuple(
        NLIPair(premise=sentence, hypothesis=hypothesis)
        for sentence in view.sentences
        for _family, hypothesis in hypotheses
    )


def collapse_nli_scores(
    view: ItemView, scores: Sequence[object]
) -> tuple[tuple[int, int, int], ...]:
    expected = len(view.sentences) * len(HYPOTHESIS_ROWS)
    if isinstance(scores, (str, bytes)) or len(scores) != expected:
        raise MavenEreLocalRuntimeError("NLI score vector length drifted")
    parsed: list[int] = []
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, int):
            raise MavenEreLocalRuntimeError("NLI score must be integer")
        parsed.append(value)
    family_positions = {
        family: tuple(
            index for index, (row_family, _template) in enumerate(HYPOTHESIS_ROWS)
            if row_family == family
        )
        for family in core.FAMILY_ORDER
    }
    result: list[tuple[int, int, int]] = []
    width = len(HYPOTHESIS_ROWS)
    for sentence_index in range(len(view.sentences)):
        offset = sentence_index * width
        result.append(
            tuple(
                max(parsed[offset + position] for position in family_positions[family])
                for family in core.FAMILY_ORDER
            )  # type: ignore[arg-type]
        )
    return tuple(result)


def _float32_hash(matrix: np.ndarray) -> str:
    values = np.asarray(matrix, dtype="<f4", order="C")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def prepare_block(
    *,
    block: str,
    views: Sequence[ItemView],
    encoder: minilm_binding.OfflineMiniLMEncoder,
    nli_pool: MavenEreNLIWorkerPool,
) -> PreparedBlock:
    if block not in BLOCK_ORDER or not views:
        raise MavenEreLocalRuntimeError("prepared block is invalid")
    if len({view.item_id for view in views}) != len(views):
        raise MavenEreLocalRuntimeError("prepared block item IDs are not unique")
    flat_texts: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    for view in views:
        start = len(flat_texts)
        flat_texts.extend((view.common_query, *view.sentences))
        slices[view.item_id] = (start, len(flat_texts))
    try:
        embedding_chunks = tuple(
            encoder.encode(
                tuple(flat_texts[start : start + minilm_binding.MAXIMUM_TEXTS_PER_CALL])
            )
            for start in range(0, len(flat_texts), minilm_binding.MAXIMUM_TEXTS_PER_CALL)
        )
        embedding_matrix = np.concatenate(embedding_chunks, axis=0)
        nli_batches: list[tuple[str, Sequence[NLIPair]]] = []
        nli_batch_keys: dict[str, list[str]] = {view.item_id: [] for view in views}
        for view in views:
            pairs = nli_pairs(view)
            for chunk_index, start in enumerate(
                range(0, len(pairs), MAXIMUM_PAIRS_PER_REQUEST)
            ):
                key = f"{view.item_id}:{chunk_index:08d}"
                nli_batch_keys[view.item_id].append(key)
                nli_batches.append(
                    (key, pairs[start : start + MAXIMUM_PAIRS_PER_REQUEST])
                )
        batch_score_map = nli_pool.score_items(tuple(nli_batches))
        score_map = {
            view.item_id: tuple(
                score
                for key in nli_batch_keys[view.item_id]
                for score in batch_score_map[key]
            )
            for view in views
        }
    except Exception as exc:
        raise MavenEreLocalRuntimeError("offline semantic preparation failed") from exc
    if embedding_matrix.shape != (len(flat_texts), minilm_binding.EMBEDDING_DIMENSION):
        raise MavenEreLocalRuntimeError("MiniLM output shape drifted")
    prepared: list[PreparedItem] = []
    for view in views:
        start, end = slices[view.item_id]
        rows = embedding_matrix[start:end]
        if rows.shape != (len(view.sentences) + 1, minilm_binding.EMBEDDING_DIMENSION):
            raise MavenEreLocalRuntimeError("item embedding slice drifted")
        nli_rows = collapse_nli_scores(view, score_map[view.item_id])
        events = tuple(
            core.Event(
                event.event_id,
                event.event_type,
                tuple(core.Mention(surface, ordinal) for surface, ordinal in event.mentions),
            )
            for event in view.events
        )
        item = core.validate_action_item(
            sentences=view.sentences,
            sentence_embeddings=rows[1:].tolist(),
            events=events,
            head_event=view.head_event,
            tail_event=view.tail_event,
            generic_relations=tuple(core.GenericRelation(*row) for row in view.generic_relations),
            common_query=view.common_query,
            query_embedding=rows[0].tolist(),
            sentence_family_nli_scores=nli_rows,
        )
        receipt = stable_hash(
            {
                "action_item_commitment": core.action_item_commitment(item),
                "embedding_float32_sha256": _float32_hash(rows),
                "item_id": view.item_id,
                "nli_score_sha256": stable_hash(nli_rows),
            }
        )
        prepared.append(PreparedItem(view, item, receipt))
    preparation = stable_hash(
        {
            "block": block,
            "items": [
                {
                    "item_id": row.view.item_id,
                    "semantic_receipt_sha256": row.semantic_receipt_sha256,
                }
                for row in prepared
            ],
        }
    )
    return PreparedBlock(block, tuple(prepared), preparation)


@dataclass(frozen=True)
class FormalRuntimeConfig:
    project: Path
    local_python: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path
    nli_asset_manifest: Path
    nli_model_root: Path
    hippo_runtime_python: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hippo_base_binding_receipt: Path
    hippo_attestation_receipt: Path
    hippo_stage_root: Path


def default_formal_runtime_config(project: str | Path) -> FormalRuntimeConfig:
    root = Path(project).resolve(strict=True)
    home = Path.home().absolute()
    return FormalRuntimeConfig(
        project=root,
        local_python=Path(sys.executable).resolve(strict=True),
        minilm_asset_manifest=root / minilm_binding.ASSET_RELATIVE_PATH,
        minilm_model_root=root / "artifacts/qasper_minilm_runtime_v1/model",
        nli_asset_manifest=root / nli_binding.ASSET_RELATIVE_PATH,
        nli_model_root=root / "artifacts/qasc_nli_runtime_v3/model",
        hippo_runtime_python=home / ".hr5/venv/bin/python",
        hippo_llm_model=home / ".hr5/models/smollm2-135m-instruct",
        hippo_embedding_model=(
            home
            / ".cache/huggingface/hub"
            / "models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots"
            / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        ),
        hippo_base_binding_receipt=(
            root / "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
        ),
        hippo_attestation_receipt=(
            root / "manifests/musique_official_hipporag_runtime_attestation_v3.json"
        ),
        hippo_stage_root=root / HIPPORAG_STAGE_RELATIVE,
    )


def preflight_formal_runtime_config(config: FormalRuntimeConfig) -> dict[str, Any]:
    if not isinstance(config, FormalRuntimeConfig):
        raise MavenEreLocalRuntimeError("runtime config type drifted")
    canonical = default_formal_runtime_config(config.project)
    if config != canonical:
        raise MavenEreLocalRuntimeError("runtime config is not canonical")
    if os.path.lexists(config.hippo_stage_root):
        raise MavenEreLocalRuntimeError("HippoRAG stage root already exists")
    try:
        minilm = minilm_binding.verify_runtime_binding(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )
        design = verify_maven_design(config.project)
        nli = nli_binding.verify_runtime_binding(
            asset_manifest_path=config.nli_asset_manifest,
            model_root=config.nli_model_root,
        )
        hippo = verify_formal_runtime_attestation_v3(
            project_root=config.project,
            attestation_receipt_path=config.hippo_attestation_receipt,
            base_binding_receipt_path=config.hippo_base_binding_receipt,
            runtime_python=config.hippo_runtime_python,
            local_llm_model=config.hippo_llm_model,
            local_embedding_model=config.hippo_embedding_model,
        )
    except Exception as exc:
        raise MavenEreLocalRuntimeError("offline runtime preflight failed") from exc
    return {
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
        "hipporag_runtime_attestation": dict(hippo),
        "minilm_runtime_binding": dict(minilm),
        "model_inference_calls": 0,
        "nli_design_binding": dict(design),
        "nli_runtime_binding": dict(nli),
        "schema": PREFLIGHT_SCHEMA,
        "version": VERSION,
    }


class OfficialHippoGateway:
    def __init__(self, config: FormalRuntimeConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._counter = 0
        self._prepared: set[str] = set()

    def prepare_blocks(self, blocks: Sequence[str]) -> None:
        if not blocks or any(block not in BLOCK_ORDER for block in blocks):
            raise MavenEreLocalRuntimeError("HippoRAG block list is invalid")
        if not self.config.hippo_stage_root.exists():
            self.config.hippo_stage_root.mkdir(mode=0o700, parents=True)
        for block in BLOCK_ORDER:
            if block not in blocks:
                continue
            path = self.config.hippo_stage_root / block
            if path.exists():
                raise MavenEreLocalRuntimeError("HippoRAG block stage already exists")
            path.mkdir(mode=0o700)
            self._prepared.add(block)

    def retrieve(self, *, block: str, view: ItemView) -> tuple[int, int, int]:
        with self._lock:
            if block not in self._prepared:
                raise MavenEreLocalRuntimeError("HippoRAG block was not prepared")
            self._counter += 1
            work = self.config.hippo_stage_root / block / (
                f"{view.item_id}.{self._counter:08d}.work"
            )
        try:
            top5 = run_item_local_official_hipporag_v1(
                query=view.common_query,
                sentence_texts=view.sentences,
                runtime_python=self.config.hippo_runtime_python,
                local_llm_model=self.config.hippo_llm_model,
                local_embedding_model=self.config.hippo_embedding_model,
                base_binding_receipt_path=self.config.hippo_base_binding_receipt,
                attestation_receipt_path=self.config.hippo_attestation_receipt,
                work_root=work,
                timeout_seconds=900,
            )
        except Exception as exc:
            raise MavenEreLocalRuntimeError("official item-local HippoRAG failed") from exc
        if os.path.lexists(work) or len(top5) != 5 or len(set(top5)) != 5:
            raise MavenEreLocalRuntimeError("official HippoRAG cleanup or output drifted")
        return tuple(top5[:3])  # type: ignore[return-value]


class RuntimeBundle(AbstractContextManager["RuntimeBundle"]):
    def __init__(self, config: FormalRuntimeConfig) -> None:
        self.config = config
        self.encoder: minilm_binding.OfflineMiniLMEncoder | None = None
        self.nli: MavenEreNLIWorkerPool | None = None
        self.hippo: OfficialHippoGateway | None = None
        self.preflight_receipt: Mapping[str, Any] | None = None
        self._stack: ExitStack | None = None

    def __enter__(self) -> "RuntimeBundle":
        self.preflight_receipt = preflight_formal_runtime_config(self.config)
        stack = ExitStack()
        try:
            self.encoder = minilm_binding.OfflineMiniLMEncoder(
                asset_manifest_path=self.config.minilm_asset_manifest,
                model_root=self.config.minilm_model_root,
            )
            self.nli = stack.enter_context(
                MavenEreNLIWorkerPool(
                    self.config.nli_model_root,
                    project_root=self.config.project,
                    runtime_python=self.config.local_python,
                )
            )
            self.hippo = OfficialHippoGateway(self.config)
            self._stack = stack
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, *_exc: object) -> None:
        if self._stack is not None:
            self._stack.close()

    def prepare_block(self, block: str, views: Sequence[ItemView]) -> PreparedBlock:
        if self.encoder is None or self.nli is None:
            raise MavenEreLocalRuntimeError("semantic runtime is not open")
        return prepare_block(
            block=block,
            views=views,
            encoder=self.encoder,
            nli_pool=self.nli,
        )


__all__ = [
    "BLOCK_ORDER",
    "FAMILY_HYPOTHESES",
    "FormalRuntimeConfig",
    "HIPPORAG_PHYSICAL_CAP",
    "HYPOTHESIS_ROWS",
    "ItemView",
    "LOCAL_TASK_PHYSICAL_CAP",
    "MavenEreLocalRuntimeError",
    "OfficialHippoGateway",
    "PreparedBlock",
    "PreparedItem",
    "RuntimeBundle",
    "collapse_nli_scores",
    "default_formal_runtime_config",
    "fixed_hypotheses",
    "load_view_pack",
    "nli_pairs",
    "preflight_formal_runtime_config",
    "prepare_block",
    "stable_hash",
]
