"""Offline semantic and item-local HippoRAG runtime for ERASER EI.

The module owns no ERASER reader and no evaluator.  It accepts already
materialized, outcome-free text views, embeds every available item in one
cross-block MiniLM call, and prepares only the immutable sentence units,
semantic tensor, graph, and normalized sentence embeddings.  R0/R7 execution
is deliberately deferred to :func:`execute_agent`, while :func:`execute_raw`
performs an independent R0 execution.  This preserves the controller's
``3 * n`` eager-submission barrier.

Pairwise redundancy is measured only after both Agent actions are fixed.  The
runtime quantizes the canonical union of their two top-five pair sets (at most
twenty pairs), never a quadratic all-sentence matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any

import numpy as np

from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_operator_v1 as operator,
)
from assumption_agent.benchmarks import semantic_assignment_operator_v1 as semantic_runtime
from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    adapter as hippo_adapter,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasper_minilm_v1 import binding as qasper_binding


VERSION = "eraser_evidence_inference_local_runtime_v1"
PREFLIGHT_SCHEMA = f"{VERSION}_preflight"
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")

FORMAL_ROOT_RELATIVE = Path(
    "artifacts/eraser_evidence_inference_r7_e3_formal_v1"
)
HIPPORAG_STAGE_PARENT_RELATIVE = (
    FORMAL_ROOT_RELATIVE / "official_hipporag_item_stage_parent"
)
# Narrow compatibility name for controllers that call every runtime output a
# stage root.  Both names denote the same private parent; there is no shared
# item index beneath it.
HIPPORAG_STAGE_RELATIVE = HIPPORAG_STAGE_PARENT_RELATIVE

MINILM_ASSET_RELATIVE = Path(
    "manifests/semantic_assignment_minilm_runtime_asset_v1.json"
)
MINILM_ASSET_MANIFEST_SHA256 = (
    "837180aeb37eaaae2ebf108d2e3e2cb381db4d80152f75ff1da178ea5e144e88"
)
MINILM_ASSET_FILE_SHA256 = (
    "035c35e6a2f6e11a20c9958a3a43b2418bc45e7d60c1943f758d31df342063ac"
)
MINILM_SNAPSHOT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
HIPPORAG_ATTESTATION_RECEIPT_SHA256 = (
    "23996f9f41f494e2fd032b285039ec9420f6a893c24081e59c1ec79f229c2c60"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_Encoder = Callable[[Sequence[str]], object]


class EraserEvidenceInferenceLocalRuntimeError(RuntimeError):
    """A frozen path, exact text view, embedding, action, or stage drifted."""


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} is not a lowercase SHA-256"
        )
    return value


def _exact_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
    ):
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} is not exact nonempty text"
        )
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ItemTextView:
    """One exact outcome-free ERASER query/ICO/article view.

    No normalization is performed.  ``sentence_texts`` is exactly the
    single-ASCII-space projection of each official token tuple.
    """

    item_commitment_sha256: str
    query: str
    intervention: str
    comparator: str
    outcome: str
    official_tokenized_sentences: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, "item commitment")
        for value, field in (
            (self.query, "query"),
            (self.intervention, "intervention"),
            (self.comparator, "comparator"),
            (self.outcome, "outcome"),
        ):
            _exact_text(value, field)
        sentences = self.official_tokenized_sentences
        if not isinstance(sentences, tuple) or len(sentences) < operator.TOP_K:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "official sentence registry must contain at least five tuples"
            )
        for sentence in sentences:
            if not isinstance(sentence, tuple) or not sentence:
                raise EraserEvidenceInferenceLocalRuntimeError(
                    "official sentence token tuple is empty or mutable"
                )
            for token in sentence:
                if not isinstance(token, str) or token == "" or "\x00" in token:
                    raise EraserEvidenceInferenceLocalRuntimeError(
                        "official sentence contains an invalid exact token"
                    )

    @property
    def sentence_texts(self) -> tuple[str, ...]:
        return tuple(" ".join(tokens) for tokens in self.official_tokenized_sentences)

    @property
    def sentence_count(self) -> int:
        return len(self.official_tokenized_sentences)


def _build_units(view: ItemTextView) -> tuple[operator.SentenceUnit, ...]:
    rows: list[operator.SentenceUnit] = []
    start = 0
    for ordinal, (tokens, text) in enumerate(
        zip(view.official_tokenized_sentences, view.sentence_texts)
    ):
        end = start + len(tokens)
        rows.append(
            operator.SentenceUnit(
                sentence_ordinal=ordinal,
                start_token=start,
                end_token=end,
                sentence_sha256=_text_sha256(text),
            )
        )
        start = end
    return tuple(rows)


def _validate_normalized_matrix(
    value: object, *, expected_rows: int, field: str
) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} is not a float32 matrix"
        ) from exc
    if matrix.shape != (expected_rows, semantic_runtime.EMBEDDING_DIMENSION):
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} has the wrong shape"
        )
    if not np.isfinite(matrix).all():
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} contains a non-finite value"
        )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} is not L2-normalized"
        )
    return matrix


def _quantized_cosine(left: object, right: object, field: str) -> int:
    try:
        value = qasper_binding.quantized_cosine_similarity(left, right)
    except Exception as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} quantization failed"
        ) from exc
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -operator.INTEGER_SCALE <= value <= operator.INTEGER_SCALE
    ):
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} quantized cosine is outside the frozen range"
        )
    return value


@dataclass(frozen=True, eq=False)
class PreparedItemArtifact:
    """Label-free semantic preparation; it contains no executed action."""

    block: str
    view: ItemTextView
    units: tuple[operator.SentenceUnit, ...]
    semantic_tensor: operator.QuerySemanticTensor
    graph: operator.QueryAnchoredSentenceGraph
    sentence_embeddings: np.ndarray

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ORDER or not isinstance(self.view, ItemTextView):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared item block or text view is invalid"
            )
        expected_units = _build_units(self.view)
        if self.units != expected_units:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared sentence units drifted from exact official tokens"
            )
        try:
            operator.verify_query_anchored_graph(
                self.graph, self.semantic_tensor
            )
        except operator.EraserR7OperatorError as exc:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared graph/tensor verification failed"
            ) from exc
        if self.graph.units != self.units:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared graph uses a different sentence registry"
            )
        if (
            not isinstance(self.sentence_embeddings, np.ndarray)
            or self.sentence_embeddings.dtype != np.dtype(np.float32)
            or self.sentence_embeddings.shape
            != (self.view.sentence_count, semantic_runtime.EMBEDDING_DIMENSION)
            or self.sentence_embeddings.flags.writeable
            or not np.isfinite(self.sentence_embeddings).all()
            or not np.allclose(
                np.linalg.norm(self.sentence_embeddings, axis=1),
                1.0,
                rtol=0.0,
                atol=1e-5,
            )
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared sentence embeddings are not immutable normalized float32"
            )

    @property
    def item_commitment_sha256(self) -> str:
        return self.view.item_commitment_sha256

    @property
    def sentence_count(self) -> int:
        return self.view.sentence_count

    def binding_payload(self) -> dict[str, object]:
        """Return the content-free JSON binding; text/embeddings stay private."""

        return {
            "schema": f"{VERSION}_prepared_item_binding",
            "version": VERSION,
            "status": "semantic_preparation_only_no_action",
            "block": self.block,
            "item_commitment_sha256": self.item_commitment_sha256,
            "sentence_count": self.sentence_count,
            "semantic_tensor_sha256": self.semantic_tensor.tensor_sha256,
            "graph_sha256": self.graph.graph_sha256,
            "action_execution_count": 0,
            "exact_text_or_embedding_persisted": False,
        }


@dataclass(frozen=True)
class PreparedBatchArtifact:
    """One cross-block encoder call and its canonically ordered item outputs."""

    items: tuple[PreparedItemArtifact, ...]
    encoded_text_count: int
    encoder_call_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or not self.items:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared batch is empty or mutable"
            )
        if self.encoder_call_count != 1 or self.encoded_text_count != sum(
            4 + item.sentence_count for item in self.items
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared batch encoder accounting drifted"
            )
        commitments = [item.item_commitment_sha256 for item in self.items]
        if len(commitments) != len(set(commitments)):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared batch item commitment is duplicated"
            )
        block_positions = [BLOCK_ORDER.index(item.block) for item in self.items]
        if block_positions != sorted(block_positions):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "prepared batch block order is noncanonical"
            )

    def items_for_block(self, block: str) -> tuple[PreparedItemArtifact, ...]:
        if block not in BLOCK_ORDER:
            raise EraserEvidenceInferenceLocalRuntimeError("block is invalid")
        return tuple(item for item in self.items if item.block == block)

    def binding_payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_prepared_batch_binding",
            "version": VERSION,
            "status": "one_cross_block_encode_no_action",
            "encoder_call_count": self.encoder_call_count,
            "encoded_text_count": self.encoded_text_count,
            "item_count": len(self.items),
            "items": [item.binding_payload() for item in self.items],
            "action_execution_count": 0,
            "exact_text_or_embedding_persisted": False,
        }


def _canonical_block_items(
    items_by_block: Mapping[str, Sequence[ItemTextView]],
) -> tuple[tuple[str, ItemTextView], ...]:
    if not isinstance(items_by_block, Mapping) or not items_by_block:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "cross-block item registry is empty or invalid"
        )
    if any(block not in BLOCK_ORDER for block in items_by_block):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "cross-block item registry contains an invalid block"
        )
    rows: list[tuple[str, ItemTextView]] = []
    for block in BLOCK_ORDER:
        if block not in items_by_block:
            continue
        values = items_by_block[block]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "block item registry is not a sequence"
            )
        block_rows = tuple(values)
        if not block_rows or any(not isinstance(row, ItemTextView) for row in block_rows):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "block item registry is empty or contains a wrong view type"
            )
        rows.extend((block, row) for row in block_rows)
    commitments = [row.item_commitment_sha256 for _block, row in rows]
    if len(commitments) != len(set(commitments)):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "cross-block item commitment is duplicated"
        )
    return tuple(rows)


def prepare_semantic_batch(
    *,
    items_by_block: Mapping[str, Sequence[ItemTextView]],
    encoder: _Encoder,
) -> PreparedBatchArtifact:
    """Embed all supplied blocks once and prepare no R0/R7 action."""

    if not callable(encoder):
        raise EraserEvidenceInferenceLocalRuntimeError("MiniLM encoder is not callable")
    rows = _canonical_block_items(items_by_block)
    schedule: list[str] = []
    offsets: list[tuple[int, int]] = []
    for _block, view in rows:
        start = len(schedule)
        schedule.extend(
            (
                view.query,
                view.intervention,
                view.comparator,
                view.outcome,
                *view.sentence_texts,
            )
        )
        offsets.append((start, len(schedule)))
    try:
        encoded = encoder(tuple(schedule))
    except Exception as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "cross-block offline MiniLM encode failed"
        ) from exc
    matrix = _validate_normalized_matrix(
        encoded, expected_rows=len(schedule), field="cross-block embedding matrix"
    )

    prepared: list[PreparedItemArtifact] = []
    try:
        for (block, view), (start, stop) in zip(rows, offsets):
            item_matrix = matrix[start:stop]
            sentence_matrix = item_matrix[4:]
            facets = operator.make_official_ico_facets(
                intervention_sha256=_text_sha256(view.intervention),
                comparator_sha256=_text_sha256(view.comparator),
                outcome_sha256=_text_sha256(view.outcome),
            )
            dense = tuple(
                _quantized_cosine(
                    item_matrix[0], sentence, "full-query sentence"
                )
                for sentence in sentence_matrix
            )
            facet_rows = tuple(
                tuple(
                    _quantized_cosine(
                        item_matrix[facet_i + 1],
                        sentence,
                        f"{operator.FACET_TYPES[facet_i]} sentence",
                    )
                    for sentence in sentence_matrix
                )
                for facet_i in range(len(operator.FACET_TYPES))
            )
            tensor = operator.make_query_semantic_tensor(
                query_sha256=_text_sha256(view.query),
                facets=facets,
                facet_similarity_ints=facet_rows,
                dense_relevance_ints=dense,
            )
            units = _build_units(view)
            graph = operator.build_query_anchored_graph(
                units=units, semantic_tensor=tensor
            )
            retained = np.array(sentence_matrix, dtype=np.float32, copy=True, order="C")
            retained.setflags(write=False)
            prepared.append(
                PreparedItemArtifact(
                    block=block,
                    view=view,
                    units=units,
                    semantic_tensor=tensor,
                    graph=graph,
                    sentence_embeddings=retained,
                )
            )
    except EraserEvidenceInferenceLocalRuntimeError:
        raise
    except (operator.EraserR7OperatorError, ValueError, TypeError) as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "semantic artifact construction failed"
        ) from exc
    return PreparedBatchArtifact(
        items=tuple(prepared),
        encoded_text_count=len(schedule),
    )


# The explicit name is useful at call sites that treat preparation as an item
# batch operation rather than a semantic-runtime operation.
prepare_item_batch = prepare_semantic_batch


def _canonical_pair_union(
    r0_top5: Sequence[int], r7_top5: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    def pairs(selected: Sequence[int]) -> tuple[tuple[int, int], ...]:
        values = tuple(selected)
        if (
            len(values) != operator.TOP_K
            or len(set(values)) != operator.TOP_K
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "action output is not an exact top five"
            )
        return tuple(combinations(sorted(values), 2))

    return tuple(sorted(set(pairs(r0_top5)) | set(pairs(r7_top5))))


@dataclass(frozen=True)
class AgentExecutionArtifact:
    """Two independently hashed actions plus their selected pair measurements."""

    item_commitment_sha256: str
    graph_sha256: str
    semantic_tensor_sha256: str
    r0_action: operator.ActionTrace
    r7_action: operator.ActionTrace
    pair_rows: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        for value, field in (
            (self.item_commitment_sha256, "Agent item commitment"),
            (self.graph_sha256, "Agent graph hash"),
            (self.semantic_tensor_sha256, "Agent tensor hash"),
        ):
            _require_sha256(value, field)
        if (
            not isinstance(self.r0_action, operator.ActionTrace)
            or not isinstance(self.r7_action, operator.ActionTrace)
            or self.r0_action.recipe_id != operator.RECIPE_IDS[0]
            or self.r7_action.recipe_id != operator.RECIPE_IDS[1]
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "Agent action registry drifted"
            )
        try:
            operator.verify_action_trace(self.r0_action)
            operator.verify_action_trace(self.r7_action)
        except operator.EraserR7OperatorError as exc:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "Agent action trace verification failed"
            ) from exc
        expected_pairs = _canonical_pair_union(
            self.r0_action.output_top5, self.r7_action.output_top5
        )
        if (
            not isinstance(self.pair_rows, tuple)
            or len(self.pair_rows) > 20
            or tuple(
                (row[0], row[1])
                for row in self.pair_rows
                if isinstance(row, tuple) and len(row) == 3
            )
            != expected_pairs
            or len(self.pair_rows) != len(expected_pairs)
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "Agent selected-pair registry is incomplete or noncanonical"
            )
        for left, right, value in self.pair_rows:
            if (
                type(left) is not int
                or type(right) is not int
                or type(value) is not int
                or not 0 <= left < right
                or not -operator.INTEGER_SCALE <= value <= operator.INTEGER_SCALE
            ):
                raise EraserEvidenceInferenceLocalRuntimeError(
                    "Agent selected-pair row is malformed"
                )

    @property
    def selected_pair_rows(self) -> tuple[tuple[int, int, int], ...]:
        return self.pair_rows

    def payload(self) -> dict[str, object]:
        """Return the exact JSON-safe Agent result without private vectors."""

        return {
            "schema": f"{VERSION}_agent_execution",
            "version": VERSION,
            "item_commitment_sha256": self.item_commitment_sha256,
            "graph_sha256": self.graph_sha256,
            "semantic_tensor_sha256": self.semantic_tensor_sha256,
            "r0_action_trace_sha256": self.r0_action.trace_sha256,
            "r0_operator_behavior_sha256": self.r0_action.behavior_sha256,
            "r0_top5": list(self.r0_action.output_top5),
            "r7_action_trace_sha256": self.r7_action.trace_sha256,
            "r7_operator_behavior_sha256": self.r7_action.behavior_sha256,
            "r7_top5": list(self.r7_action.output_top5),
            "selected_pair_rows": [list(row) for row in self.pair_rows],
            "selected_pair_count": len(self.pair_rows),
            "full_square_pair_scan_performed": False,
        }


@dataclass(frozen=True)
class RawExecutionArtifact:
    """An independent R0/RAW action result for one logical RAW future."""

    item_commitment_sha256: str
    graph_sha256: str
    semantic_tensor_sha256: str
    r0_action: operator.ActionTrace

    def __post_init__(self) -> None:
        for value, field in (
            (self.item_commitment_sha256, "RAW item commitment"),
            (self.graph_sha256, "RAW graph hash"),
            (self.semantic_tensor_sha256, "RAW tensor hash"),
        ):
            _require_sha256(value, field)
        if (
            not isinstance(self.r0_action, operator.ActionTrace)
            or self.r0_action.recipe_id != operator.RECIPE_IDS[0]
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "RAW action registry drifted"
            )
        try:
            operator.verify_action_trace(self.r0_action)
        except operator.EraserR7OperatorError as exc:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "RAW action trace verification failed"
            ) from exc

    @property
    def top5(self) -> tuple[int, int, int, int, int]:
        return self.r0_action.output_top5

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_raw_execution",
            "version": VERSION,
            "item_commitment_sha256": self.item_commitment_sha256,
            "graph_sha256": self.graph_sha256,
            "semantic_tensor_sha256": self.semantic_tensor_sha256,
            "r0_action_trace_sha256": self.r0_action.trace_sha256,
            "r0_operator_behavior_sha256": self.r0_action.behavior_sha256,
            "top5": list(self.top5),
            "independent_r0_execution": True,
        }


def execute_agent(prepared: PreparedItemArtifact) -> AgentExecutionArtifact:
    """Run R0/R7 and at most twenty action-induced pair measurements."""

    if not isinstance(prepared, PreparedItemArtifact):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "Agent input is not a prepared item"
        )
    try:
        r0_action, r7_action = operator.run_all_actions(
            graph=prepared.graph,
            semantic_tensor=prepared.semantic_tensor,
        )
        pairs = _canonical_pair_union(
            r0_action.output_top5, r7_action.output_top5
        )
        pair_rows = tuple(
            (
                left,
                right,
                _quantized_cosine(
                    prepared.sentence_embeddings[left],
                    prepared.sentence_embeddings[right],
                    "selected sentence pair",
                ),
            )
            for left, right in pairs
        )
    except EraserEvidenceInferenceLocalRuntimeError:
        raise
    except operator.EraserR7OperatorError as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "Agent R0/R7 execution failed"
        ) from exc
    return AgentExecutionArtifact(
        item_commitment_sha256=prepared.item_commitment_sha256,
        graph_sha256=prepared.graph.graph_sha256,
        semantic_tensor_sha256=prepared.semantic_tensor.tensor_sha256,
        r0_action=r0_action,
        r7_action=r7_action,
        pair_rows=pair_rows,
    )


def execute_raw(prepared: PreparedItemArtifact) -> RawExecutionArtifact:
    """Run an independent R0 action; never reuse an Agent future's action."""

    if not isinstance(prepared, PreparedItemArtifact):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "RAW input is not a prepared item"
        )
    try:
        action = operator.run_action(
            recipe_id=operator.RECIPE_IDS[0],
            graph=prepared.graph,
            semantic_tensor=prepared.semantic_tensor,
        )
    except operator.EraserR7OperatorError as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "independent RAW/R0 execution failed"
        ) from exc
    return RawExecutionArtifact(
        item_commitment_sha256=prepared.item_commitment_sha256,
        graph_sha256=prepared.graph.graph_sha256,
        semantic_tensor_sha256=prepared.semantic_tensor.tensor_sha256,
        r0_action=action,
    )


@dataclass(frozen=True)
class FormalRuntimeConfig:
    """The sole authorized local path binding for the formal lifecycle."""

    project: Path
    minilm_asset_manifest: Path
    minilm_snapshot_root: Path
    hippo_runtime_python: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hippo_base_binding_receipt: Path
    hippo_attestation_receipt: Path
    hippo_stage_parent_root: Path


def _canonical_project(project: str | Path) -> Path:
    try:
        lexical = Path(project).expanduser().absolute()
        if lexical.is_symlink():
            raise EraserEvidenceInferenceLocalRuntimeError(
                "project root is a symlink"
            )
        root = lexical.resolve(strict=True)
    except EraserEvidenceInferenceLocalRuntimeError:
        raise
    except (OSError, RuntimeError) as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "project root is unavailable"
        ) from exc
    if not root.is_dir():
        raise EraserEvidenceInferenceLocalRuntimeError(
            "project root is not a directory"
        )
    return root


def default_formal_runtime_config(project: str | Path) -> FormalRuntimeConfig:
    """Return exact frozen paths without verifying or loading a model."""

    root = _canonical_project(project)
    home = Path.home().expanduser().absolute()
    return FormalRuntimeConfig(
        project=root,
        minilm_asset_manifest=root / MINILM_ASSET_RELATIVE,
        minilm_snapshot_root=(
            home
            / ".cache/huggingface/hub"
            / "models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots"
            / MINILM_SNAPSHOT_REVISION
        ),
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
        hippo_stage_parent_root=root / HIPPORAG_STAGE_PARENT_RELATIVE,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_minilm_asset(path: Path) -> dict[str, Any]:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise EraserEvidenceInferenceLocalRuntimeError(
                "MiniLM asset path contains a symlink"
            )
    if (
        not absolute.is_file()
        or absolute.stat().st_size > 256 * 1024
        or _sha256_file(absolute) != MINILM_ASSET_FILE_SHA256
    ):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "frozen MiniLM asset manifest drifted"
        )
    try:
        payload = json.loads(absolute.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "frozen MiniLM asset manifest is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "frozen MiniLM asset manifest is not an object"
        )
    return payload


def _require_receipt(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} verifier returned no receipt"
        )
    return dict(value)


def _assert_output_path(config: FormalRuntimeConfig, project: Path) -> None:
    path = config.hippo_stage_parent_root.absolute()
    try:
        relative = path.relative_to(project)
    except ValueError as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "HippoRAG stage parent escaped the project"
        ) from exc
    cursor = project
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG stage parent contains a symlink component"
            )
        if cursor.exists() and not cursor.is_dir():
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG stage parent contains a nondirectory component"
            )
    if os.path.lexists(path):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "HippoRAG formal stage parent already exists"
        )


def preflight_formal_runtime_config(
    config: FormalRuntimeConfig,
) -> dict[str, Any]:
    """Hash both runtimes with zero model, benchmark, or network calls."""

    if not isinstance(config, FormalRuntimeConfig):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "formal runtime config type drifted"
        )
    project = _canonical_project(config.project)
    if config != default_formal_runtime_config(project):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "formal runtime config is not canonical"
        )
    _assert_output_path(config, project)
    try:
        asset = _load_minilm_asset(config.minilm_asset_manifest)
        minilm = _require_receipt(
            semantic_runtime.verify_runtime_asset(
                asset, snapshot_root=config.minilm_snapshot_root
            ),
            "MiniLM runtime",
        )
        hippo = _require_receipt(
            verify_formal_runtime_attestation_v3(
                project_root=project,
                attestation_receipt_path=config.hippo_attestation_receipt,
                base_binding_receipt_path=config.hippo_base_binding_receipt,
                runtime_python=config.hippo_runtime_python,
                local_llm_model=config.hippo_llm_model,
                local_embedding_model=config.hippo_embedding_model,
            ),
            "official HippoRAG runtime",
        )
    except EraserEvidenceInferenceLocalRuntimeError:
        raise
    except Exception as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "offline runtime preflight failed"
        ) from exc
    if (
        minilm.get("runtime_asset_manifest_hash")
        != MINILM_ASSET_MANIFEST_SHA256
        or minilm.get("snapshot_revision") != MINILM_SNAPSHOT_REVISION
    ):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "MiniLM runtime receipt drifted from the ERASER design"
        )
    if (
        hippo.get("attestation_receipt_sha256")
        != HIPPORAG_ATTESTATION_RECEIPT_SHA256
    ):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "HippoRAG attestation drifted from the ERASER design"
        )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "version": VERSION,
        "semantic_assignment_minilm_runtime_asset": minilm,
        "official_hipporag_runtime_attestation": hippo,
        "model_inference_calls": 0,
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
    }


def _inspect_private_directory(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} cannot be inspected"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise EraserEvidenceInferenceLocalRuntimeError(
            f"{field} is not a private directory"
        )


def _create_parent_chain(project: Path, parent: Path) -> None:
    try:
        relative = parent.absolute().relative_to(project)
    except ValueError as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "HippoRAG stage parent escaped the project"
        ) from exc
    cursor = project
    for component in relative.parts:
        cursor = cursor / component
        if os.path.lexists(cursor):
            if cursor.is_symlink() or not cursor.is_dir():
                raise EraserEvidenceInferenceLocalRuntimeError(
                    "HippoRAG stage path contains an unsafe component"
                )
            continue
        try:
            os.mkdir(cursor, 0o700)
        except OSError as exc:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG stage parent chain cannot be created"
            ) from exc


@dataclass(frozen=True)
class HippoExecutionArtifact:
    """Ordinal-only item-local HippoRAG result safe for JSON persistence."""

    block: str
    item_commitment_sha256: str
    top5: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ORDER:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG result block is invalid"
            )
        _require_sha256(self.item_commitment_sha256, "HippoRAG item commitment")
        if (
            not isinstance(self.top5, tuple)
            or len(self.top5) != operator.TOP_K
            or len(set(self.top5)) != operator.TOP_K
            or any(type(value) is not int or value < 0 for value in self.top5)
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG result is not an ordinal top five"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_hipporag_execution",
            "version": VERSION,
            "block": self.block,
            "item_commitment_sha256": self.item_commitment_sha256,
            "top5": list(self.top5),
            "item_local_fresh_index": True,
            "exact_text_or_index_persisted": False,
        }


class OfficialHippoGateway:
    """Thread-safe allocator for fresh item-local official HippoRAG roots."""

    def __init__(self, config: FormalRuntimeConfig) -> None:
        if not isinstance(config, FormalRuntimeConfig):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG gateway config type drifted"
            )
        project = _canonical_project(config.project)
        if config != default_formal_runtime_config(project):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG gateway config is not canonical"
            )
        self.config = config
        self._lock = threading.Lock()
        self._root_ready = False
        self._prepared_blocks: set[str] = set()
        self._call_index = 0

    def prepare_blocks(self, blocks: Sequence[str]) -> tuple[Path, ...]:
        """Prebuild exact-mode private parents before any item future runs."""

        if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG block preparation registry is invalid"
            )
        requested = tuple(blocks)
        if (
            not requested
            or len(requested) != len(set(requested))
            or any(block not in BLOCK_ORDER for block in requested)
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG block preparation registry drifted"
            )
        canonical = tuple(block for block in BLOCK_ORDER if block in requested)
        with self._lock:
            root = self.config.hippo_stage_parent_root
            if not self._root_ready:
                if os.path.lexists(root):
                    raise EraserEvidenceInferenceLocalRuntimeError(
                        "HippoRAG stage parent was not freshly allocated"
                    )
                _create_parent_chain(self.config.project, root.parent)
                try:
                    os.mkdir(root, 0o700)
                except OSError as exc:
                    raise EraserEvidenceInferenceLocalRuntimeError(
                        "HippoRAG private stage parent cannot be created"
                    ) from exc
                _inspect_private_directory(root, "HippoRAG stage parent")
                self._root_ready = True
            else:
                _inspect_private_directory(root, "HippoRAG stage parent")
            result: list[Path] = []
            for block in canonical:
                parent = root / block
                if block not in self._prepared_blocks:
                    if os.path.lexists(parent):
                        raise EraserEvidenceInferenceLocalRuntimeError(
                            "HippoRAG block parent was not freshly allocated"
                        )
                    try:
                        os.mkdir(parent, 0o700)
                    except OSError as exc:
                        raise EraserEvidenceInferenceLocalRuntimeError(
                            "HippoRAG block parent cannot be created"
                        ) from exc
                    self._prepared_blocks.add(block)
                _inspect_private_directory(parent, "HippoRAG block parent")
                result.append(parent)
            return tuple(result)

    def retrieve(
        self,
        *,
        block: str,
        view: ItemTextView,
        timeout_seconds: int = 900,
    ) -> tuple[int, ...]:
        if block not in BLOCK_ORDER or not isinstance(view, ItemTextView):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG item block or exact text view is invalid"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 3600
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "HippoRAG timeout is outside the frozen bound"
            )
        with self._lock:
            if block not in self._prepared_blocks:
                raise EraserEvidenceInferenceLocalRuntimeError(
                    "HippoRAG block parent was not prebuilt"
                )
            parent = self.config.hippo_stage_parent_root / block
            _inspect_private_directory(parent, "HippoRAG block parent")
            self._call_index += 1
            work_root = parent / (
                f"{view.item_commitment_sha256}.{self._call_index:08d}.work"
            )
            if os.path.lexists(work_root):
                raise EraserEvidenceInferenceLocalRuntimeError(
                    "HippoRAG per-item work root is not fresh"
                )
        try:
            result = hippo_adapter.run_item_local_official_hipporag_v1(
                query=view.query,
                sentence_texts=view.sentence_texts,
                runtime_python=self.config.hippo_runtime_python,
                local_llm_model=self.config.hippo_llm_model,
                local_embedding_model=self.config.hippo_embedding_model,
                base_binding_receipt_path=self.config.hippo_base_binding_receipt,
                attestation_receipt_path=self.config.hippo_attestation_receipt,
                work_root=work_root,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            raise EraserEvidenceInferenceLocalRuntimeError(
                "item-local official HippoRAG execution failed"
            ) from exc
        if os.path.lexists(work_root):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "item-local HippoRAG adapter did not destroy its work root"
            )
        if (
            not isinstance(result, tuple)
            or len(result) != operator.TOP_K
            or len(set(result)) != operator.TOP_K
            or any(
                type(value) is not int or not 0 <= value < view.sentence_count
                for value in result
            )
        ):
            raise EraserEvidenceInferenceLocalRuntimeError(
                "item-local HippoRAG output is not an in-corpus top five"
            )
        return result

    retrieve_item = retrieve

    def retrieve_artifact(
        self,
        *,
        block: str,
        view: ItemTextView,
        timeout_seconds: int = 900,
    ) -> HippoExecutionArtifact:
        return HippoExecutionArtifact(
            block=block,
            item_commitment_sha256=view.item_commitment_sha256,
            top5=self.retrieve(
                block=block,
                view=view,
                timeout_seconds=timeout_seconds,
            ),
        )


@dataclass(frozen=True)
class RuntimeBundle:
    encoder: semantic_runtime.OfflineMiniLMEncoder
    hippo: OfficialHippoGateway

    def prepare(
        self, items_by_block: Mapping[str, Sequence[ItemTextView]]
    ) -> PreparedBatchArtifact:
        return prepare_semantic_batch(
            items_by_block=items_by_block, encoder=self.encoder
        )


def open_runtime(config: FormalRuntimeConfig) -> RuntimeBundle:
    """Load the frozen semantic encoder only after canonical path validation."""

    if not isinstance(config, FormalRuntimeConfig):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "runtime config type drifted before model load"
        )
    project = _canonical_project(config.project)
    if config != default_formal_runtime_config(project):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "runtime config drifted before model load"
        )
    if os.path.lexists(config.hippo_stage_parent_root):
        raise EraserEvidenceInferenceLocalRuntimeError(
            "runtime output exists before model load"
        )
    try:
        encoder = semantic_runtime.OfflineMiniLMEncoder(
            runtime_asset_path=config.minilm_asset_manifest,
            snapshot_root=config.minilm_snapshot_root,
        )
    except Exception as exc:
        raise EraserEvidenceInferenceLocalRuntimeError(
            "offline MiniLM load/canary failed"
        ) from exc
    return RuntimeBundle(encoder=encoder, hippo=OfficialHippoGateway(config))


__all__ = [
    "AgentExecutionArtifact",
    "BLOCK_ORDER",
    "EraserEvidenceInferenceLocalRuntimeError",
    "FORMAL_ROOT_RELATIVE",
    "FormalRuntimeConfig",
    "HIPPORAG_ATTESTATION_RECEIPT_SHA256",
    "HIPPORAG_STAGE_PARENT_RELATIVE",
    "HIPPORAG_STAGE_RELATIVE",
    "HippoExecutionArtifact",
    "ItemTextView",
    "MINILM_ASSET_FILE_SHA256",
    "MINILM_ASSET_MANIFEST_SHA256",
    "MINILM_ASSET_RELATIVE",
    "MINILM_SNAPSHOT_REVISION",
    "OfficialHippoGateway",
    "PREFLIGHT_SCHEMA",
    "PreparedBatchArtifact",
    "PreparedItemArtifact",
    "RawExecutionArtifact",
    "RuntimeBundle",
    "VERSION",
    "default_formal_runtime_config",
    "execute_agent",
    "execute_raw",
    "open_runtime",
    "preflight_formal_runtime_config",
    "prepare_item_batch",
    "prepare_semantic_batch",
]
