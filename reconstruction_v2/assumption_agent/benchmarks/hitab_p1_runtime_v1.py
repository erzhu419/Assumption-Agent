"""Source-free runtime adapter for the frozen HiTab DMC-1 core.

The adapter accepts only a question, one canonical ordered list of unit
strings, source-native unit types, and source-native typed edges.  It has no
source loader, split, item identity, family, answer, qrel, RAW rank input, or
HippoRAG rank input.

There are three intentionally narrow model capabilities:

* a byte runner for the existing ``bright_query_generator_v2`` semantic
  contract;
* a pair scorer backed in production by the same immutable local BERT asset as
  ``bright_cross_encoder_v1``; and
* the existing offline ``BrightMiniLMEncoder.encode`` capability.

The BRIGHT query-generator contract exposes four expansions but no operation
field.  Therefore this adapter preserves those four expansions and freezes
``UNSPECIFIED`` as the operation; an invalid generation falls back to the full
question alone, also with ``UNSPECIFIED``, without repair or retry.  Likewise,
the BRIGHT cross-encoder IPC contract is hard-coded to 32 documents and one
two-query mean vector, so HiTab cannot truthfully reuse that IPC envelope.
``BrightCrossEncoderProductionScorer`` reuses its immutable model asset and
deterministic inference settings through a dynamic, item-local pair scorer.

RAW is the full-question cross-encoder top five.  It is returned beside, and
never inserted into, :class:`~hitab_p1_dmc1_core_v1.PrecomputedView`.  The
official HippoRAG arm has a separate adapter around the generic BIRCO contract:
it receives the exact same full question and ordered unit strings and returns
five unique local ordinals.  Its output has no route into Agent features.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import threading
from typing import Callable, Protocol, Sequence
import unicodedata

import numpy as np

from assumption_agent.benchmarks import hitab_p1_dmc1_core_v1 as core
from replication_runtime.birco_official_hipporag_v1 import contract as hippo_contract
from replication_runtime.bright_minilm_v1 import encoder as bright_minilm_encoder
from replication_runtime.bright_cross_encoder_v1 import worker as bright_ce_worker
from replication_runtime.bright_query_generator_v1 import contract as planner_contract
from replication_runtime.bright_query_generator_v1 import worker as planner_v1_worker
from replication_runtime.bright_query_generator_v2 import worker as planner_v2_worker


VERSION = "hitab_p1_runtime_v1"
TOP_K = core.TOP_K
MINIMUM_UNIT_COUNT = hippo_contract.MIN_CANDIDATE_COUNT
MAXIMUM_UNIT_COUNT = hippo_contract.MAX_CANDIDATE_COUNT
MAXIMUM_QUESTION_CHARACTERS = planner_contract.MAXIMUM_QUERY_CHARACTERS
MAXIMUM_UNIT_CHARACTERS = 24_000
MAXIMUM_TOTAL_CHARACTERS = 1_500_000
EMBEDDING_DIMENSION = bright_minilm_encoder.EMBEDDING_DIMENSION
CE_BATCH_SIZE = bright_ce_worker.BATCH_SIZE
PHYSICAL_GPUS = (0, 1)
AGENT_FORMATION_PHYSICAL_GPU = 0
OFFICIAL_HIPPO_CONCURRENCY = 2
CPU_THREAD_LIMIT_PER_GPU_LANE = 4
INFERRED_OPERATION = "UNSPECIFIED"
HIPPORAG_OBJECTIVE = (
    "Retrieve the five atomic hierarchical table cells most useful as "
    "evidence for the question."
)
PRODUCTION_OFFLINE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}


class HitabP1RuntimeError(RuntimeError):
    """The source-free runtime or a narrow model contract failed closed."""


class PlannerByteRunner(Protocol):
    """Exactly one local planner invocation over canonical BRIGHT input bytes."""

    def __call__(self, canonical_input: bytes) -> bytes: ...


class CrossEncoderPairScorer(Protocol):
    """Return one finite raw logit for each ordered query/passage pair."""

    def __call__(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[Real]: ...


class MiniLMEncoder(Protocol):
    """The existing offline encoder capability."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class OfficialHippoByteRunner(Protocol):
    """Launch one fresh item-local official-core process on one physical GPU."""

    def __call__(
        self,
        canonical_input: bytes,
        *,
        physical_gpu: int,
        cpu_thread_limit: int,
        launch_ack: Callable[[], None],
    ) -> bytes: ...


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HitabP1RuntimeError("runtime value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _canonical_text(value: object, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise HitabP1RuntimeError(f"{field} must be NUL-free text")
    try:
        normalized = unicodedata.normalize("NFKC", value)
    except UnicodeError as exc:
        raise HitabP1RuntimeError(f"{field} is invalid Unicode") from exc
    normalized = " ".join(normalized.split())
    if not normalized or len(normalized) > maximum:
        raise HitabP1RuntimeError(f"{field} is empty or oversized")
    return normalized


@dataclass(frozen=True)
class RuntimeItem:
    """The complete source-free input visible to the runtime adapter."""

    question: str
    ordered_unit_strings: tuple[str, ...]
    corpus_commitment: str
    unit_types: tuple[str, ...]
    typed_edges: tuple[core.TypedEdge, ...]

    def __post_init__(self) -> None:
        question = _canonical_text(
            self.question,
            field="question",
            maximum=MAXIMUM_QUESTION_CHARACTERS,
        )
        if (
            not isinstance(self.ordered_unit_strings, tuple)
            or not MINIMUM_UNIT_COUNT
            <= len(self.ordered_unit_strings)
            <= MAXIMUM_UNIT_COUNT
        ):
            raise HitabP1RuntimeError("ordered unit count is outside contract")
        units = tuple(
            _canonical_text(
                value,
                field=f"ordered_unit_strings[{index}]",
                maximum=MAXIMUM_UNIT_CHARACTERS,
            )
            for index, value in enumerate(self.ordered_unit_strings)
        )
        expected_corpus_commitment = ordered_corpus_commitment(units)
        if self.corpus_commitment != expected_corpus_commitment:
            raise HitabP1RuntimeError("ordered corpus commitment mismatched")
        if len(self.unit_types) != len(units):
            raise HitabP1RuntimeError("unit type count drifted")
        if not isinstance(self.typed_edges, tuple):
            raise HitabP1RuntimeError("typed edges must be a canonical tuple")
        if (
            len(question) + sum(len(value) for value in units)
            > MAXIMUM_TOTAL_CHARACTERS
        ):
            raise HitabP1RuntimeError("runtime item text exceeds the frozen bound")

        # Reuse the core's source-native type/edge validation without creating
        # any semantic tensor or baseline output.
        placeholder_facets = ((0,) * len(units),)
        placeholder_pairwise = tuple(
            tuple(
                core.QUANT_SCALE if left == right else 0
                for right in range(len(units))
            )
            for left in range(len(units))
        )
        checked = core.PrecomputedView(
            corpus_commitment=expected_corpus_commitment,
            question_facets=(question,),
            unit_keys=tuple(f"U:{index}" for index in range(len(units))),
            unit_types=tuple(self.unit_types),
            typed_edges=tuple(self.typed_edges),
            ce_facet_unit=placeholder_facets,
            minilm_facet_unit=placeholder_facets,
            minilm_unit_unit=placeholder_pairwise,
        )
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "ordered_unit_strings", units)
        object.__setattr__(
            self, "corpus_commitment", expected_corpus_commitment
        )
        object.__setattr__(self, "unit_types", checked.unit_types)
        object.__setattr__(self, "typed_edges", checked.typed_edges)


@dataclass(frozen=True)
class PlannerOutput:
    """Five possible semantic queries plus a non-feature operation receipt."""

    question_facets: tuple[str, ...]
    inferred_operation: str
    generation_valid: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.question_facets, tuple)
            or not 1 <= len(self.question_facets) <= 5
            or len({value.casefold() for value in self.question_facets})
            != len(self.question_facets)
            or self.inferred_operation != INFERRED_OPERATION
            or not isinstance(self.generation_valid, bool)
            or (self.generation_valid and len(self.question_facets) != 5)
            or (not self.generation_valid and len(self.question_facets) != 1)
        ):
            raise HitabP1RuntimeError("planner output contract drifted")


def planner_input_bytes(question: str) -> bytes:
    """Build the exact one-item input consumed by BRIGHT v2."""

    checked = _canonical_text(
        question,
        field="question",
        maximum=MAXIMUM_QUESTION_CHARACTERS,
    )
    items = [{"ordinal": 0, "query": checked}]
    planner_contract.validate_items(items)
    return planner_contract.canonical_json_bytes(
        {"items": items, "schema": planner_contract.INPUT_SCHEMA}
    )


def ordered_corpus_commitment(ordered_units: Sequence[str]) -> str:
    """Commit the exact canonical ordered strings shared by all four arms."""

    if isinstance(ordered_units, (str, bytes)) or not isinstance(
        ordered_units, Sequence
    ):
        raise HitabP1RuntimeError("ordered corpus is not a sequence")
    checked = tuple(
        _canonical_text(
            value,
            field=f"ordered_unit_strings[{index}]",
            maximum=MAXIMUM_UNIT_CHARACTERS,
        )
        for index, value in enumerate(ordered_units)
    )
    if not MINIMUM_UNIT_COUNT <= len(checked) <= MAXIMUM_UNIT_COUNT:
        raise HitabP1RuntimeError("ordered unit count is outside contract")
    return stable_hash(list(checked))


def plan_question(
    question: str, planner_runner: PlannerByteRunner
) -> PlannerOutput:
    """Invoke the frozen local planner exactly once, without repair or retry."""

    if not callable(planner_runner):
        raise HitabP1RuntimeError("planner runner is unavailable")
    checked_question = _canonical_text(
        question,
        field="question",
        maximum=MAXIMUM_QUESTION_CHARACTERS,
    )
    canonical_input = planner_input_bytes(checked_question)
    try:
        raw = planner_runner(canonical_input)
    except Exception as exc:
        raise HitabP1RuntimeError("local planner execution failed") from exc
    if not isinstance(raw, bytes):
        raise HitabP1RuntimeError("local planner output is not bytes")
    try:
        payload = planner_contract.parse_output(raw)
    except Exception as exc:
        raise HitabP1RuntimeError("local planner output contract drifted") from exc
    rows = payload["items"]
    if not isinstance(rows, list) or len(rows) != 1:
        raise HitabP1RuntimeError("local planner item count drifted")
    row = rows[0]
    valid = bool(row["generation_valid"])
    expansions = tuple(row["expansions"]) if valid else ()
    normalized = (
        checked_question.casefold(),
        *(value.casefold() for value in expansions),
    )
    if (
        valid
        and (
            len(expansions) != len(planner_contract.EXPANSION_KEYS)
            or len(set(normalized)) != len(normalized)
        )
    ):
        # This is an invalid semantic completion, not a transport failure.  It
        # receives the single frozen fallback and is never resubmitted.
        valid = False
        expansions = ()
    facets = (
        (checked_question, *expansions)
        if valid
        else (checked_question,)
    )
    return PlannerOutput(
        question_facets=facets,
        inferred_operation=INFERRED_OPERATION,
        generation_valid=valid,
    )


def _stable_sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _cross_encoder_tensor(
    *,
    queries: Sequence[str],
    units: Sequence[str],
    scorer: CrossEncoderPairScorer,
) -> tuple[tuple[int, ...], ...]:
    if not callable(scorer):
        raise HitabP1RuntimeError("cross-encoder scorer is unavailable")
    pairs = tuple((query, unit) for query in queries for unit in units)
    try:
        raw = scorer(pairs)
        values = tuple(raw)
    except Exception as exc:
        raise HitabP1RuntimeError("offline cross-encoder execution failed") from exc
    if len(values) != len(pairs):
        raise HitabP1RuntimeError("cross-encoder score count drifted")
    quantized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise HitabP1RuntimeError("cross-encoder logit is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise HitabP1RuntimeError("cross-encoder logit is nonfinite")
        quantized.append(
            int(round(_stable_sigmoid(numeric) * core.QUANT_SCALE))
        )
    width = len(units)
    return tuple(
        tuple(quantized[start : start + width])
        for start in range(0, len(quantized), width)
    )


def minilm_input_texts(
    planner: PlannerOutput, units: Sequence[str]
) -> tuple[str, ...]:
    return (*planner.question_facets, *tuple(units))


def _validated_embeddings(
    value: object, *, expected_rows: int
) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise HitabP1RuntimeError("MiniLM output is not an array") from exc
    if (
        matrix.dtype != np.dtype(np.float32)
        or matrix.shape != (expected_rows, EMBEDDING_DIMENSION)
        or not np.isfinite(matrix).all()
    ):
        raise HitabP1RuntimeError("MiniLM output tensor drifted")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
        raise HitabP1RuntimeError("MiniLM output is not L2 normalized")
    return np.ascontiguousarray(matrix)


def _q6_cosine(left: np.ndarray, right: np.ndarray) -> int:
    value = float(np.asarray(left @ right, dtype=np.float32).item())
    if not math.isfinite(value) or not -1.000002 <= value <= 1.000002:
        raise HitabP1RuntimeError("MiniLM cosine escaped its range")
    return int(round(max(-1.0, min(1.0, value)) * core.QUANT_SCALE))


def _minilm_tensors(
    *,
    planner: PlannerOutput,
    units: Sequence[str],
    encoder: MiniLMEncoder,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    encode = getattr(encoder, "encode", None)
    if not callable(encode):
        raise HitabP1RuntimeError("MiniLM encoder capability is unavailable")
    texts = minilm_input_texts(planner, units)
    try:
        raw = encode(texts)
    except Exception as exc:
        raise HitabP1RuntimeError("offline MiniLM execution failed") from exc
    matrix = _validated_embeddings(raw, expected_rows=len(texts))
    facet_count = len(planner.question_facets)
    facets = matrix[:facet_count]
    unit_matrix = matrix[facet_count:]
    facet_unit = tuple(
        tuple(_q6_cosine(facet, unit) for unit in unit_matrix)
        for facet in facets
    )
    pairwise_rows: list[tuple[int, ...]] = []
    for left_index, left in enumerate(unit_matrix):
        row = []
        for right_index, right in enumerate(unit_matrix):
            row.append(
                core.QUANT_SCALE
                if left_index == right_index
                else _q6_cosine(left, right)
            )
        pairwise_rows.append(tuple(row))
    pairwise = tuple(pairwise_rows)
    # Remove possible last-bit asymmetry from separate float32 dot products.
    symmetric = tuple(
        tuple(
            core.QUANT_SCALE
            if left == right
            else pairwise[min(left, right)][max(left, right)]
            for right in range(len(units))
        )
        for left in range(len(units))
    )
    return facet_unit, symmetric


@dataclass(frozen=True)
class CompiledRuntime:
    """Agent tensor plus the independently held RAW action."""

    planner: PlannerOutput
    view: core.PrecomputedView
    raw_top5: tuple[int, ...]
    physical_gpu: int
    tensor_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.raw_top5) != TOP_K
            or len(set(self.raw_top5)) != TOP_K
            or any(
                type(value) is not int
                or not 0 <= value < self.view.unit_count
                for value in self.raw_top5
            )
            or self.physical_gpu != AGENT_FORMATION_PHYSICAL_GPU
            or not isinstance(self.tensor_sha256, str)
            or len(self.tensor_sha256) != 64
        ):
            raise HitabP1RuntimeError("compiled runtime contract drifted")


def compile_runtime(
    item: RuntimeItem,
    *,
    planner_runner: PlannerByteRunner,
    cross_encoder_scorer: CrossEncoderPairScorer,
    minilm_encoder: MiniLMEncoder,
    physical_gpu: int,
) -> CompiledRuntime:
    """Compile one item with one planner, one CE, and one MiniLM invocation."""

    if not isinstance(item, RuntimeItem):
        raise HitabP1RuntimeError("runtime item type drifted")
    checked_gpu = _physical_gpu(physical_gpu)
    if checked_gpu != AGENT_FORMATION_PHYSICAL_GPU:
        raise HitabP1RuntimeError(
            "planner/CE/MiniLM/Agent formation is frozen to physical GPU 0"
        )
    planner = plan_question(item.question, planner_runner)
    ce = _cross_encoder_tensor(
        queries=planner.question_facets,
        units=item.ordered_unit_strings,
        scorer=cross_encoder_scorer,
    )
    facet_minilm, unit_minilm = _minilm_tensors(
        planner=planner,
        units=item.ordered_unit_strings,
        encoder=minilm_encoder,
    )
    view = core.PrecomputedView(
        corpus_commitment=item.corpus_commitment,
        question_facets=planner.question_facets,
        unit_keys=tuple(f"U:{index}" for index in range(len(item.ordered_unit_strings))),
        unit_types=item.unit_types,
        typed_edges=item.typed_edges,
        ce_facet_unit=ce,
        minilm_facet_unit=facet_minilm,
        minilm_unit_unit=unit_minilm,
    )
    raw_top5 = tuple(
        sorted(
            range(view.unit_count),
            key=lambda ordinal: (-ce[0][ordinal], ordinal),
        )[:TOP_K]
    )
    body = {
        "inferred_operation": planner.inferred_operation,
        "planner_generation_valid": planner.generation_valid,
        "physical_gpu": checked_gpu,
        "corpus_commitment": item.corpus_commitment,
        "raw_top5": list(raw_top5),
        "runtime": VERSION,
        "view": view.payload(),
    }
    return CompiledRuntime(
        planner=planner,
        view=view,
        raw_top5=raw_top5,
        physical_gpu=checked_gpu,
        tensor_sha256=stable_hash(body),
    )


def _hippo_projection(
    question: str, ordered_units: Sequence[str]
) -> tuple[str, str, tuple[hippo_contract.CandidateDocument, ...], str]:
    checked_question = _canonical_text(
        question,
        field="HippoRAG question",
        maximum=MAXIMUM_QUESTION_CHARACTERS,
    )
    documents = tuple(
        hippo_contract.CandidateDocument(index, value)
        for index, value in enumerate(ordered_units)
    )
    mapping = [
        {"ordinal": row.ordinal, "text": row.text} for row in documents
    ]
    projection = hippo_contract.common_projection_sha256(
        objective=HIPPORAG_OBJECTIVE,
        query=checked_question,
        documents=mapping,
    )
    work_id = stable_hash(
        {
            "query": checked_question,
            "schema": f"{VERSION}_official_hippo_work_v1",
            "unit_strings": list(ordered_units),
        }
    )
    return work_id, checked_question, documents, projection


def hippo_input_bytes(question: str, ordered_units: Sequence[str]) -> bytes:
    """Build the exact generic official-HippoRAG input envelope."""

    work_id, checked_question, documents, projection = _hippo_projection(
        question, ordered_units
    )
    body = {
        "common_projection_sha256": projection,
        "documents": [
            {"ordinal": row.ordinal, "text": row.text} for row in documents
        ],
        "objective": HIPPORAG_OBJECTIVE,
        "query": checked_question,
        "schema": hippo_contract.INPUT_SCHEMA,
        "work_id": work_id,
    }
    hippo_contract.validate_input(
        work_id,
        HIPPORAG_OBJECTIVE,
        checked_question,
        body["documents"],
        projection,
    )
    return hippo_contract.canonical_json_bytes(body)


@dataclass(frozen=True)
class OfficialHippoAction:
    top5_ordinals: tuple[int, ...]
    corpus_commitment: str
    physical_gpu: int
    input_sha256: str
    output_sha256: str
    complete_rank_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.top5_ordinals) != TOP_K
            or len(set(self.top5_ordinals)) != TOP_K
            or any(type(value) is not int or value < 0 for value in self.top5_ordinals)
            or not isinstance(self.corpus_commitment, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.corpus_commitment) is None
            or self.physical_gpu not in PHYSICAL_GPUS
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (
                    self.input_sha256,
                    self.output_sha256,
                    self.complete_rank_sha256,
                )
            )
        ):
            raise HitabP1RuntimeError("official HippoRAG action drifted")


def run_official_hippo(
    question: str,
    ordered_units: Sequence[str],
    runner: OfficialHippoByteRunner,
    *,
    physical_gpu: int,
    launch_ack: Callable[[], None] | None = None,
) -> OfficialHippoAction:
    """Run the independent official arm once and return its first five.

    ``launch_ack`` is emitted exactly once by the byte runner immediately
    before it crosses into the fresh official process.  Wrapping it here makes
    a missing or duplicate acknowledgement a hard contract failure even for
    injected qualification runners.
    """

    if not callable(runner) or (
        launch_ack is not None and not callable(launch_ack)
    ):
        raise HitabP1RuntimeError("official HippoRAG runner is unavailable")
    checked_gpu = _physical_gpu(physical_gpu)
    canonical_input = hippo_input_bytes(question, ordered_units)
    ack_lock = threading.Lock()
    ack_count = 0

    def acknowledge_launch_once() -> None:
        nonlocal ack_count
        with ack_lock:
            ack_count += 1
            if ack_count != 1:
                raise HitabP1RuntimeError(
                    "official HippoRAG launch acknowledgement repeated"
                )
        if launch_ack is not None:
            launch_ack()

    try:
        raw = runner(
            canonical_input,
            physical_gpu=checked_gpu,
            cpu_thread_limit=CPU_THREAD_LIMIT_PER_GPU_LANE,
            launch_ack=acknowledge_launch_once,
        )
    except Exception as exc:
        raise HitabP1RuntimeError("official HippoRAG execution failed") from exc
    with ack_lock:
        acknowledged = ack_count == 1
    if not acknowledged:
        raise HitabP1RuntimeError(
            "official HippoRAG launch was not acknowledged"
        )
    if not isinstance(raw, bytes):
        raise HitabP1RuntimeError("official HippoRAG output is not bytes")
    try:
        payload = hippo_contract.parse_output(raw)
    except Exception as exc:
        raise HitabP1RuntimeError(
            "official HippoRAG output contract drifted"
        ) from exc
    work_id, _query, _documents, projection = _hippo_projection(
        question, ordered_units
    )
    if (
        payload["work_id"] != work_id
        or payload["common_projection_sha256"] != projection
        or payload["candidate_count"] != len(ordered_units)
    ):
        raise HitabP1RuntimeError("official HippoRAG binding drifted")
    ranking = tuple(payload["rank_ordinals"])
    top5 = ranking[:TOP_K]
    if len(top5) != TOP_K or len(set(top5)) != TOP_K:
        raise HitabP1RuntimeError("official HippoRAG top five drifted")
    return OfficialHippoAction(
        top5_ordinals=top5,
        corpus_commitment=ordered_corpus_commitment(ordered_units),
        physical_gpu=checked_gpu,
        input_sha256=hashlib.sha256(canonical_input).hexdigest(),
        output_sha256=hashlib.sha256(raw).hexdigest(),
        complete_rank_sha256=stable_hash(list(ranking)),
    )


def run_official_hippo_queue(
    items: Sequence[RuntimeItem],
    runner: OfficialHippoByteRunner,
    *,
    block: str,
) -> tuple[OfficialHippoAction, ...]:
    """Execute A_hold/M as two bounded per-GPU queues.

    Every runner call must create and close one item-local official process.
    The two thread lanes are scheduling queues, not persistent HippoRAG cores.
    A_form is rejected because its frozen contract has no Hippo arm.
    """

    if block not in {"A_hold", "M_search"}:
        raise HitabP1RuntimeError(
            "official HippoRAG queue is authorized only for A_hold or M_search"
        )
    if (
        isinstance(items, (str, bytes))
        or not isinstance(items, Sequence)
        or not items
        or any(not isinstance(item, RuntimeItem) for item in items)
        or not callable(runner)
    ):
        raise HitabP1RuntimeError("official HippoRAG queue input drifted")
    output: list[OfficialHippoAction | None] = [None] * len(items)

    def lane(physical_gpu: int, ordinals: tuple[int, ...]) -> None:
        for ordinal in ordinals:
            item = items[ordinal]
            output[ordinal] = run_official_hippo(
                item.question,
                item.ordered_unit_strings,
                runner,
                physical_gpu=physical_gpu,
            )

    lane_ordinals = tuple(
        tuple(
            index
            for index in range(len(items))
            if index % OFFICIAL_HIPPO_CONCURRENCY == physical_gpu
        )
        for physical_gpu in PHYSICAL_GPUS
    )
    with ThreadPoolExecutor(
        max_workers=OFFICIAL_HIPPO_CONCURRENCY,
        thread_name_prefix="hitab-official-hippo-gpu-lane",
    ) as executor:
        futures = [
            executor.submit(lane, physical_gpu, lane_ordinals[physical_gpu])
            for physical_gpu in PHYSICAL_GPUS
            if lane_ordinals[physical_gpu]
        ]
        for future in futures:
            future.result()
    if any(value is None for value in output):
        raise HitabP1RuntimeError("official HippoRAG queue output is incomplete")
    return tuple(value for value in output if value is not None)


class BrightPlannerProductionRunner:
    """Bind the existing local BRIGHT v2 worker to the narrow byte protocol."""

    def __init__(self, model_root: Path, *, physical_gpu: int) -> None:
        _require_production_offline_environment()
        self.physical_gpu = _require_visible_physical_gpu(physical_gpu)
        self._model, self._tokenizer = planner_v1_worker._load_model(
            _direct_model_root(model_root, field="planner model root")
        )

    def __call__(self, canonical_input: bytes) -> bytes:
        items = planner_contract.parse_input(canonical_input)
        payload, _schedule_receipt = planner_v2_worker.generate(
            items=items,
            model=self._model,
            tokenizer=self._tokenizer,
        )
        return planner_contract.canonical_json_bytes(payload)


class BrightCrossEncoderProductionScorer:
    """Dynamic pair scorer using the exact local BRIGHT CE asset."""

    def __init__(self, model_root: Path, *, physical_gpu: int) -> None:
        _require_production_offline_environment()
        self.physical_gpu = _require_visible_physical_gpu(physical_gpu)
        root = bright_ce_worker._model_root(
            _direct_model_root(model_root, field="cross-encoder model root")
        )
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise HitabP1RuntimeError("cross-encoder CUDA device is unavailable")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self._tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            root,
            local_files_only=True,
            use_safetensors=True,
        ).eval().cuda()
        if (
            self._model.__class__.__name__
            != "BertForSequenceClassification"
            or self._model.num_labels != 1
        ):
            raise HitabP1RuntimeError(
                "cross-encoder model architecture drifted"
            )

    def __call__(
        self, pairs: Sequence[tuple[str, str]]
    ) -> tuple[float, ...]:
        import torch

        if not pairs:
            raise HitabP1RuntimeError("cross-encoder pair slate is empty")
        output: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), CE_BATCH_SIZE):
                batch = pairs[start : start + CE_BATCH_SIZE]
                encoded = self._tokenizer(
                    [row[0] for row in batch],
                    [row[1] for row in batch],
                    max_length=bright_ce_worker.MAXIMUM_SEQUENCE_LENGTH,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )
                encoded = {
                    key: value.cuda() for key, value in encoded.items()
                }
                logits = self._model(**encoded).logits
                if logits.ndim != 2 or logits.shape != (len(batch), 1):
                    raise HitabP1RuntimeError(
                        "cross-encoder output shape drifted"
                    )
                output.extend(
                    float(value) for value in logits[:, 0].detach().cpu()
                )
        if len(output) != len(pairs) or not all(
            math.isfinite(value) for value in output
        ):
            raise HitabP1RuntimeError("cross-encoder output drifted")
        return tuple(output)


def bind_bright_minilm_production_encoder(
    *,
    asset_manifest: Path,
    model_root: Path,
    physical_gpu: int,
) -> bright_minilm_encoder.BrightMiniLMEncoder:
    """Bind the existing verified local MiniLM asset on an explicit GPU."""

    _require_production_offline_environment()
    _require_visible_physical_gpu(physical_gpu)
    if (
        not isinstance(asset_manifest, Path)
        or not asset_manifest.is_absolute()
        or asset_manifest.is_symlink()
        or not asset_manifest.is_file()
        or asset_manifest.resolve() != asset_manifest
    ):
        raise HitabP1RuntimeError(
            "MiniLM asset manifest is not a direct local file"
        )
    root = _direct_model_root(model_root, field="MiniLM model root")
    return bright_minilm_encoder.BrightMiniLMEncoder(
        asset_manifest=asset_manifest,
        model_root=root,
    )


def _require_production_offline_environment() -> None:
    for key, expected in PRODUCTION_OFFLINE_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise HitabP1RuntimeError(f"{key} production binding drifted")


def _physical_gpu(value: object) -> int:
    if type(value) is not int or value not in PHYSICAL_GPUS:
        raise HitabP1RuntimeError("physical GPU must be explicitly 0 or 1")
    return int(value)


def _require_visible_physical_gpu(value: object) -> int:
    physical_gpu = _physical_gpu(value)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise HitabP1RuntimeError(
            "CUDA_VISIBLE_DEVICES does not bind the requested physical GPU"
        )
    return physical_gpu


def _direct_model_root(path: Path, *, field: str) -> Path:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve() != path
    ):
        raise HitabP1RuntimeError(f"{field} is not a direct local directory")
    return path


__all__ = [
    "BrightCrossEncoderProductionScorer",
    "BrightPlannerProductionRunner",
    "CompiledRuntime",
    "CrossEncoderPairScorer",
    "HIPPORAG_OBJECTIVE",
    "HitabP1RuntimeError",
    "INFERRED_OPERATION",
    "MiniLMEncoder",
    "OfficialHippoAction",
    "OfficialHippoByteRunner",
    "PlannerByteRunner",
    "PlannerOutput",
    "RuntimeItem",
    "VERSION",
    "bind_bright_minilm_production_encoder",
    "compile_runtime",
    "hippo_input_bytes",
    "minilm_input_texts",
    "ordered_corpus_commitment",
    "plan_question",
    "planner_input_bytes",
    "run_official_hippo",
    "run_official_hippo_queue",
]
