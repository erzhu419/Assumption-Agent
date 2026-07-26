"""One source-free production-isomorphic canary for HiTab DMC-1.

The fixture is a fixed public synthetic hierarchical cell corpus.  It contains
no HiTab row, table, question, identifier, family, answer, or score.  A canary
pass crosses the injected local planner, cross-encoder, MiniLM compiler, RAW
formation, E0 A_form registry seal, *then* a synthetic DNF qrel open, one E1
fit, E0/E1 action construction, and the independent official-HippoRAG adapter.
As in formal A_hold/M_search, the GPU1 odd Hippo lane enters first and overlaps
GPU0 feature/action formation; the GPU0 Hippo lane remains blocked until
formation, action construction, and the unused-cache release are complete.

The whole path is evaluated twice and must be exact.  This is only an
implementation/capability qualification: E1 may equal E0, either Agent action
may equal RAW or HippoRAG, and every residual may be zero.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import hmac
import re
import threading
from typing import Callable, Mapping

from assumption_agent.benchmarks import hitab_p1_dmc1_core_v1 as core
from assumption_agent.benchmarks import hitab_p1_runtime_v1 as runtime


VERSION = "hitab_p1_public_canary_v1"
SCHEMA = f"{VERSION}_receipt"
REPEAT_COUNT = 2
PUBLIC_EXPLORATION_KEY = hashlib.sha256(
    b"HiTab DMC-1 public synthetic A_form exploration key v1"
).digest()
_HEX64 = re.compile(r"[0-9a-f]{64}")
GPU0CacheReleaser = Callable[[], Mapping[str, object]]
FORMAL_PHASE_ORDER = (
    "gpu1_hippo_lane_live",
    "gpu1_hippo_call_begin",
    "gpu0_feature_formation_begin",
    "gpu0_feature_formation_complete",
    "gpu0_unused_cache_release",
    "gpu0_hippo_lane_release",
    "gpu0_hippo_call_begin",
    "hipporag_queue_joined",
)

SYNTHETIC_QUESTION = (
    "Across the synthetic North and South regions, which displayed renewable "
    "share is larger and what year does the table describe?"
)
SYNTHETIC_UNITS = (
    "VALUE type=percentage surface=42 | LEFT_PATH Region > North | "
    "TOP_PATH Energy > Renewable share > 2024",
    "VALUE type=percentage surface=37 | LEFT_PATH Region > South | "
    "TOP_PATH Energy > Renewable share > 2024",
    "VALUE type=integer surface=2024 | LEFT_PATH Metadata > Year | "
    "TOP_PATH Observation",
    "VALUE type=text surface=North | LEFT_PATH Region | TOP_PATH Label",
    "VALUE type=text surface=South | LEFT_PATH Region | TOP_PATH Label",
    "VALUE type=percentage surface=58 | LEFT_PATH Region > North | "
    "TOP_PATH Energy > Non-renewable share > 2024",
    "VALUE type=percentage surface=63 | LEFT_PATH Region > South | "
    "TOP_PATH Energy > Non-renewable share > 2024",
    "VALUE type=text surface=percent | LEFT_PATH Units | TOP_PATH Measure",
    "VALUE type=integer surface=2023 | LEFT_PATH Metadata > Prior year | "
    "TOP_PATH Observation",
    "VALUE type=text surface=synthetic | LEFT_PATH Provenance | "
    "TOP_PATH Fixture",
)
SYNTHETIC_UNIT_TYPES = (
    "PERCENTAGE",
    "PERCENTAGE",
    "INTEGER",
    "TEXT",
    "TEXT",
    "PERCENTAGE",
    "PERCENTAGE",
    "TEXT",
    "INTEGER",
    "TEXT",
)
SYNTHETIC_EDGES = tuple(
    sorted(
        core.TypedEdge(left, right, core.SOURCE_NATIVE_EDGE_TYPE)
        for left, right in (
            (0, 2),
            (0, 3),
            (0, 5),
            (1, 2),
            (1, 4),
            (1, 6),
            (3, 5),
            (4, 6),
        )
    )
)


class HitabP1PublicCanaryError(RuntimeError):
    """The public synthetic path failed capability qualification."""


@dataclass(frozen=True)
class CanaryPass:
    aform_compiled_tensor_sha256: str
    compiled_tensor_sha256: str
    planner_generation_valid: bool
    registry_seal_sha256: str
    prelabel_archive_sha256: str
    model_sha256: str
    raw_top5: tuple[int, ...]
    e0_top5: tuple[int, ...]
    e1_top5: tuple[int, ...]
    hippo_top5: tuple[tuple[int, ...], ...]
    hippo_physical_gpus: tuple[int, ...]
    hippo_input_sha256s: tuple[str, ...]
    hippo_output_sha256s: tuple[str, ...]
    four_arm_corpus_commitment_exact: bool
    gpu0_cache_release_receipt_sha256: str
    gpu0_hippo_started_after_formation_and_cache_release: bool
    gpu1_hippo_overlapped_gpu0_feature_formation: bool
    hippo_maximum_active_calls_per_gpu: tuple[int, int]
    formal_phase_order: tuple[str, ...]
    qrel_opened_after_seal: bool
    prelabel_archive_contains_no_qrel: bool


def synthetic_item() -> runtime.RuntimeItem:
    """Return a fresh immutable copy of the public hierarchical fixture."""

    return runtime.RuntimeItem(
        question=SYNTHETIC_QUESTION,
        ordered_unit_strings=tuple(SYNTHETIC_UNITS),
        corpus_commitment=runtime.ordered_corpus_commitment(SYNTHETIC_UNITS),
        unit_types=tuple(SYNTHETIC_UNIT_TYPES),
        typed_edges=tuple(SYNTHETIC_EDGES),
    )


def second_synthetic_item() -> runtime.RuntimeItem:
    """A distinct public item used only to qualify the second Hippo lane."""

    units = tuple(
        value.replace("North", "East")
        .replace("South", "West")
        .replace("2024", "2025")
        .replace("renewable", "employment")
        for value in SYNTHETIC_UNITS
    )
    return runtime.RuntimeItem(
        question=(
            "For the separate synthetic East and West regions, retrieve the "
            "displayed 2025 employment cells needed for comparison."
        ),
        ordered_unit_strings=units,
        corpus_commitment=runtime.ordered_corpus_commitment(units),
        unit_types=tuple(SYNTHETIC_UNIT_TYPES),
        typed_edges=tuple(SYNTHETIC_EDGES),
    )


def _late_synthetic_qrel() -> core.ProofDNF:
    # This function is called only after the registry payload and seal hash have
    # been materialized.  Alternatives remain DNF alternatives and are never
    # unioned.
    return core.ProofDNF(
        alternatives=(
            ((0,), (1,), (2,)),
        ),
        corpus_commitment=runtime.ordered_corpus_commitment(
            SYNTHETIC_UNITS
        ),
    )


def _cache_release_receipt_sha256(
    value: Mapping[str, object],
) -> str:
    if not isinstance(value, Mapping):
        raise HitabP1PublicCanaryError(
            "GPU0 cache release receipt is not an object"
        )
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        set(value)
        != {
            "model_offload_or_reload",
            "physical_gpu",
            "schema",
            "self_sha256",
            "study_id",
            "torch_cuda_empty_cache_called",
        }
        or value.get("schema")
        != "hitab_p1_gpu0_unused_cuda_cache_release_v1"
        or value.get("study_id") != core.STUDY_ID
        or value.get("physical_gpu") != 0
        or value.get("torch_cuda_empty_cache_called") is not True
        or value.get("model_offload_or_reload") is not False
        or not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or not hmac.compare_digest(runtime.stable_hash(body), claimed)
    ):
        raise HitabP1PublicCanaryError(
            "GPU0 cache release receipt drifted"
        )
    return claimed


def _one_pass(
    *,
    planner_runner: runtime.PlannerByteRunner,
    cross_encoder_scorer: runtime.CrossEncoderPairScorer,
    minilm_encoder: runtime.MiniLMEncoder,
    hippo_runner: runtime.OfficialHippoByteRunner,
    gpu0_cache_releaser: GPU0CacheReleaser,
) -> CanaryPass:
    item = synthetic_item()
    # A_form has no Hippo arm.  Complete its source-free feature archive,
    # registry seal, late label open, and E1 fit before A_hold scheduling.
    aform_compiled = runtime.compile_runtime(
        item,
        planner_runner=planner_runner,
        cross_encoder_scorer=cross_encoder_scorer,
        minilm_encoder=minilm_encoder,
        physical_gpu=runtime.AGENT_FORMATION_PHYSICAL_GPU,
    )
    registry = core.build_and_seal_aform_registry(
        aform_compiled.view,
        exploration_key=PUBLIC_EXPLORATION_KEY,
    )
    prelabel_payload = core.registry_payload(registry)
    prelabel_archive_sha256 = core.stable_hash(prelabel_payload)
    serialized_prelabel = runtime.canonical_bytes(prelabel_payload).lower()
    no_qrel = all(
        token not in serialized_prelabel
        for token in (b"qrel", b"gold", b"proof", b"answer")
    )
    sealed_before_qrel = bool(
        registry.seal_sha256 and prelabel_archive_sha256 and no_qrel
    )
    proof = _late_synthetic_qrel()
    labelled = core.label_sealed_registry(registry, proof)
    model = core.fit_e1((labelled,))

    # A_hold now mirrors the formal phase: GPU1's odd Hippo lane starts first
    # and overlaps a fresh GPU0 compile plus RAW/E0/E1 construction using the
    # already-fitted A_form model.  GPU0 Hippo remains blocked until release.
    hippo_items = (item, second_synthetic_item())
    hippo_output: list[runtime.OfficialHippoAction | None] = [None, None]
    release_gpu0 = threading.Event()
    gpu1_ack_or_terminal = threading.Event()
    gpu1_launch_acknowledged = threading.Event()
    gpu1_active = threading.Event()
    cache_release_complete = threading.Event()
    abort = threading.Event()
    phase_lock = threading.Lock()
    active_lock = threading.Lock()
    phase_events: list[str] = []
    active_by_gpu = [0, 0]
    maximum_active_by_gpu = [0, 0]
    gpu0_started_after_release = False

    def record(event: str) -> None:
        with phase_lock:
            phase_events.append(event)

    def hippo_lane(physical_gpu: int) -> None:
        nonlocal gpu0_started_after_release
        acknowledged = False

        def acknowledge_launch() -> None:
            nonlocal acknowledged, gpu0_started_after_release
            acknowledged = True
            if physical_gpu == 0:
                gpu0_started_after_release = (
                    release_gpu0.is_set()
                    and cache_release_complete.is_set()
                )
                record("gpu0_hippo_call_begin")
            else:
                record("gpu1_hippo_call_begin")
            with active_lock:
                active_by_gpu[physical_gpu] += 1
                maximum_active_by_gpu[physical_gpu] = max(
                    maximum_active_by_gpu[physical_gpu],
                    active_by_gpu[physical_gpu],
                )
            if physical_gpu == 1:
                gpu1_active.set()
                gpu1_launch_acknowledged.set()
                gpu1_ack_or_terminal.set()

        if physical_gpu == 0:
            release_gpu0.wait()
            if abort.is_set():
                return
        else:
            record("gpu1_hippo_lane_live")
        try:
            hippo_item = hippo_items[physical_gpu]
            hippo_output[physical_gpu] = runtime.run_official_hippo(
                hippo_item.question,
                hippo_item.ordered_unit_strings,
                hippo_runner,
                physical_gpu=physical_gpu,
                launch_ack=acknowledge_launch,
            )
        finally:
            if acknowledged:
                if physical_gpu == 1:
                    gpu1_active.clear()
                with active_lock:
                    active_by_gpu[physical_gpu] -= 1
            if physical_gpu == 1:
                # A pre-ack runner failure must wake the main thread without
                # fabricating launch or overlap evidence.
                gpu1_ack_or_terminal.set()

    formation_error: Exception | None = None
    overlap_observed = False
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix=(
            "hitab-public-canary-official-hippo-physical-gpu"
        ),
    ) as executor:
        futures = (
            executor.submit(hippo_lane, 0),
            executor.submit(hippo_lane, 1),
        )
        gpu1_ack_or_terminal.wait()
        if not gpu1_launch_acknowledged.is_set():
            formation_error = HitabP1PublicCanaryError(
                "GPU1 HippoRAG terminated before launch acknowledgement"
            )
            abort.set()
            release_gpu0.set()
        else:
            try:
                overlap_observed = gpu1_active.is_set()
                if not overlap_observed:
                    raise HitabP1PublicCanaryError(
                        "GPU1 HippoRAG did not overlap GPU0 feature formation"
                    )
                record("gpu0_feature_formation_begin")
                ahold_compiled = runtime.compile_runtime(
                    item,
                    planner_runner=planner_runner,
                    cross_encoder_scorer=cross_encoder_scorer,
                    minilm_encoder=minilm_encoder,
                    physical_gpu=runtime.AGENT_FORMATION_PHYSICAL_GPU,
                )
                raw = ahold_compiled.raw_top5
                e0 = core.select_e0(ahold_compiled.view)
                e1 = core.select_e1(ahold_compiled.view, model)
                record("gpu0_feature_formation_complete")
                cache_release_receipt_sha256 = (
                    _cache_release_receipt_sha256(gpu0_cache_releaser())
                )
                cache_release_complete.set()
                record("gpu0_unused_cache_release")
            except Exception as exc:
                formation_error = exc
                abort.set()
            finally:
                if formation_error is None:
                    record("gpu0_hippo_lane_release")
                release_gpu0.set()
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                if formation_error is None:
                    formation_error = exc
                abort.set()

    if formation_error is not None:
        raise formation_error
    if any(row is None for row in hippo_output):
        raise HitabP1PublicCanaryError(
            "public synthetic HippoRAG output is incomplete"
        )
    record("hipporag_queue_joined")
    phase_order = tuple(phase_events)
    if (
        phase_order != FORMAL_PHASE_ORDER
        or maximum_active_by_gpu != [1, 1]
        or not gpu0_started_after_release
        or not overlap_observed
    ):
        raise HitabP1PublicCanaryError(
            "public synthetic formal phase order drifted"
        )
    hippo = tuple(
        row for row in hippo_output if row is not None
    )
    actions = (raw, e0, e1, *(row.top5_ordinals for row in hippo))
    if any(len(row) != core.TOP_K or len(set(row)) != core.TOP_K for row in actions):
        raise HitabP1PublicCanaryError(
            "public synthetic action cardinality drifted"
        )
    four_arm_corpus_commitment_exact = (
        item.corpus_commitment
        == aform_compiled.view.corpus_commitment
        == ahold_compiled.view.corpus_commitment
        == registry.corpus_commitment
        == hippo[0].corpus_commitment
    )
    if not four_arm_corpus_commitment_exact:
        raise HitabP1PublicCanaryError(
            "public synthetic four-arm corpus binding drifted"
        )
    return CanaryPass(
        aform_compiled_tensor_sha256=aform_compiled.tensor_sha256,
        compiled_tensor_sha256=ahold_compiled.tensor_sha256,
        planner_generation_valid=(
            aform_compiled.planner.generation_valid
            and ahold_compiled.planner.generation_valid
        ),
        registry_seal_sha256=registry.seal_sha256,
        prelabel_archive_sha256=prelabel_archive_sha256,
        model_sha256=core.stable_hash(core.model_payload(model)),
        raw_top5=raw,
        e0_top5=e0,
        e1_top5=e1,
        hippo_top5=tuple(row.top5_ordinals for row in hippo),
        hippo_physical_gpus=tuple(row.physical_gpu for row in hippo),
        hippo_input_sha256s=tuple(row.input_sha256 for row in hippo),
        hippo_output_sha256s=tuple(row.output_sha256 for row in hippo),
        four_arm_corpus_commitment_exact=four_arm_corpus_commitment_exact,
        gpu0_cache_release_receipt_sha256=(
            cache_release_receipt_sha256
        ),
        gpu0_hippo_started_after_formation_and_cache_release=(
            gpu0_started_after_release
        ),
        gpu1_hippo_overlapped_gpu0_feature_formation=(
            overlap_observed
        ),
        hippo_maximum_active_calls_per_gpu=tuple(
            maximum_active_by_gpu
        ),
        formal_phase_order=phase_order,
        qrel_opened_after_seal=sealed_before_qrel,
        prelabel_archive_contains_no_qrel=no_qrel,
    )


def _pass_payload(value: CanaryPass) -> dict[str, object]:
    return {
        "A_form_compiled_tensor_sha256": (
            value.aform_compiled_tensor_sha256
        ),
        "A_hold_compiled_tensor_sha256": value.compiled_tensor_sha256,
        "compiled_tensor_sha256": value.compiled_tensor_sha256,
        "E0_top5": list(value.e0_top5),
        "E1_top5": list(value.e1_top5),
        "HippoRAG_top5": [list(row) for row in value.hippo_top5],
        "RAW_top5": list(value.raw_top5),
        "hippo_input_sha256s": list(value.hippo_input_sha256s),
        "hippo_observed_physical_GPU_set": sorted(
            set(value.hippo_physical_gpus)
        ),
        "hippo_output_sha256s": list(value.hippo_output_sha256s),
        "four_arm_corpus_commitment_exact": (
            value.four_arm_corpus_commitment_exact
        ),
        "gpu0_cache_release_receipt_sha256": (
            value.gpu0_cache_release_receipt_sha256
        ),
        "gpu0_hippo_started_after_formation_and_cache_release": (
            value.gpu0_hippo_started_after_formation_and_cache_release
        ),
        "gpu1_hippo_overlapped_gpu0_feature_formation": (
            value.gpu1_hippo_overlapped_gpu0_feature_formation
        ),
        "hippo_maximum_active_calls_per_GPU": list(
            value.hippo_maximum_active_calls_per_gpu
        ),
        "formal_phase_order": list(value.formal_phase_order),
        "model_sha256": value.model_sha256,
        "planner_generation_valid": value.planner_generation_valid,
        "prelabel_archive_contains_no_qrel": (
            value.prelabel_archive_contains_no_qrel
        ),
        "prelabel_archive_sha256": value.prelabel_archive_sha256,
        "qrel_opened_after_seal": value.qrel_opened_after_seal,
        "registry_seal_sha256": value.registry_seal_sha256,
    }


def run_public_canary(
    *,
    planner_runner: runtime.PlannerByteRunner,
    cross_encoder_scorer: runtime.CrossEncoderPairScorer,
    minilm_encoder: runtime.MiniLMEncoder,
    hippo_runner: runtime.OfficialHippoByteRunner,
    gpu0_cache_releaser: GPU0CacheReleaser,
) -> dict[str, object]:
    """Run the complete public path twice and return a self-hashed receipt."""

    first = _one_pass(
        planner_runner=planner_runner,
        cross_encoder_scorer=cross_encoder_scorer,
        minilm_encoder=minilm_encoder,
        hippo_runner=hippo_runner,
        gpu0_cache_releaser=gpu0_cache_releaser,
    )
    second = _one_pass(
        planner_runner=planner_runner,
        cross_encoder_scorer=cross_encoder_scorer,
        minilm_encoder=minilm_encoder,
        hippo_runner=hippo_runner,
        gpu0_cache_releaser=gpu0_cache_releaser,
    )
    if first != second:
        raise HitabP1PublicCanaryError(
            "public synthetic production path is not exact-repeat deterministic"
        )
    if (
        not first.qrel_opened_after_seal
        or not first.prelabel_archive_contains_no_qrel
    ):
        raise HitabP1PublicCanaryError("late-qrel boundary drifted")

    body: dict[str, object] = {
        "actions_need_not_differ": True,
        "behavior_or_efficacy_gate": False,
        "E1_minus_E0_nonzero_required": False,
        "E1_outside_RAW_required": False,
        "pass": _pass_payload(first),
        "qualified": True,
        "repeat_count": REPEAT_COUNT,
        "repeat_exact": True,
        "residual_nonzero_required": False,
        "schema": SCHEMA,
        "source_or_HiTab_rows_accessed": False,
        "study_id": core.STUDY_ID,
        "version": VERSION,
    }
    body["self_sha256"] = runtime.stable_hash(body)
    return body


def validate_receipt(value: Mapping[str, object]) -> str:
    """Validate the minimal immutable canary receipt boundary."""

    if not isinstance(value, Mapping):
        raise HitabP1PublicCanaryError("canary receipt is not an object")
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    pass_row = body.get("pass")
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or runtime.stable_hash(body) != claimed
        or body.get("schema") != SCHEMA
        or body.get("study_id") != core.STUDY_ID
        or body.get("qualified") is not True
        or body.get("repeat_count") != REPEAT_COUNT
        or body.get("repeat_exact") is not True
        or body.get("behavior_or_efficacy_gate") is not False
        or body.get("source_or_HiTab_rows_accessed") is not False
        or not isinstance(pass_row, Mapping)
        or pass_row.get("formal_phase_order")
        != list(FORMAL_PHASE_ORDER)
        or pass_row.get(
            "gpu1_hippo_overlapped_gpu0_feature_formation"
        )
        is not True
        or pass_row.get(
            "gpu0_hippo_started_after_formation_and_cache_release"
        )
        is not True
        or pass_row.get("hippo_maximum_active_calls_per_GPU")
        != [1, 1]
        or pass_row.get("hippo_observed_physical_GPU_set")
        != [0, 1]
        or not isinstance(
            pass_row.get("A_form_compiled_tensor_sha256"), str
        )
        or _HEX64.fullmatch(
            str(pass_row.get("A_form_compiled_tensor_sha256"))
        )
        is None
        or pass_row.get("A_hold_compiled_tensor_sha256")
        != pass_row.get("compiled_tensor_sha256")
        or not isinstance(
            pass_row.get("A_hold_compiled_tensor_sha256"), str
        )
        or _HEX64.fullmatch(
            str(pass_row.get("A_hold_compiled_tensor_sha256"))
        )
        is None
    ):
        raise HitabP1PublicCanaryError("canary receipt binding drifted")
    return claimed


__all__ = [
    "CanaryPass",
    "FORMAL_PHASE_ORDER",
    "HitabP1PublicCanaryError",
    "PUBLIC_EXPLORATION_KEY",
    "SCHEMA",
    "SYNTHETIC_EDGES",
    "SYNTHETIC_QUESTION",
    "SYNTHETIC_UNITS",
    "VERSION",
    "run_public_canary",
    "second_synthetic_item",
    "synthetic_item",
    "validate_receipt",
]
