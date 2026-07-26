from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading

import numpy as np
import pytest

from assumption_agent.benchmarks import hitab_p1_public_canary_v1 as canary
from assumption_agent.benchmarks import hitab_p1_runtime_v1 as runtime
from replication_runtime.birco_official_hipporag_v1 import contract as hippo_contract
from replication_runtime.bright_query_generator_v1 import contract as planner_contract


class _Planner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, canonical_input: bytes) -> bytes:
        items = planner_contract.parse_input(canonical_input)
        self.calls += 1
        completion = json.dumps(
            {
                "entity_query": "North and South synthetic regions",
                "relation_query": "compare displayed renewable shares",
                "mechanism_query": "identify the larger displayed percentage",
                "constraint_query": "use the synthetic 2024 cells",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        row = planner_contract.build_output_item(
            ordinal=0,
            completion=completion,
            completion_token_count=28,
            query=items[0].query,
        )
        return planner_contract.canonical_json_bytes(
            planner_contract.output_payload((row,))
        )


class _Scorer:
    def __init__(self, *, drift: bool = False) -> None:
        self.calls = 0
        self.drift = drift

    def __call__(self, pairs):
        rows = tuple(pairs)
        offset = self.calls if self.drift else 0
        self.calls += 1
        return tuple(
            (index % len(canary.SYNTHETIC_UNITS)) / 3.0
            + offset * (index + 1) / 100.0
            for index in range(len(rows))
        )


def _vector(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    row = np.frombuffer(digest * 12, dtype=np.uint8)[:384].astype(np.float32)
    row -= np.float32(127.5)
    row /= np.float32(np.linalg.norm(row.astype(np.float64)))
    return row


class _Encoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.stack([_vector(value) for value in texts]).astype(np.float32)


class _CacheReleaser:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        body = {
            "model_offload_or_reload": False,
            "physical_gpu": 0,
            "schema": "hitab_p1_gpu0_unused_cuda_cache_release_v1",
            "study_id": canary.core.STUDY_ID,
            "torch_cuda_empty_cache_called": True,
        }
        return {**body, "self_sha256": runtime.stable_hash(body)}


class _Hippo:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.barrier = threading.Barrier(2, timeout=5)
        self.lock = threading.Lock()
        self.active_total = 0
        self.active_by_gpu = {0: 0, 1: 0}
        self.maximum_active_total = 0
        self.maximum_active_by_gpu = {0: 0, 1: 0}

    def __call__(
        self,
        canonical_input: bytes,
        *,
        physical_gpu: int,
        cpu_thread_limit: int,
        launch_ack,
    ) -> bytes:
        value = json.loads(canonical_input.decode("ascii"))
        validated = hippo_contract.validate_input(
            value["work_id"],
            value["objective"],
            value["query"],
            value["documents"],
            value["common_projection_sha256"],
        )
        with self.lock:
            self.calls.append((physical_gpu, cpu_thread_limit))
            self.active_total += 1
            self.active_by_gpu[physical_gpu] += 1
            self.maximum_active_total = max(
                self.maximum_active_total, self.active_total
            )
            self.maximum_active_by_gpu[physical_gpu] = max(
                self.maximum_active_by_gpu[physical_gpu],
                self.active_by_gpu[physical_gpu],
            )
        try:
            launch_ack()
            self.barrier.wait()
            count = len(validated[3])
            payload = hippo_contract.output_payload(
                work_id=validated[0],
                common_projection_sha256=validated[4],
                candidate_count=count,
                rank_ordinals=tuple(range(count)),
                graph_nodes=12,
                graph_edges=11,
            )
            return hippo_contract.canonical_json_bytes(payload)
        finally:
            with self.lock:
                self.active_total -= 1
                self.active_by_gpu[physical_gpu] -= 1


def test_public_canary_crosses_complete_late_qrel_path_twice() -> None:
    planner = _Planner()
    scorer = _Scorer()
    encoder = _Encoder()
    hippo = _Hippo()
    cache = _CacheReleaser()
    receipt = canary.run_public_canary(
        planner_runner=planner,
        cross_encoder_scorer=scorer,
        minilm_encoder=encoder,
        hippo_runner=hippo,
        gpu0_cache_releaser=cache,
    )
    assert receipt["qualified"] is True
    assert receipt["repeat_exact"] is True
    assert receipt["behavior_or_efficacy_gate"] is False
    assert receipt["E1_minus_E0_nonzero_required"] is False
    assert receipt["E1_outside_RAW_required"] is False
    assert receipt["residual_nonzero_required"] is False
    assert receipt["source_or_HiTab_rows_accessed"] is False
    assert receipt["pass"]["qrel_opened_after_seal"] is True
    assert receipt["pass"]["prelabel_archive_contains_no_qrel"] is True
    assert receipt["pass"]["four_arm_corpus_commitment_exact"] is True
    assert all(
        len(receipt["pass"][field]) == 5
        for field in ("RAW_top5", "E0_top5", "E1_top5")
    )
    assert len(receipt["pass"]["HippoRAG_top5"]) == 2
    assert all(len(row) == 5 for row in receipt["pass"]["HippoRAG_top5"])
    assert receipt["pass"]["hippo_observed_physical_GPU_set"] == [0, 1]
    assert receipt["pass"]["formal_phase_order"] == list(
        canary.FORMAL_PHASE_ORDER
    )
    assert (
        receipt["pass"][
            "gpu1_hippo_overlapped_gpu0_feature_formation"
        ]
        is True
    )
    assert (
        receipt["pass"][
            "gpu0_hippo_started_after_formation_and_cache_release"
        ]
        is True
    )
    assert receipt["pass"]["hippo_maximum_active_calls_per_GPU"] == [1, 1]
    assert len(set(receipt["pass"]["hippo_input_sha256s"])) == 2
    assert planner.calls == scorer.calls == encoder.calls == 4
    assert len(hippo.calls) == 4
    assert {row[0] for row in hippo.calls} == {0, 1}
    assert all(row[1] == 4 for row in hippo.calls)
    assert hippo.maximum_active_total == 2
    assert hippo.maximum_active_by_gpu == {0: 1, 1: 1}
    assert cache.calls == 2
    assert len(receipt["pass"]["A_form_compiled_tensor_sha256"]) == 64
    assert len(receipt["pass"]["A_hold_compiled_tensor_sha256"]) == 64
    assert canary.validate_receipt(receipt) == receipt["self_sha256"]


class _PhaseWitness:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[str] = []
        self.gpu1_active = False
        self.formation_observed = False
        self.cache_released = False
        self.gpu0_started = False


class _PhaseHippo(_Hippo):
    def __init__(self, witness: _PhaseWitness) -> None:
        super().__init__()
        self.witness = witness

    def __call__(
        self,
        canonical_input: bytes,
        *,
        physical_gpu: int,
        cpu_thread_limit: int,
        launch_ack,
    ) -> bytes:
        def phase_launch_ack() -> None:
            with self.witness.lock:
                if physical_gpu == 1:
                    self.witness.gpu1_active = True
                    self.witness.formation_observed = False
                    self.witness.cache_released = False
                    self.witness.gpu0_started = False
                    self.witness.events.append("gpu1_begin")
                else:
                    assert self.witness.formation_observed
                    assert self.witness.cache_released
                    self.witness.gpu0_started = True
                    self.witness.events.append("gpu0_begin")
            launch_ack()

        try:
            return super().__call__(
                canonical_input,
                physical_gpu=physical_gpu,
                cpu_thread_limit=cpu_thread_limit,
                launch_ack=phase_launch_ack,
            )
        finally:
            if physical_gpu == 1:
                with self.witness.lock:
                    self.witness.gpu1_active = False


class _PhasePlanner(_Planner):
    def __init__(self, witness: _PhaseWitness) -> None:
        super().__init__()
        self.witness = witness

    def __call__(self, canonical_input: bytes) -> bytes:
        with self.witness.lock:
            if self.calls % 2 == 0:
                # A_form is completed before either Hippo lane exists.
                assert not self.witness.gpu1_active
            else:
                assert self.witness.gpu1_active
                assert not self.witness.gpu0_started
                self.witness.formation_observed = True
                self.witness.events.append("formation")
        return super().__call__(canonical_input)


class _PhaseCache(_CacheReleaser):
    def __init__(self, witness: _PhaseWitness) -> None:
        super().__init__()
        self.witness = witness

    def __call__(self):
        with self.witness.lock:
            assert self.witness.gpu1_active
            assert self.witness.formation_observed
            assert not self.witness.gpu0_started
            self.witness.cache_released = True
            self.witness.events.append("cache_release")
        return super().__call__()


def test_public_canary_mirrors_formal_gpu_phase_order() -> None:
    witness = _PhaseWitness()
    hippo = _PhaseHippo(witness)
    receipt = canary.run_public_canary(
        planner_runner=_PhasePlanner(witness),
        cross_encoder_scorer=_Scorer(),
        minilm_encoder=_Encoder(),
        hippo_runner=hippo,
        gpu0_cache_releaser=_PhaseCache(witness),
    )
    assert witness.events == [
        "gpu1_begin",
        "formation",
        "cache_release",
        "gpu0_begin",
        "gpu1_begin",
        "formation",
        "cache_release",
        "gpu0_begin",
    ]
    assert hippo.maximum_active_by_gpu == {0: 1, 1: 1}
    assert receipt["pass"]["formal_phase_order"] == list(
        canary.FORMAL_PHASE_ORDER
    )


def test_public_canary_waits_for_real_gpu1_launch_ack_before_formation() -> None:
    class DelayedLaunchHippo(_Hippo):
        def __init__(self) -> None:
            super().__init__()
            self.before_gpu1_ack = threading.Event()
            self.release_gpu1_ack = threading.Event()

        def __call__(
            self,
            canonical_input: bytes,
            *,
            physical_gpu: int,
            cpu_thread_limit: int,
            launch_ack,
        ) -> bytes:
            if physical_gpu == 1 and not self.release_gpu1_ack.is_set():
                self.before_gpu1_ack.set()
                self.release_gpu1_ack.wait()
            return super().__call__(
                canonical_input,
                physical_gpu=physical_gpu,
                cpu_thread_limit=cpu_thread_limit,
                launch_ack=launch_ack,
            )

    hippo = DelayedLaunchHippo()
    cache = _CacheReleaser()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            canary.run_public_canary,
            planner_runner=_Planner(),
            cross_encoder_scorer=_Scorer(),
            minilm_encoder=_Encoder(),
            hippo_runner=hippo,
            gpu0_cache_releaser=cache,
        )
        assert hippo.before_gpu1_ack.wait(5)
        assert cache.calls == 0
        assert not future.done()
        hippo.release_gpu1_ack.set()
        receipt = future.result(timeout=20)
    assert receipt["qualified"] is True
    assert cache.calls == 2


def test_public_canary_pre_ack_failure_wakes_without_false_overlap() -> None:
    class PreAckFailure:
        def __call__(self, _canonical_input: bytes, **_kwargs) -> bytes:
            raise RuntimeError("synthetic failure before launch ack")

    with pytest.raises(
        canary.HitabP1PublicCanaryError,
        match="terminated before launch acknowledgement",
    ):
        canary.run_public_canary(
            planner_runner=_Planner(),
            cross_encoder_scorer=_Scorer(),
            minilm_encoder=_Encoder(),
            hippo_runner=PreAckFailure(),
            gpu0_cache_releaser=_CacheReleaser(),
        )


def test_aform_registry_qrel_label_and_fit_finish_before_any_hippo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hippo = _Hippo()
    events: list[str] = []
    pass_baselines: list[int] = []
    fitted_models: list[object] = []
    registries: list[object] = []
    original_build = canary.core.build_and_seal_aform_registry
    original_qrel = canary._late_synthetic_qrel
    original_label = canary.core.label_sealed_registry
    original_fit = canary.core.fit_e1
    original_select = canary.core.select_e1

    def build_registry(view, *, exploration_key):
        pass_baselines.append(len(hippo.calls))
        registry = original_build(
            view, exploration_key=exploration_key
        )
        assert registry.seal_sha256
        registries.append(registry)
        events.append("registry_sealed")
        return registry

    def open_late_qrel():
        assert len(hippo.calls) == pass_baselines[-1]
        assert registries[-1].seal_sha256
        events.append("late_qrel")
        return original_qrel()

    def label_registry(registry, proof):
        assert registry is registries[-1]
        assert len(hippo.calls) == pass_baselines[-1]
        events.append("label")
        return original_label(registry, proof)

    def fit_model(labelled):
        assert len(hippo.calls) == pass_baselines[-1]
        model = original_fit(labelled)
        fitted_models.append(model)
        events.append("fit")
        return model

    def select_with_fitted_model(view, model):
        assert model is fitted_models[-1]
        events.append("A_hold_E1")
        return original_select(view, model)

    monkeypatch.setattr(
        canary.core, "build_and_seal_aform_registry", build_registry
    )
    monkeypatch.setattr(canary, "_late_synthetic_qrel", open_late_qrel)
    monkeypatch.setattr(
        canary.core, "label_sealed_registry", label_registry
    )
    monkeypatch.setattr(canary.core, "fit_e1", fit_model)
    monkeypatch.setattr(canary.core, "select_e1", select_with_fitted_model)
    receipt = canary.run_public_canary(
        planner_runner=_Planner(),
        cross_encoder_scorer=_Scorer(),
        minilm_encoder=_Encoder(),
        hippo_runner=hippo,
        gpu0_cache_releaser=_CacheReleaser(),
    )
    assert receipt["qualified"] is True
    assert pass_baselines == [0, 2]
    assert events == [
        "registry_sealed",
        "late_qrel",
        "label",
        "fit",
        "A_hold_E1",
        "registry_sealed",
        "late_qrel",
        "label",
        "fit",
        "A_hold_E1",
    ]


def test_public_canary_rejects_any_repeat_drift() -> None:
    with pytest.raises(
        canary.HitabP1PublicCanaryError, match="not exact-repeat"
    ):
        canary.run_public_canary(
            planner_runner=_Planner(),
            cross_encoder_scorer=_Scorer(drift=True),
            minilm_encoder=_Encoder(),
            hippo_runner=_Hippo(),
            gpu0_cache_releaser=_CacheReleaser(),
        )


def test_receipt_self_hash_fails_closed() -> None:
    receipt = canary.run_public_canary(
        planner_runner=_Planner(),
        cross_encoder_scorer=_Scorer(),
        minilm_encoder=_Encoder(),
        hippo_runner=_Hippo(),
        gpu0_cache_releaser=_CacheReleaser(),
    )
    receipt["qualified"] = False
    with pytest.raises(
        canary.HitabP1PublicCanaryError, match="binding drifted"
    ):
        canary.validate_receipt(receipt)
