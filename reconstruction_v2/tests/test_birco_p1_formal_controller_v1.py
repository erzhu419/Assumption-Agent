from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import threading
import time
from typing import Mapping

import pytest

from assumption_agent.benchmarks import birco_p1_action_integration_v1 as integration
from assumption_agent.benchmarks import birco_p1_formal_controller_v1 as controller
from assumption_agent.benchmarks import birco_p1_private_selection_v1 as selection
from assumption_agent.benchmarks import birco_p1_typed_constraint_e4_core_v1 as core
from replication_runtime.birco_gpt54_semantic_v1 import contract as semantic
from replication_runtime.birco_official_hipporag_v1 import contract as hippo


def _provider() -> dict[str, object]:
    return {
        "api_key_hmac_sha256": "2" * 64,
        "api_origin": integration.PROVIDER_ORIGIN,
        "key_commitment_version": integration.KEY_COMMITMENT_VERSION,
        "model": semantic.MODEL_ID,
        "provider_label": "formal-plus",
        "secret_persisted": False,
    }


def _runtime_policy() -> dict[str, object]:
    return {
        "model_alias_cwd_relative": "runtime/model_aliases",
        "llm_model_alias": "smollm2",
        "embedding_model_alias": "minilm",
        "aliases_are_single_relative_components": True,
        "subprocess_cwd_is_model_alias_cwd": True,
        "absolute_model_path_argument_count": 0,
        "logical_slot_count": 4,
        "gpu_assignment": ["0", "1", "0", "1"],
        "maximum_processes_per_gpu": 2,
        "cpu_threads_per_process": 2,
    }


def _runtime_receipt() -> dict[str, object]:
    return {
        "model_alias_cwd_relative": "runtime/model_aliases",
        "subprocess_cwd_relative": "runtime/model_aliases",
        "llm_model_argument": "smollm2",
        "embedding_model_argument": "minilm",
        "model_arguments_are_single_relative_components": True,
        "absolute_model_path_argument_count": 0,
        "logical_slot_ordinal": 0,
        "visible_gpu": "0",
        "configured_cpu_threads": 2,
        "external_network_call_count": 0,
    }


def _action_item(block: str, ordinal: int, candidate_count: int = 10) -> dict[str, object]:
    work_id = "birco-work-v1-" + hashlib.sha256(
        f"{block}:{ordinal}".encode("ascii")
    ).hexdigest()
    objective = "Rank every candidate against the complete typed objective."
    query = f"Require alpha {ordinal}; exclude gamma."
    candidates = tuple(
        semantic.project_candidate_text(
            f"Candidate {candidate} has alpha evidence and beta detail; gamma is absent.",
            candidate_ordinal=candidate,
        )
        for candidate in range(candidate_count)
    )
    documents = [
        {"ordinal": row.ordinal, "text": row.projection_text} for row in candidates
    ]
    common = semantic.semantic_hash(
        {"documents": documents, "objective": objective, "query": query}
    )
    return {
        "schema": integration.SELECTOR_ACTION_ITEM_SCHEMA,
        "block_ordinal": ordinal,
        "work_id": work_id,
        "candidate_count": candidate_count,
        "common_projection_sha256": common,
        "hipporag_input": {
            "schema": integration.HIPPORAG_INPUT_SCHEMA,
            "work_id": work_id,
            "objective": objective,
            "query": query,
            "documents": documents,
            "common_projection_sha256": common,
        },
    }


def _action_pack(block: str, count: int = 30) -> dict[str, object]:
    body = {
        "schema": f"{selection.VERSION}_label_free_action_pack_v1",
        "version": selection.VERSION,
        "study_id": selection.STUDY_ID,
        "block": block,
        "item_count": count,
        "common_action_projection_fields": [
            "hipporag_input.objective",
            "hipporag_input.query",
            "hipporag_input.documents.ordinal",
            "hipporag_input.documents.text",
            "hipporag_input.common_projection_sha256",
        ],
        "hipporag_exact_input_field": "hipporag_input",
        "source_qid_or_candidate_id_included": False,
        "numeric_qrel_value_included": False,
        "items": [_action_item(block, ordinal) for ordinal in range(count)],
    }
    return selection.self_hashed(body, "action_pack_sha256")


def _qrel_pack(block: str, action_pack_sha256: str, count: int = 30) -> dict[str, object]:
    items = []
    for ordinal in range(count):
        items.append(
            {
                "block_ordinal": ordinal,
                "work_id": _action_item(block, ordinal)["work_id"],
                "family": selection.FAMILIES[
                    ordinal // selection.PER_FAMILY_QUOTA
                ],
                "qrel_values": [
                    {"candidate_ordinal": candidate, "value": float(candidate == 0)}
                    for candidate in range(10)
                ],
            }
        )
    return selection.self_hashed(
        {
            "schema": f"{selection.VERSION}_sealed_qrel_pack_v1",
            "version": selection.VERSION,
            "study_id": selection.STUDY_ID,
            "block": block,
            "item_count": count,
            "action_pack_sha256": action_pack_sha256,
            "source_qid_or_candidate_id_included": False,
            "numeric_qrel_values_sealed_separately": True,
            "items": items,
        },
        "qrel_pack_sha256",
    )


def _write(path: Path, value: Mapping[str, object], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(controller._canonical_bytes(value))
    os.chmod(path, mode)


def _file_binding(root: Path, relative: str, semantic_sha256: str) -> dict[str, str]:
    path = root / relative
    return {
        "relative_path": relative,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "semantic_sha256": semantic_sha256,
    }


def _freeze_fixture(root: Path) -> tuple[Path, str, str]:
    member = root / "implementation/member.txt"
    member.parent.mkdir(parents=True)
    member.write_text("frozen\n", encoding="ascii")
    implementation = controller._self_hashed(
        {
            "schema": controller.IMPLEMENTATION_FREEZE_SCHEMA,
            "study_id": selection.STUDY_ID,
            "implementation_bindings": [
                {
                    "relative_path": "implementation/member.txt",
                    "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
                }
            ],
        },
        "self_sha256",
    )
    implementation_path = root / "manifests/implementation.json"
    _write(implementation_path, implementation)

    packs: dict[str, dict[str, object]] = {}
    private_bindings: dict[str, object] = {}
    action_bindings: dict[str, object] = {}
    for block in selection.BLOCK_ORDER:
        pack = _action_pack(block)
        packs[block] = pack
        relative = f"private/{block}.actions.json"
        _write(root / relative, pack)
        binding = _file_binding(root, relative, str(pack["action_pack_sha256"]))
        action_bindings[block] = binding
        private_bindings[block] = {"action": dict(binding)}
    receipt = selection.self_hashed(
        {
            "schema": f"{selection.VERSION}_public_receipt_v1",
            "version": selection.VERSION,
            "study_id": selection.STUDY_ID,
            "status": "private_query_disjoint_four_block_selection_complete",
            "raw_qid_cid_query_document_or_qrel_value_published": False,
            "private_pack_bindings": private_bindings,
        },
        "acquisition_sha256",
    )
    receipt_path = root / "manifests/selection.json"
    _write(receipt_path, receipt)
    policy = controller.ConcurrencyPolicy(2, 2, 2, 1)
    execution = controller._self_hashed(
        {
            "schema": controller.EXECUTION_FREEZE_SCHEMA,
            "version": controller.VERSION,
            "study_id": selection.STUDY_ID,
            "implementation_freeze_binding": _file_binding(
                root, "manifests/implementation.json", str(implementation["self_sha256"])
            ),
            "selection_receipt_binding": _file_binding(
                root, "manifests/selection.json", str(receipt["acquisition_sha256"])
            ),
            "action_pack_bindings": action_bindings,
            "provider_identity": _provider(),
            "hipporag_runtime_policy": _runtime_policy(),
            "concurrency_policy": policy.payload(),
            "stage_order": list(selection.BLOCK_ORDER),
            "arms_by_block": {
                block: list(controller.BLOCK_ARMS[block])
                for block in selection.BLOCK_ORDER
            },
            "offline_only_scoring": True,
            "online_evaluator_call_count": 0,
            "retry_replay_resample_or_provider_switch_count": 0,
            "official_hipporag_commit": hippo.OFFICIAL_HIPPORAG_COMMIT,
        },
        "self_sha256",
    )
    execution_path = root / "manifests/execution.json"
    _write(execution_path, execution)
    return (
        execution_path,
        hashlib.sha256(execution_path.read_bytes()).hexdigest(),
        str(execution["self_sha256"]),
    )


def _bindings(
    packs: Mapping[str, Mapping[str, object]], policy: controller.ConcurrencyPolicy
) -> controller.PrerunBindings:
    return controller.PrerunBindings(
        execution_freeze_self_sha256="a" * 64,
        execution_freeze_file_sha256="b" * 64,
        implementation_freeze_self_sha256="c" * 64,
        selection_receipt_self_sha256="d" * 64,
        provider_identity=_provider(),
        hipporag_runtime_policy=_runtime_policy(),
        concurrency=policy,
        action_packs=packs,
        action_pack_sha256s={
            block: str(pack["action_pack_sha256"]) for block, pack in packs.items()
        },
    )


def _terminal(
    mode: str,
    expected_input: Mapping[str, object],
    action: Mapping[str, object],
) -> dict[str, object]:
    body: dict[str, object] = {
        "action": dict(action),
        "attempt_count": 1,
        "generation_valid": True,
        "input_sha256": semantic.semantic_hash(expected_input),
        "mode": mode,
        "model_request_sha256": "3" * 64,
        "provider": _provider(),
        "raw_completion_persisted": False,
        "response_sha256": "4" * 64,
        "retry_replay_resample_or_provider_switch_count": 0,
        "schema": semantic.TERMINAL_OUTPUT_SCHEMA,
        "terminal_category": "success",
        "transport": integration.SEMANTIC_TRANSPORT_ID,
        "transport_succeeded": True,
        "work_id": expected_input["work_id"],
    }
    if mode in {"matrix", "raw"}:
        for field in (
            "batch_count",
            "batch_ordinal",
            "batch_common_projection_sha256",
            "pool_candidate_count",
            "pool_common_projection_sha256",
        ):
            body[field] = expected_input[field]
    return {**body, "self_sha256": semantic.semantic_hash(body)}


class _SharedTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.plan_finished: set[str] = set()

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self.lock:
            self.current -= 1


class _SemanticFake:
    def __init__(self, tracker: _SharedTracker) -> None:
        self.tracker = tracker
        self.calls = 0

    def __call__(
        self, *, mode: str, payload: Mapping[str, object]
    ) -> Mapping[str, object]:
        self.tracker.enter()
        try:
            time.sleep(0.005)
            self.calls += 1
            work_id = str(payload["work_id"])
            if mode == "plan":
                plan = semantic.Plan(
                    facets=(
                        semantic.Facet(0, "REQUIRED", "alpha is present", 4),
                        semantic.Facet(1, "EXCLUDED", "gamma is present", 3),
                    ),
                    edges=(),
                    generation_valid=True,
                )
                with self.tracker.lock:
                    self.tracker.plan_finished.add(work_id)
                return _terminal("plan", payload, {"plan": plan.payload()})
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            if mode == "matrix":
                with self.tracker.lock:
                    assert work_id in self.tracker.plan_finished
                rows = [
                    {
                        "ordinal": row["ordinal"],
                        "rows": [[4, 0, 0], [0, 0, None]],
                    }
                    for row in candidates
                ]
                return _terminal("matrix", payload, {"matrix": rows})
            rows = [
                {"ordinal": row["ordinal"], "score": 100 - row["ordinal"]}
                for row in candidates
            ]
            return _terminal("raw", payload, {"scores": rows})
        finally:
            self.tracker.leave()


class _HippoFake:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        *,
        payload: Mapping[str, object],
        runtime_policy: Mapping[str, object],
    ) -> Mapping[str, object]:
        base = _runtime_policy()
        assert {
            key: runtime_policy[key] for key in base
        } == base
        self.calls += 1
        documents = payload["documents"]
        assert isinstance(documents, list)
        output = hippo.output_payload(
            work_id=payload["work_id"],
            common_projection_sha256=payload["common_projection_sha256"],
            candidate_count=len(documents),
            rank_ordinals=tuple(range(len(documents))),
            graph_nodes=1,
            graph_edges=0,
        )
        receipt = _runtime_receipt()
        receipt["logical_slot_ordinal"] = runtime_policy["logical_slot_ordinal"]
        receipt["visible_gpu"] = runtime_policy["visible_gpu"]
        return {"output": output, "runtime_receipt": receipt}


def _zero_model() -> core.E4Model:
    width = len(core.FEATURE_ORDER)
    return core.E4Model(
        population_mean=(0.0,) * width,
        population_std=(1.0,) * width,
        coefficients=(0.0,) * width,
        laplace_covariance=tuple((0.0,) * width for _ in range(width)),
        solver="synthetic",
        iterations=0,
        converged=True,
        objective=0.0,
    )


def _direct_controller(
    tmp_path: Path,
    bindings: controller.PrerunBindings,
    agent: object,
    raw: object,
    hipporag: object,
    opener: object,
) -> controller.FormalController:
    instance = controller.FormalController(
        project_root=tmp_path,
        control_root=tmp_path / "control",
        execution_freeze_path=tmp_path / "unused.execution.json",
        expected_execution_freeze_file_sha256="e" * 64,
        expected_execution_freeze_self_sha256="f" * 64,
        agent_executor=agent,  # type: ignore[arg-type]
        raw_executor=raw,  # type: ignore[arg-type]
        hipporag_executor=hipporag,  # type: ignore[arg-type]
        qrel_opener=opener,  # type: ignore[arg-type]
    )
    instance._bindings = bindings
    instance._api_semaphore = threading.BoundedSemaphore(
        bindings.concurrency.total_api_call_cap
    )
    return instance


def test_prerun_verifies_three_freezes_and_every_action_pack(tmp_path: Path) -> None:
    execution, file_sha, self_sha = _freeze_fixture(tmp_path)
    verified = controller.verify_prerun_freezes(
        project_root=tmp_path,
        execution_freeze_path=execution,
        expected_execution_freeze_file_sha256=file_sha,
        expected_execution_freeze_self_sha256=self_sha,
    )
    assert verified.execution_freeze_self_sha256 == self_sha
    assert verified.implementation_freeze_self_sha256
    assert set(verified.action_packs) == set(selection.BLOCK_ORDER)
    assert verified.hipporag_runtime_policy["llm_model_alias"] == "smollm2"

    (tmp_path / "implementation/member.txt").write_text("tampered\n", encoding="ascii")
    with pytest.raises(controller.BircoP1FormalControllerError, match="member drifted"):
        controller.verify_prerun_freezes(
            project_root=tmp_path,
            execution_freeze_path=execution,
            expected_execution_freeze_file_sha256=file_sha,
            expected_execution_freeze_self_sha256=self_sha,
        )


def test_concurrency_policy_has_max_useful_64_shared_and_hippo_four() -> None:
    assert controller.ConcurrencyPolicy().payload() == {
        "agent_api_workers": 64,
        "raw_api_workers": 64,
        "total_api_call_cap": 64,
        "hipporag_workers": 4,
    }
    with pytest.raises(controller.BircoP1FormalControllerError):
        controller.ConcurrencyPolicy(total_api_call_cap=65)
    with pytest.raises(controller.BircoP1FormalControllerError):
        controller.ConcurrencyPolicy(hipporag_workers=5)


def test_a_hold_three_arms_share_api_cap_and_planner_precedes_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller, "BLOCK_ITEM_COUNT", 4)
    pack = _action_pack("A_hold", count=4)
    policy = controller.ConcurrencyPolicy(8, 8, 2, 1)
    bindings = _bindings({"A_hold": pack}, policy)
    tracker = _SharedTracker()
    agent = _SemanticFake(tracker)
    raw = _SemanticFake(tracker)
    hipporag = _HippoFake()
    instance = _direct_controller(
        tmp_path,
        bindings,
        agent,
        raw,
        hipporag,
        lambda **_kwargs: pytest.fail("qrels opened while materializing actions"),
    )
    archive = instance._materialize_stage(block="A_hold", e4_model=_zero_model())
    assert archive["arms"] == ["Agent", "RAW", "HippoRAG"]
    assert archive["observed_concurrency"]["total_api_peak"] <= 2
    assert tracker.peak <= 2
    assert hipporag.calls == 4
    assert all(
        row["all_present_arms_common_projection_sha256_equal"] is True
        and row["RAW_ranking"] is not None
        and row["HippoRAG_ranking"] is not None
        for row in archive["items"]
    )
    assert stat_mode(tmp_path / "control/stages/A_hold/action_archive.json") == 0o400

    # Complete immutable stage archives are the only stage-level recovery unit.
    no_call = lambda **_kwargs: pytest.fail("executor repeated after archive recovery")
    resumed = _direct_controller(
        tmp_path, bindings, no_call, no_call, no_call, no_call
    )
    assert resumed._materialize_stage(
        block="A_hold", e4_model=_zero_model()
    )["archive_sha256"] == archive["archive_sha256"]


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def test_consumed_terminal_recovery_never_repeats_http_and_incomplete_claim_fails(
    tmp_path: Path,
) -> None:
    pack = _action_pack("A_form", count=1)
    bindings = _bindings({"A_form": pack}, controller.ConcurrencyPolicy(1, 1, 1, 1))
    tracker = _SharedTracker()
    fake = _SemanticFake(tracker)
    instance = _direct_controller(tmp_path, bindings, fake, fake, _HippoFake(), fake)
    prepared = integration.prepare_canonical_action_inputs(pack["items"][0])
    first = instance._semantic_call(
        block="A_form",
        ordinal=0,
        action_name="plan",
        role="Agent",
        mode="plan",
        payload=prepared.planner_input,
        executor=fake,
    )
    calls = fake.calls
    second = instance._semantic_call(
        block="A_form",
        ordinal=0,
        action_name="plan",
        role="Agent",
        mode="plan",
        payload=prepared.planner_input,
        executor=lambda **_kwargs: pytest.fail("HTTP repeated"),  # type: ignore[arg-type]
    )
    assert second == first
    assert fake.calls == calls

    claim, _terminal_path, _failure = instance._attempt_paths(
        "A_form", 0, "raw.000"
    )
    claim_value = controller._self_hashed(
        {
            "schema": controller.ATTEMPT_CLAIM_SCHEMA,
            "version": controller.VERSION,
            "study_id": controller.STUDY_ID,
            "block": "A_form",
            "block_ordinal": 0,
            "action_name": "raw.000",
            "role": "RAW",
            "input_sha256": semantic.semantic_hash(prepared.raw_inputs[0]),
            "attempt_count": 1,
            "retry_replay_resample_or_provider_switch_count": 0,
        },
        "claim_sha256",
    )
    controller._atomic_write_once(claim, claim_value, final_mode=0o400)
    with pytest.raises(controller.BircoP1FormalControllerError, match="consumed"):
        instance._semantic_call(
            block="A_form",
            ordinal=0,
            action_name="raw.000",
            role="RAW",
            mode="raw",
            payload=prepared.raw_inputs[0],
            executor=lambda **_kwargs: pytest.fail("consumed attempt repeated"),  # type: ignore[arg-type]
        )


def test_qrel_opener_runs_only_after_read_only_archive_and_never_for_f(
    tmp_path: Path,
) -> None:
    pack = _action_pack("A_form")
    bindings = _bindings({"A_form": pack}, controller.ConcurrencyPolicy(1, 1, 1, 1))
    seen: list[str] = []

    def opener(**kwargs: object) -> Mapping[str, object]:
        archive_path = tmp_path / "control/stages/A_form/action_archive.json"
        assert archive_path.is_file()
        assert stat_mode(archive_path) == 0o400
        seen.append(str(kwargs["block"]))
        return _qrel_pack("A_form", str(pack["action_pack_sha256"]))

    instance = _direct_controller(tmp_path, bindings, opener, opener, opener, opener)
    archive = controller._self_hashed(
        {
            "schema": controller.STAGE_ARCHIVE_SCHEMA,
            "version": controller.VERSION,
            "study_id": controller.STUDY_ID,
            "block": "A_form",
            "items": [],
        },
        "archive_sha256",
    )
    controller._seal_read_only(
        tmp_path / "control/stages/A_form/action_archive.json",
        archive,
        "archive_sha256",
    )
    assert instance._qrels_after_archive(
        block="A_form", archive=archive
    )["block"] == "A_form"
    assert seen == ["A_form"]
    with pytest.raises(controller.BircoP1FormalControllerError, match="permanently"):
        instance._qrels_after_archive(block="F_search", archive=archive)
    assert seen == ["A_form"]


def _promotion_receipt(
    archive_sha: str, f_sha: str
) -> tuple[dict[str, object], core.RealityPrimaryDecision, core.E4PromotionDecision]:
    ones = [1] * 30
    zeros = [0] * 30
    families = [family for family in selection.FAMILIES for _ in range(10)]
    reality = core.decide_a_hold_reality_primary(ones, zeros, zeros, families)
    promotion = core.decide_a_hold_e4_promotion(
        ones, zeros, f_identifiability_passed=True
    )
    value = controller._self_hashed(
        {
            "schema": f"{controller.VERSION}_A_hold_score_and_promotion_v1",
            "version": controller.VERSION,
            "study_id": controller.STUDY_ID,
            "status": "offline_A_hold_reality_and_single_challenger_promotion_complete",
            "A_hold_action_archive_sha256": archive_sha,
            "A_hold_qrel_pack_sha256": "1" * 64,
            "F_search_identifiability_receipt_sha256": f_sha,
            "reality_primary": {
                "agent_minus_RAW": reality.agent_minus_raw.payload(),
                "agent_minus_HippoRAG": reality.agent_minus_hipporag.payload(),
                "RAW_family_integer_deltas": list(reality.raw_family_integer_deltas),
                "HippoRAG_family_integer_deltas": list(
                    reality.hipporag_family_integer_deltas
                ),
                "passed": reality.passed,
            },
            "E4_promotion": promotion.payload(),
            "online_evaluator_call_count": 0,
        },
        "promotion_receipt_sha256",
    )
    return value, reality, promotion


def test_crash_resume_reuses_a_hold_score_without_second_qrel_open(tmp_path: Path) -> None:
    pack = _action_pack("A_hold")
    bindings = _bindings({"A_hold": pack}, controller.ConcurrencyPolicy(1, 1, 1, 1))
    instance = _direct_controller(
        tmp_path,
        bindings,
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: pytest.fail("A_hold qrels reopened"),
    )
    archive = {"archive_sha256": "a" * 64}
    f_sha = "b" * 64
    receipt, expected_reality, expected_promotion = _promotion_receipt(
        str(archive["archive_sha256"]), f_sha
    )
    _write(tmp_path / "control/A_hold_score_and_promotion.json", receipt, mode=0o400)
    reality, promotion, receipt_sha = instance._score_a_hold(
        archive, f_passed=True, f_receipt_sha256=f_sha
    )
    assert reality == expected_reality
    assert promotion == expected_promotion
    assert receipt_sha == receipt["promotion_receipt_sha256"]


def test_crash_resume_reuses_a_form_model_without_second_qrel_open(tmp_path: Path) -> None:
    pack = _action_pack("A_form")
    bindings = _bindings({"A_form": pack}, controller.ConcurrencyPolicy(1, 1, 1, 1))
    instance = _direct_controller(
        tmp_path,
        bindings,
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: pytest.fail("A_form qrels reopened"),
    )
    archive = {"archive_sha256": "a" * 64}
    model = _zero_model()
    receipt = controller._self_hashed(
        {
            "schema": f"{controller.VERSION}_A_form_e4_model_v1",
            "version": controller.VERSION,
            "study_id": controller.STUDY_ID,
            "status": "single_E4_fit_after_immutable_A_form_actions",
            "A_form_action_archive_sha256": archive["archive_sha256"],
            "A_form_qrel_pack_sha256": "1" * 64,
            "training_slate_count": 30,
            "model": controller._model_payload(model),
            "online_evaluator_call_count": 0,
        },
        "model_receipt_sha256",
    )
    _write(tmp_path / "control/A_form_e4_model.json", receipt, mode=0o400)
    observed, receipt_sha = instance._load_or_fit_e4(archive)
    assert observed == model
    assert receipt_sha == receipt["model_receipt_sha256"]


def test_crash_resume_reuses_m_search_score_without_second_qrel_open(tmp_path: Path) -> None:
    pack = _action_pack("M_search")
    bindings = _bindings({"M_search": pack}, controller.ConcurrencyPolicy(1, 1, 1, 1))
    instance = _direct_controller(
        tmp_path,
        bindings,
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: pytest.fail("M_search qrels reopened"),
    )
    ones = [1] * 30
    zeros = [0] * 30
    families = [family for family in selection.FAMILIES for _ in range(10)]
    decision = core.decide_m_search_e4_improvement(ones, zeros, families)
    archive = {"archive_sha256": "a" * 64}
    promotion_sha = "b" * 64
    receipt = controller._self_hashed(
        {
            "schema": f"{controller.VERSION}_M_search_score_v1",
            "version": controller.VERSION,
            "study_id": controller.STUDY_ID,
            "status": "offline_untouched_M_search_complete",
            "M_search_action_archive_sha256": archive["archive_sha256"],
            "M_search_qrel_pack_sha256": "1" * 64,
            "A_hold_promotion_receipt_sha256": promotion_sha,
            "comparison": decision.comparison.payload(),
            "family_integer_deltas": list(decision.family_integer_deltas),
            "passed": decision.passed,
            "online_evaluator_call_count": 0,
        },
        "score_receipt_sha256",
    )
    _write(tmp_path / "control/M_search_score.json", receipt, mode=0o400)
    observed, receipt_sha = instance._score_m_search(
        archive, promotion_sha256=promotion_sha
    )
    assert observed == decision
    assert receipt_sha == receipt["score_receipt_sha256"]


def test_existing_f_receipt_checks_all_bindings_and_never_opens_qrels(
    tmp_path: Path,
) -> None:
    pack = _action_pack("F_search")
    bindings = _bindings({"F_search": pack}, controller.ConcurrencyPolicy(1, 1, 1, 1))
    instance = _direct_controller(
        tmp_path,
        bindings,
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: pytest.fail("F qrels opened"),
    )
    archive = {"archive_sha256": "a" * 64}
    model_sha = "b" * 64
    receipt = controller._self_hashed(
        {
            "schema": f"{controller.VERSION}_F_search_identifiability_v1",
            "version": controller.VERSION,
            "study_id": controller.STUDY_ID,
            "status": "label_free_permutation_identifiability_complete",
            "F_search_action_archive_sha256": archive["archive_sha256"],
            "A_form_model_receipt_sha256": model_sha,
            "item_count": 30,
            "differing_ranking_count": 3,
            "differing_family_count": 2,
            "passed": True,
            "F_search_qrel_open_count": 0,
        },
        "receipt_sha256",
    )
    path = tmp_path / "control/F_search_identifiability.json"
    _write(path, receipt, mode=0o400)
    result, _sha = instance._f_identifiability(archive, model_sha)
    assert result.passed

    forged = copy.deepcopy(receipt)
    forged["F_search_qrel_open_count"] = 1
    forged.pop("receipt_sha256")
    forged = controller._self_hashed(forged, "receipt_sha256")
    os.chmod(path, 0o600)
    _write(path, forged, mode=0o400)
    with pytest.raises(controller.BircoP1FormalControllerError, match="drifted"):
        instance._f_identifiability(archive, model_sha)


def test_lifecycle_stops_after_unidentifiable_f_without_a_hold_or_m(
    tmp_path: Path,
) -> None:
    pack = _action_pack("A_form")
    bindings = _bindings({"A_form": pack}, controller.ConcurrencyPolicy(1, 1, 1, 1))
    instance = _direct_controller(
        tmp_path,
        bindings,
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: {},
        lambda **_kwargs: pytest.fail("qrels unexpectedly opened"),
    )
    calls: list[str] = []
    instance._verify_before_execution = lambda: bindings  # type: ignore[method-assign]

    def materialize(*, block: str, e4_model: object) -> Mapping[str, object]:
        calls.append(block)
        return {"archive_sha256": hashlib.sha256(block.encode()).hexdigest()}

    instance._materialize_stage = materialize  # type: ignore[method-assign]
    instance._load_or_fit_e4 = lambda _archive: (_zero_model(), "1" * 64)  # type: ignore[method-assign]
    instance._f_identifiability = lambda _archive, _model: (  # type: ignore[method-assign]
        core.FIdentifiabilityResult(30, 0, 0, False),
        "2" * 64,
    )
    final = instance.run()
    assert calls == ["A_form", "F_search"]
    assert final["status"] == "terminal_F_search_label_free_unidentifiable"
    assert final["F_search_qrel_open_count"] == 0
