from __future__ import annotations

from fractions import Fraction
import hashlib
import inspect
import itertools
import json
from pathlib import Path

import pytest

import assumption_agent.benchmarks.qasc_evaluator_direct_action_coevolution_v1 as runner


def _brute_sign_flip(deltas: list[int]) -> Fraction:
    magnitudes = [abs(value) for value in deltas if value]
    observed = sum(deltas)
    values = [
        sum(sign * magnitude for sign, magnitude in zip(signs, magnitudes))
        for signs in itertools.product((-1, 1), repeat=len(magnitudes))
    ]
    return Fraction(sum(value >= observed for value in values), len(values))


@pytest.mark.parametrize(
    "deltas",
    (
        [0, 0, 0],
        [1, 1, 1, 1, 1],
        [3, -1, 2, -3, 0, 1],
        [-3, -1, 0, 1],
    ),
)
def test_exact_magnitude_sign_flip_matches_brute_force(deltas: list[int]) -> None:
    result = runner.exact_magnitude_preserving_sign_flip(deltas)
    expected = _brute_sign_flip(deltas)
    assert (result["p_value_numerator"], result["p_value_denominator"]) == (
        expected.numerator,
        expected.denominator,
    )
    assert result["promoted"] is (
        sum(deltas) > 0 and expected <= Fraction(1, 10)
    )


def test_item_utility_and_paired_comparison_use_U_not_hits_only() -> None:
    gold = [(0, 1), (0, 1), (0, 1)]
    incumbent = [(0, 2, 3, 4, 5), (2, 3, 4, 5, 6), (0, 1, 2, 3, 4)]
    challenger = [(0, 1, 2, 3, 4), (0, 2, 3, 4, 5), (0, 2, 3, 4, 5)]
    assert runner.item_utility(incumbent[2], gold[2]) == (2, 1, 3)
    comparison = runner.paired_utility_comparison(
        left_arm_id="challenger",
        right_arm_id="incumbent",
        left_rankings=challenger,
        right_rankings=incumbent,
        gold_rows=gold,
        confirmatory=True,
    )
    # Per-item U deltas are +2, +1, -2.
    assert comparison["net_U"] == 1
    assert comparison["paired_delta_vector_sha256"] == runner.stable_hash([2, 1, -2])


def test_descriptive_comparison_can_never_promote() -> None:
    gold = [(0, 1)] * 5
    winning = [(0, 1, 2, 3, 4)] * 5
    losing = [(2, 3, 4, 5, 6)] * 5
    comparison = runner.paired_utility_comparison(
        left_arm_id="control",
        right_arm_id="baseline",
        left_rankings=winning,
        right_rankings=losing,
        gold_rows=gold,
        confirmatory=False,
    )
    assert comparison["paired_test"]["promoted"] is False
    assert comparison["paired_test"]["descriptive_positive_and_p_at_or_below_alpha"] is True


def test_epoch_transition_is_selective_and_M_is_conditional() -> None:
    anchor = "a" * 64
    retained = runner.evaluator_epoch_transition(
        a_decision_sha256=anchor, promoted=False
    )
    promoted = runner.evaluator_epoch_transition(
        a_decision_sha256=anchor, promoted=True
    )
    assert retained["next_epoch_id"] == retained["previous_epoch_id"]
    assert retained["M_search_open_authorized"] is False
    assert promoted["next_epoch_index"] == 1
    assert promoted["dependent_evaluator_scores_invalidated"] is True
    assert promoted["independent_source_actions_retained"] is True
    assert promoted["M_search_can_rollback_epoch"] is False


def test_public_payload_rejects_rows_labels_and_host_paths() -> None:
    runner._assert_public_safe(
        {
            "private_evidence_binding": {"file_sha256": "b" * 64},
            "raw_content_persisted": False,
        }
    )
    for unsafe in (
        {"answerKey": "A"},
        {"documents": []},
        {"nested": {"view_sha256": "c" * 64}},
        {"runtime": "/tmp/private-runtime"},
    ):
        with pytest.raises(runner.QASCCoevolutionError):
            runner._assert_public_safe(unsafe)


def test_atomic_exclusive_persistence_has_no_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    runner._write_json_exclusive(
        target,
        {"schema": "synthetic", "raw_content_persisted": False},
        public=True,
    )
    first = target.read_bytes()
    with pytest.raises(runner.QASCCoevolutionError, match="not fresh"):
        runner._write_json_exclusive(
            target,
            {"schema": "replacement", "raw_content_persisted": False},
            public=True,
        )
    assert target.read_bytes() == first


def test_control_failures_are_item_local_and_primary_independent() -> None:
    views = tuple({"ordinal": index} for index in range(4))

    def control(index: int, _view: object) -> tuple[int, ...]:
        if index in {1, 3}:
            raise RuntimeError("synthetic isolated control failure")
        return (0, 1, 2, 3, 4)

    result = runner._run_failure_isolated_control(
        control_id="synthetic",
        views=views,
        function=control,
        maximum_workers=4,
    )
    assert result.status == "unavailable_or_partial"
    assert result.rankings == (
        (0, 1, 2, 3, 4),
        None,
        (0, 1, 2, 3, 4),
        None,
    )
    summary = result.public_summary()
    assert summary["failed_item_count"] == 2
    assert summary["affects_primary_or_epoch"] is False


def test_official_control_uses_qualified_maximum_concurrency() -> None:
    assert runner.OFFICIAL_CONCURRENCY_CAP == 8


def test_view_hash_boundary_rejects_identity_or_label_proxy() -> None:
    safe = {
        "schema": "view",
        "block": "A_hold",
        "source_member": "TRAIN",
        "formatted_question": "Synthetic?",
        "choices": [],
        "documents": [],
        "raw_ranking": [],
    }
    assert runner._view_item_key(safe) == runner.stable_hash(safe)
    for field in ("identity_commitment_sha256", "label_envelope_sha256"):
        with pytest.raises(runner.QASCCoevolutionError, match="label proxy"):
            runner._view_item_key({**safe, field: "d" * 64})


def test_real_recipe_view_runs_through_exact_two_wave_pool_interface() -> None:
    recipe = runner._recipe_module()
    mapping = runner._synthetic_view_mapping()
    view = recipe.load_retrieval_view(mapping)

    class FakePool:
        calls = 0

        def score_items(self, items):
            self.calls += 1
            return {
                key: tuple(
                    int.from_bytes(
                        hashlib.sha256(f"{premise}\0{hypothesis}".encode()).digest()[:4],
                        "big",
                        signed=False,
                    )
                    - 2**31
                    for premise, hypothesis in pairs
                )
                for key, pairs in items
            }

    pool = FakePool()
    actions, execution = runner._score_recipe_views_two_waves(
        views=(view,), recipe_ids=None, pool=pool
    )
    assert pool.calls == 2
    assert execution["two_score_waves_exact"] is True
    assert execution["recipe_action_terminal_count"] == 16
    terminal = actions[view.view_sha256]
    assert len(terminal) == 16
    assert all(len(action.ordered_top5) == 5 for action in terminal)
    query, paragraphs = runner._official_inputs(view)
    assert "[CHOICES]" in query
    assert len(paragraphs) == 32
    assert paragraphs[0]["paragraph_text"] == mapping["documents"][0]["text"]


def test_formal_surface_and_cli_cover_all_frozen_stages(tmp_path: Path) -> None:
    assert runner.formal_signatures_have_no_injection_surface() is True
    parser = runner._parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(subparsers.choices) == {
        "diagnose",
        "freeze-formation",
        "run-formation",
        "verify-formation",
        "freeze-a",
        "run-a",
        "verify-a",
        "freeze-m",
        "run-m",
        "verify-m",
    }
    runner._CLEAN_MODULE_CLI_ACTIVE = False
    for function in (
        runner.execute_formation,
        runner.execute_a_hold,
        runner.execute_m_search,
    ):
        arguments = {
            name: tmp_path / name
            for name in inspect.signature(function).parameters
        }
        with pytest.raises(runner.QASCCoevolutionError, match="clean CLI"):
            function(**arguments)


def test_control_orchestration_exception_is_isolated_but_interrupt_propagates() -> None:
    class FailedFuture:
        def result(self):
            raise RuntimeError("synthetic control executor failure")

    outcomes, postflight = runner._join_controls_failure_isolated(
        FailedFuture(), item_count=64
    )
    assert set(outcomes) == {
        "canonical_RAW",
        "retained_P",
        "official_HippoRAG_item_local_32",
    }
    assert all(outcome.rankings == (None,) * 64 for outcome in outcomes.values())
    assert postflight["passed"] is False

    class InterruptedFuture:
        def result(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner._join_controls_failure_isolated(InterruptedFuture(), item_count=64)


def _valid_measurement_execution(block: str) -> dict[str, object]:
    postflight = {
        "passed": False,
        "failure_type_sha256": "f" * 64,
    }
    execution: dict[str, object] = {
        "view_count": 64,
        "recipe_count_per_view": 2,
        "first_wave_actual_NLI_pair_count": 100,
        "first_wave_conceptual_request_count": 32768,
        "first_wave_item_terminal_count": 64,
        "second_wave_actual_NLI_pair_count": 200,
        "second_wave_conceptual_request_count": 126976,
        "second_wave_item_terminal_count": 64,
        "recipe_action_terminal_count": 128,
        "all_first_wave_items_submitted_before_first_wave_join": True,
        "second_wave_built_only_after_complete_first_wave_join": True,
        "all_second_wave_items_submitted_before_second_wave_join": True,
        "labels_loaded_or_scored": False,
        "two_score_waves_exact": True,
        "block": block,
        "item_count": 64,
        "NLI_worker_count": 8,
        "torch_threads_per_worker": 4,
        "NLI_postflight_before_label_open": True,
        "all_primary_action_terminals_before_label_open": True,
        "control_actions_joined_before_label_open": True,
        "label_rows_opened_after_terminals": 64,
        "official_postflight": postflight,
        "controls_failure_isolated": True,
        "network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }
    if block == "A_hold":
        execution["M_search_view_or_label_rows_opened"] = 0
    return execution


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("network_calls", 1),
        ("retries", 1),
        ("first_wave_conceptual_request_count", 32767),
        ("NLI_worker_count", 7),
        ("M_search_view_or_label_rows_opened", 1),
    ),
)
def test_measurement_execution_rejects_protocol_drift(
    field: str, bad_value: object
) -> None:
    postflight = {"passed": False, "failure_type_sha256": "f" * 64}
    valid = _valid_measurement_execution("A_hold")
    runner._verify_measurement_execution(
        execution=valid,
        block="A_hold",
        official_postflight=postflight,
    )
    changed = {**valid, field: bad_value}
    with pytest.raises(runner.QASCCoevolutionError):
        runner._verify_measurement_execution(
            execution=changed,
            block="A_hold",
            official_postflight=postflight,
        )


def _measurement_evidence(recipe_ids: tuple[str, str]) -> dict[str, object]:
    rows = []
    for ordinal in range(64):
        identity = runner.stable_hash({"identity": ordinal})
        view = runner.stable_hash({"view": ordinal})

        def scored(recipe_id: str) -> dict[str, object]:
            return {
                "identity_commitment_sha256": identity,
                "view_sha256": view,
                "recipe_id": recipe_id,
                "invalid": False,
                "support_hits_at_5": 2,
                "complete": True,
                "U": 3,
                "auc2": 0,
                "top1": False,
                "gold_pair": False,
                "ordered_top5": [0, 1, 2, 3, 4],
                "action_sha256": runner.stable_hash(
                    {"ordinal": ordinal, "recipe_id": recipe_id}
                ),
            }

        rows.append(
            {
                "identity_commitment_sha256": identity,
                "view_sha256": view,
                "gold_document_ids": [0, 1],
                "incumbent": scored(recipe_ids[0]),
                "challenger": scored(recipe_ids[1]),
            }
        )
    available = runner.ControlOutcome(
        control_id="canonical_RAW",
        status="available",
        rankings=((0, 1, 2, 3, 4),) * 64,
        failure_type_hashes=(None,) * 64,
    )
    retained = runner.ControlOutcome(
        control_id="retained_P",
        status="available",
        rankings=((0, 1, 2, 3, 4),) * 64,
        failure_type_hashes=(None,) * 64,
    )
    official = runner.ControlOutcome(
        control_id="official_HippoRAG_item_local_32",
        status="unavailable_or_partial",
        rankings=(None,) * 64,
        failure_type_hashes=("e" * 64,) * 64,
    )
    return runner._measurement_private_evidence(
        schema=f"{runner.VERSION}_A_hold_private_evidence",
        freeze_sha256="a" * 64,
        block="A_hold",
        primary_rows=rows,
        controls={
            "canonical_RAW": available,
            "retained_P": retained,
            "official_HippoRAG_item_local_32": official,
        },
        official_postflight={
            "passed": False,
            "failure_type_sha256": "e" * 64,
        },
    )


def test_private_evidence_rejects_selected_recipe_swap_and_postflight_drift(
    tmp_path: Path,
) -> None:
    registry = tuple(row.recipe_id for row in runner._recipe_module().recipe_registry())
    recipe_ids = (registry[0], registry[1])
    evidence = _measurement_evidence(recipe_ids)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    runner._load_measurement_evidence(
        path=path,
        field="synthetic evidence",
        schema=f"{runner.VERSION}_A_hold_private_evidence",
        freeze_sha256="a" * 64,
        block="A_hold",
        expected_incumbent_recipe_id=recipe_ids[0],
        expected_challenger_recipe_id=recipe_ids[1],
    )

    tampered = json.loads(json.dumps(evidence))
    tampered["primary_rows"][0]["incumbent"]["recipe_id"] = recipe_ids[1]
    body = dict(tampered)
    body.pop("evidence_sha256")
    tampered["evidence_sha256"] = runner.stable_hash(body)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.QASCCoevolutionError, match="scored item"):
        runner._load_measurement_evidence(
            path=path,
            field="synthetic evidence",
            schema=f"{runner.VERSION}_A_hold_private_evidence",
            freeze_sha256="a" * 64,
            block="A_hold",
            expected_incumbent_recipe_id=recipe_ids[0],
            expected_challenger_recipe_id=recipe_ids[1],
        )

    postflight_tamper = json.loads(json.dumps(evidence))
    postflight_tamper["controls"]["official_HippoRAG_item_local_32"]["rankings"][0] = [0, 1, 2, 3, 4]
    postflight_tamper["controls"]["official_HippoRAG_item_local_32"]["failure_type_hashes"][0] = None
    body = dict(postflight_tamper)
    body.pop("evidence_sha256")
    postflight_tamper["evidence_sha256"] = runner.stable_hash(body)
    path.write_text(json.dumps(postflight_tamper), encoding="utf-8")
    with pytest.raises(runner.QASCCoevolutionError, match="postflight"):
        runner._load_measurement_evidence(
            path=path,
            field="synthetic evidence",
            schema=f"{runner.VERSION}_A_hold_private_evidence",
            freeze_sha256="a" * 64,
            block="A_hold",
            expected_incumbent_recipe_id=recipe_ids[0],
            expected_challenger_recipe_id=recipe_ids[1],
        )


def test_frozen_unavailable_controls_are_not_retried(monkeypatch, tmp_path: Path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen unavailable control was retried")

    monkeypatch.setattr(runner, "_load_retained_p", forbidden)
    monkeypatch.setattr(runner, "_prepare_official_runtime", forbidden)
    retained, prepared = runner._load_descriptive_runtime_for_execution(
        project=tmp_path,
        frozen_retained_binding={"status": "preflight_unavailable"},
        frozen_official_binding={"status": "preflight_unavailable"},
        capability_receipt_path=tmp_path,
        runtime_python=tmp_path,
        local_llm_model=tmp_path,
        local_embedding_model=tmp_path,
        base_binding_receipt_path=tmp_path,
        attestation_receipt_path=tmp_path,
    )
    assert retained is None
    assert prepared is None


def test_m_freeze_without_A_promotion_never_touches_acquisition(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner,
        "_canonical_public_output",
        lambda *_args, **_kwargs: tmp_path / "m-freeze.json",
    )
    monkeypatch.setattr(
        runner,
        "_canonical_stage_root",
        lambda *_args, **_kwargs: tmp_path / "m-root",
    )
    monkeypatch.setattr(
        runner,
        "_load_design",
        lambda _project: ({}, {"design_sha256": runner.DESIGN_SHA256}),
    )
    monkeypatch.setattr(
        runner,
        "_load_diagnostic",
        lambda *_args, **_kwargs: ({}, {"diagnostic_sha256": "d" * 64}),
    )
    monkeypatch.setattr(
        runner,
        "reverify_a_hold_report",
        lambda **_kwargs: (
            {
                "evaluator_epoch_transition": {
                    "promoted": False,
                    "M_search_open_authorized": False,
                },
                "M_search_disposition": {"opened_during_A_hold": False},
            },
            {"report_sha256": "a" * 64},
        ),
    )

    def acquisition_must_remain_unopened(**_kwargs):
        raise AssertionError("M acquisition was touched without promotion")

    monkeypatch.setattr(runner, "_load_acquisition_live", acquisition_must_remain_unopened)
    with pytest.raises(runner.QASCCoevolutionError, match="must remain unopened"):
        runner.build_m_search_freeze(
            project_root=tmp_path,
            diagnostic_path=tmp_path,
            formation_freeze_path=tmp_path,
            formation_receipt_path=tmp_path,
            a_hold_freeze_path=tmp_path,
            a_hold_report_path=tmp_path,
            acquisition_receipt_path=tmp_path,
            selection_secret_path=tmp_path,
            nli_model_path=tmp_path,
            capability_receipt_path=tmp_path,
            runtime_python=tmp_path,
            local_llm_model=tmp_path,
            local_embedding_model=tmp_path,
            base_binding_receipt_path=tmp_path,
            attestation_receipt_path=tmp_path,
            execution_root=tmp_path,
            authorization_hash="b" * 64,
            output_path=tmp_path,
        )


def test_selection_secret_delegates_to_strict_acquisition_decoder(
    monkeypatch, tmp_path: Path
) -> None:
    secret = bytes(range(32))
    calls = []

    class Acquisition:
        @staticmethod
        def load_selection_secret(*, project, selection_secret_path):
            calls.append((project, selection_secret_path))
            return secret

    monkeypatch.setattr(runner, "_acquisition_module", lambda: Acquisition)
    receipt = {
        "selection": {
            "selection_secret_commitment_sha256": hashlib.sha256(secret).hexdigest()
        }
    }
    assert runner._selection_secret(
        project=tmp_path,
        supplied=tmp_path / "selection.key",
        receipt=receipt,
    ) == secret
    assert calls == [(tmp_path, (tmp_path / "selection.key").absolute())]


def test_retained_P_requires_frozen_file_hash(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(runner.QASCCoevolutionError, match="file identity"):
        runner._load_retained_p(tmp_path)
