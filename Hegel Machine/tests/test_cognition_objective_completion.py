"""Evidence and budgets decide when a collected goal can face future labels."""

from copy import deepcopy
import json
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.objective_completion import (
    objective_completion, validation_compute_reserve,
)
from hegel_machine.cognition.objective_research import ObjectiveResearch, _fit


def spec():
    return TaskSpec("completion", "Complete a paid goal question", ("x1", "x2", "x3"), (), (),
        feature_bounds={name: (0., 1.) for name in ("x1", "x2", "x3")},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0"})


def observations(count=20, *, affine=False):
    random = Random(127)
    result = []
    for index in range(count):
        point = {name: random.random() for name in spec().feature_names}
        x, y, z = point["x1"] - .2, point["x2"] - .7, point["x3"] - .8
        target = (1 + .4*x + 2*y + z if affine else
                  .35 + 2*x*x + 3*y*y + 4*z*z + 2*x*y - 2*y*z)
        result.append({"observation": {"features": point, "target": -1.},
                       "context": {"objective": target, "global_index": index}})
    return result


def assess(rows, budget, state, *, calls=15, fit=_fit):
    return objective_completion(spec(), rows, budget, fit=fit, reference_loss=.0001,
                                remaining_calls=calls, state=state)


def test_adequate_low_order_candidate_starts_validation_without_cross_term_coverage():
    rows = observations(9, affine=True)
    before = deepcopy(rows)
    budget, state = SearchBudget(20000), {}
    result = assess(rows, budget, state)
    assert result["ready"] and result["candidate"]["degree"] == 1
    assert result["status"] == "adequate_paid_residual"
    assert state["fit_readiness"] == [True, True, False]
    assert budget.spent > 0 and budget.remaining >= result["validation_reserve"]
    assert rows == before


def test_unexplained_residual_waits_for_measured_interaction_comparison():
    rows = observations()
    budget, state = SearchBudget(20000), {}
    before = assess(rows[:12], budget, state)
    assert before["candidate"] is None and before["need_interactions"]
    assert before["status"] == "unresolved_low_order_residual"
    after = assess(rows[:13], budget, state)
    assert after["ready"] and after["candidate"]["interactions"]
    assert after["candidate"]["cv_loss"] < 1e-8
    assert budget.remaining >= after["validation_reserve"]


def test_cached_insufficient_fit_waits_for_new_identifiability_or_four_new_controls():
    rows = observations()
    budget, state = SearchBudget(20000), {}
    assess(rows[:9], budget, state)
    for count in (10, 11, 12):
        result = assess(rows[:count], budget, state)
        assert result["fit_compute"] == 0
    assert len(state["fits"]) == 1
    assert assess(rows[:13], budget, state)["candidate"]["interactions"]
    assert len(state["fits"]) == 2


def test_cache_is_json_restorable_and_replicates_do_not_create_new_control_groups():
    rows = observations(12)
    budget, state = SearchBudget(20000), {}
    assess(rows, budget, state)
    state = json.loads(json.dumps(state))
    spent = budget.spent
    result = assess(rows, budget, state)
    assert result["fit_compute"] == result["readiness_compute"] == 0
    assert budget.spent == spent
    repeat = deepcopy(rows[-1])
    repeat["context"]["global_index"] = 12
    assess([*rows, repeat], budget, state)
    assert len(state["fits"]) == 1 and budget.spent == spent


def saved_failed_richer_fit():
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_joint_identification_20260911/revision5/fresh_confirmation/runs/"
        "joint_identification_3_center_interaction_change_hegel_joint_identification.json")
    record = json.loads(path.read_text())
    rows = [{"observation": {"features": dict(zip(spec().feature_names,
                    (value / 100 for value in item["point"]))), "target": item["output"][1]},
             "context": {"objective": item["output"][0], "global_index": 31 + item["new_call"]}}
            for item in record["simulations"] if 40 <= 31 + item["new_call"] < 54]
    return rows, record["objective_research"]["scope_research"]["trigger"][0]["threshold"] / 9


def test_failed_richer_fit_preserves_actual_earlier_hypothesis_and_its_training_prefix():
    rows, reference = saved_failed_richer_fit()
    state, budget = {}, SearchBudget(20000)
    objective_completion(spec(), rows[:9], budget, fit=_fit,
        reference_loss=reference, remaining_calls=15, state=state)
    earlier = deepcopy(state["model"])
    assert earlier["degree"] == 1
    assert earlier["cv_loss"] == pytest.approx(.10552732950092655)
    for count in (13, 14):
        result = objective_completion(spec(), rows[:count], budget, fit=_fit,
            reference_loss=reference, remaining_calls=24-count, state=state)
        assert not result["ready"]
        assert state["model"] == earlier
        assert state["training_indices"] == list(range(40, 49))
        assert state["fits"][-1]["training_indices"] == list(range(40, 40+count))
        assert state["fits"][-1]["degree"] is None
        assert state["fits"][-1]["compute_cost"] > 0
        assert state["fit_control_count"] == count
    assert state["fit_readiness"] == [True, True, True]
    state = json.loads(json.dumps(state))

    def cannot_repeat(*args):
        pytest.fail("a failed fit must not repeat on the same paid controls")

    repeated = objective_completion(spec(), rows, budget, fit=cannot_repeat,
        reference_loss=reference, remaining_calls=10, state=state)
    assert repeated["fit_compute"] == repeated["readiness_compute"] == 0
    # A failed attempt with no earlier model must also wait for new evidence.
    failed = {}
    objective_completion(spec(), rows[:5], SearchBudget(20000), fit=_fit,
        reference_loss=reference, remaining_calls=19, state=failed)
    assert not failed.get("model")
    objective_completion(spec(), rows[:5], SearchBudget(20000), fit=cannot_repeat,
        reference_loss=reference, remaining_calls=19, state=failed)


def test_retained_hypothesis_does_not_claim_later_failed_fit_labels_as_its_cv_evidence():
    rows, reference = saved_failed_richer_fit()
    state, budget = {}, SearchBudget(20000)
    for count in (9, 13, 14):
        objective_completion(spec(), rows[:count], budget, fit=_fit,
            reference_loss=reference, remaining_calls=24-count, state=state)
    original = deepcopy(state)

    def cannot_refit(*args):
        pytest.fail("the remaining compute belongs to future validation")

    result = objective_completion(spec(), rows, SearchBudget(700), fit=cannot_refit,
        reference_loss=reference, remaining_calls=4, state=state)
    assert result["ready"] and result["status"] == "budgeted_low_order_validation"
    assert result["residual_evidence"]["count"] == 9
    assert result["candidate"] == original["model"]
    assert state["training_indices"] == list(range(40, 49))
    changed = deepcopy(rows)
    for row in changed[9:]:
        row["context"]["objective"] = 1e6
    other = objective_completion(spec(), changed, SearchBudget(700), fit=cannot_refit,
        reference_loss=reference, remaining_calls=4, state=deepcopy(original))
    assert other["residual_evidence"] == result["residual_evidence"]


def test_identification_cannot_consume_the_next_complete_fit_and_validation():
    rows = observations()
    budget, state = SearchBudget(20000), {}
    result = assess(rows[:12], budget, state)
    assert result["identification_allowance"] > 0
    assert budget.spend(result["identification_allowance"])
    after = assess(rows[:13], budget, state, calls=14)
    assert after["ready"] and after["candidate"]["interactions"]
    assert budget.remaining >= after["validation_reserve"]
    assert budget.limit == 20000


def test_unaffordable_completion_does_not_spend_the_validation_reserve():
    rows = observations(13)
    reserve = validation_compute_reserve(spec(), len(rows), 9)
    budget = SearchBudget(reserve + 10)
    result = assess(rows, budget, {}, calls=9)
    assert result["candidate"] is None
    assert result["status"] == "insufficient_completion_compute"
    assert result["identification_allowance"] == 0 and budget.spent == 0


def test_three_remaining_labels_do_not_begin_a_four_label_validation():
    budget = SearchBudget(20000)
    result = assess(observations(13), budget, {}, calls=3)
    assert result["status"] == "insufficient_validation_calls"
    assert not result["ready"] and budget.spent == 0


def test_saved_nine_control_affine_repair_can_finish_without_identifying_interactions():
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_objective_reentry_v052_20260911/runs/"
        "research_scope_3_objective_only_change_noise_0.03_objective_reentry_v052.json")
    record = json.loads(path.read_text())
    revision = record["objective_scope_revisions"][0]
    indices = revision["training_indices"]
    rows = [{"observation": {"features": {name: value / 100 for name, value in
                 zip(spec().feature_names, item["point"])}, "target": item["output"][1]},
             "context": {"objective": item["output"][0], "global_index": 31 + item["new_call"]}}
            for item in record["simulations"] if 31 + item["new_call"] in indices]
    trigger = next(event for event in record["objective_research"]["events"]
                   if event["action"] == "research_objective_scope")
    budget = SearchBudget(20000)
    result = objective_completion(spec(), rows, budget, fit=_fit,
        reference_loss=trigger["threshold"] / 9, remaining_calls=11, state={})
    assert len(rows) == 9
    assert result["ready"] and result["candidate"]["degree"] == 1
    assert result["candidate"]["cv_loss"] == pytest.approx(.0005216771010919763)
    assert budget.remaining >= result["validation_reserve"]


def saved_small_cv_repair():
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_goal_completion_20260911/runs/"
        "objective_reentry_1_mechanism_change_noise_0.03_hegel_goal_completion.json")
    record = json.loads(path.read_text())
    research = record["objective_research"]["scope_research"]
    indices = research["completion"]["training_indices"]
    rows = [{"observation": {"features": {name: value / 100 for name, value in
                 zip(spec().feature_names, item["point"])}, "target": item["output"][1]},
             "context": {"objective": item["output"][0], "global_index": 31 + item["new_call"]}}
            for item in record["simulations"] if 31 + item["new_call"] in indices]
    return rows, research["trigger"][0]["threshold"] / 9


@pytest.mark.parametrize("calls, ready", [(6, True), (8, False)])
def test_saved_small_cv_repair_can_validate_only_when_richer_study_cannot_finish(calls, ready):
    rows, reference = saved_small_cv_repair()
    before = deepcopy(rows)
    budget, state = SearchBudget(20000), {}
    result = objective_completion(spec(), rows, budget, fit=_fit,
        reference_loss=reference, remaining_calls=calls, state=state)
    assert state["training_indices"] == [54, 55, 56, 57, 58]
    assert state["fit_readiness"] == [True, False, False]
    assert state["model"]["cv_loss"] == pytest.approx(.004575429306227963)
    assert result["error_threshold"] == pytest.approx(.001167877479484694)
    assert result["ready"] == ready
    if ready:
        assert result["status"] == "budgeted_low_order_validation"
        assert result["residual_unresolved"]
        assert result["identification_calls_required"] == 4
        assert result["residual_evidence"]["decision"] == "supported"
        assert result["residual_evidence"]["count"] == 5
        assert result["candidate"]["degree"] == 1
    else:
        assert result["candidate"] is None
        assert result["need_interactions"]
    assert budget.remaining >= result["validation_reserve"]
    assert budget.spent == result["fit_compute"] + result["readiness_compute"]
    assert budget.limit == 20000 and rows == before


def test_unresolved_interactions_can_face_future_labels_only_at_completion_deadline():
    budget = SearchBudget(20000)
    result = assess(observations(12), budget, {}, calls=4)
    assert result["candidate"] is not None and result["residual_unresolved"]
    assert result["identification_calls_required"] == 1
    assert result["residual_evidence"]["count"] == 12
    assert result["residual_evidence"]["decision"] == "inconclusive"
    assert result["residual_evidence"]["mean_gain"] > 0
    assert not assess(observations(12), SearchBudget(20000), {}, calls=5)["ready"]
    assert budget.remaining >= result["validation_reserve"]


def test_paid_improvement_reuses_its_frozen_groups_after_json_restore_and_new_labels():
    rows, reference = saved_small_cv_repair()
    budget, state = SearchBudget(20000), {}
    first = objective_completion(spec(), rows, budget, fit=_fit,
        reference_loss=reference, remaining_calls=6, state=state)
    state = json.loads(json.dumps(state))
    spent = budget.spent
    repeated = objective_completion(spec(), rows, budget, fit=_fit,
        reference_loss=reference, remaining_calls=6, state=state)
    assert repeated == first | {"fit_compute": 0, "readiness_compute": 0}
    assert budget.spent == spent
    # Later observations cannot be reassigned to the cached model's CV folds.
    extra = {"observation": {"features": {"x1": 0., "x2": 0., "x3": 1.}, "target": 0.},
             "context": {"objective": 100., "global_index": 59}}
    after = objective_completion(spec(), [*rows, extra], budget, fit=_fit,
        reference_loss=reference, remaining_calls=5, state=state)
    assert after["ready"] and after["fit_compute"] == 0
    assert after["residual_evidence"] == first["residual_evidence"]
    assert state["training_indices"] == [54, 55, 56, 57, 58]


def test_replicates_cannot_inflate_independent_improvement_support():
    rows, reference = saved_small_cv_repair()
    repeats = []
    for index, row in enumerate(rows, start=59):
        repeated = deepcopy(row)
        repeated["context"]["global_index"] = index
        repeats.append(repeated)
    result = objective_completion(spec(), [*rows, *repeats], SearchBudget(20000), fit=_fit,
        reference_loss=reference, remaining_calls=6, state={})
    assert result["ready"]
    assert result["residual_evidence"]["count"] == 5
    assert result["identification_calls_required"] == 4


def saved_compute_limited_repair():
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_goal_recompetition_20260911/runs/"
        "goal_completion_2_center_interaction_change_hegel_goal_recompetition.json")
    record = json.loads(path.read_text())
    rows = [{"observation": {"features": {name: value / 100 for name, value in
                 zip(spec().feature_names, item["point"])}, "target": item["output"][1]},
             "context": {"objective": item["output"][0], "global_index": 31 + item["new_call"]}}
            for item in record["simulations"] if 40 <= 31 + item["new_call"] < 60]
    return record["objective_research"], rows


def test_saved_compute_limited_repair_uses_only_original_frozen_prefix():
    research, rows = saved_compute_limited_repair()
    state = deepcopy(research["scope_research"]["completion"])
    before = deepcopy(state)
    reference = research["scope_research"]["trigger"][0]["threshold"] / 9
    budget = SearchBudget(research["compute_budget"] - research["compute_cost"])

    def cannot_refit(*args):
        pytest.fail("new fitting cannot consume the future validation reserve")

    result = objective_completion(spec(), rows, budget, fit=cannot_refit,
        reference_loss=reference, remaining_calls=4, state=state)
    assert result["ready"] and result["status"] == "budgeted_low_order_validation"
    assert result["residual_unresolved"] and "compute" in result["completion_limited_by"]
    assert result["residual_evidence"]["count"] == 13
    assert result["residual_evidence"]["mean_gain"] == pytest.approx(.3083120450455559)
    assert result["residual_evidence"]["decision"] == "inconclusive"
    assert result["candidate"] == before["model"]
    assert state["training_indices"] == list(range(40, 53))
    assert state["fits"] == before["fits"] and state["checked_controls"] == before["checked_controls"]
    assert budget.limit == 2220 and budget.spent == result["fit_compute"] == 26
    assert result["readiness_compute"] == 0
    assert budget.remaining >= result["validation_reserve"]
    # Newly available labels are not retroactive support for the frozen folds.
    altered = deepcopy(rows)
    for row in altered:
        if row["context"]["global_index"] > 52:
            row["context"]["objective"] = 1e6
    other = objective_completion(spec(), altered, SearchBudget(2220), fit=cannot_refit,
        reference_loss=reference, remaining_calls=4, state=deepcopy(before))
    assert other["residual_evidence"] == result["residual_evidence"]


def test_compute_limited_candidate_still_needs_support_from_future_shared_labels():
    state, rows = saved_compute_limited_repair()
    research = ObjectiveResearch(spec(), state=state)
    budget = SearchBudget(2220)
    result = research._collect_candidate(rows, budget, remaining_calls=4)
    research.state["compute_cost"] += budget.spent
    assert result["ready"] and research.accepted_revision is None
    for index in range(60, 64):
        features = {"x1": .1 * (index - 59), "x2": .2, "x3": .3}
        request = research.prepare_shared_control(rows,
            lambda point: {"features": point, "objective_mean": 0., "objective_noise": .001},
            features, scope_start=40, stage=index, remaining_calls=64-index, purpose="optimization")
        assert request is not None
        research.observe(features, [0., -1.],
            context={"global_index": index, "stage": index + 1, "purpose": "optimization"})
        rows.append({"observation": {"features": features, "target": -1.},
                     "context": {"global_index": index, "objective": 0.}})
    assert research.accepted_revision is None
    attempt = research.state["scope_attempts"][-1]
    assert attempt["status"] == "future_predictions_do_not_improve"
    assert [row["global_index"] for row in attempt["validation"]] == list(range(60, 64))
    assert research.state["compute_cost"] <= 20000


def test_negative_paid_improvement_does_not_start_future_validation():
    rows = observations(12)
    model = _fit(spec(), rows, SearchBudget(20000))
    # A frozen poor proposal can be assessed but gains no right to another query.
    model["coefficients"] = [10., 0., 0., 0., 0., 0., 0.]
    model["folds"] = [{"degree": 2, "coefficients": model["coefficients"]} for _ in range(5)]
    state = {"model": model, "training_indices": list(range(12)),
        "checked_controls": [list(row["observation"]["features"].values()) for row in rows],
        "checked_readiness": [True, True, False], "fit_readiness": [True, True, False],
        "fit_control_count": 12}
    result = assess(rows, SearchBudget(20000), state, calls=4)
    assert result["residual_evidence"]["mean_gain"] < 0
    assert result["candidate"] is None and not result["ready"]


def saved_stage52_optimizer(profile, record, monkeypatch):
    """Recover the actual opponent using paid labels; physical calls are forbidden."""
    import numpy as np
    from types import MethodType

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import run_sc_kg_goal_completion_study as loader
    protocol = json.loads((root / "artifacts/sc_kg_goal_recompetition_20260911/revision2/protocol.json").read_text())
    group = protocol["groups"]["unknown_previous_confirmation"]
    case = next(case for case in group["cases"] if case["case_id"] == record["case_id"])
    settings = json.loads(Path(group["source_protocol"]).read_text())
    source = json.loads(Path(case["initial_research_origin"]).read_text())
    modules, _ = loader.load_method({"method_source": protocol["source_v053"]}, f"unknown_completion_test_{profile}")
    module, Policy, Recovery = modules[loader.METHOD]
    report = source["result"]
    payload = report.get("program") or report["research_state"]["incumbent"]["candidate"]["program"]
    environment = module.SCKGEnvironment(settings["sc_kg_project_root"], **case["task"])
    environment.restore(source.get("optimizer_initial_environment_state", report["research_state"]["environment_state"]))
    adapter = loader.study.controlled_problem(module.HegelProgramProblem, environment,
        loader.study.Program.from_dict(payload), case)

    def physical_disabled(*args, **kwargs):
        pytest.fail("this paid-prefix regression test must not request a physical label")

    monkeypatch.setattr(adapter.base.base, "simulate", physical_disabled)
    monkeypatch.setattr(adapter.next_base.base, "simulate", physical_disabled)

    def paid_label(self, point, rng):
        row = record["simulations"][self.simulator_calls]
        assert list(point) == row["point"]
        rng.normal(0., case["task"]["noise"], size=3)
        if self.simulator_calls >= case["change_after_new_calls"]:
            self.base = self.next_base
        self.simulator_calls += 1
        self.simulation_log.append(row)
        return np.asarray(row["output"])

    monkeypatch.setattr(adapter, "simulate", MethodType(paid_label, adapter))
    from algorithms.single_olhkg import SingleOLHKGConfig
    config = SingleOLHKGConfig(N=settings["total_calls_per_arm"],
        seed=case["task"]["seed"] + settings["optimizer_seed_offset"], **settings["optimizer_settings"])
    recovery = Recovery(environment.spec, minimum_error=settings["revision"]["minimum_error"],
        batch_size=settings["revision"]["batch_size"],
        seed=case["task"]["seed"] + settings["revision"]["seed_offset"],
        compute_budget=settings["revision"]["compute_budget"])
    algorithm = module.initialize_sc_kg_optimizer(adapter, config, recovery=recovery,
        adoption_policy=Policy(minimum_gain=settings["policy"]["minimum_gain"],
            batch_size=settings["policy"]["minimum_new_observations"]))
    generate = algorithm._generate_candidates

    class PaidPrefixComplete(Exception):
        pass

    def stop_at_paid_prefix(*args, **kwargs):
        if adapter.simulator_calls == 20:
            raise PaidPrefixComplete
        return generate(*args, **kwargs)

    monkeypatch.setattr(algorithm, "_generate_candidates", stop_at_paid_prefix)
    with pytest.raises(PaidPrefixComplete):
        algorithm.run()
    assert adapter.simulator_calls == 20
    return algorithm


@pytest.mark.parametrize("profile", ["center_change", "center_interaction_change"])
def test_actual_stage52_budget_preserves_identifiable_joint_control(profile, monkeypatch):
    import numpy as np
    from hegel_machine.cognition.decision_experiments import _design

    path = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_goal_completion_20260911/runs/"
        f"unknown_confirmation_1_{profile}_hegel_goal_completion.json")
    record = json.loads(path.read_text())
    paid = [*zip(record["initial_points"], record["initial_outputs"]),
            *((item["point"], item["output"]) for item in record["simulations"][:20])]
    rows = [{"observation": {"features": dict(zip(spec().feature_names,
                    (value / 100 for value in point))), "target": output[1]},
             "context": {"objective": output[0], "global_index": index}}
            for index, (point, output) in enumerate(paid)]
    current = rows[40:]
    state = {"model": _fit(spec(), current[:9], SearchBudget(20000)),
        "training_indices": list(range(40, 49)), "fit_readiness": [True, True, False],
        "fit_control_count": 9, "checked_readiness": [True, True, False],
        "checked_controls": [list(row["observation"]["features"].values()) for row in current[:11]]}
    # Recovered by replaying the original paid prefix through SC-KG: no new labels.
    budget = SearchBudget(4070)
    result = objective_completion(spec(), current, budget, fit=_fit,
        reference_loss=.0030078549502106726 / 9, remaining_calls=12, state=state)
    assert result["candidate"] is None and result["need_interactions"]
    assert result["fit_compute"] == 0 and result["readiness_compute"] == 120
    assert result["next_fit_reserve"] == 1534
    assert result["next_validation_reserve"] == 627
    assert result["identification_allowance"] == 1789
    research = ObjectiveResearch(spec())
    research.state["scope_start"] = 0

    algorithm = saved_stage52_optimizer(profile, record, monkeypatch)
    allowance = SearchBudget(result["identification_allowance"])
    _, candidates = research._identifying_controls(current, allowance, centers=rows)
    # Selection may improve; require the actual paid-prefix GP to certify the
    # selected quantized control while preserving fitting and validation money.
    forecast = next(prediction for point in candidates
        if (prediction := algorithm._objective_forecast(point))["feasible"])
    actual = forecast["features"]
    assert 0 < allowance.spent <= result["identification_allowance"]
    assert result["next_fit_reserve"] + result["next_validation_reserve"] <= budget.remaining - allowance.spent

    def fold_ranks(points):
        keys = [tuple(point[name] for name in spec().feature_names) for point in points]
        groups = {key: index % 5 for index, key in enumerate(dict.fromkeys(keys))}
        design = np.asarray([_design(spec(), point, 2, ((0, 1), (0, 2), (1, 2))) for point in points])
        return [np.linalg.matrix_rank(design[[groups[key] != group for key in keys]])
                for group in range(5)]

    points = [row["observation"]["features"] for row in current]
    assert fold_ranks(points) == [9, 9, 10, 9, 10]
    assert fold_ranks([*points, actual]) == [10] * 5
    assert fold_ranks([*points, {"x1": .19, "x2": .41, "x3": .59}]) == [10] * 5
