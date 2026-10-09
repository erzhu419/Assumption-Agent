"""Research purchases must use the paid support of the remaining meter gap."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_research_returns import consider_condition_research, record_condition_research
from hegel_machine.cognition.condition_response_actions import research_observed_condition_competition
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def paid_prefix(experiment_count=7, *, earlier=False):
    with gzip.open(Path(__file__).parent / "fixtures/condition_coverage_paid_root05.json.gz", "rt") as f:
        recorded = json.load(f)
    episodes = []
    for episode in recorded["episodes"][:4 + experiment_count]:
        rows, history = [], []
        for item in episode["observations"]:
            rows.append(Observation(item["features"], item["target"], tuple(history)))
            history.append({**item["features"], "__target__": item["target"]})
        episodes.append(rows)
    validation = episodes[3]
    training = [row for index, episode in enumerate(episodes) if index != 3 for row in episode]
    spec = TaskSpec("paid_condition_coverage", "Predict measured responses under legal commands.",
        tuple(recorded["feature_names"]), (), (),
        feature_bounds={key: tuple(value) for key, value in recorded["feature_bounds"].items()},
        max_probe_length=recorded["max_probe_length"])
    program = Program.from_dict(recorded["earlier_program" if earlier else "program"])
    steps = deepcopy(recorded["selection_steps"])
    if earlier:
        # Restore this actual earlier recipe's gate from its paid report. Later
        # recipe revisions must not conceal the counterexample to the old cut.
        condition = recorded["earlier_condition"]

        def restore_gate(items):
            for item in items:
                if item.get("gate_coordinate", {}).get("name") == condition["condition_field"]:
                    item.update(threshold=condition["threshold"],
                                threshold_interval=condition["threshold_interval"])
                restore_gate(item.get("inner_selection_steps", []))
        restore_gate(steps)
    selected = {"candidate": Candidate("paid_response", program, "research_program"),
        "research_program": {"selection_steps": steps},
        "training_loss": loss(program, training, 0., .5),
        "validation_loss": loss(program, validation, 0., .5), "fallback": 0.}
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        spec, [*training, *validation], selected, SearchBudget(15000))
    actions = deepcopy(recorded["paid_actions"][:2 * experiment_count])
    experiments = [{**actions[index], "research_return": actions[index + 1]["research_return"]}
                   for index in range(0, len(actions), 2)]
    if experiment_count == 7:
        returns = deepcopy(recorded["condition_research_returns"])
    else:
        # These are actual purchased outcomes, truncated at this paid prefix.
        outcomes = [item["research_return"] for item in experiments]
        state = selected["condition_uncertainty"]
        returns = {"batches": len(outcomes),
            "information_batches": sum(item["informative"] for item in outcomes),
            "predicted_information_batches": sum(item["predicted_information_probability"] for item in outcomes),
            **{f"{resource}_cost": sum(item[f"{resource}_cost"] for item in outcomes)
               for resource in ("observation", "measurement", "compute")},
            "last_question": {"status": state["status"], "program": state["program"],
                "original_target_loss": state["original_target_loss"],
                "conditions": [{"condition_field": item["condition_field"],
                    "threshold_intervals": item["threshold_intervals"],
                    "needs_threshold_revision": item.get("needs_threshold_revision", False)}
                    for item in state["conditions"]]},
            "last_outcome": outcomes[-1]}
    selected["condition_research_returns"] = returns
    return recorded, spec, training, validation, selected, actions, experiments


def test_narrow_paid_gap_prices_new_local_forecast_without_erasing_old_paid_misses():
    for experiment_count in (4, 7):
        recorded, spec, training, validation, selected, actions, previous = paid_prefix(experiment_count)
        rows = [*training, *validation]
        state = selected["condition_uncertainty"]
        interval = state["conditions"][0]["threshold_interval"]
        assert len(rows) == 48 + 24 * experiment_count
        assert state["status"] == "unresolved"
        assert state["original_target_loss"] < 1e-7

        planner_budget = SearchBudget(15000)
        plan = plan_observed_condition_probe(spec, rows, selected, planner_budget,
                                            previous_experiments=previous)
        assert plan is not None
        assert plan["threshold_interval"] == interval
        assert plan["sampling_basis"] == "paid_local_residual_density"
        assert plan["previous_experiment_count"] == experiment_count
        assert all(plan["fixed_control_background"] != item["fixed_control_background"]
                   for item in previous)
        assert all(set(point) == set(spec.feature_bounds) for point in plan["probe"].sequence)
        assert 0 < plan["compute_cost"] == planner_budget.spent <= planner_budget.limit

        # The interval forecast is supported by actual paid residual neighbors,
        # rather than a CDF jump selected by a tiny change in background.
        exposure, = plan["interval_exposure"]
        assert exposure["threshold_interval"] == interval
        assert exposure["probability_basis"] == "paid_local_residual_density"
        assert 0 < exposure["estimated_probability"] < 1
        assert 0 < exposure["local_residual_rows"] <= len(rows)
        mean = Program.from_dict(plan["reading_mean_program"])
        background_mean = mean.predict(Observation(plan["fixed_control_background"], 0.))
        support_left, support_right = exposure["residual_support_interval"]
        assert support_left < interval[0] - background_mean < interval[1] - background_mean < support_right
        residuals = [row.features["meter"] - mean.predict(row) for row in rows]
        assert any(value == pytest.approx(support_left) for value in residuals)
        assert any(value == pytest.approx(support_right) for value in residuals)

        agent = UnifiedResearchAgent(ResearchConfig(observation_budget=320, compute_budget=320000,
            use_memory=False, minimum_improvement=1e-7), measurement_budget=320)
        purchases = []

        def acquire(channels, purpose, *, probe):
            purchases.append(probe)
            return []

        budget = SearchBudget(320000)
        result, updated, _ = research_observed_condition_competition(agent, spec, selected,
            training, validation, [], budget, actions, acquire=acquire,
            remaining=lambda: {"observation_remaining": recorded["observation_budget"] - len(rows),
                               "measurement_remaining": recorded["measurement_budget"] - len(rows)},
            channels=("meter",), measurement_rate=1)
        choice, = [item for item in actions if item["action"] == "choose_condition_research_investment"]
        # These historical batches forecast from globally pooled residuals.
        # The local model nominates a different experiment: at 144 rows it is
        # payable, while the 216-row prefix still stops on remaining value.
        assert choice["buy"] == (experiment_count == 4)
        assert bool(purchases) == choice["buy"]
        if experiment_count == 7:
            assert choice["reason"] == "retain_question_for_more_valuable_future_evidence"
        assert choice["previous_experiment_count"] == experiment_count
        assert choice["current_question_trials"] == 0
        assert choice["current_question_information_batches"] == 0
        assert choice["current_question_predicted_information_batches"] == 0.
        assert choice["probability_of_new_information"] > 0
        if experiment_count == 7:
            assert choice["calibrated_information_probability"] < choice["resource_share"]
        if experiment_count == 7:
            assert selected["condition_research_returns"]["information_batches"] == 3
            assert choice["current_question_trials"] == 0
            assert all(not item["research_return"]["informative"] for item in previous[4:])
            assert all(not interval[0] < item["features"]["meter"] < interval[1]
                       for episode in recorded["episodes"][8:11] for item in episode["observations"])
            # Persist outcomes of those real paid batches with the new summary
            # API. JSON continuation without the action log retains the same
            # current-gap calibration and all earlier lifecycle costs.
            for count in (5, 6, 7):
                _, _, _, _, after, _, _ = paid_prefix(count)
                experiment = previous[count - 1]
                outcome = experiment["research_return"]
                persisted = record_condition_research(persisted, last_state,
                    after["condition_uncertainty"], experiment,
                    observation_cost=outcome["observation_cost"],
                    measurement_cost=outcome["measurement_cost"],
                    compute_cost=outcome["compute_cost"], changed_program=outcome["changed_program"])
                last_state = after["condition_uncertainty"]
            restored = json.loads(json.dumps(persisted))
            assert restored["current_question_returns"]["batches"] == 3
            assert restored["current_question_returns"]["information_batches"] == 0
            for resource in ("observation", "measurement", "compute"):
                assert restored[f"{resource}_cost"] == selected["condition_research_returns"][f"{resource}_cost"]
            continued = consider_condition_research(state, plan, restored,
                observation_remaining=recorded["observation_budget"] - len(rows),
                measurement_remaining=recorded["measurement_budget"] - len(rows),
                compute_remaining=320000 - planner_budget.spent, measurement_rate=1,
                compute_quote=choice["quoted_compute_cost"], observation_reserve=24,
                measurement_reserve=24, compute_reserve=96)
            assert not continued["buy"]
            for field in ("current_question_trials", "current_question_information_batches",
                    "current_question_predicted_information_batches", "sampling_reliability",
                    "calibrated_information_probability"):
                assert continued[field] == pytest.approx(choice[field])
        else:
            persisted = deepcopy(selected["condition_research_returns"])
            last_state = state
        assert updated == training
        assert result["candidate"].program == selected["candidate"].program
        assert result["condition_research_returns"] == selected["condition_research_returns"]
        assert result["condition_uncertainty"]["status"] == "unresolved"
        assert result["condition_uncertainty"]["expected_prediction_ambiguity"] > 1e-7
        assert 0 < budget.spent <= budget.limit


@pytest.mark.parametrize("remaining_rows", [100, 176])
def test_actual_new_paid_counterexample_reopens_even_when_an_ordinary_purchase_would_be_declined(remaining_rows):
    _, _, old_training, old_validation, before, actions, _ = paid_prefix(3, earlier=True)
    recorded, spec, training, validation, selected, _, _ = paid_prefix(4, earlier=True)
    assert len(old_training) + len(old_validation) == recorded["earlier_paid_rows"] == 120
    assert max(before["training_loss"], before["validation_loss"]) < 1e-7
    assert selected["candidate"].program == before["candidate"].program
    assert selected["training_loss"] == pytest.approx(6.39016656671864)
    assert selected["condition_uncertainty"]["conditions"][0]["needs_threshold_revision"]
    selected["condition_research_returns"] = deepcopy(before["condition_research_returns"])
    agent = UnifiedResearchAgent(ResearchConfig(observation_budget=320, compute_budget=320000,
        use_memory=False, minimum_improvement=1e-7), measurement_budget=320)
    purchases = []

    def acquire(channels, purpose, *, probe):
        # The observations are already paid. Stop at the actual next purchase
        # boundary without creating another lab experiment or stubbing a fit.
        purchases.append((channels, purpose, probe))
        return []

    budget = SearchBudget(320000)
    result, updated, _ = research_observed_condition_competition(agent, spec, selected,
        training, validation, [], budget, actions, acquire=acquire,
        remaining=lambda: {"observation_remaining": remaining_rows,
                           "measurement_remaining": remaining_rows},
        channels=("meter",), measurement_rate=1)
    choice, = [item for item in actions if item["action"] == "choose_condition_research_investment"]
    if remaining_rows == 100:
        assert choice["calibrated_information_probability"] < choice["resource_share"]
    assert choice["buy"] and choice["reopened"]
    assert choice["reason"] == "reopen_from_new_original_response_evidence"
    for channels, purpose, probe in purchases:
        assert channels == ("meter",)
        assert purpose == "distinguish_competing_observed_conditions"
        assert all(set(point) == set(spec.feature_bounds) for point in probe.sequence)
    repair, = [item for item in actions if item["action"] == "revise_condition_on_paid_response"]
    assert repair["changed_program"] and repair["measured_rows"] == 0
    assert repair["requires_fresh_confirmation"]
    assert repair["after_loss"] < 1e-7 < repair["before_loss"]
    assert actions.index(repair) < next(index for index, item in enumerate(actions)
        if item["action"] == "choose_condition_research_investment")
    assert updated == training
    assert result["candidate"].program != before["candidate"].program
    assert loss(result["candidate"].program, [*training, *validation], 0., .5) < 1e-7
    assert result["condition_uncertainty"]["status"] == "unresolved"
    assert not result["condition_uncertainty"]["conditions"][0]["needs_threshold_revision"]
    assert 0 < budget.spent <= budget.limit
