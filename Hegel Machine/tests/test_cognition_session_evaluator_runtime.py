"""A value-model observation must come from an executed continuation."""

from dataclasses import asdict, replace
import json

from hegel_machine.cognition.session import ResearchSession
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from test_cognition_experiment_frontier_runtime import _saved_question


def test_paid_forecast_receives_actual_return_and_report_reads_do_not_retrain():
    original, environment, _ = _saved_question()
    config = replace(original.config, use_memory=True, learn_research_policy=True)
    saved = original.to_dict()
    saved["config"] = asdict(config)
    session = ResearchSession(config, [environment], observation_budget=128, compute_budget=80000,
                              max_visits=2, state=json.loads(json.dumps(saved)))
    initial_compute = saved["total_compute_cost"]
    report = session.run(max_steps=1)
    episode = report["episodes"][0]
    selected = episode["scheduler"]["research_opportunity"]
    outcome = episode["scheduler"]["research_return"]
    assert selected["evaluator_forecast"]["forecast"] is not None
    actual = next(action["details"] for action in episode["actions"] if action["action"] == "continuation_feedback")
    assert outcome["reference_rows"] == actual["reference_rows"] > 0
    model = report["knowledge"]["research_opportunities"]["evaluator"]
    assert model["records"][-1]["reported_normalized_gain"] == actual["normalized_gain"]
    assert report["evaluator_compute_cost"] > 0
    assert report["total_compute_cost"] == initial_compute + episode["compute_cost"] + report["evaluator_compute_cost"]
    assert episode["observation_cost"] == sum(probe.cost for probe in environment.queries)

    before = session.library.research_opportunities.to_dict()
    costs = report["total_compute_cost"]
    session.to_dict()
    session._pending()
    paused = session.run(max_steps=0)
    assert session.library.research_opportunities.to_dict() == before
    assert paused["total_compute_cost"] == costs


def test_evaluator_budget_preserves_the_last_complete_visit():
    session, _, _ = _saved_question()
    session.config = replace(session.config, use_memory=True, learn_research_policy=True)
    session.limits["compute_budget"] = session.state["total_compute_cost"] + session.config.compute_budget
    assert session._evaluator_budget(256, 100).limit == 0
    assert session._evaluator_budget(1200, 20).limit == 0


def test_model_rejected_payable_experiment_is_not_resurrected_as_unfunded():
    session, environment, _ = _saved_question()
    # Isolate routing with a restored deployed evaluator. The experiment and
    # its payable old quote are real; model induction is tested separately.
    session.library.research_opportunities.evaluator.current = {
        "model_id": 1, "program": Program("const", value=-.5).to_dict()}
    assert session._pending() and session._visit_config(session.state["tasks"][0]) is not None
    budget = SearchBudget(256)
    assert session._pending(evaluator_budget=budget) == []
    assert budget.spent > 0 and environment.queries == []
    estimates = session.state["tasks"][0]["evaluator_estimates"]
    assert estimates and all(quote["basis"] == "revised_research_evaluator" for quote in estimates)
    assert all(quote["expected_gain"] == -.5 and quote["evaluator_forecast"] for quote in estimates)
    assert session.state["tasks"][0]["next_research"] is None
