"""Real observation forecasts, future model choice and changed experiment design."""
import json

import pytest

from hegel_machine.cognition.experiment_simulator import ExperimentSimulator
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.simulator_research import ObservationApparatus, run_simulator_research


def test_model_rewrite_requires_future_trajectories_then_changes_executed_probe():
    proposed = run_simulator_research(max_steps=2)
    state = proposed["simulator_state"]
    assert state["current"] is None
    assert len(state["candidates"]) == 2
    assert all(model["training_rows"] == 16 and not model["trials"] for model in state["candidates"])
    almost = run_simulator_research(state=proposed, max_steps=3)
    assert almost["simulator_state"]["current"] is None
    assert all(len(model["trials"]) == 3 for model in almost["simulator_state"]["candidates"])
    adopted = run_simulator_research(state=almost, max_steps=1)
    assert adopted["simulator_state"]["current"] is not None
    event = next(item for item in adopted["revisions"] if item["event"] == "adopt_simulation_model")
    assert event["comparison"]["trials"] == 4 and event["comparison"]["gain"] > 0
    completed = run_simulator_research(state=adopted)
    selection = completed["selections"][0]
    assert selection["previous_probe"] == "loud-discrimination"
    assert selection["selected_probe"] == selection["executed_probe"]["probe_id"] == "quiet-discrimination"
    assert completed["selected_probe_changed"]
    assert selection["revised_design"]["simulator"]["model_id"] == event["model_id"]
    assert completed["costs"]["observation_cost"] == completed["apparatus_state"]["rows"] == 51
    assert completed["costs"]["compute_cost"] > 0
    assert completed["task_objective"] == proposed["task_objective"]


def test_model_predicts_systematic_mean_and_conditional_variance_for_a_paid_rollout():
    state = run_simulator_research(max_steps=6)["simulator_state"]
    simulator = ExperimentSimulator(state)
    probe = Probe("two_conditions", ({"x": -.2}, {"x": 1.}))
    blocked = SearchBudget(3)
    assert simulator.adjusted(probe, [0., 0.], baseline_variance=1., budget=blocked) is None
    assert blocked.spent == 0
    paid = SearchBudget(4)
    distribution = simulator.adjusted(probe, [0., 0.], baseline_variance=1., budget=paid)
    assert distribution["mean"] == pytest.approx([.2, .8], abs=1e-6)
    assert distribution["variance"] == pytest.approx([.01, 4.], abs=1e-6)
    assert paid.spent == 4


def test_no_compute_preserves_actual_feedback_until_scoring_and_fitting_can_be_paid():
    simulator, apparatus = ExperimentSimulator(), ObservationApparatus()
    probe = Probe("calibration", tuple({"x": x} for x in (-.2,) * 4 + (1.,) * 4))
    assert simulator.forecast(probe, [0.] * 8, baseline_variance=1., budget=SearchBudget(7)) is None
    for _ in range(2):
        forecast = simulator.forecast(probe, [0.] * 8, baseline_variance=1., budget=SearchBudget(8))
        result = simulator.observe(forecast, apparatus.observe(probe), budget=SearchBudget(0))
        assert result["status"] == "pending_scoring" and result["compute_cost"] == 0
    assert not simulator.rows and len(simulator.deferred) == 2 and not simulator.candidates
    restored = ExperimentSimulator(json.loads(json.dumps(simulator.to_dict())))
    work = SearchBudget(2000)
    result = restored.flush(budget=work)
    assert result["compute_cost"] == work.spent > 16
    assert len(restored.rows) == 16 and not restored.deferred and len(restored.candidates) == 2
    assert restored.current is None


def test_subsequent_counterexamples_withdraw_the_frozen_model_and_restore_original_channel():
    simulator = ExperimentSimulator(run_simulator_research(max_steps=6)["simulator_state"])
    probe = Probe("new_disturbance", ({"x": -.2},))
    for index in range(4):
        forecast = simulator.forecast(probe, [0.], baseline_variance=1., budget=SearchBudget(2))
        work = SearchBudget(2)
        simulator.observe(forecast, [Observation({"x": -.2}, -1.)], budget=work)
        assert work.spent == 2
        if index < 3:
            assert simulator.current is not None
    assert simulator.current is None
    assert simulator.events[-1]["event"] == "withdraw_simulation_model"
    assert simulator.events[-1]["comparison"]["refuted"]
    assert simulator.adjusted(probe, [0.], baseline_variance=1., budget=SearchBudget(0)) == {
        "mean": [0.], "variance": [1.]}


def test_restoring_pending_comparison_preserves_execution_and_total_cost():
    uninterrupted = run_simulator_research()
    partial = json.loads(json.dumps(run_simulator_research(max_steps=4)))
    restored = run_simulator_research(state=partial)
    assert restored == uninterrupted


def test_ordinary_controller_learns_from_its_own_queries_and_uses_restored_channel(monkeypatch):
    from hegel_machine.cognition import controller
    from hegel_machine.cognition.models import ResearchConfig, TaskSpec

    class Lab(ObservationApparatus):
        spec = TaskSpec("channel_runtime", "Predict the measured response", ("x",),
            (Probe("initial", tuple({"x": x} for x in (-.2,) * 4 + (1.,) * 4)),),
            tuple(Probe(f"calibration_{index}", tuple({"x": x} for x in
                  ((-.2, 1.) * 4 if index % 2 else (-.2,) * 4 + (1.,) * 4))) for index in range(20))
                  + (Probe("quiet", ({"x": -.19},)), Probe("loud", ({"x": .99},))),
            max_probe_length=8)

    environment = Lab()
    config = ResearchConfig(observation_budget=32, compute_budget=8000, max_rounds=2,
        use_memory=False, seed=19, max_history=1, state_search=False, transition_probes=False)
    workspace, episodes, simulated = None, [], []
    original = controller.choose_valuable_probe

    def capture(*args, **kwargs):
        selected, details = original(*args, **kwargs)
        if details.get("simulator"):
            simulated.append((selected, details))
        return selected, details

    monkeypatch.setattr(controller, "choose_valuable_probe", capture)
    for _ in range(4):
        result = controller.CognitiveAgent(config).run(environment, research_state=workspace)
        episodes.append(result)
        workspace = json.loads(json.dumps(result.research_state))
    assert all(episode.compute_cost <= config.compute_budget for episode in episodes)
    assert sum(episode.observation_cost for episode in episodes) == environment.state["rows"]
    assert all(any(action.action == "forecast_experiment_observations" for action in episode.actions)
               for episode in episodes)
    simulator = workspace["simulator_state"]
    assert simulator["batches"] >= 8 and simulator["current"] is not None
    assert any(event["event"] == "adopt_simulation_model" for event in simulator["events"])
    assert simulated and simulated[-1][0] is not None
    assert simulated[-1][1]["simulator"]["model_id"] == simulator["current"]["model_id"]
    assert simulated[-1][1]["evaluated_probes"]
