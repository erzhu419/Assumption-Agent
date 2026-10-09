"""A paid validation residual may propose a tail revision, never certify it."""

from dataclasses import replace

import pytest

from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _view
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


CONFIG = ResearchConfig(observation_budget=384, compute_budget=240000,
    minimum_improvement=1e-7, active_probes=False, max_rounds=4, seed=7907)


def claim(program):
    return {"candidate": Candidate("existing", program, "fit"), "kind": "current_affine",
            "family_id": "T02", "training_loss": 0., "validation_loss": 1.}


@pytest.mark.parametrize("confirmation_target,accepted", [(0., False), (8., True)])
def test_calibration_uses_old_residuals_and_new_confirmation_can_reject_it(
        confirmation_target, accepted):
    agent = UnifiedResearchAgent(CONFIG)
    spec = UnknownTaskLab(12).spec
    original = claim(Program("const", value=0.))
    original["research_action_origin"] = {"context": "pinball:0.9:multiple:unified:coordinates:unexplained",
        "action": "program_revision", "trial": 1, "credited_gain": .4,
        "baseline_program": Program("const", value=-1.).to_dict(), "fallback": -1., "risk_quantile": .9}
    validation = [Observation({"u": 0.}, float(i)) for i in range(10)]
    confirmation = [Observation({"u": 0.}, confirmation_target) for _ in range(10)]
    budget, actions = SearchBudget(100), []

    calibrated = agent._tail_calibration(original, validation, spec, budget, actions)
    # The empirical .9 quantile of 0,...,9 is 8 regardless of the new targets.
    assert actions[0]["offset"] == 8.
    assert calibrated["candidate"].program.predict(confirmation[0]) == 8.
    assert original["candidate"].program.predict(confirmation[0]) == 0.
    chosen, fixed_loss = agent._compare_calibration(original, calibrated, confirmation,
        spec, budget, 0., actions)

    assert (chosen is calibrated) == accepted
    assert chosen is (calibrated if accepted else original)
    assert chosen["research_action_origin"] == original["research_action_origin"]
    assert fixed_loss == 0.
    if accepted:
        assert chosen["validation_loss"] == fixed_loss
        assert chosen["calibration_parent_validation_loss"] == 1.
    assert original["validation_loss"] == 1.
    comparison = actions[-1]
    assert comparison["accepted"] == accepted
    assert comparison["original_loss"] == loss(original["candidate"].program,
                                               confirmation, 0., .9)
    assert comparison["calibrated_loss"] == loss(calibrated["candidate"].program,
                                                 confirmation, 0., .9)
    assert budget.spent == 30


@pytest.mark.parametrize("task,config", [
    (0, CONFIG),
    (12, replace(CONFIG, risk_learning=False)),
    (12, replace(CONFIG, strategy="fixed")),
])
def test_disabled_calibration_keeps_the_original_loss_and_compute(task, config):
    agent = UnifiedResearchAgent(config)
    spec = UnknownTaskLab(task).spec
    original = claim(Program("const", value=0.))
    rows = [Observation({"u": 0.}, 2.), Observation({"u": .5}, 3.)]
    budget, actions = SearchBudget(20), []
    calibrated = agent._tail_calibration(original, rows, spec, budget, actions)
    assert calibrated is None and not actions and budget.spent == 0
    chosen, fixed_loss = agent._compare_calibration(original, calibrated, rows,
        spec, budget, 0., actions)
    assert chosen is original
    assert fixed_loss == loss(original["candidate"].program, rows, 0., spec.risk_quantile)
    assert budget.spent == len(rows) and not actions


def test_unknown_noisy_task_calibrates_paid_validation_before_new_confirmation(monkeypatch):
    laboratory = UnknownTaskLab(12, CONFIG.seed)
    agent = UnifiedResearchAgent(CONFIG, measurement_budget=1536)
    original_propose = agent._tail_calibration
    original_compare = agent._compare_calibration
    proposals, comparisons = [], []

    def propose(current, rows, spec, budget, actions):
        proposed = original_propose(current, rows, spec, budget, actions)
        if proposed is not None:
            proposals.append({"rows": list(rows), "spec": spec,
                "old": current["candidate"].program, "new": proposed["candidate"].program,
                "paid_rows": laboratory.observation_cost,
                "acquisitions": len(laboratory.acquisitions), "action": dict(actions[-1])})
        return proposed

    def compare(current, proposed, rows, spec, budget, fallback, actions):
        if proposed is not None:
            last = proposals[-1]
            assert laboratory.observation_cost == last["paid_rows"] + len(rows)
            assert len(laboratory.acquisitions) == last["acquisitions"] + 1
            assert spec.risk_quantile == .9
            assert proposed["candidate"].program == last["new"]
        chosen, fixed_loss = original_compare(current, proposed, rows, spec,
            budget, fallback, actions)
        if proposed is not None:
            comparisons.append((dict(actions[-1]), list(rows), fallback))
        return chosen, fixed_loss

    monkeypatch.setattr(agent, "_tail_calibration", propose)
    monkeypatch.setattr(agent, "_compare_calibration", compare)
    result = agent.run(laboratory)

    assert proposals and len(proposals) == len(comparisons)
    assert result["risk_quantile"] == .9 and result["loss_kind"] == "pinball"
    acquired = {action["probe_id"]: action for action in result["actions"]
                if action["action"] == "acquire"}
    for proposal, (comparison, confirmation, fallback) in zip(proposals, comparisons):
        paid_validation = []
        for batch in result["batches"][:proposal["acquisitions"]]:
            if acquired[batch["probe"]["probe_id"]]["purpose"] == "compare_on_new_trajectory":
                rows = [Observation(row["features"], row["target"], tuple(row["history"]))
                        for row in batch["observations"]]
                paid_validation.append(_view(rows, proposal["spec"].feature_names))
        assert proposal["rows"] in paid_validation
        assert proposal["rows"] != confirmation
        residuals = sorted(row.target - predict(proposal["old"], row, 0.)
                           for row in proposal["rows"])
        assert proposal["action"]["offset"] == pytest.approx(residuals[21])
        assert proposal["action"]["source_rows"] == 24
        assert comparison["original_loss"] == loss(proposal["old"], confirmation, fallback, .9)
        assert comparison["calibrated_loss"] == loss(proposal["new"], confirmation, fallback, .9)
    final_validation = [action for action in result["actions"]
                        if action["action"] == "validate_frozen_program"][-1]
    assert result["confirmation_loss"] == final_validation["loss"]
    assert result["costs"]["observation_cost"] == laboratory.observation_cost


def test_supported_short_trial_compares_both_frozen_tails_on_its_new_confirmation(monkeypatch):
    source = UnifiedResearchAgent(CONFIG, measurement_budget=1536)
    source.run(UnknownTaskLab(12, CONFIG.seed))
    agent = UnifiedResearchAgent(replace(CONFIG, seed=7908), measurement_budget=1536,
                                 library=source.library)
    # Isolate the short-trial wiring from the separate noisy-adequacy decision.
    # This forced support is not evidence that this noisy task merits adoption.
    monkeypatch.setattr(agent, "_support", lambda *args: {
        "adequate": True, "reason": "isolated_short_trial_wiring"})
    result = agent.run(UnknownTaskLab(12, 7908))
    actions = result["actions"]
    proposal_index = next(index for index, action in enumerate(actions)
                          if action["action"] == "propose_tail_calibration")
    proposal = actions[proposal_index]
    previous = [action for action in actions[:proposal_index] if action["action"] == "acquire"][-1]
    following = [action for action in actions[proposal_index:] if action["action"] == "acquire"][0]
    assert previous["purpose"] == "validate_short_learned_trial"
    assert following["purpose"] == "confirm_short_frozen_program"
    assert previous["probe_id"] != following["probe_id"]
    assert proposal["source_rows"] == previous["observation_cost"]
    confirmation_batch = next(batch for batch in result["batches"]
                              if batch["probe"]["probe_id"] == following["probe_id"])
    rows = [Observation(row["features"], row["target"], tuple(row["history"]))
            for row in confirmation_batch["observations"]]
    comparison = next(action for action in actions[proposal_index:]
                      if action["action"] == "compare_tail_calibration")
    original = Program.from_dict(proposal["parent_program"])
    calibrated = Program.from_dict(proposal["program"])
    assert comparison["original_loss"] == loss(original, rows, result["fallback"], .9)
    assert comparison["calibrated_loss"] == loss(calibrated, rows, result["fallback"], .9)
    assert comparison["accepted"] == (comparison["calibrated_loss"] + CONFIG.minimum_improvement
                                      < comparison["original_loss"])
    assert any(action.get("short_trial") for action in actions)
