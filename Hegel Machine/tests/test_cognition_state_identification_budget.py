"""A fully funded state-identification trial precedes expensive expansion."""

import pytest

from hegel_machine.cognition.continuous_state import continuous_state_candidates, state_identification_cost
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_continuous_state import trajectories


def _data(two_modes):
    parameters = ({"decay": .91, "second_decay": .36} if two_modes else
                  {"decay": .873, "initial": .8})
    lengths = [17, 29, 37] if two_modes else [7, 15, 25]
    training = trajectories(lengths, seed=84 if two_modes else 41, **parameters)
    validation = trajectories([11, 19], seed=619, **parameters)
    terminal = trajectories([1, 3, 43, 63], seed=894, **parameters)
    no_outputs = [Observation(row.features, row.target,
        tuple({name: value for name, value in previous.items() if name != "__target__"}
              for previous in row.history)) for row in terminal]
    return training, validation, no_outputs


@pytest.mark.parametrize("two_modes", [False, True])
def test_complete_identification_quote_fits_states_that_run_without_past_outputs(two_modes):
    training, _, terminal = _data(two_modes)
    budget = SearchBudget(state_identification_cost(training))
    candidates = continuous_state_candidates(training, budget=budget)
    assert candidates
    # Model choice sees training only; neither the final trajectories nor
    # their past targets can supply poles, initial values or coefficients.
    best = min(candidates, key=lambda candidate: loss(candidate.program, training, 0., .5)).program
    assert loss(best, terminal, 0., .5) < 1e-10
    assert best.describe().count("accumulate") == (2 if two_modes else 1)
    assert all(name != "__target__" for name, _ in best.observed_variables())
    assert 0 < budget.spent < budget.limit  # Early success saves the unused cold-fit allowance.


@pytest.mark.parametrize("funded", [True, False])
def test_native_direction_requires_the_complete_trial_and_validation_bill(funded):
    training, validation, terminal = _data(False)
    spec = TaskSpec("unlabeled_temporal", "Predict new trajectories", ("u", "sensor"), (), (),
        feature_bounds={"u": (-1., 1.), "sensor": (-1., 1.)}, max_probe_length=96)
    search_quote = state_identification_cost(training)
    whole_trial = search_quote + 6 * (len(training) + len(validation))
    # Both limits are small compared with the old general search allowance.
    # The smaller one cannot purchase the complete identified-state proposal
    # and its new-trajectory comparisons, even before the initial affine fit.
    budget = SearchBudget(2 * whole_trial if funded else whole_trial - 1)
    agent = UnifiedResearchAgent(ResearchConfig(compute_budget=budget.limit,
        minimum_improvement=1e-7, max_candidates=16, active_probes=False))
    actions = []
    claims = agent._research_view(spec, training, validation, budget, actions, [], ResearchGraph(),
                                 observation_remaining=0, measurement_remaining=0)
    choices = [item["selected_action"] for item in actions if item["action"] == "choose_research_action"
               and item["selected_action"] is not None]
    identification = [item for item in actions if item["action"] == "identify_state_before_expansion"]
    if funded:
        assert choices[0] == "state_identification"
        assert len(identification) == 1 and identification[0]["candidate_count"] > 0
        assert identification[0]["compute_cost"] <= whole_trial
        assert "coordinate_reconstruction" not in choices
        assert "conditional_reconstruction" not in choices
        best = agent._best(claims)
        assert best["kind"] == "continuous_latent_state"
        assert best["validation_loss"] < 1e-10
        assert loss(best["candidate"].program, terminal, 0., .5) < 1e-10
        # Candidate comparison is not a task-level fresh confirmation.
        assert not any(item["action"] == "validate_frozen_program" for item in actions)
    else:
        assert "state_identification" not in choices
        assert identification == []
    assert budget.spent <= budget.limit
