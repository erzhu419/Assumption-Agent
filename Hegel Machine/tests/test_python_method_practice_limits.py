"""Method practice honors the experiment's actual observation and confirmation limits."""

from dataclasses import replace
from random import Random

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.anonymous_method_learning import paid_scene_trial_preparer
from hegel_machine.cognition.python_method_learning import PythonArchiveMethodLearner
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH


class PublicLab:
    def public_spec(self):
        return {'controls': ['u'], 'max_probe_length': 24}


def make_learner(tmp_path, monkeypatch, practice_limits=None):
    calls = []

    class PracticeAgent:
        confirmation_rng = Random(0)

        def execute_method(self, laboratory, max_steps):
            calls[-1]['steps'] = max_steps
            return {'model': None, 'status': 'method_stopped'}

    def prepare_trial(case, workspace, source, limits):
        calls.append({'case': case, 'workspace': workspace, 'source': source, 'limits': limits})
        return {'agent': PracticeAgent(), 'laboratory': PublicLab(), 'inherited_candidates': []}

    learner = PythonArchiveMethodLearner(None, tmp_path,
        knowledge={'files': {METHOD_PATH: 'def select_actions(spec,records,memory): return {"stop":"done"}\n'}},
        cases={'sample': {'confirmation_seed': 17}}, lab_factory=lambda _: PublicLab(),
        config=InquiryLLMConfig(observation_budget=1800, compute_budget=900000,
                               execution_seconds=500.),
        prepare_trial=prepare_trial, practice_limits=practice_limits)
    monkeypatch.setattr(learner, '_record_practice',
                        lambda case, directory, source, agent, result, selected: result)
    return learner, calls


def test_legacy_practice_limits_and_public_contract_are_preserved(tmp_path, monkeypatch):
    learner, calls = make_learner(tmp_path, monkeypatch)
    learner._practice('sample')
    limits = calls[0]['limits']
    assert (limits.observation_budget, limits.confirmation_rows, limits.compute_budget,
            limits.execution_seconds, limits.max_model_calls, limits.token_budget) == (
                384, 24, 360000, 120., 0, 0)
    assert calls[0]['steps'] == 12
    assert 'Confirmation costs 24 fresh rows.' in learner.common_interface
    assert '24 new confirmation rows.' in learner.common_interface


def test_configured_576_rows_and_96_confirmation_reach_prepared_trial(tmp_path, monkeypatch):
    requested = InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
        compute_budget=250000, execution_seconds=300., max_model_calls=0, token_budget=0)
    learner, calls = make_learner(tmp_path, monkeypatch, requested)
    learner._practice('sample')
    assert calls[0]['limits'] == requested
    assert 'Confirmation costs 96 fresh rows.' in learner.common_interface
    assert '96 new confirmation rows.' in learner.common_interface
    assert 'Three paid 24-row control trajectories' in learner.common_interface
    assert calls[0]['steps'] == 12


def test_practice_cannot_exceed_remaining_author_account(tmp_path, monkeypatch):
    requested = InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
        compute_budget=250000, execution_seconds=300., max_model_calls=0, token_budget=0)
    learner, calls = make_learner(tmp_path, monkeypatch, requested)
    learner.costs.update(observation_cost=1400, compute_cost=700000, python_execution_seconds=450.)
    learner._practice('sample')
    limits = calls[0]['limits']
    assert (limits.observation_budget, limits.compute_budget, limits.execution_seconds) == (400, 200000, 50.)
    assert limits.confirmation_rows == 96


def test_restored_scene_can_use_exactly_remaining_confirmation_allowance(tmp_path, monkeypatch):
    requested = InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
        compute_budget=250000, execution_seconds=300., max_model_calls=0, token_budget=0)
    learner, calls = make_learner(tmp_path, monkeypatch, requested)
    restored = paid_scene_trial_preparer({}, tmp_path / 'retained.json', lambda _: PublicLab())
    learner.prepare_trial.initial_observation_cost = restored.initial_observation_cost
    learner.costs['observation_cost'] = learner.config.observation_budget - 96
    outcome = learner._practice('sample')
    assert outcome['status'] == 'method_stopped'
    assert len(calls) == 1
    assert calls[0]['limits'].observation_budget == calls[0]['limits'].confirmation_rows == 96


@pytest.mark.parametrize('exhausted', ['author_observations', 'author_compute', 'practice_observations'])
def test_insufficient_prefix_and_confirmation_budget_does_not_prepare_or_purchase(tmp_path, monkeypatch, exhausted):
    requested = InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
        compute_budget=250000, execution_seconds=300., max_model_calls=0, token_budget=0)
    if exhausted == 'practice_observations':
        requested = replace(requested, observation_budget=167)
    learner, calls = make_learner(tmp_path, monkeypatch, requested)
    if exhausted == 'author_observations':
        learner.costs['observation_cost'] = learner.config.observation_budget - 167
    if exhausted == 'author_compute':
        learner.costs['compute_cost'] = learner.config.compute_budget - 167
    before = dict(learner.costs)
    result = learner._practice('sample')
    assert 'paid prefix and fresh confirmation' in result['error']
    assert calls == []
    assert learner.costs == before
