"""Paid view work reaches the real parent account even when research raises."""
import json

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.models import ResearchConfig


@pytest.mark.parametrize('fails', [False, True])
def test_actual_view_budget_settles_once_on_success_and_failure(tmp_path, monkeypatch, fails):
    agent = CommonObjectResearchAgent(ResearchConfig(seed=73, observation_budget=96,
        compute_budget=80000, minimum_improvement=1e-7, max_rounds=2, active_probes=True),
        workspace=tmp_path, organization='fixed_pr2', measurement_budget=1152,
        source_task_ref={'target': 'view_accounting_control'})
    observed = {}

    def research(spec, training, validation, local, *args, **kwargs):
        parent = agent._paid_native_state['budget']
        assert local is not parent
        observed.update(parent=parent, before=parent.spent, debits=[], local=local)
        spend = parent.spend

        def record(cost):
            observed['debits'].append(cost)
            return spend(cost)

        parent.spend = record
        assert local.spend(1173)
        if fails:
            raise RuntimeError('control: paid response view interrupted')
        return []

    monkeypatch.setattr(agent, '_research_view', research)
    with AnonymousObjectLab(objects=3, readout='focus', apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as laboratory:
        if fails:
            with pytest.raises(RuntimeError, match='paid response view interrupted'):
                agent.run(laboratory)
        else:
            agent.run(laboratory)
    assert observed['debits'].count(1173) == 1
    assert observed['parent'].spent >= observed['before'] + 1173
    if fails:
        saved = json.loads((tmp_path / 'failed_paid_state.json').read_text())
        assert saved['confirmed_compute_lower_bound'] == observed['before'] + 1173
        assert agent.failure_compute_lower_bound == saved['confirmed_compute_lower_bound']
