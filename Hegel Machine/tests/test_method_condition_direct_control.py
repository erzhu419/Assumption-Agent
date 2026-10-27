"""Condition-only treatment shares paid fits in both version directions."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import research_programs
from test_method_condition_fit_reuse import useful_conditioned_scene, route, fitted_execution
from test_research_program_consolidation import compile_sources


def test_old_and_new_conditions_share_the_same_paid_fit_in_both_directions(tmp_path, monkeypatch):
    agent, lab, old, question, original, calls = useful_conditioned_scene(tmp_path, monkeypatch)
    new = deepcopy(agent.theory_consolidation['candidates'][old['id']])
    route(agent, lab, question)
    cached = deepcopy(fitted_execution(agent, question, old))
    assert cached['basis']['binding']['candidate_version'] == new['version']
    assert cached['basis']['fit_method_ref'] == {'id': old['id'], 'version': old['version']}
    paid_programs = {item['theory_id']: deepcopy(agent.theory_index[item['theory_id']]['program'])
        for item in cached['result']['generated_theories']}
    costs = deepcopy(agent.costs)
    monkeypatch.setattr(research_programs, 'replay_research_program',
        lambda *a, **k: pytest.fail('condition-only rollback repeated a paid fit'))
    for method in (old, new, old):
        agent.theory_consolidation['candidates'][old['id']] = deepcopy(method)
        agent._save('shared/theory_library.json', json.dumps(agent.theory_consolidation))
        route(agent, lab, question)
        current = fitted_execution(agent, question, old)
        assert current['basis']['binding']['candidate_version'] == method['version']
        assert current['basis']['fit_method_ref'] == cached['basis']['fit_method_ref']
        assert current['result']['seed_replay_work'] == original['result']['seed_replay_work']
        assert {key: agent.theory_index[key]['program'] for key in paid_programs} == paid_programs
        assert lab.paid == 0 and agent.costs['observation_cost'] == costs['observation_cost']
    assert calls == [8]


def test_existing_precondition_paid_basis_stays_usable_without_refitting(tmp_path, monkeypatch):
    agent, lab, old, question, original, calls = useful_conditioned_scene(tmp_path, monkeypatch)
    execution = fitted_execution(agent, question, old)
    execution['basis'].pop('fit_method_ref')
    agent._save('shared/theory_library.json', json.dumps(agent.theory_consolidation))
    monkeypatch.setattr(research_programs, 'replay_research_program',
        lambda *a, **k: pytest.fail('existing precondition fit was discarded'))
    route(agent, lab, question)
    current = fitted_execution(agent, question, old)
    assert current['basis']['fit_method_ref'] == {'id': old['id'], 'version': old['version']}
    assert current['result']['seed_replay_work'] == original['result']['seed_replay_work']
    assert calls == [8]


def test_changed_method_body_does_not_borrow_a_fit_via_its_usage_version(tmp_path, monkeypatch):
    agent, lab, old, question, original, calls = useful_conditioned_scene(tmp_path, monkeypatch)
    route(agent, lab, question)
    conditions = deepcopy(agent.theory_consolidation['candidates'][old['id']]['method_use_conditions'])
    learner, compiled = compile_sources(tmp_path / 'different_steps', conditional=True)
    replacement = deepcopy(learner.theory_consolidation['candidates'][
        compiled['frozen_candidates'][0]['candidate_id']])
    replacement.update(id=old['id'], version=old['version'] + 2, method_use_conditions=conditions)
    assert replacement['research_program_contract'] != old['research_program_contract']
    agent.theory_consolidation['candidates'][old['id']] = replacement
    agent._save('shared/theory_library.json', json.dumps(agent.theory_consolidation))
    route(agent, lab, question)
    current = fitted_execution(agent, question, old)
    assert calls == [8, 8, 8]
    assert current['result']['new_compute_cost'] > 0
    assert not current['result'].get('fit_cache_hit')
