"""A paid failed method becomes a paired step study and an ordinary method."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.method_feedback import _read, method_step_research_inputs
from test_method_use_question_route import measured_method, publish


def step_question(tmp_path):
    agent, lab, method, parent, binding, _ = measured_method(tmp_path)
    feedback, = agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs']
    question = publish(agent, method, parent, binding)
    question['method_use_plan']['step_revision'] = {'feedback_ref': feedback, 'max_candidates': 8}
    return agent, lab, method, parent, question


def test_step_quote_reads_original_split_without_another_fit_or_world_measurement(tmp_path, monkeypatch):
    agent, lab, method, _, question = step_question(tmp_path)
    import hegel_machine.cognition.research_programs as programs

    def forbidden(*args, **kwargs):
        raise AssertionError('quoting must not refit or collect new world evidence')

    monkeypatch.setattr(programs, 'replay_research_program', forbidden)
    before = deepcopy(agent.costs)
    quote = method_step_research_inputs(agent, method['id'],
        question['method_use_plan']['step_revision']['feedback_ref'])
    assert quote['status'] == 'quoted'
    assert quote['required_compute'] > 0 and quote['new_fits'] == 0
    assert len(quote['training']) == len(quote['validation']) == 8
    assert agent.costs == before and lab.paid == 0


def test_unfunded_step_question_keeps_paid_records_and_does_not_refit(tmp_path, monkeypatch):
    agent, lab, method, _, question = step_question(tmp_path)
    import hegel_machine.cognition.research_programs as programs

    def forbidden(*args, **kwargs):
        raise AssertionError('partial allocation must not repeat an old fit')

    monkeypatch.setattr(programs, 'replay_research_program', forbidden)
    before, original = deepcopy(agent.costs), deepcopy(method)
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 1}}, lab)
    assert result['status'] == 'scoped_unknown' and result['required_compute'] > 1
    assert result['new_compute_cost'] == 0 and agent.costs == before
    assert agent.theory_consolidation['candidates'][method['id']] == original
    assert 'completed_agenda_basis' not in question


def test_actual_step_revision_is_consumed_by_ordinary_transfer_without_claiming_independence(tmp_path, monkeypatch):
    agent, lab, method, parent, question = step_question(tmp_path)
    original, pool = deepcopy(method), set(agent.theory_consolidation['candidates'])
    from hegel_machine.cognition import research_programs as programs
    quote = method_step_research_inputs(agent, method['id'],
        question['method_use_plan']['step_revision']['feedback_ref'])
    old_inputs = [seed['paid_baseline']['basis']['replay_inputs'] for seed in quote['seeds']]
    replay = programs.replay_research_program

    def changed_step_only(payload, *args, **kwargs):
        assert programs.research_program_replay_inputs(payload) not in old_inputs
        return replay(payload, *args, **kwargs)

    monkeypatch.setattr(programs, 'replay_research_program', changed_step_only)
    result = agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 2000}}, lab)
    assert result['status'] == 'method_steps_revised'
    study = result['method_step_research']
    revised = agent.theory_consolidation['candidates'][method['id']]
    assert _read(agent, revised['previous_version_ref']) == original
    assert set(agent.theory_consolidation['candidates']) == pool
    assert revised['source_ids'] == original['source_ids']
    assert revised['independent_source_count'] == original['independent_source_count']
    assert revised['research_program_source_refs'] != original['research_program_source_refs']
    assert 'method_use_conditions' not in revised
    assert not study['independent_support'] and not result['support_updated']
    assert all(seed['methods'][0]['reused'] and seed['methods'][0]['new_compute_cost'] == 0
               for seed in study['seed_studies'])
    proof = _read(agent, revised['method_step_revision']['actual_step_revision_ref'])
    assert proof['status'] == 'completed' and proof['data_role'] == 'exposed_development'
    # The next condition-learning call cannot relabel old-body failures as
    # measured outcomes of the revised, as-yet-untried executable method.
    retained = deepcopy(revised)
    learned = agent._execute('run', {'operator': 'theory_consolidation',
        'object_type': 'method_use_condition', 'candidate_ids': [method['id']],
        'mission': {'purpose': 'method_learning', 'allocation': {'compute': 10}},
        'allocation': {'compute': 10}}, lab)
    assert not learned['frozen_candidates']
    assert agent.theory_consolidation['candidates'][method['id']] == retained
    follow = result['next_action']
    executed = agent._execute(follow['action'], follow['arguments'], lab)
    assert executed['generated_theories'] and executed['search_completed']
    assert all(item['binding']['candidate_version'] == revised['version']
               for item in executed['generated_theories'])
    assert lab.paid == agent.costs['model_calls'] == 0
    assert not executed['independent_confirmation']


def test_complete_step_study_is_not_repeated_by_ordinary_agenda(tmp_path, monkeypatch):
    agent, lab, _, _, question = step_question(tmp_path)
    arguments = {'operator': 'research_route', 'research_question': {'id': question['id']},
                 'allocation': {'compute': 2000}}
    first = agent._execute('run', arguments, lab)
    assert first['status'] == 'method_steps_revised'
    import hegel_machine.cognition.research_programs as programs

    def forbidden(*args, **kwargs):
        raise AssertionError('completed development step study repeated')

    monkeypatch.setattr(programs, 'replay_research_program', forbidden)
    repeated = agent._execute('run', arguments, lab)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0


def test_registered_quote_reuses_paid_gate_readouts(tmp_path, monkeypatch):
    agent, _, method, _, question = step_question(tmp_path)
    ref = question['method_use_plan']['step_revision']['feedback_ref']
    quote = method_step_research_inputs(agent, method['id'], ref)
    saved = agent.workspace / 'step_quote.json'
    saved.write_text(json.dumps({'feedback_ref': ref, 'max_candidates': 8,
                                'seeds': quote['seeds']}))

    def forbidden(*args, **kwargs):
        raise AssertionError('registered paid quote readouts repeated')

    monkeypatch.setattr('hegel_machine.cognition.research_programs.quote_research_step_revisions', forbidden)
    reused = method_step_research_inputs(agent, method['id'], ref,
        quote_ref={'file': str(saved), 'json_path': []})
    assert reused['status'] == 'quoted' and reused['coordinate_readout_compute'] == 0
    assert reused['required_compute'] == quote['required_compute']


def test_common_checkpoint_retains_completed_slot_before_an_interrupted_study(tmp_path, monkeypatch):
    agent, _, method, _, question = step_question(tmp_path)
    from hegel_machine.cognition import method_feedback as feedback
    ref = question['method_use_plan']['step_revision']['feedback_ref']
    original_store, saved = feedback._store, []

    def interrupted(owner, name, value, **kwargs):
        result = original_store(owner, name, value, **kwargs)
        if name.startswith('step_study_'):
            saved.append(result)
            raise RuntimeError('stop after the paid slot checkpoint committed')
        return result

    monkeypatch.setattr(feedback, '_store', interrupted)
    with pytest.raises(RuntimeError, match='paid slot checkpoint'):
        feedback.research_method_steps(agent, method['id'], ref, 2000)
    partial = _read(agent, saved[0])
    assert partial['total_new_compute_cost'] > 0 and not partial['execution_complete']
    first = partial['seed_studies'][0]
    assert first['methods'][0]['reused'] and first['methods'][1]['execution_complete']
    assert first['methods'][1]['new_compute_cost'] > 0
    assert first['selected_revision'] is None
    assert method['version'] == agent.theory_consolidation['candidates'][method['id']]['version']
