"""Usage-only method revisions preserve real Common's paid parameter fits."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import question_queries, research_programs
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_feedback import _read
from hegel_machine.cognition.models import Observation
from test_common_method_feedback import freeze_usage, scene, use_and_compare
from test_common_method_future_returns import acquire_new_rows
from test_inquiry_revision import PublicLab
from test_research_program_consolidation import compile_sources
from test_theory_consolidation import agent_scene


def route(agent, lab, question, *, target=None):
    return agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 3000},
        'mission': {},  # Resume the task, not the freeze operation's 10-unit theory mission.
        **({'target': target} if target is not None else {})}, lab)


def fitted_execution(agent, question, method):
    key = f'research_program_transfer:frozen_recall:{question["id"]}@1:{method["id"]}'
    return agent.theory_consolidation['executions'][key]


def useful_conditioned_scene(tmp_path, monkeypatch):
    calls = []
    replay = research_programs.replay_research_program

    def counted(*args, **kwargs):
        calls.append(len(args[1]))
        return replay(*args, **kwargs)

    monkeypatch.setattr(research_programs, 'replay_research_program', counted)
    agent, lab, method, question = scene(tmp_path, useful=True)
    # Both role bindings below are actual measured inputs, present before fitting.
    for record in agent.observations:
        for row in record['rows']:
            row['features']['v'] = row['features']['u'] ** 2
    agent._write_evidence()
    use_and_compare(agent, lab, question)
    execution = deepcopy(fitted_execution(agent, question, method))
    assert execution['result']['search_completed'] and calls == [8]
    feedback = agent.theory_consolidation['method_feedback'][method['id']]
    original = _read(agent, feedback['outcome_refs'][0])
    assert original['working_comparison']['after_loss'] < original['working_comparison']['before_loss']
    freeze_usage(agent, lab, method)
    revised = agent.theory_consolidation['candidates'][method['id']]
    assert revised['version'] == method['version'] + 1
    assert revised['method_use_conditions']['usage_version'] == {
        'id': method['id'], 'version': method['version']}
    assert revised['research_program_contract'] == method['research_program_contract']
    return agent, lab, method, question, execution, calls


def test_condition_only_parent_route_reuses_paid_fit_and_only_pays_for_new_checks(tmp_path, monkeypatch):
    agent, lab, method, question, original, calls = useful_conditioned_scene(tmp_path, monkeypatch)
    programs = {item['theory_id']: deepcopy(agent.theory_index[item['theory_id']]['program'])
                for item in original['result']['generated_theories']}
    assert programs and original['result']['new_compute_cost'] > 0
    route(agent, lab, question)
    reused = fitted_execution(agent, question, method)
    assert reused['basis']['binding']['candidate_version'] == method['version'] + 1
    assert reused['result']['fit_cache_hit'] and reused['result']['new_compute_cost'] == 0
    assert reused['result']['seed_replay_work'] == original['result']['seed_replay_work']
    assert calls == [8]

    checks = []
    evaluate = question_queries.evaluate_on_model

    def counted_check(expression, predict, *args, **kwargs):
        def counted(value):
            checks.append(value)
            return predict(value)
        return evaluate(expression, counted, *args, **kwargs)

    monkeypatch.setattr(question_queries, 'evaluate_on_model', counted_check)
    target = deepcopy(question['theory_target'])
    target['evidence_checks'] = [{'source_ref': {
        'file': str(agent.workspace / 'laboratory_observations.json'), 'json_path': [0, 'rows', 0]},
        'context_ref': deepcopy(target['context_ref']),
        'observation': deepcopy(agent.observations[0]['rows'][0]), 'tolerance': 1e-6}]
    paid = agent.costs['compute_cost']
    route(agent, lab, question, target=target)
    checked = fitted_execution(agent, question, method)['result']
    assert checked['fit_cache_hit'] and checked['new_compute_cost'] == len(programs) == len(checks)
    assert agent.costs['compute_cost'] - paid >= checked['new_compute_cost']
    assert all(item['checks'][0]['status'] == 'passed' for item in checked['generated_theories'])
    assert checked['seed_replay_work'] == original['result']['seed_replay_work']
    assert {key: agent.theory_index[key]['program'] for key in programs} == programs
    repeated = route(agent, lab, question, target=target)
    assert len(checks) == len(programs) and calls == [8]
    assert repeated['frozen_recall']['new_compute_cost'] == 0
    assert fitted_execution(agent, question, method)['result'] == checked
    assert lab.paid == 0 and not agent.confirmations and agent.costs['model_calls'] == 0


@pytest.mark.parametrize('changed', ['paid_data', 'split', 'binding', 'method_contract'])
def test_real_fit_changes_still_refit_after_condition_revision(tmp_path, monkeypatch, changed):
    agent, lab, method, question, original, calls = useful_conditioned_scene(tmp_path, monkeypatch)
    route(agent, lab, question)
    assert fitted_execution(agent, question, method)['result']['fit_cache_hit'] and calls == [8]
    target = deepcopy(question['theory_target'])
    if changed == 'paid_data':
        lab = acquire_new_rows(agent, lab, question)
    elif changed == 'split':
        target['research_program_fitting']['validation_records'] = [0]
    elif changed == 'binding':
        target['role_signature'] = {'v': target['role_signature']['u']}
    else:
        learner, result = compile_sources(tmp_path / 'changed-method', conditional=True)
        replacement = deepcopy(learner.theory_consolidation['candidates'][
            result['frozen_candidates'][0]['candidate_id']])
        replacement.update(id=method['id'], version=method['version'] + 2)
        assert replacement['research_program_contract'] != method['research_program_contract']
        agent.theory_consolidation['candidates'][method['id']] = replacement
        agent._save('shared/theory_library.json', json.dumps(agent.theory_consolidation))
    paid = agent.costs['compute_cost']
    route(agent, lab, question, target=target)
    refreshed = fitted_execution(agent, question, method)['result']
    assert calls == {'paid_data': [8, 10], 'split': [8, 8],
                     'binding': [8, 8], 'method_contract': [8, 8, 8]}[changed]
    assert not refreshed.get('fit_cache_hit')
    assert refreshed['new_compute_cost'] > 0
    assert agent.costs['compute_cost'] - paid >= refreshed['new_compute_cost']
    assert sum(item['compute_cost'] for item in original['result']['seed_replay_work']) > 0
    assert not agent.confirmations and agent.costs['model_calls'] == 0


def test_new_episode_same_count_buys_its_own_rows_before_reusing_an_archived_method(tmp_path, monkeypatch):
    source, lab, method, question, original, calls = useful_conditioned_scene(tmp_path, monkeypatch)
    route(source, lab, question)
    fitted_execution(source, question, method)['basis'].pop('paid_evidence_ref')
    source._save('shared/theory_library.json', json.dumps(source.theory_consolidation))
    source._prepare_program_state()
    memory = source.result('researching')['outgoing_knowledge']
    source_library = (source.workspace / 'shared/theory_library.json').read_bytes()
    source_costs = deepcopy(source.costs)
    target = CommonArchiveInquiryAgent(None, tmp_path / 'new-episode', arm='hegel',
        source_task=False, knowledge=memory, config=source.config)
    assert target.rows == [] and target.costs['observation_cost'] == 0
    measured = PublicLab()
    measured.public_spec = lab.public_spec
    current = target._open_question({key: deepcopy(question[key]) for key in
        ('id', 'question', 'parent_id', 'depends_on', 'theory_target')})
    target._execute('query', {'research_question': {'id': current['id']}, 'actions': [
        {'kind': 'trajectory', 'sequence': [{'u': i + record * .1} for i in range(8)]}
        for record in range(2)]}, measured)
    assert len(target.rows) == len(source.rows) == measured.paid == 16
    assert [row.target for row in target.rows] != [row.target for row in source.rows]
    route(target, measured, current)
    result = fitted_execution(target, current, method)['result']
    assert calls == [8, 8] and not result.get('fit_cache_hit') and not result.get('cache_hit')
    assert result['new_compute_cost'] > 0
    fitted, = result['generated_theories']
    program = Program.from_dict(target.theory_index[fitted['theory_id']]['program'])
    assert program.predict(Observation({'u': 3.}, 0.)) == pytest.approx(3., abs=1e-8)
    assert source.costs == source_costs
    assert (source.workspace / 'shared/theory_library.json').read_bytes() == source_library


def test_old_condition_fit_recovers_its_paid_view_from_feedback_after_restore(tmp_path, monkeypatch):
    source, lab, method, question, original, calls = useful_conditioned_scene(tmp_path, monkeypatch)
    # This is the real saved pre-identity basis, with its original paid feedback intact.
    fitted_execution(source, question, method)['basis'].pop('paid_evidence_ref')
    source._save('shared/theory_library.json', json.dumps(source.theory_consolidation))
    source._prepare_program_state()
    previous = source.result('researching')
    restored, _ = agent_scene(tmp_path / 'restored')
    fresh = PublicLab()
    fresh.costs = {'observation_cost': 0}
    fresh.public_spec = lab.public_spec
    restored.restore_practice_research(previous, source.workspace, fresh, confirmation_seed=17)
    paid = deepcopy(restored.costs)
    route(restored, fresh, question)
    current = fitted_execution(restored, question, method)
    assert calls == [8] and current['result']['fit_cache_hit']
    assert current['result']['new_compute_cost'] == 0
    assert current['result']['seed_replay_work'] == original['result']['seed_replay_work']
    group = restored.theory_consolidation['method_feedback'][method['id']]
    original_feedback = _read(restored, group['outcome_refs'][0])
    assert _read(restored, current['basis']['paid_evidence_ref']) == _read(
        restored, original_feedback['evidence_basis']['paid_evidence_ref'])
    assert restored.costs['observation_cost'] == paid['observation_cost'] and fresh.paid == 0
