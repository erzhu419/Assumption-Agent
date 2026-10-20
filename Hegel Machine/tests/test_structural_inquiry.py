"""Authored structural research is executable; fixtures do not test LLM ability."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_llm_inquiry import RestorableConfirmationLab
from test_common_archive_online_research import ScriptedProvider
from test_inquiry_revision import action
from test_scientific_generalization import execute


ROOT = {'id': 'task', 'question': 'Predict the public response',
        'parent_id': None, 'depends_on': []}
RELATION = {'id': 'structural_response', 'question': 'Which authored relation survives?',
            'parent_id': 'task', 'depends_on': []}
U = Program('feature', name='u')
SQUARE = Program('mul', children=(U, U))


def binding_source(hypotheses, *, additions=(), derived=False):
    declarations = json.dumps({'hypotheses': hypotheses,
                               'candidate_additions': list(additions)})
    if derived:
        transform = """def represent(row):
    return {'features': {'shifted': row['features']['u'] + 5.},
            'history': [{'shifted': p['u'] + 5.} for p in row['history']]}
"""
    else:
        transform = """def represent(row):
    return {'features': row['features'], 'history': row['history']}
"""
    return f"""import json
DECLARATIONS = json.loads({declarations!r})
{transform}
def bind(spec, observations, memory):
    assert spec['controls'] == ['u']
    assert memory['question_id'] == 'structural_response'
    previous = memory['previous_research']
    if previous:
        assert previous['round_index'] >= 0 and previous['candidates']
    rows = []
    for record in observations:
        for row, index in zip(record['rows'], record['row_indices']):
            rows.append({{**represent(row), 'target': row['target'], 'evidence': [index]}})
    return {{**DECLARATIONS, 'rows': rows,
            'probes': [{{'id': 'same', 'features': {{'u': 1.}}, 'actions': [{{'kind': 'trajectory', 'sequence': [{{'u': 1.}}]}}]}},
                       {{'id': 'separate', 'features': {{'u': 2.}}, 'actions': [{{'kind': 'trajectory', 'sequence': [{{'u': 2.}}]}}]}}] if not {derived!r} else []}}
"""


def prepared(path, *, arm='hegel', source=None, provider=None, calls=20):
    return CommonArchiveInquiryAgent(provider or ScriptedProvider([]), path,
        arm=arm, source_task=False, confirmation_seed=1709,
        config=InquiryLLMConfig(observation_budget=30, confirmation_rows=2,
            compute_budget=3000, execution_seconds=30., max_model_calls=calls),
        knowledge={'files': {
            'shared/binding.py': source or binding_source([U.to_dict(), SQUARE.to_dict()]),
            'shared/knowledge.json': json.dumps({'hypotheses': [], 'research_programs': [],
                                                'support_questions': []})}})


def update(agent, lab):
    return execute(agent, lab, 'run', {'operator': 'research_update',
        'path': 'shared/binding.py', 'research_question': RELATION})


@pytest.mark.parametrize('arm', ['hegel', 'code_react'])
def test_authored_probe_refutes_frozen_alternative_and_preserves_paid_history(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm=arm), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [action(0., 1.)]})
    before_paid, before_actions = lab.paid, deepcopy(lab.actions)
    first = update(agent, lab)
    bank = deepcopy(agent.relation_research[RELATION['id']])

    assert bank['diagnosis'] == 'discriminating_experiment_available'
    assert {item['fit_mse'] for item in bank['candidates']} == {0.}
    assert bank['selected_probe']['predictions'] == [2., 4.]
    assert lab.paid == before_paid and lab.actions == before_actions
    assert agent.model is None and not agent.confirmations
    assert first['next_action']['arguments']['actions'] == [action(2.)]

    selected = first['next_action']
    bought = execute(agent, lab, selected['action'], selected['arguments'])
    assert 'after_observation_error' not in bought
    assert bought['after_observation_results'][0]['operator'] == 'research_update'
    revised = agent.relation_research[RELATION['id']]
    assert revised['round_index'] == 1
    assert revised['fit_evidence'] == [0, 1]
    assert revised['validation_evidence'] == [2]
    assert revised['available_evidence'] == [0, 1, 2]  # two records, three responses
    candidates = {json.dumps(item['program'], sort_keys=True): item for item in revised['candidates']}
    old_square = candidates[json.dumps(SQUARE.to_dict(), sort_keys=True)]
    assert old_square['status'] == 'refuted' and old_square['validation_mse'] == 4.
    assert old_square['fit_evidence'] == old_square['proposed_after'] == [0, 1]
    assert revised['rejected_candidates'][0]['program'] == SQUARE.to_dict()
    assert revised['selected_program'] == U.to_dict()
    assert not revised['confirmation_evidence']
    assert lab.paid == 3 and len(lab.actions) == 2
    assert agent.model is None and not agent.confirmations

    # A second update does not invent evidence, resurrect the loser or repurchase.
    update(agent, lab)
    retained = agent.relation_research[RELATION['id']]
    assert retained['rejected_candidates'][0]['program'] == SQUARE.to_dict()
    assert retained['candidates'][0]['proposed_after'] == [0, 1]
    assert lab.paid == 3 and len(lab.actions) == 2


def test_explicit_addition_is_screened_without_replacing_existing_bank(tmp_path):
    agent, lab = prepared(tmp_path), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [action(0., 1.)]})
    update(agent, lab)
    original = deepcopy(agent.relation_research[RELATION['id']]['candidates'])
    zero = Program('const', value=0.).to_dict()
    execute(agent, lab, 'write', {'path': 'shared/binding.py',
        'code': binding_source([zero], additions=[zero])})
    update(agent, lab)
    current = agent.relation_research[RELATION['id']]
    for retained, old in zip(current['candidates'][:2], original):
        for key in ('program', 'fit_mse', 'fit_rows', 'fit_evidence', 'proposed_after', 'status'):
            assert retained[key] == old[key]
    added = current['candidates'][2]
    assert added['program'] == zero and added['status'] == 'refuted'
    assert added['fit_mse'] == .5 and added['validation_mse'] is None
    assert current['rejected_candidates'][0]['program'] == zero
    assert len(current['exported_models']) == 2
    assert agent.model is None and lab.paid == 2


def test_exported_derived_coordinates_use_frozen_binding_on_public_inputs(tmp_path):
    shifted = Program('feature', name='shifted')
    law = Program('sub', children=(shifted, Program('const', value=5.)))
    source = binding_source([law.to_dict()], derived=True)
    agent, lab = prepared(tmp_path, source=source), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [action(0., 3.)]})
    result = update(agent, lab)
    models = result['relation_research']['exported_models']
    assert len(models) == 1 and not result['relation_research']['export_errors']
    model = models[0]['model']
    assert (agent.workspace / model).is_file()
    execute(agent, lab, 'write', {'path': 'shared/binding.py',
        'code': source.replace("row['features']['u'] + 5.", "row['features']['u'] + 50.")})
    execute(agent, lab, 'query', {'actions': [action(7.)]})
    scored = execute(agent, lab, 'evaluate', models[0]['evaluate_action']['arguments'])
    assert scored['returncode'] == 0 and scored['mse'] == pytest.approx(0.)
    assert agent.evaluated_model['evaluated_rows'] == 3
    assert agent.model is None and not agent.confirmations


class HistoryResponseLab(RestorableConfirmationLab):
    """The public fixture distinguishes a missing predecessor from a zero one."""

    def execute(self, selected):
        record = super().execute(selected)
        for row in record['rows']:
            row['target'] = row['history'][-1]['u'] if row['history'] else 17.
        return record


def test_history_binding_and_export_preserve_missing_and_zero_predecessors(tmp_path):
    law = Program('if', children=(Program('available', lag=1),
        Program('lag', name='u', lag=1), Program('const', value=17.)))
    source = binding_source([law.to_dict()])
    agent, lab = prepared(tmp_path, source=source), HistoryResponseLab()
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [action(0., 4., 9.)]})
    result = update(agent, lab)
    entries = agent.relation_research[RELATION['id']]['bound_rows']
    assert entries[0]['history'] == [] and entries[0]['target'] == 17.
    assert entries[1]['history'] == [{'u': 0.}] and entries[1]['target'] == 0.
    assert entries[2]['history'] == [{'u': 0.}, {'u': 4.}]
    model = result['relation_research']['exported_models'][0]['model']
    scored = execute(agent, lab, 'evaluate', {'model': model})
    assert scored['returncode'] == 0 and scored['mse'] == 0.
    assert not agent.confirmations


def test_paid_response_references_are_not_laboratory_record_positions(tmp_path):
    agent, lab = prepared(tmp_path), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [action(0., 1.)]})
    # Row 1 is valid although record 1 does not yet exist.
    assert update(agent, lab)['relation_research']['fit_evidence'] == [0, 1]
    source = (agent.workspace / 'shared/binding.py').read_text()
    source = source.replace("'evidence': [index]", "'evidence': [index + 2]")
    execute(agent, lab, 'write', {'path': 'shared/binding.py', 'code': source})
    with pytest.raises(ValueError, match='actual observation indices'):
        update(agent, lab)
    assert lab.paid == 2 and len(lab.actions) == 1


def test_child_return_compact_factcard_and_restore_retain_same_bank(tmp_path):
    original, lab = prepared(tmp_path / 'original'), RestorableConfirmationLab()
    execute(original, lab, 'query', {'question': ROOT, 'actions': [action(0., 1.)]})
    first = update(original, lab)
    selected = first['next_action']
    execute(original, lab, selected['action'], selected['arguments'])
    original._prepare_program_state()
    previous = original.result('interrupted_after_structural_research')
    returned = original._request(lab)
    parent = next(item for item in returned['online_research']['parent_returns']
                  if item['parent_id'] == 'task')
    answer = parent['answers'][-1]
    assert answer['question_id'] == RELATION['id'] and answer['operator'] == 'research_update'
    assert answer['answer']['program'] == U.to_dict()
    assert answer['evidence'] == [0, 1, 2]
    fact = returned['evidence_notebook']['facts']['relation:structural_response']
    assert fact['kind'] == 'executed_structural_research'
    assert fact['bound_row_count'] == 3 and 'bound_rows' not in fact
    assert len(fact['rejected_candidates']) == 1

    restored, fresh = prepared(tmp_path / 'restored'), RestorableConfirmationLab()
    report = restored.restore_practice_research(previous, original.workspace, fresh,
                                                confirmation_seed=1709)
    assert report['new_observations'] == 0
    assert restored.relation_research == original.relation_research
    assert restored.costs == original.costs and fresh.paid == 3 and fresh.actions == []
    round_index = restored.relation_research[RELATION['id']]['round_index']
    update(restored, fresh)
    current = restored.relation_research[RELATION['id']]
    assert current['round_index'] == round_index + 1
    assert current['rejected_candidates'][0]['program'] == SQUARE.to_dict()
    assert current['candidates'][0]['proposed_after'] == [0, 1]
    assert fresh.paid == 3 and fresh.actions == [] and restored.model is None


def test_structural_child_executes_in_shared_llm_loop_and_returns_to_parent(tmp_path):
    provider = ScriptedProvider([
        ('run', {'operator': 'inquire', 'question': RELATION, 'max_model_calls': 3}),
        ('run', {'operator': 'research_update', 'path': 'shared/binding.py'}),
        ('finish', {'lesson': 'The declared alternatives need the returned separating probe.'}),
        ('run', {'code': "print('Parent receives the child result before deciding acquisition')"}),
    ])
    agent, lab = prepared(tmp_path, provider=provider, calls=4), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [action(0., 1.)]})
    result = agent.run(lab)
    assert result['status'] == 'model_call_budget_exhausted'
    assert provider.requests[1]['llm_inquiry']['question_id'] == RELATION['id']
    parent = provider.requests[3]
    assert not parent.get('llm_inquiry')
    returned = next(item for item in parent['online_research']['parent_returns']
                    if item['parent_id'] == ROOT['id'])['answers'][-1]
    assert returned['operator'] == 'research_update'
    assert returned['question_id'] == RELATION['id']
    assert returned['answer']['diagnosis'] == 'discriminating_experiment_available'
    assert returned['evidence'] == [0, 1] and len(returned['answer']['models']) == 2
    assert result['relation_research'][RELATION['id']]['round_index'] == 0
    assert lab.paid == 2 and len(lab.actions) == 1
    assert result['model'] is None and not agent.confirmations


def test_refit_factcards_keep_each_actual_model_and_complete_trajectory_split(tmp_path):
    agent, lab = prepared(tmp_path), RestorableConfirmationLab()
    execute(agent, lab, 'write', {'path': 'identity.py', 'code':
        "def fit(rows): pass\ndef predict(row): return row['features']['u']\n"})
    execute(agent, lab, 'write', {'path': 'zero.py', 'code':
        "def fit(rows): pass\ndef predict(row): return 0.\n"})
    execute(agent, lab, 'query', {'question': ROOT,
        'actions': [action(0., 1.), action(2., 3.)]})
    first = execute(agent, lab, 'run', {'operator': 'refit', 'model': 'identity.py',
        'path': 'refits/identity', 'records': [0, 1], 'validation_records': [1]})
    execute(agent, lab, 'query', {'actions': [action(4., 5., 6.)]})
    second = execute(agent, lab, 'run', {'operator': 'refit', 'model': 'zero.py',
        'path': 'refits/zero', 'records': [0, 1, 2], 'validation_records': [0, 2],
        'notes': 'The identity model validated zero error on record 1; '
                 'do not mistake that result for this separate zero model.'})
    assert first['validation']['mse'] == 0.
    assert second['validation']['mse'] > 0.
    assert first['split']['validation_records'] == [1]
    assert second['split']['validation_records'] == [0, 2]
    facts = agent._request(lab)['evidence_notebook']['facts']
    for index, result in enumerate(agent.operator_results):
        if result['operator'] != 'refit':
            continue
        card = facts[f'refit:{index}']
        assert card['kind'] == 'paid_trajectory_validation'
        assert card['source_model'] == result['source_model']
        assert card['tested_model'] == result['tested_model']
        assert card['validation_mse'] == result['validation']['mse']
        for key in ('train_rows', 'holdout_rows', 'validation_records'):
            assert card[key] == result['split'][key]
        assert card['source']['json_path'] == ['operator_results', index]
    assert lab.paid == 7 and len(lab.actions) == 3 and not agent.confirmations
