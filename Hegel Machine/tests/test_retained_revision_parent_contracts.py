"""Retained constructor typing and fixed-parent identity; no world evaluation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition import question_routing, revision_relations as relations
from hegel_machine.cognition import theory_consolidation as consolidation


def node(op, *children, **kwargs):
    return Program(op, children=children, **kwargs)


@pytest.mark.parametrize('changed_contract', [None, 'comparison_rows', 'parent_version', 'fitting'])
def test_transfer_continues_pending_fits_when_only_compute_changes(
        retained_parent, monkeypatch, changed_contract):
    agent, parent, _, candidate, _ = retained_parent
    seen = []
    pending = {'layer': 2, 'offset': 1, 'pending_fits': [
        {'indices': [0, 0], 'position': {'layer': 0, 'offset': 0}}]}

    def propose(candidate, supplied, components, allocation, **kwargs):
        seen.append(deepcopy(kwargs.get('cursor')))
        return {'status': 'UNKNOWN', 'proposals': [], 'excluded': [],
                'actual_compute_cost': 1, 'budget_exhausted': False,
                'search_completed': False, 'cursor': deepcopy(pending)}

    monkeypatch.setattr(relations, 'propose_revision', propose)
    arguments = {'parent_theory_id': parent['id'], 'theory_id': 'pending_resume',
        'allocation': {'compute': 1000, 'comparison_rows': 97},
        'fitting': {'max_history': 4}}
    consolidation._transfer_relation(agent, arguments, candidate, consolidation.LIBRARY_PATH)
    original_basis = deepcopy(agent.theory_consolidation['executions'][
        'relation_transfer:pending_resume']['request_basis'])
    arguments['allocation']['compute'] = 2000
    if changed_contract == 'comparison_rows':
        arguments['allocation']['comparison_rows'] = 0
    elif changed_contract == 'parent_version':
        parent['version'] = 2
    elif changed_contract == 'fitting':
        arguments['fitting']['max_history'] = 3
    consolidation._transfer_relation(agent, arguments, candidate, consolidation.LIBRARY_PATH)
    assert seen == [None, pending if changed_contract is None else None]
    assert original_basis['allocation']['compute'] == 1000


@pytest.fixture
def retained_parent(tmp_path, monkeypatch):
    u = node('feature', name='u')
    one, zero = node('const', value=1.), node('const', value=0.)
    predicate = node('ge', u, one)
    coefficient = node('if', predicate, one, zero)
    signal = node('if', predicate, u, zero)
    program = node('mul', coefficient, signal)
    holes = [{'name': 'coefficient', 'output_unit': 'dimensionless', 'time_contract': ['current']},
             {'name': 'signal', 'output_unit': '1', 'time_contract': ['current']}]
    pattern = node('mul', node('arg', name='coefficient'), node('arg', name='signal'))
    constructor = {'id': 'old0', 'version': 1, 'kind': 'program_revision_relation',
        'definition': Definition('old0', ('coefficient', 'signal'), pattern).to_dict(),
        'before_pattern': pattern.to_dict(), 'holes': holes, 'output_unit': '1',
        'fixed_role_signature': {}, 'learned_constraints': [],
        'constraint_hypotheses': [{'id': 'unchanged', 'assignments': [], 'constraints': []}],
        'trigger_checks': [{'before': {'failed': True}}]}
    identifier = 'frozen_recall:q@1:old0:binding:1:5:fit:1'
    binding = {'library_path': 'shared/theory_library.json', 'candidate_id': 'old0',
        'candidate_version': 1, 'components': {
            name: {'component_id': name, 'version': 1} for name in ('coefficient', 'signal')}}
    parent = {'id': identifier, 'version': 1, 'program': program.to_dict(),
        'consolidation_ref': deepcopy(binding), 'role_signature': {
            'u': {'role': 'drive', 'unit': '1', 'time': 'current'}},
        'output_unit': '1', 'context_ref': {'scene': 'retained_fixture'}}
    program_ref = {'file': 'research_state.json', 'json_path': ['theory_index', identifier, 'program'],
                   'version': 1}
    transfer_ref = {'file': 'shared/theory_library.json', 'json_path': ['transfers', 0]}
    ledger = {'theory_ref': {'id': identifier, 'version': 1}, 'program_source_ref': program_ref,
              'transfer_ref': transfer_ref, 'model': 'retained_fixture.py'}
    question = {'id': 'q', 'meaning_version': 1, 'revision_candidate_exports': [ledger],
                'theory_target': {key: deepcopy(parent[key]) for key in
                                  ('role_signature', 'output_unit', 'context_ref')}}
    library = {'candidates': {'old0': constructor}, 'transfers': [
        {'theory_ref': deepcopy(ledger['theory_ref']), 'binding': deepcopy(binding)}]}
    shared = tmp_path / 'shared'
    shared.mkdir()
    (shared / 'theory_library.json').write_text(json.dumps(library))
    (tmp_path / 'research_state.json').write_text(json.dumps({'theory_index': {identifier: parent}}))
    agent = SimpleNamespace(workspace=tmp_path, files={}, questions=[question], theory_index={identifier: parent},
        theory_consolidation=deepcopy(library), costs={'compute_cost': 0},
        config=SimpleNamespace(compute_budget=10000), research_mission={'purpose': 'task'})
    agent._save = lambda name, text: (agent.workspace / name).write_text(text)
    monkeypatch.setattr(question_routing, 'resolved_theories',
                        lambda agent: [deepcopy(value) for value in agent.theory_index.values()])
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('metadata typing must not predict'))
    return agent, parent, program, constructor, ledger


@pytest.mark.parametrize('thin_restore', [False, True])
@pytest.mark.parametrize('extra_evidence', [False, True])
def test_transfer_recovers_retained_parent_from_its_own_exact_constructor(retained_parent, monkeypatch,
                                                                         thin_restore, extra_evidence):
    agent, parent, program, constructor, ledger = retained_parent
    original_workspace = agent.workspace
    if extra_evidence:
        failure = deepcopy(agent.theory_consolidation['transfers'][0])
        failure.update(status='refuted', ordinary_comparison_failure={'mse': 1.})
        agent.theory_consolidation['transfers'].append(failure)
        consolidation._save(agent)
    if thin_restore:
        agent.workspace = original_workspace / 'restored'
        (agent.workspace / 'shared').mkdir(parents=True)
        agent.theory_consolidation = consolidation.install_theory_library(agent, {
            'file': str(original_workspace / consolidation.LIBRARY_PATH), 'json_path': []})
        consolidation._save(agent)
        assert set(json.loads((agent.workspace / consolidation.LIBRARY_PATH).read_text())) == {'source_ref'}
    raw = relations._prepare(parent, program, SearchBudget(1000))
    assert raw['contracts'][(0,)]['output_unit'] is None
    assert raw['contracts'][(1,)]['output_unit'] is None
    captured = {}

    def propose(candidate, supplied, components, allocation, **kwargs):
        budget = SearchBudget(allocation['compute'])
        prepared = relations._component(supplied, budget)['_prepared']
        assert prepared['contracts'][(0,)]['output_unit'] == 'dimensionless'
        assert prepared['contracts'][(1,)]['output_unit'] == '1'
        assert supplied.get('source_ref') is None
        expected_ref = consolidation._absolute_reference(ledger['program_source_ref'], original_workspace)
        assert supplied['program_source_ref'] == expected_ref
        assert prepared['contract_recovery']['sources'][0]['transfer_ref']['file'] == str(
            original_workspace / consolidation.LIBRARY_PATH)
        assert prepared['contract_recovery']['sources'][0]['transfer_ref']['json_path'] == ['transfers', 0]
        captured.update(prepared=prepared, spent=budget.spent)
        return {'status': 'completed', 'proposals': [], 'excluded': [],
                'actual_compute_cost': budget.spent, 'budget_exhausted': False}

    monkeypatch.setattr(relations, 'propose_revision', propose)
    result = consolidation._transfer_relation(agent, {
        'parent_theory_id': parent['id'], 'theory_id': 'new_revision',
        'allocation': {'compute': 1000}}, constructor, consolidation.LIBRARY_PATH)
    assert result['new_compute_cost'] == captured['spent'] > raw['nodes']
    assert captured['prepared']['contract_recovery']['occurrence_count'] == 2


def test_fixed_subtrees_rebase_occurrence_paths_and_drop_whole_parent_pattern(retained_parent, monkeypatch):
    agent, parent, program, candidate, ledger = retained_parent
    parent = deepcopy(parent)
    parent['program_source_ref'] = deepcopy(ledger['program_source_ref'])
    parent['_contract_source'] = {'_pattern': candidate['definition']['body'], '_holes': candidate['holes'],
        '_bound_holes': parent['consolidation_ref']['components'],
        'source_ref': {'program_ref': deepcopy(parent['program_source_ref']),
                       'pattern_ref': {'file': 'shared/theory_library.json',
                                       'json_path': ['candidates', 'old0', 'definition', 'body']}}}

    def discrepancy(program, example, budget):
        assert budget.spend(1)
        return {'status': 'computed', 'failed': True}

    monkeypatch.setattr(relations, '_discrepancy', discrepancy)
    observed = []
    original = relations._component

    def component(record, budget):
        result = original(record, budget)
        if 'program_path' in record:
            observed.append(record)
            assert '_contract_source' not in record
            assert record['subtree_contracts']['']['output_unit'] == record['output_unit']
            assert all(not key.startswith('1.1.') for key in record['subtree_contracts'])
        return result

    monkeypatch.setattr(relations, '_component', component)
    result = relations.propose_revision(candidate, parent, [], allocation={'compute': 3000},
        evidence_checks=[{'source_ref': {'file': 'retained_fixture.json', 'json_path': [0]},
                          'context_ref': parent['context_ref'],
                          'observation': {'features': {'u': 2.}, 'target': 3., 'history': []}}])
    assert result['proposals'], result
    assert len(observed) == 2
    assert {tuple(item['program_path']) for item in observed} == {(0,), (1,)}
    assert all(item.get('source_ref') == parent.get('source_ref') for item in observed)
    assert all(item.get('subtree_contract_sources') for item in observed)
    assert Program.from_dict(result['proposals'][0]['program']).expanded() == program


def test_retained_export_version_mismatch_stays_unknown(retained_parent):
    agent, parent, _, constructor, ledger = retained_parent
    ledger['theory_ref']['version'] = 2
    result = consolidation._transfer_relation(agent, {
        'parent_theory_id': parent['id'], 'theory_id': 'rejected_revision',
        'allocation': {'compute': 1000}}, constructor, consolidation.LIBRARY_PATH)
    assert result['reason'] == 'retained_parent_export_version_mismatch'
    assert result['generated_theories'] == []
    assert result['status'] == 'UNKNOWN'


def test_recall_freezes_parent_version_across_same_id_generation_and_next_basis(retained_parent, monkeypatch):
    agent, parent, _, constructor, _ = retained_parent
    # Different consumed fitting contracts keep two real work items; exact
    # constructor aliases now correctly share one execution opportunity.
    agent.theory_consolidation['candidates']['later'] = {
        **deepcopy(constructor), 'id': 'later', 'purpose': 'parameter_proposal'}
    question = agent.questions[0]
    seen = []

    def apply(agent, arguments):
        seen.append(arguments['_parent_snapshot']['version'])
        agent.theory_index[parent['id']]['version'] = 2
        return {'status': 'UNKNOWN', 'generated_theories': [], 'new_compute_cost': 0}

    monkeypatch.setattr(consolidation, 'transfer_theory', apply)
    monkeypatch.setattr(consolidation, 'read_theory_library', lambda agent, path, **kwargs: agent.theory_consolidation)
    arguments = {'parent_theory_id': parent['id'], 'allocation': {'compute': 1000}}
    first = consolidation.recall_frozen_theories(agent, question, arguments)
    assert seen == [1, 1]
    assert question['frozen_recall_execution']['basis']['parent']['version'] == 1
    second = consolidation.recall_frozen_theories(agent, question, arguments)
    assert seen == [1, 1, 2, 2]
    assert not first['cache_hit'] and not second['cache_hit']
    assert question['frozen_recall_execution']['basis']['parent']['version'] == 2
