"""Cohort aggregation contracts on saved toy Common sources, not science."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.structural_inquiry import _register_native_theories
from scripts import run_anonymous_revision_inquiry as study
from test_registered_native_author import (
    DRIVER, LocalLab, knowledge, registration, runtime_root,
)
from test_inquiry_revision import action
from test_theory_consolidation import affine


def save_source(workspace, organization, *, identity, program, spent=17, mission_spent=3, bad=False):
    agent = study.RegisteredOrganizationAgent(None, workspace, arm='pr3',
        research_organization=organization, knowledge=knowledge(), source_task=False,
        config=InquiryLLMConfig(compute_budget=5000, max_model_calls=0, confirmation_rows=1))
    context = {'source_id': identity}
    question = agent._open_question({'id': 'local', 'question': 'Supplied toy source relation',
        'context_ref': context, 'theory_target': {
            'role_signature': {'u': {'role': 'drive', 'unit': '1', 'time': 'current'}},
            'output_unit': '1', 'context_ref': context, 'context_family_ref': 'toy_affine'}})
    question['native_condition_research'] = [{'round_index': 0,
        'status': 'search_failed' if bad else 'candidates_require_evaluation',
        'fit_evidence': [], 'fit_record_indices': [], 'exported_models': [], 'compute_cost': spent,
        'candidates': [{'program': program.to_dict(), 'status': 'refuted' if bad else 'candidate'}]}]
    receipt = _register_native_theories(agent, question, agent.questions.index(question), 0)
    local_id = receipt['registered'][0]['theory_id']
    agent._organization_state['consolidation_basis'] = {
        'source_versions': [{'id': local_id, 'version': agent.theory_index[local_id]['version']}]}
    agent._organization_state['theory_mission']['resource_usage'] = {'compute': mission_spent}
    agent.costs['compute_cost'] = spent
    if bad:
        agent.events.append({'action': 'finish', 'arguments': {'model': 'old.py'},
            'feedback': {'research_question_id': 'local', 'question_meaning_version': 1,
                'error': 'Retained synthetic failure',
                'fresh_confirmation': {'passed': False, 'mse': 9.}},
            'usage': {'backend': 'saved_fixture', 'total_tokens': 0}})
    agent._prepare_program_state()
    result = agent.result('method_error' if bad else 'method_stopped')
    study.write_json(agent.workspace / 'training.json', result)
    study.write_json(agent.workspace / 'archive.json', result['outgoing_knowledge'])
    return {'case': {'id': identity}, 'arm': 'pr3', 'result': result}


def cohort(tmp_path, *, count=3, compute=5000, funding=1000):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frozen = tmp_path / 'initial.json'
    frozen.write_text('{"candidates":{}}')
    organization = registration() | {'initial_library_path': str(frozen)}
    organization['consolidation']['mission']['theory_learning_allocation']['compute'] = funding
    configured = study.protocol(kinds=('toy',), cases=count, arms=study.ORGANIZATION_ARMS,
        research_organization=organization,
        config=InquiryLLMConfig(compute_budget=compute, max_model_calls=0, token_budget=0))
    programs = [affine(2., 1.), affine(4., -3.), Program('const', value=0.)]
    trials = [save_source(tmp_path / f'source_{i}', organization,
        identity=f'physical_{i}', program=programs[i], spent=17 + i * 6,
        mission_spent=3 + i, bad=i == 2) for i in range(count)]
    return configured, organization, trials


def aggregate(output, configured, organization, trials, previous=None):
    output.mkdir(parents=True, exist_ok=True)
    return study._aggregate_registered_sources(configured, output, trials, organization,
        knowledge(), {}, previous_study=previous)


@pytest.mark.parametrize('count, cap, funding, requested, reserve, stages', [
    (2, 250000, 25000, 25000, 25000, [237500, 237500]),
    (3, 11, 2, 5, 5, [9, 9, 10]),
    (2, 10, 1, 100, 2, [9, 9]),
])
def test_new_source_contract_reserves_only_original_funded_compute(
        count, cap, funding, requested, reserve, stages):
    organization = registration()
    work = organization['consolidation']
    work['allocation']['compute'] = requested
    work['mission']['theory_learning_allocation']['compute'] = funding
    configured = study.protocol(kinds=('toy',), cases=count, arms=study.ORGANIZATION_ARMS,
        research_organization=organization,
        config=InquiryLLMConfig(compute_budget=cap, max_model_calls=0))
    contract = configured['source_cohort_allocation']
    assert contract['aggregation_compute'] == reserve
    assert list(contract['source_stage_compute'].values()) == stages
    assert sum(stages) + reserve == contract['source_compute_limit'] == cap * count
    assert contract['local_consolidation_compute'] == 0
    assert configured['research_organization'] == organization
    assert configured['limits']['compute_budget'] == cap


def test_single_source_and_target_keep_the_original_per_case_allocation():
    organization = registration()
    single = study.protocol(kinds=('toy',), arms=study.ORGANIZATION_ARMS,
                            research_organization=organization)
    target = study.protocol(kinds=('toy',), cases=2, arms=study.ORGANIZATION_ARMS,
                           research_organization=organization | {'phase': 'target'})
    assert 'source_cohort_allocation' not in single
    assert 'source_cohort_allocation' not in target
    assert single['research_organization']['consolidation']['allocation']['compute'] == 1000


def test_qualified_sources_freeze_two_source_family_and_keep_bad_outcome_and_costs(tmp_path):
    configured, organization, trials = cohort(tmp_path / 'inputs')
    ignored = deepcopy(trials[0])
    ignored['arm'] = 'archive'
    ignored['result']['costs']['compute_cost'] = 99999
    receipt = aggregate(tmp_path / 'aggregate', configured, organization, [*trials, ignored])
    assert receipt['source_compute_limit'] == 3 * configured['limits']['compute_budget']
    assert receipt['source_costs']['compute_cost'] == receipt['source_compute_before'] == 17 + 23 + 29
    assert receipt['mission_compute_before'] == 3 + 4 + 5
    assert 0 < receipt['new_compute_cost'] <= organization['consolidation']['allocation']['compute']
    assert receipt['source_compute_after'] == receipt['source_compute_before'] + receipt['new_compute_cost']
    assert receipt['mission_compute_after'] == receipt['mission_compute_before'] + receipt['new_compute_cost']
    library = json.loads(Path(receipt['library_path']).read_text())
    frozen = [c for c in library['candidates'].values() if c['independent_source_count'] == 2]
    assert frozen
    assert all(item['exact'] for c in frozen for item in c['source_reconstruction'])
    raw_index = json.loads(Path(receipt['index_path']).read_text())
    indexed = raw_index.get('theory_index', raw_index)
    assert len(indexed) == 3
    assert len({identifier.split('::')[-1] for identifier in indexed}) == 1
    assert all('::' in identifier for identifier in indexed)
    bad_state = str(Path(trials[2]['result']['workspace']) / 'research_state.json')
    bad_training = str(Path(trials[2]['result']['workspace']) / 'training.json')
    bad_episodes = [e for e in library['episodes'] if e['source_id'] == 'physical_2']
    assert bad_episodes and any(ref.get('file') in {bad_state, bad_training}
                              for e in bad_episodes for ref in e['failure_refs'])
    failed = [result for e in bad_episodes for result in e['independent_results']
              if result.get('passed') is False]
    assert failed and failed[0]['source_ref']['file'] == bad_training
    assert receipt['new_observations'] == receipt['new_model_calls'] == 0


@pytest.mark.parametrize('exhausted', ['source', 'mission'])
def test_exhausted_original_combined_balance_cannot_receive_new_allowance(tmp_path, exhausted):
    configured, organization, trials = cohort(tmp_path / 'inputs', count=2)
    for trial in trials:
        result = trial['result']
        if exhausted == 'source':
            result['costs']['compute_cost'] = configured['limits']['compute_budget']
        else:
            result['research_organization']['theory_mission']['resource_usage']['compute'] = 1000
        study.write_json(Path(result['workspace']) / 'training.json', result)
    receipt = aggregate(tmp_path / 'exhausted', configured, organization, trials)
    assert receipt['new_compute_cost'] == 0
    assert receipt['cumulative_aggregation_compute'] == 0
    assert receipt['source_compute_after'] == receipt['source_compute_before']
    assert not receipt['feedback']['frozen_candidates']
    assert receipt['new_observations'] == receipt['new_model_calls'] == 0


def test_same_source_resume_reuses_and_changed_source_retains_old_aggregate_debit(tmp_path):
    configured, organization, trials = cohort(tmp_path / 'inputs', count=2)
    first_root = tmp_path / 'first'
    first = aggregate(first_root, configured, organization, trials)
    second_root = tmp_path / 'same'
    second = aggregate(second_root, configured, organization, trials, previous=first_root)
    assert second['reused'] and second['new_compute_cost'] == 0
    assert second['library_path'] == first['library_path']
    assert second['cumulative_aggregation_compute'] == first['new_compute_cost']
    # Same address and total cost, changed source evidence/version: the original
    # stat-based source contract must invalidate the outer cohort early return.
    file = Path(trials[1]['result']['workspace']) / 'research_state.json'
    state = json.loads(file.read_text())
    identifier, = state['theory_index']
    state['theory_index'][identifier]['version'] = 2
    state['questions'][1]['native_condition_research'][0]['candidates'][0]['program'] = affine(5., -4.).to_dict()
    study.write_json(file, state)
    third = aggregate(tmp_path / 'changed', configured, organization, trials, previous=second_root)
    assert third['reused'] is False
    assert third['source_compute_before'] == sum(t['result']['costs']['compute_cost'] for t in trials) + first['new_compute_cost']
    assert third['mission_compute_before'] == 3 + 4 + first['cumulative_aggregation_mission_compute']
    assert third['cumulative_aggregation_compute'] == first['new_compute_cost'] + third['new_compute_cost']
    assert third['source_compute_after'] <= third['source_compute_limit']
    assert third['mission_compute_after'] <= third['mission_compute_limit']
    assert third['source_versions'] != first['source_versions']


def test_run_study_aggregates_all_c_sources_before_any_terminal_rows(tmp_path, runtime_root, monkeypatch):
    order, agents = [], {}
    driver = "def select_actions(spec, observations, memory): return {'model':'new.py'}\n"
    organization = registration()
    organization['consolidation']['source_question_refs'] = [{'question_id': 'local', 'meaning_version': 1}]
    configured = study.protocol(kinds=('toy',), cases=2, arms=study.ORGANIZATION_ARMS,
        research_organization=organization,
        author={'kind': 'finite_native', 'path': DRIVER, 'max_steps': 1, 'confirm': True},
        config=InquiryLLMConfig(observation_budget=10, confirmation_rows=1,
            compute_budget=5000, execution_seconds=30., max_model_calls=0, token_budget=0))
    configured['evaluation']['rows'] = 2
    actual = study._aggregate_registered_sources

    def observed(*args, **kwargs):
        trials = args[2]
        assert len(trials) == 6
        assert all((Path(t['result']['workspace']) / 'training.json').is_file() for t in trials)
        order.append('aggregate')
        return actual(*args, **kwargs)

    monkeypatch.setattr(study, '_aggregate_registered_sources', observed)

    class OrderedLab(LocalLab):
        def evaluation_rows(self, length, seed):
            assert order == ['aggregate']
            return super().evaluation_rows(length, seed)

    def prepare(agent, lab, case):
        agents[case['id'], agent.organization_group] = agent
        agent._execute('query', {'actions': [action(1.)]}, lab)
        context = {'source_id': case['id']}
        question = agent._open_question({'id': 'local', 'question': 'Supplied toy source relation',
            'context_ref': context, 'theory_target': {
                'role_signature': {'u': {'role': 'drive', 'unit': '1', 'time': 'current'}},
                'output_unit': '1', 'context_ref': context, 'context_family_ref': 'toy_affine'}})
        program = affine(2. if case['id'] == 'case_000' else 4., 1.)
        question['native_condition_research'] = [{'round_index': 0, 'status': 'candidates_require_evaluation',
            'compute_cost': 0, 'fit_evidence': [], 'exported_models': [], 'candidates': [{'program': program.to_dict()}]}]
        _register_native_theories(agent, question, agent.questions.index(question), 0)
        return {}

    result = study.run_study(configured, tmp_path / 'formal', knowledge=knowledge(driver),
        prepare_prefix=prepare, laboratory_factory=lambda _: OrderedLab())
    receipt = result['source_aggregation']
    assert order == ['aggregate']
    assert len(receipt['source_refs']) == 2
    assert all('/pr3/training.json' in ref['file'] for ref in receipt['source_refs'])
    assert receipt['source_compute_limit'] == 2 * configured['limits']['compute_budget']
    assert receipt['source_costs']['compute_cost'] == sum(t['costs']['compute_cost']
        for t in result['trials'] if t['arm'] == 'pr3')
    assert all(t['terminal']['passed'] for t in result['trials'])
    # The final cross-source work owns part of the unchanged total allowance;
    # a complete local search must not be able to spend its reserved compute.
    for (case_id, group), agent in agents.items():
        assert agent.config.compute_budget == (4500 if group == 'pr3' else 5000)
        if group == 'pr3':
            local = agent._organization_registration['consolidation']['allocation']
            assert local['compute'] == 0
            selected = agent._organization_state['consolidation_basis']['source_versions']
            assert len(selected) == 1
            assert selected[0]['id'] in agent.theory_index
            assert agent._organization_state['theory_mission']['resource_usage']['compute'] == 0
            steps = [e for e in agent.events if e.get('origin') == study.ORGANIZATION_ORIGIN
                     and e['arguments']['operator'] == 'theory_consolidation']
            assert steps and all(e['arguments']['allocation']['compute'] == 0
                                 and e['feedback']['new_compute_cost'] == 0 for e in steps)
    assert configured['research_organization']['consolidation']['allocation']['compute'] == 1000
    assert 0 < receipt['new_compute_cost'] <= 1000
    assert receipt['source_compute_after'] <= 10000
    assert receipt['mission_compute_before'] == 0
    assert receipt['mission_compute_after'] == receipt['new_compute_cost'] <= 2000
    library = json.loads(Path(receipt['library_path']).read_text())
    families = [c for c in library['candidates'].values() if c['independent_source_count'] == 2]
    assert families and all(item['exact'] for c in families for item in c['source_reconstruction'])
    assert result['source_cohort_allocation'] == configured['source_cohort_allocation']

    # Completed source training and aggregation remain spent on actual runner
    # recovery; the reserve is a partition, never a second allowance.
    resumed = study.run_study(configured, tmp_path / 'resumed', previous_study=tmp_path / 'formal',
        prepare_prefix=lambda *args: pytest.fail('Recovery must not buy a new prefix'),
        provider_factory=lambda *args: pytest.fail('Recovery must not construct a provider'),
        laboratory_factory=lambda _: LocalLab())
    retained = resumed['source_aggregation']
    assert retained['reused'] and retained['new_compute_cost'] == 0
    assert retained['cumulative_aggregation_compute'] == receipt['new_compute_cost']
    assert retained['source_compute_after'] == receipt['source_compute_after']
    assert retained['mission_compute_after'] == receipt['mission_compute_after']
    assert [t['costs'] for t in resumed['trials']] == [t['costs'] for t in result['trials']]

    changed = deepcopy(configured)
    changed['source_cohort_allocation']['aggregation_compute'] += 1
    with pytest.raises(ValueError, match='preserve registered source_cohort_allocation'):
        study.run_study(changed, tmp_path / 'changed', previous_study=tmp_path / 'formal')
    legacy = tmp_path / 'legacy_registration'
    legacy.mkdir()
    original = deepcopy(configured)
    original.pop('source_cohort_allocation')
    study.write_json(legacy / 'protocol.json', original)
    with pytest.raises(ValueError, match='preserve registered source_cohort_allocation'):
        study.run_study(configured, tmp_path / 'retroactive_reserve', previous_study=legacy)


def test_source_cohort_without_new_native_objects_exports_a_readable_empty_index(tmp_path, runtime_root):
    driver = "def select_actions(spec, observations, memory): return {'model':'new.py'}\n"
    configured = study.protocol(kinds=('toy',), cases=2, arms=study.ORGANIZATION_ARMS,
        research_organization=registration(),
        author={'kind': 'finite_native', 'path': DRIVER, 'max_steps': 1, 'confirm': True},
        config=InquiryLLMConfig(observation_budget=10, confirmation_rows=1,
            compute_budget=5000, execution_seconds=30., max_model_calls=0, token_budget=0))
    configured['evaluation']['rows'] = 2

    def prefix(agent, lab, case):
        agent._execute('query', {'actions': [action(1.)]}, lab)
        return {}

    result = study.run_study(configured, tmp_path / 'no_new_object',
        knowledge=knowledge(driver), prepare_prefix=prefix, laboratory_factory=lambda _: LocalLab())
    receipt = result['source_aggregation']
    assert not receipt['feedback']['frozen_candidates']
    assert json.loads(Path(receipt['index_path']).read_text()) == {}
    assert receipt['new_compute_cost'] == 0
    assert all(t['terminal']['passed'] for t in result['trials'])
