"""The ordinary native author can reconstruct or reuse the same finite method."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.method_action_research import partition_followup_body
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.source_method_reconstruction import reconstruct_partition_source
from hegel_machine.cognition.transfer_partitions import propose_transfer_partitions
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_transfer_partitions import plan, observations


def source_fixture(path):
    before, after = plan(dormant=True), plan(.85, weight=3., dormant=True)
    for name, body in [('before', before), ('after', after)]:
        (path / (name + '.json')).write_text(json.dumps(body))
    raw = {'source_concept_id': 'source', 'source_ref': {'file': 'paid_revision.json'},
        'before_ref': {'file': 'before.json'}, 'after_ref': {'file': 'after.json'}}
    proposal = propose_transfer_partitions(before, after, source_ref=raw['source_ref'],
        before_ref=raw['before_ref'], after_ref=raw['after_ref'])
    return raw, before, proposal


def test_archive_and_fixed_execute_reconstruction_and_replay_same_content_as_C(tmp_path, monkeypatch):
    raw, before, proposal = source_fixture(tmp_path)
    frozen = {'source_concept_id': 'source', 'source_ref': raw['source_ref'],
        'research_program': proposal['candidates'][0]['research_program']}
    baseline = {**frozen, 'research_program': proposal['baseline_research_program']}
    rows = observations()
    names = tuple(rows[0].features)
    template = SimpleNamespace(source_features=names, roles=names)
    mapping = {name: name for name in names}
    monkeypatch.setattr(UnifiedResearchAgent, '_replay_program_queue',
        lambda *args, **kwargs: [('source', before, template, [mapping])])
    results, setup = {}, {}
    for arm in ('archive', 'fixed_pr2', 'pr3'):
        agent = CommonObjectResearchAgent(workspace=tmp_path, organization=arm,
            program_sources=[raw], learned_programs=[frozen, baseline] if arm == 'pr3' else [])
        budget = SearchBudget(100000)
        setup[arm] = agent._prepare_program_sources(budget)
        plans = agent._replay_program_queue(TaskSpec('test', '', names, (), ()), [])
        assert plans[0][1] == frozen['research_program']
        assert plans[-1][1] == before  # Old executable procedure is retained.
        claims = agent._replay_program_claims(SimpleNamespace(risk_quantile=.5), rows,
            budget, [], prepared=plans)
        results[arm] = [claim['candidate'].program for claim in claims]
        assert claims and budget.spent > setup[arm]
        assert agent._prepare_program_sources(budget) == 0
    assert setup['archive'] == setup['fixed_pr2'] > 0 and setup['pr3'] == 0
    assert results['archive'] == results['fixed_pr2'] == results['pr3']


def test_second_stage_is_reconstructed_only_from_actual_completed_fit_return(tmp_path):
    raw, _, proposal = source_fixture(tmp_path)
    (tmp_path / 'proposal.json').write_text(json.dumps({'proposal': proposal}))
    fitted = proposal['candidates'][0]['research_program']
    study = {'basis': {'minimum_improvement': 1e-7}, 'methods': [{
        'index': 1, 'execution_complete': True, 'training_loss': 0.,
        'validation_loss': .1, 'research_program': fitted}]}
    (tmp_path / 'study.json').write_text(json.dumps(study))
    raw.update(proposal_ref={'file': 'proposal.json'}, study_ref={'file': 'study.json'})
    result = reconstruct_partition_source(raw, tmp_path, SearchBudget(100000))
    assert result['methods'][0]['research_program'] == partition_followup_body(fitted)
    study['methods'][0]['execution_complete'] = False
    (tmp_path / 'study.json').write_text(json.dumps(study))
    result = reconstruct_partition_source(raw, tmp_path, SearchBudget(100000))
    assert len(result['methods']) == 2
    assert not result['methods'][0]['research_program']['transfer_partition'].get('second_stage')


def test_frozen_bank_retains_the_same_first_stage_methods_as_reconstruction(tmp_path):
    from scripts.run_partial_role_successors import frozen_program_bank
    raw, _, proposal = source_fixture(tmp_path)
    fitted = proposal['candidates'][0]['research_program']
    method = {'research_program': partition_followup_body(fitted)}
    (tmp_path / 'proposal.json').write_text(json.dumps({'proposal': proposal}))
    (tmp_path / 'study.json').write_text(json.dumps({
        'basis': {'minimum_improvement': 1e-7}, 'methods': [{
            'index': 1, 'execution_complete': True, 'training_loss': 0.,
            'validation_loss': .1, 'research_program': fitted}]}))
    raw.update(proposal_ref={'file': 'proposal.json'}, study_ref={'file': 'study.json'})
    reconstructed = reconstruct_partition_source(raw, tmp_path, SearchBudget(100000))
    frozen = frozen_program_bank(raw, method, proposal)
    assert [item['research_program'] for item in frozen] == [
        item['research_program'] for item in reconstructed['methods']]
    assert fitted in [item['research_program'] for item in frozen]


def test_native_failure_keeps_paid_targets_once_but_normal_run_does_not_copy(tmp_path, monkeypatch):
    from hegel_machine.cognition.models import ResearchConfig
    paid = {'probe': {'probe_id': 'paid', 'sequence': [{'u': .4}]}, 'channels': ['sensor'],
            'observations': [{'features': {'q': 2.}, 'target': 7.5, 'history': []}],
            'transitions': []}
    batches, actions = [], []
    def research(agent, laboratory, channels, acquire, supplied, budget, trace, **kwargs):
        assert agent._paid_native_state['batches'] is supplied
        assert agent._paid_native_state['actions'] is trace
        supplied.append(deepcopy(paid))
        trace.append({'action': 'acquire', 'observation_cost': 1})
        budget.spend(13)
        agent._object_models = {tuple(channels): {'complete': False, 'groups': [{'q': 'q'}]}}
        if laboratory.fail:
            raise KeyError('f')
        return {'complete': False}
    def run(agent, laboratory, **kwargs):
        budget = SearchBudget(100)
        agent._research_objects(laboratory, ('sensor',), None, batches, budget, actions)
        return {'status': 'unresolved', 'program': None, 'costs': {'compute_cost': budget.spent}}
    monkeypatch.setattr(UnifiedResearchAgent, '_research_objects', research)
    monkeypatch.setattr(UnifiedResearchAgent, 'run', run)
    for fail in (False, True):
        batches.clear()
        actions.clear()
        workspace = tmp_path / str(fail)
        agent = CommonObjectResearchAgent(ResearchConfig(compute_budget=100),
            workspace=workspace, organization='archive')
        if fail:
            with pytest.raises(KeyError, match='f'):
                agent.run(SimpleNamespace(fail=True))
            state = json.loads(Path(agent.failure_evidence_ref['file']).read_text())
            assert state['batches'] == [paid]
            assert state['batches'][0]['observations'][0]['target'] == 7.5
            assert state['confirmed_compute_lower_bound'] == 13
            assert state['compute_upper_bound'] == 100
            assert state['object_continuation_state']['models'][0]['model']['complete'] is False
            assert len(list(workspace.glob('*.json'))) == 1
        else:
            agent.run(SimpleNamespace(fail=False))
            assert not workspace.exists()
            assert agent.failure_evidence_ref is None
        assert agent._paid_native_state is agent._current_native_frame is None


def test_current_pool_scores_do_not_inherit_source_training_errors(tmp_path, monkeypatch):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Candidate, Observation
    rows = [Observation({'x': 1.}, 2.), Observation({'x': 2.}, 4.)]
    old = {'candidate': Candidate('retained', Program('feature', name='x'), 'retained'),
           'training_loss': 0., 'validation_loss': 0., 'kind': 'retained'}
    base = {**old, 'candidate': replace(old['candidate'], candidate_id='baseline',
        program=Program('const', value=0.)), 'training_loss': 10., 'validation_loss': 10.}
    frame = {'assessed': [base], 'uncompared': [], 'comparison': rows,
        'comparison_complete': True, 'terminal': False, 'fallback': 0., 'prepare_compute_cost': 0}
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision',
        lambda *args, **kwargs: deepcopy(frame))
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='pr3', retained_pool=[old])
    budget, actions = SearchBudget(100), []
    actual = agent.prepare_research_decision(TaskSpec('test', '', ('x',), (), ()), rows, rows, budget, actions)
    scored = actual['assessed'][-1]
    assert scored['training_loss'] == scored['validation_loss'] == 2.5
    assert budget.spent == 4 and not actual['terminal']
    assert old['training_loss'] == old['validation_loss'] == 0.
    assert actions[-1]['full_retained_pool_comparison']


def test_prepared_caller_executes_condition_consumer_once_on_actual_profile(tmp_path, monkeypatch):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Candidate, Observation
    import hegel_machine.cognition.method_conditions as module
    rows = [Observation({'x': 1.}, 2.)]
    old, new = {'old': True}, {'new': True}
    plans = [('source', body, SimpleNamespace(), [{}]) for body in (old, new)]
    class Policy:
        def set_action_knowledge(self, action, knowledge):
            self.last = action, knowledge
    frame = {'assessed': [{'candidate': Candidate('best', Program('const', value=0.), 'retained'),
                          'training_loss': 4., 'validation_loss': 4.}],
        'uncompared': [], 'comparison': rows, 'comparison_complete': True, 'terminal': False,
        'fallback': 0., 'prepare_compute_cost': 0, 'replay_plans': plans, 'policy': Policy()}
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision', lambda *a, **k: frame)
    called = []
    def select(records, profile, conditions):
        called.append(profile)
        return {'programs': [records[1]], 'decisions': [{'selected': 'challenger'}], 'compute_cost': 1}
    monkeypatch.setattr(module, 'select_native_method_programs', select)
    monkeypatch.setattr(CommonObjectResearchAgent, '_replay_knowledge', lambda *a: ['actual-new'])
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='pr3',
        native_method_conditions=[{'research': {'candidates': [{}]}}])
    budget, actions = SearchBudget(100), []
    result = agent.prepare_research_decision(TaskSpec('test', '', ('x',), (), ()), rows, rows, budget, actions)
    assert result['replay_plans'] == [plans[1]] and budget.spent == 1 and len(called) == 1
    assert called[0]['residual']['loss'] == 4.
    assert called[0]['research']['budget']['remaining_compute'] == 100
    assert actions[-1]['action'] == 'select_native_research_program'
    assert frame['policy'].last == ('program_replay', ['actual-new'])


def test_actual_parent_revision_creates_a_method_proposal_through_ordinary_return(tmp_path):
    from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract
    before, after = plan(dormant=True), plan(.85, weight=3., dormant=True)
    initial = tmp_path / 'actual_before.json'
    write_json_checkpoint(initial, {'research_program': before})
    agent = CommonObjectResearchAgent(ResearchConfig(compute_budget=10000), workspace=tmp_path,
        organization='pr3', source_task_ref={'source': 'real_return_control'})
    agent._native_revision_before_ref = {'file': str(initial)}
    report = {'program': research_program_fit_contract(after)['program'].to_dict(),
        'research_program': after, 'status': 'unresolved', 'costs': {'compute_cost': 250}}
    assert agent._record_actual_program_return(report)['status'] == 'actual_before_after_addressed'
    agent._consume_program_return(report, [])
    returned = report['native_method_return']
    assert returned['new_compute_cost'] > 0
    assert report['costs']['compute_cost'] == 250 + returned['new_compute_cost']
    proposals = list((tmp_path / 'response_method/method_feedback').glob('transfer_partition_*.json'))
    assert len(proposals) == 1
    bundle = json.loads(proposals[0].read_text())
    assert bundle['proposal']['candidates'][0]['research_program']['transfer_partition']['kind'] == 'Pi1'
    assert bundle['source_operation'] == 'ordinary_parent_research_return'
    assert bundle['independent_support'] is False


def test_unchanged_parent_is_recorded_without_manufacturing_method_change(tmp_path):
    from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract
    before = plan(dormant=True)
    initial = tmp_path / 'actual_before.json'
    write_json_checkpoint(initial, {'research_program': before})
    agent = CommonObjectResearchAgent(ResearchConfig(compute_budget=10000), workspace=tmp_path,
        organization='pr3', source_task_ref={'source': 'unchanged_return_control'})
    agent._native_revision_before_ref = {'file': str(initial)}
    report = {'program': research_program_fit_contract(before)['program'].to_dict(),
        'research_program': before, 'status': 'completed', 'costs': {'compute_cost': 250}}
    agent._record_actual_program_return(report)
    agent._consume_program_return(report, [])
    proposal, = (tmp_path / 'response_method/method_feedback').glob('transfer_partition_*.json')
    assert json.loads(proposal.read_text())['proposal']['candidates'] == []


def test_retained_candidate_and_fit_body_use_the_same_role_binding(tmp_path, monkeypatch):
    from hegel_machine.cognition.models import Candidate
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract
    from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
    body = plan(dormant=True)
    fit = research_program_fit_contract(body)
    fields = sorted(fit['program'].required_features())
    mapping = {name: 'target_' + name for name in fields}
    renamed = bind_measurement_arguments(fit['program'], mapping)
    rows = [replace(row, features={mapping.get(k, k): v for k, v in row.features.items()})
            for row in observations(count=12)]
    old = {'candidate': Candidate('retained', fit['program'], 'retained'),
        'research_program': body, 'training_loss': 0., 'validation_loss': 0.,
        'retained_candidate_ref': {'file': 'original.json'}}
    baseline = {**old, 'candidate': replace(old['candidate'], candidate_id='baseline', program=renamed),
                'training_loss': 1., 'validation_loss': 1.}
    frame = {'assessed': [], 'uncompared': [], 'comparison': rows,
        'comparison_complete': True, 'terminal': False, 'fallback': 0., 'prepare_compute_cost': 0}
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision', lambda *a, **k: frame)
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='pr3', retained_pool=[old])
    monkeypatch.setattr(agent, '_program_body_bindings', lambda *a: (
        SimpleNamespace(source_features=fields, roles=fields), [mapping]))
    result = agent.prepare_research_decision(TaskSpec('test', '', tuple(rows[0].features), (), ()),
        rows, rows, SearchBudget(10000), [])
    selected = result['assessed'][0]
    rebuilt = research_program_fit_contract(selected['research_program'])
    assert selected['candidate'].program == renamed == rebuilt['program']
    assert len(rebuilt['fit_terms']) == len(fit['fit_terms'])
    assert selected['canonical_research_entry_ref'] == old['retained_candidate_ref']
    assert old['research_program'] == body


def test_archive_reconstructs_conditions_from_actual_paired_records(tmp_path):
    from test_native_method_condition_selection import setup_case
    from hegel_machine.cognition.method_conditions import native_condition_bindings
    pairs, records, binding = setup_case()
    refs = []
    for index, pair in enumerate(pairs):
        saved = {key: value for key, value in pair.items() if key not in ('baseline', 'challenger')}
        for side in ('baseline', 'challenger'):
            value = deepcopy(pair[side])
            profile = tmp_path / f'{index}_{side}_profile.json'
            profile.write_text(json.dumps(value.pop('profile')))
            value.update(profile_ref={'file': str(profile)}, **value.pop('basis'))
            path = tmp_path / f'{index}_{side}.json'
            path.write_text(json.dumps(value))
            saved[side + '_ref'] = {'file': str(path)}
        path = tmp_path / f'pair_{index}.json'
        path.write_text(json.dumps(saved))
        refs.append({'file': str(path)})
    group = {**binding['research'], 'method_ref': {'id': 'method', 'version': 1}, 'pair_refs': refs}
    for side, record in zip(('baseline', 'challenger'), records):
        body = tmp_path / f'{side}_body.json'
        body.write_text(json.dumps(record['research_program']))
        artifact = tmp_path / f'{side}_method.json'
        artifact.write_text(json.dumps({'execution': {'entry': 'replay_research_program',
            'research_program_ref': {'file': str(body)}}}))
        group['baseline_method_artifact_ref' if side == 'baseline' else 'method_artifact_ref'] = {'file': str(artifact)}
    spent = []
    for arm in ('archive', 'fixed_pr2', 'pr3'):
        agent = CommonObjectResearchAgent(workspace=tmp_path, organization=arm, condition_sources=[group],
            native_method_conditions=native_condition_bindings([group], tmp_path) if arm == 'pr3' else [])
        budget = SearchBudget(100000)
        spent.append(agent._prepare_program_sources(budget))
        assert agent.native_method_conditions[0]['research']['general_guard_supported'] is True
        assert agent._prepare_program_sources(budget) == 0
    assert spent[0] == spent[1] > 0 and spent[2] == 0
