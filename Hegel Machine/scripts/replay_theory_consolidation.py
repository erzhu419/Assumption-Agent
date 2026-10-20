#!/usr/bin/env python3
"""Replay one typed Program theme on retained Episodes, without acquisition.

The archived sequence is read into memory.  Only a small receipt is exported;
the original Programs, research actions and measured counterexamples remain at
their retained source.  This is a historical adoption/transfer replay, not a
blind scientific experiment or evidence of autonomous theme discovery.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.effective_history import HistoryStore
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_queries import evaluate_on_model
from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition.theory_consolidation import INDEX_PATH, LIBRARY_PATH


ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / 'artifacts/cognition_v3_sequence_20260908.json.ARCHIVE_POINTER.json'
OUTPUT = ROOT / 'docs/evidence/theory_consolidation_replay_20261020.json'
MEMBER = 'cognition_v3_sequence_20260908.json'
UNIT = 'original_public_numeric_coordinate:response'
FAMILY = 'cognitive_task_sequence:ordered_affine_numeric_coordinate'


class NoAcquisitionLab:
    def __init__(self):
        self.execute_calls = 0

    def execute(self, action):
        self.execute_calls += 1
        raise AssertionError('Historical consolidation must never purchase observations')

    def public_spec(self):
        return {'controls': [{'name': name} for name in ('x', 'z', 'u', 'v')]}


def load_source(pointer):
    reference = json.loads(pointer.read_text())
    store = HistoryStore(reference['archive'], readonly=True)
    try:
        entries = store.connection.execute(
            'SELECT metadata,sequence FROM entries WHERE unit=? ORDER BY ordinal',
            (reference['archive_unit'],))
        for raw, sequence in entries:
            metadata = json.loads(raw)
            if metadata['name'] == MEMBER:
                payload = b''.join(store.read_sequence(sequence))
                return json.loads(payload), {
                    'pointer': str(pointer), 'archive': reference['archive'],
                    'unit': reference['archive_unit'], 'member': MEMBER,
                    'logical_member_bytes': len(payload), 'materialized_source_bytes': 0}
    finally:
        store.close()
    raise KeyError(f'{MEMBER} is missing from its retained archive unit')


def source_ref(source, field):
    return {key: deepcopy(source[key]) for key in ('archive', 'unit', 'member')} | {'field': field}


def context(task_id):
    return {'episode': f'cognition_v3_sequence_20260908/{task_id}',
            'measurement': 'response', 'coordinate_contract': 'original_public_numeric_coordinates',
            'output_unit': UNIT}


def roles(name):
    return {name: {'role': 'public_numeric_input',
                   'unit': 'original_public_numeric_coordinate:input', 'time': 'current'}}


def declaration(source, episode, position):
    task_id = episode['task_id']
    return {'id': f'history:{task_id}', 'program': deepcopy(episode['program']),
            'role_signature': roles('x'), 'output_unit': UNIT,
            'context_ref': context(task_id), 'context_family_ref': FAMILY,
            'parameter_paths': [[0], [1, 0]],
            'episode_ref': {'question_id': task_id, 'meaning_version': 1},
            'declared_scope_ref': source_ref(source, f'episodes[{position}].evaluation.task_metadata'),
            'measured_coverage_ref': source_ref(source, f'episodes[{position}].actions'),
            'support': {'status': 'historical_execution', 'context_ref': context(task_id),
                        'evidence': [source_ref(source, f'episodes[{position}].validation_loss')]},
            'program_source_ref': source_ref(source, f'episodes[{position}].program')}


def measured_checks(source, episode, position):
    return [{'source_ref': source_ref(source, f'episodes[{position}].evaluation.prediction_examples[{i}]'),
             'observation': {'features': deepcopy(row['features']), 'target': row['target'], 'history': []},
             'context_ref': context(episode['task_id']), 'tolerance': 1e-6}
            for i, row in enumerate(episode['evaluation']['prediction_examples'])]


def mse(program, rows):
    return sum((program.predict(Observation(row['features'], row['target'],
               tuple(row.get('history', [])))) - row['target']) ** 2 for row in rows) / len(rows)


def model_readout(program, row, task_id):
    # No historical acquisition is invented for an evaluation example. This is
    # explicitly a model query at its saved public coordinates, without a target.
    query = {'op': 'Readout', 'record_index': 0, 'step': 0, 'target': 'response',
             'context_ref': context(task_id), 'units': UNIT,
             'binding': {'action': {'sequence': [deepcopy(row['features'])]},
                         'features': deepcopy(row['features']), 'history': []}}
    return evaluate_on_model(query, lambda public: program.predict(
        Observation(public['features'], 0., tuple(public['history']))),
        context=context(task_id))


def episode_facts(source, episode, position):
    revision_index, revision = next((i, action['details'])
        for i, action in enumerate(episode['actions']) if action['action'] == 'revise')
    return {'id': f'episode:{episode["task_id"]}',
            'source_ref': source_ref(source, f'episodes[{position}]'),
            'question_ref': source_ref(source, f'episodes[{position}].questions'),
            'initial_program': deepcopy(revision['before_program']),
            'program_change_ref': source_ref(source, f'episodes[{position}].actions[{revision_index}]'),
            'actual_adoption': revision['operation'],
            'result': {'status': episode['status'], 'validation_loss': episode['validation_loss'],
                       'evaluation_loss': episode['evaluation']['loss']},
            'costs': {key: episode[key] for key in ('observation_cost', 'compute_cost')},
            'unresolved': 'Historical validation does not establish universal applicability'}


def extracted_episode_facts(episode):
    # Every address below is relative to this one retained Episode reference.
    # Repeating its archive/member path for every old action adds no evidence.
    def action_index(reference):
        path = reference['json_path']
        return path[path.index('actions') + 1]

    return {'episode_id': episode['episode_id'], 'source_id': episode['source_id'],
            'source_ref': episode['source_ref'], 'status': episode['status'],
            'adoption': {'occurred': episode['adoption']['occurred'],
                'execution_action_indices': [action_index(ref)
                    for ref in episode['adoption']['execution_refs']]},
            'program_change_action_indices': [action_index(change['source_ref'])
                for change in episode['actual_changes']],
            'behavior_action_indices': [action_index(change['source_ref'])
                for change in episode['behavior_changes']],
            'failure_action_indices': [action_index(ref) for ref in episode['failure_refs']],
            'cost_fields': ['observation_cost', 'compute_cost'],
            'retained_final_evaluation': {'relative_json_path': ['evaluation'],
                'independent_confirmation': False}}


def replay_frozen_installation(pointer, frozen_directory):
    """Check the independent-workspace input path on the retained source objects."""
    payload, origin = load_source(Path(pointer))
    positions = (0, 5)
    episodes = [payload['episodes'][i] for i in positions]
    assert [episode['task_id'] for episode in episodes] == ['t01', 't06']
    counter_index, counter = next((i, value) for i, value in enumerate(
        payload['knowledge']['concepts'][0]['counterexamples']) if value['task_id'] == 't13')
    check = {'source_ref': source_ref(origin, f'knowledge.concepts[0].counterexamples[{counter_index}]'),
        'observation': deepcopy(counter['observation']), 'context_ref': context('t13'), 'tolerance': 1e-6}
    config = InquiryLLMConfig(compute_budget=50000, observation_budget=0,
                             max_model_calls=0, execution_seconds=30.)
    directory = Path(frozen_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lab = NoAcquisitionLab()
    with tempfile.TemporaryDirectory(prefix='hegel-frozen-installation-') as temporary:
        source = CommonArchiveInquiryAgent(None, Path(temporary) / 'source', arm='hegel',
                                           source_task=False, config=config)
        declarations = [declaration(origin, episode, i) for i, episode in zip(positions, episodes)]
        compiled = source._execute('run', {'operator': 'theory_consolidation',
            'theme': 'ordered_affine', 'theories': declarations,
            'source_theory_ids': [item['id'] for item in declarations],
            'episode_sources': [source_ref(origin, f'episodes[{i}]') for i in positions],
            'counterexamples': {'history:t01': [check]},
            'mission': {'purpose': 'theory_learning', 'mission_id': 'frozen_installation_source',
                'theory_learning_allocation': {'compute': 30000, 'author_calls': 0, 'observations': 0}},
            'allocation': {'compute': 30000, 'max_groups': 4}}, lab)
        candidate_id = compiled['frozen_candidates'][0]['candidate_id']
        candidate = source.theory_consolidation['candidates'][candidate_id]
        parameters = next(item['parameters'] for item in candidate['instances']
                          if item['source_id'] == context('t01')['episode'])
        failed = source._execute('run', {'operator': 'theory_transfer',
            'candidate_id': candidate_id, 'theory_id': 'history:t13:failed_frozen_instance',
            'mapping': {'role_0': 'x'}, 'parameters': parameters,
            'role_signature': roles('x'), 'output_unit': UNIT,
            'context_ref': context('t13'), 'evidence_checks': [check]}, lab)
        assert failed['status'] == 'refuted'
        frozen_library = deepcopy(source.theory_consolidation)

        def freeze_state_references(value):
            if isinstance(value, dict):
                if value.get('file') and (source.workspace / value['file']).resolve() == (
                        source.workspace / 'research_state.json').resolve():
                    value['file'] = str(directory / 'index.json')
                for item in value.values():
                    freeze_state_references(item)
            elif isinstance(value, list):
                for item in value:
                    freeze_state_references(item)

        freeze_state_references(frozen_library)
        files = {directory / 'library.json': json.dumps(frozen_library, ensure_ascii=False, indent=2),
                 directory / 'index.json': json.dumps({'theory_index': source.theory_index},
                                                       ensure_ascii=False, indent=2)}
        for path, text in files.items():
            if path.exists():
                if path.read_text() != text:
                    raise ValueError('retain earlier frozen objects; choose a new frozen directory')
            else:
                path.write_text(text)
        knowledge = {'files': {},
            'theory_consolidation': {'file': str(directory / 'library.json'), 'json_path': []},
            'theory_index': {'file': str(directory / 'index.json'), 'json_path': ['theory_index']}}
        target = CommonArchiveInquiryAgent(None, Path(temporary) / 'target', arm='hegel',
            source_task=False, config=config, knowledge=knowledge)
        assert target.rows == target.observations == []
        assert target.costs['compute_cost'] == target.costs['observation_cost'] == 0
        assert set(source.theory_index).issubset(target.theory_index)
        assert target.theory_consolidation['candidates'][candidate_id] == candidate
        point = deepcopy(payload['episodes'][14]['actions'][0]['details']['sequence'][0])
        question = {'id': 'retained_installation_response', 'parent_id': 'task',
            'question': 'Which retained candidate applies at the original public successor input?',
            'depends_on': [], 'kind': 'value', 'answer_type': 'number',
            'context_ref': context('t15'), 'answer_resolution_ref': {'atol': 1e-6},
            'theory_target': {'role_signature': {**roles('u'), **roles('v')},
                             'output_unit': UNIT, 'context_ref': context('t15')},
            'query_semantics': {'mode': 'model_query', 'query_expression': {
                'op': 'Readout', 'record_index': 0, 'step': 0, 'target': 'response',
                'context_ref': context('t15'), 'units': UNIT,
                'binding': {'action': {'sequence': [point]}, 'features': point, 'history': []}}},
            'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
        alias = deepcopy(declarations[0])
        alias.update(id='ordinary_alias:failed_t13', context_ref=context('t13'))
        alias['support'] = {'status': 'candidate', 'context_ref': context('t13'), 'evidence': []}
        routed = target._execute('run', {'operator': 'research_route',
            'research_question': question, 'theories': [alias],
            'mission': {'purpose': 'task', 'mission_id': 'frozen_installation_target'}}, lab)
        recalled = routed['frozen_recall']['generated_theory_ids']
        assert recalled
        pool = {item['id']: item for item in resolved_theories(target)}
        assert pool[alias['id']]['compatible'] is False
        assert pool['history:t01'].get('compatible') is not False
        assert all(pool[name]['support']['status'] == 'candidate' and
                   pool[name]['support']['evidence'] == [] for name in recalled)
        for path, text in files.items():
            assert path.read_text() == text
        local = json.loads((target.workspace / LIBRARY_PATH).read_text())
        assert local['source_ref'] == knowledge['theory_consolidation']
        assert candidate_id not in local.get('candidates', {})
        assert not local.get('episodes') and not local.get('materials')
        outgoing = target.result('installation_checked')['outgoing_knowledge']
        assert LIBRARY_PATH not in outgoing['files'] and INDEX_PATH not in outgoing['files']
        local_index = json.loads((target.workspace / INDEX_PATH).read_text())
        assert not set(source.theory_index).intersection(local_index['entries'])
        successor = CommonArchiveInquiryAgent(None, Path(temporary) / 'successor', arm='hegel',
            source_task=False, config=config, knowledge=outgoing)
        assert successor.rows == successor.observations == []
        assert successor.theory_consolidation['candidates'][candidate_id] == candidate
        assert next(item for item in resolved_theories(successor)
                    if item['id'] == alias['id'])['compatible'] is False
        assert lab.execute_calls == source.costs['model_calls'] == target.costs['model_calls'] == 0
        return {'status': 'passed', 'scope': 'retained_object_installation_not_new_gain_comparison',
            'source': origin, 'frozen_knowledge': knowledge,
            'frozen_candidate': {'id': candidate_id, 'version': candidate['version']},
            'source_construction_compute': source.costs['compute_cost'],
            'target_compute': target.costs['compute_cost'], 'target_initial_paid_rows': 0,
            'automatic_recalled_ids': recalled, 'ordinary_failed_alias_stays_refuted': True,
            'source_context_remains_available': True, 'source_files_unchanged': True,
            'target_support': 'candidate_without_source_confirmation',
            'local_library_keys': list(local),
            'frozen_library_bytes': (directory / 'library.json').stat().st_size,
            'local_library_bytes': (target.workspace / LIBRARY_PATH).stat().st_size,
            'local_library_allocated_bytes': (target.workspace / LIBRARY_PATH).stat().st_blocks * 512,
            'local_index_bytes': (target.workspace / INDEX_PATH).stat().st_size,
            'outgoing_refs_install_in_another_fresh_workspace': True,
            'frozen_objects_repeated_in_target_library': False,
            'new_provider_calls': 0, 'new_scientific_acquisitions': 0,
            'independent_successor_gain': 'not_tested'}


def replay(pointer, output):
    payload, source = load_source(pointer)
    episodes = payload['episodes']
    sources = [(0, episodes[0]), (5, episodes[5])]
    target_position, target = 14, episodes[14]
    assert [episode['task_id'] for _, episode in sources] == ['t01', 't06']
    assert target['task_id'] == 't15'
    original_sources = [deepcopy(episode) for _, episode in sources]
    counter = next(item for item in payload['knowledge']['concepts'][0]['counterexamples']
                   if item['task_id'] == 't13')
    counter_index = payload['knowledge']['concepts'][0]['counterexamples'].index(counter)
    counter_ref = source_ref(source, f'knowledge.concepts[0].counterexamples[{counter_index}]')
    counter_check = {'source_ref': counter_ref, 'observation': deepcopy(counter['observation']),
                     'context_ref': context('t13'), 'tolerance': 1e-6}
    declarations = [declaration(source, episode, position) for position, episode in sources]
    mission = {'purpose': 'theory_learning', 'mission_id': 'historical_ordered_affine_replay',
               'theory_learning_allocation': {'compute': 30000, 'author_calls': 0, 'observations': 0}}
    lab = NoAcquisitionLab()
    with tempfile.TemporaryDirectory(prefix='hegel-theory-consolidation-replay-') as temporary:
        agent = CommonArchiveInquiryAgent(None, temporary, arm='hegel', source_task=False,
            config=InquiryLLMConfig(compute_budget=50000, observation_budget=0,
                                   max_model_calls=0, execution_seconds=30.))
        before_costs = deepcopy(agent.costs)
        arguments = {'operator': 'theory_consolidation', 'theme': 'ordered_affine',
            'theories': declarations, 'source_theory_ids': [item['id'] for item in declarations],
            'episode_sources': [source_ref(source, f'episodes[{position}]')
                                for position, episode in sources],
            'counterexamples': {'history:t01': [counter_check]},
            'materials': [{'id': 'known_transfer_counterexample', 'kind': 'error',
                'source_ref': counter_ref,
                'claim': {'status': 'failed_frozen_instance',
                          'scope': 'cross-context numerical reuse; not refutation of every refittable affine model'}}],
            'mission': mission, 'allocation': {'compute': 30000, 'max_groups': 4}}
        consolidated = agent._execute('run', deepcopy(arguments), lab)
        assert consolidated['frozen_candidates'], consolidated
        candidate_id = consolidated['frozen_candidates'][0]['candidate_id']
        candidate = agent.theory_consolidation['candidates'][candidate_id]
        assert len(candidate['instances']) == 2
        assert all(item['exact'] for item in candidate['source_reconstruction'])
        assert candidate['transfer_status'] == 'not_established'
        assert candidate['global_support'] is False
        frozen_source = next(item for item in candidate['instances']
                             if item['source_id'] == context('t01')['episode'])
        frozen_parameters = deepcopy(frozen_source['parameters'])
        # Frozen source values are chosen before any target example is checked.
        # The target Program/parameters are never passed to the transfer route.
        pool_before = [item['id'] for item in resolved_theories(agent)]
        transfer_args = {'operator': 'theory_transfer', 'library_path': consolidated['library_path'],
            'candidate_id': candidate_id, 'theory_id': 'replay:t15:frozen_source_t01',
            'mapping': {'role_0': 'u'}, 'parameters': frozen_parameters,
            'role_signature': roles('u'), 'output_unit': UNIT, 'context_ref': context('t15')}
        registered = agent._execute('run', deepcopy(transfer_args), lab)
        assert registered['status'] == 'candidate_registered', registered
        pool = resolved_theories(agent)
        transferred = next(item for item in pool if item['id'] == transfer_args['theory_id'])
        assert transferred['compatible'] is True and transferred['support']['evidence'] == []
        transferred_program = Program.from_dict(transferred['program'])
        rows = target['evaluation']['prediction_examples']
        answers = [model_readout(transferred_program, row, 't15') for row in rows]
        assert all(answer['status'] == 'answered' and answer['evidence'] == [] for answer in answers)
        revision = next(action['details'] for action in target['actions'] if action['action'] == 'revise')
        before_program = Program.from_dict(revision['before_program'])
        source_program = Program.from_dict(sources[0][1]['program'])
        assert transferred_program == source_program.remap_features({'x': 'u'})
        before_loss, transfer_loss = mse(before_program, rows), mse(transferred_program, rows)
        assert transfer_loss < 1e-12 and transfer_loss < before_loss
        checked = agent._execute('run', {**deepcopy(transfer_args),
            'evidence_checks': measured_checks(source, target, target_position)}, lab)
        assert all(check['status'] == 'passed' for check in checked['checks'])

        failed = agent._execute('run', {**deepcopy(transfer_args),
            'theory_id': 'replay:t13:failed_frozen_source_t01', 'mapping': {'role_0': 'x'},
            'role_signature': roles('x'), 'context_ref': context('t13'),
            'evidence_checks': [counter_check]}, lab)
        assert failed['status'] == 'refuted' and failed['checks'][0]['status'] == 'failed'
        assert next(item for item in resolved_theories(agent)
                    if item['id'] == failed['theory_id'])['compatible'] is False
        assert agent.theory_consolidation['candidates'][candidate_id]['source_reconstruction'] == candidate['source_reconstruction']
        after_work = deepcopy(agent.costs)
        repeated = agent._execute('run', deepcopy(arguments), lab)
        assert repeated['cache_hit'] is True and repeated['new_compute_cost'] == 0
        assert after_work == agent.costs
        assert [episode for _, episode in sources] == original_sources
        assert lab.execute_calls == agent.costs['model_calls'] == 0
        assert agent.costs['observation_cost'] == before_costs['observation_cost'] == 0
        assert agent.costs['measurement_cost'] == before_costs['measurement_cost'] == 0
        receipt = {'status': 'passed', 'scope': 'historical_adoption_and_frozen_transfer_replay',
            'source': source, 'theme': {'name': 'ordered_affine', 'selection': 'operator_selected',
                'discovery_claim': False, 'context_family_ref': FAMILY,
                'coordinate_contract': 'Original public numeric coordinates; no physical units inferred'},
            'source_episodes': [episode_facts(source, episode, position) for position, episode in sources],
            'lineage': {'distinct_measured_task_episodes': 2,
                'independent_discoveries': False,
                'reason': 't06 selected_operation is abstract_reuse and its accepted reuse cites concept_0001 from t01'},
            'consolidation': consolidated,
            'candidate': {key: deepcopy(candidate[key]) for key in (
                'id', 'kind', 'frozen', 'parameter_paths', 'source_reconstruction',
                'source_contexts', 'source_ids', 'counterexample_checks',
                'blocked_transfer_bindings', 'encoding_cost', 'active_library',
                'transfer_status', 'global_support')},
            'retained_episode_extraction': [extracted_episode_facts(episode)
                for episode in agent.theory_consolidation['episodes']
                if episode['episode_id'] in {'t01@meaning1', 't06@meaning1'}],
            'frozen_transfer': {'target_source_ref': source_ref(source, f'episodes[{target_position}]'),
                'source_parameter_episode': frozen_source['source_id'], 'parameters': frozen_parameters,
                'mapping': transfer_args['mapping'], 'target_parameters_used': False,
                'target_fit_calls': 0, 'pool_before': pool_before,
                'pool_after_registration': [item['id'] for item in pool],
                'new_candidate_registered': registered['candidate_pool_changed'],
                'model_readouts': [{key: answer.get(key) for key in ('status', 'value', 'evidence')}
                                   for answer in answers],
                'same_saved_evaluation_examples': len(rows), 'before_program_mse': before_loss,
                'frozen_source_program_mse': transfer_loss, 'checks': checked['checks'],
                'support_updated': checked['support_updated'],
                'independent_confirmation': checked['independent_confirmation']},
            'counterexample': {'source_ref': counter_ref,
                'historical_full_segment_concept_loss': counter['concept_loss'],
                'historical_replacement_loss': counter['replacement_loss'],
                'single_saved_row_replay': failed['checks'],
                'target_instance_status': failed['status'], 'target_excluded_from_compatible_pool': True,
                'source_reconstruction_preserved': failed['source_reconstruction_preserved'],
                'entire_refittable_family_refuted': False},
            'repeat_call': {'cache_hit': repeated['cache_hit'], 'new_compute_cost': 0,
                            'costs_unchanged': True},
            'no_acquisition': {'execute_calls': lab.execute_calls, 'model_calls': agent.costs['model_calls'],
                'new_observations': 0, 'new_observation_cost': 0, 'new_measurement_cost': 0},
            'limitations': ['The theme and historical cases are operator-selected; this is not autonomous discovery or a blind transfer trial.',
                'The two measured source Episodes share an abstraction lineage, so they are not two independent discoveries.',
                'Only four already-saved target evaluation examples are replayed; no new confirmation or whole-task gain is claimed.',
                'The new-domain counterexample rejects the frozen numerical instance there, not every refittable affine predictor.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = replay(args.source_pointer, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output),
        'frozen_source_program_mse': result['frozen_transfer']['frozen_source_program_mse'],
        'before_program_mse': result['frozen_transfer']['before_program_mse'],
        'no_acquisition': result['no_acquisition']}, ensure_ascii=False))
