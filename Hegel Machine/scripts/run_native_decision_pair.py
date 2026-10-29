#!/usr/bin/env python3
"""Reconstruct one paid native decision and compare its recorded alternatives."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from math import isclose
from pathlib import Path

from scripts.native_decision_scene import load_native_decision_scene
from scripts.run_partial_role_parent import load, save, plain
from scripts.run_partial_role_successors import knowledge_changes, apply_knowledge_changes
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return,
)
from hegel_machine.cognition.experiment_runtime import (
    ExecutionTiming, python_execution_deadline, run_process_trials,
)
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.method_feedback import ensure_python_method_topic
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.question_routing import consume_method_return_agenda
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


ROOT = Path(__file__).resolve().parents[1]
_SHARED = None


def ref(path, *keys):
    return {'file': str(path), **({'json_path': list(keys)} if keys else {})}


def bind_native_method_artifacts(output, registration, decision_ref):
    """Keep method identity fixed while decision provenance changes."""
    from hegel_machine.cognition.method_action_research import read_method_reference

    retained_ref = registration.get('retained_method_pair_ref')
    retained = read_method_reference(retained_ref, output) if retained_ref else None
    bindings = {}
    for name, operation in registration['operations'].items():
        identity = {'id': 'native:' + operation,
                    'version': registration.get('method_versions', {}).get(
                        name, registration.get('method_version', 1))}
        execution = {'entry': 'UnifiedResearchAgent.execute_research_decision',
                     'research_action': operation}
        if registration.get('knowledge_contract_ref') is not None:
            execution['knowledge_ref'] = deepcopy(registration['knowledge_contract_ref'])
        revision = registration.get('method_program_revisions', {}).get(name)
        if revision is not None:
            execution['research_program_revision_ref'] = deepcopy(revision)
        artifact_ref = registration.get('retained_method_artifact_refs', {}).get(name)
        if artifact_ref is not None:
            artifact = read_method_reference(artifact_ref, output)
            if ({key: artifact.get(key) for key in ('id', 'version')} != identity
                    or artifact.get('kind') != 'native_research_segment'
                    or artifact.get('execution') != execution):
                raise ValueError('Retained native method body differs from the declared operation')
        elif retained is not None:
            outcome = read_method_reference(retained[f'{name}_ref'], output)
            artifact_ref = outcome['method_artifact_ref']
            artifact = read_method_reference(artifact_ref, output)
            if (outcome['method_ref'] != identity
                    or {key: artifact.get(key) for key in ('id', 'version')} != identity
                    or artifact.get('kind') != 'native_research_segment'
                    or artifact.get('execution') != execution):
                raise ValueError('Retained native method body differs from the declared operation')
        else:
            artifact_ref = ref(output / f'{name}_method.json')
            save(output / f'{name}_method.json', {**identity,
                'kind': 'native_research_segment', 'execution': execution,
                'source_ref': deepcopy(decision_ref)})
        bindings[name] = {'method_ref': identity,
                         'method_artifact_ref': deepcopy(artifact_ref),
                         'execution': execution}
    save(output / 'method_bindings.json', {'methods': bindings,
        'decision_ref': deepcopy(decision_ref), 'retained_method_pair_ref': retained_ref})
    return bindings


def register(source, prior_review, output, *, seed=106029101, arm='pr3',
             decision_index=None, retained_method_pair_ref=None, knowledge_root=None):
    scene = load_native_decision_scene(source, seed=seed, arm=arm, decision_index=decision_index,
                                      **({'knowledge_root': knowledge_root} if knowledge_root else {}))
    # Hold the two executable methods fixed across different paid states. The
    # archived ordinary selector can choose another action in the later state.
    operations = {'baseline': 'program_replay', 'challenger': 'structural_research'}
    if not set(operations.values()) <= set(scene['choice']['eligible_actions']):
        raise ValueError('Both fixed research segments must be available at this choice')
    prior_topic = next(q for q in load(prior_review / 'common/questions.json')
                       if q['id'] == 'task:method_choice')
    review_compute = 2 + len(prior_topic.get('method_use_plan', {}).get('native_return_refs', []))
    output.mkdir(parents=True, exist_ok=False)
    registration = {'source': str(source), 'source_seed': seed, 'source_arm': arm,
        'source_task_ref': scene['source_task_ref'],
        'prior_method_review': str(prior_review), 'choice_ref': scene['references']['choice'],
        'operations': operations, 'source_references': scene['references'],
        'decision_index': scene['choice_index'],
        'recorded_ordinary_operation': scene['ordinary_operation'],
        'contrast_scope': 'fixed_native_segments_not_a_replay_of_the_ordinary_policy',
        'retained_method_pair_ref': retained_method_pair_ref,
        'historical_knowledge_ref': scene['historical_knowledge_ref'],
        'preparation_compute_limit': 90000, 'per_arm_compute_limit': scene['remaining_compute'],
        'per_arm_active_wall_seconds': 1800, 'method_review_compute_limit': review_compute,
        'common_compute_limit': 90000 + review_compute,
        'common_allocation': 'unused_preparation_allowance_funds_method_conditions_before_arm_results',
        'training_rows': len(scene['training']), 'comparison_rows': len(scene['validation']),
        'quality_threshold': scene['research_config']['minimum_improvement'],
        'evaluation_role': 'paid_working_comparison', 'maximum_new_observations': 0,
        'scope': 'one reconstructed development decision; original nine runs unchanged',
        'restoration_scope': scene['restoration_scope'],
        'parent_reconstruction': scene.get('parent_reconstruction'),
        'stop_rule': 'retain incomplete work; no second decision, new data or budget extension'}
    if knowledge_root is not None:
        registration.update(knowledge_root=str(knowledge_root),
            knowledge_contract_ref=ref(Path(knowledge_root) / 'knowledge_delta.json'),
            method_version=3, knowledge_scope='retrospective_development_with_frozen_later_knowledge',
            source_cost_refs=[ref(Path(knowledge_root) / 'receipt.json')],
            object_identity_remap=scene['object_identity_remap'],
            source_successful_research_repeated=False)
    save(output / 'registration.json', registration)
    names = [p.relative_to(ROOT) for p in (ROOT / 'src/hegel_machine').rglob('*')
             if p.is_file() and p.suffix in {'.py', '.json'}]
    names.extend(Path('scripts') / name for name in (
        'run_native_decision_pair.py', 'native_decision_scene.py',
        'accept_partial_role_structural_seed.py', 'run_partial_role_parent.py',
        'run_partial_role_successors.py'))
    if knowledge_root is not None:
        names.extend(Path('scripts') / name for name in (
            'successor_interval_scene.py', 'revise_learned_structure_successor.py',
            'run_learned_structure_successor.py', 'retain_confirmed_structure.py',
            'partial_role_paid_scene.py'))
    for name in names:
        target, base = output / 'runtime' / name, source / 'runtime' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_incrementally(target, (ROOT / name).read_text(), base=base if base.exists() else None)
    return registration


def method_review_allocation(registration, preparation_spent):
    """Share the registered common allowance without changing either arm's cap."""
    unused = max(0, registration['preparation_compute_limit'] - preparation_spent)
    return {'preparation_compute': preparation_spent,
        'unused_preparation_compute': unused,
        'metadata_review_reserved': registration['method_review_compute_limit'],
        'method_review_compute_limit': registration['method_review_compute_limit'] + unused,
        'common_compute_limit': (registration['preparation_compute_limit']
            + registration['method_review_compute_limit']),
        'allocation_time': 'after_common_preparation_before_paired_research'}


def preparation_matches(frame, scene):
    """Compare the paid historical scores, without predicting a second time."""
    expected = {item['kind']: item['validation_loss']
                for item in scene['expected_preparation']['partial_claims']}
    expected.update(current_affine=scene['expected_preparation']['affine_validation_loss'],
        retained_response=scene['expected_preparation']['retained_response_validation_loss'])
    retained = frame.get('retained_on_current')
    actual = {item['kind']: item['validation_loss'] for item in frame['assessed']
              if item is not retained}
    # A retained partial-object program can have the same kind as its current
    # refit. They are distinct candidates with different paid scores.
    if retained is not None:
        actual['retained_response'] = retained['validation_loss']
    missing = sorted(set(expected) - set(actual))
    changed = {kind: {'saved': value, 'reconstructed': actual[kind]}
        for kind, value in expected.items() if kind in actual and
        not isclose(value, actual[kind], rel_tol=1e-8, abs_tol=1e-10)}
    return {'matched': not missing and not changed, 'missing': missing, 'changed': changed,
        'saved_candidates': len(expected), 'prepared_candidates': len(frame['assessed']),
        'scope': 'same public inputs and stored scores; old executable checkpoint was absent'}


def save_prepared(output, agent, frame, scene, actions):
    # All paid observations and the complete original library stay referenced.
    # Only the newly reconstructed pool and small decision bindings are written.
    entries = frame['assessed']
    save(output / 'reconstruction_ledger.json', {
        'compute_cost': frame['prepare_compute_cost'], 'new_observations': 0,
        'comparison_complete': frame['comparison_complete'],
        'old_source_costs_ref': ref(Path(scene['references']['report']['file']), 'costs')})

    def encode(item):
        encoded = encode_entry(item)
        for key in ('research_action_origin', 'object_action_origin', 'risk_action_origin'):
            if key in item:
                encoded[key] = plain(item[key])
        return encoded

    # Persist the paid fit results before assembling ancillary metadata. A
    # metadata failure must not lose the completed comparison or its charge.
    save(output / 'prepared_pool.json', [encode(item) for item in entries])

    def entry_index(item):
        if item is None:
            return None
        return next(i for i, saved in enumerate(entries) if saved is item)

    replay_plans = []
    for source, payload, template, mappings in frame.get('replay_plans', []):
        prior_index, payload_field = next((i, field)
            for i, prior in enumerate(agent.library.meta_priors)
            if prior['source_concept_id'] == source
            for field in ('input_research_program', 'research_program')
            if prior.get(field) is payload)
        replay_plans.append({'source': source, 'prior_index': prior_index,
            'payload_field': payload_field, 'template': template.to_dict(),
            'mappings': [{role: {'program': value.to_dict()} if hasattr(value, 'to_dict')
                         else {'feature': value} for role, value in mapping.items()}
                        for mapping in mappings]})
    state = {'source_references': scene['references'],
        'historical_knowledge_ref': scene['historical_knowledge_ref'],
        'knowledge_base_ref': deepcopy(scene.get('knowledge_base_ref', scene['historical_knowledge_ref'])),
        'knowledge_additions': scene['knowledge_additions'],
        'knowledge_delta': knowledge_changes(scene['knowledge'], plain(agent.library.to_dict())),
        'runtime_state': {'learn_instruments': agent._learn_instruments,
            'revise_inputs': agent._revise_inputs, 'object_view': ['fine'],
            'object_model_ref': scene['references']['object_model'],
            'callable_fit_serial': agent._callable_fit_serial,
            'measurement_budget': agent.measurement_budget,
            'downstream_demands': agent.downstream_demands},
        'assessed': [ref(output / 'prepared_pool.json', i) for i in range(len(entries))],
        'uncompared': [encode(item) for item in frame['uncompared']], 'actions': actions,
        'parent_ref': scene['references']['parent_original'],
        'parent_metadata': {key: plain(value) for key, value in (frame['parent'] or {}).items()
                            if key != 'candidate' and (scene['parent'] or {}).get(key) != value},
        'spec': {key: value for key, value in asdict(frame['spec']).items() if key != 'context'},
        'spec_context_refs': {'object_model': scene['references']['object_model']},
        'calls': [call.to_dict() for call in frame.get('calls', [])],
        'call_origins': [{'program': program.to_dict(), 'origin': origin}
                         for program, origin in frame.get('call_origins', {}).items()],
        'replay_plans': replay_plans,
        'relation_kernels': [{'program': program.to_dict(), 'sources': plain(sources)}
                             for program, sources in frame.get('relation_kernels', {}).items()],
        'directions': frame.get('directions', []), 'fallback': frame['fallback'],
        'prepare_compute_cost': frame['prepare_compute_cost'],
        'remaining_compute': frame['remaining_compute'],
        'terminal': frame['terminal'], 'independent_validation': frame['independent_validation'],
        'observation_remaining': frame['observation_remaining'],
        'measurement_remaining': frame['measurement_remaining'],
        'baseline_index': entry_index(frame.get('baseline')),
        'retained_on_current_index': entry_index(frame.get('retained_on_current')),
        'decision_state': {key: frame[key] for key in ('structured', 'usable_parent',
            'parent_fitted_comparison', 'state_identification', 'experiment_compute', 'learn_policy')
            if key in frame},
        'comparison_complete': frame['comparison_complete'],
        'old_comparison_scores_are_paid': True}
    if frame.get('policy'):
        policy = frame['policy']
        state['policy'] = {'base_context': policy.base_context, 'context': policy.context,
            'action_contexts': policy._action_contexts, 'last_decision': policy.last_decision}
    save(output / 'prepared_state.json', state)


def load_prepared(output, scene=None):
    """Restore the paid decision without fitting, scoring or replay planning."""
    from hegel_machine.cognition.abstractions import AbstractTemplate
    from hegel_machine.cognition.action_policy import ResearchActionPolicy
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.object_continuation import decode_entry

    output = Path(output)
    state = load(output / 'prepared_state.json')
    if scene is None:
        registration = load(output / 'registration.json')
        scene = load_native_decision_scene(registration['source'],
            seed=registration.get('source_seed', 106029101), arm=registration.get('source_arm', 'pr3'),
            decision_index=registration.get('decision_index'),
            **({'knowledge_root': registration['knowledge_root']} if registration.get('knowledge_root') else {}))
    knowledge = apply_knowledge_changes(scene['knowledge'], state['knowledge_delta'])
    runtime = state['runtime_state']
    agent = UnifiedResearchAgent(ResearchConfig(**scene['research_config']),
        library=KnowledgeLibrary.from_dict(knowledge),
        measurement_budget=runtime['measurement_budget'],
        downstream_demands=runtime['downstream_demands'])
    agent._learn_instruments, agent._revise_inputs = runtime['learn_instruments'], runtime['revise_inputs']
    agent._callable_fit_serial = runtime['callable_fit_serial']
    view = tuple(runtime['object_view'])
    agent._object_models[view] = deepcopy(scene['spec'].context['object_model'])
    if scene['object_model'].get('schema') is not None:
        agent._object_schemas[view] = deepcopy(scene['object_model']['schema'])
    entries = [decode_entry(item) for item in load(output / 'prepared_pool.json')]
    parent = deepcopy(scene['parent'])
    if parent is not None:
        parent.update(deepcopy(state['parent_metadata']))
    frame = {key: state[key] for key in ('fallback', 'prepare_compute_cost', 'remaining_compute',
        'terminal', 'independent_validation', 'observation_remaining', 'measurement_remaining',
        'directions', 'comparison_complete')}
    frame.update(spec=scene['spec'], training=scene['training'], validation=scene['validation'],
        comparison=scene['validation'] or scene['training'], transitions=scene['transitions'],
        parent=parent, assessed=entries,
        uncompared=[decode_entry(item) for item in state['uncompared']], **state['decision_state'])
    for key in ('baseline', 'retained_on_current'):
        index = state[key + '_index']
        frame[key] = entries[index] if index is not None else None
    frame['calls'] = [Program.from_dict(item) for item in state['calls']]
    frame['call_origins'] = {Program.from_dict(item['program']): item['origin']
                             for item in state['call_origins']}
    frame['relation_kernels'] = {Program.from_dict(item['program']): item['sources']
                                 for item in state['relation_kernels']}
    frame['replay_plans'] = [(item['source'],
        agent.library.meta_priors[item['prior_index']][item['payload_field']],
        AbstractTemplate.from_dict(item['template']),
        [{role: Program.from_dict(value['program']) if 'program' in value else value['feature']
          for role, value in mapping.items()} for mapping in item['mappings']])
        for item in state['replay_plans']]
    frame['policy'] = None
    if state.get('policy'):
        values = state['policy']
        policy = ResearchActionPolicy(frame['training'], frame['spec'],
            agent.library.action_experience if frame['learn_policy'] else {})
        policy.base_context, policy.context = values['base_context'], values['context']
        policy.experience = policy._store.setdefault(policy.context, {})
        policy._action_contexts, policy.last_decision = values['action_contexts'], values['last_decision']
        frame['policy'] = policy
    return agent, frame, state['actions']


def consume_saved_structural_return(output, retained):
    """Consume an existing same-input structural return, without rerunning it.

    This is a paid-result consumption receipt, not a new method pair or proof
    that the old search covered a different allowance.
    """
    from scripts.accept_partial_role_structural_seed import recover_paid_scene
    from hegel_machine.cognition.object_continuation import decode_entry

    output, retained = Path(output), Path(retained)
    registration = load(output / 'registration.json')
    scene = load_native_decision_scene(registration['source'],
        seed=registration.get('source_seed', 106029101), arm=registration.get('source_arm', 'pr3'),
        decision_index=registration.get('decision_index'),
        **({'knowledge_root': registration['knowledge_root']} if registration.get('knowledge_root') else {}))
    old_registration, old_result = load(retained / 'registration.json'), load(retained / 'receipt.json')
    if old_registration['source_report_ref'] != scene['references']['report']:
        raise ValueError('Saved structural return belongs to another source')
    earlier = recover_paid_scene(load(Path(old_registration['source_report_ref']['file'])))
    agent, frame, _ = load_prepared(output, scene=scene)
    if (earlier['training'] != frame['training'] or earlier['validation'] != frame['comparison']
            or earlier['transitions'] != frame['transitions']
            or earlier['spec'].feature_names != frame['spec'].feature_names
            or earlier['spec'].risk_quantile != frame['spec'].risk_quantile
            or earlier['spec'].context['object_schema'] != frame['spec'].context['object_schema']):
        raise ValueError('Saved structural fit inputs or comparison differ')
    actions = []
    selected = agent._select_structure_response_parent(frame['spec'], frame['parent'],
        frame['assessed'], frame['retained_on_current'], frame['usable_parent'],
        actions, len(frame['comparison']))
    prior = decode_entry(old_result['selected_response'])
    if (selected['candidate'].program != prior['candidate'].program
            or any(selected.get(key) != prior.get(key)
                   for key in ('research_program', 'structure', 'partial_object_knowledge'))):
        raise ValueError('Saved structural return has a different response starting point')
    source_config = load(Path(old_registration['source_registration_ref']['file']))['research_config']
    if source_config['minimum_improvement'] != agent.config.minimum_improvement:
        raise ValueError('Saved structural return has a different fitting tolerance')
    claims = [decode_entry(item) for item in load(retained / 'new_claims.json')]
    budget = SearchBudget(0)
    result = agent.compare_research_decision_return(frame, 'structural_research', claims,
        budget, actions, paid_scores={item['candidate'].program: item['validation_loss'] for item in claims},
        reused_paid_return=True)
    receipt = {'scope': 'paid_return_reconsumption_not_new_method_pair',
        'start_ref': ref(output / 'prepared_state.json'),
        'paid_claims_ref': ref(retained / 'new_claims.json'),
        'source_cost_ref': ref(retained / 'receipt.json'),
        'old_candidates': len(frame['assessed']), 'consumed_candidates': len(claims),
        'full_comparison': result['full_comparison'], 'loss': result['selected']['validation_loss'],
        'compute_cost': budget.spent, 'new_fits': 0, 'new_observations': 0,
        'search_exhaustiveness_established': False, 'independent_confirmation': False,
        'actions': actions}
    save(output / 'paid_return_consumption.json', receipt)
    return receipt


def apply_method_program_revision(agent, frame, patch, actions):
    """Bind an already retained method revision to its existing replay slots."""
    index = patch['prior_index']
    original = agent.library.meta_priors[index]
    retained = patch['prior']
    if (original['source_concept_id'] != patch['source_concept_id']
            or retained['source_concept_id'] != original['source_concept_id']):
        raise ValueError('The paired method revision must preserve its world knowledge')
    # Common preparation may already update use/evidence metadata. Only the
    # two verified procedure fields belong to this intervention.
    revised = deepcopy(original)
    for field in ('research_program', 'input_research_program'):
        if field in retained:
            revised[field] = deepcopy(retained[field])
    if not any(payload is original.get(key)
            for _, payload, _, _ in frame['replay_plans']
            for key in ('research_program', 'input_research_program')):
        raise ValueError('The repaired method is absent from the shared replay queue')
    agent.library.meta_priors[index] = revised
    # A zero-weight source column can require inputs absent from its old final
    # answer. Reuse ordinary binding over the complete revised input template.
    frame['replay_plans'] = agent._replay_program_queue(
        frame['spec'], frame['calls'], frame['training'])
    replaced = sum(payload is revised.get(key)
        for _, payload, _, _ in frame['replay_plans']
        for key in ('research_program', 'input_research_program'))
    if not replaced:
        raise ValueError('The revised method has no binding in this public representation')
    actions.append({'action': 'bind_retained_method_revision',
        'source_concept_id': patch['source_concept_id'], 'repaired_entry_ref': patch['repaired_entry_ref'],
        'rebound_plans': replaced, 'new_fits': 0, 'new_predictions': 0})


def run_arm(name):
    output, registration = _SHARED['output'], _SHARED['registration']
    agent, frame = deepcopy((_SHARED['agent'], _SHARED['frame']))
    directory = output / name
    directory.mkdir(exist_ok=False)
    budget, actions = SearchBudget(registration['per_arm_compute_limit']), []
    timing = ExecutionTiming(('research', 'save'))
    try:
        revision = registration.get('method_program_revisions', {}).get(name)
        if revision is not None:
            from hegel_machine.cognition.method_action_research import read_method_reference
            apply_method_program_revision(agent, frame, read_method_reference(revision, output), actions)
        with timing.phase('research'), python_execution_deadline(
                registration['per_arm_active_wall_seconds'], activity=True):
            result = agent.execute_research_decision(frame, registration['operations'][name], budget, actions)
        with timing.phase('save'):
            save(directory / 'new_claims.json', [encode_entry(item) for item in result['new_claims']])
            count = len(frame['assessed'])
            candidates = [ref(output / 'prepared_pool.json', i) for i in range(count)]
            candidates.extend(ref(directory / 'new_claims.json', i)
                              for i in range(len(result['new_claims'])))
            chosen = next((i for i, item in enumerate(result['claims']) if item is result['selected']), None)
            if chosen is None and result['selected'] is not None:
                chosen = result['claims'].index(result['selected'])
            selected_ref = candidates[chosen] if chosen is not None else None
            loss = result['selected']['validation_loss'] if result['selected'] is not None else None
            save(directory / 'comparison.json', {'full_comparison': result['full_comparison'],
                'scores': [{'candidate_ref': candidate_ref, 'loss': item.get('validation_loss')}
                    for candidate_ref, item in zip(candidates, result['claims'])],
                'selected_ref': selected_ref, 'actions': actions, 'quote': result.get('quote')})
            outcome = {'status': result['status'], 'execution_complete': result['execution_complete'],
                'completed': bool(result['execution_complete'] and loss is not None
                    and loss <= registration['quality_threshold']),
                'coverage': len(frame['comparison']), 'loss': loss,
                'method_ref': deepcopy(_SHARED['methods'][name]['method_ref']),
                'method_artifact_ref': deepcopy(_SHARED['methods'][name]['method_artifact_ref']),
                'profile_ref': ref(output / 'profile.json'),
                **{key + '_ref': ref(output / (file + '.json')) for key, file in (
                    ('start_state', 'prepared_state'), ('data', 'data_contract'),
                    ('split', 'split_contract'), ('quality_contract', 'quality_contract'),
                    ('resource_contract', 'resource_contract'), ('evaluation', 'evaluation_contract'))},
                'comparison_ref': ref(directory / 'comparison.json'), 'selected_ref': selected_ref,
                'old_candidates': count, 'new_candidates': len(result['new_claims']),
                'full_comparison': result['full_comparison'],
                'costs': {'compute_cost': budget.spent, 'observation_cost': 0, 'measurement_cost': 0,
                         'python_execution_seconds': 0, 'model_calls': 0, 'reported_tokens': 0},
                'search_compute_cost': result['search_compute_cost'],
                'comparison_compute_cost': result['comparison_compute_cost'],
                **{key: result.get(key) for key in ('completion_scope', 'search_budget_denied',
                    'last_denied_request', 'structure_search_complete')},
                'timing': {'activity_wall_seconds': timing.result()['wall_seconds']},
                'execution_timing': timing.result(), 'independent_confirmation': False}
            save(directory / 'outcome.json', outcome)
        return {'arm': name, 'outcome_ref': ref(directory / 'outcome.json'),
                'compute_cost': budget.spent, 'execution_complete': result['execution_complete']}
    except BaseException as error:
        save(directory / 'failure.json', {'error': repr(error), 'compute_known': budget.spent,
            'compute_upper_bound': budget.limit, 'execution_timing': timing.result()})
        raise


def run(output):
    global _SHARED
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Use the newly frozen current entry')
    registration = load(output / 'registration.json')
    if (output / 'reconstruction_ledger.json').exists():
        raise ValueError('Keep the existing paid reconstruction; do not rerun this unit')
    scene = load_native_decision_scene(registration['source'],
        seed=registration.get('source_seed', 106029101), arm=registration.get('source_arm', 'pr3'),
        decision_index=registration.get('decision_index'),
        **({'knowledge_root': registration['knowledge_root']} if registration.get('knowledge_root') else {}))
    cfg = ResearchConfig(**scene['research_config'])
    agent = UnifiedResearchAgent(cfg, library=KnowledgeLibrary.from_dict(scene['knowledge']))
    agent._object_models[('fine',)] = deepcopy(scene['spec'].context['object_model'])
    agent._object_schemas[('fine',)] = deepcopy(scene['object_model']['schema'])
    # The registered public laboratory has the measured fine channel. Restore
    # the same ordinary run preconditions, not the constructor's cold defaults.
    agent._learn_instruments = cfg.strategy != 'fixed'
    agent._revise_inputs = cfg.use_memory and agent._learn_instruments
    actions, preparation = [], SearchBudget(registration['preparation_compute_limit'])
    timing = ExecutionTiming(('preparation', 'paired_research', 'method_return'))
    stage = 'common_preparation'
    try:
        with timing.phase('preparation'), python_execution_deadline(1800, activity=True):
            frame = agent.prepare_research_decision(scene['spec'], scene['training'], scene['validation'],
                preparation, actions, parent=scene['parent'], transitions=scene['transitions'],
                observation_remaining=scene['observation_remaining'],
                measurement_remaining=scene['measurement_remaining'])
        stage = 'save_prepared_state'
        save_prepared(output, agent, frame, scene, actions)
    except BaseException as error:
        save(output / 'failure.json', {'stage': stage, 'error': repr(error),
            'compute_known': preparation.spent, 'compute_upper_bound': preparation.limit,
            'actions': actions, 'execution_timing': timing.result()})
        raise
    # Both arms actually start from the saved state, not an unsaved live object.
    agent, frame, actions = load_prepared(output, scene=scene)
    check = preparation_matches(frame, scene)
    save(output / 'preparation_receipt.json', {**check, 'compute_cost': preparation.spent,
        'historical_pool_reconstruction': True, 'new_observations': 0,
        'old_source_costs_ref': ref(Path(scene['references']['report']['file']), 'costs')})
    if not check['matched'] or not frame['comparison_complete'] or frame['terminal']:
        result = {'status': 'preparation_requires_review', 'preparation': check,
            'compute_cost': preparation.spent, 'arms_started': False}
        save(output / 'receipt.json', result)
        return result
    return finish_prepared_pair(output, registration, scene, agent, frame, preparation.spent, timing)


def finish_prepared_pair(output, registration, scene, agent, frame, preparation_spent, timing):
    """Execute matched native segments from a restored, complete common state."""
    global _SHARED
    if not frame['comparison_complete'] or frame['terminal']:
        result = {'status': 'prepared_pool_requires_no_paired_research' if frame['terminal']
                  else 'prepared_comparison_incomplete', 'arms_started': False,
                  'compute_cost': preparation_spent}
        save(output / 'receipt.json', result)
        return result
    methods = bind_native_method_artifacts(output, registration, scene['references']['choice'])
    for operation in registration['operations'].values():
        if operation not in frame['directions']:
            raise ValueError('The archived alternative is unavailable in the reconstructed state')
    review_allocation = method_review_allocation(registration, preparation_spent)
    save(output / 'method_review_allocation.json', review_allocation)
    review_limit = review_allocation['method_review_compute_limit']
    save(output / 'data_contract.json', {'training': scene['references']['training'],
        'comparison': scene['references']['validation'], 'new_observations': 0})
    save(output / 'split_contract.json', {'training_rows': len(scene['training']),
        'validation_rows': len(scene['validation']), 'validation_is_exposed': True,
        'cutoff_choice_ref': scene['references']['choice']})
    save(output / 'quality_contract.json', {'threshold': registration['quality_threshold'],
        'metric': 'MSE_on_original_exposed_validation', 'coverage': len(scene['validation'])})
    save(output / 'resource_contract.json', {'compute_limit': registration['per_arm_compute_limit'],
        'observation_limit': 0, 'time_limits': {
            'active_wall_seconds': registration['per_arm_active_wall_seconds']}})
    save(output / 'evaluation_contract.json', {'role': 'paid_working_comparison',
        'data_ref': scene['references']['validation'], 'labels_already_exposed': True})
    incumbent = agent._best(frame['assessed'])
    profile = method_profile(scene['spec'], [*scene['training'], *scene['validation']],
        incumbent=incumbent['candidate'] if incumbent else None,
        residual_loss=incumbent['validation_loss'] if incumbent else None,
        baseline_loss=frame['baseline']['validation_loss'] if frame['baseline'] else None,
        minimum_improvement=registration['quality_threshold'],
        research_state={'stage': 'native_direction_choice',
            'comparison': {'stale': False, 'paid_rows': len(frame['comparison']),
                'models_requested': len(frame['assessed']), 'models_completed': len(frame['assessed'])},
            'budget': {'remaining_compute': registration['per_arm_compute_limit']}})
    save(output / 'profile.json', profile)
    _SHARED = {'output': output, 'registration': registration, 'agent': agent, 'frame': frame,
               'methods': methods}
    with timing.phase('paired_research'):
        arms = run_process_trials(run_arm, ['baseline', 'challenger'], 2)
    _SHARED = None
    pair = {'status': 'returned', 'comparison_kind': 'matched_native_segments',
        'baseline_ref': ref(output / 'baseline/outcome.json'),
        'challenger_ref': ref(output / 'challenger/outcome.json'),
        'baseline_action_content': methods['baseline']['execution'],
        'challenger_action_content': methods['challenger']['execution'],
        'decision_ref': deepcopy(scene['references']['choice']),
        'physical_source_ref': scene['source_task_ref'], 'data_role': 'development',
        'evaluation_role': 'paid_working_comparison',
        'identification_target': 'fixed_data_method_effect', 'origin_ref': ref(output / 'registration.json')}
    save(output / 'pair.json', pair)
    result = {'status': 'paired_segments_returned', 'source_task_ref': scene['source_task_ref'],
        'method_pair_ref': ref(output / 'pair.json'),
        'method_ref': deepcopy(methods['challenger']['method_ref']),
        'method_artifact_ref': deepcopy(methods['challenger']['method_artifact_ref']),
        'preparation_compute': preparation_spent, 'arm_results': arms,
        'method_review_allocation_ref': ref(output / 'method_review_allocation.json'),
        'research_compute': preparation_spent + sum(arm['compute_cost'] for arm in arms),
        'prior_attempt_cost_ref': registration.get('prior_attempt_cost_ref'),
        'observation_cost': 0, 'measurement_cost': 0, 'model_calls': 0,
        'scope': registration.get('scope', 'reconstructed exposed source decision')}
    save(output / 'research_receipt.json', result)
    with timing.phase('method_return'):
        previous = Path(registration['prior_method_review'])
        questions = load(previous / 'common/questions.json')
        owner = next(q for q in questions if q['id'] == 'task:method_choice')
        save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
        common = CommonObjectDependencyMethods(output / 'common', [],
            budget=SearchBudget(review_limit), actions=[],
            source_task_ref=scene['source_task_ref']).common
        ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
            topic_ref=ref(output / 'previous_topic.json'))
        mission = deepcopy(owner['mission'])
        mission['allocation']['compute'] += review_limit
        feedback = record_native_parent_research_return(common,
            source_task_ref=scene['source_task_ref'], result_ref=ref(output / 'research_receipt.json'),
            before_ref=ref(output / 'prepared_state.json'), cost_refs=[ref(output / 'research_receipt.json')],
            mission=mission)
        learned = consume_method_return_agenda(common, feedback, None,
            compute_limit=review_limit)
    result.update(method_return=learned, method_review_compute=common.costs['compute_cost'],
        common_compute_cost=preparation_spent + common.costs['compute_cost'],
        common_compute_limit=review_allocation['common_compute_limit'],
        compute_cost=result['research_compute'] + common.costs['compute_cost'],
        execution_timing=timing.result())
    save(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'consume-paid-return'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--seed', type=int, default=106029101)
    parser.add_argument('--arm', default='pr3', choices=('archive', 'fixed_pr2', 'pr3'))
    parser.add_argument('--prior-review', type=Path)
    parser.add_argument('--retained-return', type=Path)
    parser.add_argument('--decision-index', type=int)
    parser.add_argument('--retained-method-pair', type=Path)
    parser.add_argument('--knowledge-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(register(args.source.resolve(), args.prior_review.resolve(), args.output.resolve(),
              seed=args.seed, arm=args.arm,
              decision_index=args.decision_index,
              knowledge_root=args.knowledge_root.resolve() if args.knowledge_root else None,
              retained_method_pair_ref=ref(args.retained_method_pair.resolve())
                  if args.retained_method_pair else None)
          if args.mode == 'prepare' else consume_saved_structural_return(
              args.output.resolve(), args.retained_return.resolve())
          if args.mode == 'consume-paid-return' else run(args.output.resolve()))
