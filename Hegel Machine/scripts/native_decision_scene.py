"""Restore paid inputs for a stage's first native choice, without later fit leakage.

The old run did not retain its complete pre-choice candidate pool. The caller
must reconstruct the common preparation once, charge it, and freeze that pool
before making a paired method comparison. This loader performs no research.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from scripts.accept_partial_role_structural_seed import _rows
from scripts.run_partial_role_parent import load
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import initial_prediction
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate


def recover_native_decision_scene(report, registration, model, *, public_spec,
                                  report_ref, registration_ref, model_ref, decision_index=None,
                                  reconstructed_parent=None):
    """Read the first choice of a paid-data stage, not an intermediate pool."""
    actions = report['actions']
    first_index, _ = next((i, action) for i, action in enumerate(actions)
        if action.get('action') == 'choose_research_action'
        and action.get('selected_action') in action.get('eligible_actions', [])
        and len(set(action['eligible_actions'])) > 1)
    choice_index = first_index if decision_index is None else decision_index
    choice = actions[choice_index]
    partial_index = next(i for i in range(choice_index - 1, -1, -1)
                         if actions[i].get('action') == 'compare_partial_object_claims')
    if (choice.get('action') != 'choose_research_action'
            or choice.get('selected_action') not in choice.get('eligible_actions', [])
            or len(set(choice.get('eligible_actions', []))) < 2
            or any(a.get('action') == 'choose_research_action'
                   for a in actions[partial_index + 1:choice_index])):
        raise ValueError('Only a stage first choice has a reconstructable complete initial pool')
    feedback_index = next(i for i in range(choice_index + 1, len(actions))
        if actions[i].get('action') == 'research_action_feedback')
    feedback = actions[feedback_index]
    if (feedback['research_action'] != choice['selected_action']
            or feedback['evidence_basis'] != 'common_unfitted_validation'
            or any(a.get('action') == 'acquire' for a in actions[choice_index:feedback_index])):
        raise ValueError('The first feedback does not bind the original choice evidence')
    acquired = {a['probe_id'] for a in actions[:choice_index] if a.get('action') == 'acquire'}
    feature_names = set(public_spec.feature_names)
    batches = [(i, batch) for i, batch in enumerate(report['batches'])
        if batch['probe']['probe_id'] in acquired and batch.get('observations')
        and set(batch['observations'][0]['features']) == feature_names]
    if len(batches) < 2:
        raise ValueError('A distinct original fitting and comparison trajectory is required')
    training_batches, (validation_index, validation_batch) = batches[:-1], batches[-1]
    training = [row for _, batch in training_batches for row in _rows(batch['observations'])]
    validation = _rows(validation_batch['observations'])
    if (len(training) != feedback['fitting_rows'] or len(validation) != feedback['evidence_rows']
            or any(row in training for row in validation)):
        raise ValueError('Original trajectory cutoffs do not match the paid feedback')
    adoption = next(a for a in actions[:choice_index] if a.get('action') == 'adopt_discovered_objects')
    role_rows = adoption.get('rows', model['support_rows'])
    if (model['model_id'] != adoption['object_model_id'] or model['groups'] != adoption['groups']
            or model['support_rows'] != role_rows or role_rows > len(training)
            or model['acquisition_rows'] != role_rows):
        raise ValueError('The role model is not the model available before this choice')

    # The fixed parent is retained by a later dependent question in the SAME
    # view. Its original AST also occurs in the earlier mechanism revision;
    # these are archival carriers of an unchanged input, not later fitted wins.
    view_end = next(i for i in range(choice_index + 1, len(actions))
                    if actions[i].get('action') == 'compare_research_directions')
    if choice_index == first_index:
        dependent_index = next((i for i in range(choice_index + 1, view_end)
            if actions[i].get('action') == 'execute_dependent_research'), None)
        if dependent_index is not None:
            dependent = actions[dependent_index]
            original_parent = dependent['parent_program']
            parent_id = dependent['parent_candidate_id']
            parent_carrier_path = ['actions', dependent_index, 'parent_program']
            carrier_stop = dependent_index
        else:
            if reconstructed_parent is None:
                raise ValueError('The original command parent must be rebound before recovery')
            original_parent = reconstructed_parent['candidate'].program.to_dict()
            parent_id = reconstructed_parent['candidate'].candidate_id
            parent_carrier_path = None
            carrier_stop = view_end
        parent_kind = 'retained_response'
    else:
        confirmation_index = next(i for i in range(partial_index - 1, -1, -1)
            if actions[i].get('action') == 'validate_frozen_program')
        settled_index = next(i for i in range(confirmation_index + 1, choice_index)
            if actions[i].get('action') == 'object_research_feedback')
        original_parent = actions[confirmation_index]['program']
        if actions[settled_index]['program'] != original_parent:
            raise ValueError('The prior confirmation does not bind the next parent')
        previous_view = next(i for i in range(confirmation_index - 1, -1, -1)
            if actions[i].get('action') == 'compare_research_directions')
        parent_kind = actions[previous_view]['selected_kind']
        parent_id = f'restored:before_choice_{choice_index}'
        parent_carrier_path = ['actions', confirmation_index, 'program']
        carrier_stop = view_end
    carriers = [(i, j) for i, action in enumerate(actions[:carrier_stop])
        for j, candidate in enumerate(action.get('alternatives', []))
        if (candidate.get('revision') or {}).get('original_program') == original_parent]
    if not carriers:
        raise ValueError('The bound parent lacks its unchanged original-program carrier')
    if parent_carrier_path is None:
        ci, cj = carriers[0]
        parent_carrier_path = ['actions', ci, 'alternatives', cj, 'revision', 'original_program']
    parent_program = Program.from_dict(original_parent)
    if not parent_program.required_features() <= feature_names:
        raise ValueError('The archived bound parent is not readable in this instrument')
    partial_scores = actions[partial_index]['alternatives']
    # q=.5 has affine, the partial claims, then the carried parent. A parent
    # may have the same kind as a newly refitted partial claim, but not its score.
    parent_index = 1 + len(partial_scores)
    parent_score_row = actions[view_end]['alternatives'][parent_index]
    if parent_score_row['kind'] != parent_kind:
        raise ValueError('Original prelude order does not identify the retained parent')
    parent_score = parent_score_row['validation_loss']
    affine_score = next(a['validation_loss'] for a in actions[view_end]['alternatives']
                        if a['kind'] == 'current_affine')
    parent = {'candidate': Candidate(parent_id, parent_program, 'retained_response'),
              'kind': parent_kind, 'family_id': 'T03', 'validation_loss': parent_score}
    parent_metadata_refs = []
    if choice_index != first_index:
        graph_i, graph_j = next((i, j) for i, j in carriers
            if actions[i]['alternatives'][j]['revision'].get('original_research_program'))
        graph = deepcopy(actions[graph_i]['alternatives'][graph_j]['revision']['original_research_program'])
        if graph['graph']['nodes'][graph['output_node']]['answer'] != original_parent:
            raise ValueError('The saved original research graph does not produce the bound parent')
        parent['research_program'] = graph
        parent['training_loss'] = graph['graph']['nodes'][graph['output_node']]['loss']
        parent_metadata_refs.append(['actions', graph_i, 'alternatives', graph_j,
                                     'revision', 'original_research_program'])
        if parent_kind.startswith('partial_object:'):
            fi, ei, coverage = next((fi, ei, entry['partial_object_knowledge'])
                for fi, frontier in enumerate(report['object_continuation_state']['candidate_frontiers'])
                for ei, entry in enumerate(frontier['entries'])
                if entry.get('kind') == parent_kind
                and entry.get('partial_object_knowledge', {}).get('source_model_id') == model['model_id']
                and set(entry['partial_object_knowledge']['used_fields']) == parent_program.required_features())
            # This is the original object's immutable role/field coverage,
            # not this later entry's fitted coefficients or current scores.
            parent['partial_object_knowledge'] = deepcopy(coverage)
            parent_metadata_refs.append(['object_continuation_state', 'candidate_frontiers', fi,
                                         'entries', ei, 'partial_object_knowledge'])
    # _research_objects stores the agent's measured view before adding the
    # membership witnesses to its library copy. Both derive from these same
    # paid rows; retain that original distinction instead of loading final state.
    measured_model = deepcopy(model)
    measured_model.pop('membership_witnesses', None)
    spec = replace(public_spec, context={**public_spec.context,
        'object_schema': json.dumps(model['schema']), 'object_model': measured_model})
    i, j = carriers[0]
    reference = lambda path: {**deepcopy(report_ref), 'json_path': path}
    return {'spec': spec, 'training': training, 'validation': validation,
        'transitions': [deepcopy(t) for _, batch in training_batches
                        for t in batch.get('transitions', [])],
        'parent': parent, 'object_model': deepcopy(model),
        'fallback': initial_prediction(training, spec.risk_quantile),
        'choice_index': choice_index, 'choice': deepcopy(choice),
        'ordinary_operation': choice['selected_action'],
        'fallback_operation': choice.get('fallback_action'),
        'remaining_compute': choice['compute_remaining'],
        'observation_remaining': choice['observation_remaining'],
        'measurement_remaining': choice.get('measurement_remaining'),
        'historical_knowledge_ref': deepcopy(registration['historical_knowledge_ref']),
        'research_config': deepcopy(registration['research_config']),
        'profile': {'evidence_rows': len(validation), 'fitting_rows': len(training),
            'selected_operation': choice['selected_action'],
            'eligible_operations': list(choice['eligible_actions']),
            'context': choice['context']},
        'expected_preparation': {
            'partial_claims': deepcopy(partial_scores),
            'affine_validation_loss': affine_score,
            'retained_response_validation_loss': parent_score,
            'retained_response_index': parent_index,
            'retained_response_kind': parent_kind,
            'scope': 'saved same-view scores for checking reconstruction, not a method preference'},
        'references': {'report': deepcopy(report_ref), 'registration': deepcopy(registration_ref),
            'choice': reference(['actions', choice_index]),
            'training': [reference(['batches', i]) for i, _ in training_batches],
            'validation': reference(['batches', validation_index]),
            'object_model': deepcopy(model_ref),
            'parent_original': reference(['actions', i, 'alternatives', j, 'revision', 'original_program']),
            'parent_carrier': reference(parent_carrier_path),
            'parent_metadata': [reference(path) for path in parent_metadata_refs],
            'preparation_partial_scores': reference(['actions', partial_index, 'alternatives']),
            'original_scores': reference(['actions', view_end, 'alternatives'])},
        'complete_historical_pool_available': False,
        'preparation_required': 'reconstruct_common_pool_once_before_any_selected_direction',
        'preparation_cost_scope': 'new_shared_preparation_not_free_historical_cache',
        'excluded_later_state': ['candidate_frontiers', 'finalists', 'selected',
                                 'research_drafts', 'terminal_confirmation'],
        'restoration_scope': 'paid_inputs_and_unchanged_parent_not_an_exact_prechoice_checkpoint'}


def rebound_original_command_parent(knowledge, research_config, model, laboratory):
    """Repeat only the original role binding, never a fitted target revision."""
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.relation_input_diagnosis import measured_relation_entry, bind_relation_entry
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    agent = UnifiedResearchAgent(ResearchConfig(**research_config),
        library=KnowledgeLibrary.from_dict(deepcopy(knowledge)))
    agent._learn_instruments = agent.config.strategy != 'fixed'
    agent._revise_inputs = agent.config.use_memory and agent._learn_instruments
    agent._object_models[('fine',)] = deepcopy(model)
    agent._object_models[('fine',)].pop('membership_witnesses', None)
    agent._object_schemas[('fine',)] = deepcopy(model['schema'])
    calls = agent._recall(agent._spec(laboratory, ()))
    if not calls:
        raise ValueError('The original library has no command parent to restore')
    entry = {'candidate': Candidate('retained:object_recovery', calls[0], 'retained_response'),
        'kind': 'retained_response', 'family_id': 'T03', 'validation_loss': None}
    spec = agent._spec(laboratory, ('fine',))
    prepared = measured_relation_entry(entry, spec, agent.library)
    if prepared is None:
        raise ValueError('The original command parent has no measured input contract')
    bound = bind_relation_entry(prepared, {name: Program('feature', name=name)
                                         for name in spec.feature_names})
    if bound is None:
        raise ValueError('The original parent requires unavailable measured inputs')
    return bound


def load_native_decision_scene(root, *, seed=106029101, arm='pr3', decision_index=None,
                               knowledge_root=None):
    from scripts.run_partial_role_successors import historical_knowledge

    root = Path(root)
    path = root / 'runs' / str(seed) / arm
    report_ref = {'file': str(path / 'response_report.json')}
    registration_ref = {'file': str(root / 'registration.json')}
    model_ref = {'file': str(path / 'common/object_question_000/parent_result.json')}
    report, registration, model = (load(Path(ref['file']))
        for ref in (report_ref, registration_ref, model_ref))
    settings = next(item for item in registration['tasks'] if item['seed'] == seed)
    # The original run constructs config(seed) per task; the registration's
    # common config example carries the first task's seed only.
    registration = deepcopy(registration)
    registration['research_config']['seed'] = settings['seed']
    knowledge, costs, sequence = historical_knowledge(registration['historical_knowledge_ref'])
    knowledge = deepcopy(knowledge)
    models = knowledge.setdefault('object_models', [])
    if any(item.get('model_id') == model['model_id'] for item in models):
        raise ValueError('The original K0 already contains this target object identity')
    models.append(deepcopy(model))
    actions = report['actions']
    first_index = next(i for i, action in enumerate(actions)
        if action.get('action') == 'choose_research_action'
        and action.get('selected_action') in action.get('eligible_actions', [])
        and len(set(action['eligible_actions'])) > 1)
    index = first_index if decision_index is None else decision_index
    end = next(i for i in range(index + 1, len(actions))
               if actions[i].get('action') == 'compare_research_directions')
    needs_binding = index == first_index and not any(
        a.get('action') == 'execute_dependent_research' for a in actions[index + 1:end])
    # Only the laboratory's public interface is read. No world response,
    # hidden mechanism parameter, purchase, fit or prediction enters recovery.
    with AnonymousObjectLab(**settings) as laboratory:
        public_spec = laboratory.spec_for(('fine',))
        reconstructed_parent = rebound_original_command_parent(knowledge,
            registration['research_config'], model, laboratory) if needs_binding else None
    scene = recover_native_decision_scene(report, registration, model,
        public_spec=public_spec, report_ref=report_ref,
        registration_ref=registration_ref, model_ref=model_ref, decision_index=decision_index,
        reconstructed_parent=reconstructed_parent)
    # The ordinary author appended this measured object at the recorded
    # adoption before the first response choice. No final draft or refutation is used.
    if needs_binding:
        scene['parent_reconstruction'] = {
            'entry': 'ObjectResearchActions._structure_view',
            'operations': ['recall_original_command_parent', 'measured_relation_entry', 'bind_relation_entry'],
            'knowledge_ref': deepcopy(scene['historical_knowledge_ref']),
            'object_model_ref': deepcopy(model_ref),
            'bound_original_ref': deepcopy(scene['references']['parent_original']),
            'new_fits': 0, 'new_predictions': 0}
    # These snapshots were recorded at each actual return. Concept/definition
    # changes and target refutations occur only later in this source run.
    # Replay the already-recorded policy values, not any scientific operation.
    policy_refs = []
    for i, action in enumerate(report['actions'][:scene['choice_index']]):
        if action.get('action') == 'research_action_feedback':
            operation, feedback = action['research_action'], action.get('feedback')
        elif action.get('action') == 'object_research_feedback':
            operation, feedback = action['direction'], action.get('feedback')
        elif action.get('action') == 'propose_object_research':
            operation, feedback = action['direction'], action.get('unscored_feedback')
        else:
            continue
        if isinstance(feedback, dict) and action.get('context'):
            knowledge.setdefault('action_experience', {}).setdefault(action['context'], {})[operation] = deepcopy(feedback)
            policy_refs.append({**report_ref, 'json_path': ['actions', i]})
    scene.update(knowledge=knowledge, historical_costs=costs,
                 historical_sequence=sequence,
                 knowledge_additions=[deepcopy(model_ref), *policy_refs],
                 action_experience_restored_from=policy_refs,
                 source_task_ref={**registration_ref,
                    'json_path': ['tasks', registration['tasks'].index(settings)]})
    if knowledge_root is not None:
        from scripts.successor_interval_scene import load_confirmed_knowledge

        frozen = deepcopy(load_confirmed_knowledge(Path(knowledge_root)).to_dict())
        current = deepcopy(scene['object_model'])
        original_id = current['model_id']
        current['model_id'] = f"objects_{len(frozen['object_models']) + 1:04d}"
        frozen['object_models'].append(deepcopy(current))
        measured = deepcopy(scene['spec'].context['object_model'])
        measured['model_id'] = current['model_id']
        scene['spec'] = replace(scene['spec'], context={**scene['spec'].context,
                                                      'object_model': measured})
        coverage = scene['parent'].get('partial_object_knowledge')
        if coverage and coverage.get('source_model_id') == original_id:
            coverage['source_model_id'] = current['model_id']
        scene.update(knowledge=frozen, object_model=current,
            knowledge_base_ref={'file': str(Path(knowledge_root) / 'knowledge_delta.json')},
            knowledge_source_scope='retrospective_development_with_frozen_later_knowledge',
            knowledge_additions=[deepcopy(model_ref)], action_experience_restored_from=[],
            historical_action_experience_refs=policy_refs,
            object_identity_remap={'original_model_id': original_id,
                'current_model_id': current['model_id'], 'source_ref': deepcopy(model_ref),
                'reason': 'retain_target_object_in_frozen_later_library'})
    return scene
