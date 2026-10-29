"""Restore a registered source comparison through the existing Common agent.

Material loading is separate from restoration so both arms share one batched
read of the development Programs. No research action is executed here.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from .anonymous_research_archive import resolve_workspace_archive
from .checkpoint_storage import read_json_checkpoint, write_json_checkpoint
from .code_agent import archived_paid_trace_references
from .question_routing import register_theories, resolved_theories
from .research_episodes import load_source_values
from .theory_consolidation import (
    LIBRARY_PATH, _read_library, _save as save_theory_library,
    save_theory_index,
)


def _read(reference, workspace):
    return load_source_values([reference], workspace)[0]['value']


def load_retained_common_checkpoint(source):
    """Reuse the admitted case002 projection, not its portable file archive.

    This is the selected-field restore used by prepare_case_scene in the closed
    paid-analysis run. Its one state tree passes directly to Common; the old
    index and retained score graph are never deep-copied for an extra material
    view. Call separately for each mutable branch.
    """
    from contextlib import ExitStack
    import os
    from .checkpoint_storage import _JsonSource
    from .code_agent import _model_context_ignored
    from .common_archive_inquiry import retained_checkpoint_result

    source = Path(source).resolve()
    fields = ('costs', 'paid_observations', 'events', 'model', 'online_questions',
        'revision_proposals', 'inherited_files', 'method_results', 'program_contexts',
        'experiment_comparisons', 'observable_models', 'conditional_models', 'model_dependencies',
        'research_candidates', 'research_input_bindings', 'relation_research', 'theory_index',
        'research_mission', 'research_agenda', 'theory_consolidation', 'research_organization',
        'llm_inquiries', 'inquiry_stack', 'evaluated_model', 'fresh_confirmation', 'method_path')
    previous = read_json_checkpoint(source / 'training.json', keys=fields)
    with ExitStack() as opened:
        saved = _JsonSource(source / 'training.json', opened)
        outgoing = saved.members(keys={'outgoing_knowledge'})['outgoing_knowledge']
        manifest = saved.members(outgoing, keys={'manifest'})['manifest']
        manifest = json.loads(saved.text[manifest[1]:manifest[2]])
    names = set(previous['inherited_files'])
    for directory, subdirs, files in os.walk(source):
        base = Path(directory)
        subdirs[:] = [name for name in subdirs if name not in {'_evaluated_model', '_submitted_models'}
            and not _model_context_ignored(base / name, source)]
        names.update(str((base / name).relative_to(source)) for name in files
            if not _model_context_ignored(base / name, source))
    previous['outgoing_knowledge'] = {'manifest': manifest, 'files': dict.fromkeys(sorted(names))}
    thin = any(isinstance(question.get('native_condition_research'), dict)
               for question in previous['online_questions'])
    state = read_json_checkpoint(source / 'research_state.json', keys=[
        'operator_results', 'exported_candidate_paths', 'online_research', 'costs',
        *(['questions'] if thin else [])])
    canonical = {question['id']: question.get('native_condition_research')
                 for question in state.get('questions', [])}
    for question in previous['online_questions']:
        rounds = question.get('native_condition_research')
        if isinstance(rounds, dict) and rounds.get('full_content_file') == 'research_state.json':
            question['native_condition_research'] = canonical[question['id']]
    state['questions'], state['revisions'] = previous['online_questions'], previous['revision_proposals']
    for field in ('costs', 'evaluated_model', 'method_results', 'program_contexts',
            'experiment_comparisons', 'observable_models', 'conditional_models', 'model_dependencies',
            'research_candidates', 'research_input_bindings', 'relation_research', 'theory_index',
            'research_mission', 'research_agenda', 'theory_consolidation', 'research_organization'):
        if field in previous:
            state[field] = previous[field]
    return retained_checkpoint_result(previous, state), state


def preserve_completed_pool_order(agent, models):
    """Keep a paid complete comparison's order when restoration found its pool.

    Reconstructing the same models from exports and native rounds can change
    their insertion order. Preserve the actual paid execution rather than
    asking the strict ordinary comparison consumer to score that pool again.
    """
    expected = set(models)
    for report in reversed(agent.operator_results):
        if (report.get('operator') != 'research_choice'
                or report.get('paid_rows') != len(agent.rows)
                or report.get('error') or report.get('truncated')
                or report.get('models_requested') != len(models)
                or report.get('models_completed') != len(models)):
            continue
        order = report.get('arguments', {}).get('models', [])
        evaluations = report.get('evaluations', [])
        if (len(order) == len(models) == len(set(order))
                and set(order) == expected
                and [entry.get('path') for entry in evaluations] == order
                and not any(entry.get('truncated') for entry in evaluations)):
            return list(order)
    return models


def restore_retained_common_scene(source, workspace, *, plan, case, knowledge, limits,
                                  starting_costs):
    """Restore one exact paid Common scene through the existing current factory."""
    from scripts import run_anonymous_revision_inquiry as study
    from scripts.prepare_pr3_source_successor import _prepared_callbacks, _retain_scene_score_index

    source, workspace = Path(source).resolve(), Path(workspace).resolve()
    if workspace.exists():
        raise ValueError('restore into a fresh workspace and preserve the paid source')
    previous, state = load_retained_common_checkpoint(source)
    if (any(starting_costs[key] < value for key, value in previous['costs'].items())
            or any(starting_costs[key] != previous['costs'][key] for key in
                   ('observation_cost', 'measurement_cost', 'model_calls'))):
        raise ValueError('the authoritative account changed paid acquisitions or lost old costs')
    previous['costs'] = state['costs'] = dict(starting_costs)
    knowledge['finite_native_author']['question'] = deepcopy(plan['prepared_public_questions'][case['id']])
    factory, _ = _prepared_callbacks(plan, case_id=case['id'])
    paid = archived_paid_trace_references(knowledge.get('files', {}),
        workspace.parent / 'reference_archive', source_workspace=source)
    agent = factory(None, workspace, arm='pr3',
        research_organization=deepcopy(plan['research_organization']), knowledge=knowledge,
        config=limits, confirmation_seed=case['confirmation_seed'], source_task=False,
        paid_trace_references=paid)
    lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
    restored = agent.restore_research(previous, source, lab,
        confirmation_seed=case['confirmation_seed'], retained_state=state)
    if agent.costs != starting_costs or agent.observations != previous['paid_observations']:
        raise ValueError('restoration changed the actual paid evidence or account')
    agent._organization_state['automatic_library_path'] = LIBRARY_PATH
    agent._organization_registration['pr3_library_path'] = str(workspace / LIBRARY_PATH)
    agent.knowledge['theory_consolidation'] = {'file': str(workspace / LIBRARY_PATH), 'json_path': []}
    agent.knowledge['finite_native_author']['question']['frozen_library_path'] = LIBRARY_PATH
    for question in agent.questions:
        if ('frozen_library_path' in question or 'frozen_recall_execution' in question
                or question.get('theory_target', {}).get('revision_fitting')):
            question['frozen_library_path'] = LIBRARY_PATH
    score_index = _retain_scene_score_index(source, workspace, paid_rows=len(agent.rows))
    models = list(agent.knowledge['finite_native_author']['predictor_paths'])
    from .finite_native_author import _comparison_models
    rounds = [round_ for question in agent.questions for round_ in question.get('native_condition_research', [])]
    models = _comparison_models(models, agent.candidate_paths, rounds)
    from .python_research_routing import validated_candidates
    models = list(dict.fromkeys([*models, *validated_candidates(agent)]))
    models = preserve_completed_pool_order(agent, models)
    agent.knowledge['finite_native_author']['predictor_paths'] = models
    save_theory_library(agent)
    return {'agent': agent, 'lab': lab, 'models': models, 'source': source,
        'starting_costs': dict(starting_costs), 'validation': {**restored,
            'paid_rows': len(agent.rows), 'complete_model_count': len(models),
            'theory_count': len(agent.theory_index), 'retained_score_index': score_index,
            'finite_author_cursor': deepcopy(agent._organization_state['finite_author']),
            'new_fits': 0, 'new_scores': 0, 'new_predictions': 0, 'new_observations': 0}}


def admit_exposed_representation_evidence(scene, material):
    """Reuse a finished frozen readout as development evidence for its parent.

    This imports paid reset trajectories, not new laboratory measurements or a
    replacement full-data score. The original partial diagnostic keeps its own
    row scope and source; the retained parent and score index remain untouched.
    """
    from .finite_native_author import _retained_representation_parent
    from .method_action_research import read_method_reference
    from .models import Observation
    from .temporal_operator_agent import TemporalOperatorAgent
    from types import MethodType, SimpleNamespace

    agent = scene['agent']
    research_plan = {key: deepcopy(material[key]) for key in
        ('validation_records', 'compute_limit', 'max_candidates')}
    research_plan.update({key: deepcopy(material[key]) for key in
        ('retained_research_program_refs', 'retained_replay_scope') if key in material})
    retained = agent._organization_state.setdefault('exposed_representation_evidence', [])
    prior = next((entry for entry in retained if entry['panel_ref'] == material['panel_ref']
                  and entry['evaluation_ref'] == material['evaluation_ref']), None)
    if prior is not None:
        if prior['research_plan'] != research_plan:
            raise ValueError('the admitted development material has a different registered research plan')
        agent._active_question_id = prior['question_ref']['id']
        return {**deepcopy(prior), 'cache_hit': True}
    panel, evaluated = [item['value'] for item in load_source_values(
        [material['panel_ref'], material['evaluation_ref']], agent.workspace)]
    evaluation = evaluated['evaluation']
    diagnostic = evaluation.get('representation_diagnostic', {})
    if (not panel.get('all_predictors_frozen_before_labels')
            or evaluation.get('returncode') != 0
            or diagnostic.get('evidence_state') != 'proven_insufficient_on_evidence'
            or diagnostic.get('rows') != len(panel['rows'])):
        raise ValueError('reuse a completed frozen readout with its actual scoped conflict')
    model = agent.evaluated_model['source_model']
    frozen = evaluated['frozen_model']
    if (agent.workspace / model).read_bytes() != (Path(frozen['workspace']) / frozen['model']).read_bytes():
        raise ValueError('the exposed diagnostic belongs to a different actual parent')
    owner = agent.knowledge['finite_native_author']['question']
    parent = _retained_representation_parent(
        {'questions': agent.questions, 'theory_index': agent.theory_index},
        agent.method_results, model, owner['theory_target'])
    if parent is None:
        raise ValueError('retain the actual parent Program address before representation research')
    parents = [parent]
    for source_ref in material.get('retained_research_program_refs', []):
        candidate = read_method_reference(source_ref, agent.workspace)
        if not candidate.get('program') or not candidate.get('research_program') or not candidate.get('model'):
            raise ValueError('retained refits require their actual saved Program, procedure and model')
        identifier = f'retained_structure_parent:{candidate["model"]}'
        if any(entry['program'] == candidate['program'] for entry in parents):
            continue
        reference = {**deepcopy(source_ref),
            'json_path': [*source_ref.get('json_path', []), 'program']}
        parents.append({**{key: deepcopy(parent[key]) for key in
            ('role_signature', 'output_unit', 'context_ref', 'support')},
            'id': identifier, 'program': deepcopy(candidate['program']), 'program_source_ref': reference,
            'source_snapshot': {'model': candidate['model'], 'source_ref': reference}})
    # Split only at real resets and retain every full public history. Appending
    # a terminal fragment to the previous paid trajectory would change its use.
    records, history = [], []
    row_start, record_start = len(agent.rows), len(agent.observations)
    for offset, row in enumerate(panel['rows']):
        if not row.get('history'):
            history = []
            records.append({'observation_index': row_start + offset,
                'action': {'kind': 'trajectory', 'sequence': []}, 'rows': [], 'row_indices': [],
                'origin': 'exposed_evaluation_development_not_purchase',
                'source_ref': deepcopy(material['panel_ref']), 'observation_cost': 0,
                'measurement_cost': 0})
        if not records or row.get('history', []) != history:
            raise ValueError('development readout must preserve complete reset trajectories')
        records[-1]['rows'].append(deepcopy(row))
        records[-1]['row_indices'].append(row_start + offset)
        records[-1]['action']['sequence'].append(deepcopy(row['features']))
        history.append(deepcopy(row['features']))
    prospective = SimpleNamespace(observations=[*agent.observations, *records])
    prospective._record_indices = MethodType(TemporalOperatorAgent._record_indices, prospective)
    _, _, split = TemporalOperatorAgent._split(prospective,
        {'validation_records': material['validation_records']})
    if set(split['validation_records']).intersection(range(record_start, len(prospective.observations))):
        raise ValueError('the counterexamples used to propose representations are training development evidence')
    agent.observations.extend(records)
    agent.rows.extend(Observation(dict(row['features']), float(row['target']),
        tuple(dict(point) for point in row.get('history', []))) for row in panel['rows'])
    register_theories(agent, parents)
    question = agent._open_question({'id': f'representation:{model}',
        'question': 'Which public distinctions did this actual retained predictor lose?',
        'kind': 'representation', 'origin': 'computed_representation_conflict',
        'parent_id': owner['id'], 'depends_on': [], 'answer_type': 'program',
        'subject_refs': [{'type': 'candidate', 'path': model}],
        'context_ref': deepcopy(owner.get('context_ref')),
        'query_semantics': {'mode': 'solution_relation'},
        'answer_resolution_ref': {'evaluator': 'representation_diagnostic_and_paid_comparison',
                                  'parent_model': model},
        'representation_diagnostic': deepcopy(diagnostic),
        'representation_evidence_ref': {**deepcopy(material['evaluation_ref']),
            'json_path': [*material['evaluation_ref'].get('json_path', []),
                          'evaluation', 'representation_diagnostic']},
        'use_sites': [{'kind': 'supply_component', 'model': model, 'slot': 'representation'}],
        'native_representation_plan': {**research_plan, 'parent_theory_ids': [item['id'] for item in parents],
            'validation_records': split['validation_records']}})
    question['status'] = 'open'
    agent._active_question_id = question['id']
    receipt = {'status': 'exposed_evaluation_admitted_as_development',
        'panel_ref': deepcopy(material['panel_ref']), 'evaluation_ref': deepcopy(material['evaluation_ref']),
        'source_costs_ref': {**deepcopy(material['panel_ref']),
            'json_path': [*material['panel_ref'].get('json_path', []), 'costs_once']},
        'original_parent_model': model, 'parent_theory_id': parent['id'],
        'parent_theory_ids': [item['id'] for item in parents], 'research_plan': research_plan,
        'question_ref': {'id': question['id'], 'meaning_version': question.get('meaning_version', 1)},
        'appended_records': list(range(record_start, len(agent.observations))),
        'appended_rows': list(range(row_start, len(agent.rows))),
        'split': split, 'new_observations': 0, 'new_fits': 0, 'new_scores': 0,
        'independent_support': False, 'cache_hit': False}
    if material.get('parent_assessment_ref') is not None:
        from .paid_score_history import retain_frozen_readout_score
        receipt['paid_readout_reuse'] = retain_frozen_readout_score(agent,
            model=model, parent_assessment_ref=material['parent_assessment_ref'],
            panel_ref=material['panel_ref'], evaluation_ref=material['evaluation_ref'],
            public_spec=scene['lab'].public_spec(), question_id=owner['id'])
    question['development_material_ref'] = {'file': 'research_state.json', 'json_path': [
        'research_organization', 'exposed_representation_evidence', len(retained)]}
    retained.append(deepcopy(receipt))
    return receipt


def prepare_registered_parent_revision(scene, registration):
    """Bind the registered parent use without rewriting its answered Readout."""
    from .models import Observation
    from .question_continuation import parent_revision_obligation
    from .question_routing import _resolved_theory_versions
    from .registered_source_budget import retained_grammar_preparation
    from .theory_consolidation import _resolve_paid_checks, _retained_parent_contract

    contract = registration['parent_research']
    agent = scene['agent']
    source = next(q for q in agent.questions if q['id'] == contract['source_question_id'])
    original = deepcopy(source)
    subject = contract['parent_theory_ref']
    parent = deepcopy(agent.theory_index[subject['id']])
    if parent.get('version', 1) != subject['version']:
        raise ValueError('registered parent version changed')
    parent = _retained_parent_contract(agent, parent)
    if parent.get('_contract_source', {}).get('reason'):
        raise ValueError('registered parent typed contract is unresolved: '
                         + parent['_contract_source']['reason'])
    fields = ('evidence_checks_ref', 'fitting_ref', 'component_roster_ref',
              'grammar_preparation_ref', 'grammar_context_ref')
    values = [item['value'] for item in load_source_values(
        [contract[field] for field in fields], agent.workspace)]
    raw_checks, fitting, roster, grammar_state, source_request = values
    checks = _resolve_paid_checks(agent, raw_checks)
    rows = [Observation(item['observation']['features'], item['observation']['target'],
        tuple(item['observation'].get('history', []))) for item in checks
        if 'observation' in item and not item.get('resolution_reason')]
    if rows != agent.rows or any(item.get('criterion', {}).get('kind') != 'paid_mse'
            or item['criterion'].get('evaluated_rows') != len(agent.rows)
            or item['criterion'].get('mse_threshold') != agent.config.confirmation_mse_threshold
            for item in checks):
        raise ValueError('registered parent revision must retain the complete original paid-row contract')
    versions = {(item['id'], item.get('version', 1)): item
                for item, _, _ in _resolved_theory_versions(agent)}
    components = []
    for reference in roster:
        item = versions[(reference['id'], reference.get('version', 1))]
        if not item.get('program') or item.get('source_ref') != reference.get('source_ref'):
            raise ValueError('registered structural component version or source changed')
        components.append(deepcopy(item))
    grammar = retained_grammar_preparation(source_request, grammar_state,
        components=components, evidence_checks=checks, fitting=fitting,
        source_ref=contract['grammar_preparation_ref'])
    target = {key: deepcopy(parent[key]) for key in
              ('role_signature', 'output_unit', 'context_ref')}
    target.update(parent_theory_id=parent['id'], revision_fitting=deepcopy(fitting),
                  evidence_checks=deepcopy(raw_checks), paid_evidence_scope={
                      'records': len(agent.observations), 'rows': len(agent.rows)})
    if parent.get('context_family_ref') is not None:
        target['context_family_ref'] = deepcopy(parent['context_family_ref'])
    intention = {**deepcopy(source), 'theory_target': target}
    obligation = parent_revision_obligation(agent, intention, {'parent_revision_completed': False})
    if obligation is None:
        raise ValueError('registered parent has no unresolved paid revision obligation')
    actual_ref = {'question_id': obligation['id'],
                  'meaning_version': obligation.get('meaning_version', 1)}
    if actual_ref != {'question_id': contract['question_id'], 'meaning_version': contract['meaning_version']}:
        raise ValueError('registered parent revision meaning differs from the actual obligation')
    obligation['paid_score_question_ref'] = {'question_id': source['id'],
                                            'meaning_version': source.get('meaning_version', 1)}
    obligation['frozen_library_path'] = LIBRARY_PATH
    retained = {'_retained_parent': parent, '_retained_components': components,
                'component_theory_ids': [item['id'] for item in components]}
    inputs = {**retained, '_retained_execution_inputs':
        _registered_retained_parent_inputs(scene, retained, actual_ref),
        '_grammar_preparation': grammar, 'library_path': LIBRARY_PATH,
        '_retained_fit_constructors': scene['retained_fit_constructors'],
        'parent_theory_id': parent['id'], 'fitting': deepcopy(fitting),
        'evidence_checks': deepcopy(raw_checks)}
    order_receipt = _reconcile_registered_parent_order(scene, registration, obligation, actual_ref)
    if source != original:
        raise ValueError('registered parent setup changed the original answered Readout')
    prefix = f'relation_transfer:frozen_recall:{obligation["id"]}@{actual_ref["meaning_version"]}:'
    receipt = {'status': 'registered_parent_revision_bound_without_scientific_execution',
        'research_question_ref': actual_ref, 'parent_theory_ref': deepcopy(subject),
        'paid_score_question_ref': deepcopy(obligation['paid_score_question_ref']),
        'paid_rows': len(rows), 'ordered_structural_components': len(components),
        'full_strong_models': len(scene['models']),
        'original_Readout_unchanged': True, 'original_parent_work_retained': True,
        'current_parent_scope_has_prior_executions': any(
            key.startswith(prefix) for key in agent.theory_consolidation.get('executions', {})),
        'grammar_preparation_ref': deepcopy(contract['grammar_preparation_ref']),
        'historical_fit_library_ref': deepcopy(scene['historical_fit_library_ref']),
        'historical_fit_constructor_count': len(scene['retained_fit_constructors']),
        'old_layer_offset_and_pending_imported': False,
        'retained_execution_groups': scene.get('registered_parent_input_groups', []),
        'fair_order_reconciliation': order_receipt,
        'new_fits': 0, 'new_predictions': 0, 'new_scores': 0, 'paid_calls': 0}
    scene.update(parent=parent, retained_components=components,
        source_component_theory_ids=retained['component_theory_ids'],
        registered_revision_inputs=inputs, research_question_ref=actual_ref,
        parent_revision_acceptance=receipt)
    write_json_checkpoint(agent.workspace / 'parent_revision_acceptance.json', receipt, ensure_ascii=False)
    return receipt


def _same_restored_value(left, right, left_workspace, right_workspace):
    """Compare scientific values, allowing the actual local reference rebase."""
    if left == right:
        return True
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        for key in left:
            a, b = left[key], right[key]
            if key == 'file' and ('json_path' in left or 'field' in left):
                a_prefix, b_prefix = str(left_workspace) + '/', str(right_workspace) + '/'
                if isinstance(a, str) and isinstance(b, str) and a.startswith(a_prefix) and b.startswith(b_prefix):
                    if a[len(a_prefix):] != b[len(b_prefix):]:
                        return False
                    continue
            if key == 'workspace' and a == str(left_workspace) and b == str(right_workspace):
                continue
            if not _same_restored_value(a, b, left_workspace, right_workspace):
                return False
        return True
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_same_restored_value(a, b, left_workspace, right_workspace) for a, b in zip(left, right))
    return False


def _registered_retained_parent_inputs(scene, retained, question_ref):
    """Keep the verified original constructor address of each own195 cursor."""
    agent, parent = scene['agent'], retained['_retained_parent']
    prefix = (f'relation_transfer:frozen_recall:{question_ref["question_id"]}'
              f'@{question_ref["meaning_version"]}:')
    inputs, parents = {}, []
    for identifier in agent.theory_consolidation['candidates']:
        inputs[identifier] = retained
        previous = agent.theory_consolidation['executions'].get(prefix + identifier)
        if previous is None:
            continue
        original = previous['request_basis']['parent']
        if (original['id'] != parent['id']
                or original.get('version', 1) != parent.get('version', 1)):
            raise ValueError('registered parent cursor belongs to a different parent version')
        current = parent.get('_contract_source', {})
        source_ref = original.get('contract_source')
        if source_ref == current.get('source_ref'):
            continue
        from .theory_consolidation import _reference_workspace
        source_workspace = _reference_workspace(source_ref['program_ref'])
        if not _same_restored_value(source_ref, current.get('source_ref'),
                                    source_workspace, agent.workspace):
            raise ValueError('registered parent constructor differs beyond local relocation')
        # Full restoration already loaded this exact version and transfer. Read
        # their existing RAM values rather than decode either source again.
        saved_program = scene['previous']['theory_index'][parent['id']]['program']
        if saved_program != parent['program']:
            raise ValueError('relocated parent Program differs from its saved version')
        bound_ref = source_ref['components_ref']
        if Path(bound_ref['file']) == source_workspace / LIBRARY_PATH:
            bound = agent.theory_consolidation
            for part in bound_ref['json_path']:
                bound = bound[part]
            if not _same_restored_value(bound, current['_bound_holes'],
                                        source_workspace, agent.workspace):
                raise ValueError('relocated parent constructor binding changed')
        old_parent = next((value for reference, value in parents
                           if reference == source_ref), None)
        if old_parent is None:
            old_parent = deepcopy(parent)
            old_parent['program_source_ref'] = deepcopy(source_ref['program_ref'])
            old_parent['_contract_source']['source_ref'] = deepcopy(source_ref)
            parents.append((source_ref, old_parent))
        inputs[identifier] = {**retained, '_retained_parent': old_parent}
    scene['registered_parent_input_groups'] = []
    if not any(key.startswith(prefix) for key in agent.theory_consolidation['executions']):
        return inputs
    from .theory_consolidation import _recall_work_groups, _shared_relation_progress
    library = agent.theory_consolidation
    common = {identifier: retained for identifier in library['candidates']}
    groups = _recall_work_groups(agent, library['candidates'], library, common,
        prefix=prefix.removeprefix('relation_transfer:'))
    for group in groups:
        prior = {identifier: library['executions'][prefix + identifier]
                 for identifier in group if prefix + identifier in library['executions']}
        if not prior:
            scene['registered_parent_input_groups'].append({
                'candidate_ids': list(group), 'selected_execution_key': None})
            continue
        # Compare only the already verified constructor aliases in this small
        # metadata view. Original executions, paid cursors and fees stay intact.
        canonical = parent.get('_contract_source', {}).get('source_ref')
        equivalent = {prefix + identifier: {**execution, 'request_basis': {
            **execution['request_basis'], 'parent': {**execution['request_basis']['parent'],
                **({'contract_source': canonical} if canonical is not None else {})}}}
            for identifier, execution in prior.items()}
        owner, first = group[0], next(iter(equivalent.values()))
        candidate = library['candidates'][owner]
        basis = {**first['request_basis'], 'candidate_id': owner,
                 'candidate_version': candidate['version']}
        selected, matches = _shared_relation_progress(agent,
            {'_execution_candidate_ids': group, '_execution_candidates': library['candidates']},
            candidate, basis, prefix + owner, equivalent)
        if selected is None or len(matches) != len(prior):
            raise ValueError('registered alias executions have different research contracts')
        chosen = selected['execution_key'].removeprefix(prefix)
        shared = inputs[chosen]
        for identifier in group:
            inputs[identifier] = shared
        scene['registered_parent_input_groups'].append({
            'candidate_ids': list(group), 'selected_execution_key': selected['execution_key'],
            'retained_prior_positions': {identifier: {key: (execution.get('cursor') or {}).get(key)
                for key in ('layer', 'offset')} for identifier, execution in prior.items()}})
    return inputs


def _reconcile_registered_parent_order(scene, registration, question, question_ref):
    """Restore only the uncompleted owner rejected by the recorded source pass."""
    agent = scene['agent']
    arm = scene.get('registered_arm', agent.workspace.name)
    units = registration['parent_research'].get('complete_unit_compute_limits', {}).get(arm, [])
    if not units:
        return {'status': 'no_registered_remaining_unit'}
    expected = units[0]['candidate_id']
    execution = question.get('frozen_recall_execution')
    if execution is None:
        prefix = (f'relation_transfer:frozen_recall:{question_ref["question_id"]}'
                  f'@{question_ref["meaning_version"]}:')
        if any(key.startswith(prefix) for key in agent.theory_consolidation['executions']):
            raise ValueError('registered prior parent research lost its outer fair cursor')
        return {'status': 'first_parent_duty_without_prior_cursor'}
    current = execution.get('cursor', {}).get('next_candidate_id')
    if current == expected:
        return {'status': 'registered_fair_order_preserved', 'before': current, 'after': current}
    actions = json.loads((scene['source'] / 'registered_actions.json').read_text())
    failure = actions[-1]
    previous = next((item for item in reversed(actions)
        if item['action'] == 'frozen_recall_on_registered_parent_duty'), None)
    completed = ([item for item in previous['feedback'].get('complete_research_visits', [])
        if item.get('status') == 'complete_visit_executed'] if previous else [])
    if (failure.get('action') != 'execution_failure'
            or failure.get('exception') != 'ValueError' or failure.get('reason') !=
            'actual complete unit differs from the registered persistent-order plan'
            or previous is None or previous.get('research_question_ref') != question_ref
            or previous['registered_unit']['candidate_id'] != expected
            or previous['registered_unit']['search_position'] != units[0]['search_position']
            or len(completed) != 1 or completed[0]['candidate_id'] != expected
            or completed[0]['search_position'] == units[0]['search_position']
            or previous['feedback'].get('object_cursor', {}).get('next_candidate_id') != current):
        raise ValueError('registered outer fair cursor differs without the recorded rejected unit')
    if agent.costs != scene['starting_costs']:
        raise ValueError('registered rejected-unit fees changed during restoration')
    execution['cursor']['next_candidate_id'] = expected
    return {'status': 'recorded_wrong_position_owner_restored',
        'before': current, 'after': expected, 'source_actions_ref': str(
            scene['source'] / 'registered_actions.json'),
        'source_failed_unit_index': previous['unit_index'],
        'expected_position': deepcopy(units[0]['search_position']),
        'rejected_actual_position': deepcopy(completed[0]['search_position']),
        'inner_cursors_and_paid_costs_unchanged': True}


def _interrupted_parent_progress(source, source_registration, state, progress):
    """Bind the audited CP8 prefix to its original arm and fair unit plan."""
    if progress is None:
        raise ValueError('interrupted research state requires its audited paid prefix')
    if Path(progress['source_workspace']).resolve() != source:
        raise ValueError('interrupted progress names a different source workspace')
    if progress['state_costs'] != state['costs']:
        raise ValueError('interrupted progress differs from the committed state account')
    arm = source.name
    plan = source_registration['parent_research']['complete_unit_compute_limits'][arm]
    receipts = progress['unit_receipts']
    complete = progress['fully_committed_new_units']
    if (len(receipts) != progress['native_units_executed']
            or len(receipts) - complete != progress[
                'comparison_returned_but_complete_checkpoint_uncommitted_units']):
        raise ValueError('interrupted checkpoint prefix counts disagree')
    for index, row in enumerate(receipts):
        unit = plan[index]
        if (row['remaining_plan_index'] != index
                or row['candidate_id'] != unit['candidate_id']
                or row['registered_position'] != unit['search_position']
                or row['actual_persisted_position'] != unit['search_position']
                or not row['native_unit_completed']
                or not row['comparison_returned_and_checkpoint_started']
                or row['complete_checkpoint_committed'] != (index < complete)):
            raise ValueError('interrupted paid unit differs from the registered fair prefix')
    return arm


def _interrupted_comparison_roster(registration, state, source):
    """Recover the last comparison through its declared paid-task binding.

    Common consumes research_question before the lower operator records its
    arguments. The saved numerical scope retains that association explicitly.
    """
    from .research_episodes import SEMANTIC_FIELDS

    contract = registration['parent_research']
    questions = {row['id']: row for row in state['questions']}
    duty = questions[contract['question_id']]
    source_question = questions[contract['source_question_id']]
    paid_ref = {'question_id': source_question['id'],
                'meaning_version': source_question.get('meaning_version', 1)}
    if (duty.get('kind') != 'program_revision'
            or duty.get('meaning_version', 1) != contract['meaning_version']
            or duty.get('paid_score_question_ref') != paid_ref
            or duty.get('native_research_ref') != paid_ref
            or duty.get('context_ref') != source_question.get('context_ref')
            or duty.get('theory_target', {}).get('parent_theory_id')
                != contract['parent_theory_ref']['id']):
        raise ValueError('interrupted parent paid-score task differs from its declared source')
    expected_question = json.loads(json.dumps({key: source_question[key]
        for key in ('id', 'meaning_version', *SEMANTIC_FIELDS) if key in source_question}))
    comparison = next((row for row in reversed(state['operator_results'])
        if row.get('arguments', {}).get('operator') == 'research_choice'), None)
    if comparison is None:
        raise ValueError('interrupted parent research lacks a completed strong-pool comparison')
    arguments = comparison['arguments']
    models, evaluations = arguments.get('models'), comparison.get('evaluations', [])
    if (not isinstance(models, list) or comparison.get('truncated', False)
            or 'error' in comparison or comparison.get('paid_rows') != registration['material']['paid_rows']
            or comparison.get('models_requested') != len(models)
            or comparison.get('models_completed') != len(models)
            or [row.get('path') for row in evaluations] != models):
        raise ValueError('interrupted parent research lacks a completed strong-pool comparison')
    scopes = {}
    for evaluation in evaluations:
        reference = evaluation.get('paid_score_reuse', {}).get('scope_ref')
        if reference is None:
            raise ValueError('interrupted comparison lacks its saved paid-score scope')
        key = json.dumps(reference, sort_keys=True)
        if key not in scopes:
            scope = _read(reference, source)
            if (scope.get('question') != expected_question
                    or len(scope.get('rows', [])) != registration['material']['paid_rows']):
                raise ValueError('interrupted comparison paid-score scope differs from its source task')
            scopes[key] = reference
    if not scopes:
        raise ValueError('interrupted comparison lacks its saved paid-score scope')
    return models, {'file': str(source / 'research_state.json'),
        'json_path': ['operator_results', comparison['index'], 'arguments', 'models']}, {
            'parent_research_question_ref': {'question_id': duty['id'],
                'meaning_version': duty.get('meaning_version', 1)},
            'paid_score_question_ref': paid_ref, 'scope_refs': list(scopes.values()),
            'models_requested_and_completed': len(models)}


def load_completed_source_material(registration, *, retained_state=None,
                                   interrupted_progress=None):
    """Read a closed scene or the explicitly audited committed state of CP8."""
    source = Path(registration['material']['same_closed_scene']).resolve()
    previous = _read(registration['material']['training_ref'], source)
    retained_training_costs = deepcopy(previous['costs'])
    source_registration = json.loads((source.parent / 'registration.json').read_text())
    if retained_state is not None:
        from .common_archive_inquiry import retained_checkpoint_result
        original_arm = _interrupted_parent_progress(
            source, source_registration, retained_state, interrupted_progress)
        previous = retained_checkpoint_result(previous, retained_state)
        starting = deepcopy(retained_state['costs'])
    else:
        if interrupted_progress is not None:
            raise ValueError('interrupted progress requires its already read research state')
        completed = json.loads((source.parent / 'summary.json').read_text())
        original = (next(item for item in completed['arms']
                         if Path(item['workspace']).resolve() == source)
                    if 'arms' in completed else completed)
        if Path(original['workspace']).resolve() != source:
            raise ValueError('completed summary names a different source workspace')
        original_arm = original.get('arm', source.name)
        paid_delta = json.loads((source / 'final_costs.json').read_text())
        starting = {key: original['starting_costs'][key] + paid_delta[key] for key in paid_delta}
        if any(starting[key] < value for key, value in retained_training_costs.items()):
            raise ValueError('completed-scene account precedes its retained checkpoint')
        previous['costs'] = deepcopy(starting)
    # Completed controls can themselves be the next registered source. Follow
    # their declared source until the original constructor/roster is reached.
    work_registration = source_registration
    while 'source_training_ref' not in work_registration['material']:
        parent_scene = Path(work_registration['material']['same_closed_scene']).resolve()
        work_registration = json.loads((parent_scene.parent / 'registration.json').read_text())
    origin = Path(work_registration['material']['source_training_ref']['file']).resolve().parent
    plan = json.loads((origin.parent.parent / 'protocol.json').read_text())
    case = next(item for item in plan['cases'] if item['id'] == origin.parent.name)
    knowledge = _read({'file': str(origin.parent.parent / 'initial_archive.json'), 'json_path': []}, origin)
    while set(knowledge) == {'source_ref'}:
        knowledge = _read(knowledge['source_ref'], origin)
    knowledge = resolve_workspace_archive(knowledge)
    knowledge['finite_native_author']['question'] = deepcopy(plan['prepared_public_questions'][case['id']])
    new_candidate = _read(registration['material']['new_object_ref'], source)
    models = _read(work_registration['material']['common_executable_model_roster_ref'], source)
    # Registered controls save their direct research actions separately; their
    # training method_results can still contain only inherited source research.
    # Retain the actual completed joint comparison and its source namespaces.
    actions = source / 'registered_actions.json'
    comparisons = (json.loads(actions.read_text()) if actions.is_file()
                   else previous.get('method_results', [])) if retained_state is None else []
    retained_comparison, retained_score_binding = None, None
    if retained_state is not None:
        roster, retained_comparison, retained_score_binding = _interrupted_comparison_roster(
            registration, retained_state, source)
        models = list(dict.fromkeys([*models, *roster]))
    for result in reversed(comparisons):
        roster = result.get('complete_model_roster_ref')
        if result.get('action') == 'research_choice' and roster:
            models = list(dict.fromkeys([*models, *_read(roster, source)]))
            break
        feedback = result.get('feedback', {})
        if (feedback.get('operator') == 'research_choice'
                and feedback.get('paid_rows') == registration['material']['paid_rows']):
            models = list(dict.fromkeys([*models, *(
                item['path'] for item in feedback.get('evaluations', []) if item.get('path'))]))
            break
    constructor = work_registration['material']['parent_constructor_source']
    pattern, holes, bound = [item['value'] for item in load_source_values(
        [constructor[field] for field in ('pattern_ref', 'holes_ref', 'components_ref')], source)]
    return {'source': source, 'previous': previous, 'plan': plan, 'case': case,
        'knowledge': knowledge, 'new_candidate': new_candidate,
        'retained_training_costs': retained_training_costs, 'starting_costs': starting,
        'retained_state': retained_state, 'interrupted_progress': interrupted_progress,
        'retained_comparison_roster_ref': retained_comparison,
        'retained_score_binding': retained_score_binding,
        'source_registration': work_registration, 'completed_registration': source_registration,
        'source_candidate_versions': {row['id']: row['version'] for row in
            source_registration['arms'][original_arm]['candidate_versions']},
        'closed_scene_theory_count': len(previous['theory_index']),
        'models': models,
        'parent_contract': {'source_ref': deepcopy(constructor),
            '_pattern': pattern, '_holes': holes, '_bound_holes': bound}}


def restore_completed_source_scene(registration, arm, workspace, *, material):
    """Change frozen availability while retaining every completed-scene search."""
    from scripts import run_anonymous_revision_inquiry as study
    from scripts.prepare_pr3_source_successor import _retain_scene_score_index
    from .question_routing import _retained_revision_inputs
    from .theory_consolidation import _retained_parent_contract, _same_relation_work
    from .common_archive_inquiry import retained_checkpoint_result

    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise ValueError('restore into a fresh workspace and preserve the completed scene')
    source, previous = material['source'], material['previous']
    plan, case = material['plan'], material['case']
    starting = material['starting_costs']
    retained_state = material.get('retained_state')
    if retained_state is None:
        # Restoration must read this state, but it need not also deepcopy the
        # full historical score graph. Transfer the one decoded graph through
        # Common's existing retained-state path. Final readouts may have been
        # billed after its last checkpoint; keep that authoritative paid tail.
        retained_state = read_json_checkpoint(source / 'research_state.json')
        checkpoint_costs = retained_state['costs']
        if any(starting[key] < value for key, value in checkpoint_costs.items()):
            raise ValueError('completed-scene account precedes its committed research state')
        if any(starting[key] != checkpoint_costs[key]
               for key in ('observation_cost', 'measurement_cost', 'model_calls')):
            raise ValueError('completed-scene state differs from its paid acquisitions')
        retained_state['costs'] = deepcopy(starting)
        previous = retained_checkpoint_result(previous, retained_state)
    limits = replace(study.InquiryLLMConfig(**plan['limits']),
        compute_budget=starting['compute_cost'] + registration['budget']['additional_compute_per_arm'],
        execution_seconds=starting['python_execution_seconds']
            + registration['budget']['additional_Python_seconds_per_arm'],
        observation_budget=starting['observation_cost'], max_model_calls=0, token_budget=0)
    knowledge = deepcopy(material['knowledge'])
    paid = archived_paid_trace_references(knowledge.get('files', {}),
        workspace.parent / 'reference_archive', source_workspace=source)
    agent = study.RegisteredOrganizationAgent(None, workspace, arm='pr3',
        research_organization=deepcopy(plan['research_organization']), knowledge=knowledge,
        config=limits, confirmation_seed=case['confirmation_seed'], source_task=False,
        paid_trace_references=paid)
    lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
    restored = agent.restore_research(previous, source, lab,
        confirmation_seed=case['confirmation_seed'], retained_state=retained_state)
    if (agent.costs != starting or agent.observations != previous['paid_observations']
            or len(agent.rows) != registration['material']['paid_rows']
            or len(agent.theory_index) != material.get('closed_scene_theory_count',
                registration['material'].get('closed_scene_theory_count'))):
        raise ValueError('completed scene paid material or account changed during restoration')
    library = agent.theory_consolidation
    executions, transfers = library['executions'], library['transfers']
    # Historical producers stay available to the exact-fit query even when
    # this arm removes their future construction opportunity.
    fit_constructors = dict(library['candidates'])
    # An ended REMOVE view still needs the frozen producer declaration to
    # query its already-paid fits; this does not enable new construction.
    fit_constructors.setdefault(material['new_candidate']['id'], material['new_candidate'])
    original_versions = {key: value.get('version', 1) for key, value in library['candidates'].items()}
    expected_old = {row['id']: row['version'] for row in registration['arms']['remove']['candidate_versions']}
    joint = 'complete_common_bank_manifest_ref' in registration['material']
    source_versions = material.get('source_candidate_versions',
        {row['id']: row['version'] for row in registration['arms']['keep']['candidate_versions']}
        if joint else expected_old)
    if original_versions != source_versions:
        raise ValueError('completed-scene objects differ from the registered old19 view')
    new = material['new_candidate']
    new_execution_keys = [key for key in executions if key.endswith(':' + new['id'])]
    new_cursors = {key: deepcopy(executions[key].get('cursor')) for key in new_execution_keys}
    if new_execution_keys and not joint:
        raise ValueError('new20 unexpectedly has a pre-existing execution')
    question = next(q for q in agent.questions if q['id'] == 'native_public_response')
    inputs = _retained_revision_inputs(agent, question)
    old_registration = material['source_registration']
    old_ids = {row['id'] for row in old_registration['arms']['keep']['candidate_versions']} - {
        row['id'] for row in old_registration['arms']['remove']['candidate_versions']}
    if len(old_ids) != 1:
        raise ValueError('original old19 source work is not uniquely identified')
    old_id = old_ids.pop()
    key = f'relation_transfer:frozen_recall:{question["id"]}@{question.get("meaning_version", 1)}:{old_id}'
    execution = executions[key]
    original_inputs = inputs['_retained_execution_inputs'][old_id]
    parent = _retained_parent_contract(agent, original_inputs['_retained_parent'])
    components = original_inputs['_retained_components']
    basis = execution['request_basis']
    if basis.get('parent', {}).get('contract_source'):
        contract = material['parent_contract']
        if basis['parent']['contract_source'] != contract['source_ref']:
            raise ValueError('original native parent constructor differs from the saved work')
        parent['_contract_source'] = deepcopy(contract)
        parent['program_source_ref'] = deepcopy(old_registration['material']['parent_Program_ref'])
    recovered = {**basis, 'parent': {field: deepcopy(parent.get(field)) for field in
        ('id', 'version', 'source_ref', 'source_status')}, 'components': [
            {field: deepcopy(item.get(field)) for field in
                ('id', 'version', 'source_ref', 'compatible', 'source_status')} for item in components]}
    if parent.get('_contract_source'):
        recovered['parent']['contract_source'] = deepcopy(parent['_contract_source']['source_ref'])
    if not _same_relation_work(agent, basis, recovered):
        raise ValueError('restoration changed the old19 parent or component versions')
    cursor = execution['cursor']
    pending = [f'{item["position"]["layer"]}:{item["position"]["offset"]}'
               for item in cursor.get('pending_fits', [])]
    expected_pending = registration['restoration_admission'].get('old19_original_pending_positions')
    if expected_pending is not None and pending != expected_pending:
        raise ValueError('old19 pending cursor differs from the registered completed scene')
    expected = {row['id']: row['version'] for row in registration['arms'][arm]['candidate_versions']}
    if new['id'] in expected:
        library['candidates'][new['id']] = deepcopy(new)
    else:
        library['candidates'].pop(new['id'], None)
    if {key: value.get('version', 1) for key, value in library['candidates'].items()} != expected:
        raise ValueError('only the registered new20 availability may differ')
    # Saving a view must not replace the target's local executions with those
    # from the source-learning library that supplied the new Definition.
    agent._theory_library_candidate_versions = None
    agent._theory_library_entry_references = None
    save_theory_library(agent)
    if library['executions'] is not executions or library['transfers'] is not transfers:
        raise ValueError('availability installation replaced completed research work')
    agent._organization_state['automatic_library_path'] = LIBRARY_PATH
    agent.knowledge['finite_native_author']['question']['frozen_library_path'] = LIBRARY_PATH
    for q in agent.questions:
        if 'frozen_library_path' in q or 'frozen_recall_execution' in q:
            q['frozen_library_path'] = LIBRARY_PATH
    # Use every final compared model, including source and development outputs.
    models = list(dict.fromkeys([*material['models'], *agent.candidate_paths]))
    expected_models = registration['material'].get('closed_scene_scored_models')
    if expected_models is not None and len(models) != expected_models:
        raise ValueError('completed-scene strong model pool is incomplete')
    missing = [name for name in models if not (workspace / name).is_file()]
    if missing:
        raise ValueError(f'completed-scene strong pool has {len(missing)} unavailable models')
    agent.knowledge['finite_native_author']['predictor_paths'] = models
    score_index = _retain_scene_score_index(source, workspace, paid_rows=len(agent.rows))
    winner = previous['evaluated_model']
    if (agent.evaluated_model['source_model'] != winner['source_model']
            or agent.evaluated_model['current_mse'] != winner['mse']):
        raise ValueError('completed-scene task winner changed during restoration')
    arguments = {field: deepcopy(basis.get(field)) for field in (
        'fitting', 'evidence_checks', 'role_signature', 'output_unit', 'context_ref')}
    arguments.update(candidate_id=new['id'], library_path=LIBRARY_PATH,
        parent_theory_id=parent['id'], component_theory_ids=deepcopy(basis['component_theory_ids']),
        allocation={'compute': registration['budget']['per_consumption_compute'],
                    'comparison_rows': len(agent.rows)})
    save_theory_index(agent)
    agent._prepare_program_state()
    validation = {'status': 'completed_scene_restored_without_scientific_execution', **restored,
        'source': str(source), 'paid_rows': len(agent.rows), 'complete_theory_count': len(agent.theory_index),
        'complete_model_count': len(models), 'active_candidate_versions': expected,
        'starting_costs': deepcopy(agent.costs), 'retained_training_costs': material['retained_training_costs'],
        'post_checkpoint_fees_carried': {key: starting[key]-material['retained_training_costs'][key]
                                      for key in starting},
        'task_winner_retained': True, 'old19_execution_key': key,
        'old19_cursor_layer': cursor['layer'], 'old19_cursor_offset': cursor['offset'],
        'old19_pending_positions': pending, 'source_component_count': len(components),
        'exact_historical_component_versions_verified': True,
        'new20_has_prior_cursor': bool(new_execution_keys),
        'new20_original_cursor_positions': {name: {
            'layer': value.get('layer'), 'offset': value.get('offset'),
            'pending_positions': [item['position'] for item in value.get('pending_fits', [])]}
            for name, value in new_cursors.items()},
        'original_execution_count': len(executions), 'original_transfer_count': len(transfers),
        'retained_score_index': score_index,
        'new_fits': 0, 'new_scores': 0, 'new_predictions': 0, 'paid_model_calls': 0}
    if material.get('retained_state') is not None:
        validation['authoritative_scientific_checkpoint'] = {
            'file': str(source / 'research_state.json'),
            'compact_training_is_supplemental': True,
            'completed_comparison_roster_ref': material['retained_comparison_roster_ref'],
            'paid_score_binding': material['retained_score_binding'],
            'paid_prefix': material['interrupted_progress'],
            'inherited_terminal_actions_and_costs_used': False}
    write_json_checkpoint(workspace / 'restoration_validation.json', validation, ensure_ascii=False)
    scene = {'agent': agent, 'lab': lab, 'parent': parent, 'retained_components': components,
        'source_component_theory_ids': deepcopy(basis['component_theory_ids']),
        'transfer_arguments': arguments, 'models': models, 'validation': validation,
        'source': source, 'previous': previous, 'starting_costs': deepcopy(starting),
        'original_new_cursors': new_cursors,
        'retained_fit_constructors': fit_constructors,
        'historical_fit_library_ref': {'file': str(source / LIBRARY_PATH), 'json_path': []}}
    if joint:
        scene['joint_bank_validation'] = materialize_joint_exposed_bank(
            agent, registration, shared_material=material)
        scene['models'] = agent.knowledge['finite_native_author']['predictor_paths']
    return scene


def prepare_completed_source_control(registration, output):
    """Accept literal restoration and numerical reuse before either arm runs."""
    if 'same_closed_scene_by_arm' in registration['material']:
        return prepare_parent_continuation_control(registration, output)
    from unittest.mock import patch
    from . import abstractions, revision_relations
    from .language import Program
    from .models import SearchBudget
    from .theory_consolidation import _retained_revision_fits

    output = Path(output).resolve()
    material = load_completed_source_material(registration)
    scenes = {}
    for arm in registration['arms']:
        scenes[arm] = restore_completed_source_scene(registration, arm, output / arm, material=material)
        print(json.dumps({'stage': 'completed_scene_restored', 'arm': arm,
            'validation': scenes[arm]['validation']}, ensure_ascii=False), flush=True)
    left, right = [scenes[arm]['agent'] for arm in ('remove', 'keep')]
    pairs = {'theory_index': (left.theory_index, right.theory_index),
        'paid_observations': (left.observations, right.observations), 'costs': (left.costs, right.costs),
        'executions': (left.theory_consolidation['executions'], right.theory_consolidation['executions']),
        'transfers': (left.theory_consolidation['transfers'], right.theory_consolidation['transfers'])}
    equal = {key: _same_restored_value(a, b, left.workspace, right.workspace)
             for key, (a, b) in pairs.items()}
    if not all(equal.values()):
        write_json_checkpoint(output / 'scene_equivalence_failure.json', equal, ensure_ascii=False)
        raise ValueError('two restored arms differ before frozen availability is used')
    new = material['new_candidate']
    work = SearchBudget(300000)
    original_key = scenes['keep']['validation']['old19_execution_key']
    original_cursor = deepcopy(right.theory_consolidation['executions'][original_key]['cursor'])

    def no_scientific_execution(*args, **kwargs):
        raise AssertionError('restoration acceptance must not fit or predict')

    with patch.object(Program, 'predict', no_scientific_execution), \
            patch.object(abstractions, 'fit_parameter_slots', no_scientific_execution), \
            patch.object(revision_relations, '_fit_revision_binding', no_scientific_execution):
        records = _retained_revision_fits(right, new, scenes['keep']['parent'], work)
        if not records:
            raise ValueError('new20 cannot retrieve the completed old generic fits')
        first = records[0]
        reused = revision_relations._reuse_revision_fit(first['template'], first['rows'],
            first['risk_quantile'], scenes['keep']['parent'], records, work)
        if reused is None or reused[2].get('actual_compute_cost') != 0:
            raise ValueError('new20 cannot reuse an equal complete old fit without solving')
    prior_new_cursors = scenes['keep'].get('original_new_cursors', {})
    current_new_cursors = {key: deepcopy(value.get('cursor')) for key, value in
        right.theory_consolidation['executions'].items() if key.endswith(':' + new['id'])}
    if (right.theory_consolidation['executions'][original_key]['cursor'] != original_cursor
            or current_new_cursors != prior_new_cursors):
        raise ValueError('fit-reference acceptance changed object search progress')
    reuse = {'status': 'passed_actual_complete_fit_reference_reuse_without_solve',
        'available_complete_fit_batches': len(records),
        'restored_methods': reused[2]['candidate_methods'],
        'fit_reference': reused[2]['fit_reuse_ref'],
        'metadata_acceptance_compute': work.spent, 'new_fits': 0, 'new_predictions': 0,
        'old19_cursor_unchanged': True, 'new20_cursor_absent': not prior_new_cursors,
        'new20_own_cursor_unchanged': True}
    write_json_checkpoint(output / 'fit_reuse_acceptance.json', reuse, ensure_ascii=False)
    acceptance = {'status': 'accepted_same_completed_scene_old19_keep20',
        'arms': {arm: scene['validation'] for arm, scene in scenes.items()},
        'same_complete_pool_paid_rows_old_work_and_fees': True,
        'same_scene_fields': equal, 'local_reference_relocation_only': True,
        'fit_reuse_acceptance_ref': str(output / 'fit_reuse_acceptance.json'),
        'metadata_acceptance_compute': work.spent, 'new_fits': 0, 'new_scores': 0,
        'new_predictions': 0, 'new_observations': 0, 'paid_model_calls': 0}
    if 'complete_common_bank_manifest_ref' in registration['material']:
        acceptance['joint_exposed_banks'] = {arm: scene['joint_bank_validation']
                                             for arm, scene in scenes.items()}
        acceptance['paid_score_reuse'] = {arm: quote_joint_paid_scores(scene)
                                           for arm, scene in scenes.items()}
        if scenes['remove']['models'] != scenes['keep']['models']:
            raise ValueError('joint exposed model rosters differ')
    if 'parent_research' in registration:
        acceptance['registered_parent_research'] = {arm: prepare_registered_parent_revision(
            scene, registration) for arm, scene in scenes.items()}
    write_json_checkpoint(output / 'registration.json', registration, ensure_ascii=False)
    write_json_checkpoint(output / 'acceptance.json', acceptance, ensure_ascii=False)
    return scenes, acceptance


def prepare_parent_continuation_control(registration, output):
    """Continue each ended195 scene instead of restarting the common scene."""
    from .registered_source_budget import retained_four_round_progress

    output = Path(output).resolve()
    original = _read(registration['continuation']['original_registration_ref'], output)
    scenes, receipts = {}, {}
    for arm, source in registration['material']['same_closed_scene_by_arm'].items():
        source = Path(source).resolve()
        actions = json.loads((source / 'registered_actions.json').read_text())
        progress = retained_four_round_progress(
            original['parent_research']['complete_unit_compute_limits'][arm], actions)
        if (progress['remaining_units'] != registration['parent_research'][
                'complete_unit_compute_limits'][arm]
                or progress['completed_unit_counts_by_group'] != registration['parent_research'][
                    'completed_unit_counts_by_group'][arm]):
            raise ValueError('registered continuation differs from retained195 fair progress')
        local = {**registration, 'material': {**registration['material'],
            'same_closed_scene': str(source),
            'training_ref': {'file': str(source / 'training.json'), 'json_path': []},
            'research_state_ref': {'file': str(source / 'research_state.json'), 'json_path': []}}}
        material = load_completed_source_material(local)
        scene = restore_completed_source_scene(local, arm, output / arm, material=material)
        parent = prepare_registered_parent_revision(scene, local)
        if not parent['current_parent_scope_has_prior_executions']:
            raise ValueError('ended195 continuation lost its independent research work')
        scores = quote_joint_paid_scores(scene)
        scenes[arm] = scene
        receipts[arm] = {'restoration': scene['validation'], 'parent_research': parent,
            'consumed_units': progress['consumed_units'],
            'comparison_of_committed_work_required': progress['comparison_of_committed_work_required'],
            'remaining_units': len(progress['remaining_units']), 'paid_scores': scores}
    acceptance = {'status': 'accepted_own_ended195_scenes_without_restarting_fits',
        'arms': receipts, 'metadata_acceptance_compute': 0,
        'same_original_paid_material_and_strong_baseline': True,
        'each_arm_own195_progress_and_expenses_preserved': True,
        'new_fits': 0, 'new_scores': 0, 'new_predictions': 0, 'new_observations': 0,
        'paid_model_calls': 0, 'common_repair_is_PR3_gain': False}
    write_json_checkpoint(output / 'registration.json', registration, ensure_ascii=False)
    write_json_checkpoint(output / 'acceptance.json', acceptance, ensure_ascii=False)
    return scenes, acceptance


def resume_completed_source_scene(registration, arm, prior, workspace, *, material,
                                  original_execution):
    """Complete the unused second allowance, keeping the first pass and its bill."""
    from scripts import run_anonymous_revision_inquiry as study
    from scripts.prepare_pr3_source_successor import _retain_scene_score_index
    from .question_routing import _retained_revision_inputs
    from .theory_consolidation import _same_relation_work

    prior, workspace = Path(prior).resolve(), Path(workspace).resolve()
    if workspace.exists():
        raise ValueError('resume into a fresh workspace and retain the first pass')
    actions = json.loads((prior / 'registered_actions.json').read_text())
    routes = [item for item in actions if item['action'] == 'research_route']
    if len(routes) != 1 or routes[0]['compute_delta'] != 0:
        raise ValueError('the second registered allowance was already consumed')
    old_summary = json.loads((prior / 'arm_summary.json').read_text())
    original = deepcopy(material['starting_costs'])
    if old_summary['starting_costs'] != original:
        raise ValueError('the first pass does not have the registered original account')
    paid_delta = json.loads((prior / 'final_costs.json').read_text())
    starting = {key: original[key] + value for key, value in paid_delta.items()}
    previous = _read({'file': str(prior / 'training.json'), 'json_path': []}, prior)
    previous['costs'] = deepcopy(starting)  # include the two post-checkpoint readouts
    plan, case = material['plan'], material['case']
    limits = replace(study.InquiryLLMConfig(**plan['limits']),
        compute_budget=original['compute_cost'] + registration['budget']['additional_compute_per_arm'],
        execution_seconds=original['python_execution_seconds']
            + registration['budget']['additional_Python_seconds_per_arm'],
        observation_budget=original['observation_cost'], max_model_calls=0, token_budget=0)
    knowledge = deepcopy(material['knowledge'])
    paid = archived_paid_trace_references(knowledge.get('files', {}),
        workspace.parent / 'reference_archive', source_workspace=material['source'])
    agent = study.RegisteredOrganizationAgent(None, workspace, arm='pr3',
        research_organization=deepcopy(plan['research_organization']), knowledge=knowledge,
        config=limits, confirmation_seed=case['confirmation_seed'], source_task=False,
        paid_trace_references=paid)
    lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
    restored = agent.restore_research(previous, prior, lab,
        confirmation_seed=case['confirmation_seed'])
    expected = {row['id']: row['version'] for row in registration['arms'][arm]['candidate_versions']}
    library = agent.theory_consolidation
    if (agent.costs != starting or agent.observations != previous['paid_observations']
            or {key: item.get('version', 1) for key, item in library['candidates'].items()} != expected):
        raise ValueError('resume changed retained fees, evidence or frozen availability')

    # Recover only the demonstrated missing-constructor execution. Its failed
    # first pass remains a separate complete record, and its cost is not refunded.
    basis = original_execution['request_basis']
    old_id = basis['candidate_id']
    key = f'relation_transfer:frozen_recall:native_public_response@1:{old_id}'
    current = library['executions'][key]
    preserved_key = None
    if not _same_relation_work(agent, current['request_basis'], basis):
        repaired_basis = deepcopy(current['request_basis'])
        repaired_basis['parent']['contract_source'] = deepcopy(basis['parent']['contract_source'])
        if (current['request_basis']['parent'].get('contract_source') is not None
                or not _same_relation_work(agent, repaired_basis, basis)):
            raise ValueError('old19 differs by more than the diagnosed missing constructor')
        suffix = 0
        while f'{key}:misbound:{suffix}' in library['executions']:
            suffix += 1
        preserved_key = f'{key}:misbound:{suffix}'
        library['executions'][preserved_key] = deepcopy(current)
        library['executions'][key] = deepcopy(original_execution)
    question = next(q for q in agent.questions if q['id'] == 'native_public_response')
    inputs = _retained_revision_inputs(agent, question)
    retained = inputs['_retained_execution_inputs'][old_id]
    parent = retained['_retained_parent']
    if parent.get('_contract_source', {}).get('source_ref') != basis['parent']['contract_source']:
        raise ValueError('ordinary retained recall still loses the saved parent constructor')
    models = list(dict.fromkeys([*material['models'], *agent.candidate_paths]))
    compared = next(item['models_requested'] for item in reversed(actions)
                    if item['action'] == 'research_choice')
    if len(models) != compared:
        raise ValueError('resume lost part of the complete first-pass model pool')
    agent._organization_state['automatic_library_path'] = LIBRARY_PATH
    agent.knowledge['finite_native_author']['predictor_paths'] = models
    agent.knowledge['finite_native_author']['question']['frozen_library_path'] = LIBRARY_PATH
    for q in agent.questions:
        if 'frozen_library_path' in q or 'frozen_recall_execution' in q:
            q['frozen_library_path'] = LIBRARY_PATH
    score_index = _retain_scene_score_index(prior, workspace, paid_rows=len(agent.rows))
    save_theory_library(agent)
    agent._prepare_program_state()
    new_id = material['new_candidate']['id']
    arguments = {field: deepcopy(basis.get(field)) for field in (
        'fitting', 'evidence_checks', 'role_signature', 'output_unit', 'context_ref')}
    arguments.update(candidate_id=new_id, library_path=LIBRARY_PATH,
        parent_theory_id=parent['id'], component_theory_ids=deepcopy(basis['component_theory_ids']),
        allocation={'compute': registration['budget']['per_consumption_compute'],
                    'comparison_rows': len(agent.rows)})
    receipt = {'status': 'retained_first_pass_unused_second_allocation_accepted',
        'source': str(prior), **restored, 'starting_costs': starting,
        'original_starting_costs': original, 'original_compute_cap': limits.compute_budget,
        'original_Python_cap': limits.execution_seconds, 'retained_model_count': len(models),
        'retained_score_index': score_index, 'old19_execution_key': key,
        'preserved_misbound_execution_key': preserved_key,
        'old19_cursor': deepcopy(library['executions'][key]['cursor']),
        'new20_execution_cursor': deepcopy(library['executions'].get(
            f'relation_transfer:frozen_recall:native_public_response@1:{new_id}', {}).get('cursor')),
        'second_allocation_compute': registration['budget']['per_consumption_compute'],
        'third_native_allocation': False, 'new_fits': 0, 'new_scores': 0, 'new_predictions': 0}
    write_json_checkpoint(workspace / 'resume_validation.json', receipt, ensure_ascii=False)
    return {'agent': agent, 'lab': lab, 'models': models, 'parent': parent,
        'retained_components': retained['_retained_components'],
        'source_component_theory_ids': deepcopy(basis['component_theory_ids']),
        'transfer_arguments': arguments, 'resume_phase': 'last_research_allocation',
        'first_allocation_remaining': registration['budget']['per_consumption_compute'],
        'prior_continuation_costs': paid_delta, 'original_starting_costs': original,
        'validation': receipt}


def load_registered_source_material(registration):
    """Read the registered retained inputs once, without fitting or scoring."""
    material = registration['material']
    source = Path(material['source_training_ref']['file']).resolve().parent
    previous = _read(material['source_training_ref'], source)
    plan = json.loads((source.parent.parent / 'protocol.json').read_text())
    case_id = source.parent.name
    case = next(item for item in plan['cases'] if item['id'] == case_id)
    initial = {'file': str(source.parent.parent / 'initial_archive.json'), 'json_path': []}
    knowledge = _read(initial, source)
    while set(knowledge) == {'source_ref'}:
        knowledge = _read(knowledge['source_ref'], source)
    knowledge = resolve_workspace_archive(knowledge)
    declarations = _read(material['common_additional_theory_bank_ref'], source)
    values = load_source_values([item['program_source_ref'] for item in declarations], source)
    for declaration, loaded in zip(declarations, values):
        declaration['program'] = loaded['value']
    models = _read(material['common_executable_model_roster_ref'], source)
    library = _read_library(registration['arms']['keep']['library_ref'], source)
    versions = {key: value.get('version', 1) for key, value in library['candidates'].items()}
    for arm in registration['arms'].values():
        if any(versions.get(item['id']) != item['version'] for item in arm['candidate_versions']):
            raise ValueError('registered object versions differ from the frozen library')
    constructor = material['parent_constructor_source']
    references = [constructor[key] for key in ('pattern_ref', 'holes_ref', 'components_ref')]
    pattern, holes, components = [item['value'] for item in load_source_values(references, source)]
    return {'source': source, 'previous': previous, 'plan': plan, 'case': case,
        'knowledge': knowledge, 'declarations': declarations, 'models': models,
        'frozen_candidates': library['candidates'], 'parent_contract': {
            'source_ref': deepcopy(constructor), '_pattern': pattern, '_holes': holes,
            '_bound_holes': components}}


def restore_registered_source_scene(registration, arm, workspace, *, material=None):
    """Restore unchanged paid evidence and shared development work for one arm."""
    from scripts import run_anonymous_revision_inquiry as study
    from scripts.prepare_pr3_source_successor import _retain_scene_score_index

    retained = material or load_registered_source_material(registration)
    source, previous = retained['source'], retained['previous']
    case, plan = retained['case'], retained['plan']
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise ValueError('restore into a fresh workspace and preserve the original scene')
    starting_costs = deepcopy(previous['costs'])
    starting_costs['compute_cost'] += registration['costs'][
        'new_induction_and_development_compute_once']
    limits = replace(study.InquiryLLMConfig(**plan['limits']),
        compute_budget=starting_costs['compute_cost'] + registration['budget']['additional_compute_per_arm'],
        execution_seconds=starting_costs['python_execution_seconds']
            + registration['budget']['additional_Python_seconds_per_arm'],
        observation_budget=previous['costs']['observation_cost'], max_model_calls=0, token_budget=0)
    knowledge = deepcopy(retained['knowledge'])
    knowledge['finite_native_author']['question'] = deepcopy(plan['prepared_public_questions'][case['id']])
    organization = deepcopy(plan['research_organization'])
    paid = archived_paid_trace_references(knowledge.get('files', {}),
        workspace.parent / 'reference_archive', source_workspace=source)
    agent = study.RegisteredOrganizationAgent(None, workspace, arm='pr3',
        research_organization=organization, knowledge=knowledge, config=limits,
        confirmation_seed=case['confirmation_seed'], source_task=False,
        paid_trace_references=paid)
    laboratory = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
    restored = agent.restore_research(previous, source, laboratory,
        confirmation_seed=case['confirmation_seed'])
    if agent.costs != previous['costs']:
        raise ValueError('Common restoration changed the retained source account')
    original_index = deepcopy(agent.theory_index)
    original_library = agent.theory_consolidation
    original_execution_count = len(original_library.get('executions', {}))
    components = [item['id'] for item in resolved_theories(agent) if item.get('program')]
    bank = retained['declarations']
    if (len(agent.rows) != registration['material']['same_paid_rows']
            or agent.observations != previous['paid_observations']
            or len(original_index) != registration['material']['old_archive_entries']
            or len(components) != registration['material']['old_native_entries']):
        raise ValueError('restored source material differs from the registered strong pool')
    register_theories(agent, bank)
    expected = registration['arms'][arm]['candidate_versions']
    # Availability changes, while all source work, failures and cursors survive.
    original_library['candidates'] = {item['id']: deepcopy(retained['frozen_candidates'][item['id']])
                                      for item in expected}
    agent._theory_library_candidate_versions = None
    agent._theory_library_entry_references = None
    save_theory_library(agent)
    # The working-library token uses the installed in-memory view. An absolute
    # path to this same file would expand its retained archive on every recall.
    view = LIBRARY_PATH
    agent._organization_state['automatic_library_path'] = view
    agent.knowledge['finite_native_author']['question']['frozen_library_path'] = view
    for question in agent.questions:
        if ('frozen_library_path' in question or 'frozen_recall_execution' in question
                or question.get('theory_target', {}).get('revision_fitting')):
            question['frozen_library_path'] = view
    parent_id = registration['material']['parent_theory_id']
    parent = deepcopy(next(item for item in resolved_theories(agent) if item['id'] == parent_id))
    parent['program_source_ref'] = deepcopy(registration['material']['parent_Program_ref'])
    parent['_contract_source'] = deepcopy(retained['parent_contract'])
    agent.costs = deepcopy(starting_costs)
    score_index = _retain_scene_score_index(source, workspace, paid_rows=len(agent.rows))
    save_theory_index(agent)
    agent._prepare_program_state()
    expected_parent = previous['evaluated_model']
    validation = {'status': 'restored_without_research_execution', **restored,
        'source_case': deepcopy(case), 'paid_record_count': len(agent.observations),
        'paid_rows': len(agent.rows), 'unchanged_old_theories': all(
            agent.theory_index[key] == value for key, value in original_index.items()),
        'shared_development_entries': len(bank), 'complete_theory_count': len(agent.theory_index),
        'complete_model_count': len(retained['models']), 'source_component_count': len(components),
        'current_winner_retained': agent.evaluated_model['source_model'] == expected_parent['source_model']
            and agent.evaluated_model['current_mse'] == expected_parent['mse'],
        'original_library_execution_count': original_execution_count,
        'retained_library_execution_count': len(agent.theory_consolidation.get('executions', {})),
        'retained_score_index': score_index, 'active_candidate_versions': deepcopy(expected),
        'original_source_costs': deepcopy(previous['costs']), 'starting_costs': deepcopy(starting_costs),
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'paid_model_calls': 0}
    if (not validation['unchanged_old_theories'] or not validation['current_winner_retained']
            or len(agent.theory_index) != registration['material']['common_archive_entries_after_materialization']
            or len(retained['models']) != registration['material']['common_model_count']):
        raise ValueError('source restoration did not preserve the registered comparison material')
    write_json_checkpoint(workspace / 'restoration_validation.json', validation, ensure_ascii=False)
    return {'agent': agent, 'lab': laboratory, 'source': source, 'starting_costs': starting_costs,
        'validation': validation, 'models': retained['models'], 'parent': parent,
        'source_component_theory_ids': components, 'case': case, 'plan': plan,
        'previous': previous}


def resume_registered_source_scene(registration, arm, failed, workspace, *, material):
    """Resume the actual failed action, retaining its costs and structural cursor."""
    from scripts import run_anonymous_revision_inquiry as study
    from scripts.prepare_pr3_source_successor import _retain_scene_score_index

    failed, workspace = Path(failed).resolve(), Path(workspace).resolve()
    previous = json.loads((failed/'training.json').read_text())
    old_summary = json.loads((failed/'arm_summary.json').read_text())
    paid_delta = json.loads((failed/'final_costs.json').read_text())
    plan, case = material['plan'], material['case']
    original = deepcopy(material['previous']['costs'])
    original['compute_cost'] += registration['costs']['new_induction_and_development_compute_once']
    actions = json.loads((failed/'registered_actions.json').read_text())
    resume_phase = ('final_comparison' if any(item['action']=='research_route' for item in actions)
        else 'first_comparison' if any(item['action'] in {
            'frozen_recall','export_committed_interrupted_work'} for item in actions) else 'first_recall')
    limits = replace(study.InquiryLLMConfig(**plan['limits']),
        compute_budget=original['compute_cost']+registration['budget']['additional_compute_per_arm'],
        execution_seconds=original['python_execution_seconds']+registration['budget'][
            'additional_Python_seconds_per_arm'], observation_budget=previous['costs']['observation_cost'],
        max_model_calls=0, token_budget=0)
    knowledge = deepcopy(material['knowledge'])
    knowledge['finite_native_author']['question'] = deepcopy(plan['prepared_public_questions'][case['id']])
    paid = archived_paid_trace_references(knowledge.get('files', {}),
        workspace.parent/'reference_archive', source_workspace=material['source'])
    agent = study.RegisteredOrganizationAgent(None, workspace, arm='pr3',
        research_organization=deepcopy(plan['research_organization']), knowledge=knowledge,
        config=limits, confirmation_seed=case['confirmation_seed'], source_task=False,
        paid_trace_references=paid)
    lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
    restored = agent.restore_research(previous, failed, lab, confirmation_seed=case['confirmation_seed'])
    agent.costs = {key: original[key]+value for key, value in paid_delta.items()}
    agent.config = limits
    agent._organization_state['automatic_library_path'] = LIBRARY_PATH
    agent.knowledge['finite_native_author']['predictor_paths'] = list(material['models'])
    agent.knowledge['finite_native_author']['question']['frozen_library_path'] = LIBRARY_PATH
    for question in agent.questions:
        if 'frozen_library_path' in question:
            question['frozen_library_path'] = LIBRARY_PATH
    pool = {item['id']: item for item in resolved_theories(agent)}
    new_id = next(item['id'] for item in registration['arms']['keep']['candidate_versions']
        if item not in registration['arms']['remove']['candidate_versions'])
    execution_key = 'relation_transfer:frozen_recall:native_public_response@1:'+new_id
    basis = agent.theory_consolidation['executions'][execution_key]['request_basis']
    component_ids = basis['component_theory_ids']
    parent = deepcopy(pool[registration['material']['parent_theory_id']])
    parent['program_source_ref'] = deepcopy(registration['material']['parent_Program_ref'])
    parent['_contract_source'] = deepcopy(material['parent_contract'])
    owner = next(q for q in agent.questions if q['id']=='native_public_response')
    # Object0 committed its internal cursor before object1 failed. The outer
    # pass had not committed; resume its actual next object rather than replay0.
    if resume_phase == 'first_recall':
        owner['frozen_recall_execution'].setdefault('cursor', {})['next_candidate_id'] = (
            'consolidated:0:relation:revision:1')
    first_spent = paid_delta['compute_cost']-2  # two completed public readouts
    arguments = {key: deepcopy(basis.get(key)) for key in (
        'library_path', 'parent_theory_id', 'fitting', 'evidence_checks', 'allocation', 'candidate_id')}
    arguments['library_path'] = LIBRARY_PATH
    _retain_scene_score_index(failed, workspace, paid_rows=len(agent.rows))
    agent._prepare_program_state()
    receipt = {'status':'resumed_actual_failed_scene','source':str(failed), **restored,
        'retained_theory_count':len(agent.theory_index), 'retained_paid_rows':len(agent.rows),
        'prior_costs_including_completed_readouts':deepcopy(agent.costs),
        'first_allocation_remaining':(registration['budget']['per_consumption_compute']-first_spent
            if resume_phase=='first_recall' else 0), 'resume_phase':resume_phase,
        'original_500k_cap':limits.compute_budget,'new_fits':0,'new_scores':0}
    write_json_checkpoint(workspace/'resume_validation.json',receipt,ensure_ascii=False)
    return {'agent':agent,'lab':lab,'models':material['models'],'parent':parent,
        'retained_components':[pool[key] for key in component_ids],
        'source_component_theory_ids':component_ids,'transfer_arguments':arguments,
        'first_allocation_remaining':receipt['first_allocation_remaining'],
        'resume_phase':resume_phase,
        'prior_continuation_costs':paid_delta,'original_starting_costs':original,
        'validation':receipt}


def pin_registered_development_work(scene, registration, original_execution):
    """Restore the registered search's exact versions while retaining later work."""
    from .question_routing import _resolved_theory_versions
    from .theory_consolidation import _same_relation_work, _shared_relation_progress

    agent = scene['agent']
    original_basis = original_execution['request_basis']
    identifier = original_basis['candidate_id']
    new_ids = {item['id'] for item in registration['arms']['keep']['candidate_versions']} - {
        item['id'] for item in registration['arms']['remove']['candidate_versions']}
    if new_ids != {identifier}:
        raise ValueError('original development execution is not the registered object difference')
    execution_key = f'relation_transfer:frozen_recall:native_public_response@1:{identifier}'
    versions = {(item['id'], item.get('version', 1)): item
                for item, _, _ in _resolved_theory_versions(agent)}
    components, pinned = [], []
    for position, reference in enumerate(original_basis['components']):
        key = reference['id'], reference.get('version', 1)
        found = versions.get(key)
        if found is None or not found.get('program'):
            raise ValueError('registered development component Program version is unavailable')
        if found.get('source_ref') != reference.get('source_ref'):
            raise ValueError('registered development component source premise changed')
        components.append(deepcopy(found))
        current_version = agent.theory_index[reference['id']].get('version', 1)
        if current_version != key[1]:
            pinned.append({'position': position, 'id': key[0], 'version': key[1],
                           'current_pool_version_preserved': current_version})

    parent = scene['parent']
    arguments = deepcopy(scene['transfer_arguments'])
    arguments.update(candidate_id=identifier, theory_id=execution_key.removeprefix('relation_transfer:'),
        library_path=original_basis['library_path'], parent_theory_id=parent['id'],
        component_theory_ids=deepcopy(original_basis['component_theory_ids']),
        role_signature=deepcopy(parent['role_signature']), output_unit=parent['output_unit'],
        context_ref=deepcopy(parent['context_ref']))
    basis = {key: deepcopy(arguments.get(key)) for key in (
        'component_theory_ids', 'component_bindings', 'role_signature', 'output_unit',
        'context_ref', 'evidence_checks', 'allocation', 'parent_theory_id', 'fitting')}
    basis.update(library_path=arguments['library_path'], candidate_id=identifier,
        candidate_version=original_basis['candidate_version'],
        parent={key: deepcopy(parent.get(key)) for key in ('id', 'version', 'source_ref', 'source_status')},
        components=[{key: deepcopy(item.get(key)) for key in
            ('id', 'version', 'source_ref', 'compatible', 'source_status')} for item in components])
    if parent.get('_contract_source'):
        basis['parent']['contract_source'] = deepcopy(parent['_contract_source'].get(
            'source_ref', parent['_contract_source']))
    if not _same_relation_work(agent, original_basis, basis):
        raise ValueError('exact historical components do not recover the registered parent problem')

    executions = agent.theory_consolidation.setdefault('executions', {})
    current = executions.get(execution_key)
    preserved_key = None
    if current is not None and not _same_relation_work(agent, current['request_basis'], original_basis):
        prefix = execution_key + ':misbound:'
        suffix = 0
        while prefix + str(suffix) in executions:
            suffix += 1
        preserved_key = prefix + str(suffix)
        executions[preserved_key] = deepcopy(current)
    executions[execution_key] = deepcopy(original_execution)
    candidate = {'id': identifier, 'version': original_basis['candidate_version']}
    recovered, _ = _shared_relation_progress(agent, arguments, candidate, basis,
                                            execution_key, executions)
    if recovered is None or recovered.get('cursor') != original_execution.get('cursor'):
        raise ValueError('ordinary transfer did not select the original development cursor')
    scene.update(retained_components=components,
        source_component_theory_ids=deepcopy(original_basis['component_theory_ids']),
        transfer_arguments=arguments)
    save_theory_library(agent)
    cursor = recovered.get('cursor') or {}
    receipt = {'status': 'registered_development_versions_and_cursor_pinned',
        'execution_key': execution_key, 'preserved_misbound_execution_key': preserved_key,
        'original_component_count': len(components), 'pinned_historical_versions': pinned,
        'pinned_historical_version_count': len(pinned),
        'cursor_layer': cursor.get('layer'), 'cursor_offset': cursor.get('offset'),
        'pending_fit_positions': len(cursor.get('pending_fits', [])),
        'ordinary_transfer_same_work_verified': True,
        'global_current_versions_preserved': True, 'existing_transfers_preserved': True,
        'existing_costs_preserved': True, 'new_fits': 0, 'new_scores': 0,
        'new_predictions': 0, 'new_observations': 0}
    write_json_checkpoint(agent.workspace / 'pinned_development_work.json', receipt, ensure_ascii=False)
    return receipt


def quote_joint_paid_scores(scene):
    """Quote the ordinary comparator's unpaid rows without scoring or saving."""
    from copy import copy
    from unittest.mock import patch
    from . import temporal_operator_agent as scoring
    from .code_agent import python_model_context_matches
    from .python_research_routing import validated_candidates

    agent = scene['agent']
    models = list(dict.fromkeys([*scene['models'], *agent.candidate_paths,
                                *validated_candidates(agent)]))
    rows = len(agent.rows)
    index = agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    if not index.is_file():
        return {'models': len(models), 'paid_rows': rows, 'reused_score_count': 0,
                'unpaid_score_count': len(models),
                'unpaid_scoring_compute': rows * len(models),
                'missing_models': models, 'new_scores': 0, 'new_fits': 0,
                'compute_cost': 0, 'python_execution_seconds': 0.}

    # Build exactly the live comparator's scope on a shallow view. A new scope
    # means a miss; suppress only its writes while keeping the real predicate.
    view = copy(agent)
    view._active_question_id = scene.get('research_question_ref', {}).get(
        'question_id', 'native_public_response')
    with patch.object(scoring, 'write_text_incrementally'), \
            patch.object(scoring, 'write_json_checkpoint'):
        comparison = view._paid_python_score_scope(scene['lab'].public_spec())
    reused, missing = [], []
    for model in models:
        found = None
        for reference, assessment in reversed(comparison['assessments'].get(model, [])):
            if comparison['scopes'][assessment['scope_ref']['file']] != comparison['basis']:
                continue
            context = assessment['execution_context']
            if python_model_context_matches(agent.workspace, model,
                                             context['workspace'], context['model']):
                found = reference, assessment
                break
        if found is None:
            found = scoring._matching_literal_score(agent.workspace, model, comparison)
        if found is None:
            missing.append(model)
        else:
            reused.append({'model': model, 'execution_basis_ref': found[0]})
    return {'models': len(models), 'paid_rows': rows,
            'reused_score_count': len(reused), 'unpaid_score_count': len(missing),
            'unpaid_scoring_compute': rows * len(missing),
            'paid_execution_refs': reused, 'missing_models': missing,
            'new_scores': 0, 'new_fits': 0, 'compute_cost': 0,
            'python_execution_seconds': 0., 'retention_decisions_replayed': False}


def _joint_exposed_diagnostic_programs(agent, manifest):
    """Reconstruct recorded outputs, never solve a template or replay a cursor."""
    from .abstractions import AbstractTemplate, _bind_constants
    from .language import Program
    from .theory_consolidation import _snapshot_index

    output_ref = manifest['constructor_outputs_ref']
    result_ref = {**output_ref, 'json_path': output_ref['json_path'][:-1]}
    result = _read(result_ref, agent.workspace)
    outputs = result['outputs']
    representatives = json.loads(Path(result['representative_outputs_ref']).read_text())
    by_index = {item['index']: item for item in representatives}
    references = []
    for item in outputs:
        reference = item['fitting'].get('fit_reuse_output_ref')
        if reference:
            references.append({**reference, 'json_path': reference['json_path'][:-2]})
    transfers = iter(load_source_values(references, agent.workspace))
    programs = []
    for item in outputs:
        if item['fitting'].get('fit_reuse_output_ref'):
            transfer = next(transfers)['value']
            fitting = transfer['binding']['fitting']
            if any(fitting.get(key) != item['fitting'].get(key) for key in
                   ('parameters', 'parameter_paths', 'fitting_method', 'training_loss')):
                raise ValueError('recorded diagnostic fit changed')
            identity = transfer['theory_ref']
            indexed = agent.theory_index[identity['id']]
            theory = next(value for value in (indexed, *indexed.get('history', []))
                          if value.get('version', 1) == identity.get('version', 1))
            if not theory.get('program'):
                theory = _snapshot_index({identity['id']: {
                    key: value for key, value in theory.items() if key != 'history'}},
                    {'questions': agent.questions, 'relation_research': agent.relation_research,
                     'events': agent.events}, {'file': str(agent.workspace / 'research_state.json'),
                     'json_path': ['theory_index']})[identity['id']]
            program = Program.from_dict(theory['program'])
        else:
            recorded = by_index[item['index']]
            if recorded['fitting'] != item['fitting']:
                # The compact summary replaces only the long evidence list by
                # its address; fitting coordinates and values must agree.
                if any(recorded['fitting'].get(key) != item['fitting'].get(key) for key in
                       ('parameters', 'parameter_paths', 'fitting_method', 'training_loss')):
                    raise ValueError('diagnostic representative changed')
            program = Program.from_dict(recorded['program'])
        if program.expanded().size != item['nodes']:
            raise ValueError('diagnostic Program size changed')
        programs.append({'group': 'constructor', 'index': item['index'], 'program': program,
            'output_ref': {**output_ref, 'json_path': [*output_ref['json_path'], item['index']]}})
    fixed = _read(manifest['fixed_fit_outputs_ref'], agent.workspace)
    template = AbstractTemplate.from_dict(_read(manifest['fixed_fit_template_ref'], agent.workspace))
    for index, item in enumerate(fixed):
        programs.append({'group': 'fixed_fit', 'index': index,
            'program': template.instantiate({}, item['parameters']),
            'output_ref': {**manifest['fixed_fit_outputs_ref'],
                'json_path': [*manifest['fixed_fit_outputs_ref']['json_path'], index]}})
    if len(outputs) != manifest['constructor_output_count'] or len(fixed) != manifest['fixed_fit_output_count']:
        raise ValueError('registered diagnostic bank is incomplete')
    templates, recovered_count = {}, 0
    for group, reference in enumerate(manifest.get('recovered_fit_output_refs', [])):
        for index, item in enumerate(_read(reference, agent.workspace)):
            recovery = item['recovery']
            template_ref = recovery['template_ref']
            key = template_ref['file'], tuple(template_ref.get('json_path', []))
            if key not in templates:
                templates[key] = AbstractTemplate.from_dict(_read(template_ref, agent.workspace))
            template = templates[key]
            paths = [tuple(path) for path in recovery['parameter_paths']]
            values = recovery['parameters']
            if (len(paths) != len(values) or values != item['parameters']
                    or not set(paths).issubset(template.parameter_paths)):
                raise ValueError('recorded diagnostic coordinates changed')
            bindings = dict(zip(paths, values))
            if 'fixed_threshold_path' in recovery:
                path = tuple(recovery['fixed_threshold_path'])
                if path not in template.parameter_paths or path in bindings:
                    raise ValueError('recorded diagnostic threshold changed')
                bindings[path] = recovery['fixed_threshold']
            if set(bindings) != set(template.parameter_paths):
                raise ValueError('recorded diagnostic coordinates are incomplete')
            programs.append({'group': f'recovered_fit_{group}', 'index': index,
                'program': _bind_constants(template.pattern, bindings),
                'output_ref': {**reference, 'json_path': [*reference['json_path'], index]}})
            recovered_count += 1
    if recovered_count != manifest.get('recovered_fit_output_count', 0):
        raise ValueError('registered recovered diagnostic bank is incomplete')
    return programs


def materialize_joint_exposed_bank(agent, registration, *, shared_material=None):
    """Expose the complete named bank with original source fitting contexts.

    Later REMOVE models keep absolute source paths, which are their namespace.
    Restored KEEP models keep their existing paths. The ordinary scorer checks
    every imported paid reference against its exact question and model context.
    """
    from .checkpoint_storage import write_text_incrementally
    from .temporal_operator_agent import export_temporal_program
    from .theory_consolidation import _read_index

    manifest_ref = registration['material']['complete_common_bank_manifest_ref']
    manifest = _read(manifest_ref, agent.workspace)
    shared = shared_material if shared_material is not None else {}
    if '_joint_exposed_export_rosters' not in shared:
        shared['_joint_exposed_export_rosters'] = [item['value'] for item in load_source_values(
            [item['ref'] for item in manifest['later_export_rosters']], agent.workspace)]
    retained_models = agent.knowledge['finite_native_author'].get('predictor_paths', [])
    models = _read(manifest['base_model_roster_ref'], agent.workspace)
    if len(models) != manifest['base_count']:
        raise ValueError('registered base model bank is incomplete')
    source = Path(registration['material']['same_closed_scene']).resolve()
    sources, native_archives, additional_sources = [], [], []
    for roster, paths in zip(manifest['later_export_rosters'], shared['_joint_exposed_export_rosters']):
        if len(paths) != roster['count']:
            raise ValueError('registered later export bank is incomplete')
        origin = Path(roster['workspace']).resolve()
        native_archives.append({'namespace': roster['namespace'],
            'research_state_ref': {'file': str(origin / 'research_state.json'), 'json_path': []}})
        if origin == source:
            if paths != agent.candidate_paths:
                raise ValueError('restored KEEP candidate paths changed')
            models.extend(paths)
        else:
            models.extend(str(origin / path) for path in paths)
            additional_sources.append({'file': str(origin / 'research_state.json'), 'json_path': []})
        sources.append(Path(roster['paid_score_index']) / '_evaluated_model/assessments.json')

    # Archive admission keeps every actual native version. It does not decide
    # whether those versions form independent induction support.
    if '_joint_exposed_bank_material' not in shared:
        theories, archives = [], []
        for reference in additional_sources:
            index_ref = {**reference, 'json_path': ['theory_index']}
            snapshot = _read_index(index_ref, agent.workspace)
            for identifier, theory in snapshot.items():
                if not any(item.get('program') for item in (theory, *theory.get('history', []))):
                    continue  # The complete opaque Python bank is already retained by KEEP.
                theories.append({**theory, 'id': f'{reference["file"]}::{identifier}'})
            source_workspace = Path(reference['file']).parent
            archives.append({'research_state_ref': reference, 'theory_index_ref': index_ref,
                'original_failures_and_transfers_ref': {
                    'file': str(source_workspace / LIBRARY_PATH), 'json_path': ['transfers']},
                'checkpoint_costs_ref': {**reference, 'json_path': ['costs']},
                'post_checkpoint_costs_ref': {
                    'file': str(source_workspace / 'final_costs.json'), 'json_path': []}})
        shared['_joint_exposed_bank_material'] = {
            'native': {'theories': theories, 'archives': archives},
            'diagnostics': _joint_exposed_diagnostic_programs(agent, manifest)}
    exposed = shared['_joint_exposed_bank_material']
    native = exposed['native']
    missing_theories, reused_theories = [], []
    for theory in native['theories']:
        retained = agent.theory_index.get(theory['id'])
        if retained is None:
            missing_theories.append(theory)
        elif retained != theory:
            raise ValueError('retained joint source namespace differs from its original version or contract')
        else:
            reused_theories.append(theory)
    register_theories(agent, missing_theories)
    for theory in missing_theories:
        retained = agent.theory_index[theory['id']]
        retained['version'] = theory.get('version', 1)
        if theory.get('history'):
            retained['history'] = deepcopy(theory['history'])
    agent.knowledge['joint_exposed_native_archive'] = deepcopy(native['archives'])

    target = agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    index = json.loads(target.read_text()) if target.exists() else {
        'scopes': [], 'assessments': [], 'contexts': []}
    seen = {name: {(item['file'], tuple(item.get('json_path', [])))
                   for item in index[name]} for name in ('scopes', 'assessments')}
    for source_index in sources:
        retained = json.loads(source_index.read_text())
        for reference in retained['scopes']:
            if len(_read(reference, agent.workspace)['rows']) > registration['material']['paid_rows']:
                raise ValueError('joint bank includes later unpaid score rows')
        for name in seen:
            for reference in retained[name]:
                key = reference['file'], tuple(reference.get('json_path', []))
                if key not in seen[name]:
                    seen[name].add(key)
                    index[name].append(reference)
    write_json_checkpoint(target, index, indent=None)

    question = next(item for item in agent.questions if item['id'] == 'native_public_response')
    feature_names = tuple(question['theory_target']['role_signature'])
    diagnostics, reused_diagnostics = [], 0
    directory = Path('_experiment_models/joint_exposed_diagnostics/_evaluated_model')
    (agent.workspace / directory).mkdir(parents=True, exist_ok=True)
    for item in exposed['diagnostics']:
        model = str(directory / f'{item["group"]}_{item["index"]:04d}.py')
        code = export_temporal_program(item['program'], feature_names=feature_names)
        destination = agent.workspace / model
        if destination.is_file():
            if destination.read_text() != code:
                raise ValueError('retained joint diagnostic model differs from its original Program')
            reused_diagnostics += 1
        else:
            write_text_incrementally(destination, code)
        models.append(model)
        if model not in agent.candidate_paths:
            agent.candidate_paths.append(model)
        diagnostics.append({'model': model, 'output_ref': item['output_ref'],
            'nodes': item['program'].expanded().size, 'support_updated': False})
    if len(models) != manifest['named_entry_count_before_exact_program_and_context_reuse']:
        raise ValueError('joint named comparison bank differs from registration')
    manifest_models = set(models)
    retained_additions = [model for model in retained_models if model not in manifest_models]
    models = list(dict.fromkeys([*retained_models, *models]))
    missing = [model for model in models if not (agent.workspace / model).is_file()]
    if missing:
        raise ValueError(f'joint model bank has {len(missing)} unavailable models')
    agent.knowledge['finite_native_author']['predictor_paths'] = models
    receipt = {'status': 'complete_exposed_model_bank_materialized_without_execution',
        'manifest_ref': deepcopy(manifest_ref), 'named_models': len(models),
        'manifest_named_entries': manifest['named_entry_count_before_exact_program_and_context_reuse'],
        'retained_completed_scene_additions': retained_additions,
        'base_models': manifest['base_count'], 'diagnostic_models': diagnostics,
        'reused_diagnostic_models': reused_diagnostics,
        'later_native_archives_retained': native_archives,
        'additional_current_native_theories': len(missing_theories),
        'reused_current_native_theories': len(reused_theories),
        'additional_historical_native_versions': sum(len(item.get('history', [])) for item in missing_theories),
        'reused_historical_native_versions': sum(len(item.get('history', [])) for item in reused_theories),
        'additional_original_failure_archives': len(native['archives']),
        'paid_score_scopes': len(index['scopes']), 'paid_assessment_refs': len(index['assessments']),
        'paid_reference_context_verified_by_ordinary_scorer': True,
        'new_fits': 0, 'new_predictions': 0, 'new_scores': 0, 'support_updated': False,
        'scientific_cursors_changed': False}
    agent.knowledge['joint_exposed_model_bank'] = deepcopy(receipt)
    write_json_checkpoint(agent.workspace / 'joint_bank_validation.json', receipt)
    return receipt
