"""Restore the one registered paid scene; this file never starts research."""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
from time import monotonic

REPOSITORY = Path(__file__).resolve().parent.parent / 'source_at_run'
sys.path[:0] = [str(REPOSITORY / 'src'), str(REPOSITORY)]
from hegel_machine.cognition.anonymous_research_archive import resolve_workspace_archive
from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint, _JsonSource
from hegel_machine.cognition.code_agent import archived_paid_trace_references, _model_context_ignored
from hegel_machine.cognition.common_archive_inquiry import retained_checkpoint_result
from hegel_machine.cognition.experiment_runtime import checkpoint_execution
from hegel_machine.cognition.finite_native_author import (
    export_finite_native_author, paid_experiment_record_indices, select_actions,
)
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.structural_inquiry import _native_continuation
from hegel_machine.cognition.theory_consolidation import (
    LIBRARY_PATH, _read_library, _merge_library_document, _save as save_library,
)
from scripts import run_anonymous_revision_inquiry as study
from scripts.prepare_pr3_source_successor import _prepared_callbacks, _retain_scene_score_index

ROOT = Path(__file__).resolve().parent.parent
BUDGET = ROOT / 'prepared/budget_registration.json'
SOURCE = Path('/mnt/hegel-snapshot-cow/pr3_v7_condition_successor_recovery_20261026/target/case_002/pr3')
LEARNED = Path('/mnt/hegel-snapshot-cow/pr3_v7_method_feedback_after_condition_20261026/feedback/learner/shared/theory_library.json')
METHOD = 'consolidated:0:method:0'
RESTORE_FIELDS = (
    'costs', 'paid_observations', 'events', 'model', 'online_questions', 'revision_proposals',
    'inherited_files', 'method_results', 'program_contexts', 'experiment_comparisons',
    'observable_models', 'conditional_models', 'model_dependencies', 'research_candidates',
    'research_input_bindings', 'relation_research', 'theory_index', 'research_mission',
    'research_agenda', 'theory_consolidation', 'research_organization', 'llm_inquiries',
    'inquiry_stack', 'evaluated_model', 'fresh_confirmation', 'method_path',
)


def authoritative_saved_scene(source):
    previous = read_json_checkpoint(source / 'training.json', keys=RESTORE_FIELDS)
    # Restoration takes executable files from their retained workspace. It only
    # needs the result's manifest, not a second decoded portable files archive.
    with ExitStack() as opened:
        saved = _JsonSource(source / 'training.json', opened)
        outgoing = saved.members(keys={'outgoing_knowledge'})['outgoing_knowledge']
        node = saved.members(outgoing, keys={'manifest'})['manifest']
        manifest = json.loads(saved.text[node[1]:node[2]])
    # The existing restore consumes only these keys, then refreshes each file
    # from disk. Keep the ordinary authored file view without decoding copies
    # of its text inside the final portable result.
    names = set(previous['inherited_files'])
    for directory, subdirs, files in os.walk(source):
        base = Path(directory)
        subdirs[:] = [name for name in subdirs if name not in {'_evaluated_model', '_submitted_models'}
            and not _model_context_ignored(base / name, source)]
        names.update(str((base / name).relative_to(source)) for name in files
            if not _model_context_ignored(base / name, source))
    previous['outgoing_knowledge'] = {'manifest': manifest, 'files': dict.fromkeys(sorted(names))}
    thin_rounds = any(isinstance(q.get('native_condition_research'), dict)
        for q in previous['online_questions'])
    fields = ['operator_results', 'exported_candidate_paths', 'online_research', 'costs']
    if thin_rounds:
        fields.append('questions')
    state = read_json_checkpoint(source / 'research_state.json', keys=fields)
    central_costs = dict(state['costs'])
    canonical = {q['id']: q.get('native_condition_research') for q in state.get('questions', [])}
    for question in previous['online_questions']:
        rounds = question.get('native_condition_research')
        if isinstance(rounds, dict) and rounds.get('full_content_file') == 'research_state.json':
            question['native_condition_research'] = canonical[question['id']]
    state['questions'] = previous['online_questions']
    state['revisions'] = previous['revision_proposals']
    for field in ('costs', 'evaluated_model', 'method_results', 'program_contexts',
            'experiment_comparisons', 'observable_models', 'conditional_models', 'model_dependencies',
            'research_candidates', 'research_input_bindings', 'relation_research', 'theory_index',
            'research_mission', 'research_agenda', 'theory_consolidation', 'research_organization'):
        if field in previous:
            state[field] = previous[field]
    previous = retained_checkpoint_result(previous, state)
    gc.collect()
    return previous, state, central_costs


def prepare_scene(workspace, budget, *, source=SOURCE):
    workspace = Path(workspace)
    source = Path(source)
    if workspace.exists():
        raise ValueError('preserve this restoration rather than overwrite its accepted state')
    started = monotonic()
    previous, state, central_costs = authoritative_saved_scene(source)
    plan_path = (SOURCE.parent.parent / 'protocol.json' if source == SOURCE
                 else ROOT / 'prepared/current_plan.json')
    plan = read_json_checkpoint(plan_path)
    case = next(c for c in plan['cases'] if c['id'] == 'case_002')
    knowledge = read_json_checkpoint(SOURCE.parent.parent / 'initial_archive.json')
    while set(knowledge) == {'source_ref'}:
        knowledge = load_source_values([knowledge['source_ref']], SOURCE)[0]['value']
    knowledge = resolve_workspace_archive(knowledge)
    limits = replace(study.InquiryLLMConfig(**plan['limits']),
        compute_budget=budget['registered_unit']['known_cumulative_compute_ceiling_for_this_unit'],
        execution_seconds=previous['costs']['python_execution_seconds']
            + budget['registered_unit']['active_wall_limit_seconds'])
    factory, _ = _prepared_callbacks(plan)
    paid = archived_paid_trace_references(knowledge.get('files', {}),
        workspace.parent / 'reference_archive', source_workspace=source)
    agent = factory(None, workspace, arm='pr3', research_organization=deepcopy(plan['research_organization']),
        knowledge=knowledge, config=limits, confirmation_seed=case['confirmation_seed'],
        source_task=False, paid_trace_references=paid)
    lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
    restored = agent.restore_research(previous, source, lab,
        confirmation_seed=case['confirmation_seed'], retained_state=state)
    old_costs = dict(previous['costs'])
    if agent.costs != old_costs or len(agent.rows) != 145:
        raise ValueError('the actual final paid account or rows changed during restoration')
    library = agent.theory_consolidation
    executions, transfers = library['executions'], library['transfers']
    original_candidates = dict(library['candidates'])
    counts = {'executions': len(executions), 'transfers': len(transfers),
        'candidate_count': len(original_candidates), 'theory_count': len(agent.theory_index)}
    learned = _read_library({'file': str(LEARNED), 'json_path': []}, agent.workspace)
    _merge_library_document(library, {'candidates': {METHOD: learned['candidates'][METHOD]}})
    del learned
    agent._theory_library_candidate_versions = None
    agent._theory_library_entry_references = None
    save_library(agent)
    agent._organization_state['automatic_library_path'] = LIBRARY_PATH
    agent._organization_registration['pr3_library_path'] = str(agent.workspace / LIBRARY_PATH)
    plan['research_organization']['pr3_library_path'] = str(agent.workspace / LIBRARY_PATH)
    agent.knowledge['theory_consolidation'] = {'file': str(agent.workspace / LIBRARY_PATH), 'json_path': []}
    agent.knowledge['finite_native_author']['question']['frozen_library_path'] = LIBRARY_PATH
    for question in agent.questions:
        if ('frozen_library_path' in question or 'frozen_recall_execution' in question
                or question.get('theory_target', {}).get('revision_fitting')):
            question['frozen_library_path'] = LIBRARY_PATH
    path = plan['author']['path']
    agent._save(path, export_finite_native_author())
    scores = _retain_scene_score_index(source, workspace)
    agent._prepare_program_state()
    question = next(q for q in agent.questions
        if q['id'] == agent.knowledge['finite_native_author']['question']['id'])
    latest = question['native_condition_research'][-1]
    quote = _native_continuation(agent, question, latest)
    actions = [deepcopy(agent.observations[i]['action']) for i in (3, 4)]
    matches = paid_experiment_record_indices(actions, agent.observations, lab.public_spec(),
        question, agent.knowledge['finite_native_author'], prediction_rows=lab.prediction_rows)
    before = os.getcwd()
    try:
        os.chdir(workspace)
        proposal = select_actions(lab.public_spec(), agent.observations, agent.knowledge)
    finally:
        os.chdir(before)
    acceptance = {'status': 'zero_science_scene_restored', 'source_workspace': str(source),
        'source_training_ref': {'file': str(source / 'training.json'), 'json_path': []},
        'original_central_costs': central_costs, 'restored_authoritative_final_costs': old_costs,
        'finite_author_cursor': deepcopy(agent._organization_state['finite_author']),
        'library_before': counts,
        'library_after': {'executions': len(library['executions']), 'transfers': len(library['transfers']),
            'candidate_count': len(library['candidates']), 'theory_count': len(agent.theory_index)},
        'execution_map_preserved': executions is library['executions'],
        'transfer_history_preserved': transfers is library['transfers'],
        'other_candidates_unchanged': all(original_candidates[k] == library['candidates'][k]
            for k in original_candidates if k != METHOD),
        'method_condition_version': library['candidates'][METHOD]['version'],
        'current_library_path': agent._organization_state['automatic_library_path'],
        'retained_score_index': scores, 'native_quote': quote,
        'paid_pair_original_record_indices': matches,
        'selected_action': {**{key: deepcopy(proposal[key]) for key in ('action', 'stop') if key in proposal},
            'operator': proposal.get('arguments', {}).get('operator'),
            'model_count': len(proposal.get('arguments', {}).get('models', []))},
        'selected_action_ref': {'file': str(workspace.parent / 'selected_action.json'), 'json_path': []},
        'costs_unchanged': agent.costs == old_costs,
        'restoration_engineering_wall_seconds': monotonic() - started,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'new_scientific_compute': 0}
    if not acceptance['other_candidates_unchanged'] or matches != [3, 4] or not acceptance['costs_unchanged']:
        raise ValueError('restoration acceptance failed')
    write_json_checkpoint(workspace.parent / 'selected_action.json', proposal, ensure_ascii=False)
    write_json_checkpoint(workspace.parent / 'restoration_acceptance.json', acceptance, ensure_ascii=False)
    if source == SOURCE:
        write_json_checkpoint(ROOT / 'prepared/current_plan.json', plan, ensure_ascii=False)
    with checkpoint_execution():
        write_json_checkpoint(workspace / 'training.json', agent.checkpoint_result('researching'),
            base=source / 'training.json', ensure_ascii=False)
    return {'agent': agent, 'lab': lab, 'plan': plan, 'case': case, 'acceptance': acceptance,
        'before_parent': deepcopy(agent._evaluated_summary()), 'before_costs': old_costs,
        'selected_action': proposal}


if __name__ == '__main__':
    budget = read_json_checkpoint(BUDGET)
    result = prepare_scene(ROOT / 'prepared/case_002/pr3', budget)
    agent, acceptance = result['agent'], result['acceptance']
    print(acceptance['status'], 'rows', len(agent.rows), 'condition', acceptance['method_condition_version'],
        'selector', acceptance['selected_action'].get('action'),
        acceptance['selected_action'].get('operator'), flush=True)
