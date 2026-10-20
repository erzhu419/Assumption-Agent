#!/usr/bin/env python3
"""Compile confirmed Luna research with the retained author/practice machinery.

This bounded development study keeps the ten retention cases and selects four
previously registered conditional targets. It does not restart the 56-case study.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.code_agent import (
    CodexCodeProvider, archived_paid_trace_references, evaluate_python_model,
)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, register_published_method
from hegel_machine.cognition.temporal_paired_practice import (
    practice_execution_summary, restore_paid_practice_summaries)
from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from scripts.run_anonymous_method_learning import (
    frozen_transfers, laboratory, score_frozen, trial_preparer)
from scripts.run_anonymous_method_revision import attach_online_research_experience
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json, final_checkpoint_alias
from scripts.run_python_method_learning_study import LIMIT_COSTS, usage_summary
from scripts.run_temporal_counterexample_learning import supply_counterexample


def read_json(path):
    return json.loads(Path(path).read_text())


def paid_source_repair_protocol(checkpoint):
    """Repair one unfinished paid source; this is not method publication."""
    checkpoint = Path(checkpoint).resolve()
    previous = read_json(checkpoint)
    workspace = checkpoint.parent
    study = workspace.parent.parent
    registered = read_json(study / 'protocol.json')
    case, = registered['cases']
    if case['id'] != workspace.parent.name:
        raise ValueError('the paid source must match its registered single case')
    if previous.get('model') and (previous.get('fresh_confirmation') or {}).get('passed'):
        raise ValueError('the source is already confirmed; retain its recorded result')
    for path in (workspace / METHOD_PATH, workspace / 'research_context.json',
                 study / 'initial_archive.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    limits = dict(observation_budget=4096, compute_budget=2_000_000,
        execution_seconds=7200., run_timeout_seconds=300.,
        confirmation_rows=96, confirmation_mse_threshold=1e-6,
        max_model_calls=0, token_budget=0)
    return {'study': 'luna_paid_source_procedure_repair',
        'source_checkpoint': str(checkpoint), 'source_study': str(study),
        'source_workspace': str(workspace),
        'initial_archive': str(study / 'initial_archive.json'),
        'model': registered['model'], 'provider': deepcopy(registered['provider']),
        'practice_cases': {case['id']: deepcopy(case)},
        'author_limits': {**limits, 'confirmation_rows': 0,
                          'max_model_calls': 24, 'token_budget': 12_000_000},
        'limits': limits, 'source_costs': deepcopy(previous['costs']),
        'source_status': previous['status'],
        'source_confirmation': deepcopy(previous.get('fresh_confirmation')),
        'runtime_snapshot': sorted({str(path.relative_to(ROOT)) for path in
            (ROOT / 'src/hegel_machine').rglob('*.py')} | {
            str(path.relative_to(ROOT)) for path in (ROOT / 'scripts').glob('*.py')}),
        'scope': 'One source author edits the retained executable research method and '
            'practices it on the same paid source, retaining subsequent paid results. '
            'The program must deliver its candidate before ordinary fresh confirmation. '
            'Source confirmation preserves a draft; it does not publish a method or run targets.',
        'accounting': 'Author limits cover only additional work. The restored source account '
            'and prior paid records remain separately referenced. No prefix is repurchased.'}


def run_paid_source_repair(plan, output, *, provider_factory):
    """Bind actual source work to the retained edit/practice author loop."""
    from hegel_machine.cognition.anonymous_method_learning import (
        SourceProcedureRepairLearner, paid_scene_trial_preparer)

    output = Path(output).resolve()
    if (output / 'protocol.json').exists():
        raise ValueError('preserve earlier attempts; use a new study directory')
    restored = read_json(plan['continued_author']) if plan.get('continued_author') else None
    checkpoint = Path(plan['source_checkpoint'])
    if restored and restored['practice_trials']:
        checkpoint = (Path(restored['workspace']) /
            restored['practice_trials'][-1]['directory'] / 'result.json')
    previous = read_json(checkpoint)
    scene = checkpoint.parent
    knowledge = read_json(plan['initial_archive'])
    method_source = (restored['outgoing_knowledge']['files'][METHOD_PATH]
        if restored else (scene / METHOD_PATH).read_text())
    knowledge.setdefault('files', {})[METHOD_PATH] = method_source
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', plan)
    # Reference the existing 40+ MB archive rather than exporting another copy.
    (output / 'initial_archive.json').symlink_to(Path(plan['initial_archive']).resolve())
    reference = output / 'reference_archive'
    if restored:
        previous_reference = Path(restored['workspace']).parent / 'reference_archive'
        if previous_reference.is_dir():
            reference.symlink_to(previous_reference.resolve(), target_is_directory=True)
    paid_traces = archived_paid_trace_references(knowledge.get('files', {}), reference,
                                                source_workspace=scene)
    for name in plan['runtime_snapshot']:
        path = output / 'source_at_run' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    author = SourceProcedureRepairLearner(provider_factory(output / 'author'), output / 'author',
        knowledge=knowledge, cases=plan['practice_cases'], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan['author_limits']),
        practice_limits=InquiryLLMConfig(**plan['limits']),
        paid_trace_references=paid_traces,
        prepare_trial=paid_scene_trial_preparer(knowledge, checkpoint, laboratory,
            paid_trace_references=paid_traces,
            retained_comparison_reports=plan.get('retained_comparison_reports', ())))
    if restored is not None:
        author.restore_practice(restored)
    if plan.get('research_brief'):
        # Keep the operator's evidence/task handoff distinct from Luna's source
        # and earlier paid findings. This uses the existing author task channel.
        author.common_interface = ('CURRENT OPERATOR RESEARCH BRIEF\n'
            + plan['research_brief']['text'] + '\n\n' + author.common_interface)
    (author.workspace / 'scene_source').symlink_to(scene, target_is_directory=True)
    initial_checkpoint = Path(plan['source_checkpoint']).resolve()
    initial_scene = initial_checkpoint.parent
    source_index = {'initial_scene': {
        'working_directory': str(initial_scene), 'training': str(initial_checkpoint),
        'observations': str(initial_scene / 'laboratory_observations.json'),
        'rows': str(initial_scene / 'observations.json'),
        'program_state': str(initial_scene / 'research_state.json'),
        'retained_method': str(initial_scene / METHOD_PATH)},
        'working_directory': 'scene_source',
        'training': 'scene_source/' + checkpoint.name,
        'observations': 'scene_source/laboratory_observations.json',
        'rows': 'scene_source/observations.json',
        'program_state': 'scene_source/research_state.json',
        'retained_method': 'scene_source/' + METHOD_PATH,
        'author_method': METHOD_PATH, 'status': previous['status'],
        'costs': deepcopy(previous['costs']),
        'evaluated_model': deepcopy(previous.get('evaluated_model')),
        'fresh_confirmation': deepcopy(previous.get('fresh_confirmation')),
        'scope': 'Actual unfinished source, read-only. These results do not establish '
            'a successful explanation. Edit the author method and inspect actual practices; '
            'later trial results retain the accumulated paid observations.'}
    source_text = json.dumps(source_index, indent=2)
    (author.workspace / 'scene_source.json').write_text(source_text)
    author.files['scene_source.json'] = source_text
    previous_trial_count = len(author.practice_trials)
    if restored is None:
        write_json(output / 'progress.json', {'stage': 'retained_method_source_practice'})
        case, = plan['practice_cases']
        baseline = author._practice(case)
        author.events.append({'step': -1, 'action': 'run',
            'arguments': {'operator': 'practice', 'case': case}, 'feedback': baseline,
            'usage': {'backend': 'retained_method_source_practice', 'total_tokens': 0}})
    write_json(output / 'author/active.json', author.result('researching'))
    write_json(output / 'progress.json', {'stage': 'source_procedure_authoring'})
    learned = author.run(on_checkpoint=lambda result:
        write_json(output / 'author/active.json', result))
    write_json(output / 'author/training.json', learned)
    final_checkpoint_alias(output / 'author/training.json', output / 'author/active.json')
    latest = learned['practice_trials'][-1] if learned['practice_trials'] else None
    summary = {'study': plan['study'], 'status': learned['status'],
        'source_checkpoint': str(checkpoint),
        'initial_source_checkpoint': plan['source_checkpoint'],
        'source_prior_costs': previous['costs'],
        'author_costs': learned['costs'], 'author_usage': usage_summary(learned),
        'prior_author_phases': deepcopy(plan.get('prior_author_phases', [])),
        'prior_unknown_usage_calls': plan.get('prior_unknown_usage_calls', 0),
        'author_total_limits': deepcopy(plan.get('author_total_limits', plan['author_limits'])),
        'author_phase_limits': deepcopy(plan['author_limits']),
        'additional_author_calls': plan.get('additional_author_calls', 0),
        'additional_author_execution_seconds': plan.get('additional_author_execution_seconds', 0.),
        'additional_author_tokens': plan.get('additional_author_tokens', 0),
        'practice_trials': len(learned['practice_trials']),
        'new_practice_trials': len(learned['practice_trials']) - previous_trial_count,
        'latest_practice_result': str(author.workspace / latest['directory'] / 'result.json')
            if latest else None,
        'latest_practice_status': latest['status'] if latest else None,
        'fresh_confirmation': latest.get('fresh_confirmation') if latest else None,
        'method_draft': str(author.workspace / METHOD_PATH),
        'source_changed': (author.workspace / METHOD_PATH).read_text() != method_source,
        'method_published': False, 'target_episodes_run': 0, 'scope': plan['scope']}
    write_json(output / 'summary.json', summary)
    write_json(output / 'progress.json', {'stage': learned['status']})
    return summary


class SourceComparisonMethodLearner(AnonymousResearchMethodLearner):
    """Keep the same author loop and preserve either arm's unfinished draft."""

    def result(self, status):
        result = super().result(status)
        result['outgoing_knowledge'].setdefault('files', {})[METHOD_PATH] = (
            self.workspace / METHOD_PATH).read_text()
        return result


def protocol(online_study):
    study = Path(online_study).resolve()
    previous = read_json(study / 'protocol.json')
    summary = read_json(study / 'summary.json')
    source_trial, = summary['trials']
    if (source_trial['status'] != 'completed'
            or not source_trial['fresh_confirmation_passed']
            or not source_trial['terminal']['passed']):
        raise ValueError('compile only after actual source confirmation and terminal success')
    plan = deepcopy(previous['method'])
    old_ids = set(plan['old_target_ids'])
    plan['practice_cases'] = {name: case for name, case in plan['practice_cases'].items()
        if case.get('kind') != 'conditional_window' or case['stage'] in (1, 2)}
    plan['transfer_cases'] = [case for case in plan['transfer_cases']
        if case['id'] in old_ids or (case['parameter_pair'] < 2 and case['stage'] in (1, 2))]
    # Method compilation is a predictive transfer task. The earlier design
    # objective and its random/variance interventions are separate experiments.
    for name in ('design_experiments', 'prospective_design', 'conditional_design'):
        plan.pop(name, None)
    plan.update(study='luna_native_research_method_compilation',
        online_research_experience=str(Path(source_trial['archive']).parent),
        source_model=source_trial['model'],
        online_study=str(study), model='gpt-6-luna',
        provider={**plan['provider'], 'reasoning_effort': 'high'},
        original_summary=str(Path(plan['source']) / 'summary.json'),
        source_research_costs=source_trial['costs'],
        source_cost_scope='Cumulative online costs overlap prior_author_phases; do not sum them.',
        scope='One Luna method author; six unchanged practices and two conditional/window '
            'practices. Freeze only after the existing publication gate. Ten retention '
            'targets and four registered new-parameter targets; one final confirmation '
            'per target. Source predictions alone are a transfer control, not a Code baseline.',
        failure_policy='Keep every paid draft and failed practice. An unpublished method '
            'does not transfer. Original 10/10 and Astra56 stay unchanged.',
        transfer='Only the published program and original archive enter targets; the successful '
            'online source and practice fits are author-only. All research stops before scoring.',
        runtime_snapshot=sorted(set(previous['method']['runtime_snapshot']) |
            {str(path.relative_to(ROOT)) for path in
            (ROOT / 'src/hegel_machine').rglob('*.py')} | {
                'scripts/run_native_method_compilation.py',
                'scripts/run_anonymous_method_learning.py',
                'scripts/run_anonymous_method_revision.py',
                'scripts/experimental_design_control.py'}))
    plan['author_limits'].update(max_model_calls=32, token_budget=12_000_000)
    return plan


def restore_published_author(frozen_study):
    """Reconstruct all paid drafts from the existing compact publication index."""
    frozen_study = Path(frozen_study).resolve()
    plan = read_json(frozen_study / 'protocol.json')
    if (frozen_study / 'published_result.json').is_file():
        publication = read_json(frozen_study / 'published_result.json')
        source_study = Path(plan['frozen_publication']['source_study'])
    else:
        plan, publication, source_metadata = published_source(frozen_study)
        source_study = Path(source_metadata['source_study'])
    workspace = source_study / 'author'
    candidates = read_json(workspace / 'method_candidates.json')
    current, = [candidate for candidate in candidates if candidate.get('current')]
    if (publication['status'] != 'completed' or not publication.get('method_source')
            or (workspace / current['source_path']).read_text() != publication['method_source']):
        raise ValueError('restore the actual frozen publication and its indexed source')
    trials = {}
    for candidate in candidates:
        source = (workspace / candidate['source_path']).read_text()
        for trial in candidate['trials']:
            directory = trial['directory']
            if not (workspace / directory / 'result.json').is_file():
                raise FileNotFoundError(f'Retained paid trial unavailable: {directory}')
            trials[directory] = {**deepcopy(trial), 'method_source': source}
    # The candidate index deliberately omits large execution histories. Reuse
    # the author's already saved recent views so the first request does not
    # reopen a hundreds-of-megabytes practice just to rebuild its summary.
    decision = read_json(workspace / 'method_decision.json')
    recent = [decision.get(key) for key in (
        'latest_practice', 'latest_evaluated_practice', 'current_source_latest_failure',
        'current_source_latest_evaluation', 'current_source_latest_no_progress')]
    recent.extend((decision.get('latest_incumbent_comparison') or {}).get('practices', []))
    recent.extend((decision.get('failed_case_paid_comparison') or {}).values())
    for view in recent:
        if not isinstance(view, dict) or view.get('directory') not in trials:
            continue
        trial = trials[view['directory']]
        if view.get('source_path') != f"{trial['directory']}/{METHOD_PATH}":
            raise ValueError('saved paid-practice view must identify its indexed draft')
        for key in ('costs', 'execution_summary', 'no_progress'):
            if key in view:
                trial[key] = deepcopy(view[key])
    restored = {**deepcopy(publication), 'workspace': str(workspace),
        'practice_trials': sorted(trials.values(),
            key=lambda trial: int(Path(trial['directory']).name.removeprefix('trial_'))),
        'author_research_notes': (workspace / 'research_notes.md').read_text(),
        'events': []}
    supported = {trial['case'] for trial in restored['practice_trials']
        if trial['method_source'] == publication['method_source']
        and trial['status'] == 'completed'
        and (trial.get('fresh_confirmation') or {}).get('passed') is True}
    if supported != set(plan['practice_cases']):
        raise ValueError('retain published support for every earlier practice')

    metadata = {'frozen_study': str(frozen_study), 'source_study': str(source_study),
        'support_index': str(workspace / 'method_candidates.json'),
        'retained_paid_practices': len(trials), 'retained_drafts': len(candidates)}
    if not (frozen_study / 'published_result.json').is_file():
        metadata.update(publication_study=str(frozen_study),
            frozen_study=plan['published_revision']['frozen_study'])
    return plan, publication, restored, metadata


def prepare_revision_compilation(online_study, frozen_study, output):
    """Restore the published author and add one actually confirmed source practice."""
    online_study, frozen_study, output = map(
        lambda path: Path(path).resolve(), (online_study, frozen_study, output))
    online = read_json(online_study / 'protocol.json')
    source_trial, = read_json(online_study / 'summary.json')['trials']
    if (source_trial['status'] != 'completed'
            or not source_trial['fresh_confirmation_passed']
            or not source_trial['terminal']['passed']):
        raise ValueError('compile only after actual source confirmation and terminal success')
    source_case, = online['cases']
    if source_case['kind'] != 'nested_response':
        raise ValueError('this method revision uses the confirmed nested-response source')
    return _prepare_confirmed_source_compilation(
        online_study, frozen_study, output, online, source_trial, source_case)


def prepare_source_comparison_compilation(online_study, frozen_study, output, *, case_id, arm):
    """Compile one arm's confirmed source with the same retained author machinery."""
    online_study, frozen_study, output = map(
        lambda path: Path(path).resolve(), (online_study, frozen_study, output))
    if arm not in {'hegel', 'code_react'}:
        raise ValueError('source comparison requires hegel or code_react')
    online = read_json(online_study / 'protocol.json')
    source_trial, = [trial for trial in read_json(online_study / 'summary.json')['trials']
                    if trial['case'] == case_id and trial['arm'] == arm]
    if (source_trial['status'] != 'completed'
            or not source_trial['fresh_confirmation_passed']
            or not source_trial['terminal']['passed']):
        raise ValueError('compile only after actual source confirmation and terminal success')
    registered, = [case for case in online['cases'] if case['id'] == case_id]
    confirmation_seed = registered['compilation_confirmation_seed']
    if confirmation_seed in {registered['confirmation_seed'], registered['evaluation_seed']}:
        raise ValueError('compilation requires its separately registered confirmation stream')
    source_case = {key: deepcopy(value) for key, value in registered.items()
                   if key not in {'arms', 'compilation_confirmation_seed'}}
    source_case['confirmation_seed'] = confirmation_seed
    # Use the registered fresh-source allowance, not that investigation's spent
    # account; the shared helper retains every older case's effective ceilings.
    online = {**online, 'online_allowance': online.get('online_allowance', online['limits'])}
    plan = _prepare_confirmed_source_compilation(
        online_study, frozen_study, output, online, source_trial, source_case)
    for key in ('published_revision', 'design_experiments', 'prospective_design',
                'conditional_design'):
        plan.pop(key, None)
    plan.update(model=online['model'], provider=deepcopy(online['provider']),
        author_arm=arm, transfer_cases=[], author_only=True,
        source_comparison={'case': case_id, 'arm': arm,
            'baseline_publication': str(frozen_study),
            'compilation_confirmation_seed': confirmation_seed,
            'scope': 'Same initial archive, retained drafts, edit/practice behavior and publication '
                'requirements. Each author sees only its own new source investigation.'},
        scope='Matched source-to-method compilation with the retained publication as initial '
            'program. Both arms use the same author, practice limits and publication gate; '
            'only their own source research is attached. No transfer is scheduled.')
    plan['author_limits'].update(max_model_calls=32, token_budget=12_000_000,
                                 execution_seconds=36000.)
    plan['runtime_snapshot'] = sorted(set(plan['runtime_snapshot']) |
                                     set(online.get('runtime_snapshot', [])))
    return plan


def _prepare_confirmed_source_compilation(
        online_study, frozen_study, output, online, source_trial, source_case):
    """Reuse the publication restore and append a confirmed source's practice."""
    if (output / 'protocol.json').exists():
        raise ValueError('preserve earlier attempts; use a new study directory')

    plan, publication, restored, restoration = restore_published_author(frozen_study)
    prior_case_count = len(plan['practice_cases'])
    name = f"design_case_{prior_case_count:03d}"
    # A resumed online limit includes the already paid account. For a fresh
    # practice retain only that phase's allowance, while keeping every earlier
    # case's effective ceiling unchanged. trial_preparer takes the minimum of
    # the shared allowance and each case, so raising only the new case fails.
    allowance = online.get('online_allowance')
    if allowance:
        resources = ('observation_budget', 'compute_budget', 'execution_seconds')
        for case in plan['practice_cases'].values():
            ceilings = case.setdefault('limits', {})
            for key in resources:
                ceilings[key] = min(plan['limits'][key], ceilings.get(key, plan['limits'][key]))
        for key in resources:
            plan['limits'][key] = max(plan['limits'][key], allowance[key])
    plan['practice_cases'][name] = {key: deepcopy(value) for key, value in source_case.items()
        if key not in {'id', 'parameter_pair', 'evaluation_seed'}}
    if allowance:
        plan['practice_cases'][name]['limits'] = {
            **{key: allowance[key] for key in (*resources, 'confirmation_rows',
                'confirmation_mse_threshold', 'run_timeout_seconds') if key in allowance},
            **source_case.get('limits', {}), 'max_model_calls': 0, 'token_budget': 0}
    # Replacing the current source pointer must not drop the preceding online
    # investigation from the existing paid-phase ledger.
    previous_source = plan.get('online_research_experience')
    if previous_source:
        previous_result = str(Path(previous_source) / 'training.json')
        if not any(phase.get('result') == previous_result for phase in plan['prior_author_phases']):
            plan['prior_author_phases'].append({
                'phase': 'online_source_research', 'result': previous_result,
                'costs': deepcopy(plan['source_research_costs']),
                'cost_scope': plan['source_cost_scope']})
    for key in ('frozen_publication', 'paired_baseline', 'parameter_registration',
                'restore_accepted', 'repair_study', 'repair_development_cases', 'operator_practice',
                'complete_practices'):
        plan.pop(key, None)
    plan.update(author_only=True, continued_author=str(output / 'author_resume.json'),
        restored_publication=restoration,
        online_study=str(online_study),
        online_research_experience=str(Path(source_trial['archive']).parent),
        source_model=source_trial['model'], source_research_costs=source_trial['costs'],
        source_cost_scope='Confirmed online development costs and prior author phases are retained separately.',
        prior_author_phases=[*plan['prior_author_phases'], {
            'phase': 'published_research_method_author',
            'result': publication['full_result'], 'status': publication['status'],
            'costs': publication['costs'], 'usage': publication['author_usage']}],
        scope=f'One method-author phase after confirmed online structural revision. Preserve '
            f'all {prior_case_count} earlier practices and add the same source apparatus as one '
            f'development practice. Publication requires all {prior_case_count + 1} cases under '
            'the exact same source. '
            'No transfer is scheduled; earlier target results remain preserved evidence.',
        runtime_snapshot=sorted(set(plan['runtime_snapshot']) |
            {str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
            {'scripts/run_native_method_compilation.py', 'scripts/run_nonlinear_mechanism_revision.py',
             'scripts/run_anonymous_method_learning.py'}))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'author_resume.json', restored)
    return plan


def continuation_protocol(author_result, *, additional_author_calls=0,
                          additional_author_execution_seconds=0., additional_author_tokens=0):
    """Resume an exhausted compilation author, preserving drafts and target slots."""
    additions = {'additional_author_calls': additional_author_calls,
        'additional_author_execution_seconds': additional_author_execution_seconds,
        'additional_author_tokens': additional_author_tokens}
    for name, value in additions.items():
        if value < 0:
            raise ValueError(f'{name} must be nonnegative')
    path = Path(author_result).resolve()
    previous = read_json(path)
    study = Path(previous['workspace']).parent
    plan = read_json(study / 'protocol.json')
    if plan['study'] == 'luna_paid_source_procedure_repair':
        unresolved = previous.get('source_repair', {}) or {}
        if (previous['status'] not in {'provider_failed', 'runtime_interrupted',
                'model_call_budget_exhausted', 'token_budget_exhausted', 'source_repair_inconclusive',
                'python_execution_budget_exhausted'}
                or (unresolved and unresolved.get('status') != 'source_repair_inconclusive')
                or previous.get('method_source') is not None):
            raise ValueError('continue an actual unfinished source-repair author')
        phases = [*plan.get('prior_author_phases', []), {
            'phase': 'paid_source_procedure_repair', 'result': str(path),
            'status': previous['status'], 'costs': deepcopy(previous['costs']),
            'usage': usage_summary(previous)}]
        total = deepcopy(plan.get('author_total_limits', plan['author_limits']))
        total['max_model_calls'] += additional_author_calls
        total['execution_seconds'] += additional_author_execution_seconds
        total['token_budget'] += additional_author_tokens
        remaining = deepcopy(total)
        for limit, cost in LIMIT_COSTS:
            remaining[limit] = max(0, total[limit] - sum(
                phase['costs'][cost] for phase in phases))
        remaining['token_budget'] = max(0, total['token_budget'] - sum(
            phase['usage']['known_tokens'] for phase in phases))
        plan.update(continued_author=str(path), prior_author_phases=phases,
            author_total_limits=total, author_limits=remaining,
            **additions,
            prior_unknown_usage_calls=sum(phase['usage']['missing_usage_calls'] for phase in phases),
            continuation='Restore the same source author, its exact current draft and all paid '
                'practices through restore_practice. Keep remaining observation and computation '
                'allowances; enlarge cumulative author call, execution and token limits only '
                'by their explicitly recorded additions, then subtract every prior phase. '
                'Do not replay the baseline or repurchase the source prefix. Subtract known '
                'reported tokens while retaining unknown usage explicitly. No publication or targets.')
        return plan
    if any(additions.values()):
        raise ValueError('additional author allowances apply only to paid-source author continuation')
    summary = read_json(study / 'summary.json')
    if (plan['study'] != 'luna_native_research_method_compilation'
            or summary['method_published'] or previous.get('method_source')
            or previous['status'] not in {'model_call_budget_exhausted',
                                         'token_budget_exhausted', 'provider_failed'}):
        raise ValueError('continue an actual unpublished compilation author')
    # Restoring the accepted baseline was an explicit intervention in the prior
    # phase. Ordinary continuation keeps the author's latest scientific edits.
    plan.pop('restore_accepted', None)
    plan.update(continued_author=str(path),
        prior_author_phases=[*plan['prior_author_phases'], {
            'phase': 'native_method_compilation',
            'result': previous.get('full_result', str(path)),
            'status': previous['status'], 'costs': previous['costs'],
            'usage': summary['author_usage']}],
        continuation=f"New author allowance, unchanged {len(plan['practice_cases'])} practices "
            f"and {len(plan['transfer_cases'])} target slots. Restore every paid draft, failure "
            'and current source. Shared runtime '
            'tool-result recording and factual practice feedback are repaired; Luna must '
            'revise its own method. Previous costs are retained, not reset or recharged.')
    return plan


def repair_protocol(study):
    """Revise a published procedure using paid transfer experience as development."""
    study = Path(study).resolve()
    plan = read_json(study / 'protocol.json')
    summary = read_json(study / 'summary.json')
    previous = read_json(study / 'author/training.json')
    if (plan['study'] != 'luna_native_research_method_compilation'
            or summary['status'] != 'completed' or not summary['method_published']
            or previous['status'] != 'completed' or not previous.get('method_source')):
        raise ValueError('repair requires a completed published compilation study')
    old_ids = set(plan['old_target_ids'])
    development = [case for case in plan['transfer_cases'] if case['id'] not in old_ids]
    added = []
    for case in development:
        name = f"design_case_{len(plan['practice_cases']):03d}"
        practice = {key: deepcopy(value) for key, value in case.items()
                    if key not in {'id', 'parameter_pair', 'evaluation_seed'}}
        practice['prefix_seed'] += 20_000_003
        practice['confirmation_seed'] += 40_000_037
        plan['practice_cases'][name] = practice
        added.append({'practice_case': name, 'prior_case': case['id']})
    targets = [case for case in plan['transfer_cases'] if case['id'] in old_ids]
    # Registered before authoring, without sampling outcomes or choosing by fit.
    for pair, seed in enumerate((162011031, 162021040)):
        for stage in (1, 2):
            targets.append({'id': f'boundary_target_{pair * 2 + stage - 1:03d}',
                'parameter_pair': pair, 'kind': 'conditional_window',
                'seed': seed, 'apparatus_seed': seed, 'stage': stage, 'noise_sigma': 0.,
                'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                'evaluation_seed': seed + 300007,
                'limits': {'confirmation_mse_threshold': 1e-6}})
    plan.update(continued_author=str(study / 'author/training.json'),
        repair_study=str(study), repair_development_cases=added, transfer_cases=targets,
        prior_author_phases=[*plan['prior_author_phases'], {
            'phase': 'native_method_compilation', 'result': str(study / 'author/training.json'),
            'status': previous['status'], 'costs': previous['costs'],
            'usage': summary['author_usage']}],
        prior_transfer_phases=[*plan.get('prior_transfer_phases', []), {
            'summary': str(study / 'summary.json'),
            'cases': [{'case': row['case'], 'costs': row['costs']}
                      for row in [*summary['retention'], *summary['new_parameters']]]}],
        scope='Repair one Luna-authored procedure using four completed paid transfer traces '
            'as development experience. Keep eight earlier practices and add four changed-prefix '
            'practices. Freeze before ten retention and four new-parameter targets; no live '
            'consumer LLM. Earlier terminal outcomes remain analysis-only.',
        continuation='Restore the published source, notes and every paid practice. Added cases '
            'require their own same-source support; earlier transfer outcomes are not publication '
            'coverage. Prior targets are development, not fresh transfer replications.')
    return plan


def prepare_published_revision(study, output):
    """Revise a publication using retained paid comparisons, without new targets."""
    study, output = Path(study).resolve(), Path(output).resolve()
    if (output / 'protocol.json').exists():
        raise ValueError('preserve earlier attempts; use a new study directory')
    summary = read_json(study / 'summary.json')
    author_publication = summary['status'] == 'author_development_completed'
    if summary['status'] != 'completed' and not (
            author_publication and summary.get('method_published')):
        raise ValueError('published revision requires completed frozen research or a published author')
    restored_plan, publication, restored, restoration = restore_published_author(study)
    source_study = Path(restoration['source_study'])
    plan = read_json(source_study / 'protocol.json')
    comparison_study = Path(plan['published_revision']['frozen_study']) if author_publication else study
    frozen = read_json(comparison_study / 'protocol.json') if author_publication else restored_plan
    baseline = (comparison_study if author_publication else
                Path(frozen['paired_baseline']['source_study']))
    old_publication = read_json(baseline / 'published_result.json')
    # Use the first already registered task from the latest retention group and
    # the new cohort. These are paid development contrasts, not new holdouts.
    retained_case = next(iter(list(frozen['regression_groups'].values())[-1]))
    new_case = next(case['id'] for case in frozen['transfer_cases']
                    if case['id'] not in set(frozen['old_target_ids']))
    if author_publication:
        records = deepcopy(plan['repair_development_cases'])
    else:
        records = [{'practice_case': label, 'prior_case': case, 'source_directory': str(directory)}
            for label, case, directory in (
                ('previous_retention', retained_case, baseline / 'frozen' / retained_case),
                ('current_retention', retained_case, study / 'frozen' / retained_case),
                ('previous_new_cohort', new_case,
                 study / frozen['paired_baseline']['directory'] / 'frozen' / new_case),
                ('current_new_cohort', new_case, study / 'frozen' / new_case))]
    for record in records:
        directory = Path(record['source_directory'])
        if not (directory / 'training.json').is_file():
            raise FileNotFoundError(f'Retained paid investigation unavailable: {directory}')

    # Restore the paid evidence for the latest two cases of the two actual
    # published sources. All other drafts remain available in the same index.
    evidence_trials = []
    for source in (old_publication['method_source'], publication['method_source']):
        latest = {trial['case']: trial for trial in restored['practice_trials']
                  if trial['method_source'] == source and trial['case'] in plan['practice_cases']}
        evidence_trials.extend(latest[name]['directory'] for name in sorted(latest)[-2:])
    restored_evidence = restore_paid_practice_summaries(
        restored['practice_trials'], Path(restored['workspace']), evidence_trials)
    for key in ('frozen_publication', 'paired_baseline', 'parameter_registration',
                'restore_accepted', 'accepted_draft', 'operator_practice', 'complete_practices'):
        plan.pop(key, None)
    plan.update(author_only=True, transfer_cases=[],
        continued_author=str(output / 'author_resume.json'),
        restored_publication=restoration,
        published_revision={'frozen_study': str(comparison_study),
            'previous_publication': str(baseline),
            **({'parent_author_publication': str(study)} if author_publication else {}),
            'restored_paid_summaries': restored_evidence,
            'scope': 'Completed comparison supplies paid development experience; no new targets.'},
        repair_study=str(comparison_study), repair_development_cases=records,
        prior_author_phases=[*plan['prior_author_phases'], {
            'phase': 'published_research_method_author', 'result': publication['full_result'],
            'status': publication['status'], 'costs': publication['costs'],
            'usage': publication['author_usage']}],
        prior_transfer_phases=[*plan.get('prior_transfer_phases', []), *([] if author_publication else [{
            'summary': str(study / 'summary.json'),
            'scope': 'Completed frozen comparison retained in place; not recharged to this author.'}])],
        scope='One Luna author phase starting from the actual published method and all paid drafts. '
            'Use the completed paid comparison to decide when to reuse a research program, '
            'acquire evidence or revise it. Preserve every earlier practice and its allowance. '
            'Unchanged-source republication is retention, not a new policy revision. '
            'No transfer or automatic follow-up is scheduled.',
        runtime_snapshot=sorted(set(plan['runtime_snapshot']) |
            {str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
            {'scripts/run_native_method_compilation.py', 'scripts/run_anonymous_method_revision.py'}))
    plan['author_limits'].update(max_model_calls=32, token_budget=12_000_000)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'author_resume.json', restored)
    return plan


def attach_transfer_repair_experience(author, plan):
    """Reuse the paid counterexample ledger, without loading terminal evaluations."""
    entries = []
    for item in plan['repair_development_cases']:
        source = (Path(item['source_directory']) if item.get('source_directory') else
                  Path(plan['repair_study']) / 'frozen' / item['prior_case'])
        prior = read_json(source / 'training.json')
        relative = Path('repair_evidence') / item['practice_case']
        directory = author.workspace / relative
        directory.mkdir(parents=True)
        view = SimpleNamespace(workspace=directory, files={}, common_interface='')
        supply_counterexample(view, prior, case_label=item['practice_case'])
        # Keep large paid histories in place. Only this paid result and its
        # executable contexts are exposed, never the adjacent evaluation file.
        (directory / 'result.json').symlink_to(source / 'training.json')
        for name in ('model.py', 'shared', '_experiment_models', '_evaluated_model', '_submitted_models'):
            if (source / name).exists():
                (directory / name).symlink_to(source / name,
                                             target_is_directory=(source / name).is_dir())
        summary = practice_execution_summary(prior, str(relative))
        # Keep the four-case index inline. The exact historical predictor is
        # already in its linked result; current failed-candidate detail belongs
        # to method_decision rather than four repeated expression previews.
        summary.pop('latest_submitted_predictor', None)
        entries.append({key: value for key, value in item.items() if key != 'source_directory'} | {
            'status': prior['status'], 'costs': deepcopy(prior['costs']),
            'fresh_confirmation': {key: prior.get('fresh_confirmation', {})[key]
                for key in ('passed', 'mse', 'rows', 'mse_threshold')
                if key in (prior.get('fresh_confirmation') or {})},
            'ledger': str(relative / 'counterexample.json'),
            'observations': str(relative / 'counterexample_observations.json'),
            'execution_summary': summary})
    evidence = {'records': entries,
        'scope': 'Author-only paid development experience, not new practice support. '
            'These completed records are not new independent evaluation. '
            'Inspect recorded predictions, proposed experiments, actual purchases and confirmation. '
            'Choose which research steps to retain or revise; only the method and original '
            'archive transfer to new tasks.'}
    if plan.get('published_revision'):
        # The inherited four-record comparison predates the latest author
        # revision. Add its actual same-case outcomes, so a passed sampling
        # edit is not mistaken for a measured change in research decisions.
        reference_study = Path(plan['published_revision']['frozen_study'])
        reference_source = read_json(reference_study / 'published_result.json')['method_source']
        current_source = (author.workspace / METHOD_PATH).read_text()
        latest = [{trial['case']: trial for trial in author.practice_trials
                   if trial['method_source'] == source}
                  for source in (reference_source, current_source)]
        pairs = []
        if current_source != reference_source:
            for case in sorted(set(latest[0]) & set(latest[1])):
                pair = {'case': case}
                for label, trials in zip(('frozen_parent', 'current'), latest):
                    trial = trials[case]
                    pair[label] = {key: deepcopy(trial[key]) for key in
                        ('directory', 'status', 'costs', 'fresh_confirmation') if key in trial}
                    pair[label].update(
                        source_path=str(Path(trial['directory']) / METHOD_PATH),
                        result_file=str(Path(trial['directory']) / 'result.json'),
                        action_counts=deepcopy(trial.get('execution_summary', {}).get('action_counts')))
                pairs.append(pair)
        evidence['completed_revision_comparison'] = {
            'reference_study': str(reference_study),
            'source_unchanged': current_source == reference_source, 'pairs': pairs,
            'scope': 'Already paid development outcomes for matching cases, not same-randomness '
                'replications. Missing cost/action fields are unknown, not zero. Full results '
                'and executable source remain at the listed paths. Compare accuracy, actions '
                'and the separate resource costs; no program is selected by this summary.'}
    text = json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))
    author.files['repair_evidence.json'] = text
    (author.workspace / 'repair_evidence.json').write_text(text)
    comparison = evidence.get('completed_revision_comparison')
    comparison_context = ''
    if comparison is not None:
        # The full four histories exceed the ordinary file-preview size.
        # Keep this small completed outcome comparison in the actual task;
        # source locations and detailed records stay in repair_evidence.json.
        compact = {'source_unchanged': comparison['source_unchanged'], 'pairs': [
            {'case': pair['case'], **{label: {key: value for key, value in pair[label].items()
                if key in ('status', 'costs', 'fresh_confirmation', 'action_counts')}
                for label in ('frozen_parent', 'current')}} for pair in comparison['pairs']]}
        comparison_context = ('completed_revision_comparison: ' +
            json.dumps(compact, separators=(',', ':')) + '\n\n')
    author.common_interface = (f'repair_evidence.json indexes {len(entries)} paid investigations '
        'and their actual research steps, costs and confirmation outcomes. These are development '
        'evidence for method revision, not support for the current source on a new practice. '
        'Read each linked result in its own directory. counterexample.json.steps and '
        'counterexample_trace.json are event records: action and arguments are top-level '
        'fields; proposal is optional extra information, not the location of every action. '
        'counterexample.json.paid_records is a summary: rows is an integer count. Actual '
        'trajectory records are the list in counterexample_observations.json, each with a '
        'rows array. Resolve executed_method.source_file and full_trace_file relative to '
        'that counterexample.json directory; a shared path denotes a runtime filename, '
        'not the earlier source in the author root. The full output of a truncated Python '
        'inspection is already saved at its feedback.stdout_file. Read the needed portion '
        'there instead of printing all sources again.\n\n' + comparison_context + author.common_interface)


def attach_task(author, plan):
    public = {name: {'public_spec': laboratory(case).public_spec(),
        'limits': {**plan['limits'], **case.get('limits', {}),
                   'max_model_calls': 0, 'token_budget': 0}}
        for name, case in plan['practice_cases'].items()}
    text = json.dumps(public, indent=2)
    author.files['practice_interfaces.json'] = text
    (author.workspace / 'practice_interfaces.json').write_text(text)
    author.common_interface = author.common_interface.replace(
        'The initial method is only an unresolved placeholder.',
        'The initial source is the retained draft; earlier supported drafts remain available.').replace(
        "case:'case_000'", f"case:'{next(iter(plan['practice_cases']))}'").replace(
        '12 program invocations', '32 program invocations').replace(
        'A trial receives 2240 observations',
        'A trial receives the per-case observation allowance in practice_interfaces.json').replace(
        'at MSE <= 1e-10.', 'at the per-case confirmation_mse_threshold in research_state.json.').replace(
        'Failed confirmation becomes paid evidence on the next invocation.',
        'Failed confirmation returns to the author for method revision.').replace(
        'Failed confirmation contributes paid counterexamples, visible on the next invocation.',
        'Fresh confirmation ends this practice; its outcome returns to the method author.').replace(
        "After a program's failed confirmation, continuation\n"
        'identifies the submitted predictor, the new paid evidence and an action to invoke\n'
        'that program again. It will receive the expanded records. You decide whether to\n'
        'continue, edit revision_path, change the research route or stop.',
        'After the enclosing program submits, confirmation ends its frozen research.\n'
        'The author receives its paid result and may edit, restore or practice a method.\n'
        'A new practice starts from its own paid prefix; earlier confirmation rows stay\n'
        'in the author archive, outside that new practice.')
    if plan.get('source_comparison'):
        author.common_interface = author.common_interface.replace(
            f'A trial receives {author.practice_limits.observation_budget} observations including its prefix and 96 fresh\n'
            'confirmation rows, subject to the remaining author account.',
            'practice_interfaces.json gives each case\'s actual observation, confirmation, compute\n'
            'and execution allowances, subject to the remaining author account.')
    objective = ('CURRENT DELIVERABLE: generalize your confirmed investigation into an executable '
        'research procedure that can learn from the evidence of a new apparatus. Inspect ')
    if plan.get('published_revision'):
        objective = (
            'CURRENT DELIVERABLE: use the retained paid comparisons to improve how your '
            'research procedure chooses an existing program, additional evidence or structural '
            'revision. Inspect the actual competing source programs, their paid predictions, '
            'actions, confirmation outcomes and costs. Preserve supported behavior and use '
            'current-task evidence to decide what applies. Observation and execution costs '
            'may trade off; no fixed scientific route or lower observation threshold is required. '
            'The existing source already passed every listed practice; publishing it unchanged '
            'retains that method but does not establish a new research-policy revision. '
            'repair_evidence.json also records the completed source-revision comparison. '
            'Investigate an actual decision before another sampling edit: use the existing '
            'same-interface paid practices to compare competing procedures on the same '
            'evidence available at a decision point. You choose the cutoffs and hypotheses. '
            'Same evidence means supplying both programs the same chosen prefix from one '
            'retained paid run; it need not be the intersection of two historical logs or '
            'a decision fork already executed by both. Through the existing Python run '
            'action, invoke fitting code or replay select_actions with its required state '
            'in separate scratch directories. Calling select_actions directly returns a '
            'proposal; it does not dispatch that proposal or purchase observations. Record '
            'the actual returned action or candidate and execution cost. A program returning '
            'no candidate at one cutoff is a result about that cutoff; decide whether another '
            'retained cutoff warrants a test. '
            'Refit with only that prefix; use later paid trajectories as development checks, '
            'not as fitting evidence for the earlier decision. State the predicted decision '
            'and its test before running it, then inspect the measured consequence. '
            'A change in sample coordinates alone does not answer this research question. '
            'Retain the supported source if evidence does not justify a decision-policy '
            'change. The existing publication gate still tests predictive support, not '
            'policy benefit; both outcomes must be distinguished. Inspect ')
    task = (objective +
        'the actual source research and retained failed drafts, then author and practice '
        'select_actions. The shared native_condition_research operator is available inside '
        'practices under the tool contract below; your program decides whether to use it, '
        'which evidence and candidates to use, what to purchase, and when to submit. '
        'practice_interfaces.json lists public interfaces and actual allowances. Preserve '
        'the inherited method where its evidence supports it. Source-task fitted constants '
        'are not new-task evidence. Only your method and original archive will transfer. '
        'This phase requires predictive publication on every listed case; it does not '
        'require the earlier prospective-design objective. A program stops at its first '
        'final submission; confirmation feedback returns to the author for revision, '
        'not to the frozen program.')
    if plan.get('complete_practices'):
        task = ('CURRENT DELIVERABLE: complete the restored source\'s missing practice '
            'support and independently decide whether to publish it. Keep this source '
            'unchanged during this completion phase. method_decision.current_source_coverage '
            'is the current record; retain confirmed cases and practice only missing cases. '
            'Earlier notebook plans may precede the latest completed practice. The existing '
            'finish action checks missing cases before its ordinary publication gate. '
            'Choose publication from the actual outcomes; a failed case remains a failure. '
            'This phase adds no transfer tasks or scientific strategy-revision assignment.')
    if plan.get('source_comparison'):
        task = (f'CURRENT DELIVERABLE: use your confirmed source investigation and retained '
            f'archive to form an executable research procedure. The same source must pass all '
            f'{len(plan["practice_cases"])} listed practices. Each practice allows 32 program '
            'invocations and stops at its first model submission; its confirmation feedback '
            'returns to you. practice_interfaces.json contains the actual per-case limits. '
            'Choose whether to retain, edit or restore the program from the observed evidence. '
            'The existing publication requirements and earlier same-source support apply. '
            'Only your own new source investigation is attached. No transfer is scheduled.')
    author.common_interface = task + '\n\n' + author.common_interface
    attach_online_research_experience(author, plan['online_research_experience'])
    if plan.get('operator_practice'):
        evidence = Path(plan['operator_practice']).read_text()
        author.files['native_operator_practice.json'] = evidence
        (author.workspace / 'native_operator_practice.json').write_text(evidence)
        author.common_interface = ('native_operator_practice.json records a host-executed, '
            'zero-LLM practice of the available operator on paid development evidence. '
            'It is tool execution evidence, not your authored method or an autonomous '
            'discovery. Inspect its inputs, outputs and costs alongside retained paid '
            'practices before deciding how to revise your procedure.\n\n' + author.common_interface)


def restore_accepted_method(author, study):
    """Restore the selected executable draft without discarding later paid failures."""
    study = Path(study).resolve()
    selected = read_json(study / 'accepted_draft.json')
    acceptance = read_json(study / 'protocol.json')['accepted_draft']
    if selected['status'] != 'user_accepted_draft':
        raise ValueError('restore requires an explicitly accepted development draft')
    source = selected['method_source']
    author.files[METHOD_PATH] = source
    (author.workspace / METHOD_PATH).write_text(source)
    restored = {'accepted_study': str(study),
        'source_author': acceptance['source_author'],
        'confirmed_cases': acceptance['confirmed_cases'],
        'unconfirmed_cases': acceptance['unconfirmed_cases'],
        'retained_paid_practices': len(author.practice_trials),
        'scope': 'Operator restores the user-accepted method as the starting source. '
            'Later failed drafts and their paid practices remain available; this does '
            'not publish the restored source or add confirmation support.'}
    author.files['restored_method.json'] = json.dumps(restored, indent=2)
    (author.workspace / 'restored_method.json').write_text(author.files['restored_method.json'])
    context = json.loads(author.files['resume_context.json'])
    context['current_source_coverage'] = author._current_source_coverage()
    context['restored_method'] = restored
    author.files['resume_context.json'] = json.dumps(context, indent=2)
    (author.workspace / 'resume_context.json').write_text(author.files['resume_context.json'])
    author._refresh_method_candidates()
    author.common_interface = ('restored_method.json identifies the user-accepted starting '
        'source. It replaces the latest exhausted draft only in the current working file; '
        'all later failed practices and research notes remain available for comparison.\n\n'
        + author.common_interface)


def run_compilation(plan, output, *, provider_factory):
    output, source = Path(output).resolve(), Path(plan['source'])
    if (output / 'protocol.json').exists():
        raise ValueError('preserve earlier attempts; use a new study directory')
    # Check the actual retained dependencies before any paid provider call.
    restored = read_json(plan['continued_author'])
    publication_restore = plan.get('restored_publication')
    if publication_restore and publication_restore.get('publication_study'):
        previous = published_source(publication_restore['publication_study'])[1]
    elif publication_restore:
        previous = read_json(Path(publication_restore['frozen_study']) / 'published_result.json')
    else:
        previous = read_json(source / 'author/training.json')
    original = read_json(source / 'initial_archive.json')
    sources = read_json(source / 'source_outcomes.json')
    for trial in restored['practice_trials']:
        if not (Path(restored['workspace']) / trial['directory'] / 'result.json').is_file():
            raise FileNotFoundError(f"Retained paid trial unavailable: {trial['directory']}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', plan)
    for name in plan['runtime_snapshot']:
        path = output / 'source_at_run' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    knowledge = register_published_method(original, previous, max_steps=plan['method_steps'])
    write_json(output / 'initial_archive.json', knowledge)
    write_json(output / 'source_outcomes.json', sources)
    reference = output / 'reference_archive'
    for name, content in original['files'].items():
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    practice_agent = ExperimentalDesignMethodAgent
    if plan.get('source_comparison'):
        def practice_agent(*args, **kwargs):
            kwargs['arm'] = plan['author_arm']
            return ExperimentalDesignMethodAgent(*args, **kwargs)
    learner_type = (SourceComparisonMethodLearner if plan.get('source_comparison')
                    else AnonymousResearchMethodLearner)
    author = learner_type(provider_factory(output / 'author'), output / 'author',
        knowledge=knowledge, cases=plan['practice_cases'], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan['author_limits']),
        practice_limits=InquiryLLMConfig(**plan['limits']),
        prepare_trial=trial_preparer(knowledge, sources, reference,
                                    agent_type=practice_agent))
    if plan.get('source_comparison'):
        author.arm = plan['author_arm']
    author.restore_practice(restored)
    if plan.get('restore_accepted'):
        restore_accepted_method(author, plan['restore_accepted'])
    attach_task(author, plan)
    if plan.get('repair_study'):
        attach_transfer_repair_experience(author, plan)
    write_json(output / 'progress.json', {'stage': 'method_compilation'})
    learned = author.run(on_checkpoint=lambda result: write_json(output / 'author/active.json', result))
    write_json(output / 'author/training.json', learned)
    final_checkpoint_alias(output / 'author/training.json', output / 'author/active.json')
    published = learned['status'] == 'completed' and bool(learned.get('method_source'))
    write_json(output / 'published_method.json', {'published': published,
        'source': learned.get('method_source'), 'lesson': learned.get('lesson')})
    if published and not plan.get('author_only'):
        return run_targets(plan, output, learned, original, sources)
    result = {'status': 'author_development_completed' if published else 'method_unpublished',
        'method_published': published,
        'author_status': learned['status'], 'author_costs': learned['costs'],
        'author_usage': usage_summary(learned), 'scope': plan['scope'],
        'prior_author_phases': plan['prior_author_phases'],
        'target_episodes_pending': 0 if plan.get('author_only') else len(plan['transfer_cases'])}
    if plan.get('author_only'):
        result.update(author_only=True, target_episodes_run=0,
            target_scope='No transfer is scheduled in this author-development phase. '
                'Previously scored targets remain development evidence, not unseen replications.')
    if plan.get('published_revision'):
        restored_source = restored.get('method_source') or restored['outgoing_knowledge']['files'][METHOD_PATH]
        source_changed = learned.get('method_source') != restored_source if published else (
            (output / 'author' / METHOD_PATH).read_text() != restored_source)
        unchanged = ('same_source_completion_published' if plan.get('complete_practices') else
                     'unchanged_method_retained')
        result['published_revision'] = {
            'frozen_study': plan['published_revision']['frozen_study'],
            'source_changed': source_changed,
            'outcome': ('revised_source_published' if source_changed else unchanged)
                if published else 'method_unpublished'}
    if plan.get('source_comparison'):
        baseline = restored['method_source']
        current = (output / 'author' / METHOD_PATH).read_text()
        result['source_comparison'] = {**plan['source_comparison'],
            'source_changed': current != baseline,
            'new_practice_trials': len(learned['practice_trials']) - len(restored['practice_trials']),
            'outcome': ('revised_source_published' if current != baseline else 'unchanged_method_retained')
                if published else 'method_unpublished'}
        result['author_arm'] = plan['author_arm']
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', {'stage': result['status']})
    return result


def prepare_accepted_draft(study, output):
    """Freeze a user-selected development draft without changing author publication."""
    study, output = Path(study).resolve(), Path(output).resolve()
    if (output / 'protocol.json').exists():
        raise ValueError('preserve earlier attempts; resume the frozen target directory')
    plan = read_json(study / 'protocol.json')
    previous = read_json(study / 'author/training.json')
    source = previous['outgoing_knowledge']['files'][METHOD_PATH]
    latest = {trial['case']: trial for trial in previous['practice_trials']
              if trial['method_source'] == source}
    cases = [{key: deepcopy(trial[key]) for key in
              ('case', 'directory', 'status', 'fresh_confirmation', 'costs') if key in trial}
             for trial in latest.values()]
    confirmed = sorted(trial['case'] for trial in cases if trial['status'] == 'completed'
                       and (trial.get('fresh_confirmation') or {}).get('passed') is True)
    acceptance = {'basis': 'user_accepted_development_draft',
        'source_author': str(study / 'author/training.json'),
        'source_author_status': previous['status'], 'author_publication_passed': False,
        'confirmed_cases': confirmed,
        'unconfirmed_cases': sorted(set(plan['practice_cases']) - set(confirmed)),
        'practice_cases': cases, 'retained_paid_practices': len(previous['practice_trials']),
        'author_usage': read_json(study / 'summary.json')['author_usage']}
    plan.update(accepted_draft=acceptance,
        scope='User-accepted development draft frozen before the registered ten retention '
            'and four new-parameter targets. Partial practice support remains partial; '
            'this is not autonomous publication or independent author replication.',
        failure_policy='Retain every completed target, including failures; no author revision '
            'or automatic retry in this frozen evaluation.',
        transfer='Only the selected program and original archive enter targets; '
            'source/practice fits and author history remain outside target workspaces.')
    selected = {'status': 'user_accepted_draft', 'method_source': source,
                'costs': previous['costs'], 'practice_trials': cases}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', plan)
    write_json(output / 'accepted_draft.json', selected)
    (output / 'accepted_method.py').write_text(source)
    for name in plan['runtime_snapshot']:
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    # Reuse the original reference files; never copy the author/trial archives.
    (output / 'reference_archive').symlink_to(study / 'reference_archive', target_is_directory=True)
    write_json(output / 'progress.json', {'stage': 'accepted_draft_frozen'})
    return plan, selected


def published_source(study):
    """Read an actual publication and its small, exact-source support index."""
    study = Path(study).resolve()
    plan = read_json(study / 'protocol.json')
    author_study = Path(plan['frozen_publication']['source_study']) if (
        study / 'published_result.json').is_file() else study
    published = read_json(author_study / 'published_method.json')
    summary = read_json(author_study / 'summary.json')
    candidates = read_json(author_study / 'author/method_candidates.json')
    current, = [candidate for candidate in candidates if candidate.get('current')]
    source = published.get('source')
    if (not published.get('published') or not summary.get('method_published') or not source
            or (author_study / 'author' / current['source_path']).read_text() != source):
        raise ValueError('freeze requires the actual published source and its support')
    trials = [{**deepcopy(trial), 'method_source': source} for trial in current['trials']]
    latest = {trial['case']: trial for trial in trials}
    confirmed = {trial['case'] for trial in latest.values() if trial['status'] == 'completed'
                 and (trial.get('fresh_confirmation') or {}).get('passed') is True}
    if confirmed != set(plan['practice_cases']):
        raise ValueError('published source must cover every listed practice')
    retained_count = len({trial['directory'] for candidate in candidates
                          for trial in candidate['trials']})
    learned = {'status': 'completed', 'method_source': source, 'method_path': METHOD_PATH,
        'method_entry': {'id': 'paired_evidence_method', 'path': METHOD_PATH},
        'lesson': published.get('lesson', ''), 'practice_trials': trials,
        'costs': summary['author_costs'], 'author_usage': summary['author_usage'],
        'retained_paid_practices': retained_count,
        'full_result': str(author_study / 'author/training.json')}
    metadata = {'source_study': str(author_study), 'confirmed_cases': sorted(confirmed),
        'retained_paid_practices': retained_count,
        'support_index': str(author_study / 'author/method_candidates.json')}
    return plan, learned, metadata


def prepare_published_targets(study, output, *, baseline_study=None, evaluation_registration=None):
    """Freeze published procedures using retained support and registered tasks."""
    study, output = Path(study).resolve(), Path(output).resolve()
    if (output / 'protocol.json').exists():
        raise ValueError('preserve earlier attempts; resume the frozen target directory')
    plan, learned, publication = published_source(study)
    source = learned['method_source']
    if evaluation_registration:
        registration_path = Path(evaluation_registration).resolve()
        registration = read_json(registration_path)['frozen_evaluation_registration']
        required_count = registration.get('required_practice_count', 13)
        if len(publication['confirmed_cases']) != required_count:
            raise ValueError(f'registered transfer requires actual {required_count}-case publication')
        if baseline_study:
            raise ValueError('registered comparison uses its published baseline, not an accepted draft')
        regression = read_json(registration['regression_protocol'])
        by_id = {case['id']: case for case in regression['transfer_cases']}
        opened = [deepcopy(by_id[name]) for name in registration['regression_case_ids']]
        # A new task can need a larger shared allowance. Retained regressions
        # keep their original effective ceilings under trial_preparer's minimum.
        for case in opened:
            ceilings = case.get('limits', {})
            for key in ('observation_budget', 'compute_budget', 'execution_seconds'):
                old_ceiling = min(regression['limits'][key],
                                  ceilings.get(key, regression['limits'][key]))
                if min(registration['limits'][key],
                       ceilings.get(key, registration['limits'][key])) != old_ceiling:
                    case.setdefault('limits', {})[key] = old_ceiling
        targets = opened + deepcopy(registration['new_transfer_cases'])
        old_ids = set(registration['regression_case_ids'])
        baseline_source_plan, baseline_selected, baseline_publication = published_source(
            registration['paired_baseline'])
        if baseline_source_plan['source'] != plan['source']:
            raise ValueError('published pair must use the same original archive')
        baseline_count = len(baseline_publication['confirmed_cases'])
        new_target_count = len(registration['new_transfer_cases'])
        plan.update(old_target_ids=registration['regression_case_ids'],
            regression_groups=deepcopy(registration['regression_groups']),
            limits=deepcopy(registration['limits']), method_steps=registration['method_steps'],
            evaluation=deepcopy(registration['evaluation']),
            parameter_registration={'registration_file': str(registration_path),
                **deepcopy(registration),
                'independent_parameter_groups': registration['new_transfer_parameter_groups'],
                'previously_opened_cases': opened,
                'evaluation_scope': 'Each laboratory retains its public trajectory horizon; '
                    'nested_response confirmation and terminal segments have up to64steps.'},
            scope=f'Frozen published {required_count}-case method revision: {len(opened)} '
                f'previously opened regression tasks and {new_target_count} registered targets '
                f'from {registration["new_transfer_parameter_groups"]} parameter groups. '
                f'Compare the published {baseline_count}-case baseline on the same '
                f'{new_target_count} targets. '
                'No consumer LLM, author revision, source fits or author-history transfer.')
    else:
        old_ids = set(plan['old_target_ids'])
        opened = [case for case in plan['transfer_cases'] if case['id'] not in old_ids]
        targets = [case for case in plan['transfer_cases'] if case['id'] in old_ids]
        # Existing adaptive-boundary registration, unchanged for its original entry.
        seeds = (172011031, 172021040, 172031049, 172041058)
        used = {case['apparatus_seed'] for case in [*plan['practice_cases'].values(),
                                                   *plan['transfer_cases']]}
        if used.intersection(seeds):
            raise ValueError('fresh parameter groups overlap development apparatus')
        for pair, seed in enumerate(seeds):
            for stage in (1, 2):
                targets.append({'id': f'adaptive_boundary_target_{pair * 2 + stage - 1:03d}',
                    'parameter_pair': pair, 'kind': 'conditional_window',
                    'seed': seed, 'apparatus_seed': seed, 'stage': stage, 'noise_sigma': 0.,
                    'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                    'evaluation_seed': seed + 300007,
                    'limits': {'confirmation_mse_threshold': 1e-6}})
        plan.update(parameter_registration={'apparatus_seeds': list(seeds), 'stages': [1, 2],
            'independent_parameter_groups': len(seeds), 'previously_opened_cases': opened,
            'scope': 'Four fresh parameter groups, two dependent stages each. Prior '
                'scored boundary targets are development and excluded from unseen outcomes.'},
            scope='Frozen published adaptive-boundary method: ten unchanged retention targets '
                'and eight fresh targets from four parameter groups, two stages per group. '
                'No consumer LLM calls or author revision. Same budgets and confirmation rules.')
    for name in ('author_only', 'restore_accepted', 'accepted_draft'):
        plan.pop(name, None)
    plan.update(transfer_cases=targets, frozen_publication=publication,
        failure_policy='Retain every completed success and failure; resume only unfinished '
            'targets. All registered arms stop research before terminal scoring.')
    output.mkdir(parents=True, exist_ok=True)
    if evaluation_registration:
        baseline_plan = deepcopy(baseline_source_plan)
        for name in ('paired_baseline', 'author_only', 'accepted_draft', 'restore_accepted'):
            baseline_plan.pop(name, None)
        baseline_plan.update(transfer_cases=deepcopy(registration['new_transfer_cases']),
            old_target_ids=list(plan['old_target_ids']), limits=deepcopy(plan['limits']),
            method_steps=plan['method_steps'], evaluation=deepcopy(plan['evaluation']),
            frozen_publication=baseline_publication,
            parameter_registration=deepcopy(plan['parameter_registration']),
            scope=f'Frozen published {baseline_count}-case method on the identical '
                f'{new_target_count} registered targets, with the same runtime, archive and allowances.')
        baseline_directory = f'baseline_{baseline_count}of{baseline_count}'
        baseline = output / baseline_directory
        baseline.mkdir()
        write_json(baseline / 'protocol.json', baseline_plan)
        write_json(baseline / 'published_result.json', baseline_selected)
        (baseline / 'published_method.py').write_text(baseline_selected['method_source'])
        (baseline / 'reference_archive').symlink_to(study / 'reference_archive', target_is_directory=True)
        plan['paired_baseline'] = {'directory': baseline_directory,
            'source_study': registration['paired_baseline'], 'target_count': len(baseline_plan['transfer_cases']),
            'method_published': True,
            'scope': 'Complete both published research arms before either terminal scoring.'}
    elif baseline_study:
        baseline_study = Path(baseline_study).resolve()
        baseline_plan = deepcopy(plan)
        acceptance = read_json(baseline_study / 'protocol.json')['accepted_draft']
        selected = read_json(baseline_study / 'accepted_draft.json')
        if selected['status'] != 'user_accepted_draft':
            raise ValueError('paired baseline requires the retained accepted draft')
        baseline_plan.pop('frozen_publication')
        baseline_plan.update(accepted_draft=acceptance,
            transfer_cases=[case for case in targets if case['id'] not in old_ids],
            scope='Frozen accepted 11/12 procedure on the identical eight new-parameter '
                'tasks, using the same runtime, archive, prefixes, budgets and scoring.')
        baseline = output / 'baseline_11of12'
        baseline.mkdir()
        write_json(baseline / 'protocol.json', baseline_plan)
        write_json(baseline / 'accepted_draft.json', selected)
        (baseline / 'accepted_method.py').write_text(selected['method_source'])
        (baseline / 'reference_archive').symlink_to(study / 'reference_archive',
                                                  target_is_directory=True)
        plan['paired_baseline'] = {'directory': 'baseline_11of12',
            'source_study': str(baseline_study), 'target_count': 8,
            'scope': 'Complete both frozen research arms before either terminal scoring.'}
    write_json(output / 'protocol.json', plan)
    write_json(output / 'published_result.json', learned)
    (output / 'published_method.py').write_text(source)
    for name in plan['runtime_snapshot']:
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    (output / 'reference_archive').symlink_to(study / 'reference_archive', target_is_directory=True)
    write_json(output / 'progress.json', {'stage': 'published_method_frozen'})
    return plan, learned


def transfer_memory(plan, learned, original):
    """Register either the actually published source or explicitly accepted draft."""
    accepted = plan.get('accepted_draft')
    if accepted:
        memory = deepcopy(original)
        memory['files'][METHOD_PATH] = learned['method_source']
        memory.setdefault('research_roles', {})['method'] = METHOD_PATH
        programs = memory.setdefault('manifest', {}).setdefault('research_programs', [])
        programs[:] = [entry for entry in programs if entry.get('path') != METHOD_PATH
                       and entry.get('id') != 'paired_evidence_method']
        programs.append({'id': 'paired_evidence_method', 'path': METHOD_PATH,
            'max_steps': plan['method_steps'], 'admission': accepted['basis'],
            'prior_execution': {'scope': 'Development practice; user accepted partial coverage.',
                                'cases': accepted['practice_cases']}})
        if 'shared/knowledge.json' in memory['files']:
            memory['files']['shared/knowledge.json'] = json.dumps(memory['manifest'], indent=2)
    else:
        if learned['status'] != 'completed' or not learned.get('method_source'):
            raise ValueError('frozen targets require the actual published method')
        memory = register_published_method(original, learned, max_steps=plan['method_steps'])
    return memory


def run_targets(plan, output, learned, original, sources):
    """Resume frozen research in all registered arms before terminal scoring."""
    output = Path(output)
    accepted = plan.get('accepted_draft')
    memory = transfer_memory(plan, learned, original)
    write_json(output / 'transfer_archive.json', memory)
    write_json(output / 'progress.json', {'stage': 'frozen_transfer'})
    frozen = frozen_transfers(plan, output, memory, sources, learned['method_source'],
                              agent_type=ExperimentalDesignMethodAgent)
    baseline_frozen = None
    if plan.get('paired_baseline'):
        baseline = output / plan['paired_baseline']['directory']
        baseline_plan = read_json(baseline / 'protocol.json')
        selected = read_json(baseline / ('published_result.json'
            if baseline_plan.get('frozen_publication') else 'accepted_draft.json'))
        baseline_memory = transfer_memory(baseline_plan, selected, original)
        write_json(baseline / 'transfer_archive.json', baseline_memory)
        write_json(output / 'progress.json', {'stage': 'paired_baseline_transfer'})
        baseline_frozen = frozen_transfers(baseline_plan, baseline, baseline_memory, sources,
            selected['method_source'], agent_type=ExperimentalDesignMethodAgent)
    records = score_frozen(plan, frozen)
    baseline_records = score_frozen(baseline_plan, baseline_frozen) if baseline_frozen is not None else []
    if baseline_frozen is not None:
        baseline_summary = {'status': 'completed',
            'method_published': bool(baseline_plan.get('frozen_publication')),
            'new_parameters': baseline_records, 'live_consumer_model_calls': 0}
        if baseline_plan.get('frozen_publication'):
            baseline_summary.update(frozen_publication=baseline_plan['frozen_publication'],
                author_costs=selected['costs'], author_usage=selected['author_usage'])
        else:
            baseline_summary['accepted_draft'] = baseline_plan['accepted_draft']
        write_json(baseline / 'summary.json', baseline_summary)
    old_ids = set(plan['old_target_ids'])
    original_results = read_json(plan['original_summary'])['frozen']['revised']
    source_workspace = Path(plan['online_research_experience'])
    source_model = plan['source_model']
    answer_controls = []
    for case in plan['transfer_cases']:
        if case['id'] in old_ids:
            continue
        rows = laboratory(case).evaluation_rows(length=plan['evaluation']['rows'],
                                                seed=case['evaluation_seed'])
        score = evaluate_python_model(source_workspace, source_model, rows,
                                      timeout_seconds=plan['limits']['execution_seconds'])
        score.pop('predictions', None)
        score['passed'] = (score.get('returncode') == 0 and
            score.get('mse', float('inf')) <= case['limits']['confirmation_mse_threshold'])
        answer_controls.append({'case': case['id'], 'terminal': score})
    frozen_publication = plan.get('frozen_publication')
    restored_count = (accepted['retained_paid_practices'] if accepted else
        frozen_publication['retained_paid_practices'] if frozen_publication else
        len(read_json(plan['continued_author'])['practice_trials']))
    result = {'status': 'completed', 'method_published': not bool(accepted),
        'author_costs': learned['costs'],
        'author_usage': accepted['author_usage'] if accepted else
            learned['author_usage'] if frozen_publication else usage_summary(learned),
        'prior_author_phases': plan['prior_author_phases'],
        'prior_transfer_phases': plan.get('prior_transfer_phases', []),
        'restored_practice_trials': restored_count,
        'new_practice_trials': 0 if accepted or frozen_publication else
            len(learned['practice_trials']) - restored_count,
        'retention': [row for row in records if row['case'] in old_ids],
        'new_parameters': [row for row in records if row['case'] not in old_ids],
        'retained_original_results': original_results,
        'frozen_source_answer': answer_controls, 'live_consumer_model_calls': 0,
        'scope': plan['scope']}
    if accepted:
        result['accepted_draft'] = accepted
    if frozen_publication:
        result['frozen_publication'] = frozen_publication
        result['parameter_registration'] = plan['parameter_registration']
    if plan.get('regression_groups'):
        result['retention_groups'] = {name: [row for row in records if row['case'] in names]
            for name, names in plan['regression_groups'].items()}
    if baseline_frozen is not None:
        result['paired_baseline'] = {'registration': plan['paired_baseline'],
            'new_parameters': baseline_records}
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', {'stage': 'completed'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--online-study', type=Path)
    inputs.add_argument('--repair-paid-source', type=Path, metavar='TRAINING',
                        help='Edit and practice a retained method on one unfinished paid source')
    inputs.add_argument('--continue-author', type=Path)
    inputs.add_argument('--repair-study', type=Path)
    inputs.add_argument('--revise-published-study', type=Path,
                        help='Revise a frozen or author-only publication using retained paid comparisons')
    inputs.add_argument('--resume-targets', type=Path)
    inputs.add_argument('--accept-draft', type=Path,
                        help='Freeze a user-accepted partial-development draft for targets only')
    inputs.add_argument('--freeze-published', type=Path,
                        help='Freeze a published source from its compact support index')
    parser.add_argument('--baseline-draft', type=Path,
                        help='Pair the frozen publication with this accepted 11/12 draft')
    parser.add_argument('--evaluation-registration', type=Path,
                        help='With --freeze-published, use this saved frozen_evaluation_registration')
    parser.add_argument('--compile-revision-frozen', type=Path,
                        help='With --online-study, restore this frozen publication for author-only revision')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--restore-accepted', type=Path,
                        help='Start author development from this accepted draft, retaining later practices')
    parser.add_argument('--author-only', action='store_true',
                        help='Finish after one author phase, including when it publishes')
    parser.add_argument('--complete-practices', action='store_true',
                        help='With --continue-author, complete the unchanged source and decide publication')
    parser.add_argument('--additional-author-calls', type=int, default=0,
                        help='Explicitly add cumulative calls only to a paid-source --continue-author')
    parser.add_argument('--additional-author-execution-seconds', type=float, default=0.,
                        help='Explicitly add cumulative Python seconds only to a paid-source --continue-author')
    parser.add_argument('--additional-author-tokens', type=int, default=0,
                        help='Explicitly add cumulative tokens only to a paid-source --continue-author')
    parser.add_argument('--operator-practice', type=Path,
                        help='Attach a saved zero-LLM native-operator practice for author inspection')
    parser.add_argument('--research-brief', type=Path,
                        help='Pass a recorded evidence/task brief to the paid-source author')
    parser.add_argument('--retain-comparison-report', action='append', default=[], metavar='PATH',
                        help='Reference completed comparison parameters from this source-relative report')
    args = parser.parse_args()
    if args.complete_practices and not args.continue_author:
        parser.error('--complete-practices requires --continue-author')
    for option in ('additional_author_calls', 'additional_author_execution_seconds', 'additional_author_tokens'):
        value = getattr(args, option)
        if value < 0 or (value and not args.continue_author):
            parser.error(f'--{option.replace("_", "-")} must be nonnegative and requires --continue-author')
    if args.evaluation_registration and not args.freeze_published:
        parser.error('--evaluation-registration requires --freeze-published')
    if args.resume_targets:
        out = args.resume_targets.resolve()
        plan = read_json(out / 'protocol.json')
        source = Path(plan['source'])
        selected = ('accepted_draft.json' if plan.get('accepted_draft') else
                    'published_result.json' if plan.get('frozen_publication') else 'author/training.json')
        run_targets(plan, out, read_json(out / selected),
                    read_json(source / 'initial_archive.json'),
                    read_json(source / 'source_outcomes.json'))
        return
    if args.output is None:
        parser.error('--output is required')
    if args.freeze_published:
        plan, selected = prepare_published_targets(args.freeze_published, args.output,
            baseline_study=args.baseline_draft, evaluation_registration=args.evaluation_registration)
        if not args.prepare_only:
            source = Path(plan['source'])
            run_targets(plan, args.output, selected, read_json(source / 'initial_archive.json'),
                        read_json(source / 'source_outcomes.json'))
        return
    if args.accept_draft:
        plan, selected = prepare_accepted_draft(args.accept_draft, args.output)
        if not args.prepare_only:
            source = Path(plan['source'])
            run_targets(plan, args.output, selected, read_json(source / 'initial_archive.json'),
                        read_json(source / 'source_outcomes.json'))
        return
    if args.compile_revision_frozen and not args.online_study:
        parser.error('--compile-revision-frozen requires --online-study')
    plan = (paid_source_repair_protocol(args.repair_paid_source) if args.repair_paid_source else
            prepare_revision_compilation(args.online_study, args.compile_revision_frozen, args.output)
            if args.compile_revision_frozen else
            prepare_published_revision(args.revise_published_study, args.output)
            if args.revise_published_study else
            continuation_protocol(args.continue_author,
                additional_author_calls=args.additional_author_calls,
                additional_author_execution_seconds=args.additional_author_execution_seconds,
                additional_author_tokens=args.additional_author_tokens) if args.continue_author
            else repair_protocol(args.repair_study) if args.repair_study
            else protocol(args.online_study))
    if args.restore_accepted:
        plan['restore_accepted'] = str(args.restore_accepted.resolve())
    if args.operator_practice:
        plan['operator_practice'] = str(args.operator_practice.resolve())
    if args.author_only:
        plan.update(author_only=True,
            scope='One bounded Luna author-development phase using retained paid practices. '
                'Preserve supported earlier procedures; determine from actual experiments '
                'whether to continue measurement, revise parameters or revise structure. '
                'No frozen target batch or automatic follow-up author phase is scheduled.')
    if args.complete_practices:
        plan.update(complete_practices=True, author_only=True,
            scope='Complete only the restored source\'s missing practices, retaining its '
                'already confirmed cases. Luna independently decides publication under the '
                'existing gate. Keep scientific source unchanged; no transfer is scheduled.')
    if args.research_brief:
        if plan['study'] != 'luna_paid_source_procedure_repair':
            parser.error('--research-brief requires a paid-source author')
        brief = args.research_brief.resolve()
        plan['research_brief'] = {'path': str(brief), 'text': brief.read_text()}
    if args.retain_comparison_report:
        if plan['study'] != 'luna_paid_source_procedure_repair':
            parser.error('--retain-comparison-report requires a paid-source author')
        plan['retained_comparison_reports'] = list(dict.fromkeys([
            *plan.get('retained_comparison_reports', []), *args.retain_comparison_report]))
    if args.prepare_only:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'planned_protocol.json', plan)
        return
    plan['provider']['auth_mode'] = verify_codex_subscription()
    def provider(workspace):
        return CodexCodeProvider(workspace, model=plan['model'],
            reasoning_effort=plan['provider']['reasoning_effort'],
            timeout_seconds=plan['provider']['timeout_seconds'])
    run = (run_paid_source_repair if plan['study'] == 'luna_paid_source_procedure_repair'
           else run_compilation)
    result = run(plan, args.output, provider_factory=provider)
    print(json.dumps({'status': result['status'], 'method_published': result['method_published']}), flush=True)


if __name__ == '__main__':
    main()
