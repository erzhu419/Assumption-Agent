#!/usr/bin/env python3
"""Compare autonomous source research with one shared executable archive."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import register_published_method
from scripts.run_anonymous_online_research import (
    RESEARCH_TASK, prepare_representation_prefix, representation_laboratory, research_trace)
from scripts.run_anonymous_method_learning import laboratory
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_online_inquiry_llm import (
    final_checkpoint_alias, verify_codex_subscription, write_json)
from scripts.run_python_method_learning_study import LIMIT_COSTS


def protocol(source, *, cases=None):
    source = Path(source).resolve()
    original = json.loads((source / 'protocol.json').read_text())
    selected = deepcopy(cases)
    if selected is not None:
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(case, dict) for case in selected)):
            raise ValueError('register a nonempty list of source cases')
        if len({case['id'] for case in selected}) != len(selected):
            raise ValueError('each registered source needs its own case id')
    cases = []
    for block, seed in enumerate((71231031, 71232040)):
        modes = ('recurrent', 'finite2') if block == 0 else ('finite2', 'recurrent')
        for mode in modes:
            cases.append({'id': f'case_{len(cases):03d}', 'parameter_pair': block,
                'kind': mode, 'seed': seed, 'apparatus_seed': seed,
                'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                'evaluation_seed': seed + 300007})
    snapshot = set(original.get('runtime_snapshot', [])) | {
        str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py')}
    snapshot.update({'scripts/run_anonymous_revision_choice.py',
        'scripts/run_anonymous_method_learning.py',
        'scripts/run_anonymous_online_research.py', 'scripts/run_anonymous_revision_inquiry.py',
        'scripts/run_anonymous_method_revision.py', 'scripts/run_online_inquiry_llm.py'})
    if selected is not None:
        snapshot.add('scripts/run_nonlinear_mechanism_revision.py')
    plan = {'study': 'anonymous_learned_method_revision_choice', 'source': str(source),
        'model': 'gpt-6-luna', 'provider': {'backend': 'codex_cli',
            'reasoning_effort': 'medium', 'timeout_seconds': 600.},
        'arms': ['code_react', 'hegel'], 'cases': cases, 'execution_workers': 2,
        'limits': asdict(InquiryLLMConfig(max_model_calls=96, token_budget=10_000_000,
            execution_seconds=3600., observation_budget=1152, confirmation_rows=96,
            compute_budget=250_000)),
        'prefix_rows': 72,
        'prefix_protocol': 'Buy 24 independent three-step trajectories without meter readings. '
            'Each parameter pair shares controls and apparatus coefficients. The paired worlds '
            'agree on the short prefix; their public 24-step horizon permits distinguishing '
            'experiments. World labels and task seeds do not enter the researcher context.',
        'binding': 'Both arms receive the same complete original source archive and unchanged '
            'published program with its actual source practice support. No earlier target '
            'observations, fitted target models or terminal outcomes are transferred. The '
            'researcher may invoke, inspect, revise or decline the program.',
        'discovery': 'Reuse ordinary Python, compare_experiments and prospective query revision. '
            'The researcher authors every scientific candidate and proposed experiment. No '
            'fitted alternative, mechanism label, supplied probe or automatic retry is added. '
            'Direct task success counts even without a comparison or revision interface.',
        'evaluation': {'rows': 512, 'trajectory_length': 24, 'mse_threshold': 1e-10},
        'terminal_order': 'All eight live investigations stop before any terminal evaluation. '
            'Terminal outcomes never return to research.',
        'failure_policy': 'Keep every registered episode and actual failure. Limits are identical '
            'and fixed before research; no performance-dependent retries or additional allocation.',
        'outcome_protocol': 'Report task success and costs separately from actual method execution, '
            'method source changes, prospective comparisons, acquisitions and subsequent decisions. '
            'A declaration, a method stop, or unequal forecasts alone is not successful discrimination.',
        'scope': 'Two fresh parameter blocks, two historical apparatus families and eight live '
            'Code/Hegel episodes using one previously learned method. This tests autonomous '
            'refit-versus-revision behavior in a limited development setting. It does not isolate '
            'recursive organization, establish open-world discovery or replace the preserved '
            'six-target frozen-transfer evidence.',
        'runtime_snapshot': sorted(snapshot), 'method_steps': original.get('method_steps', 12)}
    if selected is not None:
        plan.update(cases=selected, arms=selected[0].get('arms', ['hegel']), execution_workers=1,
            provider={'backend': 'codex_cli', 'reasoning_effort': 'high',
                      'timeout_seconds': 900.},
            limits=asdict(InquiryLLMConfig(max_model_calls=48, token_budget=12_000_000,
                execution_seconds=7200., observation_budget=4096, confirmation_rows=96,
                compute_budget=2_000_000, confirmation_mse_threshold=1e-6)),
            evaluation={'rows': 512, 'mse_threshold': 1e-6},
            prefix_protocol='Use the registered apparatus and prefix. If continuing a paid '
                'failed procedure, restore its actual observations, candidates, costs and '
                'confirmation stream without repurchase.',
            binding='The current published method and original archive remain available. '
                'Only the selected source\'s paid research may be restored; terminal outcomes '
                'and other targets never enter its research.',
            terminal_order='Stop online source research before generating independent terminal labels.',
            scope='One registered structural development source using the preserved published '
                'procedure and original archive. Live Luna chooses experiments, explanations '
                'and whether to revise the method. Source success is not method publication '
                'or independent transfer evidence.')
        if len(selected) > 1:
            arms = ['code_react', 'hegel']
            plan['runtime_snapshot'] = sorted(set(plan['runtime_snapshot']) | {
                'scripts/run_native_method_compilation.py',
                'scripts/experimental_design_control.py',
                'scripts/run_astra_code_frozen_comparison.py',
                'scripts/run_candidate_retention_cycle.py',
                'scripts/run_candidate_retention_cycle_llm.py',
                'scripts/run_legacy_session_bridge_study.py',
                'scripts/run_online_inquiry_study.py',
                'scripts/run_python_advice_ablation.py',
                'scripts/run_python_pending_advice_study.py',
                'scripts/run_python_procedure_study.py',
                'scripts/run_python_refit_diagnosis.py',
                'scripts/run_python_research_routing.py',
                'scripts/run_temporal_calibration_transfer.py',
                'scripts/run_temporal_inquiry_llm.py',
                'scripts/run_temporal_method_learning.py',
                'scripts/run_temporal_operator_llm.py',
                'scripts/run_temporal_paired_method.py',
                'scripts/run_temporal_paired_retention.py'})
            for index, case in enumerate(selected):
                case['arms'] = arms[index % 2:] + arms[:index % 2]
            plan.update(study='independent_source_same_archive_research', arms=arms,
                execution_workers=4,
                provider={'backend': 'codex_cli', 'reasoning_effort': 'high',
                          'timeout_seconds': 1800.},
                prefix_protocol='Both arms buy identical 72-row public prefixes for each source. '
                    'Historical finite2/recurrent apparatuses use 24 independent three-step '
                    'trajectories without meter readings; other apparatuses use the existing '
                    'confirmation-action generator with the registered prefix seed.',
                binding='Every source and arm starts from the same fixed executable archive '
                    'and published program. New source observations, code and live results '
                    'are never shared across arms or sources. Sources may retain, refit, '
                    'revise or decline the inherited method.',
                controls='Both arms use Luna with identical model effort, budgets, tools, '
                    'archive, paid prefix and fresh-confirmation contract. Code may freely '
                    'organize the archive and invoke every shared research operator. Hegel '
                    'adds its automatic research root, failed-confirmation review and '
                    'parent synthesis. This compares the complete organized runtime with '
                    'strong Code, not recursion alone.',
                terminal_order='All registered source investigations stop before any terminal '
                    'evaluation. No source terminal labels return to research.',
                method_formation={'status': 'planned_after_source_research',
                    'entry_points': ['scripts/run_native_method_compilation.py',
                                     'scripts/run_astra_code_frozen_comparison.py'],
                    'contract': 'Retain each arm\'s actual source research for a later common '
                        'method-authoring and practice protocol. A confirmed predictor is '
                        'not a published method. This source cohort launches no compilation '
                        'or target transfer.'},
                scope='New independent source instances within existing apparatus families, '
                    'paired Code/Hegel investigations from a fixed common archive. The unit '
                    'is the source instance; failures remain in the registered denominator. '
                    'This tests hypothesis revision and source research, not unseen-family '
                    'discovery, isolated recursion, method publication or target transfer.')
            plan['limits']['max_model_calls'] = 96
    return plan


def record_research(summary, output, author):
    """Summarize actual executions without deciding the scientific explanation."""
    output = Path(output)
    method = author['method_path']
    for trial in summary['trials']:
        workspace = Path(trial['archive']).parent
        host_workspace = output / trial['case'] / trial['arm']
        host_workspace.mkdir(parents=True, exist_ok=True)
        training = json.loads((workspace / 'training.json').read_text())
        trace = research_trace(training)
        write_json(host_workspace / 'research_trace.json', trace)
        final_checkpoint_alias(workspace / 'training.json', host_workspace / 'active.json')
        calls = [event for event in training['events']
                 if event.get('arguments', {}).get('path') == method]
        steps = [step for step in training['method_results'] if step['path'] == method]
        executed = [step for step in steps if step.get('proposal') is not None]
        comparisons = training.get('experiment_comparisons', [])
        revisions = training.get('revision_proposals', [])
        trial.update(
            published_method_entry_attempts=sum(event['action'] == 'query' or
                (event['action'] == 'run' and event['arguments'].get('operator') == 'procedure')
                for event in calls),
            published_method_executed_steps=len(executed),
            published_method_stops=sum(isinstance(step.get('proposal'), dict) and
                'stop' in step['proposal'] for step in executed),
            unchanged_published_steps=sum(step['feedback'].get('inherited_source_unchanged') is True
                for step in executed),
            published_method_confirmed=any(
                step['feedback'].get('inherited_source_unchanged') is True and
                (step['feedback'].get('fresh_confirmation') or {}).get('passed') is True
                for step in executed),
            published_method_present_at_end=(workspace / method).is_file(),
            published_method_changed_at_end=((workspace / method).read_text() != author['method_source']
                if (workspace / method).is_file() else None),
            other_executed_method_paths=sorted({step['path'] for step in training['method_results']
                if step['path'] != method and step.get('proposal') is not None}),
            experiment_comparison_attempts=len(comparisons),
            completed_experiment_comparisons=sum(row.get('status') == 'compared' for row in comparisons),
            prospective_revision_attempts=len(revisions),
            acquired_prospective_revisions=sum(bool(row.get('observations')) for row in revisions),
            prospective_revisions_with_different_forecasts=sum(
                row['predictions_differ'] and row['acquisition_step'] is not None
                for row in trace['prospective_revisions']),
            research_trace=str(host_workspace / 'research_trace.json'))
    summary['research_interpretation'] = (
        'Counts describe actual recorded interface use, not inferred scientific success. '
        'Inspect frozen forecasts, paid outcomes, authored readouts and subsequent actions '
        'in research_trace.json and training.json. An unchanged published source may coexist '
        'with a separately authored revised predictor or method.')
    write_json(output / 'summary.json', summary)
    return summary


def run_choice(source, output, *, provider_factory, auth_mode=None, cases=None,
               resume_result=None, form_methods=False, resume_cohort=None,
               initial_inquiry=None):
    output = Path(output).resolve()
    if initial_inquiry is not None and (resume_result is None or resume_cohort is not None):
        raise ValueError('a host-scheduled inquiry is a separate paid-source development continuation')
    if resume_cohort is not None:
        if cases is not None or resume_result is not None:
            raise ValueError('cohort recovery uses the original registered cases and paid states')
        resume_cohort = Path(resume_cohort).resolve()
        plan = json.loads((resume_cohort / 'protocol.json').read_text())
        if plan['study'] != 'independent_source_same_archive_research':
            raise ValueError('recover the registered paired independent-source cohort')
        source = Path(plan['source']).resolve()
        cases = deepcopy(plan['cases'])
        form_methods = plan.get('method_formation', {}).get('status') == (
            'scheduled_after_all_source_research')
        plan['infrastructure_resume'] = {
            'previous_study': str(resume_cohort),
            'scope': 'Continue the same registered cohort after question-declaration handling '
                'failed before evaluation. Retain all completed and exhausted episodes; '
                'recover saved paid replies under unchanged total limits; start queued episodes. '
                'No source, confirmation stream, model, allowance or scientific instruction changes.'}
    else:
        source = Path(source).resolve()
        plan = protocol(source, cases=cases)
    if form_methods and resume_cohort is None:
        if not cases or len(cases) < 2 or resume_result is not None:
            raise ValueError('method comparison follows a registered fresh paired source cohort')
        if any('compilation_confirmation_seed' not in case for case in cases):
            raise ValueError('register separate method-practice confirmation streams before research')
        plan['method_formation'] = {
            'status': 'scheduled_after_all_source_research',
            'entry_point': 'scripts/run_native_method_compilation.py:'
                'prepare_source_comparison_compilation/run_compilation',
            'execution_workers': 2,
            'author_limits': {'max_model_calls': 32, 'token_budget': 12_000_000,
                              'execution_seconds': 36000.},
            'contract': 'For each independently confirmed and terminally passed source, '
                'both arms use the same author, edit/practice and publication contract. '
                'Only that arm\'s own online evidence is attached to the common prior. '
                'Keep all earlier practice support and require one additional source practice '
                'with its preregistered confirmation stream. Source failures remain in the '
                'denominator. No target transfer is scheduled.'}
    if auth_mode is not None and resume_cohort is None:
        plan['provider']['auth_mode'] = auth_mode
    if (source / 'published_result.json').is_file():
        author = json.loads((source / 'published_result.json').read_text())
        memory = json.loads((source / 'transfer_archive.json').read_text())
        if memory['files'][author['method_path']] != author['method_source']:
            raise ValueError('use the actual frozen publication and its transfer archive')
    else:
        author = json.loads((source / 'author/training.json').read_text())
        memory = register_published_method(
            json.loads((source / 'initial_archive.json').read_text()), author,
            max_steps=plan['method_steps'])
    if resume_cohort is not None:
        memory = json.loads((resume_cohort / 'initial_archive.json').read_text())
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh output directory to preserve all earlier investigations')
    output.mkdir(parents=True, exist_ok=True)
    prepare = prepare_representation_prefix
    lab_factory = representation_laboratory
    if cases is not None:
        lab_factory = laboratory

        def prepare(agent, lab, case):
            if case.get('kind') in {'finite2', 'recurrent'}:
                return prepare_representation_prefix(agent, lab, case)
            actions = lab.confirmation_actions(plan['prefix_rows'], Random(case['prefix_seed']))
            feedback = agent._execute('query', {'actions': actions}, lab)
            if feedback.get('error'):
                raise ValueError(feedback['error'])
            agent.events.append({'step': -1, 'action': 'query',
                'arguments': {'actions': actions}, 'feedback': feedback,
                'usage': {'backend': 'common_paid_prefix', 'total_tokens': 0}})
            agent.common_interface += RESEARCH_TASK
            return {}
    if resume_result is not None:
        if cases is None or len(plan['arms']) != 1:
            raise ValueError('restore one explicitly registered source and arm')
        resume_path = Path(resume_result).resolve()
        previous = json.loads(resume_path.read_text())
        if previous['status'] == 'completed':
            raise ValueError('continue a genuinely unresolved paid source')
        plan.update(resume_result=str(resume_path),
            resume_mode=('online_research' if
                (resume_path.parent / 'research_context.json').is_file() else 'frozen_practice'),
            online_allowance=deepcopy(plan['limits']),
            prior_research_cost=deepcopy(previous['costs']),
            cost_scope='Total costs retain the previous paid account; continuation_cost '
                'reports only added online work. The source prefix is not bought again.')
        if initial_inquiry is not None:
            plan['initial_inquiry'] = deepcopy(initial_inquiry)
        for limit, cost in LIMIT_COSTS:
            plan['limits'][limit] += previous['costs'][cost]
        plan['limits']['token_budget'] += previous['costs']['reported_tokens']

        def prepare(agent, lab, case):
            restore = (agent.restore_research if plan['resume_mode'] == 'online_research'
                       else agent.restore_practice_research)
            restore(previous, resume_path.parent, lab,
                    confirmation_seed=case['confirmation_seed'])
            if initial_inquiry is not None:
                arguments = {'operator': 'inquire', **deepcopy(initial_inquiry)}
                inquiry = agent._execute('run', arguments, lab)
                agent.events.append({'step': len(agent.events), 'action': 'run',
                    'arguments': arguments, 'feedback': inquiry,
                    'origin': 'host_scheduled_development',
                    'usage': {'backend': 'host_scheduled_development', 'total_tokens': 0}})
            agent.common_interface += RESEARCH_TASK
            write_json(agent.workspace / 'active.json', agent.result('researching'))
            return {'prior_research_cost': deepcopy(previous['costs']),
                    'resume_origin': str(resume_path)}
    for name in plan['runtime_snapshot']:
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    if form_methods:
        write_json(output / 'campaign_progress.json', {
            'stage': 'source_research', 'registered_sources': len(plan['cases']),
            'registered_source_episodes': 2 * len(plan['cases']),
            'target_transfer_scheduled': False})
    summary = run_study(plan, output, provider_factory=provider_factory, knowledge=memory,
        laboratory_factory=lab_factory, prepare_prefix=prepare,
        workers=plan['execution_workers'], previous_study=resume_cohort,
        resume_task=RESEARCH_TASK)
    summary = record_research(summary, output, author)
    if form_methods:
        run_source_method_comparison(source, output, summary, provider_factory=provider_factory)
    return summary


def run_source_method_comparison(source, output, summary, *, provider_factory):
    """Use the existing compiler separately on each arm's own successful inquiry."""
    from scripts.run_native_method_compilation import (
        prepare_source_comparison_compilation, run_compilation)

    output = Path(output)
    write_json(output / 'campaign_progress.json', {
        'stage': 'method_formation', 'registered_source_episodes': len(summary['trials'])})

    def compile_trial(trial):
        record = {'case': trial['case'], 'arm': trial['arm'],
                  'source_status': trial['status']}
        if not (trial['status'] == 'completed' and trial['fresh_confirmation_passed']
                and trial['terminal']['passed']):
            return {**record, 'status': 'source_not_confirmed', 'method_published': False,
                    'author_calls': 0}
        destination = output / 'methods' / trial['case'] / trial['arm']
        plan = prepare_source_comparison_compilation(
            output, source, destination, case_id=trial['case'], arm=trial['arm'])
        result = run_compilation(plan, destination, provider_factory=provider_factory)
        return {**record, 'status': result['status'],
                'method_published': result['method_published'],
                'author_costs': result['author_costs'],
                'source_comparison': result.get('source_comparison'),
                'result': str(destination / 'summary.json')}

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(compile_trial, summary['trials']))
    result = {'status': 'completed', 'trials': records,
              'registered_source_episodes': len(records),
              'scope': 'Methods formed from independently paired source inquiries. '
                  'Publication uses development practices, not new-target transfer. '
                  'Count source failures in the formation denominator.'}
    write_json(output / 'method_formation_summary.json', result)
    write_json(output / 'campaign_progress.json', {
        'stage': 'completed', 'registered_source_episodes': len(records)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-file', type=Path,
        help='One registered case object, or a nonempty list for a paired independent-source cohort')
    parser.add_argument('--resume-result', type=Path,
        help='Actual unresolved frozen or online result to continue without rebuying paid evidence')
    parser.add_argument('--resume-cohort', type=Path,
        help='Recover the same interrupted paired cohort under its original total allowances')
    parser.add_argument('--form-methods', action='store_true',
        help='After the paired source cohort, compile each successful arm with the same publication contract')
    args = parser.parse_args()
    if args.source is None and args.resume_cohort is None:
        parser.error('provide --source or --resume-cohort')
    cases = json.loads(args.case_file.read_text()) if args.case_file else None
    if isinstance(cases, dict):
        cases = [cases]
    plan = (json.loads((args.resume_cohort / 'protocol.json').read_text())
            if args.resume_cohort else protocol(args.source, cases=cases))
    summary = run_choice(args.source, args.output, auth_mode=verify_codex_subscription(),
        cases=cases, resume_result=args.resume_result, form_methods=args.form_methods,
        resume_cohort=args.resume_cohort,
        provider_factory=lambda workspace: CodexCodeProvider(workspace, model=plan['model'],
            reasoning_effort=plan['provider']['reasoning_effort'],
            timeout_seconds=plan['provider']['timeout_seconds']))
    print(json.dumps({'completed': len(summary['trials']),
        'terminal_passed': sum(row['terminal']['passed'] for row in summary['trials'])}), flush=True)


if __name__ == '__main__':
    main()
