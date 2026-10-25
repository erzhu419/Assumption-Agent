#!/usr/bin/env python3
"""Author from actual archives, practice, freeze, and test on new apparatuses."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from random import Random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.anonymous_method_learning import (
    AnonymousPreparedMethodAgent, AnonymousResearchMethodLearner, EMPTY_METHOD)
from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.code_agent import (
    _copy_model_file,
    CodexCodeProvider, archived_paid_trace_references, evaluate_python_model,
)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, register_published_method
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_python_method_learning_study import usage_summary


def protocol(source_root, *, model='gpt-6-luna', timeout_seconds=300.):
    original = json.loads((Path(source_root) / 'protocol.json').read_text())
    cases = original['cases']
    representation = 'apparatus_seed' in cases[0]
    practice, targets = {}, []
    # All case identities and all target slots are fixed before the author runs.
    # Order interleaves exact prior returns and changed objects; no labels enter
    # the author-facing case names or public apparatus capabilities.
    for index, case in enumerate(cases):
        for offset in (0, 1100003):
            position = len(practice)
            practice[f'case_{position:03d}'] = {
                'kind': case['kind'], 'seed': case['seed'] + offset,
                'prefix_seed': case['seed'] + 1200007 + position * 101,
                'confirmation_seed': case['seed'] + 1300021 + position * 101}
            if representation:
                practice[f'case_{position:03d}']['apparatus_seed'] = case['apparatus_seed'] + offset
        for replicate in range(3):
            position = len(targets)
            target_offset = (position if representation else replicate) * 1009
            targets.append({'id': f'target_{position:03d}', 'kind': case['kind'],
                'seed': case['seed'] + 2100011 + target_offset,
                'prefix_seed': case['seed'] + 2200003 + target_offset,
                'confirmation_seed': case['seed'] + 2300027 + target_offset,
                'evaluation_seed': case['seed'] + 2400007 + target_offset})
            if representation:
                targets[-1]['apparatus_seed'] = case['apparatus_seed'] + 2100011 + target_offset
    limits = InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
        max_model_calls=48, token_budget=1200000, compute_budget=250000,
        execution_seconds=300.)
    plan = {'study': 'anonymous_archive_method_transfer_development',
        'source_root': str(Path(source_root).resolve()), 'model': model,
        'provider': {'backend': 'codex_cli', 'timeout_seconds': timeout_seconds,
                     'reasoning_effort': 'medium'},
        'author_limits': asdict(InquiryLLMConfig(max_model_calls=36, token_budget=1800000,
            observation_budget=6000, compute_budget=4000000, execution_seconds=900.,
            confirmation_rows=0)),
        'limits': asdict(limits), 'prefix_rows': 72, 'practice_cases': practice,
        'transfer_cases': targets, 'live_control_cases': [case['id'] for case in targets[::3]],
        'evaluation': {'rows': 512, 'trajectory_length': 48, 'mse_threshold': 1e-10},
        'archive': 'All registered Code and Hegel source research is pooled without success filtering. '
            'Case kinds, seeds and terminal outcomes are not supplied to author or consumers.',
        'authoring': 'One Luna author writes and practices one self-contained select_actions program. '
            'Reuse existing PythonArchiveMethodLearner publication and supported-incumbent comparison. '
            'No host fitting/state solver or hypothesis is supplied. Preserve all drafts and failures.',
        'transfer': 'Freeze the published source before any of nine new apparatuses execute. '
            'Zero LLM calls inside frozen transfer. Only original archives and the method transfer; '
            'no practice-fitted models or observations. Three source families, not nine independent authors.',
        'control': 'After freezing, ordinary Code and Hegel receive the SAME nonempty archive AND '
            'published method on three preregistered targets. Same model, public tools, paid prefix '
            'and limits. Both can execute, inspect, revise or ignore the program. These live controls '
            'test adoption; a frozen method versus live Code is not a matched policy comparison.',
        'failure_policy': 'One author attempt. If it does not publish, retain its paid failures and '
            'mark all planned transfer/control slots unavailable; do not silently substitute a draft.',
        'terminal_order': 'No target terminal labels until all frozen and live target research stops.',
        'scope': 'Deterministic finite-control development, within the three source mechanism families. '
            'One pooled author; neither independent replications of method learning nor open-world discovery.'}
    if representation:
        plan.update(study='recurrent_archive_method_transfer_development',
            prefix_protocol='Buy 24 independent three-step trajectories, 72 response rows and no '
                'meter readings, using the same prefix adapter as the source research.',
            transfer='Freeze one published program before six new apparatuses execute. Zero LLM '
                'calls inside frozen transfer. Only all original source archives and the published '
                'method transfer; no practice-fitted models or observations.',
            control='After freezing, ordinary Code and Hegel receive the SAME complete archive AND '
                'published program on two preregistered targets. Same model, tools, paid prefix and '
                'limits. Both may execute, inspect, revise or ignore it. Report actual adoption '
                'separately from frozen execution, without attributing it to recursive organization.',
            scope='One method author using four complete investigations from two recurrent apparatus '
                'settings, including failures. Four practice settings include two source returns '
                'and two new parameters; six frozen targets have new parameters in the same family. '
                'Not independent method-learning replications or unseen-family discovery.',
            execution_workers=2,
            runtime_snapshot=sorted({str(path.relative_to(ROOT))
                for path in (ROOT / 'src/hegel_machine').rglob('*.py')} | {
                'scripts/run_anonymous_method_learning.py', 'scripts/run_anonymous_online_research.py',
                'scripts/run_anonymous_method_revision.py', 'scripts/run_anonymous_revision_inquiry.py',
                'scripts/run_online_inquiry_llm.py', 'scripts/run_legacy_session_llm.py',
                'scripts/run_python_method_learning_study.py'}))
        plan['author_limits'].update(max_model_calls=96, token_budget=10_000_000,
            observation_budget=12_000, execution_seconds=3600.)
        plan['limits'].update(max_model_calls=96, token_budget=10_000_000, execution_seconds=600.)
        plan['evaluation']['trajectory_length'] = 24
    return plan


def laboratory(case):
    if case.get('kind') == 'nested_response':
        from scripts.run_nonlinear_mechanism_revision import laboratory as nested_laboratory
        return nested_laboratory(case)
    if case.get('kind') == 'conditional_window':
        from scripts.run_conditional_discovery_study import laboratory as conditional_laboratory
        return conditional_laboratory(case)
    if 'apparatus_seed' in case:
        from scripts.run_anonymous_online_research import representation_laboratory
        return representation_laboratory(case)
    return AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])


def trial_preparer(knowledge, sources, reference_archive, *, agent_type=AnonymousPreparedMethodAgent):
    paid_traces = archived_paid_trace_references(knowledge.get('files', {}), reference_archive)
    candidates = []
    for source in sources:
        if source.get('model'):
            candidates.append({'snapshot_workspace': str(Path(reference_archive) /
                source['model_context_directory']), 'inner_model': source['inner_model']})

    def prepare(case, path, method_source, limits):
        # The author account may already be almost exhausted. Per-case ceilings
        # can reduce that remaining allowance, never replenish it.
        overrides = case.get('limits', {})
        limits = replace(limits, **{key: min(getattr(limits, key), value)
            if key in {'observation_budget', 'compute_budget', 'execution_seconds',
                       'max_model_calls', 'token_budget'} else value
            for key, value in overrides.items()})
        agent = agent_type(None, path, arm='hegel', knowledge=deepcopy(knowledge),
            source_task=False, config=limits, confirmation_seed=case['confirmation_seed'],
            paid_trace_references=paid_traces)
        lab = laboratory(case)
        if ('apparatus_seed' in case
                and case.get('kind') not in {'conditional_window', 'nested_response'}):
            from scripts.run_anonymous_online_research import prepare_representation_prefix
            prepare_representation_prefix(agent, lab, case)
        else:
            actions = lab.confirmation_actions(72, Random(case['prefix_seed']))
            feedback = agent._execute('query', {'actions': actions}, lab)
            if feedback.get('error'):
                raise ValueError(feedback['error'])
            agent.events.append({'step': -1, 'action': 'query', 'arguments': {'actions': actions},
                'feedback': feedback, 'usage': {'backend': 'common_paid_prefix', 'total_tokens': 0}})
        agent.bind_method(method_source)
        return {'agent': agent, 'laboratory': lab, 'inherited_candidates': candidates}
    return prepare


def frozen_transfers(plan, output, knowledge, sources, method_source, *,
                     agent_type=AnonymousPreparedMethodAgent):
    prepare = trial_preparer(knowledge, sources, Path(output) / 'reference_archive',
                            agent_type=agent_type)
    limits = InquiryLLMConfig(**{**plan['limits'], 'max_model_calls': 0, 'token_budget': 0})
    results = []
    for case in plan['transfer_cases']:
        path = Path(output) / 'frozen' / case['id']
        training = path / 'training.json'
        if training.exists():
            # A completed failure is an outcome too; never spend again to replace it.
            result = json.loads(training.read_text())
        else:
            max_steps = plan.get('method_steps', 12)
            if path.exists() and any(path.iterdir()):
                checkpoint = path / 'active.json'
                if not checkpoint.exists():
                    raise ValueError(f'Preserved partial target has no completed-action checkpoint: {path}')
                previous = json.loads(checkpoint.read_text())
                archived = Path(output) / 'interrupted_targets'
                archived.mkdir(exist_ok=True)
                attempt = len(list(archived.glob(case['id'] + '_attempt_*')))
                retained = archived / f"{case['id']}_attempt_{attempt:03d}"
                path.rename(retained)
                overrides = case.get('limits', {})
                restored_limits = replace(limits, **{key: min(getattr(limits, key), value)
                    if key in {'observation_budget', 'compute_budget', 'execution_seconds',
                               'max_model_calls', 'token_budget'} else value
                    for key, value in overrides.items()})
                agent = agent_type(None, path, arm='hegel', knowledge=deepcopy(knowledge),
                    source_task=False, config=restored_limits,
                    confirmation_seed=case['confirmation_seed'])
                lab = laboratory(case)
                agent.restore_practice_research(previous, retained, lab,
                    confirmation_seed=case['confirmation_seed'])
                agent.bind_method(method_source)
                payload = {'agent': agent, 'laboratory': lab}
                max_steps -= len(agent.method_results)
                write_json(path / 'target_continuation.json', {'previous_workspace': str(retained),
                    'completed_steps': len(agent.method_results), 'remaining_steps': max(0, max_steps),
                    'retained_costs': deepcopy(previous['costs']),
                    'scope': 'Continue completed-action checkpoint without replay or a new prefix. '
                        'The interrupted directory retains any unfinished-action files.'})
                if previous['status'] != 'researching':
                    result = agent.result(previous['status'])
                    if previous.get('no_progress'):
                        result['no_progress'] = deepcopy(previous['no_progress'])
                elif max_steps <= 0 and not (agent.method_results and
                        agent.method_results[-1]['feedback'].get('status') == 'confirmation_not_authorized'):
                    result = agent.result('method_step_limit')
                else:
                    result = None
            else:
                payload = prepare(case, path, method_source, limits)
                result = None
            if result is None:
                result = payload['agent'].execute_method(payload['laboratory'], max_steps=max_steps,
                    on_checkpoint=lambda data, path=path: write_json(path / 'active.json', data))
            write_json(training, result)
        results.append((case, result))
        print(json.dumps({'phase': 'frozen_transfer', 'target': case['id'],
            'status': result['status'], 'costs': result['costs']}), flush=True)
    return results


def score_frozen(plan, results):
    records = []
    for case, result in results:
        score = {'status': 'no_submitted_model', 'mse': None, 'passed': False}
        saved_score = Path(result['workspace']) / 'evaluation.json'
        if saved_score.exists():
            score = json.loads(saved_score.read_text())
        elif result['model']:
            rows = laboratory(case).evaluation_rows(length=plan['evaluation']['rows'],
                seed=case['evaluation_seed'])
            score = evaluate_python_model(result['workspace'], result['model'], rows,
                timeout_seconds=plan['limits']['execution_seconds'])
            score.pop('predictions', None)
            threshold = case.get('limits', {}).get('confirmation_mse_threshold',
                plan['limits'].get('confirmation_mse_threshold', 1e-10))
            score['mse_threshold'] = threshold
            score['passed'] = score.get('returncode') == 0 and score.get('mse', float('inf')) <= threshold
        write_json(Path(result['workspace']) / 'evaluation.json', score)
        records.append({'case': case['id'], 'kind': case['kind'], 'status': result['status'],
            'model': result['model'], 'costs': result['costs'], 'terminal': score,
            'fresh_confirmation_passed': (result.get('fresh_confirmation') or {}).get('passed', False),
            'method_steps': len(result['method_results'])})
    return records


def run_learning(plan, output, *, provider_factory):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh directory; preserve previous author attempts')
    write_json(output / 'protocol.json', plan)
    for name in plan.get('runtime_snapshot', []):
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    knowledge, sources = load_research_archive(plan['source_root'])
    knowledge['files'][METHOD_PATH] = EMPTY_METHOD
    write_json(output / 'initial_archive.json', knowledge)
    write_json(output / 'source_outcomes.json', sources)
    reference_archive = output / 'reference_archive'
    previous_reference = Path(plan['source_root']) / 'reference_archive'
    paid_traces = archived_paid_trace_references(knowledge['files'], reference_archive,
        source_workspace=previous_reference if previous_reference.is_dir() else None)
    for name, content in knowledge['files'].items():
        path = reference_archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in paid_traces:
            if path.resolve() != paid_traces[name]:
                path.symlink_to(paid_traces[name])
        elif ((previous_reference / name).is_file()
                and (previous_reference / name).read_text(encoding='utf-8') == content):
            _copy_model_file(previous_reference / name, path)
        else:
            write_text_incrementally(path, content)
    workspace = output / 'author'
    author = AnonymousResearchMethodLearner(provider_factory(workspace), workspace,
        knowledge=knowledge, paid_trace_references=paid_traces, cases=plan['practice_cases'], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan['author_limits']),
        practice_limits=InquiryLLMConfig(**plan['limits']),
        prepare_trial=trial_preparer(knowledge, sources, reference_archive))
    if plan.get('continued_author'):
        previous = json.loads(Path(plan['continued_author']).read_text())
        author.restore_practice(previous)
    learned = author.run(on_checkpoint=lambda data: write_json(workspace / 'active.json', data))
    write_json(workspace / 'training.json', learned)
    published = learned['status'] == 'completed' and learned.get('method_source') is not None
    write_json(output / 'published_method.json', {'published': published,
        'source': learned.get('method_source'), 'lesson': learned.get('lesson'),
        'practice_trials': learned['practice_trials']})
    print(json.dumps({'phase': 'author', 'status': learned['status'],
        'published': published, 'costs': learned['costs']}), flush=True)
    frozen, controls = [], None
    if published:
        transfer_memory = register_published_method(knowledge, learned, max_steps=12)
        transfer_memory['research_roles']['method'] = METHOD_PATH
        write_json(output / 'transfer_archive.json', transfer_memory)
        frozen = frozen_transfers(plan, output, transfer_memory, sources, learned['method_source'])
        control_plan = {'study': 'anonymous_common_learned_archive_live_adoption',
            'arms': ['code_react', 'hegel'], 'cases': [case for case in plan['transfer_cases']
                if case['id'] in plan['live_control_cases']],
            'limits': plan['limits'], 'prefix_rows': plan['prefix_rows'],
            'model': plan['model'], 'provider': plan['provider'],
            'evaluation': plan['evaluation'], 'scope': plan['scope']}
        # run_study scores only after both arms finish all of its target research.
        prepare_prefix = None
        if 'apparatus_seed' in plan['transfer_cases'][0]:
            from scripts.run_anonymous_online_research import prepare_representation_prefix
            prepare_prefix = prepare_representation_prefix
        controls = run_study(control_plan, output / 'controls', provider_factory=provider_factory,
            knowledge=transfer_memory, laboratory_factory=laboratory,
            prepare_prefix=prepare_prefix,
            workers=plan.get('execution_workers', 1))
    frozen_records = score_frozen(plan, frozen)
    summary = {'study': plan['study'], 'status': 'completed', 'method_published': published,
        'author_status': learned['status'], 'author_costs': learned['costs'],
        'earlier_source_costs': total_costs([source['costs'] for source in sources]),
        'author_usage': usage_summary(learned), 'practice_trials': len(learned['practice_trials']),
        'prior_author_phases': plan.get('prior_author_phases', []),
        'frozen': frozen_records, 'live_controls': controls,
        'unavailable_frozen_targets': 0 if published else len(plan['transfer_cases']),
        'unavailable_live_episodes': 0 if published else 2 * len(plan['live_control_cases']),
        'new_costs': total_costs([learned['costs'], *[result['costs'] for _, result in frozen],
            *[row['costs'] for row in (controls or {}).get('trials', [])]]),
        'scope': plan['scope']}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--timeout-seconds', type=float, default=300.)
    parser.add_argument('--continue-author', type=Path,
        help='Additional development phase preserving an unpublished author and all paid practice')
    args = parser.parse_args()
    plan = protocol(args.source, model=args.model, timeout_seconds=args.timeout_seconds)
    if args.continue_author:
        previous = json.loads(args.continue_author.read_text())
        if previous.get('method_source') is not None or previous['status'] not in {
                'provider_failed', 'token_budget_exhausted', 'model_call_budget_exhausted'}:
            raise ValueError('continue an actual unpublished interrupted or exhausted author')
        old_plan = json.loads((args.continue_author.parent.parent / 'protocol.json').read_text())
        for key in ('source_root', 'practice_cases', 'transfer_cases', 'live_control_cases'):
            plan[key] = old_plan[key]
        plan['continued_author'] = str(args.continue_author.resolve())
        plan['prior_author_phases'] = [*old_plan.get('prior_author_phases', []), {
            'result': plan['continued_author'], 'status': previous['status'],
            'costs': previous['costs'], 'usage': usage_summary(previous)}]
        plan['continuation'] = ('Additional development allowance after the preceding attempt ended. '
            'Restore actual source and paid trials without replay; unchanged task cases and seeds. '
            'Repair code serialization, attach paid practice after method edits, and remove irrelevant '
            'instrument suggestions. Earlier costs and failures remain separate, not fresh replication.')
    plan['provider']['auth_mode'] = verify_codex_subscription()
    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model, reasoning_effort='medium',
            timeout_seconds=args.timeout_seconds)
    summary = run_learning(plan, args.output, provider_factory=provider)
    print(json.dumps({'method_published': summary['method_published'],
        'frozen_passed': sum(item['terminal']['passed'] for item in summary['frozen']),
        'new_costs': summary['new_costs']}), flush=True)


if __name__ == '__main__':
    main()
