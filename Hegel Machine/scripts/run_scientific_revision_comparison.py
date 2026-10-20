#!/usr/bin/env python3
"""Same paid Luna scene: local scientific revision versus an in-place review."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.common_archive_inquiry import (
    CommonArchiveInquiryAgent, FlatScientificRevisionAgent)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.evidence_readout_proposal import prepare_evidence_readout
from hegel_machine.cognition.scientific_generalization import resume_generalization_review
from scripts.run_anonymous_method_learning import laboratory
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_conditional_online_method_repair import rescue_protocol
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_python_method_learning_study import LIMIT_COSTS


def read_json(path):
    return json.loads(Path(path).read_text())


def protocol(source_study, *, resume_arm=None, resume_result=None, previous=None,
             resume_question=None):
    if resume_arm is not None:
        source = Path(source_study).resolve()
        plan = read_json(source / 'protocol.json')
        summary_path = source / 'summary.json'
        if summary_path.exists():
            summary = read_json(summary_path)
            trial = next(t for t in summary['trials'] if t['arm'] == resume_arm)
            if trial.get('model') is not None or trial['fresh_confirmation_passed']:
                raise ValueError('Continue an unresolved paid source, not a confirmed submission')
            case = next(c for c in plan['online']['cases'] if c['id'] == trial['case'])
            costs = trial['costs']
        else:
            if resume_result is None:
                raise ValueError('An unfinished source requires its actual --resume-result checkpoint')
            if len(plan['online']['cases']) != 1 or resume_arm not in plan['online']['arms']:
                raise ValueError('Checkpoint recovery requires the registered single source case and arm')
            case = plan['online']['cases'][0]
            if previous is None:
                previous = read_json(resume_result)
        if previous is not None:
            if previous.get('model') is not None or (previous.get('fresh_confirmation') or {}).get('passed'):
                raise ValueError('Continue an unresolved paid source, not a confirmed submission')
            costs = previous['costs']
        plan.pop('resume_question', None)
        plan.pop('initial_inquiry', None)
        if resume_question is not None:
            if previous is None or not any(q['id'] == resume_question
                                          for q in previous['online_questions']):
                raise ValueError('--resume-question must identify an existing checkpoint question')
        workspace = source / 'online' / case['id'] / resume_arm
        allowance = asdict(InquiryLLMConfig(max_model_calls=48, token_budget=12_000_000,
            observation_budget=4096, execution_seconds=7200., compute_budget=1_000_000,
            confirmation_rows=plan['online']['limits']['confirmation_rows'],
            confirmation_mse_threshold=plan['online']['limits']['confirmation_mse_threshold']))
        plan.update(study='luna_paid_source_executable_synthesis_continuation',
            source_study=str(source), resume_result=str(Path(resume_result).resolve()
                if resume_result is not None else workspace / 'training.json'),
            resume_workspace=str(workspace),
            initial_archive=str(source / 'online' / 'initial_archive.json'),
            online_allowance=allowance,
            comparison='Single development continuation from one unresolved Luna arm checkpoint; '
                'not a new matched comparison or an independent task.',
            scope='All previously paid source evidence and model work are retained. '
                'No Astra solution, host-authored scientific model or selected formula is supplied.')
        if resume_question is not None:
            plan['resume_question'] = resume_question
        online = plan['online']
        online.update(study=plan['study'], cases=[case], arms=[resume_arm],
            limits=deepcopy(allowance), scope=plan['scope'], comparison=plan['comparison'])
        for limit, cost in LIMIT_COSTS:
            online['limits'][limit] += costs[cost]
        online['limits']['token_budget'] += costs['reported_tokens']
        online['runtime_snapshot'] = sorted(set(online['runtime_snapshot']) | {
            str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py')})
        return plan
    plan = rescue_protocol(source_study, model='gpt-6-luna', timeout_seconds=1800.,
        online_limits=InquiryLLMConfig(max_model_calls=96, token_budget=16_000_000,
            observation_budget=16384, execution_seconds=14400., compute_budget=4_000_000,
            confirmation_rows=96, confirmation_mse_threshold=1e-6))
    plan.update(study='luna_same_scene_scientific_revision_comparison',
        comparison='Same failed confirmation, scientific review assignment, tools, total '
            'allowance and local review limit. Recursive review uses the existing isolated '
            'inquiry notebook and parent return; flat review remains in the running notebook. '
            'Both retain the same explicit inquiry tools and executable candidate bindings.',
        scope='One post-selected development source restored before Astra rescue. '
            'No Astra predictor, successful research trace or selected scientific formula '
            'is provided. Source confirmation tests adoption; this is not transfer evidence.',
        next_stage='Compare source research and independent confirmation first. '
            'No automatic method compilation or repetition of the completed 56 targets.')
    plan['online'].update(study=plan['study'], arms=['recursive', 'flat'],
                          scope=plan['scope'], comparison=plan['comparison'])
    plan['online']['provider']['reasoning_effort'] = 'high'
    plan['online']['runtime_snapshot'] = sorted(set(plan['online']['runtime_snapshot']) | {
        'scripts/run_scientific_revision_comparison.py',
        'scripts/run_anonymous_revision_inquiry.py'})
    return plan


def comparison_agent(provider, workspace, *, arm, **kwargs):
    cls = {'recursive': CommonArchiveInquiryAgent, 'flat': FlatScientificRevisionAgent}[arm]
    # Keep all ordinary Hegel tools and admission requirements identical. The
    # cohort name changes only where this failed-confirmation review executes.
    return cls(provider, workspace, arm='hegel', **kwargs)


def run_comparison(plan, output, *, provider_factory, workers=2, previous=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Retain this paid comparison; unfinished calls require checkpoint recovery')
    write_json(output / 'protocol.json', plan)
    for name in plan['online']['runtime_snapshot']:
        dest = output / 'source_at_run' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    if previous is None:
        previous = read_json(plan['resume_result'])

    def restore(agent, lab, case):
        restored = agent.restore_research(previous, plan['resume_workspace'], lab,
            confirmation_seed=case['confirmation_seed'])
        restored.update(resume_origin=plan['resume_result'],
                        restored_events=len(previous['events']),
                        restored_model_calls=previous['costs']['model_calls'])
        if plan.get('initial_inquiry') is not None:
            arguments = {'operator': 'inquire', **deepcopy(plan['initial_inquiry'])}
            inquiry = agent._execute('run', arguments, lab)
            event_index = len(agent.events)
            agent.events.append({'step': event_index, 'action': 'run',
                'arguments': arguments, 'feedback': inquiry,
                'origin': 'host_method_practice',
                'usage': {'backend': 'host_method_practice', 'total_tokens': 0}})
            restored['host_method_practice'] = {
                'event_index': event_index, 'question_id': arguments['question']['id'],
                'max_model_calls': arguments['max_model_calls'], 'status': inquiry['status']}
        elif plan.get('resume_question') is not None:
            identifier = plan['resume_question']
            question = next(q for q in agent.questions if q['id'] == identifier)
            refit_retry = resume_generalization_review(agent, question, lab)
            if refit_retry is not None:
                restored['generalization_review_retry'] = refit_retry
            parent_id = question['assignment'].get('parent_id')
            inquiry = agent._start_inquiry({'question': {
                'id': identifier, 'parent_id': parent_id},
                'max_model_calls': 12}, parent_event_index=len(agent.events) - 1)
            restored['host_scheduled_resume'] = {
                'question_id': identifier, 'parent_id': parent_id,
                'max_model_calls': 12, 'status': inquiry['status']}
            readout = prepare_evidence_readout(agent, question)
            if readout is not None:
                restored['host_scheduled_resume']['evidence_readout_proposal'] = readout
        write_json(agent.workspace / 'restoration.json', restored)
        write_json(agent.workspace / 'active.json', agent.result('researching'))
        return {'prior_research_cost': deepcopy(previous['costs']),
                'resume_origin': plan['resume_result']}

    write_json(output / 'progress.json', {'stage': ('paired_source_research'
        if len(plan['online']['arms']) > 1 else 'source_research_continuation'),
        'arms': plan['online']['arms'], 'model': plan['online']['model']})
    result = run_study(plan['online'], output / 'online', provider_factory=provider_factory,
        knowledge=read_json(plan.get('initial_archive',
            Path(plan['source_study']) / 'initial_archive.json')),
        prepare_prefix=restore, laboratory_factory=laboratory,
        workers=workers, agent_factory=comparison_agent)
    result.update(comparison=plan['comparison'], next_stage=plan['next_stage'])
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', {'stage': 'completed', 'trials': [
        {'arm': r['arm'], 'status': r['status'],
         'fresh_confirmation_passed': r['fresh_confirmation_passed'],
         'terminal_passed': r['terminal']['passed'], 'additional_costs': r['continuation_cost']}
        for r in result['trials']]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume-arm', choices=['recursive', 'flat'],
                        help='Continue this completed unresolved arm with all its paid evidence')
    parser.add_argument('--resume-result', type=Path,
                        help='Recovered result for the same paid source workspace')
    parser.add_argument('--resume-question',
                        help='Reopen this existing question with its original notebook and parent')
    args = parser.parse_args()
    previous = read_json(args.resume_result) if args.resume_result is not None else None
    plan = protocol(args.source_study, resume_arm=args.resume_arm,
                    resume_result=args.resume_result, previous=previous,
                    resume_question=args.resume_question)
    cfg = plan['online']['provider']
    cfg['auth_mode'] = verify_codex_subscription()
    provider = lambda workspace: CodexCodeProvider(workspace,
        model=plan['online']['model'], reasoning_effort=cfg['reasoning_effort'],
        timeout_seconds=cfg['timeout_seconds'])
    result = run_comparison(plan, args.output, provider_factory=provider, previous=previous)
    print(json.dumps({'status': result['status'], 'trials': len(result['trials'])}), flush=True)


if __name__ == '__main__':
    main()
