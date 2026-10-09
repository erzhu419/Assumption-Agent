#!/usr/bin/env python3
"""Continue an already opened frozen failure through the existing child LLM."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_discoveryworld_pilot import write_json
from scripts.run_online_inquiry_llm import verify_codex_subscription


def run(source_result, output, calls=None, resume=None):
    resume_plan = None
    if resume:
        resume = Path(resume).resolve()
        resume_plan = json.loads((resume.parent / 'protocol.json').read_text())
        source_result = resume_plan['source_result']
        if calls is not None and calls != resume_plan['limits']['max_model_calls']:
            raise ValueError('Resume keeps the original total model-call allowance')
        calls = resume_plan['limits']['max_model_calls']
    elif calls is None:
        calls = 64
    source_result, output = Path(source_result).resolve(), Path(output).resolve()
    previous_path = resume or source_result
    previous = json.loads(previous_path.read_text())
    plan = json.loads((source_result.parents[2] / 'protocol.json').read_text())
    task = next(t for t in plan['tasks'] if t['id'] == source_result.parent.parent.name)
    if previous['score']['completed']:
        raise ValueError('Select an unfinished frozen program execution')
    original_source = Path(json.loads(Path(plan['method_bundle']).read_text())['source'])
    model = json.loads((original_source.parents[1] / 'protocol.json').read_text())['model']
    limits = {**plan['limits'], 'max_model_calls': calls}
    task = {**task, 'split': 'development'}
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'source_result': str(source_result), 'task': task, 'model': model,
        'runtime': plan['runtime'], 'limits': limits, 'auth_mode': verify_codex_subscription(),
        'prior_costs': previous['costs'],
        'resume_result': str(resume) if resume else None,
        'scope': 'Already opened task reused as development. Replay the paid public prefix and '
                 'replace only the exhausted-family stop with existing inquire. The model chooses '
                 'the new variables, forms and tests; preserve the original frozen failure.'}
    write_json(output / 'protocol.json', protocol)
    workspace = output / 'agent'
    provider = CodexCodeProvider(workspace, model=model['requested_alias'],
        reasoning_effort=model['reasoning_effort'],
        timeout_seconds=limits['provider_request_timeout_seconds'])
    agent = DiscoveryWorldAgent(provider, workspace, arm='hegel', limits=limits)
    with DiscoveryWorldLab(task, plan['runtime'], workspace) as lab:
        agent.lab = lab
        replay = restore_repair(agent, lab,
            {**previous, 'episode_status': previous['status']}, previous_path.parent / 'agent')
        write_json(output / 'replay.json', replay)
        inherited = (workspace / METHOD_PATH).read_text()
        agent._execute('write', {'path': 'shared/frozen_parent_before_extension.py', 'code': inherited}, lab)
        source = (ROOT / 'src/hegel_machine/cognition/world_reference_recalibration.py').read_text()
        agent._execute('write', {'path': METHOD_PATH, 'code': source}, lab)
        # Earlier frozen runs already record rejected out-of-range predictions in
        # their relation research, even when no physical test was possible.
        state_path = workspace / 'shared/reference_recalibration.json'
        state = json.loads(state_path.read_text())
        for qid, research in agent.relation_research.items():
            for prediction in (research.get('selected_probe') or {}).get('predictions', []):
                if not 0 <= prediction <= 10000:
                    state.setdefault('out_of_range', []).append({
                        'prediction': prediction, 'allowed_range': [0, 10000], 'fit_question': qid})
        state_path.write_text(json.dumps(state, indent=2))
        prefix_steps = lab.environment_steps
        costs_before = deepcopy(agent.costs)
        prior_child_answers = agent.child_answers_returned

        def checkpoint(data):
            write_json(output / 'active.json', {**data, 'study_status': 'running',
                'replayed_environment_steps': prefix_steps,
                'new_environment_steps': lab.environment_steps - prefix_steps})
            print(json.dumps({'calls': agent.costs['model_calls'],
                'actions': lab.environment_steps - prefix_steps,
                'score': data['score']['scoreNormalized'],
                'completed': data['score']['completedSuccessfully']}), flush=True)

        agent._on_checkpoint = checkpoint
        execution = agent._execute('run', {'operator': 'procedure', 'path': METHOD_PATH,
            'max_steps': 256, 'question': {'id': 'task'}}, lab)
        result = agent.result()
        result['costs'].update(observation_cost=lab.environment_steps,
            compute_cost=result['costs']['research_compute_units'])
        result.update(study_status='finished', task_id=task['id'], execution=execution,
            replayed_environment_steps=prefix_steps,
            new_environment_steps=lab.environment_steps - prefix_steps,
            prior_child_answers_returned=prior_child_answers,
            new_child_answers_returned=agent.child_answers_returned - prior_child_answers,
            new_costs={key: value - costs_before[key] for key, value in result['costs'].items()},
            entry_source_revised=(workspace / METHOD_PATH).read_text() != source)
        write_json(output / 'result.json', result)
        brief = {key: result[key] for key in ('study_status', 'status', 'task_id',
            'new_environment_steps', 'new_costs', 'entry_source_revised')}
        brief.update(procedure_status=execution['status'],
            completedSuccessfully=result['score']['completedSuccessfully'],
            scoreNormalized=result['score']['scoreNormalized'],
            prior_child_answers_returned=result['prior_child_answers_returned'],
            new_child_answers_returned=result['new_child_answers_returned'])
        write_json(output / 'brief.json', brief)
    return brief


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument('--source-result', type=Path)
    start.add_argument('--resume', type=Path, help='Continue a stopped development result under its original account')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calls', type=int, help='Total model calls, default64; retained on resume')
    args = parser.parse_args()
    print(json.dumps(run(args.source_result, args.output, args.calls, args.resume)))
