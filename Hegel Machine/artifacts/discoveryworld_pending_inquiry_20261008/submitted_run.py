"""Resume the stopped development world through the existing live researcher."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_discoveryworld_pilot import write_json
from scripts.run_online_inquiry_llm import verify_codex_subscription


def run(resume=None, output=None, program=None):
    output = Path(output).resolve() if output else OUT / 'context_repair' if resume else OUT
    output.mkdir(exist_ok=True)
    source = ROOT / 'artifacts/discoveryworld_response_ablation_20261008'
    previous_dir = Path(resume).resolve() if resume else source / 'challenge_s0_initial/with_response'
    previous = json.loads((previous_dir / 'result.json').read_text())
    old_plan = json.loads((source / 'protocol.json').read_text())
    model = json.loads((ROOT / 'artifacts/discoveryworld_observed_settings_20261008'
                       '/continued/protocol.json').read_text())['model']
    task = next(c['task'] for c in old_plan['cases'] if c['id'] == 'challenge_s0_initial')
    limits = {**old_plan['limits'], 'max_model_calls': 32}
    question = {
        'id': 'stopped_procedure_research', 'parent_id': 'task', 'depends_on': [],
        'question': 'The current retained procedure stopped without completing the task. '
                    'Inspect its saved execution and current public evidence, decide what '
                    'research should happen next, execute it, and retain any revised '
                    'executable procedure with an evidence-based account of its result.'}
    if program:
        question['parent_program_path'] = program
    workspace = output / 'agent'
    workspace.mkdir(exist_ok=False)
    protocol = {
        'source_result': str(previous_dir / 'result.json'), 'task': task,
        'runtime': old_plan['runtime'], 'model': model, 'limits': limits,
        'auth_mode': verify_codex_subscription(), 'question': question,
        'entry': {'action': 'run', 'operator': 'inquire', 'max_model_calls': 32},
        'retained_procedure': METHOD_PATH,
        'local_research_scheduling': 'Existing analysis, child-return and stopped-program review '
            'are scoped to the current inquiry and descendants. A retitled local answer does '
            'not complete the original assignment. Declared unmeasured, unbound children enter '
            'the existing inquiry loop once per parent invocation, within the shared account '
            'and retaining a parent decision. Host scheduling has a separate event origin. '
            'Scientific proposals remain model-authored.',
        'scope': 'Opened seed0 development continuation after 348 paid actions. '
                 'The host initiates a neutral local research assignment and schedules its '
                 'declared unmeasured children. The model authors the questions, scientific '
                 'actions, programs, interpretations and stopping decisions. '
                 'No forced parent rerun or scientific hints are added during execution. '
                 'Inherited tools and candidate families were designed by humans/assistants; '
                 'this trial tests model-directed continuation, not tool invention or transfer.'}
    if resume:
        protocol['continuation'] = 'Retain the rejected first request and original total 32-call account. '
        protocol['continuation'] += ('Bind the retained procedure through the existing question metadata. '
            'Scientific source, assignment, controls and account match the prior neutral inquiry.' if program else
            'Only the existing context preview adapter changed before restart.')
    write_json(output / 'protocol.json', protocol)
    archived = output / 'host_sources'
    archived.mkdir()
    for name in ('code_agent', 'code_provider_context', 'discoveryworld_agent', 'world_llm_inquiry',
                 'world_procedure_repair',
                 'world_reference_recalibration', 'world_reference_hypothesis',
                 'world_reference_expansion', 'world_reference_response',
                 'world_response_hypotheses', 'world_relation_research', 'world_experiment_choice'):
        shutil.copyfile(ROOT / f'src/hegel_machine/cognition/{name}.py', archived / f'{name}.py')
    initial_source = (previous_dir / 'agent' / METHOD_PATH).read_text()
    (output / 'initial_method.py').write_text(initial_source)
    shutil.copyfile(Path(__file__), output / 'submitted_run.py')
    provider = CodexCodeProvider(workspace, model=model['requested_alias'],
        reasoning_effort=model['reasoning_effort'],
        timeout_seconds=limits['provider_request_timeout_seconds'])
    agent = DiscoveryWorldAgent(provider, workspace, arm='hegel', limits=limits)
    with DiscoveryWorldLab(task, old_plan['runtime'], workspace) as lab:
        agent.lab = lab
        replay = restore_repair(agent, lab, {**previous, 'episode_status': 'running'},
                                previous_dir / 'agent')
        write_json(output / 'replay.json', replay)
        before = deepcopy(agent.result()['costs'])
        prefix = lab.environment_steps

        def checkpoint(data):
            write_json(output / 'active.json', {**data, 'study_status': 'running',
                       'prior_costs': before, 'replayed_environment_steps': prefix})
            print(json.dumps({'model_calls': data['costs']['model_calls'],
                'environment_steps': lab.environment_steps, 'score': data['score'],
                'last_action': data['events'][-1]['action'] if data['events'] else None}), flush=True)

        agent._on_checkpoint = checkpoint
        checkpoint(agent.result())
        inquiry = agent._execute('run', {'operator': 'inquire', 'question': question,
                                        'max_model_calls': 32}, lab)
        final = agent.result()
        final['costs'].update(observation_cost=lab.environment_steps,
                             compute_cost=final['costs']['research_compute_units'])
        final.update(study_status='finished', episode_status=final['status'],
            inquiry=inquiry, prior_costs=before, replayed_environment_steps=prefix,
            new_costs={k: v - before[k] for k, v in final['costs'].items()})
        write_json(output / 'result.json', final)
        brief = {'study_status': 'finished', 'inquiry_status': inquiry['status'],
            'episode_status': final['status'], 'score': final['score'],
            'new_costs': final['new_costs'], 'replayed_environment_steps': prefix,
            'retained_procedure_changed': (workspace / METHOD_PATH).read_text() != initial_source,
            'inquiries': [{'id': k, 'status': v['status'], 'model_calls': v['model_calls']}
                         for k, v in final['research']['llm_inquiries'].items()]}
        write_json(output / 'brief.json', brief)
        print(json.dumps(brief), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--program', help='Existing research program to bind to the delegated question')
    args = parser.parse_args()
    run(args.resume, args.output, args.program)
