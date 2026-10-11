"""Resume the stopped development world through the existing live researcher."""
from copy import deepcopy
import argparse
import importlib.util
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


def run(resume=None, output=None, program=None, replay_first_event=None, focus_source=None,
        max_model_calls=32, arm='hegel', proposal_calls=0):
    output = Path(output).resolve() if output else OUT / 'context_repair' if resume else OUT
    output.mkdir(exist_ok=True)
    source = ROOT / 'artifacts/discoveryworld_response_ablation_20261008'
    previous_dir = Path(resume).resolve() if resume else source / 'challenge_s0_initial/with_response'
    previous = json.loads((previous_dir / 'result.json').read_text())
    replayed = json.loads(Path(replay_first_event).read_text()) if replay_first_event else None
    old_plan = json.loads((source / 'protocol.json').read_text())
    if focus_source:
        # The paired development control changes only this historical scheduler;
        # both arms keep current context, analysis, tools and restoration code.
        spec = importlib.util.spec_from_file_location(
            'hegel_machine.cognition._research_focus_baseline', Path(focus_source).resolve())
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        DiscoveryWorldAgent._prepare_research_focus = baseline.DiscoveryWorldAgent._prepare_research_focus
    model = json.loads((ROOT / 'artifacts/discoveryworld_observed_settings_20261008'
                       '/continued/protocol.json').read_text())['model']
    task = next(c['task'] for c in old_plan['cases'] if c['id'] == 'challenge_s0_initial')
    limits = {**old_plan['limits'], 'max_model_calls': max_model_calls}
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
        'runtime': old_plan['runtime'], 'model': model, 'limits': limits, 'arm': arm,
        'auth_mode': verify_codex_subscription(), 'question': question,
        'entry': {'action': 'run', 'operator': 'inquire', 'max_model_calls': max_model_calls},
        'retained_procedure': METHOD_PATH,
        'local_research_scheduling': 'Existing analysis, child-return and stopped-program review '
            'are scoped to the current inquiry and descendants. A retitled local answer does '
            'not complete the original assignment. Declared unmeasured, unbound children enter '
            'the existing inquiry loop once per parent invocation, within the shared account '
            'and retaining a parent decision. Host scheduling has a separate event origin. '
            'Scientific proposals remain model-authored.',
        'scope': f"Opened seed0 development continuation after {previous['costs']['environment_steps']} paid actions. "
                 'The host initiates a neutral local research assignment and schedules its '
                 'declared unmeasured children. The model authors the questions, scientific '
                 'actions, programs, interpretations and stopping decisions. '
                 'No forced parent rerun or scientific hints are added during execution. '
                 'Inherited tools and candidate families were designed by humans/assistants; '
                 'this trial tests model-directed continuation, not tool invention or transfer.'}
    if resume:
        protocol['continuation'] = (f'Retain all saved decisions and cumulative costs under the explicit '
                                    f'total {max_model_calls}-call ceiling. ')
        protocol['continuation'] += ('Bind the retained procedure through the existing question metadata. '
            'Scientific source, assignment, controls and account match the prior neutral inquiry.' if program else
            'Only the existing context preview adapter changed before restart.')
    if replayed:
        protocol['replayed_first_decision'] = {
            'source_result': replayed['source_result'],
            'source_model_call': replayed['event']['model_call'],
            'scope': 'Replay one unchanged, previously paid model decision through the current '
                     'executor. Its original usage joins the inherited account before new '
                     'model calls. This is a conditioned development continuation.'}
        write_json(output / 'replayed_first_event.json', replayed)
    protocol['parent_direction_policy'] = {
        'source': str(Path(focus_source).resolve()) if focus_source else 'current production source',
        'overridden_callable': 'DiscoveryWorldAgent._prepare_research_focus' if focus_source else None,
        'scope': 'Completed physical comparisons return to parent-level research choice in current '
                 'production. A supplied historical focus source overrides only that scheduler; '
                 'all other tools, model, public evidence and account are shared.'}
    if arm == 'code_archive':
        protocol['local_research_scheduling'] = (
            'Ordinary Code strategy without automatic Hegel research focus or pending-child delegation. '
            'The same callable inquiry, contrast, numerical research and Python/world tools remain available.')
        protocol['parent_direction_policy']['scope'] = (
            'The ordinary Code arm does not schedule Hegel research focus. It chooses its own next action.')
        protocol['scope'] = (
            f"Opened seed0 Code continuation after {previous['costs']['environment_steps']} paid actions. "
            'The host initiates the same neutral local assignment; the model chooses scientific content '
            'and research organization using the same tools and inherited evidence. '
            'No mid-run scientific hints or source edits are added.')
    if proposal_calls:
        protocol['proposal_stage'] = {
            'operator': 'propose_explanations', 'path': 'shared/explanation_proposal.py',
            'max_model_calls': proposal_calls,
            'scope': 'A host-initiated proposal stage on the same paid prefix and model account, '
                     'then the unchanged parent assignment chooses adoption and execution. '
                     'Proposal content, predictions and interpretations are model-authored. '
                     'This tests the connection, not autonomous selection of the new operator.'}
    write_json(output / 'protocol.json', protocol)
    archived = output / 'host_sources'
    archived.mkdir()
    if focus_source:
        shutil.copyfile(focus_source, archived / 'research_focus_baseline.py')
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
    agent = DiscoveryWorldAgent(provider, workspace, arm=arm, limits=limits)
    with DiscoveryWorldLab(task, old_plan['runtime'], workspace) as lab:
        agent.lab = lab
        replay = restore_repair(agent, lab, {**previous, 'episode_status': 'running'},
                                previous_dir / 'agent')
        write_json(output / 'replay.json', replay)
        before = deepcopy(agent.result()['costs'])
        prefix = lab.environment_steps
        replay_checkpoint_pending = bool(replayed)
        if replayed:
            live_next_action = provider.next_action

            def replay_then_live(request):
                provider.next_action = live_next_action
                event = replayed['event']
                agent.started -= event['usage']['wall_time_seconds']
                return {key: deepcopy(event[key]) for key in ('action', 'arguments', 'usage')}

            provider.next_action = replay_then_live

        def checkpoint(data):
            nonlocal before, replay_checkpoint_pending
            if replay_checkpoint_pending and data['costs']['model_calls'] > before['model_calls']:
                before = deepcopy(data['costs'])
                write_json(output / 'restored_decision.json', data)
                replay_checkpoint_pending = False
            write_json(output / 'active.json', {**data, 'study_status': 'running',
                       'prior_costs': before, 'replayed_environment_steps': prefix})
            print(json.dumps({'model_calls': data['costs']['model_calls'],
                'environment_steps': lab.environment_steps, 'score': data['score'],
                'last_action': data['events'][-1]['action'] if data['events'] else None}), flush=True)

        agent._on_checkpoint = checkpoint
        checkpoint(agent.result())
        proposal = None
        if proposal_calls:
            params = {**protocol['proposal_stage'], 'question': question}
            params.pop('scope')
            # Keep a parent decision available on the shared account.
            params['max_model_calls'] = min(proposal_calls, max_model_calls - agent.costs['model_calls'] - 1)
            proposal = agent._execute('run', params, lab)
            write_json(output / 'proposal.json', proposal)
            handoff = {'action': 'run', 'arguments': params,
                'trigger_origin': 'host_proposal_stage',
                'feedback': {key: deepcopy(value) for key, value in proposal.items() if key != 'steps'}}
            agent.events.append(handoff)
            if question['id'] in agent.llm_inquiries:
                agent.llm_inquiries[question['id']]['events'].append(deepcopy(handoff))
            agent._save_state()
            checkpoint(agent.result())
        inquiry = agent._execute('run', {'operator': 'inquire', 'question': question,
                                        'max_model_calls': max_model_calls}, lab)
        final = agent.result()
        final['costs'].update(observation_cost=lab.environment_steps,
                             compute_cost=final['costs']['research_compute_units'])
        final.update(study_status='finished', episode_status=final['status'],
            inquiry=inquiry, prior_costs=before, replayed_environment_steps=prefix,
            new_costs={k: v - before[k] for k, v in final['costs'].items()})
        if proposal is not None:
            final['proposal_stage'] = proposal
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
    parser.add_argument('--replay-first-event', type=Path,
                        help='Saved paid first decision to replay before the live continuation')
    parser.add_argument('--focus-source', type=Path,
                        help='Historical scheduler source for the matched development control')
    parser.add_argument('--max-model-calls', type=int, default=32,
                        help='Total cumulative model-call ceiling, including restored calls')
    parser.add_argument('--arm', choices=['hegel', 'code_archive'], default='hegel',
                        help='Existing scientific-agent condition')
    parser.add_argument('--proposal-calls', type=int, default=0,
                        help='Optional initial explanation authoring stage, within the same total account')
    args = parser.parse_args()
    run(args.resume, args.output, args.program, args.replay_first_event, args.focus_source,
        args.max_model_calls, args.arm, args.proposal_calls)
