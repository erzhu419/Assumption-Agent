"""Replay the paid state and test native Boolean proposals and their LLM adoption."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.world_llm_inquiry import refresh_condition_proposal
from hegel_machine.cognition.world_reference_response import response_premise_design_action
from scripts.run_response_ambiguity_continuation import restore_saved
from scripts.run_response_family_recovery import account
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json

SOURCE = ROOT / 'artifacts/response_design_feedback_20261009/continued/result.json'
OUT = Path(__file__).parent
AUTHOR = 'shared/response_representation_proposal_premise.py'


def run(branch):
    output = OUT / branch
    source = (OUT / 'continued/result.json' if branch == 'parent'
              else OUT / 'live/result.json' if branch == 'continued' else SOURCE)
    agent, lab, context = restore_saved(source, output, additional_calls=2 if branch == 'parent' else 0)
    protocol = json.loads((output / 'protocol.json').read_text())
    protocol.update(study='finite_boolean_condition_synthesis', branch=branch,
        scope='One previously opened development condition, same paid 451-observation prefix. '
              'Native finite relation/composition generates rules from actual Boolean outcomes. '
              'The existing controller executes the selected experimental setting. '
              'The offline branch tests the algorithm with zero new LLM calls. '
              'The live branch explicitly reopens the existing proposal and tests model adoption '
              'using its remaining call allowance. Neither branch is an independent performance sample; '
              'no scientific program, experimental point or answer is supplied by this runner.')
    if branch == 'parent':
        protocol['scope'] += (' After the first parent declined, correct the current proposal status '
            'and explicitly resume that parent with two additional calls. This is a separate '
            'development continuation, not a same-budget comparison.')
    write_json(output / 'protocol.json', protocol)

    def checkpoint(data):
        latest = data['events'][-1] if len(data['events']) > context['prefix_events'] else {}
        brief = {'status': 'running', 'new_model_calls': data['costs']['model_calls'] - context['costs_before']['model_calls'],
            'new_observations': len(agent.observations) - context['prefix_observations'],
            'latest_action': latest.get('action'), 'latest_operator': latest.get('arguments', {}).get('operator')}
        write_json(output / 'active.json', brief)
        print(json.dumps({'branch': branch, **brief}), flush=True)

    agent._on_checkpoint = checkpoint
    if branch == 'offline':
        before = refresh_condition_proposal(agent, AUTHOR)
        write_json(output / 'conditions_before.json', before)
        action = before['experiment_action']
        execution = agent._execute(action['action'], action['arguments'], lab) if action else {
            'status': 'no_discriminating_experiment_proposed'}
        write_json(output / 'trial.json', execution)
        write_json(output / 'conditions_after.json', agent.llm_inquiries['proposal:' + AUTHOR]['condition_research'])
    else:
        protocol['auth_mode'] = verify_codex_subscription()
        write_json(output / 'protocol.json', protocol)
        agent.provider = CodexCodeProvider(agent.workspace, model='gpt-6-luna',
            reasoning_effort='medium', timeout_seconds=180)
        if branch == 'parent':
            refresh_condition_proposal(agent, AUTHOR)
            proposal = agent.llm_inquiries['proposal:' + AUTHOR]['proposal']
        else:
            comparison = agent.llm_inquiries['proposal:shared/response_representation_proposal.py']['proposal']['comparison_result']
            action = response_premise_design_action(comparison)
            proposal = agent._execute(action['action'], {**action['arguments'], 'max_model_calls': 3}, lab)
            agent.events.append({'action': 'run', 'arguments': action['arguments'],
                'trigger_origin': 'operator_reopens_existing_premise_author', 'feedback': deepcopy(proposal)})
        write_json(output / 'proposal.json', proposal)
        agent._save_state()
        task = agent.run(lab, on_checkpoint=checkpoint)
        execution = {'operator': 'task_continuation', 'status': task['status'], 'proposal': proposal}
    agent._save_state()
    result = account(agent)
    result.update(study_status='finished', episode_status=result['status'], execution=execution,
        costs_before=context['costs_before'],
        new_costs={key: value - context['costs_before'][key] for key, value in result['costs'].items()},
        observations=deepcopy(agent.observations), replay=context['replay'])
    write_json(output / 'result.json', result)
    write_json(output / 'active.json', {'status': 'finished', 'score': result['score'],
        'new_costs': result['new_costs'], 'episode_status': result['status']})
    print(json.dumps({'branch': branch, 'status': result['status'], 'score': result['score'],
                      'new_costs': result['new_costs']}), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
