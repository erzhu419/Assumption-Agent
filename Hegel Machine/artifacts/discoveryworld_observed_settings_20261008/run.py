"""Continue the paid Challenge prefix using observed operational settings."""
import ast
import argparse
from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_discoveryworld_pilot import SELECTION, write_json
from scripts.run_online_inquiry_llm import verify_codex_subscription


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
TRIAL = ROOT / 'artifacts/discoveryworld_estimator_research_20261008/author/trials/trial_001'


def revised_source(previous):
    source = (ROOT / 'src/hegel_machine/cognition/world_reference_recalibration.py').read_text()
    old_tree, new_tree = ast.parse(previous), ast.parse(source)
    changes = []
    for name in ('FAMILIES', 'ESTIMATORS', 'patterns'):
        def assignment(tree):
            return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        old, new = assignment(old_tree), assignment(new_tree)
        text = ast.get_source_segment(previous, old.value)
        replacement = ' ' * new.col_offset + name + ' = ' + text + '\n'
        changes.append((new.lineno - 1, new.end_lineno, replacement))
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(changes, reverse=True):
        lines[start:end] = [replacement]
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous_path = args.resume.resolve() if args.resume else TRIAL / 'result.json'
    previous_workspace = previous_path.parent / 'agent' if args.resume else TRIAL
    previous = json.loads(previous_path.read_text())
    selection = json.loads(SELECTION.read_text())
    task = next(t for t in selection['tasks'] if t['id'] == 'dw_04_challenge_s1')
    original_plan = json.loads((ROOT / 'artifacts/discoveryworld_computed_phase_repair_20261008/protocol.json').read_text())
    model = original_plan['model']
    prior_protocol = json.loads((previous_path.parent / 'protocol.json').read_text()) if args.resume else None
    limits = prior_protocol['limits'] if args.resume else {
        **original_plan['limits'], 'max_model_calls': previous['costs']['model_calls'] + 32}
    workspace = output / 'agent'
    workspace.mkdir(exist_ok=False)
    inherited = (previous_workspace / METHOD_PATH).read_text() if args.resume else previous['executed_source']
    source = revised_source(inherited)
    (output / 'candidate.py').write_text(source)
    write_json(output / 'protocol.json', {
        'task': task, 'source_result': str(previous_path), 'model': model,
        'runtime': original_plan['runtime'], 'limits': limits,
        'additional_model_calls': limits['max_model_calls'] - previous['costs']['model_calls'],
        'prior_costs': previous['costs'],
        'auth_mode': verify_codex_subscription(),
        'scope': 'Opened development continuation. Inherit all public records, research and model-declared '
                 'estimators; update the host evidence adapter only. Successful settings are operational '
                 'labels, not exact latent resonance or an inferred response interval. No new scientific '
                 'variable or formula is supplied. Old paid costs remain cumulative; replay is separate.'})
    provider = CodexCodeProvider(workspace, model=model['requested_alias'],
        reasoning_effort=model['reasoning_effort'],
        timeout_seconds=limits['provider_request_timeout_seconds'])
    agent = DiscoveryWorldAgent(provider, workspace, arm='hegel', limits=limits)
    with DiscoveryWorldLab(task, original_plan['runtime'], workspace) as lab:
        agent.lab = lab
        replay = restore_repair(agent, lab, {**previous, 'episode_status': 'running'}, previous_workspace)
        write_json(output / 'replay.json', replay)
        agent._execute('write', {'path': 'shared/parent_before_evidence_integration.py',
                                'code': inherited}, lab)
        agent._execute('write', {'path': METHOD_PATH, 'code': source}, lab)
        before = deepcopy(agent.result()['costs'])
        prefix = lab.environment_steps

        def checkpoint(data):
            write_json(output / 'active.json', {**data, 'study_status': 'running',
                'prior_costs': before, 'replayed_environment_steps': prefix,
                'new_environment_steps': lab.environment_steps - prefix})
            print(json.dumps({'calls': agent.costs['model_calls'] - before['model_calls'],
                'new_actions': lab.environment_steps - prefix,
                'score': data['score']['scoreNormalized'],
                'completed': data['score']['completedSuccessfully']}), flush=True)

        agent._on_checkpoint = checkpoint
        checkpoint(agent.result())
        if args.resume:
            # Resume the existing failed provider request with its original total
            # allowance; do not reopen an already exhausted scientific question.
            question = next(q for q, session in agent.llm_inquiries.items()
                            if session.get('status') == 'provider_failed')
            execution = agent._execute('run', {'operator': 'inquire', 'question': {'id': question},
                'max_model_calls': limits['max_model_calls'] - agent.costs['model_calls']}, lab)
            if agent.status == 'running':
                execution = {'resumed_inquiry': execution, 'continuation': agent._execute('run', {
                    'operator': 'procedure', 'path': METHOD_PATH, 'max_steps': 512,
                    'question': {'id': 'task'}}, lab)}
        else:
            execution = agent._execute('run', {'operator': 'procedure', 'path': METHOD_PATH,
                'max_steps': 512, 'question': {'id': 'task'}}, lab)
        result = agent.result()
        result.update(study_status='finished', task_id=task['id'], execution=execution,
            prior_costs=before, replayed_environment_steps=prefix,
            new_environment_steps=lab.environment_steps - prefix,
            new_costs={key: value - before[key] for key, value in result['costs'].items()})
        write_json(output / 'result.json', result)
        state = json.loads((workspace / 'shared/reference_recalibration.json').read_text())
        brief = {key: result[key] for key in ('status', 'study_status', 'task_id', 'score',
            'new_environment_steps', 'new_costs', 'replayed_environment_steps')}
        brief.update(tests=state.get('tests', []), fit_refutations=state.get('fit_refutations', []),
            response_constraints=state.get('response_constraints', []),
            setting_recoveries=state.get('setting_recoveries', []),
            research_table=state.get('research_table', {}))
        write_json(output / 'brief.json', brief)
        print(json.dumps({k: brief[k] for k in ('status', 'score', 'new_costs')}), flush=True)


if __name__ == '__main__':
    main()
