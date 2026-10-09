"""Connect finite response hypotheses to the existing physical controller."""

import ast
from copy import deepcopy
import json
from time import monotonic

from .world_reference_recalibration import refresh_response_evidence
from .world_response_hypotheses import propose_response_research


def execute_reference_response(agent, laboratory, arguments):
    path = arguments['path']
    source = (agent.workspace / path).read_text()
    assignment = next(node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'ESTIMATORS' for target in node.targets))
    estimators = list(ast.literal_eval(assignment.value))
    state_path = agent.workspace / 'shared/reference_recalibration.json'
    state = json.loads(state_path.read_text())
    refresh_response_evidence(state, agent.observations, laboratory.public_observation(), estimators)
    number = next((n for n in state['target_numbers'] if n not in state['completed']), None)
    started = monotonic()
    try:
        result = propose_response_research(state, estimators, path)
    finally:
        agent.costs['python_execution_seconds'] += monotonic() - started
    agent.costs['research_compute_units'] += result['compute_cost']
    round_number = len(state.get('response_research_history', [])) + 1
    question = agent._question({'id': f'reference_response_{number}_{round_number}',
        'question': 'Which coefficient and shared activation-response hypotheses remain '
                    'compatible with the measured successes and failures?',
        'parent_id': 'task', 'depends_on': []})
    result['question_id'] = question['id']
    agent._record_answer(question, {'status': 'inconclusive', 'evidence': result['evidence'],
        'answer': {'diagnosis': result['status'], 'assumptions': result['assumptions'],
                   'selected_setting': result['selected_setting'],
                   'selected_number': result.get('selected_number'),
                   **({'joint_parameter_search': result['joint_parameter_search']}
                      if 'joint_parameter_search' in result else {}),
                   'scope': 'Finite response hypotheses for a new physical test; no law confirmation'}})
    state['response_research'] = result
    state['response_context'] = deepcopy(arguments.get('context'))
    state.setdefault('response_research_history', []).append({
        'question_id': question['id'], 'status': result['status'],
        'response_count': result['response_count'], 'compute_cost': result['compute_cost'],
        'candidate_count': len(result['candidates']),
        'compatible_count': sum(c['status'] == 'compatible' for c in result['candidates']),
        'selected_experiment': (result.get('selection') or {}).get('selected_experiment'),
        **({'joint_parameter_search': result['joint_parameter_search']}
           if 'joint_parameter_search' in result else {}),
        'hypotheses': result['hypotheses'], 'evidence': result['evidence']})
    state_path.write_text(json.dumps(state, indent=2))
    return {'operator': 'reference_response', **result}
