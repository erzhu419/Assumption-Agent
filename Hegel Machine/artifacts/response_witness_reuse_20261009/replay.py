"""Compare witness consumption from the exact pre-proposal public state."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from hegel_machine.cognition import world_response_consequences as consequences
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from scripts import consolidate_response_measurement as study
from scripts.run_discoveryworld_prerequisite_check import replay_prefix
from scripts.run_response_family_recovery import declarations
from scripts.run_online_inquiry_llm import write_json


if __name__ == '__main__':
    output = Path(__file__).resolve().parent
    source = ROOT / 'artifacts/response_controller_adoption_20261009'
    scope = ROOT / 'artifacts/response_revised_scope_20261009'
    protocol = json.loads((source / 'protocol.json').read_text())
    previous = json.loads((source / 'task_01/live/result.json').read_text())
    proposal = previous['events'][5]
    prefix = min(proposal['feedback']['new_observation_indices'])
    records = previous['observations'][:prefix]
    state = json.loads((scope / 'task_01/revised/prefix_state.json').read_text())
    bundle = json.loads((source / 'initial_bundle.json').read_text())
    code = Path(recipe.__file__).read_text()
    for hypothesis in declarations(bundle['files'][study.METHOD_PATH]):
        code = _retain_declaration(code, hypothesis)
    spec = importlib.util.spec_from_file_location('hegel_machine.cognition._scope_only_consequences',
                                                 scope / 'scope_only_consequences.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    repaired = consequences.assess_response_consequences
    study.CASES = {'task_01': protocol['cases']['task_01']['config']}
    write_json(output / 'protocol.json', {'source': str(source), 'prefix': prefix,
        'prefix_state': str(scope / 'task_01/revised/prefix_state.json'),
        'new_external_model_calls': 0, 'hypothesis': proposal['arguments']['hypothesis'],
        'limits': {**protocol['limits'], 'max_model_calls': 0},
        'scope': 'Both arms replay the same paid physical records and saved controller state; '
                 'numerical analysis caches are rebuilt from those paid rows. Only consumption '
                 'of executable family witnesses differs. All reported new costs start at the '
                 'proposal; no future scientific observations or new model decisions are supplied.'})
    rows = []
    try:
        for arm, assessment in [('scope_only', old.assess_response_consequences), ('witness_reuse', repaired)]:
            consequences.assess_response_consequences = assessment
            agent = DiscoveryWorldAgent(None, output / arm / 'agent', arm='hegel',
                limits={**protocol['limits'], 'max_model_calls': 0})
            lab = study.TransferSignalLab('task_01')
            agent.lab = lab
            replay = replay_prefix(lab, {'records': records, 'environment_steps': prefix})
            agent.observations = deepcopy(records)
            agent.costs['environment_steps'] = prefix
            agent._execute('write', {'path': study.METHOD_PATH, 'code': code}, lab)
            write_json(agent.workspace / 'shared/reference_recalibration.json', deepcopy(state))
            feedback = agent._execute('run', deepcopy(proposal['arguments']), lab)
            agent._save_state()
            final = json.loads((agent.workspace / 'shared/reference_recalibration.json').read_text())
            row = {'arm': arm, 'status': feedback['status'], 'success': lab.score()['completedSuccessfully'],
                'new_actions': lab.environment_steps - prefix, 'new_costs': {
                    **agent.result()['costs'], 'environment_steps': lab.environment_steps - prefix},
                'feedback': feedback, 'replay': replay,
                'new_history': final['response_research_history'][len(state['response_research_history']):],
                'new_observations': agent.observations[prefix:]}
            rows.append(row)
            write_json(output / 'results.json', rows)
            print(json.dumps({k: row[k] for k in ['arm','status','success','new_actions']}), flush=True)
    finally:
        consequences.assess_response_consequences = repaired
