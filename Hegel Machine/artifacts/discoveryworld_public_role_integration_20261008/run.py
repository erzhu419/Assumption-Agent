"""Apply the public-role adapter to a frozen learned recipe; practice without a model."""
import ast
import json
from pathlib import Path
from hegel_machine.cognition.world_method_learning import WorldMethodLearner
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_pilot import SELECTION, write_json

output = Path(__file__).resolve().parent
old_path = Path('artifacts/discoveryworld_spectral_frozen_consolidation_20261008/method_bundle.json')
bundle = json.loads(old_path.read_text())
learned = bundle['files'][METHOD_PATH]
adapter = Path('src/hegel_machine/cognition/world_reference_recalibration.py').read_text()
# Preserve the exact two learned assay declarations; all other changes are the host adapter.
for name in ('FAMILIES', 'patterns'):
    def assignment(source):
        return next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
    adapter = adapter.replace(ast.get_source_segment(adapter, assignment(adapter)),
                              ast.get_source_segment(learned, assignment(learned)))
bundle['files'][METHOD_PATH] = adapter
bundle['methods'][0].update(provenance='host_public_role_adapter_of_model_learned_recipe',
    applicability='Current public initial reference roles; existing learned measurement families require current confirmation')
bundle['validation'] = 'Host integration under development; see this directory practice results, not a new autonomous discovery'
bundle['development_task_ids'] = list(dict.fromkeys([*bundle['development_task_ids'], *[f'dw_04_challenge_s{i}' for i in range(1, 5)]]))
write_json(output / 'candidate_bundle.json', bundle)
source = Path(bundle['source'])
plan = json.loads((source.parents[1] / 'protocol.json').read_text())
source_result = json.loads((source / 'result.json').read_text())
task = next(t for t in plan['tasks'] if t['id'] == source_result['task_id'])
records = json.loads((source / 'agent/laboratory_observations.json').read_text())
archive = json.loads(bundle['files']['shared/source_episode.json'])
registered = {t['id']: t for t in json.loads(SELECTION.read_text())['tasks']}
ids = ['dw_04_normal_s2', 'dw_04_normal_s4', *[f'dw_04_challenge_s{i}' for i in range(1, 5)]]
agent = WorldMethodLearner(None, output / 'practice', knowledge={'files': bundle['files']},
    task=task, records=records, prefix_steps=source_result['prefix_environment_steps'],
    public_spec=archive['public_spec'], limits=plan['limits'], calls=48,
    runtime=plan['runtime'], provider_factory=lambda _: None, max_steps=256,
    additional_tasks=[registered[i] for i in ids])
write_json(output / 'protocol.json', {'source_bundle': str(old_path),
    'actor': 'host public-role adapter plus unchanged model-learned family declarations',
    'model': None, 'practice_cases': agent.cases, 'procedure_steps': 256,
    'scope': 'Source retention and already-opened Challenge diagnostics; not fresh held-out confirmation',
    'author_revision_result': 'artifacts/discoveryworld_challenge_role_revision_20261008/result.json'})
rows = []
for case in agent.cases:
    trial = agent._practice(case)
    raw = json.loads((agent.workspace / trial['directory'] / 'result.json').read_text())
    role_path = agent.workspace / trial['directory'] / 'shared/reference_recalibration.json'
    roles = json.loads(role_path.read_text()) if role_path.exists() else {}
    rows.append({'case': case, 'trial': trial['directory'], 'status': trial['status'],
        'completedSuccessfully': raw['score']['completedSuccessfully'],
        'scoreNormalized': raw['score']['scoreNormalized'],
        'new_environment_steps': raw['new_environment_steps'],
        'replayed_environment_steps': raw['replayed_environment_steps'],
        'model_calls': raw['costs']['model_calls'],
        'reference_numbers': roles.get('reference_numbers'), 'target_numbers': roles.get('target_numbers'),
        'fit_refutations': roles.get('fit_refutations', []),
        'stop': raw['execution']['steps'][-1]['feedback'].get('reason') if raw['execution']['steps'] else None})
    write_json(output / 'brief.json', rows)
    write_json(output / 'result.json', agent.result('host_adapter_practice'))
    print(json.dumps(rows[-1]), flush=True)
