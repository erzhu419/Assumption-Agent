"""Extract one known paid prefix; no new learning, acquisitions or terminal data."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts/independent_confirmation_20261003'
SOURCE = PRIOR / 'successor_regression/runs/new_cumulative/root12/checkpoint.json.gz'
sys.path.insert(0, str(PRIOR / 'runtime/src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from hegel_machine.cognition.unified_research import ResearchConfig
from run_observed_condition_decoding_study import laboratory

with gzip.open(ROOT / 'artifacts/instrument_whole_visit_20261003/paid_confirmation_root07.json.gz', 'rt') as stream:
    original = json.load(stream)
for case in original['cases']:
    case['root'] = 'root07'
    case['source'] = original['source']

# The only read of the large root12 state in this preparation.
with gzip.open(SOURCE, 'rt') as stream:
    state = json.load(stream)
position = 9
report = next(item['research'] for item in state['results'] if item['task_position']==position)
visit_index, visit = [(index, item) for index, item in enumerate(state['revisits'])
    if item['task_position']==position][-1]
actions = report['actions']
paired_index, paired = [(index, event) for index, event in enumerate(actions)
    if event['action']=='compare_method_followup_on_fresh_response'][-1]
plan = next(event for event in reversed(actions[:paired_index])
    if event['action']=='plan_condition_confirmation' and event['probe_id']==paired['probe_id'])
revision = next(event for event in reversed(actions[:paired_index])
    if event['action']=='revise_competing_observed_conditions')
batch_index = next(index for index, batch in enumerate(report['batches'])
    if batch['probe']['probe_id']==paired['probe_id'])
fresh = report['batches'][batch_index]
paid = report['batches'][:batch_index]
protocol = json.loads((PRIOR / 'successor_regression/protocol.json').read_text())
task = protocol['roots']['root12']['tasks'][position]
channels = fresh['channels']
with laboratory(task) as lab:
    spec = asdict(lab.spec_for(tuple(channels)))
case = {
    'root':'root12', 'task_position':position, 'source':str(SOURCE.relative_to(ROOT)),
    'task':task, 'config':asdict(ResearchConfig(**protocol['config'])), 'spec':spec,
    'visit_previous_report':visit['previous'], 'continuation_candidate':visit['candidate'],
    'resource_limits':visit['resource_limits'],
    **{key:paired[key] for key in ('baseline_program','baseline_fallback','program','fallback')},
    'channels':channels, 'paid_batches_before_confirmation':paid,
    'recorded_confirmation_plan':plan, 'recorded_fresh_batch':fresh,
    'paid_revision_event':revision, 'paired_event':paired,
    'diagnosis':{'source_visit_index':visit_index, 'paired_action_index':paired_index,
        'fresh_probe_id':paired['probe_id'], 'paid_batches':len(paid)},
    'boundary':'No terminal data or final knowledge. Saved fresh outcomes are evidence only and excluded from planning. Intermediate condition state and exact compute balance were not saved; reconstruct from paid prefix with explicit charging.',
}
output = {'source':'Two existing root07 paid fixtures plus one authorized read of frozen0.171 root12 final state.',
    'scope':'Three known pre-confirmation paid revisions; partial historical reconstruction, no full session restoration.',
    'cases':[deepcopy(item) for item in original['cases']]+[case]}
target = BASE / 'paid_confirmation_cases.json.gz'
target.write_bytes(gzip.compress(json.dumps(output,separators=(',',':')).encode(),compresslevel=1))
inventory = {'new_learning_runs':0,'terminal_scores':0,'large_checkpoint_reads':1,
    'fixture':str(target.relative_to(ROOT)), 'bytes':target.stat().st_size,
    'cases':[{'root':item['root'], 'position':item['task_position'],
        'paid_batches':len(item['paid_batches_before_confirmation']),
        'fresh_rows':len(item['recorded_fresh_batch']['observations']),
        'confirmation_compute_quote':item['continuation_candidate']['instrument_condition_experiment']['confirmation_compute_quote']}
        for item in output['cases']]}
(BASE / 'fixture_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(json.dumps(inventory))
