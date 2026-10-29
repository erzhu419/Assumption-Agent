"""Consume an archived object experiment through Common without repurchasing it.

This is paid-evidence execution acceptance, not independent transfer evaluation.
Only a compact public projection and new consumption receipts are written.
"""

from copy import deepcopy
import argparse
import json
from pathlib import Path
from time import monotonic

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.common_object_lab import CommonObjectLab, measured_selector_values
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint, write_text_incrementally
from hegel_machine.cognition.effective_history import HistoryStore
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_induction_continuation import paid_object_records
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


DEFAULT_UNIT = ('C/dependency_method_revision_20260920/development/runs/'
                'rewritten_methods/95902002/checkpoint.json')


def read_archived_scene(store_path, unit, task_position):
    store = HistoryStore(store_path, readonly=True)
    try:
        row = store.connection.execute('SELECT sequence FROM entries WHERE unit=?', (unit,)).fetchone()
        if row is None:
            raise ValueError('Historical object checkpoint is absent')
        checkpoint = json.loads(b''.join(store.read_sequence(row[0])))
    finally:
        store.close()
    result = next(row for row in checkpoint['results'] if row['task_position'] == task_position)
    task = checkpoint['tasks'][task_position]
    settings = {key: value for key, value in task.items()
                if key not in {'environment', 'research_limits'}}
    research = result['research']
    questions = research['object_research']['dependency_questions']
    membership = {entry['probe_id'] for entry in questions
                  if entry['question'].get('purpose') == 'distinguish_membership_dependency'}
    batches = research['batches']
    cutoff = next(i for i, batch in enumerate(batches)
                  if batch['probe']['probe_id'] in membership)
    with AnonymousObjectLab(**settings) as lab:
        public = CommonObjectLab(lab).public_spec()
        records = paid_object_records(lab, batches[:cutoff], ('fine',))
        complete = paid_object_records(lab, batches, ('fine',))
    return {'public': public, 'records': records, 'paid_records': complete,
        'source_ref': {'store': str(store_path), 'unit': unit,
                       'json_path': ['results', task_position, 'research', 'batches']},
        'source_costs': deepcopy(checkpoint['costs']),
        'source_cost_ref': {'store': str(store_path), 'unit': unit, 'json_path': ['costs']},
        'source_scope': 'historical_dependency_method_development',
        'source_success_threshold': 1e-6}


def replay(scene, destination, *, compute_limit=400000, max_steps=8, sources=()):
    from types import SimpleNamespace

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    root = Path(__file__).resolve().parents[1]
    for relative in ('scripts/replay_partial_role_common.py',
                     'src/hegel_machine/cognition/object_dependency_methods.py',
                     'src/hegel_machine/cognition/common_object_lab.py',
                     'src/hegel_machine/cognition/common_archive_inquiry.py'):
        snapshot = destination / 'source_at_run' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        write_text_incrementally(snapshot, (root / relative).read_text())
    spec = SimpleNamespace(**scene['public'])
    records = deepcopy(scene['records'])
    budget = SearchBudget(compute_limit)
    model = infer_object_model(spec, records, budget)
    agent = CommonArchiveInquiryAgent(None, destination, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=192, compute_budget=compute_limit,
            confirmation_mse_threshold=1e-6, execution_seconds=60.))
    agent.costs['compute_cost'] = budget.spent
    agent._method_public_spec = scene['public']
    agent._save('source_projection.json', json.dumps(scene, separators=(',', ':')))
    agent._save('spec.json', json.dumps(scene['public']))
    agent._save('sources.json', json.dumps(sources))
    lookup = {ObjectDependencyMethods._key([*row['history'], row['controls']]): row
              for row in scene['paid_records']}
    steps, previous_ref = [], None
    for serial in range(max_steps):
        agent._save('roles.json', json.dumps(model, separators=(',', ':')))
        agent._save('paid.json', json.dumps(records, separators=(',', ':')))
        agent._save('selected_values.json', json.dumps(measured_selector_values(model, records)))
        arguments = {'operator': 'partial_role_experiment',
            'allocation': {'compute': compute_limit - agent.costs['compute_cost']},
            **{name + '_ref': {'file': file} for name, file in
               [('spec', 'spec.json'), ('model', 'roles.json'),
                ('records', 'paid.json'), ('selected_values', 'selected_values.json'),
                ('sources', 'sources.json')]}}
        if previous_ref:
            arguments['previous_ref'] = previous_ref
        proposal = agent._execute('run', arguments, None)
        receipt = f'consumption_{serial:02d}.json'
        agent._save(receipt, json.dumps(proposal, separators=(',', ':')))
        previous_ref = {'file': receipt}
        step = {'result_ref': previous_ref, 'status': proposal['status'],
            'recipe': proposal.get('relation'), 'completed_relation': proposal.get('completed_relation'),
            'compute_cost': proposal['compute_cost'], 'consumed_paid_rows': 0}
        steps.append(step)
        if proposal.get('completed_relation', {}).get('status') == 'discriminating':
            method = deepcopy(proposal['method_object'])
            method.update(model_id='paid_partial_role_method', source_ref={
                'file': str(destination / receipt), 'json_path': ['method_object']})
            agent._save('formed_method.json', json.dumps(method, separators=(',', ':')))
            break
        if proposal['status'] != 'proposed_complete_partial_role_experiment':
            break
        required = {ObjectDependencyMethods._key(action['sequence']): action['sequence']
                    for action in proposal['actions']}
        if any(key not in lookup for key in required):
            step['status'] = 'historical_paid_pair_unavailable'
            break
        known = {ObjectDependencyMethods._key([*row['history'], row['controls']]) for row in records}
        for sequence in required.values():
            for end in range(1, len(sequence) + 1):
                key = ObjectDependencyMethods._key(sequence[:end])
                if key not in known:
                    records.append(deepcopy(lookup[key]))
                    known.add(key)
                    step['consumed_paid_rows'] += 1
    summary = {'status': 'paid_complete_pair_consumed' if (destination / 'formed_method.json').exists()
               else 'paid_replay_incomplete', 'steps': steps,
        'new_costs': dict(agent.costs), 'source_costs': scene['source_costs'],
        'source_cost_ref': scene['source_cost_ref'], 'source_ref': scene['source_ref'],
        'elapsed_seconds': monotonic() - start, 'independent_transfer_measured': False,
        'object_assignment_transferred': False, 'source_success_threshold': 1e-6,
        'root_response_confirmation_performed': False,
        'method_sources': [{key: source[key] for key in ('model_id', 'source_ref') if key in source}
                           for source in sources]}
    write_json_checkpoint(destination / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--store', default='/mnt/d/Hegel-History-Archive-20261012/effective_history.sqlite3')
    parser.add_argument('--unit', default=DEFAULT_UNIT)
    parser.add_argument('--task-position', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-method', type=Path)
    args = parser.parse_args()
    scene = read_archived_scene(args.store, args.unit, args.task_position)
    sources = [json.loads(args.source_method.read_text())] if args.source_method else []
    result = replay(scene, args.output, sources=sources)
    print(json.dumps({key: result[key] for key in ('status', 'new_costs', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
