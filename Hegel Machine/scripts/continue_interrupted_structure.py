"""Finish one recorded partial-object structure choice on its paid trajectories."""
import argparse
from pathlib import Path

from scripts.native_decision_scene import recover_interrupted_structure_scene
from scripts.run_learned_structure_successor import freeze_runtime
from scripts.run_partial_role_parent import load, save
from scripts.run_partial_role_successors import config, program_knowledge
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.evaluation import initial_prediction
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_alternatives import restore_paid_schema_alternatives
from hegel_machine.cognition.object_continuation import encode_entry, restore_object_state


def restore(source, seed, arm):
    registered = load(source / 'registration.json')
    path = source / 'runs' / str(seed) / arm / 'common/failed_paid_state.json'
    state = load(path)
    settings = next(t for t in registered['tasks'] if t['seed'] == seed)
    with AnonymousObjectLab(**settings) as lab:
        spec = lab.spec_for(tuple(state['selected_channels']))
    scene = recover_interrupted_structure_scene(state, public_spec=spec,
        state_ref={'file': str(path)}, retained_pool_size=registered['source_pool_size'])
    for returned in sorted(path.parent.glob('object_question_*/parent_result.json')):
        if restore_paid_schema_alternatives(scene['object_model'], load(returned), state['actions']):
            scene['references']['paid_enumeration_parent'] = {'file': str(returned)}
            break
    return registered, state, scene


def register(source, seed, arm, accounts, output):
    registered, _, scene = restore(source, seed, arm)
    prior = load(accounts)
    limit = scene['remaining_compute']
    if limit <= 0 or limit > prior['complete_relation_unit_remaining']:
        raise ValueError('Cannot fund the recorded choice from the retained development allowance')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', {
        'source': str(source), 'seed': seed, 'arm': arm, 'choice': scene['references']['choice'],
        'source_pool_ref': registered['source_pool_ref'], 'accounts': str(accounts),
        'development_allowance': 'complete_relation_unit_remaining', 'compute_limit': limit,
        'fit_rows': len(scene['training']), 'comparison_rows': len(scene['validation']),
        'threshold': registered['research_config']['minimum_improvement'],
        'max_candidates': registered['research_config']['max_candidates'],
        'stop_rule': 'One saved structural choice. Use the paid winner and original comparison rows. '
            'Stop after comparing the delivered alternatives and retaining all search returns.',
        'new_observation_limit': 0, 'model_call_limit': 0})
    freeze_runtime(output, source)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, Path(__file__).read_text())


def run(output):
    if Path(__file__).resolve().parents[1] != output / 'runtime':
        raise ValueError('Execute the registered source snapshot')
    declaration = load(output / 'registration.json')
    timing = ExecutionTiming(('restore', 'research', 'save'))
    with timing.phase('restore'):
        source = Path(declaration['source'])
        seed, arm = declaration['seed'], declaration['arm']
        registered, state, scene = restore(source, seed, arm)
        prior = load(declaration['accounts'])
        agent = CommonObjectResearchAgent(config(seed, registered['research_config']),
            library=program_knowledge(Path(registered['knowledge_root']), registered['knowledge_repair_ref']),
            workspace=output, organization=arm)
        restore_object_state(agent, state, state['selected_channels'])
        parent, rows = scene['selected'], scene['validation']
        # Only the selected executable was saved. The old comparison remains
        # addressable. Replaying its fitting prelude would duplicate paid work.
        frame = dict(spec=scene['spec'], training=scene['training'], validation=rows,
            comparison=rows, transitions=scene['transitions'], assessed=[parent],
            parent=None, retained_on_current=None, usable_parent=False,
            comparison_complete=True, native_full_pool_comparison=False, uncompared=[], terminal=False,
            fallback=initial_prediction(scene['training'], scene['spec'].risk_quantile),
            independent_validation=all(row not in scene['training'] for row in rows),
            directions=['structural_research'])
        budget = SearchBudget(declaration['compute_limit'])
        actions = []
    # Preserve an interrupted attempt even if no complete receipt is written.
    (output / 'attempt').mkdir()
    try:
        with timing.phase('research'), python_execution_deadline(registered['activity_seconds'], activity=True):
            result = agent.execute_research_decision(frame, 'structural_research', budget, actions)
        with timing.phase('save'):
            competition = agent._object_competitions.get(tuple(scene['spec'].feature_names), {})
            save(output / 'competition.json', competition)
            # Shared competition is kept once. Candidate entries refer to it.
            drafts = [encode_entry({k: v for k, v in item.items() if k != 'object_competition'})
                      for item in result['new_claims']]
            save(output / 'drafts.json', drafts)
            winner = result['selected']
            new_index = next((i for i, item in enumerate(result['new_claims'])
                              if item is winner), None)
            receipt = {'status': result['status'], 'choice': scene['references']['choice'],
                'input_parent': scene['references']['parent'], 'paid_pool': scene['references']['pool'],
                'paid_enumeration_parent': scene['references'].get('paid_enumeration_parent'),
                'scope': 'exposed development, saved winner versus admitted new programs',
                'old_bound_count': state['actions'][-1]['retained_candidates'],
                'old_scored_partial_claims': len(scene['existing_scores']),
                'unbound_old_pool': registered['source_pool_size'],
                'all_original_executables_restored': False,
                'native_comparison_complete': result['full_comparison'],
                'competition_ref': 'competition.json', 'drafts_ref': 'drafts.json',
                'before_loss': parent['validation_loss'], 'selected_loss': winner['validation_loss'],
                'selected_ref': scene['references']['parent'] if new_index is None else
                    {'file': str(output / 'drafts.json'), 'json_path': [new_index]},
                'actions': actions, 'quote': result['quote'], 'new_claims': len(drafts),
                'search_compute_cost': result['search_compute_cost'],
                'comparison_compute_cost': result['comparison_compute_cost'],
                'compute_cost': budget.spent, 'new_observations': 0, 'model_calls': 0,
                'terminal_measured': False}
        receipt['execution_timing'] = timing.result()
        save(output / 'receipt.json', receipt)
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_lower': budget.spent,
            'compute_upper': budget.limit, 'new_observations': 0, 'model_calls': 0})
        raise
    carried = {k: prior[k] for k in (
        'remaining_by_source', 'merge_remaining', 'diagnostic_suballocation_remaining',
        'original_account_unchanged', 'joint_continuation_remaining', 'unit_remaining_compute_conservative',
        'parent_separate_suballocation_remaining', 'method_return_unit_remaining', 'complete_relation_allocation_ref',
        'successor_allocation_remaining', 'successor_balance_scope', 'abc_allocation_limit',
        'abc_initial_allocation_limit', 'abc_observation_limit', 'abc_remaining_observations',
        'abc_remaining_compute_interval', 'abc_actual_compute_spent_to_date',
        'original_protocol_unassigned_conservative', 'full_lifecycle_unknowns_preserved')}
    save(output / 'closed_source_costs.json', {**carried,
        'status': 'closed_development_unit', 'new_observations': 0, 'new_measurement_cost': 0,
        'new_model_calls': 0, 'new_compute_interval': [budget.spent, budget.spent],
        'previous_source_costs_ref': {'file': declaration['accounts']},
        'receipt_ref': {'file': str(output / 'receipt.json')}, 'new_compute_cost': budget.spent,
        'cumulative_protocol_compute_interval': [v + budget.spent for v in prior['cumulative_protocol_compute_interval']],
        'cumulative_protocol_observations': prior['cumulative_protocol_observations'],
        'complete_relation_unit_remaining': prior['complete_relation_unit_remaining'] - budget.spent,
        'abc_remaining_compute': prior['abc_remaining_compute'],
        'other_accounts_inherited_from': declaration['accounts']})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--accounts', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm', choices=('archive', 'fixed_pr2', 'pr3'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'register':
        register(args.source.resolve(), args.seed, args.arm, args.accounts.resolve(), args.output.resolve())
    else:
        result = run(args.output.resolve())
        print(result['status'], result['new_claims'], result['compute_cost'])
