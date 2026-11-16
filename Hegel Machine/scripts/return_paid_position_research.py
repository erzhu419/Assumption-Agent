"""Return measured position knowledge and paid response comparisons to Common."""
import argparse
from dataclasses import replace
from pathlib import Path

from scripts.continue_interrupted_structure import restore
from scripts.run_learned_structure_successor import freeze_runtime
from scripts.run_partial_role_parent import load, save
from scripts.run_partial_role_successors import config
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.evaluation import initial_prediction
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_alternatives import enumerate_object_schemas
from hegel_machine.cognition.object_continuation import decode_entry
from hegel_machine.cognition.object_induction_continuation import paid_object_records
from hegel_machine.cognition.object_interventions import plan_paid_position_contrast


def register(unit, accounts, output):
    cfg, prior = load(unit / 'registration.json'), load(accounts)
    old = Path(cfg['previous'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', {
        'paid_unit': str(unit), 'source': cfg['original_arm_source'],
        'seed': cfg['seed'], 'arm': cfg['arm'], 'costs_ref': {'file': str(accounts)},
        'compute_limit': prior['complete_relation_unit_remaining'],
        'source_pool_ref': cfg['source_pool_ref'],
        'stages': [{'plan_ref': {'file': str(p / 'plan.json')},
                    'assessment_ref': {'file': str(p / 'role_return.json')}} for p in (old, unit)],
        'new_observations': 0, 'new_fits': 0, 'model_calls': 0,
        'scope': 'Paid position returns constrain structural explanations; compare saved fits '
                 'on their original exposed working rows, retain the old predictor, then return '
                 'the remaining research question. Preserve the original full pool by reference.',
        'stop_rule': 'One paid return and one next-experiment plan; no acquisition or independent validation.'})
    freeze_runtime(output, unit)
    for name in ('continue_interrupted_structure.py', Path(__file__).name):
        path = Path(__file__).with_name(name)
        base = unit / 'runtime/scripts' / name
        write_text_incrementally(output / 'runtime/scripts' / name, path.read_text(),
                                 base=base if base.exists() else None)


def consume_saved_owner_fits(agent, frame, model, payloads, assignments, saved, budget, actions):
    """Apply measured membership to saved structural claims, preserving every source."""
    if saved['parent_working_loss'] != frame['assessed'][0]['validation_loss']:
        raise ValueError('Restore the original parent and working comparison')
    admitted, scores, disposition = [], {}, []
    for payload in payloads:
        entry = decode_entry(payload)
        schema_index = entry['working_role_revision']['schema_index']
        paid = next(row for row in saved['owner_response_fits']
                    if row['candidate_id'] == entry['candidate'].candidate_id)
        if paid['schema_index'] != schema_index or paid['validation_loss'] != entry['validation_loss']:
            raise ValueError('The saved fit and its comparison must identify the same candidate')
        fields = assignments['schemas'][schema_index]['fields']
        if not budget.spend(len(fields)):
            raise ValueError('Insufficient allowance to consume the measured membership')
        conflicts = [{'position': group['x'], 'owner': group['q'],
                      'allowed': model['position_candidates'][group['x']]['quantities']}
                     for group in fields.values()
                     if group['q'] not in model['position_candidates'][group['x']]['quantities']]
        disposition.append({'candidate_id': entry['candidate'].candidate_id,
            'schema_index': schema_index, 'status': 'excluded_by_measured_position' if conflicts else
                'compatible_with_measured_position', 'conflicts': conflicts,
            'paid_validation_loss': paid['validation_loss']})
        if not conflicts:
            admitted.append(entry)
            scores[entry['candidate'].program] = paid['validation_loss']
    result = agent.compare_research_decision_return(frame, 'structural_research', admitted,
        budget, actions, paid_scores=scores, reused_paid_return=True)
    # Only the saved winning parent and the delivered paid fits are present.
    # The original full library remains addressable, not newly rebound/scored.
    result['native_full_pool_comparison'] = False
    result['completion_scope'] = 'paid_winner_and_delivered_response_comparison'
    return result, disposition


def run(output):
    if Path(__file__).resolve().parents[1] != output / 'runtime':
        raise ValueError('Execute the frozen return entry')
    cfg = load(output / 'registration.json')
    prior = load(cfg['costs_ref']['file'])
    budget = SearchBudget(cfg['compute_limit'])
    clock = ExecutionTiming(('restore', 'paid_return', 'next_question'))
    (output / 'attempt').mkdir()
    receipt = {'status': 'incomplete_paid_return', 'new_observations': 0,
               'new_fits': 0, 'new_predictions': 0, 'model_calls': 0,
               'independent_confirmation': False}
    try:
        with clock.phase('restore'):
            registration, state, scene = restore(Path(cfg['source']), cfg['seed'], cfg['arm'])
            unit = Path(cfg['paid_unit'])
            saved = load(unit / 'response_comparison.json')
            payloads, assignments = load(unit / 'responses.json'), load(unit / 'assignments.json')
            source_ref = {'file': str(Path(cfg['source']) / 'registration.json'),
                          'json_path': ['tasks', next(i for i, task in enumerate(registration['tasks'])
                                                    if task['seed'] == cfg['seed'])]}
            agent = CommonObjectResearchAgent(config(cfg['seed'], registration['research_config']),
                workspace=output / 'common', organization=cfg['arm'], source_task_ref=source_ref)
            actions = []
            consumer = agent._dependency_methods([], budget=budget, actions=actions)
            model = scene['object_model']
            original_schemas = model['schema_alternatives']['schemas']
        with clock.phase('paid_return'):
            for stage in cfg['stages']:
                model = consumer.consume_paid_position_return(model, **stage)
            parent = scene['selected']
            frame = dict(spec=replace(scene['spec'], context={**scene['spec'].context, 'object_model': model}),
                training=scene['training'], comparison=scene['validation'], assessed=[parent],
                comparison_complete=True, native_full_pool_comparison=False, uncompared=[],
                fallback=initial_prediction(scene['training'], scene['spec'].risk_quantile),
                independent_validation=True)
            compared, dispositions = consume_saved_owner_fits(agent, frame, model, payloads,
                assignments, saved, budget, actions)
            winner = compared['selected']
            save(output / 'paid_comparison.json', {
                'source_comparison_ref': {'file': str(unit / 'response_comparison.json')},
                'source_candidates_ref': {'file': str(unit / 'responses.json')},
                'source_pool_ref': cfg['source_pool_ref'], 'source_pool_size': registration['source_pool_size'],
                'candidate_dispositions': dispositions, 'selected_candidate_id': winner['candidate'].candidate_id,
                'selected_loss': winner['validation_loss'], 'original_parent_ref': scene['references']['parent'],
                'old_predictor_retained': winner['candidate'].program == parent['candidate'].program,
                'completion_scope': compared['completion_scope'], 'native_full_pool_comparison': False,
                'new_comparison_compute': compared['comparison_compute_cost'],
                'saved_prediction_count': sum(len(values) for values in saved['comparison_predictions'])})
        with clock.phase('next_question'):
            enumeration = enumerate_object_schemas(model, budget,
                max_schemas=registration['research_config']['max_candidates'], preferred_schemas=original_schemas)
            save(output / 'constrained_assignments.json', enumeration)
            model = {**model, 'schema_alternatives': enumeration}
            settings = registration['tasks'][source_ref['json_path'][-1]]
            batches = [*state['batches']]
            for stage in cfg['stages']:
                directory = Path(stage['plan_ref']['file']).parent
                for i in load(directory / 'purchase.json')['missing_corners']:
                    batches.append(load(directory / f'corner_{i}.json'))
            with AnonymousObjectLab(**settings) as lab:
                records = paid_object_records(lab, batches, state['selected_channels'])
            next_plan = plan_paid_position_contrast(frame['spec'], model, records, budget,
                schemas=enumeration['schemas'], remaining_rows=0)
            save(output / 'next_position_question.json', next_plan)
            agent._object_models[tuple(state['selected_channels'])] = model
            consumer.record_parent_result(model)
            result = {'status': 'paid_position_parent_returned',
                'execution_complete': compared['execution_complete'],
                'program': winner['candidate'].program.to_dict(), 'object_research': model,
                'selected_channels': state['selected_channels'], 'fallback': frame['fallback'],
                'comparison_ref': {'file': str(output / 'paid_comparison.json')},
                'costs': {'compute_cost': budget.spent, 'observation_cost': 0,
                          'measurement_cost': 0, 'model_calls': 0}, 'actions': actions}
            feedback = consumer.record_parent_response(result)
            threshold = registration['research_config']['minimum_improvement']
            response_ready = max(winner['training_loss'], winner['validation_loss']) <= threshold
            receipt.update(status=result['status'], execution_complete=result['execution_complete'],
                position_domains={key: item['quantities'] for key, item in model['position_candidates'].items()},
                unresolved_roles=model['unresolved'], complete_object_schema=model['complete'],
                candidate_dispositions=dispositions, selected_candidate_id=winner['candidate'].candidate_id,
                selected_loss=winner['validation_loss'], old_predictor_retained=winner is parent,
                original_parent_ref=scene['references']['parent'], source_pool_ref=cfg['source_pool_ref'],
                source_pool_size=registration['source_pool_size'], new_comparison_compute=compared['comparison_compute_cost'],
                next_plan_status=next_plan['status'], next_plan_ref={'file': str(output / 'next_position_question.json')},
                method_return=feedback, independent_validation_admitted=False,
                original_threshold=threshold, response_meets_threshold=response_ready,
                validation_reason=('Response still exceeds the original threshold.' if not response_ready
                    else 'Response meets the working threshold; new observations require a separate decision.'))
            save(output / 'actions.json', actions)
    except BaseException as exc:
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt.update(compute_cost=budget.spent, execution_timing=clock.result())
        save(output / 'receipt.json', receipt)
        save(output / 'closed_source_costs.json', {**prior, 'status': receipt['status'],
            'previous_source_costs_ref': cfg['costs_ref'], 'receipt_ref': {'file': str(output / 'receipt.json')},
            'new_compute_cost': budget.spent, 'new_compute_interval': [budget.spent, budget.spent],
            'new_observations': 0, 'new_measurement_cost': 0, 'new_model_calls': 0,
            'complete_relation_unit_remaining': prior['complete_relation_unit_remaining'] - budget.spent,
            'cumulative_protocol_compute_interval': [v + budget.spent for v in prior['cumulative_protocol_compute_interval']]})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--accounts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'register':
        register(args.source.resolve(), args.accounts.resolve(), args.output.resolve())
    else:
        result = run(args.output.resolve())
        print(result['status'], result['compute_cost'], result['selected_loss'], result['next_plan_status'])
