#!/usr/bin/env python3
"""One program-derived experiment on a retained anonymous source apparatus."""

import argparse
from dataclasses import asdict
from math import isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.checkpoint_storage import clone_or_copy, write_json_checkpoint
from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.native_condition_research import _spec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition import program_condition_sets, temporal_control_probes


def read(reference):
    return read_method_reference(reference, ROOT)


def run(registration, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    timing = ExecutionTiming(('load', 'planning', 'measurement', 'assessment'))
    planning = SearchBudget(registration['planning_compute_cap'])
    assessment = SearchBudget(registration['assessment_compute_cap'])
    lab, result = None, {'status': 'preparing', 'fits': 0, 'model_calls': 0,
        'parent_replaced': False, 'independent_confirmation': False}
    try:
        with python_execution_deadline(registration['activity_seconds'], activity=True):
            with timing.phase('load'):
                # Case identity is for the apparatus only. The planner receives
                # public controls and the two previously learned programs.
                case = read(registration['case_ref'])
                lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
                public = lab.public_spec()
                old_records = read(registration['paid_records_ref'])
                programs = [Program.from_dict(read(item['program_ref']))
                            for item in registration['hypotheses']]
                if len(programs) != 2:
                    raise ValueError('this study compares the two registered frozen programs')
                runtime = {Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
                    if name.startswith('hegel_machine') and getattr(module, '__file__', None)}
                runtime.add(Path(__file__).resolve())
                for path in runtime:
                    destination = output / 'source_at_run' / path.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    clone_or_copy(path, destination)
                write_json_checkpoint(output / 'registration.json', {**registration,
                    'public_spec': public, 'source_files': sorted(str(path.relative_to(ROOT)) for path in runtime)})
            with timing.phase('planning'):
                proposals = propose_program_disagreement_probes(_spec(public), programs,
                    planning, serial=registration['serial'], length=public['max_probe_length'],
                    control_levels=public['control_levels'])
                known = [record['action']['sequence'] for record in old_records]
                eligible = [item for item in proposals if
                    [dict(point) for point in item['probe'].sequence] not in known
                    and item['predicted_separation'] > registration['mse_threshold']
                    and all(len(values) == item['probe'].cost
                            and all(value is not None and isfinite(value) for value in values)
                            for values in item['predictions'])]
                eligible.sort(key=lambda item: -item['predicted_separation'])
                write_json_checkpoint(output / 'planning.json', {
                    'proposals': [{**item, 'probe': asdict(item['probe'])} for item in proposals],
                    'eligible_probe_ids': [item['probe'].probe_id for item in eligible],
                    'compute_cost': planning.spent})
                result['proposed_count'], result['eligible_count'] = len(proposals), len(eligible)
                if not eligible:
                    result.update(status='no_new_distinguishing_probe',
                        next_research='retain_existing_models_and_review_design_coverage')
                    return result
                chosen = eligible[0]
                action = {'kind': 'trajectory', 'channels': [],
                          'sequence': [dict(point) for point in chosen['probe'].sequence]}
                rows = lab.action_cost(action)
                if rows > registration['observation_cap'] or not assessment.can_spend(2 * rows):
                    raise ValueError('complete frozen assessment and measurement must fit the registered limits')
                plan = {'hypotheses': [{**item, 'program': program.to_dict(), 'predictions': predictions}
                    for item, program, predictions in zip(registration['hypotheses'], programs, chosen['predictions'])],
                    'predicted_fields': {}, 'forecast_status': 'predicted_distinction',
                    'predicted_separation': chosen['predicted_separation'],
                    'proposal_source': 'program_disagreement', 'selected_action': action,
                    'sequence_mode': chosen['sequence_mode'], 'boundary': chosen['boundary']}
                # Persist the complete prospective predictions before buying a
                # response. Existing history and large snapshots remain refs.
                write_json_checkpoint(output / 'frozen_plan.json', {**plan, 'hypotheses': [
                    {key: value for key, value in hypothesis.items() if key != 'program'}
                    for hypothesis in plan['hypotheses']]})
            with timing.phase('measurement'):
                record = lab.execute(action)
                write_json_checkpoint(output / 'observation.json', record)
                measured = lab.observation_rows(record)
            with timing.phase('assessment'):
                verdict = assess_competing_object_probe(plan, [(row, row) for row in measured],
                    assessment, tolerance=registration['mse_threshold'])
                write_json_checkpoint(output / 'assessment.json', verdict)
            result.update(status='completed', assessment_status=verdict['status'],
                predicted_separation=plan['predicted_separation'], sequence_mode=plan['sequence_mode'],
                hypotheses=verdict['hypotheses'],
                next_research=('revise_competing_structures_from_shared_counterexamples'
                    if verdict['status'] == 'all_refuted' else
                    'investigate_applicability_of_locally_supported_hypothesis'
                    if verdict['status'] == 'distinguished' else 'retain_unresolved_competition'),
                new_record_append_contract={'source_record_index': 0,
                    'destination_record_index': len(old_records),
                    'destination_row_offset': sum(len(record['rows']) for record in old_records)},
                scope='One adaptively selected research probe; neither general applicability nor terminal accuracy.')
            return result
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['costs'] = {'planning_compute': planning.spent, 'assessment_compute': assessment.spent,
            'compute_cost': planning.spent + assessment.spent,
            'observation_cost': lab.costs['observation_cost'] if lab is not None else 0,
            'measurement_cost': lab.costs['measurement_cost'] if lab is not None else 0,
            'model_calls': 0, 'reported_tokens': 0, 'fits': 0}
        result['prior_costs_ref'] = registration['prior_costs_ref']
        result['timing'] = timing.result()
        write_json_checkpoint(output / 'summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(read({'file': str(args.registration), 'json_path': []}), args.output)
    print({key: result.get(key) for key in ('status', 'assessment_status', 'costs', 'next_research')})
