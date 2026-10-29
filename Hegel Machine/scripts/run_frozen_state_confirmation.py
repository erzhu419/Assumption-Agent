#!/usr/bin/env python3
"""Confirm a frozen draft through Common and its registered additional panels."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.checkpoint_storage import clone_or_copy, write_json_checkpoint
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from scripts.run_anonymous_revision_inquiry import evaluate_terminal_model
from scripts.run_paid_state_revision import append_records


def read(reference):
    return read_method_reference(reference, ROOT)


def hold_panel(public):
    """Every public reset level, then each level after either control extreme."""
    levels = public['control_levels']['u']
    length = public['evaluation_max_trajectory_length']
    panel = []
    for level in levels:
        for mode, prefix in (('constant', []), ('from_low', [min(levels)]),
                             ('from_high', [max(levels)])):
            if prefix and prefix[0] == level:
                continue  # The constant sequence already tests this endpoint.
            panel.append({'mode': mode, 'level': level,
                'sequence': [{'u': value} for value in prefix + [level] * (length - len(prefix))]})
    return panel


def compact_trajectories(rows):
    """Persist controls and outcomes once; prefix histories reconstruct exactly."""
    records = []
    for row in rows:
        if not row.history:
            records.append({'sequence': [], 'targets': []})
        records[-1]['sequence'].append(dict(row.features))
        records[-1]['targets'].append(row.target)
    return records


def restore_trajectories(records):
    """Restore paid prefix inputs without placing responses in their histories."""
    rows = []
    for record in records:
        history = []
        for point, target in zip(record['sequence'], record['targets'], strict=True):
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append(dict(point))
    return rows


def retained_hold_score(reference, model, records, threshold):
    """Reuse the score of this frozen source on these retained trajectories."""
    if Path(reference['model_file']).read_bytes() != Path(model).read_bytes():
        return None
    score = read(reference['score_ref'])
    if (score.get('returncode') != 0 or score.get('truncated')
            or score.get('evaluation_rows') != sum(len(record['targets']) for record in records)
            or score.get('mse') is None):
        return None
    source_file = Path(reference['score_ref']['file'])
    source = read({'file': str(source_file), 'json_path': []})
    material_ref = source.get('panel_materials', {}).get('long_hold48', {}).get('source_ref')
    if material_ref is None:
        # The first frozen run saved its actual material beside summary.json.
        material = source_file.parent / 'long_hold_material.json'
        if not material.is_file():
            return None
        material_ref = {'file': str(material), 'json_path': []}
    if read(material_ref) != records:
        return None
    return {**score, 'passed': score['mse'] <= threshold,
        'reused_from': deepcopy(reference['score_ref']),
        'material_ref': deepcopy(material_ref), 'new_compute_cost': 0}


def run(registration, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json_checkpoint(output / 'registration.json', registration)
    confirmation_only = registration.get('post_confirmation_panels') == []
    timing = ExecutionTiming(('prepare', 'common_confirmation', 'fixed_panels'))
    budget = SearchBudget(registration['comparison_compute_cap'])
    agent = None
    terminal_rows = 0
    terminal_seconds = 0.
    result = {'status': 'preparing', 'fits': 0, 'model_calls': 0,
        'parent_archive_replaced': False, 'independent_task_transfer': False,
        'panels': {}, 'panel_materials': {}}
    try:
        with python_execution_deadline(registration['activity_seconds'], activity=True):
            with timing.phase('prepare'):
                case = read(registration['case_ref'])  # Apparatus identity only, never passed to a learner.
                lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
                public = lab.public_spec()
                prior = read(registration['paid_registration_ref'])
                records = append_records(read(prior['paid_records_ref']),
                    [read(reference) for reference in prior['append_record_refs']])
                rows = [row for record in records for row in lab.observation_rows(record)]
                config = replace(InquiryLLMConfig(), max_model_calls=0,
                    observation_budget=registration['confirmation_rows'],
                    confirmation_rows=registration['confirmation_rows'],
                    confirmation_mse_threshold=registration['mse_threshold'],
                    compute_budget=registration['common_compute_cap'],
                    execution_seconds=registration['execution_seconds'])
                agent = CommonArchiveInquiryAgent(None, output / 'workspace', arm='hegel',
                    config=config, source_task=False, confirmation_seed=registration['confirmation_seed'])
                agent.observations, agent.rows = records, rows
                agent._write_evidence()
                # Imported row identities continue while their inherited fees
                # remain on the source ledger, not this confirmation account.
                lab.restore_paid_account(records, agent.costs)
                models = {}
                for item in registration['models']:
                    path = item['id'] + '.py'
                    if item.get('source_file'):
                        clone_or_copy(item['source_file'], agent.workspace / path)
                        agent.files[path] = (agent.workspace / path).read_text()
                    else:
                        agent._save(path, export_temporal_program(Program.from_dict(read(item['program_ref'])),
                            feature_names=tuple(public['controls'])))
                    models[item['id']] = path
                held_records = (read(registration['hold_material_ref'])
                                if not confirmation_only and registration.get('hold_material_ref') else None)
                holds = ([] if confirmation_only else hold_panel(public) if held_records is None else
                         [{'sequence': record['sequence']} for record in held_records])
                write_json_checkpoint(output / 'frozen_protocol.json', {**registration,
                    'models': models, 'public_spec': public, 'holds': holds,
                    'paid_rows': len(rows), 'fresh_row_identity_offset': len(rows),
                    'new_labels_available_at_freeze': False})
                runtime = {Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
                    if name.startswith('hegel_machine') and getattr(module, '__file__', None)}
                runtime.update(Path(module.__file__).resolve() for module in (
                    sys.modules[__name__], sys.modules['scripts.run_anonymous_revision_inquiry'],
                    sys.modules['scripts.run_paid_state_revision']))
                for path in runtime:
                    destination = output / 'source_at_run' / path.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    clone_or_copy(path, destination)
            with timing.phase('common_confirmation'):
                selected = registration['selected_id']
                finish = agent._execute('finish', {'model': models[selected]}, lab)
                write_json_checkpoint(output / 'common_finish.json', finish)
                fresh = finish.get('fresh_confirmation')
                result['common_finish_ref'] = {'file': str(output / 'common_finish.json'), 'json_path': []}
                if fresh is None:
                    result.update(status='paid_precondition_or_execution_failed',
                                  error=finish.get('error'), paid_score=finish.get('mse'))
                    return result
                fresh_rows = [row for record in fresh['observations'] for row in lab.observation_rows(record)]
                result['panels']['fresh_confirmation'] = {selected: {
                    key: fresh.get(key) for key in ('returncode', 'mse', 'passed', 'rows', 'mse_threshold')}}
                result['submitted_candidate'] = fresh['submitted_candidate']
                result['common_model_adopted_in_isolated_workspace'] = agent.model is not None
                result['selected_passed_confirmation'] = fresh['passed']
            if confirmation_only:
                # This unit validates one already selected draft. Its earlier
                # competitors and fixed panels retain their existing evidence.
                result.update(status='completed', post_confirmation_panels=[],
                    selected_passed_all=fresh['passed'],
                    next_step=('retain_confirmed_candidate_and_review_method_evidence'
                        if fresh['passed'] else 'localize_frozen_counterexamples_before_any_revision'))
                return result
            with timing.phase('fixed_panels'):
                def compare(name, material, identifiers, retained=None):
                    nonlocal terminal_seconds
                    scores = result['panels'].setdefault(name, {})
                    for identifier in identifiers:
                        reference = (retained or {}).get(identifier)
                        if reference is not None:
                            reused = retained_hold_score(reference,
                                agent.workspace / models[identifier], held_records,
                                registration['mse_threshold'])
                            if reused is not None:
                                scores[identifier] = reused
                                write_json_checkpoint(output / 'panel_scores.json', result['panels'])
                                continue
                        if not budget.spend(len(material)):
                            raise ValueError('fixed panel scoring exceeds its registered allowance')
                        score = evaluate_terminal_model(agent.workspace, models[identifier], material,
                            timeout_seconds=registration['execution_seconds'])
                        terminal_seconds += score['python_execution_seconds']
                        predictions = score.pop('predictions', None)
                        if predictions is not None:
                            path = output / 'predictions' / f'{name}_{identifier}.json'
                            write_json_checkpoint(path, {'model': models[identifier], 'predictions': predictions})
                            score['predictions_ref'] = {'file': str(path), 'json_path': ['predictions']}
                        score['passed'] = (score.get('returncode') == 0 and not score.get('truncated')
                            and score.get('mse', float('inf')) <= registration['mse_threshold'])
                        scores[identifier] = score
                        write_json_checkpoint(output / 'panel_scores.json', result['panels'])
                        if score.get('returncode') != 0 or score.get('truncated'):
                            raise RuntimeError(f'fixed scoring did not complete: {name}/{identifier}')

                # Reuse the candidate's actual Common confirmation; only the
                # frozen controls need prediction on that shared fresh material.
                compare('fresh_confirmation', fresh_rows, [key for key in models if key != selected])
                material = lab.evaluation_rows(length=registration['terminal_random_rows'],
                                               seed=registration['terminal_seed'])
                terminal_rows += len(material)
                write_json_checkpoint(output / 'terminal_random_material.json', compact_trajectories(material))
                compare('random48', material, models)
                if held_records is None:
                    material = lab.evaluation_sequences([item['sequence'] for item in holds])
                    terminal_rows += len(material)
                    path = output / 'long_hold_material.json'
                    write_json_checkpoint(path, compact_trajectories(material))
                    material_ref = {'file': str(path), 'json_path': []}
                    role = 'fresh_frozen_evaluation'
                else:
                    material = restore_trajectories(held_records)
                    material_ref = deepcopy(registration['hold_material_ref'])
                    role = 'retained_evaluation_regression'
                result['panel_materials']['long_hold48'] = {
                    'role': role, 'source_ref': material_ref, 'rows': len(material),
                    'new_response_rows': len(material) if held_records is None else 0}
                compare('long_hold48', material, models,
                    registration.get('hold_score_refs') if held_records is not None else None)
            result.update(status='completed', selected_passed_all=all(
                panel[selected]['passed'] for panel in result['panels'].values()),
                next_step=('complete_parent_integration_and_test_independent_successors'
                    if all(panel[selected]['passed'] for panel in result['panels'].values())
                    else 'localize_frozen_counterexamples_before_any_revision'))
            return result
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        common = dict(agent.costs) if agent is not None else {}
        result['costs'] = {'common': common, 'fixed_comparison_compute': budget.spent,
            'compute_cost': common.get('compute_cost', 0) + budget.spent,
            'purchased_confirmation_rows': common.get('observation_cost', 0),
            'evaluator_response_rows': terminal_rows,
            'retained_evaluation_rows': (result['panel_materials'].get('long_hold48', {}).get('rows', 0)
                if result['panel_materials'].get('long_hold48', {}).get('role') == 'retained_evaluation_regression'
                else 0),
            'unique_new_response_rows': common.get('observation_cost', 0) + terminal_rows,
            'fixed_external_python_seconds': terminal_seconds, 'fits': 0, 'model_calls': 0}
        result['prior_costs_ref'] = registration['prior_costs_ref']
        result['timing'] = timing.result()
        write_json_checkpoint(output / 'summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(read({'file': str(args.registration), 'json_path': []}), args.output)
    print({key: result.get(key) for key in ('status', 'selected_passed_all', 'costs', 'next_step')})
