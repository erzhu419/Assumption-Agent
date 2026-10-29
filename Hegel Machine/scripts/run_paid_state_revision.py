#!/usr/bin/env python3
"""Revise frozen programs from paid reset trajectories through the Common route."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.checkpoint_storage import clone_or_copy, write_json_checkpoint
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.representation_diagnostics import representation_ambiguity
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.temporal_operator_agent import (
    export_temporal_program, _PROGRAM_REPRESENTATION_RUNNER)
from hegel_machine.cognition import state_search, legacy_research_bridge


def read(reference):
    return read_method_reference(reference, ROOT)


def append_records(old, incoming):
    """Rebase separately purchased reset records without changing either source."""
    records = deepcopy(old)
    offset = sum(len(record['rows']) for record in records)
    for original in incoming:
        record = deepcopy(original)
        record['observation_index'] = len(records)
        record['row_indices'] = list(range(offset, offset + len(record['rows'])))
        records.append(record)
        offset += len(record['rows'])
    return records


def bind_restored_state_author(agent, contract):
    """Bind the saved ordinary author to the live question and complete pool."""
    from hegel_machine.cognition.finite_native_author import (
        bind_actual_representation_parent, export_finite_native_author)
    from hegel_machine.cognition.research_episodes import SEMANTIC_FIELDS

    contract = deepcopy(contract)
    owner = next(question for question in agent.questions
                 if question['id'] == contract['question']['id'])
    declaration = {key: deepcopy(owner[key]) for key in (
        'id', 'meaning_version', 'parent_id', 'depends_on', *SEMANTIC_FIELDS,
        'origin', 'theory_target', 'use_sites', 'representation_research_basis') if key in owner}
    models = list(dict.fromkeys([*contract['predictor_paths'], *agent.candidate_paths]))
    contract.update(question=declaration, predictor_paths=models)
    bound = bind_actual_representation_parent(agent, owner)
    if bound['status'] == 'not_representation_export':
        # A new source imports its saved Programs directly, so there is no
        # native-export event in this episode. Bind the actual comparison
        # winner to that existing declaration instead of the diagnostic seed.
        model = bound.get('source_model')
        identifier = next((name for name in contract['native_representation_plan']['parent_theory_ids']
            if model == f'{name}.py' and model in models), None)
        if identifier is not None:
            theory = agent.theory_index[identifier]
            expected = export_temporal_program(Program.from_dict(theory['program']),
                feature_names=tuple(owner['theory_target']['role_signature'])).encode('utf-8')
            if ((agent.workspace / model).read_bytes() != expected
                    or (agent.workspace / agent._evaluated_summary()['model']).read_bytes() != expected):
                raise ValueError('actual registered parent no longer matches its saved Program')
            bound = {'status': 'bound', 'binding_kind': 'existing_registered_program',
                'source_model': model, 'theory_id': identifier, 'version': theory['version'],
                'source_ref': {'file': str(agent.workspace / 'research_state.json'),
                    'json_path': ['theory_index', identifier, 'program']}}
    plan = deepcopy(owner.get('native_representation_plan') or contract['native_representation_plan'])
    if bound['status'] == 'bound':
        for question in (owner, declaration):
            question['theory_target']['parent_theory_id'] = bound['theory_id']
            question['use_sites'] = [{'kind': 'supply_component',
                'model': bound['source_model'], 'slot': 'representation'}]
        plan['parent_theory_ids'] = list(dict.fromkeys([
            *plan['parent_theory_ids'], bound['theory_id']]))
    elif bound['status'] != 'not_representation_export':
        raise ValueError('actual retained representation parent cannot be bound')
    contract['native_representation_plan'] = plan
    owner['native_representation_plan'] = deepcopy(plan)
    declaration['native_representation_plan'] = deepcopy(plan)
    agent.knowledge['finite_native_author'] = contract
    agent._active_question_id = owner['id']
    agent._save('ordinary.py', export_finite_native_author())
    return {'models': models, 'actual_parent_binding': bound}


def restore_exposed_common_state(registration, workspace, laboratory, *, config):
    """Append paid Common counterexamples to the full historical research scene.

    The source checkpoint supplies the complete candidate bank and score cache;
    a confirmation-only workspace is evidence, never a replacement archive.
    No predictions, fitting or purchases occur during this restoration.
    """
    previous = read(registration['training_ref'])
    agent = CommonArchiveInquiryAgent(None, workspace, arm='hegel', source_task=False,
        config=config, knowledge=previous['outgoing_knowledge'])
    agent.restore_practice_research(previous, previous['workspace'], laboratory,
        confirmation_seed=registration['restore_confirmation_seed'])
    before = len(agent.rows)
    incoming = [read(reference) for reference in registration['append_record_refs']]
    agent.observations = append_records(agent.observations, incoming)
    agent.rows = [row for record in agent.observations for row in laboratory.observation_rows(record)]
    if len(agent.rows) != registration['expected_rows']:
        raise ValueError('exposed Common records differ from the registered row count')
    for reference in registration.get('additional_paid_cost_refs', []):
        paid_costs = read(reference)
        for key in agent.costs:
            agent.costs[key] += paid_costs[key]
    laboratory.restore_paid_account(agent.observations, agent.costs)
    bound = bind_restored_state_author(agent, read(registration['finite_native_author_ref']))
    # Earlier confirmation remains in history. It does not submit a model for
    # the enlarged development scene; the next common comparison chooses it.
    agent.model = None
    agent._write_evidence()
    agent._prepare_program_state()
    receipt = {'status': 'full_common_scene_restored_with_paid_counterexamples',
        'training_ref': deepcopy(registration['training_ref']),
        'append_record_refs': deepcopy(registration['append_record_refs']),
        'additional_paid_cost_refs': deepcopy(registration.get('additional_paid_cost_refs', [])),
        'restored_rows': before, 'exposed_rows': len(agent.rows) - before,
        'total_rows': len(agent.rows), 'restored_costs': deepcopy(agent.costs),
        'paid_score_reuse': 'retained_complete_prefixes; unrecorded_suffix_predictions_remain_unscored',
        'new_observations': 0, 'new_compute_cost': 0, 'new_fits': 0, **bound}
    write_json_checkpoint(agent.workspace / 'paid_scene_restoration.json', receipt)
    return {'agent': agent, 'lab': laboratory, 'models': bound['models'], 'restoration': receipt}


def restore_completed_state(registration, workspace, laboratory, *, config):
    """Resume a completed Common scene with its already exposed frozen panel.

    Restore the full paid archive, then offer its actual current parent to the
    ordinary author. This function performs no query, fit, prediction or method
    selection. The caller supplies total limits and a new confirmation seed.
    """
    from dataclasses import asdict
    from hegel_machine.cognition.finite_native_author import (
        bind_actual_representation_parent, export_finite_native_author)
    from hegel_machine.cognition.models import Observation
    from hegel_machine.cognition.paid_score_history import retain_frozen_readout_score
    from hegel_machine.cognition.research_episodes import SEMANTIC_FIELDS

    previous = read(registration['training_ref'])
    outcome = read(registration['outcome_ref'])
    panel = read(registration['panel_ref'])
    evaluation = read(registration['evaluation_ref'])
    if not panel['labels_opened_after_both_outputs_freeze']:
        raise ValueError('resume only a completed, exposed frozen panel')
    # The training snapshot precedes the terminal readout. Its paid execution
    # belongs to the old account even though the old training file lacks it.
    previous = {**previous, 'costs': deepcopy(outcome['costs_after'])}
    agent = CommonArchiveInquiryAgent(None, workspace, arm='hegel', source_task=False,
        config=config, knowledge=previous['outgoing_knowledge'])
    agent.restore_practice_research(previous, previous['workspace'], laboratory,
        confirmation_seed=registration['confirmation_seed'])
    contract = deepcopy(read(registration['finite_native_author_ref']))
    owner = next(question for question in agent.questions
                 if question['id'] == contract['question']['id'])
    # Actions carry a declaration, not a stale copy of the live question. Its
    # completed research rounds continue to live in agent.questions; sending
    # them back through _open_question would overwrite newly completed work.
    for key in ('id', 'meaning_version', *SEMANTIC_FIELDS, 'theory_target', 'use_sites'):
        if key in owner:
            contract['question'][key] = deepcopy(owner[key])
    agent.knowledge['finite_native_author'] = contract
    agent._active_question_id = owner['id']
    model = agent.evaluated_model['source_model']
    assessment = next(entry['paid_score_reuse']['execution_basis_ref']
        for comparison in reversed(agent.operator_results)
        for entry in comparison.get('evaluations', []) if entry.get('path') == model)

    incoming, exposed = [], []
    for source in panel['material']:
        history, rows = [], []
        for features, target in zip(source['sequence'], source['targets'], strict=True):
            row = Observation(dict(features), target, tuple(history))
            rows.append(asdict(row))
            exposed.append(row)
            history.append(dict(features))
        incoming.append({'action': {'kind': 'trajectory', 'sequence': deepcopy(source['sequence'])},
            'rows': rows, 'origin': 'exposed_evaluation_development_not_purchase',
            'source_ref': deepcopy(registration['panel_ref']),
            'observation_cost': 0, 'measurement_cost': 0})
    if len(exposed) != panel['new_response_rows']:
        raise ValueError('the completed frozen panel must retain all purchased rows')
    agent.observations = append_records(agent.observations, incoming)
    agent.rows.extend(exposed)
    restore_account = getattr(laboratory, 'restore_paid_account', None)
    if restore_account is not None:
        restore_account(agent.observations, agent.costs)
    directory = agent.workspace / '_exposed_frozen_readout'
    directory.mkdir(exist_ok=True)
    panel_path, evaluation_path = directory / 'panel.json', directory / 'evaluation.json'
    write_json_checkpoint(panel_path, {'rows': [asdict(row) for row in exposed],
        'source_ref': deepcopy(registration['panel_ref']),
        'all_predictors_frozen_before_labels': True})
    write_json_checkpoint(evaluation_path, {'evaluation': evaluation,
        'frozen_model': outcome['frozen_model'],
        'source_ref': deepcopy(registration['evaluation_ref'])})
    union = retain_frozen_readout_score(agent, model=model,
        parent_assessment_ref=assessment,
        panel_ref={'file': str(panel_path), 'json_path': []},
        evaluation_ref={'file': str(evaluation_path), 'json_path': []},
        public_spec=laboratory.public_spec(), question_id=owner['id'])

    # A completed native search has added predictors beyond the initial catalog.
    # Keep all of them, and bind the actual retained export instead of silently
    # researching only the initial five transferred parents again.
    models = list(dict.fromkeys([*contract['predictor_paths'], *agent.candidate_paths]))
    contract['predictor_paths'] = models
    bound = bind_actual_representation_parent(agent, owner)
    if bound['status'] != 'bound':
        raise ValueError('completed native parent has no retained executable Program address')
    for declaration in (owner, contract['question']):
        declaration['theory_target']['parent_theory_id'] = bound['theory_id']
        declaration['use_sites'] = [{'kind': 'supply_component', 'model': model,
                                    'slot': 'representation'}]
    plan = deepcopy(contract['native_representation_plan'])
    plan['parent_theory_ids'] = list(dict.fromkeys([*plan['parent_theory_ids'], bound['theory_id']]))
    contract['native_representation_plan'] = plan
    owner['native_representation_plan'] = deepcopy(plan)
    contract['question']['native_representation_plan'] = deepcopy(plan)
    catalog_path = directory / 'predictor_catalog.json'
    write_json_checkpoint(catalog_path, {'models': models,
        'source_catalog_ref': deepcopy(contract['predictor_catalog_ref']),
        'training_ref': deepcopy(registration['training_ref'])})
    contract['predictor_catalog_ref'] = {'file': str(catalog_path), 'json_path': []}
    agent._save('ordinary.py', export_finite_native_author())
    agent._prepare_program_state()
    receipt = {'status': 'completed_scene_restored_with_exposed_readout',
        'training_ref': deepcopy(registration['training_ref']),
        'outcome_ref': deepcopy(registration['outcome_ref']),
        'panel_ref': deepcopy(registration['panel_ref']),
        'evaluation_ref': deepcopy(registration['evaluation_ref']),
        'restored_costs': deepcopy(agent.costs), 'model': model, 'models': models,
        'paid_readout_reuse': union, 'actual_parent_binding': bound,
        'exposed_rows': len(exposed), 'total_rows': len(agent.rows),
        'new_observations': 0, 'new_compute_cost': 0, 'new_fits': 0,
        'independent_confirmation': False}
    write_json_checkpoint(directory / 'restoration.json', receipt)
    return {'agent': agent, 'lab': laboratory, 'models': models, 'restoration': receipt}


def restore_paid_state(registration, workspace, *, config=None):
    """Restore this small public scene for native research or ordinary procedures."""
    records = append_records(read(registration['paid_records_ref']),
        [read(reference) for reference in registration['append_record_refs']])
    rows = [row for record in records for row in record['rows']]
    if len(rows) != registration['expected_rows']:
        raise ValueError('paid material differs from the registered row count')
    public = read(registration['public_spec_ref'])
    parents = {item['candidate_id']: Program.from_dict(read(item['program_ref']))
               for item in registration['parents']}
    config = config or replace(InquiryLLMConfig(), max_model_calls=0,
        compute_budget=registration['research_compute_cap'] + 2 * len(rows),
        confirmation_mse_threshold=registration['mse_threshold'])
    agent = CommonArchiveInquiryAgent(None, workspace, arm='hegel', source_task=False,
        config=config)
    agent.observations, agent.rows = records, rows
    for identifier, program in parents.items():
        agent._save(f'{identifier}.py', export_temporal_program(program,
            feature_names=tuple(public['controls'])))
    return agent, public, parents


def run(registration, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json_checkpoint(output / 'registration.json', registration)
    timing = ExecutionTiming(('load', 'diagnostic', 'research', 'exported_readout'))
    agent = None
    result = {'status': 'preparing', 'independent_confirmation': False,
        'parent_replaced': False, 'full_archive_comparison': 'not_run',
        'comparison_scope': 'registered_parents_and_current_native_candidates'}
    diagnostic_compute = delivery_compute = 0
    try:
        with python_execution_deadline(registration['activity_seconds'], activity=True):
            with timing.phase('load'):
                agent, public, parents = restore_paid_state(registration, output / 'workspace')
                rows = agent.rows
                runtime = {Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
                    if name.startswith('hegel_machine') and getattr(module, '__file__', None)}
                runtime.add(Path(__file__).resolve())
                for path in runtime:
                    destination = output / 'source_at_run' / path.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    clone_or_copy(path, destination)
            with timing.phase('diagnostic'):
                parent_id = registration['diagnostic_parent_id']
                namespace = {}
                exec((agent.workspace / f'{parent_id}.py').read_text()
                     + _PROGRAM_REPRESENTATION_RUNNER, namespace)
                agent._charge(len(rows))
                diagnostic_compute = len(rows)
                diagnostic = representation_ambiguity([
                    namespace['represent']({'features': row['features'], 'history': row['history']})
                    for row in rows], rows, scope='registered_paid_reset_trajectories')
                write_json_checkpoint(output / 'representation_diagnostic.json', diagnostic)
            with timing.phase('research'):
                question = {'id': 'paid_representation_revision', 'kind': 'representation',
                    'question': 'Which executable history representation explains the paid counterexamples?',
                    'origin': {'kind': 'frozen_forecasts_refuted',
                        'evidence_ref': registration['counterexample_ref']},
                    'parent_id': 'task', 'depends_on': [], 'answer_type': 'program',
                    'representation_diagnostic': diagnostic,
                    'use_sites': [{'kind': 'supply_component',
                        'model': f'{parent_id}.py', 'slot': 'representation'}]}
                route = agent._execute('run', {'operator': 'research_route',
                    'research_question': question,
                    'theories': [{'id': identifier, 'program': program.to_dict()}
                                 for identifier, program in parents.items()],
                    'native_representation_plan': {'parent_theory_ids': list(parents),
                        'validation_records': registration['validation_records'],
                        'compute_limit': registration['research_compute_cap'],
                        'max_candidates': registration['max_candidates']}},
                    SimpleNamespace(public_spec=lambda: public))
                write_json_checkpoint(output / 'route.json', route)
                native = load_native_representation_result(route['native_representation_ref'], agent.workspace)
                result.update(native_ref=route['native_representation_ref'],
                    native_status=native['status'], split=native['split'],
                    search_directions=native.get('search_directions'),
                    retained_comparison=native['retained_comparison'],
                    candidate_comparison=[{key: item[key] for key in (
                        'id', 'nodes', 'training_mse', 'validation_mse', 'development_mse',
                        'search_direction') if key in item} for item in native['candidates']])
            with timing.phase('exported_readout'):
                selected = native.get('selected')
                if selected is not None:
                    model = route['selected_model'] or f"{selected['id']}.py"
                    namespace = {}
                    exec((agent.workspace / model).read_text(), namespace)
                    agent._charge(len(rows))
                    delivery_compute = len(rows)
                    predictions = [namespace['predict']({'features': row['features'],
                        'history': row['history']}) for row in rows]
                    mse = sum((prediction - row['target']) ** 2
                              for prediction, row in zip(predictions, rows)) / len(rows)
                    write_json_checkpoint(output / 'selected_predictions.json', {
                        'model': model, 'rows': len(rows), 'predictions': predictions,
                        'mse': mse, 'data_role': 'exposed_development',
                        'source_records_ref': {'file': str(output / 'registration.json'),
                                               'json_path': []}})
                    result['selected'] = {key: value for key, value in selected.items()
                        if key not in {'program', 'research_program', 'validation_predictions',
                                       'best_retained_validation_predictions'}}
                    result['selected'].update(exported_model=str(agent.workspace / model),
                        exported_development_mse=mse,
                        export_matches_native_mse=abs(mse - (
                            selected['training_mse'] * native['fitting_rows']
                            + selected['validation_mse'] * native['validation_rows']) / len(rows)) < 1e-12)
            result['status'] = 'completed'
            return result
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['costs'] = {**(agent.costs if agent is not None else {}),
            'representation_compute': diagnostic_compute, 'delivery_compute': delivery_compute,
            'new_observations': 0}
        result['prior_costs_ref'] = registration['prior_costs_ref']
        result['timing'] = timing.result()
        write_json_checkpoint(output / 'summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(read({'file': str(args.registration), 'json_path': []}), args.output)
    print({key: result.get(key) for key in ('status', 'native_status', 'selected', 'costs')})
