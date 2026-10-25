#!/usr/bin/env python3
"""Statically re-export saved Programs and check the real finite-author roster.

No laboratory, fitter, predictor or study loop is executed. Common baseline
contexts stay at their original references; only the small colliding native
context and newly exported code occupy the independent audit workspace.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition import finite_native_author, structural_inquiry
from hegel_machine.cognition.checkpoint_storage import clone_or_copy
from hegel_machine.cognition.language import Program


def _read(path):
    return json.loads(Path(path).read_text())


def _reference(path, *keys):
    return {'file': str(path), 'json_path': list(keys)}


def _baseline(workspace, inventory):
    contexts = {}
    for entry in inventory['models']:
        contexts[entry['destination_context']] = Path(entry['context'])
    colliding = '_experiment_models/native_conditions_0001_0001/_evaluated_model'
    for name, source in sorted(contexts.items(), key=lambda item: len(Path(item[0]).parts)):
        destination = workspace / name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == colliding:
            destination.mkdir()
            # This actual inherited context contains six standalone .py files.
            # Keep their writable audit copies independent even on a regression.
            for path in source.iterdir():
                if not path.is_file() or path.suffix != '.py':
                    raise ValueError('unexpected native baseline dependency; retain its full context')
                clone_or_copy(path, destination / path.name)
        else:
            destination.symlink_to(source, target_is_directory=True)
    paths = [entry['destination'] for entry in inventory['models']]
    return paths, {name: (workspace / name).read_bytes() for name in paths}


def audit(study_directory, output):
    study_directory, output = Path(study_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('retain earlier audits and use a fresh output directory')
    output.mkdir(parents=True)
    inventory = _read(study_directory / 'baseline_inventory.json')
    plan = _read(study_directory / 'source_plan.json')
    registration = _read(study_directory / 'registration.json')
    cases = []
    for case in plan['cases']:
        origin = study_directory / 'source' / case['id'] / 'archive'
        state_ref = origin / 'research_state.json'
        state = _read(state_ref)
        qi = next(i for i, q in enumerate(state['questions']) if q['id'] == 'native_public_response')
        saved_rounds = state['questions'][qi]['native_condition_research']
        observations = _read(origin / 'laboratory_observations.json')
        workspace = output / case['id']
        workspace.mkdir()
        old_paths, old_bodies = _baseline(workspace, inventory)
        question = {'id': 'native_public_response', 'question': 'Retained static export audit',
                    'native_condition_research': [{'candidates': []}]}
        agent = SimpleNamespace(workspace=workspace, questions=[{'id': 'task'}, question],
            costs={'compute_cost': 0, 'observation_cost': 0, 'python_execution_seconds': 0.},
            config=SimpleNamespace(compute_budget=250000, observation_budget=576, confirmation_rows=96),
            rows=[], observations=observations, evaluated_model=None, _inquiry_stack=[],
            _return_structural_answer=lambda *args: None)
        public_question = deepcopy(plan['prepared_public_questions'][case['id']])
        controls = list(public_question['theory_target']['role_signature'])
        laboratory = SimpleNamespace(public_spec=lambda: {'controls': controls})
        invocations, previous_outputs = [], {}
        generated_rounds = [(i, r) for i, r in enumerate(saved_rounds) if r['candidates']]

        def export_saved(round_index, saved, mode):
            source_candidates = []
            source_refs = []
            for i, item in enumerate(saved['candidates']):
                source_candidates.append({key: deepcopy(item[key]) for key in
                                          ('program', 'fit_mse', 'source')})
                source_refs.append(_reference(state_ref, 'questions', qi,
                    'native_condition_research', round_index, 'candidates', i, 'program'))
            result = {'status': 'candidates_require_evaluation',
                'finite_prior': 'retained historical Programs; static export only',
                'candidates': source_candidates, 'proposed_experiments': [],
                'fit_evidence': [], 'evidence': []}
            with patch.object(structural_inquiry, 'research_native_conditions',
                              return_value=deepcopy(result)), patch.object(Program, 'predict',
                              side_effect=AssertionError('prediction forbidden')):
                structural_inquiry.execute_native_condition_research(
                    agent, {'max_compute': 0}, laboratory, question)
            entry = question['native_condition_research'][-1]
            models = [c['model'] for c in entry['candidates']]
            assert not set(models).intersection(old_paths)
            assert not set(models).intersection(previous_outputs)
            assert all((workspace / path).read_bytes() == text for path, text in old_bodies.items())
            assert all((workspace / path).read_bytes() == text for path, text in previous_outputs.items())
            previous_outputs.update({name: (workspace / name).read_bytes() for name in models})
            # Exercise the existing finite procedure itself, stopping at the
            # proposed research_choice action. Its comparison is never run.
            current = {'questions': [{'id': 'task'}, public_question], 'operator_results': [
                {'operator': 'native_condition_research', 'research_question_id': question['id'],
                 'question_meaning_version': 1, 'native_condition_research': {
                     'observed_record_count': len(observations),
                     'candidates': [{'model': path} for path in previous_outputs]}}]}
            memory = {'finite_native_author': {**registration['spec']['native_search'],
                'question': public_question, 'predictor_paths': old_paths,
                'predictor_catalog_ref': {'file': str(study_directory / 'baseline_inventory.json')}},
                'program_results': []}
            with patch.object(finite_native_author, '_state', return_value=current):
                before = Path.cwd()
                try:
                    import os
                    os.chdir(workspace)
                    action = finite_native_author.select_actions({'controls': controls}, observations, memory)
                finally:
                    os.chdir(before)
            assert action['action'] == 'run' and action['arguments']['operator'] == 'research_choice'
            roster = action['arguments']['models']
            assert roster[:len(old_paths)] == old_paths
            assert all(name in roster for name in previous_outputs)
            invocations.append({'mode': mode, 'historical_round': round_index,
                'source_program_refs': source_refs, 'bank_round_index': entry['round_index'],
                'export_directory': str(Path(models[0]).parent), 'export_count': len(models),
                'old_roster_count': len(old_paths), 'proposed_roster_count': len(roster),
                'all_original_paths_and_bodies_preserved': True,
                'all_previous_new_exports_preserved': True,
                'real_select_actions_operator': action['arguments']['operator'],
                'all_originals_and_exports_in_real_roster': True})

        for i, saved in generated_rounds:
            export_saved(i, saved, 'chronological_bank')
        if generated_rounds:
            i, saved = generated_rounds[-1]
            question['native_condition_research'] = []
            export_saved(i, saved, 'reopened_question')
            # Simulate restoring the actual retained local bank into the same
            # execution workspace; directory allocation must not reset with it.
            agent.questions[1] = question = deepcopy(question)
            export_saved(i, saved, 'restored_bank')
        cases.append({'case': case['id'], 'group_material': 'archive',
            'historical_state_ref': _reference(state_ref),
            'paid_records_ref': _reference(origin / 'laboratory_observations.json'),
            'original_roster': len(old_paths), 'source_saved_endpoints': sum(len(r['candidates']) for _, r in generated_rounds),
            'invocations': invocations, 'audit_workspace': str(workspace)})
    result = {'status': 'static_export_and_real_roster_verified',
        'baseline_inventory_ref': _reference(study_directory / 'baseline_inventory.json'),
        'implementation_ref': {'file': str(ROOT / 'src/hegel_machine/cognition/structural_inquiry.py'),
                               'symbol': 'execute_native_condition_research'},
        'cases': cases, 'researcher_stubbed': True, 'exporter_and_select_actions_real': True,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'new_model_calls': 0,
        'new_scientific_compute': 0,
        'scope': 'Metadata/code preservation mechanism audit on consumed historical Programs, '
                 'not new science or a source rerun. Proposed research_choice actions were not executed.'}
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.study_dir, args.output)
    print(json.dumps({'status': result['status'], 'cases': len(result['cases']),
                      'source_saved_endpoints': sum(c['source_saved_endpoints'] for c in result['cases'])}))


if __name__ == '__main__':
    main()
