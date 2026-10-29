#!/usr/bin/env python3
"""Pair existing state-research methods on one paid Common scene."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.checkpoint_storage import clone_or_copy, write_json_checkpoint
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.finite_native_author import (
    export_finite_native_author, publish_research_options, research_segment_options)
from hegel_machine.cognition.method_feedback import (begin_method_feedback,
    ensure_python_method_topic, export_python_method_topic, record_python_method_feedback,
    restore_python_method_bodies)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.research_programs import (
    propose_research_action_revisions, export_research_selection_program)
from hegel_machine.cognition.theory_consolidation import (
    install_python_research_method, theory_library_candidate_versions)
from scripts.run_anonymous_revision_inquiry import evaluate_terminal_model
from scripts.run_frozen_state_confirmation import compact_trajectories, restore_trajectories
from scripts.run_paid_state_revision import read, restore_paid_state, bind_restored_state_author
from scripts.run_method_action_pairing import (
    GLOBAL_COMPONENTS, quote_method_action_pairing, run_method_action_pairing)


def save(path, value):
    write_json_checkpoint(path, value)
    return {'file': str(Path(path).resolve()), 'json_path': []}


def _scene(registration, workspace):
    paid = read(registration['paid_registration_ref'])
    cfg = replace(InquiryLLMConfig(), max_model_calls=0, token_budget=0,
        compute_budget=registration['per_cell_compute_cap'], observation_budget=0,
        execution_seconds=registration['time_limits']['billed_python_seconds'],
        confirmation_mse_threshold=paid['mse_threshold'])
    initial_ref = registration.get('initial_scene_ref')
    if initial_ref is not None:
        previous = read(initial_ref)
        costs = previous['costs']
        cfg = replace(cfg, compute_budget=costs['compute_cost'] + cfg.compute_budget,
            observation_budget=costs['observation_cost'],
            execution_seconds=costs['python_execution_seconds'] + cfg.execution_seconds)
        case = read(registration['evaluation']['case_ref'])
        lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'],
            **({'startup_seed': case['startup_seed']} if 'startup_seed' in case else {}))
        public = read(paid['public_spec_ref'])
        if lab.public_spec() != public:
            raise ValueError('initial scene and registered public apparatus differ')
        agent = CommonArchiveInquiryAgent(None, workspace, arm='hegel', source_task=False,
            config=cfg)
        agent.restore_practice_research(previous, previous['workspace'], lab,
            confirmation_seed=registration.get('restore_confirmation_seed', registration['evaluation']['seed']))
        parents = {item['candidate_id']: Program.from_dict(read(item['program_ref']))
                   for item in paid['parents']}
        question = next(q for q in agent.questions if q['id'] == agent._active_question_id)
    else:
        agent, public, parents = restore_paid_state(paid, workspace, config=cfg)
        lab = SimpleNamespace(public_spec=lambda: public)
        question = agent._open_question(deepcopy(registration['question']))
    agent.rows = [row if isinstance(row, Observation) else
                  Observation(row['features'], row['target'], tuple(row['history']))
                  for row in agent.rows]
    register_theories(agent, [{'id': identifier, 'program': program.to_dict()}
                              for identifier, program in parents.items()])
    agent._active_question_id = question['id']
    contract = deepcopy(registration['finite_native_author'])
    if initial_ref is not None:
        bound = bind_restored_state_author(agent, contract)
        models = bound['models']
    else:
        contract['question'] = deepcopy(question)
        agent.knowledge['finite_native_author'] = contract
        agent._save('ordinary.py', export_finite_native_author())
        models = contract['predictor_paths']
    return agent, lab, [model.removesuffix('.py') for model in models]


def prepare(paid_registration_ref, output, *, case_ref, evaluation_seed,
            compute_cap=60000, research_compute_cap=30000, max_candidates=4,
            activity_seconds=180., python_seconds=120., evaluation_rows=512,
            approval_ref=None, initial_scene_ref=None, restore_confirmation_seed=None,
            finite_native_author_ref=None, method_pair_ref=None, prior_pair_refs=(),
            data_role='development', method_condition_compute_cap=1000, method_topic_ref=None):
    """Freeze actual ordinary/alternative proposals before any paired fitting."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'registration.json'
    ref = lambda key: {'file': str(path), 'json_path': [key]}
    paid = read(paid_registration_ref)
    public = read(paid['public_spec_ref'])
    parents = [item['candidate_id'] for item in paid['parents']]
    context = {'domain': 'public_paid_state_research'}
    question = {'id': 'paid_state_method_revision', 'kind': 'program_revision',
        'question': 'Which executable revision explains the paid counterexamples?',
        'meaning_version': 1, 'parent_id': 'task', 'depends_on': [], 'answer_type': 'program',
        'context_ref': context, 'origin': {'kind': 'paid_prediction_counterexamples',
            'evidence_ref': deepcopy(paid['counterexample_ref'])},
        'theory_target': {'parent_theory_id': paid['diagnostic_parent_id'],
            'role_signature': {name: {'role': 'control', 'unit': '1', 'time': 'current'}
            for name in public['controls']}, 'output_unit': '1', 'context_ref': context},
        'use_sites': [{'kind': 'supply_component', 'model': paid['diagnostic_parent_id'] + '.py',
                      'slot': 'representation'}]}
    plan = {'parent_theory_ids': parents, 'validation_records': paid['validation_records'],
            'compute_limit': research_compute_cap, 'max_candidates': max_candidates}
    registration = {'paid_registration_ref': deepcopy(paid_registration_ref),
        'data_role': data_role,
        'question': question, 'per_cell_compute_cap': compute_cap,
        'time_limits': {'billed_python_seconds': python_seconds, 'active_wall_seconds': activity_seconds},
        'finite_native_author': {'question': question, 'predictor_paths': [p + '.py' for p in parents],
            'predictor_catalog_ref': ref('parent_catalog'), 'max_compute': research_compute_cap,
            'max_candidates': max_candidates, 'boundary_resolution': .01,
            'native_representation_plan': plan},
        'parent_catalog': deepcopy(paid['parents']),
        'evaluation': {'case_ref': deepcopy(case_ref), 'seed': evaluation_seed,
            'rows': evaluation_rows, 'threshold': paid['mse_threshold'],
            'role': ('independent_successor_evaluation_after_both_outputs_freeze'
                if data_role == 'independent_successor' else
                'new_development_evaluation_revealed_after_both_outputs_freeze')},
        'quality_contract': {'threshold': paid['mse_threshold'], 'loss': 'mse'},
        'resource_contract': {'compute_cap': compute_cap, 'observation_cap': 0,
            'time_limits': {'billed_python_seconds': python_seconds, 'active_wall_seconds': activity_seconds}},
        'paid_scene': {key: deepcopy(paid[key]) for key in
            ('paid_records_ref', 'append_record_refs', 'expected_rows')},
        'physical_source_ref': deepcopy(case_ref), 'D1': {'legal': False},
        'approval_ref': deepcopy(approval_ref), 'execution_authorized': approval_ref is not None,
        'comparison_scope': 'registered_parents_and_current_native_exports_not_full_historical_archive',
        'prior_costs_ref': deepcopy(paid['prior_costs_ref'])}
    if initial_scene_ref is not None:
        registration['initial_scene_ref'] = deepcopy(initial_scene_ref)
        if restore_confirmation_seed is not None:
            registration['restore_confirmation_seed'] = restore_confirmation_seed
        if finite_native_author_ref is not None:
            registration['finite_native_author'] = deepcopy(read(finite_native_author_ref))
    save(path, registration)
    agent, lab, parents = _scene(registration, output / 'preparation')
    if initial_scene_ref is not None:
        contract = agent.knowledge['finite_native_author']
        contract.update(max_compute=research_compute_cap, max_candidates=max_candidates)
        contract['native_representation_plan'].update(
            compute_limit=research_compute_cap, max_candidates=max_candidates)
        question = next(q for q in agent.questions if q['id'] == agent._active_question_id)
        question['native_representation_plan'] = deepcopy(contract['native_representation_plan'])
        contract['question']['native_representation_plan'] = deepcopy(contract['native_representation_plan'])
        registration['question'] = deepcopy(contract['question'])
        registration['finite_native_author'] = deepcopy(contract)
        registration['parent_catalog'] = {'models': deepcopy(contract['predictor_paths']),
            'source_catalog_ref': deepcopy(contract['predictor_catalog_ref'])}
        registration['finite_native_author']['predictor_catalog_ref'] = ref('parent_catalog')
        agent.knowledge['finite_native_author']['predictor_catalog_ref'] = ref('parent_catalog')
        agent._prepare_program_state()
        save(path, registration)
    _, _, split = agent._split({'validation_records': paid['validation_records']})
    options = publish_research_options(agent, lab.public_spec())
    view = research_segment_options(lab.public_spec(), agent.observations, agent.knowledge,
        state=agent._program_state(), decision={'research_options': options})
    ordinary = next(item for item in view['options'] if item['id'] == 'native_representation:ordinary')
    proposed = propose_research_action_revisions(ordinary['proposal'], view,
        source_refs=[paid_registration_ref], max_candidates=1)
    if not proposed['candidates']:
        raise ValueError('no distinct executable alternative: retain the ordinary method')
    candidate = proposed['candidates'][0]
    frozen_methods = read(method_pair_ref) if method_pair_ref is not None else None
    continued_topic = method_topic_ref or (frozen_methods or {}).get('method_topic_ref')
    if continued_topic is not None:
        registration['method_topic_ref'] = deepcopy(continued_topic)
    cell_names = ('00', '10', '20') if (frozen_methods is not None
        and set(frozen_methods['cells']) == {'00', '10', '20'}) else ('00', '10')
    if len(cell_names) == 3:
        registration['comparison_kind'] = 'three_method_same_data'
        registration['evaluation']['role'] = 'independent_successor_evaluation_after_all_outputs_freeze'
    if frozen_methods is not None:
        ordinary_source = Path(frozen_methods['ordinary_source_ref']['file']).read_text(encoding='utf-8')
        if ordinary_source != (agent.workspace / 'ordinary.py').read_text(encoding='utf-8'):
            raise ValueError('the frozen method cohort requires its original ordinary source')
        selection_plans = {name: deepcopy(frozen_methods['cells'][name]['selection_plan'])
                           for name in cell_names}
    else:
        selection_plans = {'00': {'ordinary_method_path': 'ordinary.py'},
            '10': {**candidate['research_selection_plan'], 'ordinary_method_path': 'ordinary.py'}}
    evidence_ref = save(output / 'static_selection.json', {
        'ordinary': ordinary, 'view': view, 'proposals': proposed,
        'scientific_fits': 0, 'new_observations': 0,
        'frozen_method_pair_ref': deepcopy(method_pair_ref)})
    exports = max(item['complete_quote']['max_exported_models'] for item in view['options'])
    rows = paid['expected_rows']
    cells = {}
    existing_ids = {item['id'] for item in theory_library_candidate_versions(agent)}
    for name in cell_names:
        base = (frozen_methods['cells'][name]['id'] if frozen_methods is not None else
                'state_method_P0' if name == '00' else 'state_method_P1')
        identifier, revision = base, 1
        while frozen_methods is None and identifier in existing_ids:
            revision += 1
            identifier = f'{base}_revision_{revision}'
        cells[name] = {'method_ref': {'id': identifier, 'version': 1},
            'method_path': f'shared/{identifier}.py', 'selection_plan': selection_plans[name],
            'scope': {'paid_rows': rows, 'pool_models': len(parents), 'max_exports': exports},
            'components': {'search': {'compute': research_compute_cap + rows * len(parents), 'observations': 0},
                'export': {'compute': 0, 'observations': 0},
                'complete_comparison': {'compute': rows * (len(parents) + exports), 'observations': 0},
                'returned_parent': {'compute': rows + agent.config.confirmation_rows, 'observations': 0}},
            'planned_native_compute': research_compute_cap,
            'start_state_ref': deepcopy(initial_scene_ref or paid_registration_ref), 'data_ref': ref('paid_scene'),
            'split_ref': ref('split'), 'quality_contract_ref': ref('quality_contract'),
            'resource_contract_ref': ref('resource_contract'), 'evaluation_ref': ref('evaluation'),
            'return_contract': 'retained_parent', 'compute_cap': compute_cap, 'observation_cap': 0,
            'time_limits': deepcopy(registration['time_limits']), 'max_steps': 8}
        if frozen_methods is not None and frozen_methods['cells'][name].get('selection_ref'):
            cells[name]['selection_ref'] = deepcopy(frozen_methods['cells'][name]['selection_ref'])
    components = {key: {'compute': 0, 'observations': 0} for key in GLOBAL_COMPONENTS}
    components['projection_proposals']['compute'] = 1 + len(options)
    components['Qeval_predictions'] = {'compute': len(cell_names) * evaluation_rows, 'observations': evaluation_rows}
    components['feedback_preservation']['compute'] = method_condition_compute_cap
    registration.update(cells=cells, split=split, components=components,
        method_ref=deepcopy(cells['10']['method_ref']), static_selection_ref=evidence_ref,
        method_condition_compute_cap=method_condition_compute_cap,
        preparation_costs={'compute_cost': 1 + len(options), 'observation_cost': 0, 'model_calls': 0},
        compute_cap=len(cell_names) * compute_cap + method_condition_compute_cap + 1 + len(options),
        observation_cap=evaluation_rows)
    if frozen_methods is not None:
        registration['method_pair_ref'] = deepcopy(method_pair_ref)
        registration['prior_pair_refs'] = deepcopy(list(prior_pair_refs))
    registration['quote'] = quote_method_action_pairing(registration)
    if registration['quote']['status'] != 'quoted':
        raise ValueError(f"incomplete registered quote: {registration['quote']}")
    save(path, registration)
    return registration


def make_factories(registration, output):
    output = Path(output).resolve()
    evaluation = {}
    accounts, method_starts = {}, {}

    def restore_cell(name, contract, destination):
        resumed = registration.get('continuation')
        previous = read(resumed['outcome_refs'][name]) if resumed else None
        physical = registration
        if previous is not None:
            physical = {**registration, 'initial_scene_ref': previous['continuation_state_ref'],
                'per_cell_compute_cap': contract['compute_cap'], 'time_limits': contract['time_limits']}
        agent, lab, parents = _scene(physical, destination / 'workspace')
        topic_ref = registration.get('method_topic_ref')
        # Restoring a frozen method must not divert its unfinished scientific
        # invocation into method learning. Its owner resumes after both returns.
        if topic_ref is not None:
            topic = read(topic_ref)
            restore_python_method_bodies(agent, topic['methods'])
        if previous is None:
            plan = contract['selection_plan']
            method_source_refs = ([registration['static_selection_ref']]
                if registration.get('method_pair_ref') is None else
                read(registration['method_pair_ref'])['cells'][name]['source_refs'])
            candidate = agent.theory_consolidation.get('candidates', {}).get(contract['method_ref']['id'])
            if candidate is not None:
                retained = {key: value for key, value in candidate['selection_plan'].items()
                            if key != 'ordinary_method_path'}
                supplied = {key: value for key, value in plan.items() if key != 'ordinary_method_path'}
                if retained != supplied or candidate['version'] != contract['method_ref']['version']:
                    raise ValueError('frozen method differs from the continuing topic')
            else:
                candidate = install_python_research_method(agent, candidate_id=contract['method_ref']['id'],
                    source=export_research_selection_program(plan), selection_plan=plan,
                    source_refs=method_source_refs, max_steps=contract['max_steps'],
                    selection_ref=contract.get('selection_ref'))
            profile = begin_method_feedback(agent, lab, candidate=candidate)
            profile_ref = save(destination / 'profile.json', profile['profile'])
            artifact_ref = save(destination / 'method_artifact.json', candidate)
            origin = save(destination / 'invocation.json', {'method_ref': contract['method_ref'],
                'start_state_ref': contract['start_state_ref'], 'profile_ref': profile_ref})
        else:
            candidate = read(previous['method_artifact_ref'])
            profile_ref, artifact_ref, origin = (previous[key] for key in
                ('profile_ref', 'method_artifact_ref', 'origin_ref'))
        accounts[name] = (agent, lab)
        method_starts[name] = len(agent.method_results)
        return {'agent': agent, 'lab': lab, 'path': candidate['execution']['path'],
            'models': [name + '.py' for name in parents], 'scope': deepcopy(contract['scope']),
            'question_ref': {'id': registration['question']['id'], 'meaning_version': 1},
            'profile_ref': profile_ref, 'method_artifact_ref': artifact_ref, 'origin_ref': origin,
            'native_attempt_limit': 1, 'cost_refs': [registration['prior_costs_ref']],
            **{key: deepcopy(contract[key]) for key in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}}

    def prepare_evaluation():
        contract = registration['evaluation']
        paid_material = registration.get('paid_evaluation_material_ref')
        if paid_material is not None:
            if not registration.get('retrospective_completion'):
                raise ValueError('already exposed evaluation material requires retrospective completion')
            evaluation['rows'] = restore_trajectories(read(paid_material)['material'])
            if len(evaluation['rows']) != contract['rows']:
                raise ValueError('paid evaluation material differs from the original row contract')
            return save(output / 'evaluation_material.json', {
                'paid_original_ref': deepcopy(paid_material), 'new_response_rows': 0,
                'retrospective_completion': True, 'labels_opened_after_new_outputs_freeze': True})
        case = read(contract['case_ref'])
        lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'],
            **({'startup_seed': case['startup_seed']} if 'startup_seed' in case else {}))
        evaluation['rows'] = lab.evaluation_rows(length=contract['rows'], seed=contract['seed'])
        return save(output / 'evaluation_material.json', {'material': compact_trajectories(evaluation['rows']),
            'new_response_rows': len(evaluation['rows']), 'labels_opened_after_both_outputs_freeze': True,
            'labels_opened_after_all_outputs_freeze': True})

    def evaluate_frozen(name, frozen, evaluation_ref, account):
        rows = evaluation['rows']
        if account.config.compute_budget - account.costs['compute_cost'] < len(rows):
            raise ValueError('frozen prediction allowance exhausted')
        account.costs['compute_cost'] += len(rows)
        scored = evaluate_terminal_model(frozen['workspace'], frozen['model'], rows,
            timeout_seconds=max(.001, account.config.execution_seconds - account.costs['python_execution_seconds']))
        account.costs['python_execution_seconds'] += scored['python_execution_seconds']
        save(output / name / 'frozen_evaluation.json', scored)
        if (scored.get('returncode') != 0 or scored.get('mse') is None
                or scored.get('truncated') or scored.get('evaluation_rows') != len(rows)):
            raise ValueError('frozen predictor did not return complete measured scores')
        loss = scored['mse']
        threshold = registration['evaluation']['threshold']
        return {'loss': loss, 'coverage': len(rows), 'completed': loss <= threshold,
            'quality': {'loss': loss, 'threshold': threshold}}

    def ordinary_use(pair_refs, outcome_refs, destination):
        # Return the actual paired research to a Common method question. The
        # existing learner may retain P0; neither a winner nor a guard is supplied.
        consumer = '10' if '10' in accounts else '00'
        agent, lab = accounts[consumer]
        before = agent.costs['compute_cost']
        allowance = registration['method_condition_compute_cap']
        agent.config = replace(agent.config, compute_budget=before + allowance)
        checkpoint = save(agent.workspace / 'training.json',
            agent.checkpoint_result('paired_research_complete'))
        challenger = read(outcome_refs['10'])
        start = method_starts[consumer]
        segments = ([{'root_result_ref': {**checkpoint, 'json_path': [
                'method_results', len(agent.method_results) - 1]},
            'decision_start_ref': {**checkpoint, 'json_path': ['method_results', start]}}]
            if len(agent.method_results) > start else [])
        ordinary_ref = deepcopy(registration['static_selection_ref'])
        ordinary_ref['json_path'] = [*ordinary_ref.get('json_path', []), 'ordinary', 'proposal']
        accumulated_pairs = [*registration.get('prior_pair_refs', []), *pair_refs]
        request = {'ordinary_method_path': 'ordinary.py', 'ordinary_proposal_ref': ordinary_ref,
            'segments': segments, 'tested_candidate_ref': challenger['method_artifact_ref'],
            'paired_candidate_id': registration['method_ref']['id'], 'pair_refs': accumulated_pairs,
            'revised_candidate_id': registration['method_ref']['id'] + f'_conditions_{len(accumulated_pairs)}',
            'max_steps': registration['cells']['10']['max_steps']}
        excluded = []
        if registration.get('continuation', {}).get('execution_repair'):
            excluded = deepcopy(accumulated_pairs)
            request['pair_refs'] = []
        owner = ensure_python_method_topic(agent,
            parent_question_ref={'question_id': registration['question']['id'], 'meaning_version': 1},
            topic_ref=registration.get('method_topic_ref'), candidate_refs=[registration['method_ref']],
            mission=None if registration.get('method_topic_ref') else {
                'purpose': 'method_learning', 'mission_id': str(output / 'method_learning'),
                'allocation': {'compute': registration['method_condition_compute_cap']}})
        owner.setdefault('method_use_plan', {})['candidate_id'] = registration['method_ref']['id']
        routed = agent._execute('run', {'operator': 'research_route',
            'research_question': deepcopy(owner), 'method_action_revision': request,
            'allocation': {'compute': allowance}}, lab)
        route_ref = save(destination / 'method_research.json', routed)
        action = routed.get('method_action_research', {})
        selected = action.get('paired_selection', {})
        reference = save(destination / 'method_selection.json', selected)
        # Deliver the complete actual invocations after the paired analysis.
        # New sources extend one method problem, not one disconnected topic
        # per episode. Each invocation keeps its P0/P1 identity and preprofile.
        sources = [{'candidate_ref': read(outcome_ref)['method_artifact_ref'],
                    'workspace': str(accounts[name][0].workspace)}
                   for name, outcome_ref in outcome_refs.items()]
        members = ([{'id': action['installed_candidate']['id'],
                     'version': action['installed_candidate']['version']}]
                   if action.get('installed_candidate') else [])
        owner = ensure_python_method_topic(agent,
            parent_question_ref={'question_id': registration['question']['id'], 'meaning_version': 1},
            topic_ref=registration.get('method_topic_ref'),
            candidate_refs=members, candidate_sources=sources)
        ordinary_returns = []
        for name, outcome_ref in outcome_refs.items():
            actual = read(outcome_ref)
            if not actual.get('method_invocation_binding'):
                ordinary_returns.append({'status': 'no_ordinary_invocation',
                    'outcome_ref': deepcopy(outcome_ref), 'new_compute_cost': 0})
                continue
            candidate = agent.theory_consolidation['candidates'][actual['method_ref']['id']]
            binding = deepcopy(actual['method_invocation_binding'])
            records = agent.theory_consolidation['method_feedback'][candidate['id']]['outcome_refs']
            if not any(read(item)['origin_ref'] == binding['origin_ref'] for item in records):
                record_python_method_feedback(agent, candidate, binding,
                    {'status': actual['status'], 'execution_complete': False},
                    {'profile': read(actual['profile_ref']),
                     'question_ref': {'question_id': registration['question']['id'], 'meaning_version': 1},
                     'theory_target': deepcopy(registration['question'].get('theory_target', {}))})
            ordinary_returns.append(agent._execute('run', {'operator': 'method_action_feedback',
                'candidate_id': candidate['id'], 'binding': binding, 'result_ref': outcome_ref,
                'allocation': {'compute': max(0, allowance - (agent.costs['compute_cost'] - before))}}, lab))
        topic_ref = export_python_method_topic(agent, owner)
        save(agent.workspace / 'training.json', agent.checkpoint_result('method_evidence_consumed'))
        return {'status': routed.get('status', 'method_actions_unresolved'),
            'research_route_ref': route_ref, 'selection_ref': reference,
            'method_topic_ref': topic_ref, 'ordinary_returns': ordinary_returns,
            'new_compute_cost': agent.costs['compute_cost'] - before,
            'local_program_supported': selected.get('local_program_supported', False),
            'general_guard_supported': selected.get('general_guard_supported', False),
            'ordinary_method_retained': not selected.get('local_program_supported', False),
            'installed_candidate_id': (action.get('installed_candidate') or {}).get('id'),
            'excluded_condition_pair_refs': excluded,
            'condition_exclusion_reason': ('public_execution_repair_intervened_in_original_pair'
                if excluded else None),
            'consumed_pair_refs': deepcopy(request['pair_refs']),
            'ordinary_research_route_executed': True,
            'method_execution_after_feedback': False, 'successors_started': False}

    return {'restore_cell': restore_cell, 'prepare_evaluation': prepare_evaluation,
            'evaluate_frozen': evaluate_frozen, 'ordinary_use': ordinary_use}


def run(registration, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', registration)
    sources = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        if (name.startswith('hegel_machine') or name.startswith('scripts.run_')) and getattr(module, '__file__', None):
            sources.add(Path(module.__file__).resolve())
    for source in sources:
        if source.is_relative_to(ROOT):
            destination = output / 'source_at_run' / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            clone_or_copy(source, destination)
    return run_method_action_pairing(registration, output, **make_factories(registration, output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(read({'file': str(args.registration), 'json_path': []}), args.output)
    print({'status': result['status'], 'ordinary_use': result['ordinary_use']})
