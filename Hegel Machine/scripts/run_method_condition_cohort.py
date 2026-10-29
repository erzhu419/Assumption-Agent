#!/usr/bin/env python3
"""Accumulate declared method pairs, then freeze a policy for new successors."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from scripts import run_independent_state_method_source as source_entry
from scripts.run_state_method_pairing import read, save
from hegel_machine.cognition.checkpoint_storage import clone_or_copy


def source_receipt(result, destination):
    """Reference paid scientific artifacts and total each actual charge once."""
    costs = {'compute_cost': 0, 'observation_cost': 0, 'measurement_cost': 0,
             'python_execution_seconds': 0., 'model_calls': 0, 'reported_tokens': 0}

    def add(values):
        for key in costs:
            costs[key] += values.get(key, 0)

    if not result.get('prefix_reused'):
        add(result.get('prefix_costs', {}))
    add(result.get('shared_comparison_costs', {}))
    add(result.get('static_proposal_costs', {}))
    receipt = {'status': result['status'],
        'source_result_ref': {'file': str(destination / 'source_result.json'), 'json_path': []},
        'prefix_reused': result.get('prefix_reused', False), 'arms': {},
        'pair_refs': [], 'method_feedback': None, 'costs': costs}
    if result.get('initial_comparison_ref'):
        receipt['initial_comparison_ref'] = result['initial_comparison_ref']
    if not result.get('study_ref'):
        return receipt
    summary = read(result['study_ref'])
    receipt.update(study_ref=result['study_ref'], pair_refs=summary['pair_refs'],
        method_feedback=summary['ordinary_use'],
        evaluation_lifecycle=summary['evaluation_lifecycle'])
    for name, reference in summary['outcome_refs'].items():
        outcome = read(reference)
        add(outcome.get('new_stage_costs', outcome['costs']))
        receipt['arms'][name] = {'outcome_ref': reference,
            **{key: deepcopy(outcome.get(key)) for key in (
                'method_ref', 'loss', 'quality', 'coverage', 'completed', 'execution_complete',
                'status', 'profile_ref', 'comparison', 'costs', 'new_stage_costs', 'frozen_model')}}
    lifecycle = summary['evaluation_lifecycle']
    if lifecycle.get('labels_revealed') and lifecycle.get('receipt_ref'):
        costs['observation_cost'] += read(lifecycle['receipt_ref'])['new_response_rows']
    costs['compute_cost'] += summary['ordinary_use'].get('new_compute_cost', 0)
    return receipt


def freeze_successor_policy(method_pair_ref, feedback, output):
    """Use actual Common selection, without inventing a condition or preference."""
    selection = read(feedback['selection_ref'])
    if not selection.get('local_program_supported'):
        return None
    original = read(method_pair_ref)
    if feedback.get('method_topic_ref'):
        original['method_topic_ref'] = deepcopy(feedback['method_topic_ref'])
    policy_selection_ref = deepcopy(feedback['selection_ref'])
    if selection.get('general_guard_supported'):
        action = read(feedback['research_route_ref'])['method_action_research']
        candidate = action.get('installed_candidate')
        if not candidate or not candidate['selection_plan'].get('condition'):
            raise ValueError('supported condition must have an actual installed method')
        # Retaining a conditional method also retains the evidence that
        # defines where that guard may run. The latest whole-method analysis
        # is not a replacement for the original branch support.
        policy_selection_ref = deepcopy(action.get('policy_selection_ref')
            or candidate.get('selection_ref') or policy_selection_ref)
        plan = {**deepcopy(candidate['selection_plan']), 'ordinary_method_path': 'ordinary.py'}
        original['cells']['10'] = {'id': candidate['id'], 'selection_plan': plan,
            'source_refs': deepcopy(candidate['source_evidence_refs']),
            'selection_ref': deepcopy(policy_selection_ref)}
        kind = 'learned_condition'
    else:
        kind = 'locally_supported_fixed_method'
    pair_ref = save(output / 'frozen_successor_method_pair.json', original)
    return {'selected_policy_kind': kind, 'method_pair_ref': pair_ref,
        'selection_ref': policy_selection_ref,
        'analysis_selection_ref': deepcopy(feedback['selection_ref']),
        'research_route_ref': deepcopy(feedback['research_route_ref']),
        'frozen_before_successor_labels': True}


def complete_frozen_three_way(source, method_bundle_ref, output, *, approval_ref):
    """Run one new method, retaining two completed arms and their paid Q."""
    from scripts import run_state_method_pairing as pairing
    from scripts.run_method_action_pairing import GLOBAL_COMPONENTS, quote_method_action_pairing

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    registration = deepcopy(read(source['registration_ref']))
    previous = read(source['study_ref'])
    bundle = read(method_bundle_ref)
    original_cells = deepcopy(registration['cells'])
    retained = source.get('retained_cells', {'00': '00', '20': '10'})
    missing = {'00', '10', '20'} - set(retained)
    if len(retained) != 2 or len(missing) != 1:
        raise ValueError('a retrospective completion retains two cells and executes one')
    new_cell = missing.pop()
    registration['cells'] = {name: original_cells[old] for name, old in retained.items()}
    fixed = deepcopy(original_cells.get(new_cell, original_cells['10']))
    selected = bundle['cells'][new_cell]
    fixed.update(method_ref={'id': selected['id'], 'version': 1},
        method_path=f"shared/{selected['id']}.py",
        selection_plan=deepcopy(selected['selection_plan']))
    fixed.pop('selection_ref', None)
    if selected.get('selection_ref') is not None:
        fixed['selection_ref'] = deepcopy(selected['selection_ref'])
    registration['cells'][new_cell] = fixed
    registration.update(comparison_kind='three_method_same_data', retrospective_completion=True,
        retained_registration_ref=deepcopy(source['registration_ref']),
        retained_outcome_refs={name: previous['outcome_refs'][old] for name, old in retained.items()},
        method_pair_ref=deepcopy(method_bundle_ref), method_ref=deepcopy(fixed['method_ref']),
        prior_pair_refs=[], data_role='independent_successor', method_condition_compute_cap=0,
        paid_evaluation_material_ref=deepcopy(previous['evaluation_lifecycle']['receipt_ref']),
        approval_ref=deepcopy(approval_ref), execution_authorized=True)
    registration['components'] = {key: {'compute': 0, 'observations': 0} for key in GLOBAL_COMPONENTS}
    # Keep the original per-arm evaluation reserve, including for retained arms.
    # Only the new arm will incur this charge; old cost is evidence, not repayment.
    registration['components']['Qeval_predictions']['compute'] = 3 * registration['evaluation']['rows']
    registration['compute_cap'] = sum(c['compute_cap'] for c in registration['cells'].values())
    registration['observation_cap'] = 0
    registration['quote'] = quote_method_action_pairing(registration)
    save(output / 'registration.json', registration)
    if registration['quote']['status'] != 'quoted':
        raise ValueError(f"missing-arm quote incomplete: {registration['quote']['missing']}")
    study = output / 'study'
    summary = pairing.run(registration, study)
    result = {'status': summary['status'], 'prefix_reused': True,
        'study_ref': {'file': str(study / 'summary.json'), 'json_path': []},
        'original_source_ref': deepcopy(source), 'source_role': 'retrospective_missing_arm_completion'}
    save(output / 'source_result.json', result)
    receipt = source_receipt(result, output)
    receipt['new_cell'] = new_cell
    return receipt


def run_three_way_study(registration, output):
    """Execute a declared three-method panel or a bounded pre-use stratum."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    entry = output / 'source_at_run' / 'scripts' / Path(__file__).name
    entry.parent.mkdir(parents=True)
    clone_or_copy(Path(__file__).resolve(), entry)
    started = monotonic()
    registration_ref = save(output / 'registration.json', registration)
    result = {'status': 'running', 'registration_ref': registration_ref,
        'sources': [], 'branch_attempts': 0,
        'costs': {'compute_cost': 0, 'observation_cost': 0, 'measurement_cost': 0,
                  'python_execution_seconds': 0., 'model_calls': 0, 'reported_tokens': 0}}

    def retain(receipt, source_id, role):
        receipt.update(source_id=source_id, evidence_role=role)
        result['sources'].append(receipt)
        for key, value in receipt['costs'].items():
            result['costs'][key] += value
        result['wall_seconds'] = monotonic() - started
        save(output / 'result.json', result)
        print({'source': source_id, 'role': role, 'status': receipt['status'],
               'new_costs': receipt['costs'],
               'losses': {k: v['loss'] for k, v in receipt['arms'].items()}}, flush=True)

    source_requests = registration.get('sources', registration.get('branch_sources', []))
    shared_request = (read(registration['source_template_ref'])
                      if registration.get('source_template_ref') else {})
    attempt_limit = registration.get('max_branch_attempts')
    try:
        for index, source in enumerate(registration.get('supplements', [])):
            name = f'paid_completion_{index:03d}'
            receipt = complete_frozen_three_way(source, registration['method_bundle_ref'],
                output / name, approval_ref=registration_ref)
            retain(receipt, name, 'retrospective_completion')
            new = receipt['arms'][receipt['new_cell']]
            if not new.get('execution_complete') or new['loss'] is None:
                result['status'] = 'missing_arm_incomplete_no_new_sources'
                return result
        for template in source_requests:
            if attempt_limit is not None and result['branch_attempts'] >= attempt_limit:
                break
            request = {**deepcopy(shared_request), **deepcopy(template)}
            request.update(data_role='independent_successor',
                method_pair_ref=registration['method_bundle_ref'], prior_pair_refs=[])
            destination = output / request['source_id']
            actual = source_entry.run(request, destination)
            receipt = source_receipt(actual, destination)
            if template.get('factorial_cell') is not None:
                receipt['factorial_cell'] = deepcopy(template['factorial_cell'])
            receipt['pre_use_condition_ref'] = deepcopy(actual.get('pre_use_condition_ref'))
            if receipt['pre_use_condition_ref']:
                decision = read(receipt['pre_use_condition_ref'])
                receipt['pre_use_condition'] = decision
                # A failed admitted attempt consumes a slot too; no search for
                # a successful outcome after the declared attempt limit.
                result['branch_attempts'] += int(decision['matched'])
            retain(receipt, request['source_id'],
                template.get('evidence_role', 'independent_pre_use_branch_stratum'))
        result['status'] = 'completed_registered_three_method_study'
        remaining = len(source_requests) - sum(
            s['evidence_role'] != 'retrospective_completion' for s in result['sources'])
        result['source_candidates_not_opened'] = remaining
        if 'branch_sources' in registration:
            result['branch_candidates_not_opened'] = remaining
    except Exception as error:
        result.update(status='interrupted_no_retry', error=f'{type(error).__name__}: {error}',
            interrupted_source_costs='not_summarized; retained source artifacts require review')
        raise
    finally:
        result['wall_seconds'] = monotonic() - started
        save(output / 'result.json', result)
    return result


def rebuild_paid_profiles(request, output):
    """Reproject pre-call paid state and relearn conditions, without world research.

    Each source supplies its original pair and prepared registration. Outcome
    returns, physical sources, action records and original identities stay fixed;
    only the newly reconstructed pre-call profile is used for method learning.
    The final source's already registered, unspent feedback allowance pays for
    the new condition pass.
    """
    from dataclasses import replace
    from scripts import run_state_method_pairing as pairing
    from hegel_machine.cognition.method_feedback import begin_method_feedback
    from hegel_machine.cognition.research_programs import export_research_selection_program
    from hegel_machine.cognition.theory_consolidation import install_python_research_method

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    entry = output / 'source_at_run' / 'scripts' / Path(__file__).name
    entry.parent.mkdir(parents=True)
    clone_or_copy(Path(__file__).resolve(), entry)
    started = monotonic()
    request_ref = save(output / 'request.json', request)
    derived, source_receipts = [], []
    agent = lab = registration = original_pair = None
    for index, source in enumerate(request['sources']):
        registration = read(source['registration_ref'])
        original_pair = read(source['pair_ref'])
        if original_pair['data_role'] != 'development':
            raise ValueError('profile reconstruction must not consume independent successor results')
        directory = output / f'source_{index:03d}'
        agent, lab, _ = pairing._scene(registration, directory / 'workspace')
        before = deepcopy(agent.costs)
        contract = registration['cells']['10']
        bundle = read(registration['method_pair_ref'])
        plan = contract['selection_plan']
        candidate = install_python_research_method(agent, candidate_id=contract['method_ref']['id'],
            source=export_research_selection_program(plan), selection_plan=plan,
            source_refs=bundle['cells']['10']['source_refs'],
            max_steps=contract['max_steps'], selection_ref=contract.get('selection_ref'))
        profile = begin_method_feedback(agent, lab, candidate=candidate)
        if profile is None or agent.costs != before:
            raise ValueError('paid profile projection must return a profile without scientific execution')
        profile_ref = save(directory / 'profile.json', profile['profile'])
        pair = deepcopy(original_pair)
        for side in ('baseline', 'challenger'):
            old_ref = original_pair[side + '_ref']
            old = read(old_ref)
            outcome = {**old, 'profile_ref': profile_ref,
                'original_outcome_ref': deepcopy(old_ref),
                'original_profile_ref': deepcopy(old['profile_ref'])}
            pair[side + '_ref'] = save(directory / f'{side}_outcome.json', outcome)
        pair['original_pair_ref'] = deepcopy(source['pair_ref'])
        # The original invocation identity prevents the reprojected sample from
        # being counted as another actual method execution.
        pair['origin_ref'] = deepcopy(original_pair.get('origin_ref', source['pair_ref']))
        pair_ref = save(directory / 'pair.json', pair)
        derived.append(pair_ref)
        source_receipts.append({'registration_ref': source['registration_ref'],
            'original_pair_ref': source['pair_ref'], 'derived_pair_ref': pair_ref,
            'profile_ref': profile_ref, 'new_compute_cost': 0})
    if agent is None:
        raise ValueError('at least one completed development pair is required')
    previous = read(request['method_feedback_ref'])
    old_action = read(previous['research_route_ref'])['method_action_research']
    allowance = registration['method_condition_compute_cap'] - previous['new_compute_cost']
    if allowance < 1:
        raise ValueError('the original method feedback allowance is exhausted')
    agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + allowance)
    ordinary_ref = deepcopy(registration['static_selection_ref'])
    ordinary_ref['json_path'] = [*ordinary_ref.get('json_path', []), 'ordinary', 'proposal']
    actual = read(original_pair['challenger_ref'])
    method_request = {'ordinary_method_path': 'ordinary.py', 'ordinary_proposal_ref': ordinary_ref,
        'segments': [{key: deepcopy(segment[key]) for key in ('root_result_ref', 'decision_start_ref')}
                     for segment in old_action['segments']],
        'tested_candidate_ref': actual['method_artifact_ref'],
        'paired_candidate_id': actual['method_ref']['id'], 'pair_refs': derived,
        'revised_candidate_id': actual['method_ref']['id'] + '_paid_pool_conditions',
        'max_steps': registration['cells']['10']['max_steps']}
    question = {'id': registration['question']['id'] + ':paid_pool_method_choice',
        'kind': 'method_use', 'meaning_version': 1,
        'question': 'Which tested research program is supported by the reconstructed paid pre-use pool?',
        'parent_id': registration['question']['id'], 'answer_type': 'research_method',
        'method_use_plan': {'candidate_id': actual['method_ref']['id'], 'binding': {}}}
    before = agent.costs['compute_cost']
    projection_wall_seconds = monotonic() - started
    routed = agent._execute('run', {'operator': 'research_route',
        'research_question': question, 'method_action_revision': method_request,
        'allocation': {'compute': allowance}}, lab)
    route_ref = save(output / 'method_research.json', routed)
    choice = routed.get('method_action_research', {}).get('paired_selection', {})
    selection_ref = save(output / 'method_selection.json', choice)
    feedback = {'research_route_ref': route_ref, 'selection_ref': selection_ref,
        'new_compute_cost': agent.costs['compute_cost'] - before,
        'local_program_supported': choice.get('local_program_supported', False),
        'general_guard_supported': choice.get('general_guard_supported', False),
        'consumed_pair_refs': derived, 'method_execution_after_feedback': False}
    result = {'status': routed.get('status'), 'request_ref': request_ref,
        'source_receipts': source_receipts, 'pair_refs': derived,
        'ordinary_use': feedback, 'remaining_original_compute_allowance': allowance - feedback['new_compute_cost'],
        'new_compute_cost': feedback['new_compute_cost'], 'new_observations': 0,
        'new_predictions': 0, 'new_fits': 0, 'model_calls': 0,
        'profile_projection_wall_seconds': projection_wall_seconds,
        'wall_seconds': monotonic() - started}
    save(output / 'result.json', result)
    return result


def _method_dependency(plan, workspace):
    return (Path(workspace) / plan['ordinary_method_path']).resolve()


def _static_method_action(plan, dependency, public, observations, memory):
    """Execute this method with its own retained ordinary implementation."""
    from hegel_machine.cognition.research_programs import export_research_selection_program

    compiled = {}
    exec(export_research_selection_program({**deepcopy(plan),
        'ordinary_method_path': str(dependency)}), compiled)
    own_memory = {**deepcopy(memory), 'ordinary_method_path': str(dependency),
                  'research_selection_plan': deepcopy(plan)}
    return compiled['select_actions'](public, observations, own_memory)


def form_method_from_paid_returns(request, output):
    """Form a method through Common, then check its paid selection contexts.

    This entry performs no fitting or laboratory query. Equality to an already
    tested finite body ends the unit instead of repeating its scientific panel.
    """
    import os
    from scripts import run_state_method_pairing as pairing
    from hegel_machine.cognition.finite_native_author import publish_research_options
    from hegel_machine.cognition.method_conditions import method_condition_features, method_condition_support
    from hegel_machine.cognition.research_programs import export_research_selection_program
    from hegel_machine.cognition.theory_consolidation import install_python_research_method

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    request_ref = save(output / 'request.json', request)
    for relative in ('scripts/run_method_condition_cohort.py',
                     'src/hegel_machine/cognition/method_action_research.py',
                     'src/hegel_machine/cognition/research_programs.py'):
        destination = output / 'source_at_run' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(ROOT / relative, destination)
    registration = read(request['registration_ref'])
    agent, lab, _ = pairing._scene(registration, output / 'workspace')
    before = deepcopy(agent.costs)
    question = {'id': registration['question']['id'] + ':content_from_paid_returns',
        'kind': 'method_use', 'meaning_version': 1, 'answer_type': 'research_method',
        'question': 'What reusable research content is supported as a hypothesis by these actual returns?',
        'parent_id': registration['question']['id'],
        'method_use_plan': {'candidate_id': request['candidate_id'], 'binding': {}}}
    routed = agent._execute('run', {'operator': 'research_route',
        'research_question': question, 'method_action_revision': request['method_request'],
        'allocation': {'compute': request['analysis_compute_cap']}}, lab)
    route_ref = save(output / 'method_research.json', routed)
    action = routed.get('method_action_research', {})
    proposed = next((item for item in action.get('proposals', {}).get('candidates', [])
        if item.get('producer') == 'finite_paid_guard_mature_startup_relation_induction'), None)
    result = {'status': 'no_method_content_formed', 'request_ref': request_ref,
        'research_route_ref': route_ref, 'selection_checks': [], 'new_compute_cost': 0,
        'new_predictions': 0, 'new_fits': 0, 'new_observations': 0, 'model_calls': 0,
        'new_scientific_support': False, 'old_method_retained': True}
    if proposed is not None:
        plan = {**deepcopy(proposed['research_selection_plan']), 'ordinary_method_path': 'ordinary.py'}
        candidate = install_python_research_method(agent, candidate_id=request['candidate_id'],
            source=export_research_selection_program(plan), selection_plan=plan,
            source_refs=proposed['source_refs'], selection_ref=request['condition_selection_ref'],
            max_steps=registration['cells']['20']['max_steps'],
            method_question_ref={'question_id': question['id'], 'meaning_version': 1})
        result['formed_candidate_ref'] = save(output / 'formed_method.json', candidate)
        old = read(request['verified_method_ref'])

        def executable_plan(value):
            value = {**deepcopy(value), 'ordinary_method_path': 'ordinary.py'}
            value.pop('source_refs', None)
            return value

        expected = executable_plan(old['selection_plan'])
        actual = executable_plan(plan)
        result['same_finite_selection_body'] = (
            export_research_selection_program(actual) == export_research_selection_program(expected))
        result['verified_method_ref'] = deepcopy(request['verified_method_ref'])
        actual_dependency = _method_dependency(candidate['selection_plan'], agent.workspace)
        expected_dependency = _method_dependency(old['selection_plan'],
            Path(request['verified_method_ref']['file']).parent / 'workspace')
        result['same_dependency'] = actual_dependency.read_bytes() == expected_dependency.read_bytes()
        result['dependency_refs'] = {'formed': {'file': str(actual_dependency)},
                                    'verified': {'file': str(expected_dependency)}}
        agent.costs['compute_cost'] += 3  # Compilation, finite-body and frozen dependency comparison.
        selection = read(request['condition_selection_ref'])
        for index, item in enumerate(request.get('selection_checks', [])):
            context, apparatus, _ = pairing._scene(read(item['registration_ref']),
                output / f'selection_{index:02d}')
            profile = read(item['profile_ref'])
            state = context._program_state()
            memory = {**context.knowledge, 'research_state': state, 'program_results': [],
                'method_decision': {'research_options': publish_research_options(context, apparatus.public_spec())},
                'method_condition_features': method_condition_features(profile, plan['condition_feature_paths']),
                'method_condition_support': method_condition_support(profile, selection),
                'method_candidate_pool': deepcopy(profile.get('research', {}).get('candidate_pool'))}
            choices = []
            previous_directory = Path.cwd()
            try:
                os.chdir(context.workspace)
                for selected_plan, dependency in ((candidate['selection_plan'], actual_dependency),
                                                   (old['selection_plan'], expected_dependency)):
                    choices.append(_static_method_action(selected_plan, dependency,
                        apparatus.public_spec(), context.observations, memory))
            finally:
                os.chdir(previous_directory)
            agent.costs['compute_cost'] += 3  # Two selection calls and their actual action comparison.
            native = choices[0].get('arguments', {}).get('native_representation_plan', {})
            result['selection_checks'].append({'source_ref': deepcopy(item),
                'same_action': choices[0] == choices[1],
                'actions_available': all(choice.get('action') and not choice.get('stop') for choice in choices),
                'condition_support': memory['method_condition_support'],
                'parent_count': len(native.get('parent_theory_ids', [])),
                'max_candidates': native.get('max_candidates'),
                'startup_parent_priority': native.get('startup_parent_priority', []),
                'formed_action': choices[0], 'verified_action': choices[1]})
        same = (result['same_finite_selection_body'] and result['same_dependency']
                and bool(result['selection_checks'])
                and all(item['same_action'] and item['actions_available'] for item in result['selection_checks']))
        result['status'] = ('formed_equivalent_method_no_repeat_research' if same
                            else 'formed_method_requires_separate_execution_evidence')
    result['new_compute_cost'] = agent.costs['compute_cost'] - before['compute_cost']
    result['wall_seconds'] = monotonic() - started
    result['formation_cost_scope'] = 'Common projection and finite relation analysis, compilation and selection checks; no world execution'
    save(agent.workspace / 'training.json', agent.checkpoint_result('method_content_formation_completed'))
    save(output / 'result.json', result)
    return result


def form_method_from_ordinary_returns(request, output):
    """Replay paid returns through ordinary feedback and its published questions.

    Existing method artifacts keep their invocation identities. The request
    supplies no revision program: Common must recover content sources and the
    old conditional body from the shared method question itself.
    """
    import os
    from scripts import run_state_method_pairing as pairing
    from hegel_machine.cognition.finite_native_author import publish_research_options
    from hegel_machine.cognition.method_conditions import method_condition_features, method_condition_support
    from hegel_machine.cognition.method_feedback import finalize_python_method_feedback
    from hegel_machine.cognition.research_programs import export_research_selection_program
    from hegel_machine.cognition.theory_consolidation import _save

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    request_ref = save(output / 'request.json', request)
    for relative in ('scripts/run_method_condition_cohort.py',
                     'src/hegel_machine/cognition/method_feedback.py',
                     'src/hegel_machine/cognition/method_action_research.py',
                     'src/hegel_machine/cognition/research_programs.py'):
        destination = output / 'source_at_run' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(ROOT / relative, destination)
    registration = read(request['registration_ref'])
    agent, lab, _ = pairing._scene(registration, output / 'workspace')
    before = deepcopy(agent.costs)
    existing = agent.theory_consolidation.setdefault('candidates', {})
    methods = []
    for reference in request['existing_method_refs']:
        candidate = deepcopy(read(reference))
        identifier = candidate['id']
        if identifier in existing and existing[identifier] != candidate:
            raise ValueError('the restored method differs from its original artifact')
        existing[identifier] = candidate
        methods.append({'id': identifier, 'version': candidate['version']})
        # These are the actual frozen executables, not a recompilation with
        # today's ordinary source. No research method is executed here.
        original = Path(reference['file']).parent / 'workspace'
        paths = {candidate['execution']['path']}
        ordinary = candidate.get('selection_plan', {}).get('ordinary_method_path')
        if ordinary:
            paths.add(ordinary)
        for relative in paths:
            source, destination = original / relative, agent.workspace / relative
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise ValueError('a restored method source differs at its original path')
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                clone_or_copy(source, destination)
    owner = agent._open_question({'id': registration['question']['id'] + ':ordinary_method_returns',
        'kind': 'method_use', 'meaning_version': 1, 'answer_type': 'research_method',
        'question': 'Which existing method content should these actual research returns revise?',
        'parent_id': registration['question']['id'], 'candidate_method_refs': methods})
    owner_ref = {'question_id': owner['id'], 'meaning_version': owner.get('meaning_version', 1)}
    for method in methods:
        group = agent.theory_consolidation.setdefault('method_feedback', {}).setdefault(
            method['id'], {'outcome_refs': []})
        group['method_question_ref'] = deepcopy(owner_ref)
    _save(agent)
    result = {'status': 'no_method_content_formed', 'request_ref': request_ref,
        'owner_question_ref': owner_ref, 'returns': [], 'selection_checks': [],
        'new_compute_cost': 0, 'new_predictions': 0, 'new_fits': 0,
        'new_observations': 0, 'model_calls': 0, 'new_scientific_support': False,
        'old_method_retained': True, 'new_method_installed': False,
        'source_costs_recharged': False}

    def consume(item, label):
        actual = read(item['outcome_ref'])
        candidate = read(actual['method_artifact_ref'])
        identity = {'id': candidate['id'], 'version': candidate['version']}
        if identity not in methods or actual['method_ref'] != identity:
            raise ValueError('the original return must retain its installed method identity')
        binding = deepcopy(actual['method_invocation_binding'])
        context = {'profile': read(actual['profile_ref']),
            'source_task_ref': deepcopy(item['source_task_ref']),
            'question_ref': {'question_id': registration['question']['id'],
                'meaning_version': registration['question'].get('meaning_version', 1)},
            'theory_target': deepcopy(registration['question'].get('theory_target', {}))}
        compute_before = agent.costs['compute_cost']
        feedback = finalize_python_method_feedback(agent, candidate, binding, context,
            result_ref=item['outcome_ref'], comparison_ref={**deepcopy(item['outcome_ref']),
                'json_path': [*item['outcome_ref'].get('json_path', []), 'comparison']})
        published = feedback['method_question_refs'][-1]
        question = next(value for value in agent.questions if value['id'] == published['question_id'])
        static_reserve = 2 + 3 * len(request.get('selection_checks', []))
        remaining = (request['analysis_compute_cap'] - static_reserve
            - (agent.costs['compute_cost'] - before['compute_cost']))
        routed = agent._execute('run', {'operator': 'research_route',
            'research_question': deepcopy(question), 'allocation': {'compute': max(0, remaining)}}, lab)
        route_ref = save(output / f'{label}_method_research.json', routed)
        action = routed.get('method_action_research', {})
        receipt = {'source': deepcopy(item), 'invoked_method_ref': identity,
            'feedback_ref': feedback['feedback_ref'], 'question_ref': published,
            'research_route_ref': route_ref, 'status': action.get('status', routed.get('status')),
            'ordinary_returns': len(question['method_use_plan']['ordinary_return_refs']),
            'new_compute_cost': agent.costs['compute_cost'] - compute_before,
            'source_costs_ref': {**deepcopy(item['outcome_ref']),
                'json_path': [*item['outcome_ref'].get('json_path', []), 'costs']}}
        return receipt, action

    proposed = None
    sources = request['ordinary_return_sources']
    for index, item in enumerate(sources):
        receipt, action = consume(item, f'return_{index:02d}')
        result['returns'].append(receipt)
        proposed = next((candidate for candidate in action.get('proposals', {}).get('candidates', [])
            if candidate.get('producer') == 'finite_paid_guard_mature_startup_relation_induction'), proposed)
    if sources:
        result['duplicate_return'], _ = consume(sources[-1], 'duplicate_return')
        result['duplicate_return_reused'] = (
            result['duplicate_return']['new_compute_cost'] == 0
            and result['duplicate_return']['ordinary_returns'] == result['returns'][-1]['ordinary_returns'])
    if proposed is not None:
        result['proposed_method_ref'] = save(output / 'proposed_method.json', proposed)
        plan = deepcopy(proposed['research_selection_plan'])
        old = read(request['verified_method_ref'])
        expected = deepcopy(old['selection_plan'])
        plan.pop('source_refs', None)
        expected.pop('source_refs', None)
        result['same_finite_selection_body'] = (
            export_research_selection_program({**plan, 'ordinary_method_path': 'ordinary.py'})
            == export_research_selection_program({**expected, 'ordinary_method_path': 'ordinary.py'}))
        actual_dependency = _method_dependency(plan, agent.workspace)
        expected_dependency = _method_dependency(expected,
            Path(request['verified_method_ref']['file']).parent / 'workspace')
        result['same_dependency'] = actual_dependency.read_bytes() == expected_dependency.read_bytes()
        result['dependency_refs'] = {'formed': {'file': str(actual_dependency)},
                                    'verified': {'file': str(expected_dependency)}}
        result['verified_method_ref'] = deepcopy(request['verified_method_ref'])
        agent.costs['compute_cost'] += 2
        selection = read(request['condition_selection_ref'])
        for index, item in enumerate(request.get('selection_checks', [])):
            context, apparatus, _ = pairing._scene(read(item['registration_ref']),
                output / f'selection_{index:02d}')
            profile = read(item['profile_ref'])
            memory = {**context.knowledge, 'research_state': context._program_state(), 'program_results': [],
                'method_decision': {'research_options': publish_research_options(context, apparatus.public_spec())},
                'method_condition_features': method_condition_features(profile, plan['condition_feature_paths']),
                'method_condition_support': method_condition_support(profile, selection),
                'method_candidate_pool': deepcopy(profile.get('research', {}).get('candidate_pool'))}
            choices = []
            previous_directory = Path.cwd()
            try:
                os.chdir(context.workspace)
                for selected_plan, dependency in ((plan, actual_dependency), (expected, expected_dependency)):
                    choices.append(_static_method_action(selected_plan, dependency,
                        apparatus.public_spec(), context.observations, memory))
            finally:
                os.chdir(previous_directory)
            agent.costs['compute_cost'] += 3
            native = choices[0].get('arguments', {}).get('native_representation_plan', {})
            result['selection_checks'].append({'source_ref': deepcopy(item),
                'same_action': choices[0] == choices[1],
                'actions_available': all(choice.get('action') and not choice.get('stop') for choice in choices),
                'parent_count': len(native.get('parent_theory_ids', [])),
                'max_candidates': native.get('max_candidates'),
                'startup_parent_priority': native.get('startup_parent_priority', []),
                'formed_action': choices[0], 'verified_action': choices[1]})
        same = (result['same_finite_selection_body'] and result['same_dependency']
                and all(item['same_action'] and item['actions_available']
                for item in result['selection_checks']))
        result['status'] = ('automatic_method_content_matches_verified_body' if same
                            else 'automatic_method_content_requires_separate_execution_evidence')
    result['new_compute_cost'] = agent.costs['compute_cost'] - before['compute_cost']
    result['remaining_analysis_compute'] = request['analysis_compute_cap'] - result['new_compute_cost']
    result['wall_seconds'] = monotonic() - started
    result['formation_cost_scope'] = 'Automatic return projection and method content analysis; static selection only'
    save(agent.workspace / 'training.json', agent.checkpoint_result('ordinary_method_returns_completed'))
    save(output / 'result.json', result)
    return result


def form_method_from_ordinary_session(request, output):
    """Deliver retained returns to a continuing topic through ordinary Common.

    The topic is registered/exported before this replay. Its mission carries
    the original spent allowance; each source retains its actual invoked body.
    Common owns agenda selection, proposal reuse and any validation request.
    """
    from dataclasses import replace
    from scripts import run_state_method_pairing as pairing
    from hegel_machine.cognition.method_feedback import (
        ensure_python_method_topic, export_python_method_topic,
        record_python_method_feedback)
    from hegel_machine.cognition.method_action_research import read_method_reference
    from hegel_machine.cognition.question_agenda import mission_compute_remaining

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    request_ref = save(output / 'request.json', request)
    for relative in ('scripts/run_method_condition_cohort.py',
                     'src/hegel_machine/cognition/method_feedback.py',
                     'src/hegel_machine/cognition/method_action_research.py',
                     'src/hegel_machine/cognition/question_routing.py',
                     'src/hegel_machine/cognition/common_archive_inquiry.py'):
        destination = output / 'source_at_run' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(ROOT / relative, destination)
    registration = read(request['registration_ref'])
    agent, lab, _ = pairing._scene(registration, output / 'workspace')
    before = deepcopy(agent.costs)
    owner = ensure_python_method_topic(agent,
        parent_question_ref={'question_id': registration['question']['id'], 'meaning_version': 1},
        topic_ref=request['method_topic_ref'])
    # This cap is only the already granted remainder, further bounded by the
    # imported mission's cumulative debit. Restoring an episode never funds it.
    agent.config = replace(agent.config,
        compute_budget=before['compute_cost'] + request['analysis_compute_cap'])
    result = {'status': 'no_method_content_formed', 'request_ref': request_ref,
        'owner_question_ref': {'question_id': owner['id'], 'meaning_version': owner.get('meaning_version', 1)},
        'returns': [], 'new_predictions': 0, 'new_fits': 0, 'new_observations': 0,
        'model_calls': 0, 'source_costs_recharged': False,
        'prior_cost_refs': deepcopy(request.get('prior_cost_refs', [])),
        'topic_registration_ref': deepcopy(request.get('topic_registration_ref'))}

    def consume(item, label):
        actual = read(item['outcome_ref'])
        method = actual['method_ref']
        candidate = agent.theory_consolidation['candidates'][method['id']]
        if candidate['version'] != method['version']:
            raise ValueError('ordinary return version differs from the continuing topic')
        if actual.get('method_artifact_ref') and read(actual['method_artifact_ref']) != candidate:
            raise ValueError('ordinary return body differs from its installed original')
        binding = deepcopy(actual['method_invocation_binding'])
        if binding.get('candidate_id') != method['id'] or binding.get('candidate_version') != method['version']:
            raise ValueError('ordinary return must retain its real invocation identity')
        if item.get('source_task_ref'):
            binding['source_task_ref'] = deepcopy(item['source_task_ref'])
        group = agent.theory_consolidation.get('method_feedback', {}).get(method['id'], {})
        previous = [read_method_reference(ref, agent.workspace) for ref in group.get('outcome_refs', [])]
        if not any(value['origin_ref'] == binding['origin_ref'] for value in previous):
            record_python_method_feedback(agent, candidate, binding,
                {'status': actual['status'], 'execution_complete': False},
                {'profile': read(actual['profile_ref']),
                 'question_ref': {'question_id': registration['question']['id'], 'meaning_version': 1},
                 'theory_target': deepcopy(registration['question'].get('theory_target', {}))})
        cost_before = agent.costs['compute_cost']
        feedback = agent._execute('run', {'operator': 'method_action_feedback',
            'candidate_id': method['id'], 'binding': binding,
            'result_ref': item['outcome_ref'],
            'allocation': {'compute': mission_compute_remaining(agent, owner.get('mission', {}))}}, lab)
        feedback_ref = save(output / f'{label}_feedback.json', feedback)
        research = feedback.get('method_research', {})
        routed = research.get('research_result', {})
        action = routed.get('method_action_research', {})
        receipt = {'source': deepcopy(item), 'invoked_method_ref': deepcopy(method),
            'feedback_ref': feedback_ref, 'method_feedback_ref': feedback['feedback_ref'],
            'question_refs': feedback['method_question_refs'],
            'status': action.get('status', research.get('status')),
            'next_action': deepcopy(research.get('next_action')),
            'ordinary_returns': len(owner.get('method_action_returns', [])),
            'new_compute_cost': agent.costs['compute_cost'] - cost_before,
            'source_costs_ref': {**deepcopy(item['outcome_ref']),
                'json_path': [*item['outcome_ref'].get('json_path', []), 'costs']}}
        return receipt, action

    sources = request['ordinary_return_sources']
    for index, item in enumerate(sources):
        receipt, action = consume(item, f'return_{index:02d}')
        result['returns'].append(receipt)
        if action:
            result['status'] = action['status']
            result['latest_method_action'] = deepcopy(action)
        result['method_topic_ref'] = export_python_method_topic(agent, owner)
        save(output / 'result.json', result)
    if sources:
        result['duplicate_return'], _ = consume(sources[-1], 'duplicate_return')
        result['duplicate_return_reused'] = (
            result['duplicate_return']['new_compute_cost'] == 0
            and result['duplicate_return']['ordinary_returns'] == result['returns'][-1]['ordinary_returns'])
    result['method_topic_ref'] = export_python_method_topic(agent, owner)
    result['new_compute_cost'] = agent.costs['compute_cost'] - before['compute_cost']
    result['remaining_analysis_compute'] = mission_compute_remaining(agent, owner.get('mission', {}))
    result['wall_seconds'] = monotonic() - started
    save(agent.workspace / 'training.json', agent.checkpoint_result('ordinary_method_session_complete'))
    save(output / 'result.json', result)
    return result


def run(registration, output):
    """Run a fixed development/validation/successor list once, without retries."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    entry = output / 'source_at_run' / 'scripts' / Path(__file__).name
    entry.parent.mkdir(parents=True)
    clone_or_copy(Path(__file__).resolve(), entry)
    started = monotonic()
    result = {'status': 'running', 'registration_ref': save(output / 'registration.json', registration),
        'sources': [], 'phase_status': {}, 'selected_policy': None,
        'condition_pair_refs': deepcopy(registration.get('prior_pair_refs', [])),
        'costs': {'compute_cost': 0, 'observation_cost': 0, 'measurement_cost': 0,
                  'python_execution_seconds': 0., 'model_calls': 0, 'reported_tokens': 0}}
    latest_feedback = (read(registration['initial_method_feedback_ref'])
                       if registration.get('initial_method_feedback_ref') else None)
    method_topic_ref = deepcopy(registration.get('method_topic_ref')
        or (latest_feedback or {}).get('method_topic_ref'))
    try:
        for role in ('development', 'condition_validation', 'independent_successor'):
            requests = registration['phases'].get(role, [])
            choice = read(latest_feedback['selection_ref']) if latest_feedback else {}
            if role == 'condition_validation' and not choice.get('candidates'):
                result['phase_status'][role] = 'not_run_no_development_condition'
                continue
            bundle = registration['method_pair_ref']
            if role == 'independent_successor':
                frozen = (freeze_successor_policy(bundle, latest_feedback, output)
                          if latest_feedback else None)
                result['selected_policy'] = frozen
                if frozen is None:
                    result['phase_status'][role] = 'not_run_no_supported_method'
                    continue
                bundle = frozen['method_pair_ref']
                save(output / 'result.json', result)
            result['phase_status'][role] = 'running'
            for index, template in enumerate(requests):
                request = deepcopy(template)
                request.update(data_role=role, method_pair_ref=bundle,
                    prior_pair_refs=([] if role == 'independent_successor'
                                     else deepcopy(result['condition_pair_refs'])))
                if method_topic_ref is not None:
                    request['method_topic_ref'] = deepcopy(method_topic_ref)
                name = request.get('source_id', f'{role}_{index:03d}')
                destination = output / name
                actual = source_entry.run(request, destination)
                receipt = source_receipt(actual, destination)
                receipt.update(source_id=name, data_role=role, case=deepcopy(request['case']))
                result['sources'].append(receipt)
                for key, value in receipt['costs'].items():
                    result['costs'][key] += value
                if role != 'independent_successor':
                    result['condition_pair_refs'].extend(receipt['pair_refs'])
                    feedback = receipt['method_feedback']
                    if feedback and feedback.get('selection_ref'):
                        latest_feedback = feedback
                    if feedback and feedback.get('method_topic_ref'):
                        method_topic_ref = deepcopy(feedback['method_topic_ref'])
                result['latest_method_feedback'] = deepcopy(latest_feedback)
                result['method_topic_ref'] = deepcopy(method_topic_ref)
                result['wall_seconds'] = monotonic() - started
                save(output / 'result.json', result)
                print({'source': name, 'role': role, 'status': receipt['status'],
                    'losses': {key: arm['loss'] for key, arm in receipt['arms'].items()},
                    'costs': receipt['costs']}, flush=True)
            result['phase_status'][role] = 'completed_registered_sources'
        result['status'] = 'completed_registered_cohort'
    except Exception as error:
        result.update(status='interrupted_no_retry', error=f'{type(error).__name__}: {error}',
            interrupted_source_costs='not_summarized; retained source artifacts require review')
        raise
    finally:
        result['wall_seconds'] = monotonic() - started
        save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(read({'file': str(args.registration), 'json_path': []}), args.output)
