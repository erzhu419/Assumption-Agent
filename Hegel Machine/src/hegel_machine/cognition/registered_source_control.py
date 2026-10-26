"""Execute a registered retained-source control using ordinary Common actions."""

from copy import deepcopy
import gc
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic, process_time

from .checkpoint_storage import write_json_checkpoint, clone_or_copy, retain_checkpoint_sources
from .experiment_runtime import checkpoint_execution
from .development_revision_restore import restore_development_revision_work
from .registered_source_scene import (load_registered_source_material,
    quote_joint_paid_scores, restore_registered_source_scene)
from .question_routing import resolved_theories
from . import theory_consolidation as consolidation


def _write(path, value):
    write_json_checkpoint(path, value, ensure_ascii=False)


def _validate_registered_policy(registration):
    root = Path(__file__).resolve().parents[3]
    for item in registration['policy']['frozen_policy_files']:
        original = Path(item['file'])
        relative = (Path(item['relative_path']) if 'relative_path' in item
                    else original.relative_to(original.parents[3]))
        if (root / relative).read_bytes() != original.read_bytes():
            raise ValueError(f'registered policy changed: {relative}')


def prepare_registered_control(registration, output):
    """Restore both scenes and exercise saved-fit reuse without a new solve."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _validate_registered_policy(registration)
    material = load_registered_source_material(registration)
    new_ids = {x['id'] for x in registration['arms']['keep']['candidate_versions']} - {
        x['id'] for x in registration['arms']['remove']['candidate_versions']}
    if len(new_ids) != 1:
        raise ValueError('this registered control requires its one frozen object difference')
    new_id = new_ids.pop()
    scenes = {}
    for arm in registration['arms']:
        scene = restore_registered_source_scene(registration, arm, output / arm, material=material)
        agent = scene['agent']
        pool = resolved_theories(agent)
        by_id = {item['id']: item for item in pool}
        components = [by_id[key] for key in scene['source_component_theory_ids']]
        original_key = 'relation_transfer:frozen_recall:native_public_response@1:consolidated:0:relation:revision:0'
        basis = agent.theory_consolidation['executions'][original_key]['request_basis']
        arguments = {key: deepcopy(basis.get(key)) for key in (
            'role_signature', 'output_unit', 'context_ref', 'evidence_checks', 'fitting')}
        arguments.update(library_path=consolidation.LIBRARY_PATH, candidate_id=new_id,
            theory_id=f'frozen_recall:native_public_response@1:{new_id}',
            parent_theory_id=scene['parent']['id'],
            component_theory_ids=scene['source_component_theory_ids'],
            allocation={'compute': registration['budget']['per_consumption_compute'],
                        'comparison_rows': registration['material']['same_paid_rows']})
        receipt = restore_development_revision_work(agent, registration,
            candidate=material['frozen_candidates'][new_id], parent=scene['parent'],
            components=components, transfer_arguments=arguments,
            execution_key='relation_transfer:' + arguments['theory_id'], pool=pool)
        scene.update(development_reuse=receipt, retained_components=components,
            transfer_arguments=arguments)
        agent.knowledge['finite_native_author']['predictor_paths'] = list(scene['models'])
        agent._prepare_program_state()
        _write(output / arm / 'development_reuse_validation.json', receipt)
        scenes[arm] = scene
        print(json.dumps({'stage': 'restoration_accepted', 'arm': arm,
            'theories': len(agent.theory_index), 'models': len(scene['models']),
            'paid_rows': len(agent.rows), 'reuse': receipt}), flush=True)
    _write(output / 'registration.json', registration)
    acceptance = {'status': 'accepted', 'arms': {arm: {
        'restoration': scene['validation'], 'development_reuse': scene['development_reuse']}
        for arm, scene in scenes.items()}, 'new_fits': 0, 'new_scores': 0,
        'new_observations': 0, 'paid_model_calls': 0}
    _write(output / 'acceptance.json', acceptance)
    return scenes, acceptance


def _checkpoint(agent, status):
    with checkpoint_execution():
        agent._prepare_program_state()
        _write(agent.workspace / 'training.json', agent.checkpoint_result(status))


def _export_saved_work(agent, question):
    """Export real committed transfer outputs from an interrupted outer pass."""
    from .finite_native_author import export_revision_candidates

    previous = question['frozen_recall_execution']
    view = deepcopy(previous)
    prefix = f'relation_transfer:frozen_recall:{question["id"]}@{question.get("meaning_version",1)}:'
    work = view['result'].setdefault('work', [])
    for key, execution in agent.theory_consolidation.get('executions', {}).items():
        if key.startswith(prefix):
            identifiers = [item['theory_id'] for item in execution.get('result', {}).get(
                'generated_theories', []) if item.get('theory_id')]
            if identifiers:
                work.append({'candidate_id':key[len(prefix):], 'theory_ids':identifiers})
    # A later continuation replaces an execution's latest result, while its
    # actual earlier outputs remain in the transfer ledger.
    identifiers = list(dict.fromkeys(item.get('theory_id') for item in
        agent.theory_consolidation.get('transfers', []) if isinstance(item.get('theory_id'),str)
        and item['theory_id'].startswith(prefix[len('relation_transfer:'):])) )
    if identifiers:
        work.append({'candidate_id':'retained_transfer_outputs', 'theory_ids':identifiers})
    question['frozen_recall_execution'] = view
    try:
        return export_revision_candidates(agent,question)
    finally:
        question['frozen_recall_execution'] = previous


def _arm_run(scene, registration):
    from .finite_native_author import export_revision_candidates
    from .question_routing import _retained_revision_inputs

    if 'parent_research' in registration:
        return _registered_parent_revision_arm_run(scene, registration)
    agent, lab = scene['agent'], scene['lab']
    started = monotonic()
    before = deepcopy(agent.costs)
    question = next(q for q in agent.questions if q['id'] == 'native_public_response')
    actions = []
    total_config = agent.config
    # The original ordered component rosters and individual object cursors are
    # retained even though the shared comparison bank now also contains70 drafts.
    inputs = _retained_revision_inputs(agent, question)
    # Opaque Python entries belong to the strong comparison pool, while the
    # structural constructor consumes the original ordered Program subset.
    structural = [item for item in inputs['_retained_components'] if item.get('program')]
    for identifier in agent.theory_consolidation['candidates']:
        inputs.setdefault('_retained_execution_inputs', {}).setdefault(identifier, {
            '_retained_parent': scene['parent'], '_retained_components': structural,
            'component_theory_ids': [item['id'] for item in structural]})
    new_id = scene['transfer_arguments']['candidate_id']
    inputs.setdefault('_retained_execution_inputs', {})[new_id] = {
        '_retained_parent': scene['parent'],
        '_retained_components': scene['retained_components'],
        'component_theory_ids': scene['source_component_theory_ids']}
    inputs.update(_retained_parent=scene['parent'])
    args = {key: deepcopy(scene['transfer_arguments'][key]) for key in (
        'library_path', 'parent_theory_id', 'fitting', 'evidence_checks', 'allocation')}
    args.update(inputs)
    n_rows = len(agent.rows)
    reserve_readouts = 2

    def comparison():
        models = list(dict.fromkeys([*scene['models'], *agent.candidate_paths]))
        before_choice = agent.costs['compute_cost']
        result = agent._execute('run', {'operator': 'research_choice', 'models': models,
            'research_question': {'id': question['id']}}, lab)
        actions.append({'action': 'research_choice', 'models_requested': len(models),
            'compute_delta': agent.costs['compute_cost'] - before_choice,
            'feedback': result})
        _checkpoint(agent, 'registered_comparison_in_progress')

    status = 'research_completed'
    try:
        if scene.get('resume_phase') not in {'first_comparison','final_comparison'}:
            agent.config = replace(total_config,
                compute_budget=min(total_config.compute_budget - reserve_readouts,
                    agent.costs['compute_cost'] + scene.get('first_allocation_remaining',
                        registration['budget']['per_consumption_compute'])))
            args['allocation']['compute'] = agent.config.compute_budget-agent.costs['compute_cost']
            recalled = consolidation.recall_frozen_theories(agent, question, args)
            exports = export_revision_candidates(agent, question)
            actions.append({'action': 'frozen_recall', 'feedback': recalled, 'exports': exports})
        else:
            actions.append({'action':'export_committed_interrupted_work',
                            'feedback':_export_saved_work(agent,question)})
        agent.config = replace(total_config, compute_budget=total_config.compute_budget-reserve_readouts)
        comparison()
        # The second registered opportunity first follows the real research
        # route. A zero-cost author request leaves that allowance unconsumed.
        if scene.get('resume_phase') != 'final_comparison':
            score_quote = quote_joint_paid_scores(scene)
            remaining_score = score_quote['unpaid_scoring_compute']
            cap = min(agent.costs['compute_cost'] + registration['budget']['per_consumption_compute'],
                total_config.compute_budget - reserve_readouts - remaining_score)
            if scene.get('resume_phase') == 'last_research_allocation':
                # This recall consumes the previously unused second allocation.
                # PR2 may report its next legal action, without a third allowance.
                cap = agent.costs['compute_cost']
            agent.config = replace(total_config, compute_budget=max(agent.costs['compute_cost'], cap))
            before_route = agent.costs['compute_cost']
            routed = agent._execute('run', {'operator': 'research_route',
                'research_question': {'id': question['id']},
                'library_path': consolidation.LIBRARY_PATH,
                'allocation': {'compute': max(0, cap-agent.costs['compute_cost']),
                               'comparison_rows': n_rows}}, lab)
            actions.append({'action': 'research_route',
                'compute_delta': agent.costs['compute_cost']-before_route,
                'comparison_score_reserve': remaining_score,
                'comparison_score_quote': {key: score_quote[key] for key in (
                    'models', 'paid_rows', 'reused_score_count', 'unpaid_score_count',
                    'unpaid_scoring_compute')},
                'paid_score_index_ref': str(agent.workspace /
                    '_paid_python_scores/_evaluated_model/assessments.json'),
                'feedback': routed})
            unfinished = question.get('frozen_recall_execution', {}).get('result', {})
            if (scene.get('resume_phase') != 'last_research_allocation'
                    and agent.costs['compute_cost'] == before_route
                    and agent.costs['compute_cost'] < cap
                    and routed.get('status') == 'author_work_request'
                    and routed.get('answer_execution', {}).get('parent_revision_completed') is False
                    and routed.get('frozen_recall', {}).get('status') == 'not_needed_answer_available'
                    and unfinished.get('search_completed') is False):
                # Preserve the actual routing failure before consuming only
                # this same second allowance, using each object's saved work.
                _write(agent.workspace / 'registered_actions.json', actions)
                continuation = {**args, 'allocation': {**args['allocation'],
                    'compute': cap-agent.costs['compute_cost']}}
                before_recall = agent.costs['compute_cost']
                recalled = consolidation.recall_frozen_theories(agent, question, continuation)
                exports = export_revision_candidates(agent, question)
                actions.append({'action': 'frozen_recall_second_allocation',
                    'compute_delta': agent.costs['compute_cost']-before_recall,
                    'allocation_compute': continuation['allocation']['compute'],
                    'feedback': recalled, 'exports': exports})
            agent.config = replace(total_config, compute_budget=total_config.compute_budget-reserve_readouts)
            comparison()
    except Exception as error:
        status = 'research_failed'
        actions.append({'action': 'execution_failure', 'exception': type(error).__name__,
                        'reason': str(error), 'traceback': __import__('traceback').format_exc()})
    finally:
        agent.config = total_config
        _write(agent.workspace / 'registered_actions.json', actions)
        _checkpoint(agent, status)
    costs = {key: agent.costs[key]-before[key] for key in before}
    if costs['observation_cost'] or costs['model_calls']:
        raise ValueError('registered source control acquired new paid evidence')
    summary = {'workspace': str(agent.workspace), 'status': status,
        'starting_costs': before, 'continuation_costs': costs,
        'retained_parent': agent._evaluated_summary(), 'wall_seconds': monotonic()-started,
        'theory_count': len(agent.theory_index), 'candidate_model_count': len(agent.candidate_paths),
        'physical_cpus': sorted(__import__('os').sched_getaffinity(0)),
        'actions_ref': str(agent.workspace/'registered_actions.json')}
    _write(agent.workspace / 'arm_summary.json', summary)
    return summary


def _registered_parent_revision_arm_run(scene, registration):
    """Consume exactly the registered fair units on the fixed parent meaning."""
    from .finite_native_author import export_revision_candidates

    agent, lab = scene['agent'], scene['lab']
    contract = registration['parent_research']
    reference = scene['research_question_ref']
    question = next(q for q in agent.questions if q['id'] == reference['question_id'])
    anchor = deepcopy(question['theory_target'])
    arm = scene['registered_arm']
    units = contract['complete_unit_compute_limits'][arm]
    starting, before = monotonic(), deepcopy(agent.costs)
    total_config, actions = agent.config, []
    counts = deepcopy(contract.get('completed_unit_counts_by_group', {}).get(arm, {}))
    roster_base = None
    reserve_readouts = 2
    status = 'research_completed'

    def comparison():
        nonlocal roster_base
        models = list(dict.fromkeys([*scene['models'], *agent.candidate_paths]))
        agent.config = replace(total_config, compute_budget=total_config.compute_budget-reserve_readouts)
        before_choice = agent.costs['compute_cost']
        result = agent._execute('run', {'operator': 'research_choice', 'models': models,
            'research_question': {'id': question['id']}}, lab)
        roster = agent.workspace / 'compared_model_rosters' / f'choice_{len(actions):04d}.json'
        write_json_checkpoint(roster, models, base=roster_base, ensure_ascii=False)
        roster_base = roster
        actions.append({'action': 'research_choice', 'models_requested': len(models),
            'research_question_ref': deepcopy(reference),
            'compute_delta': agent.costs['compute_cost']-before_choice,
            'status': result.get('status'), 'reused_score_count': result.get('reused_score_count'),
            'new_scored_count': result.get('new_scored_count'),
            'models_completed': result.get('models_completed', len(models)),
            'truncated': result.get('truncated', False), 'reason': result.get('reason'),
            'retained_parent': agent._evaluated_summary(),
            'complete_model_roster_ref': {'file': str(roster), 'json_path': []},
            'complete_paid_assessment_index_ref': {'file': str(agent.workspace /
                '_paid_python_scores/_evaluated_model/assessments.json'), 'json_path': []}})
        _write(agent.workspace / 'registered_actions.json', actions)
        _checkpoint(agent, 'registered_parent_revision_comparison')
        return not result.get('truncated', False)

    try:
        # Reuse every paid strong-pool score before the first finite unit.
        initial_comparison_complete = comparison()
        if not initial_comparison_complete:
            status = 'comparison_incomplete_python_budget'
        for index, unit in enumerate(units if initial_comparison_complete else []):
            if question['theory_target'] != anchor or question.get('meaning_version', 1) != reference['meaning_version']:
                raise ValueError('registered parent revision was rebound during execution')
            owner = unit['candidate_id']
            if counts.get(owner, 0) >= contract['rounds']:
                raise ValueError('registered complete-unit group allowance was already consumed')
            score = quote_joint_paid_scores(scene)
            remaining = total_config.compute_budget-reserve_readouts-agent.costs['compute_cost']
            if remaining < unit['compute'] + score['unpaid_scoring_compute']:
                actions.append({'action': 'complete_unit_unfunded', 'unit_index': index,
                    'registered_unit': deepcopy(unit), 'remaining_compute': remaining,
                    'unpaid_score_reserve': score['unpaid_scoring_compute']})
                status = 'research_budget_incomplete'
                break
            agent.config = replace(total_config, compute_budget=agent.costs['compute_cost']+unit['compute'])
            args = {**scene['registered_revision_inputs'],
                'allocation': {'compute': unit['compute'], 'comparison_rows': len(agent.rows)}}
            before_unit = agent.costs['compute_cost']
            # The finite fixed-parent plan already names its research duty.
            # Recall consumes the original fair cursor without re-expanding
            # every native forecast to generate a new agenda on every unit.
            result = consolidation.recall_frozen_theories(agent, question, args)
            exports = export_revision_candidates(agent, question)
            completed = [visit for visit in result.get('complete_research_visits', [])
                         if visit.get('status') == 'complete_visit_executed']
            actions.append({'action': 'frozen_recall_on_registered_parent_duty', 'unit_index': index,
                'registered_unit': deepcopy(unit), 'research_question_ref': deepcopy(reference),
                'compute_delta': agent.costs['compute_cost']-before_unit,
                'feedback': result, 'exports': exports})
            _write(agent.workspace / 'registered_actions.json', actions)
            if not comparison():
                status = 'comparison_incomplete_python_budget'
                break
            if len(completed) != 1:
                status = 'registered_unit_not_completed'
                break
            actual = completed[0]
            if (actual['candidate_id'] != owner or actual['search_position'] != unit['search_position']
                    or counts.get(actual['candidate_id'], 0) >= contract['rounds']):
                raise ValueError('actual complete unit differs from the registered persistent-order plan')
            counts[actual['candidate_id']] = counts.get(actual['candidate_id'], 0) + 1
        else:
            if initial_comparison_complete:
                status = 'registered_units_completed'
    except Exception as error:
        status = 'research_failed'
        actions.append({'action': 'execution_failure', 'exception': type(error).__name__,
            'reason': str(error), 'traceback': __import__('traceback').format_exc()})
    finally:
        agent.config = total_config
        _write(agent.workspace / 'registered_actions.json', actions)
        _checkpoint(agent, status)
    costs = {key: agent.costs[key]-before[key] for key in before}
    if costs['observation_cost'] or costs['model_calls']:
        raise ValueError('registered parent control acquired new paid evidence')
    summary = {'workspace': str(agent.workspace), 'status': status,
        'starting_costs': before, 'continuation_costs': costs,
        'research_question_ref': deepcopy(reference), 'fixed_parent_theory_ref': deepcopy(
            contract['parent_theory_ref']), 'completed_units_by_group': counts,
        'registered_unit_count': len(units), 'retained_parent': agent._evaluated_summary(),
        'wall_seconds': monotonic()-starting, 'theory_count': len(agent.theory_index),
        'candidate_model_count': len(agent.candidate_paths),
        'physical_cpus': sorted(__import__('os').sched_getaffinity(0)),
        'actions_ref': str(agent.workspace / 'registered_actions.json')}
    _write(agent.workspace / 'arm_summary.json', summary)
    return summary


def _registered_prediction_panels(registration, scenes, output):
    """Use the registered readouts, including those retained by a closed scene."""
    from .research_episodes import load_source_values

    if 'measurement' in registration:
        reference = registration['measurement']['public_prediction_bindings_ref']
        return load_source_values([reference], output)[0]['value']
    panels = []
    for scene in scenes.values():
        question = next(q for q in scene['agent'].questions
                        if q['id'] == 'native_public_response')
        panels.append(question['resolution_experiments'])
    if len(panels[0]) != 2 or any(panel != panels[0] for panel in panels[1:]):
        raise ValueError('closed-scene arms must retain the same two public readouts')
    return deepcopy(panels[0])


def _registered_cost_summary(registration, output, results):
    """Carry the retained source/failure account once, separately from this run."""
    costs = registration['costs']
    if 'old_sources_and_all_failures_ref' not in costs:
        return {'common_source_and_development_compute_once': costs[
            'two_source_account_carry_once']['known_source_compute_total'] + costs[
            'new_induction_and_development_compute_once']}
    from .research_episodes import load_source_values

    reference = costs['old_sources_and_all_failures_ref']
    prior = load_source_values([{'file': reference, 'json_path': []}], output)[0]['value']
    induction = deepcopy(costs['new_mainline_induction_once'])
    acceptance_compute = costs.get('restoration_acceptance_compute_once', 0)
    engineering = deepcopy(costs.get('retained_engineering_diagnostics_once', []))
    engineering_compute = sum(item['compute_cost'] for item in engineering)
    compute = induction['compute_cost'] + sum(
        row['continuation_costs']['compute_cost'] for row in results)
    python_seconds = induction['python_execution_seconds'] + sum(
        row['continuation_costs']['python_execution_seconds'] for row in results)
    return {'prior_costs_ref': reference,
        'prior_known_scientific_compute_once': prior['known_scientific_compute_including_sources_once'],
        'prior_known_compute_including_engineering_once': prior['known_compute_including_engineering_acceptance'],
        'new_mainline_induction_once': induction,
        'restoration_acceptance_compute_once': acceptance_compute,
        'retained_engineering_diagnostics_once': engineering,
        'retained_engineering_diagnostics_compute_once': engineering_compute,
        'common_source_and_development_compute_once': prior[
            'known_source_and_development_compute_once'] + induction['compute_cost'],
        'known_scientific_compute_including_sources_once': prior[
            'known_scientific_compute_including_sources_once'] + compute,
        'known_compute_including_engineering_acceptance': prior[
            'known_compute_including_engineering_acceptance'] + compute + acceptance_compute
            + engineering_compute,
        'known_billed_Python_seconds': prior['known_billed_Python_seconds'] + python_seconds,
        'other_D3_costs': deepcopy(prior['other_D3_costs'])}


def run_registered_control(registration, output, *, workers=8, accepted_scene=None):
    """Consume the admitted RAM scenes; retain only compact parent metadata."""
    from .experiment_runtime import (run_process_trials, physical_cpu_groups,
        python_execution_deadline, PythonExecutionDeadline)
    from .code_agent import _copy_model_context, _closed_temporal_export
    from runpy import run_path
    from .question_queries import evaluate_on_model

    started = monotonic()
    output = Path(output).resolve()
    if 'parent_research' in registration:
        if registration['budget']['additional_Python_seconds_per_arm'] <= 0:
            raise ValueError('remaining parent-revision plan has no authorized Python execution budget')
        _validate_registered_policy(registration)
    root = Path(__file__).resolve().parents[3]
    if 'parent_research' in registration:
        source_files = [(Path(item['file']), Path(item['relative_path']))
                        for item in registration['policy']['frozen_policy_files']]
    else:
        source_files = [(path, path.relative_to(root)) for path in [
            *root.joinpath('src/hegel_machine').rglob('*.py'),
            *root.joinpath('scripts').glob('*.py')]]
    for path, relative in source_files:
        destination = output / 'source_at_run' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(path, destination)
    if accepted_scene is not None:
        scenes, acceptance = accepted_scene
    elif 'parent_research' in registration:
        from .registered_source_scene import prepare_completed_source_control
        scenes, acceptance = prepare_completed_source_control(registration, output)
    else:
        scenes, acceptance = prepare_registered_control(registration, output)
    retained_metadata = {}
    for arm, scene in scenes.items():
        scene['registered_arm'] = arm
        question = next(q for q in scene['agent'].questions if q['id']=='native_public_response')
        retained_metadata[arm] = {
            'prior_continuation_costs': deepcopy(scene.get('prior_continuation_costs')),
            'original_starting_costs': deepcopy(scene.get('original_starting_costs')),
            'prediction_context': deepcopy({'context_ref': question['context_ref'],
                'output_unit': question['theory_target']['output_unit']})}
    del scene, question
    experiments = _registered_prediction_panels(registration, scenes, output)
    cpu_groups = physical_cpu_groups(len(scenes), min(workers,8))
    scene_groups = {arm: index % len(cpu_groups) for index, arm in enumerate(scenes)}
    _write(output/'protocol.json', {'status': 'running',
        'scope': registration['scope'], 'workers': min(workers,8),
        'physical_cpu_groups': cpu_groups,
        'registration_ref': str(output/'registration.json'),
        'acceptance_ref': str(output/'acceptance.json'),
        'common_repairs_are_PR3_gain': False})
    def run_arm(arm):
        scene = scenes.pop(arm)
        # Fork inherited both admitted scenes. Keep only this worker's own
        # live research object before GC or scientific mutations dirty pages.
        for other in list(scenes):
            if scene_groups[other] != scene_groups[arm]:
                del scenes[other]
        # Parent inputs were prepared during admission. Their old result view
        # is no longer read by the runner; the agent owns the recovery state.
        scene.pop('previous', None)
        gc.collect()
        return {'arm': arm, **_arm_run(scene, registration)}

    results = run_process_trials(run_arm, list(scenes), min(workers,8),
        release_parent=scenes.clear)
    for row in results:
        retained = retained_metadata[row['arm']]
        if retained['prior_continuation_costs']:
            row['this_resume_costs'] = deepcopy(row['continuation_costs'])
            row['continuation_costs'] = {key: row['continuation_costs'][key]+value
                for key, value in retained['prior_continuation_costs'].items()}
            row['prior_failed_continuation_costs'] = retained['prior_continuation_costs']
            row['starting_costs'] = retained['original_starting_costs']
    # Freeze both actual winners before either arm's published readouts.
    frozen = {}
    for row in results:
        workspace = Path(row['workspace'])
        parent = row['retained_parent']
        if parent is None:
            raise ValueError('ordinary comparison did not retain its existing parent')
        inner = _copy_model_context(workspace, parent['model'], output/'frozen_parents'/row['arm'])
        frozen[row['arm']] = {'directory': str(output/'frozen_parents'/row['arm']), 'model': inner}
    _write(output/'frozen_parents.json', frozen)
    predictions = {}
    for row in results:
        arm = row['arm']
        context = frozen[arm]
        remaining_python = (registration['budget']['additional_Python_seconds_per_arm']
                            - row['continuation_costs']['python_execution_seconds'])
        answers, python_seconds = [], 0.
        unknown_reason = 'saved_fitted_prediction_required'
        if remaining_python <= 0:
            unknown_reason = 'python_execution_budget_exhausted'
            row['status'] = 'Python_budget_exhausted'
        elif _closed_temporal_export(Path(context['directory']), context['model']):
            began = monotonic()
            try:
                with python_execution_deadline(remaining_python):
                    model = run_path(str(Path(context['directory'])/context['model']))
                    predict = model['predict']
                    for experiment in experiments:
                        answers.append(evaluate_on_model(experiment['query'], predict,
                            context=retained_metadata[arm]['prediction_context'],
                            compute_budget=1))
            except PythonExecutionDeadline:
                unknown_reason = 'python_execution_budget_exhausted'
                row['status'] = 'Python_budget_exhausted'
            finally:
                python_seconds = monotonic()-began
        answers.extend({'status': 'UNKNOWN', 'reason': unknown_reason, 'compute_cost': 0}
                       for _ in range(len(experiments)-len(answers)))
        fee = sum(answer['compute_cost'] for answer in answers)
        row['continuation_costs']['compute_cost'] += fee
        row['continuation_costs']['python_execution_seconds'] += python_seconds
        if 'this_resume_costs' in row:
            row['this_resume_costs']['compute_cost'] += fee
            row['this_resume_costs']['python_execution_seconds'] += python_seconds
        predictions[arm] = {'answers': answers, 'frozen_parent_ref': context, 'compute_cost': fee}
        if row['continuation_costs']['compute_cost'] > registration['budget']['additional_compute_per_arm']:
            raise ValueError('registered arm exceeded its declared compute allowance')
        if row['continuation_costs']['python_execution_seconds'] > registration['budget'][
                'additional_Python_seconds_per_arm']:
            row['status'] = 'Python_budget_exhausted'
        _write(Path(row['workspace'])/'final_costs.json', row['continuation_costs'])
    _write(output/'frozen_predictions.json', predictions)
    # Restoration checks use their own SearchBudget rather than either arm's
    # agent costs. Their actual receipt replaces the registered estimate once.
    cost_registration = registration
    if 'metadata_acceptance_compute' in acceptance:
        cost_registration = {**registration, 'costs': {**registration['costs'],
            'restoration_acceptance_compute_once': acceptance['metadata_acceptance_compute']}}
    summary = {'status':('completed' if all(row['status'] in {'research_completed', 'registered_units_completed'}
        for row in results) else 'failed_or_partial'), 'scope':registration['scope'], 'arms': results,
        'prediction_ref':str(output/'frozen_predictions.json'),
        'wall_seconds':monotonic()-started,
        **_registered_cost_summary(cost_registration, output, results),
        'independent_future_accuracy':'UNTESTED', 'common_repairs_are_PR3_gain':False}
    _write(output/'summary.json', summary)
    return summary


def resume_registered_control(registration, prior, output, *, workers=8):
    from .registered_source_scene import resume_registered_source_scene

    material = load_registered_source_material(registration)
    scenes = {arm: resume_registered_source_scene(registration, arm,
        Path(prior)/arm, Path(output)/arm, material=material) for arm in registration['arms']}
    acceptance = {'status':'accepted_resume_with_original_500k_caps',
        'prior_failure_ref':str(Path(prior)/'summary.json'),
        'arms':{arm:scene['validation'] for arm,scene in scenes.items()}}
    _write(Path(output)/'acceptance.json',acceptance)
    _write(Path(output)/'registration.json',registration)
    return run_registered_control(registration,output,workers=workers,
        accepted_scene=(scenes,acceptance))


def complete_registered_research_allocation(registration, prior, original_execution_ref,
                                          output, *, workers=8):
    """Use the unspent second allowance after exact historical-version repair."""
    from .registered_source_scene import (
        resume_registered_source_scene, pin_registered_development_work)
    from .research_episodes import load_source_values

    material = load_registered_source_material(registration)
    original = load_source_values([original_execution_ref], output)[0]['value']
    scenes, receipts = {}, {}
    for arm in registration['arms']:
        # The earlier second route returned an author request without native work.
        actions = json.loads((Path(prior).parent/'settled'/arm/'registered_actions.json').read_text())
        routes = [item for item in actions if item['action'] == 'research_route']
        if len(routes) != 1 or routes[0]['compute_delta'] != 0:
            raise ValueError('the second registered allowance was already consumed')
        scene = resume_registered_source_scene(registration, arm,
            Path(prior)/arm, Path(output)/arm, material=material)
        pinned = pin_registered_development_work(scene, registration, original)
        scene.update(resume_phase='last_research_allocation',
            first_allocation_remaining=registration['budget']['per_consumption_compute'])
        receipts[arm] = {'restoration': scene['validation'], 'pinned_development': pinned,
            'unused_second_route_ref':str(Path(prior).parent/'settled'/arm/'registered_actions.json'),
            'second_allocation_compute':registration['budget']['per_consumption_compute'],
            'prior_expenses_preserved':True, 'third_native_allocation':False}
        scenes[arm] = scene
        print(json.dumps({'stage':'exact_cursor_and_versions_accepted', 'arm':arm,
                          'receipt':receipts[arm]}),flush=True)
    acceptance = {'status':'accepted_exact_versions_second_registered_allocation',
        'prior_result_ref':str(Path(prior)/'summary.json'),
        'original_execution_ref':original_execution_ref, 'arms':receipts,
        'common_repair_is_PR3_gain':False, 'new_fits':0,'new_scores':0}
    _write(Path(output)/'acceptance.json',acceptance)
    _write(Path(output)/'registration.json',registration)
    return run_registered_control(registration,output,workers=workers,
        accepted_scene=(scenes,acceptance))


def _frozen_parent_row_groups(rows, paid_rows):
    """Compare exact public inputs, independently of either model or labels."""
    def public_input(row):
        features = row.features if hasattr(row, 'features') else row['features']
        history = row.history if hasattr(row, 'history') else row.get('history', [])
        return tuple(sorted(features.items())), tuple(tuple(sorted(past.items())) for past in history)

    paid_inputs = {public_input(row) for row in paid_rows}
    trajectories, regions = [], {'history_lt24': [], 'history_ge24': [],
                                 'seen_public_inputs': [], 'unseen_public_inputs': []}
    for index, row in enumerate(rows):
        history = row.history if hasattr(row, 'history') else row.get('history', [])
        if not history:
            trajectories.append([])
        trajectories[-1].append(index)
        regions['history_lt24' if len(history) < 24 else 'history_ge24'].append(index)
        regions['seen_public_inputs' if public_input(row) in paid_inputs
                else 'unseen_public_inputs'].append(index)
    return trajectories, regions


def prepare_registered_parent_recursion(registration, output):
    """Cold restore the actual failed new-parent revision without paid work.

    The failed caller saved a complete result but omitted research_state.json.
    Reconstruct its known fields in a new workspace through Common's own state
    writer, then use the ordinary practice restoration. Missing internal fits
    and a search cursor are not reconstructed.
    """
    from .anonymous_history_lab import AnonymousHistoryLab
    from .code_agent import _copy_execution_context
    from .common_archive_inquiry import CommonArchiveInquiryAgent
    from .models import Observation
    from .online_inquiry_llm import InquiryLLMConfig
    from .research_episodes import load_source_values
    from .language import Program
    from .temporal_operator_agent import export_temporal_program

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    read = lambda reference: load_source_values([reference], output)[0]['value']
    previous = read(registration['source_result_ref'])
    initial = read(registration['source_initial_evaluation_ref'])
    source = Path(registration['source_workspace']).resolve()
    native_question = next(q for q in previous['online_questions']
                           if q['id'] == registration['question_id'])
    model = previous['evaluated_model']['source_model']
    paid_score = previous['research_candidates'][model]['current_evaluation']
    if (initial.get('returncode') != 0 or paid_score.get('returncode') != 0
            or initial['mse'] != paid_score['mse']
            or previous['evaluated_model']['mse'] != initial['mse']):
        raise ValueError('the actual retained initial parent assessment is inconsistent')
    if (native_question.get('kind') != 'program_revision'
            or native_question.get('origin', {}).get('row_indices')
                != registration['counterexample_rows']):
        raise ValueError('the saved revision must bind its actual measured counterexamples')
    measured = read(native_question['origin']['source_ref'])
    if (measured.get('finite_status') != 'observed_false'
            or measured.get('evidence') != registration['counterexample_rows']):
        raise ValueError('the retained revision must use its actual returned counterevidence')
    plan = native_question['native_representation_plan']
    if plan['parent_theory_ids'] != [registration['parent_theory_id']]:
        raise ValueError('the saved revision does not bind the registered actual parent')
    source_costs = deepcopy(previous['costs'])
    retained_parent = previous['theory_index'][registration['parent_theory_id']]['program']
    parent_program = Program.from_dict(retained_parent)
    if (source / model).read_text() != export_temporal_program(parent_program):
        raise ValueError('the actual parent source does not execute its registered Program')
    for reference in registration['retained_research_program_refs']:
        candidate = read(reference)
        if candidate['program'] != retained_parent or not candidate.get('research_program'):
            raise ValueError('the retained procedure must belong to the actual saved parent')
    carried = deepcopy(registration.get('starting_costs', source_costs))
    if any(carried[key] < value for key, value in source_costs.items()):
        raise ValueError('the carried account cannot remove an already reported cost')
    limits = registration['limits']
    config = InquiryLLMConfig(compute_budget=limits['compute'],
        execution_seconds=limits['python_seconds'], observation_budget=0,
        max_model_calls=0, confirmation_mse_threshold=1e-10)
    knowledge = deepcopy(previous['outgoing_knowledge'])
    bridge_workspace = output / 'retained_scene'
    _copy_execution_context(source, bridge_workspace, [model])
    # Preserve the independently writable retained predictor and fitting view.
    _copy_execution_context(source / '_evaluated_model',
        bridge_workspace / '_evaluated_model',
        [str(Path(previous['evaluated_model']['model']).relative_to('_evaluated_model'))])
    bridge = CommonArchiveInquiryAgent(None, bridge_workspace, arm='hegel',
        source_task=False, knowledge=knowledge, config=config)
    fields = {'questions': 'online_questions', 'observations': 'paid_observations',
        'revisions': 'revision_proposals', 'events': 'events', 'model': 'model',
        'inherited_files': 'inherited_files', 'method_results': 'method_results',
        'program_contexts': 'program_contexts', 'experiment_comparisons': 'experiment_comparisons',
        'observable_models': 'observable_models', 'conditional_models': 'conditional_models',
        'model_dependencies': 'model_dependencies', 'research_candidates': 'research_candidates',
        'research_input_bindings': 'research_input_bindings', 'relation_research': 'relation_research',
        'theory_index': 'theory_index', 'research_mission': 'research_mission',
        'research_agenda': 'research_agenda', 'llm_inquiries': 'llm_inquiries',
        '_inquiry_stack': 'inquiry_stack'}
    for attribute, field in fields.items():
        setattr(bridge, attribute, deepcopy(previous[field]))
    bridge.inherited_files = set(bridge.inherited_files)
    bridge.costs = carried
    bridge.rows = [Observation(dict(row['features']), row['target'], tuple(row['history']))
                   for record in bridge.observations for row in record['rows']]
    if (len(bridge.rows) != registration['frozen_split']['training_rows']
            + registration['frozen_split']['validation_rows']):
        raise ValueError('the saved material differs from the registered frozen split')
    bridge.evaluated_model = {
        'model': str(Path(previous['evaluated_model']['model']).relative_to('_evaluated_model')),
        'source_model': model, 'current_mse': initial['mse'], 'evaluated_rows': len(bridge.rows),
        **{key: deepcopy(previous['evaluated_model'][key]) for key in
           ('evaluation_scope', 'fitting_view', 'later_readings', 'selection')
           if key in previous['evaluated_model']}}
    bridge._active_question_id = native_question['id']
    # The failed route returned no exports or operator result. Its full failure
    # remains at the existing canonical reference on the question.
    studies = native_question.get('native_representation_research', [])
    if any(item.get('candidate_count', len(item.get('candidates', []))) for item in studies):
        raise ValueError('this omitted-state restoration requires the actual no-export failure')
    bridge.candidate_paths, bridge.operator_results = [], []
    bridge._write_evidence()
    bridge._prepare_program_state()
    retained = bridge.checkpoint_result('restored_failed_parent_scene')
    retain_checkpoint_sources(output / 'retained_scene_result.json',
                              registration['source_result_ref']['file'])
    _write(output / 'retained_scene_result.json', retained)
    state = bridge._program_state()
    laboratory = AnonymousHistoryLab(**registration['environment'])
    agent = CommonArchiveInquiryAgent(None, output / 'workspace', arm='hegel',
        source_task=False, knowledge=knowledge, config=config)
    restored = agent.restore_practice_research(retained, bridge_workspace, laboratory,
        confirmation_seed=registration.get('confirmation_seed', 9173), retained_state=state)
    training, held, split = agent._split({'validation_records':
        registration['frozen_split']['validation_records']})
    if (len(training) != registration['frozen_split']['training_rows']
            or len(held) != registration['frozen_split']['validation_rows']):
        raise ValueError('cold restoration changed the frozen fitting split')
    models = read(registration['retained_roster_ref'])
    if len(models) != registration['complete_old_pool_count']:
        raise ValueError('the complete retained model roster changed')
    if any(not Path(path).is_absolute() or not Path(path).is_file() for path in models):
        raise ValueError('retain each strong-pool model at its actual execution context')
    # Retain prior scopes exactly. In particular, a609-row assessment is not a
    #613-row assessment; the ordinary scorer still checks the original basis.
    index = read(registration['retained_score_index_ref'])
    _write(agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json', index)
    question = next(q for q in agent.questions if q['id'] == registration['question_id'])
    questions_preserved = agent.questions == previous['online_questions']
    question['native_representation_plan'] = {
        **deepcopy(question['native_representation_plan']),
        'validation_records': registration['frozen_split']['validation_records'],
        'compute_limit': limits['native_unit_compute'],
        'max_candidates': limits['max_candidates'],
        'retained_research_program_refs': deepcopy(registration['retained_research_program_refs'])}
    acceptance = {'status': 'actual_new_parent_cold_restore_accepted_without_scientific_execution',
        'source_result_ref': deepcopy(registration['source_result_ref']),
        'source_reported_costs': source_costs, 'carried_costs': deepcopy(agent.costs),
        'python_cost_basis': registration.get('python_cost_basis'),
        'restoration': restored, 'rows': len(agent.rows), 'records': len(agent.observations),
        'frozen_split': split, 'parent_source_model': model,
        'parent_mse': agent._evaluated_summary()['mse'],
        'actual_parent_nodes': parent_program.size,
        'measured_answer_ref': deepcopy(native_question['origin']['source_ref']),
        'measured_status': measured['finite_status'],
        'initial_parent_score_reused_without_evaluation': True,
        'retained_research_program_refs': deepcopy(registration['retained_research_program_refs']),
        'condition_negative_ref': deepcopy(registration['condition_negative_ref']),
        'complete_old_pool_count': len(models), 'retained_assessment_refs': len(index['assessments']),
        'retained_score_index_ref': deepcopy(registration['retained_score_index_ref']),
        'source_parent_bytes_equal': (source / model).read_bytes()
            == (agent.workspace / model).read_bytes(),
        'source_questions_preserved': questions_preserved,
        'source_failure_retained': bool(studies),
        'lost_intermediate_fits_recovered': False, 'old_195_or65_cursor_reused': False,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'provider_calls': 0}
    if not acceptance['source_parent_bytes_equal'] or not acceptance['source_questions_preserved']:
        raise ValueError('cold restoration changed the original parent or question record')
    _write(output / 'restoration_acceptance.json', acceptance)
    agent._prepare_program_state()
    retain_checkpoint_sources(output / 'prepared_state.json',
                              output / 'retained_scene_result.json')
    _write(output / 'prepared_state.json', agent.checkpoint_result('prepared_not_executed'))
    return {'agent': agent, 'laboratory': laboratory, 'models': models, 'question': question,
        'acceptance': acceptance, 'starting_costs': deepcopy(agent.costs), 'output': output}


def _comparison_external_python_timing(comparison):
    """Count actual fresh executor calls, retaining unknown historical meters."""
    measured, unknown, seen = 0., 0, set()
    pending = list(comparison.get('evaluations', []))
    while pending:
        evaluation = pending.pop()
        if isinstance(evaluation.get('retained_comparison'), dict):
            pending.append(evaluation['retained_comparison'])
        reuse = evaluation.get('paid_score_reuse', {})
        if reuse.get('status') == 'reused':
            continue
        reference = reuse.get('execution_basis_ref') or reuse.get('assessment_ref')
        if reference:
            key = reference['file']
            if key in seen:
                continue
            seen.add(key)
        value = evaluation.get('external_python_wall_seconds')
        if value is None:
            unknown += 1
        else:
            measured += value
    incomplete = bool(comparison.get('truncated'))
    return {'external_python_wall_seconds': measured if not unknown and not incomplete else None,
            'measured_external_python_wall_seconds': measured,
            'unmeasured_execution_count': unknown,
            'comparison_incomplete': incomplete}


def run_registered_parent_recursion(registration, output, *, accepted_scene=None):
    """Execute an admitted actual-parent revision through Common's old actions."""
    if registration.get('execution_admitted') is not True:
        raise ValueError('the complete recursive budget is not admitted for execution')
    from .paid_score_history import retain_comparison_history
    from .language import Program
    from .representation_inquiry import load_native_representation_result

    started = monotonic()
    scene = accepted_scene or prepare_registered_parent_recursion(registration, output)
    agent, laboratory, question = scene['agent'], scene['laboratory'], scene['question']
    timing = {'preparation_wall_seconds': monotonic() - started,
              'accepted_scene_reused': accepted_scene is not None,
              'native_route': {}, 'strong_pool_comparison': {}, 'frozen_predictions': {}}
    initial_parent = agent._evaluated_summary()
    result = {'status': 'started', 'source_result_ref': registration['source_result_ref'],
        'source_cost_refs': registration['prior_cost_refs'],
        'starting_costs': scene['starting_costs'], 'initial_parent': initial_parent,
        'original_world_threshold': 1e-10, 'new_observations': 0, 'provider_calls': 0,
        'timing': timing}
    try:
        native_started, native_python = monotonic(), agent.costs['python_execution_seconds']
        try:
            routing = agent._execute('run', {'operator': 'research_route',
                'research_question': {'id': question['id']}}, laboratory)
        finally:
            timing['native_route'].update(wall_seconds=monotonic() - native_started,
                python_charge_seconds=agent.costs['python_execution_seconds'] - native_python)
        _write(scene['output'] / 'routing.json', routing)
        native = load_native_representation_result(routing['native_representation_ref'], agent.workspace)
        timing['native_route']['research_body_host_seconds'] = native.get(
            'native_host_seconds_before_persistence')
        before_program = Program.from_dict(agent.theory_index[registration['parent_theory_id']]['program'])
        descendants = {item['model']: Program.from_dict(candidate['program']) for item, candidate
                       in zip(routing.get('exported_models', []), native.get('candidates', []))}
        models = list(dict.fromkeys([*scene['models'], *agent.candidate_paths]))
        compare_started, compare_python = monotonic(), agent.costs['python_execution_seconds']
        try:
            comparison = agent._execute('run', {'operator': 'research_choice',
                'research_question': {'id': question['id']}, 'models': models}, laboratory)
        finally:
            timing['strong_pool_comparison'].update(wall_seconds=monotonic() - compare_started,
                python_charge_seconds=agent.costs['python_execution_seconds'] - compare_python)
        timing['strong_pool_comparison'].update(_comparison_external_python_timing(comparison))
        external = timing['strong_pool_comparison']['external_python_wall_seconds']
        timing['strong_pool_comparison']['charged_non_executor_seconds'] = (
            None if external is None else timing['strong_pool_comparison']['python_charge_seconds'] - external)
        _write(scene['output'] / 'complete_comparison.json', retain_comparison_history(comparison))
        complete = (not comparison.get('truncated')
                    and comparison.get('models_requested') == comparison.get('models_completed'))
        final = agent._evaluated_summary()
        descendant = descendants.get(final['source_model']) if final else None
        result.update(status='completed' if complete else 'comparison_incomplete',
            native_result_ref=routing.get('native_representation_ref'),
            exported_models=routing.get('exported_models', []), final_parent=final,
            comparison_requested=comparison.get('models_requested'),
            comparison_completed=comparison.get('models_completed'),
            reused_score_count=comparison.get('reused_score_count'),
            new_scored_count=comparison.get('new_scored_count'),
            parent_changed=bool(final and final['source_model'] != initial_parent['source_model']),
            actual_recursive_revision_completed=bool(complete and descendant is not None
                and descendant != before_program
                and native['parent_theory_ids'] == [registration['parent_theory_id']]
                and native['origin']['row_indices'] == registration['counterexample_rows']),
            experiment_selection='UNTESTED_no_purchase_in_this_stage',
            independent_future_prediction='UNTESTED_exposed_development_material')
        if complete and final:
            from scripts.run_anonymous_revision_inquiry import evaluate_terminal_model
            from .code_agent import _closed_temporal_export, python_model_context

            forecasts = {}
            for name, model in [('before', initial_parent['source_model']), ('after', final['model'])]:
                context, inner = python_model_context(agent.workspace, model)
                if not _closed_temporal_export(context, inner):
                    forecasts[name] = {'status': 'not_executed_winner_is_not_a_closed_frozen_export'}
                    continue
                if (agent._remaining_compute() < len(agent.rows)
                        or agent.costs['python_execution_seconds'] >= agent.config.execution_seconds):
                    forecasts[name] = {'status': 'not_executed_remaining_budget_insufficient'}
                    continue
                agent._charge(len(agent.rows))
                assessment = evaluate_terminal_model(agent.workspace, model, agent.rows,
                    timeout_seconds=agent.config.execution_seconds
                        - agent.costs['python_execution_seconds'])
                agent.costs['python_execution_seconds'] += assessment['python_execution_seconds']
                forecasts[name] = assessment
                timing['frozen_predictions'][name] = {
                    'python_charge_seconds': assessment['python_execution_seconds'],
                    'external_python_wall_seconds': assessment.get('external_python_wall_seconds')}
            _write(scene['output'] / 'frozen_development_predictions.json', forecasts)
            result['frozen_development_prediction_ref'] = {
                'file': str(scene['output'] / 'frozen_development_predictions.json'), 'json_path': []}
    except Exception as error:
        result.update(status='failed_retained_no_retry',
            error=f'{type(error).__name__}: {error}', actual_recursive_revision_completed=False)
    finally:
        result.update(costs=deepcopy(agent.costs),
            new_costs={key: agent.costs[key] - scene['starting_costs'][key] for key in agent.costs},
            python_cost_basis=registration.get('python_cost_basis'))
        save_started = monotonic()
        agent._prepare_program_state()
        retain_checkpoint_sources(scene['output'] / 'final_state.json',
                                  scene['output'] / 'prepared_state.json')
        _write(scene['output'] / 'final_state.json', agent.checkpoint_result(result['status']))
        timing['checkpoint_wall_seconds'] = monotonic() - save_started
        result['wall_seconds'] = monotonic() - started
        _write(scene['output'] / 'summary.json', result)
    return result


def run_registered_representation_research(registration, output):
    """Run a bounded native representation question on declared exposed rows.

    The complete retained Python pool participates in the ordinary outer
    comparison. Only the named failed parents supply composition inputs to the
    local representation researcher; those two roles are recorded separately.
    """
    import ast
    from dataclasses import asdict
    from .anonymous_history_lab import AnonymousHistoryLab
    from .common_archive_inquiry import CommonArchiveInquiryAgent
    from .language import Program
    from .models import Observation
    from .online_inquiry_llm import InquiryLLMConfig
    from .paid_score_history import retain_comparison_history
    from .question_routing import register_theories

    if registration.get('execution_admitted') is not True:
        raise ValueError('register the development material and complete budget first')
    if not registration['material']['exposed_terminal_as_development']:
        raise ValueError('the old terminal rows are exposed development material')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'registration.json', registration)
    started = monotonic()
    read = lambda name: json.loads(Path(registration['material'][name]).read_text())
    records = read('paid_records')
    terminal = read('exposed_rows')
    terminal_start = len(records)
    terminal_row_start = sum(len(record['rows']) for record in records)
    for offset, row in enumerate(terminal):
        if not row.get('history'):
            records.append({'action': {'kind': 'trajectory', 'sequence': []}, 'rows': [], 'row_indices': [],
                            'origin': 'exposed_evaluator_development_not_purchase'})
        records[-1]['rows'].append(row)
        records[-1]['row_indices'].append(terminal_row_start + offset)
        records[-1]['action']['sequence'].append(dict(row['features']))
    validation = [terminal_start + index for index in
                  registration['material']['validation_trajectory_indices']]
    limits = registration['limits']
    laboratory = AnonymousHistoryLab(registration['environment']['kind'],
                                     registration['environment']['seed'])
    agent = CommonArchiveInquiryAgent(None, output / 'workspace', arm='hegel', source_task=False,
        config=InquiryLLMConfig(compute_budget=limits['compute'],
            execution_seconds=limits['python_seconds'], max_model_calls=0,
            observation_budget=limits['new_observation_rows']))
    agent.observations = records
    agent.rows = [Observation(dict(row['features']), float(row['target']),
        tuple(dict(point) for point in row.get('history', [])))
        for record in records for row in record['rows']]
    agent._save('observations.json', json.dumps([asdict(row) for row in agent.rows]))
    agent._save('laboratory_observations.json', json.dumps(records))
    parents, models = [], []
    def decode(node):
        op, value, name, lag, children = node
        return Program(op, value=value, name=name, lag=lag,
                       children=tuple(decode(child) for child in children))

    for name, source in registration['parents'].items():
        path = Path(source)
        local = agent.workspace / f'{name}.py'
        clone_or_copy(path, local)
        literal = ast.literal_eval(next(node.value for node in ast.parse(local.read_text()).body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_PROGRAM'
                                                    for t in node.targets)))
        parents.append({'id': name, 'program': decode(literal).to_dict()})
        models.append(local.name)
    register_theories(agent, parents)
    initial = {}
    for name in models:
        initial[name] = agent._execute('evaluate', {'model': name}, laboratory)
    _write(output / 'initial_evaluations.json', initial)
    complete_pool = []
    for roster in registration['retained_rosters']:
        for model in json.loads(Path(roster['file']).read_text()):
            resolved = str((Path(roster['workspace']) / model).resolve())
            if resolved not in complete_pool:
                complete_pool.append(resolved)
    _write(output / 'complete_retained_roster.json', complete_pool)
    rounds = []
    for number in range(limits['research_units']):
        conflicts = [question for question in agent.questions
            if question.get('kind') == 'representation' and question.get('status') == 'open'
            and question.get('origin') == 'computed_representation_conflict'
            and not question.get('native_representation_completed')]
        if not conflicts:
            break
        question = next((q for q in conflicts if q['id'] == agent._active_question_id), None)
        if question is None:
            if number:
                break
            question = conflicts[0]
        result = agent._execute('run', {'operator': 'research_route',
            'research_question': {'id': question['id']},
            'native_representation_plan': {'parent_theory_ids': [item['id'] for item in parents],
                'validation_records': validation, 'compute_limit': limits['per_unit_compute'],
                'max_candidates': limits['max_candidates']}}, laboratory)
        selected = result.get('selected_model')
        evaluated = agent._execute('evaluate', {'model': selected}, laboratory) if selected else None
        if evaluated is not None:
            diagnostic = evaluated.get('representation_diagnostic', {})
            question.update(returned_model=selected, representation_return={
                'model': selected, 'diagnostic': deepcopy(diagnostic),
                'mse': evaluated.get('mse'), 'scope': 'exposed_development_rows'})
            if (evaluated.get('returncode') == 0 and diagnostic.get('empirical_mse_floor') == 0):
                question.update(status='resolved_on_paid_rows', native_representation_completed=True)
                parent = next(q for q in agent.questions if q['id'] == question['parent_id'])
                parent['status'] = 'reopened'
            else:
                # Keep the actual new draft and its counterexample as the next
                # research object, rather than revisiting an unchanged parent.
                question['native_representation_completed'] = True
                native = result.get('native_representation', {}).get('selected')
                if native and 'program' not in native:
                    from .representation_inquiry import load_native_representation_result
                    native = load_native_representation_result(
                        result['native_representation_ref'], agent.workspace)['selected']
                if native:
                    declaration = {'id': f'representation_parent:{number}', 'program': native['program']}
                    register_theories(agent, [declaration])
                    parents.append(declaration)
        rounds.append({'research': result, 'actual_evaluation': evaluated})
        _write(output / f'unit_{number + 1}.json', rounds[-1])
    comparison = agent._execute('run', {'operator': 'research_choice',
        'models': list(dict.fromkeys([*complete_pool, *models, *agent.candidate_paths]))}, laboratory)
    comparison = retain_comparison_history(comparison)
    _write(output / 'complete_comparison.json', comparison)
    summary = {'status': 'completed' if not comparison.get('truncated') else 'comparison_incomplete',
        'scope': 'native_representation_development_not_independent_confirmation',
        'material': deepcopy(registration['material']),
        'complete_retained_paths': len(complete_pool), 'old_parents_retained': models,
        'research_unit_count': len(rounds), 'units': [str(output / f'unit_{n + 1}.json')
            for n in range(len(rounds))], 'final_parent': agent._evaluated_summary(),
        'comparison_requested': comparison.get('models_requested'),
        'comparison_completed': comparison.get('models_completed'),
        'costs': deepcopy(agent.costs), 'questions_ref': {
            'file': str(output / 'final_state.json'), 'json_path': ['online_questions']},
        'new_observations': 0, 'provider_calls': 0, 'wall_seconds': monotonic() - started,
        'source_costs_ref': deepcopy(registration['source_costs_ref']),
        'independent_transfer': 'UNTESTED', 'original_world_threshold': 1e-10}
    _write(output / 'summary.json', summary)
    _write(output / 'final_state.json', agent.checkpoint_result('development_completed'))
    return summary


def _frozen_parent_metrics(predictions, rows, indices):
    if predictions is None:
        return {'status': 'unscored', 'rows': len(indices), 'scored_rows': None, 'mse': None}
    return {'status': 'scored' if indices else 'empty_region',
            'rows': len(indices), 'scored_rows': len(indices), 'mse': (sum((predictions[index] - (
        rows[index].target if hasattr(rows[index], 'target') else rows[index]['target'])) ** 2
        for index in indices) / len(indices) if indices else None)}


def run_frozen_parent_evaluation(registration, output, *, workers=8):
    """Evaluate the two registered literal parents without restoring research.

    This is an evaluator-only final readout. The caller freezes both models and
    the original terminal distribution before admission; no selection, fitting,
    confirmation or research continuation is performed here.
    """
    from dataclasses import asdict
    from .anonymous_history_lab import AnonymousHistoryLab
    from .code_agent import _closed_temporal_export, python_model_context
    from .experiment_runtime import run_process_trials
    from scripts.run_anonymous_revision_inquiry import evaluate_terminal_model

    if registration.get('execution_admitted') is not True:
        raise ValueError('frozen parent evaluation is not admitted')
    if set(registration['models']) != {'remove', 'keep'}:
        raise ValueError('freeze exactly the remove and keep parents')
    contract = registration['evaluation']
    if (contract['rows'], contract['trajectory_length'], contract['mse_threshold']) != (512, 48, 1e-10):
        raise ValueError('preserve the original 512-row, 48-step, 1e-10 terminal contract')
    if workers < 1 or workers > 8:
        raise ValueError('use between one and eight physical workers')
    originals = {}
    for arm, model in registration['models'].items():
        workspace, inner = python_model_context(model['workspace'], model['model'])
        if not _closed_temporal_export(workspace, inner):
            raise ValueError(f'{arm}: terminal parent is not the closed literal frozen export')
        originals[arm] = (workspace / inner).read_bytes()
    reference = registration['paid_observations_ref']
    paid = json.loads(Path(reference['file']).read_text(encoding='utf-8'))
    for key in reference.get('json_path', []):
        paid = paid[key]
    paid_rows = [row for record in paid for row in record['rows']] if paid and 'rows' in paid[0] else paid
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = monotonic(), process_time()
    _write(output / 'registration.json', registration)
    lab = AnonymousHistoryLab(contract['kind'], contract['seed'])
    acquisition_started = monotonic()
    rows = lab.evaluation_rows(length=contract['rows'], seed=contract['evaluation_seed'])
    acquisition_seconds = monotonic() - acquisition_started
    _write(output / 'terminal_rows.json', [asdict(row) for row in rows])
    trajectories, regions = _frozen_parent_row_groups(rows, paid_rows)

    def evaluate(arm):
        model = registration['models'][arm]
        attempt_started = monotonic()
        try:
            evaluation = evaluate_terminal_model(model['workspace'], model['model'], rows,
                timeout_seconds=registration['execution_timeout_seconds'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            evaluation = {'status': 'evaluation_failed', 'mse': None,
                'error': f'{type(error).__name__}: {error}', 'evaluation_rows': len(rows),
                'python_execution_seconds': monotonic() - attempt_started}
        evaluation['passed'] = (evaluation.get('returncode') == 0 and
            evaluation.get('mse') is not None and evaluation['mse'] <= contract['mse_threshold'])
        return {'arm': arm, 'evaluation': evaluation}

    trials = run_process_trials(evaluate, list(registration['models']), workers)
    arms = {}
    for trial in trials:
        arm, evaluation = trial['arm'], trial['evaluation']
        model = registration['models'][arm]
        workspace, inner = python_model_context(model['workspace'], model['model'])
        unchanged = (workspace / inner).read_bytes() == originals[arm]
        if not unchanged:
            raise ValueError(f'{arm}: frozen model changed during terminal execution')
        predictions = evaluation.pop('predictions', None)
        arm_output = output / arm
        arm_output.mkdir()
        if predictions is not None:
            _write(arm_output / 'terminal_predictions.json', {'model_ref': model,
                'evaluation_contract_ref': {'file': str(output / 'registration.json'),
                    'json_path': ['evaluation']},
                'terminal_rows_ref': {'file': str(output / 'terminal_rows.json'), 'json_path': []},
                'predictions': predictions})
            evaluation['prediction_ref'] = {'file': str(arm_output / 'terminal_predictions.json'),
                                          'json_path': ['predictions']}
        scored = predictions is not None and evaluation.get('returncode') == 0
        arm_result = {'evaluation': evaluation, 'model_unchanged': unchanged,
            'trajectories': [_frozen_parent_metrics(predictions, rows, indices)
                             for indices in trajectories],
            'regions': {key: _frozen_parent_metrics(predictions, rows, indices)
                        for key, indices in regions.items()},
            'costs': {'compute_cost': len(rows),
                'compute_accounting': 'requested_model_row_charge_retained_on_failure',
                'requested_rows': len(rows), 'attempted_rows': len(rows),
                'scored_rows': len(rows) if scored else None,
                'python_execution_seconds': evaluation['python_execution_seconds'],
                'model_calls': 0, 'reported_tokens': 0, 'new_fits': 0,
                'agent_observation_cost': 0, 'measurement_cost': 0, 'queries': 0}}
        _write(arm_output / 'evaluation.json', arm_result)
        arms[arm] = arm_result

    def delta(left, right):
        return None if left is None or right is None else right - left

    trajectory_deltas = [delta(left['mse'], right['mse']) for left, right in zip(
        arms['remove']['trajectories'], arms['keep']['trajectories'])]
    pair = {'status': 'fully_compared' if all(value is not None for value in trajectory_deltas)
            else 'incomplete',
        'compared_trajectories': sum(value is not None for value in trajectory_deltas),
        'unscored_trajectories': sum(value is None for value in trajectory_deltas),
        'mse_keep_minus_remove': delta(arms['remove']['evaluation'].get('mse'),
                                         arms['keep']['evaluation'].get('mse')),
        'trajectory_mse_keep_minus_remove': trajectory_deltas,
        'improved_trajectories': sum(value < 0 for value in trajectory_deltas if value is not None),
        'worse_trajectories': sum(value > 0 for value in trajectory_deltas if value is not None),
        'tied_trajectories': sum(value == 0 for value in trajectory_deltas if value is not None),
        'region_mse_keep_minus_remove': {key: delta(arms['remove']['regions'][key]['mse'],
            arms['keep']['regions'][key]['mse']) for key in regions}}
    costs = {'terminal_observations_once': len(rows),
        'compute_cost': sum(item['costs']['compute_cost'] for item in arms.values()),
        'compute_accounting': 'requested_model_row_charge_retained_on_failure',
        'requested_rows': sum(item['costs']['requested_rows'] for item in arms.values()),
        'attempted_rows': sum(item['costs']['attempted_rows'] for item in arms.values()),
        'scored_rows': (sum(item['costs']['scored_rows'] for item in arms.values())
                        if all(item['costs']['scored_rows'] is not None for item in arms.values()) else None),
        'known_scored_rows': sum(item['costs']['scored_rows'] for item in arms.values()
                                 if item['costs']['scored_rows'] is not None),
        'python_execution_seconds': sum(item['costs']['python_execution_seconds'] for item in arms.values()),
        'row_generation_seconds': acquisition_seconds, 'model_calls': 0, 'reported_tokens': 0,
        'new_fits': 0, 'agent_observation_cost': 0, 'queries': 0,
        'prior_costs_ref': deepcopy(registration['prior_costs_ref'])}
    summary = {'status': 'completed' if all(item['evaluation'].get('returncode') == 0
        and item['evaluation'].get('mse') is not None for item in arms.values()) else 'completed_with_failures',
        'scope': 'frozen_model_terminal_readout', 'arms': arms, 'pair': pair, 'costs': costs,
        'evaluation_contract': deepcopy(contract), 'exposure_audit_ref': deepcopy(
            registration['exposure_audit_ref']),
        'frozen_registration_ref': {'file': str(output / 'registration.json'), 'json_path': []},
        'wall_seconds': monotonic() - started, 'coordinator_cpu_seconds': process_time() - cpu_started,
        'research_selection': 'not_performed', 'fresh_confirmation': 'not_performed',
        'independent_transfer': 'UNTESTED', 'mse_threshold': contract['mse_threshold']}
    _write(output / 'summary.json', summary)
    files = [path for path in output.rglob('*') if path.is_file()]
    _write(output / 'resources.json', {'wall_seconds': monotonic() - started,
        'coordinator_cpu_seconds': process_time() - cpu_started,
        'logical_output_bytes_before_resource_receipt': sum(path.stat().st_size for path in files),
        'allocated_output_bytes_before_resource_receipt': sum(path.stat().st_blocks * 512 for path in files),
        'disk_scope': 'allocated_output_file_blocks_not_net_filesystem_growth',
        'files_before_resource_receipt': len(files), 'max_physical_workers': workers,
        'checkpoint_storage_reused': True, 'experiment_runtime_reused': True})
    return summary
