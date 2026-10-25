"""Execute a registered retained-source control using ordinary Common actions."""

from copy import deepcopy
import gc
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic, process_time

from .checkpoint_storage import write_json_checkpoint, clone_or_copy
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
