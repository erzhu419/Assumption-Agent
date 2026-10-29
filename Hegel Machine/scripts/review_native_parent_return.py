#!/usr/bin/env python3
"""Return the saved object-parent unit to Common without repeating its science."""

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from partial_role_paid_scene import load_paid_scene
from run_partial_role_parent import load, save
from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return,
)
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.method_feedback import ensure_python_method_topic
from hegel_machine.cognition.object_continuation import decode_entry
from hegel_machine.cognition.object_method_research import publish_object_response_options
from hegel_machine.cognition.question_routing import consume_method_return_agenda


def review(source, output):
    config = load(source / 'registration.json')
    receipt = load(source / 'receipt.json')
    selection = load(source / 'selection.json')
    scene = load_paid_scene(config['root'], config['acceptance'], config['previous'])
    output.mkdir(parents=True, exist_ok=False)
    target_index = next(i for i, item in enumerate(scene['registration']['tasks'])
                        if item['seed'] == config['seed'])
    source_ref = {'file': str(Path(config['root']) / 'registration.json'),
                  'json_path': ['tasks', target_index]}
    result_ref = {'file': str(source / 'receipt.json')}
    comparison_ref = {'file': str(source / 'selection.json')}
    save(output / 'registration.json', {
        'source_task_ref': source_ref, 'source_result_ref': result_ref,
        'scope': 'retrospective native method return; no scientific rerun',
        'method_review_limit': 3, 'original_development_limit': config['compute_limit_including_previous'],
        'prior_development_compute': receipt['cumulative_development_compute'],
        'new_observations': 0, 'new_predictions': 0, 'new_fits': 0,
        'old_confirmation_role': 'already_exposed_retained_evidence'})

    parent = decode_entry(load(source / 'selected_entry.json'))
    parent['validation_loss'] = receipt['paid_mse']
    spec = SimpleNamespace(context={'object_schema': parent['structure']['schema']})
    data_ref = {**comparison_ref, 'json_path': ['source_refs', 'paid_trajectories']}
    old_parent = deepcopy(scene['parent'])
    old_parent['validation_loss'] = selection['actions'][0]['prior_loss']
    # The pre-repair view reuses saved scores on all 236 rows, including the
    # supplement already billed in the source. It is a retrospective choice
    # audit, not another method invocation or a newly frozen preference label.
    before = publish_object_response_options(spec, scene['paid_rows'], {
        'selected': old_parent, 'full_comparison': selection['full_comparison'],
        'claims': scene['pool'], 'action': {'observation_rows': receipt['paid_rows']}},
        compute_remaining=config['remaining_compute'] - receipt['additional_exposed48_compute'],
        tolerance=config['tolerance'], data_ref=data_ref,
        parent_ref=scene['source_refs']['parent'],
        pool_ref={**comparison_ref, 'json_path': ['source_refs', 'pool']},
        comparison_ref=comparison_ref)
    before['evidence_scope'] = 'retrospective_pre_repair_choice_not_a_paired_method_trial'
    save(output / 'before_options.json', before)
    current = publish_object_response_options(spec, scene['paid_rows'], {
        'selected': parent, 'full_comparison': selection['full_comparison'],
        'status': selection['status'], 'claims': selection['scores'],
        'action': selection['actions'][0]},
        compute_remaining=config['compute_limit_including_previous'] - receipt['cumulative_development_compute'],
        tolerance=config['tolerance'], data_ref=data_ref,
        parent_ref={'file': str(source / 'selected_entry.json')},
        pool_ref={**comparison_ref, 'json_path': ['scores']}, comparison_ref=comparison_ref)
    save(output / 'current_options.json', current)

    common = CommonObjectDependencyMethods(output / 'common', [],
        budget=SearchBudget(3), actions=[], source_task_ref=source_ref).common
    previous = Path(config['previous'])
    record_native_parent_research_return(common, source_task_ref=source_ref,
        result_ref={'file': str(previous / 'receipt.json')},
        comparison_ref={'file': str(previous / 'working_selection.json')},
        cost_refs=[{'file': str(previous / 'receipt.json')}],
        mission={'purpose': 'method_learning', 'allocation': {'compute': 3}})
    feedback = record_native_parent_research_return(common, source_task_ref=source_ref,
        result_ref=result_ref, before_ref={'file': str(source / 'registration.json')},
        comparison_ref=comparison_ref,
        confirmation_ref={**result_ref, 'json_path': ['confirmation']},
        cost_refs=[result_ref], decision_options_ref={'file': str(output / 'current_options.json')})
    reviewed = consume_method_return_agenda(common, feedback, None, compute_limit=3)
    # A repeated ordinary delivery exercises the actual saved agenda contract;
    # it must not purchase or execute the previously passed source again.
    repeated = record_native_parent_research_return(common,
        source_task_ref=source_ref, result_ref=result_ref)
    reused = consume_method_return_agenda(common, repeated, None, compute_limit=3)
    result = {'source_task_ref': source_ref, 'source_result_ref': result_ref,
        'prior_return_ref': {'file': str(previous / 'receipt.json')},
        'before_options_ref': {'file': str(output / 'before_options.json')},
        'current_options_ref': {'file': str(output / 'current_options.json')},
        'method_feedback': feedback, 'method_review': reviewed, 'repeated_delivery': reused,
        'costs': common.costs, 'new_predictions': 0, 'new_fits': 0,
        'cumulative_development_compute': receipt['cumulative_development_compute']
            + common.costs['compute_cost'],
        'scope': 'same-source return integration; no new method or independent gain'}
    save(output / 'receipt.json', result)
    return result


def review_retained_choices(source, previous, output):
    """Extend the paid topic with the original policy's real choice history."""
    prior = load(previous / 'receipt.json')
    registration = load(source / 'registration.json')
    report_ref = {'file': str(Path(registration['root']) / 'runs'
        / str(registration['seed']) / 'pr3/response_report.json')}
    output.mkdir(parents=True, exist_ok=False)
    # Transport the existing small topic, including its used method allowance.
    # Results and native decisions remain at their immutable original paths.
    questions = load(previous / 'common/questions.json')
    owner = next(q for q in questions if q['id'] == 'task:method_choice')
    save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
    common = CommonObjectDependencyMethods(output / 'common', [],
        budget=SearchBudget(4), actions=[], source_task_ref=prior['source_task_ref']).common
    ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
        topic_ref={'file': str(output / 'previous_topic.json')})
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    mission = deepcopy(owner['mission'])
    mission['allocation']['compute'] += 4
    feedback = record_native_parent_research_return(common,
        source_task_ref=prior['source_task_ref'], result_ref=report_ref,
        cost_refs=[{**report_ref, 'json_path': ['costs']}],
        decision_options_ref={**report_ref, 'json_path': ['actions']}, mission=mission)
    reviewed = consume_method_return_agenda(common, feedback, None, compute_limit=4)
    result = {'prior_review_ref': {'file': str(previous / 'receipt.json')},
        'source_task_ref': prior['source_task_ref'], 'source_result_ref': report_ref,
        'method_feedback': feedback, 'method_review': reviewed, 'costs': common.costs,
        'new_predictions': 0, 'new_fits': 0,
        'cumulative_development_compute': prior['cumulative_development_compute']
            + common.costs['compute_cost'],
        'scope': 'existing multi-direction decisions retained; same-state paired return still absent'}
    save(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-review', type=Path)
    args = parser.parse_args()
    print(review_retained_choices(args.source.resolve(), args.previous_review.resolve(), args.output.resolve())
          if args.previous_review else review(args.source.resolve(), args.output.resolve()))
