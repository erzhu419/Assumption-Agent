#!/usr/bin/env python3
"""Continue a learned partition on paid feedback, then freeze its method body."""

import argparse
from copy import deepcopy
from pathlib import Path

from scripts.run_learned_structure_successor import freeze_runtime
from scripts.revise_learned_structure_successor import restore, return_paid_method_study
from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_object_research import CommonObjectDependencyMethods
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import (
    read_method_reference, quote_partition_followup_analysis)
from hegel_machine.cognition.method_feedback import ensure_python_method_topic
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry


ROOT = Path(__file__).resolve().parents[1]


def freeze(output, source):
    freeze_runtime(output, source)
    name = Path('scripts/continue_partition_method.py')
    base = source / 'runtime' / name
    write_text_incrementally(output / 'runtime' / name, (ROOT / name).read_text(),
        base=base if base.exists() else None)


def entry_from_method(method):
    return {'candidate': Candidate('retained_partition', Program.from_dict(method['program']),
                'research_program'), 'kind': 'research_program', 'family_id': 'T03',
        'research_program': deepcopy(method['research_program']),
        'training_loss': method['training_loss'], 'validation_loss': method['validation_loss']}


def register(source, output):
    source, output = Path(source), Path(output)
    old, receipt = load(source / 'registration.json'), load(source / 'receipt.json')
    ledger = load(source / 'method_return/receipt.json')
    study = load(source / 'study.json')
    original = read_method_reference(receipt['proposal_ref'], source)
    choices = quote_partition_followup_analysis(original['proposal'], study)
    owner = next(q for q in load(source / 'method_return/common/questions.json') if q['id'] == 'task:method_choice')
    review_limit = 1 + len(owner['method_use_plan']['native_return_refs']) + sum(x['required_compute'] for x in choices)
    if len(choices) != 1 or review_limit + 15000 + 19 > ledger['remaining_compute']:
        raise ValueError('One evidence-supported, fully allocated followup is required')
    output.mkdir(parents=True, exist_ok=False)
    cfg = dict(source=str(source), original_source=old['original_source'],
        source_task_ref=old['source_task_ref'], previous_compute=ledger['cumulative_compute'],
        remaining_compute=ledger['remaining_compute'], total_compute_limit=old['total_compute_limit'],
        threshold=old['threshold'], review_compute_limit=review_limit, interval_compute_limit=15000,
        active_wall_seconds=1800, original_proposal_ref=receipt['proposal_ref'],
        first_method_index=choices[0]['method_index'], source_cost_refs=[ref(source / 'method_return/receipt.json')],
        exclude_fresh_confirmation_ref=old['exclude_fresh_confirmation_ref'],
        scope='resume completed first segment; paid return selects existing interval second segment',
        stop_rule='one bounded second segment; no new purchase, fit restart or alternative source',
        new_observations=0, model_calls=0)
    save(output / 'registration.json', cfg)
    freeze(output, source)
    return cfg


def run(output):
    from hegel_machine.cognition.transfer_partitions import continue_transfer_partition

    output = Path(output)
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the newly frozen followup once')
    cfg = load(output / 'registration.json')
    source = Path(cfg['source'])
    budget = SearchBudget(cfg['remaining_compute'])
    timing = ExecutionTiming(('restore', 'proposal', 'interval', 'compare'))
    save(output / 'started.json', {'source': cfg['source']})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('restore'):
                scene = restore(cfg['original_source'])
                training = scene['scene']['training']
                feedback = [*scene['scene']['validation'], *scene['exposed']]
                old_study = load(source / 'study.json')
                first = next(m for m in old_study['methods'] if m['index'] == cfg['first_method_index'])
                owner = next(q for q in load(source / 'method_return/common/questions.json')
                             if q['id'] == 'task:method_choice')
            with timing.phase('proposal'):
                save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
                common = CommonObjectDependencyMethods(output / 'common', [],
                    budget=SearchBudget(cfg['review_compute_limit']), actions=[],
                    source_task_ref=cfg['source_task_ref']).common
                ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
                    topic_ref=ref(output / 'previous_topic.json'))
                question = next(q for q in common.questions if q['id'] == owner['id'])
                question['mission']['allocation']['compute'] += cfg['review_compute_limit']
                reviewed = common._execute('run', {'operator': 'research_route',
                    'research_question': {'id': question['id']},
                    'allocation': {'compute': cfg['review_compute_limit']}}, None)
                budget.spend(common.costs['compute_cost'])
                save(output / 'proposal_return.json', reviewed)
                proposal_record = next(item for item in reviewed.get('method_action_research', {}).get(
                    'transfer_partition_proposals', []) if item.get('followup_of') == cfg['original_proposal_ref'])
                bundle = read_method_reference(proposal_record['proposal_ref'], common.workspace)
                proposal = bundle['proposal']
                current = entry_from_method(first)
                current['research_program']['transfer_partition']['second_stage'] = deepcopy(
                    proposal['candidates'][0]['research_program']['transfer_partition']['second_stage'])
            with timing.phase('interval'):
                local = SearchBudget(cfg['interval_compute_limit'])
                result = continue_transfer_partition(current, training, feedback, local,
                    tolerance=cfg['threshold'], feedback_loss=first['validation_loss'])
                budget.spend(local.spent)
                selected = result.get('selected')
                save(output / 'interval_result.json', {key: value for key, value in result.items()
                    if key not in {'selected', 'candidates'}})
                candidates = list(result.get('candidates', []))
                # The first segment exported one draft. Preserve every paid
                # interval assessment while exporting at most the other three
                # slots under the original four-draft contract.
                assessed = [{'program': item['candidate'].program.to_dict(),
                    **{key: item.get(key) for key in ('parameter_values', 'parameter_index',
                        'radius_interval', 'training_loss', 'validation_loss', 'all_paid_loss')}}
                    for item in candidates]
                save(output / 'interval_assessments.json', assessed)
                # Keep each actual newly delivered program once; the original
                # first-stage draft already remains in the complete old pool.
                exported, seen = [], {current['candidate'].program}
                if result.get('execution_complete'):
                    candidates.sort(key=lambda item: (
                        0. if item['all_paid_loss'] <= cfg['threshold'] else item['all_paid_loss'],
                        item['candidate'].program.size))
                    for item in candidates:
                        program = item['candidate'].program
                        if program not in seen and len(exported) < 3:
                            seen.add(program)
                            exported.append(item)
                elif candidates:
                    save(output / 'interval_partial_candidates.json', [encode_entry(item) for item in candidates])
                save(output / 'interval_candidates.json', [encode_entry(item) for item in exported])
                if selected is not None:
                    save(output / 'selected_entry.json', encode_entry(selected))
            with timing.phase('compare'):
                complete = result.get('execution_complete', False)
                methods = [{**deepcopy(first), 'index': 0, 'new_compute_cost': 0, 'reused': True},
                    {'index': 1, 'execution_complete': complete, 'new_compute_cost': local.spent,
                     'paid_compute_cost': first['paid_compute_cost'] + local.spent, 'reused': False,
                     'program': selected['candidate'].program.to_dict() if selected else None,
                     'research_program': selected['research_program'] if selected else None,
                     'training_loss': result.get('first_training_loss'),
                     'validation_loss': result.get('feedback_loss'),
                     'denied_request': deepcopy(local.last_denied_request)}]
                delta = methods[1]['validation_loss'] - first['validation_loss'] if complete else None
                improved = complete and delta < -cfg['threshold']
                study = {'proposal': proposal, 'basis': {'minimum_improvement': cfg['threshold'],
                    'source_refs': [ref(source / 'study.json')], 'training_rows': len(training),
                    'validation_rows': len(feedback), 'scope': 'resume_same_completed_first_segment'},
                    'methods': methods, 'pairs': [{'method_index': 1,
                        'edit': proposal['candidates'][0]['edit'], 'execution_complete': complete,
                        'validation_loss_change': delta, 'status': 'local_step_improvement' if improved else
                            'complete_step_unhelpful' if complete else 'incomplete_not_negative'}],
                    'budget_incomplete': bool(local.denied_requests), 'time_incomplete': False,
                    'status': 'development_step_revision_selected' if improved else result['status'],
                    'data_role': 'exposed_adaptive_development', 'independent_support': False}
                save(output / 'study.json', study)
                old_pool = load(source / 'pool_comparison.json')
                scores = list(old_pool['scores'])
                if complete:
                    scores.extend({'candidate_ref': ref(output / 'interval_candidates.json', index),
                        'loss': item['all_paid_loss']} for index, item in enumerate(exported))
                save(output / 'pool_comparison.json', {**old_pool, 'scores': scores,
                    'full_comparison': complete and old_pool['full_comparison'],
                    'old_scores_reused': len(old_pool['scores']),
                    'new_candidates': len(exported), 'internal_assessments': len(assessed),
                    'internal_assessments_ref': ref(output / 'interval_assessments.json'),
                    'parent_changed': False, 'selected_ref': old_pool['selected_ref']})
                passed = complete and result['all_paid_loss'] <= cfg['threshold'] and methods[1]['validation_loss'] <= cfg['threshold']
                if passed:
                    save(output / 'frozen_method.json', {'research_program': proposal['candidates'][0]['research_program'],
                        'proposal_ref': proposal_record['proposal_ref'], 'study_ref': ref(output / 'study.json'),
                        'scope': 'one-source-development-supported-two-stage-method',
                        'independent_support': False, 'source_task_ref': cfg['source_task_ref']})
                receipt = dict(status='method_ready_for_independent_successor' if passed else result['status'],
                    proposal_ref=proposal_record['proposal_ref'], interval_status=result['status'],
                    first_segment_reused=True, training_rows=len(training), feedback_rows=len(feedback),
                    old_feedback_mse=first['validation_loss'], new_feedback_mse=result.get('feedback_loss'),
                    all_paid_mse=result.get('all_paid_loss'), completed=complete,
                    frozen_method_ref=ref(output / 'frozen_method.json') if passed else None,
                    compared_candidates=len(scores), old_scores_reused=len(old_pool['scores']), parent_changed=False,
                    new_candidates=len(exported), internal_assessments=len(assessed),
                    compute_cost=budget.spent, cumulative_compute=cfg['previous_compute'] + budget.spent,
                    remaining_compute=budget.remaining, source_cost_refs=cfg['source_cost_refs'],
                    new_observations=0, model_calls=0, independent_method_support=False, execution_timing=timing.result())
                save(output / 'receipt.json', receipt)
                receipt['method_return'] = return_paid_method_study(output, output / 'method_return')
                return receipt
    except BaseException as exc:
        save(output / 'failure.json', {'error': repr(exc), 'compute_known': budget.spent,
            'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = register(args.source, args.output) if args.mode == 'prepare' else run(args.output.resolve())
    print({k: v for k, v in result.items() if k not in {'source_cost_refs', 'method_return'}})
