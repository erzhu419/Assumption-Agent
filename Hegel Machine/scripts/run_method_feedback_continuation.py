"""Prepared native feedback continuation; never run a proposed scientific action.

The final recovery summary is the complete cohort. Original training references
and same-invocation updates remain references; a physical case contributes at
most its first actual C invocation, otherwise its first actual B invocation.
"""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.experiment_runtime import (
    ActivityExecutionDeadline, ExecutionTiming, python_execution_deadline,
)
from hegel_machine.cognition.method_feedback import begin_method_feedback, procedure_use_decision
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.research_episodes import load_source_values


METHOD = 'consolidated:0:method:0'
COMPUTE_CAP = 8_662_896
NATIVE_HOST_CAP = 600.
DEFAULT_BASELINE = '/home/erzhu419/hegel-v7-condition-successor-prepared-20261026/frozen_new_conditions.json'
DEFAULT_STAGE1 = '/mnt/hegel-snapshot-cow/pr3_v7_method_step_authorized_20261026/summary.json'
DEFAULT_ORIGINAL = '/mnt/hegel-snapshot-cow/pr3_v7_condition_successor_authorized_20261026'
DEFAULT_FINAL = '/mnt/hegel-snapshot-cow/pr3_v7_condition_successor_recovery_20261026/target/summary.json'


def ref(path, *parts):
    return {'file': str(path), 'json_path': list(parts)}


def read(reference, workspace):
    return load_source_values([reference], workspace)[0]['value']


def identity(reference):
    return str(reference['file']), tuple(reference.get('json_path', []))


def _workspace(reference):
    # A retained training file may be a cold symlink; its logical parent owns
    # the actual Common library, profiles and original local references.
    return Path(reference['file']).parent


def _local_calls(workspaces, method_id):
    """Read the published outcome list, not every historical use_* pathname."""
    calls, positions = [], {}
    owners = set(workspaces)
    for workspace in workspaces:
        path = workspace / 'shared/theory_library.json'
        if not path.is_file():
            continue
        local = read_json_checkpoint(path, keys=('method_feedback',))
        group = local.get('method_feedback', {}).get(method_id, {})
        for reference in group.get('outcome_refs', []):
            record = read(reference, workspace)
            profile_owner = Path(record['profile_ref']['file']).parent.parent
            origin = record.get('origin_ref', reference)
            origin_owner = Path(origin['file']).parent.parent
            if profile_owner not in owners and origin_owner not in owners:
                continue  # An inherited source use is not this target's call.
            if record['method_ref'] != {'id': method_id, 'version': 2}:
                raise ValueError('the cohort must use the registered unchanged method body @2')
            key = identity(origin)
            item = {'feedback_ref': deepcopy(reference), 'origin_ref': deepcopy(origin),
                'status': record['status'], 'execution_complete': record.get('execution_complete'),
                'has_complete_working_comparison': record.get('working_comparison', {}).get('complete') is True,
                'has_paired_future': 'paired_future' in record}
            if key in positions:
                calls[positions[key]] = item  # Latest saved state of the same invocation.
            else:
                positions[key] = len(calls)
                calls.append(item)
    return calls


def prepare_manifest(final_summary, original_root=DEFAULT_ORIGINAL, method_id=METHOD):
    final_summary, original_root = Path(final_summary), Path(original_root)
    summary = read_json_checkpoint(final_summary)
    protocol_path = final_summary.parent / 'protocol.json'
    protocol = read_json_checkpoint(protocol_path)
    trials = summary['trials']
    sidecar_path = final_summary.parent / 'training_results.json'
    sidecar = read_json_checkpoint(sidecar_path) if sidecar_path.is_file() else []
    saved = {(item['case']['id'], item['arm']): item['result_ref'] for item in sidecar}
    index = {(item['case'], item['arm']): i for i, item in enumerate(trials)}
    expected = {(case['id'], arm) for case in protocol['cases'] for arm in protocol['arms']}
    if set(index) != expected or len(index) != len(trials):
        raise ValueError('feedback waits for the complete final cohort summary')
    original_admission = original_root / 'interruption_recovery/stopped_thin_admission.json'
    original = read_json_checkpoint(original_admission)
    original_refs = {(item['case'], item['arm']): item['checkpoint_ref']
        for item in [*original['completed_training_reuse'], *original['tasks']]}
    selected, cases = [], []
    physical = set()
    for case in protocol['cases']:
        physical_key = case['kind'], case['seed']
        if physical_key in physical:
            raise ValueError('the final cohort contains duplicate physical cases')
        physical.add(physical_key)
        calls_by_arm, trial_refs = {}, {}
        for arm in protocol['arms']:
            key = case['id'], arm
            number = index[key]
            trial = trials[number]
            result_ref = trial.get('result_ref') or saved.get(key)
            if result_ref is None:
                raise ValueError('final trial requires its retained result_ref or training_results sidecar')
            old = original_refs[key]
            workspaces = list(dict.fromkeys((_workspace(old), _workspace(result_ref))))
            trial_refs[arm] = {'summary_trial_ref': ref(final_summary, 'trials', number),
                'result_ref': deepcopy(result_ref), 'original_checkpoint_ref': deepcopy(old)}
            if arm in ('fixed_pr2', 'pr3'):
                calls_by_arm[arm] = _local_calls(workspaces, method_id)
        arm = next((name for name in ('pr3', 'fixed_pr2') if calls_by_arm.get(name)), None)
        call = calls_by_arm[arm][0] if arm else None
        entry = {'case_id': case['id'], 'physical_case': {'kind': case['kind'], 'seed': case['seed']},
            'selected_arm': arm, 'selected_actual_use': deepcopy(call),
            'all_actual_uses': calls_by_arm, 'trial_refs': trial_refs,
            'policy_comparison_scope': 'same_body_different_conditions_not_body_present_absent',
            'body_process_return_from_policy_contrast': 'UNKNOWN'}
        cases.append(entry)
        if call:
            selected.append(call['feedback_ref'])
    return {'status': 'prepared_not_executed', 'method_body_ref': {'id': method_id, 'version': 2},
        'selection_rule': 'first_actual_C_else_first_actual_B_one_vote_per_physical_case',
        'final_summary_ref': ref(final_summary), 'original_interruption_ref': ref(original_admission),
        'protocol_ref': ref(protocol_path), 'selected_actual_use_refs': selected, 'physical_cases': cases,
        'condition_contrast_attribution': 'separate_referenced_policy_result_never_an_invented_body_counterexample'}


def _decision(agent, candidate, record):
    paid = read(record['evidence_basis']['paid_evidence_ref'], agent.workspace)
    agent.rows = [Observation(row['features'], row['target'], tuple(row.get('history', []))) for row in paid]
    agent._method_use_profile = read(record['profile_ref'], agent.workspace)
    binding = deepcopy(record.get('invocation_binding') or record['evidence_basis'])
    binding.update(candidate_id=candidate['id'], candidate_version=candidate['version'])
    return procedure_use_decision(agent, candidate, binding)


def _public_paid_scene(agent, record, case):
    """Restore the original paid prefix, not the later end-of-task observations."""
    paid = read(record['evidence_basis']['paid_evidence_ref'], agent.workspace)
    agent.rows = [Observation(row['features'], row['target'], tuple(row.get('history', []))) for row in paid]
    binding = record.get('invocation_binding') or record['evidence_basis']
    count = binding['paid_cutoff']['records']
    workspace = Path(record['profile_ref']['file']).parent.parent
    references = [ref(workspace / 'laboratory_observations.json', index) for index in range(count)]
    agent.observations = [item['value'] for item in load_source_values(references, workspace)]
    if [row for item in agent.observations for row in item['rows']] != paid:
        raise ValueError('retained prefix records differ from the actual method input rows')
    agent.evaluated_model = None
    parent = record.get('theory_target', {}).get('parent_binding_obligation', {}).get('source_model')
    comparison = record.get('working_comparison', {})
    if parent and comparison.get('before_score_ref'):
        score = read(comparison['before_score_ref'], agent.workspace)
        scope = read(score['scope_ref'], agent.workspace)
        if (comparison.get('before_model') != parent or score['model'] != parent
                or scope['rows'] != paid or score['result']['returncode'] != 0):
            raise ValueError('the paid before score must describe the recorded pre-call parent and rows')
        # A before/after comparison alone does not imply that its before-model
        # was already retained. The recorded parent obligation supplies that
        # identity; the paid assessment supplies its score without evaluation.
        result = score['result']
        agent.evaluated_model = {'model': score['model'], 'source_model': score['model'],
            'current_mse': result['mse'], 'evaluated_rows': len(scope['rows']),
            **{key: deepcopy(result[key]) for key in ('evaluation_scope', 'fitting_view') if key in result}}
    agent._active_question_id = 'task'
    laboratory = AnonymousHistoryLab(**case['physical_case'])
    expected = read(record['profile_ref'], agent.workspace)
    rebuilt = begin_method_feedback(agent, laboratory)['profile']
    agent._method_use_profile = deepcopy(expected)
    return laboratory, rebuilt == expected


def run_feedback(manifest, output, *, baseline=DEFAULT_BASELINE, stage1=DEFAULT_STAGE1):
    """Execute only existing feedback/condition research inside the old cap."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('nonempty output must be reviewed, not silently restarted')
    output.mkdir(parents=True, exist_ok=True)
    timing = ExecutionTiming(('restore', 'import', 'condition_route', 'ordinary_gate_readout'))
    prior = read_json_checkpoint(stage1, keys=('costs', 'native_host_wall_seconds'))
    paid_compute = prior['costs']['compute_cost']
    paid_host = prior['native_host_wall_seconds']
    if paid_compute >= COMPUTE_CAP or paid_host >= NATIVE_HOST_CAP:
        raise ValueError('original cumulative stage1 allowance is exhausted')
    selected = manifest['selected_actual_use_refs']
    if not selected:
        raise ValueError('no actual invocation: do not manufacture a method-use trial')
    result = {'status': 'started', 'input_manifest': deepcopy(manifest), 'stage1_prior_ref': ref(stage1),
        'baseline_library_ref': ref(baseline), 'new_world_fits': 0, 'new_world_predictions': 0,
        'new_world_observations': 0, 'new_terminal_evaluations': 0,
        'policy_contrast_body_credit': 'not_inferred', 'next_scientific_action_executed': False}
    agent = None
    try:
        with python_execution_deadline(NATIVE_HOST_CAP - paid_host, activity=True):
            with timing.phase('restore'):
                agent = CommonArchiveInquiryAgent(None, output / 'learner', arm='hegel', source_task=False,
                    knowledge={'files': {}, 'theory_consolidation': ref(baseline)},
                    config=InquiryLLMConfig(observation_budget=0, max_model_calls=0, token_budget=0,
                        execution_seconds=0., compute_budget=COMPUTE_CAP))
                agent.costs['compute_cost'] = paid_compute
                before = deepcopy(agent.theory_consolidation['candidates'])
                if len(before) != 21:
                    raise ValueError('continuation requires the complete registered 21-candidate library')
                identifier = manifest['method_body_ref']['id']
                old = before[identifier]
                if old['version'] != 5 or old['method_use_conditions']['usage_version'] != manifest['method_body_ref']:
                    raise ValueError('continuation must start at the registered @5 conditions / @2 body')
                records = [read(reference, output) for reference in selected]
                selected_cases = [case for case in manifest['physical_cases'] if case['selected_actual_use']]
                previous_decisions = [_decision(agent, old, record) for record in records]
            remaining = COMPUTE_CAP - agent.costs['compute_cost']
            mission = {'mission_id': 'post_evaluation_method_feedback', 'purpose': 'method_learning',
                'theory_learning_allocation': {'compute': remaining}}
            with timing.phase('import'):
                result['import'] = agent._execute('run', {'operator': 'theory_consolidation',
                    'object_type': 'method_use_condition', 'candidate_ids': [identifier],
                    'method_feedback_source_refs': selected, 'mission': deepcopy(mission),
                    'allocation': {'compute': remaining}}, None)
            with timing.phase('condition_route'):
                record = records[0]
                # Previous readouts ended at the last case. The method question
                # and its actual Common profile must use this first binding.
                laboratory, same_profile = _public_paid_scene(agent, record, selected_cases[0])
                binding = deepcopy(record.get('invocation_binding') or record['evidence_basis'])
                binding.update(candidate_id=identifier,
                    candidate_version=agent.theory_consolidation['candidates'][identifier]['version'])
                question = agent._open_question({'id': 'post_evaluation_method_use', 'kind': 'method_use',
                    'question': 'When should this retained research method be used after its actual paid returns?',
                    'answer_type': 'research_action', 'parent_id': 'task', 'queue': 'theory',
                    'parent_question_ref': {'question_id': 'task', 'meaning_version': 1},
                    'mission': deepcopy(mission), 'theory_target': deepcopy(record.get('theory_target', {})),
                    'use_sites': [{'kind': 'select_research_action', 'parent_id': 'task'}],
                    'method_use_plan': {'candidate_id': identifier, 'binding': binding},
                    'agenda_basis': {'actual_paid_return': deepcopy(selected)}})
                result['research_route'] = agent._execute('run', {'operator': 'research_route',
                    'research_question': {'id': question['id']}, 'allocation': {
                        'compute': COMPUTE_CAP - agent.costs['compute_cost']}}, laboratory if same_profile else None)
                result['condition_route_same_pre_call_profile'] = same_profile
            with timing.phase('ordinary_gate_readout'):
                after = agent.theory_consolidation['candidates']
                current = after[identifier]
                if set(before) != set(after) or any(before[key] != after[key] for key in before if key != identifier):
                    raise AssertionError('feedback must preserve the complete strong library')
                changed = {'version', 'previous_version_ref', 'method_use_conditions', 'trigger'}
                if {k:v for k,v in old.items() if k not in changed} != {
                        k:v for k,v in current.items() if k not in changed}:
                    raise AssertionError('condition feedback must not silently revise the method body')
                decisions = []
                for case, record, old_decision in zip(selected_cases, records, previous_decisions):
                    new_decision = _decision(agent, current, record)
                    old_admission = old_decision.get('admission') if old_decision else None
                    new_admission = new_decision.get('admission') if new_decision else None
                    item = {'physical_case': case['physical_case'],
                        'actual_use_ref': case['selected_actual_use']['feedback_ref'],
                        'old_ordinary_admission': old_decision, 'new_ordinary_admission': new_decision,
                        'admission_changed': old_admission != new_admission,
                        'scope': 'same_paid_profile_ordinary_transfer_gate_not_an_executed_world_research'}
                    if new_admission == 'defer':
                        laboratory, same_profile = _public_paid_scene(agent, record, case)
                        item['ordinary_defer_same_pre_call_profile'] = same_profile
                        if same_profile:
                            invoked = record.get('invocation_binding') or record['evidence_basis']
                            item['ordinary_common_action'] = agent._execute('run', {
                                'operator': 'theory_transfer', 'candidate_id': identifier,
                                'theory_id': f"post-feedback-defer:{case['case_id']}",
                                'research_question': {'id': 'task'},
                                'mapping': deepcopy(invoked['mapping']),
                                'role_signature': deepcopy(invoked['role_signature']),
                                'output_unit': invoked['output_unit'],
                                'context_ref': deepcopy(invoked['context_ref']),
                                'context_family_ref': current['scope']['context_ref'],
                                'research_program_fitting': {'validation_records': deepcopy(
                                    invoked['fit_split']['requested_validation_records'])},
                                'allocation': {'compute': 0}}, laboratory)
                            item['ordinary_defer_executed'] = item['ordinary_common_action']['status'] == 'method_use_deferred'
                    decisions.append(item)
                unfinished = result['research_route']['status'] in {'scoped_unknown', 'research_deferred'}
                result.update(status='feedback_recorded_condition_research_incomplete' if unfinished
                        else 'completed_feedback_only', ordinary_decisions=decisions,
                    library_candidate_count=len(after), old_condition_version=old['version'],
                    new_condition_version=current['version'],
                    learned_library_ref=ref(agent.workspace / 'shared/theory_library.json'))
    except Exception as error:
        result.update(status=('time_exhausted' if isinstance(error, ActivityExecutionDeadline)
                              else 'failed_or_budget_interrupted'),
            error={'type': type(error).__name__, 'message': str(error)})
    finally:
        clocks = timing.result()
        result['timing'] = clocks
        result['cumulative_stage1_costs'] = {
            'compute_cost': agent.costs['compute_cost'] if agent else paid_compute,
            'native_host_wall_seconds': paid_host + clocks['wall_seconds'],
            'compute_cap': COMPUTE_CAP, 'native_host_cap_seconds': NATIVE_HOST_CAP,
            'external_python_execution_seconds': prior['costs']['python_execution_seconds'],
            'engine_quote_budget_scope': 'independent_not_borrowed'}
        result['new_costs'] = {'compute_cost': result['cumulative_stage1_costs']['compute_cost'] - paid_compute,
            'native_host_wall_seconds': clocks['wall_seconds'], 'external_python_execution_seconds': 0}
        if agent:
            result['actual_agent_costs'] = deepcopy(agent.costs)
        write_json_checkpoint(output / 'summary.json', result, ensure_ascii=False, indent=None)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--final-summary', default=DEFAULT_FINAL)
    parser.add_argument('--original-root', default=DEFAULT_ORIGINAL)
    parser.add_argument('--manifest-output', required=True)
    parser.add_argument('--execute-output')
    args = parser.parse_args()
    manifest = prepare_manifest(args.final_summary, args.original_root)
    write_json_checkpoint(args.manifest_output, manifest, ensure_ascii=False, indent=None)
    if args.execute_output:
        result = run_feedback(manifest, args.execute_output)
        print(result['status'], result['new_costs'])
    else:
        print('prepared_not_executed', len(manifest['selected_actual_use_refs']))


if __name__ == '__main__':
    main()
