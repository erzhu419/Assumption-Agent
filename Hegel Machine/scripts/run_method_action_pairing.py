"""Registered v8 action pairs through existing Common and procedure execution.

No default restore/evaluator factories: importing or quoting cannot launch
scientific work. Four cells are development contrasts, never four sources.
"""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import gc
from math import isfinite
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import weakref

from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.temporal_paired_method import execute_procedure


GLOBAL_COMPONENTS = ('prepare', 'projection_proposals', 'D1_analysis', 'Qeval_predictions',
    'compilation_installation', 'ordinary_use', 'feedback_preservation')
CELL_COMPONENTS = ('search', 'export', 'complete_comparison', 'returned_parent')
SCOPE_FIELDS = ('paid_rows', 'pool_models', 'max_exports')
METHOD_UNIT_PHASES = ('before_turn_hooks', 'procedure_including_hooks_and_checkpoint',
    'after_turn_hooks', 'checkpoint', 'final_comparison', 'freeze', 'final_save',
    'exception_rescue')


def comparison_cells(registration):
    if registration.get('comparison_kind') == 'three_method_same_data':
        return ('00', '10', '20')
    return ('00', '10', '01', '11') if registration.get('D1', {}).get('legal') is True else ('00', '10')


def _comparison_pairs(cells):
    if tuple(cells) == ('00', '10', '20'):
        return (('00', '10'), ('00', '20'), ('10', '20'))
    return (('00', '10'), ('01', '11')) if len(cells) == 4 else (('00', '10'),)


def _retrospective_outcomes(registration):
    """Reuse ended arm receipts; never reinterpret exposed labels as a new trial."""
    refs = registration.get('retained_outcome_refs')
    if refs is None:
        if registration.get('retrospective_completion'):
            raise ValueError('retrospective completion requires its retained arm references')
        return {}
    if (registration.get('retrospective_completion') is not True
            or registration.get('comparison_kind') != 'three_method_same_data'
            or len(refs) != 2 or not set(refs) <= set(comparison_cells(registration))
            or registration.get('continuation')):
        raise ValueError('retained completed arms require explicit three-method retrospective completion')
    from hegel_machine.cognition.method_action_research import read_method_reference
    from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
    old = read_method_reference(registration['retained_registration_ref'], Path.cwd())
    results = {}
    identity = ('method_ref', 'start_state_ref', 'data_ref', 'split_ref',
        'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')
    conditions = (*identity, 'scope', 'compute_cap', 'observation_cap', 'time_limits',
        'return_contract', 'max_steps', 'selection_plan', 'selection_ref')
    for name, reference in refs.items():
        current = registration['cells'][name]
        matches = [(key, cell) for key, cell in old['cells'].items()
                   if cell.get('method_ref') == current['method_ref']]
        if len(matches) != 1:
            raise ValueError('retained method must identify one original registered arm')
        old_name, original = matches[0]
        outcome = read_method_reference(reference, Path.cwd())
        if any(original.get(key) != current.get(key) for key in conditions):
            raise ValueError('retrospective retained arm changed its original research contract')
        if any(outcome.get(key) != original.get(key) for key in identity):
            raise ValueError('retained outcome does not belong to its original method and scene')
        if (not outcome.get('execution_complete') or outcome.get('coverage', 0) <= 0
                or not isinstance(outcome.get('loss'), (int, float))
                or not isfinite(outcome['loss']) or 'completed' not in outcome):
            raise ValueError('retrospective completion requires completed measured old outcomes')
        frozen = outcome['frozen_model']
        manifest = read_json_checkpoint(Path(reference['file']).parent.parent / 'frozen_outputs.json')
        if (manifest.get('models', {}).get(old_name) != frozen
                or not (Path(frozen['workspace']) / frozen['model']).is_file()
                or not manifest.get('evaluation_labels_opened')):
            raise ValueError('retained evaluated arm requires its original frozen file and manifest')
        results[name] = outcome
    return results


@contextmanager
def measured_execution_timing(meter=None):
    """Observe new activity without changing its scientific fee or old ledger."""
    from hegel_machine.cognition.proposers import _EXECUTION_TIMING
    if meter is None:
        meter = {'direct_external_python_seconds': 0., 'direct_external_python_calls': 0,
            'native_dispatch_seconds': 0., 'native_host_seconds': 0., 'native_calls': 0}
    token = _EXECUTION_TIMING.set(meter)
    try:
        yield meter
    finally:
        _EXECUTION_TIMING.reset(token)


def execution_timing_fields(meter, activity_seconds):
    """Disjoint parent wall intervals; parallel worker totals stay separate."""
    native = meter['native_host_seconds']
    external = meter['direct_external_python_seconds']
    return {'native_host_seconds': native, 'native_dispatch_seconds': meter['native_dispatch_seconds'],
        'native_calls': meter['native_calls'], 'external_python_seconds': external,
        'direct_external_python_calls': meter['direct_external_python_calls'],
        'other_host_activity_seconds': max(0., activity_seconds - native - external),
        'native_host_scope': 'Common native dispatch wall excluding nested direct Python executors; '
            'includes native export, return and persistence; not CPU-core seconds',
        'external_python_scope': 'direct parent subprocess executor wall including launch and cleanup; '
            'child-worker cumulative intervals are not parent elapsed wall',
        'clock_scope': 'new parent monotonic wall; native excludes nested direct executor intervals; '
            'other host includes preparation, preservation, library and laboratory work outside that dispatch; '
            'parallel worker cumulative durations are separate and not added to parent wall',
        'historical_clocks_reconstructed': False}


def _pair_continuation(registration):
    """Read the ended, no-action step pair without reopening its large state."""
    continuation = registration.get('continuation')
    if continuation is None:
        return None
    from hegel_machine.cognition.method_action_research import read_method_reference
    from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint

    if continuation.get('reuse_frozen_cells') == ['10']:
        return _paid_method_continuation(registration)
    if (registration.get('comparison_kind') != 'additional_research_vs_retained_parent'
            or continuation.get('reuse_frozen_cells') != ['00']):
        raise ValueError('only the ended retained-parent step pair can continue here')
    old = read_method_reference(continuation['registration_ref'], Path.cwd())
    if old.get('continuation') or old.get('comparison_kind') != registration['comparison_kind']:
        raise ValueError('continuation must name the original retained-parent registration')
    if (old['source_scene'] != registration['source_scene']
            or old['components'] != registration['components']
            or old.get('physical_source_ref') != registration.get('physical_source_ref')):
        raise ValueError('continuation changed the original source, evaluation or component quote')
    outcomes = {name: read_method_reference(continuation['outcome_refs'][name], Path.cwd())
                for name in ('00', '10')}
    unchanged = ('method_path', 'method_ref', 'start_state_ref', 'data_ref', 'split_ref',
        'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref', 'return_contract',
        'scope', 'components', 'execution_role', 'retained_parent', 'selection_plan',
        'method_source_refs')
    for name, outcome in outcomes.items():
        original, current = old['cells'][name], registration['cells'][name]
        if any(original.get(key) != current.get(key) for key in unchanged):
            raise ValueError('continuation changed the original cell or method contract')
        if any(outcome.get(key) != original[key] for key in (
                'method_ref', 'start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')):
            raise ValueError('continuation outcome does not belong to its original cell')
        for field, paid in (('compute_cap', 'compute_cost'), ('observation_cap', 'observation_cost')):
            if current[field] > original[field] - outcome['costs'][paid]:
                raise ValueError('continuation exceeds the original remaining account')
        for field, spent in (
                ('billed_python_seconds', outcome['costs']['python_execution_seconds']),
                ('active_wall_seconds', outcome['timing']['activity_wall_seconds'])):
            if current['time_limits'][field] > original['time_limits'][field] - spent:
                raise ValueError('continuation exceeds the original remaining time')
    research = outcomes['10']
    if (research.get('execution_complete') or research.get('actual_action_content')
            or research['timing']['native_calls'] != 0
            or research['costs']['compute_cost'] or research['costs']['observation_cost']):
        raise ValueError('this continuation requires the original no-action execution failure')
    # Each failed selector query returned no proposal and made one external
    # call. They are already paid procedure steps, not new available steps.
    spent_steps = research['timing']['direct_external_python_calls']
    remaining_steps = continuation['remaining_steps']['10']
    if (type(remaining_steps) is not int or remaining_steps <= 0
            or remaining_steps > old['cells']['10']['max_steps'] - spent_steps):
        raise ValueError('continuation resets the original paid procedure steps')
    baseline = outcomes['00']
    if (not baseline.get('execution_complete') or baseline.get('execution_role') == 'research'
            or baseline.get('method_delivery') != 'registered_retained_parent_baseline'
            or baseline.get('retained_parent_declaration') != old['cells']['00']['retained_parent']):
        raise ValueError('the original literal retained parent was not frozen')
    frozen = baseline['frozen_model']
    if not (Path(frozen['workspace']) / frozen['model']).is_file():
        raise ValueError('the retained parent frozen file is missing')
    study = Path(continuation['outcome_refs']['00']['file']).parent.parent
    manifest = read_json_checkpoint(study / 'frozen_outputs.json')
    if (manifest.get('models', {}).get('00') != frozen
            or manifest.get('evaluation_labels_opened')
            or manifest.get('evaluation_lifecycle', {}).get('labels_revealed')):
        raise ValueError('the original evaluation was exposed or its frozen parent changed')
    return {'registration': old, 'outcomes': outcomes}


def _paid_method_continuation(registration):
    """Continue the paid, incomplete P0 while its completed P1 waits for Qeval."""
    from hegel_machine.cognition.method_action_research import read_method_reference
    from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint

    continuation = registration['continuation']
    old = read_method_reference(continuation['registration_ref'], Path.cwd())
    if (old.get('comparison_kind') is not None
            or registration.get('comparison_kind') is not None
            or not continuation.get('execution_repair')):
        raise ValueError('paid method continuation needs its original pair and execution repair')
    prior = None
    if old.get('continuation'):
        if old['continuation'].get('reuse_frozen_cells') != ['10']:
            raise ValueError('paid method continuation must retain the same completed P1')
        prior = _paid_method_continuation(old)
    for key in ('paid_registration_ref', 'initial_scene_ref', 'physical_source_ref',
                'question', 'finite_native_author', 'paid_scene', 'split', 'quality_contract',
                'resource_contract', 'evaluation', 'static_selection_ref', 'method_ref'):
        if old.get(key) != registration.get(key):
            raise ValueError('continuation changed the original scientific pair')
    outcomes = {name: read_method_reference(continuation['outcome_refs'][name], Path.cwd())
                for name in ('00', '10')}
    unchanged = ('method_path', 'method_ref', 'start_state_ref', 'data_ref', 'split_ref',
        'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref', 'return_contract',
        'selection_plan', 'method_source_refs', 'max_steps')
    for name, outcome in outcomes.items():
        original, current = old['cells'][name], registration['cells'][name]
        if any(original.get(key) != current.get(key) for key in unchanged):
            raise ValueError('continuation changed the original cell or method contract')
        if any(outcome.get(key) != original[key] for key in (
                'method_ref', 'start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')):
            raise ValueError('continuation outcome does not belong to its original cell')
        for field, paid in (('compute_cap', 'compute_cost'), ('observation_cap', 'observation_cost')):
            if current[field] > original[field] - outcome['costs'][paid]:
                raise ValueError('continuation exceeds the original remaining account')
        for field, spent in (
                ('billed_python_seconds', outcome['costs']['python_execution_seconds']),
                ('active_wall_seconds', outcome['timing']['activity_wall_seconds'])):
            if current['time_limits'][field] > original['time_limits'][field] - spent:
                raise ValueError('continuation exceeds the original remaining time')
        scope, prior_scope = current['scope'], original['scope']
        if (scope['paid_rows'] != prior_scope['paid_rows']
                or scope['pool_models'] != outcome['continuation_scope']['pool_models']
                or scope['pool_models'] + scope['max_exports'] >
                    prior_scope['pool_models'] + prior_scope['max_exports']):
            raise ValueError('continuation changed the paid data or increased the candidate capacity')
    research, completed = outcomes['00'], outcomes['10']
    paid_incomplete = (research.get('native_research_complete') is False
        and research.get('registered_attempt_returned') and research.get('actual_action_refs')
        and research['costs']['compute_cost'] > 0)
    stopped_before_research = (prior is not None and research.get('status') == 'method_stopped'
        and research['timing']['native_calls'] == 0
        and research['costs']['compute_cost'] == research['costs']['observation_cost'] == 0)
    if research.get('execution_complete') or not (paid_incomplete or stopped_before_research):
        raise ValueError('P0 must be an actual paid incomplete native research attempt')
    spent_steps = research.get('procedure_steps_consumed')
    if spent_steps is None:
        # The original paid attempt has one research action. The already saved
        # follow-up stopped in its selector, so that paid call is also a step.
        spent_steps = (research['timing']['direct_external_python_calls'] if stopped_before_research
                       else len(research['actual_action_refs']))
    prior_steps = (old['continuation']['remaining_steps']['00'] if prior is not None
                   else old['cells']['00']['max_steps'])
    remaining_steps = continuation['remaining_steps']['00']
    if (type(remaining_steps) is not int or remaining_steps <= 0
            or remaining_steps > prior_steps - spent_steps):
        raise ValueError('continuation resets the original paid procedure steps')
    if not completed.get('execution_complete') or completed.get('native_research_complete') is not True:
        raise ValueError('P1 must have completed its original research')
    for key in ('search', 'export', 'complete_comparison', 'returned_parent'):
        if registration['cells']['10']['components'][key] != {'compute': 0, 'observations': 0}:
            raise ValueError('the frozen P1 must not repeat research or comparison')
    frozen = completed['frozen_model']
    if not (Path(frozen['workspace']) / frozen['model']).is_file():
        raise ValueError('the completed P1 frozen file is missing')
    study = Path(continuation['outcome_refs']['10']['file']).parent.parent
    manifest = read_json_checkpoint(study / 'frozen_outputs.json')
    if (manifest.get('models', {}).get('10') != frozen or manifest.get('evaluation_labels_opened')
            or manifest.get('evaluation_lifecycle', {}).get('labels_revealed')):
        raise ValueError('the original evaluation was exposed or its frozen P1 changed')
    summary = read_json_checkpoint(study / 'summary.json')
    shared_paid = (old.get('preparation_costs', {}).get('compute_cost', 0)
        + summary.get('ordinary_use', {}).get('new_compute_cost', 0))
    if (registration['compute_cap'] + shared_paid
            + sum(outcome['costs']['compute_cost'] for outcome in outcomes.values()) > old['compute_cap']):
        raise ValueError('continuation resets the original total compute account')
    return {'registration': old,
            'original_registration': prior['original_registration'] if prior is not None else old,
            'outcomes': outcomes, 'reuse_frozen_cells': ['10'],
            'execution_repair': deepcopy(continuation['execution_repair'])}


def _continuation_costs(outcome, previous):
    """Keep previous failures separate while reporting the original total."""
    outcome['new_stage_costs'] = deepcopy(outcome['costs'])
    outcome['prior_stage_costs'] = deepcopy(previous['costs'])
    cumulative = previous.get('cumulative_original_stage_costs', previous['costs'])
    outcome['prior_cumulative_original_stage_costs'] = deepcopy(cumulative)
    outcome['cumulative_original_stage_costs'] = {
        key: cumulative[key] + value for key, value in outcome['costs'].items()}


def _reuse_retained_parent(registration, previous, output):
    """Carry its literal frozen model and small account into pending Qeval."""
    from hegel_machine.cognition.method_action_research import read_method_reference
    from hegel_machine.cognition.models import Observation
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig

    contract = registration['cells']['00']
    plan = read_method_reference(registration['source_scene']['plan_ref'], Path.cwd())
    costs = deepcopy(previous['costs_after'])
    config = replace(InquiryLLMConfig(**plan['limits']),
        compute_budget=costs['compute_cost'] + contract['compute_cap'],
        observation_budget=costs['observation_cost'] + contract['observation_cap'],
        execution_seconds=costs['python_execution_seconds'] + contract['time_limits']['billed_python_seconds'],
        max_model_calls=0, token_budget=0)
    paid = read_method_reference(contract['data_ref'], Path.cwd())
    rows = tuple(Observation(dict(row['features']), row['target'],
        tuple(dict(past) for past in row['history'])) for record in paid for row in record['rows'])
    if len(rows) != contract['scope']['paid_rows']:
        raise ValueError('the frozen parent paid data changed before continuation')
    account = SimpleNamespace(costs=costs, config=config, workspace=Path(output), rows=rows)
    outcome = deepcopy(previous)
    outcome.update(costs={key: 0. if key == 'python_execution_seconds' else 0 for key in costs},
        costs_after=deepcopy(costs), prior_stage_timing=deepcopy(previous['timing']),
        retained_frozen_outcome_ref=deepcopy(registration['continuation']['outcome_refs']['00']),
        timing={'activity_wall_seconds': 0., 'recovery_seconds': 0.,
            'rescue_preservation_seconds': 0., 'billed_python_seconds': 0.,
            **execution_timing_fields({'native_host_seconds': 0., 'native_dispatch_seconds': 0.,
                'native_calls': 0, 'direct_external_python_seconds': 0.,
                'direct_external_python_calls': 0}, 0.)},
        host_phase_timing=ExecutionTiming(METHOD_UNIT_PHASES).result(),
        continuation_costs_ref={'file': str(Path(output) / 'outcome.json'), 'json_path': ['costs_after']})
    _continuation_costs(outcome, previous)
    return account, outcome


def _reuse_completed_method(registration, previous, output):
    """Retain the completed P1 file and ledger; only its pending readout may run."""
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig

    contract = registration['cells']['10']
    costs = deepcopy(previous['costs_after'])
    config = replace(InquiryLLMConfig(), compute_budget=costs['compute_cost'] + contract['compute_cap'],
        observation_budget=costs['observation_cost'] + contract['observation_cap'],
        execution_seconds=costs['python_execution_seconds'] + contract['time_limits']['billed_python_seconds'],
        max_model_calls=0, token_budget=0)
    account = SimpleNamespace(costs=costs, config=config, workspace=Path(output), rows=())
    outcome = deepcopy(previous)
    outcome.update(costs={key: 0. if key == 'python_execution_seconds' else 0 for key in costs},
        costs_after=deepcopy(costs), prior_stage_timing=deepcopy(previous['timing']),
        retained_frozen_outcome_ref=deepcopy(registration['continuation']['outcome_refs']['10']),
        timing={'activity_wall_seconds': 0., 'recovery_seconds': 0.,
            'rescue_preservation_seconds': 0., 'billed_python_seconds': 0.,
            **execution_timing_fields({'native_host_seconds': 0., 'native_dispatch_seconds': 0.,
                'native_calls': 0, 'direct_external_python_seconds': 0.,
                'direct_external_python_calls': 0}, 0.)},
        host_phase_timing=ExecutionTiming(METHOD_UNIT_PHASES).result(),
        continuation_costs_ref={'file': str(Path(output) / 'outcome.json'), 'json_path': ['costs_after']})
    _continuation_costs(outcome, previous)
    return account, outcome


def quote_method_action_pairing(registration, *, current_scopes=None):
    """Validate a complete component quote; missing costs never become zero."""
    missing, compute, observations = [], 0, 0
    data_role = registration.get('data_role', 'development')
    if data_role not in {'development', 'condition_validation', 'independent_successor'}:
        missing.append('registered_method_data_role_required')
    cells = comparison_cells(registration)
    if len(cells) == 3 and (data_role != 'independent_successor'
            or registration.get('D1', {}).get('legal') is True):
        missing.append('three_methods_require_one_independent_successor_data_pool')
    retained_control = registration.get('comparison_kind') == 'additional_research_vs_retained_parent'
    if retained_control and len(cells) != 2:
        missing.append('retained_parent_comparison_requires_literal_two_cell_scene')
    if len(cells) == 4 and not registration.get('D1', {}).get('proposal_ref'):
        missing.append('D1.proposal_ref')
    for owner, names, components in [('global', GLOBAL_COMPONENTS, registration.get('components', {})),
        *((name, CELL_COMPONENTS, registration.get('cells', {}).get(name, {}).get('components', {}))
          for name in cells)]:
        for name in names:
            value = components.get(name, {})
            for resource in ('compute', 'observations'):
                number = value.get(resource)
                if type(number) is not int or number < 0:
                    missing.append(f'{owner}.{name}.{resource}')
                elif resource == 'compute':
                    compute += number
                else:
                    observations += number
    scopes, capacities = {}, {}
    qeval_compute = registration.get('components', {}).get('Qeval_predictions', {}).get('compute')
    evaluation_allowance = ((qeval_compute + len(cells) - 1) // len(cells)
        if type(qeval_compute) is int and qeval_compute >= 0 else None)
    for name in cells:
        cell = registration.get('cells', {}).get(name, {})
        scope = cell.get('scope') or {}
        scopes[name] = deepcopy(scope)
        for field in SCOPE_FIELDS:
            if type(scope.get(field)) is not int or scope[field] < 0:
                missing.append(f'{name}.scope.{field}')
        if current_scopes is not None and current_scopes.get(name) != scope:
            missing.append(f'{name}.quote_scope_changed_requires_new_quote')
        for field in ('method_path', 'start_state_ref', 'data_ref', 'split_ref',
                      'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref', 'return_contract', 'method_ref'):
            if cell.get(field) is None:
                missing.append(f'{name}.{field}')
        method = cell.get('method_ref') or {}
        if not method.get('id') or type(method.get('version')) is not int or method['version'] < 1:
            missing.append(f'{name}.exact_executable_method_version_required')
        if cell.get('return_contract') not in {'retained_parent', 'method_declared'}:
            missing.append(f'{name}.original_return_contract_required')
        baseline = cell.get('execution_role') == 'retained_parent_baseline'
        if baseline:
            declaration = cell.get('retained_parent') or {}
            if not retained_control or name != '00' or cell.get('return_contract') != 'retained_parent':
                missing.append(f'{name}.retained_parent_baseline_requires_registered_narrow_comparison')
            if not isinstance(declaration.get('model'), str) or not declaration['model']:
                missing.append(f'{name}.literal_retained_parent_model_required')
            for key in ('source_ref', 'author_work_ref'):
                if not isinstance(declaration.get(key), dict) or not declaration[key].get('file'):
                    missing.append(f'{name}.retained_parent.{key}')
            if declaration.get('research_status') != 'author_work_required':
                missing.append(f'{name}.original_author_work_status_required')
            for key in ('search', 'export', 'complete_comparison'):
                if cell.get('components', {}).get(key) != {'compute': 0, 'observations': 0}:
                    missing.append(f'{name}.retained_parent_does_not_repeat_{key}')
        elif retained_control and name == '00':
            missing.append('00.registered_retained_parent_baseline_required')
        if retained_control and name == '10' and cell.get('execution_role', 'research') != 'research':
            missing.append('10.registered_research_required')
        for field in ('billed_python_seconds', 'active_wall_seconds'):
            value = cell.get('time_limits', {}).get(field)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                missing.append(f'{name}.time_limits.{field}')
        capacities[name] = {field: cell.get(field) for field in ('compute_cap', 'observation_cap')}
        for field, resource in (('compute_cap', 'compute'), ('observation_cap', 'observations')):
            cap = cell.get(field)
            prices = [cell.get('components', {}).get(key, {}).get(resource) for key in CELL_COMPONENTS]
            if type(cap) is not int or cap < 0:
                missing.append(f'{name}.{field}')
            elif all(type(price) is int and price >= 0 for price in prices):
                reserve = evaluation_allowance if resource == 'compute' else 0
                if reserve is not None and cap < sum(prices) + reserve:
                    missing.append(f'{name}.{field}.complete_components_exceed_cap')
    for left, right in _comparison_pairs(cells):
        a, b = (registration.get('cells', {}).get(name, {}) for name in (left, right))
        if any(a.get(field) != b.get(field) for field in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')):
            missing.append(f'{left}/{right}.matched_contracts_required')
        if any(a.get(field) != b.get(field) for field in ('compute_cap', 'observation_cap', 'time_limits')):
            missing.append(f'{left}/{right}.equal_resource_caps_and_time_limits_required')
    if len(cells) == 4:
        if registration['cells']['00']['data_ref'] == registration['cells']['01']['data_ref']:
            missing.append('D1.distinct_declared_data_required')
    # Component prices describe the actual algorithms. Capacity reservations
    # give both methods the same opportunity, even when those prices differ.
    # Per-model Qeval compute is already inside each cell's cap, not a second
    # global reservation. Its shared observation acquisition remains global.
    capacities_known = all(type(cap) is int and cap >= 0
        for values in capacities.values() for cap in values.values())
    global_components = registration.get('components', {})
    use = registration.get('ordinary_use')
    if retained_control and (use is not None or
            global_components.get('ordinary_use') != {'compute': 0, 'observations': 0}):
        missing.append('retained_parent_control_does_not_authorize_ordinary_method_use')
    if use is not None:
        reservation = global_components.get('ordinary_use', {})
        for field, resource in (('compute_cap', 'compute'), ('observation_cap', 'observations')):
            cap, reserved = use.get(field), reservation.get(resource)
            if (type(cap) is not int or cap < 0 or type(reserved) is not int
                    or cap > reserved):
                missing.append(f'ordinary_use.{field}.exceeds_registered_reservation')
    shared_compute = [global_components.get(key, {}).get('compute')
        for key in GLOBAL_COMPONENTS if key != 'Qeval_predictions']
    shared_observations = [global_components.get(key, {}).get('observations') for key in GLOBAL_COMPONENTS]
    reserved_compute = (sum(value['compute_cap'] for value in capacities.values()) + sum(shared_compute)
        if capacities_known and all(type(value) is int and value >= 0 for value in shared_compute) else None)
    reserved_observations = (sum(value['observation_cap'] for value in capacities.values()) + sum(shared_observations)
        if capacities_known and all(type(value) is int and value >= 0 for value in shared_observations) else None)
    for name, total in (('compute_cap', reserved_compute), ('observation_cap', reserved_observations)):
        cap = registration.get(name)
        if total is None or type(cap) not in (int, float) or not isfinite(cap) or cap < total:
            missing.append(f'{name}.complete_quote_not_admitted')
    if registration.get('continuation') is not None:
        try:
            _pair_continuation(registration)
        except (KeyError, OSError, ValueError, TypeError) as error:
            missing.append(f'continuation.{error}')
    try:
        _retrospective_outcomes(registration)
    except (KeyError, OSError, ValueError, TypeError) as error:
        missing.append(f'retrospective_completion.{error}')
    return {'status': 'quoted' if not missing else 'quote_incomplete', 'missing': missing,
        'cells': list(cells), 'compute': reserved_compute if not missing else None,
        'observations': reserved_observations if not missing else None, 'scopes': scopes,
        'component_compute': compute if not missing else None,
        'component_observations': observations if not missing else None,
        'cell_caps': capacities,
        'component_basis': deepcopy(registration.get('components')),
        'cell_component_basis': {name: deepcopy(registration.get('cells', {}).get(name, {}).get('components'))
                                 for name in cells},
        'comparison_kind': registration.get('comparison_kind'),
        **({'data_role': data_role} if 'data_role' in registration else {}),
        'cell_return_basis': {name: {key: deepcopy(registration['cells'][name].get(key))
            for key in ('return_contract', 'execution_role', 'retained_parent')} for name in cells},
        **({'continuation': deepcopy(registration['continuation'])}
           if registration.get('continuation') is not None else {}),
        **({'retrospective_completion': True,
            'retained_outcome_refs': deepcopy(registration.get('retained_outcome_refs')),
            'retained_registration_ref': deepcopy(registration.get('retained_registration_ref'))}
           if registration.get('retrospective_completion') else {}),
        'execution_authorized': False, 'cold_upper_is_actual_spend': False}


def _save(path, value):
    path = Path(path)
    write_json_checkpoint(path, value, ensure_ascii=False, indent=None)
    return {'file': str(path), 'json_path': []}


def _registered_retained_parent(agent, contract):
    """Bind the declared control to the restored incumbent before any research."""
    declared = contract['retained_parent']
    current = agent._evaluated_summary() or {}
    if current.get('model') != declared['model']:
        raise ValueError('the restored current parent differs from the registered retained baseline')
    return current


def _retained_control_research_return(agent, result):
    """Check this structural unit's actual paid return, not its attempt marker."""
    checked = []
    for reference in result.get('native_boundary_refs', []):
        location = reference['json_path']
        if not location or location[0] not in {'operator_results', 'method_results'}:
            continue
        feedback = getattr(agent, location[0])
        for key in location[1:]:
            feedback = feedback[key]
        representation = _representation_attempt_return(feedback, reference)
        if representation is not None:
            checked.append(representation)
            continue
        work = feedback.get('native_condition_research') or {}
        discrimination = work.get('structural_discrimination') or {}
        verdict = discrimination.get('forecast_verdict') or {}
        joint = verdict.get('next_research', {}).get('kind') == 'jointly_refit_frozen_structures'
        status = work.get('status', feedback.get('status'))
        complete = bool(verdict.get('complete') and not feedback.get('error') and not work.get('error'))
        if joint:
            complete = (complete and status in {'candidates_require_evaluation', 'matched_evidence_without_candidate'}
                and discrimination.get('refit_return', {}).get('status') in {
                    'candidate_supported_on_paid_rows', 'no_supported_candidate_in_completed_finite_attempt'}
                and not any(stage.get('status') == 'computation_incomplete'
                    for stage in work.get('search_stages', [])))
        else:
            complete = complete and status == 'frozen_predictions_compared'
        checked.append({'source_ref': deepcopy(reference), 'status': status,
            'frozen_readout_complete': verdict.get('complete') is True,
            'joint_refit_required': joint, 'complete': bool(complete)})
    return {'complete': bool(checked) and all(item['complete'] for item in checked),
        'actual_returns': checked,
        'scope': 'Completed frozen readout and any required joint attempt; a native attempt alone is insufficient.'}


def _freeze(agent, model, output):
    from hegel_machine.cognition.code_agent import _copy_model_context
    destination = Path(output) / '_evaluated_model'
    inner = _copy_model_context(agent.workspace, model, destination)
    return {'workspace': str(destination), 'model': inner}


class _NativeUnitReturned(Exception):
    """Leave the existing procedure at its registered one-attempt boundary."""


def _representation_attempt_return(feedback, reference):
    """Use the route's paid receipt to distinguish an attempt from delivery."""
    if not feedback.get('native_representation_ref'):
        return None
    work = feedback.get('native_representation') or {}
    failed = bool(feedback.get('error') or work.get('error')
                  or work.get('status') == 'native_representation_execution_failed')
    complete = (not failed and work.get('comparison_completed') is True
        and work.get('research_completion') == 'bounded_search_completed'
        and work.get('status') in {'native_representation_draft_selected',
                                  'native_representation_parent_retained'})
    return {'source_ref': deepcopy(reference),
        'native_result_ref': deepcopy(feedback['native_representation_ref']),
        'status': work.get('status', feedback.get('status')),
        'research_completion': work.get('research_completion'),
        'comparison_completed': work.get('comparison_completed') is True,
        'failed': failed, 'complete': complete}


def execute_registered_method_unit(agent, lab, path, *, max_steps=48, before_step=None,
                                   after_step=None, checkpoint=None, native_attempt_limit=None,
                                   research_compute_ceiling=None, research_compute_reserve=0,
                                   execution_timing=None):
    """Compose ordinary hooks and stop after the registered Q→native attempt.

    The caller still performs the common complete comparison and parent return.
    This boundary is an execution contract, never learned scientific content.
    """
    if native_attempt_limit not in (None, 1):
        raise ValueError('the registered action unit permits one native attempt')
    timing = execution_timing if execution_timing is not None else ExecutionTiming(METHOD_UNIT_PHASES)
    found, program_indices, representation_returns = [], [], []
    program_start, program_operator_start = None, None
    completed_steps = 0

    def collect(feedback, location):
        reference = {'file': str(agent.workspace / 'research_state.json'), 'json_path': location}
        representation = _representation_attempt_return(feedback, reference)
        if representation is not None:
            found.append(reference)
            representation_returns.append(representation)
        elif (feedback.get('operator') == 'native_condition_research'
                and 'structure_planning' not in feedback):
            found.append(reference)

    def inspect():
        if native_attempt_limit is None or program_operator_start is None:
            return
        for index, entry in enumerate(agent.operator_results[program_operator_start:], program_operator_start):
            collect(entry, ['operator_results', index])
        if found:
            return
        # Query callbacks retain their own actual native results even when the
        # surrounding procedure's sole top-level action was a physical query.
        for index in program_indices:
            entry = agent.method_results[index]
            feedback = entry.get('feedback') or {}
            collect(feedback, ['method_results', index, 'feedback'])
            for position, child in enumerate(feedback.get('after_observation_results', [])):
                collect(child, ['method_results', index, 'feedback', 'after_observation_results', position])

    def before(laboratory):
        nonlocal program_start, program_operator_start
        with timing.phase('before_turn_hooks'):
            opened = before_step(laboratory) if before_step else None
        program_start = len(agent.method_results)
        program_operator_start = len(getattr(agent, 'operator_results', []))
        return opened

    def after(laboratory):
        nonlocal completed_steps
        completed_steps += 1
        if program_start is not None:
            program_indices.extend(range(program_start, len(agent.method_results)))
        if after_step:
            with timing.phase('after_turn_hooks'):
                after_step(laboratory)
        if checkpoint:
            with timing.phase('checkpoint'):
                checkpoint(laboratory)
        inspect()
        if found:
            raise _NativeUnitReturned()

    previous_ceiling = getattr(agent, '_unit_research_compute_ceiling', None)
    previous_full_budget = getattr(agent, '_unit_full_compute_budget', None)
    unit, unit_returned = None, False
    if research_compute_ceiling is not None:
        agent._unit_research_compute_ceiling = research_compute_ceiling
        agent._unit_full_compute_budget = agent.config.compute_budget
    try:
        with timing.phase('procedure_including_hooks_and_checkpoint'), agent._python_method_call(path, lab) as invocation:
            begin_unit = getattr(agent, '_begin_registered_method_unit', None)
            if begin_unit is not None:
                origin_ref = (invocation['origin_ref'] if invocation is not None else
                    {'file': str(agent.workspace / 'research_state.json'),
                     'json_path': ['method_results', len(agent.method_results)]})
                unit = begin_unit(path, origin_ref, research_compute_reserve=research_compute_reserve)
            try:
                result = execute_procedure(agent, lab, path, max_steps=max_steps,
                    allow_confirmation=False, record_events=True, before_step=before, after_step=after)
            except _NativeUnitReturned:
                feedback = agent.method_results[-1].get('feedback', {}) if program_indices else {}
                result = {'status': 'method_error' if feedback.get('error') else 'registered_native_attempt_returned',
                    'model': agent.model, 'native_boundary_refs': found}
            if invocation is not None:
                result['method_invocation_origin_ref'] = deepcopy(invocation['origin_ref'])
                result['method_invocation_binding'] = deepcopy(invocation['binding'])
            unit_returned = True
    finally:
        if unit is not None:
            agent._end_registered_method_unit(unit, interrupted=not unit_returned)
        if previous_ceiling is None:
            if hasattr(agent, '_unit_research_compute_ceiling'):
                del agent._unit_research_compute_ceiling
        else:
            agent._unit_research_compute_ceiling = previous_ceiling
        if previous_full_budget is None:
            if hasattr(agent, '_unit_full_compute_budget'):
                del agent._unit_full_compute_budget
        else:
            agent._unit_full_compute_budget = previous_full_budget
    from hegel_machine.cognition.finite_native_author import research_action_semantics
    actions, action_refs = [], []
    for index in program_indices:
        entry = agent.method_results[index]
        proposal = entry.get('proposal') or {}
        action, args = proposal.get('action'), proposal.get('arguments') or {}
        if action is None or action in {'evaluate', 'finish'} or args.get('operator') == 'research_choice':
            continue
        actions.append(research_action_semantics(proposal, entry.get('program_binding')))
        action_refs.append({'file': str(agent.workspace / 'research_state.json'),
            'json_path': ['method_results', index, 'proposal']})
    result.update(actual_action_content=actions, actual_action_refs=action_refs,
        procedure_steps_consumed=completed_steps,
        method_source_path=str(Path(agent.workspace) / path), host_phase_timing=timing.result())
    if native_attempt_limit == 1:
        result['registered_attempt_returned'] = bool(found)
    if representation_returns:
        complete = all(item['complete'] for item in representation_returns)
        result.update(native_representation_returns=representation_returns,
            native_research_complete=complete)
        if not complete:
            result['status'] = ('method_error' if any(item['failed'] for item in representation_returns)
                else 'native_representation_incomplete')
    if unit is not None:
        result['registered_research_unit'] = deepcopy(unit)
    return result


def run_method_action_pairing(registration, output, *, restore_cell, evaluate_frozen, ordinary_use,
                              prepare_evaluation=None):
    """Execute only an admitted frozen pair, with caller-owned science factories.

    restore_cell(name, cell_contract, output) returns agent/lab/path plus its
    actual start/data/split/quality/resource/evaluation refs and measured scope.
    evaluate_frozen(name, frozen, evaluation_ref, account) returns measured
    loss/coverage/completed/quality. The account retains only costs/config/
    workspace and the small paid rows tuple, never a full research agent.
    ordinary_use(pair_refs, outcome_refs, output) passes actual references to the
    existing method question/selection route, including installation/feedback.
    """
    quote = quote_method_action_pairing(registration)
    if (registration.get('execution_authorized') is not True or not registration.get('approval_ref')
            or quote['status'] != 'quoted' or registration.get('quote') != quote):
        raise ValueError('explicit approval and the current complete frozen quote are required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    states, outcomes, frozen, references = {}, {}, {}, {}
    continuation = _pair_continuation(registration)
    retained = _retrospective_outcomes(registration)
    live_agents, workspaces = [], set()
    started = monotonic()
    for name in quote['cells']:
        contract = registration['cells'][name]
        restore_at = monotonic()
        if name in retained:
            outcome = deepcopy(retained[name])
            outcome['retained_outcome_ref'] = deepcopy(registration['retained_outcome_refs'][name])
            outcome['retrospective_completion'] = True
            outcome['new_stage_costs'] = {key: 0 for key in outcome['costs']}
            outcomes[name] = outcome
            frozen[name] = deepcopy(outcome['frozen_model'])
            references[name] = _save(output / name / 'outcome.json', outcome)
            continue
        if continuation is not None and name in continuation.get('reuse_frozen_cells', ['00']):
            retain = _reuse_completed_method if name == '10' else _reuse_retained_parent
            account, outcome = retain(registration,
                continuation['outcomes'][name], output / name)
            outcome['timing']['recovery_seconds'] = monotonic() - restore_at
            states[name] = {'account': account, 'before_costs': deepcopy(account.costs),
                'execution_timing': {'native_host_seconds': 0., 'native_dispatch_seconds': 0.,
                    'native_calls': 0, 'direct_external_python_seconds': 0.,
                    'direct_external_python_calls': 0}}
            outcomes[name] = outcome
            frozen[name] = deepcopy(outcome['frozen_model'])
            references[name] = _save(output / name / 'outcome.json', outcome)
            continue
        scene = restore_cell(name, deepcopy(contract), output / name)
        agent, lab, path = scene['agent'], scene['lab'], scene['path']
        mutable_fields = ('rows', 'knowledge', 'questions', 'theory_index',
            'theory_consolidation', 'method_results', 'observations')
        workspace = str(Path(agent.workspace).resolve())
        shared = False
        for reference in live_agents:
            previous = reference()
            if previous is not None and (agent is previous or any(
                    getattr(agent, field) is getattr(previous, field) for field in mutable_fields)):
                shared = True
            del previous
        if shared or workspace in workspaces:
            raise ValueError('paired arms require independent mutable state, cursors, and workspaces')
        live_agents = [reference for reference in live_agents if reference() is not None]
        live_agents.append(weakref.ref(agent))
        workspaces.add(workspace)
        if any(scene.get(field) != contract[field] for field in ('start_state_ref', 'data_ref', 'split_ref',
                'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')):
            raise ValueError('restored cell differs from its frozen start/data/evaluation contract')
        current = quote_method_action_pairing(registration, current_scopes={
            **quote['scopes'], name: scene.get('scope')})
        if current['status'] != 'quoted':
            raise ValueError('restored N/K changed: re-quote without increasing the old cap')
        recovery = monotonic() - restore_at
        before = deepcopy(agent.costs)
        available = {'compute_cap': agent.config.compute_budget - before['compute_cost'],
            'observation_cap': agent.config.observation_budget - before['observation_cost'],
            'billed_python_seconds': agent.config.execution_seconds - before['python_execution_seconds']}
        if (any(available[field] < contract[field] for field in ('compute_cap', 'observation_cap'))
                or available['billed_python_seconds'] < contract['time_limits']['billed_python_seconds']):
            raise ValueError('restored account cannot supply the registered equal per-cell capacity')
        agent.config = replace(agent.config,
            compute_budget=before['compute_cost'] + contract['compute_cap'],
            observation_budget=before['observation_cost'] + contract['observation_cap'],
            execution_seconds=before['python_execution_seconds'] + contract['time_limits']['billed_python_seconds'])
        entered = monotonic()
        phase_timing = ExecutionTiming(METHOD_UNIT_PHASES)
        from hegel_machine.cognition.experiment_runtime import (
            python_execution_deadline, PythonExecutionDeadline, ActivityExecutionDeadline,
            checkpoint_execution)

        def checkpoint(_laboratory):
            # execute_procedure(on_checkpoint=...) constructs agent.result,
            # which deep-copies the full research history before calling back.
            with checkpoint_execution():
                agent._prepare_program_state()
                _save(agent.workspace / 'training.json', agent.checkpoint_result('researching'))

        choice, model, complete, saved, rescue_seconds = {}, None, False, None, 0.
        native, models = [], []
        try:
            with measured_execution_timing() as meter, python_execution_deadline(
                    contract['time_limits']['active_wall_seconds'], activity=True):
                if contract.get('execution_role') == 'retained_parent_baseline':
                    parent = _registered_retained_parent(agent, contract)
                    model = parent['model']
                    result = {'status': 'retained_parent_baseline_frozen', 'model': model,
                        'method_delivery': 'registered_retained_parent_baseline',
                        'research_execution_complete': False,
                        'research_status': contract['retained_parent']['research_status'],
                        'author_work_ref': deepcopy(contract['retained_parent']['author_work_ref']),
                        'actual_action_content': [], 'actual_action_refs': []}
                    complete = True
                    with phase_timing.phase('freeze'):
                        saved = _freeze(agent, model, output / name / 'frozen')
                else:
                    result = execute_registered_method_unit(agent, lab, path,
                        max_steps=min(contract.get('max_steps', 48),
                            registration['continuation']['remaining_steps'][name])
                            if continuation is not None else contract.get('max_steps', 48),
                        before_step=scene.get('before_step'),
                        after_step=scene.get('after_step'), checkpoint=checkpoint,
                        native_attempt_limit=scene.get('native_attempt_limit'),
                        execution_timing=phase_timing,
                        research_compute_reserve=max(contract.get('planned_native_compute', 0),
                            contract.get('actual_candidate_complete_quote', {}).get('bounded_research_compute', 0)),
                        research_compute_ceiling=(agent.config.compute_budget
                            - contract['components']['complete_comparison']['compute']
                            - contract['components']['returned_parent']['compute']
                            - (registration['components']['Qeval_predictions']['compute']
                                + len(quote['cells']) - 1) // len(quote['cells'])))
                    narrow = registration.get('comparison_kind') == 'additional_research_vs_retained_parent'
                    if narrow:
                        result['research_return_verification'] = _retained_control_research_return(agent, result)
                    if narrow and not result['research_return_verification']['complete']:
                        result['method_delivery'] = ('research_return_incomplete'
                            if result.get('registered_attempt_returned') else 'preparation_only')
                        result['research_execution_complete'] = False
                    else:
                        # Shared closure compares the complete pool, not a favourite subset.
                        from hegel_machine.cognition.finite_native_author import _comparison_models
                        native = [round for question in agent.questions for round in question.get('native_condition_research', [])]
                        models = _comparison_models(scene['models'], agent.candidate_paths, native)
                        if len(models) > contract['scope']['pool_models'] + contract['scope']['max_exports']:
                            raise ValueError('actual candidate pool exceeded the frozen comparison quote')
                        comparison_index = len(agent.operator_results)
                        with phase_timing.phase('final_comparison'):
                            choice = agent._execute('run', {'operator': 'research_choice', 'models': models,
                                'research_question': deepcopy(scene['question_ref'])}, lab)
                        comparison_complete = (not choice.get('error') and not choice.get('truncated')
                            and choice.get('models_requested') == choice.get('models_completed') == len(models))
                        attempt_returned = (scene.get('native_attempt_limit') != 1
                            or result.get('registered_attempt_returned') is True)
                        complete = (comparison_complete and attempt_returned
                            and result.get('native_research_complete', True))
                        if narrow:
                            result['research_execution_complete'] = complete
                        result['method_delivery'] = ('research_return_incomplete'
                            if result.get('native_research_complete') is False
                            else 'registered_research_returned' if attempt_returned else 'preparation_only')
                        unit = getattr(agent, '_organization_state', {}).get('registered_method_unit')
                        if unit is not None:
                            unit['closure'] = {'status': 'complete' if comparison_complete else 'incomplete',
                                'comparison_ref': {'file': str(agent.workspace / 'research_state.json'),
                                    'json_path': ['operator_results', comparison_index]},
                                'compared_rows': len(agent.rows), 'pool_models': len(models),
                                'method_delivery': result['method_delivery']}
                        model = result.get('model')
                        if contract['return_contract'] == 'retained_parent' and complete:
                            model = (agent._evaluated_summary() or {}).get('model')
                        if model and complete:
                            with phase_timing.phase('freeze'):
                                saved = _freeze(agent, model, output / name / 'frozen')
                with phase_timing.phase('final_save'), phase_timing.phase('checkpoint'):
                    checkpoint(lab)
        except Exception as error:
            unit = getattr(agent, '_organization_state', {}).get('registered_method_unit') or {}
            status = ('research_unit_rebind_required' if unit.get('phase') == 'requires_rebind' else
                'time_exhausted' if isinstance(error, (PythonExecutionDeadline, ActivityExecutionDeadline))
                else 'execution_error')
            result = {'status': status, 'execution_error': f'{type(error).__name__}: {error}'}
            complete, model, saved = False, None, None
            rescue_started = monotonic()
            with phase_timing.phase('exception_rescue'), phase_timing.phase('checkpoint'):
                checkpoint(lab)  # Preserve paid work, without calling it within-window completion.
            rescue_seconds = monotonic() - rescue_started
        if not models:
            # A cutoff before final comparison did not erase the inherited
            # pool. Quote its real retained roster, not an unfilled local list.
            from hegel_machine.cognition.finite_native_author import _comparison_models
            native = [entry for question in agent.questions
                      for entry in question.get('native_condition_research', [])]
            models = _comparison_models(scene['models'], agent.candidate_paths, native)
        outcome = {key: deepcopy(contract[key]) for key in ('method_ref', 'start_state_ref', 'data_ref', 'split_ref',
            'quality_contract_ref', 'resource_contract_ref', 'evaluation_ref')}
        outcome.update(status=result['status'], execution_complete=bool(complete and model),
            profile_ref=deepcopy(scene['profile_ref']), origin_ref=deepcopy(scene['origin_ref']),
            cost_refs=deepcopy(scene.get('cost_refs', [])),
            costs={key: agent.costs[key] - value for key, value in before.items()},
            timing={'activity_wall_seconds': monotonic() - entered - rescue_seconds,
                'rescue_preservation_seconds': rescue_seconds, 'recovery_seconds': recovery,
                'billed_python_seconds': agent.costs.get('python_execution_seconds', 0)
                    - before.get('python_execution_seconds', 0), 'administrative_pause_seconds': 0.,
                'administrative_pause_scope': 'this new stage contains no operator-requested pause; '
                    'raw monotonic wall still includes system scheduling waits'},
            comparison={key: choice.get(key) for key in ('models_requested', 'models_completed', 'truncated')},
            world_support_updated=False, independent_support=False)
        outcome['timing'].update(execution_timing_fields(meter,
            outcome['timing']['activity_wall_seconds']))
        outcome['host_phase_timing'] = {**phase_timing.result(),
            'phase_overlap_scope': 'before/after hooks and step checkpoints are nested in procedure; '
                'final_save and exception_rescue each include their checkpoint; do not add phase totals',
            'final_save_scope': 'canonical retained training checkpoint; thin outcome receipt writes are separate',
            'historical_phase_times_reconstructed': False}
        unit = getattr(agent, '_organization_state', {}).get('registered_method_unit')
        if unit is not None:
            outcome['registered_research_unit'] = deepcopy(unit)
        state_ref = {'file': str(agent.workspace / 'training.json'), 'json_path': []}
        outcome.update(continuation_state_ref=state_ref,
            continuation_data_ref={**state_ref, 'json_path': ['paid_observations']},
            continuation_costs_ref={'file': str(output / name / 'outcome.json'), 'json_path': ['costs_after']},
            continuation_scope={'paid_rows': len(agent.rows), 'pool_models': len(models)},
            costs_after=deepcopy(agent.costs))
        if scene.get('method_artifact_ref') is not None:
            outcome['method_artifact_ref'] = deepcopy(scene['method_artifact_ref'])
        for key in ('actual_action_content', 'actual_action_refs', 'method_source_path',
                    'procedure_steps_consumed',
                    'method_invocation_origin_ref', 'method_invocation_binding', 'native_boundary_refs',
                    'registered_attempt_returned', 'method_delivery', 'research_execution_complete',
                    'native_research_complete', 'native_representation_returns',
                    'research_status', 'author_work_ref', 'research_return_verification'):
            if key in result:
                outcome[key] = deepcopy(result[key])
        if result.get('method_invocation_origin_ref') is not None:
            outcome['origin_ref'] = deepcopy(result['method_invocation_origin_ref'])
        if result.get('method_invocation_binding') is not None:
            binding = result['method_invocation_binding']
            actual_method = {'id': binding['candidate_id'], 'version': binding['candidate_version']}
            if actual_method != contract['method_ref']:
                outcome.update(status='execution_binding_mismatch', execution_complete=False,
                    registered_method_ref=deepcopy(contract['method_ref']))
            outcome['method_ref'] = actual_method
        from hegel_machine.cognition.registered_source_control import _comparison_external_python_timing
        external = _comparison_external_python_timing(choice)
        outcome['timing']['comparison_external_python_seconds'] = external['external_python_wall_seconds']
        outcome['timing']['comparison_reported_external_python_job_seconds'] = external[
            'measured_external_python_wall_seconds']
        outcome['timing']['comparison_job_time_scope'] = (
            'sum of new per-evaluation executor intervals; not parallel parent elapsed wall; '
            'not added again to native/direct-executor/other-host wall decomposition')
        outcome['comparison_external_timing'] = external
        if result.get('execution_error'):
            outcome['execution_error'] = result['execution_error']
        if contract.get('execution_role') == 'retained_parent_baseline':
            outcome['retained_parent_declaration'] = deepcopy(contract['retained_parent'])
            outcome['comparison_kind'] = registration['comparison_kind']
        if saved is not None:
            frozen[name] = saved
            outcome['frozen_model'] = saved
        states[name] = {'account': SimpleNamespace(costs=dict(agent.costs), config=agent.config,
            workspace=Path(agent.workspace), rows=tuple(agent.rows)), 'before_costs': before,
            'execution_timing': meter}
        if continuation is not None:
            _continuation_costs(outcome, continuation['outcomes'][name])
            outcome['prior_stage_timing'] = deepcopy(continuation['outcomes'][name]['timing'])
            if continuation.get('execution_repair'):
                outcome['execution_repair'] = deepcopy(continuation['execution_repair'])
                for field in ('actual_action_content', 'actual_action_refs', 'native_boundary_refs'):
                    outcome[field] = (deepcopy(continuation['outcomes'][name].get(field, []))
                                      + outcome.get(field, []))
            for field in ('physical_resume_state_ref', 'prior_outcome_ref'):
                if scene.get(field) is not None:
                    outcome[field] = deepcopy(scene[field])
        outcomes[name] = outcome
        references[name] = _save(output / name / 'outcome.json', outcome)
        # Fitting banks and paid comparison indices can occupy several GiB.
        # Only the frozen execution context and compact account survive a cell.
        del checkpoint, result, choice, native, models, scene, lab, agent
        gc.collect()
    # Commit all frozen outputs before any evaluator may open Qeval labels.
    data_role = registration.get('data_role', 'development')
    retrospective = bool(retained)
    all_frozen = len(frozen) == len(quote['cells'])
    evaluation_lifecycle = {'role': ('retrospective_completion_after_freeze' if retrospective else
            'independent_successor_after_freeze' if data_role == 'independent_successor'
            else 'development_after_freeze'),
        'all_outputs_frozen_before_labels': all_frozen and not retrospective,
        'labels_revealed': False, 'receipt_ref': None,
        'status': 'outputs_frozen' if all_frozen else 'blocked_incomplete_outputs'}
    if retrospective:
        evaluation_lifecycle.update(prior_labels_opened=True,
            new_outputs_frozen_before_label_reuse=all_frozen,
            eligible_for_condition_learning=False)
    frozen_manifest = {'models': frozen, 'outcome_refs': references,
        'evaluation_labels_opened': False, 'evaluation_lifecycle': evaluation_lifecycle}
    _save(output / 'frozen_outputs.json', frozen_manifest)
    shared_evaluation = None
    evaluation_ready = all_frozen
    if evaluation_ready and prepare_evaluation is not None:
        preparation_started = monotonic()
        shared_evaluation = prepare_evaluation()
        evaluation_lifecycle.update(labels_revealed=True, receipt_ref=shared_evaluation,
            status=('retained_labels_available_after_new_outputs_freeze' if retrospective else
                'independent_successor_labels_available_after_freeze'
                if data_role == 'independent_successor' else 'development_labels_available_after_freeze'))
        frozen_manifest['evaluation_labels_opened'] = True
        _save(output / 'frozen_outputs.json', frozen_manifest)
        _save(output / 'shared_evaluation_preparation.json', {
            'receipt': shared_evaluation, 'wall_seconds': monotonic() - preparation_started,
            'charged_to': 'global_Qeval_component_not_first_arm'})
    for name, model in (frozen.items() if evaluation_ready else ()):
        if name in retained:
            continue
        account = states[name]['account']
        remaining = (registration['cells'][name]['time_limits']['active_wall_seconds']
            - outcomes[name]['timing']['activity_wall_seconds'])
        evaluation_started = monotonic()
        try:
            with measured_execution_timing(states[name]['execution_timing']), python_execution_deadline(
                    remaining, activity=True):
                evaluation_lifecycle.update(labels_revealed=True,
                    status=('retrospective_completion_evaluation_after_freeze' if retrospective else
                        'independent_successor_evaluation_after_freeze'
                        if data_role == 'independent_successor' else 'development_evaluation_after_freeze'))
                measured = evaluate_frozen(name, deepcopy(model), registration['cells'][name]['evaluation_ref'], account)
        except Exception as error:
            measured = {}
            outcomes[name].update(status='time_exhausted' if isinstance(error, (PythonExecutionDeadline, ActivityExecutionDeadline))
                else 'execution_error', execution_complete=False, frozen_prediction_status='UNKNOWN',
                execution_error=f'{type(error).__name__}: {error}')
        outcomes[name]['timing']['activity_wall_seconds'] += monotonic() - evaluation_started
        outcomes[name]['timing'].update(execution_timing_fields(states[name]['execution_timing'],
            outcomes[name]['timing']['activity_wall_seconds']))
        for field in ('loss', 'quality', 'coverage', 'completed'):
            if field in measured:
                outcomes[name][field] = deepcopy(measured[field])
        outcomes[name]['costs'] = {key: account.costs[key] - value
            for key, value in states[name]['before_costs'].items()}
        outcomes[name]['timing']['billed_python_seconds'] = (
            account.costs['python_execution_seconds'] - states[name]['before_costs']['python_execution_seconds'])
        outcomes[name]['costs_after'] = dict(account.costs)
        if continuation is not None:
            _continuation_costs(outcomes[name], continuation['outcomes'][name])
        _save(output / name / 'outcome.json', outcomes[name])
        frozen_manifest['evaluation_labels_opened'] = evaluation_lifecycle['labels_revealed']
        _save(output / 'frozen_outputs.json', frozen_manifest)
    if retrospective:
        for name in quote['cells']:
            if name not in retained:
                outcomes[name]['new_stage_costs'] = deepcopy(outcomes[name]['costs'])
                _save(output / name / 'outcome.json', outcomes[name])
    pairs = []
    for left, right in _comparison_pairs(quote['cells']):
        pair = {'baseline_ref': references[left], 'challenger_ref': references[right],
            'data_role': data_role, 'physical_source_ref': registration.get('physical_source_ref'),
            'baseline_action_content': outcomes[left].get('actual_action_content'),
            'challenger_action_content': outcomes[right].get('actual_action_content'),
            'evaluation_lifecycle': deepcopy(evaluation_lifecycle)}
        if len(quote['cells']) == 3:
            pair['comparison_kind'] = 'three_method_same_data'
            pair['identified_effect'] = {('00', '10'): 'fixed_method_vs_original',
                ('00', '20'): 'conditional_method_vs_original',
                ('10', '20'): 'condition_vs_fixed_method'}[(left, right)]
        if retrospective:
            pair['retrospective_completion'] = True
            pair['method_condition_label'] = {'preference': 'UNKNOWN',
                'reason': 'missing_arm_completed_after_prior_evaluation_labels_opened',
                'eligible_for_condition_learning': False}
        if continuation is not None and continuation.get('execution_repair'):
            pair['execution_repair'] = deepcopy(continuation['execution_repair'])
            pair['method_condition_label'] = {'preference': 'UNKNOWN',
                'reason': 'execution_repair_intervened_in_original_method_pair',
                'eligible_for_condition_learning': False}
        if registration.get('comparison_kind') == 'additional_research_vs_retained_parent':
            pair['comparison_kind'] = registration['comparison_kind']
            pair['identified_effect'] = 'local_incremental_research_value'
            pair['baseline_research_status'] = outcomes[left].get('research_status')
            losses = [outcomes[arm].get('loss') for arm in (left, right)]
            pair['frozen_predictive_comparison'] = {
                'baseline_loss': losses[0], 'challenger_loss': losses[1],
                'challenger_minus_baseline': (losses[1] - losses[0]
                    if all(isinstance(value, (float, int)) for value in losses) else None),
                'scope': 'One declared current parent versus additional research at this scene; '
                    'not experimental-design attribution, ordinary-method support or independent transfer.'}
        pairs.append(_save(output / f'pair_{left}_{right}.json', pair))
    if data_role == 'independent_successor':
        use = {'status': ('retrospective_completion_without_method_feedback' if retrospective
                else 'frozen_successor_evaluated_without_method_feedback'),
            'consumed_pair_refs': [], 'new_compute_cost': 0,
            'method_feedback_executed': False, 'successors_started': True,
            'frozen_method_execution_completed': all(
                value.get('execution_complete') for value in outcomes.values())}
    elif registration.get('comparison_kind') == 'additional_research_vs_retained_parent':
        use = {'status': 'not_admitted_retained_parent_control',
            'reason': 'This contrast estimates local additional-research value, not a method preference.'}
    else:
        use = ordinary_use(deepcopy(pairs), deepcopy(references), output)
    return_value = {'status': ('retrospective_completed_method_contrast' if retrospective else
            'completed_independent_successor_pairing'
            if data_role == 'independent_successor' else 'completed_development_pairing'),
        'data_role': data_role, 'quote': quote,
        'outcome_refs': references, 'pair_refs': pairs, 'ordinary_use': use,
        'shared_evaluation_preparation_ref': ({'file': str(output / 'shared_evaluation_preparation.json'),
            'json_path': []} if evaluation_ready and prepare_evaluation is not None else None),
        'evaluation_lifecycle': deepcopy(evaluation_lifecycle),
        'wall_seconds': monotonic() - started, 'independent_support': False}
    if len(quote['cells']) == 3:
        return_value['comparison_kind'] = 'three_method_same_data'
    if retrospective:
        return_value.update(retrospective_completion=True,
            comparison_costs={name: deepcopy(item['costs']) for name, item in outcomes.items()},
            new_stage_costs={name: deepcopy(item['new_stage_costs']) for name, item in outcomes.items()})
    if registration.get('comparison_kind') == 'additional_research_vs_retained_parent':
        return_value['comparison_kind'] = registration['comparison_kind']
        return_value['identified_effect'] = 'local_incremental_research_value'
    _save(output / 'summary.json', return_value)
    return return_value


def main(argv=None):
    """Run the retained-scene factories only under the existing admission gate."""
    import argparse
    from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
    from scripts.prepare_pr3_source_successor import make_retained_method_pairing_factories

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args(argv)
    registration = read_json_checkpoint(arguments.registration)
    callbacks = make_retained_method_pairing_factories(registration, arguments.output)
    return run_method_action_pairing(registration, arguments.output, **callbacks)


if __name__ == '__main__':
    import json
    result = main()
    print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds']}))
