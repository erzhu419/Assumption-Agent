#!/usr/bin/env python3
"""One declared new source, retaining published models and the existing pair runner."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.checkpoint_storage import clone_or_copy
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_state_method_pairing import read, save, prepare, run as run_pair, _scene
from scripts.run_method_action_pairing import quote_method_action_pairing


def review_paid_confirmation(request, output):
    """Resolve frozen competitors using paid confirmation before any new purchase."""
    from dataclasses import asdict
    from time import monotonic
    from hegel_machine.cognition.competing_object_experiments import (
        review_paid_program_competition, assess_competing_object_probe)
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import SearchBudget, TaskSpec
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
    from scripts.run_anonymous_revision_inquiry import evaluate_terminal_model

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request_ref = save(output / 'request.json', request)
    source = read(request['source_result_ref'])
    original = read(source['request_ref'])
    previous = read(source['training_ref'])
    native = read(request['native_result_ref'])
    public = read(request['public_spec_ref'])
    fresh = source['confirmation']
    budget = SearchBudget(original['new_compute_cap'] - source['new_costs']['compute_cost'])
    observation_limit = original['new_observation_cap'] - source['new_costs']['observation_cost']
    seconds = original['new_python_seconds'] - source['new_costs']['python_execution_seconds']
    started, python_seconds, new_observations = monotonic(), 0., 0
    result = {'status': 'reviewing_paid_confirmation', 'request_ref': request_ref,
        'source_result_ref': request['source_result_ref'],
        'compute_limit': budget.limit, 'observation_limit': observation_limit,
        'fits': 0, 'model_calls': 0, 'scores': [], 'confirmed_model_changed': False}
    for name, module in tuple(sys.modules.items()):
        if ((name.startswith('hegel_machine') or name.startswith('scripts.run_'))
                and getattr(module, '__file__', None)):
            path = Path(module.__file__).resolve()
            if path.is_relative_to(ROOT):
                target = output / 'source_at_run' / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                clone_or_copy(path, target)
    entry = output / 'source_at_run/scripts' / Path(__file__).name
    entry.parent.mkdir(parents=True, exist_ok=True)
    clone_or_copy(Path(__file__).resolve(), entry)
    try:
        with python_execution_deadline(
                original['activity_seconds'] - source['activity_wall_seconds'], activity=True):
            rows = [row for record in fresh['observations']
                    for row in AnonymousHistoryLab.observation_rows(None, record)]
            submitted = Path(previous['workspace']) / fresh['submitted_candidate']['model']
            score_scope = {**source['request_ref'], 'file': request['source_result_ref']['file'],
                           'json_path': ['confirmation', 'observations']}
            hypotheses = []
            for item in request['candidates']:
                candidate = native['candidates'][item['native_index']]
                program = Program.from_dict(candidate['program'])
                path = Path(previous['workspace']) / item['model']
                # This entry requires the literal, fit-free native export. Do not
                # invoke evaluate_terminal_model's optional source-fit behavior.
                if path.read_text() != export_temporal_program(program, feature_names=tuple(public['controls'])):
                    raise ValueError('review requires the original literal native predictor')
                if path.read_bytes() == submitted.read_bytes():
                    score = {key: fresh[key] for key in ('returncode', 'mse', 'passed')}
                    score.update(evaluation_rows=fresh['rows'], reused=True)
                    reference = {**request['source_result_ref'], 'json_path': ['confirmation']}
                else:
                    if python_seconds >= seconds:
                        raise ValueError('remaining fixed-readout Python time exhausted')
                    if not budget.spend(len(rows)):
                        raise ValueError('remaining fixed-readout allowance exhausted')
                    target = output / 'models' / f"{item['native_index']}.py"
                    target.parent.mkdir(exist_ok=True)
                    clone_or_copy(path, target)
                    score = evaluate_terminal_model(target.parent, target.name, rows,
                        timeout_seconds=max(.001, seconds - python_seconds))
                    python_seconds += score['python_execution_seconds']
                    reference = save(output / f"score_{item['native_index']}.json", score)
                hypotheses.append({'candidate_id': candidate['id'], 'program': candidate['program'],
                    'score': score, 'score_ref': reference})
                result['scores'].append({'candidate_id': candidate['id'], 'score_ref': reference,
                    'mse': score.get('mse'), 'returncode': score.get('returncode'),
                    'rows': score.get('evaluation_rows'), 'reused': score.get('reused', False)})
                save(output / 'review_result.json', result)
            spec = TaskSpec('paid_confirmation_review', 'Distinguish surviving frozen explanations',
                tuple(public['controls']), (), (), feature_bounds=public['feature_bounds'],
                max_probe_length=public['max_probe_length'])
            review = review_paid_program_competition(spec, hypotheses,
                previous['paid_observations'], budget, score_scope_ref=score_scope,
                score_rows=fresh['rows'], control_levels=public['control_levels'],
                tolerance=fresh['mse_threshold'], serial=len(previous['paid_observations']))
            plan = review.get('plan')
            result['review_ref'] = save(output / 'paid_review.json', {**review,
                'plan': {**plan, 'probe': asdict(plan['probe'])} if plan else None})
            result['status'] = review['status']
            if plan is not None:
                probe = plan['probe']
                assessment_cost = probe.cost * len(plan['hypotheses'])
                if probe.cost > observation_limit or assessment_cost > budget.remaining:
                    result['status'] = 'remaining_account_cannot_complete_probe'
                else:
                    case = read(original['case_ref'])
                    lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'],
                        **({'startup_seed': case['startup_seed']} if 'startup_seed' in case else {}))
                    if lab.public_spec() != public:
                        raise ValueError('paid source and proposed experiment have different public interfaces')
                    lab.restore_paid_account(previous['paid_observations'], source['costs_after'])
                    # The proposal and all forecasts were saved before measurement.
                    record = lab.execute({'kind': 'trajectory', 'channels': [],
                                          'sequence': list(probe.sequence)})
                    new_observations += record['observation_cost']
                    result['observation_ref'] = save(output / 'discriminating_observation.json', record)
                    observed = lab.observation_rows(record)
                    assessment = assess_competing_object_probe(plan, [(row, row) for row in observed],
                        budget, tolerance=fresh['mse_threshold'])
                    result['assessment_ref'] = save(output / 'probe_assessment.json', assessment)
                    result['status'] = assessment['status']
                    result['probe_rows'] = len(observed)
    except Exception as error:
        result.update(status='interrupted', error=f'{type(error).__name__}: {error}')
    finally:
        result['new_costs'] = {'compute_cost': budget.spent, 'observation_cost': new_observations,
            'python_execution_seconds': python_seconds, 'measurement_cost': 0,
            'model_calls': 0, 'reported_tokens': 0}
        result['activity_wall_seconds'] = monotonic() - started
        save(output / 'review_result.json', result)
    return result


def run_continuation(request, output):
    """Continue one completed source with paid counterexamples and fresh confirmation."""
    from time import monotonic
    from scripts.run_paid_state_revision import restore_completed_state
    from hegel_machine.cognition.temporal_paired_method import execute_procedure
    from hegel_machine.cognition import program_disagreement_probes

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request_ref = save(output / 'request.json', request)
    sources = {Path(__file__).resolve(), Path(program_disagreement_probes.__file__).resolve()}
    sources.update(Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
        if (name.startswith('hegel_machine') or name.startswith('scripts.run_'))
        and getattr(module, '__file__', None))
    for source in sources:
        if source.is_relative_to(ROOT):
            destination = output / 'source_at_run' / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            clone_or_copy(source, destination)
    timing = ExecutionTiming(('restore', 'ordinary_research', 'preserve'))
    result = {'status': 'preparing', 'request_ref': request_ref,
        'source_role': 'same_source_exposed_counterexample_continuation',
        'previous_evaluation_role': 'development_only', 'model_calls': 0}
    agent, started, baseline = None, monotonic(), None
    try:
        with python_execution_deadline(request['activity_seconds'], activity=True):
            with timing.phase('restore'):
                prior = read(request['outcome_ref'])['costs_after']
                previous = read(request['training_ref'])
                cfg = replace(InquiryLLMConfig(**previous['limits']),
                    max_model_calls=0, token_budget=0,
                    compute_budget=prior['compute_cost'] + request['new_compute_cap'],
                    observation_budget=prior['observation_cost'] + request['new_observation_cap'],
                    execution_seconds=prior['python_execution_seconds'] + request['new_python_seconds'],
                    confirmation_rows=request['confirmation_rows'])
                case = read(request['case_ref'])
                lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'],
                    **({'startup_seed': case['startup_seed']} if 'startup_seed' in case else {}))
                scene = restore_completed_state(request, output / 'workspace', lab, config=cfg)
                agent, lab = scene['agent'], scene['lab']
                baseline = deepcopy(agent.costs)
                result['restoration_ref'] = save(output / 'restoration.json', scene['restoration'])
                result['costs_before'] = baseline
                result['paid_rows_before'] = len(agent.rows)
                result['method_result_start'] = len(agent.method_results)
                result['operator_result_start'] = len(agent.operator_results)

            def checkpoint(_=None):
                save(agent.workspace / 'training.json', agent.checkpoint_result('researching'))
                if agent.method_results:
                    feedback = agent.method_results[-1].get('feedback', {})
                    print({'step_count': len(agent.method_results) - result['method_result_start'],
                           'status': feedback.get('status'), 'error': feedback.get('error'),
                           'compute': agent.costs['compute_cost'] - baseline['compute_cost']}, flush=True)

            checkpoint()
            with timing.phase('ordinary_research'):
                procedure = execute_procedure(agent, lab, 'ordinary.py',
                    max_steps=request['max_steps'], allow_confirmation=True, record_events=True,
                    before_step=agent._before_model_turn, after_step=agent._after_model_turn,
                    on_checkpoint=checkpoint)
            result['status'] = procedure['status']
            result['procedure_ref'] = save(output / 'procedure.json', procedure)
    except Exception as error:
        result.update(status='interrupted', error=f'{type(error).__name__}: {error}')
    finally:
        with timing.phase('preserve'):
            if agent is not None:
                agent._prepare_program_state()
                result['training_ref'] = save(agent.workspace / 'training.json',
                    agent.checkpoint_result(result['status']))
                result['costs_after'] = deepcopy(agent.costs)
                result['new_costs'] = {key: value - baseline[key]
                    for key, value in agent.costs.items()} if baseline is not None else None
                result['paid_rows_after'] = len(agent.rows)
                result['selected_model'] = agent.model
                result['evaluated'] = agent._evaluated_summary()
                result['confirmation'] = deepcopy(agent.confirmations[-1]) if agent.confirmations else None
                result['retained_pool_size'] = len(set(agent.candidate_paths) | set(
                    agent.knowledge['finite_native_author']['predictor_paths']))
        result['timing'] = timing.result()
        result['activity_wall_seconds'] = monotonic() - started
        save(output / 'continuation_result.json', result)
    return result


def run_paid_method_pair(request, output):
    """Consume exposed counterexamples in a matched pair of existing procedures."""
    from scripts.run_paid_state_revision import restore_exposed_common_state, bind_restored_state_author

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request_ref = save(output / 'request.json', request)
    timing = ExecutionTiming(('restore', 'shared_comparison', 'preparation', 'paired_research'))
    result = {'status': 'preparing', 'request_ref': request_ref,
        'source_role': 'same_physical_source_exposed_counterexample_method_pairing',
        'prior_costs_ref': request['prior_costs_ref'], 'research_observations': 0}
    agent, before = None, None
    for name, module in tuple(sys.modules.items()):
        if ((name.startswith('hegel_machine') or name.startswith('scripts.run_'))
                and getattr(module, '__file__', None)):
            path = Path(module.__file__).resolve()
            if path.is_relative_to(ROOT):
                target = output / 'source_at_run' / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                clone_or_copy(path, target)
    target = output / 'source_at_run/scripts' / Path(__file__).name
    target.parent.mkdir(parents=True, exist_ok=True)
    clone_or_copy(Path(__file__).resolve(), target)
    try:
        with timing.phase('restore'):
            previous = read(request['training_ref'])
            inherited = deepcopy(previous['costs'])
            for reference in request['additional_paid_cost_refs']:
                for key, value in read(reference).items():
                    inherited[key] += value
            cfg = replace(InquiryLLMConfig(**previous['limits']),
                max_model_calls=0, token_budget=0,
                compute_budget=inherited['compute_cost'] + request['shared_compute_cap'],
                observation_budget=inherited['observation_cost'],
                execution_seconds=inherited['python_execution_seconds'] + request['per_cell_python_seconds'])
            case = read(request['case_ref'])
            lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'],
                **({'startup_seed': case['startup_seed']} if 'startup_seed' in case else {}))
            scene = restore_exposed_common_state(request, output / 'shared' / 'workspace', lab, config=cfg)
            agent, before = scene['agent'], deepcopy(scene['agent'].costs)
            result['restoration_ref'] = save(output / 'restoration.json', scene['restoration'])
        with timing.phase('shared_comparison'):
            contract = agent.knowledge['finite_native_author']
            choice = agent._execute('run', {'operator': 'research_choice',
                'models': scene['models'], 'research_question': deepcopy(contract['question'])}, lab)
            result['shared_comparison_ref'] = save(output / 'shared_comparison.json', choice)
            if (choice.get('error') or choice.get('truncated')
                    or choice.get('models_completed') != len(scene['models'])):
                result['status'] = 'shared_comparison_incomplete'
                return result
            bind_restored_state_author(agent, contract)
            agent._prepare_program_state()
            initial = save(agent.workspace / 'training.json', agent.checkpoint_result('researching'))
            result['initial_scene_ref'] = initial
            if (agent._evaluated_summary() or {}).get('mse', float('inf')) <= request['mse_threshold']:
                result['status'] = 'existing_pool_explains_evidence_retain_without_pair'
                return result
        with timing.phase('preparation'):
            source = read(request['finite_native_author_ref'])
            catalog = read(source['predictor_catalog_ref'])
            while isinstance(catalog, dict):
                catalog = read(catalog['source_catalog_ref'])
            paid = {'paid_records_ref': {**initial, 'json_path': ['paid_observations']},
                'append_record_refs': [], 'expected_rows': len(agent.rows),
                'public_spec_ref': request['public_spec_ref'], 'parents': catalog,
                'diagnostic_parent_id': source['native_representation_plan']['parent_theory_ids'][0],
                'validation_records': source['native_representation_plan']['validation_records'],
                'counterexample_ref': request['append_record_refs'][-1],
                'prior_costs_ref': request['prior_costs_ref'],
                'research_compute_cap': request['research_compute_cap'],
                'max_candidates': 4, 'mse_threshold': request['mse_threshold']}
            paid_ref = save(output / 'paid_registration.json', paid)
            author_ref = save(output / 'restored_author.json', agent.knowledge['finite_native_author'])
            registration = prepare(paid_ref, output / 'prepared', case_ref=request['case_ref'],
                evaluation_seed=request['evaluation_seed'], evaluation_rows=request['evaluation_rows'],
                compute_cap=request['per_cell_compute_cap'],
                research_compute_cap=request['research_compute_cap'],
                activity_seconds=request['per_cell_activity_seconds'],
                python_seconds=request['per_cell_python_seconds'], approval_ref=request_ref,
                initial_scene_ref=initial, restore_confirmation_seed=request['restore_confirmation_seed'],
                finite_native_author_ref=author_ref,
                data_role=request.get('data_role', 'development'),
                method_topic_ref=request.get('method_topic_ref'),
                method_condition_compute_cap=request.get('method_condition_compute_cap', 1000))
            result['static_proposal_costs'] = registration['preparation_costs']
            result['registration_ref'] = {'file': str(output / 'prepared/registration.json'), 'json_path': []}
        with timing.phase('paired_research'):
            summary = run_pair(registration, output / 'study')
        result.update(status=summary['status'], study_ref={
            'file': str(output / 'study/summary.json'), 'json_path': []})
    except Exception as error:
        result.update(status='interrupted', error=f'{type(error).__name__}: {error}')
    finally:
        if agent is not None and before is not None:
            result['inherited_costs'] = before
            result['shared_new_costs'] = {key: value - before[key] for key, value in agent.costs.items()}
            result['shared_training_ref'] = save(agent.workspace / 'training.json',
                agent.checkpoint_result(result['status']))
        result['timing'] = timing.result()
        save(output / 'source_result.json', result)
    return result


def review_pre_use_condition_scope(registration, scope, output):
    """Screen a declared branch using only the frozen method's paid pre-use view.

    The method is installed in a restored screening workspace, so its manifest
    cannot change the shared scene subsequently supplied to the research arms.
    """
    from math import isfinite
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Observation
    from hegel_machine.cognition.method_feedback import begin_method_feedback
    from hegel_machine.cognition.method_conditions import (
        method_condition_features, method_condition_support)
    from hegel_machine.cognition.research_programs import export_research_selection_program
    from hegel_machine.cognition.theory_consolidation import install_python_research_method

    if scope['branch'] not in {'when_true', 'when_false'}:
        raise ValueError('the pre-use stratum must name an actual condition branch')
    name = scope['method_cell']
    contract = registration['cells'][name]
    bundle = read(registration['method_pair_ref'])
    method = bundle['cells'][name]
    plan = contract['selection_plan']
    if plan.get('condition') is None or not contract.get('selection_ref'):
        raise ValueError('branch screening requires the frozen conditional method and its selection evidence')
    selection = read(contract['selection_ref'])
    output = Path(output)
    agent, lab, _ = _scene(registration, output / 'workspace')
    before = deepcopy(agent.costs)
    candidate = install_python_research_method(agent, candidate_id=contract['method_ref']['id'],
        source=export_research_selection_program(plan), selection_plan=plan,
        source_refs=method['source_refs'], max_steps=contract['max_steps'],
        selection_ref=contract['selection_ref'])
    context = begin_method_feedback(agent, lab, candidate=candidate)
    if context is None or agent.costs != before:
        raise ValueError('pre-use screening must read a paid profile without executing world research')
    profile = context['profile']
    support = method_condition_support(profile, selection)
    features = method_condition_features(profile, plan['condition_feature_paths'])
    condition = Program.from_dict(plan['condition'])
    required = sorted(condition.required_features())
    missing = [key for key in required if type(features.get(key)) not in (int, float)
               or not isfinite(features[key])]
    value = None
    compute = 0
    if support['can_apply'] and not missing:
        value = bool(condition.predict(Observation(features, 0.)))
        compute = 1
    matched = value is not None and value == (scope['branch'] == 'when_true')
    result = {'status': ('method_condition_stratum_admitted' if matched else
        'method_condition_stratum_unknown' if value is None else 'method_condition_other_branch'),
        'requested_scope': deepcopy(scope), 'method_ref': deepcopy(contract['method_ref']),
        'selection_ref': deepcopy(contract['selection_ref']),
        'start_state_ref': deepcopy(registration['initial_scene_ref']),
        'profile_ref': save(output / 'profile.json', profile), 'support': support,
        'condition_value': value, 'condition_features': {key: features.get(key) for key in required},
        'missing_features': missing, 'matched': matched, 'new_compute_cost': compute,
        'new_world_predictions': 0, 'new_fits': 0, 'new_observations': 0,
        'evidence_scope': 'frozen_pre_use_condition_not_research_outcome'}
    save(output / 'result.json', result)
    return result


def run(request, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    entry = output / 'source_at_run' / 'scripts' / Path(__file__).name
    entry.parent.mkdir(parents=True, exist_ok=True)
    clone_or_copy(Path(__file__).resolve(), entry)
    request_ref = save(output / 'request.json', request)
    timing = ExecutionTiming(('prefix', 'preparation', 'shared_comparison', 'paired_research'))
    result = {'status': 'preparing', 'request_ref': request_ref,
              'source_role': 'newly_registered_parameter_source',
              'comparison_scope': request['comparison_scope']}
    with python_execution_deadline(request['source_activity_seconds'], activity=True):
        with timing.phase('prefix'):
            prefix_result = request.get('paid_prefix_ref')
            if prefix_result is not None:
                prefix = read(prefix_result)
                records_ref = prefix['records_ref']
                public_ref = {'file': str(Path(prefix_result['file']).parent / 'public_spec.json'),
                              'json_path': []}
            else:
                case = request['case']
                lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'],
                    **({'startup_seed': case['startup_seed']} if 'startup_seed' in case else {}))
                config = replace(InquiryLLMConfig(), max_model_calls=0, token_budget=0,
                    observation_budget=request['prefix_rows'], compute_budget=0,
                    execution_seconds=request['source_python_seconds'])
                source = CommonArchiveInquiryAgent(None, output / 'prefix', arm='hegel',
                    config=config, source_task=False)
                actions = (read(request['prefix_actions_ref']) if request.get('prefix_actions_ref')
                    else lab.confirmation_actions(request['prefix_rows'], Random(request['prefix_seed'])))
                save(output / 'prefix_actions.json', actions)
                response = source._execute('query', {'actions': actions}, lab)
                source._write_evidence()
                records_ref = {'file': str(source.workspace / 'laboratory_observations.json'), 'json_path': []}
                prefix = {'error': response.get('error'), 'paid_rows': len(source.rows),
                          'records_ref': records_ref, 'costs': deepcopy(source.costs)}
                prefix_result = save(output / 'prefix_result.json', prefix)
                public_ref = save(output / 'public_spec.json', lab.public_spec())
            if prefix['paid_rows'] != request['prefix_rows']:
                raise ValueError('the fixed public prefix was not acquired completely')
        paid = {'paid_records_ref': records_ref, 'append_record_refs': [],
            'public_spec_ref': public_ref, 'expected_rows': request['prefix_rows'],
            'parents': deepcopy(request['parents']),
            'diagnostic_parent_id': request['diagnostic_parent_id'],
            'validation_records': request['validation_records'],
            'counterexample_ref': records_ref, 'prior_costs_ref': prefix_result,
            'research_compute_cap': request['research_compute_cap'], 'max_candidates': 4,
            'mse_threshold': request['mse_threshold']}
        paid_ref = save(output / 'paid_registration.json', paid)
        with timing.phase('preparation'):
            registration = prepare(paid_ref, output / 'prepared',
                case_ref={**request_ref, 'json_path': ['case']},
                evaluation_seed=request['evaluation_seed'], evaluation_rows=request['evaluation_rows'],
                compute_cap=request['per_cell_compute_cap'],
                research_compute_cap=request['research_compute_cap'],
                activity_seconds=request['per_cell_activity_seconds'],
                python_seconds=request['per_cell_python_seconds'], approval_ref=request_ref,
                method_pair_ref=request.get('method_pair_ref'),
                prior_pair_refs=request.get('prior_pair_refs', []),
                data_role=request.get('data_role', 'development'),
                method_topic_ref=request.get('method_topic_ref'),
                method_condition_compute_cap=request.get('method_condition_compute_cap', 1000))
            registration['comparison_scope'] = request['comparison_scope']
        with timing.phase('shared_comparison'):
            agent, public_lab, names = _scene(registration, output / 'shared_comparison' / 'workspace')
            agent.config = replace(agent.config, compute_budget=request['source_comparison_compute_upper'])
            choice = agent._execute('run', {'operator': 'research_choice',
                'models': [name + '.py' for name in names],
                'research_question': deepcopy(registration['question'])}, public_lab)
            shared_ref = save(output / 'shared_comparison' / 'result.json', choice)
            agent._prepare_program_state()
            initial_ref = save(agent.workspace / 'training.json', agent.checkpoint_result('researching'))
            result.update(initial_comparison_ref=shared_ref, initial_scene_ref=initial_ref,
                prefix_costs=deepcopy(prefix['costs']), prefix_costs_ref=prefix_result,
                prefix_reused=request.get('paid_prefix_ref') is not None,
                shared_comparison_costs=deepcopy(agent.costs),
                static_proposal_costs=registration['preparation_costs'])
            if (choice.get('error') or choice.get('truncated')
                    or choice.get('models_completed') != len(names)):
                result['status'] = 'initial_comparison_incomplete'
            elif (agent._evaluated_summary() or {}).get('mse', float('inf')) <= paid['mse_threshold']:
                result['status'] = 'retained_models_explain_paid_prefix_no_research'
            else:
                from scripts.run_paid_state_revision import bind_restored_state_author
                from hegel_machine.cognition.finite_native_author import (
                    publish_research_options, research_segment_options)
                from hegel_machine.cognition.research_programs import propose_research_action_revisions

                bound = bind_restored_state_author(agent, registration['finite_native_author'])
                contract = deepcopy(agent.knowledge['finite_native_author'])
                registration.update(question=deepcopy(contract['question']),
                    finite_native_author=contract)
                # These static options describe the coming registered arm,
                # not the already spent shared-comparison allowance.
                agent.config = replace(agent.config,
                    compute_budget=agent.costs['compute_cost'] + request['per_cell_compute_cap'],
                    execution_seconds=agent.costs['python_execution_seconds']
                        + request['per_cell_python_seconds'])
                agent._prepare_program_state()
                options = publish_research_options(agent, public_lab.public_spec())
                view = research_segment_options(public_lab.public_spec(), agent.observations,
                    agent.knowledge, state=agent._program_state(), decision={'research_options': options})
                ordinary = next(item for item in view['options']
                    if item['id'] == 'native_representation:ordinary')
                proposals = propose_research_action_revisions(ordinary['proposal'], view,
                    source_refs=[shared_ref], max_candidates=1)
                extra = 1 + len(options)
                refreshed = save(output / 'static_selection_after_comparison.json', {
                    'ordinary': ordinary, 'view': view, 'proposals': proposals,
                    'actual_parent_binding': bound['actual_parent_binding'],
                    'previous_selection_ref': registration['static_selection_ref'],
                    'previous_preparation_costs': deepcopy(registration['preparation_costs']),
                    'new_compute_cost': extra,
                    'scientific_fits': 0, 'new_observations': 0})
                if not proposals['candidates']:
                    raise ValueError('no distinct executable alternative after the actual shared comparison')
                if request.get('method_pair_ref') is None:
                    registration['cells']['10']['selection_plan'] = {
                        **proposals['candidates'][0]['research_selection_plan'],
                        'ordinary_method_path': 'ordinary.py'}
                registration['preparation_costs']['compute_cost'] += extra
                registration['components']['projection_proposals']['compute'] += extra
                registration['components']['feedback_preservation']['compute'] -= extra
                registration['method_condition_compute_cap'] -= extra
                registration['static_selection_ref'] = refreshed
                result['static_proposal_costs'] = deepcopy(registration['preparation_costs'])
                initial_ref = save(agent.workspace / 'training.json',
                    agent.checkpoint_result('researching'))
                result.update(initial_scene_ref=initial_ref,
                    actual_parent_binding=bound['actual_parent_binding'])
                registration['initial_scene_ref'] = initial_ref
                for contract in registration['cells'].values():
                    contract['start_state_ref'] = deepcopy(initial_ref)
                registration['quote'] = quote_method_action_pairing(registration)
                save(output / 'prepared' / 'registration.json', registration)
                result['status'] = 'research_admitted_by_paid_residual'
                if request.get('pre_use_condition_scope'):
                    screening = review_pre_use_condition_scope(registration,
                        request['pre_use_condition_scope'], output / 'pre_use_condition')
                    extra = screening['new_compute_cost']
                    registration['preparation_costs']['compute_cost'] += extra
                    registration['components']['projection_proposals']['compute'] += extra
                    registration['components']['feedback_preservation']['compute'] -= extra
                    registration['method_condition_compute_cap'] -= extra
                    registration['quote'] = quote_method_action_pairing(registration)
                    if registration['quote']['status'] != 'quoted':
                        raise ValueError('the predeclared source allowance cannot fund condition screening')
                    save(output / 'prepared' / 'registration.json', registration)
                    result['static_proposal_costs'] = deepcopy(registration['preparation_costs'])
                    result['pre_use_condition_ref'] = {
                        'file': str(output / 'pre_use_condition/result.json'), 'json_path': []}
                    if not screening['matched']:
                        result['status'] = screening['status']
        if result['status'] == 'research_admitted_by_paid_residual':
            with timing.phase('paired_research'):
                summary = run_pair(registration, output / 'study')
            result.update(status=summary['status'], study_ref={
                'file': str(output / 'study' / 'summary.json'), 'json_path': []})
    result['timing'] = timing.result()
    save(output / 'source_result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--continue-paid-source', action='store_true')
    modes.add_argument('--review-paid-confirmation', action='store_true')
    modes.add_argument('--pair-paid-counterexamples', action='store_true')
    args = parser.parse_args()
    runner = (run_paid_method_pair if args.pair_paid_counterexamples else
              review_paid_confirmation if args.review_paid_confirmation else
              run_continuation if args.continue_paid_source else run)
    result = runner(read({'file': args.request, 'json_path': []}), args.output)
    print(result['status'], flush=True)
