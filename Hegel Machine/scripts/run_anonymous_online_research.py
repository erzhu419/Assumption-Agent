#!/usr/bin/env python3
"""Shared online research on paid boundaries or independently paired representations."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from random import Random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts.run_anonymous_method_revision import COUNTEREXAMPLE
from scripts.run_anonymous_revision_inquiry import protocol as base_protocol, run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


RESEARCH_TASK = """\nCURRENT RESEARCH TASK
Explain the current paid observations and predict responses on unseen public
control trajectories, including longer paths. Earlier archives are research
experience, not answers for this apparatus. If explanations compatible with the
paid evidence disagree on unmeasured predictions, decide whether and how to obtain
informative evidence. Use the resulting observations to retain or revise your
explanation and next research decision. If the evidence already suffices, you may
proceed directly to confirmation. Scientific models, experiments and conclusions
are yours to construct with the available ordinary Python and laboratory tools.
"""

SNAPSHOT_FILES = [
    'scripts/run_anonymous_online_research.py',
    'scripts/run_anonymous_revision_inquiry.py',
    'scripts/run_anonymous_method_revision.py',
    *['src/hegel_machine/cognition/' + name + '.py' for name in (
        'anonymous_history_lab', 'anonymous_research_archive', 'common_archive_inquiry',
        'online_inquiry_llm', 'inquiry_revision', 'temporal_paired_method', 'code_agent',
        'probing', 'models', 'language', 'evaluation')],
]


def representation_protocol(source_archive, *, model='gpt-6-luna', timeout_seconds=300.):
    """Reuse the old short-history ambiguity without supplying its candidate menu."""
    plan = base_protocol(model=model, timeout_seconds=timeout_seconds)
    cases = []
    for block, seed in enumerate((71211031, 71212040)):
        modes = ('finite2', 'recurrent') if block == 0 else ('recurrent', 'finite2')
        for mode in modes:
            cases.append({'id': f'case_{len(cases):03d}', 'kind': mode,
                'parameter_pair': block, 'seed': seed, 'apparatus_seed': seed,
                'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                'evaluation_seed': seed + 300007})
    plan.update(study='anonymous_online_representation_ambiguity', cases=cases,
        source_archive=str(Path(source_archive).resolve()), shared_research_task=RESEARCH_TASK,
        runtime_snapshot=SNAPSHOT_FILES + ['src/hegel_machine/cognition/' + name + '.py'
            for name in ('temporal_inquiry', 'ambiguous_instrument_lab',
                         'stateful_instrument_lab', 'measurement_lab')],
        prefix_protocol='Buy 24 independent three-step response trajectories (72 rows). '
            'Both worlds of a parameter pair share the controls and apparatus coefficients. '
            'Their short responses agree to floating-point precision; longer legal experiments '
            'can distinguish them. The public horizon is 24 from the beginning. No meter '
            'readings are supplied initially; both agents may purchase them.',
        archive_protocol='Both arms receive the same complete six-source anonymous archive. '
            'No instrument candidate menu, fitted state solver or later compiled method is supplied.',
        discovery='Luna authors ordinary Python representations, hypotheses, experiments and '
            'readouts. Reused physics previously supported host-enumerated instrument research; '
            'the present question is autonomous formulation under the common archive interface.',
        cost_scope='Every prefix is freshly purchased and included in total costs. '
            'post_prefix_cost separately reports inquiry after the common 72-row allocation. '
            'No inherited observation exposure is counted as a new purchase.',
        outcome_protocol='Report task completion, actual representation changes, pre-acquisition '
            'predictions and evidence consumption separately. Direct fitting or random confirmation '
            'may solve the task but does not establish prospective discrimination or recursion.',
        evaluation={'rows': 512, 'trajectory_length': 24, 'mse_threshold': 1e-10},
        independent_parameter_pairs=2,
        scope='Two new parameter blocks, each containing finite and recurrent apparatus worlds, '
            'crossed with two agents: eight episodes, not eight independent discoveries. '
            'This reuses historical physics to test free-Python scientific adoption. '
            'Terminal paths remain within the public 24-step horizon. No method transfer or '
            'recursive advantage is inferred from this development study alone.')
    return plan


def representation_laboratory(case):
    return TemporalInquiryLab(task_index=1, seed=case['seed'],
        apparatus_seed=case['apparatus_seed'], instrument_mode=case['kind'])


def conditional_protocol(source_archive=None, *, model='gpt-6-luna', timeout_seconds=600.):
    """Test optional conditional-component adoption on fresh recurrent apparatuses."""
    source_archive = source_archive or ROOT / 'artifacts/anonymous_revision_luna_20261011_v2'
    plan = representation_protocol(source_archive, model=model, timeout_seconds=timeout_seconds)
    cases = []
    for block, seed in enumerate((71221031, 71222040)):
        cases.append({'id': f'case_{block:03d}', 'kind': 'recurrent',
            'parameter_pair': block, 'seed': seed, 'apparatus_seed': seed,
            'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
            'evaluation_seed': seed + 300007,
            'arms': ['code_react', 'hegel'] if block == 0 else ['hegel', 'code_react']})
    plan.pop('independent_parameter_pairs')
    plan.update(cohort='conditional_components_new_parameters', cases=cases,
        independent_parameter_sets=2, execution_workers=2,
        controls='Both arms receive identical paid prefixes, six-source archives, ordinary Python, '
            'experiment tools, observable-target evaluation, conditional-component binding, '
            'retention and parent model composition. Hegel additionally maintains research '
            'questions, child-result returns and dependency focus. Neither continuation arm '
            'requires a manual manifest or a new research program before submission.',
        prefix_protocol='Freshly buy 24 independent three-step response trajectories (72 rows) '
            'for each arm, with identical controls and apparatus parameters within each case. '
            'The public horizon is 24. No meter readings or fitted scientific models are '
            'supplied initially; purchasing measurements and constructing components are optional.',
        outcome_protocol='Separately report author-created conditional views and retained models, '
            'actual parent dependency calls, fresh confirmation, terminal prediction error and '
            'costs. Ordinary monolithic predictors count as task solutions. Local paid fit, '
            'component retention, automatic questions or declared bindings alone do not establish '
            'successful parent use, autonomous decomposition or a recursive advantage.',
        scope='Two new parameter settings of the existing recurrent apparatus family, each '
            'crossed with Code and Hegel: four episodes, two independent apparatus settings. '
            'No scientific files, successful source steps or partial results from the old '
            'representation rescue are inherited. This development cohort tests optional '
            'component adoption and parent prediction closure, not unseen-family transfer or '
            'an isolated causal estimate of the new interface or recursive organization.')
    plan['limits'].update(max_model_calls=96, token_budget=10_000_000,
        execution_seconds=600., observation_budget=576, confirmation_rows=96,
        compute_budget=250_000)
    plan['runtime_snapshot'] = sorted(set(plan['runtime_snapshot']) | {
        str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py')
    } | {'scripts/run_online_inquiry_llm.py'})
    return plan


def prepare_representation_prefix(agent, laboratory, case):
    rng = Random(case['prefix_seed'])
    actions = [{'kind': 'trajectory', 'channels': [],
                'sequence': [{'u': rng.uniform(-1., 1.)} for _ in range(3)]}
               for _ in range(24)]
    feedback = agent._execute('query', {'actions': actions}, laboratory)
    if feedback.get('error'):
        raise ValueError(feedback['error'])
    agent.events.append({'step': -1, 'action': 'query', 'arguments': {'actions': actions},
        'feedback': feedback, 'usage': {'backend': 'common_paid_prefix', 'total_tokens': 0}})
    agent.common_interface += RESEARCH_TASK
    return {}


def protocol(source_trial, source_archive, *, model='gpt-6-luna', timeout_seconds=300.):
    plan = base_protocol(model=model, timeout_seconds=timeout_seconds)
    plan.update(study='anonymous_online_research_paid_boundaries',
        source_trial=str(Path(source_trial).resolve()),
        source_archive=str(Path(source_archive).resolve()),
        cases=[{'id': f'paid_{rows:03d}', **deepcopy(COUNTEREXAMPLE),
                'paid_prefix_rows': rows,
                'evaluation_seed': COUNTEREXAMPLE['seed'] + 300007} for rows in (72, 168)],
        shared_research_task=RESEARCH_TASK,
        runtime_snapshot=SNAPSHOT_FILES,
        prefix_protocol='Restore only the actual first 72 or 168 paid rows. The latter includes '
            'the first 96-row confirmation and advances that random stream once. No predictor, '
            'counterexample diagnosis, later evidence or prior selection is restored. '
            'Historical rows occupy the episode observation allowance but are not purchased again.',
        archive_protocol='Both arms receive all six original source investigations, with their '
            'authored code and paid evidence. No later compiled solver or revision archive is supplied.',
        cost_scope='costs includes inherited observation exposure; historical_prefix_cost separates '
            'that exposure from additional_cost. Summing historical exposure across branches '
            'does not represent newly incurred purchases.',
        outcome_protocol='Report frozen pre-acquisition forecasts, paid fit, observed errors, '
            'authored answers and subsequent decisions. A revision declaration, answer status or '
            'successful prediction alone does not establish a scientific research loop.',
        comparison_protocol='Both agents may compare their own executable Python hypotheses on '
            'their own proposed experiments before acquisition. The existing probe chooser '
            'recommends among paid-compatible candidates; the agent chooses whether to acquire. '
            'Record comparison, subsequent acquisition and actual candidate consequences separately. '
            'Direct confirmation remains available. This developmental repeat also includes the '
            'previously repaired source/continuation instructions; it is not an isolated ablation.',
        scope='One already opened development apparatus, two nested evidence boundaries and two '
            'online agents. Not four independent discoveries, unseen-family transfer or a test '
            'of recursive superiority. Compilation and transfer are postponed.')
    return plan


def prepare_paid_prefix(source_trial):
    """Restore existing evidence, without rerunning physics or supplying its old solver."""
    prior = json.loads(Path(source_trial).read_text())
    paid = prior['paid_observations']
    original_actions = [record['action'] for record in paid[:3]]
    lab = AnonymousHistoryLab(**{key: COUNTEREXAMPLE[key] for key in ('kind', 'seed')})
    if original_actions != lab.confirmation_actions(72, Random(COUNTEREXAMPLE['prefix_seed'])):
        raise ValueError('source trial does not have the registered 72-row paid prefix')
    first_confirmation = lab.confirmation_actions(96, Random(COUNTEREXAMPLE['confirmation_seed']))
    if [record['action'] for record in paid[3:7]] != first_confirmation:
        raise ValueError('source trial does not have the registered first confirmation')

    def prepare(agent, laboratory, case):
        count = case['paid_prefix_rows']
        records = deepcopy(paid[:3 if count == 72 else 7])
        rows = [row for record in records for row in laboratory.observation_rows(record)]
        if len(rows) != count:
            raise ValueError('paid evidence does not end at the registered boundary')
        agent.observations, agent.rows = records, rows
        historical = {key: 0 for key in agent.costs}
        historical['observation_cost'] = count
        historical['measurement_cost'] = sum(record.get('measurement_cost', 0) for record in records)
        agent.costs.update(historical)
        laboratory._costs.update(observation_cost=count,
            measurement_cost=historical['measurement_cost'], queries=len(records))
        if count == 168:
            laboratory.confirmation_actions(96, agent.confirmation_rng)
        agent.common_interface += RESEARCH_TASK
        agent._write_evidence()
        agent._prepare_program_state()
        agent.events.append({'step': -1, 'action': 'query',
            'arguments': {'actions': [record['action'] for record in records]},
            'feedback': {'observations': deepcopy(records), 'restored_paid_rows': count,
                         'observation_cost': 0, 'measurement_cost': 0},
            'usage': {'backend': 'restored_paid_prefix', 'total_tokens': 0}})
        return {'historical_prefix_cost': historical,
                'prefix_origin': {'source': str(Path(source_trial).resolve()),
                                  'rows': count, 'new_purchases': 0}}

    return prepare


def research_trace(result):
    """Expose chronological evidence without converting authored verdicts into success."""
    events = result['events']

    def decision_after(position):
        if position + 1 >= len(events):
            return None
        event = events[position + 1]
        return {'step': event.get('step'), 'action': event.get('action'),
                'arguments': {key: value for key, value in event.get('arguments', {}).items()
                              if key != 'code'}}

    traces = []
    for revision in result.get('revision_proposals', []):
        forecasts = revision.get('forecasts', [])
        usable = [forecast for forecast in forecasts
                  if forecast.get('returncode') == 0 and forecast.get('predictions') is not None]
        distinct = len({tuple(forecast['predictions']) for forecast in usable})
        event_position, observed_attempts, analysis_events = None, 0, []
        for position, event in enumerate(events):
            feedback = event.get('feedback', {})
            feedbacks = [(None, feedback), *[(step.get('step'), step.get('feedback', {}))
                                            for step in feedback.get('steps', [])]]
            for procedure_step, item in feedbacks:
                snapshot = item.get('revision') or {}
                if snapshot.get('index') != revision['index']:
                    continue
                if event_position is None and snapshot.get('observations'):
                    event_position = position
                attempts = snapshot.get('analysis_attempts', [])
                for attempt_index in range(observed_attempts, len(attempts)):
                    attempt = attempts[attempt_index]
                    analysis_events.append({'step': event.get('step'),
                        'procedure_step': procedure_step, 'attempt': attempt_index,
                        'readout': attempt['path'], 'execution': deepcopy(attempt.get('execution')),
                        'error': attempt.get('error'),
                        'authored_answer': deepcopy(attempt.get('authored_answer')),
                        'next_model_decision': decision_after(position)})
                observed_attempts = max(observed_attempts, len(attempts))
        question_id = revision.get('question_id')
        question = next((question for question in result.get('online_questions', [])
                         if (question.get('id') == question_id if question_id is not None
                             else question.get('revision_index') == revision['index'])), None)
        traces.append({'index': revision['index'], 'proposal': deepcopy(revision['proposal']),
            'evidence_cutoff': revision.get('evidence_cutoff'),
            'prior_fit': [{key: fit[key] for key in ('model', 'mse', 'returncode', 'error') if key in fit}
                          for fit in revision.get('prior_fit', [])],
            'successful_forecasts': len(usable), 'distinct_prediction_vectors': distinct,
            'predictions_differ': len(usable) > 1 and distinct > 1,
            'forecasts': [{key: forecast[key] for key in (
                'model', 'predictions', 'returncode', 'error', 'prospective_mse',
                'scored_rows', 'complete_experiment') if key in forecast} for forecast in forecasts],
            'status': revision.get('status'), 'authored_answer': revision.get('authored_answer'),
            'question_id': question_id, 'question': deepcopy(question),
            'acquisition_step': events[event_position].get('step') if event_position is not None else None,
            'next_action_after_acquisition': decision_after(event_position) if event_position is not None else None,
            'analysis_events': analysis_events})
    comparisons = []
    for position, event in enumerate(events):
        feedback = event.get('feedback', {})
        if feedback.get('operator') != 'compare_experiments' or 'comparison' not in feedback:
            continue
        comparison = feedback['comparison']
        comparisons.append({'step': event.get('step'), **{key: deepcopy(comparison[key])
            for key in ('index', 'status', 'evidence_cutoff', 'models', 'prior_fit',
                        'eligible_models', 'selected_experiment', 'predicted_separation',
                        'query_arguments', 'selection_policy', 'task_value') if key in comparison},
            'next_model_decision': decision_after(position),
            'linked_revisions': [trace['index'] for trace in traces
                if trace['proposal'].get('comparison_index') == comparison.get('index')]})
    return {'experiment_comparisons': comparisons, 'prospective_revisions': traces,
            'interpretation': 'These are actual recorded forecasts and decisions, not an automatic '
                'scientific-success verdict. Ordinary Python research may establish the same '
                'process without this interface; inspect the complete event trace.'}


def experimental_design_trace(result, *, mse_threshold=1e-10):
    """Summarize actual pre-submission discrimination; never generate evidence.

    Program invocations can repeat a revision inside both events and method_results.
    Index revisions once, keep chronology from the execution events, and omit their
    source bundles, raw observations and prediction inputs from this report.
    """
    def compact(mapping, keys):
        return {key: deepcopy(mapping[key]) for key in keys if key in mapping}

    def action_summary(actions):
        return [dict(compact(action, ('kind', 'channels')),
                     sequence_length=len(action.get('sequence', []))) for action in actions]

    def entries(events):
        for event in events:
            nested = event.get('feedback', {}).get('steps', [])
            if nested:
                yield from entries(nested)
            else:
                yield event

    timeline = list(entries(result.get('events', [])))
    if not timeline:
        timeline = list(entries(result.get('method_results', [])))
    method_proposals = {(record.get('step'), record.get('path')): record['proposal']
                        for record in result.get('method_results', [])
                        if record.get('path') and record.get('proposal')}
    revisions = {record['index']: record for record in result.get('revision_proposals', [])}
    comparisons = {record['index']: record for record in result.get('experiment_comparisons', [])}
    acquisition_positions, analysis_positions, submissions, confirmations, queries = {}, {}, [], [], []
    deferred_proposals = []
    for position, event in enumerate(timeline):
        feedback = event.get('feedback', {})
        method_proposal = method_proposals.get((event.get('step'),
            event.get('arguments', {}).get('path') or feedback.get('executed_path')))
        proposed = feedback.get('program_action') or event.get('proposal') or method_proposal or event
        arguments = proposed.get('arguments', {})
        revision = feedback.get('revision')
        if revision is not None:
            revisions.setdefault(revision['index'], revision)
            if revision.get('observations'):
                acquisition_positions.setdefault(revision['index'], position)
            if revision.get('analysis_is_agent_authored') and not revision.get('readout_error'):
                analysis_positions.setdefault(revision['index'], position)
        comparison = feedback.get('comparison')
        if comparison is not None:
            comparisons.setdefault(comparison['index'], comparison)
        confirmation = feedback.get('fresh_confirmation')
        deferred = feedback.get('status') in {'confirmation_not_authorized', 'submission_proposed'}
        if confirmation and not deferred:
            confirmations.append((position, confirmation))
        model = (proposed.get('model') or feedback.get('method_submission')
                 or (arguments.get('model') if proposed.get('action') == 'finish' else None))
        next_action = feedback.get('next_action', {})
        if not model and next_action.get('action') == 'finish':
            model = next_action.get('arguments', {}).get('model')
        if model or confirmation:
            declaration = {'position': position, 'step': event.get('step'), 'model': model}
            if deferred:
                deferred_proposals.append({**declaration, 'status': feedback['status']})
            else:
                submissions.append(declaration)
        actions = proposed.get('actions') or arguments.get('actions')
        if (actions and feedback.get('observations') and event.get('step', 0) >= 0
                and event.get('usage', {}).get('backend') not in
                {'common_paid_prefix', 'restored_paid_prefix'}):
            queries.append({'position': position, 'step': event.get('step'),
                'actions': action_summary(actions),
                'paid_rows': sum(len(batch.get('row_indices', [])) for batch in feedback['observations']),
                'revision_index': revision.get('index') if revision else None,
                'classification': 'prospective_revision' if revision else 'unclassified_explicit_query'})

    first_submission = submissions[0] if submissions else None
    for query in queries:
        query['before_first_submission'] = bool(first_submission is not None
            and query['position'] < first_submission['position'])
    final_confirmation = result.get('fresh_confirmation') or {}
    final_accepted = bool(result.get('model') and final_confirmation.get('passed') is True)
    final_position = submissions[-1]['position'] if submissions else None
    traces = []
    for index, revision in sorted(revisions.items()):
        proposal = revision.get('proposal', {})
        threshold = comparisons.get(proposal.get('comparison_index'), {}).get('mse_threshold', mse_threshold)
        prior = [compact(fit, ('model', 'returncode', 'mse', 'error'))
                 for fit in revision.get('prior_fit', [])]
        supported = [fit['model'] for fit in prior if fit.get('returncode') == 0
                     and isinstance(fit.get('mse'), (int, float)) and fit['mse'] <= threshold]
        forecasts = revision.get('forecasts', [])
        usable = [forecast for forecast in forecasts if forecast.get('model') in supported
                  and forecast.get('returncode') == 0 and forecast.get('predictions')]
        vectors = {tuple(forecast['predictions']) for forecast in usable}
        witness = None
        for row_index, values in enumerate(zip(*(forecast['predictions'] for forecast in usable))):
            if len(set(values)) > 1:
                witness = {'prediction_index': row_index,
                           'values': dict(zip((forecast['model'] for forecast in usable), values))}
                break
        measured = [forecast for forecast in usable if forecast.get('complete_experiment') is True
                    and isinstance(forecast.get('prospective_mse'), (int, float))]
        survivors = [forecast['model'] for forecast in measured if forecast['prospective_mse'] <= threshold]
        refuted = [forecast['model'] for forecast in measured if forecast['prospective_mse'] > threshold]
        observations = revision.get('observations', [])
        acquired_indices = [row for batch in observations for row in batch.get('row_indices', [])]
        answer = revision.get('authored_answer') or {}
        readout = bool(revision.get('analysis_is_agent_authored') and not revision.get('readout_error')
                       and answer.get('status') in {'supported', 'refuted'}
                       and set(answer.get('evidence', [])).intersection(acquired_indices))
        position, analyzed = acquisition_positions.get(index), analysis_positions.get(index)
        before_submission = (position is not None and first_submission is not None
                             and position < first_submission['position'])
        readout_before_submission = (analyzed is not None and first_submission is not None
                                     and analyzed < first_submission['position'])
        after_confirmation = (position is not None and any(when < position for when, _ in confirmations))
        followed_by_accepted = (final_accepted and analyzed is not None and final_position is not None
                               and analyzed < final_position)
        discrimination = bool(len(supported) >= 2 and len(vectors) >= 2 and survivors and refuted and readout)
        outcome = ('hypothesis_set_refuted' if measured and len(refuted) == len(usable) else
                   'supported_and_refuted' if survivors and refuted else
                   'no_measured_rejection' if survivors else 'incomplete_prospective_evidence')
        traces.append({'index': index, 'comparison_index': proposal.get('comparison_index'),
            'target': revision.get('target', proposal.get('target', 'response')),
            'offered_models': deepcopy(proposal.get('models', [])),
            'original_models': deepcopy(proposal.get('original_models', {})),
            'experiment_path': revision.get('experiment_path'),
            'actions': action_summary(revision.get('actions', [])),
            'evidence_cutoff': deepcopy(revision.get('evidence_cutoff')),
            'mse_threshold': threshold, 'prior_fit': prior, 'supported_models': supported,
            'predictions_differ': len(vectors) >= 2, 'disagreement_witness': witness,
            'forecasts': [dict(compact(forecast, ('model', 'execution_model', 'returncode', 'error',
                'prospective_mse', 'scored_rows', 'complete_experiment')),
                prediction_count=len(forecast.get('predictions') or [])) for forecast in forecasts],
            'measured_supported_models': survivors, 'measured_refuted_models': refuted,
            'measured_outcome': outcome,
            'paid_row_indices': acquired_indices, 'authored_readout': readout,
            'readout_path': (revision.get('analysis_attempts') or [{}])[-1].get('path', proposal.get('readout')),
            'readout_status': answer.get('status'), 'readout_evidence': deepcopy(answer.get('evidence', [])),
            'readout_error': revision.get('readout_error'),
            'acquisition_position': position, 'analysis_position': analyzed,
            'before_first_submission': before_submission, 'after_framework_confirmation': after_confirmation,
            'readout_before_first_submission': readout_before_submission,
            'selected_from_recorded_comparison': bool(proposal.get('comparison_index') in comparisons
                and (comparisons[proposal['comparison_index']].get('selected_experiment') or {}).get('actions')
                == revision.get('actions')),
            'followed_by_accepted_model': followed_by_accepted,
            'complete_discrimination': bool(discrimination and before_submission
                                           and readout_before_submission and followed_by_accepted)})
    comparison_traces = []
    for index, comparison in sorted(comparisons.items()):
        selected = comparison.get('selected_experiment') or {}
        comparison_traces.append({**compact(comparison, ('index', 'target', 'status', 'diagnosis',
            'models', 'eligible_models', 'evidence_cutoff', 'mse_threshold', 'selection_policy', 'task_value')),
            'prior_fit': [compact(fit, ('model', 'returncode', 'mse', 'error'))
                          for fit in comparison.get('prior_fit', [])],
            'selected_experiment': compact(selected, ('id', 'observation_cost', 'predicted_separation')),
            'experiments': [{**compact(experiment, ('id', 'row_count', 'observation_cost', 'predicted_separation')),
                'actions': action_summary(experiment.get('actions', []))}
                for experiment in comparison.get('comparisons', [])],
            'linked_revisions': [trace['index'] for trace in traces if trace['comparison_index'] == index]})
        control = comparison.get('experiment_selection_control')
        if control:
            comparison_traces[-1]['choice_control'] = {
                **compact(control, ('policy', 'seed', 'eligible_experiments', 'selected_experiment')),
                'greedy_suggestion': compact(control.get('greedy_suggestion', {}),
                    ('id', 'observation_cost', 'predicted_separation'))}
    return {'comparisons': comparison_traces, 'revisions': traces, 'explicit_queries': queries,
        'deferred_model_proposals': deferred_proposals,
        'first_submission': first_submission, 'final_model': result.get('model'),
        'final_confirmation': compact(final_confirmation, ('passed', 'mse', 'rows', 'returncode')),
        'complete_discrimination': any(trace['complete_discrimination'] for trace in traces),
        'interpretation': 'complete_discrimination requires supported rivals, prospective disagreement, '
            'paid separation and authored readout before the first submission, followed by an accepted model. '
            'This trace does not establish causal benefit or independent terminal accuracy. '
            'Unclassified explicit queries require inspection; absent classification is not failure.'}


def continuation_protocol(source):
    """Extend only token-truncated branches under their original total allowances."""
    source = Path(source).resolve()
    plan = json.loads((source / 'protocol.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('wait for every original episode and terminal evaluation to finish')
    if plan['study'] != 'anonymous_online_representation_ambiguity':
        raise ValueError('this continuation uses the fixed temporal representation study')
    expected = {(case['id'], arm) for case in plan['cases'] for arm in case.get('arms', plan['arms'])}
    actual = {(trial['case'], trial['arm']) for trial in summary['trials']}
    if actual != expected:
        raise ValueError('the original registered cohort has not all stopped')
    selected = {(trial['case'], trial['arm']) for trial in summary['trials']
                if trial['status'] == 'token_budget_exhausted'}
    if not selected:
        raise ValueError('the finished study has no token-truncated branches')
    cases = []
    for index, case in enumerate(plan['cases']):
        arms = plan['arms']
        ordered = case.get('arms', arms[index % len(arms):] + arms[:index % len(arms)])
        continued = [arm for arm in ordered if (case['id'], arm) in selected]
        if continued:
            cases.append({**deepcopy(case), 'arms': continued})
    original_limits = deepcopy(plan['limits'])
    if original_limits['token_budget'] >= 4_000_000:
        raise ValueError('the four-million total-token continuation has already been allocated')
    plan.update(cases=cases, resume_study=str(source), original_limits=original_limits,
        continuation_protocol='After the complete initial cohort stopped, continue all and only '
            'token_budget_exhausted branches. Retain the original cumulative call, observation, '
            'compute and Python-time limits; raise only total reported tokens to 4000000. '
            'Keep scientific files, same-arm paid evidence and original confirmation streams. '
            'No terminal labels, opposite-arm outcomes or scientific directions enter research.',
        cost_scope='costs is cumulative research across both phases. prior_research_cost remains '
            'charged, and continuation_cost reports genuinely added work. Original outcomes '
            'remain separately available; this is an added development allowance.')
    plan['limits']['token_budget'] = 4_000_000
    return plan


def prepare_research_continuation(plan):
    source = Path(plan['resume_study'])

    def prepare(agent, laboratory, case):
        workspace = source / case['id'] / agent.arm
        path = workspace / 'training.json'
        previous = json.loads(path.read_text())
        if previous['status'] != 'token_budget_exhausted':
            raise ValueError('only the registered token-truncated research can continue')
        agent.common_interface += plan['shared_research_task']
        agent.restore_research(previous, workspace, laboratory,
                               confirmation_seed=case['confirmation_seed'])
        return {'prior_research_cost': deepcopy(previous['costs']),
                'resume_origin': {'study': str(source), 'training': str(path),
                                  'status': previous['status']}}

    return prepare


def run_online(plan, output, *, provider_factory):
    output = Path(output).resolve()
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh output directory; retain prior development results')
    resuming = 'resume_study' in plan
    if resuming:
        original = Path(plan['resume_study'])
        knowledge = json.loads((original / 'initial_archive.json').read_text())
        sources = json.loads((original / 'source_outcomes.json').read_text())
        prepare = prepare_research_continuation(plan)
    else:
        knowledge, sources = load_research_archive(plan['source_archive'])
        prepare = (prepare_representation_prefix
                   if plan['study'] == 'anonymous_online_representation_ambiguity'
                   else prepare_paid_prefix(plan['source_trial']))
    representation_study = plan['study'] == 'anonymous_online_representation_ambiguity'
    output.mkdir(parents=True, exist_ok=True)
    # Reuse the existing source_at_run convention, preserving executable sources without hashes.
    for name in plan['runtime_snapshot']:
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(output / 'source_outcomes.json', sources)
    summary = run_study(plan, output, provider_factory=provider_factory,
                        knowledge=knowledge, prepare_prefix=prepare,
                        laboratory_factory=representation_laboratory if representation_study else None,
                        workers=plan.get('execution_workers', 1))
    for trial in summary['trials']:
        workspace = output / trial['case'] / trial['arm']
        trace = research_trace(json.loads((workspace / 'training.json').read_text()))
        write_json(workspace / 'research_trace.json', trace)
        trial['research_trace'] = str(workspace / 'research_trace.json')
        trial['comparison_count'] = len(trace['experiment_comparisons'])
        if representation_study:
            trial['parameter_pair'] = next(case['parameter_pair'] for case in plan['cases']
                                           if case['id'] == trial['case'])
            if not resuming:
                trial['post_prefix_cost'] = deepcopy(trial['costs'])
                trial['post_prefix_cost']['observation_cost'] -= plan['prefix_rows']
    if representation_study and 'independent_parameter_pairs' in plan:
        summary['independent_parameter_pairs'] = plan['independent_parameter_pairs']
    for field in ('cohort', 'independent_parameter_sets'):
        if field in plan:
            summary[field] = plan[field]
    summary['cost_scope'] = plan['cost_scope']
    summary['outcome_protocol'] = plan['outcome_protocol']
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('paid_boundaries', 'representation_ambiguity',
                                           'conditional_components'),
                        default='paid_boundaries')
    parser.add_argument('--source-trial', type=Path)
    parser.add_argument('--source-archive', type=Path)
    parser.add_argument('--resume-study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--timeout-seconds', type=float)
    args = parser.parse_args()
    if args.resume_study:
        if args.source_trial or args.source_archive:
            parser.error('resume-study restores the original inputs; omit source overrides')
        plan = continuation_protocol(args.resume_study)
    elif args.study == 'conditional_components':
        if args.source_trial:
            parser.error('conditional_components buys fresh prefixes; omit --source-trial')
        plan = conditional_protocol(args.source_archive, model=args.model,
                                    timeout_seconds=args.timeout_seconds or 600.)
    elif args.study == 'representation_ambiguity':
        if not args.source_archive:
            parser.error('representation_ambiguity requires --source-archive')
        if args.source_trial:
            parser.error('representation_ambiguity buys its own short prefixes; omit --source-trial')
        plan = representation_protocol(args.source_archive, model=args.model,
                                       timeout_seconds=args.timeout_seconds or 300.)
    else:
        if not args.source_trial or not args.source_archive:
            parser.error('paid_boundaries requires --source-trial and --source-archive')
        plan = protocol(args.source_trial, args.source_archive,
                        model=args.model, timeout_seconds=args.timeout_seconds or 300.)
    plan['provider']['auth_mode'] = verify_codex_subscription()
    summary = run_online(plan, args.output, provider_factory=lambda workspace:
        CodexCodeProvider(workspace, model=plan['model'],
                          reasoning_effort=plan['provider']['reasoning_effort'],
                          timeout_seconds=plan['provider']['timeout_seconds']))
    print(json.dumps({'status': summary['status'], 'trials': len(summary['trials'])}), flush=True)


if __name__ == '__main__':
    main()
