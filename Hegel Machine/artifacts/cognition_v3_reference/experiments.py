"""Matched sequences, targeted interventions and evaluation after learning."""
from __future__ import annotations

from dataclasses import asdict, replace
from statistics import mean
from time import perf_counter

from .controller import CognitiveAgent
from .environments import make_suite
from .evaluation import loss, point_loss, predict
from .language import Program
from .memory import KnowledgeLibrary
from .models import ResearchConfig
from .research_protocol import final_decision_summary, fresh_evidence_summary, summarize_research


def evaluate_revision_decisions(actions, evaluation, *, fallback, risk_quantile,
                                minimum_improvement, final_program=None, final_loss=None):
    """Score only recorded revision alternatives after the agent has returned."""
    started = perf_counter()
    cached = [(final_program, final_loss)] if final_program is not None and final_loss is not None else []
    compared = []
    additional_programs = 0

    def measured(encoded):
        nonlocal additional_programs
        program = Program.from_dict(encoded)
        if program not in compared:
            compared.append(program)
        for previous, value in cached:
            if previous == program:
                return value
        value = loss(program, evaluation, fallback, risk_quantile)
        cached.append((program, value))
        additional_programs += 1
        return value

    counts = {'revision_events': 0, 'scored_revision_events': 0, 'wrongly_accepted_events': 0,
              'rollback_events': 0, 'scored_rollback_events': 0, 'wrongly_rejected_events': 0,
              'missing_program_records': 0}
    events = []
    for index, action in enumerate(actions):
        kind, details = action['action'], action.get('details', {})
        if kind not in {'revise', 'rollback'}:
            continue
        revision = kind == 'revise'
        counts['revision_events' if revision else 'rollback_events'] += 1
        first_name, second_name = (('before_program', 'after_program') if revision
                                   else ('rejected_program', 'restored_program'))
        event = {'action_index': index, 'round': action.get('round'), 'action': kind}
        if details.get(first_name) is None or details.get(second_name) is None:
            counts['missing_program_records'] += 1
            events.append({**event, 'scored': False, 'reason': 'missing_program_record'})
            continue
        first_loss, second_loss = measured(details[first_name]), measured(details[second_name])
        if revision:
            wrong = second_loss > first_loss + minimum_improvement
            counts['scored_revision_events'] += 1
            counts['wrongly_accepted_events'] += wrong
            event.update({'before_loss': first_loss, 'after_loss': second_loss,
                          'wrongly_accepted': wrong})
        else:
            wrong = first_loss < second_loss - minimum_improvement
            counts['scored_rollback_events'] += 1
            counts['wrongly_rejected_events'] += wrong
            event.update({'rejected_loss': first_loss, 'restored_loss': second_loss,
                          'wrongly_rejected': wrong})
        events.append({**event, 'scored': True})
    return {**counts, 'events': events,
            'wrong_acceptance_rate': (counts['wrongly_accepted_events'] / counts['scored_revision_events']
                                      if counts['scored_revision_events'] else None),
            'wrong_rejection_rate': (counts['wrongly_rejected_events'] / counts['scored_rollback_events']
                                     if counts['scored_rollback_events'] else None),
            'evaluation_rows': len(evaluation), 'unique_programs_compared': len(compared),
            'additional_programs_scored': additional_programs,
            'additional_evaluation_units': additional_programs * len(evaluation),
            'wall_time_seconds': perf_counter() - started,
            'minimum_improvement': minimum_improvement,
            'evidence': 'independent final episode evaluation, accessed after learning; recorded alternatives only'}


def _library_identifiers(library):
    payload = library.to_dict()
    return {row['concept_id'] for row in payload.get('concepts', [])} | {
        row['template_id'] for row in payload.get('abstracts', [])}


def run_sequence(config: ResearchConfig | None = None, *, library: KnowledgeLibrary | None = None,
                 erase_before_transfer: bool = False, remove_concept_ids: set[str] | None = None) -> dict:
    config = config or ResearchConfig()
    agent = CognitiveAgent(config, library)
    episodes, intervened = [], False
    for index, environment in enumerate(make_suite(config.seed, partition=config.partition)):
        metadata = environment.evaluation_metadata
        intervention = None
        if not intervened and metadata['phase'] == 'transfer' and (erase_before_transfer or remove_concept_ids):
            before = _library_identifiers(agent.library)
            if erase_before_transfer:
                agent.library = KnowledgeLibrary()
                action, requested = 'erase_acquired_library', sorted(before)
            else:
                requested = sorted(remove_concept_ids or set())
                agent.library = agent.library.without(set(requested))
                action = 'erase_specific_concept_and_derivatives'
            intervention = {'action': action, 'requested_ids': requested,
                            'removed_ids': sorted(before - _library_identifiers(agent.library))}
            intervened = True
        kind = 'squared_error' if environment.spec.risk_quantile == 0.5 else 'pinball'
        started = perf_counter()
        try:
            result = agent.run(environment)
        except Exception as error:
            # Failures remain in the denominator; unknown work is not zero work.
            payload = {'episode_index': index, 'task_id': environment.spec.task_id, 'status': 'failed',
                       'error_type': type(error).__name__, 'error': str(error),
                       'observation_cost': None, 'compute_cost': None,
                       'wall_time_seconds': perf_counter() - started,
                       'actions': [], 'memory_events': [], 'selected_memory_id': None,
                       'stop_reason': 'runtime_failure', 'evaluation': {
                           'task_metadata': metadata, 'rows': 0, 'loss': None,
                           'initial_prediction_loss': None, 'loss_kind': kind,
                           'risk_quantile': environment.spec.risk_quantile}}
        else:
            learning_wall = perf_counter() - started
            # No final evaluation outcome reaches learning or memory revision.
            evaluation = environment.evaluation_rows()
            test_loss = loss(result.program, evaluation, result.fallback, environment.spec.risk_quantile)
            initial_loss = mean(point_loss(result.fallback, row.target, environment.spec.risk_quantile)
                                for row in evaluation)
            payload = result.to_dict()
            payload.update({'episode_index': index, 'status': 'completed', 'wall_time_seconds': learning_wall,
                            'evaluation': {'task_metadata': metadata, 'rows': len(evaluation), 'loss': test_loss,
                                           'initial_prediction_loss': initial_loss, 'loss_kind': kind,
                                           'risk_quantile': environment.spec.risk_quantile,
                                           'underprediction_rate': mean(predict(result.program, row, result.fallback) < row.target
                                                                        for row in evaluation),
                                           'prediction_examples': [
                                               {'features': row.features, 'target': row.target,
                                                'prediction': predict(result.program, row, result.fallback)}
                                               for row in evaluation[:4]]}})
            payload['evaluation']['final_counterfactual'] = evaluate_revision_decisions(
                payload['actions'], evaluation, fallback=result.fallback,
                risk_quantile=environment.spec.risk_quantile,
                minimum_improvement=config.minimum_improvement,
                final_program=result.program, final_loss=test_loss)
            if environment.spec.decision_costs is not None and result.uncertainty_model:
                from .evaluation import decision_metrics
                payload['evaluation']['decision'] = decision_metrics(
                    result.program, evaluation, result.fallback, result.uncertainty_model,
                    environment.spec.decision_costs)
        if intervention:
            payload['intervention'] = intervention
        episodes.append(payload)
    completed = [row for row in episodes if row['status'] == 'completed']
    groups, phases = {}, {}
    for kind in sorted({row['evaluation']['loss_kind'] for row in episodes}):
        rows = [row for row in completed if row['evaluation']['loss_kind'] == kind]
        groups[kind] = {'tasks': len(rows), 'attempted_tasks': sum(row['evaluation']['loss_kind'] == kind for row in episodes),
                        'mean_test_loss': mean(row['evaluation']['loss'] for row in rows) if rows else None,
                        'mean_initial_loss': mean(row['evaluation']['initial_prediction_loss'] for row in rows) if rows else None}
    for phase in sorted({row['evaluation']['task_metadata']['phase'] for row in episodes}):
        rows = [row for row in episodes if row['evaluation']['task_metadata']['phase'] == phase]
        phases[phase] = {'tasks': len(rows), 'completed_tasks': sum(row['status'] == 'completed' for row in rows),
                         'observation_cost': sum(row['observation_cost'] or 0 for row in rows),
                         'compute_cost': sum(row['compute_cost'] or 0 for row in rows),
                         'wall_time_seconds': sum(row['wall_time_seconds'] for row in rows),
                         'direct_reuses': sum(row.get('selected_operation') == 'reuse' for row in rows),
                         'abstract_reuses': sum(row.get('selected_operation') == 'abstract_reuse' for row in rows)}
    total = len(episodes)
    rates = {'prediction_coverage': len(completed) / total if total else 0.0,
             'failure_rate': (total - len(completed)) / total if total else 0.0,
             'refusal_rate': sum(row['stop_reason'] in {'refused', 'abstained', 'rejected'} for row in episodes) / total if total else 0.0,
             'revision_rate': sum(any(action['action'] == 'revise' for action in row['actions']) for row in episodes) / total if total else 0.0,
             'rollback_rate': sum(any(action['action'] == 'rollback' for action in row['actions']) for row in episodes) / total if total else 0.0}
    cumulative, observation_cost, compute_cost, wall = [], 0, 0, 0.0
    for row in episodes:
        observation_cost += row['observation_cost'] or 0
        compute_cost += row['compute_cost'] or 0
        wall += row['wall_time_seconds']
        cumulative.append({'episode_index': row['episode_index'], 'task_id': row['task_id'],
                           'observation_cost': observation_cost, 'compute_cost': compute_cost, 'wall_time_seconds': wall})
    attempted_usage = [row['llm_usage'] for row in completed if row.get('llm_usage', {}).get('calls', 0)]
    failed_model_work_unknown = len(completed) != total and config.proposer != 'search'
    llm_usage = {'calls': None if failed_model_work_unknown else sum(usage['calls'] for usage in attempted_usage)}
    for key in ('input_tokens', 'output_tokens', 'total_tokens'):
        complete = not failed_model_work_unknown and all(usage.get(key) is not None for usage in attempted_usage)
        llm_usage[key] = sum(usage[key] for usage in attempted_usage) if complete else None
    llm_usage['costs_complete'] = all(llm_usage[key] is not None for key in ('calls', 'input_tokens', 'output_tokens', 'total_tokens'))
    decisions = [row['evaluation']['decision'] for row in completed if 'decision' in row['evaluation']]
    decision_summary = {'tasks': len(decisions), **{
        key: mean(row[key] for row in decisions) if decisions else None
        for key in ('mean_decision_loss', 'coverage', 'mean_interval_width', 'worst_case_loss')}}
    return {'experiment': 'cognitive_task_sequence', 'config': asdict(config), 'partition': config.partition,
            'task_count': total, 'completed_tasks': len(completed), 'rates': rates,
            'metrics_by_loss': groups, 'metrics_by_phase': phases,
            'total_observation_cost': observation_cost, 'total_compute_cost': compute_cost,
            'total_wall_time_seconds': wall, 'cumulative_costs': cumulative,
            'costs_complete': len(completed) == total and llm_usage['costs_complete'], 'llm_usage': llm_usage,
            'decision_summary': decision_summary,
            'fresh_evidence': fresh_evidence_summary(episodes, config.minimum_improvement),
            'final_counterfactual': final_decision_summary(episodes),
            'evaluation_rows': sum(row['evaluation']['rows'] for row in episodes),
            'concepts_learned': len(agent.library.concepts),
            'direct_reuse_tasks': sum(row.get('selected_operation') == 'reuse' for row in episodes),
            'abstract_reuse_tasks': sum(row.get('selected_operation') == 'abstract_reuse' for row in episodes),
            'memory_revisions': sum(event['action'] in {'restricted', 'revoked'} for row in episodes for event in row['memory_events']),
            'episodes': episodes, 'knowledge': agent.library.to_dict(),
            'rate_definitions': {'prediction_coverage': 'completed predictions / all attempted episodes',
                                 'refusal_rate': 'explicit refused, abstained or rejected terminal episodes / all attempted episodes',
                                 'revision_rate': 'episodes with an executed working-hypothesis revision / all attempted episodes'},
            'limitations': ['All partitions use public synthetic generators; partition labels are evaluator-only.',
                            'Adaptive validation selects programs and is not a confirmatory holdout.',
                            'Compute units count candidate-observation work; actual learning wall time is separate.',
                            'Failed episodes retain their denominator; unavailable costs are marked incomplete.']}


def compare(config: ResearchConfig | None = None, *, include_llm: bool = False,
            concept_id: str | None = None, systems: tuple[str, ...] | None = None) -> dict:
    base = replace(config or ResearchConfig(), proposer='search')
    configurations = {
        'fixed_model': replace(base, strategy='fixed', use_memory=False, active_probes=False),
        'program_search': replace(base, strategy='search', use_memory=False, active_probes=False),
        'without_active_experiments': replace(base, active_probes=False),
        'without_knowledge': replace(base, use_memory=False),
        'single_step_planning': replace(base, joint_planning=False, planning_depth=1),
        'without_abstractions': replace(base, use_abstractions=False),
        'without_structural_mapping': replace(base, structural_transfer=False),
        'without_generated_probes': replace(base, synthesize_probes=False),
        'raw_disagreement': replace(base, experiment_policy='disagreement'),
        'retrospective_selection': replace(base, selection_policy='retrospective'),
        'without_warm_start': replace(base, use_warm_start=False),
        'cognitive_agent': base}
    if include_llm:
        configurations['llm_code_agent'] = replace(base, strategy='search', use_memory=False,
                                                   joint_planning=False, proposer='codex')
    if systems is not None:
        unknown = set(systems) - set(configurations) - {'erase_acquired_knowledge', 'erase_specific_concept'}
        if unknown:
            raise ValueError(f'unknown comparison systems: {sorted(unknown)}')
        configurations = {name: value for name, value in configurations.items() if name in systems}
    runs = {name: run_sequence(configuration) for name, configuration in configurations.items()}
    if systems is None or 'erase_acquired_knowledge' in systems:
        runs['erase_acquired_knowledge'] = run_sequence(base, erase_before_transfer=True)
    if systems is None or 'erase_specific_concept' in systems:
        if concept_id is None:
            reference = runs.get('cognitive_agent')
            if reference is None:
                reference = run_sequence(base)
                runs['cognitive_agent'] = reference
            first_transfer = next((row['episode_index'] for row in reference['episodes']
                                   if row['evaluation']['task_metadata']['phase'] == 'transfer'), len(reference['episodes']))
            acquired = [identifier for row in reference['episodes'] if row['episode_index'] < first_transfer
                        for identifier in row.get('concepts_added', [])]
            concept_id = acquired[0] if acquired else None
        if concept_id is not None:
            runs['erase_specific_concept'] = run_sequence(base, remove_concept_ids={concept_id})
    comparison = [{'system': name, 'metrics_by_loss': run['metrics_by_loss'], 'rates': run['rates'],
                   'observation_cost': run['total_observation_cost'], 'compute_cost': run['total_compute_cost'],
                   'wall_time_seconds': run['total_wall_time_seconds'], 'llm_usage': run['llm_usage'],
                   'direct_reuse_tasks': run['direct_reuse_tasks'], 'abstract_reuse_tasks': run['abstract_reuse_tasks'],
                   'fresh_evidence': run['fresh_evidence'], 'final_counterfactual': run['final_counterfactual'],
                   'memory_revisions': run['memory_revisions']}
                  for name, run in runs.items()]
    return {'experiment': 'matched_resource_cognition_comparison', 'partition': base.partition,
            'observation_budget_per_task': base.observation_budget, 'compute_budget_per_task': base.compute_budget,
            'seed': base.seed, 'specific_concept_intervention_id': concept_id, 'comparison': comparison, 'runs': runs,
            'interpretation': 'Report quality and costs together; no result is required to favor the cognitive system.',
            'llm_control': {'enabled': include_llm, 'maximum_calls_per_task': base.llm_calls_per_task,
                            'maximum_tokens_per_task': base.llm_token_budget}}


def run_research(config: ResearchConfig | None = None, seeds: list[int] | tuple[int, ...] = (7, 11, 19),
                 partition: str = 'development', *, include_llm: bool = False,
                 systems: tuple[str, ...] | None = None) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('research seeds must be a nonempty sequence without duplicates')
    base = replace(config or ResearchConfig(), partition=partition)
    comparisons = [compare(replace(base, seed=seed), include_llm=include_llm, systems=systems) for seed in seeds]
    result = summarize_research(comparisons, list(seeds), partition)
    result['config'] = asdict(base)
    return result
