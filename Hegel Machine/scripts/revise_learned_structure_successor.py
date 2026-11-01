#!/usr/bin/env python3
"""Consume a failed successor's paid rows before any new confirmation."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from scripts.run_learned_structure_successor import freeze_runtime, load_successor_scene
from scripts.run_native_decision_pair import load_prepared, ref
from scripts.run_partial_role_parent import load, save, plain
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry, observations
from hegel_machine.cognition.parallel_prediction import predict_rows


ROOT = Path(__file__).resolve().parents[1]


def restore(source):
    """Restore unique pool occurrences and already exposed trajectories by reference."""
    source = Path(source)
    scene = load_successor_scene(source)
    agent, frame, _ = load_prepared(source, scene=scene)
    registration = load(source / 'registration.json')
    pool, refs, working = [], [], {}
    parent = None
    for arm in registration['operations']:
        compared = load(source / arm / 'comparison.json')
        if not compared['full_comparison']:
            raise ValueError('The retained pool was not completely compared')
        for row in compared['scores']:
            location = row['candidate_ref']
            if location in refs:
                if arm == 'challenger' and location == compared['selected_ref']:
                    parent = pool[refs.index(location)]
                continue
            entry = decode_entry(read_method_reference(location, source))
            program = entry['candidate'].program
            pool.append(entry)
            refs.append(location)
            working[program] = row['loss']
            if arm == 'challenger' and location == compared['selected_ref']:
                parent = entry
    if parent is None:
        raise ValueError('The retained structural parent is not in the complete pool')
    parent['fallback'] = frame['fallback']
    confirmation = load(source / 'confirmation/receipt.json')
    batch = load(source / 'confirmation/observations.json')['batch']
    exposed = observations(batch['observations'])
    frozen = load(source / 'confirmation/frozen_predictors.json')
    exposed_scores = {Program.from_dict(item['program']): confirmation['arms'][arm]['mse']
                      for arm, item in frozen.items()}
    if any(item['fallback'] != frame['fallback'] for item in frozen.values()):
        raise ValueError('Retained scores use a different prediction fallback')
    # Preserve entire histories. The old confirmation is now development evidence.
    paid = [*scene['training'], *scene['validation'], *exposed]
    transitions = list(scene['transitions'])
    for location in scene['references']['validation']:
        transitions.extend(read_method_reference(location, source).get('transitions', []))
    transitions.extend(batch.get('transitions', []))
    return dict(agent=agent, frame=frame, scene=scene, parent=parent, pool=pool, refs=refs,
        working=working, exposed=exposed, exposed_scores=exposed_scores, paid=paid,
        transitions=transitions, registration=registration)


def paid_pool_scores(scene, budget):
    """Reuse original training/working losses and score only missing exposed rows."""
    nt, nv, ne = len(scene['scene']['training']), len(scene['scene']['validation']), len(scene['exposed'])
    scores, exposed = {}, dict(scene['exposed_scores'])
    before = budget.spent
    for entry in scene['pool']:
        program = entry['candidate'].program
        if program in scores:
            continue
        if program not in exposed:
            if not budget.spend(ne):
                raise RuntimeError('The complete retained comparison is not funded')
            exposed[program] = loss(program, scene['exposed'], scene['frame']['fallback'], .5)
        scores[program] = (nt * entry['training_loss'] + nv * scene['working'][program]
                          + ne * exposed[program]) / (nt + nv + ne)
    return scores, exposed, budget.spent - before


def register_paid_confirmation(source, output, *, compute_limit=1000):
    """Freeze one new readout of an already completed paid joint revision."""
    source, output = Path(source), Path(output)
    completed = load(source / 'receipt.json')
    previous = load(source / 'registration.json')
    original = Path(previous['source'])
    original_cfg = load(original / 'registration.json')
    original_costs = load(original / 'receipt.json')['cumulative_costs']
    if (not completed['full_comparison'] or not completed['all_paid_passed']
            or completed['fresh_confirmation'] is not None):
        raise ValueError('A completed paid revision without a new confirmation is required')
    rows = original_cfg['confirmation_rows']
    seed = original_cfg['confirmation_seed'] + 17017
    cost_limit = previous['registered_original_total'] - completed['cumulative_current_successor_compute']
    prior_review = source.parent / 'post_freeze'
    owner = next(q for q in load(prior_review / 'common/questions.json')
                 if q['id'] == 'task:method_choice')
    review_limit = 2 + len(owner['method_use_plan']['native_return_refs'])
    unit_limit = min(compute_limit, cost_limit)
    if 2 * rows + review_limit > unit_limit:
        raise ValueError('The existing balance does not cover the complete confirmation and return')
    if (original_costs['observation_cost'] + rows > original_cfg['observation_limit']
            or original_costs['measurement_cost'] + 12 * rows > original_cfg['measurement_limit']):
        raise ValueError('The unchanged observation or measurement allowance cannot cover confirmation')
    output.mkdir(parents=True, exist_ok=False)
    cfg = dict(source=str(source), original_source=str(original),
        source_task_ref=original_cfg['source_task_ref'], prior_method_review=str(prior_review),
        threshold=completed['threshold'], confirmation_rows=rows, confirmation_seed=seed,
        previous_compute=completed['cumulative_current_successor_compute'],
        previous_observation_cost=original_costs['observation_cost'],
        previous_measurement_cost=original_costs['measurement_cost'],
        total_compute_limit=previous['registered_original_total'], remaining_compute=cost_limit,
        unit_compute_limit=unit_limit,
        method_review_compute_limit=review_limit, active_wall_seconds=1800,
        observation_limit=original_cfg['observation_limit'], measurement_limit=original_cfg['measurement_limit'],
        source_cost_refs=[ref(source / 'receipt.json'), ref(original / 'receipt.json'),
                          *previous['source_cost_refs']],
        source_selected_ref=ref(source / 'selected_entry.json'),
        complete_pool=completed['compared_candidates'],
        confirmation_comparator_ref=load(original / 'challenger/outcome.json')['selected_ref'],
        comparison_kind='additional_research_vs_retained_parent',
        scope='one frozen original-contract confirmation, knowledge retention and native return',
        stop_rule='one shared fresh readout; preserve failure; no refits, retries or successor launch',
        model_calls=0)
    save(output / 'registration.json', cfg)
    freeze_runtime(output, source)
    return cfg


def load_paid_confirmation_knowledge(output):
    """Resolve the new knowledge against its retained source and paid rows."""
    from scripts.run_partial_role_successors import apply_knowledge_changes
    from hegel_machine.cognition.memory import KnowledgeLibrary

    output = Path(output)
    saved = load(output / 'knowledge_delta.json')
    scene = restore(saved['original_source'])
    result = apply_knowledge_changes(plain(scene['agent'].library.to_dict()), saved['changes'])
    for binding in saved['support_example_refs']:
        result['concepts'][binding['concept_index']]['scopes'][binding['scope_index']][
            'support_examples'] = plain([asdict(scene['paid'][i]) for i in binding['paid_row_indices']])
    return KnowledgeLibrary.from_dict(result)


def confirm_paid_revision(output):
    """Confirm the saved selected model without repeating selection or research."""
    from scripts.retain_confirmed_structure import paid_support_mask, reference_scope_examples
    from scripts.run_partial_role_successors import knowledge_changes
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return,
    )
    from hegel_machine.cognition.method_feedback import ensure_python_method_topic
    from hegel_machine.cognition.question_routing import consume_method_return_agenda

    output = Path(output)
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Execute the freshly frozen current confirmation entry')
    if (output / 'started.json').exists():
        raise ValueError('Retain the performed readout; do not repeat confirmation')
    cfg = load(output / 'registration.json')
    source = Path(cfg['source'])
    budget = SearchBudget(cfg['unit_compute_limit'])
    timing = ExecutionTiming(('restore', 'confirmation', 'knowledge', 'method_return'))
    observation_cost, measurement_cost, stage = 0, 0, 'restore'
    save(output / 'started.json', {'source': cfg['source'], 'status': 'started'})
    try:
        with timing.phase('restore'):
            scene = restore(cfg['original_source'])
            selected = decode_entry(read_method_reference(cfg['source_selected_ref'], output))
            incumbent = decode_entry(read_method_reference(cfg['confirmation_comparator_ref'], output))
            comparison = load(source / 'comparison.json')
            if not comparison['full_comparison'] or len(comparison['squared_errors']) != len(scene['paid']):
                raise ValueError('The saved complete comparison must cover all current paid rows')
            with AnonymousObjectLab(**scene['registration']['task']) as lab:
                probe = lab.random_probe(cfg['confirmation_seed'], length=cfg['confirmation_rows'],
                                         name='paid_joint_revision_confirmation')
            paid_batches = [*scene['scene']['references']['training'],
                *scene['scene']['references']['validation'],
                ref(Path(cfg['original_source']) / 'confirmation/observations.json')]
            for location in paid_batches:
                document = read_method_reference(location, output)
                batch = document.get('batch', document)
                previous_probe = batch.get('probe') or document['probe']
                if list(probe.sequence) == previous_probe['sequence']:
                    raise ValueError('The proposed confirmation repeats an already observed trajectory')
            save(output / 'confirmation_plan.json', {'seed': cfg['confirmation_seed'],
                'probe': asdict(probe), 'paid_probe_refs': paid_batches,
                'new_sequence': True, 'responses_obtained': False})
            # Both immutable programs are saved before any new response is obtained.
            frozen = {name: {'program': entry['candidate'].program.to_dict(), 'selected_ref': reference}
                      for name, entry, reference in (
                          ('selected', selected, cfg['source_selected_ref']),
                          ('incumbent', incumbent, cfg['confirmation_comparator_ref']))}
            save(output / 'frozen_predictors.json', frozen)
        stage = 'confirmation'
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('confirmation'), AnonymousObjectLab(**scene['registration']['task']) as lab:
                rows = cfg['confirmation_rows']
                if not budget.spend(2 * rows):
                    raise RuntimeError('Confirmation prediction allowance was exhausted')
                batch = lab.acquire(probe, ('fine',))
                observation_cost, measurement_cost = lab.observation_cost, lab.measurement_cost
                save(output / 'confirmation_observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
                confirmation = {'rows': rows, 'threshold': cfg['threshold'], 'seed': cfg['confirmation_seed']}
                for name, entry in [('selected', selected), ('incumbent', incumbent)]:
                    values = predict_rows(entry['candidate'].program, batch.observations)
                    mse = mean((value - row.target) ** 2 for value, row in zip(values, batch.observations))
                    confirmation[name] = {'predictions': values, 'mse': mse, 'passed': mse <= cfg['threshold']}
                save(output / 'confirmation.json', confirmation)
            recorded = {'stored': False, 'reason': 'confirmation_not_passed'}
            stage = 'knowledge'
            if confirmation['selected']['passed']:
                with timing.phase('knowledge'):
                    agent, paid = scene['agent'], scene['paid']
                    before = plain(agent.library.to_dict())
                    mask = paid_support_mask(selected['candidate'].program, paid, comparison['squared_errors'])
                    batch_refs = [*scene['scene']['references']['training'],
                        *scene['scene']['references']['validation'],
                        ref(Path(cfg['original_source']) / 'confirmation/observations.json')]
                    batches = []
                    for location in batch_refs:
                        document = read_method_reference(location, output)
                        retained = document.get('batch', document)
                        batches.append({'probe': retained.get('probe') or document['probe'],
                                        'channels': retained['channels']})
                    recorded = agent.remember_confirmed_selection(selected,
                        agent._knowledge_spec(scene['scene']['spec']), paid, [],
                        confirmed=True, confirmation_loss=confirmation['selected']['mse'],
                        confirmation_rows=rows, selected_channels=('fine',), batches=batches,
                        confirmation_ref=ref(output / 'confirmation.json'), paid_support_matches=mask)
                    after = plain(agent.library.to_dict())
                    changes = deepcopy(knowledge_changes(before, after))
                    supports = reference_scope_examples(changes, paid, mask)
                    save(output / 'knowledge_delta.json', {'original_source': cfg['original_source'],
                        'changes': changes, 'support_example_refs': supports, 'paid_batch_refs': batch_refs,
                        'confirmation_ref': ref(output / 'confirmation.json')})
                    restored = load_paid_confirmation_knowledge(output)
                    if plain(restored.to_dict()) != after:
                        raise ValueError('Reference-backed confirmed knowledge did not restore equally')
                    recorded.update(knowledge_ref=ref(output / 'knowledge_delta.json'),
                        old_concepts=len(before['concepts']), concepts=len(after['concepts']),
                        support_examples_saved_by_reference=len(supports), readback_equal=True)
            stage = 'method_return'
            save(output / 'research_receipt.json', {
                'status': 'confirmed' if confirmation['selected']['passed'] else 'confirmation_failed',
                'execution_complete': True, 'comparison_kind': cfg['comparison_kind'],
                'source_task_ref': cfg['source_task_ref'], 'selected_ref': cfg['source_selected_ref'],
                'confirmation_ref': ref(output / 'confirmation.json'),
                'comparison_ref': ref(source / 'comparison.json'), 'method_pair_ref': None})
            # This is a revised world answer versus a retained predictor, not a
            # new same-start method pair. Return it without fabricating a label.
            with timing.phase('method_return'):
                owner = next(q for q in load(Path(cfg['prior_method_review']) / 'common/questions.json')
                             if q['id'] == 'task:method_choice')
                save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
                common = CommonObjectDependencyMethods(output / 'common', [],
                    budget=SearchBudget(cfg['method_review_compute_limit']), actions=[],
                    source_task_ref=cfg['source_task_ref']).common
                ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
                    topic_ref=ref(output / 'previous_topic.json'))
                mission = deepcopy(owner['mission'])
                mission['allocation']['compute'] += cfg['method_review_compute_limit']
                feedback = record_native_parent_research_return(common,
                    source_task_ref=cfg['source_task_ref'], result_ref=ref(output / 'research_receipt.json'),
                    before_ref=ref(source / 'comparison.json'), comparison_ref=ref(source / 'comparison.json'),
                    confirmation_ref=ref(output / 'confirmation.json'),
                    cost_refs=[ref(source / 'receipt.json'), ref(output / 'receipt.json')], mission=mission)
                returned = consume_method_return_agenda(common, feedback, None,
                    compute_limit=cfg['method_review_compute_limit'])
                budget.spend(common.costs['compute_cost'])
            result = {'status': 'confirmed' if confirmation['selected']['passed'] else 'confirmation_failed',
                'confirmation': confirmation, 'knowledge': recorded, 'method_return': returned,
                'compute_cost': budget.spent, 'cumulative_compute': cfg['previous_compute'] + budget.spent,
                'remaining_compute': cfg['remaining_compute'] - budget.spent,
                'unit_remaining_compute': budget.remaining, 'observation_cost': observation_cost,
                'measurement_cost': measurement_cost, 'new_fits': 0, 'model_calls': 0,
                'source_cost_refs': cfg['source_cost_refs'], 'full_comparison_reused': True,
                'old_candidate_pool': cfg['complete_pool'], 'old_successes_and_failures_preserved': True,
                'comparison_kind': cfg['comparison_kind'], 'independent_method_gain': None,
                'execution_timing': timing.result()}
            save(output / 'receipt.json', result)
            return result
    except BaseException as error:
        save(output / 'failure.json', {'stage': stage, 'error': repr(error),
            'compute_known': budget.spent, 'observation_cost': observation_cost,
            'measurement_cost': measurement_cost, 'execution_timing': timing.result()})
        raise


def register_method_research(source, output):
    """Keep the newly confirmed world answer frozen during paid method research."""
    source, output = Path(source), Path(output)
    receipt, previous = load(source / 'receipt.json'), load(source / 'registration.json')
    if receipt['status'] != 'confirmed':
        raise ValueError('Complete the registered world confirmation before method research')
    output.mkdir(parents=True, exist_ok=False)
    cfg = dict(source=str(source), original_source=previous['original_source'],
        joint_source=previous['source'], source_task_ref=previous['source_task_ref'],
        before_entry_ref=previous['confirmation_comparator_ref'],
        after_entry_ref=previous['source_selected_ref'],
        previous_compute=receipt['cumulative_compute'], remaining_compute=receipt['remaining_compute'],
        total_compute_limit=previous['total_compute_limit'], threshold=previous['threshold'],
        active_wall_seconds=1800, max_attempts=48, max_candidates=1,
        training_role='original123', assessment_role='old24_plus_exposed48_development',
        exclude_fresh_confirmation_ref=ref(source / 'confirmation_observations.json'),
        complete_pool_ref=ref(Path(previous['source']) / 'comparison.json'),
        source_cost_refs=[ref(source / 'receipt.json'), *previous['source_cost_refs']],
        scope='actual return to proposed transfer partition; one matched paid development pair',
        stop_rule='one generated partition, original paid baseline, complete old pool; no acquisition or retry',
        model_calls=0)
    save(output / 'registration.json', cfg)
    freeze_runtime(output, source)
    return cfg


def return_paid_method_study(source, output):
    """Consume a completed study by reference without repeating its science."""
    from hegel_machine.cognition.common_object_research import CommonObjectDependencyMethods
    from hegel_machine.cognition.method_action_research import record_native_partition_study
    from hegel_machine.cognition.method_feedback import ensure_python_method_topic
    from hegel_machine.cognition.question_agenda import debit_mission_compute
    from hegel_machine.cognition.question_routing import consume_method_return_agenda

    source, output = Path(source), Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    cfg, receipt = load(source / 'registration.json'), load(source / 'receipt.json')
    study = load(source / 'study.json')
    owner = next(q for q in load(source / 'common/questions.json') if q['id'] == 'task:method_choice')
    metadata_cost = 3 + len(study['methods']) + len(study['pairs'])
    review_cost = 1 + len(owner['method_use_plan']['native_return_refs'])
    limit = metadata_cost + review_cost
    if limit > receipt['remaining_compute']:
        raise ValueError('Original balance cannot cover the actual method return')
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
    common = CommonObjectDependencyMethods(output / 'common', [],
        budget=SearchBudget(limit), actions=[], source_task_ref=cfg['source_task_ref']).common
    ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
        topic_ref=ref(output / 'previous_topic.json'))
    question = next(q for q in common.questions if q['id'] == 'task:method_choice')
    question['mission']['allocation']['compute'] += limit
    returned = record_native_partition_study(common, question,
        proposal_ref=receipt['proposal_ref'], study_ref=ref(source / 'study.json'),
        comparison_ref=ref(source / 'pool_comparison.json'), cost_refs=[ref(source / 'receipt.json')],
        compute_limit=metadata_cost)
    debit_mission_compute(common, question['mission'], 0)
    reviewed = consume_method_return_agenda(common, {'method_question_refs': [
        {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)}]},
        None, compute_limit=review_cost)
    result = dict(status=reviewed['status'], study_return=returned, agenda_return=reviewed,
        compute_cost=common.costs['compute_cost'], new_compute_cost=common.costs['compute_cost'],
        cumulative_compute=receipt['cumulative_compute'] + common.costs['compute_cost'],
        remaining_compute=receipt['remaining_compute'] - common.costs['compute_cost'],
        source_cost_refs=[ref(source / 'receipt.json')], new_fits=0, new_predictions=0,
        new_observations=0, model_calls=0, independent_method_support=False)
    save(output / 'receipt.json', result)
    return result


def research_paid_method(output):
    """Consume actual revision evidence through Common, then test its proposal."""
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return)
    from hegel_machine.cognition.method_action_research import quote_native_partition_analysis
    from hegel_machine.cognition.method_feedback import ensure_python_method_topic
    from hegel_machine.cognition.question_routing import consume_method_return_agenda
    from hegel_machine.cognition.research_programs import (
        evaluate_research_step_revisions, quote_research_step_revisions,
        research_program_replay_inputs)
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract

    output = Path(output)
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Run the freshly frozen method unit once; reuse its saved return afterwards')
    cfg = load(output / 'registration.json')
    source, joint = Path(cfg['source']), Path(cfg['joint_source'])
    budget = SearchBudget(cfg['remaining_compute'])
    timing = ExecutionTiming(('restore', 'proposal', 'paired_research', 'comparison'))
    save(output / 'started.json', {'source': cfg['source']})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('restore'):
                scene = restore(cfg['original_source'])
                before = decode_entry(read_method_reference(cfg['before_entry_ref'], output))
                after = decode_entry(read_method_reference(cfg['after_entry_ref'], output))
                training = scene['scene']['training']
                validation = [*scene['scene']['validation'], *scene['exposed']]
                payload = before['research_program']
                # The existing baseline really used these complete columns and
                # this original training set. Do not run the paid fit again.
                if (payload['fit_rows'] != len(training)
                        or research_program_fit_contract(payload)['program'] != before['candidate'].program):
                    raise ValueError('The paid baseline is not this complete fixed-coordinate fit')
                old_mse = (len(scene['scene']['validation']) * scene['working'][before['candidate'].program]
                    + len(scene['exposed']) * scene['exposed_scores'][before['candidate'].program]) / len(validation)
                owner = next(q for q in load(source / 'common/questions.json') if q['id'] == 'task:method_choice')
                analysis_limit = quote_native_partition_analysis(payload, after['research_program'])
                review_limit = analysis_limit + 2 + len(owner['method_use_plan']['native_return_refs'])
                if not budget.can_spend(review_limit):
                    raise ValueError('Original balance cannot cover the complete proposal analysis')
            with timing.phase('proposal'):
                save(output / 'revision_evidence.json', {
                    'source_task_ref': cfg['source_task_ref'], 'before_entry_ref': cfg['before_entry_ref'],
                    'after_entry_ref': cfg['after_entry_ref'], 'comparison_ref': cfg['complete_pool_ref'],
                    'cost_refs': cfg['source_cost_refs'], 'data_role': 'exposed_development',
                    'source_operation': 'developer_selected_joint_structure_search'})
                save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
                common = CommonObjectDependencyMethods(output / 'common', [],
                    budget=SearchBudget(review_limit), actions=[], source_task_ref=cfg['source_task_ref']).common
                ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
                    topic_ref=ref(output / 'previous_topic.json'))
                mission = deepcopy(owner['mission'])
                mission['allocation']['compute'] += review_limit
                returned = record_native_parent_research_return(common,
                    source_task_ref=cfg['source_task_ref'], result_ref=ref(source / 'research_receipt.json'),
                    revision_evidence_ref=ref(output / 'revision_evidence.json'), mission=mission)
                reviewed = consume_method_return_agenda(common, returned, None, compute_limit=review_limit)
                budget.spend(common.costs['compute_cost'])
                save(output / 'proposal_return.json', reviewed)
                records = reviewed.get('research_result', {}).get(
                    'method_action_research', {}).get('transfer_partition_proposals', [])
                if not records:
                    # Keep the complete normal-route return, including a real
                    # negative or unresolved proposal, instead of forcing Pi1.
                    result = {'status': 'no_method_proposal', 'method_return_ref': ref(output / 'proposal_return.json')}
                else:
                    bundle = read_method_reference(records[-1]['proposal_ref'], common.workspace)
                    proposal = bundle['proposal']
                    result = {'status': 'proposed', 'proposal_ref': records[-1]['proposal_ref']}
            if result['status'] == 'proposed':
                with timing.phase('paired_research'):
                    if len(proposal['candidates']) > cfg['max_candidates']:
                        raise ValueError('The registered unit covers one genuinely observed group revision')
                    budget.reserve = (5 + 2 * len(proposal['candidates'])
                        + len(owner['method_use_plan']['native_return_refs']))
                    quote = quote_research_step_revisions(payload, training, validation,
                        prepared_proposal=proposal)
                    save(output / 'quote.json', quote)
                    if quote['status'] != 'quoted':
                        raise ValueError('The generated method has no complete execution quote')
                    refs = [cfg['before_entry_ref'],
                        ref(Path(cfg['original_source']) / 'prepared_state.json'),
                        ref(Path(cfg['original_source']) / 'confirmation/observations.json')]
                    baseline = {'execution_complete': True, 'program': before['candidate'].program.to_dict(),
                        'research_program': payload, 'training_loss': before['training_loss'],
                        'validation_loss': old_mse, 'fit_compute_cost': payload['search_compute'],
                        'paid_compute_cost': payload['search_compute'] + len(validation),
                        'cost_ref': cfg['before_entry_ref'], 'basis': {
                            'replay_inputs': research_program_replay_inputs(proposal['baseline_research_program']),
                            'source_refs': refs, 'training_rows': len(training),
                            'validation_rows': len(validation), 'risk_quantile': .5}}
                    study = evaluate_research_step_revisions(payload, training, validation, budget,
                        per_method_compute=quote['per_method_compute_upper'], source_refs=refs,
                        prepared_proposal=proposal, paid_baseline=baseline,
                        minimum_improvement=cfg['threshold'],
                        on_progress=lambda value: save(output / 'study_progress.json', value))
                    save(output / 'study.json', study)
                    result.update(status=study['status'], study_ref=ref(output / 'study.json'),
                        methods=[{key: row.get(key) for key in ('index', 'execution_complete', 'training_loss',
                            'validation_loss', 'new_compute_cost', 'reused')} for row in study['methods']])
                with timing.phase('comparison'):
                    previous = load(joint / 'comparison.json')
                    scores = list(previous['candidate_scores'])
                    for row in study['methods'][1:]:
                        if row['execution_complete']:
                            scores.append({'candidate_ref': ref(output / 'study.json', 'methods', row['index']),
                                'loss': (len(training) * row['training_loss']
                                    + len(validation) * row['validation_loss']) / (len(training) + len(validation))})
                    retained = load(joint / 'receipt.json')['selected_mse']
                    changed = min(row['loss'] for row in scores) + cfg['threshold'] < retained
                    save(output / 'pool_comparison.json', {'scores': scores,
                        'full_comparison': previous['full_comparison'] and all(
                            row['execution_complete'] for row in study['methods'])
                            and len(study['methods']) == len(proposal['candidates']) + 1,
                        'paid_rows': len(training) + len(validation), 'old_scores_reused': len(previous['candidate_scores']),
                        'parent_changed': changed, 'confirmed_parent_ref': cfg['after_entry_ref'],
                        'selected_ref': min(scores, key=lambda row: row['loss'])['candidate_ref'] if changed else cfg['after_entry_ref']})
                    result.update(old_scores_reused=len(previous['candidate_scores']),
                        compared_candidates=len(scores), parent_changed=changed)
            budget.release_reserve()
            result.update(compute_cost=budget.spent, cumulative_compute=cfg['previous_compute'] + budget.spent,
                remaining_compute=budget.remaining, source_cost_refs=cfg['source_cost_refs'],
                new_observations=0, model_calls=0, fresh_confirmation_used=False,
                independent_method_support=False, data_role='exposed_development',
                execution_timing=timing.result())
            save(output / 'receipt.json', result)
            if result.get('study_ref'):
                # The paid study belongs back to its original question. Keep
                # the scientific receipt and append this small return ledger.
                result = {**result, 'method_return': return_paid_method_study(output, output / 'method_return')}
            return result
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_known': budget.spent,
            'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise


def register(source, output):
    source, output = Path(source), Path(output)
    old, receipt = load(source / 'registration.json'), load(source / 'receipt.json')
    total = old['common_compute_limit'] + len(old['operations']) * old['per_arm_total_compute_limit']
    paid = receipt['cumulative_costs']
    remaining = total - paid['compute_cost']
    if remaining <= 0:
        raise ValueError('No unused allowance remains in the original unit')
    output.mkdir(parents=True, exist_ok=False)
    registration = dict(source=str(source), source_task_ref=old['source_task_ref'],
        previous_costs=paid, total_compute_limit=total, remaining_compute=remaining,
        source_cost_refs=[ref(source / 'receipt.json'), *receipt['source_cost_refs']],
        scope='one paid development scope/boundary revision; not a new method pair',
        operations=['retain_complete_pool', 'revise_object_scope', 'revise_paid_structure'],
        scope_fit_compute_limit=12000, active_wall_seconds=1800,
        threshold=old['quality_threshold'], confirmation_rows=old['confirmation_rows'],
        confirmation_seed=old['confirmation_seed'] + 17017,
        confirmation_gate='changed complete-pool winner; every paid squared residual <= original threshold',
        confirmation_comparator='unchanged structural winner from the previous independent failure',
        development_role='all original training, working and exposed confirmation rows',
        stop_rule='one bounded revision and at most one gated fresh readout; preserve any failure',
        model_calls=0)
    save(output / 'registration.json', registration)
    freeze_runtime(output, source)
    name = Path('scripts/revise_learned_structure_successor.py')
    base = source / 'runtime' / name
    write_text_incrementally(output / 'runtime' / name, (ROOT / name).read_text(),
                             base=base if base.exists() else None)
    return registration


def run(output):
    output = Path(output)
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Execute the newly frozen current entry')
    if (output / 'started.json').exists():
        raise ValueError('Keep the prior attempt and its paid result')
    cfg = load(output / 'registration.json')
    save(output / 'started.json', {'status': 'started', 'source': cfg['source']})
    timing = ExecutionTiming(('restore', 'paid_comparison', 'scope_revision', 'boundary_revision',
                              'residual_readout', 'confirmation', 'save'))
    budget = SearchBudget(cfg['remaining_compute'])
    actions, observation_cost, measurement_cost = [], 0, 0
    try:
        with timing.phase('restore'):
            scene = restore(cfg['source'])
        agent, spec, paid = scene['agent'], scene['scene']['spec'], scene['paid']
        original_parent = deepcopy(scene['parent'])
        fallback = scene['frame']['fallback']
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('paid_comparison'):
                scores, exposed, additional_scoring = paid_pool_scores(scene, budget)
            # Keep enough for an actual selected residual check and both frozen predictors.
            budget.reserve = len(paid) + 2 * cfg['confirmation_rows']
            with timing.phase('scope_revision'):
                local = SearchBudget(min(cfg['scope_fit_compute_limit'],
                    max(0, budget.remaining - agent.config.max_candidates * len(paid))))
                scope = agent._object_scope_claims(spec, paid, local, actions)
                budget.spend(local.spent)
                assessed = agent._assess(scope, paid, budget, fallback, spec.risk_quantile)
                if len(assessed) != len(scope):
                    raise RuntimeError('Object-scope comparison is incomplete')
                scores.update({entry['candidate'].program: entry['validation_loss'] for entry in assessed})
                save(output / 'scope_claims.json', [encode_entry(entry) for entry in assessed])
            with timing.phase('boundary_revision'):
                result = agent._revise_paid_structure(spec, scene['parent'], paid, budget, actions,
                    finalists=[*scene['pool'], *assessed], transitions=scene['transitions'], paid_scores=scores)
            budget.release_reserve()
            selected = result['selected']
            with timing.phase('residual_readout'):
                values = result.get('selected_predictions')
                if values is None and result['full_comparison']:
                    if not budget.spend(len(paid)):
                        raise RuntimeError('The selected residual readout is incomplete')
                    values = predict_rows(selected['candidate'].program, paid)
                errors = [(value - row.target) ** 2 for value, row in zip(values, paid)] if values else None
            changed = selected['candidate'].program != original_parent['candidate'].program
            gate = bool(result['full_comparison'] and changed and errors
                        and max(errors) <= cfg['threshold'])
            with timing.phase('save'):
                save(output / 'new_claims.json', [encode_entry(entry) for entry in result['new_claims']])
                save(output / 'selected_entry.json', encode_entry(selected))
                all_refs = [*scene['refs'], *[ref(output / 'scope_claims.json', i) for i in range(len(scope))],
                    *[ref(output / 'new_claims.json', i) for i in range(len(result['new_claims']))]]
                all_entries = [*scene['pool'], *assessed, *result['new_claims']]
                scored = {entry['candidate'].program: entry['validation_loss']
                          for entry in result['compared_claims']}
                save(output / 'selection.json', dict(status=result['status'],
                    full_comparison=result['full_comparison'], actions=actions,
                    retained_pool_refs=scene['refs'], source_rows_ref=ref(Path(cfg['source']) / 'prepared_state.json',
                        'source_references'), exposed_rows_ref=ref(Path(cfg['source']) / 'confirmation/observations.json'),
                    scores=[{'candidate_ref': location, 'loss': scored.get(entry['candidate'].program)}
                            for entry, location in zip(all_entries, all_refs)],
                    selected_ref=ref(output / 'selected_entry.json'), selected_paid_errors=errors,
                    confirmation_admitted=gate))
            confirmation = None
            if gate:
                with timing.phase('confirmation'):
                    save(output / 'frozen_predictors.json', dict(
                        selected=selected['candidate'].program.to_dict(),
                        incumbent=original_parent['candidate'].program.to_dict(), fallback=fallback))
                    if not budget.spend(2 * cfg['confirmation_rows']):
                        raise RuntimeError('The gated confirmation was not funded')
                    with AnonymousObjectLab(**scene['registration']['task']) as laboratory:
                        probe = laboratory.random_probe(cfg['confirmation_seed'], length=cfg['confirmation_rows'],
                            name='scope_boundary_revision_confirmation')
                        batch = laboratory.acquire(probe, ('fine',))
                        observation_cost, measurement_cost = laboratory.observation_cost, laboratory.measurement_cost
                        save(output / 'confirmation_observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
                        confirmation = {'rows': len(batch.observations), 'threshold': cfg['threshold']}
                        for name, entry in [('selected', selected), ('incumbent', original_parent)]:
                            predictions = predict_rows(entry['candidate'].program, batch.observations)
                            mse = mean((value - row.target) ** 2 for value, row in zip(predictions, batch.observations))
                            confirmation[name] = dict(mse=mse, predictions=predictions, passed=mse <= cfg['threshold'])
            receipt = dict(status='confirmed' if confirmation and confirmation['selected']['passed'] else 'unresolved',
                selection_status=result['status'], full_comparison=result['full_comparison'],
                retained_pool=len(scene['pool']), scope_candidates=len(scope), boundary_candidates=len(result['new_claims']),
                complete_pool=len(result['claims']), paid_rows=len(paid),
                old_parent_paid_mse=scores[original_parent['candidate'].program],
                selected_paid_mse=mean(errors) if errors else None,
                max_paid_squared_error=max(errors) if errors else None,
                old_exposed_mse=mean(errors[-len(scene['exposed']):]) if errors else None,
                additional_old_scoring=additional_scoring, reused_old_training_scores=len(scene['pool']),
                reused_old_working_scores=len(scene['pool']), reused_old_exposed_scores=len(scene['exposed_scores']),
                confirmation_admitted=gate, confirmation=confirmation, compute_cost=budget.spent,
                cumulative_compute=cfg['previous_costs']['compute_cost'] + budget.spent,
                observation_cost=observation_cost, measurement_cost=measurement_cost,
                source_cost_refs=cfg['source_cost_refs'], model_calls=0,
                method_condition_learned=False, execution_timing=timing.result())
            save(output / 'receipt.json', receipt)
            return receipt
    except BaseException as error:
        save(output / 'failure.json', dict(error=repr(error), compute_known=budget.spent,
            compute_upper_bound=cfg['remaining_compute'], observation_cost=observation_cost,
            measurement_cost=measurement_cost, actions=actions, execution_timing=timing.result()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'prepare-confirm', 'confirm-paid',
                                        'prepare-method', 'research-method', 'return-method'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compute-limit', type=int, default=1000)
    args = parser.parse_args()
    print(return_paid_method_study(args.source, args.output) if args.mode == 'return-method'
          else register_method_research(args.source, args.output) if args.mode == 'prepare-method'
          else research_paid_method(args.output.resolve()) if args.mode == 'research-method'
          else register_paid_confirmation(args.source, args.output, compute_limit=args.compute_limit)
          if args.mode == 'prepare-confirm'
          else confirm_paid_revision(args.output.resolve()) if args.mode == 'confirm-paid'
          else register(args.source, args.output) if args.mode == 'prepare' else run(args.output.resolve()))
