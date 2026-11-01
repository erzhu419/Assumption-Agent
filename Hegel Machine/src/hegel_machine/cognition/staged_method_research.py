"""Learn a next-stage decision from an earlier method's actual paid return.

Old validation labels become development material for this new question. They
do not validate a newly generated two-stage procedure on independent sources.
"""

from copy import deepcopy
from math import isfinite
from pathlib import Path
import json

from .method_action_research import read_method_reference, quote_native_partition_analysis
from .method_conditions import (
    _conflict_predictions, paired_method_preference, research_paired_method_conditions)
from .method_feedback import load_paired_method_returns
from .models import SearchBudget


def baseline_return_profile(outcome, threshold):
    """Only the just-fitted baseline's all-paid loss is a branch observable."""
    value = outcome.get('fit_loss')
    measured = type(value) in (int, float) and isfinite(value) and value >= 0
    completed = outcome.get('execution_complete') is True
    return {'residual': {'state': ('unmeasured' if not measured else
                'adequate' if value <= threshold else 'unresolved'),
            'loss': value if measured else None},
        'research': {'stage': 'after_baseline_refit', 'execution_complete': completed}}


def _fit_evidence(agent, pair, side):
    """Read the new fit's recorded score, never substitute an old pool winner."""
    original = read_method_reference(pair[side + '_ref'], agent.workspace)
    actual_ref = original.get('actual_outcome_ref')
    if not actual_ref:
        return {'status': 'unmeasured', 'reason': side + '_actual_return_missing'}
    try:
        actual = read_method_reference(actual_ref, agent.workspace)
    except FileNotFoundError:
        return {'status': 'unmeasured', 'reason': side + '_actual_return_missing',
                'missing_source_ref': deepcopy(actual_ref)}
    comparison_ref = actual.get('comparison_ref') or {
        'file': str((Path(agent.workspace) / actual_ref['file']).with_name('comparison.json'))}
    if not (Path(agent.workspace) / comparison_ref['file']).is_file():
        return {'status': 'unmeasured', 'reason': side + '_fit_comparison_missing'}
    comparison = read_method_reference(comparison_ref, agent.workspace)
    scored = comparison.get('new_scores', [])
    if (not actual.get('execution_complete') or not comparison.get('full_comparison')
            or not scored or actual.get('new_candidates') != len(scored)
            or any(type(item.get('loss')) not in (int, float)
                   or not isfinite(item['loss']) or item['loss'] < 0 for item in scored)):
        return {'status': 'unmeasured', 'reason': 'complete_' + side + '_fit_scores_required'}
    best = min(scored, key=lambda item: item['loss'])
    return {'status': 'measured', 'fit_loss': best['loss'], 'execution_complete': True,
        'score_ref': {**comparison_ref, 'json_path': [
            *comparison_ref.get('json_path', []), 'new_scores', scored.index(best)]},
        'candidate_ref': deepcopy(best['candidate_ref']), 'actual_outcome_ref': actual_ref,
        'profile_scope': 'new_baseline_fit_all_paid_rows',
        'development_rows': actual.get('development_rows'), 'costs': deepcopy(original['costs'])}


def staged_program(baseline, continuation, condition, coordinates, evidence_refs):
    body = deepcopy(baseline)
    body['staged_research'] = {'operation': 'refit_then_reidentify',
        'baseline': deepcopy(baseline), 'continuation': deepcopy(continuation),
        'condition': deepcopy(condition), 'coordinate_paths': deepcopy(coordinates),
        'evidence_refs': deepcopy(evidence_refs),
        'profile_scope': 'new_baseline_fit_all_paid_rows',
        'scope': 'exposed_development_candidate', 'independent_support': False}
    return body


def _publication(agent, question, reference, bundle, new_compute):
    count = len(bundle['proposal']['candidates'])
    result = {**bundle['receipt'], 'proposal_ref': reference, 'new_compute_cost': new_compute,
        'candidate_refs': [{**reference, 'json_path': ['proposal', 'candidates', i]} for i in range(count)],
        'research_program_refs': [{**reference, 'json_path': ['proposal', 'candidates', i, 'research_program']}
                                  for i in range(count)]}
    for ref in reversed(question['method_use_plan'].get('transfer_partition_return_refs', [])):
        returned = read_method_reference(ref, agent.workspace)
        if returned['basis']['proposal_ref'] == reference:
            result.update(status=returned['status'], study_return_ref=deepcopy(ref),
                execution_complete=returned['execution_complete'],
                remaining_validation_residual=returned['remaining_validation_residual'])
            break
    return result


def research_staged_method_conditions(agent, question, group, compute_limit):
    """Ordinary Common method review proposes, but does not install, a stage."""
    bodies = {}
    for side, key in (('baseline', 'baseline_method_artifact_ref'),
                      ('continuation', 'method_artifact_ref')):
        artifact = read_method_reference(group[key], agent.workspace)
        execution = artifact.get('execution', {})
        if execution.get('entry') != 'replay_research_program':
            return {'status': 'not_a_replayed_program_pair', 'new_compute_cost': 0}
        bodies[side] = read_method_reference(execution['research_program_ref'], agent.workspace)
    if (bodies['baseline'].get('staged_research') or bodies['continuation'].get('staged_research')
            or bodies['baseline'] == bodies['continuation']):
        return {'status': 'distinct_unstaged_programs_required', 'new_compute_cost': 0}
    kinds = {side: body.get('transfer_partition', {}).get('kind') for side, body in bodies.items()}
    if kinds != {'baseline': 'Pi0', 'continuation': 'Pi1'}:
        return {'status': 'unsupported_staged_method_pair', 'new_compute_cost': 0,
                'partition_kinds': kinds, 'supported_sequence': ['Pi0', 'Pi1']}
    basis = {'baseline_method_artifact_ref': deepcopy(group['baseline_method_artifact_ref']),
        'method_artifact_ref': deepcopy(group['method_artifact_ref']),
        'pair_refs': deepcopy(group['pair_refs']), 'profile_scope': 'new_baseline_fit_all_paid_rows',
        'profile_stage': 'after_baseline_refit'}
    refs = question['method_use_plan'].setdefault('staged_method_proposal_refs', [])
    for ref in reversed(refs):
        previous = read_method_reference(ref, agent.workspace)
        if previous['basis'] == basis and not previous.get('budget_incomplete'):
            return {**_publication(agent, question, ref, previous, 0), 'reused': True}
    budget = SearchBudget(max(0, compute_limit))
    quote = 1 + len(group['pair_refs'])
    if not budget.spend(quote):
        return {'status': 'staged_source_analysis_not_funded', 'new_compute_cost': 0,
                'budget_incomplete': True, 'required_compute': quote}
    resolved = load_paired_method_returns(agent, group['pair_refs'])
    unique = []
    for pair in resolved:
        origin = pair.get('origin_ref', pair['feedback_ref'])
        index = next((i for i, item in enumerate(unique)
            if item.get('origin_ref', item['feedback_ref']) == origin), None)
        if index is None:
            unique.append(pair)
        else:
            unique[index] = pair
    pairs, evidence, excluded = [], [], []
    for pair in unique:
        measured = _fit_evidence(agent, pair, 'baseline')
        continuation = _fit_evidence(agent, pair, 'challenger')
        if measured['status'] != 'measured' or continuation['status'] != 'measured':
            excluded.append({'pair_ref': pair['feedback_ref'],
                'baseline': measured, 'continuation': continuation})
            continue
        quality = pair['baseline'].get('working_quality', pair['baseline'].get('quality', {}))
        threshold = quality.get('threshold')
        if type(threshold) not in (int, float) or not isfinite(threshold) or threshold < 0:
            excluded.append({'pair_ref': pair['feedback_ref'], 'reason': 'original_quality_contract_missing'})
            continue
        profile = baseline_return_profile(measured, threshold)
        # A post-P0 state is legal for the next-stage choice, not for original
        # whole-method invocation. Preserve that distinction in the proposal.
        staged_pair = deepcopy(pair)
        for side in ('baseline', 'challenger'):
            staged_pair[side]['profile'] = deepcopy(profile)
        if (not measured['development_rows'] or measured['development_rows'] != continuation['development_rows']):
            excluded.append({'pair_ref': pair['feedback_ref'], 'reason': 'same_scored_rows_required'})
            continue
        losses = {'baseline': measured['fit_loss'],
                  'challenger': min(measured['fit_loss'], continuation['fit_loss'])}
        for side in ('baseline', 'challenger'):
            output = staged_pair[side]
            output.update(quality={'loss': losses[side], 'threshold': threshold}, loss=losses[side],
                completed=losses[side] <= threshold, coverage=measured['development_rows'])
            output['basis']['evaluation'] = {'role': 'paid_working_comparison',
                'score_refs': [measured['score_ref'], continuation['score_ref']]}
            output.pop('cost_intervals', None)
            output['costs'] = ({key: 0 for key in measured['costs']} if side == 'baseline'
                               else deepcopy(continuation['costs']))
        # P0 has already been paid at this choice. Keeping it costs no new
        # research; Pi1 pays its full additional execution, not the old P1-P0
        # difference. The retained full-method invoices remain in provenance.
        staged_pair['baseline_action_content'] = {'operation': 'retain_completed_baseline'}
        staged_pair['challenger_action_content'] = {'operation': 'append_and_compare',
            'method': deepcopy(pair.get('challenger_action_content'))}
        staged_pair.update(evaluation_role='paid_working_comparison',
            prefix_costs=deepcopy(measured['costs']),
            identification_target='fixed_data_method_effect')
        staged_pair.pop('evaluation_lifecycle', None)
        staged_pair.pop('paired_future_ref', None)
        staged_pair['data_role'] = 'development'
        staged_pair.pop('condition_validation_cohort_refs', None)
        pairs.append(staged_pair)
        evidence.append({'pair_ref': deepcopy(pair['feedback_ref']),
            'origin_ref': deepcopy(pair.get('origin_ref', pair['feedback_ref'])),
            'previous_data_role': pair.get('data_role'), 'data_role': 'development',
            'profile': profile, **measured, 'continuation': continuation,
            'stage_costs': {'shared_paid_prefix': deepcopy(measured['costs']),
                'baseline_increment': deepcopy(staged_pair['baseline']['costs']),
                'challenger_increment': deepcopy(staged_pair['challenger']['costs'])}})
    learned = research_paired_method_conditions(group['method_ref'], pairs, budget, max_conditions=8)
    candidates = []
    candidate_quote = (quote_native_partition_analysis(bodies['baseline'], bodies['continuation'])
                       if learned.get('candidates') else 0)
    for candidate in learned.get('candidates', []):
        if not budget.spend(candidate_quote):
            break
        candidates.append({'research_program': staged_program(bodies['baseline'], bodies['continuation'],
            candidate['condition'], learned['coordinate_paths'], [item['pair_ref'] for item in evidence]),
            'status': 'requires_paired_evaluation', 'condition_source': deepcopy(candidate),
            'source_refs': [item['score_ref'] for item in evidence]})
    incomplete = bool(learned.get('budget_incomplete') or len(candidates) < len(learned.get('candidates', [])))
    agent.costs['compute_cost'] += budget.spent
    receipt = {'status': ('staged_analysis_incomplete' if incomplete else
        'staged_program_proposed' if candidates else 'no_staged_condition_identified'),
        'new_compute_cost': budget.spent, 'budget_incomplete': incomplete,
        'required_compute': learned.get('required_compute', 0),
        'unique_origins': len(unique), 'usable_origins': len(pairs),
        'candidate_count': len(candidates), 'general_guard_supported': False,
        'independent_support': False, 'new_fits': 0, 'new_observations': 0,
        'profile_scope': basis['profile_scope'], 'profile_stage': basis['profile_stage']}
    bundle = {'basis': basis, 'status': receipt['status'], 'receipt': receipt,
        'source_task_ref': None, 'source_refs': deepcopy(group['pair_refs']),
        'budget_incomplete': incomplete, 'research': learned, 'baseline_evidence': evidence,
        'excluded_sources': excluded, 'proposal': {'baseline_research_program': bodies['baseline'],
            'candidates': candidates}, 'analysis_compute_cost': budget.spent,
        'data_role': 'exposed_development', 'independent_support': False,
        'source_support_inherited': False, 'installed_candidate': None,
        'required_work': 'execute_new_staged_method_and_freeze_before_new_source_validation'}
    directory, serial = agent.workspace / 'method_feedback', 0
    while (directory / f'staged_method_{serial:06d}.json').exists():
        serial += 1
    path = f'method_feedback/staged_method_{serial:06d}.json'
    agent._save(path, json.dumps(bundle, separators=(',', ':')))
    ref = {'file': str(agent.workspace / path)}
    refs.append(ref)
    agent._save('questions.json', json.dumps(agent.questions, separators=(',', ':')))
    return _publication(agent, question, ref, bundle, budget.spent)


def _stage_return_pair(agent, pair, staged_side, fixed_side):
    """Expose an already executed prefix and a same-data continuation contrast."""
    original = read_method_reference(pair[staged_side + '_ref'], agent.workspace)
    actual_ref = original.get('actual_outcome_ref')
    if not actual_ref:
        return None, {'reason': 'actual_staged_return_required'}
    actual = read_method_reference(actual_ref, agent.workspace)
    staged = actual.get('staged_research_result', {})
    prefix = staged.get('baseline', {})
    continuation = _fit_evidence(agent, pair, fixed_side)
    quality = pair[staged_side].get('working_quality', pair[staged_side].get('quality', {}))
    threshold, loss = quality.get('threshold'), prefix.get('all_paid_loss')
    numeric = lambda value: type(value) in (int, float) and isfinite(value) and value >= 0
    if (not prefix.get('execution_complete') or not numeric(loss) or not numeric(threshold)
            or continuation['status'] != 'measured'):
        return None, {'reason': 'complete_measured_prefix_and_continuation_required',
            'continuation': continuation, 'actual_outcome_ref': actual_ref}
    rows = actual.get('development_rows')
    if not rows or rows != continuation['development_rows']:
        return None, {'reason': 'same_scored_rows_required', 'actual_outcome_ref': actual_ref}
    profile = baseline_return_profile({'fit_loss': loss, 'execution_complete': True}, threshold)
    if staged.get('guard', {}).get('profile') != profile:
        return None, {'reason': 'recorded_prefix_profile_differs', 'actual_outcome_ref': actual_ref}
    prefix_ref = {**actual_ref, 'json_path': [*actual_ref.get('json_path', []),
        'staged_research_result', 'baseline']}
    contrast = {**deepcopy(pair), 'baseline': deepcopy(pair[staged_side]),
        'challenger': deepcopy(pair[fixed_side])}
    losses = {'baseline': loss, 'challenger': min(loss, continuation['fit_loss'])}
    for side in ('baseline', 'challenger'):
        item = contrast[side]
        item.update(profile=deepcopy(profile), quality={'loss': losses[side], 'threshold': threshold},
            loss=losses[side], completed=losses[side] <= threshold, coverage=rows)
        item['basis']['evaluation'] = {'role': 'paid_working_comparison',
            'score_refs': [prefix_ref, continuation['score_ref']]}
        item.pop('cost_intervals', None)
        item['costs'] = ({key: 0 for key in continuation['costs']} if side == 'baseline'
                         else deepcopy(continuation['costs']))
    contrast.update(evaluation_role='paid_working_comparison', data_role='development',
        identification_target='fixed_data_method_effect',
        baseline_action_content={'operation': 'retain_completed_baseline'},
        challenger_action_content={'operation': 'append_and_compare',
            'method': deepcopy(pair.get(fixed_side + '_action_content'))})
    contrast.pop('evaluation_lifecycle', None)
    contrast.pop('paired_future_ref', None)
    preference = paired_method_preference(contrast)
    evidence = {'pair_ref': deepcopy(pair['feedback_ref']),
        'origin_ref': deepcopy(pair.get('origin_ref', pair['feedback_ref'])),
        'previous_data_role': pair.get('data_role'), 'data_role': 'exposed_development',
        'profile': profile, 'prefix_ref': prefix_ref,
        'continuation_score_ref': continuation['score_ref'], 'preference': preference,
        'incremental_costs': {'keep_prefix': deepcopy(contrast['baseline']['costs']),
            'append_continuation': deepcopy(contrast['challenger']['costs'])},
        'scope': 'paid_completed_components_not_an_independent_revised_method_execution'}
    return contrast, evidence


def _revision_publication(agent, question, reference, bundle, new_compute):
    result = {**bundle['receipt'], 'revision_ref': deepcopy(reference),
        'research_program_ref': ({**reference, 'json_path': ['research_program']}
            if bundle.get('research_program') else None), 'new_compute_cost': new_compute}
    for returned_ref in reversed(question['method_use_plan'].get('staged_method_practice_refs', [])):
        returned = read_method_reference(returned_ref, agent.workspace)
        if returned['basis']['revision_ref'] == reference:
            complete = returned['execution_complete']
            result.update(revision_status=result['status'],
                status='staged_conditions_practiced' if complete else 'staged_condition_practice_incomplete',
                practice_return_ref=deepcopy(returned_ref), practice_status=returned['status'],
                execution_complete=complete,
                required_work=('assess_new_source_validation_for_revised_method' if complete
                    else 'resolve_retained_staged_practice_without_repeating_completed_sources'))
            break
    return result


def record_staged_method_practice(agent, question, *, revision_ref, practice_ref, compute_limit):
    """Return an actual paid-stage replay to its existing method question."""
    plan = question['method_use_plan']
    if revision_ref not in plan.get('staged_method_revision_refs', []):
        raise ValueError('The practice must address this method question\'s revision')
    basis = {'revision_ref': deepcopy(revision_ref), 'practice_ref': deepcopy(practice_ref)}
    references = plan.get('staged_method_practice_refs', [])
    for reference in references:
        retained = read_method_reference(reference, agent.workspace)
        if retained['basis'] == basis:
            return {**retained, 'return_ref': deepcopy(reference),
                'cache_hit': True, 'new_compute_cost': 0}
    revised = read_method_reference(revision_ref, agent.workspace)
    practice = read_method_reference(practice_ref, agent.workspace)
    path = Path(agent.workspace) / practice_ref['file']
    registration_ref = {'file': str(path.with_name('registration.json'))}
    registration = read_method_reference(registration_ref, agent.workspace)
    measured_method = read_method_reference(registration['method_ref'], agent.workspace)
    addressed = {**revision_ref, 'json_path': ['research_program']}
    if (registration['method_source_ref'] != addressed
            or measured_method != revised.get('research_program')):
        raise ValueError('The paid practice used a different frozen method revision')
    registered, sources = registration['sources'], practice.get('sources', [])
    required = 2 + len(sources)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'staged_practice_return_not_funded',
            'required_compute': required, 'new_compute_cost': 0}
    agent.costs['compute_cost'] += required
    expected = [item['source_task_ref'] for item in registered]
    returned = [item['source_task_ref'] for item in sources]
    complete = (practice.get('status') == 'paid_stage_paths_completed'
        and len(sources) == len(registered) and all(returned.count(source) == 1 for source in expected))
    outcomes = []
    for index, source in enumerate(sources):
        declared = next((item for item in registered
            if item['source_task_ref'] == source['source_task_ref']), None)
        done = bool(declared is not None and source.get('execution_complete')
            and source.get('full_comparison')
            and source.get('staged_research_result', {}).get('execution_complete')
            and source.get('old_pool_size') == declared['pool_size'])
        complete = complete and done
        outcomes.append({'source_task_ref': deepcopy(source['source_task_ref']),
            'outcome_ref': {**practice_ref, 'json_path': [*practice_ref.get('json_path', []), 'sources', index]},
            'execution_complete': done, 'full_comparison': source.get('full_comparison', False),
            'old_pool_size': source.get('old_pool_size'),
            'stage_status': source.get('staged_research_result', {}).get('status'),
            'selected_stage': source.get('staged_research_result', {}).get('selected_stage')})
    result = {'status': 'staged_method_practice_completed' if complete else 'incomplete_not_negative',
        'basis': basis, 'registration_ref': registration_ref, 'execution_complete': complete,
        'source_outcomes': outcomes, 'cost_refs': [deepcopy(practice_ref)],
        'analysis_compute_cost': required, 'data_role': 'exposed_development',
        'independent_support': False, 'general_guard_supported': False,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0}
    serial = 0
    while (agent.workspace / 'method_feedback' / f'staged_method_practice_{serial:06d}.json').exists():
        serial += 1
    relative = f'method_feedback/staged_method_practice_{serial:06d}.json'
    agent._save(relative, json.dumps(result, separators=(',', ':')))
    reference = {'file': str(agent.workspace / relative)}
    plan.setdefault('staged_method_practice_refs', []).append(reference)
    question.update(status='open', agenda_basis=deepcopy(plan))
    question.pop('completed_agenda_basis', None)
    agent._save('questions.json', json.dumps(agent.questions, separators=(',', ':')))
    return {**result, 'return_ref': reference, 'cache_hit': False, 'new_compute_cost': required}


def revise_staged_method_conditions(agent, question, groups, compute_limit):
    """Refute frozen conditions with paid returns; never fit another threshold."""
    plan = question['method_use_plan']
    for proposal_ref in reversed(plan.get('staged_method_proposal_refs', [])):
        proposal = read_method_reference(proposal_ref, agent.workspace)
        candidates = proposal.get('proposal', {}).get('candidates', [])
        if not candidates:
            continue
        original = candidates[0]['research_program']
        stage = original['staged_research']
        matching = []
        for group in groups:
            bodies = {}
            for side, key in (('baseline', 'baseline_method_artifact_ref'),
                              ('challenger', 'method_artifact_ref')):
                artifact = read_method_reference(group[key], agent.workspace)
                execution = artifact.get('execution', {})
                if execution.get('entry') == 'replay_research_program':
                    bodies[side] = read_method_reference(execution['research_program_ref'], agent.workspace)
            for staged_side, fixed_side in (('baseline', 'challenger'), ('challenger', 'baseline')):
                body = bodies.get(staged_side, {})
                current = body.get('staged_research', {})
                if (bodies.get(fixed_side) == stage['continuation']
                        and all(current.get(key) == stage[key] for key in
                            ('operation', 'baseline', 'continuation', 'coordinate_paths', 'profile_scope'))
                        and {k: v for k, v in body.items() if k != 'staged_research'}
                            == {k: v for k, v in original.items() if k != 'staged_research'}):
                    matching.extend((reference, staged_side, fixed_side) for reference in group['pair_refs'])
        if not matching:
            continue
        basis = {'proposal_ref': deepcopy(proposal_ref),
            'paired_returns': [{'pair_ref': ref, 'staged_side': staged, 'fixed_side': fixed}
                for ref, staged, fixed in matching],
            'evaluation': 'frozen_conditions_on_paid_completed_prefix_returns'}
        revisions = plan.setdefault('staged_method_revision_refs', [])
        for reference in reversed(revisions):
            previous = read_method_reference(reference, agent.workspace)
            if previous['basis'] == basis and not previous.get('budget_incomplete'):
                return {**_revision_publication(agent, question, reference, previous, 0), 'reused': True}
        retained_sources = proposal.get('baseline_evidence', [])
        required = 1 + len(retained_sources) + len(matching) * (3 + len(candidates))
        if compute_limit < required:
            return {'status': 'staged_condition_revision_not_funded', 'budget_incomplete': True,
                'required_compute': required, 'new_compute_cost': 0}
        agent.costs['compute_cost'] += required
        retained, excluded, seen = [], [], []
        # Unchanged frozen conditions already have paid predictions on their
        # development sources. Keep those witnesses; do not silently replace
        # the original support set with the latest two returns.
        for source in retained_sources:
            preference = next((item for item in proposal['research']['preferences']
                if item.get('feedback_ref') == source['pair_ref']), None)
            frozen = []
            for index, candidate in enumerate(candidates):
                paid = next((item for item in candidate['condition_source'].get('predicted_returns', [])
                    if item['feedback_ref'] == source['pair_ref']), None)
                if paid is not None:
                    frozen.append({'candidate': index, 'helpful': paid['helpful']})
            if (not preference or preference['preference'] not in ('baseline', 'challenger')
                    or len(frozen) != len(candidates)):
                excluded.append({'pair_ref': source['pair_ref'],
                    'reason': 'retained_source_condition_witnesses_incomplete'})
                continue
            seen.append(source['origin_ref'])
            retained.append({**deepcopy(source), 'preference': deepcopy(preference),
                'frozen_condition_predictions': frozen, 'original_development_support': True})
        old_source_count = len(retained)
        for pair_ref, staged_side, fixed_side in matching:
            pair, = load_paired_method_returns(agent, [pair_ref])
            origin = pair.get('origin_ref', pair_ref)
            if origin in seen:
                continue
            seen.append(origin)
            contrast, evidence = _stage_return_pair(agent, pair, staged_side, fixed_side)
            if contrast is None or evidence['preference']['preference'] not in ('baseline', 'challenger'):
                excluded.append({'pair_ref': pair_ref, **evidence})
                continue
            retained.append(evidence)
        verdicts = []
        for index, candidate in enumerate(candidates):
            condition = candidate['research_program']['staged_research']['condition']
            predictions = []
            for evidence in retained:
                paid_vote = next((item for item in evidence.get('frozen_condition_predictions', [])
                    if item['candidate'] == index), None)
                vote = ([paid_vote] if paid_vote else _conflict_predictions({
                    'coordinate_paths': stage['coordinate_paths'],
                    'candidates': [{'condition': condition}]}, evidence['profile']))
                expected = evidence['preference']['preference'] == 'challenger'
                predictions.append({'pair_ref': evidence['pair_ref'], 'expected_append': expected,
                    'predicted_append': vote[0]['helpful'] if vote else None,
                    'contradicted': bool(vote) and vote[0]['helpful'] != expected,
                    'reused_frozen_development_prediction': paid_vote is not None})
            verdicts.append({'original_candidate_index': index,
                'candidate_ref': {**proposal_ref, 'json_path': ['proposal', 'candidates', index]},
                'condition': deepcopy(condition), 'predictions': predictions,
                'status': ('refuted_by_paid_return' if any(item['contradicted'] for item in predictions)
                    else 'survives_paid_returns' if predictions and all(
                        item['predicted_append'] is not None for item in predictions)
                    else 'unmeasured')})
        survivors = [item['original_candidate_index'] for item in verdicts
                     if item['status'] == 'survives_paid_returns']
        refuted = [item['original_candidate_index'] for item in verdicts
                   if item['status'] == 'refuted_by_paid_return']
        status = ('staged_condition_evidence_incomplete' if not retained else
            'staged_conditions_exhausted' if not survivors else
            'staged_conditions_revised' if refuted else 'staged_conditions_retained')
        serial = 0
        while (agent.workspace / 'method_feedback' / f'staged_condition_revision_{serial:06d}.json').exists():
            serial += 1
        path = f'method_feedback/staged_condition_revision_{serial:06d}.json'
        reference = {'file': str(agent.workspace / path)}
        revised = deepcopy(original) if survivors else None
        if revised:
            current = revised['staged_research']
            current.update(conditions=[deepcopy(verdicts[i]['condition']) for i in survivors],
                condition=deepcopy(verdicts[survivors[0]]['condition']),
                condition_revision_ref=deepcopy(reference),
                evidence_refs=[*current.get('evidence_refs', []), *[item['pair_ref'] for item in retained]],
                scope='paid_counterexample_revision_pending_independent_validation', independent_support=False)
        receipt = {'status': status, 'new_compute_cost': required, 'required_compute': required,
            'budget_incomplete': False, 'survivor_indices': survivors, 'refuted_indices': refuted,
            'paired_sources': len(seen), 'usable_sources': len(retained),
            'original_development_sources': old_source_count,
            'new_paid_sources': len(retained) - old_source_count,
            'general_guard_supported': False, 'independent_support': False,
            'new_fits': 0, 'new_predictions': 0, 'new_observations': 0,
            'condition_resynthesis': False,
            'required_work': ('consume_revised_condition_agreement_then_assess_new_source_validation'
                if survivors else 'research_missing_method_evidence_or_condition_content')}
        bundle = {'basis': basis, 'receipt': receipt, 'status': status, 'condition_verdicts': verdicts,
            'paid_stage_evidence': retained, 'excluded_returns': excluded, 'research_program': revised,
            'original_proposal_ref': deepcopy(proposal_ref), 'budget_incomplete': False,
            'data_role': 'exposed_development', 'independent_support': False,
            'source_support_inherited': False}
        agent._save(path, json.dumps(bundle, separators=(',', ':')))
        revisions.append(reference)
        agent._save('questions.json', json.dumps(agent.questions, separators=(',', ':')))
        return _revision_publication(agent, question, reference, bundle, required)
    return {'status': 'no_matching_staged_returns', 'new_compute_cost': 0}
