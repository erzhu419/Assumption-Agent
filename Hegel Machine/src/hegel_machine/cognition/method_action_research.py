"""Connect addressed method experience to finite ordinary action revisions."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path

from .checkpoint_storage import _JsonSource, read_json_checkpoint


def read_method_reference(reference, workspace):
    """Read one declared value, leaving the surrounding archive file-backed."""
    path = Path(workspace) / reference['file']
    with ExitStack() as opened:
        source, span = _JsonSource(path, opened), None
        for part in reference.get('json_path', []):
            span = source.members(span, keys={part})[part]
        if span is None:
            # Standalone feedback/profile contracts are already one input.
            # Large research archives are addressed by the caller's json_path.
            return read_json_checkpoint(path)
        return json.loads(source.text[span[1]:span[2]])


def _topic_method_candidates(agent, question):
    owners = [question]
    topic = question.get('origin_method_question_ref')
    if topic:
        owners.extend(item for item in agent.questions
            if item['id'] == topic['question_id']
            and item.get('meaning_version', 1) == topic.get('meaning_version', 1))
    library, methods = agent.theory_consolidation.get('candidates', {}), []
    for owner in owners:
        references = list(owner.get('candidate_method_refs', []))
        identifier = owner.get('method_use_plan', {}).get('candidate_id')
        if identifier in library:
            references.append({'id': identifier, 'version': library[identifier]['version']})
        for reference in references:
            candidate = library.get(reference['id'])
            if (candidate is not None and candidate.get('version') == reference.get('version')
                    and candidate.get('selection_plan', {}).get('condition') is not None
                    and all(item['id'] != candidate['id'] for item in methods)):
                methods.append(candidate)
    return methods


def _native_method_pair_groups(agent, records):
    """Keep the measured challenger attached to its actual frozen native body."""
    fields = ('id', 'version', 'kind', 'execution', 'selection_plan',
              'ordinary_source_ref', 'source', 'source_ref')
    groups, excluded = [], []
    for reference, record in records:
        pair_ref = record.get('method_pair_ref')
        if pair_ref is None:
            continue
        method_ref, artifact_ref = record.get('method_ref'), record.get('method_artifact_ref')
        reason = None
        if method_ref is None or artifact_ref is None:
            reason = 'actual_native_method_body_required'
        else:
            artifact = read_method_reference(artifact_ref, agent.workspace)
            pair = read_method_reference(pair_ref, agent.workspace)
            challenger = read_method_reference(pair['challenger_ref'], agent.workspace)
            measured_ref = challenger.get('method_artifact_ref')
            if ({key: artifact.get(key) for key in ('id', 'version')} != method_ref
                    or challenger.get('method_ref') != method_ref):
                reason = 'paired_native_body_identity_mismatch'
            elif measured_ref is None:
                reason = 'actual_paired_native_body_required'
            else:
                measured = read_method_reference(measured_ref, agent.workspace)
                if any(measured.get(key) != artifact.get(key) for key in fields):
                    reason = 'paired_native_body_content_mismatch'
            baseline = read_method_reference(pair['baseline_ref'], agent.workspace)
            baseline_artifact_ref = baseline.get('method_artifact_ref')
            if reason is None:
                if baseline_artifact_ref is None or baseline.get('method_ref') is None:
                    reason = 'actual_baseline_method_body_required'
                else:
                    baseline_artifact = read_method_reference(baseline_artifact_ref, agent.workspace)
                    if {key: baseline_artifact.get(key) for key in ('id', 'version')} != baseline['method_ref']:
                        reason = 'paired_baseline_body_identity_mismatch'
        if reason is not None:
            excluded.append({'feedback_ref': deepcopy(reference),
                'pair_ref': deepcopy(pair_ref), 'reason': reason})
            continue
        previous = next((item for item in groups if item['method_ref'] == method_ref), None)
        body = {key: deepcopy(artifact.get(key)) for key in fields}
        if previous is not None and previous['body'] != body:
            excluded.append({'feedback_ref': deepcopy(reference),
                'pair_ref': deepcopy(pair_ref), 'reason': 'native_method_version_changed_body'})
            continue
        baseline_ref = baseline['method_ref']
        baseline_body = {key: deepcopy(baseline_artifact.get(key)) for key in fields}
        previous_baseline = next((item for item in groups if item['baseline_method_ref'] == baseline_ref), None)
        if previous_baseline is not None and previous_baseline['baseline_body'] != baseline_body:
            excluded.append({'feedback_ref': deepcopy(reference),
                'pair_ref': deepcopy(pair_ref), 'reason': 'native_baseline_version_changed_body'})
            continue
        group = next((item for item in groups if item['method_ref'] == method_ref
                      and item['baseline_method_ref'] == baseline_ref), None)
        if group is None:
            group = {'method_ref': deepcopy(method_ref), 'method_artifact_ref': deepcopy(artifact_ref),
                     'body': body, 'baseline_method_ref': deepcopy(baseline_ref),
                     'baseline_method_artifact_ref': deepcopy(baseline_artifact_ref),
                     'baseline_body': baseline_body, 'pair_refs': []}
            groups.append(group)
        if pair_ref not in group['pair_refs']:
            group['pair_refs'].append(deepcopy(pair_ref))
    return groups, excluded


def review_native_method_returns(agent, question, *, compute_limit):
    """Review actual object-parent returns without inventing Python-method uses.

    Saved decision proposals belong to their original scenes. They are possible
    matched-development inputs, never instructions to repeat a completed task.
    A returned predictor, including a confirmed one, is not a method preference.
    """
    references = question['method_use_plan']['native_return_refs']
    required = 1 + len(references)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'native_return_review_not_funded',
            'required_compute': required, 'new_compute_cost': 0, 'next_action': None}
    returns, sources, decisions, alternatives, choice_sources = [], [], [], [], []
    records = []
    for reference in references:
        record = read_method_reference(reference, agent.workspace)
        records.append((reference, record))
        source = record.get('source_task_ref')
        if source is not None and source not in sources:
            sources.append(deepcopy(source))
        returns.append({'feedback_ref': deepcopy(reference), **{key: deepcopy(record[key])
            for key in ('source_task_ref', 'actual_result_ref', 'before_ref', 'status',
                'comparison_ref', 'confirmation_ref', 'cost_refs', 'execution_complete',
                'method_pair_ref', 'method_ref', 'method_artifact_ref')
            if key in record}})
        proposal_ref = record.get('decision_options_ref')
        if not proposal_ref:
            continue
        published = read_method_reference(proposal_ref, agent.workspace)
        sequence = isinstance(published, list)
        published = published if sequence else [published]
        for index, decision in enumerate(published):
            addressed = {**deepcopy(proposal_ref), 'json_path': [
                *proposal_ref.get('json_path', []), index]} if sequence else deepcopy(proposal_ref)
            if decision.get('action') == 'choose_research_action':
                # These are the ordinary policy's real eligible alternatives,
                # not the later single-stage continuation publisher. Preserve
                # their provenance without treating the chosen action's quote
                # as a complete quote for every unexecuted alternative.
                chosen = decision.get('selected_action')
                eligible = list(dict.fromkeys(decision.get('eligible_actions', [])))
                decisions.append({'decision_ref': addressed,
                    'source_task_ref': deepcopy(source), 'status': 'recorded_native_choice',
                    'default_operation': chosen, 'operations': eligible,
                    'reason': decision.get('reason')})
                other = [name for name in eligible if name != chosen]
                if chosen in eligible and other:
                    choice_sources.append({'decision_ref': deepcopy(addressed),
                        'source_task_ref': deepcopy(source), 'ordinary_operation': chosen,
                        'alternative_operations': other,
                        'status': 'decision_state_and_complete_alternative_quotes_required'})
                continue
            if decision.get('action') not in (None, 'publish_object_method_options'):
                continue
            if 'options' not in decision:
                continue
            options = decision.get('options', [])
            default = decision.get('default_operation')
            decisions.append({'decision_ref': addressed, 'source_task_ref': deepcopy(source),
                'status': decision.get('status'), 'reason': decision.get('reason'),
                'default_operation': default,
                'operations': [item['operation'] for item in options]})
            # One legal next stage is not an alternative to the default. The
            # publisher must explicitly identify the actual ordinary action.
            if default is None or not any(item['operation'] == default for item in options):
                continue
            for position, option in enumerate(options):
                if option['operation'] == default:
                    continue
                alternatives.append({'source_task_ref': deepcopy(source),
                    'decision_ref': deepcopy(addressed), 'proposal_ref': {
                        **deepcopy(addressed), 'json_path': [
                            *addressed.get('json_path', []), 'options', position]},
                    'ordinary_operation': default, 'alternative_operation': option['operation'],
                    'status': 'matched_execution_required'})
    agent.costs['compute_cost'] += required
    result = {'status': 'complete_native_return_review',
        'reason': ('different_native_segments_require_matched_execution' if alternatives else
                   'retained_native_choices_require_same_state_binding' if choice_sources else
                   'no_new_method_content_from_recorded_native_returns'),
        'ordinary_returns': returns, 'distinct_source_refs': sources,
        'recorded_decisions': decisions, 'matched_development_inputs': alternatives,
        'native_choice_sources': choice_sources,
        'relative_labels_created': 0, 'installed_candidate': None,
        'new_compute_cost': required, 'support_updated': False,
        'independent_confirmation': False, 'independent_method_gain': None,
        'next_action': None}
    groups, excluded = _native_method_pair_groups(agent, records)
    result['paired_method_exclusions'] = excluded
    result['paired_condition_research'] = []
    if groups:
        from .method_feedback import research_paired_method_conditions

        result['reason'] = 'native_matched_returns_consumed_by_existing_condition_learner'
        for group in groups:
            learned = research_paired_method_conditions(agent, group['method_ref'],
                group['pair_refs'], max(0, compute_limit - result['new_compute_cost']),
                max_conditions=8)
            result['new_compute_cost'] += learned['new_compute_cost']
            result['paired_condition_research'].append({
                'method_ref': deepcopy(group['method_ref']),
                'baseline_method_ref': deepcopy(group['baseline_method_ref']),
                'baseline_method_artifact_ref': deepcopy(group['baseline_method_artifact_ref']),
                'method_artifact_ref': deepcopy(group['method_artifact_ref']), **learned})
            result['relative_labels_created'] += sum(
                item['preference'] in {'baseline', 'challenger'} for item in learned['preferences'])
            if learned.get('budget_incomplete'):
                result.update(status='scoped_unknown',
                    reason='native_pair_condition_allocation_incomplete',
                    required_compute=result['new_compute_cost'] + learned['required_compute'])
                break
    elif excluded:
        result['reason'] = 'native_pair_body_identity_unresolved'
    return result


def review_python_method_returns(agent, question, *, public_spec, compute_limit):
    """Accumulate ordinary returns into a method-content hypothesis, not a gain label."""
    from .finite_native_author import publish_research_options, research_segment_options
    from .method_conditions import method_delivery_outcome

    references = question['method_use_plan']['ordinary_return_refs']
    saved = question.get('routing_execution', {}).get('result', {}).get('method_action_research', {})
    if (saved.get('status') in {'method_content_proposed_pending_validation', 'scoped_unknown'}
            and saved.get('proposals', {}).get('candidates')
            and [row['feedback_ref'] for row in saved.get('ordinary_returns', [])] == references
            and all(item.get('status') != 'analysis_budget_unresolved'
                    for item in saved['proposals'].get('experience_revisions', []))):
        return resume_python_method_proposals(agent, question, saved, compute_limit=compute_limit)
    required = 1 + len(references)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'method_return_review_not_funded',
            'required_compute': required, 'new_compute_cost': 0, 'next_action': None}
    rows, sources = [], []
    for reference in references:
        record = read_method_reference(reference, agent.workspace)
        rows.append({'feedback_ref': deepcopy(reference),
            'origin_ref': deepcopy(record.get('origin_ref', reference)),
            'method_ref': deepcopy(record['method_ref']),
            'delivery': method_delivery_outcome(record),
            'profile_ref': deepcopy(record.get('profile_ref')),
            'actual_result_ref': deepcopy(record.get('actual_result_ref')),
            'complete_comparison_ref': deepcopy(record.get('complete_comparison_ref')),
            'costs': deepcopy(record.get('costs')),
            'cost_refs': deepcopy(record.get('cost_refs', []))})
        source = record.get('method_content_source')
        if source and source not in sources:
            sources.append(deepcopy(source))
    view = research_segment_options(public_spec, agent.observations,
        {**agent.knowledge, 'program_results': []}, state=agent._program_state(),
        decision={'research_options': publish_research_options(agent, public_spec)})
    agent.costs['compute_cost'] += required
    result = {'status': 'method_actions_unresolved',
        'reason': 'ordinary_returns_require_new_matched_action_evidence',
        'ordinary_returns': rows, 'current_legal_options': view['options'],
        'excluded_options': view['excluded'],
        'selected_research_action': 'ordinary_research',
        'required_work': 'fresh_current_ordinary_proposal_and_matched_alternative_action_return',
        'relative_labels_created': 0, 'installed_candidate': None, 'next_action': None,
        'new_compute_cost': required, 'support_updated': False,
        'independent_confirmation': False}
    result['method_content_sources'] = sources
    # The invoked P0 and the conditional method being revised have different
    # identities. Use the existing shared topic, not P0's absent selection plan
    # or an arbitrary method from the entire library.
    methods = _topic_method_candidates(agent, question)
    result['revisable_method_refs'] = [
        {'id': item['id'], 'version': item['version']} for item in methods]
    distinct_sources = []
    for item in sources:
        if item['source_task_ref'] not in distinct_sources:
            distinct_sources.append(item['source_task_ref'])
    if len(distinct_sources) < 2 or not methods:
        result['reason'] = ('distinct_completed_sources_required' if len(distinct_sources) < 2
                            else 'existing_method_topic_body_required')
        return result
    if compute_limit < required + len(sources):
        result.update(status='scoped_unknown', reason='method_content_projection_not_funded',
                      required_compute=required + len(sources))
        return result
    experiences, excluded = _paid_action_experiences(agent, sources)
    agent.costs['compute_cost'] += len(sources)
    result['new_compute_cost'] += len(sources)
    result['experience_exclusions'] = excluded
    from .research_programs import derive_paid_witness_binding_revision
    proposals, analyses = [], []
    for method in methods:
        analysis = derive_paid_witness_binding_revision(experiences, method['selection_plan'],
            compute_limit=compute_limit - result['new_compute_cost'])
        cost = analysis['analysis_compute_cost']
        agent.costs['compute_cost'] += cost
        result['new_compute_cost'] += cost
        method_ref = {'id': method['id'], 'version': method['version']}
        analyses.append({'method_ref': method_ref, **analysis})
        if analysis.get('candidate'):
            proposals.append({**deepcopy(analysis['candidate']),
                'previous_method_ref': method_ref})
        if analysis['status'] == 'analysis_budget_unresolved':
            result.update(status='scoped_unknown', reason='method_content_analysis_not_completed')
            break
    result['proposals'] = {'status': 'ordinary_returns_consumed', 'candidates': proposals,
                           'experience_revisions': analyses}
    if proposals and result['status'] != 'scoped_unknown':
        result.update(status='method_content_proposed_pending_validation',
            reason='actual_research_relations_produced_unmeasured_method_content',
            required_work='compare_existing_method_body_before_new_matched_execution')
        if _bind_current_method_dependencies(agent, result, compute_limit):
            _resolve_method_proposals(agent, question, result, methods, compute_limit)
    elif result['status'] != 'scoped_unknown':
        result['reason'] = 'no_new_method_content_supported_by_accumulated_returns'
    return result


def resume_python_method_proposals(agent, question, pending, *, compute_limit):
    """Resolve saved proposals without repeating their paid source induction."""
    if pending.get('status') not in {'method_content_proposed_pending_validation', 'scoped_unknown'}:
        raise ValueError('an unresolved saved method proposal is required')
    result = deepcopy(pending)
    result.update(status='method_content_proposed_pending_validation', new_compute_cost=0,
                  proposal_analysis_reused=True, installed_candidate=None, next_action=None)
    if not result.get('proposals', {}).get('candidates'):
        return {**result, 'status': 'scoped_unknown', 'reason': 'saved_content_proposal_required'}
    methods = _topic_method_candidates(agent, question)
    available = [{'id': item['id'], 'version': item['version']} for item in methods]
    if any(item['previous_method_ref'] not in available for item in result['proposals']['candidates']):
        return {**result, 'status': 'scoped_unknown', 'reason': 'original_method_topic_body_required'}
    if _bind_current_method_dependencies(agent, result, compute_limit):
        _resolve_method_proposals(agent, question, result, methods, compute_limit)
    return result


def _bind_current_method_dependencies(agent, result, compute_limit):
    """New finite plans use the registered current finite execution adapter.

    A predecessor's frozen adapter can predate a newly proposed operation. Keep
    that predecessor intact; bind the new proposal to the actual current entry
    and retain both source references before comparing execution identities.
    """
    if not agent.knowledge.get('finite_native_author'):
        return True
    from .finite_native_author import export_finite_native_author

    if result['new_compute_cost'] >= compute_limit:
        result.update(status='scoped_unknown', reason='method_dependency_resolution_not_funded')
        return False
    result['new_compute_cost'] += 1
    agent.costs['compute_cost'] += 1
    current = agent.workspace / 'ordinary.py'
    if not current.is_file() or current.read_text(encoding='utf-8') != export_finite_native_author():
        result.update(status='scoped_unknown', reason='current_registered_finite_ordinary_source_required')
        return False
    for proposal in result['proposals']['candidates']:
        plan = proposal['research_selection_plan']
        previous = agent.workspace / plan['ordinary_method_path']
        proposal.setdefault('ordinary_dependency_revision', {
            'previous_source_ref': {'file': str(previous), 'json_path': []},
            'current_source_ref': {'file': str(current), 'json_path': []},
            'basis': 'registered_finite_native_author_entry'})
        plan['ordinary_method_path'] = str(current)
    return True


def _method_execution_body(plan):
    """Compare finite executable content without treating provenance as code."""
    from .research_programs import export_research_selection_program

    def content(value):
        if isinstance(value, dict):
            return {key: ('ordinary.py' if key == 'ordinary_method_path' else content(item))
                    for key, item in value.items() if key != 'source_refs'}
        if isinstance(value, list):
            return [content(item) for item in value]
        return value

    return export_research_selection_program(content(plan))


def _resolve_method_proposals(agent, question, result, methods, compute_limit):
    """Reuse an actually supported body, or retain an executable paired proposal.

    All work here is finite evidence analysis. A new body is neither installed
    nor given its predecessor's scientific support. The ordinary pairing entry
    owns the current source, resource and evaluation registration.
    """
    def charge():
        if result['new_compute_cost'] >= compute_limit:
            result.update(status='scoped_unknown', reason='method_proposal_resolution_not_funded')
            return False
        result['new_compute_cost'] += 1
        agent.costs['compute_cost'] += 1
        return True

    def anchored(reference):
        return {**deepcopy(reference), 'file': str(agent.workspace / reference['file'])}

    topic_ref = question.get('origin_method_question_ref') or {
        'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)}
    topic = next((item for item in agent.questions if item['id'] == topic_ref['question_id']
        and item.get('meaning_version', 1) == topic_ref.get('meaning_version', 1)), question)
    pending = []
    for proposal in result['proposals']['candidates']:
        previous = next(item for item in methods if
                        {'id': item['id'], 'version': item['version']} == proposal['previous_method_ref'])
        plan = proposal['research_selection_plan']
        source_path = agent.workspace / plan['ordinary_method_path']
        proposed_body = _method_execution_body(plan)
        for candidate in methods:
            if not charge():
                return
            if (_method_execution_body(candidate['selection_plan']) != proposed_body
                    or (agent.workspace / candidate['selection_plan']['ordinary_method_path']).read_text(
                        encoding='utf-8') != source_path.read_text(encoding='utf-8')):
                continue
            method_ref = {'id': candidate['id'], 'version': candidate['version']}
            feedback = agent.theory_consolidation.get('method_feedback', {}).get(candidate['id'], {})
            for reference in feedback.get('outcome_refs', []):
                if not charge():
                    return
                record = read_method_reference(reference, agent.workspace)
                actual_ref = record.get('actual_result_ref')
                if record.get('method_ref') != method_ref or not actual_ref:
                    continue
                actual = read_method_reference(actual_ref, agent.workspace)
                comparison = actual.get('comparison', {})
                quality = actual.get('quality', {})
                if (actual.get('method_ref') != method_ref
                        or actual.get('execution_complete') is not True
                        or actual.get('completed') is not True
                        or not comparison.get('models_requested')
                        or comparison.get('models_completed') != comparison['models_requested']
                        or comparison.get('truncated') or comparison.get('error')
                        or type(quality.get('loss')) not in (int, float)
                        or type(quality.get('threshold')) not in (int, float)
                        or not quality['loss'] <= quality['threshold']):
                    continue
                parent_ref = question.get('parent_question_ref') or {
                    'question_id': question.get('parent_id')}
                parent = next((item for item in agent.questions
                    if item['id'] == parent_ref.get('question_id')
                    and item.get('meaning_version', 1) == parent_ref.get(
                        'meaning_version', item.get('meaning_version', 1))), None)
                result.update(status='complete_supported_method_retained',
                    reason='same_topic_executable_body_has_actual_completed_support',
                    installed_candidate=deepcopy(candidate),
                    existing_method_support_refs=[anchored(actual_ref)],
                    existing_support_scope='actual_completed_invocation',
                    selected_research_action='reuse_existing_method', required_work=None,
                    next_action=({'action': 'run', 'arguments': {'operator': 'procedure',
                        'path': candidate['execution']['path'],
                        'max_steps': candidate['execution']['max_steps'], 'confirm': False,
                        'research_question': {'id': parent['id'],
                            'meaning_version': parent.get('meaning_version', 1)}}}
                        if parent is not None else None))
                if candidate.get('selection_ref'):
                    result['policy_selection_ref'] = anchored(candidate['selection_ref'])
                result.pop('validation_requests', None)
                return
        if not charge():
            return
        pair_ref, bundle = None, None
        for saved in topic.get('method_content_proposal_refs', []):
            if not charge():
                return
            retained = read_method_reference(saved, agent.workspace)
            if (retained['previous_method_ref'] == proposal['previous_method_ref']
                    and _method_execution_body(retained['cells']['10']['selection_plan']) == proposed_body
                    and Path(retained['ordinary_source_ref']['file']).read_text(
                        encoding='utf-8') == source_path.read_text(encoding='utf-8')):
                pair_ref, bundle = anchored(saved), retained
                break
        if bundle is None:
            directory = agent.workspace / 'method_feedback'
            number = 0
            while (directory / f'content_proposal_{number:06d}.json').exists():
                number += 1
            path = f'method_feedback/content_proposal_{number:06d}.json'
            proposed_id = f"{previous['id']}_content_{number:06d}"
            selection = anchored(previous['selection_ref']) if previous.get('selection_ref') else None
            bundle = {'status': 'unmeasured_local_research_intervention',
                'method_question_ref': deepcopy(topic_ref),
                'ordinary_source_ref': {'file': str(source_path), 'json_path': []},
                'previous_method_ref': deepcopy(proposal['previous_method_ref']),
                'source_refs': deepcopy(proposal['source_refs']),
                'cells': {'00': {'id': f"{previous['id']}_comparison_{number:06d}",
                    'previous_method_ref': deepcopy(proposal['previous_method_ref']),
                    'selection_plan': {**deepcopy(previous['selection_plan']),
                                       'ordinary_method_path': str(source_path)},
                    'source_refs': deepcopy(previous.get('source_evidence_refs', [])),
                    'selection_ref': selection},
                    '10': {'id': proposed_id, 'selection_plan': {**deepcopy(plan),
                            'ordinary_method_path': str(source_path)},
                        'source_refs': deepcopy(proposal['source_refs']), 'selection_ref': selection}},
                'source_support_inherited': False, 'execution_authorized': False}
            agent._save(path, json.dumps(bundle, indent=2))
            pair_ref = {'file': str(agent.workspace / path), 'json_path': []}
            topic.setdefault('method_content_proposal_refs', []).append(deepcopy(pair_ref))
        proposal['method_pair_ref'] = deepcopy(pair_ref)
        pending.append({'entry_point': 'scripts.run_state_method_pairing.prepare',
            'method_pair_ref': pair_ref,
            'method_question_ref': deepcopy(topic_ref),
            'previous_method_ref': deepcopy(proposal['previous_method_ref']),
            'candidate_id': bundle['cells']['10']['id'], 'max_steps': previous['execution']['max_steps'],
            'source_outcome_refs': [anchored(item['outcome_ref'])
                                    for item in result['method_content_sources']],
            'execution_authorized': False,
            'required_work': 'quote_on_current_registered_paid_scene_before_paired_execution'})
    result.update(validation_requests=pending,
        required_work='quote_on_current_registered_paid_scene_before_paired_execution')


def _paid_action_experiences(agent, references):
    """Project actual native returns and their pre-use parent witnesses.

    Sequence rewards are not inferred from final model loss. These records
    support a content proposal, while matched returns still govern adoption.
    """
    experiences, excluded = [], []
    for supplied in references:
        reference = supplied['outcome_ref']
        outcome = read_method_reference(reference, agent.workspace)
        comparison = outcome.get('comparison', {})
        if (outcome.get('execution_complete') is not True or comparison.get('truncated')
                or not comparison.get('models_requested')
                or comparison.get('models_completed') != comparison['models_requested']):
            excluded.append({'outcome_ref': deepcopy(reference),
                'reason': 'completed_whole_research_comparison_required'})
            continue
        profile_ref = outcome['profile_ref']
        profile = read_method_reference(profile_ref, agent.workspace)
        for returned in outcome.get('native_representation_returns', []):
            native_ref = returned['native_result_ref']
            native = read_method_reference(native_ref, agent.workspace)
            identifiers = set(native.get('parent_theory_ids', []))
            parents, seen = [], set()
            for witness in profile.get('research', {}).get('candidate_pool', {}).get('witnesses', []):
                transition = witness.get('transition', {})
                if (transition.get('status') != 'observed'
                        or transition.get('guard_depth') is None
                        or transition.get('mature_first_divergences') != 0
                        or transition.get('startup_first_divergences', 0) <= 0):
                    continue
                identifier = next((name.removesuffix('.py') for name in witness.get('models', [])
                    if name.removesuffix('.py') in identifiers), None)
                if identifier is None or identifier in seen:
                    continue
                seen.add(identifier)
                program_ref = {**deepcopy(outcome['start_state_ref']), 'json_path': [
                    *outcome['start_state_ref'].get('json_path', []), 'theory_index', identifier, 'program']}
                parents.append({'id': identifier,
                    'program': read_method_reference(program_ref, agent.workspace),
                    'source_ref': program_ref})
            experiences.append({'source_task_ref': deepcopy(supplied['source_task_ref']),
                'profile': profile, 'parent_programs': parents, 'native_result': native,
                'source_refs': [deepcopy(reference), deepcopy(profile_ref), deepcopy(native_ref),
                    *[deepcopy(parent['source_ref']) for parent in parents]]})
    return experiences, excluded


def research_method_actions(agent, question, request, *, public_spec, compute_limit):
    """Prepare a real method question; only matched paid pairs can install a body.

    This route never runs an alternative to manufacture a preference. Scientific
    paired execution belongs to its separately admitted ordinary study.
    """
    from .finite_native_author import research_segment_options, publish_research_options
    from .research_episodes import extract_method_segment
    from .research_programs import (propose_research_action_revisions,
        export_research_selection_program, rebuild_research_selection_guard)
    from .method_feedback import research_paired_method_conditions
    from .theory_consolidation import install_python_research_method

    roots = request.get('segments', [])
    # Projection/static choices have an explicit finite engineering-work count;
    # neither prediction nor scientific fitting is hidden in this route.
    experience_refs = request.get('action_experience_refs', [])
    required = 1 + len(roots) + len(experience_refs)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'method_projection_not_funded',
            'required_compute': required, 'new_compute_cost': 0, 'installed_candidate': None}
    segments = [extract_method_segment(agent, root_result_ref=item['root_result_ref'],
        decision_start_ref=item['decision_start_ref']) for item in roots]
    agent.costs['compute_cost'] += required
    sequences = []
    from .research_methods import observe_common_method_segment
    for supplied in request.get('segment_return_refs', []):
        segment = segments[supplied['segment_index']]
        observed = observe_common_method_segment(segment,
            read_method_reference(supplied['actual_return_ref'], agent.workspace),
            actual_return_ref=supplied['actual_return_ref'],
            profile=read_method_reference(supplied['profile_ref'], agent.workspace),
            source_task=supplied['source_task_ref'],
            state=question.get('method_sequence_experience'))
        question['method_sequence_experience'] = observed['state']
        sequences.append({key: value for key, value in observed.items() if key != 'state'})
    missing = [item for segment in segments for item in segment['missing_dependencies']]
    if (not roots or missing or not request.get('ordinary_proposal_ref')
            or not request.get('ordinary_method_path')):
        return {'status': 'method_actions_unresolved', 'reason': 'actual_method_inputs_incomplete',
            'segments': segments, 'missing_dependencies': missing,
            'sequence_observations': sequences, 'new_compute_cost': required, 'installed_candidate': None}
    ordinary = read_method_reference(request['ordinary_proposal_ref'], agent.workspace)
    memory = {**agent.knowledge, 'program_results': []}
    state = agent._program_state()  # Existing live view; do not copy the strong index.
    view = research_segment_options(public_spec, agent.observations, memory, state=state,
        decision={'research_options': publish_research_options(agent, public_spec)})
    tested_ref = request.get('tested_candidate_ref')
    tested = read_method_reference(tested_ref, agent.workspace) if tested_ref else None
    experience_exclusions = []
    if tested is not None:
        if (tested.get('kind') != 'python_research_method'
                or tested.get('execution', {}).get('kind') != 'python_select_actions'):
            raise ValueError('paired evidence requires its actual frozen Python method')
        proposals = {'status': 'actual_frozen_body_reused', 'baseline': {'id': 'P0'},
            'candidates': [{'id': tested['id'], 'research_selection_plan': tested['selection_plan'],
                'source_refs': tested.get('source_evidence_refs', []), 'tested_candidate_ref': tested_ref}]}
    else:
        experiences, experience_exclusions = _paid_action_experiences(agent, experience_refs)
        prior_ref = request.get('prior_selection_plan_ref')
        prior_plan = read_method_reference(prior_ref, agent.workspace) if prior_ref else None
        proposals = propose_research_action_revisions(ordinary, view['options'],
            source_refs=[item['root_result_ref'] for item in roots], max_candidates=2,
            selection_states=request.get('selection_states', ()),
            method_experiences=experiences, prior_selection_plan=prior_plan,
            compute_limit=compute_limit - required)
        analysis_cost = proposals.get('analysis_compute_cost', 0)
        agent.costs['compute_cost'] += analysis_cost
        required += analysis_cost
    candidates = proposals.get('candidates', [])
    result = {'status': 'paired_study_required' if candidates else 'complete_no_distinct_legal_alternative',
        'segments': segments, 'proposals': proposals, 'excluded_options': view['excluded'],
        'sequence_observations': sequences,
        'experience_exclusions': experience_exclusions,
        'installed_candidate': None, 'new_compute_cost': required,
        'execution_authorized': False, 'independent_confirmation': False,
        'support_updated': False}
    if (not candidates and proposals.get('experience_revision', {}).get('status')
            == 'analysis_budget_unresolved'):
        result.update(status='method_actions_unresolved', reason='method_content_analysis_not_completed')
    pair_refs = request.get('pair_refs', [])
    if not candidates or not pair_refs:
        return result
    if tested is None:
        result.update(status='method_actions_unresolved', reason='actual_tested_body_reference_required')
        return result
    selected = next((item for item in candidates if item['id'] == request.get('paired_candidate_id')), None)
    if selected is None:
        result.update(status='method_actions_unresolved', reason='actual_paired_candidate_required')
        return result
    # Pair references must name this exact frozen challenger. An alternative's
    # good return cannot be transferred to an untested body with a similar name.
    for reference in pair_refs:
        pair = read_method_reference(reference, agent.workspace)
        outcome = read_method_reference(pair['challenger_ref'], agent.workspace)
        if outcome.get('method_ref') != {'id': tested['id'], 'version': tested['version']}:
            result.update(status='method_actions_unresolved', reason='paired_body_identity_mismatch')
            return result
        measured_ref = outcome.get('method_artifact_ref')
        if measured_ref is None:
            result.update(status='method_actions_unresolved', reason='actual_paired_body_reference_required')
            return result
        measured = read_method_reference(measured_ref, agent.workspace)
        body_fields = ('id', 'version', 'kind', 'execution', 'selection_plan', 'ordinary_source_ref')
        if any(measured.get(key) != tested.get(key) for key in body_fields):
            result.update(status='method_actions_unresolved', reason='paired_body_content_mismatch')
            return result
    choice = research_paired_method_conditions(agent,
        {'id': tested['id'], 'version': tested['version']}, pair_refs, compute_limit - required,
        max_conditions=8, target_profile=getattr(agent, '_method_use_profile', None))
    result['paired_selection'] = choice
    result['identified_effects'] = deepcopy(choice['identified_effects'])
    result['evaluation_use'] = choice['evaluation_use']
    result['new_compute_cost'] += choice['new_compute_cost']
    if not choice.get('local_program_supported'):
        result['status'] = ('complete_no_supported_action_revision'
            if choice.get('status') != 'UNKNOWN' else 'method_actions_unresolved')
        return result
    rebuild = rebuild_research_selection_guard(selected['research_selection_plan'], choice)
    plan = rebuild['plan']
    result['guard_reconstruction'] = {key: value for key, value in rebuild.items() if key != 'plan'}
    # A prior candidate can still use the old selection file for its coverage.
    # Rechecking the same method must not overwrite that live evidence.
    stem = f"method_feedback/action_selection_{selected['id'].replace(':', '_')}"
    number = 0
    selection_path = f'{stem}_{number:04d}.json'
    while (agent.workspace / selection_path).exists():
        number += 1
        selection_path = f'{stem}_{number:04d}.json'
    agent._save(selection_path, json.dumps(choice, indent=2))
    if rebuild['changed']:
        identifier = request.get('revised_candidate_id')
        if not identifier or identifier == selected['id']:
            result.update(status='method_actions_unresolved', reason='new_guard_body_identity_required')
            return result
        installed = install_python_research_method(agent, candidate_id=identifier,
            source=export_research_selection_program(plan), selection_plan=plan,
            source_refs=selected.get('source_refs', []), paired_support_refs=choice['local_support_refs'],
            max_steps=request.get('max_steps', 48), previous_method_ref={
                'id': tested['id'], 'version': tested['version']},
            selection_ref={'file': selection_path, 'json_path': []},
            method_question_ref={'question_id': question['id'],
                                 'meaning_version': question.get('meaning_version', 1)})
    else:
        installed = agent.theory_consolidation.get('candidates', {}).get(tested['id'])
        if installed is None or any(installed.get(key) != tested.get(key) for key in body_fields):
            result.update(status='method_actions_unresolved', reason='restore_actual_tested_body_before_use')
            return result
        # Supporting observations revise experience, not the already tested
        # body's identity or its source-level two-seed claims.
        group = agent.theory_consolidation.setdefault('method_feedback', {}).setdefault(
            tested['id'], {'outcome_refs': []})
        selection_ref = {'file': selection_path, 'json_path': []}
        supports = group.setdefault('paired_selections', [])
        if selection_ref not in supports:
            supports.append(selection_ref)
        group['method_question_ref'] = {'question_id': question['id'],
                                        'meaning_version': question.get('meaning_version', 1)}
        from .theory_consolidation import _save
        _save(agent)
        installed = deepcopy(installed)
    parent_ref = question.get('parent_question_ref') or {'question_id': question.get('parent_id')}
    parent = next((item for item in agent.questions
        if item['id'] == parent_ref.get('question_id')
        and item.get('meaning_version', 1) == parent_ref.get('meaning_version', item.get('meaning_version', 1))), None)
    result.update(status=('complete_supported_method_retained'
        if rebuild['status'] == 'conditioned_method_retained' else 'local_method_action_revision_selected'),
        installed_candidate=installed,
        next_action=({'action': 'run', 'arguments': {'operator': 'procedure',
            'path': installed['execution']['path'], 'max_steps': installed['execution']['max_steps'],
            'confirm': False, 'research_question': {'id': parent['id'],
                'meaning_version': parent.get('meaning_version', 1)}}} if parent is not None else None))
    if installed.get('selection_ref'):
        result['policy_selection_ref'] = {**deepcopy(installed['selection_ref']),
            'file': str(agent.workspace / installed['selection_ref']['file'])}
    if parent is None:
        result['reason'] = 'actual_parent_use_binding_required_before_ordinary_execution'
    return result
