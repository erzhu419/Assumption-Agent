"""Common child research inside the unchanged finite object/response author.

The comparison isolates added Common organization and source-method consumption.
The archive arm retains Unified's existing adaptive experiments and knowledge;
it is neither an unorganized agent nor an LLM Code baseline. Scientific fitting,
acquisition and response confirmation remain owned by UnifiedResearchAgent.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from .common_archive_inquiry import CommonArchiveInquiryAgent
from .models import Probe
from .object_dependency_methods import ObjectDependencyMethods, completed_position_relation, position_method_source
from .online_inquiry_llm import InquiryLLMConfig
from .unified_research import UnifiedResearchAgent


def record_native_parent_research_return(common, *, source_task_ref, result_ref,
        before_ref=None, comparison_ref=None, confirmation_ref=None, cost_refs=(),
        method_refs=(), decision_options_ref=None, revision_evidence_ref=None, mission=None):
    """Continue one method question with an actual native return.

    References identify the original target and immutable execution. A newer
    snapshot of that target is another return, not another independent source.
    This boundary records no method preference and installs no method body.
    """
    from .method_action_research import read_method_reference
    from .method_feedback import ensure_python_method_topic

    retained = next((q for q in common.questions if q['id'] == 'task:method_choice'), {})
    owner = ensure_python_method_topic(common,
        parent_question_ref={'question_id': 'task', 'meaning_version': 1},
        mission=(mission if mission is not None else retained.get('mission')
            or {'purpose': 'method_learning', 'allocation': {'compute': 0}}))
    owner['answer_type'] = 'research_action'
    plan = owner.setdefault('method_use_plan', {})
    references = plan.setdefault('native_return_refs', [])
    actual = read_method_reference(result_ref, common.workspace)
    revision_evidence_ref = revision_evidence_ref or actual.get('revision_evidence_ref')
    for reference in references:
        previous = read_method_reference(reference, common.workspace)
        if previous['actual_result_ref'] == result_ref:
            if previous['source_task_ref'] != source_task_ref:
                raise ValueError('One native return must retain its original target identity')
            if revision_evidence_ref and previous.get('revision_evidence_ref') != revision_evidence_ref:
                # Attach later-addressed evidence to the same paid return. Its
                # original record and physical-source count remain unchanged.
                additions = plan.setdefault('native_revision_evidence', [])
                addition = {'feedback_ref': deepcopy(reference),
                    'revision_evidence_ref': deepcopy(revision_evidence_ref)}
                if addition not in additions:
                    additions.append(addition)
                    owner.update(status='open', agenda_basis=deepcopy(plan))
                    owner.pop('completed_agenda_basis', None)
                    common._save('questions.json', json.dumps(common.questions, separators=(',', ':')))
            return {'status': 'reused_native_return', 'feedback_ref': deepcopy(reference),
                'method_question_refs': [{'question_id': owner['id'],
                    'meaning_version': owner.get('meaning_version', 1)}],
                'new_compute_cost': 0, 'independent_method_gain': None}

    record = {'kind': 'native_parent_research_return',
        'source_task_ref': deepcopy(source_task_ref),
        'actual_result_ref': deepcopy(result_ref), 'parent_returned': True,
        'status': actual.get('status', 'returned'),
        'source_identity_status': 'declared' if source_task_ref else 'unresolved',
        'cost_refs': deepcopy(list(cost_refs)), 'method_refs': deepcopy(list(method_refs)),
        'paired_method_effect': None, 'independent_method_gain': None,
        'world_support_updated': False}
    if 'execution_complete' in actual:
        record['execution_complete'] = actual['execution_complete']
    for field in ('method_pair_ref', 'method_ref', 'method_artifact_ref'):
        if field in actual:
            record[field] = deepcopy(actual[field])
    for name, reference in (('before_ref', before_ref), ('comparison_ref', comparison_ref),
            ('confirmation_ref', confirmation_ref), ('decision_options_ref', decision_options_ref)):
        if reference is not None:
            record[name] = deepcopy(reference)
    if revision_evidence_ref:
        record['revision_evidence_ref'] = deepcopy(revision_evidence_ref)
    path = f'native_method_returns/return_{len(references):04d}.json'
    common._save(path, json.dumps(record, separators=(',', ':')))
    reference = {'file': str(common.workspace / path)}
    references.append(reference)
    owner.update(status='open', agenda_basis={'native_return_refs': deepcopy(references)})
    owner.pop('completed_agenda_basis', None)
    common._save('questions.json', json.dumps(common.questions, separators=(',', ':')))
    return {'status': 'recorded_native_return', 'feedback_ref': deepcopy(reference),
        'method_question_refs': [{'question_id': owner['id'],
            'meaning_version': owner.get('meaning_version', 1)}],
        'new_compute_cost': 0, 'independent_method_gain': None}


def reconstruct_partial_role_source(*, previous, source_ref, model_id, budget,
                                    spec=None, model=None, records=None):
    """Publicly reconstruct a method from its actual source experiment.

    A position sequence already has two paid instrument assessments. Bind their
    frozen forecasts and actual returns. Older selector experiments reconstruct
    their assessment directly from the source proposal and measured transitions.
    """
    from .common_object_lab import measured_selector_values
    from .object_dependency_methods import form_position_owner_method, propose_partial_role_experiment

    if previous and 'position_history_refs' in previous:
        from .method_action_research import read_method_reference
        # New source declarations retain only the references. Relative entries
        # are anchored to the declared evidence file, independently of the target.
        base = Path(source_ref.get('file', '.')).parent
        history = [{key: read_method_reference(stage[key], base)
                    for key in ('position_experiment', 'completed_relation')}
                   for stage in previous['position_history_refs']]
        result = form_position_owner_method(history, budget)
    elif previous and 'position_history' in previous:
        result = form_position_owner_method(previous['position_history'], budget)
    else:
        result = propose_partial_role_experiment(
            SimpleNamespace(**spec), model, records,
            selected_values=measured_selector_values(model, records),
            previous=previous, compute_limit=budget.remaining, remaining_rows=0)
        if not budget.spend(result['compute_cost']):
            raise RuntimeError('Source reconstruction exceeded the shared allocation')
    method = result.get('method_object')
    if method is not None:
        method = {**deepcopy(method), 'model_id': model_id,
                  'source_ref': deepcopy(source_ref)}
    return {'method': method, 'assessment': result,
            'source_ref': deepcopy(source_ref), 'compute_cost': result['compute_cost'],
            'cost_scope': 'included_in_parent_search_budget'}


class CommonObjectResearchAgent(UnifiedResearchAgent):
    """Run one shared finite author with archive/fixed/Common-learned routing."""

    def __init__(self, *args, workspace, organization, learned_methods=(),
                 raw_sources=(), source_task_ref=None, program_sources=(),
                 learned_programs=(), retained_pool=(), native_method_conditions=(),
                 condition_sources=(), **kwargs):
        super().__init__(*args, **kwargs)
        if organization not in {'archive', 'fixed_pr2', 'pr3'}:
            raise ValueError('Use archive, fixed_pr2 or pr3 organization')
        if learned_methods and organization != 'pr3':
            raise ValueError('New frozen methods belong to the declared pr3 treatment')
        if learned_programs and organization != 'pr3':
            raise ValueError('A/B reconstruct methods from the shared source evidence')
        self.workspace = Path(workspace)
        self.organization = organization
        self.source_task_ref = deepcopy(source_task_ref)
        self.learned_methods = deepcopy(list(learned_methods))
        self.raw_sources = tuple(raw_sources)
        self.reconstructed_sources = []
        self.source_reconstruction_receipts = []
        self._sources_consumed = False
        self.common_object_consumers = []
        self.program_sources = tuple(program_sources)
        self.learned_programs = deepcopy(list(learned_programs))
        self.retained_pool = tuple(retained_pool)
        self.program_reconstruction_receipts = []
        self.retained_pool_receipts = []
        self._program_sources_consumed = False
        self.native_method_conditions = tuple(native_method_conditions)
        self._native_revision_before_ref = None
        self._canonical_programs = []
        self._source_program_bindings = []
        self.condition_sources = tuple(condition_sources)
        self.condition_reconstruction_receipts = []
        self._paid_native_state = None
        self._current_native_frame = None
        self._native_artifacts = []
        self._native_applications = []
        self._native_application_serial = 0
        self.failure_evidence_ref = None
        self.failure_compute_lower_bound = 0

    def _prepare_program_sources(self, budget):
        from .source_method_reconstruction import reconstruct_partition_source
        if self._program_sources_consumed:
            return 0
        started = budget.spent
        for source in self.program_sources:
            if any(item['source_ref'] == source['source_ref'] for item in self.learned_programs):
                continue
            receipt = reconstruct_partition_source(source, self.workspace, budget)
            self.program_reconstruction_receipts.append(receipt)
            self.learned_programs.extend(receipt['methods'])
        if self.condition_sources and not self.native_method_conditions:
            from .method_feedback import research_paired_method_conditions
            from .method_conditions import native_condition_bindings
            reconstructed = []
            for group in self.condition_sources:
                proxy = SimpleNamespace(workspace=self.workspace, costs={'compute_cost': 0})
                result = research_paired_method_conditions(proxy, group['method_ref'],
                    group['pair_refs'], budget.remaining)
                if not budget.spend(result['new_compute_cost']):
                    raise RuntimeError('Condition reconstruction exceeded the parent allowance')
                reconstructed.append({**group, **result})
                self.condition_reconstruction_receipts.append(result)
            self.native_method_conditions = tuple(native_condition_bindings(reconstructed, self.workspace))
        self._program_sources_consumed = True
        return budget.spent - started

    def _replay_program_queue(self, spec, calls, observations=None):
        spec = self._knowledge_spec(spec)
        ordinary = super()._replay_program_queue(spec, calls, observations)
        plans = []
        self._canonical_programs = []
        for source, payload, template, mappings in ordinary:
            additions = [item['research_program'] for item in self.learned_programs
                         if item['source_concept_id'] == source]
            for body in [*additions, payload]:
                executable = body
                if (body is not payload and not body.get('input_template')
                        and any(item.concept_id == source for item in self.library.concepts)):
                    from .research_program_inputs import abstract_research_inputs
                    record = next(item for item in self.learned_programs
                                  if item['source_concept_id'] == source and item['research_program'] == body)
                    source_spec = spec
                    if record.get('source_model_ref'):
                        from .method_action_research import read_method_reference
                        model = read_method_reference(record['source_model_ref'], self.workspace)
                        source_spec = self._knowledge_spec(replace(spec,
                            feature_names=tuple(model['raw_sensor_features']), context={'object_model': model}))
                    cached = next((item for item in self._source_program_bindings
                        if item['canonical'] == body and item['source_spec'] == source_spec), None)
                    if cached is None:
                        cached = {'canonical': body, 'source_spec': source_spec,
                            'executable': abstract_research_inputs(body, source_spec, self.library)}
                        self._source_program_bindings.append(cached)
                    executable = cached['executable'] or body
                own_template, own_mappings = self._program_body_bindings(executable, spec)
                if executable.get('input_template'):
                    bound_template, bound_mappings = own_template, own_mappings
                else:
                    bound_template, bound_mappings = template, mappings
                if not bound_mappings:
                    continue
                if not any(previous[0] == source and previous[1] == executable for previous in plans):
                    plans.append((source, executable, bound_template, bound_mappings))
                    self._canonical_programs.append({'source_concept_id': source,
                        'executable': executable, 'canonical': body})
        # All arms retain the original replay and fitting tools. A source must
        # first be recalled and bound by the ordinary author to enter this queue.
        return plans

    def _program_body_bindings(self, body, spec):
        from itertools import islice
        from .abstractions import AbstractTemplate
        from .memory import _object_schema
        from .scale_bindings import aggregate_role_binding
        if not body.get('input_template'):
            return None, []
        template = AbstractTemplate.from_dict(body['input_template'])
        schema = _object_schema(spec)
        source_schema = body.get('source_schema')
        aggregate = aggregate_role_binding(template, source_schema, schema)
        command = self._instrument_role_binding(template,
            {'object_schema': source_schema, 'source_features': list(template.source_features)}, spec) if schema is None else None
        if schema is None and template.object_roles and command is None:
            return template, []
        mappings = ([command] if command is not None else [aggregate] if aggregate is not None
            else list(islice(template.mappings(spec.feature_names,
                structural_transfer=self.config.structural_transfer, object_schema=schema), 2)))
        return template, mappings

    def prepare_research_decision(self, spec, training, validation, budget, actions, **kwargs):
        from .evaluation import loss
        research_spec = spec
        spec = self._knowledge_spec(spec)
        frame = super().prepare_research_decision(spec, training, validation, budget, actions, **kwargs)
        # Sparse roles can bind old knowledge without authorizing a complete
        # focal-structure search. Keep the original research representation.
        frame['spec'] = research_spec
        self._current_native_frame = frame
        originally_terminal = frame['terminal']
        known = {entry['candidate'].program for entry in frame['assessed']}
        available, unbound = [], []
        for entry in self.retained_pool:
            program = entry['candidate'].program
            template, mappings = self._program_body_bindings(entry.get('research_program', {}), spec)
            if template is not None and mappings:
                from .scale_bindings import bind_measurement_arguments
                from .research_drafts import _rebind_step
                variants = []
                for mapping in mappings:
                    bound = {name: mapping[role] for name, role in zip(template.source_features, template.roles)}
                    changed = {**entry, 'candidate': replace(entry['candidate'],
                        program=bind_measurement_arguments(program, bound))}
                    for field in ('research_program', 'structure'):
                        if entry.get(field):
                            changed[field] = _rebind_step(entry[field], bound)
                    if entry.get('retained_candidate_ref'):
                        changed['canonical_research_entry_ref'] = entry['retained_candidate_ref']
                    variants.append(changed)
            else:
                variants = [entry] if program.required_features() <= set(spec.feature_names) else []
            if not variants:
                unbound.append(entry['candidate'].candidate_id)
            for variant in variants:
                candidate = variant['candidate']
                if candidate.program not in known:
                    available.append(variant)
                    known.add(candidate.program)
        started = budget.spent
        checked = []
        for entry in available:
            required = len(training) + len(frame['comparison'])
            if budget.remaining < required:
                break
            budget.spend(len(training))
            current = {**entry, 'training_loss': loss(entry['candidate'].program,
                training, frame['fallback'], spec.risk_quantile)}
            checked.extend(self._assess([current], frame['comparison'], budget,
                                       frame['fallback'], spec.risk_quantile))
        frame['assessed'].extend(checked)
        frame['uncompared'].extend(available[len(checked):])
        frame['comparison_complete'] = not frame['uncompared']
        frame['native_full_pool_comparison'] = frame['comparison_complete'] and not unbound
        frame['terminal'] = originally_terminal or self._research_pool_explained(frame)
        frame['prepare_compute_cost'] += budget.spent - started
        frame['remaining_compute'] = budget.remaining
        receipt = {'action': 'compare_shared_retained_pool', 'retained_entries': len(self.retained_pool),
            'newly_scored_programs': len(checked), 'unbound_candidate_ids': unbound,
            'full_bound_pool_comparison': frame['comparison_complete'],
            'full_retained_pool_comparison': frame['comparison_complete'] and not unbound,
            'compute_cost': budget.spent - started}
        if self.retained_pool:
            actions.append(receipt)
            self.retained_pool_receipts.append(receipt)
        selected_before = self._best(frame['assessed'])
        if (self._native_revision_before_ref is None and frame['comparison_complete']
                and selected_before and selected_before.get('research_program')):
            self._native_revision_before_ref = self._save_native_artifact(
                'parent', self._native_entry(selected_before))
        plans = frame.get('replay_plans', [])
        if plans and frame['comparison_complete'] and not frame['terminal'] and selected_before:
            from .abstractions import AbstractTemplate
            from .research_programs import research_program_replay_inputs
            from .transfer_partitions import prepare_parent_parameter_revision, quote_transfer_partition
            # A saved source is a recipe, not a reason to discard the better
            # structure already selected on this task's complete working pool.
            for source, payload, _, _ in plans:
                recipe = payload.get('staged_research', {}).get('continuation', payload)
                if recipe.get('transfer_partition', {}).get('kind') != 'Pi1':
                    continue
                prepared = prepare_parent_parameter_revision(selected_before, recipe)
                if prepared.get('status') != 'ready':
                    continue
                body = prepared['research_program']
                signature = research_program_replay_inputs(body)
                if any(research_program_replay_inputs(prior) == signature for _, prior, _, _ in plans):
                    break
                quote = quote_transfer_partition(body, training, feedback_rows=frame['comparison'])
                reserve = min(4, self.config.max_candidates) * len(frame['comparison'])
                if quote.get('required_compute') is None or quote['required_compute'] + reserve > budget.remaining:
                    continue
                parent_ref = self._save_native_artifact('parent', self._native_entry(selected_before))
                recipe_ref = self._save_native_artifact('recipe', recipe)
                body = prepare_parent_parameter_revision(selected_before, recipe,
                    source_method_ref=recipe_ref, parent_ref=parent_ref)['research_program']
                # The retained winner is already in current sensor coordinates.
                # This identity adapter performs no new correspondence search.
                names = tuple(sorted(selected_before['candidate'].program.required_features()))
                template = AbstractTemplate('current_parent_parameters', source,
                    selected_before['candidate'].program, names, names, ())
                plans = [(source, body, template, [dict(zip(names, names))]), *plans]
                frame['replay_plans'] = plans
                self._canonical_programs.append({'source_concept_id': source,
                    'executable': body, 'canonical': body})
                frame['policy'].set_action_knowledge('program_replay', self._replay_knowledge(spec, plans))
                actions.append({'action': 'prepare_current_parent_parameter_replay',
                    'source_concept_id': source,
                    'parent_candidate_id': selected_before['candidate'].candidate_id,
                    'parent_loss': selected_before['validation_loss'],
                    'comparison_basis': 'complete_current_working_pool',
                    'required_compute': quote['required_compute'], 'comparison_reserve': reserve,
                    'paid_initializer_reused': False, 'new_fits': 0, 'new_observations': 0,
                    'compute_cost': 0})
                break
        if self.native_method_conditions and plans and not frame['terminal']:
            from .method_conditions import method_profile, select_native_method_programs
            quote = sum(len(item['research'].get('candidates', [])) for item in self.native_method_conditions)
            if budget.remaining >= quote:
                selected = self._best(frame['assessed'])
                rows = [*training, *(row for row in frame['comparison'] if row not in training)]
                trace = [*actions, *[{'action': 'observe', 'details': {'sequence': item['sequence']}}
                    for item in getattr(getattr(self, '_active_laboratory', None), 'acquisitions', [])]]
                profile = method_profile(self._knowledge_spec(spec), rows, selected['candidate'], trace,
                    residual_loss=selected['validation_loss'],
                    minimum_improvement=self.config.minimum_improvement,
                    research_state={'stage': 'native_direction_choice',
                        'comparison': {'stale': False, 'paid_rows': len(rows),
                            'models_requested': len(frame['assessed']) + len(frame['uncompared']),
                            'models_completed': len(frame['assessed'])},
                        'budget': {'remaining_compute': budget.remaining}})
                records = [{'source_concept_id': source, 'research_program': next((
                    entry['canonical'] for entry in self._canonical_programs
                    if entry['source_concept_id'] == source and entry['executable'] == payload), payload),
                    'queue_index': index} for index, (source, payload, _, _) in enumerate(plans)]
                choice = select_native_method_programs(records, profile, self.native_method_conditions)
                if choice['compute_cost'] > quote:
                    raise RuntimeError('Method-condition execution exceeded its actual candidate quote')
                budget.spend(choice['compute_cost'])
                frame['replay_plans'] = [plans[item['queue_index']] for item in choice['programs']]
                frame['policy'].set_action_knowledge('program_replay',
                    self._replay_knowledge(spec, frame['replay_plans']))
                frame['prepare_compute_cost'] += choice['compute_cost']
                frame['remaining_compute'] = budget.remaining
                actions.append({'action': 'select_native_research_program',
                    'decisions': choice['decisions'], 'compute_cost': choice['compute_cost'],
                    'profile': profile})
        return frame

    @staticmethod
    def _native_entry(entry):
        from .object_continuation import encode_entry
        return encode_entry({key: entry[key] for key in (
            'candidate', 'research_program', 'training_loss', 'validation_loss') if key in entry})

    def _save_native_artifact(self, kind, value):
        """Save actual input/output once, without copying its pool or history."""
        from .checkpoint_storage import write_json_checkpoint
        for previous, reference in self._native_artifacts:
            if previous == value:
                return deepcopy(reference)
        directory = self.workspace / 'native_program_returns'
        index = len(self._native_artifacts)
        while (directory / f'{kind}_{index:04d}.json').exists():
            index += 1
        path = directory / f'{kind}_{index:04d}.json'
        write_json_checkpoint(path, value)
        reference = {'file': str(path)}
        self._native_artifacts.append((deepcopy(value), reference))
        return deepcopy(reference)

    def _record_program_replay(self, payload, replayed, feature_mapping, **execution):
        binding = payload.get('parent_parameter_revision', {})
        if (not binding.get('parent_ref') or not binding.get('source_method_ref')
                or any(source != target for source, target in feature_mapping.items())):
            return
        # The current-parent adapter executes in current coordinates. Preserve
        # that actual input binding, rather than attributing the output to the
        # first model seen by the episode or to an old source fit.
        replayed['research_program']['parent_parameter_revision'] = deepcopy(binding)
        search = deepcopy(replayed.get('transfer_partition_search', {}))
        followup = deepcopy(replayed.get('transfer_partition_followup', {}))
        complete = bool(search.get('completed') and not search.get('failed_fits')
            and not execution['budget_incomplete']
            and (not payload['transfer_partition'].get('second_stage')
                 or followup.get('execution_complete')))
        research = {**execution, 'search': search, 'followup': followup,
            'parent_ref': deepcopy(binding['parent_ref']),
            'source_method_ref': deepcopy(binding['source_method_ref']),
            'execution_complete': complete,
            'status': 'completed_parameter_research' if complete else 'parameter_research_unresolved'}
        record = {'input_method_ref': self._save_native_artifact('input', payload),
            'parent_ref': deepcopy(binding['parent_ref']),
            'after_ref': self._save_native_artifact('output', self._native_entry(replayed)),
            'execution': {'action': 'program_replay', 'research_action': 'relation_parameter_revision',
                'invocation_index': self._native_application_serial,
                'research': research, 'search_complete': complete, 'full_comparison': False},
            'program': replayed['candidate'].program,
            'returned_body': deepcopy(replayed['research_program']),
            'comparison_rows': list((self._current_native_frame or {}).get('comparison', [])),
            'comparison_complete': False, 'return_ref': None}
        self._native_application_serial += 1
        record['invocation_ref'] = self._save_native_artifact('invocation', record['execution'])
        self._native_applications.append(record)
        replayed['native_program_application'] = deepcopy(record['invocation_ref'])
        self._save_program_comparison(record, full=False)

    def _save_program_comparison(self, record, *, full, loss=None, old_pool_size=0, new_count=0):
        if record['comparison_complete'] and not full:
            return
        actual = {**record['execution'], 'full_comparison': full,
            'comparison': {'full_comparison': full, 'old_pool_size': old_pool_size,
                'new_candidates': new_count, 'validation_loss': loss,
                'basis': 'current_prepared_working_pool'}}
        execution_ref = self._save_native_artifact('execution', actual)
        evidence = {'source_task_ref': deepcopy(self.source_task_ref),
            'before_entry_ref': record['parent_ref'], 'research_parent_ref': record['parent_ref'],
            'after_entry_ref': record['after_ref'], 'executed_method_ref': record['input_method_ref'],
            'method_execution_ref': execution_ref, 'source_operation': 'program_replay',
            'data_role': 'exposed_development', 'independent_support': False}
        evidence_ref = self._save_native_artifact('evidence', evidence)
        record['evidence_ref'] = evidence_ref
        record['comparison_complete'] = full
        record['return_ref'] = self._save_native_artifact('return', {
            'status': 'existing_method_execution_returned',
            'execution_complete': bool(full and actual['search_complete']),
            'revision_evidence_ref': evidence_ref, 'method_execution_ref': execution_ref})

    def _record_program_comparison(self, claims, checked, frame, *, full):
        for item in claims:
            reference = item.get('native_program_application')
            record = next((row for row in self._native_applications if row['invocation_ref'] == reference), None)
            if record is None or frame['comparison'] != record['comparison_rows']:
                continue
            measured = next((row for row in checked if row.get('native_program_application') == reference), None)
            self._save_program_comparison(record,
                full=bool(full and measured is not None),
                loss=measured.get('validation_loss') if measured else None,
                old_pool_size=len(frame['assessed']), new_count=len(claims))

    def _assess(self, claims, validation, budget, fallback, q):
        # Unfunded joint fits return None. Only delivered candidates enter
        # the pool and its native method comparison.
        claims = [claim for claim in claims if claim is not None]
        checked = super()._assess(claims, validation, budget, fallback, q)
        frame = self._current_native_frame
        if frame and validation == frame['comparison']:
            self._record_program_comparison(claims, checked, frame,
                full=frame.get('native_full_pool_comparison', frame['comparison_complete'])
                    and len(checked) == len(claims))
        return checked

    def compare_research_decision_return(self, prepared, research_action, new_claims, *args, **kwargs):
        result = super().compare_research_decision_return(prepared, research_action, new_claims,
            *args, **kwargs)
        self._record_program_comparison(new_claims, result['new_claims'], prepared,
            full=result['full_comparison'] and prepared.get('native_full_pool_comparison', True))
        return result

    def _record_actual_program_return(self, result):
        """Address the actual pre-call model and returned selected computation."""
        from .checkpoint_storage import write_json_checkpoint
        from .language import Program
        from .structural_research_program import research_program_fit_contract
        result['native_program_return_refs'] = [deepcopy(item['return_ref'])
            for item in self._native_applications if item['return_ref']]
        selected = next((item for item in reversed(self._native_applications)
            if item['program'].to_dict() == result.get('program')
            and result.get('research_program') == item['returned_body']), None)
        if selected:
            result['revision_evidence_ref'] = deepcopy(selected['evidence_ref'])
            return {'status': 'actual_program_execution_addressed',
                'revision_evidence_ref': result['revision_evidence_ref']}
        if not self._native_revision_before_ref or not result.get('research_program'):
            return None
        body = result['research_program']
        declared = research_program_fit_contract(body)['program'].expanded().simplified()
        if declared != Program.from_dict(result['program']).expanded().simplified():
            return {'status': 'selected_predictor_program_contract_unresolved'}
        after_path = self.workspace / 'native_parent_after.json'
        write_json_checkpoint(after_path, {'program': result['program'], 'research_program': body})
        evidence = {'source_task_ref': deepcopy(self.source_task_ref),
            'before_entry_ref': deepcopy(self._native_revision_before_ref),
            'after_entry_ref': {'file': str(after_path)},
            'source_operation': 'ordinary_parent_research_return',
            'data_role': 'exposed_development', 'independent_support': False}
        path = self.workspace / 'native_revision_evidence.json'
        write_json_checkpoint(path, evidence)
        result['revision_evidence_ref'] = {'file': str(path)}
        return {'status': 'actual_before_after_addressed', 'revision_evidence_ref': result['revision_evidence_ref']}

    def _consume_program_return(self, result, receipts):
        if (not result.get('revision_evidence_ref') and not result.get('native_program_return_refs')) or self.organization == 'archive':
            return
        from .question_routing import consume_method_return_agenda
        remaining = max(0, self.config.compute_budget - result['costs']['compute_cost'])
        if receipts:
            common = self.common_object_consumers[0].common
            feedback = receipts[0]['method_feedback']
        else:
            common = CommonArchiveInquiryAgent(None, self.workspace / 'response_method', arm='hegel',
                source_task=False, config=InquiryLLMConfig(max_model_calls=0, token_budget=0,
                    compute_budget=remaining, observation_budget=0))
            response = {key: result[key] for key in ('status', 'program', 'costs', 'revision_evidence_ref') if key in result}
            common._save('parent_response.json', json.dumps(response, separators=(',', ':')))
            feedback = record_native_parent_research_return(common, source_task_ref=self.source_task_ref,
                result_ref={'file': str(common.workspace / 'parent_response.json')},
                mission={'purpose': 'method_learning', 'allocation': {'compute': remaining}})
        for reference in result.get('native_program_return_refs', []):
            if any(item['return_ref'] == reference and item['evidence_ref'] == result.get('revision_evidence_ref')
                    for item in self._native_applications):
                continue
            feedback = record_native_parent_research_return(common,
                source_task_ref=self.source_task_ref, result_ref=reference,
                mission={'purpose': 'method_learning', 'allocation': {'compute': remaining}})
        # A previous child may have had a smaller mission. The returned parent
        # now supplies its actual remaining allowance; no scientific cost grows.
        owner = next(question for question in common.questions if question['id'] == 'task:method_choice')
        owner['mission'] = {'purpose': 'method_learning', 'allocation': {'compute': remaining}}
        before = common.costs['compute_cost']
        returned = consume_method_return_agenda(common, feedback, None, compute_limit=remaining)
        spent = common.costs['compute_cost'] - before
        result['costs']['compute_cost'] += spent
        result['native_method_return'] = {'feedback': feedback, 'review': returned,
            'new_compute_cost': spent, 'cost_scope': 'included_in_parent_search_budget'}

    def _reconstruct_sources(self, budget, actions):
        if not self._sources_consumed:
            from .method_action_research import read_method_reference
            for declared in self.raw_sources:
                source = (read_method_reference(declared, self.workspace)
                    if 'file' in declared else declared)
                compiled = next((method for method in self.learned_methods
                    if method.get('source_ref') == source['source_ref']), None)
                if compiled is not None:
                    continue
                receipt = reconstruct_partial_role_source(**source, budget=budget)
                self.source_reconstruction_receipts.append(receipt)
                if receipt['method'] is not None:
                    self.reconstructed_sources.append(receipt['method'])
                actions.append({'action': 'reconstruct_paid_object_method',
                    'source_ref': deepcopy(source['source_ref']),
                    'method_available': receipt['method'] is not None,
                    'compute_cost': receipt['compute_cost'],
                    'cost_scope': receipt['cost_scope']})
            self._sources_consumed = True
        return self.reconstructed_sources

    def _dependency_methods(self, sources, *, budget, actions):
        load_sources = lambda: self._reconstruct_sources(budget, actions)
        if self.organization == 'archive':
            if self.raw_sources:
                return _ReconstructingDependencyMethods(sources, budget=budget,
                    actions=actions, load_sources=load_sources, workspace=self.workspace)
            return super()._dependency_methods(sources, budget=budget, actions=actions)
        consumer = CommonObjectDependencyMethods(
            self.workspace / f'object_question_{len(self.common_object_consumers):03d}',
            [*sources, *self.learned_methods], budget=budget, actions=actions,
            load_sources=load_sources, source_task_ref=self.source_task_ref)
        self.common_object_consumers.append(consumer)
        return consumer

    def _research_objects(self, laboratory, channels, acquire, batches, budget, actions, **kwargs):
        # These are the live native objects, not another evidence snapshot.
        self._paid_native_state = {'batches': batches, 'budget': budget, 'actions': actions,
                                   'channels': channels, 'previous_model': kwargs.get('previous_model')}
        start = len(self.common_object_consumers)
        result = super()._research_objects(laboratory, channels, acquire, batches, budget, actions, **kwargs)
        for consumer in self.common_object_consumers[start:]:
            consumer.record_parent_result(result)
        return result

    def _save_failed_paid_state(self, setup_cost, compute_upper_bound):
        state = self._paid_native_state
        self.failure_compute_lower_bound = setup_cost + (state['budget'].spent if state else 0)
        if state is None:
            return
        from .checkpoint_storage import write_json_checkpoint
        from .object_continuation import encode_entry

        def plain(value):
            if hasattr(value, 'to_dict'):
                return plain(value.to_dict())
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            return value

        frame = self._current_native_frame
        selected = self._best(frame['assessed']) if frame and frame.get('assessed') else None
        path = self.workspace / 'failed_paid_state.json'
        payload = {'status': 'interrupted_native_research',
            'scope': 'actual_paid_evidence_not_a_completed_research_return',
            'batches': state['batches'], 'actions': state['actions'],
            'selected_channels': state['channels'],
            'previous_object_model': state['previous_model'],
            'confirmed_compute_lower_bound': self.failure_compute_lower_bound,
            'compute_upper_bound': compute_upper_bound,
            'compute_scope': 'native_debits_plus_source_setup; unreturned_child_spend_may_be_missing',
            'object_continuation_state': {
                'selected': encode_entry(selected),
                'schemas': [{'channels': list(channels), 'schema': schema}
                            for channels, schema in getattr(self, '_object_schemas', {}).items()],
                'models': [{'channels': list(channels), 'model': model}
                           for channels, model in getattr(self, '_object_models', {}).items()],
                'competitions': [{'features': list(features), 'competition': competition}
                    for features, competition in getattr(self, '_object_competitions', {}).items()],
                'candidate_frontiers': [{'features': list(features), 'entries': entries}
                    for features, entries in getattr(self, '_object_candidate_frontiers', {}).items()]}}
        write_json_checkpoint(path, plain(payload))
        self.failure_evidence_ref = {'file': str(path)}

    def run(self, *args, **kwargs):
        from .models import SearchBudget
        start = len(self.common_object_consumers)
        config = self.config
        previous_compute = (kwargs.get('previous') or {}).get('costs', {}).get('compute_cost', 0)
        setup = SearchBudget(max(0, config.compute_budget - previous_compute))
        setup_cost = self._prepare_program_sources(setup)
        self._paid_native_state = self._current_native_frame = None
        self._native_applications = []
        self.failure_evidence_ref = None
        try:
            self._active_laboratory = args[0] if args else kwargs.get('laboratory')
            self.config = replace(config, compute_budget=config.compute_budget - setup_cost)
            result = super().run(*args, **kwargs)
        except Exception:
            self._save_failed_paid_state(setup_cost, config.compute_budget)
            raise
        finally:
            self.config = config
            self._active_laboratory = None
            self._paid_native_state = self._current_native_frame = None
        result['costs']['compute_cost'] += setup_cost
        if self.program_reconstruction_receipts:
            result['source_program_reconstruction'] = deepcopy(self.program_reconstruction_receipts)
        if self.condition_reconstruction_receipts:
            result['source_condition_reconstruction'] = deepcopy(self.condition_reconstruction_receipts)
        if self.retained_pool_receipts:
            result['shared_retained_pool'] = deepcopy(self.retained_pool_receipts)
        evidence = self._record_actual_program_return(result)
        if evidence:
            result['native_revision_record'] = evidence
        receipts = []
        for consumer in self.common_object_consumers[start:]:
            receipts.append(consumer.record_parent_response(result))
        if receipts:
            result['common_object_research'] = receipts
        if self.source_reconstruction_receipts:
            result['source_method_reconstruction'] = deepcopy(self.source_reconstruction_receipts)
        self._consume_program_return(result, receipts)
        return result


class _ReconstructingDependencyMethods(ObjectDependencyMethods):
    """Keep the archive policy and run recalled position experiments with native measurements."""

    def __init__(self, *args, load_sources, workspace, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_sources = load_sources
        self.workspace = Path(workspace)
        self.result = None
        self._position_parent = None
        self._position_rows = 0
        self._remaining_rows = 0
        self._applied_result = None

    def propose(self, spec, model, records, serial, selected_values, *, remaining_rows=None):
        from .method_action_research import read_method_reference
        from .object_dependency_methods import _position_followup, _recalled_position_followup
        from .object_interventions import plan_paid_position_contrast

        if self._load_sources is not None:
            self.sources = [*self.sources, *self._load_sources()]
            self._load_sources = None
        source = position_method_source(self.sources)
        available = max(0, remaining_rows or 0)
        if self.result and self.result.get('actions') and not self.result.get('completed_relation'):
            # The originally offered contract still owns its remaining corners.
            result = self.result
        elif source and model.get('position_experiment_ref'):
            prior = read_method_reference(model['position_experiment_ref'], self.workspace)
            if prior.get('position_method_application'):
                return super().propose(spec, model, records, serial, selected_values, remaining_rows=remaining_rows)
            result = _recalled_position_followup(spec, model, records, prior, source, {}, self, available)
            if result.get('position_method_application'):
                decision = {'action': 'apply_position_owner_method', **result['position_method_application'],
                    'position_first_return_ref': result['position_first_return_ref'],
                    'field': result['position_experiment']['field'],
                    'excited_quantity': result['position_experiment']['excited_quantity']}
                if decision not in self.actions:
                    self.actions.append(decision)
        else:
            probe, question = super().propose(spec, model, records, serial, selected_values,
                                              remaining_rows=remaining_rows)
            if (probe is not None or source is None
                    or question['purpose'] != 'measured_dependency_methods_lack_contrast'
                    or not self.evidence or any(e['status'] != 'uninformative_pilot' for e in self.evidence)):
                return probe, question
            plan = plan_paid_position_contrast(spec, model, records, self.budget, remaining_rows=available)
            if 'corners' not in plan:
                return None, {'purpose': plan['status']}
            result = _position_followup(plan, {}, model, records, self, available, assess=False)
        self.result, self._position_parent = result, model
        self._position_rows, self._remaining_rows = len(records), available
        if not result.get('actions'):
            return None, {'purpose': result.get('reason', result['status']),
                          'position_experiment': 'position_experiment' in result}
        metadata = {**result['question'], 'research_program': {'excitation_role': 'quantity',
                    'matched_prefix_rows': 0}, 'position_experiment': True, 'interpret_dependencies': False}
        return Probe(f'object_intervention:{serial}', tuple(result['actions'][0]['sequence'])), metadata

    def observe(self, records):
        from .object_dependency_methods import _position_followup
        if self.result and self.result.get('actions') and len(records) > self._position_rows:
            available = max(0, self._remaining_rows - (len(records) - self._position_rows))
            context = {k: self.result[k] for k in ('position_method_application', 'position_first_return_ref')
                       if k in self.result}
            self.result = _position_followup(self.result['position_experiment'], context,
                self._position_parent, records, self, available, assess=True)
            self._position_rows, self._remaining_rows = len(records), available
            return True
        return super().observe(records)

    def consume_position_result(self, model):
        from .checkpoint_storage import write_json_checkpoint
        from .object_induction import apply_position_experiment
        measured = (self.result or {}).get('completed_relation')
        if measured is None or (measured['status'] != 'position_domain_narrowed'
                                and not self.result.get('position_method_application')):
            return model
        if self._applied_result is self.result:
            return self._position_parent
        directory = self.workspace / 'native_position_returns'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'return_{len(list(directory.glob("return_*.json"))):03d}.json'
        write_json_checkpoint(path, self.result)
        self._applied_result = self.result
        self._position_parent = {**apply_position_experiment(model, measured),
                                 'position_experiment_ref': {'file': str(path.resolve())}}
        return self._position_parent


class CommonObjectDependencyMethods:
    """Consume complete measured pairs while Unified owns the physical world.

    The ordinary object planner invokes this only at its membership decision
    point, after the shared calibration/role/position work. Its returned Probe
    goes through the original acquisition and inference loop. Child compute is
    debited once to that loop's SearchBudget; Common's account is a subset.
    """

    def __init__(self, workspace, sources, *, budget, actions, load_sources=None,
                 source_task_ref=None):
        self.budget, self.actions = budget, actions
        # Keep failed and old methods with their scopes. Large fitted object
        # models are not needed by the method consumer and are not recopied.
        self.sources = [{key: deepcopy(source[key]) for key in
            ('model_id', 'source_ref', 'status', 'dependency_methods', 'position_method') if key in source}
            for source in sources]
        self.common = CommonArchiveInquiryAgent(None, Path(workspace), arm='hegel',
            source_task=False, config=InquiryLLMConfig(max_model_calls=0, token_budget=0,
                compute_budget=budget.remaining, observation_budget=0))
        self.common._save('sources.json', json.dumps(self.sources))
        self.previous_ref = None
        self.result_ref = None
        self.result = None
        self.child = None
        self.serial = 0
        self.evidence = []
        self.model = None
        self._input_records = None
        self._input_model = None
        self._remaining_rows = 0
        self._proposal_row_count = 0
        self._load_sources = load_sources
        self.source_task_ref = deepcopy(source_task_ref)

    def _save(self, name, value):
        self.common._save(name, json.dumps(value, separators=(',', ':')))
        return {'file': name}

    def _consume(self, records, remaining_rows):
        if (self._input_records == records and self._input_model == self.model
                and self.result is not None):
            return self.result
        self._save('paid_records.json', records)
        before = self.common.costs['compute_cost']
        self.common.config = replace(self.common.config,
            compute_budget=before + self.budget.remaining,
            observation_budget=max(0, remaining_rows))
        arguments = {'operator': 'partial_role_experiment',
            'allocation': {'compute': self.budget.remaining},
            **{name + '_ref': {'file': path} for name, path in (
                ('spec', 'spec.json'), ('model', 'model.json'),
                ('selected_values', 'selected_values.json'),
                ('sources', 'sources.json'), ('records', 'paid_records.json'))}}
        previous = self.previous_ref
        if previous is None and position_method_source(self.sources):
            previous = self.model.get('position_experiment_ref')
        if previous:
            arguments['previous_ref'] = previous
        result = self.common._execute('run', arguments, None)
        spent = self.common.costs['compute_cost'] - before
        if not self.budget.spend(spent):
            raise RuntimeError('Common exceeded the shared object research allocation')
        self.result = result
        self._input_records = deepcopy(records)
        self._input_model = deepcopy(self.model)
        self.evidence = deepcopy(result.get('measured_methods', self.evidence))
        result_ref = self._save(f'consumption_{self.serial:03d}.json', result)
        self.serial += 1
        self.result_ref = result_ref
        self.actions.append({'action': 'consume_common_object_question',
            'question_id': self.child['id'] if self.child else None,
            'status': result['status'], 'result_ref': {
                'file': str(self.common.workspace / result_ref['file'])},
            'paid_rows': len(records), 'compute_cost': spent,
            'cost_scope': 'included_in_parent_search_budget'})
        relation = self.child.get('experiment_relation', {}) if self.child else {}
        if relation.get('kind') == 'position_membership_contrast':
            completed = completed_position_relation(result)
            if completed and (completed['field'] != relation['position_field']
                              or result['position_experiment']['excited_quantity'] != relation['excited_quantity']):
                completed = None
        else:
            completed = result.get('selector_assessment') or result.get('completed_relation')
            if completed and any(completed.get(key) != relation.get(key)
                                 for key in ('quantity_candidate', 'excitation_role', 'matched_prefix_rows')):
                completed = None
        if completed is not None and self.child is not None:
            answer = {'status': 'answered', 'answer': deepcopy(completed),
                'evidence': [result_ref]}
            self.common._return_structural_answer(self.child, answer)
            self.child = None
            self.common._active_question_id = 'task'
            self.common._save('questions.json', json.dumps(self.common.questions))
        return result

    def propose(self, spec, model, records, serial, selected_values, *, remaining_rows=None):
        if self._load_sources is not None:
            self.sources.extend(deepcopy(self._load_sources()))
            self._load_sources = None
            self._save('sources.json', self.sources)
        self.model = deepcopy(model)
        self._remaining_rows = max(0, remaining_rows or 0)
        self._proposal_row_count = len(records)
        self._save('spec.json', {'feature_bounds': spec.feature_bounds,
                               'max_probe_length': spec.max_probe_length})
        self._save('model.json', model)
        self._save('selected_values.json', selected_values)
        result = self._consume(records, self._remaining_rows)
        actions = result.get('actions', [])
        if result['status'] != 'proposed_complete_partial_role_experiment' or not actions:
            return None, {'purpose': result.get('reason', result['status']),
                          **({'position_experiment': True} if result.get('position_experiment') else {}),
                          'common_result_ref': deepcopy(self.result_ref)}
        if self.child is None:
            self.child = self.common._open_question({
                'id': f'membership_{self.serial:03d}',
                'question': 'Which remaining membership hypotheses are separated by this complete measured pair?',
                'parent_id': 'task', 'depends_on': [],
                'experiment_relation': deepcopy(result['relation']),
                'parent_model_before': {'file': f'parent_before_{self.serial:03d}.json'},
                'paired_contract': deepcopy(result['paired_contract'])})
            self._save(f'parent_before_{self.serial:03d}.json', model)
            self.common._active_question_id = self.child['id']
            self.common._save('questions.json', json.dumps(self.common.questions))
        relation = result['relation']
        metadata = {**deepcopy(result.get('question', {})),
            'purpose': ('distinguish_position_dependency' if result.get('position_experiment')
                        else 'distinguish_membership_dependency'),
            'research_program': {key: relation[key] for key in
                                 ('excitation_role', 'matched_prefix_rows')},
            'interpret_dependencies': len(actions) == 1 and not result.get('position_experiment'),
            **({'position_experiment': True} if result.get('position_experiment') else {}),
            'common_question_id': self.child['id'],
            'common_result_ref': deepcopy(self.result_ref)}
        # Only a contract actually returned for acquisition is a pending child
        # experiment. A callback's next proposal has not been chosen yet.
        self.previous_ref = deepcopy(self.result_ref)
        return Probe(f'common_object_intervention:{serial}',
                     tuple(deepcopy(actions[0]['sequence']))), metadata

    def observe(self, records):
        if self.result is None:
            return True
        remaining = max(0, self._remaining_rows - (len(records) - self._proposal_row_count))
        self._consume(records, remaining)
        return True

    def consume_position_result(self, model):
        """Apply this child's measured return to its live parent without another induction."""
        from .object_induction import apply_position_experiment
        measured = (self.result or {}).get('position_assessment')
        if not measured or (measured['status'] != 'position_domain_narrowed'
                and not (self.result.get('position_method_application') and self.result.get('completed_relation'))):
            return model
        reference = {'file': str(self.common.workspace / self.result_ref['file'])}
        if model.get('position_experiment_ref') == reference:
            return model
        updated = {**apply_position_experiment(model, measured), 'position_experiment_ref': reference}
        # A call on the very same rows with our returned parent consumes the
        # completed result already in memory, including its paid readout.
        source = position_method_source(self.sources)
        if not (source and not self.result.get('position_method_application')
                and len(measured.get('retained_owners', ())) == source['position_method']['remaining_owner_count']):
            self._input_model = deepcopy(updated)
        self.previous_ref = None
        return updated

    def consume_paid_position_return(self, model, *, plan_ref, assessment_ref):
        """Resume a completed instrument question without repeating its measured readout."""
        from .method_action_research import read_method_reference
        from .object_induction import apply_position_experiment

        plan = read_method_reference(plan_ref, self.common.workspace)
        measured = read_method_reference(assessment_ref, self.common.workspace)
        returned = {'status': measured['status'], 'position_experiment': plan,
            'position_assessment': measured, 'completed_relation': measured,
            'paid_position_refs': {'plan': deepcopy(plan_ref), 'assessment': deepcopy(assessment_ref)},
            'compute_cost': 0, 'cost_scope': 'previously_paid_readout'}
        comparison = measured.get('comparison', {})
        if (completed_position_relation(returned) is None
                or measured.get('source_model_id') != model.get('model_id')
                or (measured['status'] != 'unverified_position_intervention' and (
                    comparison.get('control') != plan['position_control']
                    or comparison.get('excited_quantity') != plan['excited_quantity']))):
            raise ValueError('The paid position answer must belong to this parent and experiment')
        child = next((q for q in self.common.questions
                      if q.get('paid_position_refs') == returned['paid_position_refs']), None)
        if child is not None and measured['status'] == 'position_domain_narrowed' and all(
                set(model['position_candidates'][name]['quantities']) <= set(domain)
                for name, domain in measured['position_domains_delta'].items()):
            # A later paid answer can refine this one's domain further. Do not
            # replay the earlier answer over that descendant on continuation.
            return model
        # Validate the actual domain application before recording an answer.
        narrowed = apply_position_experiment(model, measured)
        if child is None:
            reference = self._save(f'consumption_{self.serial:03d}.json', returned)
            self.serial += 1
            child = self.common._open_question({
                'id': f'membership_{self.serial:03d}', 'parent_id': 'task', 'depends_on': [],
                'question': 'Which position memberships survive the completed measured contrast?',
                'paid_position_refs': returned['paid_position_refs'],
                'experiment_relation': {'kind': 'position_membership_contrast',
                    'position_field': plan['field'], 'excited_quantity': plan['excited_quantity']},
                'paid_return_ref': reference})
            status = ('answered' if measured['status'] in
                      {'position_domain_narrowed', 'position_domain_unchanged'} else 'unresolved')
            self.common._return_structural_answer(child, {'status': status,
                'answer': measured, 'evidence': [deepcopy(assessment_ref)]})
            self.actions.append({'action': 'consume_paid_position_return',
                'question_id': child['id'], 'result_ref': deepcopy(assessment_ref),
                'status': measured['status'], 'compute_cost': 0, 'new_observations': 0})
        self.result, self.result_ref = returned, child['paid_return_ref']
        self.child = None
        self.common._active_question_id = 'task'
        updated = self.consume_position_result(narrowed)
        self.model = deepcopy(updated)
        self.record_parent_result(updated)
        return updated

    def snapshot(self):
        return deepcopy(self.evidence)

    def record_parent_result(self, model):
        reference = self._save('parent_result.json', model)
        root = next(question for question in self.common.questions if question['id'] == 'task')
        root['object_research_result_ref'] = reference
        root['status'] = 'object_research_returned'
        self.common._save('questions.json', json.dumps(self.common.questions))
        self._save('account.json', {'compute_cost': self.common.costs['compute_cost'],
            'cost_scope': 'included_in_parent_search_budget',
            'observation_cost': 0, 'measurement_cost': 0, 'model_calls': 0,
            'parent_search_compute': self.budget.spent,
            'parent_result_ref': reference})

    def record_parent_response(self, result):
        from .method_action_research import read_method_reference

        response = {key: deepcopy(result[key]) for key in
            ('program', 'status', 'execution_complete', 'fallback', 'selected_channels',
             'confirmation_loss', 'costs', 'method_pair_ref', 'method_ref', 'method_artifact_ref',
             'revision_evidence_ref')
            if key in result}
        response['object_model_id'] = (result.get('object_research') or {}).get('model_id')
        published = [deepcopy(action) for action in result.get('actions', [])
            if action.get('action') in {'publish_object_method_options', 'choose_research_action'}]
        if published:
            response['decision_options'] = published
        root = next(question for question in self.common.questions if question['id'] == 'task')
        history = root.setdefault('response_result_refs', [])
        previous = root.get('response_result_ref')
        if previous and previous not in history:
            history.append(deepcopy(previous))
        if previous and read_method_reference(previous, self.common.workspace) == response:
            reference = previous
        else:
            name = f'parent_response_{len(history):03d}.json' if history else 'parent_response.json'
            reference = self._save(name, response)
            history.append(deepcopy(reference))
        root['response_result_ref'] = reference
        absolute = {**reference, 'file': str(self.common.workspace / reference['file'])}
        feedback = record_native_parent_research_return(self.common,
            source_task_ref=self.source_task_ref, result_ref=absolute,
            comparison_ref=result.get('comparison_ref'),
            confirmation_ref=({**absolute, 'json_path': ['confirmation_loss']}
                if 'confirmation_loss' in response else None),
            cost_refs=[{**absolute, 'json_path': ['costs']}] if 'costs' in response else [],
            decision_options_ref=({**absolute, 'json_path': ['decision_options']}
                if published else None))
        self.common._save('questions.json', json.dumps(self.common.questions))
        return {'questions_ref': {'file': str(self.common.workspace / 'questions.json')},
            'response_ref': absolute, 'method_feedback': feedback,
            'compute_cost': self.common.costs['compute_cost'],
            'cost_scope': 'included_in_parent_search_budget'}
