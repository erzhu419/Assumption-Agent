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
from .object_dependency_methods import ObjectDependencyMethods
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


def reconstruct_partial_role_source(*, spec, model, records, previous, source_ref,
                                   model_id, budget):
    """Publicly reconstruct a method from its actual source experiment.

    All arms can use the same paid source transitions and original proposal.
    This executes the existing assessment; an unread archive path does not
    count as reconstruction. No additional measurement or fit is performed.
    """
    from .common_object_lab import measured_selector_values
    from .object_dependency_methods import propose_partial_role_experiment

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
        spec = self._knowledge_spec(spec)
        frame = super().prepare_research_decision(spec, training, validation, budget, actions, **kwargs)
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
            from .checkpoint_storage import write_json_checkpoint
            from .object_continuation import encode_entry
            path = self.workspace / 'native_parent_before.json'
            write_json_checkpoint(path, encode_entry(selected_before))
            self._native_revision_before_ref = {'file': str(path)}
        plans = frame.get('replay_plans', [])
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

    def _record_actual_program_return(self, result):
        """Address the actual pre-call model and returned selected computation."""
        from .checkpoint_storage import write_json_checkpoint
        from .language import Program
        from .structural_research_program import research_program_fit_contract
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
        if not result.get('revision_evidence_ref') or self.organization == 'archive':
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
            response = {key: result[key] for key in ('status', 'program', 'costs', 'revision_evidence_ref')}
            common._save('parent_response.json', json.dumps(response, separators=(',', ':')))
            feedback = record_native_parent_research_return(common, source_task_ref=self.source_task_ref,
                result_ref={'file': str(common.workspace / 'parent_response.json')},
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
                    actions=actions, load_sources=load_sources)
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
    """The original archive policy can actually reconstruct the same method."""

    def __init__(self, *args, load_sources, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_sources = load_sources

    def propose(self, *args, **kwargs):
        if self._load_sources is not None:
            self.sources = [*self.sources, *self._load_sources()]
            self._load_sources = None
        return super().propose(*args, **kwargs)


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
            ('model_id', 'source_ref', 'status', 'dependency_methods') if key in source}
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
        if self.previous_ref:
            arguments['previous_ref'] = self.previous_ref
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
        completed = result.get('completed_relation')
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
            'purpose': 'distinguish_membership_dependency',
            'research_program': {key: relation[key] for key in
                                 ('excitation_role', 'matched_prefix_rows')},
            'interpret_dependencies': len(actions) == 1,
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
