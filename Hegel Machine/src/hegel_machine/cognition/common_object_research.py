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
        method_refs=(), decision_options_ref=None, mission=None):
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
    for reference in references:
        previous = read_method_reference(reference, common.workspace)
        if previous['actual_result_ref'] == result_ref:
            if previous['source_task_ref'] != source_task_ref:
                raise ValueError('One native return must retain its original target identity')
            return {'status': 'reused_native_return', 'feedback_ref': deepcopy(reference),
                'method_question_refs': [{'question_id': owner['id'],
                    'meaning_version': owner.get('meaning_version', 1)}],
                'new_compute_cost': 0, 'independent_method_gain': None}

    actual = read_method_reference(result_ref, common.workspace)
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
                 raw_sources=(), source_task_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        if organization not in {'archive', 'fixed_pr2', 'pr3'}:
            raise ValueError('Use archive, fixed_pr2 or pr3 organization')
        if learned_methods and organization != 'pr3':
            raise ValueError('New frozen methods belong to the declared pr3 treatment')
        self.workspace = Path(workspace)
        self.organization = organization
        self.source_task_ref = deepcopy(source_task_ref)
        self.learned_methods = deepcopy(list(learned_methods))
        self.raw_sources = tuple(raw_sources)
        self.reconstructed_sources = []
        self.source_reconstruction_receipts = []
        self._sources_consumed = False
        self.common_object_consumers = []

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

    def _research_objects(self, *args, **kwargs):
        start = len(self.common_object_consumers)
        result = super()._research_objects(*args, **kwargs)
        for consumer in self.common_object_consumers[start:]:
            consumer.record_parent_result(result)
        return result

    def run(self, *args, **kwargs):
        start = len(self.common_object_consumers)
        result = super().run(*args, **kwargs)
        receipts = []
        for consumer in self.common_object_consumers[start:]:
            receipts.append(consumer.record_parent_response(result))
        if receipts:
            result['common_object_research'] = receipts
        if self.source_reconstruction_receipts:
            result['source_method_reconstruction'] = deepcopy(self.source_reconstruction_receipts)
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
             'confirmation_loss', 'costs', 'method_pair_ref', 'method_ref', 'method_artifact_ref')
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
