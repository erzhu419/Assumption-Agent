"""Connect finite response hypotheses to the existing physical controller."""

import ast
from copy import deepcopy
import json
from math import isfinite, isclose
from pathlib import PurePosixPath
from time import monotonic

from .world_reference_recalibration import _prepared_reference_recovery, _scan, refresh_response_evidence
from .world_response_hypotheses import propose_response_research


def response_premise_design_action(comparison):
    """Offer the existing experiment author when the bound comparison stalls."""
    research = comparison.get('family_research', {})
    if research.get('selected_role') != 'shared_premise_research':
        return None
    declaration = comparison['revision_action']['arguments']['question']
    binding = PurePosixPath(comparison['binding_path'])
    obligation = research['research_obligation']
    return {'action': 'run', 'arguments': {
        'operator': 'propose_explanations',
        'path': str(binding.with_name(binding.stem + '_premise.py')),
        'question': {'id': declaration['id'] + '_premise',
            'parent_id': declaration.get('parent_id', 'task'), 'depends_on': [],
            'question': obligation['question'] + ' Develop an executable design and answer '
                'for the existing contrast tool. Derive competing predictions for the same '
                'observable outcome from the paid evidence, and use the existing controls '
                'or feedback procedures to test them. If no available experiment separates '
                'the claims, explain that remaining ambiguity. Retain the response binding '
                'and its conditional results; choose the scientific content yourself.'}}}


def response_premise_proposal_inputs(agent, proposal_path):
    """Give the offered premise author its actual inputs, including failed trials."""
    for session in reversed(list(agent.llm_inquiries.values())):
        compared = session.get('proposal', {}).get('comparison_result')
        if not compared:
            continue
        binding = PurePosixPath(compared['binding_path'])
        if proposal_path not in (str(binding), str(binding.with_name(binding.stem + '_premise.py'))):
            continue
        research = compared['family_research']
        packet = {
            'binding_path': compared['binding_path'],
            'revision_action': deepcopy(compared['revision_action']),
            'controller_path': compared['revision_action']['arguments']['response_path'],
            'paid_state_path': 'shared/reference_recalibration.json',
            'bound_rows': deepcopy(compared.get('bound_rows', [])),
            'estimators': deepcopy(compared.get('estimators', [])),
            'assumptions': deepcopy(compared.get('assumptions', [])),
            'paid_family_checks': deepcopy(compared.get('paid_family_checks', [])),
            'response_candidates': deepcopy(compared.get('candidates', [])),
            'candidate_additions': deepcopy(compared.get('candidate_additions', [])),
            'unresolved_consequences': {key: deepcopy(research[key]) for key in
                ('panel', 'families', 'signatures', 'possible_signatures', 'research_obligation')
                if key in research},
            'comparison_evidence': deepcopy(compared.get('evidence', [])),
        }
        state_path = agent.workspace / packet['paid_state_path']
        if state_path.exists():
            state = json.loads(state_path.read_text())
            refresh_response_evidence(state, agent.observations, agent.lab.public_observation(),
                                      compared.get('estimators', []))
            table = state.get('research_table', {})
            packet['measurements'] = deepcopy(table.get('references', []) + table.get('targets', []))
            packet['targets'] = [row['crystal'] for row in table.get('targets', [])]
            packet['unfinished_targets'] = [number for number in packet['targets']
                                             if number not in state.get('completed', [])]
            packet['responses'] = deepcopy(state.get('response_settings', []))
            packet['control_grids'] = deepcopy(state.get('control_grids', {}))
            evidence = set(packet['comparison_evidence'])
            packet['responses_not_in_comparison'] = [deepcopy(row) for row in packet['responses']
                if not evidence.intersection(row['evidence'])]
        return packet
    return None


def _response_premise_context(agent, laboratory, comparison):
    """The paid scientific inputs, independent of question names and rereads."""
    from .world_response_consequences import _families

    state = json.loads((agent.workspace / 'shared/reference_recalibration.json').read_text())
    refresh_response_evidence(state, agent.observations, laboratory.public_observation(),
                              comparison['estimators'])
    table = state.get('research_table', {})
    context = {
        'bases': sorted(json.dumps(item['basis'], sort_keys=True)
                        for item in _families(comparison['estimators'])),
        'features': {str(row['crystal']): row['features'] for row in comparison['bound_rows']},
        'measurements': {str(row['crystal']): {key: value['value']
                         for key, value in row['measurements'].items()}
                         for row in table.get('references', []) + table.get('targets', [])},
        'responses': sorted([row['crystal'], row['frequency'], row['activated']]
                            for row in state['response_settings']),
        'control_grids': state['control_grids'],
    }
    evidence = set(comparison['evidence'])
    current = all(evidence.intersection(row['evidence']) for row in state['response_settings'])
    return context, current


def dispatch_response_premise(agent, laboratory):
    """Connect the existing stalled-family decision to the existing child loop."""
    if agent.agent_arm != 'hegel' or agent.provider is None or agent._inquiry_stack:
        return False
    remaining = agent.limits['max_model_calls'] - agent.costs['model_calls']
    if remaining < 2:
        return False
    authors = []
    for author in agent.llm_inquiries.values():
        comparison = author.get('proposal', {}).get('comparison_result')
        if comparison:
            context, current = _response_premise_context(agent, laboratory, comparison)
            # Ordinary physical trials need not recompare a computed binding.
            # Its old diagnosis cannot decide the stage on those new outcomes.
            if current:
                authors.append((author, comparison, context))
    # The parent may have a usable alternative binding. Its existing experiment
    # takes precedence over automatically researching an older stalled draft.
    if any(comparison.get('experiment_action') or comparison.get('selection')
           for _, comparison, _ in authors):
        return False
    for _, comparison, _ in reversed(authors):
        if (response_premise_design_action(comparison) is not None
                and _offer_response_incumbent(agent, laboratory, comparison)):
            return False
    extension = _dispatch_condition_extension(agent, laboratory, remaining)
    if extension is not None:
        return extension
    for author, comparison, context in reversed(authors):
        design = response_premise_design_action(comparison)
        if design is None:
            continue
        qid = author.get('premise_question_id', author['question_id'] + '_premise')
        previous = agent.llm_inquiries.get(qid, {})
        if previous.get('response_premise_context') == context:
            continue
        author['premise_question_id'] = qid
        declaration = design['arguments']['question']
        params = {'operator': 'inquire', 'max_model_calls': min(8, remaining - 1), 'question': {
            'id': qid, 'parent_id': declaration['parent_id'], 'depends_on': [],
            'question': comparison['family_research']['research_obligation']['question']
                + ' Investigate this unresolved premise using the paid evidence and existing tools. '
                'Develop and execute a distinguishing experiment when available, and analyze its '
                'actual outcome. Derive competing predictions from the evidence; choose the scientific '
                'content yourself. If the available observations cannot distinguish the claims, return '
                'that unresolved result. Retain the conditional response binding at '
                + comparison['binding_path'] + '. Its latest comparison and callable experiments are '
                'in method_decision.explanation_proposals. This is a research inquiry with world '
                'access, not a draft-only proposal phase.'}}
        question = next((q for q in agent.questions if q['id'] == qid), None)
        if question is not None:
            question['status'] = 'reopened'
            agent._update_dependencies(qid)
        from .world_llm_inquiry import execute_llm_inquiry
        outcome = execute_llm_inquiry(agent, laboratory, params)
        # Child-acquired evidence belongs to this attempt. It must not cause an
        # immediate automatic repeat when the child returns unknown or finishes.
        latest = author['proposal']['comparison_result']
        agent.llm_inquiries[qid]['response_premise_context'] = _response_premise_context(
            agent, laboratory, latest)[0]
        agent.events.append({'action': 'run', 'arguments': params,
            'trigger_origin': 'host_shared_premise_research',
            'feedback': {key: deepcopy(outcome[key]) for key in
                         ('operator', 'status', 'question', 'model_calls', 'notes')}})
        agent._save_state()
        if agent._on_checkpoint:
            agent._on_checkpoint(agent.result())
        return True
    return False


def _dispatch_condition_extension(agent, laboratory, remaining):
    """Resume the executable author when its finite pool cannot solve a target."""
    from .world_llm_inquiry import execute_llm_inquiry, refresh_condition_proposal

    path = (agent.research_advice or {}).get('proposal_path')
    if not path or 'proposal:' + path not in agent.llm_inquiries:
        return None
    conditions = refresh_condition_proposal(agent, path)
    if not conditions:
        return None
    if conditions.get('experiment_action'):
        return False
    author = agent.llm_inquiries['proposal:' + path]
    context = author['condition_research']['inputs']
    packet = response_premise_proposal_inputs(agent, path)
    session_id = 'proposal:' + packet['binding_path']
    previous = agent.llm_inquiries.get(session_id, {})
    if previous.get('condition_extension_context') == context:
        # An author can return after exhausting the finite pool too. That
        # completed attempt still belongs to the parent, rather than falling
        # through to another generic premise child on these same inputs.
        return False
    action = conditions.get('research_action')
    if action is None:
        return None
    params = deepcopy(action['arguments'])
    params['max_model_calls'] = min(8, remaining - 1)
    outcome = execute_llm_inquiry(agent, laboratory, params)
    current_advice = deepcopy(agent.research_advice)
    updated = refresh_condition_proposal(agent, path)
    if (current_advice and current_advice.get('proposal_path') != path
            and current_advice.get('next_action')):
        agent.research_advice = current_advice
    # The returned attempt includes its own new evidence and representation.
    # An unresolved child must not immediately schedule itself again.
    agent.llm_inquiries[session_id]['condition_extension_context'] = deepcopy(
        author['condition_research']['inputs'])
    agent.events.append({'action': 'run', 'arguments': params,
        'trigger_origin': 'host_condition_language_extension',
        'feedback': {**{key: deepcopy(outcome[key]) for key in
                      ('operator', 'status', 'question', 'model_calls', 'notes')},
                     'proposal': deepcopy(outcome['proposal']),
                     'condition_research': updated}})
    agent._save_state()
    if agent._on_checkpoint:
        agent._on_checkpoint(agent.result())
    return True


def _reference_state(agent, laboratory, path):
    from .world_response_transfer import KNOWLEDGE_PATH

    source = (agent.workspace / path).read_text()
    assignment = next(node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'ESTIMATORS' for target in node.targets))
    estimators = list(ast.literal_eval(assignment.value))
    state_path = agent.workspace / 'shared/reference_recalibration.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'completed': [], 'rejected': {}}
    knowledge_path = agent.workspace / KNOWLEDGE_PATH
    if knowledge_path.exists():
        state['frozen_response_knowledge'] = json.loads(knowledge_path.read_text())
    else:
        state.pop('frozen_response_knowledge', None)
    refresh_response_evidence(state, agent.observations, laboratory.public_observation(), estimators)
    return estimators, state, state_path


def _reference_response_inputs(state, estimators):
    """Scientific input equality lets an unchanged incumbent comparison be reused."""
    from .world_response_consequences import _families

    table = state.get('research_table', {})
    return {'bases': sorted((family['basis'] for family in _families(estimators)), key=repr),
        'measurements': {str(row['crystal']): {key: value['value']
                         for key, value in row['measurements'].items()}
                         for row in table.get('references', []) + table.get('targets', [])},
        'reference_frequencies': {str(row['crystal']): row['frequency']['value']
                                  for row in table.get('references', [])},
        'responses': sorted([row['crystal'], row['frequency'], row['activated']]
                            for row in state['response_settings']),
        'control_grids': state['control_grids'],
        'targets': state['target_numbers'], 'completed': sorted(state['completed']),
        'frozen_response_knowledge': state.get('frozen_response_knowledge')}


def _offer_response_incumbent(agent, laboratory, comparison):
    """Compare the retained raw route before assigning another stalled draft inquiry."""
    path = comparison['revision_action']['arguments']['response_path']
    estimators, state, _ = _reference_state(agent, laboratory, path)
    if not estimators:
        return False
    inputs = _reference_response_inputs(state, estimators)
    if state.get('response_research_inputs') != inputs:
        advice = deepcopy(agent.research_advice)
        result = execute_reference_response(agent, laboratory, {'path': path})
        agent.research_advice = advice
        agent.events.append({'action': 'run', 'arguments': {
            'operator': 'reference_response', 'path': path},
            'trigger_origin': 'host_retained_response_comparison',
            'feedback': {key: deepcopy(result[key]) for key in
                ('question_id', 'status', 'response_count', 'compute_cost',
                 'selected_number', 'selected_setting')}})
    else:
        result = state['response_research']
    experiment = (result.get('selection') or {}).get('selected_experiment')
    if experiment is None:
        advice = agent.research_advice or {}
        if (advice.get('diagnosis') == 'retained_response_experiment'
                and advice.get('next_action', {}).get('arguments', {}).get('path') == path):
            agent.research_advice = None
        return False
    agent.research_advice = {
        'diagnosis': 'retained_response_experiment', 'question_id': result['question_id'],
        'comparison_path': 'shared/reference_recalibration.json',
        'response_count': result['response_count'],
        'consequence_scope': result['selection'].get('consequence_scope', 'declared_response_families'),
        'selected_role': result.get('family_research', {}).get('selected_role'),
        'next_action': {'action': 'run', 'arguments': {
            'operator': 'reference_response', 'path': path,
            'experiment': {'number': experiment['number'], 'setting': experiment['prediction'],
                **({key: deepcopy(experiment[key]) for key in
                    ('control_preparation', 'restore_frequency') if key in experiment}
                   if experiment.get('control_preparation') else {})}}}}
    agent._save_state()
    return True


def execute_reference_response(agent, laboratory, arguments):
    path = arguments['path']
    estimators, state, state_path = _reference_state(agent, laboratory, path)
    if 'experiment' in arguments:
        return _execute_selected_response(agent, laboratory, arguments, state, state_path, estimators)
    if 'binding_path' in arguments:
        return _execute_bound_response(agent, arguments, state, state_path)
    if arguments.get('stage') == 'pending_band_revision':
        from .world_response_hypotheses import revise_pending_response_band

        started = monotonic()
        result = revise_pending_response_band(state)
        agent.costs['python_execution_seconds'] += monotonic() - started
        agent.costs['research_compute_units'] += result['compute_cost']
        trial = state['test']
        question = agent._question({'id': trial['question'] + '_band_revision',
            'question': 'Can a revised shared response interval preserve the pending '
                        'experiment under the retained coefficient programs?',
            'parent_id': trial['question'], 'depends_on': []})
        result['question_id'] = question['id']
        agent._record_answer(question, {'status': 'inconclusive',
            'evidence': result['evidence'], 'answer': deepcopy(result)})
        trial['band_revision'] = result
        state_path.write_text(json.dumps(state, indent=2))
        # The child answer reopens its physical parent. Reanalyse the same paid
        # observation so the controller consumes the revised prediction before
        # choosing another world action, including when no repair was found.
        agent._execute('evaluate', {'path': path, 'question_id': trial['question']}, laboratory)
        return {'operator': 'reference_response', **result}
    number = next((n for n in state['target_numbers'] if n not in state['completed']), None)
    started = monotonic()
    try:
        if arguments.get('stage') == 'frozen_recall':
            from .world_response_transfer import reuse_response_knowledge

            result = reuse_response_knowledge(state, estimators, path)
            if result is None:
                raise ValueError('Frozen recall requires an installed numerical response archive')
            result['frozen_transfer']['research_stage'] = (
                'initial_recall' if not state.get('response_research') else
                'withdrawal_recall' if result['frozen_transfer']['status'] == 'refuted' else
                'continuation_recall')
        else:
            result = propose_response_research(state, estimators, path)
    finally:
        agent.costs['python_execution_seconds'] += monotonic() - started
    agent.costs['research_compute_units'] += result['compute_cost']
    round_number = len(state.get('response_research_history', [])) + 1
    # An authored child may already use the next response-round name. Preserve
    # its actual answer instead of repurposing that question for this comparison.
    existing_ids = {question['id'] for question in agent.questions}
    while f'reference_response_{number}_{round_number}' in existing_ids:
        round_number += 1
    question = agent._question({'id': f'reference_response_{number}_{round_number}',
        'question': 'Which coefficient and shared activation-response hypotheses remain '
                    'compatible with the measured successes and failures?',
        'parent_id': 'task', 'depends_on': []})
    result['question_id'] = question['id']
    from .world_reference_expansion import pending_reference_compositions
    if pending_reference_compositions({**state, 'response_research': result}, estimators):
        result['basis_expansion_action'] = {'action': 'run', 'arguments': {
            'operator': 'reference_expansion', 'path': path}}
    agent._record_answer(question, {'status': 'inconclusive', 'evidence': result['evidence'],
        'answer': {'diagnosis': result['status'], 'assumptions': result['assumptions'],
                   'selected_setting': result['selected_setting'],
                   'selected_number': result.get('selected_number'),
                   **({'joint_parameter_search': result['joint_parameter_search']}
                      if 'joint_parameter_search' in result else {}),
                   **({'family_research': result['family_research']}
                      if 'family_research' in result else {}),
                   **({'conditional_simplification': result['conditional_simplification']}
                      if 'conditional_simplification' in result else {}),
                   **({'basis_expansion_action': result['basis_expansion_action']}
                      if 'basis_expansion_action' in result else {}),
                   **({'frozen_transfer': result['frozen_transfer']}
                      if 'frozen_transfer' in result else {}),
                   'scope': 'Finite members and declared shared response families for a new physical test; no law confirmation'}})
    state['response_research'] = result
    state['family_history_checks'] = deepcopy(result['family_history_checks'])
    state['response_research_inputs'] = _reference_response_inputs(state, estimators)
    state['response_context'] = deepcopy(arguments.get('context'))
    state.setdefault('response_research_history', []).append({
        'question_id': question['id'], 'status': result['status'],
        'response_count': result['response_count'], 'compute_cost': result['compute_cost'],
        'candidate_count': len(result['candidates']),
        'compatible_count': sum(c['status'] == 'compatible' for c in result['candidates']),
        'selected_experiment': (result.get('selection') or {}).get('selected_experiment'),
        **({'joint_parameter_search': result['joint_parameter_search']}
           if 'joint_parameter_search' in result else {}),
        **({'family_research': result['family_research']}
           if 'family_research' in result else {}),
        **({'conditional_simplification': result['conditional_simplification']}
           if 'conditional_simplification' in result else {}),
        **({'basis_expansion_action': result['basis_expansion_action']}
           if 'basis_expansion_action' in result else {}),
        **({'frozen_transfer': result['frozen_transfer']}
           if 'frozen_transfer' in result else {}),
        'hypotheses': result['hypotheses'], 'evidence': result['evidence']})
    state_path.write_text(json.dumps(state, indent=2))
    return {'operator': 'reference_response', **result}


def _compare_bound_response(agent, arguments, state, previous):
    """Compare a draft on paid responses without adopting it as a question."""
    from .world_reference_hypothesis import _declaration
    from .world_response_constraints import query_response_family
    from .world_response_consequences import QUERY_ALLOWANCE

    binding_path = arguments['binding_path']
    binding = agent._program(binding_path, 'bind')
    estimators = [_declaration(item) for item in binding['estimators']]
    if not estimators:
        raise ValueError('bind must supply at least one estimator over its computed features')
    rows = binding['rows']
    available = {row['index'] for row in agent.observations}
    by_number = {}
    for row in rows:
        number, features, evidence = row['crystal'], row['features'], row['evidence']
        if type(number) is not int or number in by_number:
            raise ValueError('bind rows need one row per public crystal number')
        if not evidence or any(type(index) is not int or index not in available for index in evidence):
            raise ValueError('computed features must cite paid observation indices')
        if not features or any(not isfinite(float(value)) for value in features.values()):
            raise ValueError('computed features must be finite numbers')
        by_number[number] = row
    working = deepcopy(state)
    # This archive describes the raw instrument coordinates. An authored
    # binding must establish its own transferable numerical interpretation.
    working.pop('frozen_response_knowledge', None)
    table = working['research_table']
    public_rows = table.get('references', []) + table.get('targets', [])
    required = {row['crystal'] for row in public_rows}
    required.update(row['crystal'] for row in state['response_settings'])
    if set(by_number) != required:
        raise ValueError('bind must return every current calibration-table and paid-response crystal')
    for row in public_rows:
        derived = by_number[row['crystal']]
        # Keep raw readings in the parent's state. These are authored computations
        # with all their input citations, not newly acquired instrument readings.
        row['measurements'] = {name: {'value': float(value), 'evidence': list(derived['evidence']),
            'computed_by': binding_path} for name, value in derived['features'].items()}
    question = arguments['question']
    working['response_research'] = deepcopy(previous)
    # Raw-coordinate history is not this binding's history, even when feature
    # names coincide. Its own preceding comparison retains reusable programs.
    working['response_research_history'] = []
    working['family_history_checks'] = deepcopy(previous.get(
        'family_history_checks', working.get('family_history_checks', {})))
    started = monotonic()
    try:
        # A proposed family gets the same paid-history allowance as an ordinary
        # estimator proposal. Actual charged work bounds its gap enumeration;
        # later consequence queries keep their separate shared allowance.
        features = {row['crystal']: row['features'] for row in rows}
        checks = {e['id']: query_response_family(features, state['response_settings'], e,
            branch_limit=None, compute_limit=QUERY_ALLOWANCE) for e in estimators}
        result = propose_response_research(working, estimators, arguments['path'], paid_checks=checks)
    finally:
        agent.costs['python_execution_seconds'] += monotonic() - started
    agent.costs['research_compute_units'] += result['compute_cost']
    result['evidence'] = sorted(set(result['evidence'])
        | {index for row in rows for index in row['evidence']})

    declaration = {key: deepcopy(question[key]) for key in
                   ('id', 'question', 'parent_id', 'depends_on') if key in question}

    def experiment_action(experiment):
        return {'action': 'run', 'arguments': {'operator': 'reference_response',
            'path': arguments['path'], 'experiment': {
                'number': experiment['number'], 'setting': experiment['prediction'],
                **({key: deepcopy(experiment[key]) for key in
                    ('control_preparation', 'restore_frequency') if key in experiment}
                   if experiment.get('control_preparation') else {})},
            'response_comparison': {'binding_path': binding_path, 'question': deepcopy(declaration)}}}

    # These alternatives were computed under the authored binding. A plain
    # parent replay still has its original features; expose the existing explicit
    # point controller instead of claiming that replay adopts this binding.
    def bind_execution(value):
        if isinstance(value, dict):
            if 'execution_binding' in value and 'number' in value and 'prediction' in value:
                value.pop('procedure', None)
                value['execution_binding'] = 'explicit response experiment from computed representation'
                value['experiment_action'] = experiment_action(value)
            for child in value.values():
                bind_execution(child)
        elif isinstance(value, list):
            for child in value:
                bind_execution(child)

    bind_execution(result)
    selected = (result.get('selection') or {}).get('selected_experiment')
    result.update(binding_path=binding_path, bound_rows=deepcopy(rows), estimators=estimators,
        candidate_additions=deepcopy(binding.get('candidate_additions', [])),
        question_id=question['id'], new_observation_indices=[],
        experiment_action=experiment_action(selected) if selected else None)
    families = result.get('family_research', {}).get('families', [])
    status = ('refuted' if families and all(f['status'] == 'UNSAT_VERIFIED' for f in families)
              else 'inconclusive')
    result['comparison_status'] = status
    old_evidence = set(previous.get('evidence', []))
    result['new_responses'] = [deepcopy(row) for row in state['response_settings']
        if previous and set(row.get('evidence', [])) - old_evidence]
    result['revision_action'] = {'action': 'run', 'arguments': {
        'operator': 'propose_explanations', 'path': binding_path,
        'response_path': arguments['path'], 'question': deepcopy(declaration)}}
    result['premise_design_action'] = response_premise_design_action(result)
    return {'operator': 'reference_response', **result}


def compare_response_proposal(agent, laboratory, arguments, previous):
    """Give the existing proposal writer paid feedback before parent adoption."""
    _, state, _ = _reference_state(agent, laboratory, arguments['path'])
    return _compare_bound_response(agent, arguments, state, previous)


def _execute_bound_response(agent, arguments, state, state_path, physical_test=None):
    binding_path = arguments['binding_path']
    author = next((frame for frame in reversed(agent._inquiry_stack)
        if frame.get('proposal_path') == binding_path
        and frame.get('response_path') == arguments['path']), None)
    saved = state.get('bound_response_research', {})
    # Question labels choose where the answer goes, not whether the same
    # executable representation has already failed to make research progress.
    session = agent.llm_inquiries.get('proposal:' + binding_path, {})
    prior = (author.get('response_comparison') if author else None)
    if prior is None:
        prior = session.get('proposal', {}).get('comparison_result')
    if prior is None:
        prior = next((entry['result'] for entry in reversed(list(saved.values()))
                      if entry['binding_path'] == binding_path), {})
    result = _compare_bound_response(agent, arguments, state, prior)
    if author is not None:
        # An author may practice its associated trial before the parent adopts
        # the draft. Return the paid comparison through the same preview path;
        # a narrow family verdict must not replace the enclosing old answer.
        return_response_to_author(agent, result, physical_test=physical_test)
        agent._save_state()
        return {'operator': 'reference_response', **result}
    question = agent._question(arguments['question'])
    agent._record_answer(question, {'status': result['comparison_status'], 'evidence': result['evidence'], 'answer': {
        'diagnosis': result['status'], 'binding_path': binding_path,
        'family_research': deepcopy(result.get('family_research')),
        'experiment_action': deepcopy(result['experiment_action']),
        'scope': 'Authored representation and shared connected band on paid responses; not fresh confirmation'}})
    saved = state.setdefault('bound_response_research', saved)
    saved.pop(question['id'], None)
    saved[question['id']] = {'binding_path': binding_path, 'result': deepcopy(result)}
    state_path.write_text(json.dumps(state, indent=2))
    return_response_to_author(agent, result, physical_test=physical_test)
    agent._save_state()
    return {'operator': 'reference_response', **result}



def return_response_to_author(agent, comparison, physical_test=None):
    """Return paid evidence to the author of this exact binding, without a model call."""
    session = agent.llm_inquiries.get('proposal:' + comparison['binding_path'])
    if session is None:
        return
    feedback = {'response_comparison': deepcopy(comparison)}
    if physical_test is not None:
        feedback['physical_test'] = deepcopy(physical_test)
    session['events'].append({'action': 'response_return', 'arguments': {},
        'trigger_origin': 'paid_response_return', 'feedback': feedback})
    session.setdefault('proposal', {}).update(comparison_result=deepcopy(comparison),
        experiment_action=deepcopy(comparison['experiment_action']),
        revision_action=deepcopy(comparison['revision_action']),
        premise_design_action=response_premise_design_action(comparison))
    for frame in agent._inquiry_stack:
        if (frame.get('proposal_path') == comparison['binding_path']
                and frame.get('response_path') == comparison['revision_action']['arguments']['response_path']):
            frame['response_comparison'] = deepcopy(comparison)


def _execute_selected_response(agent, laboratory, arguments, state, state_path, estimators):
    """Let an author choose the scientific setpoint, using the existing controller."""
    condition_proposal = arguments.get('condition_proposal')
    binding_origin = arguments.get('response_comparison', {}).get('binding_path')
    bound_condition = (agent.llm_inquiries.get('proposal:' + binding_origin, {})
                       .get('proposal', {}).get('condition_proposal')) if binding_origin else None
    if not condition_proposal and bound_condition:
        # The source binding owns this association after its author returns;
        # an unrelated current inquiry/advice must not sever empirical feedback.
        condition_proposal = bound_condition
    inquiry_proposal = next((frame['condition_proposal'] for frame in reversed(agent._inquiry_stack)
                            if frame.get('condition_proposal')), None)
    if not condition_proposal and inquiry_proposal:
        inquiry_conditions = agent.llm_inquiries['proposal:' + inquiry_proposal]['condition_research']
        if inquiry_conditions['inputs']['controller_path'] == arguments['path']:
            condition_proposal = inquiry_proposal
    advice = agent.research_advice or {}
    current_conditions = agent.llm_inquiries.get('proposal:' + advice.get('proposal_path', ''), {})
    controller = current_conditions.get('condition_research', {}).get('inputs', {}).get('controller_path')
    if (not condition_proposal and advice.get('proposal_path')
            and controller == arguments['path']):
        # A parent can choose a different point using the same current research
        # and controller. Its outcome still tests those frozen conditions.
        condition_proposal = advice['proposal_path']
    experiment = arguments['experiment']
    number, setting = experiment['number'], experiment['setting']
    if type(number) is not int or not isinstance(setting, (int, float)) or not isfinite(setting):
        raise ValueError('experiment needs a public crystal number and a finite setting')
    _, crystals, reactors, _, _ = _scan(agent.observations, laboratory.public_observation(), estimators)
    if str(number) not in crystals or str(number) not in reactors:
        raise ValueError('Read the selected crystal and reactor number through public observations first')
    pending = state.get('test')
    pending_question = next((q for q in agent.questions if pending and q['id'] == pending['question']), {})
    unfinished = pending and pending_question.get('status') not in ('supported', 'refuted') and not (
        pending.get('tuning_finished') and pending_question.get('status') == 'inconclusive')
    resuming = unfinished and pending.get('authored_experiment') and (
        pending['number'] == number and pending['prediction'] == setting)
    preparation_source = experiment if experiment.get('control_preparation') else (pending if resuming else {})
    grid = state.get('control_grids', {}).get(str(number))
    base = {'operator': 'reference_response', 'path': arguments['path'],
            'selected_number': number, 'selected_setting': setting}
    if not grid:
        return {**base, 'status': 'control_grid_unobserved',
                'reason': 'Open this reactor menu to observe its available increments and bounds'}
    preparation = preparation_source.get('control_preparation')
    if preparation:
        # This is the proposed lattice after a physical boundary operation.
        # The controller still observes and tests that operation before tuning.
        grid = preparation['grid']
    position = (setting - grid['origin']) / grid['step']
    if not grid['lower'] <= setting <= grid['upper'] or not isclose(position, round(position), abs_tol=1e-6):
        return {**base, 'status': 'unreachable_setting', 'control_grid': grid,
                'reason': 'Requested setting is outside this observed control grid; choose a reachable setting',
                'new_observation_indices': []}
    origin = arguments.get('response_comparison') or (
        pending.get('response_comparison') if resuming else None)
    superseded = None
    if unfinished and not resuming:
        paid = [{**row, 'evidence': [i for i in row['evidence'] if i >= pending['started']]}
                for row in state.get('response_settings', []) if row['crystal'] == pending['number']
                and any(i >= pending['started'] for i in row['evidence'])]
        superseded = {**deepcopy(pending), 'status': 'inconclusive',
                      'superseded_by': {'number': number, 'setting': setting}}
        agent._record_answer(pending_question, {'status': 'inconclusive',
            'evidence': sorted({i for row in paid for i in row['evidence']}),
            'answer': {'requested_setting': pending['prediction'], 'observed_responses': paid,
                       'reason': 'Researcher selected a new experiment before this trial finished'}})
        state.setdefault('tests', []).append(superseded)
        state.pop('test')
        if pending.get('control_preparation'):
            state['prepared_recovery'] = _prepared_reference_recovery(state, pending)
        elif 'restore_frequency' in pending:
            state['recovery'] = {'number': pending['number'], 'frequency': pending['restore_frequency'],
                'crystal': pending['crystal'], 'reactor': pending['reactor'],
                'response_question': pending.get('response_question', pending['question'])}
    first = len(agent.observations)
    parent_id = agent._inquiry_stack[-1]['question_id'] if agent._inquiry_stack else 'task'
    declared = arguments.get('question') or arguments.get('research_question')
    comparison_question = origin.get('question', {}) if origin else {}
    condition_author = agent.llm_inquiries.get('proposal:' + (condition_proposal or ''), {})
    returned_research = declared and (declared['id'] in agent.llm_inquiries
        or condition_author.get('question_id') == declared['id'])
    if declared and (declared['id'] == comparison_question.get('id') or returned_research
            or (declared['id'] == parent_id and (
            not arguments.get('question') or declared['id'] in (
                comparison_question.get('id'), comparison_question.get('parent_id'))))):
        # The provider carries its active research assignment on every action.
        # It may explicitly repeat the computed proposal's enclosing question.
        # This setting tests a child prediction; it cannot answer that enclosing
        # explanatory question merely by activating or failing to activate.
        # Returned inquiries retain that role after control reaches the task loop.
        parent_id = declared['id']
        declared = None
    if resuming:
        qid = pending['question']
        declared = {key: deepcopy(pending_question[key]) for key in
                    ('id', 'question', 'parent_id', 'depends_on') if key in pending_question}
    else:
        qid = declared['id'] if declared else f'reference_authored_{number}_{first}'
    if declared:
        parent_id = declared.get('parent_id', parent_id)
    state_path.write_text(json.dumps(state, indent=2))
    recoveries_before = len(state.get('setting_recoveries', []))
    execution = agent._run_procedure(laboratory, arguments['path'],
        max_steps=arguments.get('max_steps', 128), query_arguments={'response_experiment': {
            'number': number, 'setting': setting, 'question_id': qid, 'parent_id': parent_id,
            **({key: deepcopy(preparation_source[key]) for key in
                ('control_preparation', 'restore_frequency') if key in preparation_source}
               if preparation else {}),
            **({'research_question': deepcopy(declared)} if declared else {}),
            **({'response_comparison': deepcopy(origin)} if origin else {})}})
    # Retained controller sources may still use their generic question wording.
    # Keep the author's scientific declaration and the acquired answer together.
    if declared:
        agent._question(declared)
    state = json.loads(state_path.read_text())
    refresh_response_evidence(state, agent.observations, laboratory.public_observation(), estimators)
    if origin and state.get('test', {}).get('question') == qid:
        state['test']['response_comparison'] = deepcopy(origin)
    state_path.write_text(json.dumps(state, indent=2))
    question = next((q for q in agent.questions if q['id'] == qid), {})
    answer = question.get('answer')
    measured = [{**row, 'evidence': [i for i in row['evidence'] if i >= first]}
                for row in state.get('response_settings', []) if row['crystal'] == number
                and any(i >= first for i in row['evidence'])]
    result = {**base, 'status': execution['status'], 'question_id': qid,
            'prediction_status': question.get('status', 'unmeasured'),
            'answer': deepcopy(answer), 'observed_responses': measured,
            'superseded_experiment': superseded,
            'requested_setting_reached': any(isclose(row['frequency'], setting, abs_tol=1e-6) for row in measured),
            'new_observation_indices': [row['index'] for row in agent.observations[first:]],
            'setting_recoveries': state.get('setting_recoveries', [])[recoveries_before:],
            'scope': 'Actual response to an authored setting, not confirmation of an explanatory law'}

    if origin and result['requested_setting_reached']:
        result['comparison_action'] = {'action': 'run', 'arguments': {
            'operator': 'reference_response', 'path': arguments['path'], **deepcopy(origin)}}
        if agent._procedure_budget_status():
            result['response_comparison_pending'] = deepcopy(origin)
            return result
        comparison = _execute_bound_response(agent, {'path': arguments['path'], **origin},
                                             state, state_path, physical_test=result)
        result.update(response_comparison=comparison, revision_action=comparison['revision_action'])
        agent._save_state()
    if condition_proposal:
        from .world_llm_inquiry import refresh_condition_proposal
        proposal_path = condition_proposal
        # Both parent and child condition trials return to the representation
        # that supplied their coordinates. Reuse a same-source comparison
        # already performed above; drafting is not required just to recompare.
        packet = response_premise_proposal_inputs(agent, proposal_path)
        if packet and result['new_observation_indices']:
            source = packet['binding_path']
            prior = agent.llm_inquiries['proposal:' + source]['proposal']['comparison_result']
            if (result.get('response_comparison') or {}).get('binding_path') != source:
                declaration = deepcopy(packet['revision_action']['arguments']['question'])
                if agent._procedure_budget_status():
                    result['response_comparison_pending'] = {
                        'binding_path': source, 'question': declaration}
                else:
                    compared = compare_response_proposal(agent, laboratory, {
                        'path': arguments['path'], 'binding_path': source,
                        'question': declaration}, prior)
                    return_response_to_author(agent, compared, physical_test={
                        key: deepcopy(result[key]) for key in (
                            'selected_number', 'selected_setting', 'requested_setting_reached',
                            'observed_responses', 'new_observation_indices')})
                    result['response_comparison'] = compared
        conditions = refresh_condition_proposal(agent, proposal_path)
        if conditions is not None:
            result['condition_proposal_origin'] = ('explicit' if arguments.get('condition_proposal')
                else 'condition_research_inquiry' if condition_proposal == inquiry_proposal
                else 'response_binding' if condition_proposal == bound_condition
                else 'current_research_advice')
            result['condition_research'] = conditions
            session = agent.llm_inquiries['proposal:' + proposal_path]
            session['events'].append({'action': 'condition_return', 'arguments': {},
                'feedback': {'condition_research': deepcopy(conditions),
                    'physical_test': {key: deepcopy(result[key]) for key in (
                        'selected_number', 'selected_setting', 'requested_setting_reached',
                        'observed_responses', 'new_observation_indices')}}})
            agent._save_state()
    return result
