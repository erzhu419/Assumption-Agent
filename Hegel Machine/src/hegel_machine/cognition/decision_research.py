"""Continue a saved decision question using new trials and frozen plans."""

from copy import deepcopy
from dataclasses import replace
from statistics import mean
from time import perf_counter

from .decision_evidence import begin_decision_evidence, extend_decision_evidence
from .decision_frontier import decision_evidence_frontier
from .decision_probing import plan_decision_probes
from .evaluation import loss
from .language import Program
from .models import ActionRecord, EpisodeResult, SearchBudget
from .selection import representative_probe
from .uncertainty import action_and_interval, decision_loss
from .workspace import ResearchWorkspace, candidate_from_dict


def decision_quote(workspace):
    """The minimum complete visit: one representative pair and its reporting."""
    frontier = workspace.decision_frontier
    new_rows = frontier['observation_cost']
    reference_rows = len(workspace.reference) + new_rows
    return {'observation_cost': new_rows,
            'compute_cost': frontier['comparison_compute_cost'] + 2 * reference_rows + 2 * new_rows}


def _representative_pair(spec, policies, lengths, *, seed, index, used):
    probes = []
    for offset, length in enumerate(lengths):
        probe = representative_probe(spec, seed=seed, index=index + offset, remaining=length,
            used=[*used, *probes], candidates=[item[0] for item in policies], adaptive_horizon=True)
        if probe is None:
            return []
        probes.append(probe)
    return probes


def _additional_decision_pair(spec, policies, lengths, *, seed, round_number, used,
                              observation_remaining, compute_remaining,
                              old_reference_rows, new_reference_rows):
    """Offer one further pair while funding its comparison and final reporting."""
    rows = sum(lengths)
    comparison = 2 * rows * (len(policies) - 1)
    reporting = 2 * (old_reference_rows + new_reference_rows + rows) + 2 * (new_reference_rows + rows)
    if rows > observation_remaining or comparison + reporting > compute_remaining:
        return []
    return _representative_pair(spec, policies, lengths, seed=seed,
        index=round_number + 4100, used=used)


def _pending_frontier(spec, workspace, config, previous, pending):
    """Each saved challenger keeps its own frozen model and paid trial list."""
    parts = [decision_evidence_frontier(spec, workspace, config, previous, candidate, evidence)
             for candidate, evidence in pending]
    frontier = deepcopy(parts[0])
    length = max(max(part['probe_lengths']) for part in parts)
    frontier['policies'] = [frontier['policies'][0], *[part['policies'][1] for part in parts]]
    frontier['evidence_states'] = [part['evidence_states'][0] for part in parts]
    frontier['design']['alternatives'] = []
    for part in parts:
        for alternative in part['design']['alternatives']:
            if alternative not in frontier['design']['alternatives']:
                frontier['design']['alternatives'].append(alternative)
    frontier['design']['comparisons'] = [{'policy_index': index, **deepcopy(evidence['comparison'])}
                                        for index, (_, evidence) in enumerate(pending, 1)]
    frontier.update(probe_lengths=[length, length], observation_cost=2 * length,
        comparison_compute_cost=4 * length * len(pending), compute_cost=4 * length * len(pending))
    return frontier


def run_decision_research(environment, config, library, *, research_state):
    """Fund new evidence; preserve unresolved choices for a subsequent visit.

    A first question can include a history intervention before the representative
    pair. Interventions enter learning observations, never the deployment gain.
    Existing paired evidence resumes without re-fitting or re-scoring its rows.
    A still conflicting first batch can buy at most one more pair in this visit.
    """
    started = perf_counter()
    workspace = ResearchWorkspace.from_dict(research_state)
    frontier = workspace.decision_frontier
    quote = decision_quote(workspace)
    if quote['observation_cost'] > config.observation_budget or quote['compute_cost'] > config.compute_budget:
        raise ValueError('decision question needs its complete observation and comparison budget')
    policies = [(candidate_from_dict(item['candidate']), deepcopy(item['model']))
                for item in frontier['policies']]
    previous, previous_model = policies[0]
    spec, fallback = environment.spec, workspace.fallback
    saved = frontier.get('evidence_states')
    if saved:
        previous = replace(previous, program=Program.from_dict(saved[0]['previous_program']))
        previous_model = deepcopy(saved[0]['previous_model'])
        entries = [(replace(candidate, program=Program.from_dict(evidence['candidate_program'])), deepcopy(evidence))
                   for (candidate, _), evidence in zip(policies[1:], saved)]
    else:
        entries = [(candidate, begin_decision_evidence(previous.program, previous_model, candidate.program, model,
                       fallback=fallback, costs=spec.decision_costs, minimum_improvement=config.minimum_improvement))
                   for candidate, model in policies[1:]]
    policies = [(previous, previous_model), *[(candidate, evidence['candidate_model'])
                                             for candidate, evidence in entries]]
    work = SearchBudget(config.compute_budget)
    probes, reference_rows, actions = [], [], []
    old_reference_rows = len(workspace.reference)
    observation_cost = 0

    def acquire(batch, *, targeted=False):
        nonlocal observation_cost
        rows = []
        for probe in batch:
            measured = environment.observe(probe)
            rows.extend(measured)
            workspace.observations.extend(measured)
            probes.append(probe)
            observation_cost += probe.cost
            actions.append(ActionRecord(workspace.rounds_completed, 'observe',
                'Distinguish frozen explanations using different prepared histories' if targeted else
                'Compare the frozen complete decisions on a new representative trajectory',
                observation_cost=probe.cost, details={'probe_id': probe.probe_id,
                    'sequence': list(probe.sequence), 'role': 'targeted' if targeted else 'representative'}))
        if not targeted:
            reference_rows.extend(rows)
        return rows

    def compare(batch, indexed_entries):
        for index, (candidate, evidence) in indexed_entries:
            evidence = extend_decision_evidence(evidence, batch, budget=work)
            entries[index] = candidate, evidence
            actions.append(ActionRecord(workspace.rounds_completed, 'compare_decision_question',
                'Judge a saved complete decision using accumulated paired trajectories',
                compute_cost=evidence['last_compute_cost'], details={**deepcopy(evidence['comparison']),
                    'candidate_id': candidate.candidate_id, 'candidate_program': candidate.program.to_dict(),
                    'evidence_status': evidence['status'], 'last_compute_cost': evidence['last_compute_cost']}))

    package = {}
    earlier = list(workspace.observations)
    experiment_candidates = list({candidate.program: candidate for candidate, _ in policies}.values())[:6]
    if not saved:
        planning = SearchBudget(min(500, max(0, work.remaining - quote['compute_cost'])))
        package = plan_decision_probes(spec, policies, workspace.observations, fallback, planning,
            remaining=config.observation_budget, round_number=workspace.rounds_completed,
            seed=config.seed, used=workspace.queried)
        work.spend(planning.spent)
        if package:
            count = sum(probe.cost for probe in package['reference_probes'])
            needed = package['comparison_compute_cost'] + 2 * (old_reference_rows + count) + 2 * count
            if config.use_memory:
                needed += sum(len(experiment_candidates) * (min(8, len(earlier)) + min(8, probe.cost))
                              for probe in package['targeted_probes'])
            if package['observation_cost'] > config.observation_budget or needed > work.remaining:
                package = {}
        if planning.spent or package:
            actions.append(ActionRecord(workspace.rounds_completed, 'plan_decision_probe_package',
                'Price a history intervention and subsequent representative comparison',
                compute_cost=planning.spent, details={'selected': bool(package),
                    'design': package.get('design', {}), 'observation_cost': package.get('observation_cost', 0)}))
    if package:
        for probe in package['targeted_probes']:
            measured = acquire([probe], targeted=True)
            if config.use_memory:
                before = work.spent
                learned = library.experiments.record(spec, experiment_candidates, earlier, probe, measured,
                    fallback=fallback, budget=work, design=package['design'])
                actions.append(ActionRecord(workspace.rounds_completed,
                    'learn_experiment' if learned else 'measure_decision_intervention',
                    'Measure whether the acquired history contrast distinguishes the frozen explanations',
                    compute_cost=work.spent-before,
                    details={'probe_id': probe.probe_id, 'learned': bool(learned), 'experiment': learned}))
        first_pair = package['reference_probes']
    else:
        first_pair = _representative_pair(spec, policies, frontier['probe_lengths'], seed=config.seed,
            index=workspace.rounds_completed + 4000, used=workspace.queried)
    if not first_pair:
        raise ValueError('decision question requires two available representative trajectories')
    compare(acquire(first_pair), list(enumerate(entries)))

    def supported():
        return [(evidence['comparison']['after_loss'], candidate, evidence['candidate_model'])
                for candidate, evidence in entries if evidence['status'] == 'adopted']

    unresolved = [(index, entry) for index, entry in enumerate(entries) if entry[1]['status'] == 'continue']
    if not supported() and unresolved:
        remaining_policies = [(previous, previous_model), *[(candidate, evidence['candidate_model'])
                                      for _, (candidate, evidence) in unresolved]]
        extra = _additional_decision_pair(spec, remaining_policies, frontier['probe_lengths'], seed=config.seed,
            round_number=workspace.rounds_completed, used=[*workspace.queried, *probes],
            observation_remaining=config.observation_budget-observation_cost, compute_remaining=work.remaining,
            old_reference_rows=old_reference_rows, new_reference_rows=len(reference_rows))
        if extra:
            actions.append(ActionRecord(workspace.rounds_completed, 'extend_decision_question',
                'Resolve conflicting trajectory evidence with one further funded pair',
                details={'probe_ids': [probe.probe_id for probe in extra],
                         'observation_cost': sum(probe.cost for probe in extra)}))
            compare(acquire(extra), unresolved)

    eligible = supported()
    selected, model = (min(eligible, key=lambda item: item[0])[1:] if eligible else (previous, previous_model))
    workspace.reference.extend(reference_rows)
    workspace.queried.extend(probes)
    workspace.incumbent, workspace.uncertainty_model = selected, deepcopy(model)
    if selected.program != previous.program:
        workspace.parent = previous
    pending = [(candidate, evidence) for candidate, evidence in entries if evidence['status'] == 'continue']
    complete = bool(eligible) or not pending
    workspace.decision_frontier = ({} if complete else
        _pending_frontier(spec, workspace, config, previous, pending))
    if complete:
        workspace.research_intent = {}
    workspace.rounds_completed += 1
    work.spend(2 * len(workspace.reference))
    final_loss = loss(selected.program, workspace.reference, fallback, spec.risk_quantile)
    baseline_loss = loss(Program('const', value=fallback), workspace.reference, fallback, spec.risk_quantile)
    work.spend(2 * len(reference_rows))

    def actual_loss(candidate, decision_model):
        return mean(decision_loss(action_and_interval(candidate.program, row, fallback,
                    decision_model, spec.decision_costs)[0], row.target, spec.decision_costs) for row in reference_rows)

    before, after = actual_loss(previous, previous_model), actual_loss(selected, model)
    details = {'before_loss': before, 'after_loss': after, 'gain': before-after,
        'normalized_gain': (before-after) / max(workspace.initial_loss, 1e-8),
        'reference_rows': len(reference_rows), 'program_changed': selected.program != previous.program,
        'observation_cost': observation_cost, 'compute_cost': work.spent,
        'execution_complete': complete, 'evidence': 'new_trajectories_for_frozen_complete_decisions'}
    actions.append(ActionRecord(workspace.rounds_completed, 'execute_decision_question',
        frontier['question'], details={'selected_candidate': selected.candidate_id,
            'adopted': bool(eligible), 'policies_compared': len(policies), 'execution_complete': complete,
            'probe_ids': [probe.probe_id for probe in probes]}))
    actions.append(ActionRecord(workspace.rounds_completed, 'continuation_feedback',
        'Measure the complete decision return of the short research visit',
        compute_cost=2 * len(reference_rows), details=details))
    added, events = [], []
    if config.use_memory and selected.program != previous.program and final_loss + config.minimum_improvement < baseline_loss:
        if selected.memory_id and selected.operation == 'reuse':
            library.record_use(selected.memory_id, spec.task_id, spec=spec, observations=workspace.observations)
        else:
            identifier, created = library.learn(selected, spec, observations=workspace.observations)
            if created:
                added.append(identifier)
                events.append({'action': 'learn', 'concept_id': identifier, 'expression': selected.program.describe()})
    return EpisodeResult(spec.task_id, selected.program, fallback, workspace.observations,
        actions, [], observation_cost, work.spent, final_loss, workspace.initial_loss,
        selected.operation, selected.memory_id, added, events,
        'decision_question_resolved' if complete else 'decision_question_pending',
        wall_time_seconds=perf_counter()-started, uncertainty_model=model,
        validation_rows=len(workspace.reference), research_state=workspace.to_dict())
