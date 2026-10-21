"""Representative experiments and prospective comparison of executable models."""

from dataclasses import dataclass, replace
from random import Random
from statistics import mean

from .branch_support import history_branch_support, history_frontier_key
from .evaluation import loss, point_loss, predict
from .language import EvaluationContext, Program
from .models import Candidate, Observation, Probe, SearchBudget
from .synthesis import synthesize, _fit_linear


def representative_probe(spec, *, seed, index, remaining, used=(), candidates=(), adaptive_horizon=True):
    """A stratified, independently permuted control sequence, with no labels."""
    if not spec.feature_bounds:
        used_ids = {probe.probe_id for probe in used}
        available = [probe for probe in spec.available_probes
                     if probe.probe_id not in used_ids and probe.cost <= remaining]
        return available[index % len(available)] if available else None
    depth = max((history_branch_support(candidate.program, ())["required_depth"]
                 for candidate in candidates), default=0)
    desired = max(8, 4 * depth + 4) if adaptive_horizon else 8
    length = min(desired, spec.max_probe_length, remaining)
    if length < 3:
        return None
    rng = Random(seed + 700_001 + index * 1777)
    columns = {}
    for name in spec.feature_bounds:
        low, high = spec.feature_bounds[name]
        values = [low + (high - low) * (i + rng.random()) / length for i in range(length)]
        rng.shuffle(values)
        columns[name] = values
    return Probe(f"representative_{index}", tuple(
        {name: columns[name][i] for name in spec.feature_bounds} for i in range(length)))


@dataclass
class AssessedCandidate:
    candidate: Candidate
    score: float
    reference_loss: float | None
    evidence_loss: float
    fresh_loss: float | None = None
    origins: tuple[Candidate, ...] = ()
    assessment_scope: dict | None = None


def rank_candidates(candidates, observations, reference, *, fallback, q, budget,
                    complexity_weight, fresh=(), fresh_weight=0.5):
    """Screen on evidence, then compare on representative and new observations.

    Reference queries cover the public control domain. Targeted queries remain
    useful for fitting and falsification without defining the task distribution.
    """
    ranked = []
    programs = {}
    evidence_contexts = [EvaluationContext(row) for row in observations]
    reference_contexts = [EvaluationContext(row) for row in reference]
    fresh_contexts = [EvaluationContext(row) for row in fresh]
    for candidate in candidates:
        origins = programs.setdefault(candidate.program, [])
        if all(origin.candidate_id != candidate.candidate_id for origin in origins):
            origins.append(candidate)
    for origins in programs.values():
        candidate = min(origins, key=lambda item: (item.operation != "reuse", item.memory_id is None))
        units = len(observations) + len(reference) + len(fresh)
        if not budget.spend(units):
            break
        empirical = loss(candidate.program, observations, fallback, q, contexts=evidence_contexts)
        representative = loss(candidate.program, reference, fallback, q, contexts=reference_contexts) if reference else empirical
        prospective = loss(candidate.program, fresh, fallback, q, contexts=fresh_contexts) if fresh else None
        score = 0.8 * representative + 0.2 * empirical
        if prospective is not None:
            score = (1.0 - fresh_weight) * score + fresh_weight * prospective
        score += complexity_weight * candidate.program.size
        ranked.append(AssessedCandidate(candidate, score, representative, empirical, prospective, tuple(origins)))
    return sorted(ranked, key=lambda row: (row.score, row.candidate.program.size))


def _unrefuted(row):
    # Fits of the same startup relation differ around 1e-20 after parameter
    # estimation. That numerical residue is not a counterexample to the
    # unexecuted mature mechanism. Acquired prediction errors remain errors.
    return all(value is None or value <= 1e-12
               for value in (row.evidence_loss, row.reference_loss, row.fresh_loss))


def _equally_supported(row, best):
    return all(value is None or baseline is None or value <= baseline + 1e-12
               for value, baseline in zip(
                   (row.evidence_loss, row.reference_loss, row.fresh_loss),
                   (best.evidence_loss, best.reference_loss, best.fresh_loss)))


def retain_history_frontier(ranked, observations, *, maximum):
    """Keep at most two unrefuted historical alternatives beside the best fit.

    ``ranked`` must already be assessed on all acquired comparison evidence.
    The reserved alternatives influence the next actual experiment's horizon;
    they do not become the selected explanation just by needing more history.
    This is also used when pruning the bank after a paid comparison.
    """
    if not ranked or maximum <= 0:
        return [], {"history_frontier": []}
    selected = [ranked[0]]
    frontier, structures = [], set()
    for row in ranked if maximum > 1 else ():
        if not _unrefuted(row):
            continue
        support = history_branch_support(row.candidate.program, observations)
        key = history_frontier_key(support)
        if not key or key in structures:
            continue
        if row not in selected and len(selected) >= maximum:
            break
        if row not in selected:
            selected.append(row)
        structures.add(key)
        frontier.append({"candidate_id": row.candidate.candidate_id,
                         "memory_id": row.candidate.memory_id, **support})
        if len(frontier) >= min(2, maximum - 1):
            break
    # Keep the existing coordinate diversity after the bounded historical
    # opportunity. Tiny coefficient-fitting residuals do not break a tie.
    footprints = {frozenset(row.candidate.program.observed_variables()) for row in selected}
    for row in ranked[1:]:
        if len(selected) >= maximum:
            break
        footprint = frozenset(row.candidate.program.observed_variables())
        if _equally_supported(row, ranked[0]) and footprint not in footprints:
            selected.append(row)
            footprints.add(footprint)
    for row in ranked[1:]:
        if len(selected) >= maximum:
            break
        if row not in selected:
            selected.append(row)
    return selected, {"history_frontier": frontier}


def retain_research_comparison_pool(assessed, public_predictions, *, maximum):
    """Retain research alternatives without changing task ranking or adoption.

    ``public_predictions`` maps expanded Programs to complete prediction tuples
    on the same declared public panel. Missing entries and None are UNKNOWN.
    The caller has already charged these forecasts and the evidence assessment
    to its existing budget; this function never evaluates a Program. A panel
    signature is finite behavior, not a claim of global program equivalence.
    """
    by_program = {}
    for row in assessed:
        by_program.setdefault(row.candidate.program.expanded(), []).append(row)
    positions = {id(row): index for index, row in enumerate(assessed)}
    evidence_key = lambda row: (row.evidence_loss, row.candidate.program.size, positions[id(row)])
    unique, clones = [], []
    for program, rows in by_program.items():
        representative = min(rows, key=evidence_key)
        origins = []
        for row in rows:
            for origin in row.origins or (row.candidate,):
                if all(prior.candidate_id != origin.candidate_id for prior in origins):
                    origins.append(origin)
        retained = replace(representative, origins=tuple(origins))
        positions[id(retained)] = positions[id(representative)]
        unique.append(retained)
        if len(rows) > 1 or len(origins) > 1:
            clones.append({'representative_id': retained.candidate.candidate_id,
                           'origin_ids': [origin.candidate_id for origin in origins]})
    groups, unknown = {}, []
    for row in unique:
        predictions = public_predictions.get(row.candidate.program.expanded())
        if predictions is None:
            unknown.append(row.candidate.candidate_id)
        else:
            groups.setdefault(tuple(predictions), []).append(row)
    selected, reasons, covered = [], {}, set()

    def retain(row, reason):
        if row in selected:
            reasons[row.candidate.candidate_id].append(reason)
        elif len(selected) < maximum:
            selected.append(row)
            reasons[row.candidate.candidate_id] = [reason]
            signature = public_predictions.get(row.candidate.program.expanded())
            if signature is not None:
                covered.add(tuple(signature))

    if unique and maximum > 0:
        retain(min(unique, key=evidence_key), 'minimum_evidence_loss')
        retain(min(unique, key=lambda row: (
            row.candidate.program.size, row.evidence_loss, positions[id(row)])), 'minimum_complexity')
        representatives = [(signature, min(rows, key=evidence_key)) for signature, rows in groups.items()]
        for signature, row in sorted(representatives, key=lambda item: evidence_key(item[1])):
            if signature not in covered:
                retain(row, 'different_public_behavior')
        for row in unique:
            retain(row, 'task_rank_fill')
    details = {'pool_kind': 'research_comparison_only', 'maximum': maximum,
        'input_candidate_count': len(assessed), 'unique_program_count': len(unique),
        'expanded_clone_groups': clones, 'unknown_prediction_ids': unknown,
        'public_behavior_groups': [{'candidate_ids': [row.candidate.candidate_id for row in rows],
            'representative_id': min(rows, key=evidence_key).candidate.candidate_id,
            'retained': signature in covered} for signature, rows in groups.items()],
        'retained': [{'candidate_id': row.candidate.candidate_id,
            'reasons': reasons[row.candidate.candidate_id],
            'evidence_loss': row.evidence_loss, 'complexity': row.candidate.program.size,
            'assessment_scope': row.assessment_scope} for row in selected],
        'prediction_compute_cost': 0, 'task_adoption_evaluated': False}
    return selected, details


def _current_terms(program):
    if program.op == "call" and program.history_depth:
        return _current_terms(program.expanded())
    if program.op == "add":
        return [term for child in program.children for term in _current_terms(child)]
    if program.op == "mul" and any(child.op == "const" for child in program.children):
        other = next((child for child in program.children if child.op != "const"), None)
        return _current_terms(other) if other is not None else []
    return [program] if program.op != "const" and program.history_depth == 0 else []


def warm_start_candidates(candidates, observations, *, budget, q, complexity_weight):
    """Fit every historical representation's current coordinates before extras.

    Basic joint fits take turns across depths. More general startup search uses
    only the remainder, so shallow candidates cannot consume the fitting budget
    of every deeper representation before its first attempt.
    """
    by_depth, rows_by_depth, keys = {}, {}, {}
    for candidate in candidates:
        depth = candidate.program.minimum_history_depth
        if not depth:
            continue
        rows = rows_by_depth.setdefault(depth, [row for row in observations if len(row.history) < depth])
        terms = tuple(sorted(set(_current_terms(candidate.program)), key=lambda term: term.describe()))
        keys[candidate.candidate_id] = (depth, terms)
        if terms and len(rows) >= len(terms) + 1:
            bucket = by_depth.setdefault(depth, [])
            if terms not in bucket:
                bucket.append(terms)
    for bucket in by_depth.values():
        bucket.sort(key=lambda terms: (sum(term.size for term in terms), -len(terms)))
    core = SearchBudget(budget.remaining * 2 // 3)
    fitted = {}
    while any(by_depth.values()):
        for depth in sorted(by_depth):
            if not by_depth[depth]:
                continue
            terms = by_depth[depth].pop(0)
            rows = rows_by_depth[depth]
            if core.spend(len(rows)):
                fitted[(depth, terms)] = _fit_linear(terms, rows)
    budget.spend(core.spent)
    startup_models = {}
    depths = sorted(rows_by_depth)
    for index, depth in enumerate(depths):
        rows = rows_by_depth[depth]
        if len(rows) < 3:
            startup_models[depth] = []
            continue
        current_rows = [Observation(row.features, row.target) for row in rows]
        quota = budget.remaining // (len(depths) - index)
        local = SearchBudget(min(quota // 2, 240))
        generated = synthesize(current_rows, operations=("refit", "composition", "relation"),
                               risk_quantile=q, budget=local)
        budget.spend(local.spent)
        measuring = SearchBudget(min(quota - local.spent, budget.remaining))
        scored = rank_candidates(generated[:12], current_rows, [], fallback=mean(row.target for row in rows),
                                 q=q, budget=measuring, complexity_weight=complexity_weight)
        budget.spend(measuring.spent)
        startup_models[depth] = [row.candidate.program for row in scored[:2]]
    results = []
    for candidate in candidates:
        if candidate.candidate_id not in keys:
            continue
        key = keys[candidate.candidate_id]
        depth = key[0]
        starters = [*([fitted[key]] if key in fitted else []), *startup_models[depth]]
        for startup in dict.fromkeys(starters):
            program = Program("if", children=(Program("available", lag=depth), candidate.program, startup))
            results.append(Candidate(f"startup:{candidate.candidate_id}", program,
                                     "abstract_reuse" if candidate.operation == "reuse" else candidate.operation,
                                     "Learn the initial regime and retain the mature historical representation",
                                     candidate.parents, candidate.memory_id))
    return results


def screen_candidates(candidates, observations, reference, *, fallback, q, budget,
                      complexity_weight, maximum, incumbent=None, assessment_sink=None):
    """Screen a broad pool before the controller measures its shortlist fully."""
    total = len({candidate.program for candidate in candidates})
    unresolved = {candidate.program for candidate in candidates
                  if history_branch_support(candidate.program, observations)["unresolved"]}
    # A sampled fit is insufficient to promote a historical branch. Reserve
    # complete checks for only the two branches that could gain a slot, while
    # leaving at least one ordinary observation comparison per candidate.
    validation_reserve = min(2, len(unresolved), max(0, maximum - 1)) * len(observations)
    if budget.remaining < total + validation_reserve:
        validation_reserve = 0
    witnesses = []
    original_count = max(1, min(12, len(observations), budget.remaining // max(1, total)))
    # A rare acquired contradiction must not disappear behind a cheaper model
    # on the random screening sample. Keep the ordinary candidate comparisons
    # affordable; small pools already seeing every row need no extra pass.
    if (incumbent is not None and original_count < len(observations)
            and budget.remaining >= len(observations) + 3 * max(1, total)):
        budget.spend(len(observations))
        errors = sorted(((point_loss(predict(incumbent, row, fallback), row.target, q), index)
                         for index, row in enumerate(observations)), key=lambda item: (-item[0], item[1]))
        errors = [index for error, index in errors if error > 1e-12]
        if errors:
            witnesses.append(errors[0])
            # Startup errors and transition errors require different branches.
            # Include both when actually observed, then retain other large errors.
            other_regime = next((index for index in errors[1:]
                if bool(observations[index].history) != bool(observations[errors[0]].history)), None)
            if other_regime is not None:
                witnesses.append(other_regime)
            witnesses.extend(index for index in errors if index not in witnesses)
    if budget.remaining < total + validation_reserve:
        validation_reserve = 0
    sample_count = max(1, min(12, len(observations),
                              (budget.remaining - validation_reserve) // max(1, total)))
    # No second pass is needed when this pool can see every acquired row.
    if min(12, budget.remaining // max(1, total)) >= len(observations):
        sample_count = len(observations)
    if sample_count < len(observations):
        witness_count = min(len(witnesses), sample_count // 3)
        selected = witnesses[:witness_count]
        representative = Random(len(observations) + 9301).sample(range(len(observations)), sample_count)
        selected.extend(index for index in representative if index not in selected)
        indices = selected[:sample_count]
        sample = [observations[i] for i in sorted(indices)]
    else:
        sample = observations
    ranked = rank_candidates(candidates, sample, [], fallback=fallback, q=q,
                             budget=budget, complexity_weight=complexity_weight)
    if assessment_sink is not None:
        for row in ranked:
            row.assessment_scope = {'kind': 'complete_evidence' if sample_count == len(observations)
                                    else 'sampled_evidence',
                'observation_count': len(observations),
                'assessed_row_indices': list(range(len(observations)))
                    if sample_count == len(observations) else sorted(indices)}
        assessment_sink.extend(ranked)
    if not ranked:
        return []
    if sample_count == len(observations):
        return retain_history_frontier(ranked, observations, maximum=maximum)[0]
    checked, structures = set(), set()
    for row in ranked if maximum > 1 else ():
        if row.candidate.program not in unresolved or not _unrefuted(row):
            continue
        support = history_branch_support(row.candidate.program, observations)
        key = history_frontier_key(support)
        if key in structures:
            continue
        if not budget.spend(len(observations)):
            break
        empirical = loss(row.candidate.program, observations, fallback, q)
        row.evidence_loss = row.reference_loss = empirical
        row.score = empirical + complexity_weight * row.candidate.program.size
        if assessment_sink is not None:
            row.assessment_scope = {'kind': 'complete_evidence', 'observation_count': len(observations),
                                    'assessed_row_indices': list(range(len(observations)))}
        checked.add(row.candidate.program)
        structures.add(key)
        if len(checked) >= min(2, maximum - 1):
            break
    ranked.sort(key=lambda row: (row.score, row.candidate.program.size))
    # Only fully measured historical alternatives can receive reserved slots.
    promotable = [row for row in ranked if row.candidate.program in checked]
    retained, details = retain_history_frontier([ranked[0], *(
        row for row in promotable if row is not ranked[0])], observations, maximum=maximum)
    protected_ids = {item["candidate_id"] for item in details["history_frontier"]}
    protected = [row for row in retained if row.candidate.candidate_id in protected_ids
                 and row.candidate.program in checked and row is not ranked[0]]
    # Ordinary sampled screening retains its previous role for the other slots.
    ordinary = retain_history_frontier(
        [row for row in ranked if row.candidate.program not in unresolved], observations, maximum=maximum)[0]
    selected = [ranked[0], *protected]
    for row in [*ordinary, *ranked[1:]]:
        if len(selected) >= maximum:
            break
        if row not in selected:
            selected.append(row)
    return selected



class PlanAllocation:
    """Revise search allocations from measured gain per unit of work."""
    def __init__(self):
        self.experience = {}
        self.context = None

    def remember(self, observations, spec, experience):
        """Reuse research tactics for the observed task shape, across episodes.

        Names, task identifiers and environment labels do not define this
        context. Outcomes remain measured fresh prediction improvements, scaled
        by the current loss so a large-output task cannot dominate later ones.
        """
        response = ("risk" if spec.decision_costs is not None else
                    "binary" if len({row.target for row in observations}) == 2 else "numeric")
        controls = "single" if len(spec.feature_names) == 1 else "multiple"
        self.context = f"{response}:{controls}"
        self.experience = experience.setdefault(self.context, {})

    def allocate(self, plans, allowance, base_weights=None):
        weights = []
        for index, plan in enumerate(plans):
            previous = self.experience.get(plan.plan_id)
            weight = 1.0 if previous is None else 0.5 + min(3.0, max(0.0, previous) * allowance)
            weights.append(weight * (base_weights[index] if base_weights is not None else 1.0))
        total = sum(weights) or 1
        return [int(allowance * weight / total) for weight in weights]

    def update(self, plans, assessed, baseline_loss):
        for plan in plans:
            outcomes = [row.fresh_loss for row in assessed
                        if any(origin.candidate_id.removeprefix("startup:").startswith(plan.plan_id + ":")
                               for origin in (row.origins or (row.candidate,)))
                        and row.fresh_loss is not None]
            cost = sum(step["compute_cost"] for step in plan.steps)
            if cost and (outcomes or self.context is not None):
                gain = baseline_loss - min(outcomes) if outcomes else 0.0
                if self.context is not None:
                    gain = max(-1.0, min(1.0, gain / max(abs(baseline_loss), 1e-6)))
                rate = gain / cost
                self.experience[plan.plan_id] = 0.5 * self.experience.get(plan.plan_id, rate) + 0.5 * rate
