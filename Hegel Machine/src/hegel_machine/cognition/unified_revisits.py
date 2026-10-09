"""Try successor knowledge and expose distinctions beyond a completed task's evidence."""

from .evaluation import initial_prediction, loss
from .language import Definition, Program
from dataclasses import asdict, replace
from copy import deepcopy
import json

from .knowledge_returns import transfer_condition
from .models import Candidate, SearchBudget
from .research_programs import _sources
from .unified_continuation import paid_view


def _merge_joint_transfer_candidates(candidates, origins, task_id, serial):
    """Keep one executable candidate when several sources yield its same law.

    The fit and later response are shared work.  The source lineage remains a
    set of distinct origins so attribution can report redundancy without
    charging or crediting the same comparison as separate visits.
    """
    merged, merged_origins, by_program = [], {}, {}
    for candidate in candidates:
        key = candidate.program.expanded().simplified()
        if key not in by_program:
            by_program[key] = len(merged)
            merged.append(candidate)
            merged_origins[candidate.candidate_id] = list(origins.get(candidate.candidate_id, []))
            continue
        position = by_program[key]
        current = merged[position]
        merged_origins[current.candidate_id].extend(origins.get(candidate.candidate_id, []))
        parents = tuple(dict.fromkeys((*current.parents, *candidate.parents)))
        merged[position] = replace(current, parents=parents)

    for index, candidate in enumerate(merged):
        lineage = merged_origins.get(candidate.candidate_id, [])
        source_ids = sorted({item.get("source_id") for item in lineage if item.get("source_id")})
        if len(source_ids) < 2:
            continue
        lineage_id = f"joint:{task_id}:{serial}:{index}"
        for origin in lineage:
            origin["lineage_id"] = lineage_id
            origin["joint_source_ids"] = source_ids
        merged[index] = replace(candidate, parents=tuple(dict.fromkeys(
            (*candidate.parents, *source_ids))))
    return merged, merged_origins


def _successor_candidates(agent, spec, rows, budget, relevant):
    """Refit known research coordinates before rediscovering their parameters."""
    from .scale_bindings import bind_measurement_arguments
    from .research_programs import replay_research_program

    if spec.risk_quantile != .5:
        candidates = agent.library.propose(spec, rows, budget,
            structural_transfer=agent.config.structural_transfer,
            preserve_parameters=True, source_concept_ids=relevant)
        return candidates, {candidate.candidate_id: agent.library.transfer_evidence(candidate)
                            for candidate in candidates}
    candidates, replayed_sources, seen, origins = [], set(), set(), {}
    condition = transfer_condition(spec, rows)
    for source, payload, template, mappings in agent._replay_program_queue(
            spec, agent._recall(spec, rows), rows):
        if source not in relevant:
            continue
        for mapping_index, mapping in enumerate(mappings):
            features = {original: mapping[role]
                        for role, original in zip(template.roles, template.source_features)}
            started = budget.spent
            entry = replay_research_program(payload, rows, budget,
                risk_quantile=spec.risk_quantile, feature_mapping=features)
            if entry is None:
                continue
            replayed_sources.add(source)
            coordinates = {name: f"coordinate:{index}" for index, name in enumerate(spec.feature_names)}
            origin = {"source_id": source, "template_id": template.template_id,
                "mapping": {role: value.to_dict() if isinstance(value, Program) else value
                            for role, value in mapping.items()},
                "mapping_roles": {role: json.dumps(bind_measurement_arguments(value, coordinates).to_dict(),
                        sort_keys=True) if isinstance(value, Program) else spec.feature_names.index(value)
                    for role, value in mapping.items()},
                "condition": deepcopy(condition),
                "adaptation_work": [{"work_id": f"successor_replay:{spec.task_id}:{spec.feature_names}:{source}:{template.template_id}:{mapping_index}",
                    "compute_cost": budget.spent - started, "source_ids": [source],
                    "template_ids": [template.template_id], "preserved_work": []}]}
            # A short old trajectory can leave a mature branch unidentified.
            # Its learned parameters still compete with the freshly fitted plan.
            for mode, program in (("preserved", bind_measurement_arguments(template.pattern, mapping)),
                                  ("refitted", entry["candidate"].program)):
                if agent.library.transfer_refuted(spec, program):
                    continue
                key = source, program
                if key in seen:
                    continue
                seen.add(key)
                candidate = Candidate(f"{source}:successor_replay:{mode}:{len(candidates)}",
                    program, "abstract_reuse", "Retest a learned response and its research coordinates",
                    (source,), source)
                candidates.append(candidate)
                origins[candidate.candidate_id] = [deepcopy(origin)]
    if remaining := relevant - replayed_sources:
        proposed = agent.library.propose(spec, rows, budget,
            structural_transfer=agent.config.structural_transfer,
            preserve_parameters=True, source_concept_ids=remaining)
        candidates.extend(proposed)
        origins.update({candidate.candidate_id: agent.library.transfer_evidence(candidate)
                        for candidate in proposed})

    return _merge_joint_transfer_candidates(candidates, origins, spec.task_id,
        agent.library._search_serial)


def _joint_successor_candidate(agent, spec, rows, budget, scored, origins):
    """Fit one residual addition to the best paid single-source explanation."""
    from .research_programs import _Fitter, _entry

    single = []
    for candidate, measured in scored:
        evidence = origins.get(candidate.candidate_id, [])
        sources = {item["source_id"] for item in evidence}
        if len(sources) == 1:
            single.append((candidate, measured, evidence, next(iter(sources))))
    if len({item[3] for item in single}) < 2:
        return None
    single.sort(key=lambda item: (item[1], item[0].program.size, item[0].candidate_id))
    base, base_loss, base_evidence, base_source = single[0]
    if base_loss <= agent.config.minimum_improvement:
        return None

    def learned_call(candidate, source):
        definition = Definition(f"successor:{candidate.candidate_id}", (), candidate.program, source)
        return Program("call", name=definition.definition_id, definition=definition)

    fitter = _Fitter(rows, budget, spec.risk_quantile)
    base_call = learned_call(base, base_source)
    best, seen_sources = None, {base_source}
    for candidate, _, evidence, source in single[1:]:
        if source in seen_sources:
            continue
        seen_sources.add(source)
        started = budget.spent
        fit = fitter.fit((base_call, learned_call(candidate, source)))
        if fit is None:
            break
        if (fit["loss"] + agent.config.minimum_improvement >= base_loss
                or set(_sources(fit["program"])) != {base_source, source}
                or agent.library.transfer_refuted(spec, fit["program"])):
            continue
        work = budget.spent - started
        step = {"operation": "joint_successor_residual",
            "question": "Does another learned computation explain the remaining paid response?",
            "program": fit["program"], "loss": fit["loss"]}
        entry = _entry(fit, [step], len(rows), work)
        joint = replace(entry["candidate"], candidate_id=f"joint_successor:{base.candidate_id}:{candidate.candidate_id}")
        source_ids = sorted((base_source, source))
        lineage = f"joint:{spec.task_id}:{agent.library._search_serial}:{joint.candidate_id}"
        joint_evidence = deepcopy([*base_evidence, *evidence])
        for origin in joint_evidence:
            origin.update(lineage_id=lineage, joint_source_ids=source_ids,
                          research_program=deepcopy(entry["research_program"]))
            origin["adaptation_work"].append({"work_id": lineage,
                "compute_cost": work, "source_ids": source_ids,
                "template_ids": sorted({item["template_id"] for item in joint_evidence}),
                "preserved_work": []})
        if best is None or fit["loss"] < best[1]:
            best = joint, fit["loss"], joint_evidence
    return best


def offer_successor_knowledge(agent, laboratory, previous, concept_ids, budget, *, object_model_ids=(),
                              knowledge_changes=(), method_screening_actions=None,
                              remaining_resources=None):
    """Return a proposal and a resource quote, never an adoption decision."""
    if getattr(laboratory, "measurement_views", None):
        if object_model_ids and agent._object_induction(laboratory):
            from .anonymous_revisits import offer_anonymous_successor
            offer = offer_anonymous_successor(agent, laboratory, previous, object_model_ids, budget)
            if offer is not None:
                return offer
        from .object_revisits import offer_object_successor
        offer = offer_object_successor(agent, laboratory, previous, concept_ids, budget)
        if offer is not None:
            return offer
        if knowledge_changes:
            from .knowledge_revisits import offer_object_experiment
            return offer_object_experiment(agent, laboratory, previous, knowledge_changes, budget)
        return None
    if previous.get("object_research"):
        return None
    draft_ids = {change["knowledge_id"] for change in knowledge_changes
                 if change["kind"] == "draft" and change["available"]}
    if draft_ids:
        from .numeric_method_revisits import offer_numeric_method_experiment
        offer = offer_numeric_method_experiment(agent, laboratory, previous, draft_ids, budget,
            actions=method_screening_actions, remaining_resources=remaining_resources)
        if offer is not None:
            return offer
    incumbent = Program.from_dict(previous["program"])
    old_channels = set(previous["selected_channels"])
    views = list(dict.fromkeys(tuple(batch["channels"]) for batch in previous["batches"]))
    relevant = {item.concept_id for item in agent.library.concepts
                if item.concept_id in concept_ids and item.status != "revoked"
                and item.risk_quantile == laboratory.spec.risk_quantile}
    completed = previous["status"] == "completed"
    if completed and (laboratory.spec.risk_quantile != .5 or old_channels):
        return None
    old_length = max((len(batch["probe"]["sequence"]) for batch in previous["batches"]), default=0)
    if completed and old_length >= laboratory.spec.max_probe_length:
        return None
    best = None
    if not relevant:
        return None
    for index, channels in enumerate(views):
        if not old_channels <= set(channels):
            continue
        rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
        if not rows or not budget.spend(len(rows)):
            continue
        spec = agent._spec(laboratory, channels)
        baseline = loss(incumbent, rows, previous["fallback"], spec.risk_quantile)
        fallback = initial_prediction(rows, spec.risk_quantile)
        allocation = budget.remaining // max(1, len(views) - index)
        planning_reserve = allocation // 2 if completed else 0
        fitting = SearchBudget(max(0, allocation - planning_reserve - agent.config.max_candidates * len(rows)))
        candidates, origins = _successor_candidates(agent, spec, rows, fitting, relevant)
        budget.spend(fitting.spent)
        # Give genuinely multi-source programs a chance to compete before the
        # single-source replay queue fills the candidate cap.  The measured
        # loss below still rejects a composition that does not improve the
        # retained baseline.
        candidate_queue = sorted(candidates,
            key=lambda item: (len(set(_sources(item.program))) < 2,
                              item.program.size, item.candidate_id))

        def measured_candidates():
            scored = []
            for candidate in candidate_queue[:agent.config.max_candidates]:
                if not budget.spend(len(rows)):
                    break
                measured = loss(candidate.program, rows, fallback, spec.risk_quantile)
                scored.append((candidate, measured))
                yield candidate, measured
            if not completed and spec.risk_quantile == .5:
                joint = _joint_successor_candidate(agent, spec, rows, budget, scored, origins)
                if joint is not None:
                    candidate, measured, evidence = joint
                    origins[candidate.candidate_id] = evidence
                    yield candidate, measured

        for candidate, measured in measured_candidates():
            gain = baseline - measured
            if completed:
                # Old evidence only establishes compatibility. A forecast
                # disagreement proposes an experiment, never a replacement.
                if measured > baseline + agent.config.minimum_improvement:
                    continue
                challenge = _history_challenge(agent, laboratory, incumbent, candidate,
                    old_length, budget, serial=len(previous["batches"]))
                if challenge is None:
                    continue
            elif gain <= agent.config.minimum_improvement:
                continue
            names = {name for name, _ in candidate.program.observed_variables()}
            used_channels = [name for name in channels if name in names]
            sources = sorted(relevant & {candidate.memory_id, *candidate.parents})
            length = min(24, laboratory.spec.max_probe_length)
            if completed:
                length = max(length, len(challenge["probe"]["sequence"]))
            # An exact response selected on already paid observations can be
            # frozen now. It needs a new confirmation, not another selection
            # trajectory before that confirmation.
            paid_selection = not completed and spec.risk_quantile == .5 and measured <= agent.config.minimum_improvement
            new_rows = length if paid_selection else 2 * length
            rate = sum(item.cost_per_row for item in laboratory.channels
                       if spec.risk_quantile != .5 or item.name in set(used_channels) | old_channels)
            evidence = origins.get(candidate.candidate_id, [])
            procedure = next((item.get("research_program") for item in evidence
                              if item.get("research_program") is not None), None)
            proposed = {"candidate": {"program": candidate.program.to_dict(),
                "channels": used_channels, "fallback": fallback,
                "source_concept_ids": sources, "kind": "successor_knowledge",
                "transfer_evidence": deepcopy(evidence),
                "family_id": "T03", "old_loss": baseline, "selection_loss": measured},
                "old_evidence_gain": gain, "old_evidence_rows": len(rows),
                "relative_gain": gain / max(baseline, agent.config.minimum_improvement),
                "minimum_resources": {"observation_cost": new_rows,
                    "measurement_cost": new_rows * rate,
                    "compute_cost": 6 * len(rows) + 32 * length}}
            if paid_selection:
                proposed["candidate"]["paid_selection"] = {
                    "evidence_batches": [batch["probe"]["probe_id"] for batch in previous["batches"]],
                    "evidence_rows": len(rows), "channels": list(channels),
                    "baseline_program": previous["program"], "baseline_fallback": previous["fallback"],
                    "baseline_loss": baseline, "candidate_loss": measured}
                proposed["basis"] = "paid_selection_fresh_confirmation"
            if procedure is not None:
                proposed["candidate"]["research_program"] = deepcopy(procedure)
            for origin in proposed["candidate"]["transfer_evidence"]:
                origin["adaptation_work"].append({
                    "work_id": f"successor_evaluation:{spec.task_id}:{spec.feature_names}:{candidate.candidate_id}",
                    "compute_cost": len(rows), "source_ids": sources,
                    "template_ids": [origin["template_id"]], "preserved_work": []})
            if completed:
                separation = challenge["predicted_separation"]
                proposed["candidate"]["challenge"] = challenge
                proposed["basis"] = "compatible_evidence_unmeasured_history"
                proposed["relative_gain"] = separation / (separation + max(baseline, agent.config.minimum_improvement))
                proposed["minimum_resources"]["observation_cost"] = len(challenge["probe"]["sequence"]) + length
                proposed["minimum_resources"]["measurement_cost"] = proposed["minimum_resources"]["observation_cost"] * rate
            if spec.risk_quantile == .5 and not completed:
                proposed.update(expected_loss_reduction=gain, loss_kind="squared_error")
            if best is None or proposed["relative_gain"] > best["relative_gain"]:
                best = proposed
    return best


def _history_challenge(agent, laboratory, incumbent, candidate, old_length, budget, *, serial):
    from .competing_object_experiments import plan_competing_object_probe, shorten_discriminating_probe
    spec = agent._spec(laboratory, ())
    controls = set(spec.feature_bounds)
    if any(not program.required_features() <= controls for program in (incumbent, candidate.program)):
        return None
    # Start just beyond purchased histories; extend only if the frozen
    # explanations do not yet predict a measurable distinction.
    length = max(old_length + 1, 3)
    while length <= spec.max_probe_length and budget.remaining:
        plan = plan_competing_object_probe(spec,
            [{"candidate": Candidate("completed_incumbent", incumbent, "retained")},
             {"candidate": candidate}], budget, serial=serial, length=length,
            tolerance=agent.config.minimum_improvement)
        if plan["forecast_status"] == "predicted_distinction":
            plan = shorten_discriminating_probe(plan, budget, tolerance=agent.config.minimum_improvement)
            if len(plan["probe"].sequence) <= old_length:
                return None
            return {**plan, "probe": asdict(plan["probe"]), "old_max_probe_length": old_length}
        if length == spec.max_probe_length:
            break
        length = min(spec.max_probe_length, length * 2)
    return None
