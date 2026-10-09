"""Resume anonymous role questions using the task's purchased sensor bank."""

from copy import deepcopy

from .models import Probe
from .object_induction import infer_object_model
from .object_dependency_methods import ObjectDependencyMethods
from .object_interventions import _missing_states, _observed_response


def paid_object_records(laboratory, batches, channels):
    """Keep controls and resets paired with legally projected transitions."""
    evidence = []
    for batch in batches:
        source = tuple(batch["channels"])
        if tuple(channels) not in laboratory.projectable_views(source):
            continue
        transitions = laboratory.project_transitions(batch.get("transitions", []), source, channels)
        sequence = batch["probe"]["sequence"]
        for position, (point, row, transition) in enumerate(zip(sequence, batch["observations"], transitions)):
            evidence.append({"controls": dict(point), **deepcopy(transition), "target": row["target"],
                "history": deepcopy(sequence[:position]), "probe_id": batch["probe"]["probe_id"],
                "position": position})
    return evidence


def _source_probe(sources, model, records, maximum, serial):
    """Transfer a missing experiment, never the donor's role assignments."""
    for source in sources:
        questions = {item["probe_id"]: item["question"]
                     for item in source.get("dependency_questions", [])}
        experiments = [(item["sequence"], questions.get(item["probe_id"], {}))
                       for item in source.get("dependency_probes", [])]
        experiments.extend((item["sequence"], {"purpose": "distinguish_role_definition"})
                           for item in source.get("dependency_witnesses", []))
        for sequence, question in experiments:
            membership = question.get("purpose") == "distinguish_membership_dependency"
            # Current paired responses select and revise membership recipes.
            # Literal donor replay cannot judge their changed response meaning.
            if membership:
                continue
            if not question:
                points = [] if _observed_response(sequence, records) else sequence
            else:
                points = _missing_states(sequence, records)
            if not points or len(points) > maximum:
                continue
            return Probe(f"object_resume:{serial}", tuple(deepcopy(points))), {
                **question, "purpose": question.get("purpose", "test_successor_role_experiment"),
                "source_object_model_id": source["model_id"], "observation_rows": len(points)}
    return None, None


def resume_object_induction(agent, laboratory, channels, acquire, batches, budget, actions,
                            *, previous_model, remaining_rows, source_object_model_ids=()):
    """Propose roles, then share their new confirmation with response research."""
    spec = laboratory.spec_for(channels)
    start, initial_batch = budget.spent, len(batches)
    previous = deepcopy(previous_model)
    model = deepcopy(previous)
    initial_records = paid_object_records(laboratory, batches, channels)
    old_rows = len(initial_records)
    # Keep the ordinary prediction confirmation affordable after role research.
    allowance = max(0, min(192, remaining_rows - min(24, spec.max_probe_length)))
    sources = [item for item in agent.library.object_models
        if item.get("model_id") in source_object_model_ids and item.get("status") != "counterexample"
        and set(item.get("raw_control_features", [])) == set(spec.feature_bounds)
        and set(item.get("raw_sensor_features", [])) == set(spec.feature_names)]
    source_ids = [item["model_id"] for item in sources]
    methods = ObjectDependencyMethods([previous, *sources], budget=budget, actions=actions)
    actions.append({"action": "resume_object_induction", "object_model_id": previous.get("model_id"),
        "channels": list(channels), "paid_transition_rows": old_rows,
        "paid_probe_ids": list(dict.fromkeys(row["probe_id"] for row in initial_records)),
        "unresolved_before": deepcopy(previous.get("unresolved", [])),
        "source_object_model_ids": source_ids, "available_new_rows": allowance})

    def records():
        return paid_object_records(laboratory, batches, channels)

    def purchased():
        return sum(len(batch["observations"]) for batch in batches[initial_batch:])

    def infer(previous_state):
        evidence = records()
        minimum = len(evidence) * (len(spec.feature_names) + len(spec.feature_bounds))
        if not evidence or budget.remaining < minimum:
            return None
        inferred = infer_object_model(spec, evidence, budget, prior=previous_state)
        if not inferred.get("groups"):
            return None
        return inferred

    # Existing reports already contain the last inference. Only a newer batch
    # or a previously exhausted fit calls for another paid interpretation.
    if (old_rows > previous.get("support_rows", 0)
            or "independent_role_confirmation" in previous.get("unresolved", [])):
        inferred = infer(previous)
        if inferred is not None:
            model = inferred
    if not initial_records and allowance >= 12:
        if acquire(channels, "observe_anonymous_dependencies", rows_count=12) is not None:
            model = infer(previous) or model

    for _ in range(allowance):
        if model.get("complete"):
            break
        evidence = records()
        available = allowance - purchased()
        minimum = (len(evidence) + 1) * (len(spec.feature_names) + len(spec.feature_bounds))
        if available <= 0 or budget.remaining < minimum:
            break
        serial = len(batches)
        probe, question = _source_probe(sources, model, evidence,
            min(available, spec.max_probe_length), serial)
        if probe is None:
            probe, question = agent._object_probe(spec, model, evidence, serial, methods=methods,
                remaining_rows=available)
            if probe is not None:
                probe = Probe(f"object_resume:{serial}", probe.sequence)
        if probe is None or probe.cost > available:
            actions.append({"action": "defer_object_dependency_question", "question": question,
                "reason": "no_affordable_unmeasured_distinction", "remaining_rows": available})
            break
        minimum = (len(evidence) + probe.cost) * (len(spec.feature_names) + len(spec.feature_bounds))
        if budget.remaining < minimum or acquire(channels,
                "distinguish_object_dependencies", probe=probe) is None:
            break
        before = deepcopy(model.get("unresolved", []))
        interpret = question.get("interpret_dependencies", True)
        refined = infer(model) if interpret else None
        if refined is not None:
            model = refined
        if question.get("research_program"):
            methods.observe(records())
        actions.append({"action": "test_object_dependency_question", "question": question,
            "probe_id": probe.probe_id, "observation_rows": probe.cost,
            "unresolved_before": before, "unresolved_after": deepcopy(model.get("unresolved", [])),
            "resumed": True, "inference_funded": refined is not None,
            "inference_deferred_until_paired_pilot": not interpret})
        if interpret and refined is None:
            break

    confirmation = None
    if model.get("complete"):
        confirmation = {"status": "pending", "schema_before": deepcopy(model["schema"])}
        model = {**model, "complete": False, "schema": None, "status": "pending_joint_confirmation",
                 "unresolved": ["independent_role_confirmation"]}

    changed = purchased() > 0 or budget.spent > start
    if changed:
        from .object_alternatives import enumerate_object_schemas
        model = {**deepcopy(previous), **deepcopy(model),
            "model_id": f"objects_{len(agent.library.object_models) + 1:04d}",
            "parent_model_id": previous.get("model_id"), "reused": False,
            "acquisition_rows": len(records()), "compute_cost": budget.spent - start,
            "source_object_model_ids": source_ids,
            "continuation": {"paid_transition_rows": old_rows, "new_transition_rows": purchased(),
                             "confirmation": confirmation}}
        if not model.get("complete") and confirmation is None:
            model["schema_alternatives"] = enumerate_object_schemas(model, budget)
        else:
            model.pop("schema_alternatives", None)
        probes = {item["probe_id"]: deepcopy(item) for item in previous.get("dependency_probes", [])}
        new_questions = [deepcopy(action) for action in actions
            if action.get("action") == "test_object_dependency_question"
            and action.get("probe_id") in {batch["probe"]["probe_id"] for batch in batches[initial_batch:]}]
        for batch in batches[initial_batch:]:
            if any(item["probe_id"] == batch["probe"]["probe_id"] for item in new_questions):
                probes[batch["probe"]["probe_id"]] = deepcopy(batch["probe"])
        model["dependency_probes"] = list(probes.values())
        model["dependency_questions"] = [*deepcopy(previous.get("dependency_questions", [])), *new_questions]
        model["dependency_methods"] = methods.snapshot() or deepcopy(previous.get("dependency_methods", []))
        model["used_source_object_model_ids"] = sorted({item["question"]["source_object_model_id"]
            for item in new_questions if item["question"].get("source_object_model_id")})
        model["compute_cost"] = budget.spent - start
        if agent.config.use_memory:
            agent.library.object_models.append(deepcopy(model))
    agent._object_models[tuple(channels)] = deepcopy(model)
    if confirmation is not None:
        agent._object_schemas[tuple(channels)] = deepcopy(confirmation["schema_before"])
    else:
        agent._object_schemas.pop(tuple(channels), None)
    actions.append({"action": "propose_discovered_objects" if confirmation is not None else "retain_object_ambiguity",
        "object_model_id": model.get("model_id"), "rows": len(records()), "resumed": True,
        "new_observation_rows": purchased(), "groups": deepcopy(model.get("groups", [])),
        "unresolved": deepcopy(model.get("unresolved", [])), "compute_cost": budget.spent - start})
    return model


def confirm_joint_object_roles(agent, laboratory, channels, model, batches, budget, actions):
    """Check the proposed schema on the response confirmation's fresh transitions."""
    pending = (model or {}).get("continuation", {}).get("confirmation")
    if not pending or pending.get("status") != "pending":
        return None
    before = deepcopy(pending["schema_before"])
    evidence = paid_object_records(laboratory, batches, channels)
    fresh = batches[-1]
    start = budget.spent
    previous_compute = model.get("compute_cost", 0)
    revised = infer_object_model(laboratory.spec_for(channels), evidence, budget, prior=model)
    supported = (revised.get("complete") and revised.get("schema") == before
                 and not revised.get("contradictions"))
    confirmation = {"status": "supported" if supported else "unresolved",
        "supported": bool(supported), "probe_id": fresh["probe"]["probe_id"],
        "observation_rows": len(fresh["observations"]), "schema_before": before,
        "contradictions": deepcopy(revised.get("contradictions", [])),
        "compute_cost": budget.spent - start, "shared_with_response": True}
    if supported:
        model.update(revised)
        model.pop("schema_alternatives", None)
        agent._object_schemas[tuple(channels)] = deepcopy(model["schema"])
    else:
        # Keep acquired counterevidence and any still-supported roles. Another
        # proposed grouping needs another response/role comparison.
        if revised.get("groups"):
            model.update(revised)
        model.update(complete=False, schema=None, status="unresolved")
        model["unresolved"] = list(dict.fromkeys([
            *model.get("unresolved", []), "independent_role_confirmation"]))
        agent._object_schemas.pop(tuple(channels), None)
    model["continuation"]["confirmation"] = confirmation
    model["acquisition_rows"] = len(evidence)
    model["compute_cost"] = previous_compute + budget.spent - start
    agent._object_models[tuple(channels)] = deepcopy(model)
    saved = next((item for item in agent.library.object_models
                  if item.get("model_id") == model.get("model_id")), None)
    if saved is not None:
        saved.update(deepcopy(model))
    actions.append({"action": "confirm_resumed_object_roles", **confirmation})
    return bool(supported)
