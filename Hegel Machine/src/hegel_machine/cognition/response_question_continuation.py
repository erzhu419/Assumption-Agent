"""Spend later research resources on frozen, unresolved response hypotheses."""

from copy import deepcopy
from dataclasses import asdict
from statistics import mean, stdev

from .evaluation import predict
from .language import Program
from .models import Candidate, Probe, SearchBudget
from .noise_replication import assess_noise_replication
from .object_continuation import decode_entry
from .response_research_questions import refresh_response_question_quotes, response_question_offer


def run_response_question(agent, laboratory, previous, offer):
    report = deepcopy(previous)
    old_costs = dict(previous["costs"])
    question = next(item for item in report["response_questions"]
                    if item["question_id"] == offer["response_question"]["question_id"])
    parent, candidate = map(decode_entry, (question["parent"], question["candidate"]))
    length = offer["probe_length"]
    third_incumbent = previous["program"] not in (question["parent"]["candidate"]["program"],
                                                  question["candidate"]["candidate"]["program"])
    comparison_cost = (13 if third_incumbent else 10) * length
    budget = SearchBudget(agent.config.compute_budget, old_costs["compute_cost"])
    actions, batches = report["actions"], report["batches"]
    costs = dict(old_costs)
    added, sources = [], []

    def finish(outcome):
        costs["compute_cost"] = budget.spent
        report.update(costs=costs,
            visit_costs={key: value - old_costs[key] for key, value in costs.items()},
            concepts_added=added, used_sources=sources,
            continuation={"question_id": question["question_id"], "outcome": outcome,
                "previous_status": previous["status"], "previous_costs": old_costs,
                "paid_rows_reused": old_costs["observation_cost"],
                "source_concept_ids": sources})
        return report

    spec = agent._spec(laboratory, ())
    parent_refuted, candidate_refuted = [agent.config.use_memory and
        agent.library.transfer_refuted(spec, entry["candidate"].program) for entry in (parent, candidate)]
    incumbent_refuted = agent.config.use_memory and agent.library.transfer_refuted(
        spec, Program.from_dict(previous["program"]))
    if parent_refuted and candidate_refuted:
        question.update(status="refuted", outcome="both_programs_refuted")
        if incumbent_refuted:
            report["status"] = "unresolved"
            report["noise_assessment"] = {**report.get("noise_assessment", {}),
                "adequate": False, "reason": "retained_transfer_counterexample"}
        actions.append({"action": "withdraw_refuted_response_question", "question_id": question["question_id"],
            "reason": "both_bound_programs_have_paid_counterexamples",
            "incumbent_refuted": incumbent_refuted})
        return finish(question["outcome"])

    # A saved public continuation can outlive the offer that created it. Pay
    # for the same current-evidence pricing as the session before acquiring.
    is_refuted = (lambda program: agent.library.transfer_refuted(spec, program)) \
        if agent.config.use_memory else None
    actions.extend(refresh_response_question_quotes(report, [question], budget,
        minimum_improvement=agent.config.minimum_improvement, is_refuted=is_refuted))
    current_offer = response_question_offer(report, question, laboratory.spec.max_probe_length,
        {"observation_cost": agent.config.observation_budget - costs["observation_cost"],
         "measurement_cost": agent.measurement_budget - costs["measurement_cost"],
         "compute_cost": budget.remaining}, agent.config.minimum_improvement,
        resource_limits={"observation_budget": agent.config.observation_budget,
                         "compute_budget": agent.config.compute_budget}, is_refuted=is_refuted)
    if current_offer is None:
        return finish("deferred")
    length = current_offer["candidate"]["probe_length"]
    comparison_cost = (13 if third_incumbent else 10) * length
    if (question["status"] != "pending"
            or 2 * length > agent.config.observation_budget - costs["observation_cost"]
            or comparison_cost > budget.remaining):
        return finish("deferred")
    incumbent = {"candidate": Candidate("response_question:incumbent",
            Program.from_dict(previous["program"]), "retained_incumbent"),
        "fallback": previous["fallback"], "kind": previous["selected_kind"],
        "structure": previous.get("selected_structure"),
        "research_program": deepcopy(previous.get("research_program")),
        "mechanism_revision": deepcopy(previous.get("mechanism_revision")),
        "method_source_ids": list(previous.get("method_source_ids", []))}
    restore = getattr(laboratory, "restore", None)
    if restore is not None:
        restore([{"probe_id": batch["probe"]["probe_id"], "sequence": batch["probe"]["sequence"],
                  "channels": batch["channels"]} for batch in previous["batches"]])

    # Both programs and their initial predictions were saved before these
    # responses. The pilot that generated the question is never pooled into
    # the fresh comparison's evidence.
    probe = agent._probe(agent._spec(laboratory, ()), len(batches), length, (), budget, actions)
    pair = []
    for request, purpose in ((probe, "compare_response_question"),
            (Probe(f"response_repeat_{len(batches) + 1}", probe.sequence), "repeat_response_question")):
        batch = laboratory.acquire(request, ())
        pair.append(list(batch.observations))
        costs["observation_cost"] += batch.observation_cost
        costs["measurement_cost"] += batch.measurement_cost
        batches.append({"probe": asdict(request), "channels": [],
            "observations": [asdict(row) for row in batch.observations],
            "transitions": getattr(batch, "transitions", [])})
        actions.append({"action": "acquire", "purpose": purpose, "probe_id": request.probe_id,
            "channels": [], "observation_cost": batch.observation_cost,
            "measurement_cost": batch.measurement_cost})
    budget.spend(6 * length)
    reference, revised = [], []
    for left, right in zip(*pair):
        reference.append(mean((row.target - predict(parent["candidate"].program, row,
            parent.get("fallback", previous["fallback"]))) ** 2 for row in (left, right)))
        revised.append(mean((row.target - predict(candidate["candidate"].program, row,
            candidate.get("fallback", previous["fallback"]))) ** 2 for row in (left, right)))
    question["gains"].extend(old - new for old, new in zip(reference, revised))
    question["reference_losses"].extend(reference)
    question["candidate_losses"].extend(revised)
    gains = question["gains"]
    gain, margin = mean(gains), 3 * stdev(gains) / len(gains) ** .5
    use_candidate = gain > agent.config.minimum_improvement
    selected = candidate if use_candidate else parent
    if parent_refuted:
        selected = candidate
    elif candidate_refuted:
        selected = parent
    incumbent_loss = None
    if third_incumbent:
        budget.spend(3 * length)
        incumbent_loss = mean((row.target - predict(incumbent["candidate"].program, row,
            incumbent["fallback"])) ** 2 for rows in pair for row in rows)
        eligible = [entry for entry, refuted in ((incumbent, incumbent_refuted),
            (parent, parent_refuted), (candidate, candidate_refuted)) if not refuted]
        selected = min(eligible, key=lambda item:
            incumbent_loss if item is incumbent else mean(reference) if item is parent else mean(revised))
        selected_loss = (incumbent_loss if selected is incumbent else
                         mean(reference) if selected is parent else mean(revised))
        if not incumbent_refuted and selected_loss + agent.config.minimum_improvement >= incumbent_loss:
            selected = incumbent
    use_candidate = selected is candidate
    outcome = "candidate" if selected is candidate else "parent" if selected is parent else "incumbent"
    program = selected["candidate"].program
    fallback = selected.get("fallback", previous["fallback"])
    noise = assess_noise_replication(program, *pair, budget,
        minimum_improvement=agent.config.minimum_improvement, fallback=fallback)
    model_loss = max(noise["original_model_loss"], noise["repeated_model_loss"])
    adequate = (model_loss <= agent.config.minimum_improvement or
        (noise["noise_supported"] and model_loss <= noise["repeatable_loss_limit"]))
    noise["adequate"] = adequate
    question.update(pilot_only=False, observed_gain=gain, uncertainty_margin=margin,
        rows=len(gains), reference_loss=mean(question["reference_losses"]),
        candidate_loss=mean(question["candidate_losses"]),
        status="settled" if adequate or candidate_refuted or
            (not parent_refuted and gain + margin <= agent.config.minimum_improvement) else "pending",
        outcome=outcome,
        next_probe_length=min(laboratory.spec.max_probe_length, 2 * length))
    question["probe_ids"].extend([probe.probe_id, batches[-1]["probe"]["probe_id"]])
    actions.append({"action": "resolve_response_question", "question_id": question["question_id"],
        "probe_ids": question["probe_ids"][-2:], "fresh_rows": length,
        "accumulated_rows": len(gains), "gain": gain, "margin": margin,
        "reference_loss": question["reference_loss"], "candidate_loss": question["candidate_loss"],
        "incumbent_loss": incumbent_loss,
        "selection_basis": "fresh_frozen_response_comparison", "outcome": question["outcome"],
        "refuted_programs": [name for name, refuted in (("parent", parent_refuted),
            ("candidate", candidate_refuted), ("incumbent", incumbent_refuted)) if refuted],
        "status": question["status"], "adequate": adequate, "compute_cost": budget.spent - old_costs["compute_cost"]})
    report.update(program=program.to_dict(), expression=program.describe(), fallback=fallback,
        status="completed" if adequate else "unresolved", selected_kind=selected["kind"],
        selected_structure=selected.get("structure"), research_program=selected.get("research_program"),
        mechanism_revision=deepcopy(selected.get("mechanism_revision")), selected_channels=[],
        validation_loss=None, selection_loss=(incumbent_loss if selected is incumbent else
            question["candidate_loss"] if use_candidate else question["reference_loss"]),
        confirmation_loss=mean((noise["original_model_loss"], noise["repeated_model_loss"])),
        selection_basis="fresh_frozen_response_comparison", noise_assessment=noise,
        method_source_ids=list(selected.get("method_source_ids", [])))
    if (adequate and agent.config.use_memory and
            (program.to_dict() != previous["program"] or previous["status"] != "completed")):
        from .unified_continuation import paid_view
        from .meta_prior_research import _sources
        spec = agent._knowledge_spec(agent._spec(laboratory, ()))
        rows = [row for batch in paid_view(laboratory, report, ()) for row in batch]
        concept_id, is_new = agent.library.learn(selected["candidate"], spec, rows)
        added = [concept_id] if is_new else []
        sources = list(dict.fromkeys([*_sources(program), *selected.get("method_source_ids", [])]))
        for source in sources:
            agent.library.record_use(source, spec.task_id)
        if not any(item["source_concept_id"] == concept_id for item in agent.library.meta_priors):
            agent.library.meta_priors.append({"prior_id": f"meta_prior_{len(agent.library.meta_priors) + 1:04d}",
                "source_concept_id": concept_id, "family_id": selected.get("family_id", "T03"),
                "kind": selected["kind"], "claim": "A frozen response revision predicts new measured responses",
                "status": "supported", "task_id": spec.task_id, "source_features": list(spec.feature_names),
                "control_features": list(spec.feature_bounds), "object_schema": None, "structure": None,
                "support_rows": len(gains), "validation_loss": report["confirmation_loss"],
                "probe_ids": list(question["probe_ids"]), "counterexamples": [], "uses": 0,
                "parent_concept_ids": sources,
                "research_method": {"validation_rows": len(gains), "confirmation_rows": length,
                    "noise_supported": noise["noise_supported"], "risk_quantile": .5},
                "counter_prediction": {"type": "future_trajectory", "condition": "A later measured response"}})
    costs.update(compute_cost=budget.spent, queries=len(batches))
    return finish(question["outcome"])
