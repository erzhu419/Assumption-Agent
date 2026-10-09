"""Price a successor's partial method against a numeric task's paid failure."""

from copy import copy, deepcopy
from dataclasses import asdict
import json
from math import ceil

from .action_policy import ResearchActionPolicy
from .condition_research_returns import plan_condition_research_allocation
from .evaluation import initial_prediction, loss
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget
from .observed_condition_experiments import plan_observed_condition_probe
from .research_drafts import _additive_atoms, _response_terms, replay_numeric_research_drafts
from .research_resources import research_resource_share
from .unified_continuation import paid_view


METHOD_VISIT_ACTION = "successor_partial_method_visit"
METHOD_RESPONSE_ACTION = "successor_partial_method_response"


def reprice_numeric_method_offer(offer, remaining_resources):
    """Reuse paid method evidence when the visited-task pool changes.

    Screening is already charged to this pool. It belongs in visit attribution,
    but charging it again as future execution understates the frozen plan's value.
    Risk methods share the offer basis but have different evidence and quotes.
    """
    candidate = offer["candidate"]
    if candidate.get("method_experiment") is None:
        return offer
    opportunity = candidate["method_opportunity"]
    baseline = opportunity["paid_counterexample_loss"]
    separation = opportunity["predicted_separation"]
    paid_gain = max(0., offer["old_evidence_gain"]) / baseline
    expected_gain = (paid_gain + .5 * separation / (separation + baseline)
        if not opportunity.get("observed_response_trials", 0) else
        max(0., paid_gain + opportunity["observed_response_gain"]))
    allocation = opportunity["confirmation_allocation"]
    if allocation is not None:
        expected_gain *= allocation["calibrated_information_probability"]
    costs = opportunity["recommended_resources"]
    share = research_resource_share(costs, remaining_resources)
    offer.update(expected_marginal_value=expected_gain, expected_resource_costs=dict(costs),
                 expected_loss_reduction=expected_gain * baseline, loss_kind="squared_error",
                 resource_share=share, pricing_resources=dict(remaining_resources),
                 relative_gain=expected_gain / share)
    return offer


def refresh_numeric_method_quote(quote, experience, remaining_resources):
    """Reconsider a paid executable plan using actual applicable visit returns.

    Refitting and experiment planning have already been purchased. Only the
    complete resource quote and its expected return change; an adverse return
    can withdraw a queued proposal just as a useful return can reopen it.
    """
    candidate = quote["candidate"]
    opportunity = candidate["method_opportunity"]
    inputs = quote["pricing_inputs"]
    condition = inputs["method_context"].rsplit(f":{METHOD_VISIT_ACTION}:knowledge:", 1)[0]

    def matching(action):
        prefix = f"{condition}:{action}:knowledge:"
        return next((actions[action] for context, actions in (experience or {}).items()
            if context.startswith(prefix) and action in actions
            and actions[action].get("knowledge_applicability")
                == opportunity["method_applicability"]), {})

    record = matching(METHOD_VISIT_ACTION)
    response = matching(METHOD_RESPONSE_ACTION)
    execution = record.get("execution_costs", {})
    estimates = {resource: execution.get(resource, 0.) / max(1, execution.get("visits", 0))
                 for resource in ("observation_cost", "measurement_cost", "compute_cost")}
    quote["method_feedback"] = {"trials": response.get("trials", 0),
        "route_trials": record.get("trials", 0),
        "unscored_attempts": record.get("unscored_attempts", 0),
        "expected_gain": response.get("expected_gain"), "estimated_costs": estimates}
    observation_quote = inputs["observation_quote"]
    compute_quote = inputs["compute_quote"]
    whole_compute_quote = max(compute_quote, inputs["task_compute_cap"])
    if execution.get("visits", 0):
        whole_compute_quote = max(compute_quote, ceil(estimates["compute_cost"]))
        observation_quote = max(observation_quote, ceil(estimates["observation_cost"]))
    rate, length = inputs["measurement_rate"], inputs["batch_length"]
    measurement_quote = max(observation_quote * rate, ceil(estimates["measurement_cost"]))
    minimum = {"observation_cost": observation_quote,
               "measurement_cost": measurement_quote, "compute_cost": whole_compute_quote}
    experiment = candidate["method_experiment"]
    plan = {**experiment, "probe": Probe(experiment["probe"]["probe_id"],
                                       tuple(experiment["probe"]["sequence"]))}
    allocation = plan_condition_research_allocation(inputs["condition_uncertainty"], plan,
        observation_remaining=remaining_resources["observation_cost"],
        compute_remaining=remaining_resources["compute_cost"],
        measurement_remaining=remaining_resources["measurement_cost"], measurement_rate=rate,
        setup_observation_cost=observation_quote - length,
        setup_compute_cost=compute_quote - 32 * length,
        minimum_compute_cost=whole_compute_quote, batch_compute_cost=15000 + 2 * length,
        previous=inputs["condition_research_returns"], tolerance=inputs["tolerance"])
    if allocation is not None:
        observation_quote = allocation["resource_costs"]["observation_cost"]
        measurement_quote = max(allocation["resource_costs"]["measurement_cost"],
                                ceil(estimates["measurement_cost"]))
        whole_compute_quote = allocation["resource_costs"]["compute_cost"]
    opportunity.update(whole_visit_compute_quote=minimum["compute_cost"],
        recommended_resources={"observation_cost": observation_quote,
            "measurement_cost": measurement_quote, "compute_cost": whole_compute_quote},
        observed_visit_costs=estimates, observed_response_trials=response.get("trials", 0),
        observed_response_gain=response.get("expected_gain"), confirmation_allocation=allocation,
        confirmation_rows=allocation["selected_rows"] if allocation is not None else length)
    quote["minimum_resources"] = minimum
    reprice_numeric_method_offer(quote, remaining_resources)
    return quote if quote["expected_marginal_value"] > 0. else None


def _method_applicability(record, spec, incumbent):
    """Price the still executable response, independent of research narration.

    An adoption step can supersede the fitted values of a historical discovery
    while its conditional window and response coordinates still execute. The
    output computation identifies those coordinates; stale earlier steps do
    not describe the method now being replayed.
    """
    names = record["source_features"]
    roles = {name: f"command_{index}" for index, name in enumerate(
        name for name in names if name in record["control_features"])}
    roles.update({name: f"reading_{index}" for index, name in enumerate(
        name for name in names if name in record["observed_features"])})

    procedure = record["research_program"]
    response = Program.from_dict(procedure["graph"]["nodes"][procedure["output_node"]]["answer"])
    return _response_method_structure(response, roles, spec, incumbent)


def method_response_applicability(program, spec, incumbent):
    """Match the actual fitted response in the target's public input roles."""
    roles = {name: f"command_{index}" for index, name in enumerate(
        name for name in spec.feature_names if name in spec.feature_bounds)}
    roles.update({name: f"reading_{index}" for index, name in enumerate(
        name for name in spec.feature_names if name not in spec.feature_bounds)})
    return _response_method_structure(program, roles, spec, incumbent)


def _response_method_structure(response, roles, spec, incumbent):
    def structure(node, *, threshold=False):
        if node.op == "const":
            return {"op": "threshold" if threshold else "zero" if node.value == 0. else "constant"}
        result = {"op": node.op}
        if node.name is not None:
            result["name"] = roles.get(node.name, node.name)
        if node.lag:
            result["lag"] = node.lag
        if node.children:
            result["children"] = [structure(child, threshold=node.op in ("ge", "lt") and index == 1)
                for index, child in enumerate(node.children)]
        return result

    response = response.expanded()
    terms = _response_terms(response)
    if any(atom.op == "const" and atom.value != 0. for atom in _additive_atoms(response)):
        terms.append(Program("const", value=1.))
    basis = []
    for term in terms:
        item = structure(term)
        if item not in basis:
            basis.append(item)
    basis.sort(key=lambda item: json.dumps(item, sort_keys=True))
    program = incumbent["candidate"].program.expanded()
    def has_condition(node):
        return node.op == "if" or any(has_condition(child) for child in node.children)

    return {"method_steps": [{"operation": "replay_numeric_research_drafts"}],
        "response_basis": basis,
        "commands": len(spec.feature_bounds),
        "readings": len(spec.feature_names) - len(spec.feature_bounds),
        "retained_response_has_history": program.history_depth > 0,
        "retained_response_has_condition": has_condition(program)}


def method_visit_policy(library, spec, rows, applicability, *, action=METHOD_VISIT_ACTION):
    policy = ResearchActionPolicy(rows, spec, library.action_experience)
    policy.set_condition("successor_method_paid_failure")
    policy.set_action_knowledge(action, applicability)
    return policy


def record_numeric_method_visit(library, spec, previous, report, opportunity, screening_compute_cost):
    """Keep route returns apart from the responses actually tested on new rows."""
    rows = [Observation(dict(item["features"]), item["target"], tuple(item["history"]))
            for batch in previous["batches"] for item in batch["observations"]]
    policy = method_visit_policy(library, spec, rows, opportunity["method_applicability"])
    execution_costs = {key: report["visit_costs"][key] for key in
             ("observation_cost", "measurement_cost", "compute_cost")}
    costs = dict(execution_costs)
    costs["compute_cost"] += screening_compute_cost
    comparisons = report.get("continuation", {}).get("method_fresh_comparisons", [])
    paired = comparisons[-1] if comparisons else None
    if paired is None:
        policy.record_unscored_attempt(METHOD_VISIT_ACTION, status="unresolved", **costs)
    else:
        policy.update(METHOD_VISIT_ACTION, before_loss=paired["before_loss"],
            after_loss=paired["after_loss"], **costs)
    execution = policy.action_record(METHOD_VISIT_ACTION).setdefault("execution_costs", {"visits": 0})
    execution["visits"] += 1
    for resource, amount in execution_costs.items():
        execution[resource] = execution.get(resource, 0.) + amount
    outcome = {"task_id": spec.task_id, "whole_visit_costs": costs,
        "screening_compute_cost": screening_compute_cost,
        "fresh_comparison": deepcopy(paired), "status": report["status"],
        "method_context": policy.action_context(METHOD_VISIT_ACTION),
        "method_applicability": deepcopy(opportunity["method_applicability"])}
    outcome["response_confirmations"] = []
    for comparison in comparisons:
        applicability = comparison.get("attempted_method_applicability")
        source_ids = comparison.get("attempted_source_draft_ids", [])
        if applicability is None or not source_ids:
            continue
        response_policy = method_visit_policy(library, spec, rows, applicability,
                                              action=METHOD_RESPONSE_ACTION)
        # This comparison was already paid by the route. Its response evidence
        # supplies no separate execution-cost estimate or second resource bill.
        response = response_policy.update(METHOD_RESPONSE_ACTION,
            before_loss=comparison["before_loss"], after_loss=comparison["attempted_after_loss"],
            observation_cost=0, measurement_cost=0, compute_cost=0)
        evidence = {"kind": "method_response_confirmation", "task_id": spec.task_id,
            "probe_id": comparison["probe_id"], "observation_rows": comparison["observation_rows"],
            "baseline_program": deepcopy(comparison["baseline_program"]),
            "baseline_fallback": comparison["baseline_fallback"],
            "program": deepcopy(comparison["attempted_program"]),
            "fallback": comparison["attempted_fallback"],
            "channels": list(comparison["attempted_channels"]),
            "before_loss": comparison["before_loss"], "after_loss": comparison["attempted_after_loss"],
            "retained_loss": comparison["after_loss"],
            "method_context": response_policy.action_context(METHOD_RESPONSE_ACTION),
            "method_applicability": deepcopy(applicability),
            "trial": response["trials"], "gain": response["last_gain"]}
        for draft in library.research_drafts:
            if draft["draft_id"] in source_ids:
                draft.setdefault("outcomes", []).append(deepcopy(evidence))
        outcome["response_confirmations"].append({key: evidence[key] for key in (
            "probe_id", "method_context", "trial", "gain")}
            | {"source_draft_ids": list(source_ids)})
    report["continuation"]["method_visit_outcome"] = deepcopy(outcome)
    return deepcopy(policy.action_record(METHOD_VISIT_ACTION))


def offer_numeric_method_experiment(agent, laboratory, previous, draft_ids, budget, *,
                                    actions=None, remaining_resources=None):
    """New response structure can warrant an experiment before improving old loss.

    Screening fits and forecasts use only purchased rows and public controls.
    The returned experiment must still be bought, refitted and independently
    confirmed by the normal continuation; it does not support either law.
    """
    if laboratory.spec.risk_quantile != .5:
        from .risk_research_methods import offer_risk_method_experiment
        return offer_risk_method_experiment(agent, laboratory, previous, draft_ids, budget,
            actions=actions)
    tolerance = agent.config.minimum_improvement
    if (previous["status"] != "unresolved" or laboratory.spec.risk_quantile != .5
            or not agent.config.active_probes or not previous["selected_channels"]):
        return None
    # Exact point explanations with only an unresolved boundary are already
    # handled by condition research. Method opportunities concern real failures.
    if max(previous.get("selection_loss") or 0.,
           previous.get("confirmation_loss") or 0.) <= tolerance:
        return None
    drafts = [record for record in agent.library.research_drafts
        if record["draft_id"] in draft_ids and record.get("condition_uncertainty") is not None
        and record["source_task_id"] != laboratory.spec.task_id
        and not record.get("resolved_concept_id")]
    if not drafts:
        return None
    channels = tuple(previous["selected_channels"])
    spec = agent._spec(laboratory, channels)
    rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
    if len(rows) < 8 or not budget.spend(len(rows)):
        return None
    incumbent = {"candidate": Candidate("method_revisit_incumbent",
        Program.from_dict(previous["program"]), "retained_incumbent"),
        "condition_uncertainty": deepcopy(previous.get("condition_uncertainty")),
        "research_program": deepcopy(previous.get("research_program"))}
    baseline = loss(incumbent["candidate"].program, rows, previous["fallback"], .5)
    if baseline <= tolerance:
        return None
    # Restrict this screening to the new applicable opportunity. An earlier
    # failed draft cannot crowd it out or receive another independent visit.
    screening = copy(agent)
    screening.library = copy(agent.library)
    screening.library.research_drafts = drafts
    fitting = SearchBudget(max(0, budget.remaining * 3 // 4
        - agent.config.max_candidates * len(rows)))
    screening_actions = [] if actions is None else actions
    claims = replay_numeric_research_drafts(screening, spec, rows, fitting, screening_actions,
        max_candidates=min(4, agent.config.max_candidates))
    budget.spend(fitting.spent)
    if not claims:
        return None
    fallback = initial_prediction(rows, .5)
    checked = []
    for claim in claims:
        # Screening restricts the source agenda, but a measured failure of the
        # same target binding can have come from another retained procedure.
        if agent.library.transfer_refuted(spec, claim["candidate"].program, fallback=fallback):
            continue
        if not budget.spend(len(rows)):
            break
        measured = loss(claim["candidate"].program, rows, fallback, .5)
        checked.append({**claim, "training_loss": measured, "validation_loss": measured})
    if not checked:
        return None
    checked.sort(key=lambda item: (item["training_loss"], item["candidate"].program.size))
    chosen = checked[0]
    length = min(24, laboratory.spec.max_probe_length)
    plan = plan_observed_condition_probe(spec, rows, incumbent, budget,
        serial=len(previous["batches"]), length=length, finalists=checked, tolerance=tolerance)
    # Failed parameters cannot price an extrapolation, but the method can
    # already have improved the shared paid responses. Let that actual gain
    # and its observed returns price a residual experiment below.
    if plan is None:
        return None
    if any(list(plan["probe"].sequence) == batch["probe"]["sequence"]
           for batch in previous["batches"]):
        return None
    source_ids = list(dict.fromkeys(item["research_draft_id"] for item in checked))
    sources = [record["source_task_id"] for record in drafts if record["draft_id"] in source_ids]
    separation = plan["expected_disagreement"]
    gain = baseline - chosen["training_loss"]
    fit_quote = min(80000, agent.config.compute_budget // 4)
    duration_quote = 40000
    compute_quote = (duration_quote + fit_quote
        + (6 + len(checked)) * len(rows) + 32 * length)
    observation_quote = plan["probe"].cost + 2 * length
    rate = sum(channel.cost_per_row for channel in laboratory.channels if channel.name in channels)
    applicability = method_response_applicability(chosen["candidate"].program, spec, incumbent)
    # Reading experience must not create a new knowledge change or mutate a
    # source draft during this priced screening.
    experience_library = copy(agent.library)
    experience_library.action_experience = deepcopy(agent.library.action_experience)
    policy = method_visit_policy(experience_library, spec, rows, applicability)
    available = ({"observation_cost": agent.config.observation_budget,
        "measurement_cost": agent.measurement_budget,
        "compute_cost": agent.config.compute_budget} if remaining_resources is None
        else {**remaining_resources, "compute_cost": max(0,
                  remaining_resources["compute_cost"] - budget.spent)})
    opportunity = {"draft_ids": source_ids, "source_task_ids": sources,
        "paid_counterexample_loss": baseline, "candidate_paid_loss": chosen["training_loss"],
        "predicted_separation": separation, "planning_compute_cost": budget.spent,
        "duration_compute_quote": duration_quote, "response_compute_quote": fit_quote,
        "method_applicability": applicability,
        "requires_fresh_confirmation": True}
    offer = {"candidate": {"program": chosen["candidate"].program.to_dict(),
        "channels": list(channels), "fallback": fallback, "source_concept_ids": [],
        "kind": "successor_partial_method", "family_id": chosen.get("family_id", "T03"),
        "research_program": deepcopy(chosen["research_program"]),
        "research_draft_id": chosen["research_draft_id"],
        "method_experiment": {**plan, "probe": asdict(plan["probe"]),
                              "source_draft_ids": source_ids},
        "method_opportunity": opportunity},
        "basis": "new_partial_method_distinguishing_experiment",
        "old_evidence_gain": gain, "old_evidence_rows": len(rows),
        "predicted_separation": separation,
        "pricing_inputs": {"method_context": policy.action_context(METHOD_VISIT_ACTION),
            "compute_quote": compute_quote, "observation_quote": observation_quote,
            "task_compute_cap": agent.config.compute_budget, "measurement_rate": rate,
            "batch_length": length, "tolerance": tolerance,
            "condition_uncertainty": deepcopy(chosen.get("condition_uncertainty")),
            "condition_research_returns": deepcopy(previous.get("condition_research_returns"))}}
    priced = refresh_numeric_method_quote(offer, agent.library.action_experience, available)
    # The session consumes this payload into its existing screened-method
    # attempt. A failed value forecast must not discard a purchased real plan.
    if actions is not None:
        actions.append({"action": "cache_numeric_method_quote", "quote": deepcopy(offer),
            "reason": "positive_expected_gain" if priced is not None else "no_positive_expected_gain"})
    return priced
