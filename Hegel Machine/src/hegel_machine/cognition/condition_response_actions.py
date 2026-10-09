"""Let paid response counterexamples change the executable research method."""

import json

from .action_policy import ActionOption, ResearchActionPolicy
from .evaluation import initial_prediction, loss
from .models import Candidate, SearchBudget
from .memory import _object_schema


def _observed_duration_evidence(experiment, tolerance):
    supported = [item["observed_condition_evidence"] for item in experiment
        if item["observed_condition_evidence"]["status"] == "conditional_duration"
        and item["observed_condition_evidence"]["loss"] <= tolerance]
    evidence = []
    for field in sorted({item["control"] for item in supported}):
        family = sorted((item for item in supported if item["control"] == field),
                        key=lambda item: item["loss"])
        best = dict(family[0])
        best["width_candidates"] = list(dict.fromkeys(
            width for item in family for width in item["width_candidates"]))
        best["placement_candidates"] = list(dict.fromkeys(
            placement for item in family for placement in item["placement_candidates"]))
        evidence.append(best)
    return evidence


def _observed_duration_nominations(actions, tolerance):
    """Keep informative measured structures available for a full response fit."""
    # A structure identified on a fixed background is still a competitor
    # for the varied response. Later failures must not erase its proposal.
    nominations = []
    for index, action in enumerate(actions):
        if action["action"] != "distinguish_paid_condition_duration":
            continue
        for item in action.get("competing_structures", []):
            if (item.get("proposed_structure") == "conditional_duration"
                    and (item.get("gain", 0.) > tolerance or item.get("response_supported"))):
                nominations.append({**{key: value for key, value in item.items()
                    if key not in {"parameter_trials", "duration_trials"}},
                    "observed_experiment_index": index})
    return nominations


def _partial_response_competitors(agent, spec, training, budget, actions, *, reserve=0):
    """Share actual complete fits between the next experiment and response selection."""
    from .language import Program
    from .observed_partial_structure import partial_structure_competitors, partial_structure_nominations

    if not partial_structure_nominations(actions, agent.config.minimum_improvement):
        return []
    for event in reversed(actions):
        if (event["action"] == "fit_partial_structure_competitors"
                and event["fit_rows"] == len(training)
                and event["fit_features"] == list(spec.feature_names) and event["competitors"]):
            return [{**item, "candidate": Candidate(**{**item["candidate"],
                "program": Program.from_dict(item["candidate"]["program"]),
                "parents": tuple(item["candidate"]["parents"])})}
                for item in event["competitors"]]
    local = SearchBudget(min(20000, max(0, budget.remaining - reserve)))
    claims = partial_structure_competitors(spec, training, actions, local,
        max_candidates=agent.config.max_candidates, tolerance=agent.config.minimum_improvement)
    budget.spend(local.spent)
    if local.spent:
        saved = [{**claim, "candidate": {**vars(claim["candidate"]),
                    "program": claim["candidate"].program.to_dict()}} for claim in claims]
        actions.append({"action": "fit_partial_structure_competitors",
            "fit_rows": len(training), "fit_features": list(spec.feature_names),
            "competitors": saved, "candidate_count": len(claims), "compute_cost": local.spent,
            "comparison_basis": "all_paid_original_response",
            "status": "requires_independent_confirmation"})
    return claims


def research_observed_duration_experiments(agent, spec, selected, training, finalists,
        budget, actions, *, acquire, channels, observation_allowance,
        validation_rows=0, serial=0, first_plan=None, validation=()):
    """Let actual fixed-background misfit change the next paid experiment."""
    from .observed_condition_experiments import plan_observed_condition_probe
    from .observed_condition_duration import propose_observed_condition_duration
    from .unified_research import _terms

    length = min(24, spec.max_probe_length)
    tolerance = agent.config.minimum_improvement
    fitting_rows = len(training) + observation_allowance
    basis_count = len(_terms(spec.feature_names, quadratic=True)) + 1
    # Fund complete response comparisons without assigning their entire 80k
    # maximum before the duration question has received its own search budget.
    response_floor = fitting_rows * (2 * basis_count + 2) * 4
    assessment_reserve = fitting_rows + validation_rows * (agent.config.max_candidates + 1)
    duration_floor = min(40000, length * (length * 53 + 2))
    experiment_count = observation_allowance // length
    planning_reserve = 15000 * max(0, experiment_count - bool(first_plan))
    refit_reserve = assessment_reserve + max(response_floor,
        min(80000, max(0, budget.remaining - duration_floor * experiment_count
                      - planning_reserve - assessment_reserve)))
    previous = [item for item in actions if item["action"] == "distinguish_paid_condition_duration"
        and not item.get("response_supported", bool(item.get("evidence")))]
    evidence, used = [], 0
    before_partial = budget.spent
    partial = _partial_response_competitors(agent, spec, training, budget, actions,
        reserve=assessment_reserve + response_floor + duration_floor * experiment_count + planning_reserve)
    refit_reserve = max(assessment_reserve + response_floor,
                       refit_reserve - (budget.spent - before_partial))
    while acquire is not None and used + length <= observation_allowance:
        if budget.remaining < refit_reserve + duration_floor:
            break
        plan = first_plan
        first_plan = None
        if plan is None:
            planning = SearchBudget(min(15000, max(0, budget.remaining - refit_reserve - duration_floor)))
            plan = plan_observed_condition_probe(spec, training, selected, planning,
                serial=serial + len(actions), length=length, finalists=finalists,
                previous_experiments=previous, partial_competitors=partial,
                validation=validation, tolerance=tolerance)
            budget.spend(planning.spent)
        if plan is None:
            break
        actions.append({"action": "plan_observed_condition_experiment",
            **{key: value for key, value in plan.items() if key != "probe"},
            "probe_id": plan["probe"].probe_id})
        purpose = ("successor_partial_method_experiment" if plan.get("source_draft_ids")
                   else "distinguish_observed_condition_duration")
        acquired = acquire(channels, purpose, probe=plan["probe"])
        if not acquired:
            break
        used += len(acquired)
        training = [*training, *acquired]
        fitting = SearchBudget(min(40000, max(0, budget.remaining - refit_reserve)))
        experiment = propose_observed_condition_duration(spec, acquired, fitting,
            max_candidates=agent.config.max_candidates, tolerance=tolerance)
        budget.spend(fitting.spent)
        nominees = [item["observed_condition_evidence"] for item in experiment]
        evidence = _observed_duration_evidence(experiment, tolerance)
        supported = bool(evidence)
        event = {"action": "distinguish_paid_condition_duration", "measured_rows": len(acquired),
            "probe_id": plan["probe"].probe_id,
            "condition_field": plan["condition_field"],
            "fixed_controls": plan["fixed_control_background"],
            "fixed_control_background": plan["fixed_control_background"],
            "evidence": evidence, "competing_structures": nominees, "response_supported": supported,
            "candidate_count": len(experiment), "compute_cost": fitting.spent,
            "response_compute_reserve": refit_reserve,
            "comparison_basis": "original_response"}
        actions.append(event)
        if supported:
            break
        # These are complete response fits within the existing response
        # allocation. Counting their work again as an extra reserve would
        # prevent the next already affordable distinguishing experiment.
        before_partial = budget.spent
        next_experiment = used + length <= observation_allowance
        partial = _partial_response_competitors(agent, spec, training, budget, actions,
            reserve=assessment_reserve + response_floor
                + (duration_floor + 15000 if next_experiment and not supported else 0))
        refit_reserve = max(assessment_reserve + response_floor,
                           refit_reserve - (budget.spent - before_partial))
        previous.append(event)
    return training, evidence


def _observed_draft_claims(agent, spec, training, validation, budget, actions, *, evidence=()):
    from .research_drafts import replay_numeric_research_drafts

    if not agent.config.use_memory or not agent.library.research_drafts:
        return []
    reserve = len(validation) * (agent.config.max_candidates + 1)
    # The compact method gets a complete attempt at the new measured
    # structure. Its replay cannot consume the later full-language response
    # fit before that same paid structure has been compared.
    allowance = min(40000 if evidence else 80000, max(0, budget.remaining - reserve))
    if not evidence:
        allowance = min(allowance, budget.remaining // 3)
    local = SearchBudget(allowance + reserve)
    fitting = SearchBudget(allowance)
    proposals = replay_numeric_research_drafts(agent, spec, training, fitting, actions,
        evidence=evidence, validation=validation, max_candidates=agent.config.max_candidates)
    local.spend(fitting.spent)
    checked = agent._assess(proposals, validation, local, initial_prediction(training, .5), .5)
    budget.spend(local.spent)
    independent = all(row not in training for row in validation)
    for claim in checked:
        identifier = claim["research_draft_id"]
        draft = next(item for item in agent.library.research_drafts if item["draft_id"] == identifier)
        adequate = max(claim["training_loss"], claim["validation_loss"]) <= agent.config.minimum_improvement
        outcome = {"task_id": spec.task_id, "training_loss": claim["training_loss"],
            "validation_loss": claim["validation_loss"], "rows": len(validation),
            "status": ("supported_on_current_validation" if adequate and independent
                       else "current_counterexample" if independent else "pending")}
        draft["outcomes"].append(outcome)
        actions.append({"action": "assess_research_draft", "draft_id": identifier,
            "comparison_basis": "original_response", **outcome})
    return checked


def _observed_response_claims(agent, spec, training, selected, budget, actions, validation,
                              condition_evidence=()):
    from .unified_research import _terms
    from .conditional_coordinates import (propose_conditional_coordinates,
                                           propose_experimental_conditional_coordinates)
    from .decoded_conditional_coordinates import propose_decoded_conditional_coordinates

    observed = set(spec.feature_names) - set(spec.feature_bounds)
    nominations = _observed_duration_nominations(actions, agent.config.minimum_improvement)
    upper = max((len(row.history) for row in training), default=0)
    steps = (selected.get("research_program") or {}).get("selection_steps", []) if selected else []
    evidence = [trial for step in steps for key in ("width_trials", "duration_screen", "duration_trials")
                for trial in step.get(key, [])]
    preferred = [item["width"] for item in condition_evidence if item.get("width")]
    preferred.extend(item["width"] for item in nominations if item.get("width"))
    preferred.extend(step["width"] for step in steps if step.get("width"))
    preferred.extend(trial["width"] for trial in sorted(evidence,
        key=lambda trial: trial.get("loss", float("inf"))))
    widths = list(dict.fromkeys([*preferred, *range(1, upper + 1)]))
    widths = [width for width in widths if 1 <= width <= upper]
    basis = _terms(spec.feature_names, quadratic=True)
    start = budget.spent
    proposals = _observed_draft_claims(agent, spec, training, validation, budget, actions,
                                      evidence=condition_evidence)
    if any(max(item["training_loss"], item["validation_loss"]) <= agent.config.minimum_improvement
           for item in proposals):
        return proposals
    # An identified current experiment keeps its already funded complete fit.
    # Ambiguous support competes first when that identification is absent.
    partial = [] if condition_evidence else _partial_response_competitors(
        agent, spec, training, budget, actions,
        reserve=len(validation) * (agent.config.max_candidates + 1))
    proposals.extend(agent._assess(partial, validation, budget, initial_prediction(training, .5), .5))
    if any(max(item["training_loss"], item["validation_loss"]) <= agent.config.minimum_improvement
           for item in proposals):
        return proposals
    if condition_evidence or nominations:
        # The constant-background fit supplies experimental structure, never
        # the final response. Both branch laws are refitted on all varied
        # controls and compared on the same original validation responses.
        supported = SearchBudget(min(80000, max(0,
            budget.remaining - len(validation) * agent.config.max_candidates)))
        experimental = propose_experimental_conditional_coordinates(spec, training, supported,
            basis=basis, evidence=condition_evidence, nominations=nominations,
            max_candidates=agent.config.max_candidates)
        budget.spend(supported.spent)
        fitted_structures = {}
        for proposal in experimental:
            for step in proposal["research_program"]["selection_steps"]:
                for trial in step.get("parameter_trials", []):
                    key = (trial["gate_coordinate"]["name"], trial["width"], trial["placement"])
                    if key not in fitted_structures or trial["loss"] < fitted_structures[key]["loss"]:
                        fitted_structures[key] = {"control": key[0], "width": key[1],
                            "placement": key[2], "loss": trial["loss"]}
        actions.append({"action": "refit_observed_condition_response",
            "evidence": list(condition_evidence), "fitting_rows": len(training),
            "structure_nominations": list(nominations),
            "fitting_priority": ("current_supported_structure" if condition_evidence
                                 else "paid_structure_competition"),
            "candidate_count": len(experimental), "compute_cost": supported.spent,
            "fitted_structures": list(fitted_structures.values()),
            "comparison_basis": "original_response"})
        proposals.extend(agent._assess(experimental, validation, budget,
                                      initial_prediction(training, .5), .5))
        if any(max(item["training_loss"], item["validation_loss"]) <= agent.config.minimum_improvement
               for item in proposals):
            return proposals
    if condition_evidence:
        partial = _partial_response_competitors(agent, spec, training, budget, actions,
            reserve=len(validation) * (agent.config.max_candidates + 1))
        proposals.extend(agent._assess(partial, validation, budget, initial_prediction(training, .5), .5))
    decoded = SearchBudget(min(80000, max(0, budget.remaining - len(validation) * agent.config.max_candidates)))
    decoded_proposals = propose_decoded_conditional_coordinates(spec, training, decoded,
        basis=basis, gate_names=observed, width_candidates=widths,
        duration_evidence=[*condition_evidence, *nominations, *evidence], max_candidates=agent.config.max_candidates)
    budget.spend(decoded.spent)
    actions.append({"action": "propose_decoded_observed_response",
        "width_candidates": widths, "observed_condition_fields": sorted(observed),
        "candidate_count": len(decoded_proposals), "compute_cost": decoded.spent,
        "comparison_basis": "original_response"})
    proposals.extend(agent._assess(decoded_proposals, validation, budget, initial_prediction(training, .5), .5))
    if any(max(item["training_loss"], item["validation_loss"]) <= agent.config.minimum_improvement
           for item in proposals):
        actions.append({"action": "retain_supported_observed_representation",
            "comparison_basis": "original_unfitted_validation", "evidence_rows": len(validation)})
        return proposals
    # An affine pilot cannot rule out a duration for a quadratic branch.
    # Without resolving experiments, exterior structures remain competitors.
    exterior = propose_conditional_coordinates(spec, training, budget,
        basis=basis, max_candidates=agent.config.max_candidates, gate_names=observed,
        placement_candidates=("outside_window",))
    proposals.extend(agent._assess(exterior, validation, budget, initial_prediction(training, .5), .5))
    actions.append({"action": "compare_observed_response_representations",
        "candidate_count": len(proposals), "compute_cost": budget.spent - start})
    return proposals


def _observed_method_applicability(agent, spec, selected, claims):
    """Describe methods actually refitted against the current paid response.

    A failed experiment with another response basis does not settle whether
    these methods need a different condition or duration. Keep the executing
    structures and public input roles, rather than a task ID or each new batch,
    so a failed trial of this same question remains a failed trial on reuse.
    """
    from .numeric_method_revisits import _method_applicability

    methods = []
    for claim in claims:
        if not claim.get("research_draft_id") or not claim.get("research_program"):
            continue
        current = {"source_features": list(spec.feature_names),
            "control_features": list(spec.feature_bounds),
            "observed_features": [name for name in spec.feature_names
                                  if name not in spec.feature_bounds],
            "research_program": claim["research_program"]}
        method = _method_applicability(current, spec, selected)
        method["current_response_status"] = (
            "supported" if max(claim["training_loss"], claim["validation_loss"])
            <= agent.config.minimum_improvement else "misfit")
        if method not in methods:
            methods.append(method)
    return sorted(methods, key=lambda method: json.dumps(method, sort_keys=True))


def research_condition_response_actions(agent, spec, selected, training, validation,
        finalists, budget, actions, *, acquire=None, remaining=None, serial=0,
        channels=(), measurement_rate=0):
    from .unified_research import _fit, _terms

    if (agent.config.strategy == "fixed" or spec.risk_quantile != .5
            or spec.decision_costs or spec.context.get("object_model") or _object_schema(spec)
            or len(training) < 8 or not validation):
        return selected, training, finalists
    observed = set(spec.feature_names) - set(spec.feature_bounds)
    if not observed and (len(spec.feature_names) < 4 or acquire is None):
        return selected, training, finalists
    fallback = initial_prediction(training, .5)
    if selected is None:
        initial = _fit(_terms(spec.feature_names), training, budget,
                       "condition_response_baseline", q=.5)
        initial = agent._assess([initial], validation, budget, fallback, .5)
        selected = agent._best(initial)
        finalists = initial
    if selected is None:
        return selected, training, finalists
    if (observed and selected.get("condition_uncertainty") is not None
            and max(selected["training_loss"], selected["validation_loss"]) > agent.config.minimum_improvement):
        selected, training, finalists = research_observed_condition_competition(agent, spec,
            selected, training, validation, finalists, budget, actions, acquire=acquire,
            remaining=remaining, serial=serial, channels=channels, measurement_rate=measurement_rate)
    drafts = []
    if observed and max(selected["training_loss"], selected["validation_loss"]) > agent.config.minimum_improvement:
        drafts = _observed_draft_claims(agent, spec, training, validation, budget, actions)
        if drafts:
            best = min(drafts, key=lambda item: (max(item["training_loss"], item["validation_loss"]),
                                                 item["candidate"].program.size))
            if (max(best["training_loss"], best["validation_loss"]) + agent.config.minimum_improvement
                    < max(selected["training_loss"], selected["validation_loss"])):
                selected = best
                finalists = [best, *finalists]
                actions.append({"action": "select_partial_response_method",
                    "draft_id": best["research_draft_id"], "training_loss": best["training_loss"],
                    "validation_loss": best["validation_loss"], "requires_fresh_confirmation": True})
    if max(selected["training_loss"], selected["validation_loss"]) <= agent.config.minimum_improvement:
        return research_observed_condition_competition(agent, spec, selected, training,
            validation, finalists, budget, actions, acquire=acquire, remaining=remaining,
            serial=serial, channels=channels, measurement_rate=measurement_rate)
    calls = agent._recall(spec)
    replay = agent._replay_program_queue(spec, calls)
    learn = agent.config.use_memory and agent.config.learn_research_policy
    policy = ResearchActionPolicy(training, spec, agent.library.action_experience if learn else {})
    policy.set_condition("condition_response:" + ("paid_observation" if observed else "control_counterexample"))
    if replay:
        policy.set_action_knowledge("revise_response_method", agent._replay_knowledge(spec, replay))
    nominations = _observed_duration_nominations(actions, agent.config.minimum_improvement)
    if observed and nominations:
        structures = []
        for item in nominations:
            structure = {key: item[key] for key in ("control", "width", "placement")}
            if structure not in structures:
                structures.append(structure)
        policy.set_action_knowledge("joint_observed_condition_response",
                                    {"measured_structure_nominations": structures})
    methods = _observed_method_applicability(agent, spec, selected, drafts)
    if (selected.get("selection_basis") != "all_acquired_evidence"
            and all(row not in training for row in validation)):
        counterexample = {"frozen_response": selected["candidate"].program.to_dict(),
            "measurement_fields": sorted(observed),
            "unfitted_readings": [{"features": row.features, "target": row.target}
                                  for row in validation]}
        for action in ("revise_response_method", "joint_observed_condition_response"
                       if observed else "change_condition_experiment"):
            policy.set_action_counterexample(action, counterexample,
                fit_loss=selected["training_loss"], counterexample_loss=selected["validation_loss"],
                tolerance=agent.config.minimum_improvement)
    attempted = set()
    confirmation = min(24, spec.max_probe_length)
    planned_question, planned_on = None, None
    condition_planning_cost = 0

    def resources():
        current = remaining() if remaining is not None else {}
        return {"observation_remaining": current.get("observation_remaining", 0),
                "measurement_remaining": current.get("measurement_remaining", 0),
                "compute_remaining": budget.remaining}

    while len(attempted) < 2:
        current = resources()
        reserve = 4 * max(confirmation, len(validation))
        options = []
        if "revise_response_method" not in attempted:
            options.append(ActionOption("revise_response_method", 0,
                min(24000, max(0, budget.remaining - reserve))))
        if observed:
            support = 4 * (len(_terms(spec.feature_names, quadratic=True)) + 1)
            needed = min(64, max(24, support - len(training))) if acquire is not None else 0
            extra = (min(2 * confirmation, max(0,
                current["observation_remaining"] - needed - confirmation)) if acquire is not None else 0)
            if measurement_rate:
                extra = min(extra, max(0, current["measurement_remaining"] // measurement_rate - needed - confirmation))
            extra = int(extra // confirmation) * confirmation
            plan = None
            # Price a concrete experiment before transferring its old return.
            # The current refits can contradict every candidate without any
            # one candidate first fitting perfectly; that is still a research
            # question about these methods, not the old generic action average.
            if (methods and acquire is not None and extra >= confirmation
                    and "joint_observed_condition_response" not in attempted):
                from .observed_condition_experiments import plan_observed_condition_probe
                previous = [item for item in actions
                    if item["action"] == "distinguish_paid_condition_duration"]
                current_question = (selected["candidate"].program, len(training))
                planning_cost = 0
                if current_question == planned_on:
                    plan = planned_question
                else:
                    preparation_start = budget.spent
                    partial = _partial_response_competitors(agent, spec, training, budget, actions,
                        reserve=reserve + 40000 + 15000)
                    condition_planning_cost += budget.spent - preparation_start
                    planning = SearchBudget(min(15000, max(0, budget.remaining - reserve - 40000)))
                    plan = plan_observed_condition_probe(spec, training, selected, planning,
                        serial=serial + len(actions), length=confirmation,
                        finalists=(*drafts, *finalists), previous_experiments=previous,
                        partial_competitors=partial, validation=validation,
                        tolerance=agent.config.minimum_improvement)
                    planned_question, planned_on = plan, current_question
                    budget.spend(planning.spent)
                    planning_cost = planning.spent
                    condition_planning_cost += planning_cost
                current = resources()
                if plan is not None:
                    structure_nominations = []
                    for item in nominations:
                        structure = {"reading_role": [name for name in spec.feature_names
                            if name not in spec.feature_bounds].index(item["control"]),
                            "width": item["width"], "placement": item["placement"]}
                        if structure not in structure_nominations:
                            structure_nominations.append(structure)
                    structure_nominations.sort(key=lambda item:
                        (item["reading_role"], item["width"], item["placement"]))
                    description = {"executing_methods": methods,
                        "question": "distinguish_condition_and_duration_on_current_misfit",
                        "reading_role": [name for name in spec.feature_names
                            if name not in spec.feature_bounds].index(plan["condition_field"]),
                        "experiment_kind": plan["experiment_kind"],
                        "measured_structure_nominations": structure_nominations}
                    context = policy.set_action_knowledge(
                        "joint_observed_condition_response", description)
                    actions.append({"action": "assess_condition_method_applicability",
                        "context": context, "method_applicability": description,
                        "training_loss": selected["training_loss"],
                        "validation_loss": selected["validation_loss"],
                        "expected_disagreement": plan["expected_disagreement"],
                        "compute_cost": planning_cost})
            options.append(ActionOption("joint_observed_condition_response", needed + extra + (confirmation if acquire else 0),
                min(180000, max(0, budget.remaining - reserve)),
                payload=(None, {"support_rows": needed, "condition_rows": extra,
                               "condition_plan": plan}),
                measurement_cost=(needed + extra + (confirmation if acquire else 0)) * measurement_rate))
        elif acquire is not None:
            rows = (2 * (len(spec.feature_names) + 1) + 4) * min(12, spec.max_probe_length) + 28
            # The curve follow-up is a new action after a real counterexample;
            # its acquisition must still fit the remaining observation funds.
            curve_rows = 2 * min(12, spec.max_probe_length)
            terms = 2 * (len(_terms(spec.feature_names, quadratic=True)) + 1)
            compute = min(240000, (len(training) + rows + curve_rows) * (terms + 4) * 24
                          + 12 * len(validation))
            options.append(ActionOption("change_condition_experiment", rows + confirmation,
                                        compute + reserve))
        options = [option for option in options if option.name not in attempted and option.compute_cost > 0]
        choice = policy.choose(options, **current,
                               minimum_improvement=agent.config.minimum_improvement)
        actions.append({"action": "choose_condition_response_research", **policy.last_decision,
            "options": [{"name": option.name, "observation_cost": option.observation_cost,
                         "measurement_cost": option.measurement_cost,
                         "compute_cost": option.compute_cost} for option in options],
            "counterexample_loss": selected["validation_loss"],
            "observed_condition_fields": sorted(observed)})
        if choice is None:
            break
        attempted.add(choice.name)
        before, old = resources(), selected
        comparison_loss = old["validation_loss"]
        local = SearchBudget(max(0, min(choice.compute_cost,
                                       budget.remaining - reserve)))
        if choice.name == "change_condition_experiment":
            selected, training, finalists = agent._research_conditional_experiments(
                spec, selected, training, validation, finalists, local, actions,
                acquire=acquire, observation_allowance=current["observation_remaining"] - confirmation,
                serial=serial)
        else:
            if choice.name == "joint_observed_condition_response":
                if choice.payload[1]["support_rows"]:
                    rows = acquire(channels, "distinguish_observed_condition_response",
                                   rows_count=choice.payload[1]["support_rows"])
                    if rows is not None:
                        training = [*training, *rows]
                condition_evidence = []
                if acquire is not None and choice.payload[1]["condition_rows"]:
                    training, condition_evidence = research_observed_duration_experiments(agent, spec,
                        selected, training, finalists, local, actions, acquire=acquire, channels=channels,
                        observation_allowance=choice.payload[1]["condition_rows"],
                        validation_rows=len(validation), validation=validation, serial=serial,
                        first_plan=choice.payload[1]["condition_plan"])
                checked = []
                if local.spend(len(training)):
                    old = {**old, "training_loss": loss(old["candidate"].program, training, fallback, .5)}
                    proposals = _observed_response_claims(agent, spec, training, selected, local,
                        actions, validation, condition_evidence)
                    checked = [old, *proposals]
            else:
                proposals = agent._coordinate_claims(spec, training, local, actions)
                if replay and local.remaining > len(validation):
                    proposals.extend(agent._replay_program_claims(spec, training, local, calls,
                        prepared=replay)[:agent.config.max_candidates])
                checked = agent._assess([old, *proposals], validation, local, fallback, .5)
            if checked:
                best = min(checked, key=lambda entry:
                    (max(entry["training_loss"], entry["validation_loss"]),
                     entry["candidate"].program.size))
                selected = (best if max(best["training_loss"], best["validation_loss"]) + agent.config.minimum_improvement
                             < max(old["training_loss"], old["validation_loss"]) else old)
                finalists = checked
        budget.spend(local.spent)
        after = resources()
        costs = {name + "_cost": before[name + "_remaining"] - after[name + "_remaining"]
                 for name in ("observation", "measurement", "compute")}
        if choice.name == "joint_observed_condition_response":
            costs["compute_cost"] += condition_planning_cost
            condition_planning_cost = 0
        after_loss = selected["validation_loss"]
        if learn:
            independent = (old.get("selection_basis") != "all_acquired_evidence"
                           and all(row not in training for row in validation))
            origin = {"context": policy.action_context(choice.name), "action": choice.name,
                "trial": None,
                "baseline_program": old["candidate"].program.to_dict(),
                "fallback": fallback, "risk_quantile": .5}
            if independent:
                feedback = policy.update(choice.name, before_loss=comparison_loss,
                                         after_loss=after_loss, **costs)
                origin.update(trial=feedback["trials"], credited_gain=feedback["last_gain"])
                actions.append({"action": "research_action_feedback", "research_action": choice.name,
                    "context": origin["context"], "before_loss": comparison_loss,
                    "after_loss": after_loss, "baseline_program": origin["baseline_program"],
                    "evidence_basis": "common_unfitted_validation", "evidence_rows": len(validation),
                    "fitting_rows": len(training), "feedback": feedback, **costs})
            else:
                origin.update(pending_id=len(actions), pending_status="unresolved", pending_costs=costs)
                actions.append({"action": "research_action_pending", "research_action": choice.name,
                    "context": origin["context"], "pending_id": origin["pending_id"], "trial": None,
                    "evidence_basis": "all_acquired_evidence", **costs})
            if after_loss + agent.config.minimum_improvement < comparison_loss:
                selected = {**selected, "research_action_origin": origin}
        if max(selected["training_loss"], after_loss) <= agent.config.minimum_improvement:
            break
    return research_observed_condition_competition(agent, spec, selected, training,
        validation, finalists, budget, actions, acquire=acquire, remaining=remaining,
        serial=serial, channels=channels, measurement_rate=measurement_rate)


def observed_condition_followup(state, tolerance):
    """Choose the next question from all purchased response evidence."""
    if state is None or state["status"] == "supported":
        return None
    paid_loss = state.get("original_target_loss")
    if paid_loss is None:
        return None
    if (state["status"] == "unresolved" and any(
            item.get("needs_threshold_revision") and item.get("threshold_intervals")
            and item.get("conditional_loss") is not None
            and item["conditional_loss"] <= tolerance
            for item in state.get("conditions", []))):
        return "revise_paid_condition"
    if paid_loss <= tolerance:
        return "distinguish_observed_condition"
    return "reconstruct_observed_response"


def _paid_condition_threshold_claim(selected, state, training, budget, fallback):
    """Apply the cut already supported by purchased original responses."""
    from copy import deepcopy
    from dataclasses import replace
    from .conditional_coordinates import adopt_conditional_research_program
    from .observed_condition_uncertainty import _replace_threshold

    if state is None or state["status"] == "contradicted":
        return []
    program = selected["candidate"].program
    recipe = deepcopy(selected.get("research_program"))
    for condition in state["conditions"]:
        if not condition.get("needs_threshold_revision"):
            continue
        interval = condition["threshold_interval"]
        field, cut = condition["condition_field"], sum(interval) / 2.
        program = _replace_threshold(program, field, condition["threshold"], cut)

        def revise_steps(steps):
            for step in steps:
                coordinate = step.get("gate_coordinate", {})
                if coordinate.get("op") == "feature" and coordinate.get("name") == field:
                    step.update(threshold_interval=list(interval), threshold=cut)
                revise_steps(step.get("inner_selection_steps", []))
        revise_steps((recipe or {}).get("selection_steps", []))
    if program == selected["candidate"].program or not budget.spend(len(training)):
        return []
    measured_loss = loss(program, training, fallback, .5)
    if recipe is not None:
        recipe = adopt_conditional_research_program(recipe, program,
            measured_loss=measured_loss, fit_rows=len(training), compute_cost=budget.spent)
    return [{**selected, "candidate": replace(selected["candidate"], program=program),
        "research_program": recipe, "condition_uncertainty": None,
        "training_loss": measured_loss}]


def research_observed_condition_competition(agent, spec, selected, training, validation,
        finalists, budget, actions, *, acquire=None, remaining=None, serial=0,
        channels=(), measurement_rate=0, funded_experiment=None):
    """Spend on the original-response disagreement left by a fitted meter cut."""
    from .observed_condition_uncertainty import assess_observed_condition_uncertainty
    from .observed_condition_experiments import plan_observed_condition_probe
    from .observed_condition_duration import propose_observed_condition_duration
    from .condition_research_returns import (_question, consider_condition_research,
                                            record_condition_research)
    from .condition_marginal_value import (CONDITION_FOLLOWUP_ACTION,
        condition_followup_policy, record_condition_information)

    if (selected is None or spec.risk_quantile != .5 or spec.decision_costs
            or spec.context.get("object_model") or _object_schema(spec)):
        return selected, training, finalists
    length = min(24, spec.max_probe_length)
    tolerance = agent.config.minimum_improvement
    from .instrument_condition_revisits import condition_experiment_history
    previous_experiments = condition_experiment_history(actions)
    while True:
        rows = [*training, *(row for row in validation if row not in training)]
        assessment_start = budget.spent
        state = selected.get("condition_uncertainty")
        if not (funded_experiment is not None and state is not None
                and state.get("program") == selected["candidate"].program.to_dict()
                and state.get("fit_rows") == len(rows)):
            state = assess_observed_condition_uncertainty(spec, rows, selected, budget,
                previous=state, tolerance=tolerance)
        assessment_compute_cost = budget.spent - assessment_start
        selected = {**selected, "condition_uncertainty": state}
        revised_from_paid = False
        followup = (observed_condition_followup(state, tolerance)
                    if agent.config.strategy != "fixed" else None)
        if followup in ("revise_paid_condition", "reconstruct_observed_response"):
            available = remaining() if remaining is not None else {}
            actions.append({"action": "choose_observed_response_followup",
                "research_action": followup, "evidence_basis": "all_paid_original_response",
                "paid_rows": len(rows), "paid_response_loss": state["original_target_loss"],
                "current_response_loss": selected.get("validation_loss"),
                "compute_remaining": budget.remaining, **available})
        if followup == "reconstruct_observed_response":
            # The caller's response search must see the conflicting paid
            # rows, rather than an earlier small fitting view or cached loss.
            selected["training_loss"] = state["original_target_loss"]
            return selected, rows, finalists
        if followup == "revise_paid_condition":
            # Refining a model from existing responses is a different action
            # from buying another reading. Keep the new-confirmation reserve.
            revision = SearchBudget(min(15000, max(0,
                budget.remaining - 4 * max(length, len(validation)))))
            fallback = initial_prediction(training, .5)
            proposals = _paid_condition_threshold_claim(selected, state, training,
                                                        revision, fallback)
            old = selected
            if proposals and revision.spend(len(training) + len(validation)):
                old = {**selected, "training_loss": loss(selected["candidate"].program,
                    training, fallback, .5), "validation_loss": loss(
                    selected["candidate"].program, validation, fallback, .5)}
                selected = old
                checked = agent._assess(proposals, validation, revision, fallback, .5)
                best = min([old, *checked], key=lambda item: (
                    max(item["training_loss"], item["validation_loss"]),
                    item["candidate"].program.size))
                if (max(best["training_loss"], best["validation_loss"]) + tolerance
                        < max(old["training_loss"], old["validation_loss"])):
                    selected = best
                    finalists = [best, old, *finalists]
                    state = assess_observed_condition_uncertainty(spec, rows, selected,
                        revision, tolerance=tolerance)
                    selected["condition_uncertainty"] = state
            changed = selected["candidate"].program != old["candidate"].program
            if changed:
                revised_from_paid = True
                # This is a response revision, not another purchased batch.
                # Keep its cost and the old global sampling history, while
                # starting calibration of the newly retained question.
                returns = selected.get("condition_research_returns")
                if returns is not None:
                    selected["condition_research_returns"] = {**returns,
                        "compute_cost": returns.get("compute_cost", 0) + revision.spent,
                        "last_question": _question(state),
                        "current_question_returns": {"batches": 0,
                            "information_batches": 0, "predicted_information_batches": 0.}}
                previous_experiments.append({"research_return": {"changed_program": True}})
            budget.spend(revision.spent)
            actions.append({"action": "revise_condition_on_paid_response",
                "changed_program": changed, "evidence_rows": len(rows), "measured_rows": 0,
                "before_loss": old["training_loss"], "after_loss": selected["training_loss"],
                "before_validation_loss": old["validation_loss"],
                "after_validation_loss": selected["validation_loss"],
                "compute_cost": revision.spent,
                "requires_fresh_confirmation": changed,
                "comparison_basis": "paid_original_response"})
        if state is None or state["status"] == "supported":
            break
        actions.append({"action": "retain_observed_condition_question", "status": state["status"],
            "expected_prediction_ambiguity": state["expected_prediction_ambiguity"],
            "paid_rows": len(rows), "conditions": state["conditions"],
            "compute_cost": assessment_compute_cost})
        available = remaining() if remaining is not None else {}
        reserve = (funded_experiment["confirmation_compute_quote"] if funded_experiment is not None
                   else 4 * max(length, len(validation)))
        rate = measurement_rate
        if (agent.config.strategy == "fixed" or acquire is None
                or available.get("observation_remaining", 0) < 2 * length
                or available.get("measurement_remaining", 0) < 2 * length * rate
                or budget.remaining < 20000 + reserve):
            break
        local = SearchBudget(min(80000, budget.remaining - reserve))
        # A response that fits every purchased target needs distinguishing
        # evidence before another structural fit. Keep existing competitors;
        # fresh contradictions send the caller back to response reconstruction.
        fitted_response = (state.get("original_target_loss") is not None
            and state["original_target_loss"] <= tolerance
            and all(not item.get("needs_threshold_revision")
                    and item.get("status") != "contradicted"
                    for item in state.get("conditions", [])))
        partial = []
        if not fitted_response and funded_experiment is None:
            partial = _partial_response_competitors(agent, spec, training, local, actions,
                reserve=15000 + len(rows) * 4 + len(validation) * (agent.config.max_candidates + 1))
            partial = agent._assess(partial, validation, local, initial_prediction(training, .5), .5)
        viable = []
        for entry in ([] if funded_experiment is not None else [*finalists, *partial]):
            if max(entry.get("training_loss", float("inf")),
                   entry.get("validation_loss", float("inf"))) > tolerance:
                continue
            current = (state if entry["candidate"].program == selected["candidate"].program
                else assess_observed_condition_uncertainty(spec, rows, entry, local,
                    previous=entry.get("condition_uncertainty"), tolerance=tolerance))
            if current is not None and (current["status"] == "contradicted"
                    or current.get("original_target_loss") is None
                    or current["original_target_loss"] > tolerance):
                continue
            viable.append({**entry, "condition_uncertainty": current})
        plan = (funded_experiment["plan"] if funded_experiment is not None else
            plan_observed_condition_probe(spec, rows, selected, local,
                finalists=viable, serial=serial + len(actions), length=length,
                previous_experiments=previous_experiments, tolerance=tolerance))
        if plan is None:
            budget.spend(local.spent)
            break
        returns = selected.get("condition_research_returns")
        quote = min(local.remaining, 15000 + 6 * (len(rows) + length))
        marginal_policy = condition_followup_policy(spec, rows, selected,
            agent.library.action_experience if agent.config.use_memory
            and agent.config.learn_research_policy else {})
        decision = (funded_experiment["decision"] if funded_experiment is not None else
            consider_condition_research(state, plan, returns,
            observation_remaining=available.get("observation_remaining", 0),
            measurement_remaining=available.get("measurement_remaining", 0),
            compute_remaining=budget.remaining - local.spent,
            measurement_rate=rate, compute_quote=quote,
            observation_reserve=length, measurement_reserve=length * rate,
            compute_reserve=reserve, tolerance=tolerance,
            previous_experiments=previous_experiments, new_counterexample=revised_from_paid,
            marginal_record=marginal_policy.action_record(CONDITION_FOLLOWUP_ACTION),
            compute_spent_before_decision=local.spent))
        actions.append({"action": "choose_condition_research_investment", **decision,
            "investment_scope": "session_remaining" if funded_experiment is not None else "current_visit",
            **{key: plan[key] for key in ("condition_field", "threshold_intervals",
                "fixed_control_background", "experiment_revision_basis", "previous_experiment_count")},
            "planning_compute_cost": local.spent,
            "research_focus": "distinguish_current_response" if fitted_response else "refit_competing_responses",
            "marginal_context": marginal_policy.action_context(CONDITION_FOLLOWUP_ACTION)})
        if not decision["buy"]:
            budget.spend(local.spent)
            break
        actions.append({"action": "plan_competing_observed_conditions",
            **{key: value for key, value in plan.items() if key != "probe"},
            "probe_id": plan["probe"].probe_id})
        acquired = acquire(channels, "distinguish_competing_observed_conditions", probe=plan["probe"])
        if not acquired:
            budget.spend(local.spent)
            break
        training = [*training, *acquired]
        updated_rows = [*training, *(row for row in validation if row not in training)]
        updated = assess_observed_condition_uncertainty(spec, updated_rows, selected, local,
            previous=state, tolerance=tolerance)
        old = {**selected, "condition_uncertainty": updated}
        fallback = initial_prediction(training, .5)
        if local.spend(len(training)):
            old["training_loss"] = loss(old["candidate"].program, training, fallback, .5)
        else:
            old["condition_research_returns"] = record_condition_research(returns, state, updated,
                plan, observation_cost=len(acquired), measurement_cost=len(acquired) * rate,
                compute_cost=local.spent, tolerance=tolerance)
            record_condition_information(marginal_policy,
                old["condition_research_returns"]["last_outcome"])
            budget.spend(local.spent)
            selected = old
            break
        proposals = _paid_condition_threshold_claim(old, updated, training, local, fallback)
        if old["training_loss"] > tolerance and not proposals:
            duration = SearchBudget(min(40000, max(0, local.remaining - len(validation) * 4)))
            experiment = propose_observed_condition_duration(spec, acquired, duration,
                max_candidates=agent.config.max_candidates, tolerance=tolerance)
            local.spend(duration.spent)
            evidence = _observed_duration_evidence(experiment, tolerance)
            actions.append({"action": "distinguish_paid_condition_duration",
                "measured_rows": len(acquired), "condition_field": plan["condition_field"],
                "fixed_controls": plan["fixed_control_background"],
                "fixed_control_background": plan["fixed_control_background"],
                "evidence": evidence,
                "competing_structures": [item["observed_condition_evidence"] for item in experiment],
                "response_supported": bool(evidence), "candidate_count": len(experiment),
                "compute_cost": duration.spent, "response_compute_reserve": reserve,
                "comparison_basis": "original_response"})
            proposals = _observed_response_claims(agent, spec, training, old, local,
                actions, validation, evidence)
        else:
            proposals = agent._assess(proposals, validation, local, fallback, .5)
        checked = [old, *proposals]
        if checked:
            best = min(checked, key=lambda item: (max(item["training_loss"], item["validation_loss"]),
                                                 item["candidate"].program.size))
            selected = (best if max(best["training_loss"], best["validation_loss"]) + tolerance
                         < max(old["training_loss"], old["validation_loss"]) else old)
            finalists = checked
        else:
            selected = old
        changed = selected["candidate"].program != old["candidate"].program
        after_state = (assess_observed_condition_uncertainty(spec, updated_rows, selected, local,
            tolerance=tolerance) if changed else updated)
        selected["condition_uncertainty"] = after_state
        selected["condition_research_returns"] = record_condition_research(returns, state, after_state,
            plan, observation_cost=len(acquired), measurement_cost=len(acquired) * rate,
            compute_cost=local.spent, changed_program=changed, tolerance=tolerance,
            observed_after=updated)
        record_condition_information(marginal_policy,
            selected["condition_research_returns"]["last_outcome"])
        budget.spend(local.spent)
        actions.append({"action": "revise_competing_observed_conditions",
            "measured_rows": len(acquired), "before_loss": old["training_loss"],
            "after_loss": selected["training_loss"], "compute_cost": local.spent,
            "changed_program": changed,
            "research_return": selected["condition_research_returns"]["last_outcome"],
            "requires_fresh_confirmation": True, "comparison_basis": "paid_original_response"})
        previous_experiments.append({**plan,
            "research_return": selected["condition_research_returns"]["last_outcome"]})
        if funded_experiment is not None:
            # This funded visit has revised on new readings. Return to its
            # independent frozen-law confirmation before another research cycle.
            break
    return selected, training, finalists
