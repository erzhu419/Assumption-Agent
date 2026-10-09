"""Compare executable observation and mechanism research on paid response evidence."""

from copy import deepcopy
from dataclasses import replace
from itertools import combinations

from .action_policy import ActionOption, ResearchActionPolicy
from .control_revisions import RetainedControlRevisions
from .evaluation import initial_prediction, loss
from .instrument_experiment_methods import observe_instrument_method
from .instrument_hypotheses import (assess_instrument_hypotheses, instrument_candidates,
                                    plan_instrument_disagreement)
from .language import Program
from .meta_prior_research import _sources
from .models import Candidate, Observation, SearchBudget
from .observed_response_competition import compare_observed_responses
from .observed_condition_uncertainty import assess_observed_condition_uncertainty


def research_observation_actions(agent, laboratory, *, acquire, paired_for, command_batches,
                                 budget, actions, transfer, graph, remaining, length,
                                 parent=None, failed_probe=None, alternatives=()):
    """Try each feasible direction once, reserving a new frozen confirmation.

    Field counterexamples update observation explanations. Only a subsequent
    common response trajectory supplies a return to the action policy.
    """
    result = {"selected": None, "channels": (), "training": [], "validation": [],
              "confirmed": False, "handled": False, "pending_inputs": None,
              "confirmation_loss": None, "confirmation_rows": 0,
              "alternatives": deepcopy(list(alternatives))}
    workflow = getattr(agent, "_workflow_episode", None)
    command_spec = agent._spec(laboratory, ())
    channels = tuple(channel.name for channel in laboratory.channels)
    rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
    views = sorted((view for size in range(1, len(channels) + 1) for view in combinations(channels, size)),
                   key=lambda view: (sum(rates[name] for name in view), len(view)))
    measured_spec = agent._spec(laboratory, channels)
    rate = sum(channel.cost_per_row for channel in laboratory.channels)
    count = min(length, command_spec.max_probe_length)
    tolerance = agent.config.minimum_improvement
    initial_rows = [row for batch in command_batches() for row in batch]
    fallback = initial_prediction(initial_rows, command_spec.risk_quantile)
    baseline = parent["candidate"].program if parent else Program("const", value=fallback)
    result["fallback"] = fallback
    baseline_measured = bool({name for name, _ in baseline.observed_variables()}
                             - set(command_spec.feature_names) - {"__target__"})
    learn = agent.config.use_memory and agent.config.learn_research_policy
    experience = agent.library.action_experience if learn else {}
    policy = ResearchActionPolicy(initial_rows, command_spec, experience)
    methods = agent.library.program_experiment_methods if agent.config.use_memory else None
    attempted, cached_plan, cached_model = set(), None, None
    prepared = False
    control_revisions, control_seeds = None, None
    acquire_paid, remaining_paid = acquire, remaining
    package = None

    def remaining():
        current = remaining_paid()
        if package is not None:
            current = dict(current)
            for resource in ("observation", "measurement"):
                key = resource + "_remaining"
                used = package["before"][key] - current[key]
                current[key] = min(current[key], max(0, package[resource + "_cost"] - used))
        return current

    def acquire(view, purpose, *args, probe=None, rows_count=None, **kwargs):
        if package is not None:
            rows = probe.cost if probe is not None else (rows_count or count)
            current = remaining()
            confirming = purpose == "confirm_observation_research"
            held_rows = 0 if confirming else count
            held_measurements = 0 if confirming else package["confirmation_measurement_cost"]
            measurement = rows * sum(rates[name] for name in view)
            if (rows + held_rows > current["observation_remaining"]
                    or measurement + held_measurements > current["measurement_remaining"]):
                return None
        return acquire_paid(view, purpose, *args, probe=probe, rows_count=rows_count, **kwargs)

    def cost_snapshot():
        return {**remaining_paid(), "compute_spent": budget.spent}

    def required_channels(*programs):
        required = {name for program in programs for name, _ in program.observed_variables()} - {"__target__"}
        return next(view for view in ((), *views)
                    if required <= set(agent._spec(laboratory, view).feature_names))

    def data(view):
        batches = ([[pair[1] for pair in batch] for batch in paired_for(view)]
                   if view else command_batches())
        names = set(agent._spec(laboratory, view).feature_names)
        originals = batches
        batches = [[Observation({name: value for name, value in row.features.items() if name in names}, row.target,
            tuple({name: value for name, value in past.items() if name in names or name == "__target__"}
                  for past in row.history)) for row in batch] for batch in batches]
        if workflow is not None:
            for source, projected in zip(originals, batches):
                workflow.projected(source, projected)
        return ([row for batch in batches[:-1] for row in batch], list(batches[-1])) if len(batches) > 1 else (
            list(batches[0]), list(batches[0])) if batches else ([], [])

    retained = []

    def remember(entry, view, training, validation, measured_loss, probe_id=None):
        entry = deepcopy(entry)
        entry.setdefault("fallback", fallback)
        item = {"selected": entry, "channels": tuple(view), "training": list(training),
                "validation": list(validation), "fallback": entry["fallback"],
                "confirmed": False, "confirmation_loss": measured_loss,
                "confirmation_rows": 0, "confirmation_probe_id": probe_id}
        existing = next((old for old in retained if old["channels"] == item["channels"]
            and old["selected"]["candidate"].program == entry["candidate"].program), None)
        if existing is not None:
            retained.remove(existing)
        retained.append(item)
        return item

    def saved_alternatives():
        return [{"program": item["selected"]["candidate"].program.to_dict(),
            "channels": list(item["channels"]), "fallback": item["fallback"],
            "kind": item["selected"]["kind"],
            "family_id": item["selected"].get("family_id", "T03"),
            "source_concept_ids": list(item["selected"].get("method_source_ids", ())),
            "research_program": deepcopy(item["selected"].get("research_program")),
            "condition_uncertainty": deepcopy(item["selected"].get("condition_uncertainty")),
            "condition_research_returns": deepcopy(item["selected"].get("condition_research_returns")),
            "research_draft_id": item["selected"].get("research_draft_id"),
            "training_loss": item["selected"]["training_loss"],
            "validation_loss": item["selected"]["validation_loss"],
            "confirmation_loss": item["confirmation_loss"],
            "confirmation_probe_id": item["confirmation_probe_id"]}
            for item in retained]

    for payload in alternatives:
        program = Program.from_dict(payload["program"])
        view = tuple(payload["channels"])
        training, validation = data(view)
        if not training or not validation:
            continue
        sources = payload.get("source_concept_ids", ())
        entry = {"candidate": Candidate(f"observed_retained:{len(retained)}", program,
                    "retain_observed_response", parents=tuple(sources)),
            "kind": payload["kind"], "family_id": payload.get("family_id", "T03"),
            "fallback": payload["fallback"], "method_source_ids": list(sources),
            "training_loss": payload.get("training_loss", float("inf")),
            "validation_loss": payload.get("validation_loss", float("inf"))}
        if payload.get("research_program") is not None:
            entry["research_program"] = deepcopy(payload["research_program"])
        entry["condition_uncertainty"] = deepcopy(payload.get("condition_uncertainty"))
        entry["condition_research_returns"] = deepcopy(payload.get("condition_research_returns"))
        entry["research_draft_id"] = payload.get("research_draft_id")
        remember(entry, view, training, validation, payload.get("confirmation_loss"),
                 payload.get("confirmation_probe_id"))
    if parent is not None and baseline_measured:
        view = required_channels(baseline)
        training, validation = data(view)
        if training and validation:
            entry = deepcopy(parent)
            entry.setdefault("fallback", fallback)
            if budget.spend(len(training) + len(validation)):
                entry.update(training_loss=loss(baseline, training, entry["fallback"], command_spec.risk_quantile),
                             validation_loss=loss(baseline, validation, entry["fallback"], command_spec.risk_quantile))
                remember(entry, view, training, validation, None)

    def paid_costs(before):
        after = remaining_paid()
        return {**{name + "_cost": before[name + "_remaining"] - after[name + "_remaining"]
                   for name in ("observation", "measurement")},
                "compute_cost": budget.spent - before["compute_spent"]}

    def unscored(direction, before, status, origin=None):
        costs = paid_costs(before)
        if not learn or not any(costs.values()):
            return
        feedback = (policy.record_unscored_attempt(direction, status=status, context=policy.context, **costs)
                    if origin is None else policy.charge_followup(experience, origin, **costs))
        actions.append({"action": "observation_research_unscored", "research_action": direction,
            "context": policy.context, "status": status, "trial": origin["trial"] if origin else None,
            "feedback": feedback, **costs})

    def best_explanation(claims):
        adequate = [item for item in claims if max(item["training_loss"], item["validation_loss"]) <= tolerance]
        return agent._best(adequate or claims)

    def prepare_control_revision():
        nonlocal control_revisions, control_seeds
        calls = agent._recall(command_spec)
        seeds = list(dict.fromkeys([*calls, *([baseline] if not baseline_measured else [])]))
        if seeds != control_seeds:
            local = SearchBudget(min(6000, max(0, budget.remaining - 3 * count) // 4))
            control_revisions = RetainedControlRevisions(seeds, command_spec, local,
                                                        max_history=agent.config.max_history)
            control_seeds = seeds
            budget.spend(local.spent)
            actions.append({"action": "prepare_control_revision", "compute_cost": local.spent,
                            "coordinate_count": len(control_revisions.coordinates)})
        return control_revisions.applicability([row for batch in command_batches() for row in batch])

    def candidate(view, *, revise=False, prepare=False):
        nonlocal prepared
        training, validation = data(view)
        if not training or not validation:
            return None, training, validation
        spec = agent._spec(laboratory, view)
        allowance = max(0, budget.remaining - (3 + len(retained)) * count)
        local = SearchBudget(min(180000, allowance) if set(spec.feature_names) - set(spec.feature_bounds)
                             else min(48000, allowance))
        calls = agent._recall(spec)
        claims = []
        for call in calls:
            if not local.spend(len(training)):
                break
            claims.append({"candidate": Candidate(f"observation_call:{len(actions)}:{len(claims)}", call,
                "observation_dependency_rebinding", parents=tuple(_sources(call))),
                "kind": "input_dependency_revision", "family_id": "T03", "method_source_ids": _sources(call),
                "training_loss": loss(call, training, fallback, spec.risk_quantile)})
        checked = agent._assess(claims, validation, local, fallback, spec.risk_quantile)
        # A refit may remove its source entirely. Keep the frozen source's
        # actual failed prediction as separate evidence, rather than assigning
        # that source to the replacement merely to retain its counterexample.
        for claim in checked:
            if claim["validation_loss"] > tolerance:
                transfer.append({"source_ids": _sources(claim["candidate"].program),
                    "program": claim["candidate"].program.to_dict(),
                    "validation_loss": claim["validation_loss"], "validation_rows": len(validation),
                    "status": "counterexample", "accepted": False,
                    "kind": "frozen_recalled_response_counterexample"})
        best = best_explanation(checked)
        if revise:
            researched_view = False
            if view and calls and (best is None or max(
                    best["training_loss"], best["validation_loss"]) > tolerance):
                from .unified_research import _fit, _terms

                start = local.spent
                rebound = []
                for call in calls:
                    fitted = _fit([call, Program("mul", children=(call, call)),
                        *_terms(spec.feature_names)], training, local,
                        "callable_transfer", family="T03", q=spec.risk_quantile)
                    if fitted is not None:
                        fitted["method_source_ids"] = _sources(fitted["candidate"].program)
                        rebound.append(fitted)
                checked.extend(agent._assess(rebound, validation, local, fallback, spec.risk_quantile))
                execution["search_compute_cost"] += local.spent - start
                if local.spent > start:
                    execution["executed_directions"].append("callable_transfer")
                best = best_explanation(checked)
            if view and (best is None or max(best["training_loss"], best["validation_loss"]) > tolerance):
                from .unified_research import _fit, _terms

                start = local.spent
                depth_limit = min(agent.config.max_history,
                    max((len(row.history) for row in training), default=0))
                for depth in range(depth_limit + 1):
                    terms = _terms(spec.feature_names, depth)
                    quote = len(training) * (len(terms) + 2) + len(validation)
                    if local.remaining < quote:
                        break
                    ordinary = _fit(terms, training, local, "measured_history_response",
                                    family="T08", q=spec.risk_quantile)
                    checked.extend(agent._assess([ordinary], validation, local, fallback, spec.risk_quantile))
                    best = best_explanation(checked)
                    if best is not None and max(best["training_loss"], best["validation_loss"]) <= tolerance:
                        break
                execution["search_compute_cost"] += local.spent - start
                if local.spent > start:
                    execution["executed_directions"].append("measured_history_response")
            if not view and (best is None or max(best["training_loss"], best["validation_loss"]) > tolerance):
                revisions = control_revisions
                paid = [row for batch in command_batches() for row in batch]

                def refit_controls(rows):
                    start = local.spent
                    proposed = [item for temporal in (False, True)
                                for item in revisions.propose(rows, local, temporal=temporal)]
                    if local.spent > start:
                        execution["search_compute_cost"] += local.spent - start
                        execution["executed_directions"].append("control_revision")
                    return [{**item, "validation_loss": item["training_loss"],
                             "selection_basis": "all_acquired_evidence"} for item in proposed]

                revised = refit_controls(paid)
                adequate = [item for item in revised if item["training_loss"] <= tolerance]
                if not adequate and revisions.coordinates and prepare and not prepared:
                    extra = acquire((), "distinguish_control_history_revision", rows_count=count)
                    if extra is not None:
                        prepared = True
                        paid.extend(extra)
                        revised = refit_controls(paid)
                checked.extend(revised)
                best = best_explanation(checked)
                if best in revised:
                    training, validation = paid, list(paid)
            if set(spec.feature_names) - set(spec.feature_bounds):
                from .condition_response_actions import research_condition_response_actions
                if best is None or max(best["training_loss"], best["validation_loss"]) > tolerance:
                    # A newly measured input can expose an ordinary response
                    # or history law. Compare that paid evidence before buying
                    # condition/duration experiments; those remain available
                    # for residual error or an ambiguous conditional fit.
                    response_budget = SearchBudget(min(32000, local.remaining // 2))
                    view_execution = {}
                    checked.extend(agent._research_view(spec, training, validation,
                        response_budget, actions, transfer, graph, execution=view_execution))
                    local.spend(response_budget.spent)
                    execution["search_compute_cost"] += view_execution.get("search_compute_cost", 0)
                    execution["executed_directions"].extend(view_execution.get("executed_directions", ()))
                    best = best_explanation(checked)
                    researched_view = True
                # An exact point fit can still leave competing meter cuts. The
                # ordinary helper also researches this remaining uncertainty.
                best, training, condition_claims = research_condition_response_actions(agent,
                    spec, best, training, validation, checked, local, actions,
                    acquire=acquire, remaining=remaining, channels=view,
                    measurement_rate=sum(rates[name] for name in view))
                checked.extend(condition_claims)
            if (not researched_view
                    and (best is None or max(best["training_loss"], best["validation_loss"]) > tolerance)
                    and local.remaining >= len(training) * (len(spec.feature_names) + 2) + len(validation)):
                view_execution = {}
                checked.extend(agent._research_view(spec, training, validation, local, actions, transfer, graph,
                    parent=parent if not view else None, execution=view_execution,
                    observation_remaining=remaining()["observation_remaining"],
                    measurement_remaining=remaining()["measurement_remaining"]))
                execution["search_compute_cost"] += view_execution.get("search_compute_cost", 0)
                execution["executed_directions"].extend(view_execution.get("executed_directions", ()))
                best = best_explanation(checked)
        # Re-evaluating the incumbent is paid evidence, but it does not mean a
        # research direction produced or compared an alternative. A fresh fit
        # counts even when it returns the same executable response.
        compared_candidates.update(id(item["candidate"]) for item in checked
            if parent is None or item["candidate"] is not parent["candidate"])
        execution["compared_candidates"] = len(compared_candidates)
        budget.spend(local.spent)
        return best, training, validation

    while len(attempted) < 3:
        resources = remaining()
        iteration_start = cost_snapshot()
        if resources["observation_remaining"] < count or budget.remaining < 2 * count:
            break
        # Preparing an option is prediction work, not an observation or a trial.
        if "discriminate_inputs" not in attempted and cached_plan is None:
            cached_model = next((model for model in reversed(agent.library.instrument_models)
                if model.get("status") != "counterexample" and instrument_candidates(model, measured_spec)
                and all({name for name, _ in program.observed_variables()} <= set(command_spec.feature_names)
                        for choices in instrument_candidates(model, measured_spec).values()
                        for program in choices.values())), None)
            if cached_model is not None:
                if not agent.config.use_memory:
                    cached_model = deepcopy(cached_model)
                cap = min(command_spec.max_probe_length, resources["observation_remaining"] - count,
                          resources["measurement_remaining"] // max(1, rate) - count)
                local = SearchBudget(min(6000, max(0, budget.remaining - 2 * count) // 4))
                cached_plan = plan_instrument_disagreement(cached_model, command_spec, measured_spec, local,
                    max_length=max(0, cap), serial=len(actions), tolerance=tolerance, methods=methods)
                budget.spend(local.spent)
                actions.append({"action": "plan_observation_research", "model_id": cached_model["model_id"],
                    **{key: value for key, value in cached_plan.items() if key != "probe"},
                    "probe_id": cached_plan["probe"].probe_id if cached_plan["probe"] else None})
        if not command_batches() and parent is None and not (cached_plan or {}).get("probe"):
            break
        current_batches = command_batches()
        applicability = prepare_control_revision()
        missing = max(0, 2 - len(current_batches))
        input_rows = (failed_probe.cost if failed_probe is not None else count) + count
        entry_compute = max(count, sum(map(len, current_batches[:-1]))) * (len(command_spec.feature_names) + 2)
        preparation = int(bool(parent is not None and baseline.history_depth))
        confirmation_fields = required_channels(baseline,
            *(item["selected"]["candidate"].program for item in retained))
        confirmation_compute = (3 + len(retained)) * count
        options = [ActionOption("revise_mechanism", (missing + preparation + 1) * count, entry_compute + confirmation_compute,
                                measurement_cost=preparation * count * sum(rates[name] for name in required_channels(baseline))
                                    + count * sum(rates[name] for name in confirmation_fields)),
                   ActionOption("remeasure_inputs", input_rows + count, 4 * entry_compute + confirmation_compute,
                                measurement_cost=(input_rows + count) * rate)]
        if (cached_plan or {}).get("probe") is not None:
            probe = cached_plan["probe"]
            scoring = probe.cost * sum(map(len, cached_plan["predictions"].values()))
            options.append(ActionOption("discriminate_inputs", probe.cost + count, scoring + confirmation_compute,
                                        measurement_cost=(probe.cost + count) * rate))
        options = [option for option in options if option.name not in attempted]
        resources = remaining()
        # Price a bounded execution, including its final comparison. The old
        # entry fit was only a floor; candidate() can execute several complete
        # research directions, and instrument fitting is additional work.
        measurement = next((option for option in options if option.name == "remeasure_inputs"), None)
        deferred = []
        if measurement is not None:
            from .unified_research import _terms

            # One measured support trajectory plus the same final confirmation
            # used by condition-response research. Its quote already includes
            # that confirmation, so it is held only once here.
            measured_support = min(64, max(24,
                4 * (len(_terms(measured_spec.feature_names, quadratic=True)) + 1) - count))
            full_rows = input_rows + measured_support + count
            if (full_rows <= resources["observation_remaining"]
                    and full_rows * rate <= resources["measurement_remaining"]):
                measurement = replace(measurement, observation_cost=full_rows,
                    measurement_cost=full_rows * rate)
                options = [measurement if item.name == measurement.name else item for item in options]
            measurement_funded = (measurement.observation_cost <= resources["observation_remaining"]
                and measurement.measurement_cost <= resources["measurement_remaining"]
                and measurement.compute_cost <= budget.remaining)
            if measurement_funded:
                revised = []
                for option in options:
                    if (option.name == "revise_mechanism"
                            and (option.observation_cost + measurement.observation_cost > resources["observation_remaining"]
                                 or option.measurement_cost + measurement.measurement_cost > resources["measurement_remaining"])):
                        deferred.append({"action": option.name,
                            "reason": "preserve_funded_measurement_and_response_package"})
                    else:
                        revised.append(option)
                options = revised
        else:
            measurement_funded = False
        mechanism = next((option for option in options if option.name == "revise_mechanism"), None)
        measured_cap = min(228000 + confirmation_compute, budget.remaining)
        if measurement_funded and mechanism is not None:
            measured_cap = max(measurement.compute_cost, measured_cap // 2)
        priced = []
        for option in options:
            if option.name == "revise_mechanism":
                available = budget.remaining - (measured_cap if measurement_funded else 0)
                allowance = min(48000, max(0, budget.remaining - confirmation_compute) // 2,
                                max(0, available - confirmation_compute))
                cap = max(option.compute_cost, allowance + confirmation_compute)
                if cap > available:
                    continue
            elif option.name == "discriminate_inputs":
                # This has an already prepared distinguishing probe and field
                # hypotheses. Their assessment still needs a complete response
                # search afterward; the former entry-fit quote starved it.
                # Use the existing response-search ceiling, without charging
                # another instrument-discovery search.
                cap = min(measured_cap, option.compute_cost + 48000)
                cap = max(option.compute_cost, cap)
            else:
                cap = max(option.compute_cost, measured_cap)
            priced.append(replace(option, compute_cost=cap))
        options = priced
        ambiguous = cached_model is not None and bool(instrument_candidates(cached_model, measured_spec))
        condition = "observation_recovery:" + ("ambiguity" if ambiguous else "no_ambiguity")
        condition += ":paidfields" if paired_for(channels) else ":no_paidfields"
        condition += ":failedprediction" if parent is not None or failed_probe is not None else ":unresolved"
        condition += ":control_revision:" + applicability["status"]
        policy.set_condition(condition)
        selected_action = policy.choose(options, **remaining(), minimum_improvement=tolerance) if learn else None
        decision = dict(policy.last_decision) if learn else {"context": policy.context, "reason": "fixed_available_order"}
        if selected_action is None:
            resources = remaining()
            selected_action = next((option for option in options
                if option.observation_cost <= resources["observation_remaining"]
                and option.measurement_cost <= resources["measurement_remaining"]
                and option.compute_cost <= budget.remaining), None)
            if selected_action is not None and learn:
                decision["reason"] = "retry_remaining_action_on_current_evidence"
        if selected_action is None:
            break
        direction = selected_action.name
        attempted.add(direction)
        resources = remaining()
        decision.update(selected_action=direction, eligible_actions=[option.name for option in options
                        if option.observation_cost <= resources["observation_remaining"]
                        and option.measurement_cost <= resources["measurement_remaining"]
                        and option.compute_cost <= resources["compute_remaining"]],
                        quoted_observation_cost=selected_action.observation_cost,
                        quoted_measurement_cost=selected_action.measurement_cost,
                        quoted_compute_cost=selected_action.compute_cost)
        decision["options"] = [{"name": item.name, "observation_cost": item.observation_cost,
                                "measurement_cost": item.measurement_cost, "compute_cost": item.compute_cost}
                               for item in options]
        decision["control_revision_applicability"] = applicability
        decision["deferred_actions"] = deferred
        decision["preparation_compute_cost"] = budget.spent - iteration_start["compute_spent"]
        actions.append({"action": "choose_observation_research", **decision})
        package_start = cost_snapshot()
        package = {"before": remaining_paid(),
            "observation_cost": selected_action.observation_cost,
            "measurement_cost": selected_action.measurement_cost,
            "confirmation_measurement_cost": count * (rate if direction != "revise_mechanism"
                else sum(rates[name] for name in confirmation_fields))}
        previous_reserve = budget.reserve
        budget.reserve += max(0, budget.remaining - selected_action.compute_cost)
        try:
            before = iteration_start
            prepared, action_origin = False, None
            execution = {"search_compute_cost": 0, "compared_candidates": 0, "executed_directions": []}
            compared_candidates = set()
            best, training, validation, view = None, [], [], ()
            if direction == "revise_mechanism":
                for _ in range(missing):
                    if acquire((), "observation_mechanism_training", rows_count=count) is None:
                        break
                best, training, validation = candidate((), revise=True, prepare=bool(preparation))
            else:
                if direction == "remeasure_inputs":
                    first = acquire(channels, "remeasure_failed_prediction", probe=failed_probe, rows_count=count)
                    second = acquire(channels, "compare_observation_mapping", rows_count=count) if first is not None else None
                    if second is None:
                        result["handled"] |= any(paid_costs(before).values())
                        unscored(direction, before, "unresolved")
                        continue
                    if second is not None:
                        local = SearchBudget(min(48000, max(0, budget.remaining - 3 * count) // 2))
                        agent._fit_instruments(command_spec, measured_spec, paired_for(channels), local, actions)
                        budget.spend(local.spent)
                    cached_plan, cached_model = None, None
                else:
                    observed = acquire(channels, "distinguish_instrument_hypotheses", probe=cached_plan["probe"])
                    if observed is None:
                        result["handled"] |= any(paid_costs(before).values())
                        unscored(direction, before, "unresolved")
                        continue
                    if observed:
                        assessment = assess_instrument_hypotheses(cached_model, command_spec, measured_spec,
                            paired_for(channels)[-1], budget, tolerance)
                        actions.append({"action": "test_instrument_hypotheses", "model_id": cached_model["model_id"],
                            "probe_id": cached_plan["probe"].probe_id, "predictions": cached_plan["predictions"],
                            "assessment": assessment, "evidence_role": "observation_fields_only"})
                        agent._record_instrument_assessment(cached_model, assessment, measured_spec,
                                                            len(observed), actions)
                        if methods is not None:
                            feedback = observe_instrument_method(methods, cached_plan, assessment, cached_model,
                                command_spec, task_id=f"{command_spec.task_id}:{agent.config.seed}",
                                observation_rows=len(observed), measurement_cost=len(observed) * rate, tolerance=tolerance)
                            if feedback:
                                actions.append(feedback)
                best, training, validation = candidate(())
                if best is None or max(best["training_loss"], best["validation_loss"]) > tolerance:
                    alternatives = []
                    for view in views:
                        entry, train, valid = candidate(view, revise=True)
                        if entry is not None:
                            alternatives.append((entry, train, valid, view))
                            if max(entry["training_loss"], entry["validation_loss"]) <= tolerance:
                                break
                    if alternatives:
                        adequate = [item for item in alternatives
                                    if max(item[0]["training_loss"], item[0]["validation_loss"]) <= tolerance]
                        best, training, validation, view = (adequate[0] if adequate else
                                                           min(alternatives, key=lambda item: item[0]["validation_loss"]))
            result["handled"] |= any(paid_costs(before).values())
            if (direction == "revise_mechanism" and not execution["search_compute_cost"]
                    and not execution["compared_candidates"]):
                # Keep every paid cost in the report. An unaffordable direction has
                # no trial outcome to transfer to a later, adequately funded task.
                actions.append({"action": "observation_research_unexecuted", "research_action": direction,
                    "context": policy.context, "reason": "unfunded_mechanism_research",
                    "execution": deepcopy(execution), **paid_costs(before)})
                continue
            if best is None:
                actions.append({"action": "observation_research_unresolved", "research_action": direction,
                                "context": policy.context, **paid_costs(before), "reason": "no_executable_candidate"})
                unscored(direction, before, "no_candidate")
                continue
            for attempt in range(2):
                program = best["candidate"].program
                view = required_channels(program)
                inner = best.get("research_action_origin")
                baseline_programs = [baseline, program]
                baseline_programs.extend(item["selected"]["candidate"].program for item in retained)
                if inner is not None:
                    baseline_programs.append(Program.from_dict(inner["baseline_program"]))
                confirm_view = required_channels(*baseline_programs)
                confirmation_start = cost_snapshot()
                # Both predictors are frozen before acquiring the one shared trajectory.
                if workflow is not None:
                    workflow.frozen([best], training, validation, agent._spec(laboratory, view))
                    for retained_item in retained:
                        workflow.frozen([retained_item["selected"]], retained_item["training"],
                            retained_item["validation"], agent._spec(laboratory, retained_item["channels"]))
                confirmed_rows = (acquire(confirm_view, "confirm_observation_research", rows_count=count)
                                  if budget.remaining >= 2 * count else None)
                if confirmed_rows is None or not budget.spend(2 * len(confirmed_rows)):
                    actions.append({"action": "observation_research_unresolved", "research_action": direction,
                        "context": policy.context, **paid_costs(before), "reason": "unfunded_fresh_confirmation"})
                    unscored(direction, before, "unresolved", action_origin)
                    break
                before_loss = loss(baseline, confirmed_rows, fallback, command_spec.risk_quantile)
                after_loss = loss(program, confirmed_rows, fallback, command_spec.risk_quantile)
                comparison_baseline, comparison_fallback = baseline, fallback
                checked_retained = compare_observed_responses(
                    [item["selected"] for item in retained], confirmed_rows, budget, command_spec.risk_quantile)
                old_best = min(checked_retained, key=lambda item: item["loss"]) if checked_retained else None
                if old_best is not None and old_best["loss"] < before_loss:
                    before_loss = old_best["loss"]
                    comparison_baseline = old_best["entry"]["candidate"].program
                    comparison_fallback = old_best["entry"]["fallback"]
                confirmed = max(after_loss, best["training_loss"], best["validation_loss"]) <= tolerance
                current_fit_confirmed = confirmed
                if command_spec.risk_quantile == .5 and (confirmed or best.get("condition_uncertainty") is not None):
                    paid_training, paid_validation = data(view)
                    state = assess_observed_condition_uncertainty(agent._spec(laboratory, view),
                        [*paid_training, *paid_validation], best, budget,
                        previous=best.get("condition_uncertainty"), tolerance=tolerance)
                    best["condition_uncertainty"] = state
                    if state is not None and state["status"] != "supported":
                        confirmed = False
                probe_id = next((item["probe_id"] for item in reversed(actions) if item["action"] == "acquire"), None)
                actions.append({"action": "confirm_observation_research", "research_action": direction,
                    "baseline_program": comparison_baseline.to_dict(), "program": program.to_dict(),
                    "fallback": fallback, "baseline_fallback": comparison_fallback,
                    "risk_quantile": command_spec.risk_quantile, "probe_id": probe_id,
                    "before_loss": before_loss, "after_loss": after_loss, "observation_rows": len(confirmed_rows),
                    "accepted": confirmed, "current_fit_confirmed": current_fit_confirmed,
                    "condition_uncertainty": deepcopy(best.get("condition_uncertainty")),
                    "evidence_basis": "fresh_common_response_trajectory"})
                inner = best.get("research_action_origin")
                if learn and inner is not None:
                    agent._charge_action_costs(inner, paid_costs(confirmation_start), actions, "observation_confirmation")
                    inner_before = None
                    if budget.spend(len(confirmed_rows)):
                        inner_before = loss(Program.from_dict(inner["baseline_program"]), confirmed_rows,
                                            inner["fallback"], inner["risk_quantile"])
                    status = "supported" if confirmed and inner_before is not None else "refuted" if not confirmed else "unresolved"
                    if inner["trial"] is None and inner_before is not None:
                        feedback = policy.update(inner["action"], context=inner["context"],
                            before_loss=inner_before, after_loss=after_loss, **inner["pending_costs"])
                        inner.update(trial=feedback["trials"], credited_gain=feedback["last_gain"])
                    if inner["trial"] is not None:
                        feedback = policy.revise_feedback(experience, inner, status=status,
                            before_loss=inner_before, after_loss=after_loss if inner_before is not None else None,
                            compute_cost=len(confirmed_rows) if inner_before is not None else 0,
                            evidence_rows=len(confirmed_rows))
                        actions.append({"action": "research_action_confirmation", "research_action": inner["action"],
                            "context": inner["context"], "trial": inner["trial"], "status": status,
                            "before_loss": inner_before, "after_loss": after_loss if inner_before is not None else None,
                            "baseline_program": inner["baseline_program"], "program": program.to_dict(),
                            "baseline_fallback": inner["fallback"], "program_fallback": fallback,
                            "risk_quantile": inner["risk_quantile"], "probe_id": probe_id,
                            "evidence_rows": len(confirmed_rows), "compute_cost": len(confirmed_rows) if inner_before is not None else 0,
                            "evidence_basis": "fresh_frozen_confirmation", "feedback": feedback})
                    else:
                        inner["pending_status"] = status
                costs = paid_costs(before)
                if learn:
                    if action_origin is None:
                        feedback = policy.update(direction, before_loss=before_loss, after_loss=after_loss, **costs)
                        action_origin = {"context": policy.context, "action": direction, "trial": feedback["trials"],
                                         "credited_gain": feedback["last_gain"]}
                    else:
                        policy.charge_followup(experience, action_origin, **costs)
                    feedback = policy.revise_feedback(experience, action_origin, status="supported" if confirmed else "refuted",
                        before_loss=before_loss, after_loss=after_loss, evidence_rows=len(confirmed_rows))
                    actions.append({"action": "observation_research_feedback", "research_action": direction,
                        "context": policy.context, "before_loss": before_loss, "after_loss": after_loss,
                        "evidence_rows": len(confirmed_rows), **costs, "feedback": feedback})
                chosen = {"selected": best, "channels": view, "training": training,
                    "validation": validation, "confirmed": confirmed,
                    "confirmation_loss": after_loss, "confirmation_rows": len(confirmed_rows),
                    "current_fit_confirmed": current_fit_confirmed,
                    "condition_uncertainty": deepcopy(best.get("condition_uncertainty"))}
                if view:
                    remember(best, view, training, validation, after_loss, probe_id)
                if old_best is not None and old_best["loss"] < after_loss - tolerance:
                    old_entry = deepcopy(old_best["entry"])
                    old = next(item for item in retained if item["selected"]["candidate"].program
                        == old_entry["candidate"].program)
                    current_training, current_validation = data(old["channels"])
                    score_cost = len(current_training) + len(current_validation)
                    current_scored = bool(current_training and current_validation and budget.spend(score_cost))
                    if current_scored:
                        old_entry.update(training_loss=loss(old_entry["candidate"].program,
                            current_training, old["fallback"], command_spec.risk_quantile),
                            validation_loss=loss(old_entry["candidate"].program,
                            current_validation, old["fallback"], command_spec.risk_quantile))
                        old.update(selected=old_entry, training=current_training, validation=current_validation,
                                   confirmation_loss=old_best["loss"], confirmation_probe_id=probe_id)
                    current_fit = current_scored and max(old_best["loss"], old_entry["training_loss"],
                                                         old_entry["validation_loss"]) <= tolerance
                    if command_spec.risk_quantile == .5 and (current_fit or old_entry.get("condition_uncertainty") is not None):
                        state = assess_observed_condition_uncertainty(agent._spec(laboratory, old["channels"]),
                            [*current_training, *current_validation], old_entry, budget,
                            previous=old_entry.get("condition_uncertainty"), tolerance=tolerance)
                        old_entry["condition_uncertainty"] = state
                        old["selected"] = old_entry
                    state = old_entry.get("condition_uncertainty")
                    chosen = {**old, "selected": deepcopy(old_entry),
                        "confirmation_loss": old_best["loss"], "confirmation_rows": len(confirmed_rows),
                        "current_fit_confirmed": current_fit,
                        "condition_uncertainty": deepcopy(state),
                        "confirmed": current_fit and (state is None or state["status"] == "supported")}
                    actions.append({"action": "retain_observed_response", "probe_id": probe_id,
                        "before_loss": old_best["loss"], "after_loss": after_loss,
                        "program": old_entry["candidate"].program.to_dict(),
                        "channels": list(old["channels"]), "observation_rows": len(confirmed_rows),
                        "compute_cost": score_cost if current_scored else 0,
                        "current_evidence_scored": current_scored,
                        "current_fit_confirmed": current_fit,
                        "condition_uncertainty": deepcopy(state),
                        "evidence_basis": "fresh_common_response_trajectory", "confirmed": chosen["confirmed"]})
                if workflow is not None:
                    workflow.tested(program, confirmed_rows, after_loss, accepted=confirmed)
                    if chosen["selected"]["candidate"].program != program:
                        workflow.tested(chosen["selected"]["candidate"].program, confirmed_rows,
                            chosen["confirmation_loss"], accepted=chosen["confirmed"])
                result.update(chosen, alternatives=saved_alternatives())
                if chosen["confirmed"]:
                    return result
                if (direction != "revise_mechanism" or not preparation or prepared or attempt
                        or remaining()["observation_remaining"] < count or budget.remaining < 3 * count
                        or remaining()["measurement_remaining"] < count * sum(rates[name] for name in required_channels(baseline))):
                    break
                # The failed frozen trajectory is now fitting evidence. It cannot
                # confirm its own repair; the next iteration buys another trajectory.
                prepared, before = True, cost_snapshot()
                best, training, validation = candidate((), revise=True)
                actions.append({"action": "revise_mechanism_from_confirmation", "training_rows": len(training),
                                "candidate_found": best is not None})
                if best is None:
                    unscored(direction, before, "no_candidate", action_origin)
                    break
        finally:
            budget.reserve = previous_reserve
            package = None
            actions.append({"action": "finish_observation_research_package",
                "research_action": direction,
                "quoted_observation_cost": selected_action.observation_cost,
                "quoted_measurement_cost": selected_action.measurement_cost,
                "quoted_compute_cost": selected_action.compute_cost,
                **paid_costs(package_start)})
    if paired_for(channels):
        result["pending_inputs"] = data(channels)
    result["alternatives"] = saved_alternatives()
    return result
