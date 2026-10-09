"""Continue an unresolved measured question from its already purchased evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
from math import sqrt
from statistics import mean, stdev

from .evaluation import initial_prediction, loss, point_loss, predict
from .language import Program
from .memory import _object_schema
from .models import Candidate, Observation, Probe, SearchBudget
from .research_graph import ResearchGraph


def paid_view(laboratory, report, channels):
    """Recover one purchased view, keeping every experimental reset separate."""
    from .unified_research import _view
    channels, result = tuple(channels), []
    for batch in report.get("batches", []):
        source = tuple(batch["channels"])
        if not channels:
            history, rows = [], []
            for point, item in zip(batch["probe"]["sequence"], batch["observations"]):
                rows.append(Observation(dict(point), item["target"], tuple(history)))
                history.append({**point, "__target__": item["target"]})
            result.append(rows)
            continue
        if getattr(laboratory, "measurement_views", None):
            if channels not in map(tuple, laboratory.projectable_views(source)):
                continue
        elif not set(channels) <= set(source):
            continue
        rows = [Observation(dict(item["features"]), item["target"],
                            tuple(dict(past) for past in item["history"]))
                for item in batch["observations"]]
        result.append(laboratory.project_rows(rows, source, channels)
            if getattr(laboratory, "measurement_views", None) else
            _view(rows, laboratory.spec_for(channels).feature_names))
    return result


def _guard(program, fallback):
    depth = program.minimum_history_depth
    return (Program("if", children=(Program("available", lag=depth), program,
            Program("const", value=fallback))) if depth else program)


def run_continuation(agent, laboratory, previous, continuation_candidate=None):
    """Use cumulative task limits; only new acquisitions are purchased again."""
    from .unified_research import _fit, _terms
    from .risk_research_actions import RiskResearchActions
    from .risk_experiments import distinguish_risk_explanations
    from .noise_replication import assess_noise_replication
    from .observed_condition_uncertainty import assess_observed_condition_uncertainty

    offered = tuple(channel.name for channel in laboratory.channels)
    if getattr(laboratory, "measurement_views", None) or _object_schema(laboratory.spec_for(offered)):
        raise ValueError("continuation currently requires the numeric measurement interface")
    if hasattr(laboratory, "restore"):
        # Resume the purchased process before drawing a new realization. A new
        # probe ID must not restart the laboratory's initial random sequence.
        laboratory.restore([{**batch["probe"], "channels": batch["channels"]}
                            for batch in previous["batches"]])
    report = deepcopy(previous)
    report.pop("knowledge_return", None)
    old_costs = dict(previous["costs"])
    if (agent.config.observation_budget < old_costs["observation_cost"]
            or agent.measurement_budget < old_costs["measurement_cost"]
            or agent.config.compute_budget < old_costs["compute_cost"]):
        raise ValueError("continuation limits cannot be smaller than already purchased work")
    budget = SearchBudget(agent.config.compute_budget, old_costs["compute_cost"])
    actions, batches = report["actions"], report["batches"]
    transfer = report.get("transfer", [])
    graph = ResearchGraph.from_dict(report.get("research_graph"))
    rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
    spent_observations, spent_measurements = old_costs["observation_cost"], old_costs["measurement_cost"]
    serial, length = len(batches), min(24, laboratory.spec.max_probe_length)
    q = laboratory.spec.risk_quantile
    agent._learn_instruments = agent._instrument_knowledge(laboratory)
    agent._revise_inputs = agent._revisable_inputs(laboratory)

    def remaining():
        return {"observation_remaining": agent.config.observation_budget - spent_observations,
                "measurement_remaining": agent.measurement_budget - spent_measurements,
                "compute_remaining": budget.remaining}

    def acquire(channels, purpose, candidates=(), *, rows_count=None, probe=None, **unused):
        nonlocal spent_observations, spent_measurements, serial
        channels = tuple(channels)
        count = probe.cost if probe is not None else rows_count or length
        rate = sum(rates[name] for name in channels)
        if (count > remaining()["observation_remaining"]
                or count * rate > remaining()["measurement_remaining"]):
            return None
        probe = probe or agent._probe(agent._spec(laboratory, channels), serial, count,
            () if purpose in {"compare_continuation_candidates", "confirm_continuation_program"}
            else candidates, budget, actions)
        serial += 1
        batch = laboratory.acquire(probe, channels)
        spent_observations += batch.observation_cost
        spent_measurements += batch.measurement_cost
        batches.append({"probe": asdict(probe), "channels": list(channels),
            "observations": [asdict(row) for row in batch.observations],
            "transitions": getattr(batch, "transitions", [])})
        actions.append({"action": "acquire", "purpose": purpose, "probe_id": probe.probe_id,
            "channels": list(channels), "observation_cost": batch.observation_cost,
            "measurement_cost": batch.measurement_cost})
        return batch.observations

    def rows_for(channels):
        return [row for batch in paid_view(laboratory, report, channels) for row in batch]

    def view_rows(rows, source, target):
        from .unified_research import _view
        return _view(rows, laboratory.spec_for(target).feature_names)

    def paired_for(channels):
        measured = paid_view(laboratory, report, channels)
        eligible = [batch for batch in batches if not channels or set(channels) <= set(batch["channels"])]
        result = []
        for batch, rows in zip(eligible, measured):
            history, pairs = [], []
            for point, row in zip(batch["probe"]["sequence"], rows):
                pairs.append((Observation(dict(point), row.target, tuple(history)), row))
                history.append({**point, "__target__": row.target})
            result.append(pairs)
        return result

    def entry(payload, name):
        program = Program.from_dict(payload["program"])
        channels = tuple(payload.get("channels", payload.get("selected_channels", ())))
        sources = list(payload.get("source_concept_ids", payload.get("method_source_ids", ())))
        result = {"candidate": Candidate(f"continuation:{name}", program, name,
                    parents=tuple(sources)),
            "kind": payload.get("kind", payload.get("selected_kind", name)),
            "family_id": payload.get("family_id", "T03"),
            "fallback": payload.get("fallback", initial_prediction(rows_for(channels), q)),
            "channels": channels, "method_source_ids": sources,
            "transfer_evidence": deepcopy(payload.get("transfer_evidence", [])),
            "selection_basis": "all_acquired_evidence"}
        if payload.get("research_program") is not None:
            result["research_program"] = deepcopy(payload["research_program"])
        result["condition_uncertainty"] = deepcopy(payload.get("condition_uncertainty",
            payload.get("continuation_state", {}).get("condition_uncertainty")))
        result["condition_research_returns"] = deepcopy(payload.get("condition_research_returns"))
        result["pair_confirmation_returns"] = deepcopy(payload.get("pair_confirmation_returns"))
        result["research_draft_id"] = payload.get("research_draft_id")
        if payload.get("risk_method") is not None:
            result["risk_method"] = deepcopy(payload["risk_method"])
        return result

    incumbent = entry(previous, "retained_incumbent")
    challenge = ((continuation_candidate or {}).get("challenge")
                 if previous["status"] == "completed" else None)
    challenge_probe = (Probe(challenge["probe"]["probe_id"],
                             tuple(challenge["probe"]["sequence"])) if challenge else None)
    method_experiment = (continuation_candidate or {}).get("method_experiment")
    instrument_experiment = (continuation_candidate or {}).get("instrument_condition_experiment")
    risk_method = (continuation_candidate or {}).get("risk_method")
    method_known_error, method_new_error = None, 0.
    method_structure_evidence = False
    method_previous_condition = deepcopy(incumbent.get("condition_uncertainty"))
    scope_counterexamples = []
    completed_refuted = False
    candidates = [incumbent]
    training = rows_for(incumbent["channels"])
    initial_paid_rows = len(training)
    for payload in previous.get("continuation_state", {}).get("alternatives", []):
        proposed = entry(payload, "retained_alternative")
        if proposed["candidate"].program != incumbent["candidate"].program:
            candidates.append(proposed)
    if continuation_candidate is not None:
        candidates.append(entry(continuation_candidate, "successor_knowledge"))
    paid_selection = deepcopy((continuation_candidate or {}).get("paid_selection"))
    if paid_selection is not None:
        # A queued selection is reusable only for the paid frontier on which
        # its frozen response beat this incumbent. A changed frontier follows
        # the ordinary comparison/research path instead.
        common_rows = rows_for(tuple(paid_selection["channels"]))
        if not (previous["status"] != "completed" and q == .5
                and not (challenge or method_experiment or instrument_experiment or risk_method)
                and paid_selection["evidence_batches"] == [
                    batch["probe"]["probe_id"] for batch in previous["batches"]]
                and paid_selection["evidence_rows"] == len(common_rows) > 0
                and paid_selection["baseline_program"] == previous["program"]
                and paid_selection["baseline_fallback"] == incumbent["fallback"]
                and paid_selection["candidate_loss"] <= agent.config.minimum_improvement
                and paid_selection["baseline_loss"] - paid_selection["candidate_loss"]
                    > agent.config.minimum_improvement):
            paid_selection = None
    transfer_episode = (agent.config.use_memory and any(
        candidate["transfer_evidence"] for candidate in candidates))
    if transfer_episode:
        agent.library.begin_transfer_episode(agent._spec(laboratory, incumbent["channels"]), training)

    def record_source_comparison(item, rows, before, after, probe_id, *, accepted):
        nonlocal transfer_episode
        origins = item.get("transfer_evidence", [])
        if not agent.config.use_memory or not origins or item is incumbent:
            return
        spec = agent._spec(laboratory, item["channels"])
        if not transfer_episode:
            # Continued paid research can first discover a reusable call after
            # entry, without an incoming successor offer. Open its return episode
            # from the old evidence; the comparison below uses the new response.
            prior_rows = [row for batch in paid_view(laboratory, previous, item["channels"])
                          for row in batch]
            agent.library.begin_transfer_episode(spec, prior_rows)
            transfer_episode = True
        for origin in origins:
            candidate = replace(item["candidate"], memory_id=origin["source_id"])
            agent.library.record_transfer_result(candidate, spec, accepted=accepted,
                loss=after, baseline_loss=before, evidence_rows=len(rows),
                comparison_compute=2 * len(rows), comparison_id=probe_id, origin=origin)
        actions.append({"action": "compare_successor_knowledge_on_fresh_response",
            "probe_id": probe_id, "source_concept_ids": [origin["source_id"] for origin in origins],
            "lineage_id": next((origin.get("lineage_id") for origin in origins
                                 if origin.get("lineage_id") is not None), None),
            "joint_source_ids": next((origin.get("joint_source_ids") for origin in origins
                                       if origin.get("joint_source_ids") is not None),
                                      [origin["source_id"] for origin in origins]),
            "program": item["candidate"].program.to_dict(), "fallback": item.get("fallback", 0.),
            "before_loss": before, "after_loss": after, "observation_rows": len(rows)})
        if (q == .5 and not agent.config.risk_learning
                and after > before + agent.config.minimum_improvement):
            withdrawn = agent.library.record_transfer_counterexample(spec,
                replace(item["candidate"], parents=tuple(origin["source_id"] for origin in origins)),
                [origin["source_id"] for origin in origins], probe_id=probe_id,
                evidence={"status": "counterexample", "rows": len(rows),
                          "measured_loss": after, "retained_loss": before})
            if withdrawn is not None:
                actions.append(withdrawn)

    def finish_source_returns(item, final_loss, costs):
        if transfer_episode:
            report["knowledge_return"] = agent.library.finish_transfer_episode(
                agent._spec(laboratory, item["channels"]), final_candidate=item["candidate"],
                final_loss=final_loss, baseline_loss=previous.get("confirmation_loss"),
                compute_cost=costs["compute_cost"] - old_costs["compute_cost"],
                observation_cost=costs["observation_cost"] - old_costs["observation_cost"],
                validation_rows=confirmation_rows,
                final_status=report.get("status"))
    selected, selected_channels = incumbent, incumbent["channels"]
    confirmed, confirmation_loss, support, confirmation_rows = False, None, previous.get("noise_assessment"), 0
    measured_choice = (deepcopy(incumbent), incumbent["channels"], None, deepcopy(support))
    research = RiskResearchActions(agent, laboratory, acquire=acquire, paired_for=paired_for,
        command_batches=lambda: paid_view(laboratory, report, ()), view_rows=view_rows,
        transitions_for=lambda *args: [], budget=budget, actions=actions, transfer=transfer,
        graph=graph, remaining=remaining, length=length) if (q != .5 and agent.config.risk_learning
            and agent.config.strategy != "fixed") else None
    actions.append({"action": "resume_measured_research", "paid_rows": spent_observations,
        "paid_batches": len(batches), "previous_status": previous["status"],
        "source_concept_ids": (continuation_candidate or {}).get("source_concept_ids", []),
        "previous_costs": old_costs})

    def keep_completed(outcome, reason):
        if q == .5:
            paid = rows_for(incumbent["channels"])
            state = incumbent.get("condition_uncertainty")
            if (state is None or state.get("fit_rows") != len(paid)
                    or state.get("program") != incumbent["candidate"].program.to_dict()):
                state = assess_observed_condition_uncertainty(
                    agent._spec(laboratory, incumbent["channels"]), paid, incumbent, budget,
                    previous=state, tolerance=agent.config.minimum_improvement)
                incumbent["condition_uncertainty"] = state
            report["condition_uncertainty"] = deepcopy(state)
            report.setdefault("continuation_state", {})["condition_uncertainty"] = deepcopy(state)
            if state is not None and state["status"] != "supported":
                report["status"] = "unresolved"
        costs = {"observation_cost": spent_observations, "measurement_cost": spent_measurements,
                 "compute_cost": budget.spent, "queries": len(batches)}
        event = {"action": "completed_task_challenge", "outcome": outcome, "reason": reason,
                 "source_concept_ids": (continuation_candidate or {}).get("source_concept_ids", []),
                 "counterexample": False, "probe_id": challenge_probe.probe_id if challenge_probe else None}
        actions.append(event)
        if challenge:
            report["scope_revision"] = {"status": "challenge_rejected" if outcome == "rejected" else "challenge_deferred",
                "previous_program": previous["program"], "previous_channels": previous["selected_channels"],
                "previous_status": previous["status"], "previous_confirmation_loss": previous.get("confirmation_loss"),
                "validated_trajectories": [{"probe_id": batch["probe"]["probe_id"],
                    "length": len(batch["probe"]["sequence"])} for batch in previous["batches"]],
                "counterexamples": [], "attempted_challenge": deepcopy(challenge)}
        report.update(costs=costs,
            visit_costs={key: value - old_costs.get(key, 0) for key, value in costs.items()},
            concepts_added=[], used_sources=[], method_source_ids=[],
            continuation={"source_concept_ids": event["source_concept_ids"],
                "previous_status": previous["status"], "previous_costs": old_costs,
                "comparisons": [], "paid_rows_reused": old_costs["observation_cost"],
                "challenge_outcome": outcome})
        finish_source_returns(incumbent, previous.get("confirmation_loss"), costs)
        return report

    # A successor's computation is refitted on the OLD purchased response. The
    # incumbent and unchanged successor remain alternatives to this local fit.
    fitted = []
    for candidate in (() if paid_selection or challenge or method_experiment or instrument_experiment else candidates):
        channels = candidate["channels"]
        rows = rows_for(channels)
        if not rows:
            continue
        reserve = 4 * length + (len(rows) * 6 + length * 16 if research is not None else 0)
        local = SearchBudget(min(24000, max(0, budget.remaining - reserve)))
        if candidate is not incumbent:
            program = _guard(candidate["candidate"].program, candidate["fallback"])
            independent_terms = _terms(laboratory.spec_for(channels).feature_names)
            claim = _fit([program, *independent_terms],
                rows, local, "continued_callable_refit", family=candidate["family_id"], q=q)
            if claim is not None:
                claim.update(channels=channels, fallback=candidate["fallback"],
                    method_source_ids=candidate.get("method_source_ids", []),
                    selection_basis="all_acquired_evidence")
                # Credit a local combination only if the transferred expression
                # survives in its fitted computation; independent terms earn no
                # source credit merely because they shared a fit call.
                source_program = program.expanded().simplified()
                def contains_source(node):
                    return (node.expanded().simplified() == source_program
                            or any(contains_source(child) for child in node.children))
                if (candidate.get("transfer_evidence") and source_program.op != "const"
                        and source_program not in independent_terms
                        and contains_source(claim["candidate"].program)):
                    claim["transfer_evidence"] = deepcopy(candidate["transfer_evidence"])
                    work = {"work_id": f"continuation:{laboratory.spec.task_id}:{serial}:{len(fitted)}",
                        "compute_cost": local.spent,
                        "source_ids": [origin["source_id"] for origin in claim["transfer_evidence"]],
                        "template_ids": [origin["template_id"] for origin in claim["transfer_evidence"]]}
                    for origin in claim["transfer_evidence"]:
                        origin["adaptation_work"].append(deepcopy(work))
                fitted.append(claim)
                if q != .5:
                    for revised in agent._risk_revisions(agent._spec(laboratory, channels), [claim], rows, local):
                        revised.update(channels=channels, fallback=claim["fallback"],
                            method_source_ids=claim["method_source_ids"], selection_basis="all_acquired_evidence")
                        fitted.append(revised)
        budget.spend(local.spent)
    candidates.extend(fitted)
    checked = []
    for candidate in (() if paid_selection or challenge else candidates):
        rows = rows_for(candidate["channels"])
        if rows and budget.spend(len(rows)):
            if method_experiment and q == .5 and candidate is incumbent:
                errors = [point_loss(predict(candidate["candidate"].program, row,
                    candidate["fallback"]), row.target, q) for row in rows]
                measured, method_known_error = mean(errors), max(errors)
            else:
                measured = loss(candidate["candidate"].program, rows, candidate["fallback"], q)
            candidate.update(training_loss=measured, validation_loss=measured)
            origins = candidate.get("transfer_evidence", [])
            for origin in origins:
                origin["adaptation_work"].append({
                    "work_id": f"continuation_evaluation:{laboratory.spec.task_id}:{serial}:{len(checked)}",
                    "compute_cost": len(rows), "source_ids": [item["source_id"] for item in origins],
                    "template_ids": [item["template_id"] for item in origins]})
            checked.append(candidate)
    if checked:
        selected = agent._best(checked)
        selected_channels = selected["channels"]
    if paid_selection:
        selected = candidates[-1]
        selected_channels = selected["channels"]
        selected.update(training_loss=paid_selection["candidate_loss"],
                        validation_loss=paid_selection["candidate_loss"])
        actions.append({"action": "reuse_paid_continuation_selection",
            **deepcopy(paid_selection), "program": selected["candidate"].program.to_dict(),
            "fallback": selected["fallback"], "new_observations": 0})
    if challenge:
        selected = candidates[-1]
        selected_channels = selected["channels"]
        length = min(laboratory.spec.max_probe_length, max(length, challenge_probe.cost,
            incumbent["candidate"].program.history_depth + 1, selected["candidate"].program.history_depth + 1))
    if not paid_selection:
        actions.append({"action": "compare_continuation_on_paid_evidence", "selection_basis": "all_acquired_evidence",
            "alternatives": [{"kind": item["kind"], "program": item["candidate"].program.to_dict(),
                "channels": list(item["channels"]), "loss": item["validation_loss"]} for item in checked],
            "selected_kind": selected["kind"], "new_observations": 0})

    if not paid_selection and not challenge and q == .5:
        purchased_views = paid_view(laboratory, report, selected_channels)
        validation = purchased_views[-1] if purchased_views else []
        from .condition_response_actions import (research_condition_response_actions,
            research_observed_duration_experiments, _observed_response_claims)
        if instrument_experiment:
            from .instrument_condition_revisits import execute_instrument_condition_revisit
            selected, selected_channels = incumbent, incumbent["channels"]
            selected, training, candidates = execute_instrument_condition_revisit(agent,
                agent._spec(laboratory, selected_channels), selected, rows_for(selected_channels),
                validation, candidates, budget, actions, instrument_experiment, acquire=acquire,
                remaining=remaining, serial=serial, channels=selected_channels,
                measurement_rate=sum(rates[name] for name in selected_channels))
        if method_experiment:
            method_channels = tuple(continuation_candidate["channels"])
            method_spec = agent._spec(laboratory, method_channels)
            method_training = rows_for(method_channels)
            method_validation = paid_view(laboratory, report, method_channels)[-1]
            method_parent = selected if selected_channels == method_channels else incumbent
            method_proposal = entry(continuation_candidate, "successor_partial_method")
            plan = {**method_experiment, "probe": Probe(method_experiment["probe"]["probe_id"],
                tuple(method_experiment["probe"]["sequence"]))}
            start_compute, start_rows = budget.spent, len(method_training)
            experiment_budget = SearchBudget(max(0, budget.remaining - 4 * length))
            method_training, evidence = research_observed_duration_experiments(agent, method_spec,
                method_parent, method_training, [method_proposal], experiment_budget, actions, acquire=acquire,
                channels=method_channels, observation_allowance=plan["probe"].cost,
                validation_rows=len(method_validation), validation=method_validation,
                serial=serial, first_plan=plan)
            budget.spend(experiment_budget.spent)
            method_structure_evidence = bool(evidence)
            new_rows = method_training[start_rows:]
            event = {"action": "execute_successor_method_experiment",
                "source_draft_ids": method_experiment["source_draft_ids"],
                "predicted_separation": method_experiment["expected_disagreement"],
                "observation_rows": len(new_rows), "evidence": evidence,
                "incumbent_loss": None, "proposed_loss": None}
            if new_rows and budget.spend(2 * len(new_rows)):
                errors = [point_loss(predict(incumbent["candidate"].program, row,
                    incumbent["fallback"]), row.target, q) for row in new_rows]
                method_new_error = max(method_new_error, max(errors))
                event.update(incumbent_loss=mean(errors),
                    proposed_loss=loss(method_proposal["candidate"].program, new_rows, method_proposal["fallback"], q))
            if new_rows and budget.spend(len(method_training) + len(method_validation)):
                fallback = initial_prediction(method_training, q)
                old = {**method_parent, "training_loss": loss(method_parent["candidate"].program,
                    method_training, fallback, q), "validation_loss": loss(method_parent["candidate"].program,
                    method_validation, fallback, q), "selection_basis": "all_acquired_evidence"}
                quote = continuation_candidate["method_opportunity"]["response_compute_quote"]
                fitting = SearchBudget(min(quote, max(0, budget.remaining - 4 * length)))
                proposals = _observed_response_claims(agent, method_spec, method_training, old,
                    fitting, actions, method_validation, evidence)
                budget.spend(fitting.spent)
                best = min([old, *proposals], key=lambda item:
                    (max(item["training_loss"], item["validation_loss"]), item["candidate"].program.size))
                selected = (best if max(best["training_loss"], best["validation_loss"]) + agent.config.minimum_improvement
                             < max(old["training_loss"], old["validation_loss"]) else old)
                selected_channels = method_channels
                selected.update(channels=method_channels, fallback=fallback,
                    selection_basis="all_acquired_evidence")
                candidates = [selected, *proposals, *candidates]
                training = method_training
                validation = method_validation
            event["compute_cost"] = budget.spent - start_compute
            actions.append(event)
        # This visit has bought a method experiment and refitted its response.
        # Compare and confirm that frozen result before spending its remaining
        # allocation on another round of condition or representation research.
        if not method_experiment and not instrument_experiment and (selected_channels or len(laboratory.spec.feature_names) >= 4):
            selected, training, candidates = research_condition_response_actions(agent,
                agent._spec(laboratory, selected_channels), selected, rows_for(selected_channels), validation,
                candidates, budget, actions, acquire=acquire,
                remaining=remaining, serial=serial, channels=selected_channels,
                measurement_rate=sum(rates[name] for name in selected_channels))
        elif not method_experiment and not instrument_experiment:
            selected, training, candidates = agent._research_conditional_experiments(
                agent._spec(laboratory, ()), selected, rows_for(()), validation,
                candidates, budget, actions, acquire=acquire,
                observation_allowance=remaining()["observation_remaining"] - 2 * length,
                serial=serial)
        selected["channels"] = selected_channels
        selected.setdefault("fallback", initial_prediction(training, q))
        for candidate in candidates:
            candidate.setdefault("channels", selected_channels)
            candidate.setdefault("fallback", selected["fallback"])

    def adopt(proposal):
        nonlocal selected, selected_channels
        if proposal and proposal.get("selected"):
            selected, selected_channels = proposal["selected"], tuple(proposal["channels"])
            selected["channels"] = selected_channels
            selected.setdefault("fallback", initial_prediction(proposal.get("training", []), q))
            candidates.append(selected)
            return True
        return False

    if research is not None and (continuation_candidate is None
            or (q != .5 and (continuation_candidate or {}).get("risk_method") is not None)):
        if risk_method is not None:
            selected["risk_method"] = deepcopy(risk_method)
            selected["research_draft_id"] = continuation_candidate["research_draft_id"]
        adopt(research.propose(selected, channels=selected_channels, noise_assessment=support))

    comparisons = []
    method_confirmations = []
    marginal_comparisons, followup_origin = [], None
    instrument_origin, instrument_compared = None, False
    followup_experience = (agent.library.action_experience
        if agent.config.use_memory and agent.config.learn_research_policy else {})

    def current_costs():
        return {"observation_cost": spent_observations,
                "measurement_cost": spent_measurements, "compute_cost": budget.spent}

    def begin_method_followup(start_costs=None):
        nonlocal followup_origin
        if followup_origin is not None or not (method_experiment or instrument_experiment) or q != .5:
            return
        from .condition_marginal_value import condition_followup_policy
        policy = condition_followup_policy(agent._spec(laboratory, selected_channels),
            rows_for(selected_channels), selected, followup_experience)
        followup_origin = {"policy": policy, "program": selected["candidate"].program,
            "fallback": selected.get("fallback", 0.), "channels": selected_channels,
            "costs": current_costs() if start_costs is None else start_costs}

    def settle_method_followup(fresh_rows=None, probe_id=None):
        nonlocal followup_origin, instrument_compared
        if followup_origin is None:
            return
        from .condition_marginal_value import (CONDITION_FOLLOWUP_ACTION,
                                               record_condition_followup)
        origin, followup_origin = followup_origin, None
        before_loss, after_loss = None, None
        if fresh_rows and origin["channels"] == selected_channels:
            # Compare to the explanation in use when this extra work began.
            # A method's earlier repair of the original bad model cannot earn
            # another positive return for merely confirming the repaired law.
            if (origin["program"] == selected["candidate"].program
                    and origin["fallback"] == selected.get("fallback", 0.)):
                before_loss = confirmation_loss
            elif budget.spend(len(fresh_rows)):
                before_loss = loss(origin["program"], fresh_rows, origin["fallback"], q)
            if before_loss is not None:
                after_loss = confirmation_loss
                instrument_compared |= instrument_origin is not None
        costs = {key: value - origin["costs"][key] for key, value in current_costs().items()}
        if not any(costs.values()):
            return
        experience = record_condition_followup(origin["policy"], before_loss=before_loss,
            after_loss=after_loss, **costs)
        event = {"action": "compare_method_followup_on_fresh_response" if before_loss is not None
                 else "retain_unconfirmed_method_followup",
            "context": origin["policy"].action_context(CONDITION_FOLLOWUP_ACTION),
            "baseline_program": origin["program"].to_dict(),
            "baseline_fallback": origin["fallback"],
            "program": selected["candidate"].program.to_dict(),
            "fallback": selected.get("fallback", 0.), "channels": list(selected_channels),
            "before_loss": before_loss, "after_loss": after_loss,
            "probe_id": probe_id, "observation_rows": len(fresh_rows or ()),
            "stage_costs": costs, "costs_included_in_whole_visit": True,
            "trials": experience.get("trials", 0)}
        actions.append(event)
        marginal_comparisons.append(deepcopy(event))

    if instrument_experiment:
        from .condition_marginal_value import condition_followup_policy
        instrument_origin = {
            "policy": condition_followup_policy(agent._spec(laboratory, incumbent["channels"]),
                [row for batch in paid_view(laboratory, previous, incumbent["channels"]) for row in batch],
                incumbent, followup_experience),
            "program": incumbent["candidate"].program, "fallback": incumbent["fallback"],
            "channels": incumbent["channels"], "costs": {**old_costs,
                "compute_cost": old_costs["compute_cost"] - instrument_experiment["screening_compute_cost"]},
            "sampling_context": deepcopy(instrument_experiment["plan"].get("sampling_context")),
            "screening_compute_cost": instrument_experiment["screening_compute_cost"]}
        followup_origin = instrument_origin
    condition_confirmation_only = bool(q == .5 and any(
        action.get("action") == "revise_competing_observed_conditions"
        for action in actions[len(previous["actions"]):]))
    for attempt in range(agent.config.max_rounds):
        count = min(laboratory.spec.max_probe_length,
                    24 if condition_confirmation_only or paid_selection else length * 2 ** attempt)
        if condition_confirmation_only or paid_selection:
            channels = tuple(name for name in offered
                if name in {*selected_channels, *incumbent["channels"]})
            training = rows_for(channels)
            if (budget.remaining < 2 * count
                    or remaining()["observation_remaining"] < count
                    or remaining()["measurement_remaining"] < count * sum(rates[name] for name in channels)):
                break
            # The new response already won the common paid-evidence comparison
            # in the condition research action. Only a new response can now
            # confirm it; another selection trajectory would use its reserve.
            probe = None
        else:
            comparison_count = challenge_probe.cost if challenge and attempt == 0 else count
            if not challenge and q == .5 and previous["status"] != "completed":
                comparison_count = min(comparison_count,
                    max(0, remaining()["observation_remaining"] - count))
            channels = offered if research is not None else tuple(name for name in offered
                if name in {*selected_channels, *incumbent["channels"]})
            training = rows_for(channels)
            if (comparison_count == 0 or budget.remaining < 2 * (comparison_count + count)
                    or remaining()["observation_remaining"] < comparison_count + count
                    or remaining()["measurement_remaining"] < (comparison_count + count) * sum(rates[name] for name in channels)):
                if challenge and not scope_counterexamples:
                    return keep_completed("deferred", "challenge_and_confirmation_unfunded")
                break
            comparison = acquire(channels, "challenge_completed_hypotheses" if challenge and attempt == 0
                else "compare_continuation_candidates", rows_count=comparison_count,
                probe=challenge_probe if challenge and attempt == 0 else None)
            if comparison is None or not budget.spend(2 * len(comparison)):
                break
            if method_experiment and q == .5:
                errors = [point_loss(predict(incumbent["candidate"].program, row,
                    incumbent["fallback"]), row.target, q) for row in comparison]
                before = mean(errors)
                method_new_error = max(method_new_error, max(errors))
            else:
                before = loss(incumbent["candidate"].program, comparison, incumbent["fallback"], q)
            proposed = loss(selected["candidate"].program, comparison, selected.get("fallback", 0.), q)
            record_source_comparison(selected, comparison, before, proposed,
                batches[-1]["probe"]["probe_id"], accepted=proposed < before - agent.config.minimum_improvement)
            attempted_program = selected["candidate"].program.to_dict()
            if previous["status"] == "completed" and not challenge and q == .5:
                completed_refuted |= before > agent.config.minimum_improvement
            if challenge and attempt == 0:
                gains = [point_loss(predict(incumbent["candidate"].program, row, incumbent["fallback"]), row.target, q)
                         - point_loss(predict(selected["candidate"].program, row, selected.get("fallback", 0.)), row.target, q)
                         for row in comparison]
                margin = max(agent.config.minimum_improvement, 3 * stdev(gains) / sqrt(len(gains))) if len(gains) > 1 else float("inf")
                noisy = q != .5 or bool((previous.get("noise_assessment") or {}).get("adequate")
                                        and (previous.get("noise_assessment") or {}).get("reason") != "deterministic_fit")
                counterexample = mean(gains) > margin if noisy else before > agent.config.minimum_improvement
                event = {"action": "compare_completed_hypotheses", "probe_id": challenge_probe.probe_id,
                    "baseline_program": incumbent["candidate"].program.to_dict(), "baseline_fallback": incumbent["fallback"],
                    "program": attempted_program, "fallback": selected.get("fallback", 0.),
                    "before_loss": before, "after_loss": proposed, "gain": mean(gains), "gain_margin": margin,
                    "channels": list(channels), "risk_quantile": q, "observation_rows": len(comparison),
                    "source_concept_ids": continuation_candidate.get("source_concept_ids", []),
                    "predicted_separation": challenge.get("predicted_separation"), "counterexample": counterexample}
                actions.append(event)
                if not counterexample:
                    return keep_completed("rejected", "new_response_did_not_refute_completed_hypothesis")
                completed_refuted = True
                scope_counterexamples.append(deepcopy(event))
                report["scope_revision"] = {"status": "restricted_by_measured_counterexample",
                    "previous_program": previous["program"], "previous_channels": previous["selected_channels"],
                    "previous_status": previous["status"], "previous_confirmation_loss": previous.get("confirmation_loss"),
                    "validated_trajectories": [{"probe_id": batch["probe"]["probe_id"],
                        "length": len(batch["probe"]["sequence"])} for batch in previous["batches"]],
                    "counterexamples": scope_counterexamples}
                if agent.config.use_memory:
                    old_program = incumbent["candidate"].program.expanded().simplified()
                    matching = [concept for concept in agent.library.concepts
                        if concept.status != "revoked" and concept.risk_quantile == q
                        and concept.program.expanded().simplified() == old_program]
                    memory_events = []
                    # The measured old program identifies which stored law failed.
                    # A called subexpression is not blamed for its caller's error.
                    failing = [row for row in comparison if point_loss(
                        predict(incumbent["candidate"].program, row, incumbent["fallback"]), row.target, q)
                        > agent.config.minimum_improvement]
                    for concept in matching:
                        if not failing:
                            break
                        event = agent.library.record_counterexample(concept.concept_id,
                            agent._spec(laboratory, incumbent["channels"]), before, proposed, failing[0],
                            budget=budget, supporting_observations=[*training, *comparison])
                        memory_events.append(event)
                        actions.append({"action": "revise_completed_concept_scope", "evidence": deepcopy(event)})
                    report["scope_revision"]["memory_events"] = memory_events
            # An uninformative new batch cannot restore a model already
            # contradicted by the purchased response. This method's refit was
            # selected on all paid rows, including the incumbent's evidence.
            # Its summed squared errors bound the errors on the old subset;
            # compare those totals because the two fits have different counts.
            paid_repair = (method_experiment and q == .5
                and selected_channels == incumbent["channels"]
                and selected["candidate"].program != incumbent["candidate"].program
                and selected.get("selection_basis") == "all_acquired_evidence"
                and abs(before - proposed) <= agent.config.minimum_improvement
                and len(training) * max(selected["training_loss"], selected["validation_loss"])
                    < initial_paid_rows * (incumbent["training_loss"]
                                           - agent.config.minimum_improvement))
            if before <= proposed + agent.config.minimum_improvement and not paid_repair:
                selected, selected_channels = incumbent, incumbent["channels"]
                proposed = before
            elif paid_repair:
                actions.append({"action": "retain_paid_counterexample_repair",
                    "probe_id": batches[-1]["probe"]["probe_id"],
                    "incumbent_paid_loss": max(incumbent["training_loss"], incumbent["validation_loss"]),
                    "revised_paid_loss_upper_bound": len(training)
                        * max(selected["training_loss"], selected["validation_loss"]) / initial_paid_rows,
                    "previous_paid_rows": initial_paid_rows, "fitting_rows": len(training),
                    "new_incumbent_loss": before, "new_revised_loss": proposed,
                    "program": selected["candidate"].program.to_dict()})
            probe = Probe(batches[-1]["probe"]["probe_id"], tuple(batches[-1]["probe"]["sequence"]))
            paired = {"action": "compare_continuation_candidates", "probe_id": probe.probe_id,
                "baseline_program": incumbent["candidate"].program.to_dict(), "baseline_fallback": incumbent["fallback"],
                "program": selected["candidate"].program.to_dict(), "fallback": selected.get("fallback", 0.),
                "proposed_program": attempted_program, "channels": list(channels), "risk_quantile": q,
                "before_loss": before, "after_loss": proposed, "observation_rows": len(comparison)}
            actions.append(paired)
            comparisons.append(deepcopy(paired))
            measured_choice = (deepcopy(selected), selected_channels, None, None)
        # The chosen program is now frozen. Its confirmation uses both new
        # controls and new responses, independently of candidate selection.
        training = rows_for(channels)
        confirmation_plan = None
        confirmation_start_costs = current_costs()
        if (method_experiment or instrument_experiment) and q == .5 and selected_channels:
            # Confirm the revision against the explanation in use before this
            # work began. Its distinction can differ from the remaining cut
            # uncertainty of the new explanation.
            from .observed_condition_experiments import plan_observed_condition_probe
            origin = followup_origin or {"program": incumbent["candidate"].program,
                "fallback": incumbent["fallback"], "channels": incumbent["channels"]}
            confirmation_baseline = None
            if (origin["channels"] == selected_channels
                    and (origin["program"] != selected["candidate"].program
                         or origin["fallback"] != selected.get("fallback", 0.))):
                confirmation_baseline = {
                    "candidate": Candidate("confirmation_baseline", origin["program"],
                                           "explanation_before_research"),
                    "fallback": origin["fallback"], "channels": origin["channels"]}
            planning = SearchBudget(min(15000, max(0, budget.remaining - 2 * count)))
            # A frozen pair supplies its own question. Reassessing the new
            # cut before planning can consume its whole confirmation allowance;
            # fresh measured responses will be assessed below in either case.
            condition_state = (selected.get("condition_uncertainty")
                if confirmation_baseline is not None else assess_observed_condition_uncertainty(
                agent._spec(laboratory, selected_channels), training, selected, planning,
                previous=selected.get("condition_uncertainty"),
                tolerance=agent.config.minimum_improvement))
            unresolved_cut = (condition_state is not None and condition_state["status"] == "unresolved"
                    and condition_state.get("original_target_loss") is not None
                    and condition_state["original_target_loss"] <= agent.config.minimum_improvement)
            if unresolved_cut:
                selected["condition_uncertainty"] = condition_state
            if confirmation_baseline is not None or unresolved_cut:
                confirmation_plan = plan_observed_condition_probe(
                    agent._spec(laboratory, selected_channels), training, selected, planning,
                    serial=serial, length=count, tolerance=agent.config.minimum_improvement,
                    confirmation_baseline=confirmation_baseline)
                if (confirmation_plan is not None and
                        confirmation_plan["expected_disagreement"] <= agent.config.minimum_improvement):
                    confirmation_plan = None
            budget.spend(planning.spent)
            if confirmation_plan is not None:
                actions.append({"action": "plan_condition_confirmation",
                    **{key: value for key, value in confirmation_plan.items() if key != "probe"},
                    "probe_id": confirmation_plan["probe"].probe_id,
                    "planning_compute_cost": planning.spent})
        if confirmation_plan is not None:
            from .condition_confirmation import acquire_condition_confirmation
            begin_method_followup(confirmation_start_costs)
            validation = acquire_condition_confirmation(agent._spec(laboratory, selected_channels),
                selected, confirmation_plan, budget, actions, acquire=acquire,
                remaining=remaining, rows_for=rows_for, channels=channels,
                measurement_rate=sum(rates[name] for name in channels),
                tolerance=agent.config.minimum_improvement,
                experience=followup_experience)
        else:
            validation = acquire(channels, "confirm_continuation_program", rows_count=count)
        if validation is None or not budget.spend(2 * len(validation)):
            break
        confirmation_loss = loss(selected["candidate"].program, validation, selected.get("fallback", 0.), q)
        if method_experiment and q == .5:
            errors = [point_loss(predict(incumbent["candidate"].program, row,
                incumbent["fallback"]), row.target, q) for row in validation]
            incumbent_loss = mean(errors)
            method_new_error = max(method_new_error, max(errors))
        else:
            incumbent_loss = loss(incumbent["candidate"].program, validation, incumbent["fallback"], q)
        attempted_confirmation_loss = confirmation_loss
        record_source_comparison(selected, validation, incumbent_loss, confirmation_loss,
            batches[-1]["probe"]["probe_id"],
            accepted=confirmation_loss <= incumbent_loss + agent.config.minimum_improvement)
        attempted_confirmation = selected
        attempted_channels = selected_channels
        if previous["status"] == "completed" and not challenge and q == .5:
            completed_refuted |= incumbent_loss > agent.config.minimum_improvement
        confirmation_rows = len(validation)
        confirmation_probe_id = batches[-1]["probe"]["probe_id"]
        confirmation_probe = Probe(confirmation_probe_id, tuple(batches[-1]["probe"]["sequence"]))
        confirmed = confirmation_loss <= agent.config.minimum_improvement
        # A pair confirmation can distinguish two frozen responses while still
        # showing that both are wrong on the new trajectory.  Consume that
        # receipt as a response-model reopening signal; otherwise the generic
        # method loop treats the batch as an ordinary failed confirmation and
        # stops without purchasing the next response research step.
        from .condition_confirmation import pair_confirmation_reopen_required
        pair_reopen_required = pair_confirmation_reopen_required(actions, confirmation_probe_id)
        support = None
        if not confirmed and agent.config.risk_learning and not paid_selection:
            support = agent._support(_guard(selected["candidate"].program, selected.get("fallback", 0.)),
                                     training, validation, agent._spec(laboratory, channels), budget)
            if support["adequate"] and support["reason"] == "noise_compatible":
                tested = distinguish_risk_explanations(support, spec=agent._spec(laboratory, channels),
                    acquire=acquire, channels=channels, remaining=remaining, budget=budget,
                    minimum_improvement=agent.config.minimum_improvement,
                    measurement_rate=sum(rates[name] for name in channels), reserve_rows=length,
                    reserve_compute=len(training) * 6 + length * 16)
                actions.extend(tested["events"])
                support = tested["support"]
                if not tested["observation_cost"]:
                    repeated = (acquire(channels, "distinguish_noise_from_repeatable_error",
                        probe=Probe(f"repeat_noise_{serial}", confirmation_probe.sequence))
                        if budget.remaining >= (4 if q == .5 else 3) * len(validation) else None)
                    if repeated is None:
                        support.update(adequate=False, reason="noise_experiment_unfunded")
                    else:
                        evidence = assess_noise_replication(selected["candidate"].program, validation,
                            repeated, budget, minimum_improvement=agent.config.minimum_improvement,
                            risk_quantile=q, fallback=selected.get("fallback", 0.))
                        actions.append({"action": "compare_noise_replication", **evidence})
                        support["replication"] = evidence
                        if evidence["repeatable_error"] or not evidence["noise_supported"]:
                            support.update(adequate=False, reason="repeatable_unexplained_response"
                                if evidence["repeatable_error"] else "noise_experiment_unfunded"
                                if evidence["reason"] == "insufficient_budget" else "noise_replication_inconclusive")
            confirmed = support["adequate"]
        if scope_counterexamples:
            challenge_batch = next((batch for batch in batches
                if batch["probe"]["probe_id"] == challenge_probe.probe_id), None)
            challenge_rows = [Observation(item["features"], item["target"], tuple(item["history"]))
                for item in (challenge_batch or {}).get("observations", [])]
            if budget.spend(len(challenge_rows)):
                retained_loss = (loss(selected["candidate"].program, challenge_rows,
                    selected.get("fallback", 0.), q) if challenge_rows else None)
            else:
                retained_loss = None
            if retained_loss is None or (q == .5 and retained_loss > agent.config.minimum_improvement):
                scope_supported = False
                scope_reason = "retained_scope_counterexample"
                if (retained_loss is not None and confirmed and support
                        and (support.get("replication") or {}).get("noise_supported")):
                    scope_channels = tuple(challenge_batch["channels"])
                    scope_probe = Probe(f"repeat_scope_noise_{serial}",
                                        tuple(challenge_batch["probe"]["sequence"]))
                    repeated = (acquire(scope_channels, "retest_scope_counterexample_noise", probe=scope_probe)
                        if budget.remaining >= (4 if q == .5 else 3) * len(challenge_rows) else None)
                    if repeated is None:
                        scope_reason = "scope_noise_experiment_unfunded"
                    else:
                        evidence = assess_noise_replication(selected["candidate"].program,
                            challenge_rows, repeated, budget,
                            minimum_improvement=agent.config.minimum_improvement,
                            risk_quantile=q, fallback=selected.get("fallback", 0.))
                        actions.append({"action": "compare_scope_noise_replication",
                            "source_probe_id": challenge_probe.probe_id,
                            "probe_id": scope_probe.probe_id, **evidence})
                        support["scope_replication"] = evidence
                        scope_supported = evidence["noise_supported"]
                if not scope_supported:
                    confirmed = False
                    support = {**(support or {}), "adequate": False, "reason": scope_reason}
        # A candidate can win the selection batch and lose the independent
        # confirmation. Keep that failure, and retain the better old predictor.
        joint_method_mismatch = pair_reopen_required
        if pair_reopen_required:
            support = {"adequate": False, "reason": "common_response_counterexample",
                       "pair_confirmation_reopen": True}
            actions.append({"action": "reopen_response_after_pair_confirmation",
                "probe_id": confirmation_probe_id,
                "baseline_loss": incumbent_loss,
                "selected_loss": confirmation_loss,
                "reason": "both_frozen_responses_missed_fresh_trajectory"})
        if incumbent_loss + agent.config.minimum_improvement < confirmation_loss:
            rejected = selected
            rejected_support = support
            if (q == .5 and agent.config.use_memory
                    and (method_experiment or rejected.get("transfer_evidence"))):
                replication = (rejected_support or {}).get("replication")
                refuted = (not agent.config.risk_learning or
                    (not (rejected_support or {}).get("adequate") and
                     (replication is not None or (rejected_support or {}).get("reason") in {
                         "predictable_residual", "residual_loss_shift",
                         "deterministic_response_mismatch"})))
                source_ids = [origin["source_id"] for origin in rejected.get("transfer_evidence", [])]
                draft_ids = [rejected["research_draft_id"]] if rejected.get("research_draft_id") else []
                if refuted and (source_ids or draft_ids):
                    evidence = {"status": "counterexample", "rows": len(validation),
                        "measured_loss": attempted_confirmation_loss,
                        "retained_loss": incumbent_loss,
                        "noise_assessment": deepcopy(rejected_support)}
                    if replication is not None:
                        evidence["noise_replication"] = deepcopy(replication)
                    withdrawn = agent.library.record_transfer_counterexample(
                        agent._spec(laboratory, selected_channels),
                        replace(rejected["candidate"], parents=tuple(dict.fromkeys(
                            [*rejected["candidate"].parents, *source_ids]))), source_ids,
                        source_draft_ids=draft_ids,
                        fallback=rejected.get("fallback", 0.), evidence=evidence,
                        probe_id=confirmation_probe_id)
                    if withdrawn is not None:
                        actions.append({**withdrawn, "action": "withdraw_target_draft"
                            if draft_ids else "withdraw_target_transfer"})
            selected, selected_channels = incumbent, incumbent["channels"]
            confirmation_loss, confirmed = incumbent_loss, False
            support = {"adequate": False, "reason": "continuation_confirmation_regression"}
            if (method_experiment and q == .5
                    and rejected["candidate"].program != incumbent["candidate"].program
                    and incumbent_loss > agent.config.minimum_improvement
                    and not (rejected_support or {}).get("adequate")):
                # Rejecting a newly fitted explanation is new information even
                # when the retained model's error is below its historical peak.
                # Test that model on the same fresh response before deciding
                # whether its remaining error calls for reconstruction or noise.
                retained_support = (agent._support(
                    _guard(incumbent["candidate"].program, incumbent["fallback"]),
                    training, validation, agent._spec(laboratory, selected_channels), budget)
                    if agent.config.risk_learning else
                    {"adequate": False, "reason": "deterministic_response_mismatch"})
                joint_method_mismatch = retained_support["reason"] in {
                    "predictable_residual", "residual_loss_shift", "deterministic_response_mismatch"}
                actions.append({"action": "retain_method_confirmation_counterexample",
                    "probe_id": confirmation_probe_id,
                    "rejected_program": rejected["candidate"].program.to_dict(),
                    "rejected_loss": attempted_confirmation_loss,
                    "retained_program": incumbent["candidate"].program.to_dict(),
                    "retained_loss": incumbent_loss,
                    "retained_noise_assessment": deepcopy(retained_support),
                    "joint_response_mismatch": joint_method_mismatch})
                support["retained_noise_assessment"] = retained_support
        current_fit_confirmed = confirmation_loss <= agent.config.minimum_improvement
        selected["current_fit_confirmed"] = current_fit_confirmed
        if q == .5 and (confirmed or selected.get("condition_uncertainty") is not None):
            paid_rows = rows_for(selected_channels)
            condition_state = selected.get("condition_uncertainty")
            if not (confirmation_plan is not None and condition_state is not None
                    and condition_state.get("fit_rows") == len(paid_rows)
                    and condition_state.get("program") == selected["candidate"].program.to_dict()):
                condition_state = assess_observed_condition_uncertainty(
                    agent._spec(laboratory, selected_channels), paid_rows, selected, budget,
                    previous=condition_state, tolerance=agent.config.minimum_improvement)
            selected["condition_uncertainty"] = condition_state
            if condition_state is not None and condition_state["status"] != "supported":
                confirmed = False
        actions.append({"action": "validate_continued_program", "attempt": attempt,
            "program": selected["candidate"].program.to_dict(), "loss": confirmation_loss,
            "probe_id": confirmation_probe_id, "selection_probe_id": probe.probe_id if probe else None,
            "accepted": confirmed, "current_fit_confirmed": current_fit_confirmed,
            "condition_uncertainty": deepcopy(selected.get("condition_uncertainty")),
            "noise_assessment": deepcopy(support), "observation_rows": len(validation)})
        if paid_selection:
            actions[-1]["selection_evidence_batches"] = list(paid_selection["evidence_batches"])
            actions[-1]["frozen_paid_selection"] = True
        if method_experiment or risk_method is not None:
            attempted_sources = ([attempted_confirmation["research_draft_id"]]
                if attempted_confirmation is not incumbent
                and attempted_confirmation.get("research_draft_id") else [])
            paired = {"action": "compare_successor_method_on_fresh_response",
                "probe_id": confirmation_probe_id, "observation_rows": len(validation),
                "before_loss": incumbent_loss, "after_loss": confirmation_loss,
                "attempted_after_loss": attempted_confirmation_loss,
                "attempted_program": attempted_confirmation["candidate"].program.to_dict(),
                "attempted_fallback": attempted_confirmation.get("fallback", 0.),
                "attempted_channels": list(attempted_channels),
                "attempted_source_draft_ids": attempted_sources,
                "baseline_program": incumbent["candidate"].program.to_dict(),
                "baseline_fallback": incumbent["fallback"],
                "program": selected["candidate"].program.to_dict(),
                "fallback": selected.get("fallback", 0.), "channels": list(channels),
                "source_draft_ids": list(method_experiment["source_draft_ids"] if method_experiment
                    else continuation_candidate["method_opportunity"]["draft_ids"]),
                "risk_quantile": q}
            if method_experiment and q == .5 and attempted_sources:
                from .numeric_method_revisits import method_response_applicability
                paired["attempted_method_applicability"] = method_response_applicability(
                    attempted_confirmation["candidate"].program,
                    agent._spec(laboratory, attempted_channels), incumbent)
            actions.append(paired)
            method_confirmations.append(deepcopy(paired))
        settle_method_followup(validation, confirmation_probe_id)
        measured_choice = (deepcopy(selected), selected_channels, confirmation_loss, deepcopy(support))
        if research is not None:
            research.settle(selected, validation, channels, selected_channels=selected_channels,
                confirmed=confirmed, fallback=selected.get("fallback", 0.), probe_id=confirmation_probe_id)
        if confirmed or paid_selection:
            break
        if attempt + 1 < agent.config.max_rounds:
            condition_state = selected.get("condition_uncertainty")
            if method_experiment and q == .5:
                from .condition_research_returns import _evidence_change, _question
                tolerance = agent.config.minimum_improvement
                change = _evidence_change(_question(method_previous_condition),
                    _question(condition_state), tolerance)
                if incumbent_loss - confirmation_loss > tolerance:
                    reason = "fresh_response_improved"
                elif joint_method_mismatch:
                    reason = "new_revision_refuted_with_joint_response_mismatch"
                elif method_structure_evidence:
                    reason = "new_measured_structure"
                elif change["new_counterexample"] or change["interval_narrowed"]:
                    reason = "new_condition_evidence"
                elif method_known_error is None:
                    reason = "previous_error_range_unassessed"
                elif method_new_error > method_known_error + tolerance:
                    reason = "new_response_counterexample"
                elif (condition_state is not None and condition_state["status"] != "supported"
                        and condition_state.get("original_target_loss") is not None
                        and condition_state["original_target_loss"] <= tolerance):
                    reason = "unresolved_condition_with_supported_response"
                else:
                    reason = "method_failed_without_new_information"
                proceed = reason != "method_failed_without_new_information"
                actions.append({"action": "choose_method_followup_research", "continue": proceed,
                    "reason": reason, "fresh_gain": incumbent_loss - confirmation_loss,
                    "previous_max_point_loss": method_known_error,
                    "new_max_point_loss": method_new_error,
                    "condition_evidence_change": change,
                    "supported_structure": method_structure_evidence,
                    "joint_response_mismatch": joint_method_mismatch})
                if not proceed:
                    break
            from .condition_response_actions import (observed_condition_followup,
                                                    research_observed_condition_competition)
            condition_followup = (observed_condition_followup(condition_state,
                agent.config.minimum_improvement) if q == .5 else None)
            if condition_followup is not None:
                available = remaining()
                actions.append({"action": "choose_observed_response_followup",
                    "research_action": condition_followup,
                    "evidence_basis": "all_paid_original_response",
                    "paid_rows": condition_state["fit_rows"],
                    "paid_response_loss": condition_state["original_target_loss"],
                    "current_response_loss": confirmation_loss, **available})
            if condition_followup in ("revise_paid_condition", "distinguish_observed_condition"):
                begin_method_followup()
                paid_batches = paid_view(laboratory, report, selected_channels)
                checked_rows = paid_batches[-1] if paid_batches else []
                before_batches, before_program = len(batches), selected["candidate"].program
                selected, training, candidates = research_observed_condition_competition(agent,
                    agent._spec(laboratory, selected_channels), selected, rows_for(selected_channels),
                    checked_rows, candidates, budget, actions, acquire=acquire, remaining=remaining,
                    serial=serial, channels=selected_channels,
                    measurement_rate=sum(rates[name] for name in selected_channels))
                selected["channels"] = selected_channels
                selected.setdefault("fallback", initial_prediction(training, q))
                if len(batches) == before_batches and selected["candidate"].program == before_program:
                    actions.append({"action": "stop_condition_followup_research",
                        "reason": "no_new_observations_or_response_revision"})
                    break
                condition_confirmation_only = True
                continue
            if research is not None:
                research.length = min(laboratory.spec.max_probe_length, length * 2 ** (attempt + 1))
                if not adopt(research.propose(selected, channels=selected_channels,
                    failed_probe=confirmation_probe, noise_assessment=support)):
                    break
            else:
                paid_batches = paid_view(laboratory, report, selected_channels)
                rows = [row for batch in paid_batches[:-1] for row in batch]
                checked_rows = paid_batches[-1] if paid_batches else []
                next_count = min(laboratory.spec.max_probe_length, length * 2 ** (attempt + 1))
                spec = agent._spec(laboratory, selected_channels)
                reserve = 2 * next_count + agent._confirmation_compute_quote(spec, next_count,
                    training_rows=len(rows) + len(checked_rows) + next_count,
                    rivals=1, noise_check=agent.config.risk_learning)
                if (remaining()["observation_remaining"] < 2 * next_count
                        or remaining()["measurement_remaining"]
                            < 2 * next_count * sum(rates[name] for name in selected_channels)
                        or budget.remaining <= reserve):
                    actions.append({"action": "stop_method_response_reconstruction",
                        "reason": "independent_comparison_and_confirmation_unfunded",
                        "confirmation_compute_reserve": reserve, **remaining()})
                    break
                begin_method_followup()
                local = SearchBudget(budget.remaining - reserve)
                revised = agent._research_view(agent._spec(laboratory, selected_channels), rows, checked_rows,
                    local, actions, transfer, graph, parent=selected,
                    observation_remaining=remaining()["observation_remaining"],
                    measurement_remaining=remaining()["measurement_remaining"])
                budget.spend(local.spent)
                if not revised:
                    break
                next_selected = agent._best(revised)
                if (next_selected["candidate"].program == selected["candidate"].program
                        and next_selected.get("fallback", selected.get("fallback", 0.))
                            == selected.get("fallback", 0.)):
                    actions.append({"action": "stop_method_response_reconstruction",
                        "reason": "no_changed_response_proposal"})
                    break
                selected = next_selected
                selected["channels"] = selected_channels
                selected.setdefault("fallback", initial_prediction(rows, q))
                candidates.extend(revised)
    settle_method_followup()
    if research is not None:
        research.finish()
    condition_state = measured_choice[0].get("condition_uncertainty")
    condition_pending = condition_state is not None and condition_state["status"] != "supported"
    if previous["status"] == "completed" and not completed_refuted and not condition_pending:
        return keep_completed("deferred" if not comparisons else "rejected",
                              "no_measured_counterexample_to_completed_hypothesis")
    # A newly fitted tail proposal may have no affordable next experiment.
    # Preserve it as a research alternative, not as the returned best predictor.
    pending_proposal = selected
    selected, selected_channels, confirmation_loss, support = measured_choice
    program, fallback = selected["candidate"].program, selected.get("fallback", 0.)
    procedure = selected.get("research_program")
    recipe = procedure or previous.get("research_program")
    if recipe and any(step["operation"] in ("discover_conditional_coordinates", "discover_composed_conditions")
                      for step in recipe.get("selection_steps", [])):
        from .conditional_coordinates import adopt_conditional_research_program
        procedure = adopt_conditional_research_program(recipe, program,
            measured_loss=selected.get("training_loss", confirmation_loss),
            fit_rows=len(training), compute_cost=budget.spent - old_costs["compute_cost"])
    if procedure is not None:
        procedure = deepcopy(procedure)
        procedure.update(status="supported_on_fresh_trajectory" if confirmed else "requires_fresh_validation",
                         confirmation_loss=confirmation_loss, confirmation_rows=confirmation_rows)
        selected["research_program"] = procedure
    if not confirmed and procedure and agent.config.use_memory and agent.config.strategy != "fixed":
        from .research_drafts import remember_research_draft
        remember_research_draft(agent.library, selected,
            agent._spec(laboratory, selected_channels), actions)
    concept_ids, sources = [], []
    if confirmed and agent.config.use_memory and agent.config.strategy != "fixed":
        from .meta_prior_research import _sources
        spec = agent._spec(laboratory, selected_channels)
        sources = list(dict.fromkeys([*_sources(program), *selected.get("method_source_ids", [])]))
        candidate = selected["candidate"]
        candidate = Candidate(candidate.candidate_id, candidate.program, candidate.operation,
            candidate.rationale, tuple(dict.fromkeys([*candidate.parents, *sources])), candidate.memory_id)
        concept_id, added = agent.library.learn(candidate, spec, rows_for(selected_channels))
        concept_ids = [concept_id] if added else []
        if selected.get("research_draft_id") is not None:
            draft = next(item for item in agent.library.research_drafts
                         if item["draft_id"] == selected["research_draft_id"])
            if spec.task_id == draft["source_task_id"]:
                draft["resolved_concept_id"] = concept_id
                actions.append({"action": "resolve_research_draft", "draft_id": draft["draft_id"],
                    "concept_id": concept_id, "confirmation_loss": confirmation_loss,
                    "confirmation_rows": confirmation_rows})
            else:
                draft["outcomes"].append({"task_id": spec.task_id, "concept_id": concept_id,
                    "status": "supported_on_fresh_trajectory", "confirmation_loss": confirmation_loss,
                    "confirmation_rows": confirmation_rows})
        for source in sources:
            agent.library.record_use(source, spec.task_id)
        if not any(prior["source_concept_id"] == concept_id for prior in agent.library.meta_priors):
            agent.library.meta_priors.append({"prior_id": f"meta_prior_{len(agent.library.meta_priors) + 1:04d}",
                "source_concept_id": concept_id, "family_id": selected.get("family_id", "T03"),
                "kind": selected["kind"], "claim": "The continued hypothesis predicts a new measured trajectory",
                "status": "supported", "task_id": spec.task_id, "source_features": list(spec.feature_names),
                "control_features": list(spec.feature_bounds), "object_schema": None, "structure": None,
                "support_rows": len(rows_for(selected_channels)), "validation_loss": confirmation_loss,
                "probe_ids": [batch["probe"]["probe_id"] for batch in batches], "counterexamples": [], "uses": 0,
                "parent_concept_ids": sources, "research_program": deepcopy(procedure),
                "research_method": {"fit_rows": len(training),
                    "validation_rows": confirmation_rows, "confirmation_rows": confirmation_rows,
                    "noise_supported": bool(support and support["adequate"]), "risk_quantile": q},
                "counter_prediction": {"type": "future_trajectory", "condition": "A later measured response"}})
        elif procedure is not None:
            prior = next(item for item in agent.library.meta_priors
                         if item["source_concept_id"] == concept_id)
            prior["research_program"] = deepcopy(procedure)
    costs = {"observation_cost": spent_observations, "measurement_cost": spent_measurements,
             "compute_cost": budget.spent, "queries": len(batches)}
    if instrument_origin is not None and instrument_compared:
        from .condition_marginal_value import record_instrument_revisit_cost
        origin = instrument_origin
        record_instrument_revisit_cost(origin["policy"],
            **{key: value - origin["costs"][key] for key, value in current_costs().items()},
            sampling_context=origin["sampling_context"],
            screening_compute_cost=origin["screening_compute_cost"])
    alternatives = []
    for item in [incumbent, selected, pending_proposal, *candidates]:
        payload = {"program": item["candidate"].program.to_dict(), "channels": list(item.get("channels", selected_channels)),
            "fallback": item.get("fallback", 0.), "kind": item["kind"], "family_id": item.get("family_id", "T03"),
            "source_concept_ids": item.get("method_source_ids", []),
            "transfer_evidence": [{**deepcopy(origin), "adaptation_work": []}
                                  for origin in item.get("transfer_evidence", [])],
            "research_program": deepcopy(item.get("research_program")),
            "condition_uncertainty": deepcopy(item.get("condition_uncertainty"))}
        payload.update(condition_research_returns=deepcopy(item.get("condition_research_returns")),
                       pair_confirmation_returns=deepcopy(item.get("pair_confirmation_returns")),
                       research_draft_id=item.get("research_draft_id"))
        if payload not in alternatives:
            alternatives.append(payload)
    alternatives = alternatives[:agent.config.max_candidates]
    report.update(status="completed" if confirmed else "unresolved", program=program.to_dict(),
        expression=program.describe(), fallback=fallback, selected_kind=selected["kind"],
        research_program=procedure,
        selected_channels=list(selected_channels), confirmation_loss=confirmation_loss,
        current_fit_confirmed=selected.get("current_fit_confirmed"),
        condition_uncertainty=deepcopy(selected.get("condition_uncertainty")),
        condition_research_returns=deepcopy(selected.get("condition_research_returns")),
        pair_confirmation_returns=deepcopy(selected.get("pair_confirmation_returns")),
        research_draft_id=selected.get("research_draft_id"),
        risk_method=deepcopy(selected.get("risk_method")),
        validation_loss=None, selection_loss=selected.get("validation_loss"), selection_basis="all_acquired_evidence",
        noise_assessment=support, costs=costs, visit_costs={key: value - old_costs.get(key, 0) for key, value in costs.items()},
        concepts_added=concept_ids, used_sources=sources,
        method_source_ids=selected.get("method_source_ids", []) if confirmed else [], research_graph=graph.to_dict(),
        transfer=transfer, measurement_program=None, measurement_channels=None, instrument_model_id=None,
        continuation_state={"alternatives": alternatives, "noise_assessment": deepcopy(support),
                            "condition_uncertainty": deepcopy(selected.get("condition_uncertainty"))},
        continuation={"source_concept_ids": (continuation_candidate or {}).get("source_concept_ids", []),
            "previous_status": previous["status"], "previous_costs": old_costs,
            "comparisons": comparisons, "paid_rows_reused": old_costs["observation_cost"]})
    if method_experiment or risk_method is not None:
        report["continuation"]["method_fresh_comparisons"] = method_confirmations
    if paid_selection:
        report["continuation"]["paid_selection"] = deepcopy(paid_selection)
    if marginal_comparisons:
        report["continuation"]["method_marginal_comparisons"] = marginal_comparisons
    if challenge:
        outcome = "revised" if confirmed else "reopened_unresolved"
        report["continuation"]["challenge_outcome"] = outcome
        report["scope_revision"].update(status="revised_after_independent_confirmation" if confirmed
            else "restricted_with_unresolved_revision", replacement_program=program.to_dict(),
            confirmation_loss=confirmation_loss)
        actions.append({"action": "completed_task_challenge", "outcome": outcome,
            "counterexample": bool(scope_counterexamples), "probe_id": challenge_probe.probe_id,
            "source_concept_ids": continuation_candidate.get("source_concept_ids", [])})
    finish_source_returns(selected, confirmation_loss, costs)
    return report
