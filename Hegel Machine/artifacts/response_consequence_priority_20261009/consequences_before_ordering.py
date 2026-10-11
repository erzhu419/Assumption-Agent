"""Bounded queries of shared response families on unmeasured conditions.

Member votes remain useful operational suggestions. They are not samples of a
parameter posterior; family progress instead compares a fixed physical panel.
"""

from copy import deepcopy
from itertools import product

from .language import Program
from .models import Observation
from .world_response_constraints import BRANCH_LIMIT, query_response_family


QUERY_ALLOWANCE = 50000
PANEL_SIZE = 4


def _point(item):
    return item["crystal"], item["frequency"]


def _families(estimators):
    unique = {}
    for estimator in estimators:
        key = tuple(sorted(set(Program.from_dict(term).simplified()
                               for term in estimator["basis"]), key=repr))
        if key not in unique:
            unique[key] = {"id": estimator["id"], "basis": [p.to_dict() for p in key],
                           "source_ids": []}
        unique[key]["source_ids"].append(estimator["id"])
    return list(unique.values())


def _panel(state, experiments, previous, observed):
    declared = state.get("consequence_panel")
    if declared is not None:
        return [deepcopy(p) for p in declared if _point(p) not in observed][:PANEL_SIZE]
    old = previous.get("panel", [])
    if old and all(_point(p) not in observed for p in old):
        return deepcopy(old)
    # Cover different unfinished objects before adding a second setting. No
    # outcomes, hidden parameters or benchmark case labels enter this panel.
    targets = [n for n in state.get("target_numbers", []) if n not in state.get("completed", [])]
    choices = {n: sorted({e["prediction"] for e in experiments if e["number"] == n}) for n in targets}
    panel = []
    for offset in range(PANEL_SIZE):
        for number in targets:
            values = choices[number]
            if offset < len(values):
                p = {"crystal": number, "frequency": values[offset]}
                if _point(p) not in observed:
                    panel.append(p)
            if len(panel) == PANEL_SIZE:
                return panel
    return panel


def _shared_premise_test(features, responses, families, experiments, family_states):
    """Test a shared representation by transferring an actual positive setting.

    Equal basis values require equal responses at the same setting in that
    family. A negative result can refute the premise even when it was impossible
    *inside* the family; this is distinct from within-family information gain.
    The action and any reference restoration come from the existing controller.
    """
    available = {(e["number"], e["prediction"]): e for e in experiments}
    observed = {_point(r) for r in responses}
    live = {f["id"] for f in family_states if f["status"] in {"SAT_WITNESS", "UNKNOWN"}}
    proposals = {}
    for family in families:
        if family["id"] not in live:
            continue
        terms = [Program.from_dict(t) for t in family["basis"]]
        values = {n: tuple(t.predict(Observation(row, 0.)) for t in terms)
                  for n, row in features.items()}
        for source in responses:
            if not source["activated"]:
                continue
            for number in values:
                point = number, source["frequency"]
                if (number == source["crystal"] or point in observed or point not in available
                        or values[number] != values[source["crystal"]]):
                    continue
                proposal = proposals.setdefault(point, {"experiment": available[point],
                    "source_response": deepcopy(source), "required_by_families": []})
                if family["id"] not in proposal["required_by_families"]:
                    proposal["required_by_families"].append(family["id"])
    # Prefer a contrast between representations. If all surviving families make
    # the same prediction, the experiment instead challenges their shared premise.
    ordered = sorted(proposals.values(), key=lambda p: (
        set(p["required_by_families"]) == live,
        p["experiment"]["number"], p["experiment"]["prediction"]))
    return ordered[0] if ordered else None


def assess_response_consequences(state, estimators, result):
    """Retain family consequences and let them change the existing action.

    Queries use all paid responses. Hypothetical outcomes only enter temporary
    queries. UNKNOWN remains possible, and a new family/panel starts a new
    comparison epoch. Computation is bounded independently of model calls.
    """
    table = state.get("research_table", {})
    if not estimators or not table.get("references"):
        return result
    features = {r["crystal"]: {k: v["value"] for k, v in r["measurements"].items()}
                for r in table.get("references", []) + table.get("targets", [])}
    responses = state.get("response_settings", [])
    observed = {_point(r) for r in responses}
    previous = state.get("response_research", {}).get("family_research", {})
    families = _families(estimators)
    definitions = [f["basis"] for f in families]
    experiments = result.get("experiments", [])
    panel = _panel(state, experiments, previous, observed)
    # JSON object keys in saved features are strings, unlike local object ids.
    saved_features = {str(k): v for k, v in features.items()}
    same_epoch = (previous.get("panel") == panel and previous.get("bases") == definitions
                  and previous.get("features") == saved_features)
    research = {"panel": panel, "bases": definitions, "features": saved_features,
                "epoch": previous.get("epoch", 0) + int(not same_epoch),
                "response_count": len(responses), "evidence": deepcopy(result.get("evidence", [])),
                "families": [], "signatures": [], "admitted_signatures": [],
                "possible_signatures": [], "excluded_prior_signatures": [],
                "nonlocal_progress": 0, "compute_cost": 0, "lp_calls": 0,
                "query_count": 0, "query_allowance": QUERY_ALLOWANCE,
                "experiment_values": [], "selected_role": "operational_exploration",
                "completed": list(state.get("completed", [])),
                "scope": "declared_shared_connected_response_families_on_fixed_public_panel"}
    cache = {}

    def query(family, conditions):
        key = (family["id"], tuple(sorted((_point(p), bool(p["activated"])) for p in conditions)))
        if key not in cache:
            # Adding hypothetical conditions cannot restore a family already
            # excluded by the paid history. UNKNOWN supplies no such inference.
            paid = cache.get((family["id"], ()))
            if paid is not None and paid["status"] == "UNSAT_VERIFIED":
                return paid
            remaining = QUERY_ALLOWANCE - research["compute_cost"]
            # Basis evaluation is the query's first charged operation. A small
            # unusable remainder used to trigger thousands of zero-work retries.
            if remaining <= 0 or remaining < len(features) * len(family["basis"]):
                return {"status": "UNKNOWN", "reason": "consequence_query_allowance_exhausted"}
            limit = max(1, min(BRANCH_LIMIT, remaining // max(1, 2 * (len(responses) + len(conditions)))))
            answer = query_response_family(features, responses, family, queries=conditions,
                                           branch_limit=limit, compute_limit=remaining)
            research["compute_cost"] += answer["compute_cost"]
            research["lp_calls"] += answer.get("lp_calls", 0)
            research["query_count"] += 1
            cache[key] = answer
        return cache[key]

    def union(conditions):
        unknown = False
        for family in families:
            status = query(family, conditions)["status"]
            if status == "SAT_WITNESS":
                return status
            unknown |= status != "UNSAT_VERIFIED"
        return "UNKNOWN" if unknown else "UNSAT_VERIFIED"

    for family in families:
        answer = query(family, [])
        research["families"].append({"id": family["id"], "source_ids": family["source_ids"],
            "basis": family["basis"], "status": answer["status"],
            "scope": answer.get("scope"), "reason": answer.get("reason")})
    for values in product((False, True), repeat=len(panel)) if panel else ():
        status = union([{**p, "activated": y} for p, y in zip(panel, values)])
        research["signatures"].append({"outcomes": list(values), "status": status})
        if status == "SAT_WITNESS":
            research["admitted_signatures"].append(list(values))
        if status != "UNSAT_VERIFIED":
            research["possible_signatures"].append(list(values))
    if same_epoch:
        excluded = {tuple(s["outcomes"]) for s in research["signatures"]
                    if s["status"] == "UNSAT_VERIFIED"}
        research["excluded_prior_signatures"] = [s for s in previous.get("admitted_signatures", [])
                                                 if tuple(s) in excluded]
        research["nonlocal_progress"] = len(research["excluded_prior_signatures"])
    elif (previous.get("bases") == definitions and previous.get("features") == saved_features
          and previous.get("declared_panel") == state.get("consequence_panel")
          and any(_point(p) in observed for p in previous.get("panel", []))):
        # A target test may consume one panel point and constrain the others.
        # Compare the still-unobserved coordinates before rotating the panel.
        old_panel = previous["panel"]
        keep = [i for i, p in enumerate(old_panel) if _point(p) not in observed]
        remaining_panel = [old_panel[i] for i in keep]
        if remaining_panel:
            prior = {tuple(s[i] for i in keep) for s in previous.get("admitted_signatures", [])}
            research["progress_panel"] = deepcopy(remaining_panel)
            research["excluded_prior_signatures"] = [list(s) for s in sorted(prior)
                if union([{**p, "activated": y} for p, y in zip(remaining_panel, s)]) == "UNSAT_VERIFIED"]
            research["nonlocal_progress"] = len(research["excluded_prior_signatures"])
    research["declared_panel"] = deepcopy(state.get("consequence_panel"))
    actual_new = len(responses) > previous.get("response_count", len(responses))
    prior_families = previous.get("families", [])
    research["refuted_family_ids"] = [f["id"] for f in research["families"]
        if actual_new and f["status"] == "UNSAT_VERIFIED"
        and previous.get("features") == saved_features
        and any(old["basis"] == f["basis"] and old["status"] == "SAT_WITNESS"
                for old in prior_families)]
    completed_before = set(previous.get("completed", []))
    task_progress = bool(set(state.get("completed", [])) - completed_before)
    last_test = previous.get("selected_test")
    finished = [r for r in responses[previous.get("response_count", len(responses)):]
                if last_test and _point(r) == _point(last_test)]
    stagnation = previous.get("no_progress_cycles", 0)
    target_activated = any(r["activated"] and r["crystal"] in state.get("target_numbers", [])
                           for r in finished)
    if (research["nonlocal_progress"] or task_progress or target_activated
            or research["refuted_family_ids"]):
        stagnation = 0
    elif finished:
        stagnation += 1
    research["no_progress_cycles"] = stagnation
    research["new_response_received"] = actual_new

    if research["families"] and all(f["status"] == "UNSAT_VERIFIED" for f in research["families"]):
        conflict = next(([deepcopy(a), deepcopy(b)] for i, a in enumerate(responses)
            for b in responses[i + 1:] if a["frequency"] == b["frequency"]
            and a["activated"] != b["activated"]
            and features[a["crystal"]] == features[b["crystal"]]), None)
        research.update(selected_role="shared_premise_research", research_obligation={
            "reason": "declared_shared_response_families_refuted",
            "question": ("Which additional measurement or observable state distinguishes the contradictory responses?"
                         if conflict else "Which response structure or measurement revises the refuted shared families?"),
            "family_statuses": [f["status"] for f in research["families"]],
            "same_measurement_conflict": conflict})
        result.update(status="family_research_needed", selection=None,
                      selected_number=None, selected_setting=None, family_research=research)
        result["compute_cost"] += research["compute_cost"]
        return result

    # Agreement here ranges over the declared families, including unresolved
    # queries. A single representative predicting success does not qualify.
    possible = research["possible_signatures"]
    certain_target = next((p for i, p in enumerate(panel)
        if p["crystal"] in state.get("target_numbers", [])
        and p["crystal"] not in state.get("completed", [])
        and possible and research["admitted_signatures"]
        and all(signature[i] for signature in possible)), None)
    if certain_target is not None and experiments and result.get("hypotheses"):
        hypothesis = result["hypotheses"][0]
        action = {"id": "family_target_" + str(certain_target["crystal"]) + "_" + str(certain_target["frequency"]),
            "number": certain_target["crystal"], "prediction": certain_target["frequency"],
            "family": hypothesis["family"], "candidate_id": hypothesis["candidate_id"],
            "outcome": "The target activates at the family-consistent setting",
            "procedure": deepcopy(experiments[0]["procedure"]),
            "execution_binding": experiments[0]["execution_binding"],
            "predictions": {h["id"]: h["lower"] <= certain_target["frequency"] -
                h["predictions_by_crystal"][str(certain_target["crystal"])] <= h["upper"]
                for h in result["hypotheses"]}}
        result.update(status="family_target_trial", selected_number=action["number"],
            selected_setting=action["prediction"], selection={"diagnosis": "family_target_trial",
                "selected_experiment": action, "hypotheses": deepcopy(result["hypotheses"]),
                "predicted_separation": 0., "compute_cost": 0})
        research.update(selected_role="conditional_family_task_action", selected_test=deepcopy(certain_target))
        result["family_research"] = research
        result["compute_cost"] += research["compute_cost"]
        return result

    # Cycle across subjects so one object's many proposed settings cannot spend
    # the entire query allowance before references are considered.
    by_subject = {}
    for experiment in experiments:
        by_subject.setdefault(experiment["number"], []).append(experiment)
    ordered = [items[i] for i in range(max(map(len, by_subject.values()), default=0))
               for items in by_subject.values() if i < len(items)]
    winner, best_gain = None, 0
    conditional_winner, possible_gain = None, 0
    admitted = research["admitted_signatures"]
    for experiment in ordered:
        if research["compute_cost"] >= QUERY_ALLOWANCE:
            break
        point = {"crystal": experiment["number"], "frequency": experiment["prediction"]}
        keep = [i for i, p in enumerate(panel) if _point(p) != _point(point)]
        # Knowing the tested point itself never counts as nonlocal progress.
        projections = {tuple(s[i] for i in keep) for s in admitted}
        if not keep or not projections:
            continue
        gains = []
        for outcome in (False, True):
            removed = 0
            for signature in sorted(projections):
                hypothetical = [{**point, "activated": outcome}]
                hypothetical += [{**panel[i], "activated": y} for i, y in zip(keep, signature)]
                removed += union(hypothetical) == "UNSAT_VERIFIED"
            gains.append(removed)
        gain = min(gains)
        outcome_statuses = [union([{**point, "activated": value}]) for value in (False, True)]
        opportunity = max(gains) if all(s == "SAT_WITNESS" for s in outcome_statuses) else 0
        research["experiment_values"].append({"experiment_id": experiment["id"],
            "crystal": point["crystal"], "frequency": point["frequency"],
            "guaranteed_nonlocal_eliminations": gain, "by_hypothetical_outcome": gains,
            "outcome_statuses": outcome_statuses,
            "possible_nonlocal_eliminations": opportunity,
            "compared_panel_indices": keep})
        if gain > best_gain:
            winner, best_gain = experiment, gain
        if opportunity > possible_gain:
            conditional_winner, possible_gain = experiment, opportunity

    premise = (_shared_premise_test(features, responses, families, experiments, research["families"])
               if winner is None and stagnation >= 2 else None)
    role = "nonlocal_family_test"
    if premise is not None:
        winner = premise["experiment"]
        role = "shared_premise_test"
        research["premise_test"] = {"expected_activation": True,
            "source_response": premise["source_response"],
            "required_by_families": premise["required_by_families"],
            "frozen_after_response_count": len(responses)}
    if winner is None and conditional_winner is not None and stagnation < 2:
        winner = conditional_winner
        role = "conditional_family_test"
    if winner is not None:
        selection = deepcopy(result.get("selection") or {})
        forecasts = list(winner["predictions"].values())
        mean = sum(forecasts) / len(forecasts)
        diagnosis = "shared_premise_test" if premise is not None else "family_consequence_experiment"
        selection.update(selected_experiment=deepcopy(winner),
                         diagnosis=diagnosis, predicted_separation=mean * (1 - mean),
                         guaranteed_nonlocal_eliminations=best_gain,
                         consequence_selection=role)
        result.update(status=diagnosis, selection=selection,
                      selected_number=winner["number"], selected_setting=winner["prediction"])
        research["selected_role"] = role
    elif stagnation >= 2:
        result["operational_fallback"] = deepcopy(result.get("selection"))
        if conditional_winner is not None:
            result["operational_fallback"] = {"diagnosis": "conditional_family_test",
                "selected_experiment": deepcopy(conditional_winner),
                "guaranteed_nonlocal_eliminations": 0,
                "possible_nonlocal_eliminations": possible_gain}
        result.update(status="family_research_needed", selection=None,
                      selected_number=None, selected_setting=None)
        research["research_obligation"] = {
            "reason": "completed_tests_changed_members_without_task_or_nonlocal_progress",
            "question": "Which shared response assumption or missing measurement prevents a new target prediction?",
            "family_statuses": [f["status"] for f in research["families"]]}
        research["selected_role"] = "shared_premise_research"
    selected = (result.get("selection") or {}).get("selected_experiment")
    if selected is not None:
        research["selected_test"] = {"crystal": selected["number"], "frequency": selected["prediction"]}
    result["family_research"] = research
    result["compute_cost"] += research["compute_cost"]
    return result
