"""Compare executable research plans using their measured later returns.

The caller owns experiments, fitting and common new prediction comparisons.
This book learns the return of the selected continuation, including its whole
observed bill. It does not assign that bill to each constituent or each source
concept that happened to participate in the visit.
"""

from copy import deepcopy
from statistics import mean

from .method_revision import revision_condition


def opportunity_condition(offer):
    """Public evidence that determines whether an old plan faces a new problem.

    Numerical interval widths and fitted constants change during ordinary
    refinement. Neither those changes nor additional rows in the same support
    make an unsuccessful continuation an untried opportunity again.
    """
    profile = offer.get("profile") or {}
    data, representation = profile.get("data", {}), profile.get("representation", {})
    design = offer.get("design") or {}
    branches = design.get("boundary_support") or profile.get("boundary", {}).get("branches", [])
    boundary = []
    for branch in branches:
        condition = {"history_depth": branch.get("history_depth", 0),
                     "previous_state": branch.get("previous_state"),
                     "free_control_count": branch.get("free_control_count", len(branch.get("free_controls", [])))}
        if condition not in boundary:
            boundary.append(condition)
    boundary.sort(key=lambda row: (row["history_depth"], str(row["previous_state"]), row["free_control_count"]))
    needed_history = max([representation.get("history_depth", 0),
                          *(row["history_depth"] for row in boundary)])
    history = data.get("max_history", 0)
    if "continuous" not in representation.get("families", []):
        history = min(history, needed_history)
    return {"kind": offer["kind"], "action": offer["action"],
            "steps": list(offer.get("steps") or [offer["action"]]),
            "incoming_evidence": bool(offer.get("incoming_evidence")),
            "knowledge_alternatives": deepcopy(offer.get("knowledge_alternatives", [])),
            "objective": deepcopy(profile.get("objective")),
            "representation": {key: deepcopy(representation.get(key)) for key in (
                "kind", "families", "history_depth", "structure")},
            "data": {**{key: deepcopy(data.get(key)) for key in (
                "feature_dimensions", "control_dimensions", "coverage", "conditional_coverage",
                "linear_data_support", "object_fields", "response_levels", "independent_trajectories")},
                "available_history": history},
            "residual": {key: deepcopy(profile.get("residual", {}).get(key)) for key in (
                "state", "new_counterexample")},
            "decision_alternatives": deepcopy(design.get("alternatives", [])),
            "decision_followup": bool(design.get("unresolved_comparison")),
            "boundary": boundary, "source_revision": revision_condition(profile)}


def _has_open_question(offer):
    profile, design = offer.get("profile") or {}, offer.get("design") or {}
    residual = profile.get("residual", {})
    if design.get("unresolved_comparison"):
        return True
    if offer.get("incoming_evidence") or residual.get("state") == "unresolved" or residual.get("new_counterexample"):
        return True
    if design.get("candidate_disagreement", 0.) > 0. or design.get("expected_risk_reduction", 0.) > 0.:
        return True
    return bool(profile.get("boundary", {}).get("unresolved_width", 0.) > 0.
                or any(branch.get("relative_width", 0.) > 0. for branch in design.get("boundary_support", [])))


class ResearchOpportunities:
    """Price a complete proposed continuation, then learn its actual outcome.

    Offers are ordinary dictionaries with kind, action, profile,
    observation_cost and compute_cost. Experiment cost includes the follow-up
    comparison; computation includes the executable fitting and comparison.
    Additional routing fields are returned unchanged and are not learned as
    applicability conditions. Existing measured method estimates may supply
    expected_gain and separate expected_knowledge_gain/knowledge_compute_cost.
    A new interval question has no expected prediction gain until measured.
    """

    def __init__(self, state=None):
        self.outcomes = deepcopy((state or {}).get("outcomes", []))
        self._evaluator = None
        if (state or {}).get("evaluator") is not None:
            from .research_evaluator import ResearchEvaluator
            self._evaluator = ResearchEvaluator(state["evaluator"])

    @property
    def evaluator(self):
        if self._evaluator is None:
            from .research_evaluator import ResearchEvaluator
            self._evaluator = ResearchEvaluator()
        return self._evaluator

    def estimate(self, offer, *, evaluator_budget=None):
        condition = opportunity_condition(offer)
        def matches(row):
            previous = {"decision_alternatives": [], "knowledge_alternatives": [],
                        "decision_followup": False, **row["condition"]}
            if previous == condition:
                return True
            # A new delivery containing only structures already available in
            # an unproductive continuation does not renew that question.
            # Positive returns do not transfer to subsets: the missing option
            # may have been the one that made the old continuation useful.
            alternatives = condition["knowledge_alternatives"]
            return (bool(alternatives) and row["gain"] is not None and row["gain"] <= 0.
                    and all(item in previous.get("knowledge_alternatives", []) for item in alternatives)
                    and {k: v for k, v in previous.items() if k != "knowledge_alternatives"}
                    == {k: v for k, v in condition.items() if k != "knowledge_alternatives"})
        matching = [row for row in self.outcomes if matches(row)]
        measured = [row for row in matching if row["gain"] is not None]
        gain = mean(row["gain"] for row in measured) if measured else offer.get("expected_gain")
        baseline_gain = gain
        forecast = (self.evaluator.predict({**offer, "expected_gain": gain}, budget=evaluator_budget)
                    if evaluator_budget is not None else None)
        revised = bool(forecast is not None and forecast.get("expected_gain") is not None)
        if revised:
            gain = forecast["expected_gain"]

        def expected_cost(field):
            quoted = offer[field]
            # The executable quote is a minimum. A source plan that repeatedly
            # cost twice its quote must not win on that stale cheap estimate.
            ratios = [row[field] / row[f"quoted_{field}"] for row in matching if row[f"quoted_{field}"] > 0]
            extra = [row[field] for row in matching if not row[f"quoted_{field}"]]
            return max(quoted, quoted * mean(ratios) if ratios else quoted, mean(extra) if extra else 0.)

        knowledge_gain = offer.get("expected_knowledge_gain", 0.)
        observation_cost = expected_cost("observation_cost")
        compute_cost = expected_cost("compute_cost")
        unknown = gain is None and not measured
        exploration = unknown and _has_open_question(offer) and knowledge_gain <= 0.
        investment_gain = (gain or 0.) + knowledge_gain
        return {**deepcopy(offer), **({"evaluator_forecast": forecast,
                                    "baseline_expected_gain": baseline_gain} if forecast is not None else {}),
                "expected_gain": gain, "expected_knowledge_gain": knowledge_gain,
                "expected_observation_cost": observation_cost, "expected_compute_cost": compute_cost,
                "investment_gain": investment_gain,
                "investment_compute_cost": compute_cost + offer.get("knowledge_compute_cost", 0.),
                "basis": ("revised_research_evaluator" if revised else
                          "measured_continuation_return" if measured else
                          "existing_method_return" if gain is not None or knowledge_gain else
                          "untried_observable_question" if exploration else "no_observed_research_value"),
                "trials": len(measured), "incomplete_trials": len(matching) - len(measured),
                "exploration": exploration}

    def rank(self, offers, *, observation_remaining, compute_remaining,
             visit_observation_budget=None, visit_compute_budget=None, evaluator_budget=None):
        quotes = [self.estimate(offer, evaluator_budget=evaluator_budget) for offer in offers]
        return self.rank_estimates(quotes, observation_remaining=observation_remaining,
            compute_remaining=compute_remaining, visit_observation_budget=visit_observation_budget,
            visit_compute_budget=visit_compute_budget)

    def rank_estimates(self, quotes, *, observation_remaining, compute_remaining,
                       visit_observation_budget=None, visit_compute_budget=None):
        """Rank supported positive returns, then affordable untried questions.

        Exploration gets value zero: it can run after first coverage, but does
        not impersonate a measured gain that outranks an untouched question.
        Once a condition has a measured nonpositive return it needs genuinely
        changed public support, or separate measured knowledge value, to run.
        """
        ranked = []
        observation_limit = min(observation_remaining, visit_observation_budget) if visit_observation_budget is not None else observation_remaining
        compute_limit = min(compute_remaining, visit_compute_budget) if visit_compute_budget is not None else compute_remaining
        for estimated in quotes:
            quote = deepcopy(estimated)
            if quote["expected_observation_cost"] > observation_limit or quote["expected_compute_cost"] > compute_limit:
                continue
            if quote["investment_gain"] <= 0. and not quote["exploration"]:
                continue
            share = max(quote["expected_observation_cost"] / max(1, observation_remaining),
                        quote["investment_compute_cost"] / max(1, compute_remaining), 1e-8)
            quote.update(dominant_resource_share=share,
                         value=quote["investment_gain"] / share if not quote["exploration"] else 0.)
            ranked.append(quote)
        return sorted(ranked, key=lambda quote: (
            quote["exploration"], -quote["value"], quote["dominant_resource_share"]))

    def observe(self, selected_offer, feedback, *, observation_cost, compute_cost, evaluator_budget=None):
        """Record actual cost and only the completed common-evidence return.

        Normalized gains come from the caller's common new rows and baseline.
        Bound their contribution to allocation when the initial loss is zero
        or tiny, while retaining the reported ratio and absolute improvement.
        A narrower interval is useful experimental progress, but is never
        substituted for a prediction gain. Unfinished work has unknown gain.
        """
        measured = bool(feedback.get("execution_complete") and feedback.get("reference_rows")
                        and feedback.get("normalized_gain") is not None)
        record = {"condition": opportunity_condition(selected_offer),
                  "gain": max(-1., min(1., feedback["normalized_gain"])) if measured else None,
                  "raw_gain": feedback.get("gain") if measured else None,
                  "reported_normalized_gain": feedback.get("normalized_gain"),
                  "execution_complete": bool(feedback.get("execution_complete")),
                  "reference_rows": feedback.get("reference_rows", 0),
                  "observation_cost": observation_cost, "compute_cost": compute_cost,
                  "quoted_observation_cost": selected_offer["observation_cost"],
                  "quoted_compute_cost": selected_offer["compute_cost"]}
        for field in ("interval_reduction", "boundary_width_before", "boundary_width_after"):
            if field in feedback:
                record[field] = feedback[field]
        self.outcomes.append(record)
        forecast = selected_offer.get("evaluator_forecast")
        if forecast is not None and evaluator_budget is not None:
            start = evaluator_budget.spent
            update = self.evaluator.observe(forecast.get("forecast"), feedback, budget=evaluator_budget)
            record["evaluator"] = {"forecast": deepcopy(forecast), "update": update,
                                   "compute_cost": evaluator_budget.spent - start}
        return deepcopy(record)

    def to_dict(self):
        return {"outcomes": deepcopy(self.outcomes),
                **({"evaluator": self._evaluator.to_dict()} if self._evaluator is not None else {})}
