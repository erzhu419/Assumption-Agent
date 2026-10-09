"""Learn which research actions repay their actual observation and search costs.

An action's feedback arrives after the subsequent revision is compared on new
representative evidence. Fitting its own targeted observations is not feedback.
The caller supplies that measured loss in the externally specified objective.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ActionOption:
    """An available action and the resources needed to begin executing it."""

    name: str
    observation_cost: int
    compute_cost: int
    payload: Any = None
    measurement_cost: int = 0

    def __post_init__(self):
        if min(self.observation_cost, self.compute_cost, self.measurement_cost) < 0:
            raise ValueError("research action costs must be nonnegative")


class ResearchActionPolicy:
    """Persist returns for public task conditions, without task or feature IDs.

    A confirmed direction precedes unknown actions unless a cheaper, bounded
    trial leaves its entire measured follow-up affordable as a fallback.
    Otherwise unknown actions receive one affordable trial.
    A neutral first result can receive one further trial when no action has a
    measured positive return.
    Negative results do not trigger automatic retries. New or successful
    actions can therefore displace an unhelpful experiment within a few rounds.

    Among actions with positive returns, compare improvement per dominant
    remaining-resource share. Observations, computation and (when supplied)
    instrument readings retain their own budgets without a conversion rate.
    """

    def __init__(self, observations, spec, experience):
        if spec.decision_costs is not None:
            under, over = spec.decision_costs
            objective = f"decision:{under / (under + over):.8g}"
        elif spec.risk_quantile != 0.5:
            objective = f"pinball:{spec.risk_quantile:.8g}"
        else:
            response = "binary" if len({row.target for row in observations}) == 2 else "numeric"
            objective = f"squared:{response}"
        controls = "single" if len(spec.feature_names) == 1 else "multiple"
        self.base_context = f"{objective}:{controls}"
        self._store = experience
        self.context = self.base_context
        self.experience = self._store.setdefault(self.context, {})
        self._action_contexts = {}
        self.last_decision = {}

    def set_condition(self, condition):
        """Select experience for an observed research condition within this task.

        Conditions describe learning evidence, such as a newly falsified
        transfer, rather than the environment's private mechanism. The task's
        public objective remains in base_context. Pending actions retain their
        original context for feedback after the condition changes.
        """
        self.context = f"{self.base_context}:{condition}" if condition else self.base_context
        self.experience = self._store.setdefault(self.context, {})
        self._action_contexts = {}
        return self.context

    def set_action_knowledge(self, action, description):
        """Price this action under its currently executable learned programs.

        The caller supplies only knowledge this action can actually execute.
        Keep each previously observed applicability and its paid outcomes;
        other actions continue to use their ordinary task-condition record.
        """
        if not description:
            self._action_contexts.pop(action, None)
            return self.context
        prefix = f"{self.context}:{action}:knowledge:"
        applicable = [(context, actions[action]) for context, actions in self._store.items()
                      if context.startswith(prefix) and action in actions]
        context = next((context for context, record in applicable
                        if record["knowledge_applicability"] == description), None)
        if context is None:
            context = f"{prefix}{len(applicable) + 1}"
            self._store[context] = {action: {"trials": 0,
                "knowledge_applicability": deepcopy(description)}}
        self._action_contexts[action] = context
        return context

    def action_context(self, action):
        return self._action_contexts.get(action, self.context)

    def set_action_counterexample(self, action, description, *, fit_loss,
                                  counterexample_loss, tolerance):
        """Reconsider a direction when new readings refute a fitted response.

        The caller identifies an actual unfitted measurement and its frozen
        response. Reusing that evidence keeps the same applicability record,
        including any failed retry; earlier action outcomes remain available.
        """
        if description and fit_loss <= tolerance < counterexample_loss:
            return self.set_action_knowledge(action, {"measured_counterexample": description})
        return self.action_context(action)

    def action_record(self, action):
        return self._store[self.action_context(action)].get(action, {})

    @staticmethod
    def _estimated_costs(record):
        trials = record.get("trials", 0)
        attempts = trials + record.get("unscored_attempts", 0)
        return {f"{resource}_cost": (record.get(f"mean_{resource}_cost", 0.) * trials
                + record.get(f"unscored_{resource}_cost", 0.)) / max(1, attempts)
                for resource in ("observation", "measurement", "compute")}

    @staticmethod
    def _resource_share(option, observation_remaining, compute_remaining, record=None,
                        measurement_remaining=None):
        # Quoted costs determine feasibility. Real costs of earlier completed
        # actions also inform value: cheap acquisition may induce costly search.
        costs = ResearchActionPolicy._estimated_costs(record or {})
        observations = max(option.observation_cost, costs["observation_cost"])
        compute = max(option.compute_cost, costs["compute_cost"])
        shares = [observations / max(1, observation_remaining),
                  compute / max(1, compute_remaining), 1e-12]
        if measurement_remaining is not None:
            measurements = max(option.measurement_cost, costs["measurement_cost"])
            shares.append(measurements / max(1, measurement_remaining))
        return max(shares)

    def choose(self, options, *, observation_remaining, compute_remaining,
               measurement_remaining=None, minimum_improvement=.002):
        """Return an affordable ActionOption, or None when more work has no value.

        Options may contain transition, task_value, representative, refit,
        calibration, or other actions the caller can actually execute. The
        policy does not create actions or reserve resources for an absent one.
        """
        eligible = [option for option in options
                    if option.observation_cost <= observation_remaining
                    and option.compute_cost <= compute_remaining
                    and (measurement_remaining is None or option.measurement_cost <= measurement_remaining)]
        self.last_decision = {
            "context": self.context, "selected_action": None,
            "reason": "no_affordable_action", "eligible_actions": [option.name for option in eligible],
            "observation_remaining": observation_remaining, "compute_remaining": compute_remaining,
            "measurement_remaining": measurement_remaining,
        }
        if not eligible:
            return None

        def record(option):
            return self.action_record(option.name)

        def share(option):
            return self._resource_share(option, observation_remaining, compute_remaining,
                                        record(option), measurement_remaining)

        def attempts(option):
            return record(option).get("trials", 0) + record(option).get("unscored_attempts", 0)

        def adjusted_gain(option):
            measured = record(option).get("trials", 0)
            failed = record(option).get("no_candidate_attempts", 0)
            # No candidate is an observed failure to produce a new response.
            # An unconfirmed candidate remains unknown, not a measured failure.
            yield_rate = measured / (measured + failed) if measured + failed else 1.
            return record(option).get("expected_gain", 0.) * yield_rate

        # Opposite responses already bracket an unresolved boundary. A fresh
        # intervention inside it answers a concrete remaining question even
        # when every retained program fits the currently observed rows. Past
        # average returns from unrelated coverage cannot settle that question.
        def boundary_width(option):
            details = option.payload[1] if option.payload else {}
            return max((item["relative_width"] for item in details.get("boundary_support", ())), default=0.)

        boundaries = [option for option in eligible if boundary_width(option) > 2 * minimum_improvement]
        unexplored = [option for option in eligible if not attempts(option)]
        positive = [option for option in eligible if record(option).get("expected_gain", 0.0) > 0]
        def last_outcome(option):
            if record(option).get("last_unscored_trial", -1) >= record(option).get("trials", 0):
                return {}
            outcomes = record(option).get("trial_outcomes", {})
            # Earlier support cannot confirm a newer, still provisional trial.
            return outcomes.get(str(record(option).get("trials", 0)), {})

        confirmed = [option for option in positive
                     if last_outcome(option).get("status") == "supported"
                     and last_outcome(option).get("gain", 0.) > 0]
        neutral = [option for option in eligible if attempts(option) == 1
                   and record(option).get("expected_gain") == 0]
        bounded_share = None
        if boundaries:
            selected = max(boundaries, key=lambda option: boundary_width(option) / share(option))
            reason = "resolve_observed_boundary"
        elif confirmed and unexplored:
            selected = max(confirmed, key=lambda option: adjusted_gain(option) / share(option))
            reason = "confirmed_direction_before_exploration"
            measured_costs = self._estimated_costs(record(selected))
            fallback_costs = {f"{resource}_cost": max(getattr(selected, f"{resource}_cost"),
                measured_costs[f"{resource}_cost"])
                for resource in ("observation", "measurement", "compute")}
            remaining = {"observation_cost": observation_remaining, "compute_cost": compute_remaining,
                         "measurement_cost": measurement_remaining}
            bounded_trials = []
            for option in unexplored:
                details = option.payload[1] if option.payload else {}
                if "compute_limit" not in details:
                    continue
                bounded = replace(option, compute_cost=max(option.compute_cost, details["compute_limit"]))
                # An unknown direction cannot justify spending more of one
                # resource by merely omitting the known confirmation bill in
                # another. It must actually be no dearer in every resource.
                if (any(getattr(bounded, resource) > cost for resource, cost in fallback_costs.items())
                        or not any(getattr(bounded, resource) < cost
                                   for resource, cost in fallback_costs.items())):
                    continue
                if share(bounded) >= share(selected):
                    continue
                if any(limit is not None and getattr(bounded, resource) + fallback_costs[resource] > limit
                       for resource, limit in remaining.items()):
                    continue
                bounded_trials.append((share(bounded), option))
            if bounded_trials:
                fallback = selected
                bounded_share, selected = min(bounded_trials, key=lambda item: item[0])
                reason = "bounded_trial_before_confirmed_direction"
                self.last_decision.update(fallback_action=fallback.name,
                    fallback_context=self.action_context(fallback.name), fallback_costs=fallback_costs,
                    bounded_compute_limit=selected.payload[1]["compute_limit"])
        elif unexplored:
            selected = min(unexplored, key=share)
            reason = "untried_action"
        elif positive:
            selected = max(positive, key=lambda option: adjusted_gain(option) / share(option))
            reason = "measured_improvement_per_resource"
        elif neutral:
            selected = min(neutral, key=share)
            reason = "resolve_neutral_first_trial"
        else:
            self.last_decision["reason"] = "no_measured_positive_return"
            return None

        selected_record = record(selected)
        self.last_decision.update(
            context=self.action_context(selected.name), selected_action=selected.name, reason=reason,
            trials=selected_record.get("trials", 0),
            expected_gain=selected_record.get("expected_gain"),
            adjusted_gain=adjusted_gain(selected), attempt_count=attempts(selected),
            estimated_costs=self._estimated_costs(selected_record),
            dominant_resource_share=bounded_share if bounded_share is not None else share(selected),
            quoted_observation_cost=selected.observation_cost,
            quoted_compute_cost=selected.compute_cost,
            quoted_measurement_cost=selected.measurement_cost,
            unresolved_boundary_width=boundary_width(selected),
        )
        return selected

    def update(self, action_name, *, before_loss, after_loss, observation_cost,
               compute_cost, baseline_loss=None, context=None, measurement_cost=0):
        """Credit a completed action from a subsequent common-evidence comparison.

        Costs include acquisition and the fitting it caused, as supplied by the
        controller. baseline_loss can retain a task's initial predictor loss as
        a stable normalizer across successive revisions. Returns are bounded so
        one large-output task cannot dominate later tasks with the same shape.
        context credits the condition in which a pending action was selected,
        even if its feedback arrives after a different condition is observed.
        """
        values = [before_loss, after_loss, *([] if baseline_loss is None else [baseline_loss])]
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("research feedback losses must be finite and nonnegative")
        if min(observation_cost, compute_cost, measurement_cost) < 0:
            raise ValueError("research feedback costs must be nonnegative")
        normalizer = max(before_loss if baseline_loss is None else baseline_loss, 1e-8)
        gain = max(-1.0, min(1.0, (before_loss - after_loss) / normalizer))
        destination = self._store.setdefault(context if context is not None else self.action_context(action_name), {})
        previous = destination.get(action_name, {})
        count = previous.get("trials", 0) + 1
        result = {**previous,
            "trials": count,
            "expected_gain": (gain if count == 1 else 0.5 * previous["expected_gain"] + 0.5 * gain),
            "mean_gain": previous.get("mean_gain", 0.0) + (gain - previous.get("mean_gain", 0.0)) / count,
            "mean_observation_cost": previous.get("mean_observation_cost", 0.0)
            + (observation_cost - previous.get("mean_observation_cost", 0.0)) / count,
            "mean_compute_cost": previous.get("mean_compute_cost", 0.0)
            + (compute_cost - previous.get("mean_compute_cost", 0.0)) / count,
            "mean_measurement_cost": previous.get("mean_measurement_cost", 0.0)
            + (measurement_cost - previous.get("mean_measurement_cost", 0.0)) / count,
            "last_before_loss": before_loss, "last_after_loss": after_loss,
            "last_normalizer": normalizer, "last_gain": gain,
        }
        if "trial_outcomes" in previous:
            result["trial_outcomes"] = previous["trial_outcomes"]
        destination[action_name] = result
        return deepcopy(result)

    def record_unscored_attempt(self, action_name, *, status, observation_cost,
                                compute_cost, measurement_cost=0, context=None):
        """Retain a terminal execution's costs without inventing response evidence."""
        if status not in {"no_candidate", "unresolved"}:
            raise ValueError("unscored research must be no_candidate or unresolved")
        if min(observation_cost, measurement_cost, compute_cost) < 0:
            raise ValueError("research execution costs must be nonnegative")
        destination = self._store.setdefault(context if context is not None else self.action_context(action_name), {})
        record = destination.setdefault(action_name, {"trials": 0})
        record["unscored_attempts"] = record.get("unscored_attempts", 0) + 1
        record[f"{status}_attempts"] = record.get(f"{status}_attempts", 0) + 1
        record["last_unscored_trial"] = record["trials"]
        for resource, cost in (("observation", observation_cost), ("measurement", measurement_cost),
                               ("compute", compute_cost)):
            key = f"unscored_{resource}_cost"
            record[key] = record.get(key, 0.) + cost
        return deepcopy(record)

    @staticmethod
    def revise_feedback(experience, origin, *, status, before_loss=None, after_loss=None,
                        compute_cost=0, evidence_rows=0, baseline_loss=None):
        """Replace one trial's provisional return with later common-evidence feedback.

        The trial may predate newer trials in the same public context. Correct
        its original contribution to the mean and exponential average, rather
        than counting the confirmation as another successful research action.
        Stored outcomes also make a copied origin safe to revise again after
        another frozen confirmation. A status-only counterexample removes its
        confirmation support without inventing representative loss values.
        """
        if status not in {"supported", "refuted", "unresolved"}:
            raise ValueError("unknown research outcome status")
        if (before_loss is None) != (after_loss is None):
            raise ValueError("research return revision needs paired losses")
        values = [value for value in (before_loss, after_loss, baseline_loss) if value is not None]
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("research feedback losses must be finite and nonnegative")
        if compute_cost < 0 or evidence_rows < 0:
            raise ValueError("research confirmation costs must be nonnegative")
        record = experience[origin["context"]][origin["action"]]
        count, trial = record["trials"], origin["trial"]
        if not 1 <= trial <= count:
            raise ValueError("research outcome must refer to an executed trial")
        outcomes = record.setdefault("trial_outcomes", {})
        previous = outcomes.get(str(trial), {"gain": origin["credited_gain"], "status": "provisional"})
        gain = previous["gain"]
        outcome = {**previous, "status": status}
        if before_loss is not None:
            normalizer = max(before_loss if baseline_loss is None else baseline_loss, 1e-8)
            gain = max(-1., min(1., (before_loss - after_loss) / normalizer))
            delta = gain - previous["gain"]
            record["mean_gain"] += delta / count
            weight = (1. if trial == 1 else .5) * .5 ** (count - trial)
            record["expected_gain"] += weight * delta
            outcome.update(gain=gain, before_loss=before_loss, after_loss=after_loss,
                           normalizer=normalizer, evidence_rows=evidence_rows)
            if trial == count:
                record.update(last_before_loss=before_loss, last_after_loss=after_loss,
                              last_normalizer=normalizer, last_gain=gain)
        record["mean_compute_cost"] += compute_cost / count
        outcomes[str(trial)] = outcome
        return deepcopy(record)

    @staticmethod
    def charge_followup(experience, origin, *, observation_cost, measurement_cost, compute_cost):
        """Charge work caused by an existing direction trial without inventing a new trial.

        A selected explanation can cause further experiments after its initial
        independent comparison. These costs belong to that originating trial;
        its measured gain is unchanged until new common evidence is available.
        """
        if min(observation_cost, measurement_cost, compute_cost) < 0:
            raise ValueError("research followup costs must be nonnegative")
        record = experience[origin["context"]][origin["action"]]
        for resource, added in (("observation", observation_cost),
                                ("measurement", measurement_cost), ("compute", compute_cost)):
            key = f"mean_{resource}_cost"
            record[key] = record.get(key, 0.0) + added / record["trials"]
        return deepcopy(record)
