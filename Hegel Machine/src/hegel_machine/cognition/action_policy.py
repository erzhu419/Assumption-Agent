"""Learn which research actions repay their actual observation and search costs.

An action's feedback arrives after the subsequent revision is compared on new
representative evidence. Fitting its own targeted observations is not feedback.
The caller supplies that measured loss in the externally specified objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ActionOption:
    """An available action and the resources needed to begin executing it."""

    name: str
    observation_cost: int
    compute_cost: int
    payload: Any = None

    def __post_init__(self):
        if self.observation_cost < 0 or self.compute_cost < 0:
            raise ValueError("research action costs must be nonnegative")


class ResearchActionPolicy:
    """Persist returns for public task conditions, without task or feature IDs.

    Unknown actions receive one affordable trial. A neutral first result can
    receive one further trial when no action has a measured positive return.
    Negative results do not trigger automatic retries. New or successful
    actions can therefore displace an unhelpful experiment within a few rounds.

    Among actions with positive returns, compare improvement per dominant
    remaining-resource share. Observations and compute each have their own
    affordability constraint; no exchange rate converts one into the other.
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
        return self.context

    @staticmethod
    def _resource_share(option, observation_remaining, compute_remaining, record=None):
        # Quoted costs determine feasibility. Real costs of earlier completed
        # actions also inform value: cheap acquisition may induce costly search.
        observations = max(option.observation_cost,
                           (record or {}).get("mean_observation_cost", 0.0))
        compute = max(option.compute_cost,
                      (record or {}).get("mean_compute_cost", 0.0))
        return max(observations / max(1, observation_remaining),
                   compute / max(1, compute_remaining), 1e-12)

    def choose(self, options, *, observation_remaining, compute_remaining, minimum_improvement=.002):
        """Return an affordable ActionOption, or None when more work has no value.

        Options may contain transition, task_value, representative, refit,
        calibration, or other actions the caller can actually execute. The
        policy does not create actions or reserve resources for an absent one.
        """
        eligible = [option for option in options
                    if option.observation_cost <= observation_remaining
                    and option.compute_cost <= compute_remaining]
        self.last_decision = {
            "context": self.context, "selected_action": None,
            "reason": "no_affordable_action", "eligible_actions": [option.name for option in eligible],
            "observation_remaining": observation_remaining, "compute_remaining": compute_remaining,
        }
        if not eligible:
            return None

        def record(option):
            return self.experience.get(option.name, {})

        def share(option):
            return self._resource_share(option, observation_remaining, compute_remaining, record(option))

        # Opposite responses already bracket an unresolved boundary. A fresh
        # intervention inside it answers a concrete remaining question even
        # when every retained program fits the currently observed rows. Past
        # average returns from unrelated coverage cannot settle that question.
        def boundary_width(option):
            details = option.payload[1] if option.payload else {}
            return max((item["relative_width"] for item in details.get("boundary_support", ())), default=0.)

        boundaries = [option for option in eligible if boundary_width(option) > 2 * minimum_improvement]
        unexplored = [option for option in eligible if not record(option).get("trials", 0)]
        positive = [option for option in eligible if record(option).get("expected_gain", 0.0) > 0]
        neutral = [option for option in eligible if record(option).get("trials") == 1
                   and record(option).get("expected_gain") == 0]
        if boundaries:
            selected = max(boundaries, key=lambda option: boundary_width(option) / share(option))
            reason = "resolve_observed_boundary"
        elif unexplored:
            selected = min(unexplored, key=share)
            reason = "untried_action"
        elif positive:
            selected = max(positive, key=lambda option: record(option)["expected_gain"] / share(option))
            reason = "measured_improvement_per_resource"
        elif neutral:
            selected = min(neutral, key=share)
            reason = "resolve_neutral_first_trial"
        else:
            self.last_decision["reason"] = "no_measured_positive_return"
            return None

        selected_record = record(selected)
        self.last_decision.update(
            selected_action=selected.name, reason=reason,
            trials=selected_record.get("trials", 0),
            expected_gain=selected_record.get("expected_gain"),
            dominant_resource_share=share(selected),
            quoted_observation_cost=selected.observation_cost,
            quoted_compute_cost=selected.compute_cost,
            unresolved_boundary_width=boundary_width(selected),
        )
        return selected

    def update(self, action_name, *, before_loss, after_loss, observation_cost,
               compute_cost, baseline_loss=None, context=None):
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
        if observation_cost < 0 or compute_cost < 0:
            raise ValueError("research feedback costs must be nonnegative")
        normalizer = max(before_loss if baseline_loss is None else baseline_loss, 1e-8)
        gain = max(-1.0, min(1.0, (before_loss - after_loss) / normalizer))
        destination = self._store.setdefault(context if context is not None else self.context, {})
        previous = destination.get(action_name, {})
        count = previous.get("trials", 0) + 1
        result = {
            "trials": count,
            "expected_gain": (gain if count == 1 else 0.5 * previous["expected_gain"] + 0.5 * gain),
            "mean_gain": previous.get("mean_gain", 0.0) + (gain - previous.get("mean_gain", 0.0)) / count,
            "mean_observation_cost": previous.get("mean_observation_cost", 0.0)
            + (observation_cost - previous.get("mean_observation_cost", 0.0)) / count,
            "mean_compute_cost": previous.get("mean_compute_cost", 0.0)
            + (compute_cost - previous.get("mean_compute_cost", 0.0)) / count,
            "last_before_loss": before_loss, "last_after_loss": after_loss,
            "last_normalizer": normalizer, "last_gain": gain,
        }
        destination[action_name] = result
        return dict(result)
