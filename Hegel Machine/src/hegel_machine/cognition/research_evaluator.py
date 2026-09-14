"""Revise executable predictions of research return from subsequent outcomes.

The target is a completed continuation's measured normalized improvement.
It is a proxy for choosing research, never a replacement for the task's loss.
Candidate functions fit paid past returns and compete only through forecasts
made before later returns arrive. The caller retains a selected forecast in
its ordinary checkpoint; unselected proposals create no evaluator state.
"""

from copy import deepcopy
from math import log1p, sqrt
from statistics import mean, stdev

from .language import Program
from .models import Observation
from .synthesis import synthesize


def research_features(offer):
    """Use public evidence and the quoted work, with no task or seed identity."""
    profile = offer.get("profile") or {}
    data = profile.get("data") or {}
    residual = profile.get("residual") or {}
    representation = profile.get("representation") or {}
    objective = profile.get("objective") or {}
    coverage = data.get("coverage") or []
    action = offer.get("action", "")
    return {
        "residual": min(4., max(0., residual.get("normalized") or 0.)),
        "counterexample": float(bool(residual.get("new_counterexample"))),
        "new_evidence": float(bool(offer.get("incoming_evidence"))),
        "coverage": mean(len(item.get("bins", [])) / 4 for item in coverage) if coverage else 0.,
        "boundary_width": (profile.get("boundary") or {}).get("unresolved_width", 0.) or 0.,
        "history_gap": max(0., representation.get("history_depth", 0) - data.get("max_history", 0)),
        "log_observations": log1p(profile.get("observation_count", 0)),
        "log_quoted_compute": log1p(max(0., offer.get("compute_cost", 0))),
        "log_quoted_observations": log1p(max(0., offer.get("observation_cost", 0))),
        "quantile": objective.get("quantile", .5),
        "independent_trajectories": float(bool(data.get("independent_trajectories"))),
        "prepare": float(action in {"transition", "representative", "task_value", "experiment"}),
        "refit": float(action == "refit"),
        "change_representation": float(action in {"representation", "continuous_state", "state_transition"}),
    }


def _summary(trials, minimum):
    gains = [trial["gain"] for trial in trials]
    average = mean(gains) if gains else None
    margin = max(minimum, 2 * stdev(gains) / sqrt(len(gains))) if len(gains) >= 2 else None
    return {"trials": len(gains), "gain": average, "support_margin": margin,
            "supported": len(gains) >= 4 and average > margin,
            "refuted": len(gains) >= 4 and average < -margin}


class ResearchEvaluator:
    """A small revisable program model of actual research improvement.

    A proposed program is frozen for its prospective comparison. A supported
    replacement begins a new comparison generation; outstanding forecasts from
    older generations remain training evidence, not trials of the new model.
    Four later completed returns can support or refute a comparison. A current
    model that loses to the caller's frozen old estimate is withdrawn to unknown.
    An unavailable old estimate uses zero only as the internal comparison rule.
    """

    def __init__(self, state=None, *, minimum_improvement=.002):
        state = deepcopy(state or {})
        self.minimum_improvement = state.get("minimum_improvement", minimum_improvement)
        self.records = state.get("records", [])
        self.deferred = state.get("deferred", [])
        self.current = state.get("current")
        self.candidates = state.get("candidates", [])
        self.current_trials = state.get("current_trials", [])
        self.events = state.get("events", [])
        self.generation = state.get("generation", 0)
        self.next_model = state.get("next_model", 0)
        self.last_fit_count = state.get("last_fit_count", 0)

    def predict(self, offer, *, budget):
        """Return a paid, self-contained forecast without changing model state."""
        started = budget.spent
        models = ([self.current] if self.current else []) + self.candidates
        # Even an unknown forecast pays for materializing its evidence input.
        if not budget.spend(max(1, len(models))):
            return {"status": "insufficient_budget", "expected_gain": None,
                    "compute_cost": 0, "forecast": None}
        features = research_features(offer)
        row = Observation(features, 0.)
        predictions = [{"model_id": model["model_id"],
                        "value": max(-1., min(1., Program.from_dict(model["program"]).predict(row)))}
                       for model in models]
        current = predictions[0]["value"] if self.current else None
        forecast = {"features": features, "generation": self.generation,
                    "current_id": self.current["model_id"] if self.current else None,
                    "current_prediction": current, "predictions": predictions,
                    "baseline_expected_gain": offer.get("expected_gain")}
        return {"status": "predicted" if self.current else "unvalidated_model",
                "expected_gain": current, "model_id": forecast["current_id"],
                "program": deepcopy(self.current["program"]) if self.current else None,
                "forecast": forecast, "compute_cost": budget.spent-started}

    def _withdraw(self, reason, comparison):
        identifier = self.current["model_id"] if self.current else None
        self.events.append({"event": "withdraw_research_evaluator", "model_id": identifier,
                            "reason": reason, "comparison": comparison})
        self.current, self.candidates, self.current_trials = None, [], []
        self.generation += 1

    def _fit(self, budget):
        # A failed forecast asks for a different function; repeated fitting of
        # the same small evidence batch cannot supply prospective support.
        if self.candidates or len(self.records) < 6 or len(self.records)-self.last_fit_count < 4:
            return
        recent = self.records[-4:]
        if self.current and not any(record["prediction_error"] is not None
                                    and record["prediction_error"] > self.minimum_improvement for record in recent):
            return
        rows = [Observation(record["features"], record["gain"]) for record in self.records]
        if budget.remaining < 4 * len(rows):
            return
        # Constant columns carry no evidence for a coefficient. Restrict the
        # search input to measured variation rather than fitting singular terms.
        varying = [key for key in rows[0].features if len({row.features[key] for row in rows}) > 1]
        rows = [Observation({key: row.features[key] for key in varying}, row.target) for row in rows]
        before = budget.spent
        generated = synthesize(rows, operations=("refit", "relation", "composition"),
            budget=budget, max_history=0)
        existing = Program.from_dict(self.current["program"]) if self.current else Program("const", value=0.)
        seen = {existing.simplified()}
        for candidate in generated:
            program = candidate.program.simplified()
            if program in seen:
                continue
            seen.add(program)
            self.next_model += 1
            self.candidates.append({"model_id": self.next_model, "program": program.to_dict(),
                "operation": candidate.operation, "fitted_rows": len(rows), "trials": []})
            if len(self.candidates) == 2:
                break
        self.last_fit_count = len(self.records)
        self.events.append({"event": "propose_research_evaluators", "training_rows": len(rows),
            "features": varying, "compute_cost": budget.spent-before,
            "models": [{key: deepcopy(model[key]) for key in ("model_id", "program", "operation")}
                       for model in self.candidates], "reason": "unmodeled_or_mispredicted_research_return"})

    def _consume(self, forecast, feedback, *, budget):
        started = budget.spent
        predictions = {item["model_id"]: item["value"] for item in forecast["predictions"]}
        # One target update plus one scored squared error per forecast program.
        if not budget.spend(1 + len(predictions)):
            return {"status": "insufficient_budget", "compute_cost": 0}
        target = max(-1., min(1., feedback["normalized_gain"]))
        current_prediction = forecast["current_prediction"]
        loss = (target-current_prediction) ** 2 if current_prediction is not None else None
        record = {"features": deepcopy(forecast["features"]), "gain": target,
            "reported_normalized_gain": feedback["normalized_gain"], "prediction_error": loss,
            "forecast_generation": forecast["generation"], "current_id": forecast["current_id"]}
        # All squared errors use the forecasted values, never a refitted model
        # evaluated after this return was revealed.
        comparable = forecast["generation"] == self.generation
        old_prediction = forecast.get("baseline_expected_gain")
        old_prediction = max(-1., min(1., old_prediction)) if old_prediction is not None else 0.
        baseline = current_prediction if current_prediction is not None else old_prediction
        baseline_loss = (target-baseline) ** 2
        if comparable:
            for model in self.candidates:
                if model["model_id"] in predictions:
                    candidate_loss = (target-predictions[model["model_id"]]) ** 2
                    model["trials"].append({"before_loss": baseline_loss, "after_loss": candidate_loss,
                                            "gain": baseline_loss-candidate_loss})
            if self.current and forecast["current_id"] == self.current["model_id"]:
                old_loss = (target-old_prediction) ** 2
                self.current_trials.append({"before_loss": old_loss, "after_loss": loss,
                                            "gain": old_loss-loss})
                self.current_trials = self.current_trials[-4:]
            supported = [(mean(trial["after_loss"] for trial in model["trials"]), model)
                         for model in self.candidates
                         if _summary(model["trials"], self.minimum_improvement)["supported"]]
            if supported:
                chosen = min(supported, key=lambda pair: (pair[0], Program.from_dict(pair[1]["program"]).size))[1]
                previous = self.current["model_id"] if self.current else None
                self.current = {key: deepcopy(value) for key, value in chosen.items() if key != "trials"}
                self.events.append({"event": "adopt_research_evaluator", "model_id": chosen["model_id"],
                    "previous_model_id": previous,
                    "comparison": _summary(chosen["trials"], self.minimum_improvement)})
                self.candidates, self.current_trials = [], []
                self.generation += 1
            elif self.current and _summary(self.current_trials, self.minimum_improvement)["refuted"]:
                self._withdraw("subsequent_returns_refute_prediction", _summary(self.current_trials, self.minimum_improvement))
            else:
                retained = []
                for model in self.candidates:
                    comparison = _summary(model["trials"], self.minimum_improvement)
                    no_difference = len(model["trials"]) >= 4 and all(abs(trial["gain"]) <= 1e-12 for trial in model["trials"])
                    if comparison["refuted"] or no_difference:
                        self.events.append({"event": "reject_research_evaluator", "model_id": model["model_id"],
                                            "comparison": comparison})
                    else:
                        retained.append(model)
                self.candidates = retained
        self.records.append(record)
        return {"status": "measured", "gain": target, "prediction_error": loss,
                "model_id": self.current["model_id"] if self.current else None,
                "pending_comparisons": len(self.candidates), "compute_cost": budget.spent-started}

    def observe(self, forecast, feedback, *, budget):
        """Retain an actual result, then pay for comparison and model revision.

        ``forecast`` is the nested object returned by ``predict``. When the
        caller cannot fund scoring, its already acquired label remains deferred;
        it becomes neither a scored trial nor training data until a later flush.
        Unfinished research supplies no completed-return label.
        """
        if (forecast is None or not feedback.get("execution_complete")
                or not feedback.get("reference_rows") or feedback.get("normalized_gain") is None):
            return {"status": "unmeasured_return", "compute_cost": 0}
        self.deferred.append({"forecast": deepcopy(forecast), "feedback": {
            key: feedback[key] for key in ("execution_complete", "reference_rows", "normalized_gain")}})
        return self.flush(budget=budget)

    def flush(self, *, budget):
        """Consume saved real labels in arrival order without buying new data."""
        started = budget.spent
        result = {"status": "no_new_return"}
        while self.deferred:
            pending = self.deferred[0]
            update = self._consume(pending["forecast"], pending["feedback"], budget=budget)
            if update["status"] == "insufficient_budget":
                result = update
                break
            self.deferred.pop(0)
            result = update
        self._fit(budget)
        return {**result, "model_id": self.current["model_id"] if self.current else None,
                "pending_comparisons": len(self.candidates), "deferred_returns": len(self.deferred),
                "compute_cost": budget.spent-started}

    def to_dict(self):
        return deepcopy({"minimum_improvement": self.minimum_improvement, "records": self.records,
            "deferred": self.deferred,
            "current": self.current, "candidates": self.candidates, "current_trials": self.current_trials,
            "events": self.events, "generation": self.generation, "next_model": self.next_model,
            "last_fit_count": self.last_fit_count})
