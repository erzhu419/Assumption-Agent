"""Learn an executable observation channel from failed experiment forecasts.

The channel maps a proposed response and public controls to a corrected mean
and conditional simulation error. It changes simulated experiment outcomes;
it does not change the task objective or treat simulated data as observations.
"""

from copy import deepcopy
from math import log
from statistics import mean

from .language import Program
from .models import Observation, SearchBudget
from .research_evaluator import _summary
from .synthesis import synthesize


def _features(probe, baseline):
    return [{**point, "__simulated_response__": value}
            for point, value in zip(probe.sequence, baseline)]


def _distribution(model, features, baseline):
    correction = Program.from_dict(model["correction"])
    variance = Program.from_dict(model["variance"])
    rows = [Observation(point, 0.) for point in features]
    return {"mean": [value + correction.predict(row) for value, row in zip(baseline, rows)],
            "variance": [max(1e-8, variance.predict(row)) for row in rows]}


def _score(distribution, targets):
    return mean(log(variance) + (target-center) ** 2 / variance
                for target, center, variance in zip(targets, distribution["mean"], distribution["variance"]))


class ExperimentSimulator:
    def __init__(self, state=None, *, minimum_improvement=.002):
        state = deepcopy(state or {})
        self.minimum_improvement = state.get("minimum_improvement", minimum_improvement)
        self.rows = state.get("rows", [])
        self.batches = state.get("batches", 0)
        self.last_fit_batch = state.get("last_fit_batch", 0)
        self.current = state.get("current")
        self.candidates = state.get("candidates", [])
        self.current_trials = state.get("current_trials", [])
        self.deferred = state.get("deferred", [])
        self.events = state.get("events", [])
        self.generation = state.get("generation", 0)
        self.next_model = state.get("next_model", 0)
        self.misfit = state.get("misfit", True)

    def adjusted(self, probe, baseline, *, baseline_variance, budget):
        """Execute the selected observation model inside experiment simulation."""
        old = {"mean": list(baseline), "variance": [baseline_variance] * len(baseline)}
        if self.current is None:
            return old
        if not budget.spend(2 * len(baseline)):
            return None
        return _distribution(self.current, _features(probe, baseline), baseline)

    def forecast(self, probe, baseline, *, baseline_variance, budget):
        """Freeze distribution predictions before querying the real apparatus."""
        models = ([self.current] if self.current else []) + self.candidates
        if len(baseline) != probe.cost or not budget.spend(max(1, 2 * len(models)) * probe.cost):
            return None
        features = _features(probe, baseline)
        return {"generation": self.generation, "features": features, "baseline": list(baseline),
            "old_distribution": {"mean": list(baseline), "variance": [max(1e-8, baseline_variance)] * probe.cost},
            "current_id": self.current["model_id"] if self.current else None,
            "predictions": [{"model_id": model["model_id"], **_distribution(model, features, baseline)}
                            for model in models]}

    def observe(self, forecast, observations, *, budget):
        """Keep real outcomes; only paid scores enter model comparisons."""
        if forecast is None:
            return {"status": "no_forecast", "compute_cost": 0}
        if len(observations) != len(forecast["baseline"]):
            raise ValueError("the real trajectory must match the frozen experiment forecast")
        self.deferred.append({"forecast": deepcopy(forecast), "targets": [row.target for row in observations]})
        return self.flush(budget=budget)

    def flush(self, *, budget):
        started = budget.spent
        comparisons = []
        while self.deferred:
            pending = self.deferred[0]
            forecast, targets = pending["forecast"], pending["targets"]
            if not budget.spend(len(targets) * (1 + len(forecast["predictions"]))):
                break
            self.deferred.pop(0)
            old_loss = _score(forecast["old_distribution"], targets)
            scored = {item["model_id"]: _score(item, targets) for item in forecast["predictions"]}
            baseline_loss = scored.get(forecast["current_id"], old_loss)
            if forecast["generation"] == self.generation:
                for model in self.candidates:
                    if model["model_id"] in scored:
                        after = scored[model["model_id"]]
                        model["trials"].append({"before_loss": baseline_loss, "after_loss": after,
                                                "gain": baseline_loss-after})
                if self.current and forecast["current_id"] == self.current["model_id"]:
                    self.current_trials.append({"before_loss": old_loss, "after_loss": baseline_loss,
                                                "gain": old_loss-baseline_loss})
                    self.current_trials = self.current_trials[-4:]
                    self.misfit |= baseline_loss > old_loss + self.minimum_improvement
                supported = [model for model in self.candidates
                             if _summary(model["trials"], self.minimum_improvement)["supported"]]
                if supported:
                    selected = min(supported, key=lambda model: mean(trial["after_loss"] for trial in model["trials"]))
                    self.current = {key: deepcopy(value) for key, value in selected.items() if key != "trials"}
                    self.events.append({"event": "adopt_simulation_model", "model_id": selected["model_id"],
                        "comparison": _summary(selected["trials"], self.minimum_improvement)})
                    self.candidates, self.current_trials, self.misfit = [], [], False
                    self.generation += 1
                elif self.current and _summary(self.current_trials, self.minimum_improvement)["refuted"]:
                    self.events.append({"event": "withdraw_simulation_model", "model_id": self.current["model_id"],
                        "comparison": _summary(self.current_trials, self.minimum_improvement)})
                    self.current, self.candidates, self.current_trials, self.misfit = None, [], [], True
                    self.generation += 1
                else:
                    self.candidates = [model for model in self.candidates
                        if not _summary(model["trials"], self.minimum_improvement)["refuted"]]
            self.rows.extend({"features": deepcopy(features), "residual": target-baseline}
                for features, target, baseline in zip(forecast["features"], targets, forecast["baseline"]))
            self.batches += 1
            comparisons.append({"old_score": old_loss, "current_score": baseline_loss,
                                "candidate_scores": {str(key): value for key, value in scored.items()}, "rows": len(targets)})
        self._fit(budget)
        return {"status": "pending_scoring" if self.deferred else "measured",
            "compute_cost": budget.spent-started, "comparisons": comparisons,
            "model_id": self.current["model_id"] if self.current else None,
            "pending_models": len(self.candidates), "deferred_batches": len(self.deferred)}

    def _fit(self, budget):
        if self.candidates or not self.misfit or self.batches < 2 or self.batches-self.last_fit_batch < 2:
            return
        if len(self.rows) < 8 or budget.remaining < 12 * len(self.rows):
            return
        varying = [key for key in self.rows[0]["features"]
                   if len({row["features"][key] for row in self.rows}) > 1]
        rows = [Observation({key: row["features"][key] for key in varying}, row["residual"]) for row in self.rows]
        started = budget.spent
        mean_work = SearchBudget(budget.remaining // 3)
        means = synthesize(rows, operations=("refit", "relation"), budget=mean_work, max_history=0)
        budget.spend(mean_work.spent)
        corrections = [Program("const", value=0.)]
        if means:
            corrections.append(means[0].program.simplified())
        pairs = []
        for index, correction in enumerate(dict.fromkeys(corrections)):
            if not budget.spend(len(rows)):
                break
            variances = [Observation(row.features, (row.target-correction.predict(row)) ** 2) for row in rows]
            local = SearchBudget(budget.remaining // max(1, len(corrections)-index))
            fitted = synthesize(variances, operations=("refit", "relation"), budget=local, max_history=0)
            budget.spend(local.spent)
            if fitted:
                self.next_model += 1
                pairs.append({"model_id": self.next_model, "correction": correction.to_dict(),
                    "variance": fitted[0].program.simplified().to_dict(), "trials": [],
                    "training_rows": len(rows), "training_batches": self.batches})
        self.candidates = pairs
        self.last_fit_batch = self.batches
        self.events.append({"event": "propose_simulation_models", "compute_cost": budget.spent-started,
            "training_rows": len(rows), "reason": "fit_systematic_and_conditional_forecast_errors",
            "models": [{key: deepcopy(model[key]) for key in ("model_id", "correction", "variance")}
                       for model in pairs]})

    def to_dict(self):
        return deepcopy({key: getattr(self, key) for key in ("minimum_improvement", "rows", "batches",
            "last_fit_batch", "current", "candidates", "current_trials", "deferred", "events",
            "generation", "next_model", "misfit")})
