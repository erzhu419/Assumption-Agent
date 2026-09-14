"""Adopt world priors when new paired predictions show a useful improvement."""

from copy import deepcopy
from math import sqrt
from statistics import mean, variance


class PriorAdoption:
    """Compare both models before each new measurement updates either model.

    Research observations that produced the candidate are initialization data,
    not adoption evidence. Subsequent paired errors can earn adoption or revoke
    it. Predictions indistinguishable within the requested accuracy retain the
    default model rather than changing acquisition through basis geometry alone.
    """

    def __init__(self, *, minimum_gain=1e-7, batch_size=4, state=None):
        self.state = deepcopy(state) if state else {
            "minimum_gain": minimum_gain, "batch_size": batch_size,
            "selected": "default", "observations": [], "pending": [], "decisions": []}
        if self.state["minimum_gain"] < 0 or self.state["batch_size"] < 2:
            raise ValueError("adoption needs a nonnegative gain tolerance and at least two paired observations")

    @property
    def selected(self):
        return self.state["selected"]

    def observe(self, default_mean, learned_mean, target, noise_variance=0., context=None):
        row = {"default_mean": float(default_mean), "learned_mean": float(learned_mean),
            "target": float(target), "noise_variance": max(0., float(noise_variance)),
            "context": deepcopy(context or {})}
        row["default_error"] = (row["default_mean"] - row["target"]) ** 2
        row["learned_error"] = (row["learned_mean"] - row["target"]) ** 2
        row["gain"] = row["default_error"] - row["learned_error"]
        self.state["observations"].append(row)
        self.state["pending"].append(len(self.state["observations"]) - 1)
        return self._decide()

    def comparison_snapshot(self):
        """Keep the small comparison state preceding a questioned observation."""
        return {"selected": self.selected, "pending": list(self.state["pending"]),
                "decision_count": len(self.state["decisions"])}

    def repair_observation(self, observation_index, confirmation_index, *, comparison_state,
                           default_mean, learned_mean, noise_variance):
        """Reassess adoption using a trusted repeat and restored predictions.

        The original measurements, forecasts and decisions remain in the ledger.
        Only subsequent comparison uses the corrected observation interpretation.
        """
        observations = self.state["observations"]
        anchor, repeated = observations[observation_index], observations[confirmation_index]
        anchor["observation_status"] = "disputed"
        prediction = {"default_mean": float(default_mean), "learned_mean": float(learned_mean),
                      "noise_variance": max(0., float(noise_variance)), "target": repeated["target"]}
        for name in ("default", "learned"):
            prediction[f"{name}_error"] = (prediction[f"{name}_mean"] - prediction["target"]) ** 2
        prediction["gain"] = prediction["default_error"] - prediction["learned_error"]
        repeated["repaired_prediction"] = prediction
        repair = {"observation_index": observation_index, "confirmation_index": confirmation_index,
                  "previous_selected": self.selected, "restored_selected": comparison_state["selected"]}
        self.state.setdefault("observation_repairs", []).append(repair)
        for decision in self.state["decisions"][comparison_state["decision_count"]:]:
            decision["superseded_by_observation_repair"] = observation_index
        self.state["selected"] = comparison_state["selected"]
        self.state["pending"] = [*comparison_state["pending"], confirmation_index]
        return self._decide()

    def _decide(self):
        if len(self.state["pending"]) < self.state["batch_size"]:
            return self.selected

        rows = [self.state["observations"][index].get("repaired_prediction", self.state["observations"][index])
                for index in self.state["pending"]]
        n = len(rows)
        gain = mean(item["gain"] for item in rows)
        # For paired squared errors the target noise is shared. Its contribution
        # to the gain variance is 4*(mean_default-mean_learned)^2*noise_variance.
        noise_error = sum(4 * (item["default_mean"] - item["learned_mean"]) ** 2
                          * item["noise_variance"] for item in rows) / n ** 2
        error = max(variance(item["gain"] for item in rows) / n, noise_error)
        radius = 2 * sqrt(error)
        default_mse, learned_mse = (mean(item[key] for item in rows)
                                    for key in ("default_error", "learned_error"))
        tolerance = self.state["minimum_gain"]
        previous, selected = self.selected, self.selected
        if max(default_mse, learned_mse) <= tolerance:
            selected, reason = "default", "both_models_meet_accuracy"
        elif gain - radius > tolerance:
            selected, reason = "learned", "new_predictions_improve"
        elif gain + radius < -tolerance:
            selected, reason = "default", "new_predictions_refute_advantage"
        else:
            reason = "more_predictive_evidence_needed"
        self.state["selected"] = selected
        self.state["decisions"].append({"observation_indices": list(self.state["pending"]),
            "previous": previous, "selected": selected, "changed": selected != previous,
            "reason": reason, "default_mse": default_mse, "learned_mse": learned_mse,
            "mean_gain": gain, "uncertainty_radius": radius})
        if reason != "more_predictive_evidence_needed":
            self.state["pending"] = []
        return selected

    def to_dict(self):
        return deepcopy(self.state)

    def begin_scope(self, context=None, selected="learned"):
        """Start comparison of refitted models without reusing the old contest."""
        self.state.setdefault("scopes", []).append({
            "observation_start": len(self.state["observations"]),
            "previous": self.selected, "selected": selected,
            "context": deepcopy(context or {})})
        self.state["selected"] = selected
        self.state["pending"] = []
