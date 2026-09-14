"""Research goal opportunities using paid observations, independently of drift."""

from copy import deepcopy
from math import sqrt
from statistics import mean

from .continuous_state import _linear_fit
from .decision_experiments import _design, _predict, objective_pool
from .models import SearchBudget
from .language import Program


def objective_program(spec, model):
    """Compile the same measured goal model that made the future predictions."""
    terms = []
    for name in spec.feature_names:
        low, high = spec.feature_bounds[name]
        feature = Program("mul", children=(Program("const", value=1 / (high - low)),
            Program("sub", children=(Program("feature", name=name), Program("const", value=low)))))
        terms.append(feature)
    if model["degree"] == 2:
        terms.extend(Program("mul", children=(term, term)) for term in terms[:])
    result = Program("const", value=model["coefficients"][0])
    for term, coefficient in zip(terms, model["coefficients"][1:]):
        result = Program("add", children=(result,
            Program("mul", children=(Program("const", value=coefficient), term))))
    return result.simplified()


def _fit(spec, rows, budget):
    """Compare identifiable goal models on held-out groups of whole controls."""
    import numpy as np

    points = [row["observation"]["features"] for row in rows]
    targets = [float(row["context"]["objective"]) for row in rows]
    keys = [tuple(point[name] for name in spec.feature_names) for point in points]
    groups = {key: index % 5 for index, key in enumerate(dict.fromkeys(keys))}
    folds = sorted(set(groups.values()))
    if len(folds) < 3:
        return None
    models = []
    for degree in (1, 2):
        design = np.asarray([_design(spec, point, degree) for point in points])

        def fit(indices):
            if not budget.spend(len(indices) * design.shape[1]):
                return None
            if np.linalg.matrix_rank(design[indices]) < design.shape[1]:
                return None
            columns = [design[indices, j].tolist() for j in range(1, design.shape[1])]
            return {"degree": degree,
                    "coefficients": _linear_fit(columns, [targets[i] for i in indices])}

        fitted = fit(list(range(len(rows))))
        if fitted is None:
            continue
        predictions, errors, baseline = [], [], []
        for group in folds:
            training = [i for i, key in enumerate(keys) if groups[key] != group]
            testing = [i for i, key in enumerate(keys) if groups[key] == group]
            fold = fit(training)
            if fold is None or not budget.spend(len(testing)):
                break
            predictions.append(fold)
            reference = mean(targets[i] for i in training)
            errors.extend((_predict(spec, fold, points[i]) - targets[i]) ** 2 for i in testing)
            baseline.extend((reference - targets[i]) ** 2 for i in testing)
        if len(predictions) == len(folds) and mean(errors) < mean(baseline):
            models.append({**fitted, "folds": predictions, "cv_loss": mean(errors),
                           "reference_loss": mean(baseline), "training_rows": len(rows)})
    return min(models, key=lambda item: (item["cv_loss"], item["degree"])) if models else None


class ObjectiveResearch:
    """Spend on new goal evidence, and validate changes in its applicability."""

    def __init__(self, spec, *, compute_budget=20000, state=None):
        self.spec = spec
        self.state = deepcopy(state) if state is not None else {
            "compute_budget": compute_budget, "compute_cost": 0,
            "scope_start": None, "evidence_start": None, "model": None,
            "fit_rows": 0, "plan_rows": 0, "last_monitored": -1,
            "pending": None, "confirmation_queue": [], "confirmation_started": False,
            "fits": [], "requests": [], "outcomes": [], "monitoring": [],
            "scope_research": None, "scope_revisions": [], "accepted_revision": None,
            "scope_attempts": [],
            "shared_pending": None, "shared_predictions": [], "shared_outcomes": [],
            "canceled_requests": [],
            "planning_passes": 0, "skipped_unchanged_evidence": 0,
            "screened_controls": 0, "screened_gp_controls": 0,
            "completion_plans": [],
            "retained_compute": 0, "events": []}

    def to_dict(self):
        return deepcopy(self.state)

    @property
    def accepted_revision(self):
        return deepcopy(self.state["accepted_revision"])

    def acknowledge_revision(self):
        revision = self.state["accepted_revision"]
        if revision is None:
            raise ValueError("no measured objective revision awaits installation")
        self.state["scope_revisions"].append(deepcopy(revision))
        self.state["model"] = deepcopy(revision["model"])
        self.state["model"]["training_end"] = revision["validation_indices"][-1]
        self.state["evidence_start"] = revision["scope_start"]
        self.state["last_monitored"] = revision["validation_indices"][-1]
        self.state["monitoring"] = []
        self.state["scope_research"] = None
        self.state["accepted_revision"] = None
        self.state["plan_rows"] = 0
        self.state["confirmation_started"] = False

    def _point_key(self, point):
        return tuple(point[name] for name in self.spec.feature_names)

    def _cancel_request(self, name, *, stage, reason):
        request = self.state[name]
        if request is not None:
            self.state["canceled_requests"].append({"request": deepcopy(request),
                "stage": stage, "reason": reason})
            self.state[name] = None

    def _scope_rows(self, rows, *, scope_start, stage):
        if self.state["scope_start"] is None:
            self.state.update(scope_start=scope_start, evidence_start=scope_start)
        elif self.state["scope_start"] != scope_start:
            # Constraint evidence changes feasibility, not the applicability of
            # a separately tested objective explanation or its frozen candidate.
            for name in ("pending", "shared_pending"):
                self._cancel_request(name, stage=stage, reason="constraint_scope_changed")
            for point in self.state["confirmation_queue"]:
                self.state["canceled_requests"].append({"request": {"features": deepcopy(point),
                    "purpose": "objective_confirmation"}, "stage": stage,
                    "reason": "constraint_scope_changed"})
            self.state.update(scope_start=scope_start, confirmation_queue=[], confirmation_started=False)
        return [row for row in rows
                if row["context"]["global_index"] >= self.state["evidence_start"]]

    def _record_request(self, point, purpose, reason, stage, **details):
        request = {"features": deepcopy(point), "purpose": purpose, "reason": reason,
                   "stage": stage, "scope_start": self.state["scope_start"], **details}
        self.state["pending"] = request
        self.state["requests"].append(deepcopy(request))
        return deepcopy(request)

    def _budget(self, *, reserve=True):
        dimension = len(self.spec.feature_names)
        training_rows = max(12, 2 * (2 * dimension + 1), self.state["plan_rows"] + 4)
        # Keep enough work for a later complete grouped fit and its experiment
        # design. This reserve is released for a measured scope counterexample.
        validation_pool = 1 + 33 * dimension + training_rows + 4
        required = 5 * training_rows * (3 * dimension + 2) + 2 * training_rows + 5 * validation_pool
        available = self.state["compute_budget"] - self.state["compute_cost"]
        self.state["retained_compute"] = min(required, max(0, available)) if reserve else 0
        return SearchBudget(max(0, available - (required if reserve else 0)))

    def _monitor(self, rows):
        model = self.state["model"]
        if model is None or self.state["scope_research"] is not None:
            return
        threshold = 9 * max(1e-7, model["cv_loss"])
        for row in rows:
            index = row["context"]["global_index"]
            if index <= max(self.state["last_monitored"], model["training_end"]):
                continue
            if self.state["compute_cost"] >= self.state["compute_budget"]:
                break
            prediction = _predict(self.spec, model, row["observation"]["features"])
            self.state["compute_cost"] += 1
            self.state["last_monitored"] = index
            self.state["monitoring"].append({"global_index": index,
                "prediction": prediction, "target": row["context"]["objective"],
                "error": (prediction - row["context"]["objective"]) ** 2,
                "threshold": threshold})
            recent = self.state["monitoring"][-4:]
            failed = [item for item in recent if item["error"] > item["threshold"]]
            if len(recent) == 4 and len(failed) >= 2 and mean(item["error"] for item in recent) > threshold:
                start = failed[0]["global_index"]
                self.state["scope_research"] = {"scope_start": start, "phase": "collecting",
                    "trigger": deepcopy(recent), "candidate": None,
                    "validation": [], "coverage_calls": 0}
                self.state["events"].append({"action": "research_objective_scope", "scope_start": start,
                    "stage": index + 1, "threshold": threshold})
                self.state["confirmation_queue"] = []
                return

    def next_control(self, rows, forecast, *, scope_start, stage, remaining_calls):
        if self.spec.context.get("objective_direction") != "minimize" or remaining_calls <= 0:
            return None
        rows = self._scope_rows(rows, scope_start=scope_start, stage=stage)
        if self.state["pending"] is not None:
            return deepcopy(self.state["pending"])
        if not rows:
            return None
        self._monitor(rows)
        if self.state["scope_research"] is not None:
            return self._research_scope(rows, forecast, stage, remaining_calls)
        if self.state["scope_revisions"] and 1 <= remaining_calls <= 4:
            # The installed goal head now nominates competing terminal actions.
            # SC-KG chooses the actual purchase before it becomes a request.
            return None
        terminal = 2 <= remaining_calls <= 4 and not self.state["confirmation_started"]
        if not self.state["confirmation_queue"] and not terminal and len(rows) < self.state["plan_rows"] + 4:
            self.state["skipped_unchanged_evidence"] += 1
            return None
        if self.state["confirmation_queue"]:
            point = self.state["confirmation_queue"].pop(0)
            return self._record_request(point, "objective_confirmation",
                "measured_and_posterior_incumbents_disagree", stage)
        new_batch = len(rows) >= self.state["plan_rows"] + 4
        if new_batch:
            self.state["plan_rows"] = len(rows)
            self.state["planning_passes"] += 1
        budget = self._budget()
        try:
            if new_batch:
                self._fit_if_needed(rows, budget, stage)
            grouped = self._groups(rows)
            predictions = {}

            def predict(point):
                key = self._point_key(point)
                if key not in predictions:
                    result = forecast(point)
                    predictions[key] = {**result, "features": result.get("features", point)}
                return predictions[key]

            observed = self._incumbents(grouped, predict, budget, all_controls=terminal)
            if not observed:
                return None
            incumbent = min(observed, key=lambda item: item["objective_mean"])
            if terminal:
                measured = min(observed, key=lambda item: item["measured_mean"])
                if self._point_key(measured["features"]) != self._point_key(incumbent["features"]):
                    gap = measured["objective_mean"] - incumbent["objective_mean"]
                    uncertainty = sqrt(max(0., sum(item["objective_variance"] +
                        item["objective_noise"] / item["replicates"] for item in (measured, incumbent))))
                    if gap <= 2 * uncertainty:
                        self.state["confirmation_started"] = True
                        self.state["confirmation_queue"] = [deepcopy(incumbent["features"])]
                        return self._record_request(measured["features"], "objective_confirmation",
                            "measured_and_posterior_incumbents_disagree", stage,
                            posterior_gap=gap, comparison_uncertainty=uncertainty)
            if not new_batch:
                return None
            offer = self._exploration_offer(rows, grouped, incumbent, predict, budget, stage)
            if offer is not None:
                return self._record_request(offer["features"], offer["purpose"], offer["reason"], stage,
                    **{key: value for key, value in offer.items()
                       if key not in {"features", "purpose", "reason", "stage"}})
            return None
        finally:
            self.state["compute_cost"] += budget.spent

    def terminal_offers(self, rows, forecast, *, scope_start, stage, remaining_calls):
        """Nominate at most four actions for the optimizer's ordinary scoring.

        Only a goal head already accepted on future measurements can use this
        path. Offers are predictions, not purchased observations or requests.
        """
        if (self.spec.context.get("objective_direction") != "minimize"
                or not self.state["scope_revisions"] or not 1 <= remaining_calls <= 4):
            return []
        rows = self._scope_rows(rows, scope_start=scope_start, stage=stage)
        if self.state["pending"] is not None:
            return [deepcopy(self.state["pending"])]
        if not rows:
            return []
        self._monitor(rows)
        if self.state["scope_research"] is not None or self.state["model"] is None:
            return []
        available = max(0, self.state["compute_budget"] - self.state["compute_cost"])
        # Another complete fit followed by four future draws no longer fits
        # after buying this observation. Share the remaining search work among
        # the actual remaining calls instead of reserving an unusable batch.
        budget = SearchBudget(available // remaining_calls)
        self.state["retained_compute"] = available - budget.remaining
        if not budget.remaining:
            return []
        try:
            grouped = self._groups(rows)
            predictions = {}

            def predict(point):
                key = self._point_key(point)
                if key not in predictions:
                    if not budget.spend(1):
                        return {"features": point, "feasible": False}
                    result = forecast(point)
                    predictions[key] = {**result, "features": result.get("features", point)}
                return predictions[key]

            uncertified = []
            observed = self._incumbents(grouped, predict, budget, all_controls=True,
                                        uncertified=uncertified)
            if not observed:
                offer = self._certificate_offer(uncertified, None, budget, stage)
                offers = [{**offer, "scope_start": scope_start}] if offer is not None else []
                return self._plan_terminal_completion(offers, predictions, budget, stage, remaining_calls)
            incumbent = min(observed, key=lambda item: item["objective_mean"])
            measured = min(observed, key=lambda item: item["measured_mean"])
            offers = []
            for item, reason in ((measured, "measured_terminal_incumbent"),
                                 (incumbent, "posterior_terminal_incumbent")):
                key = self._point_key(item["features"])
                if any(self._point_key(offer["features"]) == key for offer in offers):
                    continue
                offers.append({"features": deepcopy(item["features"]),
                    "purpose": "objective_confirmation", "reason": reason, "stage": stage,
                    "scope_start": scope_start, "incumbent_measured_mean": item["measured_mean"],
                    "replicates": item["replicates"],
                    "prediction": {key: item[key] for key in
                                   ("objective_mean", "objective_variance", "objective_noise")}})
            certificate = self._certificate_offer(uncertified, incumbent, budget, stage)
            if certificate is not None and all(self._point_key(certificate["features"]) != self._point_key(item["features"])
                                               for item in offers):
                offers.append({**certificate, "scope_start": scope_start})
            offer = self._exploration_offer(rows, grouped, incumbent, predict, budget, stage,
                                            quantized=True)
            if offer is not None and all(self._point_key(offer["features"]) != self._point_key(item["features"])
                                         for item in offers):
                offers.append({**offer, "scope_start": scope_start})
            return self._plan_terminal_completion(offers, predictions, budget, stage, remaining_calls)
        finally:
            self.state["compute_cost"] += budget.spent

    def _completion_need(self, prediction, budget, remaining_calls):
        """Price repeats under a fixed mean/noise forecast, not a final certificate."""
        variance = max(0., prediction["constraint_variance"])
        observation_noise = max(0., prediction["constraint_observation_noise"])
        guard = max(0., prediction["constraint_epistemic_guard"])
        margin = prediction["constraint_margin"]
        floor = margin - guard
        multiplier = guard / sqrt(variance) if variance else 0.

        def latent_variance(count):
            if not count or not variance:
                return variance
            return variance * observation_noise / (observation_noise + count * variance)

        trajectory = []
        for count in range(remaining_calls + 1):
            if not budget.spend(1):
                return {"estimated_calls": None, "status": "compute_budget", "trajectory": trajectory}
            current, following = latent_variance(count), latent_variance(count + 1)
            expected = floor + multiplier * sqrt(current)
            next_upper = floor + multiplier * (sqrt(following) + sqrt(max(0., current - following)))
            trajectory.append({"additional_calls": count, "expected_margin": expected,
                               "next_observation_upper": next_upper})
            if expected <= 0. and next_upper <= 0.:
                return {"estimated_calls": count, "status": "estimated_complete", "trajectory": trajectory}
        return {"estimated_calls": None, "status": "outside_remaining_budget", "trajectory": trajectory}

    def _plan_terminal_completion(self, offers, predictions, budget, stage, remaining_calls):
        if not offers:
            return []
        forecasts = {self._point_key(item["features"]): item for item in predictions.values()}
        if any(not all(name in forecasts[self._point_key(offer["features"])] for name in
                       ("constraint_observation_noise", "constraint_epistemic_guard")) for offer in offers):
            return offers
        planned = []
        for offer in offers:
            point = self._point_key(offer["features"])
            forecast = forecasts[point]
            plan = self._completion_need(forecast, budget, remaining_calls)
            # Certificate aleatoric uncertainty, slack, and the current mean
            # remain fixed in this local planning estimate. Only actual new
            # measurements can change those quantities or grant a certificate.
            plan.update(current_margin=forecast["constraint_margin"],
                observation_noise=forecast["constraint_observation_noise"],
                certificate_noise=forecast["constraint_noise"],
                semantics="fixed_mean_noise_local_repeat_estimate")
            prior = next((outcome["request"]["prediction"] for outcome in reversed(self.state["outcomes"])
                if self._point_key(outcome["request"]["features"]) == point
                and outcome["context"]["global_index"] >= self.state["scope_start"]
                and "constraint_margin" in outcome["request"].get("prediction", {})), None)
            if prior is not None:
                plan.update(previous_margin=prior["constraint_margin"],
                    certificate_changed_sign=(prior["constraint_margin"] > 0. >= forecast["constraint_margin"]))
            planned.append({**offer, "completion_plan": plan})
        old = [offer for offer in planned if offer["purpose"] == "objective_confirmation"]
        best = min(old, key=lambda item: item["prediction"]["objective_mean"]) if old else None
        reserve = best["completion_plan"]["estimated_calls"] if best is not None else 0
        deadline = reserve is not None and reserve > 0 and reserve >= remaining_calls
        retained, deferred = [], []
        for offer in planned:
            if deadline and offer is not best:
                deferred.append(offer)
                continue
            if deadline:
                offer["completion_plan"]["must_complete_now"] = True
            if offer["reason"] == "supported_goal_with_uncertain_constraint":
                calls = offer["completion_plan"]["estimated_calls"]
                if reserve is None:
                    # An unfinishable old question is not a completed one. A
                    # different direction must itself fit and improve on it.
                    affordable = (calls is not None and max(1, calls) <= remaining_calls
                        and offer["prediction"]["objective_mean"] < best["prediction"]["objective_mean"])
                else:
                    affordable = calls is not None and max(1, calls) + reserve <= remaining_calls
                offer["completion_plan"].update(reserved_existing_calls=reserve,
                    existing_features=deepcopy(best["features"]) if reserve != 0 else None,
                    admitted=affordable)
                if not affordable:
                    deferred.append(offer)
                    continue
            retained.append(offer)
        self.state.setdefault("completion_plans", []).append({"stage": stage,
            "remaining_calls": remaining_calls, "offers": deepcopy(retained), "deferred": deepcopy(deferred)})
        return retained

    def _certificate_offer(self, candidates, incumbent, budget, stage):
        """A promising measured point may earn a test of its lost certificate."""
        model = self.state["model"]
        if not candidates or model is None:
            return None
        models = [model, *model["folds"]]
        center = None
        if incumbent is not None:
            if not budget.spend(len(models)):
                return None
            center = [_predict(self.spec, item, incumbent["features"]) for item in models]
        offers = []
        for item in candidates:
            if incumbent is not None and item["measured_mean"] >= incumbent["measured_mean"]:
                continue
            if not budget.spend(len(models)):
                break
            predictions = [_predict(self.spec, fitted, item["features"]) for fitted in models]
            comparison = {}
            if center is not None:
                gains = [a - b for a, b in zip(center[1:], predictions[1:])]
                average = mean(gains)
                uncertainty = sqrt((len(gains) - 1) / len(gains) *
                                   sum((value - average) ** 2 for value in gains))
                gain = center[0] - predictions[0]
                comparison = {"predicted_gain": gain, "gain_uncertainty": uncertainty,
                              "supported_gain": gain - uncertainty}
                if comparison["supported_gain"] <= 1e-7:
                    continue
            offers.append({**item, "model_mean": predictions[0], **comparison})
        if not offers:
            return None
        if incumbent is None:
            # With no certified incumbent there is no certified improvement
            # comparison. Research the model's best measured, nominally feasible
            # point; a successful repeat must still pass the original certificate.
            chosen = min(offers, key=lambda item: (item["model_mean"], self._point_key(item["features"])))
        else:
            chosen = max(offers, key=lambda item: item["supported_gain"])
        return {"features": deepcopy(chosen["features"]), "purpose": "objective_confirmation",
            "reason": "research_promising_constraint_certificate", "stage": stage,
            "feasible": False, "incumbent_measured_mean": chosen["measured_mean"],
            "constraint_measured_mean": chosen["constraint_measured_mean"],
            "constraint_replicates": chosen["constraint_replicates"],
            "certified_incumbent": deepcopy(incumbent["features"]) if incumbent is not None else None,
            "prediction": {key: chosen[key] for key in (
                "model_mean", "objective_mean", "objective_variance", "objective_noise",
                "constraint_mean", "constraint_variance", "constraint_noise", "constraint_margin",
                "predicted_gain", "gain_uncertainty", "supported_gain") if key in chosen}}

    def select_terminal_offer(self, offer, *, stage):
        """Register the one nominated control actually selected for purchase."""
        pending = self.state["pending"]
        if pending is not None:
            if self._point_key(pending["features"]) != self._point_key(offer["features"]):
                raise ValueError("the selected goal observation has not been acquired yet")
            return deepcopy(pending)
        return self._record_request(offer["features"], offer["purpose"], offer["reason"], stage,
            **{key: value for key, value in offer.items()
               if key not in {"features", "purpose", "reason", "stage", "scope_start"}})

    def prepare_shared_control(self, rows, forecast, features, *, scope_start, stage,
                               remaining_calls, purpose):
        """Predict a constraint-selected experiment before its real vector draw."""
        if purpose not in {"coverage", "validation", "confirmation", "applicability_probe",
                           "applicability_confirmation"}:
            raise ValueError("shared objective evidence requires an explicit constraint research purpose")
        if self.spec.context.get("objective_direction") != "minimize" or remaining_calls <= 0:
            return None
        rows = self._scope_rows(rows, scope_start=scope_start, stage=stage)
        if not rows:
            return None
        expected_index = max(row["context"]["global_index"] for row in rows) + 1
        pending = self.state["shared_pending"]
        if (pending is not None and pending["expected_global_index"] == expected_index
                and pending["purpose"] == purpose
                and self._point_key(pending["features"]) == self._point_key(features)):
            return deepcopy(pending)
        self._cancel_request("pending", stage=stage, reason="constraint_research_selected_control")
        self._cancel_request("shared_pending", stage=stage, reason="shared_control_replaced")
        self._monitor(rows)
        research = self.state["scope_research"]
        budget = self._budget(reserve=research is None)
        try:
            if research is None:
                # Establish a fixed baseline even if constraint research has
                # priority from the first online call. This fits paid rows but
                # never scans a goal experiment pool or proposes another query.
                if self.state["model"] is None and len(rows) >= self.state["fit_rows"] + 4:
                    self._fit_if_needed(rows, budget, stage)
                return None
            if research["phase"] == "collecting":
                if remaining_calls < 4:
                    return None
                current = [row for row in rows
                           if row["context"]["global_index"] >= research["scope_start"]]
                ready, _ = self._identifying_controls(current, budget, centers=rows)
                candidate = _fit(self.spec, current, budget) if ready or remaining_calls == 4 else None
                if candidate is None:
                    return None
                candidate["training_end"] = current[-1]["context"]["global_index"]
                research.update(candidate=candidate, phase="validating",
                    training_indices=[row["context"]["global_index"] for row in current])
            if research["phase"] != "validating":
                return None
            observed = {self._point_key(item["features"]) for item in research["validation"]}
            if self._point_key(features) in observed or not budget.spend(1):
                return None
            prediction = forecast(features)
            actual = prediction["features"]
            if self._point_key(actual) in observed:
                return None
            # A constraint experiment can validly test the objective outside
            # the feasible region; its actual control is already being bought.
            request = {"features": deepcopy(actual), "purpose": purpose,
                "reason": "validate_objective_scope", "stage": stage,
                "scope_start": scope_start, "objective_scope_start": research["scope_start"],
                "expected_global_index": expected_index,
                "candidate_mean": _predict(self.spec, research["candidate"], actual),
                "previous_mean": prediction["objective_mean"],
                "objective_noise": prediction["objective_noise"]}
            self.state["shared_pending"] = request
            self.state["shared_predictions"].append(deepcopy(request))
            return deepcopy(request)
        finally:
            self.state["compute_cost"] += budget.spent

    def _groups(self, rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(self._point_key(row["observation"]["features"]), []).append(row)
        return grouped

    def _incumbents(self, grouped, predict, budget, *, all_controls=False, uncertified=None):
        bound = float(self.spec.context["constraint_upper_bound"])
        candidates = []
        for values in grouped.values():
            trusted = [row for row in values if row.get("observation_status") != "disputed"
                       and row["context"]["global_index"] >= self.state["scope_start"]]
            if trusted and mean(row["observation"]["target"] for row in trusted) <= bound:
                candidates.append({"features": values[0]["observation"]["features"],
                    "measured_mean": mean(row["context"]["objective"] for row in values),
                    "replicates": len(values),
                    **({"constraint_measured_mean": mean(row["observation"]["target"] for row in trusted),
                        "constraint_replicates": len(trusted)} if uncertified is not None else {})})
        if not all_controls and len(candidates) > 6:
            measured = sorted(candidates, key=lambda item: item["measured_mean"])[:2]
            model = self.state["model"]
            if model and not budget.spend(len(candidates)):
                model = None
            ranked = (sorted(candidates, key=lambda item: _predict(self.spec, model, item["features"]))
                      if model else sorted(candidates, key=lambda item: item["measured_mean"]))
            candidates = measured + [item for item in ranked if item not in measured]
        result = []
        for item in candidates:
            prediction = predict(item["features"])
            if prediction["feasible"]:
                result.append({**item, **prediction})
                if not all_controls and len(result) >= 6:
                    break
            elif uncertified is not None and "objective_mean" in prediction:
                uncertified.append({**item, **prediction})
        return result

    def _fit_if_needed(self, rows, budget, stage):
        model = self.state["model"]
        recent = self.state["monitoring"][-4:]
        still_predicts = bool(model and recent and mean(item["error"] for item in recent)
                              <= 9 * max(model["cv_loss"], 1e-7))
        if still_predicts:
            return
        model = _fit(self.spec, rows, budget)
        self.state["fit_rows"] = len(rows)
        if model is not None:
            model["training_end"] = rows[-1]["context"]["global_index"]
            self.state["model"] = model
            self.state["monitoring"] = []
            self.state["last_monitored"] = model["training_end"]
        self.state["fits"].append({"stage": stage, "scope_start": self.state["scope_start"],
            "rows": len(rows), "supported": model is not None,
            **({key: model[key] for key in ("degree", "cv_loss", "reference_loss")}
               if model else {})})

    def _exploration_offer(self, rows, grouped, incumbent, predict, budget, stage, *, quantized=False):
        model = self.state["model"]
        if model is None or budget.remaining <= 0:
            return None
        center = incumbent["features"]
        center_predictions = [_predict(self.spec, item, center) for item in [model, *model["folds"]]]
        if not budget.spend(len(center_predictions)):
            return None
        scored = {}

        def screen(points):
            for point in points:
                key = self._point_key(point)
                if key in scored or key in grouped:
                    continue
                if not budget.spend(len(center_predictions)):
                    return
                predictions = [_predict(self.spec, item, point) for item in [model, *model["folds"]]]
                gains = [a - b for a, b in zip(center_predictions[1:], predictions[1:])]
                average = mean(gains)
                uncertainty = sqrt((len(gains) - 1) / len(gains) *
                    sum((value - average) ** 2 for value in gains))
                scored[key] = {"features": point, "model_mean": predictions[0],
                    "predicted_gain": center_predictions[0] - predictions[0],
                    "gain_uncertainty": uncertainty,
                    "supported_gain": center_predictions[0] - predictions[0] - uncertainty}
                self.state["screened_controls"] += 1

        def around(point):
            return [{**point, name: low + i * (high - low) / 32}
                    for name, (low, high) in self.spec.feature_bounds.items() for i in range(33)]

        offers = []
        checked = set()

        def neighboring_constraint_question(prediction):
            if (not quantized or prediction["feasible"]
                    or "constraint_measured_mean" not in incumbent
                    or "constraint_mean" not in prediction):
                return False
            bound = float(self.spec.context["constraint_upper_bound"])
            if incumbent["constraint_measured_mean"] > bound or prediction["constraint_mean"] > bound:
                return False
            point = prediction["features"]
            changed = [name for name in self.spec.feature_names if point[name] != center[name]]
            if len(changed) != 1:
                return False
            name = changed[0]
            low, high = self.spec.feature_bounds[name]
            # Use the same public axis grid as exploration, after the apparatus
            # has quantized the control. A remote forecast cannot turn into a
            # local experiment merely because the unquantized request was near.
            anchor_cell = round(32 * (center[name] - low) / (high - low))
            actual_cell = round(32 * (point[name] - low) / (high - low))
            return abs(actual_cell - anchor_cell) == 1

        def check_best():
            # The goal model rejects unsupported offers before any GP/HVD work.
            shortlist = sorted((item for key, item in scored.items() if key not in checked
                                and item["supported_gain"] > 1e-7),
                               key=lambda item: -item["supported_gain"])
            for item in shortlist:
                key = self._point_key(item["features"])
                checked.add(key)
                prediction = predict(item["features"])
                self.state["screened_gp_controls"] += 1
                constraint_question = neighboring_constraint_question(prediction)
                if ((prediction["feasible"] or constraint_question)
                        and self._point_key(prediction["features"]) not in grouped
                        and prediction["objective_mean"] < incumbent["objective_mean"]):
                    if quantized and prediction["features"] != item["features"]:
                        if not budget.spend(len(center_predictions)):
                            return
                        actual = [_predict(self.spec, fold, prediction["features"])
                                  for fold in [model, *model["folds"]]]
                        gains = [a - b for a, b in zip(center_predictions[1:], actual[1:])]
                        average = mean(gains)
                        uncertainty = sqrt((len(gains) - 1) / len(gains) *
                            sum((value - average) ** 2 for value in gains))
                        item = {**item, "model_mean": actual[0],
                                "predicted_gain": center_predictions[0] - actual[0],
                                "gain_uncertainty": uncertainty,
                                "supported_gain": center_predictions[0] - actual[0] - uncertainty}
                        if item["supported_gain"] <= 1e-7:
                            continue
                    offers.append({**item, **prediction,
                                   **({"constraint_question": True} if constraint_question else {})})
                    if not constraint_question:
                        break

        screen([*objective_pool(self.spec, model), *around(center)])
        check_best()
        if offers:
            screen(around(min(offers, key=lambda item: item["model_mean"])["features"]))
            check_best()
        if not offers:
            return None
        offer = max(offers, key=lambda item: item["supported_gain"])
        question = offer.get("constraint_question", False)
        return {"features": deepcopy(offer["features"]), "purpose": "objective_exploration",
            "reason": ("supported_goal_with_uncertain_constraint" if question else "supported_goal_improvement"),
            "stage": stage,
            "prediction": {key: offer[key] for key in ("model_mean", "objective_mean",
                "predicted_gain", "gain_uncertainty", *( ("constraint_mean", "constraint_variance",
                "constraint_noise", "constraint_margin") if question else ())) if key in offer},
            **({"feasible": False, "support_anchor": {"features": deepcopy(center),
                "objective_measured_mean": incumbent["measured_mean"],
                "constraint_measured_mean": incumbent["constraint_measured_mean"],
                "constraint_replicates": incumbent["constraint_replicates"]}} if question else {}),
            "incumbent_measured_mean": incumbent["measured_mean"], "training_rows": model["training_rows"]}

    def _research_scope(self, rows, forecast, stage, remaining_calls):
        research = self.state["scope_research"]
        current = [row for row in rows if row["context"]["global_index"] >= research["scope_start"]]
        budget = self._budget(reserve=False)
        try:
            if research["phase"] == "collecting":
                if remaining_calls < 4:
                    research["status"] = "insufficient_validation_budget"
                    return None
                ready, identifying = self._identifying_controls(current, budget, centers=rows)
                candidate = _fit(self.spec, current, budget) if ready or remaining_calls == 4 else None
                if candidate is not None:
                    candidate["training_end"] = current[-1]["context"]["global_index"]
                    research.update(candidate=candidate, phase="validating",
                        training_indices=[row["context"]["global_index"] for row in current])
                elif remaining_calls > 4:
                    used = {self._point_key(row["observation"]["features"]) for row in current}
                    for point in identifying:
                        prediction = forecast(point)
                        actual = prediction["features"]
                        if prediction["feasible"] and self._point_key(actual) not in used:
                            research["coverage_calls"] += 1
                            return self._record_request(actual, "objective_exploration",
                                "identify_objective_scope", stage)
                    research["status"] = "no_feasible_identifying_control"
                    return None
                else:
                    research["status"] = "insufficient_identifying_evidence"
                    return None
            if research["phase"] != "validating":
                return None
            candidate = research["candidate"]
            observed = {self._point_key(item["features"]) for item in research["validation"]}
            # The candidate stays frozen, but subsequently measured feasible
            # controls can supply a new independent repeat. Cache its scores so
            # extending this pool does not rescore all previous candidates.
            points = ([*objective_pool(self.spec, candidate)]
                      if "validation_pool" not in research else [])
            points.extend(row["observation"]["features"] for row in current)
            pool = research.setdefault("validation_pool", [])
            means = research.setdefault("validation_pool_means", [])
            used = {self._point_key(point) for point in pool}
            for point in points:
                key = self._point_key(point)
                if key in used:
                    continue
                if not budget.spend(1):
                    research["status"] = "insufficient_validation_planning_compute"
                    return None
                pool.append(deepcopy(point))
                means.append(_predict(self.spec, candidate, point))
                used.add(key)
            ordered = sorted(zip(means, pool), key=lambda item: item[0])
            research["validation_pool_means"] = [value for value, _ in ordered]
            research["validation_pool"] = [point for _, point in ordered]
            pool = research["validation_pool"]
            scored = []
            for point in pool:
                if self._point_key(point) in observed:
                    continue
                if not budget.spend(1):
                    break
                prediction = forecast(point)
                actual = prediction["features"]
                if not prediction["feasible"] or self._point_key(actual) in observed:
                    continue
                candidate_mean = _predict(self.spec, candidate, actual)
                scored.append({**prediction, "candidate_mean": candidate_mean,
                    "disagreement": abs(candidate_mean - prediction["objective_mean"])})
                if len(scored) >= 4:
                    break
            if not scored:
                research["status"] = "no_feasible_validation_control"
                return None
            selected = max(scored, key=lambda item: item["disagreement"])
            return self._record_request(selected["features"], "objective_confirmation",
                "validate_objective_scope", stage, candidate_mean=selected["candidate_mean"],
                previous_mean=selected["objective_mean"], objective_noise=selected["objective_noise"])
        finally:
            self.state["compute_cost"] += budget.spent

    def _identifying_controls(self, rows, budget, *, centers):
        """Resolve the independent directions missing from grouped fits."""
        import numpy as np
        points = [row["observation"]["features"] for row in rows]
        keys = [self._point_key(point) for point in points]
        groups = {key: index % 5 for index, key in enumerate(dict.fromkeys(keys))}
        dimension = 2 * len(self.spec.feature_names) + 1
        if not budget.spend(len(rows) * dimension):
            return False, []
        design = np.asarray([_design(self.spec, point, 2) for point in points])
        fold_ids = sorted(set(groups.values()))
        matrices = [design[[groups[key] != group for key in keys]] for group in fold_ids]
        ready = len(matrices) >= 3 and all(len(matrix) and np.linalg.matrix_rank(matrix) == dimension
                                         for matrix in matrices)
        if ready:
            return True, []
        new_group = len(groups) % 5
        affected = [matrix for group, matrix in zip(fold_ids, matrices) if group != new_group]
        if not affected:
            affected = [design]
        inverse = [np.linalg.pinv(matrix.T @ matrix + 1e-6 * np.eye(dimension)) for matrix in affected]
        missing = [np.linalg.svd(matrix, full_matrices=True)[2][np.linalg.matrix_rank(matrix):]
                   if len(matrix) else np.eye(dimension) for matrix in affected]
        needs_direction = any(len(null) for null in missing)

        # A feasible position from before the objective change remains usable
        # under the same constraint scope. Its old objective label is not added
        # to the new design. Re-evaluate the actual constraint GP at each offer.
        bound = float(self.spec.context["constraint_upper_bound"])
        feasible_centers = []
        centers = [row for row in centers
                   if row["context"]["global_index"] >= self.state["scope_start"]]
        for values in self._groups(centers).values():
            trusted = [row for row in values if row.get("observation_status") != "disputed"]
            if not trusted:
                continue
            target = mean(row["observation"]["target"] for row in trusted)
            if target <= bound:
                feasible_centers.append((target, values[0]["observation"]["features"]))
        ordered_centers = [point for _, point in sorted(feasible_centers, key=lambda item: item[0])]
        ordered_centers.extend(point for point in points if point not in ordered_centers)

        def information(point):
            vector = np.asarray(_design(self.spec, point, 2))
            rank_gain = sum(bool(np.linalg.norm(null @ vector) > 1e-8) for null in missing)
            gain = sum(float(np.log1p(max(0., vector @ covariance @ vector))) for covariance in inverse)
            return rank_gain, gain

        def controls():
            seen = set(groups)
            # Expand one measured center at a time. Exhausting an unhelpful
            # feasible slice therefore opens another slice, while the usual
            # successful case pays for only one local design pool.
            for center in ordered_centers:
                pool = [center, *({**center, name: low + i * (high - low) / 32}
                    for name, (low, high) in self.spec.feature_bounds.items() for i in range(33))]
                pool = list({self._point_key(point): point for point in pool
                    if self._point_key(point) not in seen}.values())
                if not budget.spend(len(pool) * dimension):
                    return
                seen.update(self._point_key(point) for point in pool)
                scored = [(information(point), point) for point in pool]
                for (rank_gain, _), point in sorted(scored, key=lambda item: item[0], reverse=True):
                    if rank_gain or not needs_direction:
                        yield point

        return False, controls()

    def observe(self, features, output, *, context):
        shared = self.state["shared_pending"]
        is_shared = shared is not None and shared["purpose"] == context.get("purpose")
        request = shared if is_shared else self.state["pending"]
        if request is None or context.get("purpose") != request["purpose"]:
            return
        if self._point_key(features) != self._point_key(request["features"]):
            raise ValueError("the goal research outcome must answer its actual requested control")
        if is_shared and context["global_index"] != request["expected_global_index"]:
            raise ValueError("shared objective evidence must be the next real observation after prediction")
        record = {"request": deepcopy(request), "context": deepcopy(context),
                  "objective": float(output[0]), "constraint": float(output[1])}
        if "incumbent_measured_mean" in request:
            record["measured_improvement"] = request["incumbent_measured_mean"] - float(output[0])
        self.state["shared_outcomes" if is_shared else "outcomes"].append(record)
        self.state["shared_pending" if is_shared else "pending"] = None
        if request["reason"] == "validate_objective_scope":
            research = self.state["scope_research"]
            research["validation"].append({"features": deepcopy(features),
                "global_index": context["global_index"], "target": float(output[0]),
                "source": "shared_constraint_control" if is_shared else "objective_control",
                "purpose": context["purpose"],
                "candidate_error": (float(output[0]) - request["candidate_mean"]) ** 2,
                "previous_error": (float(output[0]) - request["previous_mean"]) ** 2})
            if len(research["validation"]) >= 4:
                from statistics import variance
                validation = research["validation"]
                gains = [row["previous_error"] - row["candidate_error"] for row in validation]
                radius = 2 * sqrt(variance(gains) / len(gains))
                candidate_loss = mean(row["candidate_error"] for row in validation)
                supported = mean(gains) - radius > 1e-7
                research["status"] = "supported" if supported else "future_predictions_do_not_improve"
                research["phase"] = "completed"
                if supported:
                    self.state["accepted_revision"] = {
                        "scope_start": research["scope_start"], "model": deepcopy(research["candidate"]),
                        "program": objective_program(self.spec, research["candidate"]).to_dict(),
                        "validation_mse": candidate_loss,
                        "previous_mse": mean(row["previous_error"] for row in validation),
                        "training_indices": research["training_indices"],
                        "validation_indices": [row["global_index"] for row in validation]}
                else:
                    self.state["scope_attempts"].append(deepcopy(research))
                    self.state["scope_research"] = None
                    self.state["monitoring"] = []
                    self.state["last_monitored"] = context["global_index"]
                    self.state["plan_rows"] = context["global_index"] + 1 - self.state["evidence_start"]
