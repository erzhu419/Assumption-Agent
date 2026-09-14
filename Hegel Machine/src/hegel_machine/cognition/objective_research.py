"""Research goal opportunities using paid observations, independently of drift."""

from copy import deepcopy
from itertools import combinations
from math import erf, exp, pi, sqrt
from statistics import mean

from .continuous_state import _linear_fit
from .decision_experiments import _design, _predict, objective_pool
from .models import SearchBudget
from .language import Program
from .objective_reentry import fit_repaired_objective, select_reentry_control
from .objective_completion import objective_completion
from .objective_alternatives import prepare_interaction_prediction, supported_interaction
from .validation_continuation import evidence_summary, plan_validation_round


def repeat_decision_value(incumbent, alternative, *, repeat_incumbent):
    """Value one repeat under the current scope's measured goal distribution.

    The optimizer's two means determine its recommendation. Current-scope
    means and uncertainty evaluate that recommendation and generate the next
    noisy reading. This is a local, same-point GP update approximation; it
    neither installs the scope hypothesis nor changes the optimizer's labels.
    """
    selected = incumbent if repeat_incumbent else alternative
    other = alternative if repeat_incumbent else incumbent
    total = selected["objective_variance"] + selected["objective_noise"]
    weight = selected["objective_variance"] / total if total > 0 else 0.
    updated = selected["objective_mean"] + weight * (
        selected["current_mean"] - selected["objective_mean"])
    predictive = selected["current_variance"] + selected["objective_noise"]
    spread = weight * sqrt(predictive)
    if spread > 0:
        z = (other["objective_mean"] - updated) / spread
        below = .5 * (1. + erf(z / sqrt(2.)))
        # The measured value and its next reading share latent uncertainty.
        # Account for that dependence when valuing a changed recommendation.
        information = selected["current_variance"] / sqrt(predictive) * exp(-.5*z*z) / sqrt(2*pi)
        switch = 1. - below if repeat_incumbent else below
    else:
        switch = float(updated > other["objective_mean"] if repeat_incumbent
                       else updated < other["objective_mean"])
        information = 0.
    return {"gain": (incumbent["current_mean"] - alternative["current_mean"]) * switch + information,
        "switch_probability": switch, "gp_update_weight": weight,
        "expected_updated_mean": updated, "predictive_variance": predictive}


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
    terms.extend(Program("mul", children=(terms[first], terms[second]))
                 for first, second in model.get("interactions", ()))
    if "concept" in model:
        terms.append(Program.from_dict(model["concept"]))
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
    pairs = tuple(combinations(range(len(spec.feature_names)), 2))
    representations = [(1, ()), (2, ())] + ([(2, pairs)] if pairs else [])
    for degree, interactions in representations:
        design = np.asarray([_design(spec, point, degree, interactions) for point in points])

        def fit(indices):
            if not budget.spend(len(indices) * design.shape[1]):
                return None
            if np.linalg.matrix_rank(design[indices]) < design.shape[1]:
                return None
            columns = [design[indices, j].tolist() for j in range(1, design.shape[1])]
            return {"degree": degree, **({"interactions": list(interactions)} if interactions else {}),
                    "coefficients": _linear_fit(columns, [targets[i] for i in indices])}

        fitted = fit(list(range(len(rows))))
        if fitted is None:
            continue
        predictions, errors, baseline, by_control = [], [], [], {}
        for group in folds:
            training = [i for i, key in enumerate(keys) if groups[key] != group]
            testing = [i for i, key in enumerate(keys) if groups[key] == group]
            fold = fit(training)
            if fold is None or not budget.spend(len(testing)):
                break
            predictions.append(fold)
            reference = mean(targets[i] for i in training)
            for i in testing:
                error = (_predict(spec, fold, points[i]) - targets[i]) ** 2
                errors.append(error)
                by_control.setdefault(keys[i], []).append(error)
            baseline.extend((reference - targets[i]) ** 2 for i in testing)
        if len(predictions) == len(folds):
            models.append({**fitted, "folds": predictions, "cv_loss": mean(errors),
                           "reference_loss": mean(baseline), "training_rows": len(rows),
                           "control_errors": {key: mean(values) for key, values in by_control.items()}})
    base = [model for model in models if not model.get("interactions")]
    if not base:
        return None
    chosen = min(base, key=lambda item: (item["cv_loss"], item["degree"]))
    for candidate in (model for model in models if model.get("interactions")):
        evidence = evidence_summary([{"previous_error": chosen["control_errors"][key],
                                      "candidate_error": value}
                                     for key, value in candidate["control_errors"].items()])
        chosen["interaction_comparison"] = evidence
        if evidence["decision"] == "supported":
            candidate["interaction_comparison"] = evidence
            chosen = candidate
        elif evidence["decision"] == "inconclusive" and evidence["mean_gain"] > 1e-7:
            # Keep a promising explanation available for a future experiment.
            # The current fitted predictor remains the supported base model.
            chosen["interaction_challenger"] = {
                **{key: value for key, value in candidate.items() if key != "control_errors"},
                "interaction_comparison": evidence}
    chosen.pop("control_errors")
    return chosen if chosen["cv_loss"] < chosen["reference_loss"] else None


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
            "constraint_concept": None, "reentry_attempts": [],
            "deployment_fit_indices": [],
            "shared_pending": None, "shared_predictions": [], "shared_outcomes": [],
            "canceled_requests": [],
            "planning_passes": 0, "skipped_unchanged_evidence": 0,
            "screened_controls": 0, "screened_gp_controls": 0,
            "retained_compute": 0, "events": []}

    def to_dict(self):
        return deepcopy(self.state)

    def _start_model_deployment(self, rows, stage):
        """Compare a new executable representation with the deployed goal GP.

        Held-out fits can nominate the explanation. Only later predictions
        made before actual observations can support its deployment.
        """
        model = self.state["model"]
        fit_index = len(self.state["fits"]) - 1
        if (model is None or not (model.get("interactions") or model.get("concept"))
                or self.state["scope_research"] is not None or self.state["scope_revisions"]
                or fit_index in self.state.get("deployment_fit_indices", [])):
            return
        self.state.setdefault("deployment_fit_indices", []).append(fit_index)
        self.state["scope_research"] = {
            "scope_start": self.state["evidence_start"], "phase": "validating",
            "status": "awaiting_shared_deployment_evidence", "deployment": True,
            "candidate": deepcopy(model), "validation": [], "validation_target": 4,
            "training_indices": [row["context"]["global_index"] for row in rows],
            "coverage_calls": 0}
        self.state["events"].append({"action": "compare_goal_representation_deployment",
            "stage": int(stage), "scope_start": self.state["evidence_start"],
            "fit_index": fit_index, "training_end": model["training_end"]})

    def register_constraint_concept(self, program, *, scope_start, stage):
        """Make a measured relation available for a later representation question."""
        self.state["constraint_concept"] = {
            "program": deepcopy(program), "scope_start": int(scope_start), "stage": int(stage)}

    def _resume_rejected(self, rows, *, stage, remaining_calls):
        if (remaining_calls <= 0 or self.state["scope_research"] is not None
                or not self.state["scope_attempts"]):
            return
        index = len(self.state["scope_attempts"]) - 1
        archive = self.state["scope_attempts"][index]
        if (archive.get("reentry") or archive["scope_start"] < self.state["evidence_start"]
                or any(item["attempt_index"] == index for item in self.state.get("reentry_attempts", []))):
            return
        current = [row for row in rows
                   if row["context"]["global_index"] >= archive["scope_start"]]
        if not current or current[-1]["context"]["global_index"] < archive["validation"][-1]["global_index"]:
            return
        research = deepcopy(archive)
        entry = {"attempt_index": index, "stage": int(stage), "representation_revised": False}
        concept = self.state.get("constraint_concept")
        if concept is not None and remaining_calls >= 4:
            budget = self._budget(reserve=False)
            try:
                candidate = fit_repaired_objective(self.spec, current, concept, budget)
            finally:
                self.state["compute_cost"] += budget.spent
            if candidate is not None:
                # The parent keeps every failed prediction. A new expression
                # starts its own future test, after all of its fitting labels.
                research = {"scope_start": max(archive["scope_start"], concept["scope_start"]),
                    "candidate": candidate, "training_indices": candidate["training_indices"],
                    "validation": [], "coverage_calls": 0, "validation_target": 4}
                entry["representation_revised"] = True
        self.state.setdefault("reentry_attempts", []).append(entry)
        evidence = evidence_summary(self._comparison_rows(research))
        if (not entry["representation_revised"] and evidence["decision"] == "rejected"
                and evidence["mean_gain"] + evidence["radius"] < 0):
            entry["status"] = "refuted_without_supported_representation"
            return
        research.update(phase="validating", status="reentered_future_comparison", reentry=deepcopy(entry))
        if not entry["representation_revised"]:
            research["validation_target"] = len(research["validation"]) + 1
        self.state["scope_research"] = research
        self.state["monitoring"] = []
        self.state["confirmation_queue"] = []
        self.state["events"].append({"action": "reenter_objective_research", **deepcopy(entry),
            "scope_start": research["scope_start"], "validation_count": len(research["validation"])})

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

    def incumbent_rebuilt(self, *, stage, scope_start, comparison=()):
        """A supported replacement changes the frozen hypothesis's opponent."""
        research = self.state["scope_research"]
        if research is None or research["candidate"] is None:
            return
        start = len(research["validation"])
        research.setdefault("comparison_rebases", []).append({
            "stage": int(stage), "incumbent_scope_start": int(scope_start),
            "comparison_start": start, "prepared_comparisons": len(comparison)})
        research.update(comparison_start=start, incumbent_scope_start=int(scope_start),
            incumbent_validation=deepcopy(list(comparison)), phase="validating",
            status="awaiting_current_incumbent",
            validation_target=start + max(0, 4 - len(comparison)))
        self.state["accepted_revision"] = None
        for name in ("pending", "shared_pending"):
            self._cancel_request(name, stage=stage, reason="supported_incumbent_replacement")
        self.state["events"].append({"action": "restart_objective_comparison",
            "stage": int(stage), "scope_start": research["scope_start"],
            "incumbent_scope_start": int(scope_start), "validation_count": start,
            "prepared_comparisons": len(comparison),
            "target_count": research["validation_target"]})
        if len(comparison) >= 4:
            self._decide_validation(stage=stage, global_index=int(stage) - 1)

    def _comparison_rows(self, research):
        return [*research.get("incumbent_validation", []),
                *research["validation"][research.get("comparison_start", 0):]]

    def _limit_validation_round(self, remaining_calls, stage):
        research = self.state["scope_research"]
        if (research is not None and research["phase"] == "validating"
                and research["candidate"] is not None and remaining_calls > 0
                and research.get("validation_target", 4) - len(research["validation"]) > remaining_calls):
            previous = research.get("validation_target", 4)
            research["validation_target"] = len(research["validation"]) + remaining_calls
            self.state["events"].append({"action": "complete_available_objective_validation",
                "stage": stage, "previous_target": previous,
                "target_count": research["validation_target"], "remaining_calls": remaining_calls})

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
        research = self.state["scope_research"]
        if model is None or (research is not None and not research.get("deployment")):
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
                if research is not None:
                    research.update(phase="completed", status="deployment_superseded_by_scope_evidence")
                    if research["validation"]:
                        self.state["scope_attempts"].append(deepcopy(research))
                    self.state["events"].append({"action": "reopen_during_goal_deployment",
                        "stage": index + 1, "scope_start": start})
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
        self._resume_rejected(rows, stage=stage, remaining_calls=remaining_calls)
        self._monitor(rows)
        self._limit_validation_round(remaining_calls, stage)
        if self.state["scope_research"] is not None:
            research = self.state["scope_research"]
            if (remaining_calls <= 4 and not research.get("deployment")
                    and ((research["phase"] == "collecting" and remaining_calls < 4)
                         or len(research["validation"]) >= 4)):
                decision = self._scope_decision_confirmation(rows, forecast, stage, remaining_calls)
                if decision is not None:
                    return decision
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
                if (self.state["scope_research"] or {}).get("deployment"):
                    return None
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

    def _scope_decision_confirmation(self, rows, forecast, stage, remaining_calls):
        """A compared control can still be worth measuring for the decision."""
        research = self.state["scope_research"]
        starts = [research["scope_start"], *(item["scope_start"] for item in self.state["scope_attempts"]
                  if item["scope_start"] >= self.state["evidence_start"] and not item.get("deployment"))]
        start = min(starts)
        current = self._groups([row for row in rows if row["context"]["global_index"] >= start])
        budget = self._budget(reserve=False)
        try:
            feasible = []
            for values in self._groups(rows).values():
                if not budget.spend(1):
                    return None
                prediction = forecast(values[0]["observation"]["features"])
                if not prediction["feasible"]:
                    continue
                targets = [row["context"]["objective"]
                           for row in current.get(self._point_key(prediction["features"]), [])]
                feasible.append({**prediction, "current_count": len(targets),
                    "current_mean": mean(targets) if targets else prediction["objective_mean"],
                    "current_variance": (prediction["objective_noise"] / len(targets)
                                         if targets else prediction["objective_variance"])})
            measured = [item for item in feasible if item["current_count"]]
            if not measured:
                return None
            incumbent = min(feasible, key=lambda item: item["objective_mean"])
            alternative = min(measured, key=lambda item: item["current_mean"])
            if self._point_key(incumbent["features"]) == self._point_key(alternative["features"]):
                return None
            validating = research["phase"] == "validating" and research["candidate"] is not None
            compared = {self._point_key(row["features"]) for row in self._comparison_rows(research)}
            needed = max(0, research.get("validation_target", 4) - len(research["validation"])) if validating else 0
            if not incumbent["current_count"]:
                # An old recommendation has no measured value in the questioned
                # scope. Calibrate it before trusting a repeat-value simulator
                # centered on that same, currently disputed GP prediction.
                advances = validating and self._point_key(incumbent["features"]) not in compared
                if 0 < needed <= remaining_calls and remaining_calls - 1 < needed - int(advances):
                    return None
                return self._record_request(incumbent["features"], "objective_confirmation",
                    "calibrate_current_goal_value", stage, decision_scope=start,
                    validation_calls_needed=needed, advances_validation=advances,
                    current_incumbent=incumbent, current_alternative=alternative)
            offers = []
            for item, repeat_incumbent in ((incumbent, True), (alternative, False)):
                advances = validating and self._point_key(item["features"]) not in compared
                preserves = not (0 < needed <= remaining_calls) or remaining_calls - 1 >= needed - int(advances)
                if not budget.spend(1):
                    return None
                offers.append({"features": deepcopy(item["features"]),
                    "advances_validation": advances, "preserves_validation_budget": preserves,
                    **repeat_decision_value(incumbent, alternative, repeat_incumbent=repeat_incumbent)})
            eligible = [offer for offer in offers if offer["preserves_validation_budget"] and offer["gain"] > 1e-12]
            selected = max(eligible, key=lambda offer: offer["gain"]) if eligible else None
            validation_plan = None
            comparison = self._comparison_rows(research)
            if (validating and needed in (1, 2) and needed <= remaining_calls
                    and evidence_summary(comparison)["mean_gain"] > 1e-7):
                controls = []
                for item in feasible:
                    if not budget.spend(1):
                        break
                    controls.append({**item, "candidate_mean": _predict(self.spec, research["candidate"], item["features"])})
                validation_plan = plan_validation_round(comparison, controls,
                    target_count=len(comparison) + needed, remaining_calls=remaining_calls, budget=budget)
            if (validation_plan is not None and (selected is None
                    or validation_plan["expected_gain_per_call"] >= selected["gain"])):
                control = max(validation_plan["controls"], key=lambda item: abs(item["candidate_mean"] - item["objective_mean"]))
                return self._record_request(control["features"], "objective_confirmation",
                    "validate_objective_scope", stage, candidate_mean=control["candidate_mean"],
                    previous_mean=control["objective_mean"], objective_noise=control["objective_noise"],
                    validation_plan=validation_plan, value_offers=offers,
                    **prepare_interaction_prediction(self.spec, research, control["features"], budget))
            if selected is None:
                return None
            return self._record_request(selected["features"], "objective_confirmation",
                "resolve_current_goal_value", stage, decision_scope=start,
                validation_calls_needed=needed, expected_decision_gain=selected["gain"],
                value_offers=offers, validation_plan=validation_plan,
                current_incumbent=incumbent, current_alternative=alternative)
        finally:
            self.state["compute_cost"] += budget.spent

    def boundary_noise_evidence(self, rows, noise_prediction, *, scope_start, remaining_calls):
        """Spend existing planning work on the noise question behind a repeat."""
        from .boundary_noise import repeated_noise_scales

        if remaining_calls <= 0:
            return None
        available = max(0, self.state["compute_budget"] - self.state["compute_cost"])
        budget = SearchBudget(available // remaining_calls)
        try:
            return repeated_noise_scales(rows, noise_prediction, budget,
                scope_start=scope_start, feature_names=self.spec.feature_names)
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
                return [{**offer, "scope_start": scope_start}] if offer is not None else []
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
            return offers
        finally:
            self.state["compute_cost"] += budget.spent

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
        """Predict an already selected control before its real vector draw."""
        if purpose not in {"coverage", "validation", "confirmation", "applicability_probe",
                           "applicability_confirmation", "optimization",
                           "objective_exploration", "objective_confirmation"}:
            raise ValueError("shared objective evidence requires a research or optimization control")
        if self.spec.context.get("objective_direction") != "minimize" or remaining_calls <= 0:
            return None
        goal_control = purpose in {"objective_exploration", "objective_confirmation"}
        own_request = self.state["pending"]
        same_goal_control = (goal_control and own_request is not None
            and own_request["purpose"] == purpose
            and self._point_key(own_request["features"]) == self._point_key(features))
        if same_goal_control and own_request["reason"] == "validate_objective_scope":
            return deepcopy(own_request)
        if not goal_control:
            self._resume_rejected(rows, stage=stage, remaining_calls=remaining_calls)
        if purpose == "optimization" or goal_control:
            research = self.state["scope_research"]
            if (research is None or research["phase"] != "validating"
                    or research["candidate"] is None):
                return None
        rows = self._scope_rows(rows, scope_start=scope_start, stage=stage)
        if not rows:
            return None
        self._limit_validation_round(remaining_calls, stage)
        expected_index = max(row["context"]["global_index"] for row in rows) + 1
        pending = self.state["shared_pending"]
        if (pending is not None and pending["expected_global_index"] == expected_index
                and pending["purpose"] == purpose
                and self._point_key(pending["features"]) == self._point_key(features)):
            return deepcopy(pending)
        if not same_goal_control:
            self._cancel_request("pending", stage=stage,
                reason=("optimization_control_selected" if purpose == "optimization"
                        else "selected_goal_control_changed" if goal_control
                        else "constraint_research_selected_control"))
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
                assessment = self._collect_candidate(current, budget, remaining_calls)
                if assessment["candidate"] is None:
                    return None
            if research["phase"] != "validating":
                return None
            observed = {self._point_key(item["features"]) for item in self._comparison_rows(research)}
            if self._point_key(features) in observed or not budget.spend(1):
                return None
            prediction = forecast(features)
            actual = prediction["features"]
            if self._point_key(actual) in observed:
                return None
            # An already selected control can test the objective outside the
            # feasible region; sharing its measured vector buys no extra call.
            request = {"features": deepcopy(actual), "purpose": purpose,
                "reason": "validate_objective_scope", "stage": stage,
                "scope_start": scope_start, "objective_scope_start": research["scope_start"],
                "expected_global_index": expected_index,
                "candidate_mean": _predict(self.spec, research["candidate"], actual),
                "previous_mean": prediction["objective_mean"],
                "objective_noise": prediction["objective_noise"],
                **prepare_interaction_prediction(self.spec, research, actual, budget),
                **({"reentry": True} if research.get("reentry") else {})}
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
        if model is not None:
            self._start_model_deployment(rows, stage)

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

        screen([*objective_pool(self.spec, model, budget), *around(center)])
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

    def _collect_candidate(self, current, budget, remaining_calls):
        research = self.state["scope_research"]
        model = self.state["model"] or {}
        trigger = research.get("trigger", [])
        reference = trigger[0]["threshold"] / 9 if trigger else model.get("cv_loss", 1e-7)
        completion = research.setdefault("completion", {})

        assessment = objective_completion(self.spec, current, budget, fit=_fit,
            reference_loss=reference, remaining_calls=remaining_calls, state=completion)
        research["completion_plan"] = {key: value for key, value in assessment.items()
                                       if key != "candidate"}
        research["status"] = assessment["status"]
        if assessment["candidate"] is not None:
            candidate = assessment["candidate"]
            indices = list(completion["training_indices"])
            candidate["training_end"] = max(indices)
            research.update(candidate=candidate, phase="validating", validation_target=4,
                            training_indices=indices)
            if assessment["status"] == "budgeted_low_order_validation":
                self.state["events"].append({"action": "begin_budgeted_goal_validation",
                    "stage": max(row["context"]["global_index"] for row in current) + 1,
                    "scope_start": research["scope_start"], "training_indices": indices,
                    "remaining_calls": remaining_calls,
                    "completion_limited_by": assessment["completion_limited_by"]})
        return assessment

    def _research_scope(self, rows, forecast, stage, remaining_calls):
        research = self.state["scope_research"]
        if research.get("deployment"):
            # Its proposal already exists. Ordinary paid controls provide
            # prospective comparisons without another identification search.
            return None
        if research.get("reentry"):
            budget = self._budget(reserve=False)
            try:
                selected = select_reentry_control(self.spec, research, rows, forecast, budget)
                if selected is None:
                    research["status"] = "awaiting_shared_distinguishing_control"
                    return None
                return self._record_request(selected["features"], "objective_confirmation",
                    "validate_objective_scope", stage, candidate_mean=selected["candidate_mean"],
                    previous_mean=selected["objective_mean"], objective_noise=selected["objective_noise"],
                    reentry=True, disagreement=selected["disagreement"],
                    **prepare_interaction_prediction(self.spec, research, selected["features"], budget))
            finally:
                self.state["compute_cost"] += budget.spent
        if research["phase"] == "validating" and len(research["validation"]) >= 4:
            needed = research.get("validation_target", 4) - len(research["validation"])
            if needed <= 0 or remaining_calls != needed:
                # Share later comparisons while there is slack. At the last
                # complete window, explicitly select a new comparison control.
                return None
        current = [row for row in rows if row["context"]["global_index"] >= research["scope_start"]]
        budget = self._budget(reserve=False)
        try:
            if research["phase"] == "collecting":
                if remaining_calls < 4:
                    research["status"] = "insufficient_validation_budget"
                    return None
                assessment = self._collect_candidate(current, budget, remaining_calls)
                if assessment["candidate"] is None and remaining_calls > 4:
                    used = {self._point_key(row["observation"]["features"]) for row in current}
                    identifying_budget = SearchBudget(assessment["identification_allowance"])
                    interactions = (tuple(combinations(range(len(self.spec.feature_names)), 2))
                                    if assessment["need_interactions"] else ())
                    try:
                        _, identifying = self._identifying_controls(current, identifying_budget,
                            centers=rows, interactions=interactions)
                        for point in identifying:
                            prediction = forecast(point)
                            actual = prediction["features"]
                            if prediction["feasible"] and self._point_key(actual) not in used:
                                research["coverage_calls"] += 1
                                return self._record_request(actual, "objective_exploration",
                                    "identify_objective_scope", stage)
                    finally:
                        budget.spend(identifying_budget.spent)
                    research["status"] = ("completion_compute_reserved" if not assessment["identification_allowance"]
                                          else "no_feasible_identifying_control")
                    return None
                elif assessment["candidate"] is None:
                    research["status"] = "insufficient_identifying_evidence"
                    return None
            if research["phase"] != "validating":
                return None
            candidate = research["candidate"]
            observed = {self._point_key(item["features"]) for item in self._comparison_rows(research)}
            # The candidate stays frozen, but subsequently measured feasible
            # controls can supply a new independent repeat. Cache its scores so
            # extending this pool does not rescore all previous candidates.
            points = ([*objective_pool(self.spec, candidate, budget)]
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
                previous_mean=selected["objective_mean"], objective_noise=selected["objective_noise"],
                **prepare_interaction_prediction(self.spec, research, selected["features"], budget))
        finally:
            self.state["compute_cost"] += budget.spent

    def _identifying_controls(self, rows, budget, *, centers, interactions=None):
        """Resolve the independent directions missing from grouped fits."""
        import numpy as np
        points = [row["observation"]["features"] for row in rows]
        keys = [self._point_key(point) for point in points]
        groups = {key: index % 5 for index, key in enumerate(dict.fromkeys(keys))}
        if interactions is None:
            interactions = tuple(combinations(range(len(self.spec.feature_names)), 2))
        dimension = 2 * len(self.spec.feature_names) + 1 + len(interactions)
        if not budget.spend(len(rows) * dimension):
            return False, []
        design = np.asarray([_design(self.spec, point, 2, interactions) for point in points])
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
            vector = np.asarray(_design(self.spec, point, 2, interactions))
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
        own_request = self.state["pending"]
        if (is_shared and own_request is not None
                and own_request["purpose"] == context["purpose"]
                and self._point_key(own_request["features"]) == self._point_key(features)):
            # One vector answers the selected action and the frozen hypothesis.
            # Keep both uses, while the prospective comparison below runs once.
            owned = {**record, "request": deepcopy(own_request)}
            if "incumbent_measured_mean" in own_request:
                owned["measured_improvement"] = own_request["incumbent_measured_mean"] - float(output[0])
            self.state["outcomes"].append(owned)
            self.state["pending"] = None
        if request["reason"] == "validate_objective_scope":
            research = self.state["scope_research"]
            research["validation"].append({"features": deepcopy(features),
                "global_index": context["global_index"], "target": float(output[0]),
                "source": ("shared_optimization_control" if is_shared and context["purpose"] == "optimization"
                           else "shared_goal_control" if is_shared and context["purpose"] in {
                               "objective_exploration", "objective_confirmation"}
                           else "shared_constraint_control" if is_shared else "objective_control"),
                "purpose": context["purpose"],
                "candidate_error": (float(output[0]) - request["candidate_mean"]) ** 2,
                "previous_error": (float(output[0]) - request["previous_mean"]) ** 2,
                **({"interaction_error": (float(output[0]) - request["interaction_mean"]) ** 2}
                   if "interaction_mean" in request else {})})
            if len(research["validation"]) >= research.get("validation_target", 4):
                self._decide_validation(stage=context["stage"], global_index=context["global_index"])

    def _decide_validation(self, *, stage, global_index):
        research = self.state["scope_research"]
        validation = self._comparison_rows(research)
        evidence = evidence_summary(validation)
        alternative = None
        shared_count = sum("interaction_error" in row for row in validation)
        if research["candidate"].get("interaction_challenger") and shared_count >= 4:
            comparison_cost = 2 * shared_count
            if self.state["compute_budget"] - self.state["compute_cost"] >= comparison_cost:
                self.state["compute_cost"] += comparison_cost
                alternative = supported_interaction(research, validation, stage, base_evidence=evidence)
        if alternative is not None:
            research.setdefault("interaction_comparisons", []).append(alternative["comparison"])
            if alternative["supported"]:
                # Both explanations predicted these same future labels. Only
                # their paired evidence can replace the current nomination.
                research["base_candidate"] = deepcopy(research["candidate"])
                research["base_validation"] = deepcopy(research["validation"])
                research["base_validation_rounds"] = research.get("validation_rounds", [])
                research["validation_rounds"] = []
                research["base_comparison_rebases"] = research.pop("comparison_rebases", [])
                research["candidate"] = alternative["model"]
                research["validation"] = alternative["validation"]
                research.pop("incumbent_validation", None)
                research.pop("comparison_start", None)
                research.pop("validation_pool", None)
                research.pop("validation_pool_means", None)
                validation = research["validation"]
                evidence = alternative["comparison"]["against_incumbent"]
                self.state["events"].append({"action": "select_interaction_from_shared_validation",
                    "stage": stage, "scope_start": research["scope_start"],
                    "validation_indices": [row["global_index"] for row in validation]})
        research.setdefault("validation_rounds", []).append({
            **evidence, "stage": stage})
        reentry_uncertain = (research.get("reentry") and evidence["decision"] == "rejected"
                            and evidence["mean_gain"] + evidence["radius"] >= 0)
        if evidence["decision"] == "inconclusive" or reentry_uncertain:
            research.update(status="inconclusive_future_predictions",
                validation_target=len(research["validation"]) + (1 if research.get("reentry") else 2))
            self.state["events"].append({"action": "continue_objective_validation",
                "stage": stage, "scope_start": research["scope_start"],
                "validation_count": len(research["validation"]), "comparison_count": len(validation), "target_count": research["validation_target"],
                "mean_gain": evidence["mean_gain"], "radius": evidence["radius"]})
            return
        supported = evidence["decision"] == "supported"
        research["status"] = "supported" if supported else "future_predictions_do_not_improve"
        research["phase"] = "completed"
        if supported:
            self.state["accepted_revision"] = {
                "scope_start": research["scope_start"], "model": deepcopy(research["candidate"]),
                "program": objective_program(self.spec, research["candidate"]).to_dict(),
                "validation_mse": evidence["candidate_mse"],
                "previous_mse": evidence["previous_mse"],
                "validation_rounds": deepcopy(research["validation_rounds"]),
                "comparison_rebases": deepcopy(research.get("comparison_rebases", [])),
                **({"reentry": deepcopy(research["reentry"])} if research.get("reentry") else {}),
                **({"deployment": True} if research.get("deployment") else {}),
                "training_indices": research["training_indices"],
                "validation_indices": [row["global_index"] for row in validation]}
        else:
            self.state["scope_attempts"].append(deepcopy(research))
            self.state["scope_research"] = None
            self.state["monitoring"] = []
            self.state["last_monitored"] = global_index
            self.state["plan_rows"] = global_index + 1 - self.state["evidence_start"]
