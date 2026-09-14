"""Turn persistent joint prediction failure into new, measured hypotheses."""

from copy import deepcopy
from dataclasses import asdict
from random import Random
from statistics import mean

from .language import Program
from .models import SearchBudget
from .revision_candidates import distinguishing_controls, propose_competitors, select_competitor
from .workspace import observation_from_dict


class JointModelRevision:
    """Research on a new evidence segment while retaining the old paid ledger.

    The reference comes from predictions made before seeing their outcomes.
    It stays fixed until a replacement passes future measurements: a model's
    increasing residual variance must not explain away its own failure.
    A lone strong counterexample buys a same-control confirmation against the
    original frozen predictions. Confirmation, coverage and validation queries
    consume the caller's ordinary query budget.
    A fitted candidate is frozen throughout each validation batch.
    """

    def __init__(self, spec, *, minimum_error=1e-7, batch_size=4, seed=7,
                 compute_budget=20000, state=None):
        self.spec = spec
        self.state = deepcopy(state) if state else {
            "minimum_error": minimum_error, "batch_size": batch_size,
            "seed": seed, "compute_budget": compute_budget,
            "phase": "monitoring", "reference_scale": None, "reference_rows": [],
            "reference_noise": None,
            "pending_window": [], "trigger_window": None,
            "observations": [], "counterexamples": [], "episodes": [], "events": [],
            "controls": [], "candidate": None, "competitors": [], "accepted_revision": None,
            "confirmation": None, "confirmations": [],
            "observation_repair": None, "observation_repairs": [],
            "research_recovery_indices": [],
            "applicability_probe": None, "applicability_probes": [],
            "costs": {"compute_cost": 0, "confirmation_calls": 0,
                      "applicability_probe_calls": 0, "applicability_confirmation_calls": 0,
                      "coverage_calls": 0, "validation_calls": 0}}
        if minimum_error < 0 or batch_size < 2:
            raise ValueError("joint revision needs nonnegative accuracy and at least two observations")

    @property
    def candidate(self):
        item = self.state["candidate"]
        return Program.from_dict(item["program"]) if item else None

    @property
    def candidates(self):
        return {item["candidate_id"]: Program.from_dict(item["program"])
                for item in self.state["competitors"]}

    @property
    def accepted_revision(self):
        item = deepcopy(self.state["accepted_revision"])
        if item:
            item["program"] = Program.from_dict(item["program"])
        return item

    def next_control(self):
        return deepcopy(self.state["controls"][0]) if self.state["controls"] else None

    @property
    def observation_repair(self):
        return deepcopy(self.state["observation_repair"])

    def acknowledge_observation_repair(self):
        """Record a measured-output correction after the predictor applies it."""
        repair = self.state["observation_repair"]
        if repair is None:
            raise ValueError("no observation repair awaits application")
        self.state["observations"][repair["observation_index"]]["observation_status"] = "disputed"
        self.state["observation_repairs"].append(deepcopy(repair))
        self._event("repair_observation", **deepcopy(repair))
        self.state["observation_repair"] = None

    @property
    def control_purpose(self):
        if not self.state["controls"]:
            return None
        return {"confirming": "confirmation", "collecting": "coverage",
                "validating": "validation", "applicability_probe": "applicability_probe",
                "applicability_confirmation": "applicability_confirmation"}[self.state["phase"]]

    def start_applicability_probe(self, features, reference_observations, *, goal_scope_start, context):
        """Ask whether a previously measured constraint still applies here.

        A changed goal motivates the question; only new constraint measurements
        can answer it. The caller supplies trusted past readings at this exact
        control, including paid initialization observations when available.
        """
        phase = self.state["phase"]
        interrupted = None
        if phase in {"collecting", "validating", "unresolved"}:
            episode = self.state["episodes"][-1]
            if episode["scope_start"] >= goal_scope_start:
                return False
            interrupted = {"phase": phase, "controls": deepcopy(self.state["controls"]),
                           "episode_index": len(self.state["episodes"]) - 1,
                           "recovered_indices": list(self.state["research_recovery_indices"])}
        elif phase != "monitoring":
            return False
        if (self.state["reference_noise"] is None
                or not reference_observations or self.state["observation_repair"] is not None
                or self.state["costs"]["compute_cost"] >= self.state["compute_budget"]):
            return False
        if any(item["goal_scope_start"] == goal_scope_start for item in self.state["applicability_probes"]):
            return False
        # Two possible readings of the old control, three new controls, and
        # four later predictions leave a low-dimensional replacement testable.
        required_calls = 2 + 3 + self.state["batch_size"]
        if interrupted is not None:
            remaining_plan = len(interrupted["controls"])
            if phase == "collecting":
                remaining_plan += self.state["batch_size"]
            required_calls = max(required_calls, 2 + remaining_plan)
        if context.get("remaining_calls", 0) < required_calls:
            return False
        noise = max(self.state["minimum_error"], self.state["reference_noise"])
        references = deepcopy(reference_observations)
        reference_mean = mean(item["target"] for item in references)
        question = {"features": deepcopy(features), "reference_observations": references,
                    "reference_mean": reference_mean, "measurement_noise": noise,
                    "threshold": 9 * noise * (1 + 1 / len(references)),
                    "goal_scope_start": goal_scope_start, "context": deepcopy(context),
                    "measurements": []}
        if interrupted is not None:
            question["interrupted_research"] = interrupted
        self.state["applicability_probe"] = question
        self.state["phase"] = "applicability_probe"
        self.state["controls"] = [deepcopy(features)]
        self._event("request_constraint_applicability_probe", **deepcopy(question))
        return True

    def _measure_applicability(self, index, context):
        question = self.state["applicability_probe"]
        row = self.state["observations"][index]
        target = row["observation"]["target"]
        difference = target - question["reference_mean"]
        measurement = {"observation_index": index, "global_index": context["global_index"],
                       "target": target, "difference": difference,
                       "squared_difference": difference ** 2}
        question["measurements"].append(measurement)
        purpose = self.state["phase"]
        self.state["costs"][f"{purpose}_calls"] += 1
        self._event("measure_constraint_applicability", purpose=purpose,
                    goal_scope_start=question["goal_scope_start"], threshold=question["threshold"],
                    **deepcopy(measurement))
        if purpose == "applicability_probe" and difference ** 2 > question["threshold"]:
            self.state["phase"] = "applicability_confirmation"
            self.state["controls"] = [deepcopy(question["features"])]
            return
        first = question["measurements"][0]
        confirmed = (purpose == "applicability_confirmation"
                     and difference ** 2 > question["threshold"]
                     and first["difference"] * difference > 0)
        record = {**deepcopy(question), "confirmed": confirmed,
                  "status": ("measured_constraint_change" if confirmed else
                             "reference_retained" if difference ** 2 <= question["threshold"] else
                             "inconsistent_new_readings")}
        self.state["applicability_probes"].append(record)
        self.state["applicability_probe"] = None
        self.state["phase"] = "monitoring"
        self.state["controls"] = []
        self._event("resolve_constraint_applicability", goal_scope_start=question["goal_scope_start"],
                    confirmed=confirmed, status=record["status"],
                    observation_indices=[item["observation_index"] for item in question["measurements"]])
        if confirmed:
            indices = [item["observation_index"] for item in question["measurements"]]
            interrupted = question.get("interrupted_research")
            if interrupted is not None:
                previous = self.state["episodes"][interrupted["episode_index"]]
                previous["status"] = "superseded_by_new_evidence"
                previous["subsequent_scope_start"] = first["global_index"]
                self._event("revise_research_scope", previous_scope_start=previous["scope_start"],
                            evidence_kind="measured_constraint_change", confirmation_indices=indices,
                            goal_scope_start=question["goal_scope_start"])
            self.state["trigger_window"] = {
                "evidence_kind": "measured_constraint_change", "observation_indices": indices,
                "anomalous_indices": indices, "threshold": question["threshold"],
                "reference_mean": question["reference_mean"],
                "reference_observations": deepcopy(question["reference_observations"]),
                "goal_scope_start": question["goal_scope_start"]}
            # The old measurement established a reference, not a label for the
            # new segment. Both new readings remain paid training evidence.
            self.state["counterexamples"] = indices
            self._reopen(context)
        elif (purpose == "applicability_confirmation"
              and difference ** 2 <= question["threshold"]
              and (target - first["target"]) ** 2 > 18 * question["measurement_noise"]):
            repair = {"observation_index": first["observation_index"],
                      "global_index": first["global_index"], "confirmation_index": index,
                      "confirmation_global_index": context["global_index"], "output_index": 1,
                      "reason": "isolated_observation_conflict", "repeat_target": target,
                      "threshold": question["threshold"],
                      "separation_squared": (target - first["target"]) ** 2}
            self.state["observation_repair"] = repair
            self._event("propose_observation_repair", **deepcopy(repair))
        if not confirmed and question.get("interrupted_research") is not None:
            interrupted = question["interrupted_research"]
            episode = self.state["episodes"][interrupted["episode_index"]]
            episode["row_indices"].extend(item["observation_index"]
                                         for item in question["measurements"][:-1])
            self._resume_research(interrupted, index, context)

    def _request_confirmation(self, index, threshold, context, *, interrupted_research=None):
        anchor = self.state["observations"][index]
        if min(anchor["default_error"], anchor["learned_error"]) <= threshold:
            return
        # After a replicate, three new controls leave a univariate quadratic
        # identifiable in each training fold; keep four future validation draws.
        minimum_calls = 1 + 3 + self.state["batch_size"]
        if context.get("remaining_calls", 0) < minimum_calls:
            self._event("defer_counterexample_confirmation", observation_index=index,
                        remaining_calls=context.get("remaining_calls", 0),
                        required_calls=minimum_calls, reason="insufficient_query_budget")
            return
        self.state["confirmation"] = {
            "anchor_index": index, "features": deepcopy(anchor["observation"]["features"]),
            "anchor_target": anchor["observation"]["target"], "threshold": threshold,
            "frozen_predictions": {name: anchor[f"{name}_mean"] for name in ("default", "learned")}}
        if interrupted_research is not None:
            self.state["confirmation"]["interrupted_research"] = deepcopy(interrupted_research)
        self.state["phase"] = "confirming"
        self.state["controls"] = [deepcopy(anchor["observation"]["features"])]
        self._event("request_counterexample_confirmation", **deepcopy(self.state["confirmation"]))

    def _confirm(self, index, context):
        question = self.state["confirmation"]
        interrupted = question.get("interrupted_research")
        anchor_index = question["anchor_index"]
        row = self.state["observations"][index]
        target = row["observation"]["target"]
        errors = {name: (target - prediction) ** 2
                  for name, prediction in question["frozen_predictions"].items()}
        same_direction = all((target - prediction) * (question["anchor_target"] - prediction) > 0
                             for prediction in question["frozen_predictions"].values())
        confirmed = min(errors.values()) > question["threshold"] and same_direction
        record = {**deepcopy(question), "observation_index": index,
            "repeat_target": target, "repeat_errors": errors,
            "same_direction": same_direction, "confirmed": confirmed}
        row["confirmation_predictions"] = deepcopy(question["frozen_predictions"])
        self.state["confirmations"].append(record)
        self.state["costs"]["confirmation_calls"] += 1
        self._event("measure_counterexample_confirmation", **deepcopy(record))
        self.state["confirmation"] = None
        self.state["controls"] = []
        self.state["phase"] = "monitoring"
        if confirmed:
            if interrupted is not None:
                previous = self.state["episodes"][interrupted["episode_index"]]
                previous["status"] = "superseded_by_new_evidence"
                previous["subsequent_scope_start"] = self.state["observations"][anchor_index]["context"]["global_index"]
                self._event("revise_research_scope", previous_scope_start=previous["scope_start"],
                    recovered_indices=interrupted["recovered_indices"],
                    confirmation_indices=[anchor_index, index])
            self.state["trigger_window"] = {
                "evidence_kind": "repeated_control", "observation_indices": [anchor_index, index],
                "anomalous_indices": [anchor_index, index], "threshold": question["threshold"],
                "frozen_predictions": deepcopy(question["frozen_predictions"]),
                **{f"{name}_mse": (self.state["observations"][anchor_index][f"{name}_error"]
                                     + errors[name]) / 2 for name in errors}}
            self.state["counterexamples"] = list(range(anchor_index, index + 1))
            self._reopen(context)
        else:
            # The repeat answers the frozen question. Its residual against the
            # already updated GP would instead test assimilation of the anomaly.
            self.state["pending_window"] = [i for i in self.state["pending_window"] if i != anchor_index]
            # Two independent readings have the sum of their noise variances.
            # A repeat supporting an old predictor is evidence about the first
            # reading only when the readings themselves are incompatible.
            separation = (target - question["anchor_target"]) ** 2
            if min(errors.values()) <= question["threshold"] and separation > 2 * question["threshold"]:
                anchor = self.state["observations"][anchor_index]
                repair = {"observation_index": anchor_index,
                    "global_index": anchor["context"]["global_index"],
                    "confirmation_index": index, "confirmation_global_index": context["global_index"],
                    "output_index": 1, "reason": "isolated_observation_conflict",
                    "repeat_target": target, "threshold": question["threshold"],
                    "separation_squared": separation}
                self.state["observation_repair"] = repair
                self._event("propose_observation_repair", **deepcopy(repair))
            if interrupted is not None:
                self._resume_research(interrupted, index, context)

    def _resume_research(self, interrupted, index, context):
        """A refuted scope question leaves the frozen research plan in force."""
        episode = self.state["episodes"][interrupted["episode_index"]]
        episode["row_indices"].append(index)
        repair = self.state["observation_repair"]
        if repair is not None and repair["observation_index"] in episode["validation_indices"]:
            # The unchanged candidate already predicted this same control before
            # either draw. Use the trusted repeat for that validation position.
            anchor = self.state["observations"][repair["observation_index"]]
            self.state["observations"][index]["candidate_predictions"] = deepcopy(anchor["candidate_predictions"])
            episode["validation_indices"] = [index if i == repair["observation_index"] else i
                                             for i in episode["validation_indices"]]
        self.state["phase"] = interrupted["phase"]
        self.state["controls"] = deepcopy(interrupted["controls"])
        self.state["research_recovery_indices"] = []
        self._event("resume_research", phase=interrupted["phase"],
                    scope_start=episode["scope_start"], observation_index=index)
        if not self.state["controls"]:
            if self.state["phase"] == "collecting":
                self._propose()
            elif self.state["phase"] == "validating":
                self._validate(context)

    def _monitor_research_scope(self, index, context):
        """Test renewed joint failure after accuracy recovered during research."""
        row = self.state["observations"][index]
        threshold = 9 * self.state["reference_scale"]
        row["reference_threshold"] = threshold
        recovered = self.state["research_recovery_indices"]
        if max(row["default_error"], row["learned_error"]) <= threshold:
            point = row["observation"]["features"]
            recovered = [i for i in recovered
                         if self.state["observations"][i]["observation"]["features"] != point]
            self.state["research_recovery_indices"] = [*recovered, index][-self.state["batch_size"]:]
            return False
        self.state["research_recovery_indices"] = []
        if (len(recovered) < self.state["batch_size"]
                or min(row["default_error"], row["learned_error"]) <= threshold
                or self.state["costs"]["compute_cost"] >= self.state["compute_budget"]):
            return False
        interrupted = {"phase": self.state["phase"], "controls": deepcopy(self.state["controls"]),
                       "episode_index": len(self.state["episodes"]) - 1,
                       "recovered_indices": list(recovered)}
        self._request_confirmation(index, threshold, context, interrupted_research=interrupted)
        return self.state["phase"] == "confirming"

    def _coverage(self, count):
        episode = self.state["episodes"][-1]
        rng = Random(self.state["seed"] + 1009 * len(self.state["episodes"])
                     + 101 * len(episode["proposals"]))
        columns = {}
        for name in self.spec.feature_names:
            low, high = self.spec.feature_bounds[name]
            values = [low + (high - low) * (i + rng.random()) / count for i in range(count)]
            rng.shuffle(values)
            columns[name] = values
        self.state["controls"] = [{name: values[i] for name, values in columns.items()}
                                  for i in range(count)]

    def _event(self, action, **details):
        context = self.state["observations"][-1]["context"]
        self.state["events"].append({"action": action, "stage": context.get("stage"), **details})

    def _reopen(self, context):
        indices = list(self.state["counterexamples"])
        start = self.state["observations"][indices[0]]["context"]["global_index"]
        episode = {"scope_start": start, "trigger_indices": indices, "row_indices": indices[:],
                   "trigger_window": deepcopy(self.state["trigger_window"]),
                   "proposals": [], "status": "researching", "validation_indices": []}
        self.state["episodes"].append(episode)
        self.state["candidate"] = None
        self.state["competitors"] = []
        self.state["research_recovery_indices"] = []
        self._event("reopen_hypothesis_research", scope_start=start,
                    observation_indices=indices, reference_scale=self.state["reference_scale"],
                    window=deepcopy(self.state["trigger_window"]))
        dimension = len(self.spec.feature_names)
        # Cover enough new controls to fit a general pairwise response, including
        # the already acquired counterexamples. Four later outcomes stay unseen.
        desired = max(2 * dimension + 2, (dimension + 1) * (dimension + 2) // 2 - len(indices))
        available = context.get("remaining_calls", 0) - self.state["batch_size"]
        if available < 0:
            self._unresolved("insufficient_query_budget")
            return
        coverage = min(desired, available)
        episode["planned_coverage"] = coverage
        episode["reserved_validation"] = self.state["batch_size"]
        self.state["phase"] = "collecting"
        if coverage:
            self._coverage(coverage)
        else:
            self._propose()

    def _unresolved(self, reason):
        self.state["phase"] = "unresolved"
        self.state["episodes"][-1]["status"] = reason
        self.state["candidate"] = None
        self.state["competitors"] = []
        self.state["controls"] = []
        self._event("research_unresolved", reason=reason)

    def _propose(self):
        episode = self.state["episodes"][-1]
        repair = self.state["observation_repair"]
        indices = [i for i in episode["row_indices"]
                   if self.state["observations"][i].get("observation_status") != "disputed"
                   and (repair is None or i != repair["observation_index"])]
        rows = [observation_from_dict(self.state["observations"][i]["observation"])
                for i in indices]
        budget = SearchBudget(self.state["compute_budget"] - self.state["costs"]["compute_cost"])
        competitors = propose_competitors(self.spec, rows, budget)
        if not competitors:
            self.state["costs"]["compute_cost"] += budget.spent
            self._unresolved("insufficient_compute_budget")
            return
        for item in competitors:
            item["training_indices"] = list(indices)
        candidate = deepcopy(competitors[0])
        self.state["candidate"] = candidate
        self.state["competitors"] = competitors
        episode["proposals"].append({**deepcopy(candidate), "competitors": deepcopy(competitors)})
        episode["validation_indices"] = []
        self.state["phase"] = "validating"
        experiment_plan = {}
        decision_rows = [self.state["observations"][i] if repair is None or i != repair["observation_index"]
                         else {**self.state["observations"][i], "observation_status": "disputed"}
                         for i in episode["row_indices"]]
        self.state["controls"] = distinguishing_controls(self.spec, competitors, budget,
            seed=self.state["seed"] + 1009 * len(self.state["episodes"]) + 101 * len(episode["proposals"]),
            count=self.state["batch_size"],
            decision_rows=decision_rows,
            details=experiment_plan)
        experiment_plan["training_indices"] = list(indices)
        experiment_plan["objective_training_indices"] = list(episode["row_indices"])
        episode["proposals"][-1]["experiment_plan"] = experiment_plan
        self.state["costs"]["compute_cost"] += budget.spent
        if len(self.state["controls"]) != self.state["batch_size"]:
            self._unresolved("insufficient_experiment_planning_budget")
            return
        self._event("propose_replacement", family_id=candidate["family_id"], kind=candidate["kind"],
                    training_loss=candidate["training_loss"], training_indices=candidate["training_indices"],
                    competitors=[item["candidate_id"] for item in competitors],
                    validation_controls=deepcopy(self.state["controls"]))

    def _validate(self, context):
        episode = self.state["episodes"][-1]
        rows = [self.state["observations"][i] for i in episode["validation_indices"]]
        previous_mse = min(mean(row[key] for row in rows) for key in ("default_error", "learned_error"))
        threshold = 9 * self.state["reference_scale"]
        chosen, measured = select_competitor(self.state["competitors"], rows,
            noise_variance=self.state["reference_noise"], minimum_error=self.state["minimum_error"],
            maximum_loss=threshold)
        candidate_mse = chosen["validation_loss"] if chosen else min(
            (item["validation_loss"] for item in measured), default=float("inf"))
        training_mse = chosen["training_loss"] if chosen else self.state["candidate"]["training_loss"]
        supported = chosen is not None and previous_mse - candidate_mse > self.state["minimum_error"]
        validation = {"candidate_mse": candidate_mse, "previous_mse": previous_mse,
                      "training_mse": training_mse,
                      "chosen_candidate_id": chosen["candidate_id"] if chosen else None,
                      "competitors": measured,
                      "threshold": threshold, "supported": supported,
                      "observation_indices": list(episode["validation_indices"])}
        episode["proposals"][-1]["validation"] = validation
        self._event("validate_replacement", **validation)
        if supported:
            accepted = {**deepcopy(chosen), "scope_start": episode["scope_start"],
                        "validation_mse": candidate_mse,
                        "validation_indices": list(episode["validation_indices"])}
            self.state["accepted_revision"] = accepted
            episode["status"] = "supported"
            self.state["phase"] = "awaiting_rebase"
        elif context.get("remaining_calls", 0) >= 2 * self.state["batch_size"]:
            self.state["candidate"] = None
            self.state["competitors"] = []
            self.state["phase"] = "collecting"
            self._coverage(self.state["batch_size"])
        else:
            self._unresolved("replacement_not_supported")

    def observe(self, observation, default_mean, learned_mean, candidate_mean=None,
                noise_variance=0., context=None, candidate_predictions=None):
        context = deepcopy(context or {})
        row = {"observation": asdict(observation), "context": context,
               "default_mean": float(default_mean), "learned_mean": float(learned_mean),
               "noise_variance": max(0., float(noise_variance)),
               "default_error": (default_mean - observation.target) ** 2,
               "learned_error": (learned_mean - observation.target) ** 2}
        predictions = dict(candidate_predictions or {})
        if candidate_mean is not None and self.state["candidate"]:
            predictions.setdefault(self.state["candidate"]["candidate_id"], float(candidate_mean))
        row["candidate_predictions"] = predictions
        if candidate_mean is not None:
            row.update(candidate_mean=float(candidate_mean),
                       candidate_error=(candidate_mean - observation.target) ** 2)
        index = len(self.state["observations"])
        self.state["observations"].append(row)
        phase = self.state["phase"]
        if phase in {"applicability_probe", "applicability_confirmation"}:
            self._measure_applicability(index, context)
            return
        if phase == "confirming":
            self._confirm(index, context)
            return
        if phase in {"collecting", "validating"}:
            self.state["controls"].pop(0)
            episode = self.state["episodes"][-1]
            episode["row_indices"].append(index)
            key = "coverage_calls" if phase == "collecting" else "validation_calls"
            self.state["costs"][key] += 1
            self._event("observe_revision", purpose=phase, observation_index=index)
            if phase == "validating":
                if not predictions:
                    raise ValueError("validation needs a candidate prediction made before measurement")
                episode["validation_indices"].append(index)
            if self._monitor_research_scope(index, context):
                return
            if not self.state["controls"]:
                self._propose() if phase == "collecting" else self._validate(context)
            return
        if phase == "unresolved":
            self._monitor_research_scope(index, context)
            return
        if phase != "monitoring":
            return
        if self.state["reference_scale"] is None:
            self.state["reference_rows"].append(index)
            if len(self.state["reference_rows"]) >= self.state["batch_size"]:
                rows = [self.state["observations"][i] for i in self.state["reference_rows"]]
                error = min(mean(item[key] for item in rows) for key in ("default_error", "learned_error"))
                self.state["reference_noise"] = mean(item["noise_variance"] for item in rows)
                self.state["reference_scale"] = max(self.state["minimum_error"], error,
                                                       self.state["reference_noise"])
            return
        threshold = 9 * self.state["reference_scale"]
        row["reference_threshold"] = threshold
        self.state["pending_window"].append(index)
        if len(self.state["pending_window"]) < self.state["batch_size"]:
            self._request_confirmation(index, threshold, context)
            return
        indices = self.state["pending_window"]
        rows = [self.state["observations"][i] for i in indices]
        default_mse, learned_mse = (mean(item[key] for item in rows)
                                    for key in ("default_error", "learned_error"))
        anomalous = [i for i in indices if min(self.state["observations"][i]["default_error"],
                        self.state["observations"][i]["learned_error"]) > threshold]
        self.state["pending_window"] = indices[1:]
        # An easy control can agree with both models even after their general
        # explanation fails. Keep its neighbouring counterexamples. Requiring
        # repeated failures also prevents a single large outlier opening research.
        if min(default_mse, learned_mse) > threshold and len(anomalous) >= max(2, (len(rows) + 1) // 2):
            self.state["trigger_window"] = {"observation_indices": list(indices),
                "anomalous_indices": anomalous, "default_mse": default_mse,
                "learned_mse": learned_mse, "threshold": threshold}
            self.state["counterexamples"] = list(range(anomalous[0], index + 1))
            self._reopen(context)
        else:
            self._request_confirmation(index, threshold, context)

    def acknowledge_revision(self):
        """Called only after the optimizer installs the supported new scope."""
        if self.state["accepted_revision"] is None:
            raise ValueError("no supported replacement awaits installation")
        self._event("adopt_replacement", scope_start=self.state["accepted_revision"]["scope_start"])
        self.state["episodes"][-1]["accepted_revision"] = deepcopy(self.state["accepted_revision"])
        self.state["phase"] = "monitoring"
        self.state["candidate"] = None
        self.state["competitors"] = []
        self.state["accepted_revision"] = None
        self.state["reference_rows"] = []
        self.state["reference_scale"] = None
        self.state["reference_noise"] = None
        self.state["counterexamples"] = []
        self.state["pending_window"] = []
        self.state["trigger_window"] = None
        self.state["confirmation"] = None
        self.state["research_recovery_indices"] = []

    def to_dict(self):
        return deepcopy(self.state)
