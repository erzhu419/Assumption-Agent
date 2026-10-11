"""Turn persistent joint prediction failure into new, measured hypotheses."""

from copy import deepcopy
from dataclasses import asdict
from random import Random
from statistics import mean

from .language import Program
from .meta_prior_claims import compile_claims
from .models import SearchBudget
from .workspace import observation_from_dict


class JointModelRevision:
    """Research on a new evidence segment while retaining the old paid ledger.

    The reference comes from predictions made before seeing their outcomes.
    It stays fixed until a replacement passes future measurements: a model's
    increasing residual variance must not explain away its own failure.
    Coverage and validation queries consume the caller's ordinary query budget.
    A fitted candidate is frozen throughout each validation batch.
    """

    def __init__(self, spec, *, minimum_error=1e-7, batch_size=4, seed=7,
                 compute_budget=20000, state=None):
        self.spec = spec
        self.state = deepcopy(state) if state else {
            "minimum_error": minimum_error, "batch_size": batch_size,
            "seed": seed, "compute_budget": compute_budget,
            "phase": "monitoring", "reference_scale": None, "reference_rows": [],
            "observations": [], "counterexamples": [], "episodes": [], "events": [],
            "controls": [], "candidate": None, "accepted_revision": None,
            "costs": {"compute_cost": 0, "coverage_calls": 0, "validation_calls": 0}}
        if minimum_error < 0 or batch_size < 2:
            raise ValueError("joint revision needs nonnegative accuracy and at least two observations")

    @property
    def candidate(self):
        item = self.state["candidate"]
        return Program.from_dict(item["program"]) if item else None

    @property
    def accepted_revision(self):
        item = deepcopy(self.state["accepted_revision"])
        if item:
            item["program"] = Program.from_dict(item["program"])
        return item

    def next_control(self):
        return deepcopy(self.state["controls"][0]) if self.state["controls"] else None

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
                   "proposals": [], "status": "researching", "validation_indices": []}
        self.state["episodes"].append(episode)
        self._event("reopen_hypothesis_research", scope_start=start,
                    observation_indices=indices, reference_scale=self.state["reference_scale"])
        dimension = len(self.spec.feature_names)
        # Cover enough new controls to fit a general pairwise response, including
        # the already acquired counterexamples. Four later outcomes stay unseen.
        coverage = max(2 * dimension + 2, (dimension + 1) * (dimension + 2) // 2 - len(indices))
        if context.get("remaining_calls", 0) < coverage + self.state["batch_size"]:
            self._unresolved("insufficient_query_budget")
            return
        self.state["phase"] = "collecting"
        self._coverage(coverage)

    def _unresolved(self, reason):
        self.state["phase"] = "unresolved"
        self.state["episodes"][-1]["status"] = reason
        self.state["candidate"] = None
        self.state["controls"] = []
        self._event("research_unresolved", reason=reason)

    def _propose(self):
        episode = self.state["episodes"][-1]
        rows = [observation_from_dict(self.state["observations"][i]["observation"])
                for i in episode["row_indices"]]
        budget = SearchBudget(self.state["compute_budget"] - self.state["costs"]["compute_cost"])
        claims = compile_claims(self.spec, rows, budget)
        self.state["costs"]["compute_cost"] += budget.spent
        if not claims:
            self._unresolved("insufficient_compute_budget")
            return
        threshold = 9 * self.state["reference_scale"]
        supported = [item for item in claims if item["training_loss"] <= threshold]
        chosen = min(supported, key=lambda item: (item["candidate"].program.size, item["training_loss"])) \
            if supported else min(claims, key=lambda item: item["training_loss"])
        candidate = {key: deepcopy(chosen[key]) for key in
                     ("family_id", "kind", "claim", "counter_prediction", "training_loss")}
        candidate["program"] = chosen["candidate"].program.to_dict()
        candidate["training_indices"] = list(episode["row_indices"])
        self.state["candidate"] = candidate
        episode["proposals"].append(deepcopy(candidate))
        episode["validation_indices"] = []
        self.state["phase"] = "validating"
        self._coverage(self.state["batch_size"])
        self._event("propose_replacement", family_id=candidate["family_id"], kind=candidate["kind"],
                    training_loss=candidate["training_loss"], training_indices=candidate["training_indices"])

    def _validate(self, context):
        episode = self.state["episodes"][-1]
        rows = [self.state["observations"][i] for i in episode["validation_indices"]]
        candidate_mse = mean(row["candidate_error"] for row in rows)
        previous_mse = min(mean(row[key] for row in rows) for key in ("default_error", "learned_error"))
        threshold = 9 * self.state["reference_scale"]
        training_mse = self.state["candidate"]["training_loss"]
        supported = (max(training_mse, candidate_mse) <= threshold
                     and previous_mse - candidate_mse > self.state["minimum_error"])
        validation = {"candidate_mse": candidate_mse, "previous_mse": previous_mse,
                      "training_mse": training_mse,
                      "threshold": threshold, "supported": supported,
                      "observation_indices": list(episode["validation_indices"])}
        episode["proposals"][-1]["validation"] = validation
        self._event("validate_replacement", **validation)
        if supported:
            accepted = {**deepcopy(self.state["candidate"]), "scope_start": episode["scope_start"],
                        "validation_mse": candidate_mse,
                        "validation_indices": list(episode["validation_indices"])}
            self.state["accepted_revision"] = accepted
            episode["status"] = "supported"
            self.state["phase"] = "awaiting_rebase"
        elif context.get("remaining_calls", 0) >= 2 * self.state["batch_size"]:
            self.state["candidate"] = None
            self.state["phase"] = "collecting"
            self._coverage(self.state["batch_size"])
        else:
            self._unresolved("replacement_not_supported")

    def observe(self, observation, default_mean, learned_mean, candidate_mean=None,
                noise_variance=0., context=None):
        context = deepcopy(context or {})
        row = {"observation": asdict(observation), "context": context,
               "default_mean": float(default_mean), "learned_mean": float(learned_mean),
               "noise_variance": max(0., float(noise_variance)),
               "default_error": (default_mean - observation.target) ** 2,
               "learned_error": (learned_mean - observation.target) ** 2}
        if candidate_mean is not None:
            row.update(candidate_mean=float(candidate_mean),
                       candidate_error=(candidate_mean - observation.target) ** 2)
        index = len(self.state["observations"])
        self.state["observations"].append(row)
        phase = self.state["phase"]
        if phase in {"collecting", "validating"}:
            self.state["controls"].pop(0)
            episode = self.state["episodes"][-1]
            episode["row_indices"].append(index)
            key = "coverage_calls" if phase == "collecting" else "validation_calls"
            self.state["costs"][key] += 1
            self._event("observe_revision", purpose=phase, observation_index=index)
            if phase == "validating":
                if candidate_mean is None:
                    raise ValueError("validation needs a candidate prediction made before measurement")
                episode["validation_indices"].append(index)
            if not self.state["controls"]:
                self._propose() if phase == "collecting" else self._validate(context)
            return
        if phase != "monitoring":
            return
        if self.state["reference_scale"] is None:
            self.state["reference_rows"].append(index)
            if len(self.state["reference_rows"]) >= self.state["batch_size"]:
                rows = [self.state["observations"][i] for i in self.state["reference_rows"]]
                error = min(mean(item[key] for item in rows) for key in ("default_error", "learned_error"))
                self.state["reference_scale"] = max(self.state["minimum_error"], error,
                                                       mean(item["noise_variance"] for item in rows))
            return
        threshold = 9 * self.state["reference_scale"]
        row["reference_threshold"] = threshold
        if min(row["default_error"], row["learned_error"]) > threshold:
            self.state["counterexamples"].append(index)
            if len(self.state["counterexamples"]) >= self.state["batch_size"]:
                self._reopen(context)
        else:
            self.state["counterexamples"] = []

    def acknowledge_revision(self):
        """Called only after the optimizer installs the supported new scope."""
        if self.state["accepted_revision"] is None:
            raise ValueError("no supported replacement awaits installation")
        self._event("adopt_replacement", scope_start=self.state["accepted_revision"]["scope_start"])
        self.state["episodes"][-1]["accepted_revision"] = deepcopy(self.state["accepted_revision"])
        self.state["phase"] = "monitoring"
        self.state["candidate"] = None
        self.state["accepted_revision"] = None
        self.state["reference_rows"] = []
        self.state["reference_scale"] = None
        self.state["counterexamples"] = []

    def to_dict(self):
        return deepcopy(self.state)
