"""Compare an objective GP replacement before installing it.

Constraint scope changes propose a new objective explanation; their validation
does not establish the proposed objective model's prediction advantage. This
state records forecasts on already purchased controls for that comparison.
"""

from copy import deepcopy
from math import comb

from .validation_continuation import evidence_summary


def paired_replacement_evidence(validation):
    """Allow consistent improvement across controls with unequal error scales.

    The exact one-sided sign tail supplements the existing mean-margin rule.
    Adaptive controls and repeated decisions make this a selection heuristic,
    not a calibrated confidence statement.
    """
    evidence = evidence_summary(validation)
    gains = [row["previous_error"] - row["candidate_error"] for row in validation]
    wins = sum(gain > 1e-7 for gain in gains)
    losses = sum(gain < -1e-7 for gain in gains)
    non_ties = wins + losses
    probability = (sum(comb(non_ties, k) for k in range(wins, non_ties + 1))
                   / 2 ** non_ties if non_ties else 1.)
    basis = "original_mean_margin" if evidence["decision"] == "supported" else None
    if (basis is None and evidence["count"] >= 4
            and evidence["mean_gain"] > 1e-7 and probability <= .05):
        evidence["decision"] = "supported"
        basis = "consistent_improvement"
    return {**evidence, "wins": wins, "losses": losses,
            "ties": len(gains) - non_ties, "sign_probability": probability,
            "support_basis": basis}


class ObjectiveReplacement:
    def __init__(self, state=None):
        self.state = deepcopy(state) if state is not None else {
            "scope_start": None, "confirmed": False, "validation": [],
            "pending": None, "attempts": [], "events": [], "decisions": [],
            "comparison_rows": 0}

    def _current(self):
        return {name: deepcopy(self.state[name]) for name in
                ("scope_start", "confirmed", "validation", "pending")}

    def propose(self, scope_start, stage):
        scope_start = int(scope_start)
        if self.state["scope_start"] == scope_start:
            return self._current()
        if self.state["scope_start"] is not None:
            self.finish(stage, "proposal_scope_changed")
        self.state["scope_start"] = scope_start
        self.state["events"].append({"action": "propose_objective_replacement",
                                     "scope_start": scope_start, "stage": int(stage)})
        return self._current()

    def prepare(self, features, incumbent_mean, replacement_mean, *, stage,
                candidate=None):
        if self.state["scope_start"] is None or any(
                row["features"] == features for row in self.state["validation"]):
            return None
        request = {"features": deepcopy(features), "stage": int(stage),
                   "incumbent_mean": float(incumbent_mean),
                   "replacement_mean": float(replacement_mean)}
        if candidate is not None:
            request["candidate"] = {"tag": list(candidate["tag"]),
                                    "mean": float(candidate["mean"])}
        self.state["pending"] = request
        return deepcopy(request)

    def observe(self, features, target, *, global_index):
        request = self.state["pending"]
        if (request is None or request["features"] != features
                or request["stage"] != int(global_index)):
            return None
        row = {**deepcopy(request), "global_index": int(global_index),
               "target": float(target),
               "incumbent_error": (request["incumbent_mean"] - target) ** 2,
               "replacement_error": (request["replacement_mean"] - target) ** 2}
        if "candidate" in row:
            row["candidate"]["error"] = (row["candidate"]["mean"] - target) ** 2
        self.state["validation"].append(row)
        self.state["pending"] = None
        return deepcopy(row)

    def confirm(self, scope_start, stage):
        self.propose(scope_start, stage)
        if not self.state["confirmed"]:
            self.state["confirmed"] = True
            self.state["events"].append({"action": "confirm_objective_replacement",
                "scope_start": int(scope_start), "stage": int(stage)})
        return self._current()

    def comparisons(self, tag):
        """Return only forecasts made for this frozen goal candidate."""
        rows = []
        for row in self.state["validation"]:
            candidate = row.get("candidate")
            if candidate is not None and tuple(candidate["tag"]) == tuple(tag):
                rows.append({"features": deepcopy(row["features"]),
                    "global_index": row["global_index"], "target": row["target"],
                    "candidate_error": candidate["error"],
                    "previous_error": row["replacement_error"]})
        return rows

    def decide(self, *, stage, accepted_candidate=None):
        self.state["comparison_rows"] += len(self.state["validation"])
        evidence = paired_replacement_evidence([
            {"candidate_error": row["replacement_error"],
             "previous_error": row["incumbent_error"]}
            for row in self.state["validation"]])
        candidate_evidence = None
        if accepted_candidate is not None:
            comparisons = self.comparisons(accepted_candidate)
            self.state["comparison_rows"] += len(comparisons)
            candidate_evidence = paired_replacement_evidence([
                {"candidate_error": row["previous_error"],
                 "previous_error": row["candidate_error"]}
                for row in comparisons])
        decision = {"scope_start": self.state["scope_start"], "stage": int(stage),
            "confirmed": self.state["confirmed"], "evidence": evidence,
            "candidate_evidence": candidate_evidence,
            "replace": (self.state["confirmed"] and evidence["decision"] == "supported"
                        and (candidate_evidence is None
                             or candidate_evidence["decision"] == "supported"))}
        if accepted_candidate is not None:
            decision["accepted_candidate"] = list(accepted_candidate)
        # Keep one decision per observed comparison, rather than one per poll.
        previous = self.state["decisions"][-1] if self.state["decisions"] else None
        if previous is None or any(decision.get(key) != previous.get(key)
                                  for key in decision if key != "stage"):
            self.state["decisions"].append(deepcopy(decision))
        return decision

    def finish(self, stage, reason):
        if self.state["scope_start"] is None:
            return None
        attempt = {**self._current(), "stage": int(stage), "reason": reason}
        self.state["attempts"].append(attempt)
        self.state["events"].append({"action": "finish_objective_replacement",
            "scope_start": self.state["scope_start"], "stage": int(stage),
            "reason": reason})
        self.state.update(scope_start=None, confirmed=False, validation=[], pending=None)
        return deepcopy(attempt)

    def to_dict(self):
        return deepcopy(self.state)
