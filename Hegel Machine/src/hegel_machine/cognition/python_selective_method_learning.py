"""Keep selective refitting while learning to recover from its real counterexamples."""

import json

from .python_method_learning import PythonArchiveMethodLearner
from .temporal_method_learning import METHOD_PATH


SELECTIVE_INTERFACE = """
Publication retains the existing confirmation, unchanged-prior reuse and revision
requirements for every listed case. The same current program must also demonstrate
a separately validated refit that passes fresh confirmation without buying meter
readings in one case, and recovery in another case after a separately validated
refit is rejected by fresh confirmation. Recovery requires a later successful
fresh confirmation in that same trial. These requirements use actual execution
feedback; printed claims and successful predictions alone do not establish them.
Current-source coverage includes selective_revision with the observed cases and
the missing behavior. No case identity or repair schedule is supplied.
"""


def selective_revision_evidence(result):
    """Read executed validation/submission order, including query continuations."""
    refits, aliases, rejections, successes, recoveries = {}, {}, [], [], []
    for index, entry in enumerate(result.get("method_results", [])):
        feedback = entry.get("feedback", {})
        proposal = entry.get("proposal") or {}
        for outcome in [feedback, *feedback.get("after_observation_results", [])]:
            validation = outcome.get("validation", {})
            if (outcome.get("operator") == "refit" and outcome.get("path")
                    and validation.get("returncode") == 0
                    and validation.get("mse") is not None and validation["mse"] <= 1e-10):
                refits[outcome["path"]] = {"validation_step": index, "model": outcome["path"],
                    "validation_mse": validation["mse"]}
        evaluated = feedback.get("evaluated_model")
        if evaluated:
            original = evaluated.get("source_model")
            alias = evaluated["model"]
            if original in refits:
                aliases[alias] = original
            elif alias != original:
                aliases.pop(alias, None)
        fresh = feedback.get("fresh_confirmation")
        if not fresh:
            continue
        declared = feedback.get("program_action") or proposal
        submitted = (feedback.get("method_submission") or declared.get("model")
                     or (declared.get("arguments", {}).get("model")
                         if declared.get("action") == "finish" else None))
        model = aliases.get(submitted, submitted)
        validated = refits.get(model)
        if validated is not None and fresh.get("passed") is False:
            rejections.append({**validated, "rejection_step": index,
                "submitted_model": submitted, "fresh_mse": fresh.get("mse")})
        if fresh.get("passed") is True:
            if validated is not None:
                successes.append({**validated, "confirmation_step": index,
                    "submitted_model": submitted, "fresh_mse": fresh.get("mse")})
            recoveries.extend({**failure, "recovery_step": index, "recovery_model": submitted,
                "recovery_mse": fresh.get("mse")} for failure in rejections
                if failure["rejection_step"] < index)
    return {"zero_meter_refit_confirmations": successes
                if result.get("costs", {}).get("measurement_cost") == 0 else [],
            "validated_refit_rejections": rejections, "recoveries": recoveries,
            "evidence_scope": "Chronological execution feedback and submitted paths; final file contents "
                              "do not reconstruct model versions before later overwrites."}


class SelectivePythonMethodLearner(PythonArchiveMethodLearner):
    """Add observed selective-refit/recovery coverage before the existing publish."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_interface += SELECTIVE_INTERFACE

    def _current_source_coverage(self):
        coverage = super()._current_source_coverage()
        source = (self.workspace / METHOD_PATH).read_text()
        selective, recovered = set(), set()
        for trial in self.practice_trials:
            if trial["method_source"] != source:
                continue
            if "selective_revision" not in trial:
                result = json.loads((self.workspace / trial["directory"] / "result.json").read_text())
                trial["selective_revision"] = selective_revision_evidence(result)
            if trial["status"] != "completed" or trial["case"] not in coverage["confirmed_cases"]:
                continue
            evidence = trial["selective_revision"]
            if evidence["zero_meter_refit_confirmations"]:
                selective.add(trial["case"])
            if evidence["recoveries"]:
                recovered.add(trial["case"])
        coverage["selective_revision"] = {"zero_meter_refit_cases": sorted(selective),
            "validated_refit_recovery_cases": sorted(recovered),
            "distinct_cases": any(a != b for a in selective for b in recovered)}
        return coverage

    def _publication_error(self, coverage):
        evidence = coverage["selective_revision"]
        if not evidence["distinct_cases"]:
            return {"error": "the current method still needs a confirmed zero-meter refit and "
                "recovery after a validated refit's fresh rejection in a different case",
                "selective_revision_coverage": evidence}
        return None
