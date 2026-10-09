"""Retain current-condition refitting alongside measured structural repair."""

import json

from .python_selective_method_learning import (
    SelectivePythonMethodLearner, selective_revision_evidence,
)
from .temporal_method_learning import METHOD_PATH


SCOPED_INTERFACE = """
Publication also needs two observed recovery choices in different practice cases.
After a separately validated refit fails fresh confirmation, one trial must recover
without buying meter readings: refit using selected current records, independently
validate, then pass fresh confirmation. Its selected records must exclude rows
before the rejected confirmation window and include a newly acquired trajectory
after that window for independent validation. In another trial, recover after the
same kind of refutation by actually fitting an instrument and response from meter
readings, then pass fresh confirmation. Use the observed evidence to choose the
branch; trial labels supply no mechanism or action schedule. Existing publication
requirements, including all listed cases, remain in force. Current-source coverage
reports these executed choices under scoped_revision.
"""


def scoped_revision_evidence(result):
    """Connect actual successful submissions to intervening fits and paid rows."""
    selective = selective_revision_evidence(result)
    steps = result.get("method_results", [])
    records = result.get("paid_observations", [])
    scoped, measured = [], []
    for recovery in selective["recoveries"]:
        rejected, confirmed = recovery["rejection_step"], recovery["recovery_step"]
        fresh = steps[rejected]["feedback"]["fresh_confirmation"]
        rejected_rows = fresh.get("observation_indices", [])
        aliases, fits, responses, instruments = {}, [], [], []
        for index, step in enumerate(steps[:confirmed + 1]):
            feedback = step["feedback"]
            evaluated = feedback.get("evaluated_model")
            if evaluated and evaluated.get("source_model") != evaluated.get("model"):
                aliases[evaluated["model"]] = aliases.get(
                    evaluated["source_model"], evaluated["source_model"])
            if not rejected < index < confirmed:
                continue
            for outcome in [feedback, *feedback.get("after_observation_results", [])]:
                operator = outcome.get("operator")
                if operator == "refit":
                    fits.append((index, outcome))
                elif operator == "instrument" and outcome.get("support_rows", 0) > 0:
                    instruments.append((index, outcome))
                elif operator == "response":
                    responses.append((index, outcome))
        submitted = aliases.get(recovery["recovery_model"], recovery["recovery_model"])
        if result["costs"]["measurement_cost"] == 0 and rejected_rows:
            for index, fit in fits:
                validation = fit.get("validation", {})
                chosen = fit.get("split", {}).get("records", [])
                held = fit.get("split", {}).get("validation_records", [])
                if (fit.get("path") != submitted or validation.get("returncode") != 0
                        or validation.get("mse") is None or validation["mse"] > 1e-10
                        or not chosen or not held):
                    continue
                selected_rows = [row for record in chosen for row in records[record]["row_indices"]]
                new_validation = [record for record in held if records[record]["row_indices"]
                    and min(records[record]["row_indices"]) > max(rejected_rows)]
                if selected_rows and min(selected_rows) >= min(rejected_rows) and new_validation:
                    scoped.append({**recovery, "refit_step": index, "refit_model": fit["path"],
                        "records": chosen, "validation_records": held,
                        "new_validation_records": new_validation,
                        "refit_validation_mse": validation["mse"],
                        "rejected_observation_indices": rejected_rows})
        if result["costs"]["measurement_cost"] > 0:
            for index, response in responses:
                source = next(((step, item) for step, item in reversed(instruments)
                    if step <= index and item.get("path") == response.get("instrument")), None)
                candidates = [item for item in response.get("candidates", [])
                    if item.get("path") == submitted and item.get("validation_mse") is not None
                    and item["validation_mse"] <= 1e-10]
                if source is not None and candidates:
                    measured.append({**recovery, "instrument_step": source[0],
                        "response_step": index, "response_model": submitted,
                        "response_validation_mse": candidates[0]["validation_mse"]})
    return {"zero_meter_scoped_recoveries": scoped, "measured_recoveries": measured,
            "evidence_scope": selective["evidence_scope"]}


class ScopedPythonMethodLearner(SelectivePythonMethodLearner):
    """Require the two demonstrated repair choices before the existing publish."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_interface += SCOPED_INTERFACE

    def _current_source_coverage(self):
        coverage = super()._current_source_coverage()
        source = (self.workspace / METHOD_PATH).read_text()
        scoped, measured = set(), set()
        for trial in self.practice_trials:
            if trial["method_source"] != source:
                continue
            if "scoped_revision" not in trial:
                result = json.loads((self.workspace / trial["directory"] / "result.json").read_text())
                trial["scoped_revision"] = scoped_revision_evidence(result)
            if trial["status"] != "completed" or trial["case"] not in coverage["confirmed_cases"]:
                continue
            if trial["scoped_revision"]["zero_meter_scoped_recoveries"]:
                scoped.add(trial["case"])
            if trial["scoped_revision"]["measured_recoveries"]:
                measured.add(trial["case"])
        coverage["scoped_revision"] = {"zero_meter_scoped_recovery_cases": sorted(scoped),
            "measured_recovery_cases": sorted(measured),
            "distinct_cases": any(a != b for a in scoped for b in measured)}
        return coverage

    def _publication_error(self, coverage):
        previous = super()._publication_error(coverage)
        if previous is not None:
            return previous
        evidence = coverage["scoped_revision"]
        if not evidence["distinct_cases"]:
            return {"error": "the current method needs zero-meter current-record refit recovery "
                "and measured instrument/response recovery after validated-refit rejection "
                "in different cases", "scoped_revision_coverage": evidence}
        return None
