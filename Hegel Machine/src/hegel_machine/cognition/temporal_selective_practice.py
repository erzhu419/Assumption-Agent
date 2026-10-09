"""Learn selective repair, measured from paid research decisions and confirmation."""

import json

from .temporal_method_learning import METHOD_PATH
from .temporal_paired_practice import PairedMethodLearner


SELECTIVE_OBJECTIVE = """
ADDITIONAL EMPIRICAL OBJECTIVE: selective repair after a parent prediction fails.
Demonstrate both of these outcomes in distinct practice cases with the CURRENT
source, in addition to the prediction and unchanged-predictor reuse requirements:
* Newly purchased meter evidence after failed fresh prediction supports the
  unchanged inherited instrument; retain it, revise the response, and freshly
  confirm the selected response predictor without instrument fitting in that repair.
* Such evidence refutes the inherited instrument; reopen instrument research,
  use the resulting fitted instrument in the response, and freshly confirm the
  selected response predictor.
The old instrument is memory['research_roles']['instrument']. Earlier support
does not establish its applicability after a later prediction failure. Assessment
uses its unchanged numerical hypotheses, not a newly fitted replacement. A
decision to retain or reopen must follow the measured assessment in a later
procedure invocation: a linear continuation cannot branch on an operator result
it has not yet seen. Computing a response candidate together with the assessment
is allowed; selecting that candidate still follows the observed assessment.
Practice feedback reports these actual choices and current-source coverage.
Choose the experiments, evidence subsets, numerical conditions and repair steps
yourself. Case identities supply no failure diagnosis or experiment schedule.
"""


def selective_revision_evidence(result, instrument_path):
    """Identify confirmed repair paths without task labels or latent settings.

    Uses chronological continuation results and global paid-row indices. File
    status updates from assessment are not instrument fitting. The last export
    of the selected response is used because output paths may be reused.
    """
    methods = result.get("method_results", [])
    failures = [index for index, entry in enumerate(methods)
                if entry.get("feedback", {}).get("fresh_confirmation", {}).get("passed") is False]
    evidence = {"retained_instrument": False, "reopened_instrument": False,
                "parent_failure_steps": failures, "post_failure_assessments": [],
                "selected_response": None, "reason": "no_failed_parent_confirmation"}
    if not failures:
        return evidence
    # These tasks settle after one disclosed checkpoint. A later rejected
    # response candidate does not invalidate measurements of that same apparatus.
    failed_step = failures[0]
    failed_rows = methods[failed_step]["feedback"]["fresh_confirmation"].get("observation_indices", [])
    if not failed_rows:
        evidence["reason"] = "failed_confirmation_has_no_paid_row_indices"
        return evidence
    cutoff = max(failed_rows)
    records = result.get("paid_observations", [])

    def current_meter_records(indices):
        measured = [records[index] for index in indices
                    if any("meter" in reading["values"]
                           for reading in records[index].get("measurements", []))]
        return bool(measured) and all(record.get("row_indices")
            and min(record["row_indices"]) > cutoff for record in measured)

    operators = [(step, order, outcome) for step, entry in enumerate(methods)
                 for order, outcome in enumerate(entry.get("feedback", {}).get("after_observation_results", []))
                 if "error" not in outcome]
    prior_overwritten = None
    assessments = []
    for step, order, outcome in operators:
        if outcome["operator"] == "instrument" and outcome.get("path") == instrument_path:
            prior_overwritten = (step, order)
        if step <= failed_step or outcome["operator"] != "assess_instrument":
            continue
        if outcome.get("path") != instrument_path or prior_overwritten is not None:
            continue
        indices = outcome.get("records", [])
        if not current_meter_records(indices):
            continue
        status = ("supported" if "meter" in outcome.get("supported_fields", [])
                  and not any("meter" in outcome.get(key, [])
                              for key in ("refuted_fields", "ambiguous_fields", "unassessed_fields"))
                  else "refuted" if "meter" in outcome.get("refuted_fields", []) else "unresolved")
        assessed = {"step": step, "order": order, "records": indices, "outcome": status}
        evidence["post_failure_assessments"].append(assessed)
        assessments.append(assessed)
    selected = result.get("model")
    responses = [(step, order, outcome) for step, order, outcome in operators
                 if outcome["operator"] == "response"
                 and any(candidate.get("path") == selected for candidate in outcome.get("candidates", []))]
    confirmed = [step for step, entry in enumerate(methods)
                 if entry.get("feedback", {}).get("fresh_confirmation", {}).get("passed") is True
                 and entry.get("feedback", {}).get("method_submission") == selected]
    if (result.get("status") != "completed" or not confirmed or not responses
            or not result.get("fresh_confirmation", {}).get("passed")):
        evidence["reason"] = "no_freshly_confirmed_response_export"
        return evidence
    response_step, response_order, response = responses[-1]
    evidence["selected_response"] = {"step": response_step,
                                     "instrument": response.get("instrument"), "model": selected}
    if not failed_step < response_step < confirmed[-1]:
        evidence["reason"] = "response_not_revised_after_parent_failure"
        return evidence
    fits = [(step, order, outcome) for step, order, outcome in operators
            if outcome["operator"] == "instrument" and failed_step < step < confirmed[-1]]
    for assessment in assessments:
        assessment_step = assessment["step"]
        if not (assessment_step, assessment["order"]) < (response_step, response_order):
            continue
        if (assessment["outcome"] == "supported" and not fits
                and response.get("instrument") == instrument_path):
            evidence["retained_instrument"] = True
        if assessment["outcome"] == "refuted":
            # The response must use the last actual fit of its instrument path.
            matching = [(step, order, outcome) for step, order, outcome in fits
                        if (step, order) < (response_step, response_order)
                        and outcome.get("path") == response.get("instrument")]
            if matching:
                fit_step, _, fit = matching[-1]
                if fit_step > assessment_step and current_meter_records(fit.get("records", [])):
                    evidence["reopened_instrument"] = True
    evidence["reason"] = ("confirmed_selective_repair" if evidence["retained_instrument"]
                          or evidence["reopened_instrument"] else "no_confirmed_evidence_dependent_choice")
    return evidence


class SelectiveMethodLearner(PairedMethodLearner):
    """Require both measured retention and reopening in the practiced source."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_interface += SELECTIVE_OBJECTIVE

    def _current_source_coverage(self):
        coverage = super()._current_source_coverage()
        source = (self.workspace / METHOD_PATH).read_text()
        current = [trial for trial in self.practice_trials
                   if trial["method_source"] == source and trial["status"] == "completed"]
        for branch in ("retained_instrument", "reopened_instrument"):
            coverage[branch + "_cases"] = sorted({trial["case"] for trial in current
                if trial.get("selective_revision", {}).get(branch)})
        if "resume_context.json" in self.files:
            context = json.loads(self.files["resume_context.json"])
            context["current_source_coverage"] = coverage
            self.files["resume_context.json"] = json.dumps(context, indent=2)
            (self.workspace / "resume_context.json").write_text(self.files["resume_context.json"])
        return coverage

    def _practice(self, case):
        feedback = super()._practice(case)
        if "directory" in feedback:
            result = json.loads((self.workspace / feedback["directory"] / "result.json").read_text())
            decision = selective_revision_evidence(result, self.knowledge["research_roles"]["instrument"])
            self.practice_trials[-1]["selective_revision"] = decision
            feedback["selective_revision"] = decision
            feedback["current_source_coverage"] = self._current_source_coverage()
        return feedback

    def _execute(self, action, arguments, laboratory):
        if action == "finish":
            coverage = self._current_source_coverage()
            if not coverage["missing_cases"]:
                retained = coverage["retained_instrument_cases"]
                reopened = coverage["reopened_instrument_cases"]
                if not any(a != b for a in retained for b in reopened):
                    return {"error": "Current predictions pass, but selective repair still needs both "
                            "confirmed retention of a currently supported old instrument and confirmed "
                            "reopening after its measured refutation in distinct cases.",
                            "current_source_coverage": coverage}
        return super()._execute(action, arguments, laboratory)
