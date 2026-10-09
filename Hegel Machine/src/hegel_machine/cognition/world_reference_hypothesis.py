"""Put a proposed estimator into the existing reference experiment procedure."""

import ast
from copy import deepcopy
import json
from math import isfinite
from pprint import pformat
import re

from .language import Program


HYPOTHESIS_INTERFACE = """
For a reference-calibration procedure exposing ESTIMATORS, propose a scientific
estimator directly with run {operator:'reference_hypothesis',
path:'shared/consolidated_method.py',hypothesis:{id:'your_hypothesis_id',
basis:[Program AST expressions],rationale:'your explanation'}}.
Each basis expression describes a feature transformation; current activated
references and one observed successful setting per previously solved target fit
its coefficients. Successful settings are operational control labels, not exact
latent resonance values or an inferred activation interval. You select the
variables and functional form. Optional fit_tolerance declares the acceptable
mean squared fitting error in Hz squared (default 1e-4); fresh physical tests
still decide success. Failure of that strict fit criterion retains an approximate
candidate for physical testing. The existing controller fits all available declared
bases and prioritizes untested feasible predictions by leave-one-crystal-out error.
These diagnostics reuse calibration data; they are not confidence intervals,
inferred activation widths or independent confirmation.
Program nodes use {op:'feature',name:'measured_feature'}, {op:'const',value:number},
or {op:'add'|'sub'|'mul'|'abs',children:[nodes with the matching arity]}.
Use the existing measured feature names, or declare measurements:{feature_name:
{instrument:'public instrument name',pattern:'regex extracting the reading',group:1}}.
Patterns read public USE messages; choose a numeric capture group. A measurement
declaration adds that instrument's ordinary acquisition for references and targets.
No source editing or new control/answer program is needed for this operation.
It retains the estimator declaration in that procedure, runs its existing fitting,
control and physical-test steps, and returns their actual predictions and outcomes.
Successful fitting alone does not establish the hypothesis. Inspect physical tests
before deciding what they support. Refuted declarations and their evidence remain;
give a revised hypothesis a new id using letters, numbers and underscores.
Other inquiry, Python and experiment tools remain available for research that
cannot be expressed by this estimator interface.
run {operator:'reference_expansion',path:'shared/consolidated_method.py'} reuses
the finite refit/composition grammar on each already measured variable and retains
new basis declarations for the same empirical prediction trials. The parent
procedure uses this step when declared candidates are exhausted. These are finite
grammar proposals, not LLM-authored hypotheses or confirmed discoveries.
After these point trials are exhausted, run {operator:'reference_response',path:
'shared/consolidated_method.py'} studies finite coefficient alternatives using
all recorded successful and failed settings. A shared connected activation band
in residual coordinates is a testable subhypothesis. The existing Boolean
experiment chooser compares fixed coefficient/band hypotheses at reachable new
settings across all measured unfinished targets, using each object's control
grid and predictions. This entry refreshes its inputs from all current public observations,
including experiments performed outside the retained procedure. When no supported
target setting is available, it can distinguish these same hypotheses on an
already measured reference using that object's public control grid. The parent
tests the selected reference and restores its activation near the previous
successful setting, recording the actual readout, before resuming target research;
a reference result does not complete the target.
If these fitted candidates have no experiment left, the same declared bases can
jointly refit their coefficients and shared response interval against all actual
positive and negative responses. Reachable target support is a candidate design
condition, not evidence of activation. The bounded search reports incompleteness;
every proposed setting still requires the existing physical test.
The parent executes the chosen test and retains its physical answer;
a failed point does not refute every parameterization of the functional form.
Finite alternatives and their disagreement are not a confidence region.
"""


def _declaration(value):
    hypothesis = deepcopy(value)
    name = hypothesis.get("id")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("hypothesis id must start with a letter and use letters, numbers or underscores")
    if not isinstance(hypothesis.get("basis"), list) or not hypothesis["basis"]:
        raise ValueError("hypothesis basis must contain at least one Program AST expression")
    for expression in hypothesis["basis"]:
        Program.from_dict(expression)
    if "fit_tolerance" in hypothesis:
        tolerance = float(hypothesis["fit_tolerance"])
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("fit_tolerance must be a finite nonnegative MSE in Hz squared")
        hypothesis["fit_tolerance"] = tolerance
    for feature, measurement in hypothesis.get("measurements", {}).items():
        if not feature or not measurement.get("instrument"):
            raise ValueError("each measurement needs a feature name and a public instrument name")
        pattern = re.compile(measurement["pattern"])
        group = measurement.get("group", 1)
        if not isinstance(group, int) or not 0 <= group <= pattern.groups:
            raise ValueError("measurement group must select an existing regex capture")
    return hypothesis


def _retain_declaration(source, hypothesis):
    assignment = next((node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "ESTIMATORS" for target in node.targets)), None)
    if assignment is None:
        raise ValueError("this procedure has no ESTIMATORS declaration; select the reference-calibration procedure")
    estimators = ast.literal_eval(assignment.value)
    previous = next((item for item in estimators if item["id"] == hypothesis["id"]), None)
    if previous is not None:
        if previous != hypothesis:
            raise ValueError("that hypothesis id already records a different declaration; use a new id for the revision")
        return source
    lines = source.splitlines(keepends=True)
    replacement = "ESTIMATORS = " + pformat((*estimators, hypothesis), width=100, sort_dicts=False) + "\n"
    lines[assignment.lineno - 1:assignment.end_lineno] = [replacement]
    return "".join(lines)


def execute_reference_hypothesis(agent, laboratory, arguments):
    """Declare a candidate and return the existing procedure's measured feedback."""
    hypothesis = _declaration(arguments["hypothesis"])
    path = arguments["path"]
    source_path = agent.workspace / path
    source = _retain_declaration(source_path.read_text(encoding="utf-8"), hypothesis)
    source_path.write_text(source, encoding="utf-8")
    agent.files[path] = source
    state_path = agent.workspace / "shared/reference_recalibration.json"
    before = json.loads(state_path.read_text()) if state_path.exists() else {}
    observation_start = len(agent.observations)
    execution = agent._run_procedure(laboratory, path, max_steps=256,
                                     query_arguments={"estimator_trial": True})
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    observations = agent.observations[observation_start:]
    result = {"operator": "reference_hypothesis", "path": path,
              "hypothesis": hypothesis, "status": execution["status"],
              "new_observation_indices": [record["index"] for record in observations],
              "new_observation_count": len(observations)}
    for key in ("tests", "fit_refutations", "out_of_range", "setting_recoveries", "response_constraints"):
        result[key] = deepcopy(state.get(key, [])[len(before.get(key, [])):])
    result["research_table"] = deepcopy(state.get("research_table", {}))
    result["prediction_selection"] = deepcopy(state.get("prediction_selection"))
    prefix = "reference_fit_" + hypothesis["id"] + "_"
    result["fitted_predictions"] = [
        {"question_id": question, **{key: deepcopy(relation[key])
            for key in ("status", "diagnosis", "selected_program", "selected_probe", "fit_evidence", "fit_rows", "empirical_prediction")
            if key in relation}}
        for question, relation in agent.relation_research.items() if question.startswith(prefix)]
    if state.get("test"):
        result["pending_test"] = deepcopy(state["test"])
    if execution.get("steps"):
        last = execution["steps"][-1].get("feedback", {})
        result.update({key: deepcopy(last[key]) for key in ("error", "reason") if key in last})
    return result
