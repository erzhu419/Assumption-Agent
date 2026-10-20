"""Shared, paid-data research operators for the temporal LLM comparison.

The model chooses numerical steps or invokes the existing complete researcher.
Both paths use the current paid observations and the same resource account.
"""

import json
from copy import deepcopy
from math import nextafter
from pathlib import Path
from time import monotonic

from .code_agent import _bound_model_contexts, _copy_model_context, evaluate_python_model, python_model_context
from .history_revision import revise_history
from .knowledge_evolution import bind_answers
from .language import Program
from .online_inquiry_llm import InquiryLLMAgent
from .program_experiment_methods import ProgramExperimentMethods
from .temporal_instrument_research import (
    assess_temporal_instrument, fit_temporal_instrument,
    observe_temporal_instrument_method, select_temporal_instrument_probe,
)
from .temporal_research import fit_temporal_response, mse
from .temporal_response_experiments import plan_response_experiment, assess_response_experiment


OPERATOR_INTERFACE = """
All knowledge conditions additionally share these optional numerical operators.
Use action run with an operator key instead of Python code/path execution:
run {operator:'research_choice',hypotheses:'hypotheses.json',instrument:'instrument.json'}
compares the supplied response programs against paid evidence and checks their
instrument. It uses the existing research-action policy to choose a supported
response for confirmation, a distinguishing experiment, calibration repair, or
response revision. Multiple candidates fitting all-zero data remain ambiguous.
The returned experiment/action can be purchased and assessed without replanning.
Optional records:[0,1,2] selects the current evidence when a mechanism has changed.
For ordinary Python predictors use research_choice {models:['model.py',...]}.
It evaluates these executable candidates on current paid observations, preserves
the better evaluated snapshot and asks the existing policy whether to confirm it
or investigate its failure. A failed candidate can receive an executable refit
proposal; failed trajectory validation leads to existing instrument measurement,
assessment and response revision. Call research_choice again after executing its
next_action to use the new evidence. Validated refit/response exports are assessed
automatically by this comparison; acquisitions and finish remain your decisions.
research_choice returns next_action at the top level of its result. In contrast,
ordinary evaluation feedback can wrap the same advice in research_advice.
Evaluation and research feedback can include research_advice with a concrete
next_action from this same policy. It uses the evidence already obtained, without
executing its recommendation. You may follow it, choose another action, or call
research_choice explicitly to compare candidates.
An unresolved instrument investigation can remain in research_advice after other
Python actions, and in research_state.json, until new evidence changes the next
step or a supported alternative response is ready for confirmation.
Omit models and hypotheses to reassess the saved candidate after new observations.
This Python form uses all current paid rows. It does not parse printed fit errors.
Exported history/response paths are candidates, not confirmed explanations; write
an executable predictor and evaluate it before abandoning a promising numerical fit.
The returned next_action is advice; you still decide whether to execute finish.
run {operator:'refit',model:'archived/model.py',path:'shared/refitted'} keeps the
same Python source and dependencies, and calls its existing fit(observations)
or explicit-representation fit_features(features,targets) on training trajectories
from current paid evidence. Optional records:[3,4,5]
restricts both validation and the exported fitting view to those complete paid
laboratory records. Omit records to use all paid records. It scores separate complete
validation trajectories before making the refitted candidate available. Optional
validation_records selects whole laboratory records within that set. path names
a new directory; the returned predictor path includes its isolated fitting context.
The original predictor is unchanged. Both usual observation files contain only
training rows during validation. Afterwards the candidate's fitting view contains
all selected paid rows for a later evaluate or finish that you choose explicitly.
Measured local relations bound into fitting must not disclose those validation
labels. If they do, choose training-only relation checks or different validation
trajectories; ordinary all-paid evaluation and fresh confirmation remain available.
No callable fit means refit_unavailable, not a rejected model family. A failed
validation tests this fitting procedure on these trajectories; it does not prove
that every parameter setting or every model in its family is inadequate. Use
the existing history, instrument, response or experiment tools when more research
is warranted. Refitting charges training plus validation rows as computation
units and records Python execution time; those units are not numerical FLOPs.
run {operator:'legacy_capabilities',version:'0.93'} retrieves a historical
capability's current callable entry points; group:'scheduling' selects a family.
run {operator:'legacy_capabilities',query:'refit',compact:true} finds related
historical work and its current reuse routes without knowing a version number.
Queries match all whitespace-separated terms in capability metadata or linked
route titles and keywords. Results retain adoption status, evidence and remaining
integration gaps. Compact history entries are the default; compact:false includes
each version's full module and callable list. Search before implementing a solver
that the retained history may already provide.
run {operator:'legacy_research'} executes the existing UnifiedResearchAgent on
this current laboratory using the same paid observations and remaining budget.
It can acquire further observations and revise its concepts, instrument models,
methods and applicability through the original research loop. Full knowledge is
stored in shared/legacy_knowledge.json for subsequent tasks; the returned report
identifies its measured outcome. Its top-level path is the exported predictor,
or null when export is unavailable; export_reason explains why. You still select a
predictor and finish for this interface's fresh confirmation. Other task families
and cross-task scheduling retain the native entry points listed in the catalogue.
run {operator:'history',base_lags:2,path:'model.py'} compares complete quadratic
control-history bases with base_lags and base_lags+1 past controls. It reports
separate trajectory validation and exports the selected basis fitted on all
paid responses. Its top-level path names the exported predictor. Unsupported
extension keeps the base; this is not confirmation.
run {operator:'instrument',path:'instrument.json'} fits competing bounded
finite-history polynomial and continuous-state explanations of purchased meter
readings only. It exports all fitted hypotheses, not a known apparatus formula.
run {operator:'response',instrument:'instrument.json',output_dir:'shared/candidates'}
fits quadratic response functions of current control, each surviving learned
meter coordinate, and its previous value. It returns EVERY candidate with its
validation error and Python path, without choosing a candidate. Each export is
then refitted on all paid responses and frozen; reported validation precedes
that refit. You decide whether any candidate warrants evaluate or finish.
The result also gives hypotheses_path: a JSON list of the same frozen candidate
programs, retaining their executable structure for further experiment planning.
stored_instrument_status refers to evidence saved in the named instrument file.
Current response validation does not test current meter readings or renew that
instrument status on the current task.
The response operator also returns paid_meter_consistency: residuals of the
unchanged instrument hypotheses against already purchased current meter readings.
These separate diagnostics consume candidate-row scoring units and identify the
measured laboratory records, MSE and largest squared error. They may include
instrument-fitting evidence and are not fresh confirmation. They never update
stored support or select a response candidate. Without readings or sufficient
remaining scoring budget, this diagnostic reports unavailable.
history and response accept optional validation_records:[2,3], zero-based indices
in laboratory_observations.json, never row_indices. By default, the last two
among the longest paid trajectories are validation; the others are training.
When selecting both longest records would leave only shorter training records,
one long record stays in training and a shorter one takes its validation place.
Records are never split across training and validation. Results identify both
sets. Identical paid reset records stay in one split, with their acquisition costs
preserved. Validation status uses the task's confirmation MSE tolerance. These are
paid-data comparisons, not fresh confirmation.
instrument and response also accept records:[2,3,4] to restrict fitting to those
complete paid laboratory records, retaining the rest of the observation history.
For response, training, validation, the final refit and paid_meter_consistency all
use only this selected set; validation_records must be a subset of records.
Omitting records uses all paid records as before. Returned record indices always
refer to the original laboratory_observations.json, not positions in the subset.
run {operator:'disagreement',instrument:'instrument.json'} proposes a trajectory
separating current instrument hypotheses. It does NOT acquire it; use query if
you choose to purchase it. Optional max_length and serial control its search.
It first replays and, when needed, extends measured procedures from
shared/instrument_methods.json, using the existing experiment-method library.
This ordinary JSON file has exactly ProgramExperimentMethods.to_dict() format;
an existing legacy library's program_experiment_methods value can be used directly.
The result includes an experiment path for the proposed action, plus any method_id
or parent_method_id. Planning alone establishes no measured method support.
run {operator:'assess_instrument',instrument:'instrument.json',records:[9]}
assesses the frozen instrument hypotheses using meter readings in the selected
already purchased records and updates their support/refutation status. It does
not refit them. Keep the distinction between fitting evidence and new evidence.
To update the experiment-method book too, add experiment:the_returned_path and
select the single new meter trajectory bought from that plan. This returns
method_feedback and saves measured reuse, a supported derived procedure, or an
applicability failure through the existing library. The action and instrument
hypotheses must still match the plan. Each plan records its evidence once.
The method file remains available to later calls and ordinary shared-file transfer;
experiment plans are task-local. All arms have the same operators and file access.
run {operator:'response_disagreement',hypotheses:'shared/candidates/hypotheses.json'}
uses the existing response-competition planner and the SAME experiment-method book.
The hypotheses file is a list of {hypothesis_id,program:Program.to_dict(),path:optional}
entries, including the native hypotheses lists of earlier research plans. For laws
written in a measured meter coordinate, optional instrument:'instrument.json'
provides its uniquely supported calibration; actual meter readings are still bought.
Without a supported calibration, the old planner can request paired measurements.
For controls-only candidates no extra meter readings are required. Optional length
and serial are passed to the old planner. Returned action/experiment/forecasts do
not buy data or choose an answer. The current workspace book can contain both
instrument and response methods; no second method library is created.
run {operator:'assess_response',experiment:the_returned_path,records:[new_record]}
assesses those frozen response hypotheses and calibration on the corresponding
newly purchased trajectory. It returns separate response and calibration losses,
candidate statuses and method_feedback, and updates that same method book.
It neither refits nor adopts a candidate; select a model and use finish for fresh
confirmation. A failed calibration and a failed response law remain distinct.
The methods use only paid records and public capabilities. Their fitting,
scoring and Python time consume the common limits. The individual numerical
operators do not buy data; legacy_research can purchase through this laboratory.
You remain responsible for the manifest and final submission.
Their output files are ordinary editable files, available equally in all arms.
You can explicitly compose acquisition with these same operators in one query:
query {actions:[trajectory],after_observation:[
  {operator:'assess_instrument',instrument:'shared/meter.json',records:'acquired'},
  {operator:'response',instrument:'shared/meter.json',output_dir:'shared/revised'}]}
A select_actions program may return the same after_observation list alongside
its actions. An explicit query argument overrides the program's list. Each
declared operator runs in order after the whole acquisition succeeds; records:
'acquired' means the exact newly bought laboratory record indices. Operators
see those new paid labels. Their returned candidates still require your choice.
To branch on actual earlier operator feedback, a program may read research_state.json:
{'operator_results':[{'index':0,'arguments':{...},'operator':'response','candidates':[...]},...],
 'limits':{...},'costs':{...}}. It is refreshed immediately before invocation and
includes failed operator results with error at that same level. Each entry uses
the same flat feedback shape as after_observation_results, plus its chronological
index. memory remains the inherited snapshot.
On already-paid data, a temporal program can return {'actions':[],
'after_observation':[operator arguments]} for analysis without buying more rows.
These operators still consume compute and execution time. Both lists empty is
invalid; records:'acquired' resolves to [] for an analysis-only invocation and
never means all earlier records. Use explicit paid record indices when needed.
Feedback contains authored steps, their origin (arguments or program path),
resolved arguments, and numerical results. A failed operator stops the list;
successful purchases and earlier results remain available. Failed acquisition
runs no continuation. A live question retains the actual continuation trace
without being automatically answered. This syntax runs only the declared
shared operators, not another acquisition or finish.
"""


INSTRUMENT_METHODS_PATH = "shared/instrument_methods.json"


LEGACY_SESSION_INTERFACE = """
This workspace is joined to a paid multi-task research session. The shared
library contains actual earlier research, and unresolved tasks keep their paid
observations. research_state.json also lists session task status and pending
revisits.
run {operator:'legacy_revisit',max_tasks:1} lets the existing session scheduler
select and revisit a task using successor knowledge. Its observations and
computation use that task's remaining limits; screening uses the controller's
limits. The result identifies revisited tasks and the current task's predictor,
if exportable. Use finish for independent confirmation of a selected predictor.
run {operator:'legacy_return_confirmation',length:24} buys a new public trajectory
for this task and compares the frozen pre-revisit and post-revisit explanations.
Required instrument readings are purchased as well. Measured improvement or
failure updates the original knowledge source and its producing research method.
This validates a source's contribution; it does not submit a predictor or replace
finish. An explanation requiring meter readings can have supported source value
while still lacking a controls-only predictor.
"""


_PROGRAM_RUNNER = '''
def _eval(node, features, history):
    op, value, name, lag, children = node
    if op == "const":
        return float(value)
    if op == "feature":
        return float(features[name])
    if op == "available":
        return float(len(history) >= lag)
    if op == "lag":
        if len(history) < lag:
            raise ValueError("insufficient control history")
        return float(history[-lag][name])
    if op == "if":
        return _eval(children[1 if _eval(children[0], features, history) else 2], features, history)
    if op == "object_sum":
        return sum(_eval(child, features, history) for child in children)
    if op == "accumulate":
        state = _eval(children[1], features, history)
        for index, point in enumerate(history):
            state = value * state + _eval(children[0], point, history[:index])
        return value * state + _eval(children[0], features, history)
    if op == "event_delta":
        if len(history) < lag:
            return 0.0
        index = len(history) - lag
        return (_eval(children[0], features, history)
                - _eval(children[0], history[index], history[:index]))
    if op == "window_sum":
        start = max(0, len(history) - lag + 1)
        return _eval(children[0], features, history) + sum(
            _eval(children[0], history[index], history[:index])
            for index in range(start, len(history)))
    left = _eval(children[0], features, history)
    if op == "abs":
        return abs(left)
    if op == "not":
        return float(not left)
    right = _eval(children[1], features, history)
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "min":
        return min(left, right)
    if op == "max":
        return max(left, right)
    if op == "ge":
        return float(left >= right)
    if op == "lt":
        return float(left < right)
    if op == "and":
        return float(bool(left) and bool(right))
    if op == "or":
        return float(bool(left) or bool(right))
    raise ValueError("unsupported operator: " + op)

def predict(row):
    features = row["features"] if isinstance(row, dict) else row.features
    history = row["history"] if isinstance(row, dict) else row.history
    return float(_eval(_PROGRAM, features, history))
'''


def export_temporal_program(program, *, feature_names=("u",)):
    """Export the learned control-history program without repository imports.

    Definition calls are expanded before export. Only the scalar and temporal
    operators used by the instrument/readout repertoire are implemented here.
    The exported predictor has no fitting hook and never reads response labels.
    Callers adapting the same language to another control schema can explicitly
    supply its feature names; the original interface remains restricted to u.
    """
    supported = {"const", "feature", "lag", "available", "if", "object_sum",
                 "accumulate", "event_delta", "window_sum", "abs", "not",
                 "add", "sub", "mul", "min", "max", "ge", "lt", "and", "or"}

    def encode(node):
        if node.op not in supported:
            raise ValueError(f"unsupported temporal export operator: {node.op}")
        if node.op in {"feature", "lag"} and node.name not in feature_names:
            raise ValueError("temporal predictors can read only control "
                             + ", ".join(feature_names))
        return (node.op, node.value, node.name, node.lag,
                tuple(encode(child) for child in node.children))

    return "# Frozen paid-data predictor; controls and recorded history only.\n_PROGRAM = " + repr(
        encode(program.expanded())) + "\n" + _PROGRAM_RUNNER


def _hypotheses(model):
    return model.get("fields", {}).get("meter", {}).get("hypotheses", [])


def _summary(model):
    keys = ("hypothesis_id", "kind", "status", "fit_loss", "validation_loss")
    return [{key: hypothesis[key] for key in keys if key in hypothesis}
            for hypothesis in _hypotheses(model)]


class TemporalOperatorAgent(InquiryLLMAgent):
    """Ordinary inquiry actions plus the same optional operators in every arm."""

    operator_interface = OPERATOR_INTERFACE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator_results = []
        self.candidate_paths = []
        self._current_research_advice = None

    def _state_research_advice(self):
        return self._current_research_advice

    def _program_state(self):
        state = super()._program_state()
        state.update(operator_results=self.operator_results, exported_candidate_paths=self.candidate_paths)
        advice = self._state_research_advice()
        if advice is not None:
            state["research_advice"] = advice
        session = getattr(self, "legacy_session", None)
        if session is not None:
            native = session.session.state
            state["legacy_session"] = {
                "tasks": [{"task_position": row["task_position"],
                           "status": row["research"]["status"]} for row in native["results"]],
                "pending_revisits": [row["task_position"] for row in native["pending_revisits"]]}
        return state

    def _allows_analysis_only_query(self, continuation):
        return bool(self.observations) and isinstance(continuation, list) and bool(continuation)

    def _query(self, arguments, laboratory):
        first_question = len(self.questions)
        result = super()._query(arguments, laboratory)
        if result.get("program_action"):
            # The declared action already ran its query continuations on this
            # same agent; the surrounding program invocation only reports it.
            return result
        if result.get("error") or result.get("returncode"):
            return result
        continuation = result.get("after_observation", [])
        if continuation == []:
            return result
        current_question = self.questions[-1] if len(self.questions) > first_question else None
        outcomes = result["after_observation_results"] = []
        if not isinstance(continuation, list):
            result["after_observation_error"] = "after_observation must be a list of shared operator arguments"
            if current_question is not None:
                current_question["after_observation_error"] = result["after_observation_error"]
                result["question"] = deepcopy(current_question)
            return result
        for authored in continuation:
            resolved = deepcopy(authored)
            before_compute = self.costs["compute_cost"]
            before_seconds = self.costs["python_execution_seconds"]
            try:
                if not isinstance(resolved, dict) or "operator" not in resolved:
                    raise ValueError("each continuation step needs a shared operator")
                if resolved.get("records") == "acquired":
                    resolved["records"] = list(result["record_indices"])
                feedback = self._execute("run", resolved, laboratory)
            except (ValueError, TypeError, KeyError, OSError) as error:
                feedback = {"operator": resolved.get("operator") if isinstance(resolved, dict) else None,
                            "error": str(error),
                            "compute_cost": self.costs["compute_cost"] - before_compute,
                            "python_execution_seconds": self.costs["python_execution_seconds"] - before_seconds}
            outcome = {"arguments": resolved, **feedback}
            outcomes.append(outcome)
            if current_question is not None:
                current_question.setdefault("after_observation_results", []).append(deepcopy(outcome))
            if "error" in feedback:
                result["after_observation_error"] = feedback["error"]
                if current_question is not None:
                    current_question["after_observation_error"] = feedback["error"]
                break
        if current_question is not None:
            result["question"] = deepcopy(current_question)
        return result

    def _remaining_compute(self):
        return max(0, self.config.compute_budget - self.costs["compute_cost"])

    def _charge(self, units):
        if units > self._remaining_compute():
            raise ValueError("shared research operator compute budget exhausted")
        self.costs["compute_cost"] += units

    def _save(self, path, source):
        destination = self.workspace / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
        self.files[path] = source
        session = getattr(self, "legacy_session", None)
        if path == INSTRUMENT_METHODS_PATH and session is not None:
            session.session.library.program_experiment_methods = ProgramExperimentMethods(json.loads(source))

    def _save_instrument(self, path, model):
        self._save(path, json.dumps(model, indent=2))

    def _load_instrument(self, arguments):
        path = arguments.get("instrument", "instrument.json")
        return path, json.loads((self.workspace / path).read_text(encoding="utf-8"))

    def _record_indices(self, requested):
        indices = list(requested)
        if (len(indices) != len(set(indices)) or
                any(not isinstance(index, int) or not 0 <= index < len(self.observations)
                    for index in indices)):
            raise ValueError("records must be distinct zero-based laboratory record indices")
        return indices

    def _split(self, arguments, indices=None):
        records = self.observations
        indices = list(range(len(records))) if indices is None else indices
        if not indices:
            raise ValueError("buy complete trajectories before fitting a response")
        # Repeated reset inputs stay together even when observed labels differ;
        # target values must not decide whether an input leaks across a split.
        groups = []
        inputs = {index: [{"features": row["features"], "history": row.get("history", [])}
                          for row in records[index]["rows"]] for index in indices}
        for index in indices:
            record = records[index]
            group = next((group for group in groups
                          if records[group[0]].get("action") == record.get("action")
                          and inputs[group[0]] == inputs[index]), None)
            if group is None:
                groups.append([index])
            else:
                group.append(index)
        if "validation_records" in arguments:
            validation = self._record_indices(arguments["validation_records"])
            if not set(validation).issubset(indices):
                raise ValueError("validation_records must be a subset of selected records")
            if not validation and len(indices) > 1:
                raise ValueError("choose at least one whole validation trajectory")
        else:
            count = min(2, len(groups) - 1)
            longest = sorted((group[-1] for group in groups),
                             key=lambda index: (len(records[index]["rows"]), index))
            validation = sorted(longest[-count:]) if count else []
            if count == 2:
                train_long, held_short = longest[-3], longest[-2]
                if len(records[held_short]["rows"]) > len(records[train_long]["rows"]):
                    validation = sorted((set(validation) - {held_short}) | {train_long})
        selected_validation = list(validation)
        validation = sorted({index for group in groups if set(group).intersection(validation)
                             for index in group})
        training = [index for index in indices if index not in validation]
        if not training:
            raise ValueError("keep at least one distinct whole trajectory for training; "
                             "identical reset records cannot cross the validation split")
        train_rows = [row for index in training for row in records[index]["rows"]]
        held_rows = [row for index in validation for row in records[index]["rows"]]
        return train_rows, held_rows, {"records": indices,
                                      "training_records": training, "validation_records": validation,
                                      "requested_validation_records": selected_validation,
                                      "duplicate_groups": [group for group in groups if len(group) > 1],
                                      "train_rows": len(train_rows), "holdout_rows": len(held_rows)}

    def _history(self, arguments):
        train_rows, validation_rows, split = self._split(arguments)
        if len(self.observations) > 1 and not validation_rows:
            raise ValueError("history comparison needs at least one paid validation trajectory")
        base_lags = arguments.get("base_lags", 2)
        if not isinstance(base_lags, int) or base_lags < 0:
            raise ValueError("base_lags must be a nonnegative integer")
        widths = [(lags + 2) * (lags + 3) // 2 for lags in (base_lags, base_lags + 1)]
        # Two basis evaluations, two training fits and full-data score passes,
        # followed by the selected all-row refit and its score.
        self._charge((len(train_rows) + len(validation_rows)) * (sum(widths) + 4)
                     + 2 * len(train_rows))
        held = len(split["validation_records"])
        fraction = nextafter(held / len(self.observations), 0.) if held else 1 / 3
        proposal = revise_history(train_rows + validation_rows, base_lags=base_lags,
                                  holdout_fraction=fraction)
        path = arguments.get("path", "model.py")
        self._save(path, proposal.to_code())
        if path not in self.candidate_paths:
            self.candidate_paths.append(path)
        return {"path": path, **proposal.resultdict, "split": {**proposal.resultdict["split"], **split}}

    def _refit(self, arguments):
        indices = self._record_indices(arguments.get("records", range(len(self.observations))))
        train_rows, validation_rows, split = self._split(arguments, indices)
        if not validation_rows:
            raise ValueError("refit comparison needs separate paid validation trajectories")
        model = arguments["model"]
        source, source_model = python_model_context(self.workspace, model)
        held = {row_id: row for index in split['validation_records']
                for row_id, row in zip(self.observations[index]['row_indices'],
                                       self.observations[index]['rows'])}
        # Reuse actual bound dependencies; unrelated archive entries are not inputs.
        for directory, inner in [(source, source_model), *_bound_model_contexts(source, source_model)]:
            bindings_file = directory / 'model_research_inputs.json'
            inputs = json.loads(bindings_file.read_text()).get(inner, {}) if bindings_file.exists() else {}
            for slot, bound in inputs.items():
                if bound.get('kind') != 'local_relations':
                    continue
                overlap = set()
                for check in bound['checks']:
                    for witness in check['witnesses']:
                        row = held.get(witness['row'])
                        # Row numbers alone belong to different tasks' archives.
                        if (row is not None and witness['features'] == row['features']
                                and list(witness.get('history', [])) == list(row.get('history', []))
                                and witness['target'] == row['target']):
                            overlap.add(witness['row'])
                if overlap:
                    raise ValueError(f'refit validation labels are already bound into fitting: '
                        f'{inner}:{slot}, response rows {sorted(overlap)}. Select training-only '
                        'relation checks or different validation_records; use all-paid evaluate '
                        'or fresh confirmation for the corresponding evidence scope.')
        directory = Path(arguments.get("path", f"refits/candidate_{len(self.operator_results):04d}"))
        context = (self.workspace / directory / "_evaluated_model").resolve()
        if context.exists() or (source != self.workspace and context.is_relative_to(source)):
            raise ValueError("refit needs a new context outside the original model archive")
        self._charge(len(train_rows) + len(validation_rows))
        inner_model = _copy_model_context(self.workspace, model, context)
        # These files are actual inquiry/runner records containing current labels.
        # Ordinary model dependencies remain available in the copied context.
        for filename in ("research_context.json", "research_state.json", "active.json",
                         "preparation.json", "training.json", "evaluation.json",
                         "published_candidate.json", "result.json"):
            (context / filename).unlink(missing_ok=True)

        def fitting_view(rows, records):
            text = json.dumps(rows, allow_nan=False)
            for filename in ("model_observations.json", "observations.json"):
                (context / filename).write_text(text, encoding="utf-8")
            (context / "laboratory_observations.json").write_text(
                json.dumps([self.observations[index] for index in records], allow_nan=False),
                encoding="utf-8")

        fitting_view(train_rows, split["training_records"])
        try:
            validation = evaluate_python_model(context, inner_model, validation_rows,
                timeout_seconds=max(.001, self.config.execution_seconds
                    - self.costs["python_execution_seconds"]), require_fit=True,
                research_compute_budget=self._remaining_compute())
            self.costs['compute_cost'] += validation.get('research_prediction_count', 0)
        finally:
            fitting_view([row for index in indices for row in self.observations[index]["rows"]], indices)
        candidate_path = str(context.relative_to(self.workspace) / inner_model)
        available = validation.get("fit_available")
        valid = validation.get("returncode") == 0 and validation.get("mse") is not None
        if validation.get('truncated'):
            status = 'refit_inconclusive_budget'
        elif available is False:
            status = "refit_unavailable"
        elif not valid:
            status = "refit_execution_failed"
        elif validation["mse"] <= self.config.confirmation_mse_threshold:
            status = "refit_supported_on_validation"
        else:
            status = "refit_inadequate_on_validation"
        if available and valid:
            if candidate_path not in self.candidate_paths:
                self.candidate_paths.append(candidate_path)
            for path in context.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}:
                    self.files[str(path.relative_to(self.workspace))] = path.read_text(encoding="utf-8")
        return {"status": status, "path": candidate_path if available and valid else None,
                "tested_model": candidate_path,
                "source_model": model, "split": split, "validation": validation,
                "validation_loss_kind": "mse",
                "validation_mse_threshold": self.config.confirmation_mse_threshold,
                "final_fitting_rows": len(train_rows) + len(validation_rows),
                "paid_rows_at_refit": len(self.rows), "paid_records_at_refit": len(self.observations),
                "evidence_scope": "Same-source fitting procedure assessed on separate paid trajectories; "
                    "the exported candidate can refit on all selected paid rows. No fresh confirmation, "
                    "automatic submission, or general model-family rejection."}

    def _instrument(self, arguments, spec):
        indices = self._record_indices(arguments.get("records", range(len(self.observations))))
        if not indices:
            raise ValueError("choose paid records containing meter readings")
        model = fit_temporal_instrument([self.observations[index] for index in indices], spec,
            compute_budget=min(self._remaining_compute(), arguments.get("compute_limit", self._remaining_compute())))
        self._charge(model["compute_cost"])
        path = arguments.get("path", "instrument.json")
        self._save_instrument(path, model)
        return {"path": path, "records": indices, "status": model["status"], "support_rows": model["support_rows"],
                "hypotheses": _summary(model), "unmodeled_fields": model.get("unmodeled_fields", {})}

    def _paid_meter_consistency(self, model, spec, selected_indices):
        indices = [index for index in selected_indices
                   if any("meter" in reading["values"]
                          for reading in self.observations[index].get("measurements", []))]
        records = [self.observations[index] for index in indices]
        measured_rows = sum("meter" in reading["values"] for record in records
                            for reading in record["measurements"])
        survivors = [hypothesis for hypothesis in _hypotheses(model)
                     if hypothesis.get("status") not in {"refuted", "counterexample"}]
        required = measured_rows * len(survivors)
        result = {"status": "unavailable", "reason": None, "records": indices,
                  "measured_rows": measured_rows, "scored_rows": 0,
                  "hypotheses": [], "compute_cost": 0,
                  "evidence_scope": "Current paid readings; may include fitting evidence; not fresh confirmation."}
        if not measured_rows:
            result["reason"] = "no_paid_meter_readings"
        elif not survivors:
            result["reason"] = "no_surviving_hypotheses"
        elif required > self._remaining_compute():
            result["reason"] = "insufficient_compute_budget"
        else:
            assessed_model = deepcopy(model)
            assessment = assess_temporal_instrument(assessed_model, records, spec,
                compute_budget=self._remaining_compute())
            self._charge(assessment["compute_cost"])
            scored = assessment["field_losses"].get("meter", {})
            residuals = [{"hypothesis_id": hypothesis["hypothesis_id"],
                          "mse": hypothesis["validation_loss"],
                          "max_squared_error": hypothesis["max_squared_error"],
                          "scored_rows": hypothesis["validation_rows"]}
                         for hypothesis in _hypotheses(assessed_model)
                         if hypothesis["hypothesis_id"] in scored]
            result.update(status="available", hypotheses=residuals,
                          scored_rows=sum(item["scored_rows"] for item in residuals),
                          compute_cost=assessment["compute_cost"])
        return result

    def _response(self, arguments, spec):
        instrument_path, model = self._load_instrument(arguments)
        indices = self._record_indices(arguments.get("records", range(len(self.observations))))
        train_rows, validation_rows, split = self._split(arguments, indices)
        if not validation_rows:
            raise ValueError("response comparison needs separate paid validation trajectories")
        surviving = [hypothesis for hypothesis in _hypotheses(model)
                     if hypothesis.get("status") not in {"refuted", "counterexample"}]
        all_rows = train_rows + validation_rows
        # Ten response columns, fit and score for training; validation scoring;
        # ten all-row columns plus refit and scoring. No uncharged candidate fits.
        per_candidate = 12 * len(train_rows) + len(validation_rows) + 12 * len(all_rows)
        self._charge(per_candidate * len(surviving))
        candidates, hypotheses = [], []
        for index, hypothesis in enumerate(surviving):
            instrument = Program.from_dict(hypothesis["program"])
            expression, rank, width = fit_temporal_response(train_rows, instrument)
            predictor = bind_answers(expression, {"instrument": instrument})
            train_loss, validation_loss = mse(predictor, train_rows), mse(predictor, validation_rows)
            expression, refit_rank, _ = fit_temporal_response(all_rows, instrument)
            refitted = bind_answers(expression, {"instrument": instrument})
            path = str(Path(arguments.get("output_dir", "shared/candidates")) / f"response_{index:02d}.py")
            self._save(path, export_temporal_program(refitted))
            if path not in self.candidate_paths:
                self.candidate_paths.append(path)
            candidates.append({"hypothesis_id": hypothesis["hypothesis_id"],
                "kind": hypothesis.get("kind"), "stored_instrument_status": hypothesis.get("status"),
                "path": path, "training_rank": rank, "feature_count": width,
                "train_mse": train_loss, "validation_mse": validation_loss,
                "refit_train_mse": mse(refitted, all_rows), "refit_rank": refit_rank})
            hypotheses.append({"hypothesis_id": path, "path": path,
                               "program": refitted.expanded().to_dict()})
        hypotheses_path = str(Path(arguments.get("output_dir", "shared/candidates")) / "hypotheses.json")
        self._save(hypotheses_path, json.dumps(hypotheses, indent=2))
        return {"instrument": instrument_path,
                "instrument_status_note": "Status refers to evidence saved in this instrument file; "
                                          "current response validation does not test current meter readings.",
                "paid_meter_consistency": self._paid_meter_consistency(model, spec, indices),
                "split": split, "candidates": candidates, "hypotheses_path": hypotheses_path}

    def _disagreement(self, arguments, spec):
        instrument, model = self._load_instrument(arguments)
        result = select_temporal_instrument_probe(model, spec,
            max_length=arguments.get("max_length"), serial=arguments.get("serial", 0),
            compute_budget=min(self._remaining_compute(), arguments.get("compute_limit", self._remaining_compute())),
            methods=self._instrument_methods())
        self._charge(result["compute_cost"])
        if result["action"] is not None:
            path = f"instrument_experiments/proposal_{len(self.operator_results):04d}.json"
            self._save(path, json.dumps({**result, "instrument": instrument,
                "first_new_record": len(self.observations)}, indent=2))
            result["experiment"] = path
        # Full hypotheses stay in the task-local plan; ordinary feedback carries
        # the action, forecasts and the existing library's revision explanation.
        return {key: result[key] for key in ("action", "reason", "sequence_mode", "predictions",
                "predicted_separation", "field_disagreements", "experiment", "method_id",
                "parent_method_id", "preparation_revision", "method_attempts") if key in result}

    def _instrument_methods(self):
        session = getattr(self, "legacy_session", None)
        if session is not None:
            return ProgramExperimentMethods(session.session.library.program_experiment_methods.to_dict())
        path = self.workspace / INSTRUMENT_METHODS_PATH
        return ProgramExperimentMethods(json.loads(path.read_text(encoding="utf-8"))
                                        if path.exists() else [])

    def _python_research_choice(self, arguments, spec):
        from .python_research_routing import validated_candidates
        if "records" in arguments:
            raise ValueError("Python candidate comparison uses all current paid rows")
        models = arguments.get("models")
        if models is None:
            current = self._evaluated_summary()
            models = [current["model"]] if current else []
        models = list(models) + validated_candidates(self)
        # Explicit comparison owns scoring. Automatic feedback uses the same
        # decision below without recursively scoring those candidates again.
        evaluations = [{"path": path, **super(TemporalOperatorAgent, self)._execute(
                            "evaluate", {"model": path}, None)} for path in dict.fromkeys(models)]
        return self._python_research_decision(self._evaluated_summary(), spec, evaluations=evaluations)

    def _python_research_decision(self, retained, spec, *, evaluations=(), repair=None):
        from .action_policy import ActionOption, ResearchActionPolicy
        from .python_research_routing import (
            evaluate_action, matching_refit, revision_action, uses_refitted_training_view,
        )
        from .temporal_instrument_research import _specs
        refit = matching_refit(self, retained["model"]) if retained else None
        failed_validation = bool(refit and refit["status"] == "refit_inadequate_on_validation"
                                 and uses_refitted_training_view(self, retained["model"], refit))
        scored = bool(retained and retained["support"] in {'paid_fit', 'paid_prediction', 'paid_evaluation'})
        supported = bool(scored
                         and retained["mse"] is not None and retained["mse"] <= 1e-10
                         and not failed_validation)
        diagnosis = ("retained_response_fits_paid_evidence" if supported else
                     "response_counterexample" if scored
                     else "no_current_evaluated_response")
        knowledge = self.workspace / "shared/legacy_knowledge.json"
        experience = json.loads(knowledge.read_text()).get("action_experience", {}) if knowledge.exists() else {}
        if supported:
            repair = None
        elif repair is None and retained:
            if (retained["support"] == "new_paid_rows_unscored"
                    and retained["mse"] is not None and retained["mse"] <= 1e-10):
                repair = evaluate_action(self, retained["model"], "new_paid_rows_need_comparison")
            else:
                repair = revision_action(self, retained, spec)
        if repair is not None:
            diagnosis = repair["diagnosis"]
        policy = ResearchActionPolicy(self.rows, _specs(spec)[0], experience)
        policy.set_condition("temporal:" + diagnosis)
        options = ([ActionOption("retain_response", self.config.confirmation_rows,
                    len(self.rows) + self.config.confirmation_rows)] if supported else [])
        if repair and repair.get("next_action"):
            options.append(ActionOption(repair["name"], repair["observation_cost"],
                repair["compute_cost"], measurement_cost=repair["measurement_cost"]))
        choice = policy.choose(options,
            observation_remaining=max(0, self.config.observation_budget - self.costs["observation_cost"]),
            compute_remaining=self._remaining_compute(), minimum_improvement=1e-10)
        if not options:
            policy.last_decision["reason"] = "no_supported_confirmation_candidate"
        result = {"diagnosis": diagnosis, "evaluations": list(evaluations),
            "evaluated_model": retained, "paid_rows": len(self.rows),
            "decision": {**policy.last_decision, "options": [
                {"name": option.name, "observation_cost": option.observation_cost,
                 "compute_cost": option.compute_cost} for option in options]},
            "recommendation": None,
            "evidence_scope": "Paid candidate comparison and existing revision proposals; no automatic acquisition or submission."}
        if choice is not None:
            if supported:
                result["recommendation"] = {"operator": "retain_response", "path": retained["model"]}
                result["next_action"] = {"action": "finish", "arguments": {"model": retained["model"]}}
            else:
                result["recommendation"] = {"operator": repair["name"]}
                result["next_action"] = repair["next_action"]
        return result

    def _render_research_advice(self, advice):
        """Present a proposal; experiments may remove presentation only."""
        return advice

    def _pending_python_research_advice(self, laboratory):
        """Resume an opened question from its paid records after other actions."""
        from .python_research_routing import evaluate_action, instrument_action
        if laboratory is None or self.model is not None:
            return None
        spec = laboratory.public_spec()
        if "trajectory" not in spec.get("action_kinds", []):
            return None
        opened = any(record.get("measurements") for record in self.observations) or any(
            result.get("operator") == "instrument" or
            (result.get("operator") == "refit" and
             result.get("status") == "refit_inadequate_on_validation")
            for result in self.operator_results)
        if not opened:
            return None
        # Continue the latest validated repair, including an independently
        # refitted alternative to the instrument-based response.
        repair = None
        for result in reversed(self.operator_results):
            if (result.get("operator") == "refit"
                    and result.get("status") == "refit_supported_on_validation"
                    and result.get("final_fitting_rows") == len(self.rows)):
                repair = evaluate_action(self, result["path"], "refit_validated_needs_evaluation", validated=True)
                break
            if (result.get("operator") == "response"
                    and result.get("split", {}).get("records") == list(range(len(self.observations)))):
                candidates = [item for item in result.get("candidates", [])
                              if item.get("validation_mse") is not None and item["validation_mse"] <= 1e-10]
                if candidates:
                    repair = evaluate_action(self, min(candidates, key=lambda item: item["validation_mse"])["path"],
                        "response_validated_needs_evaluation", validated=True)
                    break
        if repair is None:
            repair = instrument_action(self, spec)
        advice = self._python_research_decision(self._evaluated_summary(), spec, repair=repair)
        advice["evidence_scope"] = "Existing paid results only; this advice performs no fitting, scoring or acquisition."
        return advice

    def _feedback_research_advice(self, action, arguments, result, laboratory):
        from .python_research_routing import evaluate_action, instrument_action
        if laboratory is None or self.model is not None:
            return None
        spec = laboratory.public_spec()
        if "trajectory" not in spec.get("action_kinds", []):
            return None
        retained, repair = self._evaluated_summary(), None
        if action == "finish" and result.get("fresh_confirmation", {}).get("passed") is False:
            # These are real fresh counterexamples to the submitted predictor,
            # not a new whole-data score for the possibly different incumbent.
            retained = {"model": arguments.get("model", "model.py"),
                "source_model": arguments.get("model", "model.py"), "mse": None,
                "support": "fresh_counterexample", "current_paid_rows": len(self.rows),
                "fresh_mse": result["fresh_confirmation"].get("mse")}
        elif action == "evaluate":
            if retained is None or result.get("returncode") != 0:
                return None
        elif action == "query":
            declared = result.get("program_action")
            if declared:
                actual = {key: value for key, value in result.items() if key != "program_action"}
                return self._feedback_research_advice(declared["action"],
                    declared.get("arguments", {}), actual, laboratory)
            if result.get("error") or result.get("after_observation_error"):
                return None
            outcomes = result.get("after_observation_results", [])
            if outcomes:
                latest = outcomes[-1]
                return self._feedback_research_advice("run", latest["arguments"], latest, laboratory)
            if any(record.get("measurements") for record in result.get("observations", [])):
                repair = instrument_action(self, spec)
            elif retained is None:
                return None
        elif action == "run" and "error" not in result:
            operator = arguments.get("operator")
            if operator == "refit":
                if result.get("status") == "refit_supported_on_validation":
                    repair = evaluate_action(self, result["path"], "refit_validated_needs_evaluation", validated=True)
                else:
                    repair = instrument_action(self, spec, result.get("status", "refit_execution_failed"))
            elif operator in {"instrument", "assess_instrument", "disagreement"}:
                repair = instrument_action(self, spec)
            elif operator == "response":
                candidates = [item for item in result.get("candidates", [])
                              if item.get("validation_mse") is not None and item["validation_mse"] <= 1e-10]
                repair = (evaluate_action(self, min(candidates, key=lambda item: item["validation_mse"])["path"],
                    "response_validated_needs_evaluation", validated=True) if candidates else
                    {"diagnosis": "response_revision_unresolved"})
            else:
                return None
        else:
            return None
        advice = self._python_research_decision(retained, spec, repair=repair)
        advice["evidence_scope"] = "Existing paid results only; this advice performs no fitting, scoring or acquisition."
        return advice

    def _attach_research_advice(self, action, arguments, result, laboratory):
        started = monotonic()
        advice = self._feedback_research_advice(action, arguments, result, laboratory)
        if advice is None:
            advice = self._pending_python_research_advice(laboratory)
        self._current_research_advice = None
        if advice is not None:
            rendered = self._render_research_advice(advice)
            if rendered is not None:
                result["research_advice"] = rendered
                self._current_research_advice = deepcopy(rendered)
        # Both presentation conditions compute proposals and pay this runtime.
        self.costs["python_execution_seconds"] += monotonic() - started
        return result

    def _research_choice(self, arguments, spec):
        if "hypotheses" not in arguments:
            return self._python_research_choice(arguments, spec)
        from .temporal_research_choice import choose_temporal_research
        hypotheses = json.loads((self.workspace / arguments["hypotheses"]).read_text())
        instrument = self._load_instrument(arguments)[1] if "instrument" in arguments else None
        indices = self._record_indices(arguments.get("records", range(len(self.observations))))
        knowledge = self.workspace / "shared/legacy_knowledge.json"
        experience = json.loads(knowledge.read_text()).get("action_experience", {}) if knowledge.exists() else {}
        result = choose_temporal_research([self.observations[index] for index in indices],
            hypotheses, spec, instrument=instrument, methods=self._instrument_methods(),
            experience=experience,
            observation_remaining=max(0, self.config.observation_budget - self.costs["observation_cost"]),
            compute_remaining=self._remaining_compute(), confirmation_rows=self.config.confirmation_rows,
            finish_scoring_rows=len(self.rows),
            task_id=str(self.workspace), serial=arguments.get("serial", 0), length=arguments.get("length", 4))
        self._charge(result["compute_cost"])
        result["records"] = indices
        response_plan, instrument_plan = result.pop("response_plan"), result.pop("instrument_plan")
        recommendation = result.get("recommendation")
        if not recommendation:
            return result
        operator = recommendation["operator"]
        plan = response_plan if operator == "response_disagreement" else instrument_plan
        if plan is not None and operator in {"response_disagreement", "disagreement"}:
            kind = "response" if operator == "response_disagreement" else "instrument"
            path = f"{kind}_experiments/proposal_{len(self.operator_results):04d}.json"
            saved = {**plan, "first_new_record": len(self.observations)}
            assessment = {"operator": f"assess_{kind}", "experiment": path, "records": "acquired"}
            if kind == "instrument":
                saved["instrument"] = arguments["instrument"]
                assessment["instrument"] = arguments["instrument"]
            self._save(path, json.dumps(saved, indent=2))
            result.update(experiment=path, action=plan["action"],
                proposal_source=plan.get("proposal_source"), proposal=plan.get("proposal"),
                fresh_synthesis_skipped=plan.get("fresh_synthesis_skipped"))
            result["next_action"] = {"action": "query", "arguments": {
                "actions": [plan["action"]], "after_observation": [assessment]}}
        elif operator == "retain_response":
            if recommendation.get("path"):
                result["next_action"] = {"action": "finish", "arguments": {"model": recommendation["path"]}}
        elif operator in {"instrument", "response"}:
            next_args = {"operator": operator, "records": indices}
            if operator == "instrument":
                next_args["path"] = arguments.get("instrument", "instrument.json")
            elif "instrument" in arguments:
                next_args["instrument"] = arguments["instrument"]
            result["next_action"] = {"action": "run", "arguments": next_args}
        return result

    def _purchased_experiment(self, experiment, indices):
        plan = json.loads((self.workspace / experiment).read_text(encoding="utf-8"))
        if "method_feedback" in plan:
            raise ValueError("this experiment already has recorded method feedback")
        if len(indices) != 1 or indices[0] < plan["first_new_record"]:
            raise ValueError("experiment feedback needs its single newly purchased trajectory")
        record = self.observations[indices[0]]
        if (record["action"] != plan["action"]
                or (plan["action"]["channels"] and len(record["measurements"]) != len(record["rows"]))):
            raise ValueError("purchased trajectory does not match the proposed experiment")
        return plan, record

    def _measured_experiment(self, experiment, instrument, model, indices, spec):
        plan, record = self._purchased_experiment(experiment, indices)
        # A later refit can overwrite the same path and recycle hypothesis IDs.
        # Attribute this experiment only to the programs whose forecasts led to it.
        planned = plan["instrument_condition"]["programs"]
        current = {name: {item["hypothesis_id"]: item["program"]
                         for item in model["fields"][name].get("hypotheses", [])
                         if item.get("status") not in {"refuted", "counterexample"}}
                   for name in planned}
        if (instrument != plan["instrument"] or current != planned
                or spec["feature_bounds"] != plan["instrument_condition"]["bounds"]):
            raise ValueError("instrument hypotheses or control bounds changed after planning")
        return plan, record

    def _assess(self, arguments, spec):
        path, model = self._load_instrument(arguments)
        indices = self._record_indices(arguments["records"])
        if not indices:
            raise ValueError("choose paid records containing meter readings")
        experiment = arguments.get("experiment")
        if experiment is not None:
            plan, record = self._measured_experiment(experiment, path, model, indices, spec)
        result = assess_temporal_instrument(model, [self.observations[index] for index in indices],
            spec, compute_budget=self._remaining_compute())
        self._charge(result["compute_cost"])
        self._save_instrument(path, model)
        if experiment is not None:
            methods = self._instrument_methods()
            feedback = observe_temporal_instrument_method(methods, plan, result, model, spec,
                task_id=str(self.workspace), record=record)
            result["method_feedback"] = feedback
            if feedback is not None:
                self._save(INSTRUMENT_METHODS_PATH, json.dumps(methods.to_dict(), indent=2))
                self._save(experiment, json.dumps({**plan, "method_feedback": feedback,
                                                  "records": indices}, indent=2))
        return {**result, "path": path, "records": indices, "status": model["status"],
                "hypotheses": _summary(model)}

    def _response_disagreement(self, arguments, spec):
        hypotheses = json.loads((self.workspace / arguments["hypotheses"]).read_text(encoding="utf-8"))
        instrument = self._load_instrument(arguments)[1] if "instrument" in arguments else None
        plan = plan_response_experiment(hypotheses, spec, methods=self._instrument_methods(),
            instrument=instrument, serial=arguments.get("serial", 0),
            length=arguments.get("length", 4), compute_budget=self._remaining_compute(),
            task_id=str(self.workspace))
        self._charge(plan["compute_cost"])
        result = {key: plan[key] for key in ("action", "forecast_status", "predicted_separation",
            "proposal_source", "proposal", "method_candidates", "fresh_synthesis_skipped",
            "predicted_fields", "missing_fields") if key in plan}
        result["hypotheses"] = [{key: item[key] for key in ("hypothesis_id", "predictions")}
                                for item in plan["hypotheses"]]
        if plan["action"] is not None:
            path = f"response_experiments/proposal_{len(self.operator_results):04d}.json"
            self._save(path, json.dumps({**plan, "first_new_record": len(self.observations)}, indent=2))
            result["experiment"] = path
        return result

    def _assess_response(self, arguments, spec):
        indices = self._record_indices(arguments["records"])
        plan, record = self._purchased_experiment(arguments["experiment"], indices)
        if spec["feature_bounds"] != plan["control_bounds"]:
            raise ValueError("control bounds changed after response experiment planning")
        methods = self._instrument_methods()
        result = assess_response_experiment(plan, record, spec, methods=methods,
            task_id=str(self.workspace), compute_budget=self._remaining_compute())
        self._charge(result["compute_cost"])
        if result["method_feedback"] is not None:
            self._save(INSTRUMENT_METHODS_PATH, json.dumps(methods.to_dict(), indent=2))
        self._save(arguments["experiment"], json.dumps({**plan, "records": indices,
            "assessment": result, "method_feedback": result["method_feedback"]}, indent=2))
        return {**result, "records": indices, "experiment": arguments["experiment"]}

    def _execute(self, action, arguments, laboratory):
        if action != "run" or "operator" not in arguments:
            session = getattr(self, "legacy_session", None)
            if action == "finish" and session is not None:
                position = next(index for index, episode in enumerate(session.episodes)
                                if episode["agent"] is self)
                session.sync_paid_observations(position)
            result = super()._execute(action, arguments, laboratory)
            if ((action == "evaluate" and result.get("mse") is not None and result["mse"] > 1e-10)
                    or (action == "finish" and result.get("fresh_confirmation", {}).get("passed") is False)):
                result["available_research_action"] = {"action": "run", "arguments": {
                    "operator": "research_choice"}}
            if action == "finish" and session is not None and result.get("fresh_confirmation"):
                result["source_confirmation"] = session.record_confirmation(position, result)
            return self._attach_research_advice(action, arguments, result, laboratory)
        before_compute = self.costs["compute_cost"]
        before_seconds = self.costs["python_execution_seconds"]
        try:
            result = self._execute_operator(arguments, laboratory)
        except (ValueError, TypeError, KeyError, OSError) as error:
            result = {"operator": arguments["operator"], "error": str(error),
                      "compute_cost": self.costs["compute_cost"] - before_compute,
                      "python_execution_seconds": self.costs["python_execution_seconds"] - before_seconds}
            self.operator_results.append({"index": len(self.operator_results),
                                          "arguments": deepcopy(arguments), **result})
            raise
        self.operator_results.append({"index": len(self.operator_results),
                                      "arguments": deepcopy(arguments), **deepcopy(result)})
        return self._attach_research_advice(action, arguments, result, laboratory)

    def _execute_operator(self, arguments, laboratory):
        remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining <= 0:
            raise ValueError("Python execution time budget exhausted")
        before, started = self.costs["compute_cost"], monotonic()
        before_seconds = self.costs["python_execution_seconds"]
        operator = arguments["operator"]
        try:
            spec = laboratory.public_spec()
            if operator == "legacy_capabilities":
                from .historical_capabilities import historical_capabilities
                result = historical_capabilities(version=arguments.get("version"), group=arguments.get("group"),
                                                 query=arguments.get("query"), compact=arguments.get("compact", True))
            elif operator == "legacy_research":
                from .legacy_research_bridge import LEGACY_KNOWLEDGE_PATH, run_legacy_research
                session = getattr(self, "legacy_session", None)
                if session is not None and "knowledge" not in arguments:
                    self._save(LEGACY_KNOWLEDGE_PATH, json.dumps(session.session.library.to_dict()))
                    self._save(INSTRUMENT_METHODS_PATH, json.dumps(
                        session.session.library.program_experiment_methods.to_dict()))
                result = run_legacy_research(self, arguments, laboratory)
                if session is not None:
                    position = next(index for index, episode in enumerate(session.episodes)
                                    if episode["agent"] is self)
                    session.record_research(position, result)
            elif operator in {"legacy_revisit", "legacy_return_confirmation"}:
                session = getattr(self, "legacy_session", None)
                if session is None:
                    raise ValueError("join paid task episodes to LegacyInquirySession first")
                position = next(index for index, episode in enumerate(session.episodes)
                                if episode["agent"] is self)
                if operator == "legacy_return_confirmation":
                    result = session.confirm_return(position, length=arguments.get("length", 24))
                else:
                    before_visits = len(session.session.state["revisits"])
                    state = session.run(max_tasks=arguments.get("max_tasks", 1))
                    report = state["results"][position]["research"]
                    result = {**state["inquiry_exports"][position],
                        "knowledge_return": report.get("knowledge_return"),
                        "revisits": [{key: visit.get(key) for key in (
                            "task_position", "source_task_position", "status", "visit_costs")}
                            for visit in state["revisits"][before_visits:]],
                        "pending_tasks": [row["task_position"] for row in state["pending_revisits"]]}
            elif operator == "research_choice":
                result = self._research_choice(arguments, spec)
            elif operator == "history":
                result = self._history(arguments)
            elif operator == "refit":
                result = self._refit(arguments)
            elif operator == "instrument":
                result = self._instrument(arguments, spec)
            elif operator == "response":
                result = self._response(arguments, spec)
            elif operator == "disagreement":
                result = self._disagreement(arguments, spec)
            elif operator == "assess_instrument":
                result = self._assess(arguments, spec)
            elif operator == "response_disagreement":
                result = self._response_disagreement(arguments, spec)
            elif operator == "assess_response":
                result = self._assess_response(arguments, spec)
            else:
                raise ValueError(f"unknown shared research operator: {operator}")
        finally:
            elapsed = monotonic() - started
            # Nested evaluate calls already record their Python runtime.
            self.costs["python_execution_seconds"] = before_seconds + elapsed
        if elapsed > remaining:
            raise ValueError("shared research operator exceeded remaining Python execution time")
        return {"operator": operator, **result, "compute_cost": self.costs["compute_cost"] - before,
                "python_execution_seconds": elapsed}
