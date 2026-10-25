"""The same external Python agent with three persistent-knowledge conditions.

The laboratory never supplies a solver to this loop.  All fitting, candidate
selection, experiment programs and memory contents are authored by the provider.
The existing code provider and Python predictor executor are shared by all arms.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from random import Random
import sys
from time import monotonic

from .code_agent import (
    _copy_model_file, evaluate_python_model, python_model_context, retain_evaluated_python_model,
)
from .checkpoint_storage import write_text_incrementally, write_json_checkpoint
from .experiment_runtime import checkpoint_execution
from .models import Observation
from .proposers import _run
from .inquiry_revision import REVISION_INTERFACE, prepare_revision, complete_revision
from .executable_evidence import compare_paid_relations
from .question_semantics import adopt_question_semantics
from .paid_score_history import compact_agent_comparisons


ARMS = ("code_react", "textual_memory", "hegel")


def open_assigned_question(agent, question, experiment_path=None):
    """Reuse one live question while retaining its original assignment."""
    if not isinstance(question, dict):
        raise ValueError("research_question must be a JSON object with id; "
                         "include question text when opening a new question")
    existing = next((q for q in agent.questions if q["id"] == question["id"]), None)
    if existing is None:
        if not question.get("question"):
            raise ValueError("a new research question needs its question text")
        existing = InquiryLLMAgent._open_question(agent, adopt_question_semantics(
            None, {"parent_id": None, "depends_on": [], **question}), experiment_path)
        existing["assignment"] = {key: deepcopy(existing[key])
                                  for key in ("id", "question", "parent_id", "depends_on")}
        existing["acquired_evidence"] = []
    else:
        version = existing.get('meaning_version', 1)
        merged = adopt_question_semantics(existing, question)
        existing.clear()
        existing.update(merged)
        if existing['meaning_version'] != version:
            for model, bindings in getattr(agent, 'research_input_bindings', {}).items():
                if existing['id'] in bindings.values():
                    state = getattr(agent, 'research_candidates', {}).setdefault(model, {})
                    state.update(status='needs_rebuild', changed_question=existing['id'])
                    state.pop('current_evaluation', None)
        if experiment_path:
            existing["experiment_path"] = experiment_path
    return existing


def _prediction_residuals(rows, predictions, model, *, evaluation_scope, row_offset=0):
    errors = [(prediction - row.target) ** 2 for prediction, row in zip(predictions, rows)]
    indices = sorted(range(len(errors)), key=errors.__getitem__, reverse=True)[:3]
    return {'evaluation_scope': evaluation_scope, 'model': model, 'evaluated_rows': len(rows),
            'examples': [{'row_index': row_offset + index, **asdict(rows[index]),
                          'prediction': predictions[index], 'squared_error': errors[index]}
                         for index in indices]}


def _paid_prediction_scope(workspace, model, rows, predictions):
    workspace = Path(workspace).resolve()
    context, _ = python_model_context(workspace, model)
    training = context / 'model_observations.json'
    if not training.exists():
        training = context / 'observations.json'
    scope = {'evaluation_scope': 'paid_evaluation', 'fitting_view': {'scope': 'unknown'}}
    if not training.exists():
        return scope
    fitted = json.loads(training.read_text())
    training_reference = (training.relative_to(workspace)
                          if training.is_relative_to(workspace) else training)
    scope['fitting_view'].update(file=str(training_reference), rows=len(fitted))
    prefix = [{'features': row.features, 'target': row.target, 'history': list(row.history)}
              for row in rows[:len(fitted)]]
    if fitted != prefix:
        return scope
    count = len(fitted)
    scope['fitting_view'].update(scope='current_paid_prefix',
        response_row_range=[0, count - 1] if count else [])
    scope['evaluation_scope'] = 'paid_fit' if count == len(rows) else 'paid_prediction'
    if count < len(rows):
        errors = [prediction - row.target for prediction, row in zip(predictions[count:], rows[count:])]
        scope['later_readings'] = {'rows': len(errors),
            'response_row_range': [count, len(rows) - 1],
            'mse': sum(error ** 2 for error in errors) / len(errors),
            'max_abs_error': max(abs(error) for error in errors)}
    return scope


def _remember_paid_scope(retained, result):
    for key in ('evaluation_scope', 'fitting_view', 'later_readings'):
        retained.pop(key, None)
        if key in result:
            retained[key] = deepcopy(result[key])


@dataclass(frozen=True)
class InquiryLLMConfig:
    observation_budget: int = 24
    max_model_calls: int = 16
    token_budget: int = 60000
    execution_seconds: float = 60.0
    compute_budget: int = 100000
    confirmation_rows: int = 3
    confirmation_mse_threshold: float = 1e-10
    run_timeout_seconds: float = 300.0


POINT_LAB_INTERFACE = """This laboratory extends the ordinary Python research interface.
All three knowledge conditions have exactly the same actions and Python access.
query {actions:[action,...]} buys any public action from capabilities. Alternatively
query {sequence:[point,...],kind:'response'|'measure'} or
query {sequence:[{level:number},...],kind:'calibrate'} buys those actions.
Every action costs one observation. The observation budget covers all kinds.
Response rows are also written to model_observations.json and observations.json;
all response/measure/calibrate outcomes are in laboratory_observations.json.
The public forward language is z=orientation*point[controls[driver]], response
y=z, z*z, abs(z), or float(z>=threshold), for the four named mechanisms.
measure returns calibration_sign*z+calibration_offset; calibrate(level) returns
calibration_sign*level+calibration_offset. Those unknown choices are in the
public finite_language. You may implement this grammar yourself in Python.
"""

CONFIRMATION_MSE_THRESHOLD = 1e-10


COMMON_INTERFACE = """\
write {path:'model.py',code:'ordinary Python source'} creates or replaces a file.
run {code:'ordinary Python source'} executes that source as a Python script.
run {path:'fit.py'} executes an existing Python script in the task directory.
Both forms default to limits.run_timeout_seconds per invocation so a long
computation returns control before consuming the whole task allowance. Set
timeout_seconds explicitly to request a longer or shorter computation, bounded
by the remaining total Python allowance. A timeout returns execution feedback;
files already written persist and you may revise or resume the work.
The code key contains Python source; path names a file, not a shell command.
Python execution has numpy/scipy available and returns stdout/stderr and exit code.
For example, run {code:"import json; rows=json.load(open('observations.json')); print(len(rows))"}
inspects paid data. All paid response rows (features, target, history) are in
observations.json and model_observations.json. Auxiliary experiment records are
in laboratory_observations.json. Files and concise numerical results persist.
evaluate {model:'model.py'} fits/scores ONLY already purchased response rows.
paid_residuals returns up to three largest squared errors from that same scoring,
with the evaluated model, exact paid row indices, controls/history and labels.
These diagnose evaluation on paid rows, not a fresh confirmation or a scientific explanation.
Retained comparisons may use score_result_ref and paid_residuals.examples_ref
to address the complete saved score and residual examples at their file/json_path.
When earlier readouts contain executed response checks, paid_relation_diagnostic
also compares this candidate's already computed predictions with those measured
contrasts. It preserves their revision, declared scope and actual term rows;
an old refuted expectation is not used as the target. This feedback requires no
bind_research call and does not select a model or establish the author's interpretation.
Its model/workspace identify the evaluated candidate, not the retained incumbent.
historical_check and evidence_status describe the original hypothesis versus the
measured data; residual describes this candidate's prediction minus that measurement.
Use residuals across both historical statuses when diagnosing the current model.
Successful paid scoring by evaluate or finish retains the better executed
candidate in evaluated_model.model, even when finish cannot confirm it.
retention_decision identifies which candidate was actually kept. Its incumbent
is the comparison baseline, not necessarily the selected model; evaluated_model.selection
retains the decision that installed the current saved source and fitting view.
That path has its own saved files and fitting data; you
can evaluate or finish it even after revising the original file. Candidates
compete on the same paid rows. New observations require charged rescoring of
the saved candidate before replacement. fitting_view identifies its selected
fitting rows; later_readings separately scores subsequently purchased rows when
that fitting view matches the paid-row prefix. evaluated_model.support separates
paid_fit, frozen paid_prediction, and paid_evaluation with unknown fitting scope.
These statuses are never fresh confirmation; retention does not submit a model.
model.py defines predict(row), where row['features'] contains public controls.
row['history'] contains the preceding public controls within that experiment.
Each row is one scalar supervised example: target labels the current step after
that history, not every step in the history. Earlier steps have their own rows
and targets; history itself contains controls, not response labels.
Optional fit(observations) gets paid response rows, never terminal targets.
fit must update the state used by predict; its return value is ignored.
finish {model:'model.py',lesson:'...',manifest:'shared/knowledge.json'} freezes
the model and ends learning. Only finish makes a terminal model available.
finish first tests that frozen model on limits.confirmation_rows new observations,
charged to the same observation budget; reserve that many observations.
Fresh confirmation passes only when mean squared error (MSE) is at most 1e-10.
Reducing error relative to another model is not sufficient to meet this criterion.
Failed confirmation returns counterexamples and you may revise if budget remains.
fresh_confirmation.residuals preserves up to three largest prediction errors from
that confirmation, with absolute paid-row indices and the frozen submitted model.
Do not import hegel_machine, read this study's repository/artifacts, or access
other arms. Those contain private implementation and held-out task parameters.
No tools provide private settings or unpurchased target labels. When the task
explicitly lists shared fitting operators, those operations use paid evidence
and their execution is charged to the same limits in every knowledge condition.

A learned experimental program can be called through query {path:'shared/plan.py',
question:{id:'q1',question:'The actual current research question',parent_id:null,
depends_on:[]}}. This records the question before its experiments are performed.
The file must define select_actions(public_spec,observations,memory), returning
an action list, or {'actions': [...], 'knowledge_sources': [ids]}. It is executed
on the actual current paid observations; resulting acquisitions are charged.
The memory argument is the inherited snapshot; observations contains current
paid laboratory records. Temporal shared-operator programs can additionally read
research_state.json, refreshed before invocation, for actual chronological
operator_results, current limits, and current costs. Each operator_results entry
has index, arguments, operator, and its feedback fields (such as candidates or
error) at the same level, matching an after_observation_results entry plus index.
run {path:'shared/file.py'} executes an ordinary script. Both paths are available
to every arm. You can create and revise such programs during the current task.
List genuinely consulted knowledge IDs in action arguments knowledge_sources.
These references are agent-authored; executed file paths are logged separately.

For the hegel condition, retain your executable hypotheses and research methods
in shared/. Write shared/knowledge.json as an object with hypotheses and
research_programs NONEMPTY lists and a support_questions list. Give entries stable IDs,
dependencies, and actual observation indices as evidence. Each hypothesis has a
path to its real Python predictor; each research_program has a path to its real
select_actions Python file. At least one hypothesis path must be the model you
finish with. These referenced files and shared/ files persist to the next task.
The manifest is an editable research record, not proof of correctness. Reuse or
revise its actual programs on later tasks when useful; record failed assumptions.
During source learning, Hegel must actually query at least once through an authored
select_actions program under an explicit current question; save that program.
Do not invent child questions for a problem already resolved by paid evidence.
Textual memory receives only earlier agent-authored lessons, not earlier code.
Code/ReAct receives no previous-task knowledge. Task-local tools remain equal.
"""

INTERFACE = POINT_LAB_INTERFACE + COMMON_INTERFACE


_PLAN_RUNNER = """import contextlib,json,runpy,sys
payload=json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    code=runpy.run_path(sys.argv[1])
    result=code['select_actions'](payload['spec'],payload['observations'],payload['memory'])
print(json.dumps(result,allow_nan=False))
"""


class InquiryLLMAgent:
    """A task-local real tool loop; persistence is determined by its caller."""

    common_interface = COMMON_INTERFACE

    def __init__(self, provider, workspace, *, arm, config=None, knowledge=None,
                 confirmation_seed=9173, source_task=True, paid_trace_references=None):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.provider, self.workspace, self.arm = provider, Path(workspace).resolve(), arm
        self.config = config or InquiryLLMConfig()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.knowledge = deepcopy(knowledge or {})
        self.files = dict(self.knowledge.get("files", {}))
        for name, source in self.files.items():
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if paid_trace_references and name in paid_trace_references:
                _copy_model_file(paid_trace_references[name], path)
            else:
                write_text_incrementally(path, source)
        self.inherited_files = set(self.files)
        self.events, self.observations, self.rows = [], [], []
        self.model, self.lesson, self.manifest = None, "", {}
        self.evaluated_model = None
        self.confirmations = []
        self.questions, self.source_task = [], source_task
        self.revisions = []
        self.confirmation_rng = Random(confirmation_seed)
        self.costs = {"observation_cost": 0, "measurement_cost": 0, "compute_cost": 0,
                      "python_execution_seconds": 0.0, "model_calls": 0,
                      "reported_tokens": 0}
        self._write_evidence()

    def _write_evidence(self):
        records = [asdict(row) for row in self.rows]
        text = json.dumps(records)
        write_text_incrementally(self.workspace / "observations.json", text)
        write_text_incrementally(self.workspace / "model_observations.json", text,
                                 base=self.workspace / "observations.json")
        write_text_incrementally(self.workspace / "laboratory_observations.json",
                                 json.dumps(self.observations))

    def _python(self, command, payload="", *, timeout_seconds=None):
        remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining <= 0:
            raise ValueError("Python execution time budget exhausted")
        timeout = remaining
        if timeout_seconds is not None:
            timeout_seconds = float(timeout_seconds)
            if not isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive and finite")
            timeout = min(timeout_seconds, remaining)
        started = monotonic()
        try:
            return _run(command, payload, timeout, str(self.workspace))
        finally:
            self.costs["python_execution_seconds"] += monotonic() - started

    def _program_state(self):
        """Let specialized research interfaces contribute to the same state."""
        compact_agent_comparisons(self)
        return {'questions': self.questions, 'revisions': self.revisions,
                'limits': asdict(self.config), 'costs': self.costs,
                'evaluated_model': self._evaluated_summary()}

    def _prepare_program_state(self):
        with checkpoint_execution():
            state = self._program_state()
            write_json_checkpoint(self.workspace / 'research_state.json', state, indent=None)

    def _program_memory(self, path, arguments):
        """Give programs their actual invocation alongside inherited knowledge."""
        return {**self.knowledge, 'program_path': path,
                'program_arguments': deepcopy(arguments)}

    def _allows_analysis_only_query(self, continuation):
        return False

    def _handle_program_proposal(self, proposed, laboratory):
        """Specialized executors may handle an explicit program submission."""
        return None

    def _open_question(self, question, experiment_path=None):
        record = {**deepcopy(question), "opened_before_observation": self.costs["observation_cost"],
                  "evidence": [], "status": "open", "experiment_path": experiment_path}
        self.questions.append(record)
        return record

    def _query(self, arguments, laboratory):
        path = arguments.get("path")
        executed, sources = None, list(arguments.get("knowledge_sources", []))
        question = arguments.get("question")
        revision = deepcopy(arguments.get('revision'))
        continuation = deepcopy(arguments.get("after_observation", []))
        continuation_origin = "arguments" if "after_observation" in arguments else None
        program_feedback = {}
        if path:
            self._prepare_program_state()
            before = self.costs["python_execution_seconds"]
            code, stdout, stderr, timed_out = self._python(
                [sys.executable, "-c", _PLAN_RUNNER, str(self.workspace / path)],
                json.dumps({"spec": laboratory.public_spec(),
                            "observations": self.observations,
                            "memory": self._program_memory(path, arguments)}))
            # Capture the program itself before executing its proposed action.
            # A returned stop may follow a substantial computation, while an
            # evaluate/query proposal performs additional work in its handler.
            program_feedback = {"program_execution": {
                "path": path, "returncode": code, "timeout": timed_out,
                "python_execution_seconds": self.costs["python_execution_seconds"] - before,
                **({"stderr": stderr[-12000:]} if stderr else {})}}
            if code:
                return {"returncode": code, "stderr": stderr[-12000:], "timeout": timed_out,
                        "executed_path": path, "inherited_program": path in self.inherited_files,
                        **program_feedback}
            proposed = json.loads(stdout)
            handled = self._handle_program_proposal(proposed, laboratory)
            if handled is not None:
                return {**handled, **program_feedback}
            actions = proposed.get("actions", []) if isinstance(proposed, dict) else proposed
            if isinstance(proposed, dict):
                sources.extend(proposed.get("knowledge_sources", []))
                question = question or proposed.get("question")
                if 'revision' in proposed and 'revision' not in arguments:
                    revision = deepcopy(proposed['revision'])
                if "after_observation" in proposed and "after_observation" not in arguments:
                    continuation = deepcopy(proposed["after_observation"])
                    continuation_origin = path
            executed = path
        elif "actions" in arguments:
            actions = arguments["actions"]
        else:
            kind = arguments.get("kind", "response")
            actions = [{"kind": kind, **({"level": point["level"]} if kind == "calibrate"
                                          else {"point": point})}
                       for point in arguments.get("sequence", [])]
        analysis_only = bool(path) and actions == [] and self._allows_analysis_only_query(continuation)
        if not isinstance(actions, list) or (not actions and not analysis_only):
            raise ValueError("query requires a nonempty action list")
        if path and self.arm == "hegel" and (not isinstance(question, dict) or not question.get("id")
                                             or not question.get("question")):
            return {"error": "state the live question (id, question, parent_id, depends_on) before program acquisition",
                    "executed_path": path, **self._provenance(path), **program_feedback}
        question_record = None
        if question:
            question_record = self._open_question(question, executed)
            if continuation_origin is not None:
                question_record.update(after_observation=deepcopy(continuation),
                                       continuation_origin=continuation_origin)
        requested = sum(laboratory.action_cost(action) for action in actions)
        if requested > self.config.observation_budget - self.costs["observation_cost"]:
            return {"error": "observation budget exhausted", "requested": requested,
                    "executed_path": executed, "inherited_program": executed in self.inherited_files,
                    "knowledge_sources": sources, "question": deepcopy(question_record),
                    **program_feedback}
        revision_record = None
        if revision is not None:
            if isinstance(revision, dict) and "readout" in arguments:
                revision.setdefault("readout", deepcopy(arguments["readout"]))
            revision_record = prepare_revision(self, revision, actions, laboratory, experiment_path=executed)
            if question_record is not None:
                revision_record.update(question_id=question_record['id'],
                    question_meaning_version=question_record.get('meaning_version', 1))
            if revision_record.get('error'):
                self._prepare_program_state()
                return {'error': revision_record['error'], 'revision': deepcopy(revision_record),
                        'executed_path': executed, 'question': deepcopy(question_record), **program_feedback}
        acquired, record_indices = [], []
        # Record each successful acquisition immediately, even if a later action
        # in the same requested batch is invalid.
        for action in actions:
            try:
                record = laboratory.execute(action)
            except (ValueError, KeyError, TypeError) as error:
                feedback = {"observations": acquired, "error": str(error),
                        "executed_path": executed, "inherited_program": executed in self.inherited_files,
                        "knowledge_sources": sources, "record_indices": record_indices,
                        "after_observation": continuation, "continuation_origin": continuation_origin,
                        "question": deepcopy(question_record), **program_feedback}
                if revision_record is not None:
                    feedback['revision'] = complete_revision(self, revision_record, acquired, record_indices,
                        laboratory, question_record, error=str(error))
                    feedback['question'] = deepcopy(question_record)
                return feedback
            record_indices.append(len(self.observations))
            self.observations.append(record)
            acquired.append(record)
            self.costs["observation_cost"] += laboratory.action_cost(action)
            self.costs["measurement_cost"] += record.get("measurement_cost", 0)
            if question_record is not None:
                question_record["evidence"].extend(record.get("row_indices", [record["observation_index"]]))
                question_record["status"] = "observed"
            self.rows.extend(laboratory.observation_rows(record))
            self._write_evidence()
        feedback = {"observations": acquired, "observation_cost": requested,
                "executed_path": executed, **self._provenance(executed),
                "question": deepcopy(question_record),
                "record_indices": record_indices, "after_observation": continuation,
                "continuation_origin": continuation_origin,
                "knowledge_sources": sources, **program_feedback}
        if revision_record is not None:
            feedback['revision'] = complete_revision(self, revision_record, acquired, record_indices,
                laboratory, question_record)
            feedback['question'] = deepcopy(question_record)
        return feedback

    def _provenance(self, path):
        inherited = path in self.inherited_files
        unchanged = inherited and (self.workspace / path).read_text() == self.knowledge["files"][path]
        return {"inherited_program": inherited, "inherited_source_unchanged": bool(unchanged)}

    def _score(self, model):
        if not self.rows:
            raise ValueError("buy response observations before model evaluation")
        if self.costs["compute_cost"] + len(self.rows) > self.config.compute_budget:
            raise ValueError("candidate-row scoring budget exhausted")
        remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining <= 0:
            raise ValueError("Python execution time budget exhausted")
        self.costs["compute_cost"] += len(self.rows)
        started = monotonic()
        result = evaluate_python_model(self.workspace, model, self.rows, timeout_seconds=remaining,
            research_compute_budget=max(0, self.config.compute_budget - self.costs['compute_cost']))
        self.costs['compute_cost'] += result.get('research_prediction_count', 0)
        self.costs["python_execution_seconds"] += monotonic() - started
        predictions = result.pop("predictions", None)
        if predictions is not None:
            result.update(_paid_prediction_scope(self.workspace, model, self.rows, predictions))
            result['paid_residuals'] = _prediction_residuals(
                self.rows, predictions, model, evaluation_scope='paid_response_evaluation')
            relations = compare_paid_relations(self.revisions, self.rows, predictions,
                compute_budget=max(0, self.config.compute_budget - self.costs['compute_cost']))
            if relations['check_count'] or relations['truncated']:
                relations.update(model=model, workspace=str(self.workspace),
                                 evaluation_scope=result['evaluation_scope'])
                result['paid_relation_diagnostic'] = relations
            self.costs['compute_cost'] += relations['term_count']
        return result

    def _evaluated_summary(self):
        retained = self.evaluated_model
        if retained is None:
            return None
        summary = {"model": str(Path("_evaluated_model") / retained["model"]),
                "source_model": retained.get("source_model", retained["model"]), "mse": retained["current_mse"],
                "evaluated_rows": retained["evaluated_rows"], "current_paid_rows": len(self.rows),
                "evaluation_scope": retained.get('evaluation_scope', 'paid_evaluation'),
                "fitting_view": deepcopy(retained.get('fitting_view', {'scope': 'unknown'})),
                "support": (retained.get('evaluation_scope', 'paid_evaluation')
                            if retained["evaluated_rows"] == len(self.rows) else "new_paid_rows_unscored")}
        if 'later_readings' in retained:
            summary['later_readings'] = deepcopy(retained['later_readings'])
        if 'selection' in retained:
            summary['selection'] = deepcopy(retained['selection'])
        return summary

    def _retain_evaluation(self, model, result):
        """Keep measured improvements while preserving the earlier fitting data."""
        if not self.rows or result["returncode"] != 0 or not isinstance(result.get("mse"), (int, float)):
            return
        candidate = {'model': model, 'mse': result['mse'], 'evaluated_rows': len(self.rows),
                     **{key: deepcopy(result[key]) for key in
                        ('evaluation_scope', 'fitting_view') if key in result}}

        def incumbent_summary():
            summary = self._evaluated_summary()
            return None if summary is None else {key: deepcopy(summary[key]) for key in
                ('model', 'source_model', 'mse', 'evaluated_rows', 'evaluation_scope', 'fitting_view')}

        def record_decision(status, incumbent):
            selected = self._evaluated_summary()
            result['retention_decision'] = {
                'status': status, 'candidate': deepcopy(candidate), 'incumbent': incumbent,
                'selected_model': selected['model'] if selected is not None else None,
                'same_paid_rows': (incumbent['evaluated_rows'] == len(self.rows)
                                   if incumbent is not None else None),
                'scope': 'Paid scoring and retention only; no independent confirmation.'}

        retained = self.evaluated_model
        incumbent = incumbent_summary()
        if retained is not None:
            retained_path = self._evaluated_summary()["model"]
            if model == retained_path:
                retained.update(current_mse=result["mse"], evaluated_rows=len(self.rows))
                _remember_paid_scope(retained, result)
                record_decision('retained_rescored', incumbent)
                return
            if retained["evaluated_rows"] != len(self.rows):
                if (self.config.compute_budget - self.costs["compute_cost"] < len(self.rows)
                        or self.config.execution_seconds <= self.costs["python_execution_seconds"]):
                    result["retention_deferred"] = "insufficient budget to compare the saved candidate on current paid rows"
                    record_decision('deferred', incumbent)
                    return
                comparison = self._score(retained_path)
                result["retained_comparison"] = comparison
                if comparison.get('mse') is None:
                    result['retention_deferred'] = 'saved candidate comparison did not finish on current paid rows'
                    record_decision('deferred', incumbent)
                    return
                retained.update(current_mse=comparison.get("mse"), evaluated_rows=len(self.rows))
                _remember_paid_scope(retained, comparison)
                incumbent = incumbent_summary()
            if retained["current_mse"] is not None and result["mse"] >= retained["current_mse"]:
                record_decision('incumbent_retained', incumbent)
                return
        execution = result.get('execution_context')
        if execution is not None:
            self.evaluated_model = retain_evaluated_python_model(
                execution['workspace'], execution['model'], [], result['mse'],
                destination=self.workspace / '_evaluated_model')
            self.evaluated_model['source_model'] = model
        else:
            self.evaluated_model = retain_evaluated_python_model(
                self.workspace, model, [], result["mse"])
        self.evaluated_model.update(current_mse=result["mse"], evaluated_rows=len(self.rows))
        _remember_paid_scope(self.evaluated_model, result)
        record_decision('candidate_retained', incumbent)
        self.evaluated_model['selection'] = deepcopy(result['retention_decision'])

    def _execute(self, action, arguments, laboratory):
        if action == "query":
            return self._query(arguments, laboratory)
        if action == "write":
            path = self.workspace / arguments["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = path.read_text(encoding="utf-8") if path.exists() else None
            path.write_text(arguments["code"], encoding="utf-8")
            self.files[arguments["path"]] = arguments["code"]
            return {"written": arguments["path"], "revised_inherited_file":
                    arguments["path"] in self.inherited_files and previous != arguments["code"]}
        if action == "run":
            path = arguments.get("path", f"scratch_{len(self.events):03d}.py")
            if "code" in arguments:
                (self.workspace / path).parent.mkdir(parents=True, exist_ok=True)
                (self.workspace / path).write_text(arguments["code"], encoding="utf-8")
            before = self.costs["python_execution_seconds"]
            code, stdout, stderr, timed_out = self._python(
                [sys.executable, str(self.workspace / path)],
                timeout_seconds=arguments.get("timeout_seconds", self.config.run_timeout_seconds))
            result = {"returncode": code, "stdout": stdout[-12000:], "stderr": stderr[-12000:],
                      "timeout": timed_out, "executed_path": path,
                      "execution_seconds": self.costs["python_execution_seconds"] - before,
                      "remaining_execution_seconds": max(0., self.config.execution_seconds
                          - self.costs["python_execution_seconds"]),
                      **self._provenance(path)}
            if timed_out:
                result["execution_notice"] = (
                    "This computation reached its execution time limit. This is not a "
                    "scientific evaluation of the model. Files already written persist; "
                    "you can inspect progress, revise the computation, or explicitly "
                    "request a longer timeout_seconds within the remaining allowance.")
            truncated = []
            for stream, output in (("stdout", stdout), ("stderr", stderr)):
                output_path = f"{path}.{stream}.txt"
                (self.workspace / output_path).write_text(output, encoding="utf-8")
                result.update({f"{stream}_file": output_path,
                               f"{stream}_characters": len(output),
                               f"{stream}_truncated": len(output) > 12000})
                if len(output) > 12000:
                    truncated.append(stream)
                    if stream == 'stdout':
                        marker = '\n... middle omitted; complete output is saved ...\n'
                        tail = 12000 - 6000 - len(marker)
                        result[stream] = output[:6000] + marker + output[-tail:]
            if truncated:
                result["output_notice"] = (
                    "Output previews are limited to 12000 characters: stdout keeps the "
                    "beginning and end, while stderr keeps the end. Complete output is "
                    "saved in the corresponding *_file; "
                    "read a selected range or compute a concise summary from that file.")
            return result
        if action in {"evaluate", "finish"}:
            model = arguments.get("model", "model.py")
            result = self._score(model)
            result.update(executed_path=model, **self._provenance(model))
            self._retain_evaluation(model, result)
            result["evaluated_model"] = self._evaluated_summary()
            if action == "finish" and result["returncode"] == 0 and not result.get('truncated'):
                precondition = getattr(laboratory, "confirmation_precondition", None)
                if callable(precondition):
                    error = precondition(result, mse_threshold=self.config.confirmation_mse_threshold)
                    if error:
                        result["error"] = error
                        return result
                manifest_path = self.workspace / (arguments.get("manifest") or "shared/knowledge.json")
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                if self.arm == "hegel":
                    error = self._manifest_error(manifest, model)
                    if error:
                        result["error"] = error
                        return result
                    self.manifest = manifest
                n = self.config.confirmation_rows
                if self.costs["observation_cost"] + n > self.config.observation_budget:
                    result["error"] = "reserve observation budget for fresh confirmation"
                    return result
                if self.costs["compute_cost"] + n > self.config.compute_budget:
                    result["error"] = "reserve candidate scoring budget for fresh confirmation"
                    return result
                # A method may overwrite the same predictor on its next call.
                # Keep this submitted explanation and its fitting view before
                # new labels arrive, so it remains executable after a revision.
                destination = Path("_submitted_models") / f"confirmation_{len(self.confirmations):04d}" / "_evaluated_model"
                retained = retain_evaluated_python_model(
                    self.workspace, model, [], result.get("mse"), destination=destination)
                submitted = {"model": str(destination / retained["model"]),
                    "source_model": model, "paid_rows": len(self.rows), "mse": result.get("mse")}
                result["submitted_candidate"] = submitted
                actions = laboratory.confirmation_actions(n, self.confirmation_rng)
                # The executing candidate may read any ordinary training file.
                # Keep new labels entirely outside that workspace until scored.
                acquired = [laboratory.execute(action) for action in actions]
                self.costs["observation_cost"] += n
                rows = [row for record in acquired for row in laboratory.observation_rows(record)]
                self.costs["compute_cost"] += n
                started = monotonic()
                remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
                fresh = evaluate_python_model(self.workspace, model, rows, timeout_seconds=max(.001, remaining),
                    research_compute_budget=max(0, self.config.compute_budget - self.costs['compute_cost']))
                self.costs['compute_cost'] += fresh.get('research_prediction_count', 0)
                self.costs["python_execution_seconds"] += monotonic() - started
                predictions = fresh.pop("predictions", None)
                if predictions is not None:
                    fresh['residuals'] = _prediction_residuals(
                        rows, predictions, submitted['model'], evaluation_scope='fresh_confirmation',
                        row_offset=len(self.rows))
                passed = (fresh.get("returncode") == 0
                          and fresh.get("mse", float("inf")) <= self.config.confirmation_mse_threshold)
                result["fresh_confirmation"] = {**fresh, "passed": passed, "rows": n,
                    "submitted_candidate": deepcopy(submitted),
                    "mse_threshold": self.config.confirmation_mse_threshold,
                    "observation_indices": [index for record in acquired for index in
                        record.get("row_indices", [record["observation_index"]])],
                    "observations": acquired}
                self.confirmations.append(result["fresh_confirmation"])
                self.observations.extend(acquired)
                self.rows.extend(rows)
                if passed:
                    self.model, self.lesson, self.manifest = model, arguments.get("lesson", ""), manifest
                else:
                    self._write_evidence()
                    result["error"] = (
                        "fresh confirmation failed; new counterexamples are now paid evidence"
                        if fresh.get("returncode") == 0 and fresh.get("mse") is not None else
                        "fresh confirmation could not be scored; acquired observations are now paid evidence")
            return result
        raise ValueError(f"unknown action: {action}")

    def _manifest_error(self, manifest, model):
        if not isinstance(manifest, dict):
            return "the research manifest must be a JSON object"
        if not all(isinstance(manifest.get(key), list) and manifest[key]
                   for key in ("hypotheses", "research_programs")) or not isinstance(manifest.get("support_questions"), list):
            return "Hegel memory requires nonempty hypotheses/research_programs and a support_questions list"
        entries = [*manifest["hypotheses"], *manifest["research_programs"]]
        if any(not isinstance(entry, dict) or not entry.get("path") or
               not (self.workspace / entry["path"]).is_file() for entry in entries):
            return "each hypothesis/research_program must reference an existing executable Python path"
        if model not in [entry["path"] for entry in manifest["hypotheses"]]:
            return "a hypothesis path must identify the actual submitted model"
        if self.source_task and not any(q["evidence"] and q["experiment_path"] in
                [entry["path"] for entry in manifest["research_programs"]] for q in self.questions):
            return "source learning must actually execute a retained select_actions program under a live question"
        return None

    def _request(self, laboratory):
        """Expose the current executable research state to the next model turn."""
        self._prepare_program_state()
        interface = getattr(laboratory, "task_interface", POINT_LAB_INTERFACE) + self.common_interface
        interface = interface.replace(
            "mean squared error (MSE) is at most 1e-10.",
            f"mean squared error (MSE) is at most {self.config.confirmation_mse_threshold}.")
        if callable(getattr(laboratory, 'prediction_rows', None)):
            interface += REVISION_INTERFACE
        return {"capabilities": laboratory.public_spec(), "task_interface": interface,
                "knowledge_condition": self.arm, "limits": asdict(self.config),
                "costs": deepcopy(self.costs), "observations": [asdict(row) for row in self.rows],
                "laboratory_observations": self.observations, "files": self.files,
                "session_knowledge": self.knowledge, "events": self.events,
                "evaluated_model": self._evaluated_summary()}

    def _before_model_turn(self, laboratory):
        pass

    def _after_model_turn(self, laboratory):
        pass

    def _terminal_status(self):
        return None

    def run(self, laboratory, *, on_checkpoint=None):
        status = "model_call_budget_exhausted"
        while True:
            terminal = self._terminal_status()
            if terminal:
                status = terminal
                break
            opened_research = self._before_model_turn(laboratory)
            if opened_research and on_checkpoint:
                on_checkpoint(self.result("researching"))
            if self.costs['model_calls'] >= self.config.max_model_calls:
                break
            step = self.costs['model_calls']
            if (not self.model and self.config.observation_budget - self.costs["observation_cost"]
                    < self.config.confirmation_rows):
                status = "confirmation_budget_exhausted"
                break
            if self.costs["python_execution_seconds"] >= self.config.execution_seconds:
                status = "python_execution_budget_exhausted"
                break
            if self.costs["reported_tokens"] >= self.config.token_budget:
                status = "token_budget_exhausted"
                break
            request = self._request(laboratory)
            self.costs["model_calls"] += 1
            decision = self.provider.next_action(request)
            usage = decision.get("usage", {})
            tokens = usage.get("total_tokens")
            if isinstance(tokens, int):
                self.costs["reported_tokens"] += tokens
            if decision["action"] == "provider_error":
                self.events.append({"step": step, **decision})
                if "decode_error" in decision.get("error", {}) and isinstance(tokens, int):
                    self.events[-1]["feedback"] = {
                        "error": decision["error"]["decode_error"],
                        "instruction": "Return exactly one research_action with action and arguments_json; "
                                       "arguments_json must encode the action's argument object. "
                                       "No action from the invalid reply was executed. This call was charged."}
                    self._after_model_turn(laboratory)
                    if on_checkpoint:
                        on_checkpoint(self.result("researching"))
                    continue
                if not isinstance(tokens, int):
                    self.costs["reported_tokens"] = None
                status = "provider_failed"
                break
            try:
                feedback = self._execute(decision["action"], decision.get("arguments", {}), laboratory)
            except (ValueError, TypeError, KeyError, OSError) as error:
                feedback = {"error": f"{type(error).__name__}: {error}"}
            self.events.append({"step": step, "action": decision["action"],
                                "arguments": decision.get("arguments", {}),
                                "feedback": feedback, "usage": usage})
            self._after_model_turn(laboratory)
            if on_checkpoint:
                on_checkpoint(self.result("researching"))
            if not isinstance(tokens, int):
                self.costs["reported_tokens"] = None
                status = "token_usage_unavailable"
                break
            if self.model:
                status = "completed"
                break
        return self.result(status)

    def _shared_files(self):
        shared = self.workspace / "shared"
        return {str(path.relative_to(self.workspace)): path.read_text(encoding="utf-8")
                for path in shared.rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}}

    def _base_result_files(self):
        files = self._shared_files()
        for entry in [*self.manifest.get("hypotheses", []), *self.manifest.get("research_programs", [])]:
            if isinstance(entry, dict) and entry.get("path") and (self.workspace / entry["path"]).is_file():
                files[entry["path"]] = (self.workspace / entry["path"]).read_text(encoding="utf-8")
        return files

    def result(self, status):
        return self._result_with_snapshot(status, deepcopy)

    def _result_with_snapshot(self, status, snapshot):
        """Build either an owned result or a synchronously borrowed checkpoint."""
        compact_agent_comparisons(self)
        files = self._base_result_files()
        if self.arm == "code_react":
            outgoing = {}
        elif self.arm == "textual_memory":
            outgoing = {"lessons": [*self.knowledge.get("lessons", []), self.lesson]}
        else:
            outgoing = {"files": files, "manifest": snapshot(self.manifest),
                        "question_traces": snapshot(self.questions),
                        "lessons": [*self.knowledge.get("lessons", []), self.lesson]}
        return {"agent_arm": self.arm, "status": status, "workspace": str(self.workspace),
                "model": self.model, "costs": snapshot(self.costs), "events": snapshot(self.events),
                "evaluated_model": self._evaluated_summary(),
                "inherited_files": sorted(self.inherited_files), "outgoing_knowledge": outgoing,
                "inherited_files_changed": sorted(name for name in self.inherited_files
                    if name in files and files[name] != self.knowledge["files"][name]),
                "fresh_confirmation": snapshot(self.confirmations[-1]) if self.confirmations else None,
                "paid_observations": snapshot(self.observations),
                "online_questions": snapshot(self.questions),
                "revision_proposals": snapshot(self.revisions),
                "limits": asdict(self.config), "token_limit_kind": "reported_usage_between_calls",
                "compute_limit_kind": "host_candidate_row_scoring; agent Python bounded by execution_seconds"}


def run_chain(tasks, *, arm, workspace, provider_factory, lab_factory, config=None,
              on_checkpoint=None, agent_class=InquiryLLMAgent):
    """Train one source and independent consumers cloned from its frozen memory."""
    results, source_memory = [], {}
    for index, task in enumerate(tasks):
        path = Path(workspace) / f"task_{index:03d}"
        agent = agent_class(provider_factory(path), path, arm=arm, config=config,
                                knowledge=source_memory if index else {},
                                confirmation_seed=task.get("confirmation_seed", task["seed"] + 7919),
                                source_task=index == 0)
        result = agent.run(lab_factory(task), on_checkpoint=on_checkpoint)
        result["task_position"] = index
        results.append(result)
        if index == 0:
            source_memory = deepcopy(result["outgoing_knowledge"])
        if result["status"] in {"provider_failed", "token_usage_unavailable"}:
            break
    return {"arm": arm, "tasks": results, "source_memory": source_memory,
            "training_complete": len(results) == len(tasks), "evaluation": None}


def evaluate_chain(chain, tasks, *, lab_factory, length=256, seed_offset=1000003,
                   timeout_seconds=60):
    """Call only after every comparison arm has stopped learning."""
    scores = []
    for task, result in zip(tasks, chain["tasks"]):
        if not result["model"]:
            scores.append({"task_position": result["task_position"], "status": "missing_model",
                           "mse": None, "rows": 0})
            continue
        seed = task.get("evaluation_seed", task["seed"] + seed_offset)
        rows = lab_factory(task).evaluation_rows(length=length, seed=seed)
        score = evaluate_python_model(result["workspace"], result["model"], rows,
                                      timeout_seconds=timeout_seconds)
        score.pop("predictions", None)
        scores.append({**score, "task_position": result["task_position"], "rows": len(rows),
                       "evaluation_seed": seed,
                       "status": "completed" if score["returncode"] == 0 else "model_failed"})
    chain["evaluation"] = scores
    return scores
