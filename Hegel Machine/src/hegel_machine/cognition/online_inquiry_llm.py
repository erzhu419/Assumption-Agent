"""The same external Python agent with three persistent-knowledge conditions.

The laboratory never supplies a solver to this loop.  All fitting, candidate
selection, experiment programs and memory contents are authored by the provider.
The existing code provider and Python predictor executor are shared by all arms.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from random import Random
import sys
from time import monotonic

from .code_agent import evaluate_python_model, retain_evaluated_python_model
from .models import Observation
from .proposers import _run
from .inquiry_revision import REVISION_INTERFACE, prepare_revision, complete_revision


ARMS = ("code_react", "textual_memory", "hegel")


@dataclass(frozen=True)
class InquiryLLMConfig:
    observation_budget: int = 24
    max_model_calls: int = 16
    token_budget: int = 60000
    execution_seconds: float = 60.0
    compute_budget: int = 100000
    confirmation_rows: int = 3
    confirmation_mse_threshold: float = 1e-10


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
The code key contains Python source; path names a file, not a shell command.
Python execution has numpy/scipy available and returns stdout/stderr and exit code.
For example, run {code:"import json; rows=json.load(open('observations.json')); print(len(rows))"}
inspects paid data. All paid response rows (features, target, history) are in
observations.json and model_observations.json. Auxiliary experiment records are
in laboratory_observations.json. Files and concise numerical results persist.
evaluate {model:'model.py'} fits/scores ONLY already purchased response rows.
An explicit successful evaluate retains the better executed candidate in
evaluated_model.model. That path has its own saved files and fitting data; you
can evaluate or finish it even after revising the original file. Candidates
compete on the same paid rows. New observations require charged rescoring of
the saved candidate before replacement. evaluated_model.support describes paid
fit evidence, never fresh confirmation; retention does not submit a model.
model.py defines predict(row), where row['features'] contains public controls.
row['history'] contains the preceding public controls within that experiment.
Optional fit(observations) gets paid response rows, never terminal targets.
fit must update the state used by predict; its return value is ignored.
finish {model:'model.py',lesson:'...',manifest:'shared/knowledge.json'} freezes
the model and ends learning. Only finish makes a terminal model available.
finish first tests that frozen model on limits.confirmation_rows new observations,
charged to the same observation budget; reserve that many observations.
Fresh confirmation passes only when mean squared error (MSE) is at most 1e-10.
Reducing error relative to another model is not sufficient to meet this criterion.
Failed confirmation returns counterexamples and you may revise if budget remains.
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
                 confirmation_seed=9173, source_task=True):
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
            path.write_text(source, encoding="utf-8")
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
        for filename in ("observations.json", "model_observations.json"):
            (self.workspace / filename).write_text(json.dumps(records), encoding="utf-8")
        (self.workspace / "laboratory_observations.json").write_text(
            json.dumps(self.observations), encoding="utf-8")

    def _python(self, command, payload=""):
        remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining <= 0:
            raise ValueError("Python execution time budget exhausted")
        started = monotonic()
        result = _run(command, payload, remaining, str(self.workspace))
        self.costs["python_execution_seconds"] += monotonic() - started
        return result

    def _prepare_program_state(self):
        """Specialized research interfaces may expose their current feedback."""
        state = {'questions': self.questions, 'revisions': self.revisions,
                 'limits': asdict(self.config), 'costs': self.costs,
                 'evaluated_model': self._evaluated_summary()}
        (self.workspace / 'research_state.json').write_text(json.dumps(state), encoding='utf-8')

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
        if path:
            self._prepare_program_state()
            code, stdout, stderr, timed_out = self._python(
                [sys.executable, "-c", _PLAN_RUNNER, str(self.workspace / path)],
                json.dumps({"spec": laboratory.public_spec(),
                            "observations": self.observations, "memory": self.knowledge}))
            if code:
                return {"returncode": code, "stderr": stderr[-12000:], "timeout": timed_out,
                        "executed_path": path, "inherited_program": path in self.inherited_files}
            proposed = json.loads(stdout)
            handled = self._handle_program_proposal(proposed, laboratory)
            if handled is not None:
                return handled
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
                    "executed_path": path, **self._provenance(path)}
        requested = sum(laboratory.action_cost(action) for action in actions)
        if requested > self.config.observation_budget - self.costs["observation_cost"]:
            return {"error": "observation budget exhausted", "requested": requested,
                    "executed_path": executed, "inherited_program": executed in self.inherited_files,
                    "knowledge_sources": sources}
        revision_record = None
        if revision is not None:
            revision_record = prepare_revision(self, revision, actions, laboratory, experiment_path=executed)
            if revision_record.get('error'):
                self._prepare_program_state()
                return {'error': revision_record['error'], 'revision': deepcopy(revision_record),
                        'executed_path': executed}
        acquired, record_indices = [], []
        question_record = None
        if question:
            question_record = self._open_question(question, executed)
            if continuation_origin is not None:
                question_record.update(after_observation=deepcopy(continuation),
                                       continuation_origin=continuation_origin)
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
                        "question": deepcopy(question_record)}
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
                "knowledge_sources": sources}
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
        result = evaluate_python_model(self.workspace, model, self.rows, timeout_seconds=remaining)
        self.costs["python_execution_seconds"] += monotonic() - started
        result.pop("predictions", None)
        return result

    def _evaluated_summary(self):
        retained = self.evaluated_model
        if retained is None:
            return None
        return {"model": str(Path("_evaluated_model") / retained["model"]),
                "source_model": retained.get("source_model", retained["model"]), "mse": retained["current_mse"],
                "evaluated_rows": retained["evaluated_rows"], "current_paid_rows": len(self.rows),
                "support": ("paid_fit" if retained["evaluated_rows"] == len(self.rows)
                            else "new_paid_rows_unscored")}

    def _retain_evaluation(self, model, result):
        """Keep measured improvements while preserving the earlier fitting data."""
        if not self.rows or result["returncode"] != 0 or not isinstance(result.get("mse"), (int, float)):
            return
        retained = self.evaluated_model
        if retained is not None:
            retained_path = self._evaluated_summary()["model"]
            if model == retained_path:
                retained.update(current_mse=result["mse"], evaluated_rows=len(self.rows))
                return
            if retained["evaluated_rows"] != len(self.rows):
                if (self.config.compute_budget - self.costs["compute_cost"] < len(self.rows)
                        or self.config.execution_seconds <= self.costs["python_execution_seconds"]):
                    result["retention_deferred"] = "insufficient budget to compare the saved candidate on current paid rows"
                    return
                comparison = self._score(retained_path)
                result["retained_comparison"] = comparison
                retained.update(current_mse=comparison.get("mse"), evaluated_rows=len(self.rows))
            if retained["current_mse"] is not None and result["mse"] >= retained["current_mse"]:
                return
        self.evaluated_model = retain_evaluated_python_model(
            self.workspace, model, [], result["mse"])
        self.evaluated_model.update(current_mse=result["mse"], evaluated_rows=len(self.rows))

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
            code, stdout, stderr, timed_out = self._python([sys.executable, str(self.workspace / path)])
            result = {"returncode": code, "stdout": stdout[-12000:], "stderr": stderr[-12000:],
                      "timeout": timed_out, "executed_path": path,
                      **self._provenance(path)}
            truncated = []
            for stream, output in (("stdout", stdout), ("stderr", stderr)):
                output_path = f"{path}.{stream}.txt"
                (self.workspace / output_path).write_text(output, encoding="utf-8")
                result.update({f"{stream}_file": output_path,
                               f"{stream}_characters": len(output),
                               f"{stream}_truncated": len(output) > 12000})
                if len(output) > 12000:
                    truncated.append(stream)
            if truncated:
                result["output_notice"] = (
                    "Only the last 12000 characters of " + ", ".join(truncated) +
                    " are shown. Earlier output is saved in the corresponding *_file; "
                    "read a selected range or compute a concise summary from that file.")
            return result
        if action in {"evaluate", "finish"}:
            model = arguments.get("model", "model.py")
            result = self._score(model)
            result.update(executed_path=model, **self._provenance(model))
            if action == "evaluate":
                self._retain_evaluation(model, result)
                result["evaluated_model"] = self._evaluated_summary()
            if action == "finish" and result["returncode"] == 0:
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
                fresh = evaluate_python_model(self.workspace, model, rows, timeout_seconds=max(.001, remaining))
                self.costs["python_execution_seconds"] += monotonic() - started
                fresh.pop("predictions", None)
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

    def run(self, laboratory, *, on_checkpoint=None):
        status = "model_call_budget_exhausted"
        for step in range(self.costs["model_calls"], self.config.max_model_calls):
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

    def result(self, status):
        shared = self.workspace / "shared"
        files = {str(path.relative_to(self.workspace)): path.read_text(encoding="utf-8")
                 for path in shared.rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}}
        for entry in [*self.manifest.get("hypotheses", []), *self.manifest.get("research_programs", [])]:
            if isinstance(entry, dict) and entry.get("path") and (self.workspace / entry["path"]).is_file():
                files[entry["path"]] = (self.workspace / entry["path"]).read_text(encoding="utf-8")
        if self.arm == "code_react":
            outgoing = {}
        elif self.arm == "textual_memory":
            outgoing = {"lessons": [*self.knowledge.get("lessons", []), self.lesson]}
        else:
            outgoing = {"files": files, "manifest": deepcopy(self.manifest),
                        "question_traces": deepcopy(self.questions),
                        "lessons": [*self.knowledge.get("lessons", []), self.lesson]}
        return {"agent_arm": self.arm, "status": status, "workspace": str(self.workspace),
                "model": self.model, "costs": deepcopy(self.costs), "events": deepcopy(self.events),
                "evaluated_model": self._evaluated_summary(),
                "inherited_files": sorted(self.inherited_files), "outgoing_knowledge": outgoing,
                "inherited_files_changed": sorted(name for name in self.inherited_files
                    if name in files and files[name] != self.knowledge["files"][name]),
                "fresh_confirmation": deepcopy(self.confirmations[-1]) if self.confirmations else None,
                "paid_observations": deepcopy(self.observations),
                "online_questions": deepcopy(self.questions),
                "revision_proposals": deepcopy(self.revisions),
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
