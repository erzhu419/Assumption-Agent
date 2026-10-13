"""Continue authored research with a common archive and live question records.

Ordinary Code and Hegel share execution, evidence, notebooks and revision tools.
Hegel additionally keeps its public research root and executable source manifest.
"""

from copy import deepcopy
import json
from pathlib import Path
from random import Random
import shutil

from .online_inquiry_llm import COMMON_INTERFACE
from .anonymous_history_lab import AnonymousHistoryLab
from .code_agent import _copy_model_file, _model_context_ignored
from .temporal_operator_agent import OPERATOR_INTERFACE
from .inquiry_revision import analyze_revision, compare_experiments
from .observable_research import (
    OBSERVABLE_INTERFACE, dependency_focus, evaluate_observable, refresh_observables,
)
from .conditional_research import (
    CONDITIONAL_INTERFACE, evaluate_component, refresh_components,
)
from .temporal_paired_method import (
    FROZEN_METHOD_INTERFACE,
    PROCEDURE_INTERFACE,
    ProcedureConsumerAgent,
)


_HOST_RECORD_FILES = {
    "active.json", "training.json", "archive.json", "evaluation.json",
    "result.json", "protocol.json", "summary.json",
}

_SOURCE_EXECUTION_REQUIREMENT = (
    "During source learning, Hegel must actually query at least once through an authored\n"
    "select_actions program under an explicit current question; save that program.\n"
)

_CONTINUATION_REQUIREMENT = """This task continues an existing executable research archive.
The Hegel manifest may retain its existing research programs. A new program or
query is optional; ordinary evaluation and direct finish are available.
For a consumer task, finish registers your explicitly selected model path in the
current manifest if missing. Existing research entries are preserved. Registration
adds no scientific support: fresh predictive confirmation still determines adoption.
"""


# Reuse the existing mechanism-independent whole-trajectory refit contract.
_REFIT_INTERFACE = OPERATOR_INTERFACE[OPERATOR_INTERFACE.index("run {operator:'refit'"):
    OPERATOR_INTERFACE.index("run {operator:'legacy_capabilities'")]


COMMON_ARCHIVE_INTERFACE = COMMON_INTERFACE.replace(
    "Textual memory receives only earlier agent-authored lessons, not earlier code.\n"
    "Code/ReAct receives no previous-task knowledge. Task-local tools remain equal.\n",
    "Both Code/ReAct and Hegel receive the same supplied archive: executable files,\n"
    "lessons, the manifest and earlier question traces. Read, execute, revise or\n"
    "decline its contents using the same ordinary Python tools. Archive entries\n"
    "can include unsuccessful investigations; their presence is not confirmation.\n"
    "Both conditions retain actual authored code, including unfinished source\n"
    "work. Code/ReAct may organize this archive freely and has no requirement to\n"
    "create a manifest or question hierarchy. Hegel retains an executable manifest\n"
    "under the current research phase's requirements.\n",
) + FROZEN_METHOD_INTERFACE + PROCEDURE_INTERFACE + _REFIT_INTERFACE


ONLINE_RESEARCH_INTERFACE = """
This is a continuing online investigation. Each ordinary action returns to the
same live researcher with its real execution feedback. research_notes is your
own persistent notebook. online_research identifies existing questions, paid
evidence awaiting analysis, and child answers not yet reviewed by their parent.
These are research records, not a host-selected scientific next step.
Use stable question IDs. A question's assignment retains its original purpose;
its question text may become a narrower hypothesis as the investigation proceeds.
Optional research_question on any action associates that decision with a question.
An association records your attention, not scientific support or a changed model.
The actual experiment question may be supplied in query.question as before.
No child or additional observation is required when the paid evidence suffices.

evaluate {model:'shared/candidate.py',code:'ordinary predictor Python source'}
saves exactly your supplied source and immediately executes ordinary paid-data
evaluation and old-candidate retention. Omitting code evaluates the existing file.
This does not submit a model or buy observations. A later finish selects the model
for fresh confirmation. Use actual evaluation and confirmation feedback to decide
whether to revise its source, construct an experiment, or stop.

run {operator:'analyze_revision',revision_index:0,path:'shared/readout.py'}
executes your readout(result,public_spec,memory) on an already purchased revision.
Use it to complete or repair analysis without repeating acquisition. The saved
pre-experiment predictions, sources, paid cutoff, and observations remain fixed;
each executed readout is retained. The resulting answer returns to that revision's
question and parent. You choose whether and how it changes the next research step.
Full questions, revision records and procedure feedback remain in research_state.json.

method_decision.methods keeps each registered program's execution action and
separate prior practice record. After a program's failed confirmation, continuation
identifies the submitted predictor, the new paid evidence and an action to invoke
that program again. It will receive the expanded records. You decide whether to
continue, edit revision_path, change the research route or stop. The failed test
evaluated that predictor; the program's ability to revise it remains a separate
question. A confirmation execution error is not a measured prediction mismatch.
submitted_candidate names the saved predictor and fitting view from before that
confirmation, even if a later method invocation overwrites its original file.
submitted_candidates keeps these attempts after a method stops. A method's stop
reason describes its own attempt; it does not exclude every model in its family.

run {operator:'compare_experiments',target:'response',models:['shared/a.py','shared/b.py'],
experiments:[{id:'an_authored_experiment',actions:[public_action,...]},...]}
executes your candidate predictors on your proposed public actions without buying
observations. It compares paid fit and prospective prediction differences, using
only candidates that meet the existing paid MSE criterion. No paid evidence means
no supported candidate. Failed or refuted candidates remain in the report.
The returned comparison includes each authored experiment's forecast, and may
suggest the one with greatest mean prediction variance. A suggestion is neither
an acquisition nor evidence that its scientific alternatives are exhaustive.
query_arguments contains ordinary actions and revision arguments you can choose
to use with query; you may add your readout and question, change the experiment,
or proceed to confirmation. The suggested query names saved candidate contexts;
query reexecutes those contexts before purchase and charges that new scoring.
Their fitting views stay fixed. A refitted candidate is a new explanation to
evaluate and compare on current paid evidence.
To value an experiment for a prediction task, supply reference_actions:[public_action,...]
in the same comparison. These describe the control histories on which future
predictions matter. They acquire no observations and expose no reference labels.
The existing finite-model risk calculation then ranks expected reference-MSE
reduction per observation, using the same supported candidates and fitting views.
task_value reports the reference risk and each option's remaining expected risk;
equal or zero values are valid. This is conditional on your candidate set and
reference inputs, not proof that a model family is true or exhausted. Offer
different supported parameter fits when the existing evidence cannot identify
one fit; a single fitted instance does not represent every member of its family.
Omit target for the response, or name a public channel such as 'meter'. A channel
comparison uses only purchased readings of that target for paid fit and freezes
them in each new candidate context. Already retained candidates keep their own
fitting views. Every proposed action must buy that channel through channels.
The suggested revision keeps the target and scores its acquired readings; these
auxiliary candidates remain separate from the retained response predictor.
Full sources and predictions persist in research_state.json experiment_comparisons;
online_research.latest_comparison summarizes the most recent comparison.
"""


class CommonArchiveInquiryAgent(ProcedureConsumerAgent):
    """Use the old Python/procedure loop with equal executable persistence."""

    common_interface = (COMMON_ARCHIVE_INTERFACE + ONLINE_RESEARCH_INTERFACE
                        + OBSERVABLE_INTERFACE + CONDITIONAL_INTERFACE)

    def __init__(self, *args, **kwargs):
        if kwargs.get("arm") not in {"code_react", "hegel"}:
            raise ValueError("common archive conditions are code_react and hegel")
        super().__init__(*args, **kwargs)
        self.experiment_comparisons = []
        self.observable_models = {}
        self.conditional_models = {}
        self.model_dependencies = []
        self._active_question_id = None
        if self.arm == "hegel":
            root = self._open_question({"id": "task", "question":
                "Infer the process from current paid observations and predict new public trajectories.",
                "parent_id": None, "depends_on": []})
            root["origin"] = "public_task"
            self._active_question_id = root["id"]

    def restore_research(self, previous, previous_workspace, laboratory, *, confirmation_seed):
        """Continue saved fixed-apparatus research, preserving its paid account.

        Construct this agent in a new workspace with the intended total limits.
        Only the original task's files and observations enter the continuation.
        """
        context = json.loads((Path(previous_workspace) / "research_context.json").read_text())
        self.knowledge = deepcopy(context["session_knowledge"])
        self.source_task = context["research_phase"] == "source_learning"
        return self._restore_paid_research(previous, previous_workspace, laboratory,
            confirmation_seed=confirmation_seed, context=context)

    def restore_practice_research(self, previous, previous_workspace, laboratory, *, confirmation_seed):
        """Continue a zero-LLM practice directly from its actual saved result/state.

        The caller supplies the original knowledge and new total limits. No online
        request is fabricated for a frozen practice that never made an LLM call.
        """
        return self._restore_paid_research(previous, previous_workspace, laboratory,
            confirmation_seed=confirmation_seed)

    def _restore_paid_research(self, previous, previous_workspace, laboratory, *,
                               confirmation_seed, context=None):
        source = Path(previous_workspace).resolve()
        if source == self.workspace:
            raise ValueError("restore into a separate workspace to retain the original run")
        if laboratory.costs["observation_cost"]:
            raise ValueError("restore with a fresh apparatus; do not repurchase the prefix")
        if getattr(laboratory, "_settings", {}).get("calibration_change") is not None:
            raise ValueError("this continuation restores a fixed apparatus")
        state = json.loads((source / "research_state.json").read_text())
        self.inherited_files = set(previous["inherited_files"])

        def ignored(directory, names):
            root = Path(directory)
            return {name for name in names if _model_context_ignored(root / name, source)
                    or (root == source and name == "research_trace.json")}

        # Keep candidate fitting views and ordinary dependencies, while sharing
        # disk extents and excluding host histories from predictor contexts.
        excluded = ignored(source, [path.name for path in source.iterdir()])
        for path in source.iterdir():
            if path.name in excluded:
                continue
            target = self.workspace / path.name
            if path.is_dir():
                shutil.copytree(path, target, dirs_exist_ok=True, symlinks=True,
                                ignore=ignored, copy_function=_copy_model_file)
            else:
                _copy_model_file(path, target)
        # The last request precedes the last executed action. Refresh its files
        # from disk and add successful final writes, not another arm's archive.
        names = set(context["files"] if context is not None else self.files)
        if context is None:
            # The frozen method can write predictors through its Python steps;
            # expose those real source files without importing a nonexistent request.
            names.update(str(path.relative_to(self.workspace))
                         for path in self.workspace.glob("*.py"))
            if previous.get("method_path"):
                names.add(previous["method_path"])
        names.update(event.get("feedback", {}).get("written") for event in previous["events"]
                     if event.get("feedback", {}).get("written"))
        self.files = {name: (self.workspace / name).read_text(encoding="utf-8")
                      for name in names if (self.workspace / name).is_file()}
        self.events = deepcopy(previous["events"])
        self.observations = deepcopy(previous["paid_observations"])
        self.rows = [row for record in self.observations for row in laboratory.observation_rows(record)]
        self.costs = deepcopy(previous["costs"])
        self.model = previous["model"]
        self.questions = deepcopy(previous["online_questions"])
        self.revisions = deepcopy(previous["revision_proposals"])
        self.experiment_comparisons = deepcopy(previous["experiment_comparisons"])
        self.observable_models = deepcopy(previous.get("observable_models", {}))
        self.conditional_models = deepcopy(previous.get("conditional_models", {}))
        self.model_dependencies = deepcopy(previous.get("model_dependencies", []))
        self.method_results = deepcopy(previous["method_results"])
        self.operator_results = deepcopy(state["operator_results"])
        self.candidate_paths = deepcopy(state["exported_candidate_paths"])
        self.manifest = deepcopy(previous["outgoing_knowledge"].get("manifest", {}))
        self._active_question_id = (context["online_research"]["active_question"]
            if context is not None else state["online_research"]["active_question"])
        question_ids = {question["id"] for question in self.questions}
        for event in self.events:
            args, feedback = event.get("arguments", {}), event.get("feedback", {})
            declaration = args.get("research_question") or args.get("question") or feedback.get("question")
            identifier = feedback.get("research_question_id") or (declaration or {}).get("id")
            if identifier in question_ids:
                self._active_question_id = identifier
        retained = previous["evaluated_model"]
        self.evaluated_model = None if retained is None else {
            "model": str(Path(retained["model"]).relative_to("_evaluated_model")),
            "source_model": retained["source_model"], "current_mse": retained["mse"],
            "evaluated_rows": retained["evaluated_rows"]}
        self.confirmations = []

        def collect_confirmations(value):
            if isinstance(value, dict):
                confirmation = value.get("fresh_confirmation")
                if confirmation and "observations" in confirmation:
                    indices = confirmation["observation_indices"]
                    if not any(item["observation_indices"] == indices for item in self.confirmations):
                        self.confirmations.append(deepcopy(confirmation))
                for key, item in value.items():
                    if key != "fresh_confirmation":
                        collect_confirmations(item)
            elif isinstance(value, list):
                for item in value:
                    collect_confirmations(item)

        for event in self.events:
            collect_confirmations(event.get("feedback", {}))
        # The result also carries the last confirmation directly. Event records
        # are normally complete; de-duplication avoids consuming the stream twice.
        collect_confirmations({"fresh_confirmation": previous.get("fresh_confirmation")})
        self.confirmation_rng = Random(confirmation_seed)
        for confirmation in self.confirmations:
            actions = laboratory.confirmation_actions(confirmation["rows"], self.confirmation_rng)
            if actions != [record["action"] for record in confirmation["observations"]]:
                raise ValueError("confirmation stream differs from the saved paid acquisitions")
        # Every temporal trajectory resets physical state. Carry its already
        # paid account forward without purchasing or simulating it again.
        restore_account = getattr(laboratory, "restore_paid_account", None)
        if restore_account is not None:
            restore_account(self.observations, self.costs)
        else:
            account = {"observation_cost": self.costs["observation_cost"],
                "measurement_cost": self.costs["measurement_cost"], "queries": len(self.observations)}
            if isinstance(laboratory, AnonymousHistoryLab):
                laboratory._costs = account
            else:
                laboratory._past_costs = account
        if self.provider is not None:
            self.provider.calls = self.costs["model_calls"]
        self._write_evidence()
        self._prepare_program_state()
        return {"restored_paid_rows": len(self.rows), "costs_before": deepcopy(self.costs),
                "restored_confirmations": len(self.confirmations), "new_observations": 0}

    def _open_question(self, question, experiment_path=None):
        """Reuse one live question; acquisition history is not a new assignment."""
        existing = next((q for q in self.questions if q["id"] == question["id"]), None)
        if existing is None:
            if not question.get("question"):
                raise ValueError("a new research question needs its question text")
            existing = super()._open_question({"parent_id": None, "depends_on": [], **question},
                                             experiment_path)
            existing["assignment"] = {key: deepcopy(existing[key])
                                      for key in ("id", "question", "parent_id", "depends_on")}
            existing["acquired_evidence"] = []
        else:
            existing.update(deepcopy(question))
            if experiment_path:
                existing["experiment_path"] = experiment_path
        return existing

    def _return_revision_answer(self, question, record, answer):
        parent = next((q for q in self.questions if q["id"] == question.get("parent_id")), None)
        if parent is not None:
            parent.setdefault("child_answers", []).append({
                "question_id": question["id"], "revision_index": record["index"], **deepcopy(answer)})
            parent["status"] = "reopened"

    def _query(self, arguments, laboratory):
        reviewed = {q["id"]: deepcopy(q.get("child_answers", [])) for q in self.questions}
        result = super()._query(arguments, laboratory)
        returned = result.get("question")
        if returned is not None:
            question = next(q for q in self.questions if q["id"] == returned["id"])
            self._active_question_id = question["id"]
            question["reviewed_child_answers"] = reviewed.get(question["id"], [])
            acquired = [index for observation in result.get("observations", [])
                        for index in observation.get("row_indices", [observation["observation_index"]])]
            question["acquired_evidence"] = sorted(set(question.get("acquired_evidence", [])) | set(acquired))
            # The inherited operator executor used append-only questions. A
            # continued question still owns its actual continuation feedback.
            for key in ("after_observation_results", "after_observation_error"):
                if key in result:
                    question[key] = deepcopy(result[key])
            revision = result.get("revision")
            if revision is not None:
                self.revisions[revision["index"]]["question_id"] = question["id"]
                result["revision"] = deepcopy(self.revisions[revision["index"]])
            result["question"] = deepcopy(question)
        reassessments = refresh_observables(self, laboratory)
        if reassessments:
            result["observable_reassessments"] = reassessments
        component_reassessments = refresh_components(self, laboratory)
        if component_reassessments:
            result["component_reassessments"] = component_reassessments
        return result

    def _execute(self, action, arguments, laboratory):
        arguments = deepcopy(arguments)
        if "notes" in arguments:
            self._save("research_notes.md", arguments.pop("notes"))
        declaration = arguments.pop("research_question", None) or arguments.get("question")
        question = self._open_question(declaration) if declaration else None
        reviewed_returns = deepcopy(question.get("child_answers", [])) if question is not None else []
        if question is not None:
            self._active_question_id = question["id"]
            if action == "query":
                arguments.setdefault("question", {key: question[key]
                    for key in ("id", "question", "parent_id", "depends_on")})
        written = {}
        if action == "evaluate" and "code" in arguments:
            model = arguments.get("model", "model.py")
            written = super()._execute("write", {"path": model, "code": arguments.pop("code")}, laboratory)
        if action in {"evaluate", "finish"} and "bindings" in arguments:
            path = self.workspace / "model_bindings.json"
            bindings = json.loads(path.read_text()) if path.exists() else {}
            bindings[arguments.get("model", "model.py")] = arguments.pop("bindings")
            self._save("model_bindings.json", json.dumps(bindings, indent=2))
        if action == "evaluate" and "component" in arguments:
            result = evaluate_component(self, arguments, laboratory, question)
        elif action == "evaluate" and arguments.get("target", "response") != "response":
            result = evaluate_observable(self, arguments, laboratory, question)
        elif action == "run" and arguments.get("operator") == "compare_experiments":
            result = {"operator": "compare_experiments", "comparison": compare_experiments(
                self, arguments.get("models"), arguments.get("experiments"), laboratory,
                mse_threshold=self.config.confirmation_mse_threshold, target=arguments.get("target", "response"),
                reference_actions=arguments.get("reference_actions"))}
        elif action == "run" and arguments.get("operator") == "analyze_revision":
            index = arguments["revision_index"]
            if type(index) is not int or not 0 <= index < len(self.revisions):
                raise ValueError("revision_index must identify an existing experiment")
            record = self.revisions[index]
            owner = next((q for q in self.questions if q["id"] == record.get("question_id")), None)
            result = {"operator": "analyze_revision", "revision": analyze_revision(
                self, record, laboratory, owner, path=arguments.get("path"))}
            if owner is not None:
                result["question"] = deepcopy(owner)
        else:
            result = super()._execute(action, arguments, laboratory)
        result = {**written, **result}
        if (action in {"evaluate", "finish"} and "component" not in arguments
                and arguments.get("target", "response") == "response"):
            self.model_dependencies = deepcopy(result.get("model_dependencies", []))
        if question is not None:
            # This records the decision's actual question association only.
            # Answers produced by this action remain new to a later parent turn.
            question["reviewed_child_answers"] = reviewed_returns
            result["research_question_id"] = question["id"]
        return result

    def _manifest_error(self, manifest, model):
        if not self.source_task:
            # Continuation can finish ordinary scientific research without
            # inventing a select_actions program merely to populate a list.
            # The inherited consumer records the selected executable model;
            # paid and fresh scoring, not this bookkeeping, determine adoption.
            return None
        return super()._manifest_error(manifest, model)

    def _online_research(self):
        pending = []
        for record in self.revisions:
            if record.get("observations") and (record.get("readout_error") or not record.get("analysis_is_agent_authored")):
                pending.append({"revision_index": record["index"], "question_id": record.get("question_id"),
                    "status": record["status"], "record_indices": record["record_indices"],
                    "readout": record["proposal"].get("readout"),
                    "error": record.get("readout_error"),
                    "record_file": "research_state.json"})
        returns = []
        for question in self.questions:
            reviewed = question.get("reviewed_child_answers", [])
            returned = [answer for answer in question.get("child_answers", []) if answer not in reviewed]
            if returned:
                returns.append({"parent_id": question["id"], "answers": deepcopy(returned)})
        questions = [{key: deepcopy(q[key]) for key in
            ("id", "question", "assignment", "parent_id", "depends_on", "status", "experiment_path",
             "answer", "child_answers", "evidence", "acquired_evidence", "revision_index")
            if key in q} for q in self.questions]
        for summary, question in zip(questions, self.questions):
            summary["unreviewed_evidence"] = [index for index in question.get("acquired_evidence", [])
                if index > question.get("reviewed_through_observation", -1)]
        latest = self.events[-1] if self.events else {}
        feedback = latest.get("feedback", {})
        comparison = self.experiment_comparisons[-1] if self.experiment_comparisons else None
        latest_comparison = ({key: deepcopy(comparison[key]) for key in
            ("index", "status", "diagnosis", "target", "evidence_cutoff", "models", "eligible_models",
             "mse_threshold", "selected_experiment", "query_arguments", "compute_cost", "error",
             "selection_policy", "task_value")
            if key in comparison} if comparison is not None else None)
        if latest_comparison is not None:
            latest_comparison["prior_fit"] = [{key: deepcopy(fit[key]) for key in
                ("model", "returncode", "mse", "error") if key in fit}
                for fit in comparison["prior_fit"]]
            latest_comparison["comparisons"] = [{key: deepcopy(experiment[key]) for key in
                ("id", "row_count", "observation_cost", "predicted_separation")}
                for experiment in comparison["comparisons"]]
            latest_comparison["current_paid_rows"] = len(self.rows)
        return {"active_question": self._active_question_id, "questions": questions,
                "pending_analyses": pending, "parent_returns": returns,
                "latest_comparison": latest_comparison,
                "latest_execution": {"action": latest.get("action"), **{key: deepcopy(feedback[key])
                    for key in ("error", "status", "executed_path", "mse", "returncode") if key in feedback}},
                "evidence_file": "research_state.json"}

    def _request(self, laboratory):
        for name in list(self.files):
            path = self.workspace / name
            if path.is_file():
                self.files[name] = path.read_text(encoding="utf-8")
        for path in (self.workspace / "shared").rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}:
                self.files[str(path.relative_to(self.workspace))] = path.read_text(encoding="utf-8")
        request = super()._request(laboratory)
        request["research_phase"] = "source_learning" if self.source_task else "continuation"
        if not self.source_task:
            # Restore the existing ProcedureConsumer contract after replacing
            # its instrument-specific interface with the common archive one.
            request["task_interface"] = request["task_interface"].replace(
                _SOURCE_EXECUTION_REQUIREMENT, _CONTINUATION_REQUIREMENT)
            start = COMMON_INTERFACE.index("For the hegel condition, retain")
            end = COMMON_INTERFACE.index("During source learning, Hegel")
            request["task_interface"] = request["task_interface"].replace(
                COMMON_INTERFACE[start:end],
                "Executable hypotheses, paid evidence and research files persist. During "
                "continuation, no manual manifest or new experimental program is needed "
                "to submit a predictor for fresh confirmation.\n")
        notes = self.workspace / "research_notes.md"
        request["research_notes"] = notes.read_text(encoding="utf-8") if notes.exists() else ""
        request["online_research"] = self._online_research()
        request["observable_models"] = deepcopy(self.observable_models)
        request["conditional_models"] = deepcopy(self.conditional_models)
        request["model_dependencies"] = deepcopy(self.model_dependencies)
        if self.arm == "hegel":
            root = next(q for q in self.questions if q["id"] == "task")
            request["research_root"] = deepcopy(root["assignment"])
            request["research_dependency_focus"] = dependency_focus(self)
        return request

    def _attach_research_advice(self, action, arguments, result, laboratory):
        # This ordinary-Python condition exposes no shared instrument solver.
        # Preserve executed feedback without proposing the old meter workflow.
        result.pop("available_research_action", None)
        self._current_research_advice = None
        return result

    def _registered_research_programs(self, manifest):
        # Submission bindings name the executed method but omit its registered
        # calling convention and paid support. Use the same merge online and
        # during export so a failed submission cannot erase that information.
        registered = {entry["path"]: entry for entry in
                      self.knowledge.get("manifest", {}).get("research_programs", [])}
        return [{**deepcopy(registered.get(entry.get("path"), {})), **entry}
                for entry in manifest.get("research_programs", [])]

    def _method_continuation(self, entry):
        if self.model is not None or not entry["current_executions"]:
            return None
        latest = entry["current_executions"][-1]
        feedback = latest["feedback"]
        confirmation = feedback.get("fresh_confirmation") or {}
        if confirmation.get("passed") is not False:
            return None
        proposal = latest.get("proposal") or {}
        scored = confirmation.get("returncode") == 0 and confirmation.get("mse") is not None
        return {
            "status": "prediction_refuted" if scored else "confirmation_execution_failed",
            "submitted_model": feedback.get("method_submission") or proposal.get("model")
                or proposal.get("arguments", {}).get("model"),
            "submitted_candidate": deepcopy(feedback.get("submitted_candidate")),
            "method_step": latest["step"],
            "confirmation": {key: deepcopy(confirmation[key]) for key in
                ("passed", "mse", "rows", "mse_threshold", "returncode", "timeout", "stderr")
                if key in confirmation},
            "new_paid_observation_indices": deepcopy(confirmation.get("observation_indices", [])),
            "current_paid_rows": len(self.rows),
            "resume_action": deepcopy(entry["execute_action"]),
            "revision_path": entry["path"],
        }

    def _prepare_program_state(self):
        super()._prepare_program_state()
        state_path = self.workspace / "research_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["experiment_comparisons"] = deepcopy(self.experiment_comparisons)
        state["online_research"] = self._online_research()
        state["observable_models"] = deepcopy(self.observable_models)
        state["conditional_models"] = deepcopy(self.conditional_models)
        state["model_dependencies"] = deepcopy(self.model_dependencies)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        manifest = self.manifest or self.knowledge.get("manifest", {})
        path = self.workspace / "shared/knowledge.json"
        if path.is_file():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(current, dict):
                    manifest = current
            except json.JSONDecodeError:
                pass  # An unfinished manifest does not erase the inherited entry.
        methods = []
        for method in self._registered_research_programs(manifest):
            if not (self.workspace / method["path"]).is_file():
                continue
            entry = deepcopy(method)
            entry["execute_action"] = ({"action": "run", "arguments": {
                "operator": "procedure", "path": method["path"],
                "max_steps": method["max_steps"], "confirm": True}}
                if "max_steps" in method else
                {"action": "query", "arguments": {"path": method["path"]}})
            entry["current_executions"] = [deepcopy(outcome) for outcome in self.method_results
                                            if outcome["path"] == method["path"]]
            entry["submitted_candidates"] = [
                {"method_step": outcome["step"], **deepcopy(outcome["feedback"]["submitted_candidate"]),
                 "confirmation": {key: deepcopy(outcome["feedback"]["fresh_confirmation"][key])
                    for key in ("passed", "mse", "returncode", "timeout", "observation_indices")}}
                for outcome in entry["current_executions"]
                if "submitted_candidate" in outcome["feedback"]]
            entry["source_unchanged"] = self._provenance(method["path"])["inherited_source_unchanged"]
            continuation = self._method_continuation(entry)
            if continuation is not None:
                entry["continuation"] = continuation
            methods.append(entry)
        self._save("method_decision.json", json.dumps({"methods": methods}, indent=2))

    def result(self, status):
        result = super().result(status)
        result["experiment_comparisons"] = deepcopy(self.experiment_comparisons)
        result["observable_models"] = deepcopy(self.observable_models)
        result["conditional_models"] = deepcopy(self.conditional_models)
        result["model_dependencies"] = deepcopy(self.model_dependencies)
        # A failed or unfinished source may already have authored a useful
        # manifest. The old result only retains self.manifest after submission.
        manifest = deepcopy(self.manifest or self.knowledge.get("manifest", {}))
        manifest_paths = ["shared/knowledge.json"]
        model_paths = [self.model] if self.model else []
        for event in self.events:
            arguments = event.get("arguments", {})
            if arguments.get("manifest"):
                manifest_paths.append(arguments["manifest"])
            if arguments.get("model"):
                model_paths.append(arguments["model"])
        for name in manifest_paths:
            path = self.workspace / name
            if path.is_file():
                try:
                    authored = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    # Keep the unfinished file itself below; checkpointing must
                    # not discard a source merely because it was mid-revision.
                    continue
                if isinstance(authored, dict):
                    manifest = authored

        # A procedure's submission creates execution bindings, while the
        # inherited manifest records how that same published method is called.
        # Keep the newly submitted hypothesis and backfill its method metadata
        # rather than replacing the publication with a path-only binding.
        for field in ("hypotheses", "research_programs"):
            additions = self.manifest.get(field, [])
            if not additions:
                continue
            entries = manifest.setdefault(field, [])
            paths = {entry.get("path") for entry in entries}
            entries.extend(deepcopy(entry) for entry in additions
                           if entry.get("path") not in paths)
        if manifest.get("research_programs"):
            manifest["research_programs"] = self._registered_research_programs(manifest)

        files = self._result_files(manifest, manifest_paths, model_paths)
        lessons = deepcopy(self.knowledge.get("lessons", []))
        if self.lesson:
            lessons.append(self.lesson)
        result["outgoing_knowledge"] = {
            "files": files,
            "manifest": deepcopy(manifest),
            "question_traces": [*deepcopy(self.knowledge.get("question_traces", [])),
                                *deepcopy(self.questions)],
            "lessons": lessons,
        }
        for key in ("research_roles", "source_catalog", "archive_instructions"):
            if key in self.knowledge:
                result["outgoing_knowledge"][key] = deepcopy(self.knowledge[key])
        result["inherited_files_changed"] = sorted(
            name for name in self.inherited_files
            if name in files and files[name] != self.knowledge["files"][name])
        result["knowledge_protocol"] = "common_executable_archive"
        return result

    def _result_files(self, manifest, manifest_paths, model_paths):
        """Export executable source research for later independent consumers."""
        names = set(self.files) | set(manifest_paths) | set(model_paths)
        # Preserve ordinary authored dependencies and the selected training view
        # inside _evaluated_model/, not only predictor source. Provider transcripts
        # and runtime state are not executable knowledge or fitting context.
        names.update(str(path.relative_to(self.workspace))
                     for path in self.workspace.rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}
                     and "__pycache__" not in path.parts
                     and path.name not in _HOST_RECORD_FILES | {
                         "research_context.json", "research_state.json", "code_agent_instructions.txt"}
                     and not path.name.startswith(("code_action_", "code_call_")))

        def referenced_paths(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "path" and isinstance(item, str):
                        names.add(item)
                    referenced_paths(item)
            elif isinstance(value, list):
                for item in value:
                    referenced_paths(item)

        referenced_paths(manifest)
        return {name: (self.workspace / name).read_text(encoding="utf-8")
                for name in sorted(names) if (self.workspace / name).is_file()
                and (self.workspace / name).name not in _HOST_RECORD_FILES}
