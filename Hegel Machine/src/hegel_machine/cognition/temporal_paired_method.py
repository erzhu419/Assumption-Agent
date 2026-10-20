"""Execute model-authored research procedures in live or frozen research loops."""

from copy import deepcopy
from dataclasses import replace
import json
import math

from .code_agent import python_model_context_changes, python_model_context_matches
from .temporal_method_learning import METHOD_PATH
from .temporal_operator_agent import TemporalOperatorAgent


PROCEDURE_INTERFACE = """
run {operator:'procedure',path:'shared/research_procedure.py',max_steps:12,confirm:true}
executes your selected select_actions program repeatedly within the current
observation, computation and execution budgets. It reads actual feedback between
steps and stops on completion, an error, a failed confirmation or its step limit.
confirm:true explicitly permits that program to select a model for ordinary fresh
confirmation. Without it, execution stops before submission. Each internal action
is charged normally; the entire invocation uses one LLM call. You can inspect or
edit the program and use query {path:...} for only one invocation instead.
During live research, two consecutive acquisitions without a prospective test or
an analysis continuation return to you before a third such acquisition. The
method_review_required result preserves those paid rows and the proposed action.
Inspect the evidence and method, then choose an analysis, a source revision, or
an explicit further query. This is a handoff, not a refutation of the method.
"""

DEFAULT_RESEARCH_PROCEDURE = '''import json
def select_actions(spec, observations, memory):
    state = json.load(open('research_state.json'))
    advice = state.get('research_advice')
    if advice is not None:
        return advice.get('next_action') or {'stop': advice.get('diagnosis', 'unresolved')}
    outcomes = state.get('operator_results', [])
    if outcomes and outcomes[-1].get('operator') == 'research_choice':
        choice = outcomes[-1]
        return choice.get('next_action') or {'stop': choice.get('diagnosis', 'unresolved')}
    current = state.get('evaluated_model')
    models = [current['model']] if current else state.get('exported_candidate_paths', [])
    if not models:
        return {'stop': 'supply_or_evaluate_a_candidate_first'}
    return {'action': 'run', 'arguments': {'operator': 'research_choice', 'models': models}}
'''


FROZEN_METHOD_INTERFACE = """
Write select_actions(public_spec, observations, memory) returning a dictionary.
The same source will execute repeatedly with the current paid laboratory records.
memory retains inherited knowledge, including any supplied research_roles paths.
As in the live world procedure, program_path and program_arguments identify this
invocation. memory.program_results contains this path's actual completed steps
since its current source was installed; a whole-source revision starts a new scope.
The scope persists across restoration. Older unscoped history remains available in
research_state.json; it is not assumed to be this draft's completed work.
memory.program_evidence gives current records/rows; each result gives evidence_before
and evidence_after. New observations preserve completed stages: your program chooses
which results remain usable and when a fit needs current evidence. Use actual feedback
to advance, retry or stop. These fields do not cache your Python or choose a next stage.
research_state.json contains flat operator_results entries, current limits/costs,
and method_results:[{step,proposal,feedback},...]. Method feedback retains actual
acquisition/operator errors and fresh_confirmation outcomes; observation payloads
are omitted there because the paid records are already supplied separately.
proposal is null when select_actions fails before returning an action. Read its
feedback for the failure; use proposal or {} when inspecting prior action fields.

Return actions and optional after_observation as in the shared temporal interface.
Empty actions with a nonempty continuation can analyze already-paid evidence.
Supply your own live question when useful; otherwise the executor records a generic
question for that invocation. It supplies no experiments, fitting steps or choices.

When ready, return {'model':'an_existing_predictor.py','lesson':'...'} with no
actions or after_observation steps. You choose the predictor; this may be an
inherited predictor or one your procedure exported. The executor creates only the
manifest bindings needed to submit that declared path to ordinary fresh confirmation.
Failed confirmation contributes paid counterexamples, visible on the next invocation.
No candidate is selected, instrument refitted, or model submitted automatically.
A repeated comparison with unchanged evidence, candidates and predictions, or a
repeated identical execution error, returns method_no_progress to the caller.
Alternatively return {'action':'evaluate','arguments':{'model':'model.py'}}, or
another ordinary query/run/write/finish action. The same agent executes exactly
that declared action and records its result before the next program invocation.
Return {'stop':'reason'} to leave an unresolved investigation to the caller.
"""


class ProcedureConsumerAgent(TemporalOperatorAgent):
    """Give both knowledge conditions the same single-invocation procedure contract."""

    common_interface = TemporalOperatorAgent.common_interface + """
Additional procedure interface, available equally with or without retained methods:
query {path:'a_program.py'} invokes select_actions ONCE. Read its actual feedback
before deciding whether to invoke it again, change it or use ordinary actions.
Programs may return acquisition/analysis steps or a final model declaration under
the following contract. Direct finish may also request fresh confirmation before
any training observations exist; the absent training score is not a confirmation.
For a consumer task, finish registers your explicitly selected model path in the
current manifest if missing. Existing research entries are preserved. Registration
adds no scientific support: fresh predictive confirmation still determines adoption.
""" + FROZEN_METHOD_INTERFACE + PROCEDURE_INTERFACE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_results = []
        self.program_contexts = {}
        self._current_proposal = None
        self._active_program_path = None
        self._procedure_confirmation_allowed = True
        self._method_no_progress = None
        self._acquisition_review = None
        self._procedure_rejected_context = None

    def _execute(self, action, arguments, laboratory):
        if action == "run" and arguments.get("operator") == "procedure":
            before = deepcopy(self.costs)
            result = execute_procedure(self, laboratory, arguments["path"],
                max_steps=arguments.get("max_steps", 12),
                allow_confirmation=arguments.get("confirm") is True)
            return {"operator": "procedure", **result,
                    "compute_cost": self.costs["compute_cost"] - before["compute_cost"],
                    "python_execution_seconds": self.costs["python_execution_seconds"] - before["python_execution_seconds"]}
        if action == "finish" and self.arm == "hegel" and not self.source_task:
            model = arguments.get("model", "model.py")
            manifest_path = arguments.get("manifest", "shared/knowledge.json")
            path = self.workspace / manifest_path
            manifest = json.loads(path.read_text()) if path.exists() else deepcopy(self.knowledge.get("manifest", {}))
            if isinstance(manifest, dict) and isinstance(manifest.get("hypotheses", []), list):
                hypotheses = manifest.setdefault("hypotheses", [])
                registered = any(isinstance(entry, dict) and entry.get("path") == model for entry in hypotheses)
                if not registered:
                    hypotheses.append({"id": f"submitted:{model}", "path": model})
                if not registered or not path.exists():
                    self._save(manifest_path, json.dumps(manifest, indent=2))
        return super()._execute(action, arguments, laboratory)

    def _score(self, model):
        if not self.rows:
            # There is no training diagnostic yet. The ordinary finish path
            # still purchases and scores fresh rows before accepting this model.
            return {"returncode": 0, "mse": None, "rows": 0,
                    "paid_score": "not_applicable_no_paid_rows"}
        return super()._score(model)

    def _program_state(self):
        state = super()._program_state()
        state["method_results"] = deepcopy(self.method_results)
        state["program_contexts"] = deepcopy(self.program_contexts)
        return state

    def _program_evidence(self):
        return {"records": len(self.observations), "rows": len(self.rows)}

    def _program_memory(self, path, arguments):
        memory = super()._program_memory(path, arguments)
        source = (self.workspace / path).read_text(encoding="utf-8")
        context = self.program_contexts.get(path)
        if context is None or context["source"] != source:
            context = {"source": source, "start": len(self.method_results)}
            self.program_contexts[path] = context
        memory.update(program_start=context["start"],
            program_results=[deepcopy(row) for row in self.method_results[context["start"]:]
                             if row["path"] == path],
            program_evidence=self._program_evidence())
        return memory

    def _handle_program_proposal(self, proposed, laboratory):
        self._current_proposal = deepcopy(proposed)
        path = self._active_program_path
        review = self._acquisition_review
        if review is not None and _acquisition_only(proposed) and len(review["steps"]) >= 2:
            return {"status": "method_review_required", "executed_path": path,
                "review": {"kind": "unassessed_acquisition_loop", "path": path,
                    "steps": [item["step"] for item in review["steps"]],
                    "record_indices": [index for item in review["steps"]
                                       for index in item["record_indices"]],
                    "observation_cost": sum(item["observation_cost"] for item in review["steps"]),
                    "current_paid_rows": len(self.rows),
                    "proposed_action": deepcopy(proposed),
                    "reason": "The live procedure acquired two batches without returning a "
                              "prospective test or analysis. Review these paid records before "
                              "another automatic acquisition; information gain is not assessed."}}
        if not isinstance(proposed, dict):
            return None  # Ordinary acquisition programs may return an action list.
        if "stop" in proposed:
            if review is not None:
                from .structural_inquiry import native_global_misfit_review, native_evidence_extension_review
                mismatch = (native_global_misfit_review(self, path)
                            or native_evidence_extension_review(self, path))
                if mismatch is not None:
                    return {"status": "method_review_required", "executed_path": path,
                            "review": {**mismatch, "proposed_stop": deepcopy(proposed)}}
            return {"status": "method_stopped", "reason": proposed["stop"], "executed_path": path}
        if "action" in proposed:
            action = proposed["action"]
            arguments = proposed.get("arguments", {})
            if action == "finish" and not self._procedure_confirmation_allowed:
                return {"status": "confirmation_not_authorized", "next_action": deepcopy(proposed),
                        "executed_path": path}
            if action == "evaluate" and self.evaluated_model is not None:
                # Compare with the already retained executable context before
                # evaluation can replace it. This also observes dependency edits
                # made inside select_actions, without creating another snapshot.
                if self._procedure_rejected_context is None:
                    self._procedure_evaluation_unchanged = python_model_context_matches(
                        self.workspace, arguments.get("model", "model.py"),
                        self.workspace / "_evaluated_model", self.evaluated_model["model"])
                else:
                    before = python_model_context_changes(
                        self.workspace, arguments.get("model", "model.py"),
                        self.workspace / "_evaluated_model", self.evaluated_model["model"])
                    self._procedure_evaluation_unchanged = before == self._procedure_rejected_context
            result = self._execute(action, arguments, laboratory)
            if action == "evaluate":
                if ((result.get("retention_decision") or {}).get("status") == "incumbent_retained"
                        and result.get("returncode") == 0):
                    after = python_model_context_changes(
                        self.workspace, arguments.get("model", "model.py"),
                        self.workspace / "_evaluated_model", self.evaluated_model["model"])
                    if self._procedure_rejected_context is not None:
                        # fit() may itself update parameters even at equal MSE.
                        self._procedure_evaluation_unchanged &= before == after
                    self._procedure_rejected_context = after
                else:
                    self._procedure_rejected_context = None
            return {**result, "program_action": deepcopy(proposed), "executed_path": path,
                    **self._provenance(path)}
        if "model" in proposed:
            if proposed.get("actions") or proposed.get("after_observation"):
                return {"error": "a model submission cannot also request acquisitions or operators",
                        "executed_path": path}
            model = proposed["model"]
            if not isinstance(model, str) or not model:
                return {"error": "model must name the predictor selected by the procedure",
                        "executed_path": path}
            if not self._procedure_confirmation_allowed:
                return {"status": "confirmation_not_authorized", "next_action": {
                    "action": "finish", "arguments": {"model": model}}, "executed_path": path}
            manifest = {"record_kind": "execution_bindings",
                        "hypotheses": [{"id": "declared_predictor", "path": model}],
                        "research_programs": [{"id": "executed_method", "path": path}],
                        "support_questions": []}
            manifest_path = "shared/frozen_method_manifest.json"
            self._save(manifest_path, json.dumps(manifest, indent=2))
            result = self._execute("finish", {"model": model, "manifest": manifest_path,
                                              "lesson": proposed.get("lesson", "")}, laboratory)
            return {**result, "method_submission": model,
                    "submitted_model_inherited": result.get("inherited_program", False),
                    "submitted_model_unchanged": result.get("inherited_source_unchanged", False),
                    "executed_path": path, **self._provenance(path)}
        if not proposed.get("question"):
            proposed["question"] = {"id": f"frozen_method_step_{len(self.method_results)}",
                "question": "Which next research step does the frozen procedure prescribe from current evidence?",
                "parent_id": None, "depends_on": []}
        return None

    def _query(self, arguments, laboratory):
        path = arguments.get("path")
        if not path:
            return super()._query(arguments, laboratory)
        self._active_program_path, self._current_proposal = path, None
        evidence_before = self._program_evidence()
        try:
            feedback = super()._query(arguments, laboratory)
        except (ValueError, TypeError, KeyError, OSError) as error:
            feedback = {"error": f"{type(error).__name__}: {error}", "executed_path": path}
        self.method_results.append({"step": len(self.method_results), "path": path,
            "proposal": deepcopy(self._current_proposal), "feedback": self._compact_feedback(feedback),
            "evidence_before": evidence_before, "evidence_after": self._program_evidence()})
        return feedback

    @staticmethod
    def _compact_feedback(feedback):
        result = deepcopy(feedback)
        result.pop("observations", None)
        if isinstance(result.get("fresh_confirmation"), dict):
            result["fresh_confirmation"].pop("observations", None)
        return result

    def result(self, status):
        result = super().result(status)
        result["method_results"] = deepcopy(self.method_results)
        result["program_contexts"] = deepcopy(self.program_contexts)
        if status == "method_no_progress" and self._method_no_progress:
            result["no_progress"] = deepcopy(self._method_no_progress)
        return result

    def _procedure_budget_status(self):
        if self.costs["python_execution_seconds"] >= self.config.execution_seconds:
            return "python_execution_budget_exhausted"
        if self.config.observation_budget - self.costs["observation_cost"] < self.config.confirmation_rows:
            return "confirmation_budget_exhausted"
        return None

    def _procedure_terminal_status(self):
        return "completed" if self.model else None


def _comparison_state(record):
    """Reuse saved scientific values, excluding new snapshot paths and charges."""
    state = {key: record.get(key) for key in (
        "models", "target", "experiments", "reference_actions", "evidence_cutoff",
        "mse_threshold", "selection_policy", "eligible_models", "diagnosis")}
    state["forecasts"] = [{key: row.get(key) for key in (
        "model", "source", "prediction_inputs", "predictions", "returncode", "timeout")}
        for row in record.get("forecasts", [])]
    state["prior_fit"] = [{key: row.get(key) for key in (
        "model", "mse", "loss", "rows", "returncode", "timeout")}
        for row in record.get("prior_fit", [])]
    return state


def _failure_state(feedback):
    keys = ("error", "after_observation_error", "returncode", "timeout")
    if not any(feedback.get(key) for key in keys):
        return None
    failure = {key: feedback.get(key) for key in keys}
    # Traceback paths vary between candidate snapshots; the actual last error
    # remains available in full in the original step feedback.
    failure["stderr"] = (feedback.get("stderr") or "").strip().splitlines()[-1:]
    return failure


def _acquisition_only(proposed):
    """Identify an opaque acquisition, not whether the experiment is useful."""
    if isinstance(proposed, list):
        return bool(proposed)
    if not isinstance(proposed, dict):
        return False
    if proposed.get("action") == "query":
        proposed = proposed.get("arguments", {})
    elif "action" in proposed:
        return False
    return bool(proposed.get("actions")) and not any(
        proposed.get(key) for key in ("revision", "readout", "after_observation"))


def execute_procedure(agent, laboratory, path, max_steps=8, *, allow_confirmation=True,
                      record_events=False, on_checkpoint=None, query_arguments=None):
    """Run the existing procedure protocol on this agent's evidence and account.

    Frozen replay records its own zero-LLM events and permits the authored method
    to react to failures. A live invocation keeps substeps in method_results and
    returns on failure, leaving the calling LLM the revision decision.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    status, steps = "method_step_limit", []
    previous_comparison, previous_failure, previous_evaluation = None, None, None
    agent._procedure_rejected_context = None
    agent._method_no_progress = None
    previous = agent._procedure_confirmation_allowed
    previous_review = agent._acquisition_review
    previous_config = agent.config
    if record_events:
        # A frozen program has no live author to receive the per-run handoff.
        # Keep its existing task allowance; authored explicit timeouts still apply.
        agent.config = replace(agent.config, run_timeout_seconds=agent.config.execution_seconds)
    agent._acquisition_review = None if record_events else {"steps": [], "source": None}
    agent._procedure_confirmation_allowed = allow_confirmation
    confirmation_start = len(agent.confirmations)
    try:
        for step in range(max_steps):
            terminal = agent._procedure_terminal_status()
            if terminal is not None:
                status = terminal
                break
            exhausted = agent._procedure_budget_status()
            if exhausted is not None:
                status = exhausted
                break
            agent._current_proposal = None
            agent._procedure_evaluation_unchanged = False
            review = agent._acquisition_review
            comparison_start = len(getattr(agent, "experiment_comparisons", []))
            arguments = {**(query_arguments or {}), "path": path}
            try:
                if review is not None:
                    source = (agent.workspace / path).read_text(encoding="utf-8")
                    if source != review["source"]:
                        review.update(source=source, steps=[])
                feedback = agent._execute("query", arguments, laboratory)
            except (ValueError, TypeError, KeyError, OSError) as error:
                feedback = {"error": f"{type(error).__name__}: {error}", "executed_path": path}
            entry = {"step": step, "action": "query", "arguments": arguments,
                     "proposal": deepcopy(agent._current_proposal),
                     "feedback": agent._compact_feedback(feedback)}
            steps.append(entry)
            if record_events:
                agent.events.append({"step": step, "action": "query", "arguments": arguments,
                    "feedback": deepcopy(feedback),
                    "usage": {"backend": "frozen_program_execution", "total_tokens": 0}})
            if on_checkpoint:
                on_checkpoint(agent.result("researching"))
            terminal = agent._procedure_terminal_status()
            if terminal is not None:
                status = terminal
                break
            if feedback.get("status") in {"confirmation_not_authorized", "method_stopped", "method_review_required"}:
                status = feedback["status"]
                break
            if review is not None:
                if (_acquisition_only(entry["proposal"]) and feedback.get("observation_cost", 0) > 0
                        and not feedback.get("revision") and not feedback.get("after_observation")):
                    review["steps"].append({"step": step,
                        "record_indices": feedback.get("record_indices", []),
                        "observation_cost": feedback["observation_cost"]})
                else:
                    review["steps"] = []
            no_progress = None
            proposal = entry["proposal"] or {}
            if (isinstance(proposal, dict) and proposal.get("action") == "evaluate"
                    and feedback.get("returncode") == 0
                    and isinstance(feedback.get("mse"), (int, float))):
                selected_model = proposal.get("arguments", {}).get("model", "model.py")
                state = {"model": selected_model, "mse": feedback["mse"],
                         "evidence_cutoff": {"records": len(agent.observations),
                                             "rows": len(getattr(agent, "rows", []))}}
                repeated = False
                roundoff = None
                if (previous_evaluation and agent._procedure_evaluation_unchanged
                        and all(state[key] == previous_evaluation["state"][key]
                                for key in ("model", "evidence_cutoff"))):
                    previous_mse = previous_evaluation["state"]["mse"]
                    repeated = state["mse"] == previous_mse
                    threshold = agent.config.confirmation_mse_threshold
                    if (not repeated and math.isfinite(state["mse"])
                            and math.isfinite(previous_mse)
                            and (state["mse"] <= threshold) == (previous_mse <= threshold)):
                        # Parallel prediction/reduction also jitters at nonzero
                        # loss. Recognize a few final bits of the MSE itself,
                        # without treating a percentage gain as roundoff or
                        # hiding a transition across the confirmation gate.
                        roundoff = 8 * math.ulp(max(state["mse"], previous_mse))
                        if max(state["mse"], previous_mse) <= threshold:
                            # Near zero, cancellation is set by target precision
                            # rather than by the much smaller ULP of its MSE.
                            scale = max((abs(row.target if hasattr(row, "target")
                                else row["target"]) for row in agent.rows), default=0.)
                            roundoff = max(roundoff, math.ulp(scale) ** 2)
                        repeated = abs(state["mse"] - previous_mse) <= roundoff
                if repeated:
                    no_progress = {"kind": "repeated_candidate_evaluation", "path": path,
                        "steps": [previous_evaluation["step"], step], **deepcopy(state),
                        "reason": "The candidate and its executable dependencies were "
                                  "evaluated twice unchanged on the same paid evidence, with the "
                                  "same result within floating-point resolution. Review the method "
                                  "before another evaluation."}
                    if roundoff is not None:
                        no_progress.update(previous_mse=previous_mse,
                                           mse_roundoff_tolerance=roundoff)
                previous_evaluation = {"state": state, "step": step}
            else:
                previous_evaluation = None
            for comparison in getattr(agent, "experiment_comparisons", [])[comparison_start:]:
                if comparison.get("status") != "compared":
                    continue
                state = _comparison_state(comparison)
                if previous_comparison and state == previous_comparison["state"]:
                    no_progress = {"kind": "repeated_experiment_comparison", "path": path,
                        "steps": [previous_comparison["step"], step],
                        "comparison_indices": [previous_comparison["index"], comparison["index"]],
                        "evidence_cutoff": deepcopy(comparison["evidence_cutoff"]),
                        "diagnosis": comparison["diagnosis"],
                        "reason": "The same experiments and candidate sources, predictions and paid-fit "
                                  "results were compared twice without new paid evidence."}
                    break
                previous_comparison = {"state": state, "step": step, "index": comparison["index"]}
            failure = _failure_state(feedback)
            if failure is not None:
                state = {"failure": failure, "proposal": entry["proposal"],
                    "evidence_cutoff": {"records": len(agent.observations),
                                        "rows": len(getattr(agent, "rows", []))}}
                if previous_failure and state == previous_failure["state"]:
                    no_progress = {"kind": "repeated_execution_error", "path": path,
                        "steps": [previous_failure["step"], step],
                        "evidence_cutoff": state["evidence_cutoff"], "failure": failure,
                        "reason": "The same proposed operation returned the same error twice "
                                  "without new paid evidence."}
                previous_failure = {"state": state, "step": step}
            else:
                previous_failure = None
            if no_progress is not None:
                status, agent._method_no_progress = "method_no_progress", no_progress
                entry["feedback"]["no_progress"] = deepcopy(no_progress)
                if getattr(agent, "method_results", None):
                    agent.method_results[-1]["feedback"]["no_progress"] = deepcopy(no_progress)
                if record_events:
                    agent.events[-1]["feedback"]["no_progress"] = deepcopy(no_progress)
                if on_checkpoint:
                    on_checkpoint(agent.result(status))
                break
            if not record_events:
                if (feedback.get("fresh_confirmation") or {}).get("passed") is False:
                    status = "confirmation_failed"
                    break
                if (feedback.get("error") or feedback.get("after_observation_error")
                        or feedback.get("returncode") or feedback.get("timeout")):
                    status = "method_error"
                    break
    finally:
        agent._procedure_rejected_context = None
        agent._procedure_confirmation_allowed = previous
        agent._acquisition_review = previous_review
        agent.config = previous_config
    return {"status": status, "path": path, "steps": steps, "model": agent.model,
            **({"no_progress": deepcopy(agent._method_no_progress)} if status == "method_no_progress" else {}),
            **({"review": deepcopy(steps[-1]["feedback"]["review"])} if status == "method_review_required" else {}),
            "fresh_confirmation": deepcopy(agent.confirmations[-1])
                if len(agent.confirmations) > confirmation_start else None}


class FrozenMethodAgent(ProcedureConsumerAgent):
    """Replay one fixed procedure without an LLM selecting intermediate actions."""

    def __init__(self, workspace, *, knowledge, method_source, config=None, confirmation_seed=9173):
        frozen_memory = deepcopy(knowledge)
        frozen_memory.setdefault("files", {})[METHOD_PATH] = method_source
        super().__init__(None, workspace, arm="hegel", knowledge=frozen_memory, config=config,
                         confirmation_seed=confirmation_seed, source_task=False)
        self.frozen_method_source = method_source

    def execute_method(self, laboratory, max_steps=8, *, on_checkpoint=None):
        """Interpret only declared steps; stop after confirmation or the step limit."""
        outcome = execute_procedure(self, laboratory, METHOD_PATH, max_steps,
                                    record_events=True, on_checkpoint=on_checkpoint)
        return self.result(outcome["status"])

    def result(self, status):
        result = super().result(status)
        result.update(method_path=METHOD_PATH, method_source=self.frozen_method_source,
                      method_results=deepcopy(self.method_results))
        return result
