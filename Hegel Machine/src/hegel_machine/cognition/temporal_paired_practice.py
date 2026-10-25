"""Learn one executable method from paid, unlabeled paired situations."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from .code_provider_context import _compact_trajectory_payload, _compact_practice_feedback
from .checkpoint_storage import write_text_incrementally
from .language import Program
from .online_inquiry_llm import InquiryLLMAgent
from .temporal_method_learning import METHOD_PATH
from .temporal_paired_method import FrozenMethodAgent
from .temporal_operator_agent import OPERATOR_INTERFACE


def practice_execution_summary(result, directory, *, method_path=None):
    """Expose executed research outcomes without copying observations or programs."""
    def paid_score(value):
        score = {key: value[key] for key in
            ("model", "source_model", "mse", "evaluation_scope", "returncode",
             "evaluated_rows", "current_paid_rows") if key in value}
        rows = value.get('paid_residuals', {}).get('evaluated_rows')
        if rows is not None:
            score['evaluated_rows'] = rows
        view = value.get("fitting_view") or {}
        if view:
            score["fitting_view"] = {key: deepcopy(view[key]) for key in
                ("scope", "file", "rows", "response_row_range") if key in view}
        if value.get('later_readings'):
            score['later_readings'] = {key: deepcopy(value['later_readings'][key]) for key in
                ('rows', 'response_row_range', 'mse', 'max_abs_error') if key in value['later_readings']}
        return score

    def evidence_range(start, stop):
        lengths, histories = [], []
        for record in result.get("paid_observations", []):
            rows = [row for index, row in zip(record.get("row_indices", []), record.get("rows", []))
                    if start <= index < stop]
            if rows:
                lengths.append(len(rows))
                histories.extend(len(row.get("history", [])) for row in rows)
        return {"row_count": sum(lengths), "trajectory_row_counts": lengths,
                "max_history_length": max(histories, default=0)}

    # Keep indices in the canonical record when selecting one procedure from
    # an online investigation that may also execute other methods.
    start = result.get('practice_start', {})
    steps = [(index, entry) for index, entry in enumerate(result.get("method_results", []))
             if index >= start.get('method_results', 0)
             and (method_path is None or entry.get("path") == method_path)]
    native_models = {}
    summary = {"status": result.get("status") if method_path is None else None,
        "step_count": len(steps),
        "result_file": str(Path(directory) / "result.json"),
        "result_steps_field": "method_results", "author_feedback_steps_field": "steps",
        "action_counts": {}, "submission_count": 0, "confirmation_count": 0,
        "research_steps": []}
    if method_path is None and 'continuation_cost' in result:
        summary['practice_execution_cost'] = {key: result['continuation_cost'][key] for key in
            ('python_execution_seconds', 'compute_cost', 'observation_cost')
            if key in result['continuation_cost']}
    for index, entry in steps:
        proposal, feedback = entry.get("proposal") or {}, entry.get("feedback") or {}
        program_error = entry.get("proposal") is None and (
            feedback.get("error") or feedback.get("timeout")
            or feedback.get("returncode") not in (None, 0))
        action = ("program_error" if program_error else
                  proposal.get("action", "stop" if "stop" in proposal else
                               "model" if "model" in proposal else "query"))
        arguments = proposal.get("arguments") or {}
        summary["action_counts"][action] = summary["action_counts"].get(action, 0) + 1
        summary["last_action"] = {"step": entry.get("step", index), "action": action,
            **{key: arguments[key] for key in ("operator", "model") if key in arguments}}
        if "model" in proposal:
            summary["last_action"]["model"] = proposal["model"]
        executed = {**summary["last_action"], "json_path": ["method_results", index]}
        if isinstance(feedback.get('program_execution'), dict):
            executed['program_execution'] = deepcopy(feedback['program_execution'])
            summary['latest_program_execution'] = deepcopy(feedback['program_execution'])
        if program_error:
            message = {key: feedback[key] for key in
                       ("error", "returncode", "timeout") if key in feedback}
            stderr = (feedback.get("stderr") or "").strip().splitlines()
            if stderr:
                message["stderr"] = stderr[-1]
            executed.update(message)
            summary["last_execution_message"] = message
        if isinstance(feedback.get('retention_decision'), dict):
            executed['retention_decision'] = deepcopy(feedback['retention_decision'])
            summary['latest_retention_decision'] = deepcopy(feedback['retention_decision'])
        if "observation_cost" in feedback:
            executed["observation_cost"] = feedback["observation_cost"]
        actions = arguments.get("actions", proposal.get("actions", []))
        lengths = [len(item["sequence"]) for item in actions if "sequence" in item]
        if lengths:
            executed["trajectory_lengths"] = lengths
        comparison = feedback.get("comparison")
        if isinstance(comparison, dict):
            selected = comparison.get("selected_experiment") or {}
            executed["comparison"] = {key: comparison[key] for key in
                ("index", "status", "diagnosis", "evidence_cutoff") if key in comparison}
            executed["comparison"].update(
                prior_fit=[paid_score(item) for item in comparison.get("prior_fit", [])],
                selected_experiment={key: selected[key] for key in
                    ("id", "predicted_separation", "observation_cost") if key in selected})
        revision = feedback.get("revision")
        if isinstance(revision, dict):
            executed["revision"] = {key: revision[key] for key in
                ("index", "status", "evidence_cutoff", "action_row_counts") if key in revision}
            executed["revision"]["forecasts"] = [{key: forecast[key] for key in
                ("model", "prospective_mse", "scored_rows", "complete_experiment", "error")
                if key in forecast} for forecast in revision.get("forecasts", [])]
        if "model" in proposal or action == "finish":
            summary["submission_count"] += 1
            if "mse" in feedback:
                executed["paid_score"] = paid_score(feedback)
                summary["latest_submission_paid_fit"] = {
                    "step": entry.get("step", index),
                    "model": proposal.get("model", arguments.get("model")),
                    **executed["paid_score"]}
        if action == "evaluate":
            summary["latest_evaluation"] = {"step": entry.get("step", index),
                "model": arguments.get("model"),
                **paid_score(feedback),
                **{key: feedback[key] for key in ("mse", "loss", "error", "returncode", "timeout")
                   if key in feedback}}
            for key in ('retained_comparison', 'evaluated_model'):
                if isinstance(feedback.get(key), dict):
                    summary['latest_evaluation'][key] = paid_score(feedback[key])
            if isinstance(feedback.get('retention_decision'), dict):
                summary['latest_evaluation']['retention_decision'] = deepcopy(feedback['retention_decision'])
            executed['paid_score'] = deepcopy(summary['latest_evaluation'])
        confirmation = feedback.get("fresh_confirmation")
        if isinstance(confirmation, dict):
            summary["confirmation_count"] += 1
            summary["latest_confirmation"] = {key: confirmation[key]
                for key in ("passed", "mse", "rows", "mse_threshold") if key in confirmation}
            executed["fresh_confirmation"] = dict(summary["latest_confirmation"])
            if method_path is not None:
                summary["status"] = ("completed" if confirmation.get("passed")
                                     else "confirmation_failed")
        if method_path is not None and "stop" in proposal:
            summary["status"] = feedback.get("status", "method_stopped")
        submitted = feedback.get("submitted_candidate") or (
            confirmation.get("submitted_candidate", {}) if isinstance(confirmation, dict) else {})
        if "model" in proposal or action == "finish":
            model = submitted.get("source_model", proposal.get("model", arguments.get("model")))
            frozen = submitted.get("model", model)
            selected = {"step": entry.get("step", index), "source_model": model,
                "model_file": str(Path(directory) / frozen) if frozen else None,
                "json_path": ["method_results", index, "feedback"],
                **{key: submitted[key] for key in ("paid_rows", "mse") if key in submitted}}
            if result.get("workspace") and submitted.get("model"):
                source_file = Path(result["workspace"]) / submitted["model"]
                if source_file.is_file():
                    source = source_file.read_text(encoding="utf-8")
                    selected["source_preview"] = {"file": str(source_file),
                        "characters": len(source), "preview": source[:800]}
            if "paid_rows" in submitted and result.get("paid_observations"):
                cutoff = submitted["paid_rows"]
                selected["submission_evidence"] = {
                    "paid_before_submission": evidence_range(0, cutoff)}
                if isinstance(confirmation, dict):
                    selected["submission_evidence"]["fresh_confirmation"] = evidence_range(
                        cutoff, cutoff + confirmation.get("rows", 0))
            if model in native_models:
                candidate, location, proposed = native_models[model]
                selected["native_candidate"] = {
                    "json_path": location, "source": candidate.get("source"),
                    "fit_mse": candidate.get("fit_mse")}
                program = candidate.get("program", {})
                if "op" in program:
                    # Describe the actually exported expression, not the author's
                    # fallback fitter or the first candidate in a preview.
                    selected["native_candidate"]["expression"] = Program.from_dict(program).describe()
                selected["native_candidate"]["fitted_structure"] = [
                    {key: item[key] for key in ("operation", "gate_coordinate", "threshold",
                        "threshold_interval", "width", "placement") if key in item}
                    for item in candidate.get("research_program", {}).get("selection_steps", [])]
                selected["experiments_proposed_with_candidate"] = proposed
            if isinstance(confirmation, dict):
                selected["fresh_confirmation"] = dict(summary["latest_confirmation"])
                if "residuals" in confirmation:
                    selected["confirmation_residuals_path"] = [
                        "method_results", index, "feedback", "fresh_confirmation", "residuals"]
            summary["latest_submitted_predictor"] = selected
        for key in ("error", "reason"):
            if key in feedback:
                executed[key] = feedback[key]
        if feedback.get('status') == 'author_work_request':
            request = {key: deepcopy(feedback[key]) for key in (
                'status', 'question_id', 'meaning_version', 'object_type', 'reason',
                'required_contract', 'author_work_ref', 'bounded_result_ref',
                'bounded_status', 'bounded_allowance') if key in feedback}
            request['source_ref'] = {'file': summary['result_file'],
                                     'json_path': ['method_results', index, 'feedback']}
            executed['author_work_request'] = request
            summary['latest_author_work_request'] = deepcopy(request)
        summary["research_steps"].append(executed)
        outcomes = [(feedback, ["method_results", index, "feedback"])]
        outcomes.extend((outcome, ["method_results", index, "feedback", "after_observation_results", j])
            for j, outcome in enumerate(feedback.get("after_observation_results", [])))
        for outcome, location in outcomes:
            native = outcome.get("native_condition_research")
            if not isinstance(native, dict):
                continue
            candidates = native.get("candidates", [])
            proposals = native.get("proposed_experiments", [])
            proposal_preview = []
            for proposal_index, suggested in enumerate(proposals[:3]):
                query = suggested.get("query_action", {}).get("arguments", {})
                proposal_preview.append({
                    **{key: suggested[key] for key in
                       ("purpose", "control", "predicted_separation", "observation_cost",
                        "threshold_interval", "interval_width")
                       if key in suggested},
                    "status_when_returned": suggested.get("status"),
                    "query_action_path": [*location, "native_condition_research",
                                          "proposed_experiments", proposal_index, "query_action"],
                    "after_observation_operators": [item["operator"] for item in
                        query.get("after_observation", []) if "operator" in item]})
            for candidate_index, candidate in enumerate(candidates):
                if candidate.get("model"):
                    associated = [deepcopy(preview) for suggested, preview in
                        zip(proposals[:3], proposal_preview)
                        if (suggested["candidate_program"] == candidate.get("program")
                            if "candidate_program" in suggested else
                            suggested.get("candidate_source") == candidate.get("source")
                            and suggested.get("candidate_source") is not None)]
                    native_models[candidate["model"]] = (candidate,
                        [*location, "native_condition_research", "candidates", candidate_index],
                        associated)
            summary["latest_native_result"] = {
                "step": entry.get("step", index), "status": native.get("status"),
                "json_path": [*location, "native_condition_research"],
                "candidate_count": len(candidates),
                "candidate_preview": [{key: candidate[key]
                    for key in ("model", "fit_mse", "evaluate_action") if key in candidate}
                    for candidate in candidates[:3]],
                "candidate_preview_order": "first three in returned order",
                "proposed_experiment_count": len(proposals),
                "proposed_experiment_preview": proposal_preview,
                "proposed_experiment_preview_order": "first three in returned order",
                # These are the researcher's actual outcome and executable
                # continuation, not conclusions inferred from the paid-fit MSE.
                # Keep them with practice feedback so the author can distinguish
                # an unresolved parameter from a refuted structure.
                **{key: deepcopy(native[key]) for key in
                   ("research_decision", "remaining_boundary", "boundary_update",
                    "completion_reserve", "compute_cost", "round_index", "search_stages",
                    "compute_continuation", "search_allowance_estimate")
                   if key in native}}
    # A method's explicit stop is retained in its execution event even when it
    # returned before another method_results entry could be recorded.
    for event in reversed(result.get("events", [])[start.get('events', 0):]):
        if method_path is not None and event.get("arguments", {}).get("path") != method_path:
            continue
        feedback = event.get("feedback") or {}
        if (feedback.get("reason") or feedback.get("error") or feedback.get("timeout")
                or feedback.get("returncode") not in (None, 0)):
            summary["last_execution_message"] = {
                key: feedback[key] for key in
                ("status", "reason", "error", "returncode", "timeout", "program_execution")
                if key in feedback}
            stderr = (feedback.get("stderr") or "").strip().splitlines()
            if stderr:
                summary["last_execution_message"]["stderr"] = stderr[-1]
            break
    return summary


def candidate_execution_summary(summary):
    """Keep paid execution facts beside a source without repeating its trace."""
    compact = {key: deepcopy(summary[key]) for key in (
        "status", "step_count", "action_counts", "submission_count", "confirmation_count",
        "last_action", "latest_confirmation", "last_execution_message", "result_file",
        "result_steps_field", "latest_retention_decision", "practice_execution_cost",
        "latest_program_execution", "latest_author_work_request") if key in summary}
    for field in ("latest_evaluation", "latest_submission_paid_fit"):
        if field in summary:
            compact[field] = {key: deepcopy(summary[field][key]) for key in (
                "step", "model", "mse", "loss", "evaluation_scope", "returncode", "timeout",
                "evaluated_rows", "fitting_view", "later_readings", "retained_comparison",
                "evaluated_model", "retention_decision")
                if key in summary[field]}
    if "latest_submitted_predictor" in summary:
        submitted = summary["latest_submitted_predictor"]
        compact["latest_submitted_predictor"] = {key: deepcopy(submitted[key]) for key in (
            "step", "source_model", "model_file", "json_path", "paid_rows", "mse",
            "source_preview", "native_candidate", "fresh_confirmation") if key in submitted}
    return compact


def restore_paid_practice_summaries(practice_trials, workspace, directories):
    """Read only explicitly selected legacy paid trials missing execution metadata."""
    trials = {trial["directory"]: trial for trial in practice_trials}
    restored = []
    for directory in dict.fromkeys(directories):
        trial = trials[directory]
        if "costs" in trial and trial.get("execution_summary"):
            continue
        result = json.loads((Path(workspace) / directory / "result.json").read_text())
        if "costs" not in trial:
            trial["costs"] = deepcopy(result["costs"])
        if not trial.get("execution_summary"):
            trial["execution_summary"] = practice_execution_summary(result, directory)
        restored.append(directory)
    return restored


PAIRED_INTERFACE = """
Learn a self-contained research program that checks whether its earlier supporting
model still applies, preserves useful knowledge, and repairs it when evidence
requires that. The listed practice cases expose the same public interface and earlier
knowledge. Their identities do not reveal whether anything changed. Choose your
own experiments, numerical conditions and repairs. Do not always rebuild first:
the question is whether your program can decide from evidence when that is useful.
You can revise the inherited program before any execution; it is a reference,
not a mandatory first experiment.

AUTHORING ACTIONS:
write {path:'shared/consolidated_method.py',code:'Python source'};
run {code:'Python source'} to inspect your files and paid practice traces;
run {operator:'practice',case:CASE_LABEL} to execute the current program
from empty observations and the SAME frozen prior knowledge in a fresh workspace;
run {operator:'restore_method',trial:'trials/trial_NNN'} to select a previously
practiced source exactly as saved in that trial;
finish {path:'shared/consolidated_method.py',lesson:'...'} to freeze your method.
Each practice costs real observations, measurements and computation. Feedback
provides the relative trial directory; its result.json and paid evidence files
are available for inspection. No automatic refitting occurs before your program.
method_candidates.json lists your practiced sources, grouped by exact source,
with confirmed and failed cases, actual trial directories, and the current source.
The same case can appear in both lists when that source had different outcomes.
Choose a candidate's trial to restore its source and existing same-source coverage.
Restoration does not publish or repair the program; missing cases still require
practice under the same publication conditions. You may also keep editing.
Any study-specific practice-after-edit rule also applies to restoring a source.
Publication needs fresh predictive confirmation on EVERY listed case with exactly the
current source. It must actually use the unchanged earlier predictor on at least
one case, and confirm a revised predictor on another case. Which case supports
which choice is not supplied: establish that from paid evidence. Solving all cases
by refitting does not demonstrate the requested reuse-and-revision capability.
A final unexecuted edit loses that coverage. Retesting any case
starts from its original prior, not the previous trial's fitted numerical models.
Write and practice feedback report current_source_coverage: the cases that actually
confirmed this exact program, the cases still missing, and its observed reuse and
revision outcomes. This is evidence about the current file, not earlier versions.
If resume_context.json is present, an earlier authoring phase ended before
publication. That file records its status, saved current-source coverage and last action; the
underlying paid trial files remain available. Publication conditions are unchanged.
Ordinary Python inspection is available; query/evaluate and individual fitting
operators are not authoring actions. Request experiments through your program.

EXECUTABLE PROGRAM CONTRACT:
def select_actions(public_spec, records, memory): ...
records is a list of trajectory RECORDS, each with action, rows, measurements and
row_indices. len(records) counts trajectories, NOT observation rows. Existing
record indices are range(len(records)); observation rows count as
sum(len(record['rows']) for record in records). No records exist at the first call.
memory['research_roles'] gives instrument and predictor file paths for the earlier
knowledge. Use these roles across apparatuses; do not infer mechanisms from case
identifiers, filenames or directories. Newly fitted files are local to a trial.
research_state.json exposes flat operator_results and method_results from actual
previous invocations, plus costs and limits. All are reset for a new trial.
Practice feedback.steps uses these same method_results entries, each containing
{step,path,proposal,feedback}; a submission's feedback
contains fresh_confirmation:{passed,mse,rows,...} or an error. The last operator
result has operator and its numerical feedback fields at the same level.
proposal is null when the program fails before returning an action; its feedback
holds the execution error. Use proposal or {} when inspecting prior action fields.

Return {'actions':[trajectory,...], 'after_observation':[operator arguments,...]}
to buy data and analyze it, or actions:[] with nonempty after_observation to analyze
already-paid records. A whole continuation runs linearly; the next invocation can
branch on its actual outcomes. Both lists empty is invalid.
Return {'model':'path/to/predictor.py','lesson':'...'} alone to request fresh
confirmation and finish. Selecting the unchanged earlier predictor is allowed.
The host never selects a candidate or an experimental strategy for you. A failed
confirmation adds its counterexamples to paid records and permits another step.
The exact same source executes up to eight steps per trial, without another LLM
choosing intermediate actions. After publication it will also run from empty
observations with another previously learned calibration. Only your source and
lesson transfer; practice-fitted models and observations do not.

OPERATOR REFERENCE FOR THE PROGRAM'S after_observation ENTRIES:
""" + OPERATOR_INTERFACE


class _PracticeInterface:
    def __init__(self, spec, cases):
        self.spec = deepcopy(spec)
        self.task_interface = ("Author and practice a research program on these cases: "
                               + ", ".join(cases) + ".\n")

    def public_spec(self):
        return deepcopy(self.spec)


class PairedMethodLearner(InquiryLLMAgent):
    common_interface = PAIRED_INTERFACE

    def __init__(self, provider, workspace, *, knowledge, cases, lab_factory, config,
                 paid_trace_references=None):
        super().__init__(provider, workspace, arm="hegel", knowledge=knowledge,
                         config=config, source_task=False, paid_trace_references=paid_trace_references)
        self.cases, self.lab_factory = deepcopy(cases), lab_factory
        self.practice_trials = []
        self.published_source = None
        self.common_interface = PAIRED_INTERFACE.replace("CASE_LABEL", repr(next(iter(cases))))
        self.interface = _PracticeInterface(lab_factory(next(iter(cases.values()))).public_spec(), cases)
        self._refresh_method_candidates()

    def run(self, *, on_checkpoint=None):
        return super().run(self.interface, on_checkpoint=on_checkpoint)

    def restore_practice(self, previous_result):
        """Reference completed paid trials; charge and write new work separately."""
        published = (previous_result["status"] == "completed"
                     and isinstance(previous_result.get("method_source"), str)
                     and bool(previous_result["method_source"]))
        if not published and previous_result["status"] not in {
                "provider_failed", "runtime_interrupted", "token_budget_exhausted", "model_call_budget_exhausted",
                "source_repair_inconclusive", "python_execution_budget_exhausted"}:
            raise ValueError("practice restoration expects an interrupted author or an actually published method")
        source = (previous_result["method_source"] if published
                  else previous_result["outgoing_knowledge"]["files"][METHOD_PATH])
        self.model, self.published_source = None, None
        (self.workspace / METHOD_PATH).write_text(source)
        self.files[METHOD_PATH] = source
        self.practice_trials = deepcopy(previous_result["practice_trials"])
        previous_workspace = Path(previous_result["workspace"])
        for trial in self.practice_trials:
            source = previous_workspace / trial["directory"]
            target = self.workspace / trial["directory"]
            # Finished trials are evidence, not writable practice workspaces.
            # Keep their stable IDs while referring to the original paid trees;
            # a later practice gets its own new numbered directory below trials.
            if source.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve(), target_is_directory=True)
        last_action = next((event for event in reversed(previous_result["events"])
                            if event["action"] != "provider_error"), None)
        last_action_summary = _compact_trajectory_payload(last_action)
        if last_action is not None:
            action_file = "resume_last_action.json"
            write_text_incrementally(self.workspace / action_file, json.dumps(last_action, indent=2))
            last_action_summary["full_action_file"] = action_file
            if last_action["action"] == "run" and "operator" not in last_action.get("arguments", {}):
                feedback = last_action_summary.get("feedback", {})
                stdout = feedback.get("stdout", "")
                if len(stdout) > 2000:
                    # A previous inspection's output must not crowd the resume
                    # status and paid-practice record out of an ordinary read.
                    feedback.update(stdout=stdout[:256], stdout_characters=len(stdout),
                                    stdout_truncated=True)
                    last_action_summary.pop("usage", None)
        context = {"resumed_from": str(previous_workspace), "previous_status": previous_result["status"],
                   "current_source_coverage": self._current_source_coverage(),
                   "previous_practice": [{key: value for key, value in trial.items() if key != "method_source"}
                                         for trial in self.practice_trials],
                   "last_action": last_action_summary,
                   "cost_scope": "Earlier paid trials are retained. Current costs and limits apply only "
                                 "to this new authoring phase; earlier costs remain in the preceding result.",
                   "history_access": "Historical trial directories reference the original paid records "
                                     "for reading. Edit or restore methods only into the current shared source; "
                                     "new practices use new numbered directories."}
        self.files["resume_context.json"] = json.dumps(context, indent=2)
        write_text_incrementally(self.workspace / "resume_context.json", self.files["resume_context.json"])
        return context

    def _refresh_method_candidates(self):
        """Expose the source snapshots and paid outcomes already held by practice."""
        current = (self.workspace / METHOD_PATH).read_text() if self.practice_trials else None
        grouped = {}
        for trial in self.practice_trials:
            source = trial["method_source"]
            if source not in grouped:
                grouped[source] = {"trial": trial["directory"],
                    "source_path": str(Path(trial["directory"]) / METHOD_PATH),
                    "current": source == current, "confirmed_cases": [], "failed_cases": [],
                    "trials": []}
            candidate = grouped[source]
            cases = candidate["confirmed_cases" if trial["status"] == "completed" else "failed_cases"]
            if trial["case"] not in cases:
                cases.append(trial["case"])
                cases.sort()
            evidence = {key: deepcopy(trial[key]) for key in (
                "case", "directory", "status", "fresh_confirmation", "selected_prior_predictor")}
            if "costs" in trial:
                evidence["costs"] = deepcopy(trial["costs"])
            if trial.get("execution_summary"):
                evidence["execution_summary"] = candidate_execution_summary(trial["execution_summary"])
            candidate["trials"].append(evidence)
        content = json.dumps(list(grouped.values()), indent=2)
        self.files["method_candidates.json"] = content
        write_text_incrementally(self.workspace / "method_candidates.json", content)

    def _latest_source_practices(self, source):
        """A later practice on the same case updates that source's support."""
        return {trial["case"]: trial for trial in self.practice_trials
                if trial["method_source"] == source and trial["case"] in self.cases}

    def _current_source_coverage(self):
        source = (self.workspace / METHOD_PATH).read_text()
        successful = [trial for trial in self._latest_source_practices(source).values()
                      if trial["status"] == "completed"
                      and (trial.get("fresh_confirmation") or {}).get("passed") is True]
        confirmed = {trial["case"] for trial in successful}
        coverage = {"confirmed_cases": sorted(confirmed),
                "missing_cases": sorted(set(self.cases) - confirmed),
                "prior_reuse_cases": sorted({trial["case"] for trial in successful
                                             if trial["selected_prior_predictor"]}),
                "revised_cases": sorted({trial["case"] for trial in successful
                                         if not trial["selected_prior_predictor"]})}
        if "resume_context.json" in self.files:
            context = json.loads(self.files["resume_context.json"])
            context["current_source_coverage"] = coverage
            self.files["resume_context.json"] = json.dumps(context, indent=2)
            write_text_incrementally(self.workspace / "resume_context.json", self.files["resume_context.json"])
        self._refresh_method_candidates()
        return coverage

    def _execute(self, action, arguments, laboratory):
        if action == "write":
            if arguments.get("path") != METHOD_PATH:
                raise ValueError("write the self-contained method at shared/consolidated_method.py")
            feedback = super()._execute(action, arguments, laboratory)
            return {**feedback, "current_source_coverage": self._current_source_coverage()}
        if action == "run" and "operator" not in arguments:
            feedback = super()._execute(action, arguments, laboratory)
            source = (self.workspace / METHOD_PATH).read_text()
            if source != self.files[METHOD_PATH]:
                # Ordinary Python can adopt or edit a stored method just as write can.
                # The next author decision must see the source that practice executes.
                self.files[METHOD_PATH] = source
                return {**feedback, "current_source_coverage": self._current_source_coverage()}
            return feedback
        if action == "run" and arguments.get("operator") == "practice":
            return self._practice(arguments["case"])
        if action == "run" and arguments.get("operator") == "restore_method":
            trial = next((item for item in self.practice_trials
                          if item["directory"] == arguments.get("trial")), None)
            if trial is None:
                raise ValueError("choose a recorded trial from method_candidates.json")
            feedback = super()._execute("write", {"path": METHOD_PATH,
                "code": trial["method_source"]}, laboratory)
            return {**feedback, "restored_method": trial["directory"],
                    "current_source_coverage": self._current_source_coverage()}
        if action != "finish":
            raise ValueError("authoring uses write, ordinary run, run operator:practice/restore_method, and finish")
        if arguments.get("path") != METHOD_PATH:
            raise ValueError("finish needs path:shared/consolidated_method.py")
        source = (self.workspace / METHOD_PATH).read_text()
        coverage = self._current_source_coverage()
        confirmed, missing = coverage["confirmed_cases"], coverage["missing_cases"]
        if missing:
            return {"error": "the current method still needs fresh confirmation on: " + ", ".join(missing),
                    "unconfirmed_cases": missing}
        selection_error = self._prior_selection_error(coverage)
        if selection_error is not None:
            return selection_error
        additional_error = self._publication_error(coverage)
        if additional_error is not None:
            return additional_error
        self.model, self.lesson, self.published_source = METHOD_PATH, arguments.get("lesson", ""), source
        return {"published": METHOD_PATH, "confirmed_cases": sorted(confirmed)}

    def _publication_error(self, coverage):
        return None

    def _prior_selection_error(self, coverage):
        reused, revised = coverage["prior_reuse_cases"], coverage["revised_cases"]
        if not any(a != b for a in reused for b in revised):
            return {"error": "prediction succeeded but the current method has not demonstrated actual "
                             "unchanged-prior reuse in one case and confirmed revision in another; "
                             "use evidence to decide rather than rebuilding in every case",
                    "confirmed_prior_reuse_cases": sorted(reused), "confirmed_revision_cases": sorted(revised)}
        return None

    def _practice(self, case):
        if case not in self.cases:
            raise ValueError("choose " + " or ".join(self.cases))
        remaining = self.config.observation_budget - self.costs["observation_cost"]
        if remaining < 48:
            return {"error": "remaining observations cannot cover fresh confirmation"}
        source = (self.workspace / METHOD_PATH).read_text()
        directory = f"trials/trial_{len(self.practice_trials):03d}"
        limits = replace(self.config, observation_budget=min(240, remaining), confirmation_rows=48,
            compute_budget=min(100000, self.config.compute_budget - self.costs["compute_cost"]),
            execution_seconds=min(60., self.config.execution_seconds - self.costs["python_execution_seconds"]))
        agent = FrozenMethodAgent(self.workspace / directory, knowledge=self.knowledge,
            method_source=source, config=limits,
            confirmation_seed=self.cases[case]["confirmation_seed"] + len(self.practice_trials))
        result = agent.execute_method(self.lab_factory(self.cases[case]), max_steps=8)
        prior_path = self.knowledge["research_roles"]["predictor"]
        selected_prior = (result["model"] == prior_path
                          and (agent.workspace / prior_path).read_text() == self.knowledge["files"][prior_path])
        return self._record_practice(case, directory, source, agent, result, selected_prior)

    def _record_practice(self, case, directory, source, agent, result, selected_prior):
        """Keep paid outcomes and current-source coverage shared across laboratories."""
        write_text_incrementally(agent.workspace / "result.json",
            json.dumps(result, separators=(',', ':')), base=result.get('resume_origin'))
        charged = result.get('continuation_cost', result['costs'])
        for key in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
            self.costs[key] += charged[key]
        continuation = {key: deepcopy(result[key]) for key in
            ('prior_research_cost', 'continuation_cost', 'resume_origin', 'practice_start',
             'research_work') if key in result}
        execution_summary = practice_execution_summary(result, directory)
        confirmation = result.get("fresh_confirmation")
        if result.get('practice_start') and not execution_summary['confirmation_count']:
            confirmation = None
        feedback = {"case": case, "directory": directory, "status": result["status"],
                    "costs": result["costs"], "model": result["model"],
                    "selected_prior_predictor": selected_prior,
                    **continuation,
                    "fresh_confirmation": {key: confirmation.get(key) for key in ("passed", "mse", "rows")}
                        if confirmation else None,
                    "steps": [_compact_practice_feedback(step,
                        ['method_results', index], str(Path(directory) / 'result.json'))
                        for index, step in enumerate(result['method_results'])
                        if index >= result.get('practice_start', {}).get('method_results', 0)],
                    "execution_summary": execution_summary}
        feedback['full_trace_file'] = str(Path(directory) / 'result.json')
        if result.get("no_progress") is not None:
            feedback["no_progress"] = deepcopy(result["no_progress"])
        self.practice_trials.append({"case": case, "directory": directory,
            "method_source": source, "status": result["status"], "costs": result["costs"],
            "selected_prior_predictor": selected_prior,
            **continuation,
            "fresh_confirmation": feedback["fresh_confirmation"],
            "execution_summary": execution_summary,
            **({"no_progress": deepcopy(result["no_progress"])} if "no_progress" in result else {})})
        feedback["current_source_coverage"] = self._current_source_coverage()
        return feedback

    def result(self, status):
        result = super().result(status)
        result.update(practice_trials=deepcopy(self.practice_trials),
                      method_source=self.published_source, method_path=self.model, lesson=self.lesson,
                      method_entry={"id": "paired_evidence_method", "path": METHOD_PATH}
                          if self.published_source else None)
        return result
