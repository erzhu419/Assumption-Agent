"""Bind the existing method author/practice loop to the anonymous apparatus."""

from copy import deepcopy

from .common_archive_inquiry import CommonArchiveInquiryAgent, ONLINE_RESEARCH_INTERFACE
from .inquiry_revision import REVISION_INTERFACE
from .observable_research import OBSERVABLE_INTERFACE
from .conditional_research import CONDITIONAL_INTERFACE
from .python_method_learning import PreparedPythonMethodAgent, PythonArchiveMethodLearner
from .temporal_method_learning import METHOD_PATH
from .temporal_paired_method import FROZEN_METHOD_INTERFACE, execute_procedure


EMPTY_METHOD = '''def select_actions(spec, records, memory):
    return {'stop': 'No research method has been authored yet.'}
'''

AUTHOR_INTERFACE = """
Turn the supplied actual research archives into one self-contained Python research
procedure. source_catalog.json identifies each earlier workspace, submitted model,
paid trace and outcome. These files include successful and unsuccessful actions;
inspect the evidence rather than assuming every archived file is supported.
Archive paths and case labels identify files, not the mechanism of a new apparatus.

AUTHOR ACTIONS:
write {path:'shared/consolidated_method.py',code:'Python source'};
run {code:'Python source'} to inspect archives and paid practice results;
run {operator:'practice',case:'case_000'} to execute your current method;
run {operator:'restore_method',trial:'trials/trial_NNN'} to restore a practiced draft;
finish {path:'shared/consolidated_method.py',lesson:'...'} to request publication.
The initial method is only an unresolved placeholder. You author its decisions,
predictors, parameter estimation and experiments using ordinary Python.

PRACTICE AND PUBLICATION:
Each anonymous practice starts from the same complete source archive and a fresh
paid 72-row prefix. The source archive is not current-task
evidence. A trial receives 576 observations including its prefix and 96 fresh
confirmation rows, subject to the remaining author account. It executes at most
12 program invocations without an LLM making intermediate decisions. Your program
can use actual feedback to repair a failure before that limit.
Every practice costs real observations and execution. Its directory contains
result.json, laboratory_observations.json and the actual generated files.
Repeated identical errors or completed comparisons on unchanged evidence return
method_no_progress to the author, with the repeated steps and their paid outcomes.
This ends that practice without confirming or refuting its scientific hypotheses.
Inspect the retained evidence and decide whether to revise or restore your method.
Changing shared/consolidated_method.py through write or ordinary run also requests
one paid practice. It uses the most recently failed case still unconfirmed for
this exact source, or the first missing case in listed order. Its actual outcome,
including execution errors and failed confirmation, returns under edit_practice.
Pure inspection and unchanged writes buy no practice. This executes your program
without changing it or publishing it; explicit practice and finish remain available.
Publication requires the SAME method source to confirm every listed practice,
including unchanged earlier-model reuse on at least one case and a revised model
on another. No case is labeled with the appropriate choice. Decide from evidence.
This is an executable transfer task: only the published method and the original
source archive transfer, not your practice-fitted files or target observations.

finish practices missing cases in their listed order. The first failure returns
its evidence to you. It also compares the latest earlier draft with paid support
and no unresolved failures, using the same account. An older draft publishes only
if it meets all the same conditions; otherwise your current draft is restored for
further work. method_decision.json and method_candidates.json expose this existing
paid support and the restore actions. Editing invalidates support for that draft.
You can inspect, revise and practice again within the account. No host-written
fitting procedure or experimental design is supplied.

PROGRAM INPUTS:
select_actions(public_spec, records, memory) receives trajectory records, each
with action, rows, measurements and row_indices. len(records) counts trajectories;
sum(len(r['rows']) for r in records) counts observations. Numeric row targets are
paid evidence. research_state.json contains method_results with exact proposals
and feedback, costs and limits. memory['research_roles']['predictors'] lists the
archived predictor paths; source_catalog.json gives their original working
directories. Evaluating an archived path retains its original training view.
Ordinary inspection scripts with relative paths must run in the catalog directory.
All source archives are available equally on every trial, never selected by the
hidden identity of the target. The program must select from observations.

PROGRAM ACTIONS:
An ordinary {'action':'run','arguments':{'code':...}} executes your Python;
{'action':'write','arguments':{'path':...,'code':...}} writes a file;
{'action':'evaluate','arguments':{'model':...}} scores already-paid evidence;
{'actions':[...]} buys your chosen new trajectories. Evaluation alone does not
confirm a model. {'model':path} requests 96 fresh rows and ordinary confirmation
at MSE <= 1e-10. Failed confirmation becomes paid evidence on the next invocation.
The program may create local predictors and dependencies. Use only its own trial
workspace, public interface and inherited files. The following apparatus and
program contracts apply INSIDE a practice, not as direct authoring actions.
""" + FROZEN_METHOD_INTERFACE + REVISION_INTERFACE + ONLINE_RESEARCH_INTERFACE + OBSERVABLE_INTERFACE + CONDITIONAL_INTERFACE


class AnonymousResearchMethodLearner(PythonArchiveMethodLearner):
    """Reuse publication, incumbent retention and paid practice without old solvers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_interface = (AUTHOR_INTERFACE.replace(
            'A trial receives 576 observations',
            f'A trial receives {self.practice_limits.observation_budget} observations')
            + self.lab_factory(next(iter(self.cases.values()))).task_interface)

    def _execute(self, action, arguments, laboratory):
        # Reconnect the existing edit/practice loop without the old meter-specific
        # publication requirements. Practice selection uses paid outcomes only.
        if action != "write" and not (action == "run" and "operator" not in arguments):
            return super()._execute(action, arguments, laboratory)
        path = self.workspace / METHOD_PATH
        before = path.read_text()
        feedback = super()._execute(action, arguments, laboratory)
        if path.read_text() == before:
            return feedback
        missing = self._current_source_coverage()["missing_cases"]
        if not missing:
            return feedback
        first_missing = next(case for case in self.cases if case in missing)
        case = next((trial["case"] for trial in reversed(self.practice_trials)
            if trial["case"] in missing and (trial["status"] != "completed" or
                (trial.get("fresh_confirmation") or {}).get("passed") is not True)), first_missing)
        outcome = self._practice(case)
        return {**feedback, "edit_practice": outcome,
                "current_source_coverage": self._current_source_coverage()}


class AnonymousPreparedMethodAgent(CommonArchiveInquiryAgent, PreparedPythonMethodAgent):
    """Practice and freeze methods with the same tools as the online researcher."""

    def _result_files(self, manifest, manifest_paths, model_paths):
        # Practice transfers the authored method and original archive, never
        # its fitted files. Keep those files in this workspace instead of also
        # embedding every candidate snapshot into the practice result.
        return {}

    def result(self, status):
        result = super().result(status)
        result["outgoing_knowledge"].pop("files")
        result["outgoing_knowledge"]["workspace"] = str(self.workspace)
        result["inherited_files_changed"] = sorted(
            name for name in self.inherited_files
            if (self.workspace / name).is_file()
            and (self.workspace / name).read_text(encoding="utf-8") != self.knowledge["files"][name])
        result["knowledge_protocol"] = "method_practice_workspace"
        return result

    def _attach_research_advice(self, action, arguments, result, laboratory):
        result.pop("available_research_action", None)
        self._current_research_advice = None
        return result


class SingleSubmissionMethodAgent(AnonymousPreparedMethodAgent):
    """End research at its first submission, then perform ordinary fresh confirmation."""

    def _handle_program_proposal(self, proposed, laboratory):
        if (not isinstance(proposed, dict) or proposed.get("action") != "query"
                or not proposed.get("arguments", {}).get("path")):
            return super()._handle_program_proposal(proposed, laboratory)
        path = self._active_program_path
        try:
            result = super()._handle_program_proposal(proposed, laboratory)
        finally:
            self._current_proposal, self._active_program_path = deepcopy(proposed), path
        result["executed_path"] = path
        if result.get("status") == "confirmation_not_authorized":
            result["procedure_status"] = result["status"]
            result["status"] = "submission_proposed"
        return result

    def _execute(self, action, arguments, laboratory):
        if action != "run" or arguments.get("operator") != "procedure":
            return super()._execute(action, arguments, laboratory)
        proposal, path = deepcopy(self._current_proposal), self._active_program_path
        try:
            result = super()._execute(action, {**arguments, "confirm": False}, laboratory)
        finally:
            self._current_proposal, self._active_program_path = proposal, path
        if result.get("status") == "confirmation_not_authorized":
            # A child's explanation is a proposal for its parent, not the
            # enclosing research program's final submission.
            result["procedure_status"] = result["status"]
            result["status"] = "submission_proposed"
        return result

    def execute_method(self, laboratory, max_steps=8, *, on_checkpoint=None):
        pending = self.method_results[-1] if self.method_results else None
        if pending and pending['feedback'].get('status') == 'confirmation_not_authorized':
            # A checkpoint may have persisted the final declaration just before
            # its confirmation. Continue that declaration without another choice.
            proposal = deepcopy(pending['proposal'])
            self._active_program_path = pending['path']
            self._current_proposal = deepcopy(proposal)
        else:
            outcome = execute_procedure(self, laboratory, METHOD_PATH, max_steps,
                allow_confirmation=False, record_events=True, on_checkpoint=on_checkpoint)
            if outcome["status"] != "confirmation_not_authorized":
                return self.result(outcome["status"])
            proposal = deepcopy(outcome["steps"][-1]["proposal"])

        # The program has already selected its explanation. Execute that saved
        # declaration without invoking it again or returning new labels to it.
        try:
            feedback = self._handle_program_proposal(proposal, laboratory)
        except (ValueError, TypeError, KeyError, OSError) as error:
            feedback = {"error": f"{type(error).__name__}: {error}",
                        "executed_path": METHOD_PATH}
        self.method_results[-1]["feedback"] = self._compact_feedback(feedback)
        self.events[-1]["feedback"] = deepcopy(feedback)
        if self.model is not None:
            status = "completed"
        elif feedback.get("fresh_confirmation", {}).get("passed") is False:
            status = "confirmation_failed"
        else:
            status = "method_error"
        self._prepare_program_state()
        result = self.result(status)
        if on_checkpoint:
            on_checkpoint(result)
        return result
