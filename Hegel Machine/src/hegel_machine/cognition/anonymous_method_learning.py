"""Bind the existing method author/practice loop to the anonymous apparatus."""

from copy import deepcopy
from dataclasses import replace
import ast
import json
from pathlib import Path

from .common_archive_inquiry import (
    CommonArchiveInquiryAgent, ONLINE_RESEARCH_INTERFACE, _REFIT_INTERFACE,
)
from .checkpoint_storage import checkpoint_file_text, read_json_checkpoint, write_text_incrementally
from .inquiry_revision import REVISION_INTERFACE
from .observable_research import OBSERVABLE_INTERFACE
from .conditional_research import CONDITIONAL_INTERFACE
from .online_inquiry_llm import InquiryLLMAgent, open_assigned_question
from .python_method_learning import PreparedPythonMethodAgent, PythonArchiveMethodLearner
from .structural_inquiry import NATIVE_CONDITION_INTERFACE
from .temporal_method_learning import METHOD_PATH
from .temporal_paired_method import FROZEN_METHOD_INTERFACE, execute_procedure
from .temporal_paired_practice import practice_execution_summary


def compact_workspace_result(agent, result):
    """Reuse the practice result protocol while retaining complete execution files on disk."""
    result["outgoing_knowledge"].pop("files")
    result["outgoing_knowledge"]["workspace"] = str(agent.workspace)
    # Practice evidence remains in its workspace. Its outgoing history is
    # a view of the complete restorable questions, not another bank copy.
    inherited = len(agent.knowledge.get("question_traces", []))
    # Checkpoints borrow the live question records. Detach only the containers
    # whose native-bank field is replaced, retaining the large bank by reference.
    traces = list(result["outgoing_knowledge"]["question_traces"])
    result["outgoing_knowledge"]["question_traces"] = traces
    result["online_questions"] = list(result["online_questions"])
    state_path = agent.workspace / "research_state.json"
    saved_questions = (read_json_checkpoint(state_path, keys={"questions"})["questions"]
        if state_path.is_file() and any("native_condition_research" in question
            for question in agent.questions) else [])
    for index, question in enumerate(agent.questions):
        if ("native_condition_research" in question and index < len(saved_questions)
                # JSON arrays also represent the native engine's tuples.
                and json.loads(json.dumps(question["native_condition_research"]))
                    == saved_questions[index].get("native_condition_research")):
            reference = {
                "full_content_file": "research_state.json",
                "json_path": ["questions", index, "native_condition_research"],
                "round_count": len(question["native_condition_research"]),
            }
            traces[inherited + index] = {**traces[inherited + index],
                "native_condition_research": deepcopy(reference)}
            result["online_questions"][index] = {**result["online_questions"][index],
                "native_condition_research": reference}
    result["inherited_files_changed"] = sorted(
        name for name in agent.inherited_files
        if (agent.workspace / name).is_file()
        and checkpoint_file_text(agent, agent.workspace / name) != agent.knowledge["files"][name])
    result["knowledge_protocol"] = "method_practice_workspace"
    return result


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
The same ordinary run can read and edit selected parts of
shared/consolidated_method.py, preserving its other source text. You choose
whether to make a local edit, restore a draft, or replace the complete program.
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
You may preserve an earlier model's fit/predict source in your self-contained
method and write it into a trial's workspace. Reuse its fitting algorithm with
that trial's paid observations; source-task fitted parameters and observations
are not evidence for the new apparatus. The existing refit operation below
validates the unchanged fitting source on separate current paid trajectories.

finish practices missing cases in their listed order. The first failure returns
its evidence to you. It also compares the latest earlier draft with paid support
and no unresolved failures, using the same account. An older draft publishes only
if it meets all the same conditions; otherwise your current draft is restored for
further work. method_decision.json and method_candidates.json expose this existing
paid support and the restore actions. Editing invalidates support for that draft.
current_source_latest_failure and current_source_latest_evaluation belong to your
currently selected method. latest_practice is chronological and may instead be an
older comparison draft; source_matches_current and latest_incumbent_comparison
identify that distinction. An execution_summary's latest_submitted_predictor names
the frozen predictor actually confirmed, its native expression and fitted structure
when available, and the corresponding residuals and experiment records. These
describe the executed candidate, which can differ from a fitter in your method source.
current_source_entry locates the effective select_actions definition and any
earlier definitions shadowed by it. Native execution summaries retain the actual
research_decision, remaining_boundary and boundary_update alongside candidate
scores, with the corresponding executable next experiment when one was returned.
You can inspect, revise and practice again within the account. You choose whether
and how to use the shared finite research operators or ordinary Python.

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
""" + FROZEN_METHOD_INTERFACE + _REFIT_INTERFACE + REVISION_INTERFACE + ONLINE_RESEARCH_INTERFACE + OBSERVABLE_INTERFACE + CONDITIONAL_INTERFACE + NATIVE_CONDITION_INTERFACE


class AnonymousResearchMethodLearner(PythonArchiveMethodLearner):
    """Reuse publication, incumbent retention and the shared online research tools."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_interface = (AUTHOR_INTERFACE.replace(
            'A trial receives 576 observations',
            f'A trial receives {self.practice_limits.observation_budget} observations')
            + self.lab_factory(next(iter(self.cases.values()))).task_interface)

    def _request(self, laboratory):
        request = super()._request(laboratory)
        request['research_notes'] = self.files.get('research_notes.md', '')
        return request

    def result(self, status):
        result = super().result(status)
        # Author reasoning survives fresh provider sessions and checkpoint restore;
        # it is not part of the shared executable source exported to consumers.
        result['author_research_notes'] = self.files.get('research_notes.md', '')
        return result

    def restore_practice(self, previous_result):
        result = super().restore_practice(previous_result)
        InquiryLLMAgent._execute(self, 'write', {'path': 'research_notes.md',
            'code': previous_result.get('author_research_notes', '')}, self.interface)
        return result

    def _skip_edit_practice(self, before, after):
        return None

    def _execute(self, action, arguments, laboratory):
        arguments = deepcopy(arguments)
        if 'notes' in arguments:
            InquiryLLMAgent._execute(self, 'write', {'path': 'research_notes.md',
                'code': arguments.pop('notes')}, laboratory)
        # Reconnect the existing edit/practice loop without the old meter-specific
        # publication requirements. Practice selection uses paid outcomes only.
        if action != "write" and not (action == "run" and "operator" not in arguments):
            return super()._execute(action, arguments, laboratory)
        path = self.workspace / METHOD_PATH
        before = path.read_text()
        feedback = super()._execute(action, arguments, laboratory)
        after = path.read_text()
        if after == before:
            return feedback
        skipped = self._skip_edit_practice(before, after)
        if skipped is not None:
            return {**feedback, 'edit_practice_skipped': skipped,
                    'current_source_coverage': self._current_source_coverage()}
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
        return compact_workspace_result(self, super().result(status))

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
                self._prepare_program_state()
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


SOURCE_REPAIR_INTERFACE = """
Repair your supplied shared/consolidated_method.py on the current paid source.
research_root retains this original assignment across local program choices.
A program question is a child investigation: rejecting one candidate or returning
from a stopped program leaves the source-repair assignment open. Historical
program outcomes in method_decision.candidates remain available for selection;
they are distinct from this work's independent-confirmation eligibility.
scene_source.json and paid_scene identify the latest paid scene and its records;
initial_scene preserves the original checkpoint. Your authoring directory is not
the laboratory workspace. Read program_state and observations at the paths in
that current scene index, including after another practice returns. The
existing method is your working artifact: inspect its effective select_actions,
preserve its supported branches, and revise its research decisions from evidence.
Distinguish an earlier search result from a completed search on current evidence.
Choose scientific hypotheses, parameters, representations and experiments yourself.

AUTHOR ACTIONS:
write {path:'shared/consolidated_method.py',code:'Python source'} edits the method;
ordinary run {code:...} may inspect evidence or edit this same file. Every actual
research-logic edit automatically practices the resulting method and returns its
execution. After a stopped practice, changing only the literal explanation in
return {'stop': '...'} saves the source without automatic practice. The feedback
identifies this skipped explanation edit and the earlier actual practice; it does
not confirm the changed source. Explicit practice remains available.
run {operator:'practice',case:CASE_LABEL} continues practice explicitly;
run {operator:'restore_method',trial:'trials/trial_NNN'} restores an earlier draft.
run {operator:'procedure',path:'a_selected_program.py',
     question:{id:'your_work',question:'your research objective',depends_on:[]}}
adopts that complete self-contained select_actions source as the current method,
then runs the existing practice executor on the latest paid scene. The selected
file can be an earlier method from method_candidates.json or a revised program
you have written with ordinary run. Its original file remains unchanged; only
the selected source is adopted, never the whole archive. An earlier program's
support does not confirm its current-scene execution. Use the same action with
another path to choose a different program; ordinary edits revise the selected
method and continue the same research work. All scientific decisions, including
which evidence to inherit or challenge, stay in YOUR executable program.
The question is retained under research_root in the scene and research_work.json.
Reusing its id preserves its original assignment while allowing its hypothesis to
change. A new question creates another child of the SAME source-repair assignment.
The question identifies work,
not a fitted parameter or proof that evidence was consumed. To make a selected
child result affect a candidate, your program must use the existing bind_research
and candidate bind_research(inputs)/fit contract, then refit or evaluate it.
Question dependency names alone do not change a predictor. Existing validated
local checks and parameter-region results are available in research_state.json;
inspect their scope and select the appropriate inputs yourself. Do not use checks
containing held-out labels as fit inputs for that same validation partition.
Return actual refit validation and candidate comparison before deciding another
experiment, a structural revision, or submission. The work result distinguishes
declared bindings from bindings actually reported by model execution.
Each practice executes at most 12 method invocations using the existing procedure
executor. Your program chooses and completes fitting, compares candidates on
current paid data, and then decides whether to experiment, revise, stop or submit.
The actual outcome returns to YOU for the next method edit, not to a free-action
parent researcher. A stopped or incomplete procedure is an unresolved trial.
program_execution reports actual Python execution, including time and printed diagnostics.
A returned stop is not evidence that no fitting ran or that a model family was refuted.
If a computed draft stopped without recorded numerical outcomes, the next deliverable
is a result-reporting repair of that draft. Keep its current scientific hypothesis and
paid data fixed for this diagnostic execution. Save the actual fit error, validation
partition errors, optimization termination, fitted parameters and candidate path where
available, including results that fail your submission criteria. Return or print the
report path and a concise numerical summary before stopping. If no fit ran, record the
actual reason instead. The host does not infer unrecorded scores.
Use that result to decide the next scientific revision. An exported candidate can be
compared through the existing evaluate action before it qualifies for confirmation;
the confirmation gate is unchanged. A repeated stop sentence is not a numerical test.
When numerical results and fitted parameters are already saved, consume them to export
your executable candidate before repeating optimization. Failing a submission threshold
does not prevent saving or evaluating that candidate to obtain its residuals. Compare
validation scores using the same explicit partitions and aggregation rule; the minimum
error from one held-out partition is not overall cross-validation performance.
paid_scene.evaluated_model identifies the current retained evaluation. Its separate
latest_submitted_predictor identifies the frozen predictor and paid-evidence range of
the last actual independent confirmation. That historical confirmation does not
automatically apply to a later refit, even when the source filename is unchanged.

Practices continue the latest paid scene, including new observations and failed
confirmations from earlier drafts. They preserve the confirmation stream and all
earlier records; costs report newly executed work separately from the cumulative
scene. Top-level limits and costs describe the author account: its confirmation_rows
may be zero because the author does not directly submit predictors. The live
practice_confirmation field gives the practice's configured fresh-row count,
confirmation threshold and remaining shared resources. resources_allow_practice
only reports whether a practice can start; it does not establish model eligibility
or confirmation success. Inside a practice, {'model': path} uses that practice's
ordinary submission gate and fresh confirmation, charged to the same observation
account. research_state.json contains prior and current procedure steps, current
costs, limits, retained candidate and search results. Read their actual evidence
cutoffs, completion state and candidate identity before deciding reuse. Current
practice summaries distinguish fitted error, later prediction and confirmation.
For a completed historical practice, its result.json method_results and continuation_cost
record the finished execution; an older research_state.json can predate its final step.
Method edits must turn those outcomes into executable research decisions.
research_work_result keeps the last executed program result across edits to the
same child question. Its method_path and question identify the executed draft;
source_matches_current and question_matches_current distinguish it from a new
revision. Revising a question does not erase completed fitting or comparison,
and retained results do not confirm the revised draft.

finish {path:'shared/consolidated_method.py',lesson:'...'} retains a repaired draft
only after this exact source passes a NEW independent confirmation on this scene.
This ends source repair. Publication still requires the separate existing
same-source practice requirements; transfer is not scheduled here.
finish {inconclusive:true,lesson:'local outcome and its consequence for the parent'}
returns the active child to the original source-repair assignment. The next author
turn chooses or revises a program from the returned evidence and historical methods.
The original goal and actual execution stay visible; a local rejection is not a
family-wide refutation. When the parent itself has no justified continuation,
finish {inconclusive:true,question_id:'source_method_repair',lesson:'specific obstacle'}
explicitly ends source repair without confirmation, publication or transfer.
Its paid work and selected program remain available for a later explicit revision.

PROGRAM INPUTS AND ACTIONS:
"""


class SourceProcedureRepairLearner(AnonymousResearchMethodLearner):
    """Use the existing edit/practice loop to repair one unresolved paid source."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_repair = None
        self.research_work = None
        self.research_work_result = None
        self.research_root = open_assigned_question(self, {
            'id': 'source_method_repair',
            'question': 'Repair the retained research method on this paid source: '
                'use measured evidence to choose or revise an executable research program, '
                'test its candidate on the matched comparison, and seek new independent '
                'confirmation. A local candidate rejection leaves this assignment open.',
            'parent_id': None, 'depends_on': []})
        self.research_root['phase'] = 'choose_or_revise_program'
        self._active_work_id = None
        self._paid_scene_scores = None
        contracts = AUTHOR_INTERFACE[AUTHOR_INTERFACE.index('PROGRAM INPUTS:'):]
        contracts = contracts.replace('96 fresh rows', f'{self.practice_limits.confirmation_rows} fresh rows')
        contracts = contracts.replace('MSE <= 1e-10',
            f'MSE <= {self.practice_limits.confirmation_mse_threshold:g}')
        self.common_interface = (SOURCE_REPAIR_INTERFACE.replace('CASE_LABEL', repr(next(iter(self.cases))))
            + contracts + self.lab_factory(next(iter(self.cases.values()))).task_interface)

    def _prepare_program_state(self):
        super()._prepare_program_state()
        if 'scene_source.json' not in self.files or not self.practice_trials:
            return
        original = json.loads(self.files['scene_source.json'])
        initial = original.get('initial_scene') or {key: original[key] for key in
            ('working_directory', 'training', 'observations', 'rows', 'program_state',
             'retained_method') if key in original}
        trial = self.practice_trials[-1]
        directory = trial['directory']
        if self._paid_scene_scores is None or self._paid_scene_scores[0] != directory:
            result = json.loads((self.workspace / directory / 'result.json').read_text())
            evaluated = deepcopy(result.get('evaluated_model'))
            if evaluated is not None and evaluated.get('model'):
                evaluated['model_file'] = str(Path(directory) / evaluated['model'])
            submitted = None
            steps = result.get('method_results', [])
            for index in range(len(steps) - 1, -1, -1):
                if not isinstance(steps[index].get('feedback', {}).get('fresh_confirmation'), dict):
                    continue
                # Reuse the existing submitted-source view, retaining its original
                # method_results index and ignoring later unconfirmed submissions.
                confirmation_result = {**result, 'method_results': steps[:index + 1],
                    'practice_start': {'method_results': index, 'events': len(result.get('events', []))}}
                submitted = practice_execution_summary(confirmation_result, directory).get(
                    'latest_submitted_predictor')
                break
            self._paid_scene_scores = (directory, {
                'evaluated_model': evaluated, 'latest_submitted_predictor': submitted})
        current = {'initial_scene': initial, 'working_directory': directory,
            'training': f'{directory}/result.json',
            'observations': f'{directory}/laboratory_observations.json',
            'rows': f'{directory}/observations.json',
            'program_state': f'{directory}/research_state.json',
            'retained_method': f'{directory}/{METHOD_PATH}', 'author_method': METHOD_PATH,
            'status': trial['status'], 'costs': deepcopy(trial['costs']),
            'fresh_confirmation': deepcopy(trial.get('fresh_confirmation')),
            **deepcopy(self._paid_scene_scores[1]),
            'scope': 'Latest completed paid practice, including all inherited evidence. '
                'initial_scene is historical; edit author_method in the author workspace. '
                'Current files and scores are in working_directory, not the author directory. '
                'The training checkpoint contains every paid observation; successful confirmation '
                'labels stay outside the predictor fitting files.'}
        content = json.dumps(current, indent=2)
        write_text_incrementally(self.workspace / 'scene_source.json', content)
        self.files['scene_source.json'] = content

    def _request(self, laboratory):
        request = super()._request(laboratory)
        if 'scene_source.json' in self.files:
            request['paid_scene'] = json.loads(self.files['scene_source.json'])
        limit = self.practice_limits
        observations = min(limit.observation_budget,
            self.config.observation_budget - self.costs['observation_cost'])
        compute = min(limit.compute_budget,
            self.config.compute_budget - self.costs['compute_cost'])
        seconds = min(limit.execution_seconds,
            self.config.execution_seconds - self.costs['python_execution_seconds'])
        required = getattr(self.prepare_trial, 'initial_observation_cost', 72) + limit.confirmation_rows
        request['practice_confirmation'] = {
            'confirmation_rows': limit.confirmation_rows,
            'confirmation_mse_threshold': limit.confirmation_mse_threshold,
            'remaining_observations': observations,
            'remaining_compute': compute,
            'remaining_execution_seconds': seconds,
            'resources_allow_practice': observations >= required and compute >= required and seconds > 0,
        }
        request['research_work'] = deepcopy(self.research_work)
        request['research_work_result'] = deepcopy(self.research_work_result)
        if self.research_work_result is not None:
            executed = request['research_work_result']
            trial = next((item for item in self.practice_trials
                          if item['directory'] == executed['directory']), None)
            executed['source_matches_current'] = (trial is not None and
                trial['method_source'] == (self.workspace / METHOD_PATH).read_text())
            executed['question_matches_current'] = (
                executed.get('question') == (self.research_work or {}).get('question'))
        request['research_root'] = deepcopy(self.research_root)
        request['active_research_question'] = self._active_work_id or self.research_root['id']
        return request

    def restore_practice(self, previous_result):
        result = super().restore_practice(previous_result)
        self.research_work = deepcopy(previous_result.get('research_work'))
        self.research_work_result = deepcopy(previous_result.get('research_work_result'))
        saved_root = previous_result.get('research_root')
        if saved_root is not None:
            self.questions = deepcopy(previous_result['online_questions'])
            self.research_root = next(q for q in self.questions if q['id'] == saved_root['id'])
            self._active_work_id = previous_result.get('active_research_question')
            if self._active_work_id == self.research_root['id']:
                self._active_work_id = None
        elif self.research_work and self.research_work.get('question'):
            # The retained pre-assignment run ended its narrower reassessment.
            # Keep that actual return; do not replay it as a fresh investigation.
            question = {**self.research_work['question'], 'parent_id': self.research_root['id']}
            opened = open_assigned_question(self, question)
            self.research_work['question'] = {key: deepcopy(opened[key]) for key in
                ('id', 'question', 'parent_id', 'depends_on', 'assignment')}
            self._active_work_id = opened['id']
            self._return_source_work((previous_result.get('source_repair') or {}).get('lesson', ''),
                                     status=previous_result['status'])
        if previous_result['status'] == 'source_repair_inconclusive':
            self.research_root.update(status='open', phase='choose_or_revise_program')
            self._active_work_id = None
        return result

    def _return_source_work(self, lesson, *, status='inconclusive'):
        """Return a local program result without replacing its parent's assignment."""
        question = next(q for q in self.questions if q['id'] == self._active_work_id)
        returned = {'question_id': question['id'], 'assignment': deepcopy(question['assignment']),
            'status': status, 'lesson': lesson,
            'selected_source': (self.research_work or {}).get('selected_source'),
            'result': deepcopy(self.research_work_result)}
        question.update(status=status, research_return=returned)
        # Only the latest compact return is inline. Earlier returns remain in
        # the existing question/event history and referenced trial results.
        self.research_root.update(status='open', phase='choose_or_revise_program',
                                  latest_child_return=deepcopy(returned))
        self._active_work_id = None
        return {'status': 'source_work_returned', 'research_question_id': question['id'],
                'parent_id': self.research_root['id'], 'research_root': deepcopy(self.research_root)}

    def _select_source_work(self, question, selected):
        """Keep authored edits and explicit procedure choices on the same work."""
        if not isinstance(question, dict) or not question.get('id'):
            raise ValueError('question needs an id and text for a new research question')
        if question['id'] == self.research_root['id']:
            # A routine action can name the parent without rewriting its assignment.
            return
        question = deepcopy(question)
        if self._active_work_id and self._active_work_id != question['id']:
            self._return_source_work('Author selected another program investigation.',
                                     status='continued_by_author')
        question['parent_id'] = self.research_root['id']
        opened = open_assigned_question(self, question)
        opened['status'] = 'open'
        self._active_work_id = opened['id']
        self.research_root['status'] = 'open'
        question = {key: deepcopy(opened[key]) for key in
                    ('id', 'question', 'parent_id', 'depends_on', 'assignment')}
        work = {'selected_source': selected, 'method': METHOD_PATH,
                'source_workspace': str(self.workspace), 'question': question}
        if work != self.research_work:
            self.research_work_result = deepcopy(opened.get('last_program_result'))
            self.research_root['phase'] = 'choose_or_revise_program'
        self.research_work = work

    def _prepare_practice(self, agent, laboratory):
        self.research_root['phase'] = 'executing_program'
        agent._open_question(deepcopy(self.research_root['assignment']))
        if self.research_work is not None:
            question = self.research_work.get('question')
            if question:
                opened = agent._open_question(deepcopy(question))
                agent._active_question_id = opened['id']
            # References and the author's assignment, not host-selected inputs.
            agent._save('research_work.json', json.dumps(self.research_work, indent=2))
            agent._prepare_program_state()

    def _record_practice(self, case, directory, source, agent, result, selected_prior):
        start = result.get('practice_start', {}).get('method_results', 0)
        current_steps = result.get('method_results', [])[start:]

        def outcomes(feedback, proposal, location):
            yield feedback, proposal, location
            for index, outcome in enumerate(feedback.get('after_observation_results', [])):
                yield from outcomes(outcome, {'action': 'run',
                    'arguments': outcome.get('arguments', {})},
                    [*location, 'after_observation_results', index])
            for index, step in enumerate(feedback.get('steps', [])):
                nested_feedback = step.get('feedback', {})
                nested_proposal = step.get('proposal') or {}
                # Procedure steps usually also occur in method_results. Use
                # that original record once, rather than counting its returned
                # copy as another fit or evidence consumption. Its live question
                # may have been updated by the time the caller made its copy.
                if any((item.get('proposal') or {}) == nested_proposal
                       and {key: value for key, value in item.get('feedback', {}).items()
                            if key != 'question'} == {key: value for key, value in
                            nested_feedback.items() if key != 'question'}
                       for item in current_steps):
                    continue
                yield from outcomes(nested_feedback, nested_proposal,
                    [*location, 'steps', index, 'feedback'])

        operations = []
        executions = ((index, feedback, proposal, location)
            for index, step in enumerate(current_steps, start)
            for feedback, proposal, location in outcomes(step.get('feedback', {}),
                step.get('proposal') or {}, ['method_results', index, 'feedback']))
        for index, feedback, proposal, location in executions:
            arguments = proposal.get('arguments', {})
            operation = {'step': index, 'action': proposal.get('action'),
                'operator': arguments.get('operator', feedback.get('operator')),
                'evidence': {'file': f'{directory}/result.json',
                             'json_path': location}}
            for key in ('status', 'error', 'returncode', 'path', 'tested_model', 'mse',
                        'retention_decision'):
                if key in feedback:
                    operation[key] = deepcopy(feedback[key])
            validation = feedback.get('validation')
            if isinstance(validation, dict):
                operation['validation'] = {key: deepcopy(validation[key]) for key in
                    ('returncode', 'mse', 'rows', 'fit_available', 'error') if key in validation}
                split = feedback.get('split') or {}
                operation['split'] = {key: split[key] for key in
                    ('train_rows', 'holdout_rows') if key in split}
                operation['split'].update({key + '_count': len(split[key]) for key in
                    ('records', 'training_records', 'validation_records') if key in split})
            # A declaration is not execution. These summaries come from the
            # actual evaluator, including validation failures nested under refit.
            execution = validation if isinstance(validation, dict) else feedback
            operation['executed_bindings'] = [{
                'model': item.get('model'),
                'inputs': {slot: {key: value.get(key) for key in
                    ('question_id', 'revision_index', 'version', 'kind') if key in value}
                    for slot, value in item.get('inputs', {}).items()},
                'actual_parameters': {key: value for key, value in
                    (item.get('actual_parameters') or {}).items()
                    if isinstance(value, (str, bool, int, float)) or value is None},
            } for item in execution.get('research_bindings', [])]
            operations.append(operation)
        result['research_work'] = deepcopy(self.research_work)
        self.research_work_result = {'directory': directory, 'status': result['status'],
            'method_path': f'{directory}/{METHOD_PATH}',
            'question': deepcopy((self.research_work or {}).get('question')),
            'operations': operations,
            'declared_binding_models': sorted(result.get('research_input_bindings', {}))}
        feedback = super()._record_practice(case, directory, source, agent, result, selected_prior)
        self.research_work_result['confirmation'] = deepcopy(feedback.get('fresh_confirmation'))
        if self._active_work_id is not None:
            question = next(q for q in self.questions if q['id'] == self._active_work_id)
            # One compact executed result per question; full evidence stays in
            # the referenced trial. A wording/source revision must not erase it.
            question['last_program_result'] = deepcopy(self.research_work_result)
        feedback['research_work_result'] = deepcopy(self.research_work_result)
        confirmed = (feedback.get('fresh_confirmation') or {}).get('passed') is True
        self.research_root['phase'] = ('awaiting_author_finish' if confirmed
                                       else 'review_program_result')
        self._paid_scene_scores = None
        return feedback

    def _latest_source_practices(self, source):
        work = getattr(self, 'research_work', None)
        return {trial['case']: trial for trial in self.practice_trials
                if trial['method_source'] == source and trial['case'] in self.cases
                and trial.get('research_work') == work}

    def _terminal_status(self):
        if self.source_repair and self.source_repair.get('status') == 'source_repair_inconclusive':
            return 'source_repair_inconclusive'
        return super()._terminal_status()

    def _skip_edit_practice(self, before, after):
        self.research_root['phase'] = 'choose_or_revise_program'
        if not self.practice_trials or self.practice_trials[-1]['status'] != 'method_stopped':
            return None

        def stop_explanations(source):
            tree = ast.parse(source)
            explanations = []
            for node in ast.walk(tree):
                value = node.value if isinstance(node, ast.Return) else None
                if (isinstance(value, ast.Dict) and len(value.keys) == 1
                        and isinstance(value.keys[0], ast.Constant) and value.keys[0].value == 'stop'
                        and isinstance(value.values[0], ast.Constant)
                        and isinstance(value.values[0].value, str)):
                    explanations.append(value.values[0].value)
                    value.values[0].value = ''
            return ast.dump(tree), explanations

        previous = self.practice_trials[-1]
        try:
            old_tree, old_messages = stop_explanations(before)
            new_tree, new_messages = stop_explanations(after)
            practiced_tree, _ = stop_explanations(previous['method_source'])
        except SyntaxError:
            return None
        # Compare with the actual stopped draft as well as this edit. Consecutive
        # explanation edits never become practice evidence for the changed source.
        if old_messages == new_messages or old_tree != new_tree or new_tree != practiced_tree:
            return None
        return {'reason': 'stop_explanation_only', 'previous_practice': previous['directory'],
            'message': 'Saved only changed literal stop explanations; automatic practice was skipped. '
                'No new execution or confirmation occurred.',
            'practice_action': {'action': 'run', 'arguments': {
                'operator': 'practice', 'case': previous['case']}}}

    def _execute(self, action, arguments, laboratory):
        if action == 'run' and arguments.get('operator') == 'procedure':
            arguments = deepcopy(arguments)
            if 'notes' in arguments:
                InquiryLLMAgent._execute(self, 'write', {'path': 'research_notes.md',
                    'code': arguments.pop('notes')}, laboratory)
            selected = arguments['path']
            source = (self.workspace / selected).read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                raise ValueError(f'selected procedure has invalid Python: {error}') from error
            if not any(isinstance(node, ast.FunctionDef) and node.name == 'select_actions'
                       for node in tree.body):
                raise ValueError('selected procedure needs a top-level select_actions function')
            question = arguments.get('research_question') or arguments.get('question')
            if question is None:
                question = {'id': f'source_program_{len(self.practice_trials)}',
                    'question': 'Execute the selected program on the paid scene and '
                        'return its measured consequences to the source-repair assignment.'}
            self._select_source_work(question, selected)
            # Use the normal canonical source identity and accounting. Avoid
            # automatic edit practice here: this action explicitly runs it once.
            InquiryLLMAgent._execute(self, 'write', {'path': METHOD_PATH, 'code': source}, laboratory)
            outcome = self._practice(arguments.get('case', next(iter(self.cases))))
            return {'operator': 'procedure', 'selected_source': selected, **outcome}
        if action == 'finish' and arguments.get('inconclusive') is True:
            if 'notes' in arguments:
                InquiryLLMAgent._execute(self, 'write', {'path': 'research_notes.md',
                    'code': arguments['notes']}, laboratory)
            scope = arguments.get('question_id') or self._active_work_id
            if scope is None:
                return {'status': 'source_repair_pending',
                    'research_root': deepcopy(self.research_root),
                    'message': 'No child is active. Choose or revise a program using its returned '
                        'evidence, or explicitly return the source_method_repair parent as inconclusive.'}
            if scope != self.research_root['id']:
                if scope != self._active_work_id:
                    raise ValueError('finish returns the active work or the source_method_repair parent')
                return self._return_source_work(arguments.get('lesson', ''))
            self.research_root.update(status='inconclusive', lesson=arguments.get('lesson', ''))
            self.source_repair = {'status': 'source_repair_inconclusive',
                'method_path': METHOD_PATH, 'lesson': arguments.get('lesson', ''),
                'latest_practice': self.practice_trials[-1]['directory'] if self.practice_trials else None,
                'published': False, 'transfer_scheduled': False}
            return deepcopy(self.source_repair)
        if action != 'finish':
            if action in {'write', 'run'} and arguments.get('research_question') is not None:
                self._select_source_work(arguments['research_question'], METHOD_PATH)
            return super()._execute(action, arguments, laboratory)
        if arguments.get('path') != METHOD_PATH:
            raise ValueError(f'finish needs path:{METHOD_PATH}')
        source = (self.workspace / METHOD_PATH).read_text()
        latest = list(self._latest_source_practices(source).values())
        if (len(latest) != len(self.cases) or any(trial['status'] != 'completed'
                or (trial.get('fresh_confirmation') or {}).get('passed') is not True
                or trial.get('research_work') != self.research_work
                for trial in latest)):
            return {'error': 'The current method has no completed new independent confirmation '
                    'on this paid scene. Inspect its actual practice and revise or continue it.',
                    'current_source_coverage': self._current_source_coverage()}
        self.model, self.lesson = METHOD_PATH, arguments.get('lesson', '')
        self.research_root.update(status='independent_confirmation_passed',
                                  phase='confirmed', lesson=self.lesson)
        self.source_repair = {'method_path': METHOD_PATH,
            'confirmed_practices': [trial['directory'] for trial in latest],
            'lesson': self.lesson, 'published': False, 'transfer_scheduled': False}
        return {'status': 'source_repair_confirmed', **deepcopy(self.source_repair)}

    def result(self, status):
        result = super().result(status)
        result.update(source_repair=deepcopy(self.source_repair), published=False,
                      research_work=deepcopy(self.research_work),
                      research_work_result=deepcopy(self.research_work_result),
                      research_root=deepcopy(self.research_root),
                      active_research_question=self._active_work_id or self.research_root['id'])
        if self.source_repair is not None:
            result.update(status=self.source_repair.get('status', 'source_repair_confirmed'),
                          model=None, method_path=None,
                          method_source=None, method_entry=None)
        return result


def paid_scene_trial_preparer(knowledge, source_checkpoint, lab_factory, *, paid_trace_references=None,
                             retained_comparison_reports=()):
    """Resume the latest real paid trial before binding the author's next draft."""
    source_checkpoint = Path(source_checkpoint).resolve()

    def prepare(case, directory, source, limits):
        directory = Path(directory)
        completed = sorted(directory.parent.glob('trial_*/result.json'),
                           key=lambda path: int(path.parent.name.split('_')[-1]))
        checkpoint = completed[-1] if completed else source_checkpoint
        previous = json.loads(checkpoint.read_text())
        prior = previous['costs']
        cumulative = replace(limits,
            observation_budget=limits.observation_budget + prior['observation_cost'],
            compute_budget=limits.compute_budget + prior['compute_cost'],
            execution_seconds=limits.execution_seconds + prior['python_execution_seconds'],
            max_model_calls=prior['model_calls'], token_budget=prior['reported_tokens'])
        agent = SingleSubmissionMethodAgent(None, directory, arm='hegel', source_task=False,
            knowledge=deepcopy(knowledge), config=cumulative,
            confirmation_seed=case['confirmation_seed'], paid_trace_references=paid_trace_references)
        lab = lab_factory(case)
        restore = agent.restore_practice_research if completed else agent.restore_research
        restore(previous, checkpoint.parent, lab, confirmation_seed=case['confirmation_seed'],
                retained_comparison_reports=retained_comparison_reports)
        # A confirmed draft may be compared with a later edit. Keep its evidence
        # and retained predictor, while the newly bound method must submit anew.
        agent.model = None
        agent.bind_method(source)
        agent._prepare_program_state()
        return {'agent': agent, 'laboratory': lab, 'inherited_candidates': [],
            'prior_research_cost': deepcopy(prior),
            'prior_method_count': len(previous['method_results']),
            'prior_event_count': len(previous['events']), 'resume_origin': str(checkpoint)}

    prepare.initial_observation_cost = 0
    return prepare
