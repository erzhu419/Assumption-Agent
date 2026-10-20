"""Public scientific-world tasks using the existing Code provider and executor.

Both conditions have the same tools and persistent task-local files. Questions
carry live child answers into authored experiment programs; neither condition
receives a scenario solver or the evaluation scorecard.
"""

from copy import deepcopy
import ast
import json
from math import isfinite
import sys
from time import monotonic

from .online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig, _PLAN_RUNNER
from .proposers import _run
from .temporal_paired_method import execute_procedure
from .world_llm_inquiry import INQUIRY_INTERFACE, execute_llm_inquiry
from .world_procedure_repair import run_world_procedure
from .world_reference_hypothesis import HYPOTHESIS_INTERFACE, execute_reference_hypothesis
from .world_reference_expansion import execute_reference_expansion
from .world_reference_response import (execute_reference_response,
                                       response_premise_design_action, dispatch_response_premise)


INTERFACE = """You are conducting scientific research inside DiscoveryWorld.
Solve the task in current_observation.taskProgress by experimenting in the world.
Only public observations and capabilities are available. Object IDs and action
names come from these observations. Read instruments, change conditions, compare
outcomes, and implement the task's requested intervention when you have evidence.

For a numerical relationship, run {operator:'research',path:'binding.py',
question:{id:'relation',question:'...',parent_id:'task',depends_on:[]}} connects
your public observations to the existing hypothesis search and experiment selector.
You may include code:'...' to write the binding and start research in one call.
Define bind(public_spec,observations,memory), returning:
{rows:[{features:{x:2},target:7,evidence:[record_indices]}],
 probes:[{id:'test',features:{x:3},actions:[official_actions]}],
 tolerance:1e-6, parent_path:'parent.py'}.
Extract rows from actual observations; features are your chosen measured inputs,
target is the measured quantity to explain. Each row cites its supporting records.
Return all available rows on every invocation and rebuild probes for the current
world state. An optional hypotheses:[Program_AST,...] supplies your own candidates;
otherwise existing refit/composition/relation search proposes them. To add new
programs to the same question, return candidate_additions:[Program_AST,...]
(or [{program:Program_AST}]); old candidates and their actual evidence remain.
Additions are screened on all bound rows and need their own later confirmation.
Repeated programs do not reset their evidence or count as additional votes.
Changing hypotheses alone does not replace this question's frozen candidates.
AST examples:
{op:'feature',name:'x'}, {op:'const',value:2},
{op:'mul',children:[{op:'feature',name:'x'},{op:'feature',name:'x'}]}.
Distinguish a chosen control value, a successful API action, and an observed
physical outcome. Reading back your chosen setting does not measure the correct
setting. When an intervention reveals acceptance/activation rather than a numeric
answer, bind its publicly observed outcome as target 0 or 1, with the applied
control and measured object properties as features. Existing search and probe
selection also support Boolean relations. An absent outcome is missing evidence,
not a negative label. Alternatively test your proposed numerical law by applying
its predicted setting through query, then use an authored after_observation
answer to judge the actual public outcome. Do not validate a predicted setting
against its own read-back. Preserve the candidate before its test; let a failed
intervention refute it or motivate a specific revised condition.
Rows may include conditioning_refs:[old_measurement_indices] and
outcome_refs:[response_indices]. Both must cite real records and together cover
evidence; outcome_refs must be nonempty. This permits a new response conditional
on previously measured properties. Only outcomes observed after the candidate's
selection can confirm it; rebinding an already public outcome is not new evidence.
Keep all citations so a later correction can revisit the extraction. Rows without
these roles still require the whole supporting evidence to be independent.
The tool compares candidates, chooses a probe by disagreement, performs its exact
actions, then re-runs your binding against new records to test FROZEN candidates.
Training fit alone is not confirmation. tolerance is the declared MSE criterion.
When a child is supported, optional parent_path executes your select_actions
program on the live child answer (answer.program is a Program AST). All numerical
features needed by candidates must be specified in each proposed probe. Only
offer actions whose resulting measured inputs match those features. If new
measurements are needed to form rows, collect them first. If no informative probe
exists, research returns that diagnosis for you to change the measurement or
representation. You choose the scientific variables and actions; no task formula
is supplied. run {operator:'research_update',path:'binding.py',question:{id:'q'}}
performs comparison only; research uses it after acquisition. Both arms have
these tools, with ordinary world/Python costs and a default 12-step procedure.

For competing explanations without a numerical binding, run
{operator:'contrast',path:'test.py',code:'Python source',question:{id:'q',
question:'Which explanation accounts for the outcome?',parent_id:'task',depends_on:[]}}.
Define design(public_spec,observations,memory), returning
{hypotheses:[{id:'a',claim:'...'}, {id:'b',claim:'...'}],
 experiments:[{id:'test',outcome:'A specific observable physical result',
 actions:[official_actions],predictions:{a:true,b:false}}]}.
Each experiment describes ONE observable Boolean outcome; all hypotheses predict
that same outcome under those same actions and conditions. Use Python True/False.
The existing probe chooser selects an experiment with differing predictions, then
executes its original actions and calls answer in the SAME file. Zero disagreement
returns observational_ambiguity without buying observations or granting support.
For controls requiring intermediate feedback, replace actions with
procedure:{path:'test.py',max_steps:12}. Define select_actions in that file using
the existing procedure contract: inspect live controls and each actual response
before choosing the next step. The selected experiment and predictions remain
fixed throughout the procedure; read them from this question's experiment_choice.
Its started_before_observation marks this invocation, so inherited trials are not
mistaken for new phases. Return {stop:'reason'} when finished or unable to continue;
the host then runs answer, which must judge whether the intended conditions and
all required phases actually occurred. A stop alone grants no support. Execution
errors or step limits leave records for analysis and revision. Ordinary steps
belong to the contrast question; an explicitly declared child retains its own ID.
The procedure may share the design file or use a previously authored program.
Define answer(public_spec,observations,memory) under the ordinary evaluate contract.
memory.questions contains this question's experiment_choice, frozen before its test;
use its selected_experiment and hypotheses when interpreting ACTUAL records.
memory.experiment_trials retains earlier contrast invocations, their frozen
predictions and exact observation_indices, including failed repeats. A later
invocation's start does not invalidate an earlier completed trial.
The predictions are forecasts, not measurements. Judge the question on observed
conditions and physical results; unsupported actions or missing outcomes may leave
it inconclusive. Cite actual observation indices. Rebuild the design from current
evidence on later calls. Both arms can use this operator.
Saving a design with write and its question returns a contrast_action. Saved designs
also appear in method_decision.questions[].contrast_actions. Choose that run action
to compare its predictions and execute the selected experiment. Saving alone does
not select or run a trial. Inspect and revise an old design for the current state
before invoking it; its original conditions may no longer hold.
An existing separately repaired analysis remains in use. To choose another answer
implementation, explicitly include question.analysis_path in the contrast call.

evidence.reference_history projects earlier interactions and public object states,
including direct containment, from the paid observations. It retains first/latest
records per distinct state, not every measurement; full records remain available.
evidence.measurement_history retains all paid USE/READ records, including failed
and inconclusive interactions. Compare previously measured quantities when a
response hypothesis fails; the latest notebook may omit useful earlier variables.
You choose which readings measure which variables and which outcomes they explain.
Select references by their observed scientific outcome, not API success alone.
If a failed prediction could reflect either a wrong law or an unmet condition,
use a known working configuration to test that condition and the physical readout.
Where possible, observe a successful baseline, change one controlled condition,
then restore it without the allegedly necessary extra operation. Freeze the
competing predictions before testing and interpret the actual intermediate states.
Success on another object shows the outcome is observable there; justify matched
conditions before transferring that conclusion to the failed object. Your authored
design/answer and the existing query/contrast tools perform this comparison.

query {actions:[{action:'ACTION_NAME',arg1:...,arg2:...},...]} executes official
environment actions in order. Inspect feedback before choosing uncertain next
actions. When in dialog use {chosen_dialog_option_int:N}, with a listed option.
query {path:'shared/experiment.py', question:{id:'q',question:'...',parent_id:null,
depends_on:[]}} executes an authored select_actions(public_spec,observations,memory)
function, which returns an action list or {actions:[...]}. public_spec contains
current_observation and capabilities. observations contains all actual previous
action records. memory contains the CURRENT questions, answers and methods.
The optional question is opened before its actions, in both query forms.
parent_id links an inquiry to the research it informs: supported, refuted and
inconclusive answers all return control for a new decision. Only explicit depends_on
and method requires demand supported premises. Refuting an assumed prerequisite
can therefore change the parent experiment without requiring that assumption true.
The program may itself return question and after_observation alongside actions.
after_observation:[{action:'evaluate',arguments:{path:'analysis.py',question_id:'q'}}]
runs authored analysis immediately after successful acquisition. These steps may
write, run or evaluate; an explicit query argument overrides the program's list.
actions:[] with a nonempty after_observation list analyzes already acquired data.
A failed acquisition retains its records and skips the follow-up analysis.

Programs can also return {action:'write'|'run'|'query'|'evaluate'|'finish',arguments:{...}}
or {stop:'reason'}. run {operator:'procedure',path:'shared/research.py',max_steps:12}
uses the existing procedure executor to repeatedly call your select_actions,
execute its declared action, and return live feedback before its next step.
It stops on completion, error, explicit stop or the step limit. memory.method_results
and research_state.json contain feedback from ALL programs, including inherited runs.
Each result carries its path; filter by your program's path for its own history.
Inside a program, memory.program_path is its current path and memory.program_results
contains only that path's results, including its earlier executions. Branch on their
actual outcomes and the live state; a call count alone does not mean an action succeeded.
The total method_results length is not this program's step count. Child analyses update
memory.answers before the next parent step. You author the questions, experiments,
analysis and branching; the executor supplies no scenario solution. Both arms can
use this interface. A procedure may be revised and continued on the same evidence.

write {path:'file.py',code:'Python source'} saves a file. run {code:'Python source'}
or run {path:'file.py'} executes ordinary Python. You may save notes, inspect data,
fit models and author reusable experiment programs. numpy/scipy are available.
run feedback gives stdout_file/stderr_file for complete output and reports any
preview truncation. Filter or summarize those files instead of repeating a scan
whose beginning was omitted from its preview.
All action records are in laboratory_observations.json, indexed from zero.
Each full record has {index,action,feedback,observation}; world messages are at
record['observation']['lastActionMessage'] and optional 'extended_action_message'.
action_observations.json contains the same indexed actions and feedback with these
messages at the TOP LEVEL, and public_objects for action participants and their
direct public containment relations (or accessible objects for dialog steps).
These are the actual contemporaneous names/descriptions, not inferred outcome
labels. An observed negative physical state differs from an absent outcome.
Program observations include this same top-level action view. Their nested
record.observation retains the original environment fields and adds the full
public_objects/public_relations projection, as current_observation does. The
laboratory_observations.json records retain the raw environment fields unchanged.
public_relations lists explicit {subject,relation,object,text}
clauses: '(in crystal reactor (uncalibrated) [uuid: 8])' on object 9 becomes
{subject:9,relation:'in',object:8,text:'(in crystal reactor (uncalibrated) [uuid: 8])'}.
Match relation UUIDs; names can include qualifiers. The text remains an observed
description for your analysis, not a host-assigned scientific outcome.
Start with this
compact file for analysis; use the full records for historical object/world state.
current_observation.json and research_state.json are refreshed after every action.
current_observation (in public_spec and current_observation.json) includes the same
public_objects/public_relations projection of currently accessible objects alongside
the original environment fields. It is a live view; use indexed action records for
past states. Object names and descriptions remain observations for your analysis.

run {operator:'question',id:'q',question:'...',parent_id:null,depends_on:[]} opens
or revises a research question before obtaining evidence. To return a child
answer, write an analysis file defining answer(public_spec,observations,memory),
then evaluate {path:'analysis.py',question_id:'q'}. The function must return
{status:'supported'|'refuted'|'inconclusive',answer:...,evidence:[record_indices]}.
evaluate may include code:'Python source' to save and execute a revised analysis
on the same paid records. This does not rerun the experiment. Use a separate
analysis path to preserve a working experiment's design and select_actions.
This records your analysis, not an oracle's judgment. Its result enters live
memory. A changed child answer reopens dependent questions and pauses methods
that require their support. Parent experiment programs can read these answers.
run {operator:'method',id:'m',path:'shared/experiment.py',requires:['q'],
applicability:'...'} registers an authored method. query {method_id:'m'} executes
it when its requirements are supported. You may inspect/revise all these files.

Register max_steps:12 for a full feedback procedure; its listed execute_action
uses run {operator:'procedure',method_id:'m'}. Optional parameters are defaults,
overridden by invocation parameters. A question on that call associates its new
observations with your inquiry. Inside select_actions, memory.program_arguments
contains parameters and procedure_start (the first observation index of this
invocation); these stay fixed across the procedure's steps. A contrast's selected
procedure may also provide parameters under the same contract.
method_decision exposes available methods and their prior_execution records to
both arms. provenance identifies who supplied a program; an evidence_file may
contain a separate public practice archive. Its indices belong to that archive,
not this episode's observations. Reusing an already tested method is an execution
or replication, not a new discriminating discovery. Select its use and scope,
then interpret fresh local observations in your own answer before changing the
parent explanation. Supplied control programs contain no scientific verdict.

finish {lesson:'Your scientific conclusion and evidence'} ends the episode.
The environment determines task success; no special manifest is required.
Limits are generous ceilings, not targets. Stop after solving the actual task.
Keep concise notes in files; recent context is a preview, full records persist.
Each decision is a fresh model session. Persist your working explanation, fitted
relations, evidence indices, unresolved alternatives and next experiment in
research_notes.md; its current contents are shown every turn. Return the complete
current notebook in the required top-level research_notes field alongside your
chosen action; it is saved without a separate call. Keep notebook text only in
that top-level field, rather than duplicating it in the action arguments.
write or run may also edit research_notes.md. Save useful computed results there
instead of repeatedly printing the records. Readouts alone do not settle a claim:
use the task's intervention and observed response to test your explanation.
For a planned question analysis you may include analysis_path:'analysis.py' in
the question. Once authored, evaluate executes it on the actual observations.
method_decision, when present, organizes your declared dependencies and available
programs; its support labels are your analyses, not environment confirmations.
Each question's answer_observations shows the actual records its answer cites,
including older evidence outside the recent window. Compare that evidence with
the recorded conclusion before reusing it; these are paid records, not new tests.
Notes and saved source do not change a recorded answer. If you discover an error
in an earlier analysis, evaluate the corrected answer on its actual paid records
before relying on that question again. You may reuse an existing analysis or send
evaluate {path:'analysis.py',code:'corrected Python source',question_id:'q'}.
When that path contains select_actions or design, evaluate saves the new analysis to
shared/analysis_<question_id>.py, retaining the original experiment program.
Use write or contrast with code when you intend to revise the experiment itself.
The revised answer then updates dependent research through the existing return path.
If research_root is supplied, return research_question for the question this
action advances (id, question, parent_id, depends_on), or null for routine actions.
Use stable IDs. The public task is the supplied root; name a child only for an
unresolved condition needed by its parent. Registration is automatic before the
action and supplies no answer. Return answers through executable evaluate.
Use only this episode's public observations and authored files. Do not read
benchmark source, private scorecards, other episodes or repository artifacts.
"""

STRATEGIES = {
    "code_archive": "Act as a capable ordinary scientific Code/ReAct agent. Use Python, "
                    "notes and reusable programs freely to solve the task. Choose your own research organization.",
    "hegel": "Organize research around revisable explanations of the question you are resolving. "
             "Retain their applicability conditions and implications across experiments in research_notes. "
             "Use paid evidence to revise those explanations within the scope actually tested. "
             "A prediction that one finite experiment succeeds or fails answers that experiment; "
             "also consider what it establishes about the parent explanation. For a further experiment, "
             "compare what the surviving explanations predict under its conditions, using the existing "
             "research or contrast tools when applicable. Choose among executable experiments on that basis. "
             "If current observations cannot distinguish the explanations, state what remains unobservable "
             "and investigate a measurement or representation that could distinguish them. If you cannot "
             "yet justify competing explanations or distinct predictions, record that limitation and "
             "choose exploratory work; do not invent alternatives merely to fill a design. When a parent's "
             "experiment depends on an uncertain instrument, representation or condition, open "
             "that child question, acquire evidence, and return an executable analysis to the "
             "parent. Let the answer change the next experiment. Retain useful procedures with "
             "their applicability conditions; reopen unsupported dependencies when evidence changes. "
             "For a measured numerical relation, use the research operator to bind public readings "
             "and executable experiments to hypothesis search; consume its returned child answer "
             "in the parent program. Revise the binding when it reports ambiguity or failed validation. "
             "For exploration without such a relation, ordinary world actions remain available.",
}

_ANSWER_RUNNER = _PLAN_RUNNER.replace("code['select_actions']", "code['answer']")
_BIND_RUNNER = _PLAN_RUNNER.replace("code['select_actions']", "code['bind']")
_DESIGN_RUNNER = _PLAN_RUNNER.replace("code['select_actions']", "code['design']")
_RESEARCH_DRIVER_PATH = "shared/world_research_driver.py"
_RESEARCH_DRIVER = '''def select_actions(spec, observations, memory):
    advice = memory.get('research_advice') or {}
    return advice.get('next_action') or {'stop': advice.get('diagnosis', 'Research needs a revised binding')}
'''


class DiscoveryWorldAgent(InquiryLLMAgent):
    def __init__(self, provider, workspace, *, arm, limits):
        if arm not in STRATEGIES:
            raise ValueError(f"Unknown DiscoveryWorld arm: {arm}")
        super().__init__(provider, workspace, arm="hegel" if arm == "hegel" else "code_react",
                         config=InquiryLLMConfig(execution_seconds=limits["python_seconds_per_task"]))
        self.agent_arm, self.limits = arm, deepcopy(limits)
        self.methods, self.score_checkpoints = [], []
        self.program_executions = self.child_answers_returned = 0
        self.method_results = []
        self.relation_research = {}
        self.research_advice = None
        self.research_focus = None
        self._prerequisite_checked_through = -1
        self._python_runs = 0
        self._current_proposal = None
        self._procedure_confirmation_allowed = True
        self._acquisition_review = None
        self.llm_inquiries, self._inquiry_stack = {}, []
        self._on_checkpoint = None
        self.started = monotonic()
        self.status, self.lab = "running", None
        self.costs.update(environment_steps=0, input_tokens=0, output_tokens=0,
                          cached_input_tokens=0, usage_missing_calls=0, wall_time_seconds=0.,
                          research_compute_units=0)

    def _write_evidence(self):
        # Override the inherited numerical-row writer, including during init.
        # This environment has action records, not an empty fitting dataset.
        (self.workspace / "laboratory_observations.json").write_text(
            json.dumps(self.observations, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _action_observation(record):
        from .world_prerequisite_research import public_object_context, public_object_relations

        observation = record["observation"]
        result = {key: deepcopy(record[key]) for key in ("index", "action", "feedback")}
        result["lastActionMessage"] = observation.get("lastActionMessage", "")
        targets = [value for key, value in record["action"].items()
                   if key in {"arg1", "arg2"} and type(value) is int]
        result["public_objects"] = public_object_context(observation, targets or None)
        result["public_relations"] = public_object_relations(result["public_objects"])
        extended = observation.get("extended_action_message", "")
        if extended and extended != result["lastActionMessage"]:
            result["extended_action_message"] = extended
        if observation.get("dialog_box", {}).get("is_in_dialog"):
            result["dialog_box"] = deepcopy(observation["dialog_box"])
        return result

    @staticmethod
    def _public_observation(observation):
        from .world_prerequisite_research import public_object_context, public_object_relations

        observation = deepcopy(observation)
        observation["public_objects"] = public_object_context(observation)
        observation["public_relations"] = public_object_relations(observation["public_objects"])
        return observation

    def _current_observation(self):
        return self._public_observation(self.lab.public_observation())

    def _program_observation(self, record):
        return {**record, **self._action_observation(record),
                "observation": self._public_observation(record["observation"])}

    def _python(self, command, payload="", *, timeout_seconds=None):
        remaining = min(self.limits["python_seconds_per_call"],
                        self.limits["python_seconds_per_task"] - self.costs["python_execution_seconds"],
                        self.limits["task_wall_seconds"] - (monotonic() - self.started))
        if remaining <= 0:
            raise ValueError("Python or episode wall-time budget exhausted")
        if timeout_seconds is not None:
            timeout_seconds = float(timeout_seconds)
            if not isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive and finite")
            remaining = min(remaining, timeout_seconds)
        started = monotonic()
        try:
            return _run(command, payload, remaining, str(self.workspace))
        finally:
            self.costs["python_execution_seconds"] += monotonic() - started

    @staticmethod
    def _execution_nodes(feedback, step_path=()):
        """Walk actual feedback containers, excluding unexecuted proposals."""
        yield feedback, list(step_path)
        for key in ("procedure", "experiment"):
            if isinstance(feedback.get(key), dict):
                yield from DiscoveryWorldAgent._execution_nodes(feedback[key], (*step_path, key))
        for index, step in enumerate(feedback.get("steps", [])):
            yield from DiscoveryWorldAgent._execution_nodes(
                step["feedback"], (*step_path, "steps", index, "feedback"))
        for index, result in enumerate(feedback.get("after_observation_results", [])):
            yield from DiscoveryWorldAgent._execution_nodes(
                result, (*step_path, "after_observation_results", index))

    def _experiment_trials(self):
        """Project completed event records; never replace a trial with its retry."""
        trials = []

        for event_index, event in enumerate(self.events):
            for feedback, step_path in self._execution_nodes(event.get("feedback", {})):
                if feedback.get("operator") != "contrast" or not feedback.get("selected_experiment"):
                    continue
                execution = feedback.get("procedure") or feedback.get("experiment", {})
                trials.append({"event_index": event_index, "step_path": step_path,
                    "question_id": feedback["question_id"], "path": feedback["path"],
                    "hypotheses": deepcopy(feedback["hypotheses"]),
                    "selected_experiment": deepcopy(feedback["selected_experiment"]),
                    "observation_indices": sorted({index for node, _ in self._execution_nodes(execution)
                        for index in node.get("observation_indices", [])}),
                    "execution_status": execution.get("status", "acquisition_error" if
                        execution.get("error") else "recorded")})
        return trials

    def _state(self):
        return {"questions": deepcopy(self.questions), "methods": deepcopy(self.methods),
                "experiment_trials": self._experiment_trials(),
                "answers": {q["id"]: {k: deepcopy(q[k]) for k in ("answer", "status", "evidence")
                                     if k in q} for q in self.questions},
                "method_results": deepcopy(self.method_results),
                "relation_research": deepcopy(self.relation_research),
                "research_advice": deepcopy(self.research_advice),
                "research_focus": deepcopy(self.research_focus),
                "llm_inquiries": {key: {name: deepcopy(value) for name, value in session.items()
                                        if name != "events"}
                                  for key, session in self.llm_inquiries.items()},
                "prerequisite_checked_through": self._prerequisite_checked_through,
                "model_available": self.provider is not None,
                "limits": self.limits, "costs": self.costs}

    def _save_state(self):
        self._write_evidence()
        for name, data in (("research_state.json", self._state()),
                           ("llm_inquiries.json", self.llm_inquiries),
                           ("current_observation.json", self._current_observation()),
                           ("action_observations.json", [self._action_observation(r) for r in self.observations])):
            (self.workspace / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self._prepare_program_state()

    def _latest_program_result(self, path):
        previous = next((row for row in reversed(self.method_results) if row["path"] == path), None)
        if previous is None:
            return None
        previous = deepcopy(previous)
        # The final function call can succeed while the enclosing procedure
        # exhausts its steps. Reuse the recorded invocation, including frozen
        # prefixes, rather than losing that result behind its last successful action.
        event = next((e for e in reversed(self.events) if e.get("arguments", {}).get("path") == path), {})
        feedback = event.get("feedback", {})
        procedure = feedback.get("procedure") or (
            feedback if event.get("arguments", {}).get("operator") == "procedure" else {})
        steps = procedure.get("steps", [])
        if steps and steps[-1]["proposal"] == previous["proposal"] and steps[-1]["feedback"] == previous["feedback"]:
            previous["procedure_status"] = procedure["status"]
            previous["procedure_steps"] = [{"proposal": step["proposal"],
                **{key: value for key, value in step["feedback"].items()
                   if key in {"observation_indices", "error", "after_observation_error", "status", "reason"}}}
                for step in steps]
        return previous

    def _program_functions(self, path):
        """Inspect a saved draft's callable interface without executing it."""
        source = self.workspace / path
        if not source.is_file():
            return set()
        try:
            return {node.name for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef)}
        except SyntaxError:
            return set()

    def _design_action(self, path, question):
        """Expose an authored design without running it or changing its support."""
        if not {"design", "answer"} <= self._program_functions(path):
            return None
        return {"action": "run", "arguments": {
            "operator": "contrast", "path": path, "question": {"id": question["id"]}}}

    def _question_design_actions(self):
        """Find the same authored comparison programs for display and return."""
        design_paths = {q["id"]: [q["design_path"]] if q.get("design_path") else []
                        for q in self.questions}
        # Reuse actual write events, including inherited drafts that have never
        # reached contrast. Registration is a view, not a trial or an answer.
        for event in self.events:
            feedback = event.get("feedback", {})
            writes = [(event, feedback), *((node.get("program_action", node), node)
                      for node, _ in self._execution_nodes(feedback))]
            for action, result in writes:
                arguments = action.get("arguments", {})
                question = arguments.get("question") or arguments.get("research_question") or {}
                path = arguments.get("path")
                if (action.get("action") == "write" and path and result.get("written") == path
                        and question.get("id") in design_paths):
                    paths = design_paths[question["id"]]
                    if path not in paths:
                        paths.append(path)
        return {q["id"]: [action for path in design_paths[q["id"]]
                          if (action := self._design_action(path, q))] for q in self.questions}

    def _prepare_program_state(self):
        """Adapt the historical method-decision view to live world questions.

        Like PythonArchiveMethodLearner, expose existing support next to real
        executable choices. Scientific questions and answers remain authored.
        """
        by_id = {q["id"]: q for q in self.questions}
        design_actions = self._question_design_actions()
        questions = []
        trials = self._experiment_trials()
        for question in self.questions:
            entry = deepcopy(question)
            entry["answer_observations"] = [self._action_observation(self.observations[index])
                                             for index in question["evidence"]]
            entry["contrast_actions"] = design_actions[question["id"]]
            entry["experiment_trials"] = [trial for trial in trials if trial["question_id"] == question["id"]]
            entry["pending_dependencies"] = self._pending_dependencies(question)
            entry["unreviewed_observation_indices"] = [index for index in question["acquired_observations"]
                if index > question.get("reviewed_through_observation", -1)]
            path = question.get("analysis_path")
            if path and (self.workspace / path).is_file():
                entry["evaluate_action"] = {"action": "evaluate", "arguments": {
                    "path": path, "question_id": question["id"]}}
            path = question.get("parent_program_path")
            if path and (self.workspace / path).is_file():
                entry["procedure_action"] = {"action": "run", "arguments": {
                    "operator": "procedure", "path": path}}
                previous = self._latest_program_result(path)
                if previous:
                    entry["last_program_result"] = deepcopy(previous)
            questions.append(entry)
        methods = []
        for method in self.methods:
            entry = deepcopy(method)
            entry["blocked_by"] = [key for key in method["requires"]
                if by_id.get(key, {}).get("status") != "supported"]
            entry["execute_action"] = ({"action": "run", "arguments": {
                "operator": "procedure", "method_id": method["id"]}} if "max_steps" in method else
                {"action": "query", "arguments": {"method_id": method["id"]}})
            if method.get("evidence_file"):
                entry["prior_execution"] = json.loads((self.workspace / method["evidence_file"]).read_text())
            methods.append(entry)
        proposals = []
        for key, session in self.llm_inquiries.items():
            if "proposal" not in session:
                continue
            proposal = {"session_id": key, "question_id": session["question_id"],
                        "notes": session["notes"], **deepcopy(session["proposal"])}
            comparison = proposal.get("comparison_result")
            if comparison:
                proposal["premise_design_action"] = response_premise_design_action(comparison)
            proposals.append(proposal)
        summary = {"questions": questions, "methods": methods,
                   "explanation_proposals": proposals,
                   "research_advice": deepcopy(self.research_advice),
                   "answer_semantics": "Agent-authored support on cited evidence; new evidence requires review."}
        self.files["method_decision.json"] = json.dumps(summary, ensure_ascii=False)
        (self.workspace / "method_decision.json").write_text(self.files["method_decision.json"], encoding="utf-8")

    def _request(self):
        from .world_prerequisite_research import public_reference_history

        self._prepare_research_focus()
        self._prepare_program_state()
        notes_path = self.workspace / "research_notes.md"
        notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
        # run may edit an already registered file; expose the executed contents.
        for name in self.files:
            path = self.workspace / name
            if path.is_file():
                self.files[name] = path.read_text(encoding="utf-8")
        root = next((question for question in self.questions if question['id'] == 'task'), None)
        return {**({"research_root": {"id": "task", "question": root["question"]}}
                   if self.agent_arm == "hegel" and root else {}),
                "task_interface": INTERFACE + INQUIRY_INTERFACE + HYPOTHESIS_INTERFACE + "\n" + STRATEGIES[self.agent_arm],
                "research_notes": notes, **self._spec(),
                **({"research_focus": deepcopy(self.research_focus)} if self.research_focus else {}),
                "evidence": {"observations_file": "laboratory_observations.json",
                             "observation_count": len(self.observations),
                             "compact_observations_file": "action_observations.json",
                             "record_fields": ["index", "action", "feedback", "observation"],
                             "message_path": ["observation", "lastActionMessage"],
                             "compact_message_path": ["lastActionMessage"],
                             "reference_history": public_reference_history(self.observations),
                             "measurement_history": [self._action_observation(r) for r in self.observations
                                 if r["action"].get("action") in {"USE", "READ"}],
                             "recent_action_results": [self._action_observation(r) for r in self.observations[-12:]]},
                "limits": self.limits, "costs": deepcopy(self.costs),
                "research": self._state(), "files": deepcopy(self.files),
                "events": [{k: deepcopy(v) for k, v in e.items() if k != "usage"}
                           for e in self.events[-6:]]}

    def _child_answer_snapshots(self, parent):
        return {q["id"]: {key: deepcopy(q[key]) for key in ("status", "answer", "evidence")}
                for q in self.questions if q.get("parent_id") == parent["id"]
                and q["status"] in {"supported", "refuted", "inconclusive"}
                and "answer" in q and "reviewed_through_observation" in q}

    def _prepare_research_focus(self):
        if self._inquiry_stack and self._inquiry_stack[-1].get("stage") == "proposal":
            self.research_focus = None
            return
        if self.agent_arm != "hegel":
            return
        from .world_prerequisite_research import next_prerequisite, operation_failures

        local_root = self._inquiry_stack[-1]["question_id"] if self._inquiry_stack else None
        questions = self.questions
        if local_root:
            # Schedule this inquiry and its descendants while keeping all paid
            # evidence and external dependencies available to the researcher.
            included = {local_root}
            while True:
                descendants = {q["id"] for q in questions if q.get("parent_id") in included}
                if descendants <= included:
                    break
                included.update(descendants)
            questions = [q for q in questions if q["id"] in included]
            if self.research_focus and self.research_focus["question"]["id"] not in included:
                self.research_focus = None

        if self.research_focus:
            question = next(q for q in self.questions if q["id"] == self.research_focus["question"]["id"])
            if self.research_focus.get("kind") == "parent_research":
                # Rebuild from unconsumed answers and execution results below.
                # A child may have returned after the old program was reviewed.
                pass
            elif self.research_focus.get("kind") == "analysis":
                if any(index > question.get("reviewed_through_observation", -1)
                       for index in question["acquired_observations"]):
                    self._save_state()
                    return
            elif self.research_focus.get("kind") == "intervention":
                if question.get("reviewed_through_observation", -1) < max(self.research_focus["evidence"]):
                    self._save_state()
                    return
            elif question["status"] != "supported":
                self.research_focus["failures"] = operation_failures(self.observations).get(
                    self.research_focus["target"], [])
                self._save_state()
                return
            self.research_focus = None

        # The model already chose this child and bought its evidence. Complete
        # the existing analysis contract before returning to its parent, without
        # supplying an answer or buying the same observations again.
        for question in reversed(questions):
            if not question.get("parent_id") or question.get("binding_path") or self._pending_dependencies(question):
                continue
            pending = [index for index in question["acquired_observations"]
                       if index > question.get("reviewed_through_observation", -1)]
            if not pending:
                continue
            analysis_path = question.get("analysis_path")
            if analysis_path in (question.get("experiment_path"), question.get("design_path")):
                # An answer-only repair must leave the executable experiment
                # available for continuation and revision.
                analysis_path = None
            self.research_focus = {"kind": "analysis", "question": deepcopy(question),
                "path": analysis_path or f"shared/analysis_{question['id']}.py",
                "evidence": pending, "origin": "host_analysis_continuation",
                "purpose": "Analyze the evidence acquired for your declared child before continuing its parent. "
                           "Write answer using actual outcomes. Inconclusive or refuted is a valid result; "
                           "this continuation does not supply the interpretation or repeat the experiment."}
            self._save_state()
            return
        # This scanner opens task-level operation questions. A local inquiry
        # must neither select those siblings nor consume the outer scan cursor.
        focus = None
        if local_root is None:
            focus = next_prerequisite(self.observations, self.questions, self.lab.public_observation(),
                                      after_index=self._prerequisite_checked_through)
            self._prerequisite_checked_through = len(self.observations) - 1
        if focus:
            question = self._question(focus["question"])
            focus.setdefault("origin", "host_execution_failure")
            question.update(origin=focus["origin"], target=focus["target"])
            if focus.get("kind") == "intervention" and "reviewed_through_observation" in question:
                question["status"] = "reopened"
                self._update_dependencies(question["id"])
            focus["question"] = deepcopy(question)
            focus.setdefault("kind", "operation")
            focus["path"] = f"shared/{question['id']}.py"
            if focus["kind"] == "intervention":
                focus["parent_question"] = deepcopy(next(
                    (q for q in self.questions if q["id"] == question["parent_id"]), {}))
                focus["purpose"] = ("Preserve the supported interface answer and test the parent intervention "
                    "or its application conditions against actual public outcomes. Reuse paid evidence where "
                    "sufficient. You supply the hypothesis, predicted outcome and analysis; an unsupported "
                    "command alone does not refute a scientific relation. Any valid verdict returns control.")
            else:
                focus["purpose"] = ("Resolve this operational prerequisite before continuing the parent. "
                    "Propose and execute a different public interaction, then interpret its actual feedback. "
                    "An unsupported action is different from an inaccessible object. The trigger is host-selected; "
                    "the operational hypothesis, test program and interpretation are yours.")
            self.research_focus = focus
        if not self.research_focus:
            for parent in questions:
                bound_local_root = parent["id"] == local_root and parent.get("parent_program_path")
                if ((parent["status"] != "reopened" and not bound_local_root)
                        or parent.get("binding_path") or self._pending_dependencies(parent)
                        or (not self.observations and not bound_local_root)):
                    continue
                children = [q for q in self.questions if q.get("parent_id") == parent["id"]
                            and "answer" in q and "reviewed_through_observation" in q]
                if not children and not bound_local_root:
                    continue
                path = parent.get("parent_program_path")
                snapshots = self._child_answer_snapshots(parent)
                changed_children = [key for key, value in snapshots.items()
                                    if parent.get("reviewed_child_answers", {}).get(key) != value]
                previous = self._latest_program_result(path) if path else None
                review_program = False
                if previous and previous["step"] > parent.get("reviewed_program_step", -1):
                    feedback = previous["feedback"]
                    review_program = (feedback.get("status") == "method_stopped" or feedback.get("error")
                            or feedback.get("after_observation_error")
                            or previous.get("procedure_status") == "method_step_limit")
                if not changed_children and not review_program:
                    continue
                self.research_focus = {"kind": "parent_research", "question": deepcopy(parent),
                    "path": path or f"shared/research_{parent['id']}.py",
                    "origin": "host_parent_research_continuation", "returned_children": deepcopy(children),
                    "changed_child_ids": changed_children,
                    "purpose": "Continue the parent investigation using its returned child answers and paid "
                               "evidence. Turn competing explanations into an executable discriminating "
                               "experiment and analysis using the existing procedure interface. "
                               "You choose hypotheses, predictions and actions; no scientific answer is supplied."}
                if path:
                    self.research_focus["stage"] = "child_return"
                if review_program:
                    self.research_focus.update(stage="review", origin="host_parent_program_review",
                        previous_execution=deepcopy(previous),
                        purpose="Review this existing program's actual stop or failure against the unresolved "
                                "parent question and current public evidence. Choose to revise, continue, analyze "
                                "paid evidence, or stop. A stopped execution is not a refuted scientific hypothesis.")
                design_actions = self._question_design_actions()
                comparisons = [q for q in children if q["id"] in changed_children and (
                    q["status"] == "inconclusive" or (q["status"] == "refuted"
                        and (q.get("experiment_choice", {}).get("selected_experiment")
                             or design_actions[q["id"]])))]
                if path and comparisons:
                    child = next((q for q in comparisons if q["id"] == parent.get("reopened_by")), comparisons[-1])
                    self.research_focus.update(comparison_question=deepcopy(child),
                        comparison_status=child["status"],
                        available_designs=deepcopy(design_actions[child["id"]]),
                        answer_observations=[self._action_observation(self.observations[index])
                                             for index in child["evidence"]],
                        prior_trials=[trial for trial in self._experiment_trials()
                                      if trial["question_id"] == child["id"]])
                    start = child.get("experiment_choice", {}).get("started_before_observation")
                    # An analysis error can hide the trial from the outer event
                    # log. The question still retains its actual acquisitions;
                    # a repaired answer can cite them and return to the parent.
                    measured = set(child["evidence"]) & set(child["acquired_observations"])
                    if start is not None and any(index >= start for index in measured):
                        self.research_focus["purpose"] = (
                            "Reconsider the parent investigation using the returned experiment's actual "
                            "outcome and scope. Decide whether to continue or revise this experiment, "
                            "investigate another explanation or measurement, revise the parent procedure, "
                            "analyze existing evidence, or stop. Existing comparison programs remain "
                            "available through their contrast_actions. You choose the next research direction.")
                    else:
                        self.research_focus.update(stage="contrast", origin="host_parent_contrast_continuation",
                            path=f"shared/contrast_{child['id']}.py",
                            purpose="Use the returned answer and its actual scope to revise the comparison. Author "
                                "different predictions for the same observable physical result and candidate "
                                "experiments. The existing contrast selector executes an informative experiment; "
                                "your answer interprets its actual outcome. Preserve the existing parent program.")
                break
        self._save_state()

    def _spec(self):
        return {"current_observation": self._current_observation(), "capabilities": self.lab.capabilities()}

    def _question(self, value):
        question_id = value["id"]
        existing = next((q for q in self.questions if q["id"] == question_id), None)
        if existing is not None:
            # Referencing a question does not erase earlier answers or evidence.
            changed = False
            for key in ("question", "parent_id", "depends_on"):
                if key in value and existing.get(key) != value[key]:
                    existing[key] = deepcopy(value[key])
                    changed = True
            if changed:
                existing["status"] = "reopened"
                self._update_dependencies(question_id)
            for key in ("analysis_path", "parent_program_path"):
                if key in value:
                    existing[key] = value[key]
            return existing
        record = {"id": question_id, "question": value["question"],
                  "parent_id": value.get("parent_id"), "depends_on": list(value.get("depends_on", [])),
                  "status": "open", "evidence": [], "acquired_observations": [],
                  "opened_before_observation": len(self.observations)}
        for key in ("analysis_path", "parent_program_path"):
            if key in value:
                record[key] = value[key]
        self.questions.append(record)
        self._update_dependencies(question_id)
        return record

    def _update_dependencies(self, changed_id):
        changed = {changed_id}
        while True:
            dependents = [q for q in self.questions if q["id"] not in changed and (
                set(q["depends_on"]) & changed or any(
                    child.get("parent_id") == q["id"] and child["id"] in changed for child in self.questions))]
            if not dependents:
                break
            for question in dependents:
                question["status"] = "reopened"
                question["reopened_by"] = changed_id
                changed.add(question["id"])
        for method in self.methods:
            method["available"] = all(any(q["id"] == required and q["status"] == "supported"
                                          for q in self.questions) for required in method["requires"])
        if self.research_advice:
            next_question = (self.research_advice.get("next_action", {}).get("arguments", {})
                             .get("question", {}).get("id"))
            if changed & {self.research_advice.get("question_id"), next_question}:
                self.research_advice = None

    def _program(self, path, function, arguments=None):
        self._save_state()
        memory = self._state()
        memory.update(program_path=path, program_results=[deepcopy(row) for row in self.method_results
                                                         if row["path"] == path])
        memory["program_arguments"] = deepcopy(arguments or {})
        code, stdout, stderr, timed_out = self._python(
            [sys.executable, "-c", {"select_actions": _PLAN_RUNNER,
                                  "answer": _ANSWER_RUNNER, "bind": _BIND_RUNNER,
                                  "design": _DESIGN_RUNNER}[function],
             str(self.workspace / path)],
            json.dumps({"spec": self._spec(), "observations": [
                self._program_observation(record) for record in self.observations],
                "memory": memory}))
        self.program_executions += 1
        if code or timed_out:
            raise ValueError(f"Program {path} failed (returncode={code}, timeout={timed_out}): {stderr[-4000:]}")
        return json.loads(stdout)

    @staticmethod
    def _compact_feedback(feedback):
        result = deepcopy(feedback)
        if isinstance(result.get("last_action"), dict):
            result["last_action"].pop("observation", None)
        return result

    def _procedure_budget_status(self):
        if self.lab.environment_steps >= self.limits["environment_steps"]:
            return "environment_budget_exhausted"
        if monotonic() - self.started >= self.limits["task_wall_seconds"]:
            return "wall_time_exhausted"
        if self.costs["python_execution_seconds"] >= self.limits["python_seconds_per_task"]:
            return "python_budget_exhausted"
        return None

    def _run_procedure(self, laboratory, path, *, max_steps=12, query_arguments=None):
        return run_world_procedure(self, laboratory, path, max_steps=max_steps,
                                   query_arguments=query_arguments)

    def _procedure_terminal_status(self):
        score = self.lab.score()
        if score["completed"]:
            self.status = "completed" if score["completedSuccessfully"] else "environment_completed_unsuccessfully"
        if self.status == "running" and self._inquiry_stack and self._inquiry_stack[-1]["stop_requested"]:
            return "inquiry_finished"
        return self.status if self.status != "running" else None

    def _query(self, arguments, laboratory):
        question_value = arguments.get("question") or arguments.get("research_question")
        if question_value:
            self._question(question_value)
        path = arguments.get("path")
        if arguments.get("method_id"):
            method = next(m for m in self.methods if m["id"] == arguments["method_id"])
            if not method["available"]:
                return {"error": "Method requirements need renewed evidence", "method": method}
            path = method["path"]
            arguments = {**arguments, "parameters": {
                **method.get("parameters", {}), **arguments.get("parameters", {})}}
        if not path:
            return self._acquire(arguments, arguments, laboratory, None)
        if question_value:
            question = self._question(question_value)
            if any(q.get("parent_id") == question["id"] for q in self.questions):
                question["parent_program_path"] = path
                # This invocation sees these answers. Answers changed by its
                # later child analysis remain new to the next parent decision.
                question["reviewed_child_answers"] = self._child_answer_snapshots(question)
                if (self.research_focus and self.research_focus.get("kind") == "parent_research"
                        and self.research_focus["question"]["id"] == question["id"]):
                    self.research_focus = None
        self._current_proposal = None
        proposed = None
        try:
            proposed = self._program(path, "select_actions", arguments)
            self._current_proposal = deepcopy(proposed)
            if isinstance(proposed, dict) and "stop" in proposed:
                feedback = {"status": "method_stopped", "reason": proposed["stop"], "executed_path": path}
            elif isinstance(proposed, dict) and "action" in proposed:
                if path == _RESEARCH_DRIVER_PATH:
                    # Only the executed result may install the next step.
                    self.research_advice = None
                if proposed.get("question"):
                    self._question(proposed["question"])
                params = deepcopy(proposed.get("arguments", {}))
                if proposed["action"] == "run" and params.get("operator") == "inquire":
                    # Existing exported programs know their caller path; carry it
                    # into the child so the retained procedure remains callable.
                    params["question"].setdefault("parent_program_path", path)
                if proposed["action"] == "query" and question_value:
                    params.setdefault("research_question", proposed.get("question") or question_value)
                feedback = self._execute(proposed["action"], params, laboratory)
                feedback = {**feedback, "program_action": deepcopy(proposed), "executed_path": path}
            else:
                feedback = self._acquire(arguments, proposed, laboratory, path)
        except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
            feedback = {"error": f"{type(error).__name__}: {error}", "executed_path": path}
        self._current_proposal = deepcopy(proposed)
        self.method_results.append({"step": len(self.method_results), "path": path,
            "proposal": deepcopy(self._current_proposal), "feedback": self._compact_feedback(feedback)})
        return feedback

    def _acquire(self, arguments, proposed, laboratory, path):
        question_value = (arguments.get("question") or
                          (proposed.get("question") if isinstance(proposed, dict) else None) or
                          arguments.get("research_question"))
        question = self._question(question_value) if question_value else None
        actions = proposed.get("actions", []) if isinstance(proposed, dict) else proposed
        continuation = deepcopy(arguments.get("after_observation",
            proposed.get("after_observation", []) if isinstance(proposed, dict) else []))
        if not isinstance(actions, list) or (not actions and not (path and self.observations and continuation)):
            raise ValueError("query requires actions, or a program analyzing existing evidence")
        indices, action_results = [], []
        for action in actions:
            if self.lab.environment_steps >= self.limits["environment_steps"] or (
                    monotonic() - self.started >= self.limits["task_wall_seconds"]):
                break
            record = laboratory.act(action)
            index = len(self.observations)
            self.observations.append({"index": index, **record})
            self.costs["environment_steps"] = laboratory.environment_steps
            indices.append(index)
            action_results.append(self._action_observation(self.observations[-1]))
            if question is not None:
                question["acquired_observations"].append(index)
                question["experiment_path"] = path
            self._save_state()
            score = laboratory.score()
            if laboratory.environment_steps in self.limits["score_checkpoints_environment_steps"]:
                self.score_checkpoints.append(deepcopy(score))
            if record["feedback"].get("success") is False or record["feedback"].get("errors") or score["completed"]:
                break
        result = {"executed_path": path, "observation_indices": indices, "action_results": action_results,
                "last_action": self.observations[-1] if indices else None,
                "remaining_environment_steps": self.limits["environment_steps"] - laboratory.environment_steps}
        failed = any(r["feedback"].get("success") is False or r["feedback"].get("errors") for r in action_results)
        if failed or (len(indices) < len(actions) and not laboratory.score()["completed"]):
            result["error"] = "Acquisition failed or exhausted its budget; analyze the saved feedback before continuing"
            return result
        if len(indices) < len(actions):
            result["after_observation_skipped"] = "Environment completed before the full proposed batch"
            return result
        if not isinstance(continuation, list):
            result["after_observation_error"] = "after_observation must be a list"
            return result
        outcomes = result["after_observation_results"] = []
        for step in continuation:
            if monotonic() - self.started >= self.limits["task_wall_seconds"]:
                result["after_observation_error"] = "Episode wall-time budget exhausted before analysis"
                break
            action, params = "run", step
            try:
                if not isinstance(step, dict):
                    raise ValueError("after_observation steps must be action objects")
                action = step.get("action", "run")
                params = step.get("arguments", step)
                if action not in {"run", "write", "evaluate"} or params.get("operator") == "procedure":
                    raise ValueError("after_observation declares analysis only; choose further experiments in the next program step")
                feedback = self._execute(action, params, laboratory)
            except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
                feedback = {"error": f"{type(error).__name__}: {error}"}
            outcomes.append({"action": action, "arguments": deepcopy(params), **feedback})
            if feedback.get("error") or feedback.get("returncode") or feedback.get("timeout"):
                result["after_observation_error"] = feedback.get("error", "Authored analysis failed")
                break
        if question is not None:
            result["question"] = deepcopy(question)
        return result

    def _record_answer(self, question, answer):
        previous = {k: question.get(k) for k in ("status", "answer", "evidence")}
        question.update({k: deepcopy(answer[k]) for k in ("status", "answer", "evidence")})
        question["reviewed_through_observation"] = len(self.observations) - 1
        if previous != {k: question.get(k) for k in previous}:
            self._update_dependencies(question["id"])
            if question.get("parent_id") and answer["status"] != "inconclusive":
                self.child_answers_returned += 1

    def _pending_dependencies(self, question):
        supported = {q["id"] for q in self.questions if q["status"] == "supported"}
        # An explicit premise must hold. A child inquiry instead returns an
        # answer that can change the parent's experiment, including a negative
        # or inconclusive answer. Reopening that child makes it pending again.
        pending = set(question["depends_on"]) - supported
        pending.update(q["id"] for q in self.questions
            if q.get("parent_id") == question["id"] and q["status"] != "supported"
            and not (q["status"] in {"refuted", "inconclusive"} and "answer" in q
                     and "reviewed_through_observation" in q))
        return sorted(pending)

    def _contrast(self, arguments, laboratory):
        from .world_experiment_choice import select_experiment

        path = arguments["path"]
        if "code" in arguments:
            super()._execute("write", {"path": path, "code": arguments["code"]}, laboratory)
        question = self._question(arguments["question"])
        blocked = self._pending_dependencies(question)
        if blocked:
            return {"operator": "contrast", "question_id": question["id"],
                    "diagnosis": "unresolved_child", "blocked_by": blocked}
        design = self._program(path, "design")
        choice = select_experiment(design)
        self.costs["research_compute_units"] += choice["compute_cost"]
        previous = question.get("experiment_choice")
        if previous and any(previous[key] != choice[key]
                            for key in ("hypotheses", "selected_experiment")):
            # An answer to the old comparison does not establish revised
            # claims, conditions or predictions. Keep its evidence while the
            # new comparison is unresolved; unchanged repeat tests retain it.
            question["status"] = "reopened"
            self._update_dependencies(question["id"])
        # These are the author's forecasts, frozen before the selected actions.
        # Only the existing answer function can interpret their actual outcome.
        question["experiment_choice"] = {**deepcopy(choice),
                                         "started_before_observation": len(self.observations)}
        # Reuse a separately repaired answer when rerunning a saved design.
        # Switching to a saved revised design also adopts its authored answer
        # when the old analysis belonged to the old experiment itself. An
        # independent repair remains selected until explicitly replaced.
        analysis_path = question.get("analysis_path")
        if ("code" in arguments or not analysis_path or (
                path != question.get("design_path")
                and analysis_path in (question.get("design_path"), question.get("experiment_path"))
                and self._design_action(path, question))):
            analysis_path = path
        question["design_path"], question["analysis_path"] = path, analysis_path
        analysis_path = question["analysis_path"]
        self._save_state()
        result = {"operator": "contrast", "question_id": question["id"], "path": path, **choice}
        selected = choice["selected_experiment"]
        if selected:
            procedure = selected.get("procedure")
            if procedure:
                question["experiment_path"] = procedure["path"]
                result["procedure"] = execute_procedure(self, laboratory, procedure["path"],
                    max_steps=procedure.get("max_steps", 12),
                    query_arguments={"research_question": {"id": question["id"]},
                        "parameters": deepcopy(procedure.get("parameters", {})),
                        "procedure_start": len(self.observations)})
                # A stop is a request to interpret evidence, never a verdict.
                # The authored answer must check achieved conditions and any
                # missing phases. Errors/limits retain their pending evidence.
                if result["procedure"]["status"] == "method_stopped":
                    result["analysis"] = self._execute("evaluate", {
                        "path": analysis_path, "question_id": question["id"]}, laboratory)
            else:
                result["experiment"] = self._acquire({"question": {"id": question["id"]},
                    "after_observation": [{"action": "evaluate", "arguments": {
                        "path": analysis_path, "question_id": question["id"]}}]},
                    {"actions": selected["actions"]}, laboratory, path)
        return result

    def _research_update(self, arguments, laboratory):
        from .world_relation_research import research_relation

        self.research_advice = None
        path = arguments["path"]
        if "code" in arguments:
            super()._execute("write", {"path": path, "code": arguments["code"]}, laboratory)
        question = self._question(arguments["question"])
        question["binding_path"] = path
        blocked = self._pending_dependencies(question)
        if blocked:
            child = next((q for q in self.questions if q["id"] in blocked and q.get("binding_path")), None)
            self.research_advice = {"diagnosis": "unresolved_child", "question_id": question["id"],
                                    "blocked_by": blocked}
            if child:
                self.research_advice["next_action"] = {"action": "run", "arguments": {
                    "operator": "research_update", "path": child["binding_path"],
                    "question": {"id": child["id"]}}}
            return deepcopy(self.research_advice)

        binding = self._program(path, "bind")
        previous = self.relation_research.get(question["id"])
        started = monotonic()
        try:
            result = research_relation(binding, self.observations, previous)
        finally:
            self.costs["python_execution_seconds"] += monotonic() - started
        self.costs["research_compute_units"] += result["compute_cost"]
        self.relation_research[question["id"]] = deepcopy(result)
        candidates = result["candidates"]
        survivor = next((item for item in candidates if item["status"] == "plausible"), None)
        answer = {"program": result.get("selected_program") or (survivor["program"] if survivor else None),
                  "candidates": candidates, "diagnosis": result["diagnosis"]}
        self._record_answer(question, {"status": result["status"], "answer": answer,
                                       "evidence": result["evidence"]})
        self.research_advice = {"question_id": question["id"], "diagnosis": result["diagnosis"],
                               "status": result["status"]}
        probe = result.get("selected_probe")
        if probe:
            self.research_advice["next_action"] = {"action": "query", "arguments": {
                "question": {"id": question["id"]}, "actions": deepcopy(probe["actions"]),
                "after_observation": [{"action": "run", "arguments": {
                    "operator": "research_update", "path": path, "question": {"id": question["id"]}}}]}}
            self.research_advice["selected_probe"] = deepcopy(probe)
        elif result["status"] == "supported" and binding.get("parent_path"):
            parent_id = question.get("parent_id")
            parent = next((q for q in self.questions if q["id"] == parent_id), None)
            pending = self._pending_dependencies(parent) if parent else []
            if pending:
                self.research_advice.update(diagnosis="parent_dependencies_unresolved", blocked_by=pending)
            else:
                self.research_advice["next_action"] = {"action": "query", "arguments": {
                    "path": binding["parent_path"], **({"question": {"id": parent_id}} if parent else {})}}
        result = {**result, "question_id": question["id"], "binding_path": path,
                  "research_advice": deepcopy(self.research_advice)}
        self._save_state()
        return result

    def _execute(self, action, arguments, laboratory):
        if action == "finish" and self._inquiry_stack:
            self._inquiry_stack[-1]["stop_requested"] = True
            return {"status": "inquiry_finished", "lesson": arguments.get("lesson", "")}
        if self._inquiry_stack and self._inquiry_stack[-1].get("stage") == "proposal":
            frame = self._inquiry_stack[-1]
            draft = self.workspace / frame['proposal_path']
            if (action == 'run' and arguments.get('operator') == 'inquire'
                    and arguments['question']['id'] not in {
                        item['question_id'] for item in self._inquiry_stack}):
                from .world_llm_inquiry import (
                    preview_design_proposal, refresh_authored_response, refresh_condition_proposal)
                available = frame['call_limit'] - self.costs['model_calls'] - 1
                if available < 1:
                    return {'operator': 'inquire', 'status': 'model_call_budget_exhausted',
                            'model_calls': 0, 'message': 'The remaining decision is reserved for the author.'}
                params = deepcopy(arguments)
                params['max_model_calls'] = min(params.get('max_model_calls', 8), available)
                # A draft revision is not a dependency of the old adopted law.
                params['question'].setdefault('parent_id', 'proposal:' + frame['proposal_path'])
                if frame.get('condition_proposal'):
                    params.setdefault('condition_proposal', frame['condition_proposal'])
                before_source = draft.read_text() if draft.exists() else None
                before_observations = len(self.observations)
                returned = execute_llm_inquiry(self, laboratory, params)
                if (before_source != (draft.read_text() if draft.exists() else None)
                        or len(self.observations) != before_observations):
                    if frame.get('response_path'):
                        returned.update(refresh_authored_response(self, laboratory, frame))
                    else:
                        compared = preview_design_proposal(self, frame)
                        if compared is not None:
                            returned['design_comparison'] = compared
                        conditions = refresh_condition_proposal(self, frame['proposal_path'])
                        if conditions is not None:
                            returned['condition_research'] = conditions
                return returned
            if (action == 'run' and arguments.get('operator') == 'reference_response'
                    and arguments.get('experiment')
                    and (arguments.get('condition_proposal', frame.get('condition_proposal'))
                         == frame.get('condition_proposal', frame['proposal_path'])
                         or (frame.get('response_path') == arguments.get('path')
                             and arguments.get('response_comparison', {}).get('binding_path')
                             == frame['proposal_path']))):
                # The author selected a test of this draft's computed rules.
                # Reuse ordinary inquiry execution for the complete controller
                # call, including its nested query/evaluate actions. A measured
                # point does not adopt the draft or answer its enclosing law.
                self._question({key: deepcopy(frame['question'][key]) for key in
                                ('id', 'question', 'parent_id', 'depends_on')})
                frame['stage'] = 'inquiry'
                try:
                    return self._execute(action, arguments, laboratory)
                finally:
                    frame['stage'] = 'proposal'
            if action != "write" and not (action == "run" and not arguments.get("operator")):
                deferred = {"action": action, "arguments": deepcopy(arguments)}
                frame["deferred_actions"].append(deferred)
                return {"status": "proposal_action_deferred", "suggested_action": deferred,
                        "message": "Explain why this evidence is needed in your notebook and "
                                   "finish to return it to the caller, or continue local drafting."}
            if action == "run":
                self._python_runs += 1
                arguments = {"path": f"scratch_{len(self.events):03d}_{self._python_runs:03d}.py", **arguments}
            draft = self.workspace / frame["proposal_path"]
            before_source = draft.read_text() if draft.exists() else None
            # Drafting does not register a scientific question or buy evidence.
            written = super()._execute(action, arguments, laboratory)
            if (frame.get("response_path") and (
                    (action == "write" and arguments["path"] == frame["proposal_path"])
                    or before_source != (draft.read_text() if draft.exists() else None))):
                from .world_llm_inquiry import refresh_authored_response
                return {**written, **refresh_authored_response(self, laboratory, frame)}
            if (not frame.get("response_path") and (
                    (action == "write" and arguments["path"] == frame["proposal_path"])
                    or before_source != (draft.read_text() if draft.exists() else None))):
                from .world_llm_inquiry import preview_design_proposal
                compared = preview_design_proposal(self, frame)
                if compared is not None:
                    return {**written, "design_comparison": compared}
            return written
        if action == "run" and arguments.get("operator") == "propose_explanations":
            return execute_llm_inquiry(self, laboratory, arguments)
        if (self.research_focus and self.research_focus.get("stage") == "contrast"
                and action == "write" and (arguments.get("path") == self.research_focus["path"]
                    or any(arguments.get("path") == design["arguments"]["path"]
                           for design in self.research_focus.get("available_designs", [])))):
            focus = self.research_focus
            if "notes" in arguments:
                (self.workspace / "research_notes.md").write_text(arguments["notes"], encoding="utf-8")
            self.research_focus = None
            try:
                comparison = arguments.get("question") or focus["comparison_question"]
                declared = arguments.get("research_question", {})
                if "question" not in arguments and declared.get("id") == comparison["id"]:
                    comparison = {**comparison, **declared}
                result = self._contrast({"path": arguments["path"], "code": arguments["code"],
                                         "question": comparison}, laboratory)
            except (ValueError, KeyError, TypeError, StopIteration, OSError):
                self.research_focus = focus
                raise
            parent = self._question({"id": focus["question"]["id"]})
            if focus.get("previous_execution"):
                parent["reviewed_program_step"] = focus["previous_execution"]["step"]
            # A valid design consumes the old return. Its newly acquired answer
            # remains different and can inform the next parent decision.
            parent.setdefault("reviewed_child_answers", {}).update({q["id"]: {
                key: deepcopy(q[key]) for key in ("status", "answer", "evidence")}
                for q in focus["returned_children"]})
            self._save_state()
            return {**result, "trigger_origin": focus["origin"]}
        if self.research_focus and self.research_focus.get("kind") == "parent_research":
            focus = self.research_focus
            if action != "write" or arguments.get("path") != focus["path"]:
                # Execute through the ordinary dispatcher before accepting the
                # review. A rejected action must leave this return available for
                # repair instead of moving the author to another parent.
                self.research_focus = None
                try:
                    result = self._execute(action, arguments, laboratory)
                except (ValueError, KeyError, TypeError, StopIteration, OSError):
                    self.research_focus = focus
                    raise
                if result.get("error"):
                    self.research_focus = focus
                    return result
                parent = self._question({"id": focus["question"]["id"]})
                if focus.get("previous_execution"):
                    parent["reviewed_program_step"] = focus["previous_execution"]["step"]
                parent.setdefault("reviewed_child_answers", {}).update({q["id"]: {
                    key: deepcopy(q[key]) for key in ("status", "answer", "evidence")}
                    for q in focus["returned_children"]})
                self._save_state()
                return result
        if "notes" in arguments:
            (self.workspace / "research_notes.md").write_text(arguments["notes"], encoding="utf-8")
        if isinstance(arguments.get("question"), dict):
            self._question(arguments["question"])
        if isinstance(arguments.get("research_question"), dict):
            self._question(arguments["research_question"])
        operator = arguments.get("operator")
        if action == "run" and operator == "inquire":
            return execute_llm_inquiry(self, laboratory, arguments)
        if action == "run" and operator == "reference_hypothesis":
            return execute_reference_hypothesis(self, laboratory, arguments)
        if action == "run" and operator == "reference_expansion":
            return execute_reference_expansion(self, laboratory, arguments)
        if action == "run" and operator == "reference_response":
            return execute_reference_response(self, laboratory, arguments)
        if action == "run" and operator == "contrast":
            return self._contrast(arguments, laboratory)
        if action == "run" and operator in {"research", "research_update"}:
            result = self._research_update(arguments, laboratory)
            if operator == "research" and self.research_advice.get("next_action"):
                super()._execute("write", {"path": _RESEARCH_DRIVER_PATH,
                                           "code": _RESEARCH_DRIVER}, laboratory)
                initial = deepcopy(result)
                procedure = execute_procedure(self, laboratory, _RESEARCH_DRIVER_PATH,
                    max_steps=arguments.get("max_steps", 12))
                latest = self.relation_research.get(arguments["question"]["id"], {})
                result = {**result, **deepcopy(latest), "initial_research": initial,
                          "procedure": procedure, "research_advice": deepcopy(self.research_advice)}
            return {"operator": operator, **result}
        if action == "run" and operator == "procedure":
            method = {}
            if arguments.get("method_id"):
                method = next(m for m in self.methods if m["id"] == arguments["method_id"])
                if not method["available"]:
                    return {"error": "Method requirements need renewed evidence", "method": deepcopy(method)}
            query_arguments = {"parameters": {
                **method.get("parameters", {}), **arguments.get("parameters", {})},
                "procedure_start": len(self.observations)}
            question = arguments.get("question") or arguments.get("research_question")
            if question:
                query_arguments["research_question"] = deepcopy(question)
            return {"operator": "procedure", **self._run_procedure(laboratory,
                method.get("path") or arguments["path"],
                max_steps=arguments.get("max_steps", method.get("max_steps", 12)),
                query_arguments=query_arguments)}
        if action == "run" and operator == "question":
            return {"question": deepcopy(self._question(arguments))}
        if action == "run" and operator == "method":
            method = {k: deepcopy(arguments[k]) for k in ("id", "path", "applicability",
                "parameters", "max_steps", "provenance", "evidence_file") if k in arguments}
            method["requires"] = list(arguments.get("requires", []))
            if not (self.workspace / method["path"]).is_file():
                raise ValueError("Write the executable method before registering it")
            self.methods = [m for m in self.methods if m["id"] != method["id"]] + [method]
            self._update_dependencies("")
            return {"method": method}
        if action == "evaluate":
            question = next(q for q in self.questions if q["id"] == arguments["question_id"])
            analysis_path = arguments["path"]
            if "code" in arguments:
                source = self.workspace / analysis_path
                functions = set()
                if source.is_file():
                    try:
                        functions = {node.name for node in ast.parse(source.read_text()).body
                                     if isinstance(node, ast.FunctionDef)}
                    except SyntaxError:
                        pass
                if functions & {"select_actions", "design"}:
                    # Analysis revises an answer; write revises an experiment.
                    analysis_path = f"shared/analysis_{question['id']}.py"
                super()._execute("write", {"path": analysis_path, "code": arguments["code"]}, laboratory)
            answer = self._program(analysis_path, "answer")
            if answer.get("status") not in ("supported", "refuted", "inconclusive"):
                raise ValueError("Analysis must return supported, refuted or inconclusive")
            evidence = answer.get("evidence", [])
            if not evidence or any(type(i) is not int or not 0 <= i < len(self.observations) for i in evidence):
                raise ValueError(
                    "Analysis must reference actual observation indices; "
                    f"returned evidence={evidence!r}, valid global indices are "
                    f"0 <= index < {len(self.observations)}. "
                    f"Actual analysis result: {answer!r}")
            question["analysis_path"] = analysis_path
            self._record_answer(question, answer)
            return {"question": deepcopy(question), "analysis_is_agent_authored": True}
        if action == "finish":
            self.lesson = arguments.get("lesson", "")
            self.status = "agent_finished"
            return {"lesson": self.lesson}
        if action == "write" and self.research_focus:
            focus = self.research_focus
            if arguments.get("path") != focus["path"]:
                if focus.get("kind") == "parent_research" or (
                        self._inquiry_stack and focus.get("kind") == "analysis"):
                    return super()._execute(action, arguments, laboratory)
                return {"error": f"Write the prerequisite experiment to {focus['path']}"}
            written = super()._execute(action, arguments, laboratory)
            question_id = focus["question"]["id"]
            if focus.get("kind") == "parent_research":
                question = self._question({"id": question_id})
                question["parent_program_path"] = focus["path"]
                # Its child steps may write/evaluate other files. They use the
                # ordinary executor, not the outer authoring constraint.
                self.research_focus = None
                if (self._inquiry_stack and self._inquiry_stack[-1]["assignment"].get(
                        "failed_program_path") == focus["path"]):
                    # The automatic repair caller already executes a returned
                    # edit with the original phase, parameters and step limit.
                    # Keep that contract; the local researcher can explicitly
                    # run a diagnostic before returning it.
                    result = written
                else:
                    try:
                        procedure = self._run_procedure(laboratory, focus["path"], max_steps=12)
                    except (ValueError, KeyError, TypeError, StopIteration, OSError):
                        self.research_focus = focus
                        raise
                    result = {**written, "procedure": procedure, "question": deepcopy(question),
                              "trigger_origin": focus["origin"]}
                if focus.get("previous_execution"):
                    question["reviewed_program_step"] = focus["previous_execution"]["step"]
                question.setdefault("reviewed_child_answers", {}).update({q["id"]: {
                    key: deepcopy(q[key]) for key in ("status", "answer", "evidence")}
                    for q in focus["returned_children"]})
                if "procedure" in result and result["procedure"]["status"] == "method_error":
                    focus.update(stage="review", origin="host_parent_program_review",
                                 previous_execution=deepcopy(self.method_results[-1]))
                    self.research_focus = focus
                if "question" in result:
                    result["question"] = deepcopy(question)
                self._save_state()
                return result
            if focus.get("kind") == "analysis":
                analysis = self._execute("evaluate", {"path": focus["path"], "question_id": question_id}, laboratory)
                self.research_focus = None
                return {**written, **analysis, "trigger_origin": "host_analysis_continuation"}
            experiment = self._query({"path": focus["path"], "question": {"id": question_id},
                "after_observation": [{"action": "evaluate", "arguments": {
                    "path": focus["path"], "question_id": question_id}}]}, laboratory)
            question = next(q for q in self.questions if q["id"] == question_id)
            if ((focus.get("kind") == "intervention" and
                 question.get("reviewed_through_observation", -1) >= max(focus["evidence"])) or
                    (focus.get("kind") != "intervention" and question["status"] == "supported")):
                self.research_focus = None
                self._prerequisite_checked_through = len(self.observations) - 1
            return {**written, "prerequisite_experiment": experiment,
                    "question": deepcopy(question), "trigger_origin": focus.get("origin", "host_execution_failure")}
        if action == "write":
            written = super()._execute(action, arguments, laboratory)
            for question in self.questions:
                if question.get("design_path") == arguments["path"]:
                    # Ordinary write -> contrast is also an authored revision.
                    # Its new answer supersedes a separate repair of old code.
                    question["analysis_path"] = arguments["path"]
            declared = arguments.get("question") or arguments.get("research_question")
            if declared:
                question = next(q for q in self.questions if q["id"] == declared["id"])
                contrast_action = self._design_action(arguments["path"], question)
                if contrast_action:
                    written["contrast_action"] = contrast_action
            return written
        if action == "run":
            # Several authored substeps can execute within one model event.
            # Keep their generated scripts and complete outputs distinct.
            self._python_runs += 1
            arguments = {"path": f"scratch_{len(self.events):03d}_{self._python_runs:03d}.py", **arguments}
        return super()._execute(action, arguments, laboratory)

    def result(self):
        self.costs["wall_time_seconds"] = monotonic() - self.started
        return {"status": self.status, "agent_arm": self.agent_arm, "costs": deepcopy(self.costs),
                "score": self.lab.score(), "score_checkpoints": deepcopy(self.score_checkpoints),
                "research": {"questions": deepcopy(self.questions), "methods": deepcopy(self.methods),
                             "relation_research": deepcopy(self.relation_research),
                             "research_advice": deepcopy(self.research_advice),
                             "research_focus": deepcopy(self.research_focus),
                             "llm_inquiries": deepcopy(self.llm_inquiries),
                             "prerequisite_checked_through": self._prerequisite_checked_through,
                             "child_answers_returned": self.child_answers_returned,
                             "program_executions": self.program_executions},
                "method_results": deepcopy(self.method_results),
                "lesson": self.lesson, "events": deepcopy(self.events)}

    def _model_decision(self, request):
        """Charge every parent/child model turn to the same existing account."""
        remaining = self.limits["task_wall_seconds"] - (monotonic() - self.started)
        if hasattr(self.provider, "timeout_seconds"):
            self.provider.timeout_seconds = min(self.limits["provider_request_timeout_seconds"], remaining)
        self.costs["model_calls"] += 1
        try:
            decision = self.provider.next_action(request)
        except Exception as error:
            decision = {"action": "provider_error", "error": f"{type(error).__name__}: {error}"}
        usage = decision.get("usage", {})
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            if isinstance(usage.get(key), int):
                self.costs[key] += usage[key]
        if isinstance(usage.get("total_tokens"), int):
            self.costs["reported_tokens"] += usage["total_tokens"]
        else:
            self.costs["usage_missing_calls"] += 1
        return decision

    def run(self, laboratory, *, on_checkpoint=None):
        self.lab = laboratory
        self.started = monotonic() - self.costs["wall_time_seconds"]
        self._on_checkpoint = on_checkpoint
        if self.agent_arm == "hegel" and not any(q['id'] == 'task' for q in self.questions):
            task = laboratory.public_observation().get("taskProgress", [])
            root = self._question({"id": "task", "question": "\n".join(
                row.get("description", row.get("taskName", "")) for row in task)})
            root["origin"] = "public_task"
        self._save_state()
        if on_checkpoint:
            on_checkpoint(self.result())
        while self.status == "running":
            score = laboratory.score()
            if score["completed"]:
                self.status = "completed" if score["completedSuccessfully"] else "environment_completed_unsuccessfully"
                break
            if laboratory.environment_steps >= self.limits["environment_steps"]:
                self.status = "environment_budget_exhausted"
                break
            if self.costs["model_calls"] >= self.limits["max_model_calls"]:
                self.status = "model_call_budget_exhausted"
                break
            remaining = self.limits["task_wall_seconds"] - (monotonic() - self.started)
            if remaining <= 0:
                self.status = "wall_time_exhausted"
                break
            if dispatch_response_premise(self, laboratory):
                continue
            request = self._request()
            decision = self._model_decision(request)
            usage = decision.get("usage", {})
            if decision["action"] == "provider_error":
                error = decision.get("error", {})
                if isinstance(error, dict) and error.get("decode_error"):
                    # Reuse the ordinary Code/Inquiry paid-repair contract:
                    # this completed reply consumed a call, but no world action.
                    self.events.append({"action": "provider_error", "arguments": {},
                        "feedback": {"error": error["decode_error"]}, "usage": usage})
                    if on_checkpoint:
                        on_checkpoint(self.result())
                    continue
                self.events.append(decision)
                self.status = "provider_failed"
                break
            arguments = decision.get("arguments", {})
            try:
                feedback = self._execute(decision["action"], arguments, laboratory)
            except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
                feedback = {"error": f"{type(error).__name__}: {error}"}
            # Current observation has its own context slot; old event previews
            # keep feedback and indices without duplicating full world states.
            if isinstance(feedback.get("last_action"), dict):
                feedback = deepcopy(feedback)
                feedback["last_action"].pop("observation", None)
            self.events.append({"action": decision["action"], "arguments": deepcopy(arguments),
                                "feedback": feedback, "usage": usage})
            self._save_state()
            if on_checkpoint:
                on_checkpoint(self.result())
        result = self.result()
        if on_checkpoint:
            on_checkpoint(result)
        return result
