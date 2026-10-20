"""An ordinary Python coding agent with real experimental and execution tools."""

from dataclasses import asdict, dataclass
from copy import deepcopy
from contextlib import nullcontext
import errno
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Protocol

from .models import Observation, Probe
from .evaluation import point_loss
from .proposers import _events, _run
from .code_provider_context import prepare_code_context
from .checkpoint_storage import clone_or_copy, write_text_incrementally
from .representation_diagnostics import REPRESENTATION_INTERFACE, representation_ambiguity


@dataclass(frozen=True)
class CodeAgentConfig:
    observation_budget: int = 128
    measurement_budget: int = 512
    compute_budget: int = 100000
    execution_seconds: float = 60.
    max_steps: int = 20
    max_model_calls: int = 16
    token_budget: int = 60000


class CodeProvider(Protocol):
    def next_action(self, request: dict) -> dict:
        """Return {action, arguments, usage}; usage includes reported tokens."""


_ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"action": {"type": "string", "enum": ["query", "write", "run", "evaluate", "finish"]},
                   "arguments_json": {"type": "string"}},
    "required": ["action", "arguments_json"]}

_INSTRUCTIONS = """You are a scientific coding agent. Infer the numerical process from paid observations.
Use the supplied query/write/run/evaluate/finish tools through one JSON action per turn.
Put the chosen action only in your final message. Only that final response is executed;
intermediate write/practice text does not change files or run a practice. Wait for the next
request's actual events and feedback before choosing a subsequent action.
You may write and execute unrestricted ordinary Python in your own working directory, fit parameters,
import available scientific libraries, inspect your own outputs, and repair programs after errors.
The environment is accessible through query only. Its implementation and final evaluation rows are not
part of this task. Do not inspect other directories or call external tools directly; return the action
for the supplied execution loop. Programs are ordinary Python, not an expression AST.
Model contract: model.py defines predict(row)->float; row contains features and history (past feature
values plus past __target__), never the current target. Optional fit(observations) runs on the acquired
training rows from the selected measurement view before predictions. Each query resets its trajectory. A fit script can also save local
parameters for model.py. Query controls must obey the public bounds. Channels are optional paid inputs.
When measurement_views are supplied, choose one listed channel combination. Each view lists its
measured features and public object roles; these can differ from the command names. For evaluate or
finish, channels selects the model's view. Compatible acquired rows are projected into that view;
coarse readings cannot recover individual readings. observations.json retains all purchased rows.
query, run, evaluate and finish refresh model_observations.json with compatible paid rows in the
selected view. Use this file for fitting and residuals; it excludes incompatible rows and projects
features and history consistently. A run action accepts channels; omitted channels use the latest
query view. The fit(observations) argument uses the same projected training rows.
Raw paid data are available in observations.json. The final model must work for new trajectories.
When a query also returns paid before/after measurements, its feedback names transitions.json.
This file retains batches with channels, observation_start (the zero-based offset in observations.json),
and transitions in the same order as that batch's observations. Iterate batches first, then each
batch["transitions"]; these are not a flat list of before/after records. Use these measured transitions to
study the process; they are part of the purchased evidence, not additional prediction inputs.
When session_knowledge is supplied, this is a continuing research programme. Reuse and revise its
ordinary Python libraries and previous lessons when the acquired evidence supports them. Files under
shared/ survive into later tasks; write reusable code and concise applicability/failure lessons there.
Other tasks can change their parameters or mechanism. Refit on the current observations instead of
blindly copying old coefficients. You choose what to preserve and how to use it. A finish action may
include lesson (text describing what was learned) and reused_files (shared paths actually used).
The request previews the latest 12 observations with their original row indices and trajectory_start
markers. observations.json contains every acquired row including its full history. research_context.json
contains the complete current request, including every earlier action, argument and tool output.
The request summarizes older actions and includes the latest tool feedback. Read the files with run
when you need more evidence, and print concise numerical summaries instead of whole datasets.
All source and current paid data remain available; the preview is not a fitting dataset.
Tools, with JSON arguments encoded in arguments_json:
query: {sequence:[{control:value},...],channels:[names]}; select probes and instrument channels yourself.
write: {path:'model.py',code:'ordinary Python source'}; also supports other local files.
run: {code:'ordinary Python to execute',channels:[names]} or {path:'fit.py',channels:[names]};
refreshes model_observations.json before execution and returns stdout/stderr and exit code.
evaluate: {model:'model.py',channels:[names]}; report training task loss in the selected view (channels optional).
finish: {model:'model.py',channels:[needed channel names]}; validates the executable then ends research.
The public risk_quantile fixes the loss: q=0.5 uses squared error; other q values use pinball loss
max(q*(target-prediction), (q-1)*(target-prediction)). Optimize the reported loss. MSE is also reported
as a squared-error diagnostic and can differ from the task loss. The risk target cannot be changed.
Prefer efficient programs and finish within the remaining calls. Use enough varied observations to
identify history dependence. All statistical fitting must use acquired observations, not guesses at
private coefficients. A bad execution is returned as feedback so you can repair it.
"""

_TASK_INTERFACE_INSTRUCTIONS = """You are a scientific coding agent. Infer the numerical process from paid observations.
Follow the supplied task_interface and capabilities for the exact action arguments, prediction inputs,
measurements, confirmation procedure, and knowledge persistence. They define this task's contract.
Return one JSON action per turn, with its argument object encoded in arguments_json.
Put the chosen action only in your final message. Only that final response is executed;
intermediate write/practice text does not change files or run a practice. Wait for the next
request's actual events and feedback before choosing a subsequent action.
Write and run ordinary Python in your working directory, use available scientific libraries, inspect
your outputs, and revise programs after errors. Access the environment through the supplied actions.
Its implementation, private settings, and final evaluation targets are outside this task. Do not inspect
other directories or call external tools directly; return an action for the supplied execution loop.
Fit and assess hypotheses using purchased evidence. Respect the public bounds and remaining budgets.
The request contains compact evidence previews and action feedback. research_context.json retains the
full request; the files named in evidence and task_interface retain the complete paid observations.
Read those files through a run action when needed. Print concise results instead of complete datasets.
Choose experiments, repair errors, and finish within the available calls and resources.
"""

_INSTRUCTIONS += REPRESENTATION_INTERFACE
_TASK_INTERFACE_INSTRUCTIONS += REPRESENTATION_INTERFACE


_CODEX_SOURCE_INSTRUCTIONS = """\nJSON RESPONSE FORMAT:
Include source_code in every final JSON response. When write or run supplies code,
put the exact source text in the top-level source_code string, not arguments_json.
The same applies to another action whose task interface explicitly accepts code.
arguments_json encodes only the remaining argument object (path, operator, etc.).
The executor maps source_code to that action's code argument without changing it.
Use source_code:null for actions without inline code, including run with a path.
The earlier tool examples describe execution arguments, not a second code encoding.
Source text, backslashes and line breaks need only the normal outer JSON encoding;
do not JSON-encode the source again or duplicate it inside arguments_json.
"""


class CodexCodeProvider:
    """Use the currently configured Codex model as the code-agent decision maker."""

    def __init__(self, workspace, *, executable=None, timeout_seconds=120, model=None, reasoning_effort=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.executable = executable or shutil.which("codex")
        self.timeout_seconds, self.model = timeout_seconds, model
        self.reasoning_effort = reasoning_effort
        self.calls = max([0, *(int(path.stem.rsplit("_", 1)[1])
                               for path in self.workspace.glob("code_action_[0-9][0-9][0-9].json"))])

    def next_action(self, request):
        if not self.executable:
            raise RuntimeError("Codex CLI was not found on PATH")
        self.calls += 1
        schema = self.workspace / "code_action_schema.json"
        response = self.workspace / f"code_action_{self.calls:03d}.json"
        instructions = self.workspace / "code_agent_instructions.txt"
        action_schema = deepcopy(_ACTION_SCHEMA)
        action_schema["properties"]["source_code"] = {
            "type": ["string", "null"],
            "description": "Exact inline source text for the action's code argument; null when absent."}
        action_schema["required"].append("source_code")
        instruction_text = _TASK_INTERFACE_INSTRUCTIONS if "task_interface" in request else _INSTRUCTIONS
        if "research_notes" in request:
            action_schema["properties"]["research_notes"] = {
                "type": "string", "description": "Complete concise current research notebook: observed evidence, "
                "working explanation, unresolved alternatives, predictions and next experiment."}
            action_schema["required"].append("research_notes")
            if 'evidence_notebook' in request:
                action_schema['properties']['research_notes']['description'] = (
                    'Current scientific interpretation and open hypotheses, citing evidence_notebook references; '
                    'the runtime preserves the empirical facts separately.')
                instruction_text += """\nInclude research_notes in every final JSON response. Each decision starts a fresh
session. evidence_notebook is empirical memory rendered from the actual executed
records; research_notes is the prior author's interpretation and can be wrong.
Use the evidence references (revision:N, record:N, confirmation:N, analysis:N) when carrying
a finding into the new notebook. The referenced record already supplies its rows,
prediction cutoff and measured result; do not reconstruct that association from
the old notebook or attach its forecasts to a different acquisition.
Correct any conflicting prior interpretation from these records. Preserve useful
working hypotheses, unresolved alternatives and the next step in research_notes.
An intended action is not an observed result. Prior failed attempts and full source
records remain on disk. You choose the scientific interpretation and next action.
Put notebook text only in this top-level field, not arguments_json.
"""
            else:
                instruction_text += """\nInclude research_notes in every final JSON response alongside action and arguments_json.
Each decision starts a fresh model session. Carry forward a concise COMPLETE current notebook,
not merely a change log: preserve useful observed results and their evidence indices, working
explanations, unresolved alternatives, predictions and the next experiment. Update it from the
actual feedback. Distinguish observations from hypotheses and planned actions; the action you
are now proposing has not happened yet. This authored text replaces research_notes.md and is
shown on the next turn. Ordinary task actions and tools are otherwise unchanged.
Put notebook text only in this top-level research_notes field, never in
arguments_json. Keep arguments_json limited to the chosen action's tool arguments.
"""
        if "research_root" in request:
            action_schema["properties"]["research_question"] = {"anyOf": [
                {"type": "null"},
                {"type": "object", "additionalProperties": False,
                 "properties": {"id": {"type": "string"}, "question": {"type": "string"},
                                "parent_id": {"type": ["string", "null"]},
                                "depends_on": {"type": "array", "items": {"type": "string"}}},
                 "required": ["id", "question", "parent_id", "depends_on"]}]}
            action_schema["required"].append("research_question")
            instruction_text += "\nReturn research_question for the live question advanced by this action, or null for routine actions. " \
                "Use the supplied task root and stable question IDs. A child is an unresolved inquiry informing " \
                "its parent; a negative answer can change the next experiment. Use depends_on only for premises " \
                "that must be supported. Registering an inquiry grants no scientific support.\n"
        if "research_focus" in request:
            if request["research_focus"].get("kind") == "analysis" and "llm_inquiry" not in request:
                action_schema["properties"]["action"]["enum"] = ["write", "run", "evaluate", "finish"]
            elif request["research_focus"].get("kind") not in {"parent_research", "analysis"}:
                action_schema["properties"]["action"]["enum"] = ["write", "run", "finish"]
            instruction_text += """\nYou may use ordinary run for calculations or code debugging before answering.
When task_interface supplies run operator:'inquire', you may instead delegate the
current research_focus.question to that continuing local LLM loop, with its ID,
parent_id and depends_on. Its tool feedback and local notebook persist; it returns
to this parent after analysis or local finish. Use the existing evidence and
execution tools; delegating a question does not establish its answer.
"""
            if request["research_focus"].get("kind") == "parent_research":
                if request["research_focus"].get("stage") == "contrast":
                    instruction_text += """\nA child answer has returned for comparison in an existing parent research program.
Choose how to resolve research_focus.comparison_question. Available registered methods
and their separate prior execution records appear in method_decision. You may invoke a
method's execute_action with your own question, analyze existing local evidence, or design
a new contrast. A supplied control procedure can acquire reference evidence directly;
replicating a known result requires no invented pair of opposing scientific predictions.
research_focus.available_designs lists retained comparison programs. An empty prior_trials
means no frozen contrast execution is recorded for this child: its draft is still a proposal,
and its returned answer may come from ordinary query/evaluate. Use answer_observations to
interpret that answer's actual scope, then revise the retained design or author a new one.
The failed tested setting does not automatically reject an entire relation or support
the other draft hypothesis. Before testing a newly fitted or extrapolated numerical
prediction, execute its calculation from cited public readings. You can return
run {code:'Python source'} to fit/compare candidates and print their coefficients,
reference residuals and target predictions, then author a contrast from those results.
Alternatively run {operator:'research',path:'binding.py',code:'Python source',
question:{id:'...',question:'...',parent_id:'...',depends_on:[]}} invokes the existing
bind/rows/probes researcher under task_interface. Choose the measured variables
yourself. A requested setting or its readback is not a measurement of the correct
setting: use actual physical outcomes to test the numerical prediction. Ordinary
Python calculation is sufficient when the intervention yields a Boolean outcome;
you need not invent a numeric target or force a numerical binding.
Compute each revised explanation's predictions and choose
an experiment that distinguishes them, rather than treating another setting as informative
merely because it has not been tried. The existing selector consumes your actual design.
To design a new contrast, return write {path:research_focus.path,code:'Python source'} defining design and answer
under the shared contrast contract. You may instead revise a retained comparison in
research_focus.available_designs by writing to that program's path. Writing either the
scheduled path or one of those retained comparison paths runs the revised experiment.
The host runs the existing disagreement selector,
executes the selected experiment's actions or existing feedback procedure, and analyzes
its ACTUAL outcome. For live controls, return procedure:{path:research_focus.path,max_steps:12}
instead of actions and define select_actions in the same file. It reads each returned
setting/state before choosing the next step; frozen predictions stay in experiment_choice.
For an ordered condition, adapt the phase-tracking pattern from an existing registered
feedback procedure in method_decision. Use this invocation's successful observations
(memory.program_arguments.procedure_start) to establish which phases actually finished.
Current inventory alone cannot distinguish before and after a completed phase: a
remove/replace controller can otherwise repeat those actions forever. Advance only
when the phase's required action and outcome occurred; inspect the current controls
to choose the next action. Reading an existing setting and changing a setting are
different interventions. Keep the hypothesis, selected experiment and executed phase
sequence consistent. If you change the tested condition, state that narrower question
explicitly; its result does not adjudicate the original stronger condition.
The parent program is preserved. This is a comparison step, not a replacement parent procedure.
If the returned status is refuted, revise the comparison using that specific
counterexample and research_focus.prior_trials. A failed conditional prediction
does not disprove its whole formula family or confirm every alternative. Identify
which claimed conditions actually held, which prediction failed, and what the
revised explanations predict differently. Existing measurement_history includes
earlier paid USE/READ results; consider alternative measured variables or conditions
before repeating a nearby setting under the same untested assumptions. Use ordinary
Python to calculate candidate predictions from those readings. No variable or law
is selected for you. To revise the comparison's scope, include question:{id:...,
question:'...',parent_id:research_focus.question.id,depends_on:[...]} in the write
arguments, or explicitly run contrast with your authored question and design.
Define competing testable claims and their predictions for the SAME observable outcome.
Use the returned uncertainty, paid evidence and reference_history to choose the comparison.
When an additional condition could explain failure, a reversible test on a known working
configuration can distinguish the condition from a wrong response law. State what can
transfer to the failed target; success elsewhere alone does not establish shared conditions.
Build actions from CURRENT public controls. Parse complete numeric readouts, including
decimals, and identify the actual object before reusing a controller from a different one.
Compute adjustments in Python, and have answer
check the actually achieved conditions and physical response before accepting or rejecting
a prediction. A mismatched applied setting does not test the intended numerical prediction.
answer returns supported/refuted/inconclusive with actual observation indices for the
comparison_question, not the entire parent task. It must read the observed records and
the frozen experiment_choice. A setting readback is not physical confirmation; an explicitly
negative physical outcome is not missing data. No task formula or verdict is supplied.
If your current explanations predict all available experiments identically, return that
design honestly; the selector reports ambiguity without buying evidence or granting support.
finish can end unresolved research. Existing programs, evidence and answers remain available.
"""
                elif request["research_focus"].get("stage") == "review":
                    instruction_text += """\nThe host is returning an existing parent program's actual execution for review.
research_focus.previous_execution contains its returned proposal and real feedback.
When available, procedure_status and procedure_steps describe the entire invocation;
a successful final action may still belong to a procedure that hit its step limit.
Inspect that program and compare what it actually did with its intended experiment.
Use the stop reason, acquired observations, actual physical outcomes and current state
to decide whether to revise the program, continue it, analyze paid evidence, or stop.
Comments describing hypotheses do not execute an experiment. Separate an operation
failure from a failed physical prediction, and choose which explanation to test next.
This review is offered once for this execution; any chosen action returns control.
The old program remains available. A stop is not automatically a scientific refutation.
Before further operational repairs, examine whether the assumed operation is actually
necessary to answer the parent question, or is itself an untested explanation.
Use an explicit child inquiry for that distinction. Paid evidence may already answer
it; otherwise contrast competing explanations on the same observable physical outcome.
The shared contrast operator accepts your design and answer functions, selects among
your executable experiments using the existing disagreement chooser, and returns actual
analysis to the parent. An experiment all explanations predict identically cannot
resolve their disagreement. You choose the explanations and experimental controls.
"""
                elif request["research_focus"].get("stage") == "child_return":
                    instruction_text += """\nNew child analysis has returned to a parent with an existing research program.
research_focus.changed_child_ids identifies the answers not yet considered by this parent.
Use their actual scope and evidence to decide the next experiment. The existing program
is available; its previous review does not settle these new answers. A refuted or
inconclusive child can change the parent's question, conditions, or experiment without
requiring the rejected assumption to become true. Choose an action; rewriting is optional.
When a child cannot distinguish a wrong response law from an unmet condition, inspect
the shared reference_history and design a matched control to resolve that uncertainty.
"""
                else:
                    instruction_text += """\nThe host is returning from answered child research to its reopened parent.
The parent has no executable research program yet. Use the paid records, computed
results and current notebook to author its next scientific experiment.
"""
                if request["research_focus"].get("stage") != "contrast":
                    instruction_text += """
research_focus.returned_children contains the actual answers; preserve their valid scope.
Choose the next research action before deciding whether a program needs rewriting:
run contrast to compare explanations by their predicted physical outcomes, evaluate paid
evidence that already answers the question, query an existing experiment, or write a
procedure when its execution needs changing. Existing programs and analyses remain usable.
If authoring or revising a procedure, return
"""
                    instruction_text += "write {path:research_focus.path,code:'Python source'} "
                    instruction_text += ("to save the repair, and define\n" if request.get("procedure_repair")
                                         else "to run it immediately, and define\n")
                    instruction_text += """
select_actions(public_spec, observations, memory). The existing procedure executor
will call it repeatedly, with fresh public state and previous method_results at each step
(up to 12 steps per invocation, charging ordinary action/Python costs).
State your competing explanations and their DIFFERENT physical predictions for the SAME
intervention in the returned proposal, e.g. hypotheses and predicted_outcomes alongside
actions. Select actions that distinguish those predictions using public controls; observe
the actual response before choosing the next step. Your Python can calculate candidate
predictions without translating arbitrary formulas into a new language.
For existing numerical search, bind candidate residuals or predictions as features and
use the research operator with executable probes and actual outcome targets. Boolean
abs/sub/lt relations are already supported. A chosen setting is not the target truth.
You can instead use query actions plus after_observation evaluate in this procedure.
Define answer in the same file if using it, register the scientific question before its
experiment, and read actual outcome fields in answer. Cite observed conditions and
physical results, rather than command success alone. A failed prediction may revise the
response model or open a specific condition question; your program chooses the branch.
If the chosen experiment uses a dialog, use its live menu: navigate/open it, select a
currently listed option, then inspect its returned setting/state. Keep the candidate
fixed during its test. memory.method_results contains ALL programs' results, including
inherited history. Filter records by path before interpreting this program's progress;
len(memory.method_results) is not a local step counter. Branch on recorded outcomes.
Inside your program, memory.program_path identifies this file and memory.program_results
already selects its own history, including earlier executions. A failed action does not
advance an intended preparation step; inspect its feedback and current public state.
Return {stop:'reason and unresolved question'} when evidence or controls are insufficient;
do not manufacture confirmation or loop on already-paid analysis. The parent is not marked
supported merely because this program runs. Its path is retained for continuation/revision.
The host supplies neither candidate formulas nor a device-specific experiment.
"""
            elif request["research_focus"].get("kind") == "analysis":
                instruction_text += """\nThe host is continuing analysis of a child you already declared and measured.
research_focus lists its unreviewed observation indices and question. Reuse an existing
analysis with evaluate {path:'analysis.py',question_id:'q'}, optionally including corrected
code, or author one with write {path: research_focus.path, code: 'Python source'}. Define
answer(public_spec, observations, memory), returning {status:'supported'|'refuted'|'inconclusive',
answer: your explanation and remaining uncertainty, evidence:[actual record indices]}.
The host executes this answer with the existing evaluate tool on the paid records;
no select_actions function or additional acquisition is needed for this decision.
The original design/experiment path remains in the question; this analysis file is
separate so a later experiment revision can still reuse the actual research program.
Inspect the actual public feedback and object state. An API action succeeding or a control
value matching your chosen setting alone does not demonstrate a physical prediction.
An absent measurement/outcome is missing evidence. Judge the declared child, not the whole task.
Inconclusive and refuted analyses also return control so you can choose the next experiment.
You may first evaluate a correction to an earlier answer on its paid evidence. That
updates its formal status; noting an error alone does not. The pending child's analysis
remains available until you address its evidence.
Do not keep collecting unrelated evidence before interpreting this child's existing experiment.
If the analysis program failed, repair it from the feedback. finish may end the unresolved task.
"""
            elif request["research_focus"].get("kind") == "intervention":
                instruction_text += """\nThe host is continuing the parent investigation after repeated failures
following an already supported operational answer. Preserve research_focus.known_operation;
it answers how to use the object, not whether the current scientific intervention is correct.
research_focus.intervention_history includes successful interventions, readbacks and
contemporaneous public object descriptions as well as errors. Use these conditions and
physical states to investigate the parent_question, rather than retesting only whether
the already rejected command is supported. Distinguish three questions: was the operation
accepted, was the intended setting/condition established, and did the physical prediction
hold? An explicitly observed contrary state is not the same as an absent outcome.
Return write with arguments {path: research_focus.path, code: 'Python source'} defining:
select_actions(public_spec, observations, memory): return {hypothesis: your specific testable
claim, predicted_outcome: the public state it predicts, actions:[official actions, if needed]}.
answer(public_spec, observations, memory): return {status:'supported'|'refuted'|'inconclusive',
answer: your interpretation and remaining uncertainty, evidence:[actual observation indices]}.
The host executes these through the existing query and after_observation evaluate tools.
Use actions:[] when the paid observations already test the claim. Inspect observations and
actual public object states in answer, not just command success or the chosen setpoint.
Compare the predicted state with the observed state under the actually observed conditions.
If several explanations remain (for example a response model versus an unmet condition),
state their different predicted observations and the experiment that separates them.
Your answer should cite the actual physical-state and condition records it used; an
inconclusive result should identify what observation would resolve that uncertainty.
An unsupported command alone does not refute a scientific law. A failed tested setting does
not by itself refute its whole approximate relation family; identify the scope of your verdict.
Missing physical outcomes remain inconclusive. Existing evidence is retrospective analysis,
not new independent confirmation. All three valid verdicts return control to the parent;
repair a failed program using its feedback. The host supplies neither a formula nor an answer.
"""
            else:
                instruction_text += """\nThe host has paused the parent task after repeated unsupported operations.
research_focus contains the actual failures and the unresolved operational child question.
This decision designs an executable test, rather than repeating a log scan or the failed action.
Return write with arguments {path: research_focus.path, code: 'Python source'}.
Define TWO functions in that file:
select_actions(public_spec, observations, memory): return an official action list, or {actions:[...]}.
answer(public_spec, observations, memory): return {status:'supported'|'refuted'|'inconclusive',
 answer: an explanation of the operation and its conditions, evidence:[actual record indices]}.
The host immediately executes select_actions, buys its observations, then executes answer against
the new feedback. Your answer must inspect what actually happened, not assume the plan succeeded.
If research_focus.successful_interactions already resolves the question, select_actions may
return [] so answer reuses that actual paid evidence instead of buying a duplicate test.
Use the exact public capabilities, accessible objects, observed roles and dialog state to choose
an as-yet untested explanation of how to interact with the object. Missing access and unsupported
operations are different failures. A device may have a public agent role; use roles, not names,
to interpret applicable actions. You supply the hypothesis and experiment, not just a new label.
If the previous program failed, revise it using that feedback. Support returns control to the
parent task with this executable child answer. finish may end an unresolved task if appropriate.
Notebook and question metadata still follow the same output schema; no source files change until
this final write action is executed.
"""
        instruction_text += _CODEX_SOURCE_INSTRUCTIONS
        context = prepare_code_context(self.workspace, request, schema=action_schema,
                                       instructions=instruction_text)
        command = [self.executable, "-a", "never", "exec", "--sandbox", "read-only",
                   "--cd", str(self.workspace), "--skip-git-repo-check", "--ephemeral", "--json",
                   "--color", "never", "--output-schema", str(schema), "--output-last-message", str(response),
                   "-c", "model_instructions_file=" + json.dumps(str(instructions)),
                   "-c", "project_doc_max_bytes=0", "-c", 'developer_instructions=""',
                   "-c", 'web_search="disabled"', "-c", "mcp_servers={}", "-c", "notify=[]"]
        # The provider selects actions; the agent loop executes its ordinary
        # Python and apparatus calls and returns their actual outputs.
        for feature in ("shell_tool", "apps", "plugins", "memories", "multi_agent",
                        "browser_use", "computer_use", "image_generation",
                        "skill_search", "hooks", "view_image"):
            command.extend(("--disable", feature))
        # The installed CLI requires its host even when only returning a JSON action.
        command.extend(("--enable", "code_mode_host"))
        if self.model:
            command.extend(("--model", self.model))
        if self.reasoning_effort:
            command.extend(("-c", "model_reasoning_effort=" + json.dumps(self.reasoning_effort)))
        command.append("-")
        # Preserve the full research record for ordinary Python inspection;
        # repeated raw trajectories and old stdout exhausted real runs in 3 calls.
        started = monotonic()
        code, stdout, stderr, timed_out = _run(command, json.dumps(context), self.timeout_seconds, str(self.workspace))
        (self.workspace / f"code_call_{self.calls:03d}.jsonl").write_text(stdout, encoding="utf-8")
        events, _ = _events(stdout)
        usage = {"backend": "codex_cli", "runtime": self.executable, "calls": 1,
                 "input_tokens": None, "output_tokens": None, "total_tokens": None,
                 "wall_time_seconds": monotonic() - started, "model": self.model or "cli_default"}
        if self.reasoning_effort:
            usage["requested_reasoning_effort"] = self.reasoning_effort
        errors = []
        for event in events:
            if event.get("model"):
                usage["model"] = event["model"]
            if event.get("type") == "turn.completed":
                usage.update({name: event.get("usage", {}).get(name)
                              for name in ("input_tokens", "output_tokens", "cached_input_tokens")})
            if event.get("type") in {"error", "turn.failed"}:
                errors.append(event.get("message") or event.get("error"))
            item = event.get("item", {})
            if item.get("type") == "error" and event.get("type") == "item.completed":
                errors.append(item.get("message"))
        if errors:
            usage["runtime_errors"] = errors
        if isinstance(usage["input_tokens"], int) and isinstance(usage["output_tokens"], int):
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if timed_out or code or not response.exists():
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"returncode": code, "timeout": timed_out, "messages": errors,
                              "stderr": stderr[-4000:]}}
        try:
            payload = json.loads(response.read_text(encoding="utf-8"))
            # Actual CLI completions sometimes retain literal line breaks in
            # this nested string. Preserve the authored code for execution.
            arguments = json.loads(payload["arguments_json"], strict=False)
            source = payload.get("source_code")
            if source is not None:
                if not isinstance(source, str):
                    raise ValueError("source_code must be a string or null")
                if "code" in arguments and arguments["code"] != source:
                    raise ValueError("source_code conflicts with arguments_json.code")
                arguments["code"] = source
            if payload.get("research_question") is not None:
                arguments["research_question"] = payload["research_question"]
            if "research_notes" in request:
                arguments["notes"] = payload["research_notes"]
        except (ValueError, KeyError, TypeError) as error:
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"decode_error": f"{type(error).__name__}: {error}"}}
        return {"action": payload["action"], "arguments": arguments, "usage": usage}


_MODEL_RUNNER = """import contextlib, json, math, os, runpy, sys
from pathlib import Path
payload = json.load(sys.stdin)
require_fit = sys.argv[3] == '1'
dependencies = []
research_bindings = []
research_prediction_count = 0
research_compute_budget = json.loads(sys.argv[4])

class ResearchComputeBudgetExhausted(Exception):
    pass

def public_row(row):
    return {'features': row['features'], 'history': row.get('history', [])}

def represented(model):
    present = callable(model.get('predict_features'))
    if present and not callable(model.get('represent')):
        raise ValueError('an explicit representation needs represent and predict_features')
    return present

def represent(model, row):
    value = model['represent'](public_row(row))
    # This is also the value passed downstream: numeric JSON is the shared
    # authored representation contract, without current evaluation labels.
    return json.loads(json.dumps(value, allow_nan=False))

def predict_row(model, row):
    if represented(model):
        return model['predict_features'](represent(model, row))
    return model['predict'](row)

@contextlib.contextmanager
def working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)

def model_context(directory, model):
    model = Path(model)
    snapshots = [index for index, part in enumerate(model.parts) if part == '_evaluated_model']
    if snapshots:
        boundary = snapshots[-1] + 1
        directory = directory / Path(*model.parts[:boundary])
        model = Path(*model.parts[boundary:])
    return directory, str(model)

def load_model(directory, path, training=None):
    global research_prediction_count
    directory, path = model_context(directory, path)
    with working_directory(directory):
        model = runpy.run_path(str(directory / path))
        inputs_file = directory / 'model_research_inputs.json'
        inputs = json.loads(inputs_file.read_text()).get(path, {}) if inputs_file.exists() else {}
        if inputs:
            if not callable(model.get('bind_research')):
                raise ValueError('research_input_not_consumed: model needs bind_research(inputs)')
            model['bind_research'](inputs)
        bindings_file = directory / 'model_bindings.json'
        bindings = json.loads(bindings_file.read_text()).get(path, {}) if bindings_file.exists() else {}
        predictors = {}
        for slot, child_path in bindings.items():
            child, child_directory = load_model(directory, child_path)
            record = {'parent': path, 'slot': slot, 'model': child_path, 'calls': 0}
            dependencies.append(record)
            def predict_child(row, child=child, child_directory=child_directory, record=record):
                record['calls'] += 1
                with working_directory(child_directory):
                    return predict_row(child, row)
            predictors[slot] = predict_child
        if bindings:
            model['bind_models'](predictors)
        fitting = model.get('fit_features') if represented(model) else model.get('fit')
        if callable(fitting):
            fitting_path = training or directory / 'model_observations.json'
            if not fitting_path.exists():
                fitting_path = directory / 'observations.json'
            training_rows = json.loads(fitting_path.read_text())
            if represented(model):
                features = [represent(model, row) for row in training_rows]
                fitting(json.loads(json.dumps(features)), [row['target'] for row in training_rows])
                model['_training_representations'] = features
            else:
                fitting(training_rows)
        if inputs:
            regions = {slot: item for slot, item in inputs.items()
                       if item.get('kind') == 'parameter_region'}
            if regions and not callable(model.get('get_parameters')):
                raise ValueError('research_input_not_consumed: model needs get_parameters()')
            parameters = model['get_parameters']() if regions else {}
            for slot, item in regions.items():
                value = parameters.get(slot)
                region = item['region']
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('research_input_not_consumed: missing finite parameter for ' + slot)
                lower, upper = region['lower'], region['upper']
                inside = ((value >= lower if region.get('lower_closed', True) else value > lower)
                          and (value <= upper if region.get('upper_closed', True) else value < upper))
                if not inside:
                    raise ValueError('research_input_not_consumed: parameter outside bound region: ' + slot)
            relations = []
            research_bindings.append({'model': path, 'inputs': inputs,
                'actual_parameters': parameters, 'local_relations': relations})
            for slot, item in inputs.items():
                if item.get('kind') != 'local_relations':
                    continue
                checks = []
                relation = {'slot': slot, 'revision_index': item['revision_index'],
                    'version': item['version'], 'checks': checks}
                relations.append(relation)
                for check in item['checks']:
                    predicted = 0.
                    for index, witness in enumerate(check['witnesses']):
                        if (research_compute_budget is not None
                                and research_prediction_count >= research_compute_budget):
                            relation['pending_check'] = {'id': check['id'],
                                'completed_witnesses': index,
                                'total_witnesses': len(check['witnesses']),
                                'partial_predicted_value': predicted}
                            raise ResearchComputeBudgetExhausted()
                        research_prediction_count += 1
                        predicted += witness['coefficient'] * float(predict_row(model, public_row(witness)))
                    checks.append({'id': check['id'], 'scope': check.get('scope'),
                        'target': check['target'], 'measured_value': check['value'],
                        'predicted_value': predicted, 'residual': predicted - check['value'],
                        'evidence_status': check['status']})
    return model, directory

fit_available, predictions, representations, model = None, None, None, {}
truncated = False
try:
    with contextlib.redirect_stdout(sys.stderr):
        root = Path.cwd()
        model, directory = load_model(root, str(Path(sys.argv[1]).relative_to(root)), Path(sys.argv[2]))
        fit_available = callable(model.get('fit_features') if represented(model) else model.get('fit'))
        with working_directory(directory):
            representations = ([represent(model, row) for row in payload] if represented(model) else None)
            if not require_fit or fit_available:
                predictions = ([float(model['predict_features'](json.loads(json.dumps(features))))
                                for features in representations]
                               if representations is not None else [float(model['predict'](row)) for row in payload])
except ResearchComputeBudgetExhausted:
    truncated = True
print(json.dumps({'fit_available': fit_available, 'predictions': predictions,
                  'model_dependencies': dependencies, 'research_bindings': research_bindings,
                  'research_prediction_count': research_prediction_count,
                  'truncated': truncated,
                  'representations': representations,
                  'training_representations': model.get('_training_representations')}, allow_nan=False))
"""


def python_model_context(workspace, model):
    """Resolve a retained model's original working directory and fitting view."""
    workspace = Path(workspace).resolve()
    model = Path(model)
    snapshots = [index for index, part in enumerate(model.parts) if part == "_evaluated_model"]
    if snapshots:
        boundary = snapshots[-1] + 1
        workspace = workspace / Path(*model.parts[:boundary])
        model = Path(*model.parts[boundary:])
    return workspace.resolve(), str(model)


def evaluate_python_model(workspace, model, observations, *, timeout_seconds=30, risk_quantile=.5,
                          require_fit=False, training_observations=None,
                          research_compute_budget=None):
    """Run input rows, optionally bounding extra bound-research witness predictions.

    Candidate-row scoring is charged by the caller. The separate allowance
    bounds research-input validation, including nested bound models.
    """
    if research_compute_budget is not None and (
            type(research_compute_budget) is not int or research_compute_budget < 0):
        raise ValueError('research_compute_budget must be a nonnegative integer')
    parts = Path(model).parts
    shared_submission = (len(parts) > 2 and parts[0] == "_submitted_models"
        and (Path(workspace) / parts[0] / parts[1]).is_symlink())
    source, inner = python_model_context(workspace, model)
    shared_submission = shared_submission or any(
        path.parent.name == "_submitted_models" and path.is_symlink()
        for context, _ in _bound_model_contexts(source, inner)
        for path in (context, *context.parents))
    if shared_submission:
        # A restored historical submission stays read-only. Its authored fit
        # may write parameters, so execute only the selected context privately.
        with TemporaryDirectory(prefix="hegel-submitted-fit-") as temporary:
            copied = Path(temporary) / "snapshot"
            _copy_execution_context(source, copied, [inner])
            return evaluate_python_model(copied, inner, observations,
                timeout_seconds=timeout_seconds, risk_quantile=risk_quantile,
                require_fit=require_fit, training_observations=training_observations,
                research_compute_budget=research_compute_budget)
    workspace, model = python_model_context(workspace, model)
    payload = [{"features": row.features, "history": row.history} if hasattr(row, "features")
               else {"features": row["features"], "history": row.get("history", [])} for row in observations]
    # A retained candidate carries its selected paid training view to terminal
    # execution. Terminal prediction rows never replace these fitting rows.
    training = workspace / "model_observations.json"
    if not training.exists():
        training = workspace / "observations.json"
    with (TemporaryDirectory(prefix="hegel-target-fit-") if training_observations is not None
          else nullcontext(None)) as fitting_directory:
        if fitting_directory is not None:
            training = Path(fitting_directory) / "model_observations.json"
            training.write_text(json.dumps([asdict(row) if hasattr(row, "features") else row
                                            for row in training_observations]), encoding="utf-8")
        code, stdout, stderr, timed_out = _run([sys.executable, "-c", _MODEL_RUNNER, str(workspace / model), str(training),
                                             "1" if require_fit else "0", json.dumps(research_compute_budget)],
                                             json.dumps(payload), timeout_seconds, str(workspace))
        execution = json.loads(stdout) if code == 0 else None
        training_rows = (json.loads(training.read_text())
                         if execution and execution.get("training_representations") is not None else [])
    result = {"returncode": code, "stderr": stderr, "timeout": timed_out,
              "risk_quantile": risk_quantile, "loss_kind": "mse" if risk_quantile == .5 else "pinball"}
    if code == 0:
        predictions = execution["predictions"]
        result["model_dependencies"] = execution["model_dependencies"]
        result["research_bindings"] = execution.get("research_bindings", [])
        result["research_prediction_count"] = execution.get("research_prediction_count", 0)
        if execution.get('truncated'):
            result.update(status='scoped_unknown', truncated=True,
                          reason='research_compute_budget_exhausted')
            return result
        representations = execution.get("representations")
        result["representation_diagnostic"] = (representation_ambiguity(representations, observations)
            if representations is not None else {"status": "not_exposed"})
        training_representations = execution.get("training_representations")
        if training_representations is not None:
            result["training_representation_diagnostic"] = representation_ambiguity(
                training_representations, training_rows, scope="selected_fitting_rows")
        if require_fit:
            result["fit_available"] = execution["fit_available"]
            if predictions is None:
                return result
        if len(predictions) != len(observations):
            raise ValueError("model returned a different number of predictions")
        targets = [row.target if hasattr(row, "target") else row["target"] for row in observations]
        result.update(predictions=predictions, mse=sum((value - target) ** 2
                      for value, target in zip(predictions, targets)) / max(1, len(targets)))
        result["loss"] = sum(point_loss(value, target, risk_quantile)
                             for value, target in zip(predictions, targets)) / max(1, len(targets))
    return result


def _archived_paid_trace(path):
    path = Path(path)
    return path.name == "paid_trace.json" and path.parent.parent.name == "archive"


def archived_paid_trace_references(files, reference_archive, *, source_workspace=None):
    """Keep identical immutable source traces once in the existing reference archive."""
    reference_archive = Path(reference_archive).resolve()
    references, originals = {}, []
    source_traces = [(Path(source_workspace) / name, content)
                     for name, content in files.items() if _archived_paid_trace(name)] \
        if source_workspace is not None else []
    for name, content in files.items():
        if not _archived_paid_trace(name):
            continue
        original = next((path for text, path in originals if text == content), None)
        if original is None:
            original = reference_archive / name
            if original.is_file():
                if original.read_text(encoding="utf-8") != content:
                    raise ValueError(f"Reference archive has different paid evidence: {original}")
            else:
                original.parent.mkdir(parents=True, exist_ok=True)
                # Historical cleanup may remove a repeated nested path while
                # retaining the same trace at another declared archive path.
                source = next((path for path, text in source_traces if text == content
                               and path.is_file() and path.read_text(encoding="utf-8") == content), None)
                if source is not None:
                    original.symlink_to(source.resolve())
                else:
                    original.write_text(content, encoding="utf-8")
            originals.append((content, original))
        references[name] = original.resolve()
    return references


def _copy_model_file(source, destination):
    """Keep writable files independent, sharing disk extents where supported."""
    if _archived_paid_trace(source):
        # Prior source evidence is immutable. Fits may write model/configuration
        # files, so only the paid trace references its retained original.
        Path(destination).unlink(missing_ok=True)
        Path(destination).symlink_to(Path(source).resolve())
        return destination
    return clone_or_copy(source, destination)


def _model_context_ignored(path, context_root):
    if path.name == "__pycache__" or path.suffix == ".pyc":
        return True
    # These names identify host records only at an execution-context boundary.
    # Identically named files in ordinary model data directories remain inputs.
    return ((path.parent == context_root or path.parent.name == "_evaluated_model")
            and (path.name in {"code_agent_instructions.txt", "research_context.json",
                               "research_state.json", "method_decision.json", "active.json",
                               "training.json", "interrupted_training.json", "archive.json", "evaluation.json", "result.json",
                               "protocol.json", "summary.json"}
                 or (path.name.startswith("interrupted_training_") and path.suffix == ".json"
                     and path.stem.removeprefix("interrupted_training_").isdigit())
                 or path.name.startswith(("code_action_", "code_call_"))))


def _model_snapshot_ignored(path, context_root):
    if _model_context_ignored(path, context_root):
        return True
    parts = path.relative_to(context_root).parts
    # Other execution contexts are not inputs of this predictor. Keep inherited
    # archives; explicitly bound children are copied separately from their roots.
    return bool(parts and parts[0] != "archive" and "_evaluated_model" in parts)


def _bound_model_contexts(directory, model):
    bindings_file = directory / "model_bindings.json"
    bindings = json.loads(bindings_file.read_text()).get(model, {}) if bindings_file.exists() else {}
    for child_path in bindings.values():
        context, inner = python_model_context(directory, child_path)
        parts = Path(child_path).parts
        snapshots = [index for index, part in enumerate(parts) if part == "_evaluated_model"]
        if snapshots:
            # Preserve the declared placement while copying a linked historical
            # submission into a new writable execution context.
            context = Path(directory) / Path(*parts[:snapshots[-1] + 1])
        yield context, inner
        yield from _bound_model_contexts(context, inner)


def _copy_model_context(workspace, model, destination):
    source, model = python_model_context(workspace, model)
    _copy_execution_context(source, destination, [model])
    return model


def _copy_execution_context(source, destination, models):
    """Copy one execution root and its selected declared model dependencies."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        return

    def ignored(directory, names):
        return {name for name in names if _model_snapshot_ignored(Path(directory) / name, source)}

    # Archives can live inside the source workspace. Stage outside that source
    # before installing the copy, so it never recursively copies itself.
    with TemporaryDirectory(prefix="hegel-model-", dir=source.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        staged.mkdir()
        # Keep ordinary code/data, including inherited archives. Conversation
        # and research-state records describe the host loop, not the fitted
        # predictor, and can themselves contain every earlier comparison.
        for path in source.iterdir():
            if (path.name in {"_evaluated_model", "_target_work", "_experiment_models", "_submitted_models"}
                    or (path.name == "_observable_models" and destination.is_relative_to(path))
                    or _model_context_ignored(path, source)):
                continue
            target = staged / path.name
            if path.is_dir():
                try:
                    shutil.copytree(path, target, ignore=ignored, copy_function=_copy_model_file)
                except shutil.Error as error:
                    failures = error.args[0]
                    failed_source, failed_target, cause = failures[0]
                    raise OSError(f"Model context copy failed for {len(failures)} paths; "
                        f"first source={failed_source}; destination={failed_target}; cause={cause}") from None
            else:
                _copy_model_file(path, target)
        # Target revisions can call an earlier retained target. Copy only
        # their declared contexts when the enclosing archive was skipped.
        children = (child for model in models for child in _bound_model_contexts(source, model))
        for child_context, inner in sorted(children, key=lambda item: len(item[0].parts)):
            if child_context == source:
                continue
            target = staged / child_context.relative_to(source)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_model_context(child_context, inner, target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staged), str(destination))


def retain_evaluated_python_model(workspace, model, channels, training_mse, *,
                                  training_loss=None, loss_kind="mse", risk_quantile=.5,
                                  destination=None, training_observations=None):
    """Keep a successfully executed candidate while ordinary work continues."""
    retained = Path(workspace).resolve() / (destination or "_evaluated_model")
    source_model = str(model)
    model = _copy_model_context(workspace, model, retained)
    if training_observations is not None:
        training_text = json.dumps([asdict(row) if hasattr(row, "features") else row
                                    for row in training_observations])
        for filename in ("model_observations.json", "observations.json"):
            write_text_incrementally(retained / filename, training_text)
    return {"workspace": str(retained), "model": model, "source_model": source_model,
            "selected_channels": list(channels),
            "training_mse": training_mse, "model_source": (retained / model).read_text(),
            "training_loss": training_mse if training_loss is None else training_loss,
            "loss_kind": loss_kind, "risk_quantile": risk_quantile}


def archive_confirmed_python_model(agent, destination):
    """Archive the submitted predictor and fitting view, without host conversation state."""
    if not agent.model or not agent.confirmations or not agent.confirmations[-1].get("passed"):
        raise ValueError("archive a submitted model only after fresh confirmation passed")
    destination = Path(destination).resolve()
    model = _copy_model_context(agent.workspace, agent.model, destination)
    for filename in ("research_context.json", "research_state.json"):
        (destination / filename).unlink(missing_ok=True)
    training = destination / "model_observations.json"
    if not training.exists():
        training = destination / "observations.json"
    return {"snapshot_workspace": str(destination), "inner_model": model,
            "model": str(Path("_evaluated_model") / model), "source_model": agent.model,
            "training_rows": len(json.loads(training.read_text()))}


def _python_model_context_files(directory):
    directory = Path(directory)
    files = {}
    for root, directories, names in os.walk(directory):
        directories[:] = [name for name in directories
            if not _model_snapshot_ignored(Path(root) / name, directory)]
        for name in names:
            path = Path(root) / name
            if path.is_file() and not _model_snapshot_ignored(path, directory):
                files[path.relative_to(directory)] = path
    return files


def python_model_context_changes(workspace, model, snapshot_workspace, inner_model):
    """Keep transient candidate differences against an existing retained context.

    Rejected evaluations have no snapshot of their own. Store only their changed
    input bytes for the next procedure step, without duplicating the unchanged
    archive or writing another predictor context.
    """
    current, model = python_model_context(workspace, model)
    submitted = _python_model_context_files(current)
    original = _python_model_context_files(snapshot_workspace)
    changes = {path: None for path in original.keys() - submitted.keys()}
    for name, path in submitted.items():
        reference = original.get(name)
        if reference is not None and path.resolve() == reference.resolve():
            continue  # The immutable paid trace already shares its original.
        content = path.read_bytes()
        if reference is None or content != reference.read_bytes():
            changes[name] = content
    dependencies = {}
    for child_context, inner in _bound_model_contexts(current, model):
        if child_context != current and _model_snapshot_ignored(child_context, current):
            name = child_context.relative_to(current)
            dependencies[name] = python_model_context_changes(
                child_context, inner, Path(snapshot_workspace) / name, inner)
    return {"model": model, "retained_model": inner_model,
            "files": changes, "dependencies": dependencies}


def python_model_context_matches(workspace, model, snapshot_workspace, inner_model):
    """Decide whether a submitted archive still contains the original predictor context."""
    current, model = python_model_context(workspace, model)
    if model != inner_model:
        return False

    submitted = _python_model_context_files(current)
    original = _python_model_context_files(snapshot_workspace)
    if (submitted.keys() != original.keys()
            or any(submitted[path].read_bytes() != original[path].read_bytes() for path in original)):
        return False
    # Explicitly selected contexts are executable dependencies, even though
    # unrelated generated contexts were omitted from the enclosing snapshot.
    for child_context, inner in _bound_model_contexts(current, model):
        if child_context != current and _model_snapshot_ignored(child_context, current):
            retained_context = Path(snapshot_workspace) / child_context.relative_to(current)
            if not python_model_context_matches(child_context, inner, retained_context, inner):
                return False
    return True


class CodeAgent:
    """Execute the model's experimental and Python decisions with real feedback."""

    def __init__(self, provider, workspace, config=None, *, state=None, session_knowledge=None):
        self.provider, self.workspace = provider, Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config or CodeAgentConfig()
        self.observations, self.events, self.files = [], [], {}
        self.transition_batches = []
        self.costs = {"observation_cost": 0, "measurement_cost": 0, "compute_cost": 0,
                      "python_execution_seconds": 0., "model_calls": 0, "reported_tokens": 0}
        self.tokens_available = True
        self.model = None
        self.selected_channels = []
        self.session_knowledge = deepcopy(session_knowledge or {})
        self.lesson, self.reused_files = "", []
        self.evaluated_model = None
        if state is not None:
            self.observations = [Observation(dict(row["features"]), row["target"],
                tuple(dict(item) for item in row.get("history", []))) for row in state["observations"]]
            self.events, self.files, self.costs = deepcopy(state["events"]), deepcopy(state["files"]), dict(state["costs"])
            self.tokens_available = state["tokens_available"]
            self.model, self.selected_channels = state["model"], list(state["selected_channels"])
            self.lesson, self.reused_files = state.get("lesson", ""), state.get("reused_files", [])
            self.evaluated_model = deepcopy(state.get("evaluated_model"))
            self.transition_batches = deepcopy(state.get("transition_batches", []))
            for name, source in self.files.items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            (self.workspace / "observations.json").write_text(json.dumps([asdict(row) for row in self.observations]))
            if self.transition_batches:
                (self.workspace / "transitions.json").write_text(json.dumps(self.transition_batches), encoding="utf-8")
        else:
            for name, source in self.session_knowledge.get("files", {}).items():
                path = self.workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
                self.files[name] = source

    def to_dict(self):
        names = set(self.files)
        names.update(str(path.relative_to(self.workspace)) for path in (self.workspace / "shared").rglob("*") if path.is_file())
        if self.model:
            names.add(self.model)
        for name in names:
            path = self.workspace / name
            if path.is_file():
                try:
                    self.files[name] = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass  # Binary fit artifacts stay in this persistent task workspace.
        return {"observations": [asdict(row) for row in self.observations],
                "transition_batches": deepcopy(self.transition_batches),
                "events": deepcopy(self.events), "files": dict(self.files), "costs": dict(self.costs),
                "tokens_available": self.tokens_available, "model": self.model,
                "selected_channels": list(self.selected_channels), "lesson": self.lesson,
                "reused_files": list(self.reused_files), "evaluated_model": deepcopy(self.evaluated_model)}

    def _execute(self, action, arguments, laboratory):
        if action == "query":
            probe = Probe(f"code_query_{len(self.events)}", tuple(arguments["sequence"]))
            channels = tuple(arguments.get("channels", ()))
            rates = {channel.name: channel.cost_per_row for channel in getattr(laboratory, "channels", ())}
            units = probe.cost * sum(rates[name] for name in set(channels))
            if self.costs["observation_cost"] + probe.cost > self.config.observation_budget:
                return {"error": "observation budget exhausted"}
            if self.costs["measurement_cost"] + units > self.config.measurement_budget:
                return {"error": "measurement budget exhausted"}
            transitions = []
            if hasattr(laboratory, "acquire"):
                batch = laboratory.acquire(probe, channels)
                rows, units = batch.observations, batch.measurement_cost
                transitions = getattr(batch, "transitions", [])
            else:
                rows = laboratory.observe(probe)
            if transitions:
                self.transition_batches.append({"channels": list(channels),
                    "observation_start": len(self.observations), "transitions": deepcopy(transitions)})
                (self.workspace / "transitions.json").write_text(json.dumps(self.transition_batches), encoding="utf-8")
            self.observations.extend(rows)
            self.selected_channels = list(channels)
            self.costs["observation_cost"] += len(rows)
            self.costs["measurement_cost"] += units
            (self.workspace / "observations.json").write_text(json.dumps([asdict(row) for row in self.observations]), encoding="utf-8")
            self._write_model_observations(laboratory, channels)
            feedback = {"observations_file": "observations.json", "model_observations_file": "model_observations.json",
                        "observation_cost": len(rows), "measurement_cost": units}
            if transitions:
                feedback.update(transitions_file="transitions.json", transition_count=len(transitions),
                    transitions_layout="batches: channels, observation_start, transitions[{before, after}]")
            if hasattr(laboratory, "measurement_views"):
                feedback.update(channels=list(channels),
                    feature_names=list(laboratory.spec_for(channels).feature_names))
            return feedback
        if action == "write":
            path = self.workspace / arguments["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["code"], encoding="utf-8")
            self.files[arguments["path"]] = arguments["code"]
            return {"written": arguments["path"]}
        remaining = self.config.execution_seconds - self.costs["python_execution_seconds"]
        if remaining <= 0:
            return {"error": "Python execution time budget exhausted"}
        if action == "run":
            channels = tuple(arguments.get("channels", self.selected_channels))
            rows = self._write_model_observations(laboratory, channels)
            path = arguments.get("path", f"scratch_{len(self.events):03d}.py")
            if "code" in arguments:
                (self.workspace / path).write_text(arguments["code"], encoding="utf-8")
                self.files[path] = arguments["code"]
            started = monotonic()
            code, stdout, stderr, timed_out = _run([sys.executable, str(self.workspace / path)], "", remaining, str(self.workspace))
            self.costs["python_execution_seconds"] += monotonic() - started
            return {"returncode": code, "stdout": stdout[-12000:], "stderr": stderr[-12000:], "timeout": timed_out,
                    "model_observations_file": "model_observations.json", "model_observation_count": len(rows),
                    "channels": list(channels)}
        if action in {"evaluate", "finish"}:
            channels = tuple(arguments.get("channels", self.selected_channels))
            observations = self._write_model_observations(laboratory, channels)
            if not observations:
                return {"error": "acquire observations before fitting a model"}
            if self.costs["compute_cost"] + len(observations) > self.config.compute_budget:
                return {"error": "candidate-row scoring budget exhausted"}
            model = arguments.get("model", "model.py")
            self.costs["compute_cost"] += len(observations)
            started = monotonic()
            result = evaluate_python_model(self.workspace, model, observations,
                timeout_seconds=remaining, risk_quantile=laboratory.spec.risk_quantile)
            self.costs["compute_cost"] += result.get("research_prediction_count", 0)
            self.costs["python_execution_seconds"] += monotonic() - started
            result.pop("predictions", None)
            if action == "evaluate" and result["returncode"] == 0:
                self.evaluated_model = retain_evaluated_python_model(self.workspace, model,
                    arguments.get("channels", self.selected_channels), result["mse"],
                    training_loss=result["loss"], loss_kind=result["loss_kind"],
                    risk_quantile=result["risk_quantile"])
            if action == "finish" and result["returncode"] == 0:
                self.model = model
                self.selected_channels = list(arguments.get("channels", self.selected_channels))
                self.lesson = arguments.get("lesson", "")
                self.reused_files = list(arguments.get("reused_files", []))
            return result
        return {"error": f"unknown tool: {action}"}

    def _write_model_observations(self, laboratory, channels):
        observations = self._model_observations(laboratory, channels)
        (self.workspace / "model_observations.json").write_text(
            json.dumps([asdict(row) for row in observations]), encoding="utf-8")
        return observations

    def _model_observations(self, laboratory, channels):
        if not hasattr(laboratory, "measurement_views"):
            if not hasattr(laboratory, "spec_for"):
                return self.observations
            names = set(laboratory.spec_for(channels).feature_names)
            return [Observation({name: row.features[name] for name in names}, row.target,
                        tuple({name: value for name, value in point.items()
                               if name in names or name == "__target__"} for point in row.history))
                    for row in self.observations if names <= set(row.features)]
        laboratory.spec_for(channels)
        views = [(view, set(laboratory.spec_for(view).feature_names))
                 for view in laboratory.measurement_views]
        observations = []
        for row in self.observations:
            source = next(view for view, names in views if set(row.features) == names)
            if channels in laboratory.projectable_views(source):
                observations.extend(laboratory.project_rows([row], source, channels))
        return observations

    def run(self, laboratory, *, on_checkpoint=None):
        spec = laboratory.spec
        capabilities = {"description": spec.description, "feature_names": spec.feature_names,
                        "feature_bounds": spec.feature_bounds, "max_probe_length": spec.max_probe_length,
                        "risk_quantile": spec.risk_quantile,
                        "loss_kind": "mse" if spec.risk_quantile == .5 else "pinball",
                        "channels": [asdict(channel) for channel in getattr(laboratory, "channels", ())],
                        "python_libraries": [name for name in ("numpy", "scipy") if importlib.util.find_spec(name)]}
        if hasattr(laboratory, "measurement_views"):
            capabilities["measurement_views"] = [{"channels": list(view),
                "feature_names": list(laboratory.spec_for(view).feature_names),
                "context": dict(laboratory.spec_for(view).context),
                "projectable_to": [list(target) for target in laboratory.projectable_views(view)]}
                for view in laboratory.measurement_views]
        status = "completed" if self.model else "step_budget_exhausted"
        for step in range(len(self.events), self.config.max_steps):
            if self.model:
                break
            if self.costs["model_calls"] >= self.config.max_model_calls:
                status = "model_call_budget_exhausted"
                break
            if not self.tokens_available or self.costs["reported_tokens"] >= self.config.token_budget:
                status = "token_usage_unavailable" if not self.tokens_available else "token_budget_exhausted"
                break
            request = {"capabilities": capabilities, "limits": asdict(self.config), "costs": dict(self.costs),
                       "observations": [{"features": row.features, "target": row.target,
                                          "trajectory_start": not row.history} for row in self.observations],
                       "files": dict(self.files),
                       "session_knowledge": self.session_knowledge,
                       "events": self.events}
            self.costs["model_calls"] += 1
            try:
                decision = self.provider.next_action(request)
            except Exception as error:
                self.tokens_available = False
                self.events.append({"action": "provider_error", "error": f"{type(error).__name__}: {error}"})
                status = "provider_failed"
                break
            usage = decision.get("usage", {})
            tokens = usage.get("total_tokens")
            if isinstance(tokens, int):
                self.costs["reported_tokens"] += tokens
            else:
                self.tokens_available = False
            action = decision["action"]
            if action == "provider_error":
                if decision.get("error", {}).get("decode_error"):
                    # A completed model reply with malformed tool arguments
                    # remains a paid turn. Let the model repair its own JSON
                    # within the same task's remaining call/token budget.
                    self.events.append({"step": step, "action": action, "arguments": {},
                        "feedback": {"error": decision["error"]["decode_error"]}, "usage": usage})
                    if on_checkpoint:
                        on_checkpoint(self.to_dict())
                    continue
                self.events.append(decision)
                status = "provider_failed"
                break
            try:
                feedback = self._execute(action, decision.get("arguments", {}), laboratory)
            except Exception as error:
                feedback = {"error": f"{type(error).__name__}: {error}"}
            self.events.append({"step": step, "action": action, "arguments": decision.get("arguments", {}),
                                "feedback": feedback, "usage": usage})
            if on_checkpoint:
                on_checkpoint(self.to_dict())
            if self.model is not None:
                status = "completed"
                break
        usable = ({"model": self.model, "workspace": str(self.workspace),
                   "model_source": (self.workspace / self.model).read_text(), "selected_channels": self.selected_channels}
                  if self.model else self.evaluated_model or {})
        return {"agent": "ordinary_python_code_agent", "status": status, "model": usable.get("model"),
                "risk_quantile": spec.risk_quantile,
                "loss_kind": "mse" if spec.risk_quantile == .5 else "pinball",
                "model_source": usable.get("model_source"),
                "model_origin": "finish" if self.model else "last_successful_evaluation" if self.evaluated_model else None,
                "selected_channels": usable.get("selected_channels", self.selected_channels),
                "workspace": usable.get("workspace", str(self.workspace)), "research_workspace": str(self.workspace),
                "lesson": self.lesson, "reused_files": self.reused_files,
                "costs": {**self.costs, "reported_tokens": self.costs["reported_tokens"] if self.tokens_available else None},
                "token_limit_kind": "reported_usage_between_calls", "config": asdict(self.config),
                "token_budget_exceeded": (self.costs["reported_tokens"] > self.config.token_budget
                                           if self.tokens_available else None),
                "events": self.events}
