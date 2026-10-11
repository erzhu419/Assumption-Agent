"""Task-local model research using the existing world tools and model account."""

from copy import deepcopy
import json
from time import monotonic


INQUIRY_INTERFACE = """
run {operator:'inquire',question:{id:'q',question:'...',parent_id:'task',depends_on:[]},
     max_model_calls:8} delegates a question to a continuing local LLM work loop.
The researcher can write/run Python, buy observations, analyze, repair and retry,
or delegate another question through the same inquire tool. Every model call and
world action uses the same episode account. A program can return this run action.
Each question retains its own notebook and actual execution feedback across calls.
The question may include parent_program_path:'shared/procedure.py' to bind an
existing research procedure. Program-authored delegations retain their caller path
automatically. method_decision.questions exposes its procedure_action and latest
execution, when available. Binding a program neither runs it nor supports its claims.
The same question ID resumes that research. finish inside this loop returns to
the caller; it does not finish the episode or establish a scientific answer.
Only the existing evaluate/measurement contracts can support or refute a question.
Use this when a procedure needs reasoning or repair instead of another fixed step.
In the Hegel execution policy, an actual procedure error or repeated public
navigation/dialogue cycle opens this local researcher automatically. The failed
source and trace are retained. A tested child answer, or a locally returned code
revision, lets the parent execute again in the same world and invocation phase.
Code revisions alone grant no scientific support. Within a local researcher,
execution errors return to that same researcher for its next decision.
Inside a Hegel inquiry, a declared unresolved child with no acquired observations
or numerical binding enters this same local researcher before the parent's next
decision. The host schedules the declared question without adding scientific content.
Each child is automatically visited at most once per parent invocation; an unresolved
return remains available for the parent to revise or explicitly revisit. Existing
paid observations can answer it without buying new data. Child calls share the same
account, retaining at least one remaining decision for the parent.

run {operator:'propose_explanations',question:{id:'q',question:'...',parent_id:'task'},
     path:'shared/new_design.py',max_model_calls:8} uses this same LLM account to
form explanations from paid evidence before choosing another world action. It
returns a draft design/answer program and its contrast_action for the caller to
inspect, revise or execute. A numerical draft can instead define bind, returning
actual-evidence rows, executable Program hypotheses and prospective probes under
the existing research contract. Its comparison_action runs research_update on
paid data; its research_action starts the existing research loop. Compare the
paid rows first, inspect contradictions and remaining alternatives, then choose
whether to execute a proposed experiment. It can compute, write and repair; the
condition-test exception below also permits its selected physical trial. It can
delegate a distinct prerequisite through inquire and inspect its actual return
before continuing the same draft. The child shares the account and retains one
decision for its author. Other direct world experiments and scientific verdicts
remain with the caller. A missing observation can be
returned as a deferred action with its reason. The question is only registered
when the caller adopts a returned comparison or experiment; drafting creates no
unresolved dependency and does not replace an existing binding or answer.
Use a separate draft path to retain earlier programs and their actual results.
Reusing that path resumes its proposal notebook. No draft is a scientific answer.
For computed calibration representations, add response_path:'shared/consolidated_method.py'
to propose_explanations. This selects the existing Boolean response contract:
bind returns rows with crystal/features/evidence and estimators over those features.
Writing the requested draft immediately compares all paid positive and negative
responses and returns response_comparison to this same author for revision. It
buys no observation and does not adopt the draft into the parent's questions.
The returned comparison_action uses reference_response, not numerical research_update.
Returned experiment_action retains response_comparison linking the physical trial
to this binding and author. The caller or this associated author can execute that
action with its association intact. Author practice does not adopt the draft or
establish its answer. After
the selected point is observed, its new responses update the same comparison and
author notebook. revision_action resumes that author; a failed point alone does
not refute the family. An explicit proposal resume refreshes its saved comparison
before the author's next decision, then lets the author keep, revise or reject it.
When shared-premise research is needed, premise_design_action opens the existing
ordinary design/answer proposal at a separate path. Its contrast_action can test
an authored premise beyond the fixed connected-band comparison. The caller chooses
whether to run it; no scientific prediction or experimental setting is supplied.

condition_research contains finite executable Boolean rules screened on paid
activation outcomes, with frozen predictions in selected_experiment. Their status
unconfirmed describes what the evidence establishes. A provisional hypothesis can
justify a test of its distinguishing prediction before that hypothesis is confirmed;
the candidate set need not exhaust all possible laws. Assess the actual forecasts,
their applicability and how either outcome changes the remaining task decisions.
The returned experiment_action uses reference_response and the existing physical
controller. Preserve condition_proposal to associate its real outcomes with the
same proposal. The parent or its condition author may select this action; frozen
rules are then rechecked, contradictions withdrawn and current results returned
before the next decision. Surviving compatibility alone is still unconfirmed.
target_consequences projects possible target settings under the retained rules
and how each predicted experiment outcome changes that set. Use those conditional
consequences to compare a discriminating experiment, a direct operational trial,
further explanation work or stopping. Uncertainty is a reason to test a useful
prediction; a claimed law requires subsequent evidence. A successful operational
trial answers the task's intervention without establishing the whole explanation.
An empty target support means this finite pool offers no activating setting;
compatible rules may be retained while broader hypotheses or measurements are
investigated. An unresolved projection leaves that support unknown.
A condition research_obligation exposes the existing source revision_action and
a propose_explanations research_action that resumes that executable author for
the current extension assignment. Actual changes to measured
or computed coordinates admit new rules alongside the retained frozen rules;
repeating the same inputs reuses the existing result.
The response binding can also return candidate_additions: explicit Boolean Program
rules for activation, including their applicability conditions. These enter this
same condition bank, independently of the estimator's shared-band premise. Each
rule is screened on the paid outcomes, retained with its original coordinates,
and compared at prospective settings by the existing experiment selector.
"""


PROPOSAL_INTERFACE = """You are proposing explanations for llm_inquiry.assignment
from the existing paid evidence. Your deliverable is a Python draft at
llm_inquiry.proposal_path, with bind for the caller's existing numerical researcher,
design and answer for its contrast tool, or an explanation of the evidence still
needed. For a numerical or Boolean response, bind returns measured rows with
actual observation citations, hypotheses as executable Program ASTs, and probes
whose features and actions describe possible new measurements. The same Program
must predict across the declared rows and probe conditions. For a new relation
question, the existing researcher compares these predictions with all returned
paid rows before proposing an experiment. Resuming an existing relation tests its
frozen candidates on eligible new evidence. To compare a revision in that same
question, explicitly return candidate_additions containing new Program ASTs;
editing hypotheses alone does not replace the retained candidates. The researcher
screens additions on paid rows while retaining old programs and their results.
Rows may distinguish conditioning_refs (reused measured properties) from
outcome_refs (actual response), whose union is evidence. A later response can
test a frozen candidate using earlier conditioning measurements; a response
already public when that candidate was admitted is not fresh confirmation.
Use actual observed responses as targets, including failed outcomes when observed;
a forecast or a requested setting is not a measured response.
Explain the observed successes and failures, identify unresolved alternatives,
and derive their observable predictions. Use computations to check whether the
claimed predictions actually follow from those alternatives. Choose the scientific
content yourself; an executable file alone does not establish an explanation.
A new hypothesis may be a provisional assumption rather than a conclusion already
established by the observations. Specify its quantitative rule and applicability
conditions, check its consequences against the paid counterexamples, and derive
prospective predictions from that rule. An unspecified label such as a missing
feature is a research question; refine it into a testable conditional model when
possible. Keep its proposed assumptions separate from measured facts and support.
For a design/answer draft, every write runs design and the existing experiment
chooser on paid evidence. design_comparison returns its actual result or execution
error to your next turn. Use it to repair or revise the same draft before returning.
This calculation buys no observations and grants no scientific support. A broad
alternative that permits both outcomes does not justify a Boolean forecast: state
an explicit conditional hypothesis and compute its consequence, or report what
remains unspecified. A missing draft is NO_EXECUTABLE_EXPLANATION_PROPOSED.
An empty experiment list is no_experiments_proposed: no candidate predictions were
compared. observational_ambiguity means agreement on the authored experiments only,
not that all reachable experiments fail to distinguish the hypotheses.
For a response-premise proposal, the shared condition_research contract in
execution_interface also applies here. You may select its reference_response
experiment with this proposal's condition_proposal association to test your rules
and receive actual outcomes before your next decision.

This phase provides write {path,code}, ordinary run {code} or run {path} for local
Python calculations and debugging, and finish {lesson} to return to the caller.
Read paid observations and retained sources using these tools. Their file paths
are in evidence and files. Save a complete research_notes notebook each turn.
Write the deliverable to a separate draft; retain the earlier scientific programs.
For a prerequisite needing its own computation or observation, run inquire with
a question ID distinct from the active questions. That researcher uses the existing
tools and returns to this same draft; its finish does not end your assignment.
Its default parent is this proposal session, leaving earlier scientific answers
intact. Child calls share the account and retain one decision for you to inspect
their return. Otherwise describe the needed observation and return its suggested
action. Other direct world queries and research operators are deferred to the caller.
The existing execution_interface describes the downstream bind, design/answer and
procedure contracts for your draft. A bind draft returns comparison_action
(research_update, no acquisition) and research_action (the existing acquire/update
loop); a design/answer draft returns contrast_action. These actions use the declared
question and run only when the caller selects them. The caller can inspect the paid
comparison's candidate errors, diagnosis and next_action before buying evidence.
Fitting paid data is not fresh confirmation; existing confirmation rules still apply.
Drafting leaves scientific verdicts unchanged. A condition experiment records its
own observed point response; it does not confirm the enclosing explanation.
finish returns your file, notebook and any test results without acquiring further
evidence or ending the episode. The caller decides whether to adopt, revise or
reject your proposal.
"""


RESPONSE_PROPOSAL_INTERFACE = """You are proposing a computed representation for
llm_inquiry.assignment using the paid calibration and Boolean response evidence.
Write a complete Python draft at llm_inquiry.proposal_path, defining
bind(public_spec, observations, memory). Return:
{rows:[{crystal:PUBLIC_NUMBER,features:{YOUR_FEATURE:COMPUTED_VALUE},
        evidence:[PAID_INPUT_INDICES]}],
 estimators:[{id:'your_family',basis:[Program AST expressions over YOUR_FEATURES]}]}.
Return every crystal in the current calibration table, including targets. The
latest raw measurements are in shared/reference_recalibration.json. Transform
measured inputs with ordinary Python; the declared bases operate on your computed
features. Choose the representation and scientific hypothesis yourself. Successful
control settings are Boolean responses, not exact labels for a latent function.

To propose a response law beyond that estimator premise, also return
candidate_additions:[{id:'your_rule',claim:'your quantitative hypothesis',
                     program:BOOLEAN_PROGRAM_AST}]. The rule predicts activation
for every supplied crystal and setting. Express applicability in its Boolean
branches; use ge/lt/and/or/not with the existing arithmetic Program operations.
Inputs are raw measured feature names, bound_YOUR_FEATURE for computed features,
and frequency for the controlled setting. Compute constants or transformations
from the paid data in Python; the executable rule must determine its predictions.
These additions are compared in condition_research using the actual Boolean
outcomes, separately from response_comparison's connected-band estimator checks.
Retain estimators for that existing comparison; their premise does not restrict
the added Boolean rules. An added rule faces all paid counterexamples, can supply
a distinguishing experiment, and is frozen for later observations. Editing or
removing an addition cannot erase its previous version or its refutations.
Rules are provisional, so they can motivate an experiment before confirmation.

Each write to this draft runs the existing shared connected-band comparison
against all paid positive and negative responses. The next feedback contains
response_comparison: family status, exact conflicts or remaining uncertainty,
and an experiment_action when available. Use that actual result to revise the
representation or premise, or explain precisely what remains unresolved. Do not
replace a computationally unattempted comparison with a prose assertion.
SAT_WITNESS means compatibility, not confirmation; UNKNOWN is unfinished work;
UNSAT_VERIFIED excludes only that basis plus the declared response premise.

Use write and ordinary run to calculate, inspect paid data, repair and revise.
For a prerequisite needing its own computation or observation, run inquire with
a question ID distinct from the active questions. Its researcher uses the existing
tools and returns to this same draft, reserving one author decision in the shared
account. New measurements or draft changes refresh your comparison and condition
bank before your next decision. Its default parent is this proposal session;
earlier scientific answers remain intact. You may execute this draft's returned
reference_response experiment_action with its response_comparison association
intact. The physical result returns to this author and refreshes its paid comparison;
this practice does not adopt the draft or support its answer. The existing condition
experiment exception also remains available. Other direct operators and world
actions are deferred to the caller. Save research_notes.
When resuming this draft, the latest response_return event contains its comparison
against the current paid responses and any newly measured responses. Evaluate
what changed before revising; a single failed point does not refute a still
compatible family. You may keep its structure, revise its parameters or representation,
or state what evidence would resolve the remaining uncertainty.
This bind comparison assumes a shared connected response band; changing features
does not change that assumption. A returned premise_design_action offers a separate
ordinary experiment draft if that premise needs research. You may return that action
to the caller through the existing deferred-action mechanism, retaining this binding.
When the assignment extends a finite condition pool, writes also return
condition_research on the revised computed coordinates, alongside frozen old rules.
Construct provisional quantitative hypotheses before they are confirmed; compare
their consequences with the paid responses. Choose the formulas yourself. If a new
observation is necessary, return its concrete action and the alternatives it tests.
The associated condition_proposal also permits its reference_response experiment
through the existing controller, returning the actual outcome before the next turn.
That outcome also refreshes this source response_comparison and its condition rules
on the new evidence. Inspect the returned comparison in this same author session.
finish returns the latest draft, its actual comparison and notebook; the parent
chooses adoption and any physical experiment. Draft comparisons do not overwrite
its supported answers. Old drafts and feedback remain in the session events.
"""


DESIGN_SCAFFOLD = '''def design(public_spec, observations, memory):
    # Derive hypotheses and their predictions from the paid evidence.
    # Return {"hypotheses": [{"id": ..., "claim": ...}], "experiments": [
    # {"id": ..., "outcome": ..., "predictions": {hypothesis_id: bool, ...},
    #  "procedure": {"path": existing_controller, "parameters": {...}, "max_steps": ...}}]}.
    # An experiment may instead provide "actions": [official_actions].
    raise NotImplementedError("Supply the scientific hypotheses and computed forecasts")

def answer(public_spec, observations, memory):
    # Interpret actual observations after the selected experiment, with citations.
    # Return {"status": "supported"/"refuted"/"inconclusive",
    #         "answer": your_analysis, "evidence": [actual_observation_indices]}.
    raise NotImplementedError("Supply the physical-outcome analysis")
'''


def _proposal_summary(agent, frame):
    path, question = frame["proposal_path"], frame["question"]
    declaration = {key: deepcopy(question[key])
                   for key in ("id", "question", "parent_id", "depends_on")}
    action = agent._design_action(path, question)
    design_comparison = frame.get("design_comparison")
    if design_comparison and design_comparison["diagnosis"] in {
            "design_error", "no_experiments_proposed"}:
        action = None
    if action:
        action["arguments"]["question"] = deepcopy(declaration)
        # Adoption explicitly chooses the new draft's answer too;
        # the old experiment and any separate analysis stay on disk.
        action["arguments"]["question"]["analysis_path"] = path
    bound = "bind" in agent._program_functions(path)
    draft = agent.workspace / path
    result = {"path": path, "contrast_action": action,
                           "draft_changed": frame["initial_draft"] != (
                               draft.read_text() if draft.exists() else None),
                           "author_work_attempted": frame["author_work_attempted"],
                           "comparison_action": ({"action": "run", "arguments": {
                               "operator": "research_update", "path": path,
                               "question": deepcopy(declaration)}} if bound else None),
                           "research_action": ({"action": "run", "arguments": {
                               "operator": "research", "path": path,
                               "question": deepcopy(declaration)}} if bound else None),
                           "deferred_actions": frame["deferred_actions"]}
    if not frame.get("response_path"):
        result.update(
            design_comparison=deepcopy(design_comparison),
            deliverable_status=(design_comparison["diagnosis"] if design_comparison and
                design_comparison["diagnosis"] in {"design_error", "no_experiments_proposed"} else
                "design_computed" if action and design_comparison else
                "binding_proposed" if bound else "NO_EXECUTABLE_EXPLANATION_PROPOSED"))
        conditions = agent.llm_inquiries['proposal:' + path].get('condition_research')
        if conditions is not None:
            _attach_condition_result(result, conditions)
    return result


def refresh_authored_response(agent, laboratory, frame):
    """Return a changed draft or child evidence through the existing comparison."""
    from .world_reference_response import compare_response_proposal, return_response_to_author
    previous = frame.pop('response_comparison', {})
    compared = compare_response_proposal(agent, laboratory, {
        'path': frame['response_path'], 'binding_path': frame['proposal_path'],
        'question': frame['question']}, previous)
    frame['response_comparison'] = compared
    feedback = {'response_comparison': deepcopy(compared)}
    if compared.get('candidate_additions') and not frame.get('condition_proposal'):
        frame['condition_proposal'] = frame['proposal_path']
    if frame.get('condition_proposal'):
        # Publish the revised source before its retained bank consumes the inputs.
        return_response_to_author(agent, compared)
        feedback['condition_research'] = refresh_condition_proposal(
            agent, frame['condition_proposal'])
    return feedback


def _attach_condition_result(proposal, conditions):
    """An empty authored draft does not erase a computed native experiment."""
    proposal.setdefault('authored_deliverable_status', proposal.get('deliverable_status'))
    proposal['condition_research'] = {key: deepcopy(value) for key, value in conditions.items()
                                      if key != 'inputs'}
    action = conditions.get('experiment_action')
    if action:
        proposal.update(deliverable_status='condition_experiment_available',
                        experiment_action=deepcopy(action))
    else:
        proposal['deliverable_status'] = proposal['authored_deliverable_status']
        proposal.pop('experiment_action', None)


def refresh_condition_proposal(agent, path, packet=None):
    """Adapt paid Boolean responses to the retained finite condition synthesizer."""
    from .world_reference_response import response_premise_proposal_inputs
    from .world_response_conditions import synthesize_response_conditions

    session = agent.llm_inquiries['proposal:' + path]
    packet = response_premise_proposal_inputs(agent, path) if packet is None else packet
    if packet is None:
        return None
    started = monotonic()
    try:
        result = synthesize_response_conditions(packet, session.get('condition_research'))
    finally:
        agent.costs['python_execution_seconds'] += monotonic() - started
    agent.costs['research_compute_units'] += result['compute_cost']
    action = result.get('experiment_action')
    if action is not None:
        action['arguments']['condition_proposal'] = path
    # A task goal is not a positive training label or a refutation of compatible
    # rules. It can instead motivate a broader research question.
    result.pop('research_obligation', None)
    result.pop('research_action', None)
    empty_targets = [target['number'] for target in
        result.get('target_consequences', {}).get('targets', [])
        if target['number'] in packet.get('unfinished_targets', [])
        and target['status'] == 'complete' and target['support']['grid_count'] == 0]
    if empty_targets:
        obligation = {'status': 'condition_language_extension_needed',
            'targets': empty_targets, 'revision_action': deepcopy(packet['revision_action'])}
        result['research_obligation'] = obligation
        # Resume the author that actually computes this representation. Its
        # latest revision_action may still name an old physical trial; keep the
        # executable binding but give it the current research assignment.
        result['research_action'] = deepcopy(packet['revision_action'])
        result['research_action']['arguments'].update(condition_proposal=path,
            question={'id': session['question_id'] + '_target_extension',
                'parent_id': 'task', 'depends_on': [],
                'question': 'The current finite response rules remain compatible with the paid '
                    'observations but predict no activating setting on the legal grid for unfinished '
                    'targets ' + str(empty_targets) + '. This limits the current hypothesis pool; '
                    'it does not establish physical impossibility. Extend the executable representation '
                    'at ' + packet['binding_path'] + ' using the paid measurements. Construct and '
                    'compute a quantitative proposal, compare it with the paid counterexamples, '
                    'and inspect the returned target predictions. You can propose explicit Boolean '
                    'candidate_additions to revise the response law or its applicability independently '
                    'of the estimator comparison, which keeps its shared-band premise. The previous source and comparisons '
                    'are in this author session; compatible old condition rules retain their original '
                    'coordinates and evidence. The current condition pool is attached to ' + path + '. '
                    'Choose the scientific content yourself. If the shared response premise or an '
                    'unmeasured distinction prevents this calculation, return a concrete research '
                    'action for that prerequisite, with the competing predictions it would test. '
                    'Return the computed proposal or the specific unresolved prerequisite to the parent.'})
    session['condition_research'] = result
    view = {key: deepcopy(value) for key, value in result.items() if key != 'inputs'}
    _attach_condition_result(session.setdefault('proposal', {}), result)
    source = agent.llm_inquiries.get('proposal:' + packet['binding_path'])
    if source is not None:
        source_proposal = source.setdefault('proposal', {})
        source_proposal['condition_proposal'] = path
        # The source author and its premise researcher expose the same current
        # bank. An old returned proposal must not keep recommending a refuted rule.
        if source is not session:
            _attach_condition_result(source_proposal, result)
    # Like _research_update, publish the result of the current research in the
    # existing advice channel. A superseded numerical fit is still retained in
    # relation_research; its old verdict is not advice for these new rules.
    agent.research_advice = {
        'question_id': session['question_id'], 'proposal_path': path,
        'diagnosis': result['status'], 'status': result['confirmation_status'],
        'compatible_candidates': sum(item['status'] == 'compatible' for item in result['candidates']),
        'refuted_candidates': sum(item['status'] == 'refuted' for item in result['candidates']),
    }
    if action is not None:
        agent.research_advice['next_action'] = deepcopy(action)
    elif result.get('research_action'):
        agent.research_advice.update(diagnosis='condition_language_extension_needed',
            next_action=deepcopy(result['research_action']))
    if result.get('research_obligation'):
        agent.research_advice['research_obligation'] = deepcopy(result['research_obligation'])
    if result.get('target_consequences'):
        agent.research_advice['target_consequences'] = deepcopy(result['target_consequences'])
    return view


def preview_design_proposal(agent, frame):
    """Feed the existing chooser back to its author without adopting a question."""
    from .world_experiment_choice import select_experiment

    frame.pop("design_comparison", None)
    if "design" not in agent._program_functions(frame["proposal_path"]):
        agent.llm_inquiries["proposal:" + frame["proposal_path"]]["proposal"] = _proposal_summary(agent, frame)
        return None
    try:
        result = select_experiment(agent._program(frame["proposal_path"], "design"))
        agent.costs["research_compute_units"] += result["compute_cost"]
    except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
        result = {"diagnosis": "design_error", "error": f"{type(error).__name__}: {error}"}
    frame["design_comparison"] = result
    # The author reads both event feedback and the current proposal view.
    # Publish one current result before its next decision, retaining old events.
    agent.llm_inquiries["proposal:" + frame["proposal_path"]]["proposal"] = _proposal_summary(agent, frame)
    return deepcopy(result)


def inquiry_request(agent, session, frame):
    request = agent._request()
    request["enclosing_research_root"] = request.get("research_root")
    request["research_root"] = {key: deepcopy(frame["assignment"][key])
                                for key in ("id", "question", "parent_id")}
    request["task_interface"] = request["task_interface"].replace(
        "Solve the task in current_observation.taskProgress by experimenting in the world.",
        "Resolve llm_inquiry.assignment and return its result to the caller. "
        "current_observation.taskProgress describes the enclosing task.")
    request["research_notes"] = session["notes"]
    request["events"] = deepcopy(session["events"][-8:])
    request["llm_inquiry"] = {
        "question_id": frame["question_id"], "parent_id": frame["question"].get("parent_id"),
        "depth": len(agent._inquiry_stack), "question": deepcopy(frame["question"]),
        "assignment": deepcopy(frame["assignment"]),
        "remaining_model_calls": frame["call_limit"] - agent.costs["model_calls"],
        "invocation": session["invocations"],
    }
    if frame.get('condition_proposal'):
        request['llm_inquiry']['condition_proposal'] = frame['condition_proposal']
    if frame.get("stage") == "proposal":
        request.pop("research_root", None)
        request.pop("research_focus", None)
        request["execution_interface"] = request["task_interface"]
        request["task_interface"] = (RESPONSE_PROPOSAL_INTERFACE if frame.get("response_path")
                                     else PROPOSAL_INTERFACE)
        request["enclosing_research_notes"] = frame["parent_notes"]
        request["llm_inquiry"].update(stage="proposal", proposal_path=frame["proposal_path"])
        if frame.get("response_path"):
            request["llm_inquiry"]["response_path"] = frame["response_path"]
        else:
            from .world_reference_response import response_premise_proposal_inputs
            request["llm_inquiry"]["draft_scaffold"] = DESIGN_SCAFFOLD
            inputs = response_premise_proposal_inputs(agent, frame["proposal_path"])
            if inputs:
                request["llm_inquiry"]["proposal_inputs"] = inputs
        return request
    request["task_interface"] += """
You are the local researcher for llm_inquiry.assignment, within the same current world.
That assignment retains the caller's purpose and deliverable. llm_inquiry.question
tracks your current hypothesis or experiment and can change as evidence arrives.
Work toward the assignment through actual execution, using your local notebook and the
returned errors. Repair failed code and rerun it; distinguish an execution failure
from a refuted prediction. You may run calculations before choosing an experiment.
Distinguish missing observations from calculations not yet attempted. Transforming
already measured numbers is a computation, not a new instrument reading. If a
narrow declaration tool cannot express your proposed calculation, use ordinary
run {code:'Python source'} on the paid values and inspect the actual result before
returning that calculation as unavailable. Your notebook records past attempts;
the current tool interface determines what you can execute now.
research_focus, when present, schedules evidence analysis or procedure review only
within this inquiry and its descendants. Write/evaluate an evidence-based answer
when ready. A supported or refuted answer returns automatically when the current
question still addresses the assignment and its dependencies are resolved. An
answer to a retitled, narrower question stays local; continue toward the original
assignment or finish with what is missing;
this only returns the local work, not a claim that the overall task is complete.
Prior-world experience is a candidate explanation, not a current measurement.
"""
    if frame["assignment"].get("parent_program_path"):
        request["task_interface"] += """
llm_inquiry.assignment.parent_program_path identifies the retained research procedure.
Its existing procedure_action and last_program_result are in method_decision.questions
under llm_inquiry.question_id. Use its actual execution feedback to decide whether
to inspect, continue or revise it. A retained stopped procedure is available evidence,
not a successful answer. You choose any revision and execute it through the same tools.
"""
    if frame["question"].get("failed_program_path"):
        path = frame["question"]["failed_program_path"]
        request["procedure_repair"] = {
            "path": path, "question_id": frame["question_id"],
            "failure_evidence_file": frame["question"]["failure_evidence_file"],
            "latest_execution": next((deepcopy(row) for row in reversed(agent.method_results)
                                      if row["path"] == path), None)}
        request["task_interface"] += """
The local deliverable is a repaired procedure at procedure_repair.path and its
actual execution result. Direct interactions can diagnose a prerequisite, but
incorporate the resulting repair into that procedure and execute it before
claiming the procedure works. Use the existing evaluate contract to analyze this
local question, then let the parent resume the remaining task. Scientific task
completion is the parent's goal; resolve the local execution prerequisite first.
"""
    return request


def _unmeasured_child(agent, frame):
    if agent.agent_arm != "hegel" or frame.get("stage") == "proposal":
        return None
    pending = set(agent._pending_dependencies(frame["question"]))
    active = {item["question_id"] for item in agent._inquiry_stack}
    return next((question for question in agent.questions
                 if question.get("parent_id") == frame["question_id"]
                 and question["id"] in pending
                 and question["id"] not in active | frame["delegated_children"]
                 and question["status"] in {"open", "reopened"}
                 and not question.get("binding_path")
                 and not question["acquired_observations"]), None)


def _executable_parent_trial(response, hypothesis_id):
    return (response.get("status") == "family_research_needed"
            and (response.get("operational_fallback") or {}).get("selected_experiment")
            and not any(
                family.get("status") == "UNSAT_VERIFIED"
                and hypothesis_id in family.get("source_ids", [family.get("id")])
                for family in response.get("family_research", {}).get("families", [])))


def _resume_deferred_parent_trial(agent, laboratory, frame):
    pending = frame.get("pending_parent_trial")
    if pending is None:
        return False
    # A child analysis can discharge the blocker without changing the acquired
    # experiment. A changed method or unreviewed evidence instead belongs to the
    # author's next decision, not this earlier comparison's handoff.
    path = pending["path"]
    if (frame["question"].get("parent_program_path") != path
            or (agent.workspace / path).read_text() != pending["source"]):
        frame.pop("pending_parent_trial")
        return False
    state = json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())
    response = state.get("response_research", {})
    if (len(agent.observations) != pending["observation_count"]
            or response.get("question_id") != pending["response_question_id"]):
        from .world_reference_response import _reference_response_inputs, _reference_state
        estimators, state, _ = _reference_state(agent, laboratory, path)
        if state.get("response_research_inputs") != _reference_response_inputs(state, estimators):
            frame.pop("pending_parent_trial")
            return False
        response = state.get("response_research", {})
    if not _executable_parent_trial(response, pending["hypothesis_id"]):
        frame.pop("pending_parent_trial")
        return False
    if agent._pending_dependencies(frame["question"]):
        return False
    frame.pop("pending_parent_trial")
    return True


def execute_llm_inquiry(agent, laboratory, arguments):
    calls = arguments.get("max_model_calls", 8)
    if type(calls) is not int or calls < 1:
        raise ValueError("max_model_calls must be a positive integer")
    if agent.provider is None:
        raise ValueError("inquire needs the live model provider, not a frozen zero-LLM replay")
    value = deepcopy(arguments["question"])
    if agent._inquiry_stack:
        value.setdefault("parent_id", agent._inquiry_stack[-1]["question_id"])
    proposal = arguments.get("operator") == "propose_explanations"
    if proposal:
        path = arguments["path"]
        # A writing assignment is not an unresolved scientific child. Keep
        # even existing supported questions intact until the caller adopts it.
        existing = next((q for q in agent.questions if q["id"] == value["id"]), {})
        question = {"parent_id": None, "depends_on": [], "status": "open",
                    **deepcopy(existing), **value}
        session_id = "proposal:" + path
    else:
        question = agent._question(value)
        session_id = question["id"]
    # Explicit model delegations count as visits too. A local finish should not
    # immediately trigger the same child again in an enclosing invocation.
    if not proposal:
        for parent in agent._inquiry_stack:
            parent["delegated_children"].add(question["id"])
    session = agent.llm_inquiries.setdefault(session_id, {
        "question_id": question["id"], "notes": "", "events": [],
        "invocations": 0, "model_calls": 0, "status": "open",
    })
    # Proposal sessions follow a draft path across revised assignments. Keep
    # their current advice and new trials attached to this caller; old event
    # question IDs and earlier scientific answers retain their original scope.
    session['question_id'] = question['id']
    session["invocations"] += 1
    start_calls = agent.costs["model_calls"]
    frame = {"question_id": question["id"], "question": question,
             "assignment": deepcopy(question),
             "call_limit": min(start_calls + calls, agent.limits["max_model_calls"],
                               *(item["call_limit"] for item in agent._inquiry_stack)),
             "stop_requested": False, "delegated_children": set()}
    if "assignment" in arguments:
        # Resuming the caller's research need must not overwrite the author's
        # current scientific hypothesis or its already acquired answer.
        frame["assignment"]["question"] = arguments["assignment"]
    if proposal:
        draft = agent.workspace / path
        frame.update(stage="proposal", proposal_path=path, deferred_actions=[],
                     initial_draft=draft.read_text() if draft.exists() else None,
                     author_work_attempted=False)
        if arguments.get("response_path"):
            frame["response_path"] = arguments["response_path"]
        frame["call_limit"] = min(frame["call_limit"], agent.limits["max_model_calls"] - 1)
        if agent._inquiry_stack:
            frame["call_limit"] = min(frame["call_limit"],
                                      agent._inquiry_stack[-1]["call_limit"] - 1)
    if arguments.get('condition_proposal'):
        frame['condition_proposal'] = arguments['condition_proposal']
    elif proposal and frame.get('response_path') and session.get('proposal', {}).get('condition_proposal'):
        frame['condition_proposal'] = session['proposal']['condition_proposal']
    elif proposal and frame.get('response_path') and session.get('condition_research'):
        frame['condition_proposal'] = path
    notes_path = agent.workspace / "research_notes.md"
    parent_notes = notes_path.read_text() if notes_path.exists() else None
    frame["parent_notes"] = parent_notes
    notes_path.write_text(session["notes"])
    parent_focus = agent.research_focus
    parent_proposal = agent._current_proposal
    agent.research_focus = None
    agent._inquiry_stack.append(frame)
    steps, status = [], "running"
    try:
        previous = session.get("proposal", {}).get("comparison_result")
        if proposal and frame.get("response_path") and previous is not None:
            from .world_reference_response import compare_response_proposal, return_response_to_author
            # Restore this author's previous comparison and refresh it on actual
            # evidence before asking for another decision; no world action.
            frame["response_comparison"] = deepcopy(previous)
            try:
                current = compare_response_proposal(agent, laboratory, {
                    "path": frame["response_path"], "binding_path": path, "question": question}, previous)
                frame["response_comparison"] = current
                return_response_to_author(agent, current)
                if current.get('candidate_additions') and not frame.get('condition_proposal'):
                    frame['condition_proposal'] = path
                if frame.get('condition_proposal'):
                    conditions = refresh_condition_proposal(agent, frame['condition_proposal'])
                    session['events'].append({'action': 'condition_return', 'arguments': {},
                        'feedback': {'condition_research': conditions}})
            except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
                frame.pop("response_comparison", None)
                session["events"].append({"action": "response_return", "arguments": {},
                    "feedback": {"error": f"{type(error).__name__}: {error}"}})
        elif proposal and not frame.get("response_path"):
            compared = preview_design_proposal(agent, frame)
            if compared is not None:
                session["events"].append({"action": "design_return", "arguments": {},
                                         "feedback": {"design_comparison": compared}})
            conditions = refresh_condition_proposal(agent, path)
            if conditions is not None:
                session['events'].append({'action': 'condition_synthesis', 'arguments': {},
                                         'feedback': {'condition_research': conditions}})
        while True:
            terminal = agent._procedure_terminal_status()
            if terminal:
                status = terminal
                break
            same_assignment = all(question.get(key) == frame["assignment"].get(key)
                                  for key in ("question", "parent_id", "depends_on"))
            if (not proposal and question["status"] in {"supported", "refuted"} and same_assignment
                    and not agent._pending_dependencies(question)):
                status = "answered"
                break
            exhausted = agent._procedure_budget_status()
            if exhausted:
                status = exhausted
                break
            if agent.costs["model_calls"] >= frame["call_limit"]:
                status = "model_call_budget_exhausted"
                break
            remaining_calls = frame["call_limit"] - agent.costs["model_calls"]
            child = _unmeasured_child(agent, frame) if remaining_calls > 1 else None
            if child is not None:
                frame["delegated_children"].add(child["id"])
                params = {"operator": "inquire", "question": {key: deepcopy(child[key])
                          for key in ("id", "question", "parent_id", "depends_on")},
                          "max_model_calls": min(8, remaining_calls - 1)}
                outcome = execute_llm_inquiry(agent, laboratory, params)
                # Child model events already carry their real usage. This is a
                # host scheduling event, not another paid model decision, and
                # the complete child trace stays in its existing session.
                event = {"action": "run", "arguments": params,
                         "trigger_origin": "host_pending_child_research",
                         "inquiry_question_id": question["id"],
                         "feedback": {key: deepcopy(outcome[key]) for key in (
                             "operator", "status", "question", "model_calls", "notes")}}
                session["events"].append(deepcopy(event))
                steps.append(deepcopy(event))
                agent.events.append(event)
                if _resume_deferred_parent_trial(agent, laboratory, frame):
                    frame["stop_requested"] = True
                agent._save_state()
                if agent._on_checkpoint:
                    agent._on_checkpoint(agent.result())
                continue
            # A new author decision replaces a still-blocked proposal's pending
            # handoff. Only its own measured proposal can establish another one.
            frame.pop("pending_parent_trial", None)
            decision = agent._model_decision(inquiry_request(agent, session, frame))
            call_index = agent.costs["model_calls"]
            action, params = decision["action"], deepcopy(decision.get("arguments", {}))
            if "notes" in params:
                session["notes"] = params["notes"]
                notes_path.write_text(session["notes"])
            fatal = False
            if action == "provider_error":
                error = decision.get("error", {})
                feedback = {"error": error.get("decode_error", error) if isinstance(error, dict) else error}
                fatal = not (isinstance(error, dict) and error.get("decode_error"))
            else:
                if proposal and action != 'finish':
                    frame['author_work_attempted'] = True
                try:
                    feedback = agent._execute(action, params, laboratory)
                except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
                    feedback = {"error": f"{type(error).__name__}: {error}"}
            if (not proposal and action == "run"
                    and params.get("operator") == "reference_hypothesis"
                    and params.get("path") == frame["assignment"].get("parent_program_path")
                    and feedback.get("status") == "method_stopped"
                    and feedback.get("new_observation_count", 0) > 0
                    and _executable_parent_trial(feedback.get("response_research", {}),
                                                 feedback.get("hypothesis", {}).get("id"))):
                # The proposed hypothesis has reached an actual experiment and
                # returned an executable parent continuation. Yield this local
                # invocation; the unresolved scientific question stays open.
                # A verified exclusion of this author's family instead stays
                # with the author. An unrelated old fallback is not an extension.
                if not agent._pending_dependencies(question):
                    frame["stop_requested"] = True
                else:
                    frame["pending_parent_trial"] = {
                        "path": params["path"],
                        "hypothesis_id": feedback.get("hypothesis", {}).get("id"),
                        "response_question_id": feedback["response_research"].get("question_id"),
                        "observation_count": len(agent.observations),
                        "source": (agent.workspace / params["path"]).read_text(),
                    }
            if notes_path.exists():
                session["notes"] = notes_path.read_text()
            event = {"action": action, "arguments": params,
                     "feedback": agent._compact_feedback(feedback),
                     "usage": deepcopy(decision.get("usage", {})),
                     "model_call": call_index, "inquiry_question_id": question["id"]}
            session["events"].append(deepcopy(event))
            steps.append(deepcopy(event))
            agent.events.append(event)
            session["model_calls"] += 1
            agent._save_state()
            if agent._on_checkpoint:
                agent._on_checkpoint(agent.result())
            if fatal:
                status = agent.status = "provider_failed"
                break
    finally:
        agent._inquiry_stack.pop()
        agent.research_focus = parent_focus
        agent._current_proposal = parent_proposal
        if parent_notes is None:
            notes_path.unlink(missing_ok=True)
        else:
            notes_path.write_text(parent_notes)
        session["status"] = status
        if proposal:
            session["proposal"] = _proposal_summary(agent, frame)
            declaration = {key: deepcopy(question[key])
                           for key in ("id", "question", "parent_id", "depends_on")}
            bound = "bind" in agent._program_functions(path)
            if frame.get("response_path"):
                from .world_reference_response import response_premise_design_action
                comparison = frame.get("response_comparison")
                session["proposal"].update(
                    comparison_action=({"action": "run", "arguments": {
                        "operator": "reference_response", "path": frame["response_path"],
                        "binding_path": path, "question": deepcopy(declaration)}} if bound else None),
                    research_action=None,
                    comparison_result=deepcopy(comparison),
                    experiment_action=deepcopy(comparison.get("experiment_action")) if comparison else None,
                    revision_action=deepcopy(comparison.get("revision_action")) if comparison else None,
                    premise_design_action=response_premise_design_action(comparison) if comparison else None)
                condition_path = frame.get('condition_proposal')
                if condition_path:
                    session['proposal']['condition_proposal'] = condition_path
                    conditions = agent.llm_inquiries['proposal:' + condition_path].get('condition_research')
                    if conditions is not None:
                        _attach_condition_result(session['proposal'], conditions)
        agent._save_state()
    return {"operator": "propose_explanations" if proposal else "inquire",
            "status": status, "question": deepcopy(question),
            "model_calls": agent.costs["model_calls"] - start_calls,
            "steps": steps, "notes": session["notes"],
            **({"proposal": deepcopy(session["proposal"])} if proposal else {})}
