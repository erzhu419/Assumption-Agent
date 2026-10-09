"""Task-local model research using the existing world tools and model account."""

from copy import deepcopy


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
whether to execute a proposed experiment. It can compute, write and repair, but leaves world
experiments and scientific verdicts to the caller. A missing observation can be
returned as a deferred action with its reason. The question is only registered
when the caller adopts a returned comparison or experiment; drafting creates no
unresolved dependency and does not replace an existing binding or answer.
Use a separate draft path to retain earlier programs and their actual results.
Reusing that path resumes its proposal notebook. No draft is a scientific answer.
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
frozen candidates on eligible new evidence; editing a draft does not replace that
candidate set. Keep a new candidate family in a distinct relation question so the
parent retains the earlier explanation and its evidence for comparison.
Use actual observed responses as targets, including failed outcomes when observed;
a forecast or a requested setting is not a measured response.
Explain the observed successes and failures, identify unresolved alternatives,
and derive their observable predictions. Use computations to check whether the
claimed predictions actually follow from those alternatives. Choose the scientific
content yourself; an executable file alone does not establish an explanation.

This phase provides write {path,code}, ordinary run {code} or run {path} for local
Python calculations and debugging, and finish {lesson} to return to the caller.
Read paid observations and retained sources using these tools. Their file paths
are in evidence and files. Save a complete research_notes notebook each turn.
Write the deliverable to a separate draft; retain the earlier scientific programs.
If you need a new observation, describe what it would distinguish and return the
suggested action. World queries and research operators are deferred to the caller.
The existing execution_interface describes the downstream bind, design/answer and
procedure contracts for your draft. A bind draft returns comparison_action
(research_update, no acquisition) and research_action (the existing acquire/update
loop); a design/answer draft returns contrast_action. These actions use the declared
question and run only when the caller selects them. The caller can inspect the paid
comparison's candidate errors, diagnosis and next_action before buying evidence.
Fitting paid data is not fresh confirmation; existing confirmation rules still apply.
Scientific questions and their verdicts remain unchanged while you propose.
finish returns your file and notebook without running an experiment or ending
the episode. The caller decides whether to adopt, revise or reject your proposal.
"""


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
    if frame.get("stage") == "proposal":
        request.pop("research_root", None)
        request.pop("research_focus", None)
        request["execution_interface"] = request["task_interface"]
        request["task_interface"] = PROPOSAL_INTERFACE
        request["enclosing_research_notes"] = frame["parent_notes"]
        request["llm_inquiry"].update(stage="proposal", proposal_path=frame["proposal_path"])
        return request
    request["task_interface"] += """
You are the local researcher for llm_inquiry.assignment, within the same current world.
That assignment retains the caller's purpose and deliverable. llm_inquiry.question
tracks your current hypothesis or experiment and can change as evidence arrives.
Work toward the assignment through actual execution, using your local notebook and the
returned errors. Repair failed code and rerun it; distinguish an execution failure
from a refuted prediction. You may run calculations before choosing an experiment.
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
    session["invocations"] += 1
    start_calls = agent.costs["model_calls"]
    frame = {"question_id": question["id"], "question": question,
             "assignment": deepcopy(question),
             "call_limit": min(start_calls + calls, agent.limits["max_model_calls"],
                               *(item["call_limit"] for item in agent._inquiry_stack)),
             "stop_requested": False, "delegated_children": set()}
    if proposal:
        frame.update(stage="proposal", proposal_path=path, deferred_actions=[])
        frame["call_limit"] = min(frame["call_limit"], agent.limits["max_model_calls"] - 1)
        if agent._inquiry_stack:
            frame["call_limit"] = min(frame["call_limit"],
                                      agent._inquiry_stack[-1]["call_limit"] - 1)
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
                agent._save_state()
                if agent._on_checkpoint:
                    agent._on_checkpoint(agent.result())
                continue
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
                try:
                    feedback = agent._execute(action, params, laboratory)
                except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
                    feedback = {"error": f"{type(error).__name__}: {error}"}
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
            declaration = {key: deepcopy(question[key])
                           for key in ("id", "question", "parent_id", "depends_on")}
            action = agent._design_action(path, question)
            if action:
                action["arguments"]["question"] = deepcopy(declaration)
                # Adoption explicitly chooses the new draft's answer too;
                # the old experiment and any separate analysis stay on disk.
                action["arguments"]["question"]["analysis_path"] = path
            bound = "bind" in agent._program_functions(path)
            session["proposal"] = {"path": path, "contrast_action": action,
                                   "comparison_action": ({"action": "run", "arguments": {
                                       "operator": "research_update", "path": path,
                                       "question": deepcopy(declaration)}} if bound else None),
                                   "research_action": ({"action": "run", "arguments": {
                                       "operator": "research", "path": path,
                                       "question": deepcopy(declaration)}} if bound else None),
                                   "deferred_actions": frame["deferred_actions"]}
        agent._save_state()
    return {"operator": "propose_explanations" if proposal else "inquire",
            "status": status, "question": deepcopy(question),
            "model_calls": agent.costs["model_calls"] - start_calls,
            "steps": steps, "notes": session["notes"],
            **({"proposal": deepcopy(session["proposal"])} if proposal else {})}
