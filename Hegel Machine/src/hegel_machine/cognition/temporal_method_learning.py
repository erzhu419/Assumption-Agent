"""LLM-authored research-method consolidation from completed paid experience."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .code_provider_context import _compact_trajectory_payload
from .online_inquiry_llm import InquiryLLMAgent, _PLAN_RUNNER
from .temporal_operator_agent import OPERATOR_INTERFACE
from .temporal_research import observation


METHOD_PATH = "shared/consolidated_method.py"
CONSOLIDATION_INTERFACE = """
Author a reusable research procedure from this actual completed research, including
its unsuccessful attempts. Paid response and meter records are in observations.json
and laboratory_observations.json; experience.json contains the preceding actions
and their feedback. Existing hypotheses, instrument files and procedures are frozen.
Choose the strategy yourself. No acquisition schedule or repair pipeline is supplied.
Available actions in THIS consolidation phase are only:
write {path:'shared/consolidated_method.py',code:'ordinary Python source'};
run {code:'ordinary Python source'} or run {path:'shared/consolidated_method.py'}
to inspect supplied evidence and check your code; and
finish {path:'shared/consolidated_method.py',entry:{id:'your_method_id',...},lesson:'...'}.
Do not fit or evaluate new predictors, change existing files, or request new data.
Write a self-contained select_actions(public_spec,observations,memory) function.
It will later receive the new task's paid records and public interface.
memory is the inherited snapshot; observations contains the current paid records.
Operator-updated instrument files on disk reflect their latest fitted or assessed
state. A continuation is linear; branching on its results requires a later
invocation of your procedure. Return
{'actions':[public trajectory actions], 'after_observation':[operator arguments],
 'knowledge_sources':[ids]} when useful. An actual acquisition must be nonempty.
Only your declared downstream operators run, in order after successful acquisition.
They use paid evidence; candidate selection and submission remain decisions for
the consuming agent. Learn applicable conditions
and useful steps from the supplied experience, and leave unresolved choices explicit.
No hegel_machine imports or access to other workspaces, artifacts or terminal data.
finish publishes only this procedure and its description. Publication calls
select_actions with the actual public_spec, empty current observations, and inherited
memory, and requires a nonempty action list. It does not acquire the proposed actions,
fit anything, or imply predictive confirmation or transfer benefit.

DOWNSTREAM INTERFACE REFERENCE: The following operators and query syntax are for
the later consuming research agent and your returned procedure, not for actions
available during this consolidation phase.
""" + OPERATOR_INTERFACE + """
END DOWNSTREAM REFERENCE. During consolidation, use only write, ordinary run,
and the method-publication finish described above. No query, evaluate, or fitting
operator is available in this phase.
"""


def load_training_experience(source):
    """Select completed Hegel target training; discard any terminal evaluation."""
    payload = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
    if "tasks" in payload:
        eligible = [task for task in payload["tasks"] if task.get("agent_arm") == "hegel"
                    and task.get("status") == "completed" and task.get("task_position", 0) > 0]
        if not eligible:
            raise ValueError("no completed Hegel target training experience")
        payload = eligible[-1]
    if (payload.get("agent_arm") != "hegel" or payload.get("status") != "completed"
            or payload.get("task_position", 0) <= 0):
        raise ValueError("method learning requires completed Hegel target training")
    keys = ("agent_arm", "status", "task_position", "paid_observations", "events", "outgoing_knowledge")
    return {key: deepcopy(payload[key]) for key in keys}


class _ExperienceLab:
    task_interface = "Consolidate a research method from supplied paid experience. No new laboratory queries are available.\n"

    def __init__(self, spec):
        self.spec = deepcopy(spec)

    def public_spec(self):
        return deepcopy(self.spec)


class MethodConsolidationAgent(InquiryLLMAgent):
    common_interface = CONSOLIDATION_INTERFACE

    def __init__(self, provider, workspace, *, experience, public_spec, config=None):
        experience = load_training_experience(experience)
        knowledge = experience["outgoing_knowledge"]
        if METHOD_PATH in knowledge.get("files", {}):
            raise ValueError("the selected consolidation path already exists in inherited memory")
        super().__init__(provider, workspace, arm="hegel", knowledge=knowledge, config=config,
                         source_task=False)
        self.method_entry, self.method_source = None, None
        self.laboratory = _ExperienceLab(public_spec)
        self.observations = deepcopy(experience["paid_observations"])
        self.rows = [observation(row) for record in self.observations for row in record["rows"]]
        self._write_evidence()
        trace = [{"action": event.get("action"),
                  "arguments": {key: value for key, value in event.get("arguments", {}).items() if key != "code"},
                  "feedback": _compact_trajectory_payload(event.get("feedback", {}))}
                 for event in experience["events"]]
        provenance = {"task_position": experience["task_position"], "paid_records": len(self.observations),
                      "paid_rows": len(self.rows), "events": trace}
        source = json.dumps(provenance)
        (self.workspace / "experience.json").write_text(source, encoding="utf-8")
        self.files["experience.json"] = source

    def run(self, *, on_checkpoint=None):
        return super().run(self.laboratory, on_checkpoint=on_checkpoint)

    def _execute(self, action, arguments, laboratory=None):
        if action == "write":
            if arguments.get("path") != METHOD_PATH:
                raise ValueError("only the new consolidated method may be written")
            return super()._execute(action, arguments, self.laboratory)
        if action == "run":
            if "operator" in arguments:
                raise ValueError("consolidation has no fitting or research operators")
            return super()._execute(action, arguments, self.laboratory)
        if action != "finish":
            raise ValueError("consolidation permits only write, ordinary run, and finish")
        path, entry = arguments.get("path"), deepcopy(arguments.get("entry", {}))
        if path != METHOD_PATH or not isinstance(entry, dict) or not entry.get("id"):
            raise ValueError("finish needs the new method path and an entry with an id")
        if any(item.get("id") == entry["id"] for item in self.knowledge.get("manifest", {}).get("research_programs", [])):
            raise ValueError("the consolidated method needs a new research-program id")
        state = {"operator_results": [], "limits": asdict(self.config), "costs": self.costs}
        (self.workspace / "research_state.json").write_text(json.dumps(state), encoding="utf-8")
        code, stdout, stderr, timed_out = self._python(
            [sys.executable, "-c", _PLAN_RUNNER, str(self.workspace / path)],
            json.dumps({"spec": self.laboratory.public_spec(), "observations": [],
                        "memory": self.knowledge}))
        if code:
            return {"error": "method publication check failed", "returncode": code,
                    "stdout": stdout[-2000:], "stderr": stderr[-4000:], "timeout": timed_out}
        proposed = json.loads(stdout)
        actions = proposed.get("actions", []) if isinstance(proposed, dict) else proposed
        if not isinstance(actions, list) or not actions:
            return {"error": "select_actions must return a nonempty action list for the actual public_spec "
                             "and a new task with observations=[]; no actions were acquired"}
        for name, source in self.knowledge.get("files", {}).items():
            if (self.workspace / name).read_text(encoding="utf-8") != source:
                raise ValueError(f"existing knowledge was modified during consolidation: {name}")
        self.method_source = (self.workspace / path).read_text(encoding="utf-8")
        self.method_entry = {**entry, "path": path}
        self.model, self.lesson = path, arguments.get("lesson", "")
        return {"published": path, "entry": deepcopy(self.method_entry)}

    def result(self, status):
        return {"agent_arm": "method_consolidation", "status": status, "workspace": str(self.workspace),
                "method_path": self.model, "method_source": self.method_source,
                "method_entry": deepcopy(self.method_entry), "lesson": self.lesson,
                "costs": deepcopy(self.costs), "events": deepcopy(self.events), "limits": asdict(self.config)}


def register_published_method(original_knowledge, result, *, max_steps=None):
    """Bind an actual publication to its executable archive and source support.

    A practiced author may replace an inherited placeholder or earlier draft at
    the same path. This registers that publication; it does not run or select it.
    """
    if result.get("status") != "completed" or not result.get("method_source") or not result.get("method_entry"):
        raise ValueError("a published consolidated method is required; no fallback is generated")
    path = result["method_path"]
    retained = deepcopy(original_knowledge)
    retained.setdefault("lessons", []).append(result.get("lesson", ""))
    retained.setdefault("files", {})[path] = result["method_source"]
    retained.setdefault("research_roles", {})["method"] = path
    entry = {**deepcopy(result["method_entry"]), "path": path}
    if max_steps is not None:
        entry["max_steps"] = max_steps
    # Coverage belongs to this exact source, not every draft the author tried.
    latest = {trial["case"]: trial for trial in result.get("practice_trials", [])
              if trial["method_source"] == result["method_source"]}
    if latest:
        entry["prior_execution"] = {
            "scope": "Source paid practice; these are not current-task observation indices.",
            "cases": [{key: deepcopy(trial[key]) for key in
                       ("case", "status", "fresh_confirmation", "selected_prior_predictor", "costs")
                       if key in trial} for trial in latest.values()],
        }
    manifest = retained.setdefault("manifest", {})
    programs = manifest.setdefault("research_programs", [])
    programs[:] = [old for old in programs
                   if old.get("path") != path and old.get("id") != entry["id"]]
    programs.append(entry)
    if "shared/knowledge.json" in retained["files"]:
        retained["files"]["shared/knowledge.json"] = json.dumps(manifest, indent=2)
    return retained


def method_memory_pair(original_knowledge, consolidation_result):
    """Retain/delete only the learned procedure while sharing its verbal lesson."""
    result = consolidation_result
    retained = register_published_method(original_knowledge, result)
    path = result["method_path"]
    if path in original_knowledge.get("files", {}):
        raise ValueError("the learned method must be a new file")
    # Keep the historical intervention limited to source and manifest: the
    # deleted arm must not carry a role pointing to the removed procedure.
    if "research_roles" in original_knowledge:
        retained["research_roles"] = deepcopy(original_knowledge["research_roles"])
    else:
        retained.pop("research_roles", None)
    deleted = deepcopy(original_knowledge)
    deleted.setdefault("lessons", []).append(result.get("lesson", ""))
    if "shared/knowledge.json" in deleted.get("files", {}):
        deleted["files"]["shared/knowledge.json"] = json.dumps(deleted.get("manifest", {}), indent=2)
    return retained, deleted
