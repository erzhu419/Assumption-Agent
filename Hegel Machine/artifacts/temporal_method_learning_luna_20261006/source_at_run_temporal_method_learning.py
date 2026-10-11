"""LLM-authored research-method consolidation from completed paid experience."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .code_provider_context import _compact_trajectory_payload
from .online_inquiry_llm import InquiryLLMAgent
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
Downstream shared operators are history (base_lags,path,validation_records),
instrument (path), response (instrument,output_dir,validation_records),
disagreement (instrument,max_length,serial), and assess_instrument (instrument,records).
Each operator dictionary includes operator:'name'; records:'acquired' binds to
the newly bought laboratory record indices. Only your declared operators run,
in order after successful acquisition. They use paid evidence; candidate selection
and submission remain decisions for the consuming agent. Learn applicable conditions
and useful steps from the supplied experience, and leave unresolved choices explicit.
No hegel_machine imports or access to other workspaces, artifacts or terminal data.
finish publishes only this procedure and its description. Publication checks a callable
select_actions; it does not imply predictive confirmation or transfer benefit.
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
        code, stdout, stderr, timed_out = self._python([sys.executable, "-c",
            "import runpy,sys; module=runpy.run_path(sys.argv[1]); "
            "assert callable(module.get('select_actions')), 'select_actions must be callable'", str(self.workspace / path)])
        if code:
            return {"error": "method publication check failed", "returncode": code,
                    "stdout": stdout[-2000:], "stderr": stderr[-4000:], "timeout": timed_out}
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


def method_memory_pair(original_knowledge, consolidation_result):
    """Retain/delete only the learned procedure while sharing its verbal lesson."""
    result = consolidation_result
    if result.get("status") != "completed" or not result.get("method_source") or not result.get("method_entry"):
        raise ValueError("a published consolidated method is required; no fallback is generated")
    path = result["method_path"]
    if path in original_knowledge.get("files", {}):
        raise ValueError("the learned method must be a new file")
    deleted = deepcopy(original_knowledge)
    deleted.setdefault("lessons", []).append(result.get("lesson", ""))
    retained = deepcopy(deleted)
    retained.setdefault("files", {})[path] = result["method_source"]
    retained.setdefault("manifest", {}).setdefault("research_programs", []).append(deepcopy(result["method_entry"]))
    for memory in (retained, deleted):
        if "shared/knowledge.json" in memory.get("files", {}):
            memory["files"]["shared/knowledge.json"] = json.dumps(memory.get("manifest", {}), indent=2)
    return retained, deleted
