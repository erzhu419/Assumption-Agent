"""Execute the supplied control method on the frozen public prefix without an LLM."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tests")]
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from scripts.run_discoveryworld_prerequisite_check import replay_prefix
from test_discoveryworld_agent import ScriptedProvider

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/discoveryworld_adaptive_reference_20261007"
prefix = json.loads((SOURCE / "prefix.json").read_text())
plan = json.loads((SOURCE / "protocol.json").read_text())
source = (ROOT / "src/hegel_machine/cognition/public_reference_method.py").read_text()
method = {"id": "reference_control", "path": "shared/reference_control.py",
          "parameters": {"target": 50759, "perturb_label": "Increase frequency by 100 Hz",
                         "restore_label": "Decrease frequency by 100 Hz"},
          "max_steps": 12, "requires": [], "provenance": "operator_authored",
          "evidence_file": "shared/reference_control_practice.json",
          "applicability": "An operator-authored public-menu procedure for a caller-selected reference and reversible menu controls. Its archive records one actual execution; the caller must interpret physical outcomes and transfer conditions."}
question = {"id": "reference_recovery", "parent_id": "task",
            "question": "Does reversing one public control restore the reference physical state?"}
result = {"scope": "operator-authored procedure practice; no LLM and no automatic scientific answer",
          "prior_observation_count": len(prefix["records"]),
          "prior_environment_steps": prefix["environment_steps"],
          "source_prefix": str(SOURCE / "prefix.json"), "model_calls": 0}

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))

try:
    with DiscoveryWorldLab(prefix["task"], plan["runtime"], OUT / "runtime") as lab:
        result["replay"] = replay_prefix(lab, prefix)
        agent = DiscoveryWorldAgent(ScriptedProvider([]), OUT / "agent", arm="hegel", limits=plan["limits"])
        agent.lab = lab
        agent.observations = deepcopy(prefix["records"])
        agent.questions = deepcopy(prefix["research"]["questions"])
        agent.costs["environment_steps"] = lab.environment_steps
        agent._execute("write", {"path": method["path"], "code": source}, lab)
        # This practice will create the archive; it cannot claim it beforehand.
        agent._execute("run", {"operator": "method", **{key: value for key, value in method.items()
                                                      if key != "evidence_file"}}, lab)
        execution = agent._execute("run", {"operator": "procedure", "method_id": method["id"],
                                           "question": question}, lab)
        new_records = agent.observations[len(prefix["records"]):]
        save("new_observations.json", new_records)
        compact = []
        for row in new_records:
            public = agent._action_observation(row)
            public["source_observation_index"] = public.pop("index")
            compact.append(public)
        result.update(status=execution["status"], execution=execution,
                      method_results=agent.method_results, new_observations=compact,
                      new_environment_steps=lab.environment_steps - prefix["environment_steps"],
                      program_executions=agent.program_executions,
                      research_question=next(q for q in agent.questions if q["id"] == question["id"]),
                      score=lab.score())
        practice = {"record_namespace": OUT.name,
                    "source_author": "operator", "source_task": prefix["task"],
                    "scope": "Actual prior execution of the supplied public-control procedure; external practice records, not this consuming agent's observations or a task-law answer.",
                    "index_semantics": "source_observation_index refers only to this named practice archive, never to a consuming episode's laboratory_observations.json",
                    "procedure_status": execution["status"], "method": method,
                    "source_prefix_observation_count": len(prefix["records"]),
                    "new_environment_steps": result["new_environment_steps"],
                    "model_calls": 0, "observed_states": compact}
        save("reference_control_practice.json", practice)
        save("method_bundle.json", {"files": {method["path"]: source,
            method["evidence_file"]: json.dumps(practice, indent=2, ensure_ascii=False)}, "methods": [method]})
except Exception as error:
    result.update(status="error", error=f"{type(error).__name__}: {error}")
    save("result.json", result)
    raise
save("result.json", result)
print(json.dumps({key: result[key] for key in ("status", "model_calls", "new_environment_steps", "program_executions")}))
