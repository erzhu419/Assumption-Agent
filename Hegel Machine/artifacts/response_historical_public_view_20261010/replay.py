"""Execute unchanged recorded analyses against their existing paid observations."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import InstrumentLab, LIMITS, ScriptedProvider

OUT = Path(__file__).resolve().parent
inputs = json.loads((OUT / "recorded_cases.json").read_text())
history = json.loads((ROOT / inputs["source_observations"]).read_text())
results = []
for case in inputs["cases"]:
    index = case["observation_index"]
    assert history[index] == case["raw_observation_record"]
    original = json.loads((ROOT / case["source_action"]).read_text())
    assert case["analysis"] == json.loads(original["arguments_json"])["code"]
    namespace = {}
    exec(case["analysis"], namespace)
    for arm in ("hegel", "code_archive"):
        with TemporaryDirectory() as workspace:
            lab = InstrumentLab()
            lab.public_observation = lambda: deepcopy(case["raw_observation_record"]["observation"])
            agent = DiscoveryWorldAgent(ScriptedProvider([]), workspace, arm=arm, limits=LIMITS)
            agent.lab = lab
            agent.observations = deepcopy(history[:index + 1])
            agent.questions = deepcopy(case["memory"].get("questions", []))
            raw_history = deepcopy(agent.observations)
            before = namespace["answer"](agent._spec(), [
                {**record, **agent._action_observation(record)} for record in raw_history], agent._state())
            agent._execute("write", {"path": "recorded.py", "code": case["analysis"]}, lab)
            after = agent._program("recorded.py", "answer")
            assert before["status"] == "inconclusive"
            assert after["status"] == "refuted" and after["answer"]["activated"] is False
            assert agent.observations == raw_history
            assert json.loads((Path(workspace) / "laboratory_observations.json").read_text()) == raw_history
            assert not lab.actions and agent.costs["model_calls"] == 0
            results.append({"model_call": case["model_call"], "observation_index": index,
                "arm": arm, "before": before, "after": after,
                "new_world_actions": 0, "new_model_calls": 0,
                "analysis_unchanged": True, "raw_history_unchanged": True})
(OUT / "summary.json").write_text(json.dumps({"scope": "Recorded analysis-only replay; no new scientific prediction or task completion",
    "cases": results, "new_model_calls": 0, "new_world_actions": 0}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"analyses": len(results), "before_inconclusive": len(results),
    "after_refuted": len(results), "new_model_calls": 0, "new_world_actions": 0}))
