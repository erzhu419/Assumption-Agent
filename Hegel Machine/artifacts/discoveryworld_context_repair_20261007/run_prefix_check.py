"""Bounded context diagnostic from an already acquired public action prefix."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from scripts.run_discoveryworld_pilot import make_plan, summarize, write_json
from scripts.run_online_inquiry_llm import verify_codex_subscription

OUTPUT = ROOT / "artifacts/discoveryworld_persistent_notes_20261007"
SOURCE = ROOT / "artifacts/discoveryworld_pilot_20261007_v2/dw_04_normal_s0/hegel/agent/laboratory_observations.json"


def run_prefix(plan, arm):
    task = plan["tasks"][0]
    directory = OUTPUT / task["id"] / arm
    workspace = directory / "agent"
    workspace.mkdir(parents=True)
    records = json.loads(SOURCE.read_text())
    provider = CodexCodeProvider(workspace, model=plan["model"]["requested_alias"],
        reasoning_effort=plan["model"]["reasoning_effort"],
        timeout_seconds=plan["limits"]["provider_request_timeout_seconds"])
    agent = DiscoveryWorldAgent(provider, workspace, arm=arm, limits=plan["limits"])
    with DiscoveryWorldLab(task, plan["runtime"], workspace) as lab:
        for record in records:
            replay = {"index": len(agent.observations), **lab.act(record["action"])}
            # A changed public prefix would no longer test the original stall.
            if replay["observation"] != record["observation"]:
                raise ValueError(f"Public prefix replay differs at {record['index']}")
            agent.observations.append(replay)
        initial_score = lab.score()
        agent.costs["environment_steps"] = lab.environment_steps
        result = agent.run(lab, on_checkpoint=lambda d: write_json(directory / "active.json", d))
    result.update(task_id=task["id"], arm=arm, inherited_observation_count=len(records),
                  inherited_score=initial_score,
                  new_environment_steps=result["costs"]["environment_steps"] - len(records))
    write_json(directory / "result.json", result)
    return result


if __name__ == "__main__":
    plan = make_plan(["dw_04_normal_s0"], ["code_archive", "hegel"],
                     "/home/erzhu419/.cache/hegel-machine/discoveryworld", 2)
    plan["study"] = "discoveryworld_persistent_notes_prefix_diagnostic"
    plan["limits"].update(max_model_calls=12, task_wall_seconds=1800,
        note="12 fresh decisions after replaying the same previously acquired public prefix in both arms.")
    plan.update(auth_mode=verify_codex_subscription(), inherited_public_prefix=str(SOURCE),
        purpose="Test persistent authored notes and scientific continuation at the prior read-loop state. "
                "This Hegel-selected prefix is shared by both arms; not a fresh task or performance comparison.")
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / "protocol.json", plan)
    results = []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(run_prefix, plan, arm): arm for arm in plan["arms"]}
        for future in as_completed(jobs):
            results.append(future.result())
            write_json(OUTPUT / "summary.json", summarize(plan, results))
            print(results[-1]["arm"], results[-1]["status"], flush=True)
