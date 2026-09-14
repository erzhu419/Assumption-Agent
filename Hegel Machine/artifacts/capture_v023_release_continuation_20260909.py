"""Capture the real pre-visit state of the two diagnosed release regressions."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts/v023_release_source_20260909/hegel_machine"
PARTITIONS = ("development", "confirmation", "forward", "continual", "stateful")
CASES = {4507: (36, 30), 7919: (42, 24)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=CASES, required=True)
    args = parser.parse_args()
    before, target = CASES[args.seed]
    output = ROOT / f"artifacts/v023_release_seed{args.seed}_precontinuation_20260909.json"
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location("hegel_machine", SNAPSHOT / "__init__.py",
        submodule_search_locations=[str(SNAPSHOT), str(ROOT / "src/hegel_machine")])
    package = importlib.util.module_from_spec(spec)
    sys.modules["hegel_machine"] = package
    spec.loader.exec_module(package)
    from hegel_machine.cognition.environments import make_suite
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition import session as module

    sources = [(partition, index, environment) for partition in PARTITIONS
               for index, environment in enumerate(make_suite(args.seed, partition=partition))]
    config = ResearchConfig(seed=args.seed)
    session = module.ResearchSession(config, [row[2] for row in sources],
        observation_budget=65 * 96 * 2, compute_budget=65 * 24000 * 2, max_visits=2)
    session.state["partitions"] = list(PARTITIONS)

    def progress(checkpoint):
        if checkpoint["completed_visits"] % 10 == 0:
            print(json.dumps({"seed": args.seed, "prefix_visits": checkpoint["completed_visits"]}), flush=True)

    session.run(max_steps=before, on_checkpoint=progress)

    class Captured(Exception):
        pass

    class CaptureAgent:
        def __init__(self, visit_config, library):
            self.config, self.library = visit_config, library

        def run(self, environment, *, research_state):
            partition, local_index, expected = sources[target]
            if environment.spec.task_id != expected.spec.task_id:
                raise ValueError(f"Expected task {target}, selected {environment.spec.task_id}")
            task = session.state["tasks"][target]
            bundle = {"purpose": "Known release regression context; diagnostic, not independent confirmation",
                "source": str(SNAPSHOT), "suite_seed": args.seed, "prefix_visits": before,
                "task_index": target, "task_id": environment.spec.task_id,
                "partition": partition, "local_index": local_index, "spec": asdict(environment.spec),
                "session_config": asdict(config), "visit_config": asdict(self.config),
                "workspace": research_state, "library": self.library.to_dict(),
                "next_research": task.get("next_research"), "next_method": task.get("next_method"),
                "prefix_cost": {"observation_cost": session.state["total_observation_cost"],
                                "compute_cost": session.state["total_compute_cost"]},
                "evaluation_seed": self.config.seed + 2000003, "evaluation_trajectories": 16}
            with output.open("x") as stream:
                json.dump(bundle, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({"captured": str(output), "task_id": bundle["task_id"],
                "config": bundle["visit_config"], "source": bundle["source"]}), flush=True)
            raise Captured

    module.CognitiveAgent = CaptureAgent
    try:
        session.run(max_steps=1)
    except Captured:
        pass
    else:
        raise RuntimeError("Expected a cognitive continuation after the captured prefix")


if __name__ == "__main__":
    main()
