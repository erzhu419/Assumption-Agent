"""Execute one unchanged paid continuation under release or corrected code."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(4507, 7919), required=True)
    parser.add_argument("--source", choices=("release", "corrected"), required=True)
    args = parser.parse_args()
    output = ROOT / f"artifacts/v023_{args.source}_seed{args.seed}_continuation_replay_20260909.json"
    if output.exists():
        raise FileExistsError(output)
    if args.source == "release":
        snapshot = ROOT / "artifacts/v023_release_source_20260909/hegel_machine"
        spec = importlib.util.spec_from_file_location("hegel_machine", snapshot / "__init__.py",
            submodule_search_locations=[str(snapshot), str(ROOT / "src/hegel_machine")])
        package = importlib.util.module_from_spec(spec)
        sys.modules["hegel_machine"] = package
        spec.loader.exec_module(package)
    else:
        sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from hegel_machine.cognition.controller import CognitiveAgent
    from hegel_machine.cognition.environments import make_suite
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from run_method_continuation_study import _new_evaluation_rows

    source = ROOT / f"artifacts/v023_release_seed{args.seed}_precontinuation_20260909.json"
    bundle = json.loads(source.read_text())
    environment = make_suite(args.seed, partition=bundle["partition"])[bundle["local_index"]]
    library = KnowledgeLibrary.from_dict(bundle["library"])
    config = ResearchConfig(**bundle["visit_config"])
    episode = CognitiveAgent(config, library).run(environment, research_state=bundle["workspace"]).to_dict()
    rows = _new_evaluation_rows(environment, bundle["evaluation_seed"], bundle["evaluation_trajectories"])
    measured = loss(Program.from_dict(episode["program"]), rows, episode["fallback"], environment.spec.risk_quantile)
    action_costs = defaultdict(int)
    for action in episode["actions"]:
        action_costs[action["action"]] += action["compute_cost"]
    summary = {"source": args.source, "bundle": str(source), "suite_seed": args.seed,
        "task_id": bundle["task_id"], "visit_config": asdict(config),
        "observation_cost": episode["observation_cost"], "compute_cost": episode["compute_cost"],
        "validation_loss": episode["validation_loss"], "independent_loss": measured,
        "expression": episode["expression"], "stop_reason": episode["stop_reason"],
        "action_compute": dict(action_costs),
        "evaluation_cost": {"observation_rows": len(rows), "candidate_row_compute": len(rows),
                            "trajectories": bundle["evaluation_trajectories"], "seed": bundle["evaluation_seed"]},
        "interpretation": "Same frozen pre-visit evidence, library, selected method and paid visit budget; these two known failure cases diagnose the repair."}
    report = {"summary": summary, "episode": episode}
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
