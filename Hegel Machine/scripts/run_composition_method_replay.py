"""Development-only ablation of saved research procedures after coefficient change.

The same learned concepts and callable definitions are retained in both arms.
The private apparatus changes its output law; only saved research_program fields
are removed from the ablation. This is separate from the frozen eight-task study.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_session import UnifiedResearchSession


class ChangedReadout(StructureLab):
    """Private response wiring; no output coefficients enter the public spec."""

    def __init__(self, *, a, b, **settings):
        super().__init__(**settings)
        self._a, self._b = a, b
        self._totals = []

    def _step(self, point):
        state = super()._step(point)
        self._totals.append(sum(state[-1]))
        return state

    def acquire(self, probe, *, objects=None, scale=1):
        self._totals = []
        batch = super().acquire(probe, objects=objects, scale=scale)
        rows, history = [], []
        for row, total in zip(batch.observations, self._totals):
            target = self._a * row.target + self._b * total
            rows.append(Observation(row.features, target, tuple(history)))
            history.append({**row.features, "__target__": target})
        return replace(batch, observations=rows)


class ChangedObjectLab(UnifiedObjectLab):
    def __init__(self, *, a, b, **settings):
        super().__init__(**settings)
        self._changed_settings = dict(settings, a=a, b=b)
        self._lab = ChangedReadout(a=a, b=b, objects=settings["objects"],
            readout=settings["readout"], variant=settings["variant"],
            seed=settings["apparatus_seed"])
        self.spec = self.spec_for()

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        # Rebuild the changed apparatus, including its response law, for scoring.
        with ChangedObjectLab(**self._changed_settings) as separate:
            return [row for index in range(trajectories)
                    for row in separate.observe(separate.random_probe(seed + 10007 * index,
                        length=length, name=f"evaluation_{index}"), channels)]


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
        default=Path("artifacts/unified_composition_20260913/development_working/default.json"))
    parser.add_argument("--output", type=Path,
        default=Path("artifacts/unified_composition_20260913/method_replay"))
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    config = replace(ResearchConfig(**source["config"]), observation_budget=384,
        compute_budget=360000, active_probes=False)
    task = {"seed": 80127, "objects": 4, "variant": "exchange", "apparatus_seed": 9013,
            "readout": "total_event", "rename_prefix": "changed_", "a": -.6, "b": 1.8}
    protocol = {"purpose": "development_method_replay_after_coefficient_change",
        "source": str(args.source), "task": task,
        "config": config.__dict__, "measurement_budget": 4608,
        "arms": ["with_research_program", "without_research_program"],
        "difference": "Delete only each meta_prior research_program field in the ablation.",
        "probe_policy": "Existing independent-random policy, same seed in both arms.",
        "terminal_evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "source_learned_procedures": sum(bool(prior.get("research_program"))
                                         for prior in source["knowledge"]["meta_priors"])}
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "protocol.json", protocol)
    summaries, batches = {}, {}
    for arm in protocol["arms"]:
        knowledge = deepcopy(source["knowledge"])
        if arm == "without_research_program":
            for prior in knowledge["meta_priors"]:
                prior.pop("research_program", None)
        session = UnifiedResearchSession(config, tasks=[task], measurement_budget=4608,
            library=KnowledgeLibrary.from_dict(knowledge),
            lab_factory=lambda item: ChangedObjectLab(**item))
        started = time.monotonic()
        session.run()
        report = session.evaluate(length=64, trajectories=4)
        write(args.output / f"{arm}.json", report)
        result = report["results"][0]["research"]
        plan = result.get("research_program") or {}
        method_actions = [action for action in result["actions"]
                          if action["action"] == "execute_research_programs"]
        summaries[arm] = {"status": result["status"], "costs": result["costs"],
            "selected_kind": result["selected_kind"], "selected_channels": result["selected_channels"],
            "selected_reused_procedure": plan.get("reused_procedure", False),
            "method_source_ids": result["method_source_ids"],
            "replayed_proposals": sum(alternative["reused_procedure"]
                for action in method_actions for alternative in action["alternatives"]),
            "research_program_compute": sum(action["compute_cost"] for action in method_actions),
            "loss": report["evaluation"][0]["loss"],
            "evaluation": report["evaluation"][0], "elapsed_seconds": time.monotonic() - started}
        batches[arm] = result["batches"]
        print(json.dumps({"arm": arm, **summaries[arm]}, ensure_ascii=False), flush=True)
        write(args.output / "summary.json", {"protocol": protocol, "arms": summaries})
    full, ablated = (summaries[arm] for arm in protocol["arms"])
    comparison = {"identical_acquired_batches": batches[protocol["arms"][0]] == batches[protocol["arms"][1]],
        "both_solved": max(full["loss"], ablated["loss"]) < 1e-7,
        "full_minus_ablated": {key: full["costs"][key] - ablated["costs"][key]
                               for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "compute_reduction_fraction": 1. - full["costs"]["compute_cost"] / ablated["costs"]["compute_cost"]}
    write(args.output / "summary.json", {"protocol": protocol, "arms": summaries, "comparison": comparison})
    print(json.dumps({"comparison": comparison}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
