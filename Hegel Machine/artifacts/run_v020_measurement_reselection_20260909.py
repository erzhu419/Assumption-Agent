"""Actual process apparatus research with a fixed initial measurement control."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import InstrumentedEnvironment, MeasurementResearchAgent
from hegel_machine.cognition.measurement_session import MeasurementSession, session_summary
from hegel_machine.cognition.models import ResearchConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/measurement_reselection_v020_20260909.json"
SUMMARY = ROOT / "artifacts/measurement_reselection_v020_20260909_summary.json"
CONFIG = ResearchConfig(observation_budget=128, compute_budget=100000, max_rounds=3, seed=313)


def evaluate(lab, channels, episode, seed=1441):
    rows = lab.evaluation_rows(channels, seed=seed, length=64, trajectories=4)
    return {"seed": seed, "rows": len(rows), "compute_cost": len(rows),
            "mse": loss(Program.from_dict(episode["program"]), rows, episode["fallback"], .5)}


def main():
    with make_lab(3, 313, backend="process") as adaptive_lab, make_lab(3, 313, backend="process") as fixed_lab:
        agent = MeasurementResearchAgent(CONFIG, measurement_budget=512)
        initial = agent.run(adaptive_lab, max_steps=1)
        saved = json.loads(json.dumps(initial["research_state"]))
        adaptive = agent.run(adaptive_lab, research_state=saved)
        fixed_lab.restore(saved["apparatus_acquisitions"])
        rate = sum(item.cost_per_row for item in fixed_lab.channels if item.name in initial["selected_channels"])
        remaining_observations = CONFIG.observation_budget - initial["costs"]["observation_cost"]
        if rate:
            remaining_observations = min(remaining_observations,
                (512 - initial["costs"]["measurement_cost"]) // rate)
        fixed_episode = CognitiveAgent(replace(CONFIG, observation_budget=remaining_observations,
            compute_budget=CONFIG.compute_budget - initial["costs"]["compute_cost"])).run(
                InstrumentedEnvironment(fixed_lab, initial["selected_channels"]),
                research_state=saved["workspace"])
        fixed = {"selected_channels": initial["selected_channels"], "episode": fixed_episode.to_dict(),
                 "costs": {**fixed_lab.costs,
                    "compute_cost": initial["costs"]["compute_cost"] + fixed_episode.compute_cost},
                 "apparatus_acquisitions": fixed_lab.acquisitions}
        # Both research arms have stopped before either independent stream is requested.
        adaptive["evaluation"] = evaluate(adaptive_lab, adaptive["selected_channels"], adaptive["episode"])
        fixed["evaluation"] = evaluate(fixed_lab, fixed["selected_channels"], fixed["episode"])

    stable = MeasurementSession(CONFIG, tasks=[0, 1, 2], backend="process", measurement_budget=512).run()
    limited = MeasurementSession(replace(CONFIG, observation_budget=96, compute_budget=60000),
        tasks=[3], backend="process", measurement_budget=320).run()
    restored = MeasurementSession(CONFIG, measurement_budget=512,
                                  state=json.loads(json.dumps(limited))).run()
    report = {
        "experiment": "measurement_reselection_v020", "date": "2026-09-09",
        "config": asdict(CONFIG), "measurement_budget": 512, "backend": "process",
        "protocol": {
            "development_task_seed": 313, "apparatus_index": 3,
            "shared_initial_pilots": True,
            "fixed_view": "Same saved initial selected channel and workspace; ordinary CognitiveAgent receives all remaining observation and computation caps. No new instrument comparison.",
            "evaluation": "Both arms stop before evaluation. Four fresh 64-row trajectories use the final reached operating regime; evaluation costs are separate.",
            "instrument_replay": "Public command replay reconstructs apparatus state and historical billed totals, with no new learning rows. Reacquiring a previous trajectory with additional instruments is a new billed request.",
        },
        "initial": initial, "adaptive": adaptive, "fixed_initial_view": fixed,
        "stable_tasks": session_summary(stable),
        "budget_recovery": {"before": session_summary(limited), "after": session_summary(restored),
            "resumed_result": restored["results"][0]},
        "development_failure_retained": {
            "phase": "Before final representation comparison was implemented",
            "finding": "A short refit discarded a better existing explanation and preferred sensor_b alone, while the saved incumbent still required sensor_a. The following research step raised KeyError.",
            "paid_validation_losses": {"refitted_sensor_a": 2.0675, "refitted_sensor_b": 1.1256,
                "refitted_both": 1.9548, "existing_sensor_a": .17501625929332423},
            "correction": "Compare compatible existing explanations, refit actual measured response-history terms, and update incompatible workspace hypotheses after dropping a channel.",
            "independent_evaluation_used_for_correction": False,
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    compact = {"experiment": report["experiment"], "date": report["date"], "protocol": report["protocol"],
        "config": report["config"], "measurement_budget": 512,
        "initial_channels": initial["selected_channels"],
        "initial_validation_loss": initial["research_state"]["view_validation_loss"],
        "arms": {name: {"selected_channels": value["selected_channels"], "costs": value["costs"],
            "evaluation": value["evaluation"], "expression": value["episode"]["expression"]}
            for name, value in (("adaptive", adaptive), ("fixed_initial_view", fixed))},
        "measurement_events": adaptive["measurement_events"],
        "stable_tasks": report["stable_tasks"],
        "budget_recovery": {"before": report["budget_recovery"]["before"],
                            "after": report["budget_recovery"]["after"]},
        "source": str(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": str(SUMMARY), "arms": compact["arms"],
                      "budget_recovery_status": restored["status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
