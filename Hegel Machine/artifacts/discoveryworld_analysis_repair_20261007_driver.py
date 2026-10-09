#!/usr/bin/env python3
"""Operator-assigned analysis repair on an existing paid DiscoveryWorld prefix."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_discoveryworld_prerequisite_check import (
    check_summary, continue_arm, failed_result, freeze_prefix,
    verify_codex_subscription, write_json,
)

SOURCE = ROOT / "artifacts/discoveryworld_supplied_reference_20261007/dw_04_normal_s0/hegel"
OUTPUT = ROOT / "artifacts/discoveryworld_analysis_repair_20261007"
ASSIGNMENT = """# Operator-assigned review of existing child analyses

Before acquiring further observations, re-examine the two existing child
analyses below against their already paid records. This review was requested by
the operator; discovering the need for this review is not part of this trial.
No scientific answer or frequency law is supplied.

1. `test_crystal4_870`: inspect the frozen selected 916-Hz experiment and its
   observations 390 through 400. Its identifier and some inherited wording refer
   to 870 Hz; interpret the selected experiment actually executed. The repeat
   attempt at 401 is a separate attempt, not the start of the preceding trial.
2. `intervention_34810`: inspect the empty-setting/insertion trial at observations
   370 through 380 and its frozen prediction. Distinguish this trial from the
   earlier 363 through 369 trial and the later unsuccessful repeat. Do not let
   the latest invocation boundary erase evidence from the completed trial.

Read the actual records and the current analysis source. Compare their field
paths, object descriptions and UUID references, intended conditions, achieved
controls and physical outcomes. A missing observation and an observed negative
outcome have different meanings. Interpret the evidence yourself; an outcome
may support, refute or leave the specified prediction unresolved. State only the
scope established by these records, and cite their observation indices.

Repair each analysis through the existing `evaluate` action with its own
`question_id`, a separate analysis `path`, and inline `code` defining
`answer(public_spec, observations, memory)`. Do not run `contrast` or acquire
another observation merely to repair analysis of an existing trial. Preserve
the original experiment design and procedure source. Inspect the actual
evaluation result; repair remaining analysis errors on the same paid evidence.

After both child analyses are resolved as far as the existing evidence allows,
return to the parent research and let their scoped answers inform the next
scientific decision. Further experiments are then available if justified.
"""


def main():
    auth_mode = verify_codex_subscription()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source_plan, prefix = freeze_prefix(SOURCE, OUTPUT)
    if len(prefix["records"]) != 421:
        raise ValueError("This assigned review requires the recorded 421-observation prefix")
    if source_plan["model"]["backend"] != "codex_cli":
        raise ValueError("The source must use the authorized subscription CLI provider")
    prefix["notes"] = ASSIGNMENT + "\n\n# Inherited research notebook\n\n" + prefix["notes"]
    prefix["operator_assignment"] = ASSIGNMENT
    write_json(OUTPUT / "prefix.json", prefix)
    (OUTPUT / "analysis_repair_assignment.md").write_text(ASSIGNMENT, encoding="utf-8")
    arms = ["code_archive", "hegel"]
    plan = {
        **source_plan,
        "study": "discoveryworld_operator_assigned_paid_evidence_analysis_repair",
        "tasks": [prefix["task"]], "arms": arms,
        "jobs": [{"task_id": prefix["task"]["id"], "arm": arm} for arm in arms],
        "limits": {**source_plan["limits"], "max_model_calls": 8}, "workers": 2,
        "auth_mode": auth_mode, "prefix_source": prefix["source"], "prefix_calls": None,
        "source_checkpoint_costs": prefix["source_checkpoint_costs"],
        "prefix_cost_scope": prefix["cost_scope"],
        "prefix_environment_steps": prefix["environment_steps"],
        "prefix_prior_costs": prefix["prior_costs"],
        "prefix_question_ids": [q["id"] for q in prefix["research"]["questions"]],
        "operator_assignment": {"text": ASSIGNMENT, "file": "analysis_repair_assignment.md",
            "origin": "operator_selected_review", "shared_by_both_arms": True},
        "purpose": "Test model-authored repair of existing analyses on the same paid evidence, "
                   "then parent continuation. This is an operator-assigned review, not autonomous "
                   "error detection or a full-task performance comparison.",
        "replay_policy": "Each arm replays the frozen official actions and compares public "
                         "success/messages before model calls, using the existing continuation runner.",
        "accounting": "New model calls and tokens start at zero. The environment ceiling includes "
                      "the prefix; prior costs and new environment steps are reported separately.",
    }
    write_json(OUTPUT / "protocol.json", plan)
    results = []
    write_json(OUTPUT / "summary.json", check_summary(plan, results))
    with ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(continue_arm, plan, arm, str(OUTPUT)): arm for arm in arms}
        for future in as_completed(pending):
            arm = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                workspace = OUTPUT / prefix["task"]["id"] / arm
                workspace.mkdir(parents=True, exist_ok=True)
                result = failed_result(prefix["task"], arm, workspace, exc)
            results.append(result)
            write_json(OUTPUT / "summary.json", check_summary(plan, results))
            print(json.dumps({"arm": arm, "status": result["status"],
                              "finished": len(results)}), flush=True)
    print(json.dumps(check_summary(plan, results), indent=2))


if __name__ == "__main__":
    main()
