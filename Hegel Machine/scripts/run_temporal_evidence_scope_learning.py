#!/usr/bin/env python3
"""Learn from an explicit chronology of the failed method's paid evidence."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_counterexample_learning import run_study


FAILED = ROOT / "artifacts/temporal_failure_selectivity_luna_20261006"


def run_experiment(output, *, provider_factory, auth_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = json.loads((FAILED / "protocol.json").read_text())
    failed = json.loads((FAILED / "study/learning/result.json").read_text())
    trial = json.loads((FAILED / "study/learning/trials/trial_003/result.json").read_text())
    counterexample = output / "paid_counterexample.json"
    write_json(counterexample, {"tasks": [trial], "evaluation": None})
    plan.update(study="llm_temporal_evidence_scope_revision",
        counterexample=str(counterexample.resolve()), counterexample_case="case_b",
        preceding_author_phase={"path": str(FAILED / "study/learning/result.json"),
            "status": failed["status"], "costs": deepcopy(failed["costs"])},
        learning="A new author phase starts from the last successfully published method. "
            "An actual failed practice from the preceding author phase supplies its own source, "
            "paid observations, and a compact chronology of record ranges and fitting scopes. "
            "This increases the salience of existing evidence; it does not supply a repaired program, "
            "cause labels, held-out outcomes, or new fitting algorithms. The earlier unsuccessful "
            "author phase remains a separate negative result and cost. Four-case publication and "
            "the preselected thirty-two transfer tasks are unchanged.")
    write_json(output / "protocol.json", plan)
    shutil.copy2(Path(__file__), output / "runner_at_start.py")
    result = run_study(output / "study", provider_factory=provider_factory,
                       auth_mode=auth_mode, study_protocol=plan)
    result["preceding_author_phase"] = plan["preceding_author_phase"]
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = verify_codex_subscription()
    print(json.dumps(run_experiment(args.output, auth_mode=auth,
        provider_factory=lambda path: CodexCodeProvider(path, model="gpt-6-luna",
            reasoning_effort="medium", timeout_seconds=180)), indent=2))


if __name__ == "__main__":
    main()
