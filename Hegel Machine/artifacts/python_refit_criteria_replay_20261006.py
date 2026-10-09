"""Disclosed same-task development replay after publishing the unchanged criterion."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_python_refit_diagnosis import protocol, run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription
from hegel_machine.cognition.code_agent import CodexCodeProvider

plan = protocol("live")
plan.update(development_replay_of=str(ROOT / "artifacts/python_refit_diagnosis_luna_20261006"),
    repair="Publish the existing MSE <= 1e-10 confirmation criterion in the common task "
           "interface and fresh-confirmation feedback. No threshold, budget, task or source change.",
    scope="All four previously attempted tasks replayed after a public-criterion repair. "
          "The original two failures remain. This is development evidence, not a new independent "
          "sample or an isolated causal estimate of the interface change.")
output = ROOT / "artifacts/python_refit_criteria_luna_20261006"
result = run_study(output, mode="live", study_protocol=plan,
    auth_mode=verify_codex_subscription(), provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180))
shutil.copy2(__file__, output / "source_at_run" / Path(__file__).name)
print({"confirmed_and_correct": result["confirmed_and_correct"],
       "new_target_costs": result["new_target_costs"]})
