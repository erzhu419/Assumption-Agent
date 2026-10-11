"""One disclosed development replay of the previously unsubmitted Luna task."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.code_agent import CodexCodeProvider
from scripts.run_legacy_session_llm import protocol, run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription

plan = protocol()
plan.update(study="candidate_retention_known_failure_development_replay",
    cases=[case for case in plan["cases"] if case["apparatus_seed"] == 1120077],
    conditions=["retained"],
    previous_attempt="artifacts/legacy_session_luna_20261006/case_1120077/retained/training.json",
    evaluation_order="Stop this one registered learning attempt before terminal evaluation.",
    scope="One previously observed failure with the repaired candidate interface and the same "
          "paid prefix and resource limits. This is development replay, not independent "
          "confirmation or causal isolation of the code change. No old attempt is replaced.")
result = run_study(Path(__file__).parent, study_protocol=plan,
    auth_mode=verify_codex_subscription(), max_workers=1,
    provider_factory=lambda path: CodexCodeProvider(path, model="gpt-6-luna",
        reasoning_effort="medium", timeout_seconds=180))
print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
