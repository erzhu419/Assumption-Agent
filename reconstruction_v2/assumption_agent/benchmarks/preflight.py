from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .skilllearnbench import SkillLearnBenchAdapter
from .skilllearn_lifecycle import VERIFIER_ISOLATION_VERSION
from ..provider_chain import proposal_provider_status
from ..secure_env import (
    LOCKED_MODEL,
    alternate_model_allowed,
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
    resolve_codex_auth_path,
)


def build_preflight(
    root: str | Path,
    *,
    trial_provider_mode: str | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    checks: dict[str, dict[str, Any]] = {}
    checks["python_supported"] = {
        "passed": sys.version_info >= (3, 10),
        "version": ".".join(str(value) for value in sys.version_info[:3]),
    }
    docker = shutil.which("docker")
    checks["docker_cli"] = {"passed": bool(docker), "path_present": bool(docker)}
    docker_daemon = False
    if docker:
        try:
            completed = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            docker_daemon = completed.returncode == 0 and bool(completed.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            docker_daemon = False
    checks["docker_daemon"] = {"passed": docker_daemon}
    dataclaw = shutil.which("dataclaw")
    checks["dataclaw_cli"] = {
        "passed": bool(dataclaw),
        "path_present": bool(dataclaw),
        "required_for_v2_execution": False,
    }
    json_repair_present = importlib.util.find_spec("json_repair") is not None
    checks["skilllearn_runtime_dependencies"] = {
        "passed": json_repair_present,
        "json_repair_present": json_repair_present,
    }
    try:
        inventory = SkillLearnBenchAdapter(root).inventory_summary()
        repository_ok = inventory["instance_count"] == 100 and inventory["family_count"] == 20
    except (FileNotFoundError, ValueError) as exc:
        inventory = {"error_type": type(exc).__name__}
        repository_ok = False
    checks["local_benchmark_inventory"] = {"passed": repository_ok, **inventory}
    provider_status = proposal_provider_status()
    model = provider_status["model"]
    allow_alternate = alternate_model_allowed()
    model_policy_passed = model == LOCKED_MODEL or allow_alternate
    checks["proposal_model_config"] = {
        **provider_status,
        "passed": bool(provider_status["passed"] and model_policy_passed),
        "model_policy_passed": model_policy_passed,
        "alternate_model_allowed": allow_alternate,
        "secret_value_persisted": False,
    }
    mode = trial_provider_mode or configured_skilllearn_provider_mode()
    if mode == "codex_subscription":
        auth_present = resolve_codex_auth_path() is not None
        trial_auth_passed = bool(
            auth_present and provider_status["codex_chatgpt_login_present"]
        )
        auth_check = {
            "codex_auth_file_present": auth_present,
            "codex_chatgpt_login_present": provider_status["codex_chatgpt_login_present"],
            "api_key_required": False,
        }
    elif mode == "openai_compatible":
        trial_auth_passed = bool(provider_status["openai_compatible_config_present"])
        auth_check = {
            "codex_auth_file_present": False,
            "codex_chatgpt_login_present": False,
            "api_key_required": True,
        }
    else:
        trial_auth_passed = False
        auth_check = {
            "codex_auth_file_present": False,
            "codex_chatgpt_login_present": False,
            "api_key_required": False,
        }
    checks["skilllearn_trial_auth"] = {
        "passed": trial_auth_passed,
        "provider_mode": mode,
        "ephemeral_codex_home_bind": mode == "codex_subscription",
        "secret_value_persisted": False,
        **auth_check,
    }
    checks["verifier_isolation"] = {
        "passed": VERIFIER_ISOLATION_VERSION == "post_agent_verifier_copy_v1",
        "version": VERIFIER_ISOLATION_VERSION,
        "verifier_visible_during_agent": False,
    }
    required_checks = (
        "python_supported",
        "docker_cli",
        "docker_daemon",
        "local_benchmark_inventory",
        "proposal_model_config",
        "skilllearn_runtime_dependencies",
        "skilllearn_trial_auth",
        "verifier_isolation",
    )
    blockers = [name for name in required_checks if not checks[name].get("passed")]
    return {
        "preflight_version": "skilllearnbench_preflight_v1",
        "checks": checks,
        "blockers": blockers,
        "ready_for_inventory_and_manifest": repository_ok,
        "ready_for_live_skill_generation": not blockers,
        "optional_warnings": ["dataclaw_cli"] if not checks["dataclaw_cli"]["passed"] else [],
        "raw_content_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local SkillLearnBench prerequisites without network calls.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--trial-provider-mode",
        choices=("codex_subscription", "openai_compatible"),
    )
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
        map_legacy_model_env()
    print(
        json.dumps(
            build_preflight(args.root, trial_provider_mode=args.trial_provider_mode),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
