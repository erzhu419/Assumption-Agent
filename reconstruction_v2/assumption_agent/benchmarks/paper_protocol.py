from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..models import HypothesisProgram, stable_hash
from ..evolution import (
    COUNTERFACTUAL_REPLAY_POLICY_VERSION,
    TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
)
from ..provider_chain import configured_provider_chain, proposal_provider_status
from ..secure_env import (
    configured_api_origin,
    configured_model,
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
)
from ..splits import SplitManifest
from .preflight import build_preflight
from .skilllearn_compiler import SKILL_ROUTING_VERSION
from .prewarm import DEVELOPMENT_PREWARM_VERSION
from .skilllearn_lifecycle import (
    EPHEMERAL_AUTH_CLEANUP_VERSION,
    INVALID_TRIAL_RETRY_POLICY_VERSION,
    LOCAL_EVIDENCE_TRANSPORT_VERSION,
    NETWORK_SCOPE_AUDIT_VERSION,
    OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION,
    PREBUILT_IMAGE_POLICY_VERSION,
    PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION,
    PROVIDER_ROUTE_POLICY_VERSION,
    PROVIDER_FAILURE_POLICY_VERSION,
    RUNNER_AGENT_REGISTRY_ISOLATION_VERSION,
    SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
    SHARED_CODEX_CLI_PACKAGE,
    SHARED_CODEX_CLI_VERSION,
    TRAINING_EVIDENCE_POLICY_VERSION,
    TRAINING_EVIDENCE_REPLAY_POLICY_VERSION,
    TRIAL_TIMEOUT_POLICY_VERSION,
    VERIFIER_ISOLATION_VERSION,
)
from .skilllearnbench import SkillLearnBenchAdapter


PAPER_ROUTES_BY_MAJOR: dict[int, dict[str, Any]] = {
    1: {
        "model": "gpt-5.3-codex-spark",
        "proposal_provider_chain": ["codex_app_server", "openai_compatible"],
        "trial_provider_mode": "codex_subscription",
    },
    2: {
        "model": "gpt-5.3-codex-spark",
        "proposal_provider_chain": ["codex_app_server", "openai_compatible"],
        "trial_provider_mode": "codex_subscription",
    },
    3: {
        "model": "gpt-5.4-mini",
        "proposal_provider_chain": ["openai_compatible"],
        "trial_provider_mode": "openai_compatible",
        "provider_endpoint_origin": "https://ruoli.dev",
    },
}


@dataclass(frozen=True)
class PaperProtocol:
    path: Path
    payload: Mapping[str, Any]

    @property
    def id(self) -> str:
        return str(self.payload.get("protocol_id") or "")

    @property
    def protocol_hash(self) -> str:
        return stable_hash(self.payload)

    @classmethod
    def read(cls, path: str | Path) -> "PaperProtocol":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("paper protocol must contain one JSON object")
        protocol = cls(path=source, payload=payload)
        issues = protocol.validate_structure()
        if issues:
            raise ValueError(f"invalid paper protocol: {issues}")
        return protocol

    def validate_structure(self) -> list[str]:
        issues: list[str] = []
        major = _protocol_major(self.payload.get("protocol_version"))
        if not self.id:
            issues.append("protocol_id_missing")
        if self.payload.get("benchmark") != "skilllearnbench":
            issues.append("benchmark_mismatch")
        route = PAPER_ROUTES_BY_MAJOR.get(major)
        if route is None:
            issues.append("unsupported_protocol_version")
        else:
            if self.payload.get("model") != route["model"]:
                issues.append("paper_model_route_mismatch")
            if list(self.payload.get("proposal_provider_chain") or []) != route[
                "proposal_provider_chain"
            ]:
                issues.append("proposal_provider_route_mismatch")
            if self.payload.get("trial_provider_mode") != route["trial_provider_mode"]:
                issues.append("trial_provider_route_mismatch")
            if route.get("provider_endpoint_origin") and self.payload.get(
                "provider_endpoint_origin"
            ) != route["provider_endpoint_origin"]:
                issues.append("provider_endpoint_route_mismatch")
        execution = self.payload.get("execution")
        if not isinstance(execution, Mapping):
            issues.append("execution_policy_missing")
        else:
            if execution.get("prebuilt_image_policy") != PREBUILT_IMAGE_POLICY_VERSION:
                issues.append("prebuilt_image_policy_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("runner_agent_registry_isolation")
                != RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ):
                issues.append("runner_agent_registry_isolation_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("development_prewarm") != DEVELOPMENT_PREWARM_VERSION
            ):
                issues.append("development_prewarm_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("trial_timeout_policy")
                != TRIAL_TIMEOUT_POLICY_VERSION
            ):
                issues.append("trial_timeout_policy_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("provider_failure_policy")
                != PROVIDER_FAILURE_POLICY_VERSION
            ):
                issues.append("provider_failure_policy_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("ephemeral_auth_cleanup")
                != EPHEMERAL_AUTH_CLEANUP_VERSION
            ):
                issues.append("ephemeral_auth_cleanup_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("training_evidence_policy")
                != TRAINING_EVIDENCE_POLICY_VERSION
            ):
                issues.append("training_evidence_policy_mismatch")
            if execution.get("agent_runtime_builder") != SHARED_AGENT_RUNTIME_BUILDER_IMAGE:
                issues.append("agent_runtime_builder_mismatch")
            if execution.get("agent_runtime_package") != SHARED_CODEX_CLI_PACKAGE:
                issues.append("agent_runtime_package_mismatch")
            if execution.get("agent_runtime_version") != SHARED_CODEX_CLI_VERSION:
                issues.append("agent_runtime_version_mismatch")
            if (
                execution.get("proposal_candidate_selection")
                != TRAIN_ONLY_CANDIDATE_SELECTION_VERSION
            ):
                issues.append("proposal_candidate_selection_mismatch")
            if execution.get("runtime_candidate_kinds") != ["task", "policy"]:
                issues.append("runtime_candidate_kinds_mismatch")
            if (
                execution.get("evaluator_hypothesis_mode")
                != "separate_epoch_challenger_not_in_primary_runtime"
            ):
                issues.append("evaluator_hypothesis_mode_mismatch")
            if execution.get("skill_routing") != SKILL_ROUTING_VERSION:
                issues.append("skill_routing_mismatch")
            if execution.get("verifier_isolation") != VERIFIER_ISOLATION_VERSION:
                issues.append("verifier_isolation_mismatch")
            if execution.get("parallel_unit") != "benchmark_item":
                issues.append("parallel_unit_invalid")
            if execution.get("within_pair_execution") != "sequential_balanced_order":
                issues.append("within_pair_execution_invalid")
            if (
                major is not None
                and major >= 3
                and execution.get("provider_route_policy")
                != PROVIDER_ROUTE_POLICY_VERSION
            ):
                issues.append("provider_route_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("openai_compatible_codex_config")
                != OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
            ):
                issues.append("openai_compatible_codex_config_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("counterfactual_replay_policy")
                != COUNTERFACTUAL_REPLAY_POLICY_VERSION
            ):
                issues.append("counterfactual_replay_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("training_evidence_replay_policy")
                != TRAINING_EVIDENCE_REPLAY_POLICY_VERSION
            ):
                issues.append("training_evidence_replay_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("invalid_trial_retry_policy")
                != INVALID_TRIAL_RETRY_POLICY_VERSION
            ):
                issues.append("invalid_trial_retry_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and not 1 <= int(execution.get("invalid_trial_max_attempts") or 0) <= 5
            ):
                issues.append("invalid_trial_max_attempts_invalid")
            if (
                major is not None
                and major >= 3
                and not _is_nonnegative_number(
                    execution.get("invalid_trial_retry_backoff_seconds")
                )
            ):
                issues.append("invalid_trial_retry_backoff_invalid")
            if (
                major is not None
                and major >= 3
                and not 1 <= int(execution.get("invalid_trial_retry_workers") or 0) <= 4
            ):
                issues.append("invalid_trial_retry_workers_invalid")
            if (
                major is not None
                and major >= 3
                and execution.get("local_evidence_transport")
                != LOCAL_EVIDENCE_TRANSPORT_VERSION
            ):
                issues.append("local_evidence_transport_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("network_scope_audit")
                != NETWORK_SCOPE_AUDIT_VERSION
            ):
                issues.append("network_scope_audit_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("proposal_failure_isolation_policy")
                != PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION
            ):
                issues.append("proposal_failure_isolation_policy_mismatch")
        evolution = self.payload.get("evolution")
        if not isinstance(evolution, Mapping):
            issues.append("evolution_budget_missing")
        else:
            if not 1 <= int(evolution.get("max_generations") or 0) <= 10:
                issues.append("evolution_generation_budget_invalid")
            if not 1 <= int(evolution.get("max_consecutive_non_promotions") or 0) <= int(
                evolution.get("max_generations") or 0
            ):
                issues.append("evolution_early_stop_invalid")
            if not 1 <= int(evolution.get("proposal_candidates_per_generation") or 0) <= 10:
                issues.append("evolution_candidate_budget_invalid")
        phases = self.payload.get("phases")
        if not isinstance(phases, Mapping):
            issues.append("phases_missing")
        else:
            sealed = phases.get("sealed_test")
            if not isinstance(sealed, Mapping) or sealed.get("single_access") is not True:
                issues.append("sealed_test_not_single_access")
            if not isinstance(sealed, Mapping) or int(sealed.get("repeats") or 0) < 1:
                issues.append("sealed_test_repeats_missing")
        controls = self.payload.get("controls")
        if not isinstance(controls, list) or not controls:
            issues.append("controls_missing")
        else:
            control_ids = [str(row.get("id") or "") for row in controls if isinstance(row, Mapping)]
            if len(control_ids) != len(set(control_ids)):
                issues.append("duplicate_control_id")
            for required in ("raw_no_skill", "promoted_v2"):
                if required not in control_ids:
                    issues.append(f"required_control_missing:{required}")
        statistics = self.payload.get("statistics")
        if not isinstance(statistics, Mapping):
            issues.append("statistics_missing")
        elif statistics.get("analysis_unit") != "benchmark_item":
            issues.append("analysis_unit_not_item")
        subset = self.payload.get("benchmark_subset")
        if subset is not None:
            if not isinstance(subset, Mapping):
                issues.append("benchmark_subset_invalid")
            elif subset.get("policy") not in {
                "full_inventory_v1",
                "exclude_external_credentials_by_family_v1",
            }:
                issues.append("benchmark_subset_policy_invalid")
        return sorted(set(issues))


def _protocol_major(value: Any) -> int | None:
    try:
        return int(str(value).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _is_nonnegative_number(value: Any) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def build_protocol_lock(
    protocol: PaperProtocol,
    *,
    project_root: str | Path,
    benchmark_root: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    benchmark = Path(benchmark_root).expanduser().resolve()
    primary_path = project / str(protocol.payload["primary_manifest"])
    secondary_path = project / str(protocol.payload["secondary_manifest"])
    primary = SplitManifest.read(primary_path)
    secondary = SplitManifest.read(secondary_path)
    adapter = SkillLearnBenchAdapter(benchmark)
    inventory = adapter.discover()
    subset_policy = dict(protocol.payload.get("benchmark_subset") or {})
    if subset_policy.get("policy") == "exclude_external_credentials_by_family_v1":
        eligible_inventory = adapter.credential_independent_items()
        subset_summary = adapter.credential_independent_summary()
    else:
        eligible_inventory = inventory
        subset_summary = {
            "policy": "full_inventory_v1",
            "eligible_instance_count": len(inventory),
            "excluded_instance_count": 0,
            "excluded_families": [],
            "excluded_required_env_names": [],
            "secret_value_persisted": False,
        }
    eligible_ids = {row.id for row in eligible_inventory}
    primary_ids = {*primary.train_ids, *primary.validation_ids, *primary.test_ids}
    secondary_ids = {*secondary.train_ids, *secondary.validation_ids, *secondary.test_ids}
    issues: list[str] = []
    if primary_ids != eligible_ids:
        issues.append("primary_manifest_inventory_mismatch")
    if secondary_ids != eligible_ids:
        issues.append("secondary_manifest_inventory_mismatch")
    for key in (
        "policy",
        "eligible_instance_count",
        "excluded_instance_count",
        "excluded_families",
        "excluded_required_env_names",
    ):
        if key in subset_policy and subset_policy.get(key) != subset_summary.get(key):
            issues.append(f"benchmark_subset_mismatch:{key}")
    if _counts(primary) != dict(protocol.payload["expected_primary_counts"]):
        issues.append("primary_count_mismatch")
    if _counts(secondary) != dict(protocol.payload["expected_secondary_counts"]):
        issues.append("secondary_count_mismatch")
    if configured_model() != protocol.payload["model"]:
        issues.append("configured_model_mismatch")
    if list(configured_provider_chain()) != list(protocol.payload["proposal_provider_chain"]):
        issues.append("proposal_provider_chain_mismatch")
    trial_provider_mode = configured_skilllearn_provider_mode()
    if trial_provider_mode != protocol.payload["trial_provider_mode"]:
        issues.append("trial_provider_mode_mismatch")
    api_origin = configured_api_origin()
    expected_api_origin = str(protocol.payload.get("provider_endpoint_origin") or "")
    if expected_api_origin and api_origin != expected_api_origin:
        issues.append("configured_provider_endpoint_origin_mismatch")
    static_program_path = project / "baselines" / "static_generic_program.json"
    static_program = HypothesisProgram.from_dict(
        json.loads(static_program_path.read_text(encoding="utf-8"))
    )
    static_issues = static_program.validate()
    if static_issues:
        issues.extend(f"static_program:{issue}" for issue in static_issues)
    source_issues = _validate_control_sources(protocol, project)
    issues.extend(source_issues)
    code_fingerprint = _code_fingerprint(project)
    git_state = _git_state(project)
    preflight = build_preflight(
        benchmark,
        trial_provider_mode=trial_provider_mode,
        item_ids=eligible_ids,
    )
    provider_status = proposal_provider_status()
    claim_eligible = not issues and not git_state["scoped_dirty"] and not preflight["blockers"]
    lock = {
        "lock_version": (
            "paper_protocol_lock_v2" if expected_api_origin else "paper_protocol_lock_v1"
        ),
        "protocol_id": protocol.id,
        "protocol_hash": protocol.protocol_hash,
        "primary_manifest_hash": primary.manifest_hash,
        "secondary_manifest_hash": secondary.manifest_hash,
        "primary_counts": _counts(primary),
        "secondary_counts": _counts(secondary),
        "inventory_count": len(inventory),
        "inventory_hash": stable_hash(
            {"item_hashes": sorted(row.id_hash for row in inventory)}
        ),
        "eligible_inventory_count": len(eligible_inventory),
        "eligible_inventory_hash": stable_hash(
            {"item_hashes": sorted(row.id_hash for row in eligible_inventory)}
        ),
        "benchmark_subset": subset_summary,
        "model": configured_model(),
        "proposal_provider_chain": list(configured_provider_chain()),
        "trial_provider_mode": trial_provider_mode,
        "provider_endpoint_origin": api_origin or None,
        "provider_status": provider_status,
        "max_steps": int(protocol.payload["max_steps"]),
        "execution": dict(protocol.payload["execution"]),
        "evolution": dict(protocol.payload["evolution"]),
        "static_program_hash": static_program.payload_hash,
        "code_fingerprint": code_fingerprint,
        "git": git_state,
        "preflight": preflight,
        "validation_issues": sorted(set(issues)),
        "claim_eligible": claim_eligible,
        "sealed_test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
    }
    lock["lock_hash"] = stable_hash(lock)
    return lock


def _counts(manifest: SplitManifest) -> dict[str, int]:
    return {
        "train": len(manifest.train_ids),
        "validation": len(manifest.validation_ids),
        "test": len(manifest.test_ids),
    }


def _validate_control_sources(
    protocol: PaperProtocol,
    project_root: Path,
) -> list[str]:
    issues: list[str] = []
    dynamic = {"none", "no_recursive_archive_incumbent", "frozen_archive_incumbent"}
    for row in protocol.payload["controls"]:
        if not isinstance(row, Mapping):
            issues.append("malformed_control")
            continue
        source = str(row.get("source") or "")
        if source in dynamic:
            continue
        if not (project_root / source).exists():
            issues.append(f"control_source_missing:{row.get('id')}")
    return issues


def _code_fingerprint(project_root: Path) -> dict[str, Any]:
    roots = (
        project_root / "assumption_agent",
        project_root / "tests",
        project_root / "scripts",
        project_root / "baselines",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    files.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "ARCHITECTURE.md",
            project_root / "BENCHMARK_PROTOCOL.md",
        )
        if path.is_file()
    )
    rows = [
        {
            "path": str(path.relative_to(project_root)),
            "content_hash": stable_hash({"bytes": path.read_bytes().hex()}),
        }
        for path in sorted(set(files))
    ]
    return {
        "file_count": len(rows),
        "tree_hash": stable_hash(rows),
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    repository_root = project_root.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        relative = str(project_root.relative_to(repository_root))
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"commit": "", "scoped_dirty": True, "scoped_change_count": -1}
    return {
        "commit": commit,
        "scoped_dirty": bool(status),
        "scoped_change_count": len(status),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and audit the paper experiment protocol.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-claim-eligible", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    map_legacy_model_env()
    protocol = PaperProtocol.read(args.protocol)
    lock = build_protocol_lock(
        protocol,
        project_root=args.project_root,
        benchmark_root=args.benchmark_root,
    )
    _write_json(args.out, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    if args.require_claim_eligible and not lock["claim_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
