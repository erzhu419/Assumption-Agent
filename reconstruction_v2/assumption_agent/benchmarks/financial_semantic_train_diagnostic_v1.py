from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

from ..events import JsonlEventSink
from ..models import stable_hash
from .financial_semantic_operator_v1 import (
    FINANCIAL_QUERY_RECEIPT_VERSION,
    OfflineFinancialQA,
    build_financial_semantic_plan,
    load_financial_semantic_asset,
)
from .offline_verifier import (
    OFFLINE_VERIFIER_MOUNT,
    SkillLearnOfflineVerifierRuntimeCache,
)
from .semantic_assignment_operator_v1 import OfflineMiniLMEncoder
from .skilllearn_lifecycle import (
    SkillLearnPrebuiltImage,
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
)
from .train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
    V320_MANIFEST_RELATIVE_PATH,
    _load_raw_projection,
)
from .v320_train_candidate_material_v2 import V320_SOURCE_RELATIVE_ROOT


FINANCIAL_TRAIN_DIAGNOSTIC_VERSION = (
    "financial_semantic_consumed_train_offline_diagnostic_v1"
)
FORMATION_ITEM_IDS = (
    "financial-analysis-1",
    "financial-analysis-3",
    "financial-analysis-5",
)
MAXIMUM_CONCURRENT_CONTAINERS = 3
REPORT_FILENAME = "financial_semantic_train_diagnostic.report.json"
EVENTS_FILENAME = "financial_semantic_train_diagnostic.events.jsonl"


class FinancialTrainDiagnosticError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_file_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FinancialTrainDiagnosticError(
                "diagnostic closure contains a symbolic link"
            )
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not rows:
        raise FinancialTrainDiagnosticError("diagnostic closure is empty")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_checked(
    command: Sequence[str],
    *,
    timeout: float = 900.0,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise FinancialTrainDiagnosticError(
            "container command failed: "
            + hashlib.sha256(
                (completed.stdout + completed.stderr).encode("utf-8")
            ).hexdigest()
        )
    return completed


def _verify_json_self_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = payload.get(field)
    body = dict(payload)
    body.pop(field, None)
    if declared != stable_hash(body):
        raise FinancialTrainDiagnosticError(f"{field} verification failed")


def _run_one(
    *,
    item_id: str,
    item_path: Path,
    plan_path: Path,
    item_output: Path,
    image: SkillLearnPrebuiltImage,
    verifier_volume: str,
    verifier_command: str,
    operator_source: Path,
    expected_operator_source_sha256: str,
    expected_candidate_id: str,
    expected_candidate_manifest_hash: str,
    expected_plan_hash: str,
    execution_closure: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    container_name = (
        "financial-semantic-train-"
        + stable_hash({"item_id": item_id, "nonce": uuid.uuid4().hex})[:20]
    )
    created = False
    try:
        _run_checked(
            [
                "docker",
                "run",
                "--detach",
                "--pull",
                "never",
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "4096m",
                "--name",
                container_name,
                "-v",
                f"{verifier_volume}:{OFFLINE_VERIFIER_MOUNT}:ro",
                image.image_id,
                "sleep",
                "infinity",
            ]
        )
        created = True
        inspected_image = _run_checked(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Image}}",
                container_name,
            ]
        ).stdout.strip()
        if inspected_image != image.image_id:
            raise FinancialTrainDiagnosticError(
                "diagnostic container image identity drifted"
            )
        container_operator = "/tmp/financial_semantic_operator_v1.py"
        container_plan = "/tmp/financial_semantic_plan.json"
        container_receipt = "/tmp/financial_semantic_query_receipt.json"
        _run_checked(
            ["docker", "cp", str(operator_source), f"{container_name}:{container_operator}"]
        )
        _run_checked(
            ["docker", "cp", str(plan_path), f"{container_name}:{container_plan}"]
        )
        readback = _run_checked(
            ["docker", "exec", container_name, "sha256sum", container_operator]
        ).stdout.split()[0]
        if readback != expected_operator_source_sha256:
            raise FinancialTrainDiagnosticError("container operator source drifted")
        _run_checked(
            [
                "docker",
                "exec",
                container_name,
                "python3",
                container_operator,
                "execute",
                "--plan",
                container_plan,
                "--q2-root",
                "/root/2025-q2",
                "--q3-root",
                "/root/2025-q3",
                "--output",
                "/root/answers.json",
                "--receipt-output",
                container_receipt,
            ],
        )
        item_output.mkdir(parents=True, exist_ok=False)
        receipt_path = item_output / "query.receipt.json"
        answers_path = item_output / "answers.json"
        _run_checked(
            ["docker", "cp", f"{container_name}:{container_receipt}", str(receipt_path)]
        )
        _run_checked(
            ["docker", "cp", f"{container_name}:/root/answers.json", str(answers_path)]
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise FinancialTrainDiagnosticError("query receipt is malformed")
        _verify_json_self_hash(receipt, "receipt_hash")
        if (
            receipt.get("receipt_version") != FINANCIAL_QUERY_RECEIPT_VERSION
            or receipt.get("operator_source_sha256")
            != expected_operator_source_sha256
            or receipt.get("candidate_id") != expected_candidate_id
            or receipt.get("candidate_manifest_hash")
            != expected_candidate_manifest_hash
            or receipt.get("plan_hash") != expected_plan_hash
            or receipt.get("network_calls") != 0
            or receipt.get("verifier_content_accessed") is not False
        ):
            raise FinancialTrainDiagnosticError("query receipt boundary failed")
        if receipt.get("output_sha256") != _sha256_file(answers_path):
            raise FinancialTrainDiagnosticError("archived answers binding failed")
        # Materialize tests only after the typed query has completed.  This
        # preserves the same no-verifier-before-action boundary as production.
        _run_checked(["docker", "exec", container_name, "mkdir", "/tests"])
        _run_checked(
            [
                "docker",
                "cp",
                f"{item_path / 'tests'}/.",
                f"{container_name}:/tests",
            ]
        )
        _run_checked(
            ["docker", "exec", container_name, "sh", "-lc", verifier_command],
            timeout=930.0,
        )
        _run_checked(
            [
                "docker",
                "cp",
                f"{container_name}:/logs/verifier",
                str(item_output),
            ]
        )
        reward_path = item_output / "verifier" / "reward.txt"
        ctrf_path = item_output / "verifier" / "ctrf.json"
        if not reward_path.is_file() or not ctrf_path.is_file():
            raise FinancialTrainDiagnosticError("offline verifier receipt is incomplete")
        reward = reward_path.read_text(encoding="utf-8").strip()
        if reward not in {"0", "1"}:
            raise FinancialTrainDiagnosticError("offline verifier reward is invalid")
        return {
            "item_id": item_id,
            "item_id_hash": stable_hash({"item_id": item_id}),
            "candidate_success": reward == "1",
            "candidate_score": float(reward),
            "plan_sha256": _sha256_file(plan_path),
            "plan_hash": receipt["plan_hash"],
            "query_receipt_hash": receipt["receipt_hash"],
            "output_sha256": receipt["output_sha256"],
            "archived_answers_sha256": _sha256_file(answers_path),
            "verifier_reward_sha256": _sha256_file(reward_path),
            "verifier_ctrf_sha256": _sha256_file(ctrf_path),
            "offline_verifier": True,
            "container_network": "none",
            "container_image_id": inspected_image,
            "execution_closure": dict(execution_closure),
            "execution_closure_hash": stable_hash(dict(execution_closure)),
            "duration_seconds": round(time.monotonic() - started, 6),
        }
    finally:
        if created:
            subprocess.run(
                ["docker", "rm", "--force", container_name],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )


def run_financial_train_diagnostic_v1(
    *,
    project_root: str | Path,
    output_root: str | Path,
    asset_path: str | Path,
    minilm_runtime_asset_path: str | Path,
    minilm_snapshot_root: str | Path,
    qa_runtime_asset_path: str | Path,
    qa_snapshot_root: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("financial diagnostic output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    event_sink = JsonlEventSink(destination / EVENTS_FILENAME)
    try:
        benchmark = (project / SKILLLEARN_BENCHMARK_RELATIVE_ROOT).resolve(
            strict=True
        )
        asset = load_financial_semantic_asset(
            asset_path,
            minilm_runtime_asset_path=minilm_runtime_asset_path,
            qa_runtime_asset_path=qa_runtime_asset_path,
        )
        encoder = OfflineMiniLMEncoder(
            runtime_asset_path=minilm_runtime_asset_path,
            snapshot_root=minilm_snapshot_root,
        )
        qa = OfflineFinancialQA(
            runtime_asset_path=qa_runtime_asset_path,
            snapshot_root=qa_snapshot_root,
        )
        plans_root = destination / "plans"
        plans_root.mkdir()
        plan_paths: dict[str, Path] = {}
        plans: dict[str, Mapping[str, Any]] = {}
        extraction_receipts: dict[str, Mapping[str, Any]] = {}
        extraction_root = destination / "extraction_receipts"
        extraction_root.mkdir()
        for item_id in FORMATION_ITEM_IDS:
            instruction = (
                benchmark
                / "tasks"
                / "financial-analysis"
                / item_id
                / "instruction.md"
            ).read_text(encoding="utf-8")
            plan, extraction_receipt = build_financial_semantic_plan(
                instruction=instruction,
                asset=asset,
                encoder=encoder,
                qa=qa,
                minilm_runtime_receipt=encoder.runtime_receipt,
                qa_runtime_receipt=qa.runtime_receipt,
            )
            plan_path = plans_root / f"{item_id}.plan.json"
            _write_json(plan_path, plan)
            plan_paths[item_id] = plan_path
            plans[item_id] = plan
            extraction_receipts[item_id] = extraction_receipt
            _write_json(
                extraction_root / f"{item_id}.extraction.receipt.json",
                extraction_receipt,
            )
        del encoder, qa

        prebuilt_cache = SkillLearnPrebuiltImageCache(
            benchmark,
            cache_only=True,
            event_sink=event_sink,
        )
        verifier_cache = SkillLearnOfflineVerifierRuntimeCache(
            event_sink=event_sink
        )
        backend = SkillLearnSubprocessBackend(
            benchmark,
            prebuilt_cache=prebuilt_cache,
            offline_verifier_cache=verifier_cache,
            event_sink=event_sink,
        )
        environments: dict[str, tuple[SkillLearnPrebuiltImage, Any]] = {}
        for item_id in FORMATION_ITEM_IDS:
            image, verifier = backend.prewarm_trial_environment(
                family="financial-analysis",
                item_id=item_id,
                trace_id=f"financial-semantic-train:{item_id}:prewarm",
            )
            if verifier is None:
                raise FinancialTrainDiagnosticError(
                    "financial offline verifier runtime is absent"
                )
            environments[item_id] = (image, verifier)

        operator_source = Path(
            __file__
        ).resolve().with_name("financial_semantic_operator_v1.py")
        if _sha256_file(operator_source) != asset["operator_source_sha256"]:
            raise FinancialTrainDiagnosticError("operator source changed after freeze")
        benchmark_commit = _run_checked(
            ["git", "-C", str(benchmark), "rev-parse", "HEAD"]
        ).stdout.strip()
        if len(benchmark_commit) != 40:
            raise FinancialTrainDiagnosticError("benchmark commit is malformed")
        trials_root = destination / "trials"
        trials_root.mkdir()
        futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAXIMUM_CONCURRENT_CONTAINERS
        ) as pool:
            for item_id in FORMATION_ITEM_IDS:
                image, verifier = environments[item_id]
                item_path = (
                    benchmark / "tasks" / "financial-analysis" / item_id
                )
                environment_rows = _directory_file_rows(
                    item_path / "environment"
                )
                test_rows = _directory_file_rows(item_path / "tests")
                execution_closure = {
                    "closure_version": (
                        "financial_semantic_formation_execution_closure_v1"
                    ),
                    "benchmark_commit": benchmark_commit,
                    "item_id_hash": stable_hash({"item_id": item_id}),
                    "instruction_sha256": _sha256_file(
                        item_path / "instruction.md"
                    ),
                    "task_toml_sha256": _sha256_file(item_path / "task.toml"),
                    "environment_file_rows": environment_rows,
                    "environment_file_set_hash": stable_hash(environment_rows),
                    "test_file_rows": test_rows,
                    "test_file_set_hash": stable_hash(test_rows),
                    "image": {
                        "image_id": image.image_id,
                        "cache_key": image.cache_key,
                        "environment_hash": image.environment_hash,
                        "source_environment_hash": image.source_environment_hash,
                        "agent_runtime_key": image.agent_runtime_key,
                        "agent_runtime_version": image.agent_runtime_version,
                        "task_input_closure_required": (
                            image.task_input_closure_required
                        ),
                        "task_input_closure_policy": (
                            image.task_input_closure_policy
                        ),
                        "task_input_closure_hash": image.task_input_closure_hash,
                        "task_input_build_context_receipt_hash": (
                            image.task_input_build_context_receipt_hash
                        ),
                        "task_input_integrity_receipt_hash": (
                            image.task_input_integrity_receipt_hash
                        ),
                    },
                    "offline_verifier": {
                        "profile_id": verifier.profile.profile_id,
                        "profile_hash": verifier.profile.profile_hash,
                        "runtime_key": verifier.runtime_key,
                        "base_image_id": verifier.base_image_id,
                        "volume_name_hash": stable_hash(
                            {"volume_name": verifier.volume_name}
                        ),
                        "verifier_command_hash": stable_hash(
                            {"command": verifier.profile.verifier_command}
                        ),
                    },
                }
                future = pool.submit(
                    _run_one,
                    item_id=item_id,
                    item_path=item_path,
                    plan_path=plan_paths[item_id],
                    item_output=trials_root / item_id,
                    image=image,
                    verifier_volume=verifier.volume_name,
                    verifier_command=verifier.profile.verifier_command,
                    operator_source=operator_source,
                    expected_operator_source_sha256=asset[
                        "operator_source_sha256"
                    ],
                    expected_candidate_id=asset["candidate_id"],
                    expected_candidate_manifest_hash=asset["manifest_hash"],
                    expected_plan_hash=str(plans[item_id]["plan_hash"]),
                    execution_closure=execution_closure,
                )
                futures[future] = item_id
            rows = [future.result() for future in concurrent.futures.as_completed(futures)]
        rows.sort(key=lambda row: row["item_id"])

        raw_projection = _load_raw_projection(
            project_root=project,
            source_root=(project / V320_SOURCE_RELATIVE_ROOT).resolve(strict=True),
            manifest_path=(project / V320_MANIFEST_RELATIVE_PATH).resolve(
                strict=True
            ),
        )
        historical = {
            row.item_id: row
            for row in raw_projection.baseline_set.rows
            if row.item_id in FORMATION_ITEM_IDS
        }
        if set(historical) != set(FORMATION_ITEM_IDS):
            raise FinancialTrainDiagnosticError("historical RAW cohort is incomplete")
        for row in rows:
            baseline = historical[row["item_id"]]
            row.update(
                {
                    "historical_raw_success": baseline.success,
                    "historical_raw_score": baseline.score_units / 1_000_000,
                    "historical_raw_evidence_hash": baseline.baseline_evidence_hash,
                    "gain": bool(row["candidate_success"] and not baseline.success),
                    "harm": bool(not row["candidate_success"] and baseline.success),
                    "extraction_receipt_hash": extraction_receipts[
                        row["item_id"]
                    ]["receipt_hash"],
                }
            )
        candidate_successes = sum(bool(row["candidate_success"]) for row in rows)
        raw_successes = sum(bool(row["historical_raw_success"]) for row in rows)
        event_ledger_path = destination / EVENTS_FILENAME
        if not event_ledger_path.is_file():
            raise FinancialTrainDiagnosticError("event ledger is absent")
        evidence_tree_rows = _directory_file_rows(destination)
        report: dict[str, Any] = {
            "report_version": FINANCIAL_TRAIN_DIAGNOSTIC_VERSION,
            "candidate_id": asset["candidate_id"],
            "candidate_manifest_hash": asset["manifest_hash"],
            "operator_source_sha256": asset["operator_source_sha256"],
            "formation_item_ids": list(FORMATION_ITEM_IDS),
            "formation_item_count": len(FORMATION_ITEM_IDS),
            "rows": rows,
            "row_set_hash": stable_hash(rows),
            "candidate_success_count": candidate_successes,
            "historical_raw_success_count": raw_successes,
            "gain_count": sum(bool(row["gain"]) for row in rows),
            "harm_count": sum(bool(row["harm"]) for row in rows),
            "all_candidate_trials_passed": candidate_successes == len(rows),
            "retrospective_formation_replay_gain": (
                candidate_successes > raw_successes
            ),
            "in_sample_formation_replay": True,
            "cross_fit": False,
            "causal_gain_claim_authorized": False,
            "prospective_claim_authorized": False,
            "offline_verifier_only": True,
            "generative_agent_calls": 0,
            "online_judge_calls": 0,
            "online_calls": 0,
            "offline_semantic_extractor": "MiniLM_plus_DistilBERT_QA",
            "maximum_concurrent_containers": MAXIMUM_CONCURRENT_CONTAINERS,
            "benchmark_commit": benchmark_commit,
            "historical_raw_projection_receipt_hash": (
                raw_projection.receipt.receipt_hash
            ),
            "historical_raw_projected_observation_set_hash": (
                raw_projection.receipt.projected_observation_set_hash
            ),
            "event_ledger_sha256": _sha256_file(event_ledger_path),
            "execution_closure_set_hash": stable_hash(
                sorted(row["execution_closure_hash"] for row in rows)
            ),
            "evidence_tree_rows": evidence_tree_rows,
            "evidence_tree_hash": stable_hash(evidence_tree_rows),
            "validation_content_accessed": False,
            "sealed_content_accessed": False,
            "raw_instruction_logged": False,
        }
        report["report_hash"] = stable_hash(report)
        _write_json(destination / REPORT_FILENAME, report)
        return report
    except Exception:
        failure = {
            "report_version": "financial_semantic_train_diagnostic_failure_v1",
            "validation_content_accessed": False,
            "sealed_content_accessed": False,
            "online_calls": 0,
        }
        failure["report_hash"] = stable_hash(failure)
        _write_json(destination / "financial_semantic_train_diagnostic.failure.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--asset",
        type=Path,
        default=Path("manifests/financial_semantic_operator_asset_v1.json"),
    )
    parser.add_argument(
        "--minilm-runtime-asset",
        type=Path,
        default=Path("manifests/semantic_assignment_minilm_runtime_asset_v1.json"),
    )
    parser.add_argument("--minilm-snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--qa-runtime-asset",
        type=Path,
        default=Path("manifests/financial_distilbert_qa_runtime_asset_v1.json"),
    )
    parser.add_argument("--qa-snapshot-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_financial_train_diagnostic_v1(
        project_root=args.project_root,
        output_root=args.output_root,
        asset_path=args.asset,
        minilm_runtime_asset_path=args.minilm_runtime_asset,
        minilm_snapshot_root=args.minilm_snapshot_root,
        qa_runtime_asset_path=args.qa_runtime_asset,
        qa_snapshot_root=args.qa_snapshot_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
