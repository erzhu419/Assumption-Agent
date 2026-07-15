"""One-shot, offline TRAIN diagnostic for the frozen SC-100 operator.

The diagnostic is intentionally not an OOF estimate.  It replays three already
consumed TRAIN cases (two historical RAW failures and one success control) to
ask a narrow causal question: can the frozen executable operator gain on both
failures without harming the control?  Candidate generation is completed for
all items before any verifier path is supplied, and the three official offline
verifiers then run concurrently with Docker networking disabled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from ..models import stable_hash
from .offline_verifier import (
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
    probe_offline_verifier_runtime,
)
from .sc100_typed_operator_v1 import execute


DIAGNOSTIC_VERSION = "sc100_historically_informed_train_diagnostic_v1"
ITEM_IDS = (
    "court-form-filling-3",
    "court-form-filling-4",
    "court-form-filling-6",
)


class SC100DiagnosticError(RuntimeError):
    """Raised when the frozen diagnostic contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SC100DiagnosticError(f"JSON object required: {path}")
    return payload


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = payload.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        raise SC100DiagnosticError(f"missing {field}")
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise SC100DiagnosticError(f"{field} mismatch")
    return declared


def _resolve_under(project: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SC100DiagnosticError("relative path is missing")
    candidate = (project / relative).resolve()
    try:
        candidate.relative_to(project)
    except ValueError as exc:
        raise PermissionError("path escapes project root") from exc
    return candidate


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_bound_files(project: Path, preregistration: Mapping[str, Any]) -> None:
    bindings = preregistration.get("file_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise SC100DiagnosticError("file bindings are missing")
    seen: set[str] = set()
    for row in bindings:
        if not isinstance(row, Mapping):
            raise SC100DiagnosticError("file binding row is malformed")
        relative = row.get("path")
        expected = row.get("sha256")
        if not isinstance(relative, str) or relative in seen:
            raise SC100DiagnosticError("file binding path is invalid")
        seen.add(relative)
        path = _resolve_under(project, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise SC100DiagnosticError(f"bound file mismatch: {relative}")


def _preflight(project: Path, preregistration: Mapping[str, Any]) -> None:
    if preregistration.get("preregistration_version") != DIAGNOSTIC_VERSION:
        raise SC100DiagnosticError("diagnostic version mismatch")
    if preregistration.get("decision_budget") != 1:
        raise SC100DiagnosticError("decision budget must equal one")
    _verify_bound_files(project, preregistration)

    data = preregistration.get("data_boundary")
    if not isinstance(data, Mapping) or tuple(data.get("train_item_ids") or ()) != ITEM_IDS:
        raise SC100DiagnosticError("TRAIN item cohort mismatch")
    if any(
        data.get(flag) is not False
        for flag in (
            "validation_split_accessed",
            "test_split_accessed",
            "sealed_split_accessed",
        )
    ):
        raise PermissionError("non-TRAIN split is not closed")

    split_path = _resolve_under(project, data.get("split_manifest_path"))
    split_payload = _read_json(split_path)
    if split_payload.get("manifest_hash") != data.get("split_manifest_hash"):
        raise SC100DiagnosticError("split manifest identity mismatch")
    train = set(split_payload.get("train_ids") or ())
    validation = set(split_payload.get("validation_ids") or ())
    test = set(split_payload.get("test_ids") or ())
    if not set(ITEM_IDS).issubset(train) or set(ITEM_IDS) & (validation | test):
        raise PermissionError("SC-100 cohort is not TRAIN-only")

    runtime = preregistration.get("runtime")
    if not isinstance(runtime, Mapping):
        raise SC100DiagnosticError("runtime binding is missing")
    host_assets = runtime.get("host_assets")
    if not isinstance(host_assets, list) or not host_assets:
        raise SC100DiagnosticError("host runtime assets are missing")
    for asset in host_assets:
        if not isinstance(asset, Mapping):
            raise SC100DiagnosticError("host runtime asset is malformed")
        path = Path(str(asset.get("path") or "")).expanduser().resolve(strict=True)
        if not path.is_file() or _sha256(path) != asset.get("sha256"):
            raise SC100DiagnosticError("host runtime asset drift")
    import fitz

    if fitz.__version__ != runtime.get("pymupdf_version"):
        raise SC100DiagnosticError("PyMuPDF version drift")
    profile = offline_verifier_profile_for_family("court-form-filling")
    if (
        profile.profile_id != runtime.get("offline_verifier_profile_id")
        or profile.profile_hash != runtime.get("offline_verifier_profile_hash")
        or offline_verifier_runtime_key(profile=profile)
        != runtime.get("offline_verifier_runtime_key")
    ):
        raise SC100DiagnosticError("offline verifier profile drift")

    image = subprocess.run(
        ["docker", "image", "inspect", str(runtime.get("base_image_id"))],
        check=False,
        capture_output=True,
        text=True,
    )
    if image.returncode != 0:
        raise SC100DiagnosticError("frozen offline image is unavailable")
    image_payload = json.loads(image.stdout)[0]
    if image_payload.get("Id") != runtime.get("base_image_id"):
        raise SC100DiagnosticError("offline image ID drift")
    volume = subprocess.run(
        ["docker", "volume", "inspect", str(runtime.get("offline_verifier_volume"))],
        check=False,
        capture_output=True,
        text=True,
    )
    if volume.returncode != 0:
        raise SC100DiagnosticError("frozen offline verifier volume is unavailable")

    for row in preregistration.get("historical_raw_baseline", ()):  # type: ignore[arg-type]
        path = _resolve_under(project, row.get("result_path"))
        if _sha256(path) != row.get("result_sha256"):
            raise SC100DiagnosticError("historical RAW result drift")
        result = _read_json(path)
        if int(result.get("reward")) != int(row.get("reward")):
            raise SC100DiagnosticError("historical RAW reward drift")


def _formal_paths(
    project: Path, preregistration: Mapping[str, Any]
) -> dict[str, Path]:
    formal = preregistration.get("formal_paths")
    if not isinstance(formal, Mapping):
        raise SC100DiagnosticError("formal paths are missing")
    paths = {
        name: _resolve_under(project, formal.get(name))
        for name in ("root", "report", "decision_lock")
    }
    if paths["report"].parent != paths["root"] or paths["decision_lock"].parent != paths["root"]:
        raise PermissionError("formal report paths are not rooted together")
    return paths


def _input_rows(preregistration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = preregistration.get("items")
    if not isinstance(rows, list) or tuple(row.get("item_id") for row in rows) != ITEM_IDS:
        raise SC100DiagnosticError("item bindings are malformed")
    return {str(row["item_id"]): row for row in rows}


def _baseline_rows(preregistration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = preregistration.get("historical_raw_baseline")
    if not isinstance(rows, list) or tuple(row.get("item_id") for row in rows) != ITEM_IDS:
        raise SC100DiagnosticError("historical baseline bindings are malformed")
    return {str(row["item_id"]): row for row in rows}


def run_sc100_train_diagnostic(
    *,
    project_root: str | Path,
    preregistration_path: str | Path,
    operator: Callable[..., Mapping[str, Any]] = execute,
    verifier: Callable[..., Mapping[str, Any]] = probe_offline_verifier_runtime,
) -> dict[str, Any]:
    """Consume the sole preregistered diagnostic decision."""

    project = Path(project_root).expanduser().resolve(strict=True)
    prereg_path = Path(preregistration_path).expanduser().resolve(strict=True)
    preregistration = _read_json(prereg_path)
    preregistration_hash = _verify_self_hash(preregistration, "manifest_hash")
    _preflight(project, preregistration)
    paths = _formal_paths(project, preregistration)
    if paths["root"].exists():
        raise FileExistsError("formal SC-100 diagnostic root already exists")
    paths["root"].mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        paths["decision_lock"],
        {
            "lock_version": "sc100_train_diagnostic_decision_lock_v1",
            "state": "reserved",
            "decision_ordinal": 1,
            "preregistration_manifest_hash": preregistration_hash,
            "retry_authorized": False,
        },
    )

    item_rows = _input_rows(preregistration)
    baseline_rows = _baseline_rows(preregistration)
    workspace_root = paths["root"] / "workspaces"
    receipt_root = paths["root"] / "operator_receipts"
    verifier_root = paths["root"] / "verifier_reports"
    workspace_root.mkdir()
    receipt_root.mkdir()
    verifier_root.mkdir()

    def generate(item_id: str) -> dict[str, Any]:
        row = item_rows[item_id]
        workspace = workspace_root / item_id
        workspace.mkdir()
        blank_source = _resolve_under(project, row["blank_pdf_path"])
        blank_target = workspace / "sc100-blank.pdf"
        output_target = workspace / "sc100-filled.pdf"
        shutil.copyfile(blank_source, blank_target)
        try:
            receipt = dict(
                operator(
                    instruction=_resolve_under(project, row["instruction_path"]).read_text(
                        encoding="utf-8"
                    ),
                    blank_pdf=blank_target,
                    output_pdf=output_target,
                )
            )
        except Exception as exc:  # A generation error is a consumed FAIL decision.
            return {
                "item_id": item_id,
                "generated": False,
                "error_type": type(exc).__name__,
                "error_hash": stable_hash({"error": str(exc)}),
            }
        receipt_path = receipt_root / f"{item_id}.json"
        if receipt.get("plan_hash") != row.get("expected_plan_hash"):
            raise SC100DiagnosticError("compiled plan differs from the frozen item plan")
        _write_json_exclusive(receipt_path, receipt)
        if set(path.name for path in workspace.iterdir()) != {
            "sc100-blank.pdf",
            "sc100-filled.pdf",
        }:
            raise SC100DiagnosticError("verifier workspace contains unexpected files")
        return {
            "item_id": item_id,
            "generated": True,
            "operator_receipt_path": str(receipt_path.relative_to(project)),
            "operator_receipt_sha256": _sha256(receipt_path),
            "output_sha256": receipt["output_sha256"],
            "plan_hash": receipt["plan_hash"],
        }

    generation: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(generate, item_id): item_id for item_id in ITEM_IDS}
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                generation[item_id] = future.result()
            except Exception as exc:
                generation[item_id] = {
                    "item_id": item_id,
                    "generated": False,
                    "error_type": type(exc).__name__,
                    "error_hash": stable_hash({"error": str(exc)}),
                }

    # This is the first point at which a verifier tests directory is handed to
    # any invoked function.  All three generation futures are already joined.
    runtime = preregistration["runtime"]
    profile = offline_verifier_profile_for_family("court-form-filling")

    def verify(item_id: str) -> dict[str, Any]:
        report_path = verifier_root / f"{item_id}.json"
        try:
            report = dict(
                verifier(
                    profile=profile,
                    base_image_tag=str(runtime["base_image_id"]),
                    workspace=workspace_root / item_id,
                    tests_dir=_resolve_under(project, item_rows[item_id]["tests_dir"]),
                    report_path=report_path,
                    trace_id=f"sc100-typed-train-diagnostic:{item_id}",
                )
            )
        except Exception as exc:
            return {
                "item_id": item_id,
                "verifier_completed": False,
                "error_type": type(exc).__name__,
                "error_hash": stable_hash({"error": str(exc)}),
            }
        return {
            "item_id": item_id,
            "verifier_completed": True,
            "probe_passed": report.get("probe_passed") is True,
            "reward": report.get("reward"),
            "test_count": report.get("test_count"),
            "container_network": report.get("container_network"),
            "base_image_id": report.get("base_image_id"),
            "profile_hash": report.get("profile_hash"),
            "runtime_key": report.get("runtime_key"),
            "verifier_report_path": str(report_path.relative_to(project)),
            "verifier_report_sha256": _sha256(report_path),
            "verifier_receipt_hash": report.get("receipt_hash"),
        }

    verification: dict[str, dict[str, Any]] = {}
    generated_ids = [item_id for item_id in ITEM_IDS if generation[item_id]["generated"]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(verify, item_id): item_id for item_id in generated_ids}
        for future in as_completed(futures):
            item_id = futures[future]
            verification[item_id] = future.result()

    rows: list[dict[str, Any]] = []
    for item_id in ITEM_IDS:
        generated = generation[item_id]
        verified = verification.get(
            item_id,
            {
                "item_id": item_id,
                "verifier_completed": False,
                "not_run_reason": "generation_failed",
            },
        )
        baseline_reward = int(baseline_rows[item_id]["reward"])
        candidate_reward = verified.get("reward") if verified.get("probe_passed") else None
        rows.append(
            {
                "item_id": item_id,
                "baseline_reward": baseline_reward,
                "candidate_reward": candidate_reward,
                "gain": candidate_reward == 1 and baseline_reward == 0,
                "harm": candidate_reward == 0 and baseline_reward == 1,
                "generation": generated,
                "verification": verified,
            }
        )

    infrastructure_valid = all(
        row["verification"].get("probe_passed") is True
        and row["verification"].get("container_network") == "none"
        and row["verification"].get("base_image_id") == runtime["base_image_id"]
        and row["verification"].get("profile_hash")
        == runtime["offline_verifier_profile_hash"]
        and row["verification"].get("runtime_key")
        == runtime["offline_verifier_runtime_key"]
        for row in rows
    )
    gain_count = sum(bool(row["gain"]) for row in rows)
    harm_count = sum(bool(row["harm"]) for row in rows)
    all_candidate_pass = all(row["candidate_reward"] == 1 for row in rows)
    control_preserved = next(
        row for row in rows if row["item_id"] == "court-form-filling-3"
    )["candidate_reward"] == 1
    diagnostic_passed = bool(
        infrastructure_valid
        and all_candidate_pass
        and gain_count >= 2
        and harm_count == 0
        and control_preserved
    )

    report: dict[str, Any] = {
        "report_version": DIAGNOSTIC_VERSION,
        "preregistration_manifest_hash": preregistration_hash,
        "decision_ordinal": 1,
        "decision_budget": 1,
        "evaluation_class": "historically_informed_consumed_TRAIN_diagnostic",
        "candidate_claim": "closed_SC100_executable_representation_feasibility_only",
        "rows": rows,
        "item_count": len(rows),
        "generation_parallelism": 3,
        "verifier_parallelism": 3,
        "all_generation_joined_before_any_verifier_call": True,
        "agent_calls": 0,
        "model_calls": 0,
        "online_judge_calls": 0,
        "ruoli_calls": 0,
        "verifier_network": "none",
        "infrastructure_valid": infrastructure_valid,
        "gain_count": gain_count,
        "harm_count": harm_count,
        "all_candidate_pass": all_candidate_pass,
        "historical_success_control_preserved": control_preserved,
        "diagnostic_passed": diagnostic_passed,
        "oof_claim_authorized": False,
        "independent_holdout_claim_authorized": False,
        "incumbent_claim_authorized": False,
        "promotion_authorized": False,
        "validation_access_authorized": False,
        "sealed_access_authorized": False,
        "same_train_cohort_tuning_authorized": False,
        "raw_instruction_or_test_content_persisted": False,
        "report_hash": "",
    }
    report["report_hash"] = stable_hash({**report, "report_hash": ""})
    _write_json_exclusive(paths["report"], report)

    _replace_json(
        paths["decision_lock"],
        {
            "lock_version": "sc100_train_diagnostic_decision_lock_v1",
            "state": "completed",
            "decision_ordinal": 1,
            "preregistration_manifest_hash": preregistration_hash,
            "report_hash": report["report_hash"],
            "diagnostic_passed": diagnostic_passed,
            "retry_authorized": False,
        },
    )
    return report


def verify_existing_sc100_train_diagnostic(
    *, project_root: str | Path, preregistration_path: str | Path
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    preregistration = _read_json(
        Path(preregistration_path).expanduser().resolve(strict=True)
    )
    preregistration_hash = _verify_self_hash(preregistration, "manifest_hash")
    paths = _formal_paths(project, preregistration)
    report = _read_json(paths["report"])
    report_hash = _verify_self_hash(report, "report_hash")
    lock = _read_json(paths["decision_lock"])
    if (
        lock.get("state") != "completed"
        or lock.get("decision_ordinal") != 1
        or lock.get("preregistration_manifest_hash") != preregistration_hash
        or lock.get("report_hash") != report_hash
        or lock.get("diagnostic_passed") != report.get("diagnostic_passed")
        or lock.get("retry_authorized") is not False
    ):
        raise SC100DiagnosticError("decision lock does not bind the report")
    for row in report.get("rows", ()):
        generation = row.get("generation") or {}
        verification = row.get("verification") or {}
        if generation.get("generated"):
            path = _resolve_under(project, generation["operator_receipt_path"])
            if _sha256(path) != generation["operator_receipt_sha256"]:
                raise SC100DiagnosticError("operator receipt drift")
        if verification.get("verifier_completed"):
            path = _resolve_under(project, verification["verifier_report_path"])
            if _sha256(path) != verification["verifier_report_sha256"]:
                raise SC100DiagnosticError("verifier receipt drift")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_existing:
        report = verify_existing_sc100_train_diagnostic(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
        )
    else:
        report = run_sc100_train_diagnostic(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
