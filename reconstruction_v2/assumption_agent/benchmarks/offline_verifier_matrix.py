from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from ..events import JsonlEventSink
from ..models import stable_hash
from ..splits import SplitManifest
from .offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    OFFLINE_VERIFIER_PROFILES,
    probe_offline_verifier_runtime,
)
from .skilllearn_lifecycle import (
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
)
from .skilllearnbench import SkillLearnBenchAdapter


def probe_offline_verifier_matrix(
    *,
    benchmark_root: Path,
    manifest: SplitManifest,
    output_root: Path,
    events_path: Path,
) -> dict[str, Any]:
    benchmark_root = benchmark_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    sink = JsonlEventSink(events_path)
    adapter = SkillLearnBenchAdapter(benchmark_root)
    inventory = {item.id: item for item in adapter.discover()}
    cache = SkillLearnPrebuiltImageCache(
        benchmark_root,
        cache_only=True,
        event_sink=sink,
    )
    backend = SkillLearnSubprocessBackend(
        benchmark_root,
        record_upstream=False,
        prebuilt_cache=cache,
        event_sink=sink,
    )
    train_by_family: dict[str, str] = {}
    for item_id in manifest.train_ids:
        item = inventory[item_id]
        train_by_family.setdefault(item.family, item_id)
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="assumption-v2-empty-verifier-workspace-"
    ) as raw_workspace, tempfile.TemporaryDirectory(
        prefix="assumption-v2-synthetic-verifier-tests-"
    ) as raw_synthetic_tests:
        workspace = Path(raw_workspace)
        synthetic_tests = Path(raw_synthetic_tests)
        pass_tests = synthetic_tests / "pass"
        fail_tests = synthetic_tests / "fail"
        pass_tests.mkdir()
        fail_tests.mkdir()
        (pass_tests / "test_outputs.py").write_text(
            "def test_known_pass():\n    assert True\n",
            encoding="utf-8",
        )
        (fail_tests / "test_outputs.py").write_text(
            "def test_known_fail():\n    assert False\n",
            encoding="utf-8",
        )
        for profile in OFFLINE_VERIFIER_PROFILES:
            representative_family = profile.families[0]
            representative_item_id = train_by_family.get(representative_family)
            if representative_item_id is None:
                profile_rows.append(
                    {
                        "profile_id": profile.profile_id,
                        "profile_hash": profile.profile_hash,
                        "passed": False,
                        "error_type": "train_profile_representative_missing",
                    }
                )
            else:
                profile_trace_id = (
                    "offline-verifier-profile-contract:"
                    + stable_hash(
                        {
                            "profile_id": profile.profile_id,
                            "item_id": representative_item_id,
                        }
                    )[:20]
                )
                try:
                    representative_image = backend.prewarm_environment(
                        family=representative_family,
                        item_id=representative_item_id,
                        trace_id=profile_trace_id,
                    )
                    contract_receipts: dict[str, dict[str, Any]] = {}
                    for expected, tests_dir in (
                        ("pass", pass_tests),
                        ("fail", fail_tests),
                    ):
                        contract_receipts[expected] = dict(
                            probe_offline_verifier_runtime(
                                profile=profile,
                                base_image_tag=representative_image.tag,
                                workspace=workspace,
                                tests_dir=tests_dir,
                                report_path=(
                                    output_root
                                    / "profiles"
                                    / f"{profile.profile_id}-{expected}.json"
                                ),
                                event_sink=sink,
                                trace_id=f"{profile_trace_id}:{expected}",
                            )
                        )
                    pass_receipt = contract_receipts["pass"]
                    fail_receipt = contract_receipts["fail"]
                    contract_passed = bool(
                        pass_receipt.get("probe_passed")
                        and pass_receipt.get("reward") == 1
                        and int(pass_receipt.get("test_count") or 0) == 1
                        and fail_receipt.get("probe_passed")
                        and fail_receipt.get("reward") == 0
                        and int(fail_receipt.get("test_count") or 0) == 1
                    )
                    profile_rows.append(
                        {
                            "profile_id": profile.profile_id,
                            "profile_hash": profile.profile_hash,
                            "representative_family": representative_family,
                            "representative_item_id_hash": stable_hash(
                                {"item_id": representative_item_id}
                            ),
                            "prebuilt_image_id": representative_image.image_id,
                            "known_pass_receipt_hash": pass_receipt.get(
                                "receipt_hash"
                            ),
                            "known_fail_receipt_hash": fail_receipt.get(
                                "receipt_hash"
                            ),
                            "passed": contract_passed,
                            "error_type": (
                                None
                                if contract_passed
                                else "profile_pass_fail_contract_failed"
                            ),
                        }
                    )
                except Exception as exc:
                    profile_rows.append(
                        {
                            "profile_id": profile.profile_id,
                            "profile_hash": profile.profile_hash,
                            "passed": False,
                            "error_type": type(exc).__name__,
                            "error_message_hash": stable_hash(
                                {"message": str(exc)}
                            ),
                        }
                    )
            for family in profile.families:
                item_id = train_by_family.get(family)
                if item_id is None:
                    rows.append(
                        {
                            "family": family,
                            "profile_id": profile.profile_id,
                            "passed": False,
                            "error_type": "train_family_representative_missing",
                        }
                    )
                    continue
                trace_id = (
                    "offline-verifier-matrix:"
                    + stable_hash({"family": family, "item_id": item_id})[:20]
                )
                try:
                    image = backend.prewarm_environment(
                        family=family,
                        item_id=item_id,
                        trace_id=trace_id,
                    )
                    family_report = (
                        output_root / "families" / f"{family}.json"
                    )
                    probe = probe_offline_verifier_runtime(
                        profile=profile,
                        base_image_tag=image.tag,
                        workspace=workspace,
                        tests_dir=(
                            benchmark_root
                            / "tasks"
                            / family
                            / item_id
                            / "tests"
                        ),
                        report_path=family_report,
                        event_sink=sink,
                        trace_id=f"{trace_id}:probe",
                    )
                    passed = bool(
                        probe.get("probe_passed")
                        and int(probe.get("test_count") or 0) > 0
                        and probe.get("reward") == 0
                    )
                    rows.append(
                        {
                            "family": family,
                            "profile_id": profile.profile_id,
                            "profile_hash": profile.profile_hash,
                            "item_id_hash": stable_hash({"item_id": item_id}),
                            "prebuilt_image_id": image.image_id,
                            "runtime_key": probe.get("runtime_key"),
                            "test_count": probe.get("test_count"),
                            "reward": probe.get("reward"),
                            "probe_receipt_hash": probe.get("receipt_hash"),
                            "passed": passed,
                            "error_type": (
                                None if passed else "offline_probe_contract_failed"
                            ),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "family": family,
                            "profile_id": profile.profile_id,
                            "item_id_hash": stable_hash({"item_id": item_id}),
                            "passed": False,
                            "error_type": type(exc).__name__,
                            "error_message_hash": stable_hash(
                                {"message": str(exc)}
                            ),
                        }
                    )
    payload: dict[str, Any] = {
        "report_version": "offline_verifier_family_matrix_v2",
        "policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "manifest_hash": manifest.manifest_hash,
        "split": "train",
        "model_executed": False,
        "sealed_test_content_accessed": False,
        "family_count": len(rows),
        "passed_family_count": sum(bool(row["passed"]) for row in rows),
        "failed_family_count": sum(not bool(row["passed"]) for row in rows),
        "profile_count": len(profile_rows),
        "passed_profile_count": sum(
            bool(row["passed"]) for row in profile_rows
        ),
        "failed_profile_count": sum(
            not bool(row["passed"]) for row in profile_rows
        ),
        "passed": (
            all(bool(row["passed"]) for row in rows)
            and all(bool(row["passed"]) for row in profile_rows)
        ),
        "profile_contracts": profile_rows,
        "families": rows,
        "raw_content_persisted": False,
    }
    payload["receipt_hash"] = stable_hash(payload)
    (output_root / "matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    payload = probe_offline_verifier_matrix(
        benchmark_root=args.root,
        manifest=SplitManifest.read(args.manifest),
        output_root=args.output_root,
        events_path=args.events,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
