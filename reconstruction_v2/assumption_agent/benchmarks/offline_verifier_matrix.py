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
    OFFLINE_VERIFIER_PROFILE_CATALOG,
    OFFLINE_VERIFIER_POLICY_VERSION,
    OFFLINE_VERIFIER_PROFILES,
    offline_verifier_activation_blocker_for_family,
    offline_verifier_profile_for_family,
    probe_offline_verifier_runtime,
    test_script_requires_offline_profile,
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
    incomplete_train_verifier_ids: list[str] = []
    activation_blocked_train_ids: list[str] = []
    missing_profile_train_ids: list[str] = []
    for item_id in manifest.train_ids:
        item = inventory[item_id]
        test_script = (
            benchmark_root
            / "tasks"
            / item.family
            / item_id
            / "tests"
            / "test.sh"
        )
        if (
            offline_verifier_profile_for_family(item.family) is None
            and test_script_requires_offline_profile(test_script)
        ):
            if offline_verifier_activation_blocker_for_family(item.family):
                activation_blocked_train_ids.append(item_id)
            else:
                missing_profile_train_ids.append(item_id)
        test_outputs = (
            benchmark_root
            / "tasks"
            / item.family
            / item_id
            / "tests"
            / "test_outputs.py"
        )
        if not test_outputs.is_file():
            incomplete_train_verifier_ids.append(item_id)
            continue
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
        secret_audit_source = (
            "from pathlib import Path\n"
            "import os\n\n"
            "SECRET_NAMES = {b'OPENAI_API_KEY', b'ASSUMPTION_V2_API_KEY'}\n\n"
            "def assert_model_secrets_absent():\n"
            "    assert not {name.decode() for name in SECRET_NAMES} & set(os.environ)\n"
            "    for proc in Path('/proc').iterdir():\n"
            "        if not proc.name.isdigit():\n"
            "            continue\n"
            "        try:\n"
            "            command = (proc / 'cmdline').read_bytes()\n"
            "            if b'org.apache.druid.cli.Main' not in command:\n"
            "                continue\n"
            "            environ = (proc / 'environ').read_bytes().split(b'\\0')\n"
            "            names = {row.split(b'=', 1)[0] for row in environ if b'=' in row}\n"
            "            assert not SECRET_NAMES & names\n"
            "        except (FileNotFoundError, PermissionError, ProcessLookupError):\n"
            "            continue\n\n"
        )
        (pass_tests / "test_outputs.py").write_text(
            secret_audit_source
            + "def test_known_pass():\n"
            + "    assert_model_secrets_absent()\n",
            encoding="utf-8",
        )
        (fail_tests / "test_outputs.py").write_text(
            secret_audit_source
            + "def test_known_fail():\n"
            + "    assert_model_secrets_absent()\n"
            + "    assert False\n",
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
                            "model_secret_env_canary_verified": bool(
                                pass_receipt.get(
                                    "model_secret_env_canary_injected"
                                )
                                and fail_receipt.get(
                                    "model_secret_env_canary_injected"
                                )
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
    profile_matrix_passed = bool(
        all(bool(row["passed"]) for row in rows)
        and all(bool(row["passed"]) for row in profile_rows)
    )
    manifest_execution_ready = not (
        incomplete_train_verifier_ids
        or activation_blocked_train_ids
        or missing_profile_train_ids
    )
    blockers = []
    if incomplete_train_verifier_ids:
        blockers.append("incomplete_train_verifier_payload")
    if activation_blocked_train_ids:
        blockers.append("inactive_offline_verifier_profile")
    if missing_profile_train_ids:
        blockers.append("missing_offline_verifier_profile")
    inactive_profiles = [
        {
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "family_count": len(profile.families),
            "activation_blocker": profile.activation_blocker,
        }
        for profile in OFFLINE_VERIFIER_PROFILE_CATALOG
        if profile.activation_blocker is not None
    ]
    payload: dict[str, Any] = {
        "report_version": "offline_verifier_family_matrix_v4",
        "policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "manifest_hash": manifest.manifest_hash,
        "split": "train",
        "representative_selection_policy": "train_first_complete_verifier_v2",
        "model_executed": False,
        "sealed_test_content_accessed": False,
        "incomplete_train_verifier_item_count": len(
            incomplete_train_verifier_ids
        ),
        "incomplete_train_verifier_item_set_hash": stable_hash(
            sorted(
                stable_hash({"item_id": item_id})
                for item_id in incomplete_train_verifier_ids
            )
        ),
        "activation_blocked_train_item_count": len(
            activation_blocked_train_ids
        ),
        "activation_blocked_train_item_set_hash": stable_hash(
            sorted(
                stable_hash({"item_id": item_id})
                for item_id in activation_blocked_train_ids
            )
        ),
        "missing_profile_train_item_count": len(missing_profile_train_ids),
        "missing_profile_train_item_set_hash": stable_hash(
            sorted(
                stable_hash({"item_id": item_id})
                for item_id in missing_profile_train_ids
            )
        ),
        "inactive_profile_count": len(inactive_profiles),
        "inactive_profiles": inactive_profiles,
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
        "profile_matrix_passed": profile_matrix_passed,
        "manifest_execution_ready": manifest_execution_ready,
        "blockers": blockers,
        "passed": profile_matrix_passed and manifest_execution_ready,
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
