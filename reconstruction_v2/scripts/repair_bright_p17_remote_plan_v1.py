#!/usr/bin/env python3
"""Replace only the invalid P17 plan's corrected freeze identities, once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "reconstruction_v2")]

from reconstruction_v2.assumption_agent.benchmarks import (  # noqa: E402
    bright_p14_acquisition_v1 as p14_acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (  # noqa: E402
    bright_p17_all_remote_c_confirm_v1 as p17,
)


OLD_PLAN_FILE_SHA256 = (
    "c18c65ab8eae13cb0da36c803ff3e62aaf07aafc8a3417d2229450ec2204c4b2"
)
OLD_PLAN_PACK_SHA256 = (
    "8a0bad29e20aa13b24fe728b15e0a926d323db4cb995eafe72c8f0fc5ea22cc1"
)


class P17PlanRepairError(RuntimeError):
    """The prelaunch-invalid P17 plan cannot be corrected exactly once."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return p14_acquisition.utilities.canonical_json_bytes(value)


def _read(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P17PlanRepairError(f"{name} is unavailable") from exc
    if not isinstance(value, Mapping) or path.read_bytes() != _canonical(value):
        raise P17PlanRepairError(f"{name} is not canonical")
    return value


def _verify(value: Mapping[str, Any], field: str, name: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if (
        not isinstance(declared, str)
        or p14_acquisition.utilities.stable_hash(body) != declared
    ):
        raise P17PlanRepairError(f"{name} identity drifted")
    return declared


def repair(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    plan_path = base / p17.PLAN_RELATIVE
    archive_path = plan_path.parent / "prelaunch_invalid/remote_execution.plan.json"
    fingerprint_path = base / p17.FINGERPRINT_RELATIVE
    freeze_path = base / p17.FREEZE_RELATIVE

    archive_exists = archive_path.exists() or archive_path.is_symlink()
    if archive_exists and (
        archive_path.is_symlink()
        or not archive_path.is_file()
        or _file_sha256(archive_path) != OLD_PLAN_FILE_SHA256
    ):
        raise P17PlanRepairError("the existing invalid P17 plan archive drifted")
    if _file_sha256(plan_path) != OLD_PLAN_FILE_SHA256:
        raise P17PlanRepairError("the invalid P17 plan was already consumed")
    plan = _read(plan_path, "invalid P17 plan")
    if (
        _verify(plan, "pack_sha256", "invalid P17 plan")
        != OLD_PLAN_PACK_SHA256
    ):
        raise P17PlanRepairError("the invalid P17 plan pack drifted")

    fingerprint = _read(fingerprint_path, "corrected P17 fingerprint")
    freeze = _read(freeze_path, "corrected P17 implementation freeze")
    fingerprint_self = _verify(
        fingerprint, "self_sha256", "corrected P17 fingerprint"
    )
    freeze_self = _verify(freeze, "self_sha256", "corrected P17 implementation freeze")
    if (
        fingerprint.get("status")
        != "corrected_after_P17_item_identity_staging_before_any_model_or_comparator_action"
        or freeze.get("status")
        != "outcome_blind_prelaunch_runtime_repair_frozen_after_identity_staging"
    ):
        raise P17PlanRepairError("the corrected P17 freeze boundary is absent")

    body = dict(plan)
    body.pop("pack_sha256")
    body["implementation_freeze_self_sha256"] = freeze_self
    body["remote_runtime_fingerprint_self_sha256"] = fingerprint_self
    corrected = p14_acquisition.utilities.self_hashed(body, field="pack_sha256")
    unchanged = set(plan) - {
        "implementation_freeze_self_sha256",
        "pack_sha256",
        "remote_runtime_fingerprint_self_sha256",
    }
    if any(corrected.get(key) != plan.get(key) for key in unchanged):
        raise P17PlanRepairError("a scientific P17 plan field changed")

    if not archive_exists:
        archive_path.parent.mkdir(mode=0o700, parents=True)
        p14_acquisition.utilities._write_exclusive(
            archive_path, plan_path.read_bytes(), mode=0o600
        )
    temporary = plan_path.with_name(plan_path.name + ".repair.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise P17PlanRepairError("the corrected plan temporary already exists")
    p14_acquisition.utilities._write_json(temporary, corrected)
    temporary.replace(plan_path)
    return {
        "corrected_plan_file_sha256": _file_sha256(plan_path),
        "corrected_plan_pack_sha256": corrected["pack_sha256"],
        "old_plan_file_sha256": _file_sha256(archive_path),
        "old_plan_pack_sha256": plan["pack_sha256"],
        "status": "P17_prelaunch_plan_corrected_freeze_bindings_only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            repair(arguments.project_root),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
