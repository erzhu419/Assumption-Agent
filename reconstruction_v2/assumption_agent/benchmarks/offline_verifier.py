from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import Event, EventSink, NullEventSink
from ..models import stable_hash


OFFLINE_VERIFIER_POLICY_VERSION = "family_profile_readonly_volume_v1"
OFFLINE_VERIFIER_MOUNT = "/opt/assumption-v2-verifier"
TUNA_PYPI_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
TUNA_PYPI_HOST = "pypi.tuna.tsinghua.edu.cn"
_RUNTIME_NETWORK_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:apt(?:-get)?|pip3?|uvx|curl|wget|npm|pnpm|yarn|npx)\b"
    r"|\bpython(?:3(?:\.\d+)?)?\s+-m\s+pip\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class OfflineVerifierProfile:
    profile_id: str
    families: tuple[str, ...]
    requirements: tuple[str, ...]
    import_probe: str
    python_version: str = "3.12"
    python_abi: str = "cp312"
    platform: str = "manylinux_2_17_x86_64"

    @property
    def profile_hash(self) -> str:
        return stable_hash(
            {
                "policy": OFFLINE_VERIFIER_POLICY_VERSION,
                "profile_id": self.profile_id,
                "families": self.families,
                "requirements": self.requirements,
                "import_probe": self.import_probe,
                "python_version": self.python_version,
                "python_abi": self.python_abi,
                "platform": self.platform,
            }
        )

    @property
    def verifier_command(self) -> str:
        site = f"{OFFLINE_VERIFIER_MOUNT}/site"
        return (
            "set -u; mkdir -p /logs/verifier; "
            "rm -f /logs/verifier/ctrf.json /logs/verifier/reward.txt; "
            f"PYTHONPATH={site} PYTHONNOUSERSITE=1 PIP_NO_INDEX=1 "
            "PIP_DISABLE_PIP_VERSION_CHECK=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 python3 -m pytest "
            "--ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA -v; "
            "status=$?; if [ \"$status\" -eq 0 ]; then "
            "echo 1 > /logs/verifier/reward.txt; else "
            "echo 0 > /logs/verifier/reward.txt; fi; exit 0"
        )


POSTER_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="anthropic-poster-py312-v1",
    families=("anthropic-poster-design",),
    requirements=(
        "pytest==8.4.1",
        "pytest-json-ctrf==0.3.5",
        "python-docx==1.1.2",
        "numpy==2.2.6",
        "Pillow==11.3.0",
    ),
    import_probe=(
        "import pytest, PIL, numpy, docx; "
        "assert pytest.__version__ == '8.4.1'; "
        "assert PIL.__version__ == '11.3.0'; "
        "assert numpy.__version__ == '2.2.6'; "
        "assert docx.__version__ == '1.1.2'"
    ),
)

OFFLINE_VERIFIER_PROFILES = (POSTER_VERIFIER_PROFILE,)


def offline_verifier_profile_for_family(
    family: str,
) -> OfflineVerifierProfile | None:
    for profile in OFFLINE_VERIFIER_PROFILES:
        if family in profile.families:
            return profile
    return None


def test_script_requires_offline_profile(test_script: Path) -> bool:
    if not test_script.is_file():
        return True
    executable_lines = "\n".join(
        line
        for line in test_script.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if not line.lstrip().startswith("#")
    )
    return _RUNTIME_NETWORK_COMMAND.search(executable_lines) is not None


@dataclass(frozen=True)
class OfflineVerifierRuntime:
    profile: OfflineVerifierProfile
    runtime_key: str
    volume_name: str
    base_image_id: str
    reused: bool


class SkillLearnOfflineVerifierRuntimeCache:
    def __init__(
        self,
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self.event_sink = event_sink or NullEventSink()

    def ensure(
        self,
        *,
        profile: OfflineVerifierProfile,
        base_image_tag: str,
        base_image_id: str,
        delegate: Any,
        trace_id: str,
    ) -> OfflineVerifierRuntime:
        runtime_key = offline_verifier_runtime_key(
            profile=profile,
            base_image_id=base_image_id,
        )
        volume_name = offline_verifier_volume_name(runtime_key)
        inspected = delegate.run(
            ["docker", "volume", "inspect", volume_name],
            capture_output=True,
            text=True,
        )
        if int(getattr(inspected, "returncode", 1)) != 0:
            self._emit(
                "skilllearn_offline_verifier_runtime_missing",
                trace_id,
                profile,
                runtime_key,
                volume_name,
                base_image_id,
                reused=False,
            )
            raise RuntimeError("offline_verifier_runtime_missing_cache_only")
        try:
            payload = json.loads(str(getattr(inspected, "stdout", "") or ""))[0]
            labels = payload.get("Labels") or {}
            if labels.get("org.assumption-agent.verifier.key") != runtime_key:
                raise PermissionError("offline verifier volume key mismatch")
            if labels.get("org.assumption-agent.verifier.policy") != OFFLINE_VERIFIER_POLICY_VERSION:
                raise PermissionError("offline verifier volume policy mismatch")
        except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("offline verifier volume metadata is malformed") from exc
        verified = delegate.run(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "-v",
                f"{volume_name}:{OFFLINE_VERIFIER_MOUNT}:ro",
                base_image_tag,
                "sh",
                "-lc",
                (
                    f"PYTHONPATH={OFFLINE_VERIFIER_MOUNT}/site "
                    "PYTHONNOUSERSITE=1 PIP_NO_INDEX=1 python3 -c "
                    + _shell_single_quote(profile.import_probe)
                ),
            ],
            capture_output=True,
            text=True,
        )
        if int(getattr(verified, "returncode", 1)) != 0:
            raise RuntimeError("offline_verifier_runtime_probe_failed")
        runtime = OfflineVerifierRuntime(
            profile=profile,
            runtime_key=runtime_key,
            volume_name=volume_name,
            base_image_id=base_image_id,
            reused=True,
        )
        self._emit(
            "skilllearn_offline_verifier_runtime_ready",
            trace_id,
            profile,
            runtime_key,
            volume_name,
            base_image_id,
            reused=True,
        )
        return runtime

    def _emit(
        self,
        event: str,
        trace_id: str,
        profile: OfflineVerifierProfile,
        runtime_key: str,
        volume_name: str,
        base_image_id: str,
        *,
        reused: bool,
    ) -> None:
        self.event_sink.emit(
            Event(
                event=event,
                stage="benchmark.skilllearn.offline_verifier",
                trace_id=trace_id,
                payload={
                    "policy": OFFLINE_VERIFIER_POLICY_VERSION,
                    "profile_id": profile.profile_id,
                    "profile_hash": profile.profile_hash,
                    "runtime_key": runtime_key,
                    "runtime_volume_hash": stable_hash({"volume": volume_name}),
                    "base_image_id": base_image_id,
                    "reused": reused,
                    "runtime_network": "none",
                    "runtime_mount_read_only": True,
                    "raw_content_persisted": False,
                },
            )
        )


def offline_verifier_runtime_key(
    *,
    profile: OfflineVerifierProfile,
    base_image_id: str,
) -> str:
    return stable_hash(
        {
            "policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "profile_hash": profile.profile_hash,
            "base_image_id": base_image_id,
        }
    )


def offline_verifier_volume_name(runtime_key: str) -> str:
    return f"assumption-v2-verifier-{runtime_key[:24]}"


def prepare_offline_verifier_runtime(
    *,
    profile: OfflineVerifierProfile,
    base_image_tag: str,
    report_path: Path,
    refresh_wheels: bool = False,
    delegate: Any = subprocess,
) -> Mapping[str, Any]:
    image = delegate.run(
        ["docker", "image", "inspect", base_image_tag],
        capture_output=True,
        text=True,
    )
    if int(getattr(image, "returncode", 1)) != 0:
        raise RuntimeError("offline verifier base image is unavailable")
    image_payload = json.loads(str(getattr(image, "stdout", "") or ""))[0]
    base_image_id = str(image_payload.get("Id") or "")
    if not base_image_id.startswith("sha256:"):
        raise RuntimeError("offline verifier base image has no immutable ID")
    runtime_key = offline_verifier_runtime_key(
        profile=profile,
        base_image_id=base_image_id,
    )
    volume_name = offline_verifier_volume_name(runtime_key)
    cache_root = Path(
        os.environ.get(
            "ASSUMPTION_V2_OFFLINE_VERIFIER_CACHE",
            Path.home() / ".cache" / "assumption-agent-v2" / "offline-verifier",
        )
    ).expanduser().resolve()
    wheelhouse = cache_root / profile.profile_hash / "wheels"
    if refresh_wheels and wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    wheels = tuple(sorted(wheelhouse.glob("*.whl")))
    wheelhouse_reused = bool(wheels) and not refresh_wheels
    online_download_attempted = False
    runtime_cache = SkillLearnOfflineVerifierRuntimeCache()
    runtime_reused = False
    if not refresh_wheels:
        try:
            runtime_cache.ensure(
                profile=profile,
                base_image_tag=base_image_tag,
                base_image_id=base_image_id,
                delegate=delegate,
                trace_id="offline-verifier-prepare-reuse",
            )
            runtime_reused = True
        except RuntimeError as exc:
            if str(exc) not in {
                "offline_verifier_runtime_missing_cache_only",
                "offline_verifier_runtime_probe_failed",
            }:
                raise
    if not runtime_reused:
        if not wheels:
            online_download_attempted = True
            download_command = [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--only-binary=:all:",
                "--platform",
                profile.platform,
                "--implementation",
                "cp",
                "--python-version",
                profile.python_version,
                "--abi",
                profile.python_abi,
                "--dest",
                str(wheelhouse),
                "--index-url",
                TUNA_PYPI_INDEX_URL,
                "--trusted-host",
                TUNA_PYPI_HOST,
                *profile.requirements,
            ]
            downloaded = delegate.run(
                download_command,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PIP_INDEX_URL": TUNA_PYPI_INDEX_URL,
                    "PIP_TRUSTED_HOST": TUNA_PYPI_HOST,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_NO_CACHE_DIR": "1",
                },
            )
            if int(getattr(downloaded, "returncode", 1)) != 0:
                raise RuntimeError("offline verifier wheel download failed")
            wheels = tuple(sorted(wheelhouse.glob("*.whl")))
            wheelhouse_reused = False
        if not wheels:
            raise RuntimeError("offline verifier wheelhouse is empty")
        delegate.run(
            ["docker", "volume", "rm", "-f", volume_name],
            capture_output=True,
            text=True,
        )
        created = delegate.run(
            [
                "docker",
                "volume",
                "create",
                "--label",
                f"org.assumption-agent.verifier.key={runtime_key}",
                "--label",
                f"org.assumption-agent.verifier.policy={OFFLINE_VERIFIER_POLICY_VERSION}",
                "--label",
                f"org.assumption-agent.verifier.profile={profile.profile_hash}",
                volume_name,
            ],
            capture_output=True,
            text=True,
        )
        if int(getattr(created, "returncode", 1)) != 0:
            raise RuntimeError("offline verifier volume creation failed")
        install_command = (
            "set -eu; rm -rf /runtime/site; mkdir -p /runtime/site; "
            "python3 -m pip install --no-index --no-cache-dir "
            "--disable-pip-version-check --find-links=/wheels "
            "--target=/runtime/site "
            + " ".join(profile.requirements)
        )
        installed = delegate.run(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "-v",
                f"{wheelhouse}:/wheels:ro",
                "-v",
                f"{volume_name}:/runtime",
                base_image_tag,
                "sh",
                "-lc",
                install_command,
            ],
            capture_output=True,
            text=True,
        )
        if int(getattr(installed, "returncode", 1)) != 0:
            delegate.run(
                ["docker", "volume", "rm", "-f", volume_name],
                capture_output=True,
                text=True,
            )
            raise RuntimeError("offline verifier wheel installation failed")
        runtime_cache.ensure(
            profile=profile,
            base_image_tag=base_image_tag,
            base_image_id=base_image_id,
            delegate=delegate,
            trace_id="offline-verifier-prepare-verify",
        )
    report: dict[str, Any] = {
        "report_version": "offline_verifier_preparation_receipt_v2",
        "policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "profile_id": profile.profile_id,
        "profile_hash": profile.profile_hash,
        "runtime_key": runtime_key,
        "runtime_volume_hash": stable_hash({"volume": volume_name}),
        "base_image_tag": base_image_tag,
        "base_image_id": base_image_id,
        "python_version": profile.python_version,
        "python_abi": profile.python_abi,
        "platform": profile.platform,
        "package_index_origin": TUNA_PYPI_INDEX_URL,
        "docker_install_network": "none",
        "runtime_reused": runtime_reused,
        "wheelhouse_reused": wheelhouse_reused,
        "online_download_attempted": online_download_attempted,
        "wheel_count": len(wheels),
        "wheel_total_bytes": sum(path.stat().st_size for path in wheels),
        "wheels": [
            {
                "filename": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in wheels
        ],
        "probe_passed": True,
        "raw_content_persisted": False,
    }
    report["receipt_hash"] = stable_hash(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def probe_offline_verifier_runtime(
    *,
    profile: OfflineVerifierProfile,
    base_image_tag: str,
    workspace: Path,
    tests_dir: Path,
    report_path: Path,
    delegate: Any = subprocess,
) -> Mapping[str, Any]:
    workspace = workspace.expanduser().resolve()
    tests_dir = tests_dir.expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError("offline verifier probe workspace is missing")
    if not (tests_dir / "test_outputs.py").is_file():
        raise FileNotFoundError("offline verifier probe tests are missing")
    image = delegate.run(
        ["docker", "image", "inspect", base_image_tag],
        capture_output=True,
        text=True,
    )
    if int(getattr(image, "returncode", 1)) != 0:
        raise RuntimeError("offline verifier probe base image is unavailable")
    image_payload = json.loads(str(getattr(image, "stdout", "") or ""))[0]
    base_image_id = str(image_payload.get("Id") or "")
    runtime_key = offline_verifier_runtime_key(
        profile=profile,
        base_image_id=base_image_id,
    )
    volume_name = offline_verifier_volume_name(runtime_key)
    SkillLearnOfflineVerifierRuntimeCache().ensure(
        profile=profile,
        base_image_tag=base_image_tag,
        base_image_id=base_image_id,
        delegate=delegate,
        trace_id="offline-verifier-probe",
    )
    with tempfile.TemporaryDirectory(
        prefix="assumption-v2-verifier-probe-"
    ) as raw_logs_root:
        logs_root = Path(raw_logs_root)
        completed = delegate.run(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "-v",
                f"{volume_name}:{OFFLINE_VERIFIER_MOUNT}:ro",
                "-v",
                f"{workspace}:/root",
                "-v",
                f"{tests_dir}:/tests:ro",
                "-v",
                f"{logs_root}:/logs",
                base_image_tag,
                "sh",
                "-lc",
                profile.verifier_command,
            ],
            capture_output=True,
            text=True,
        )
        reward_file = logs_root / "verifier" / "reward.txt"
        ctrf_file = logs_root / "verifier" / "ctrf.json"
        reward = (
            int(reward_file.read_text(encoding="utf-8").strip())
            if reward_file.is_file()
            else None
        )
        ctrf = (
            json.loads(ctrf_file.read_text(encoding="utf-8"))
            if ctrf_file.is_file()
            else None
        )
        results = ctrf.get("results") if isinstance(ctrf, Mapping) else None
        summary = results.get("summary") if isinstance(results, Mapping) else None
        test_count = int(summary.get("tests") or 0) if isinstance(summary, Mapping) else 0
        report: dict[str, Any] = {
            "report_version": "offline_verifier_probe_receipt_v1",
            "policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "runtime_key": runtime_key,
            "runtime_volume_hash": stable_hash({"volume": volume_name}),
            "base_image_id": base_image_id,
            "container_network": "none",
            "runtime_mount_read_only": True,
            "original_online_test_script_executed": False,
            "container_exit": int(getattr(completed, "returncode", 1)),
            "reward": reward,
            "test_count": test_count,
            "ctrf_sha256": _sha256(ctrf_file) if ctrf_file.is_file() else None,
            "workspace_hash": _directory_hash(workspace),
            "tests_hash": _directory_hash(tests_dir),
            "probe_passed": (
                int(getattr(completed, "returncode", 1)) == 0
                and reward in {0, 1}
                and test_count > 0
            ),
            "raw_content_persisted": False,
        }
    report["receipt_hash"] = stable_hash(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hash(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return stable_hash(rows)


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--base-image-tag", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-workspace", type=Path)
    parser.add_argument("--probe-tests", type=Path)
    parser.add_argument("--refresh-wheels", action="store_true")
    args = parser.parse_args(argv)
    profile = next(
        (row for row in OFFLINE_VERIFIER_PROFILES if row.profile_id == args.profile),
        None,
    )
    if profile is None:
        raise ValueError(f"unknown offline verifier profile: {args.profile}")
    if (args.probe_workspace is None) != (args.probe_tests is None):
        raise ValueError("offline verifier probe requires workspace and tests together")
    if args.probe_workspace is not None:
        report = probe_offline_verifier_runtime(
            profile=profile,
            base_image_tag=args.base_image_tag,
            workspace=args.probe_workspace,
            tests_dir=args.probe_tests,
            report_path=args.report,
        )
    else:
        report = prepare_offline_verifier_runtime(
            profile=profile,
            base_image_tag=args.base_image_tag,
            report_path=args.report,
            refresh_wheels=args.refresh_wheels,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
