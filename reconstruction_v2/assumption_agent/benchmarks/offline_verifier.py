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

from ..events import Event, EventSink, JsonlEventSink, NullEventSink
from ..models import stable_hash


OFFLINE_VERIFIER_POLICY_VERSION = "family_profile_readonly_volume_v3"
OFFLINE_VERIFIER_MOUNT = "/opt/assumption-v2-verifier"
TUNA_PYPI_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
TUNA_PYPI_HOST = "pypi.tuna.tsinghua.edu.cn"
_RUNTIME_NETWORK_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:apt(?:-get)?|pip3?|uvx|curl|wget|npm|pnpm|yarn|npx)\b"
    r"|\bpython(?:3(?:\.\d+)?)?\s+-m\s+pip\b",
    re.IGNORECASE | re.MULTILINE,
)
_DIAGNOSTIC_SECRET = re.compile(
    r"(?i)(?:(?:sk-|s2k-|ghp_|hf_)[a-z0-9_-]{8,}|bearer\s+\S+|"
    r"https?://[^\s/:@]+:[^\s/@]+@)"
)
_DIAGNOSTIC_SIGNAL = re.compile(
    r"(?i)(?:error|failed|requires-python|ignored|no matching distribution|not found)"
)
_PRELUDE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_PRELUDE_DETAIL_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PRELUDE_DETAIL_VALUE = re.compile(r"^[a-zA-Z0-9_.:/+-]{1,160}$")
_MODEL_SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ASSUMPTION_V2_API_KEY",
    "ASSUMPTION_V2_API_BASE",
    "GPT5_API_KEY",
    "GPT5_BASE_URL",
    "RUOLI_API_KEY",
    "RUOLI_BASE_URL",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OPENALEX_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)
_MODEL_SECRET_UNSET_COMMAND = "unset " + " ".join(_MODEL_SECRET_ENV_NAMES)

_WEIGHTED_GDP_PRELUDE_COMMAND = (
    "rm -f /root/sheet.csv /root/sheet.csv.*; "
    "ssconvert_status=1; sheet_count=0; "
    "if [ -f /root/gdp.xlsx ] && command -v ssconvert >/dev/null 2>&1; then "
    "ssconvert -S /root/gdp.xlsx /root/sheet.csv >/dev/null 2>&1; "
    "ssconvert_status=$?; "
    "sheet_count=$(find /root -maxdepth 1 -type f -name 'sheet.csv.*' | wc -l); "
    "fi; "
    "printf 'tool=ssconvert\\ncommand_exit=%s\\nsheet_count=%s\\n' "
    '"$ssconvert_status" "$sheet_count" '
    "> /logs/verifier/semantic_prelude_details.txt; "
    '[ "$ssconvert_status" -eq 0 ] && [ "$sheet_count" -gt 0 ]'
)
_WEIGHTED_GDP_ARTIFACT_COMMAND = (
    "cp /root/gdp.xlsx /logs/verifier/gdp_modified.xlsx 2>/dev/null || true; "
    "cp /root/sheet.csv.* /logs/verifier/ 2>/dev/null || true"
)
_DRUID_SECURITY_PRELUDE_COMMAND = (
    "export DRUID_VERSION=0.20.0 DRUID_HOME=/opt/druid WORKSPACE=/root "
    "DRUID_HOST=localhost DRUID_PORT=8888 DRUID_SKIP_PORT_CHECK=1; "
    "pkill -9 -f 'org.apache.druid.cli.Main' 2>/dev/null || true; "
    "pkill -9 -f 'org.apache.zookeeper' 2>/dev/null || true; "
    "pkill -9 -f supervise 2>/dev/null || true; "
    "pkill -9 -f runsvdir 2>/dev/null || true; "
    "sleep 1; rm -rf /opt/druid/var/sv; "
    "druid_hostname=$(hostname); "
    "grep -F \" $druid_hostname\" /etc/hosts >/dev/null 2>&1 || "
    "printf '127.0.0.1 %s\\n' \"$druid_hostname\" >> /etc/hosts; "
    "deployed_jar_count=0; "
    "for module in core processing server indexing-service; do "
    "built_jar=$(find /root/druid/$module/target -name \"druid-$module-*.jar\" "
    "-not -name '*sources*' -not -name '*tests*' 2>/dev/null | head -n 1); "
    "if [ -n \"$built_jar\" ] && [ -f \"$built_jar\" ]; then "
    "original_jar=$(find /opt/druid/lib -name \"druid-$module-*.jar\" "
    "2>/dev/null | head -n 1); "
    "if [ -n \"$original_jar\" ]; then "
    "cp \"$original_jar\" \"${original_jar}.backup\" 2>/dev/null || true; "
    "rm -f \"$original_jar\"; fi; "
    "if cp -f \"$built_jar\" /opt/druid/lib/; then "
    "deployed_jar_count=$((deployed_jar_count + 1)); fi; fi; done; "
    "cd /opt/druid; "
    "nohup ./bin/start-single-server-small > /var/log/druid.log 2>&1 & "
    "druid_pid=$!; druid_ready=0; readiness_attempts=0; "
    "while [ \"$readiness_attempts\" -lt 90 ]; do "
    "readiness_attempts=$((readiness_attempts + 1)); "
    "if curl -fsS http://127.0.0.1:8888/status >/dev/null 2>&1; then "
    "druid_ready=1; break; fi; sleep 2; done; "
    "printf 'hostname_mapped=1\\ndeployed_jar_count=%s\\ndruid_ready=%s\\n"
    "readiness_attempts=%s\\ndruid_pid=%s\\n' "
    '"$deployed_jar_count" "$druid_ready" "$readiness_attempts" "$druid_pid" '
    "> /logs/verifier/semantic_prelude_details.txt; "
    '[ "$druid_ready" -eq 1 ]'
)
_DRUID_SECURITY_ARTIFACT_COMMAND = (
    "mkdir -p /logs/verifier/patches; "
    "if [ -d /root/patches ]; then "
    "find /root/patches -type f \\( -name '*.patch' -o -name '*.diff' \\) "
    "-exec cp {} /logs/verifier/patches/ \\; 2>/dev/null || true; fi; "
    "if [ -d /root/druid/.git ]; then "
    "git -C /root/druid diff HEAD > /logs/verifier/patches/druid-changes.diff "
    "2>/dev/null || true; "
    "git -C /root/druid diff --cached HEAD "
    ">> /logs/verifier/patches/druid-changes.diff 2>/dev/null || true; fi; "
    "cp /var/log/druid.log /logs/verifier/druid.log 2>/dev/null || true; "
    "cp /opt/druid/var/sv/*.log /logs/verifier/ 2>/dev/null || true"
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
    semantic_prelude_id: str | None = None
    semantic_prelude_command: str = ""
    artifact_command: str = ""
    probe_workspace_mode: str = "empty_mount"
    activation_blocker: str | None = None

    def __post_init__(self) -> None:
        if self.semantic_prelude_id is not None and not _PRELUDE_ID.fullmatch(
            self.semantic_prelude_id
        ):
            raise ValueError("offline verifier semantic prelude ID is invalid")
        if bool(self.semantic_prelude_id) != bool(self.semantic_prelude_command):
            raise ValueError("offline verifier semantic prelude is incomplete")
        if self.probe_workspace_mode not in {"empty_mount", "image_root"}:
            raise ValueError("offline verifier probe workspace mode is invalid")
        if self.activation_blocker is not None and not _PRELUDE_ID.fullmatch(
            self.activation_blocker
        ):
            raise ValueError("offline verifier activation blocker is invalid")

    @property
    def profile_hash(self) -> str:
        payload = {
            "policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "profile_id": self.profile_id,
            "families": self.families,
            "requirements": self.requirements,
            "import_probe": self.import_probe,
            "python_version": self.python_version,
            "python_abi": self.python_abi,
            "platform": self.platform,
            "semantic_prelude_id": self.semantic_prelude_id,
            "semantic_prelude_command_hash": stable_hash(
                {"command": self.semantic_prelude_command}
            ),
            "artifact_command_hash": stable_hash(
                {"command": self.artifact_command}
            ),
            "probe_workspace_mode": self.probe_workspace_mode,
        }
        if self.activation_blocker is not None:
            payload["activation_blocker"] = self.activation_blocker
        return stable_hash(payload)

    @property
    def wheelhouse_key(self) -> str:
        return stable_hash(
            {
                "requirements": self.requirements,
                "python_version": self.python_version,
                "python_abi": self.python_abi,
                "platform": self.platform,
            }
        )

    @property
    def verifier_command(self) -> str:
        site = f"{OFFLINE_VERIFIER_MOUNT}/site"
        semantic_prelude = "prelude_status=0; "
        if self.semantic_prelude_id is not None:
            semantic_prelude += (
                f"{self.semantic_prelude_command} || prelude_status=$?; "
                "printf '{\"prelude_id\":\"%s\",\"exit_code\":%s}\\n' "
                f"'{self.semantic_prelude_id}' \"$prelude_status\" "
                "> /logs/verifier/semantic_prelude.json; "
            )
        artifact_command = (
            f"{self.artifact_command}; " if self.artifact_command else ""
        )
        return (
            "set -u; mkdir -p /logs/verifier; cd /root; "
            f"{_MODEL_SECRET_UNSET_COMMAND}; "
            "rm -f /logs/verifier/ctrf.json /logs/verifier/reward.txt "
            "/logs/verifier/semantic_prelude.json "
            "/logs/verifier/semantic_prelude_details.txt; "
            "export RESULTS_PATH=/root/results.json WORKING_DIR=/root; "
            f"{semantic_prelude}"
            f"PYTHONPATH={site} PYTHONNOUSERSITE=1 PIP_NO_INDEX=1 "
            "PIP_DISABLE_PIP_VERSION_CHECK=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 python3 -m pytest "
            "--ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA -v; "
            "status=$?; "
            "cp /root/sc100-filled.pdf /logs/verifier/sc100-filled.pdf 2>/dev/null || true; "
            "cp /root/sc100-blank.pdf /logs/verifier/sc100-blank.pdf 2>/dev/null || true; "
            "cp /root/security_audit.csv /logs/verifier/security_audit.csv "
            "2>/dev/null || true; "
            "cp /app/output/itinerary.json /logs/verifier/itinerary.json "
            "2>/dev/null || true; "
            f"{artifact_command}"
            "if [ \"$status\" -eq 0 ] && [ \"$prelude_status\" -eq 0 ]; then "
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

COMMON_PY312_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="common-pytest-ctrf-py312-v1",
    families=(
        "court-form-filling",
        "enterprise-information-search",
        "financial-analysis",
        "offer-letter-generator",
        "organize-messy-files",
        "schedule-planning",
        "stock-data-visualization",
        "video-object-counting",
    ),
    requirements=("pytest==8.4.1", "pytest-json-ctrf==0.3.5"),
    import_probe=(
        "import pytest; assert pytest.__version__ == '8.4.1'"
    ),
)

CHINESE_POEM_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="chinese-poem-py312-v1",
    families=("chinese-poem-generator",),
    requirements=(
        "pytest==8.4.1",
        "pytest-json-ctrf==0.3.5",
        "pypinyin==0.55.0",
    ),
    import_probe=(
        "import pytest, pypinyin; "
        "assert pytest.__version__ == '8.4.1'; "
        "assert pypinyin.__version__ == '0.55.0'"
    ),
)

COMMON_PY310_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="common-pytest-ctrf-py310-v1",
    families=(
        "dependency-vulnerability-check",
        "earthquake-plate-calculation",
    ),
    requirements=("pytest==8.4.1", "pytest-json-ctrf==0.3.5"),
    import_probe=(
        "import pytest; assert pytest.__version__ == '8.4.1'"
    ),
    python_version="3.10",
    python_abi="cp310",
)

COMMON_PY311_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="common-pytest-ctrf-py311-v1",
    families=("travel-planning",),
    requirements=("pytest==8.4.1", "pytest-json-ctrf==0.3.5"),
    import_probe=(
        "import pytest; assert pytest.__version__ == '8.4.1'"
    ),
    python_version="3.11",
    python_abi="cp311",
)

COMMON_PY38_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="common-pytest-ctrf-py38-v1",
    families=("temperature-simulation",),
    requirements=("pytest==8.3.5", "pytest-json-ctrf==0.4.1"),
    import_probe=(
        "import pytest; assert pytest.__version__ == '8.3.5'"
    ),
    python_version="3.8",
    python_abi="cp38",
)

WEIGHTED_GDP_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="weighted-gdp-ssconvert-py312-v1",
    families=("weighted-gdp-calculation",),
    requirements=("pytest==8.4.1", "pytest-json-ctrf==0.3.5"),
    import_probe=(
        "import openpyxl, pytest, shutil; "
        "assert pytest.__version__ == '8.4.1'; "
        "assert openpyxl.__version__ == '3.1.5'; "
        "assert shutil.which('ssconvert')"
    ),
    semantic_prelude_id="weighted_gdp_ssconvert_v1",
    semantic_prelude_command=_WEIGHTED_GDP_PRELUDE_COMMAND,
    artifact_command=_WEIGHTED_GDP_ARTIFACT_COMMAND,
    probe_workspace_mode="image_root",
)

DRUID_SECURITY_VERIFIER_PROFILE = OfflineVerifierProfile(
    profile_id="druid-security-py312-v1",
    families=("fix-security-bug",),
    requirements=(
        "pytest==8.4.1",
        "pytest-json-ctrf==0.3.5",
        "requests==2.31.0",
    ),
    import_probe=(
        "import pytest, requests; "
        "assert pytest.__version__ == '8.4.1'; "
        "assert requests.__version__ == '2.31.0'"
    ),
    semantic_prelude_id="druid_deploy_restart_readiness_v1",
    semantic_prelude_command=_DRUID_SECURITY_PRELUDE_COMMAND,
    artifact_command=_DRUID_SECURITY_ARTIFACT_COMMAND,
    probe_workspace_mode="image_root",
    activation_blocker="druid_maven_cache_incomplete",
)

OFFLINE_VERIFIER_PROFILE_CATALOG = (
    POSTER_VERIFIER_PROFILE,
    COMMON_PY312_VERIFIER_PROFILE,
    CHINESE_POEM_VERIFIER_PROFILE,
    COMMON_PY310_VERIFIER_PROFILE,
    COMMON_PY311_VERIFIER_PROFILE,
    COMMON_PY38_VERIFIER_PROFILE,
    WEIGHTED_GDP_VERIFIER_PROFILE,
    DRUID_SECURITY_VERIFIER_PROFILE,
)
OFFLINE_VERIFIER_PROFILES = tuple(
    profile
    for profile in OFFLINE_VERIFIER_PROFILE_CATALOG
    if profile.activation_blocker is None
)


@dataclass(frozen=True)
class SemanticPreludeReceipt:
    required: bool
    valid: bool
    succeeded: bool
    error_type: str | None
    prelude_id: str | None
    exit_code: int | None
    details: Mapping[str, str]
    receipt_hash: str


def offline_verifier_profile_for_family(
    family: str,
) -> OfflineVerifierProfile | None:
    for profile in OFFLINE_VERIFIER_PROFILES:
        if family in profile.families:
            return profile
    return None


def offline_verifier_catalog_profile_for_family(
    family: str,
) -> OfflineVerifierProfile | None:
    for profile in OFFLINE_VERIFIER_PROFILE_CATALOG:
        if family in profile.families:
            return profile
    return None


def offline_verifier_activation_blocker_for_family(family: str) -> str | None:
    profile = offline_verifier_catalog_profile_for_family(family)
    return profile.activation_blocker if profile is not None else None


def inspect_semantic_prelude_receipt(
    *,
    profile: OfflineVerifierProfile | None,
    verifier_dir: Path,
) -> SemanticPreludeReceipt:
    required = bool(profile and profile.semantic_prelude_id)
    if not required:
        payload = {
            "required": False,
            "profile_hash": profile.profile_hash if profile is not None else None,
        }
        return SemanticPreludeReceipt(
            required=False,
            valid=True,
            succeeded=True,
            error_type=None,
            prelude_id=None,
            exit_code=None,
            details={},
            receipt_hash=stable_hash(payload),
        )

    assert profile is not None
    receipt_file = verifier_dir / "semantic_prelude.json"
    details_file = verifier_dir / "semantic_prelude_details.txt"
    evidence: dict[str, Any] = {
        "required": True,
        "profile_hash": profile.profile_hash,
        "expected_prelude_id": profile.semantic_prelude_id,
    }
    error_type: str | None = None
    observed_id: str | None = None
    exit_code: int | None = None
    details: dict[str, str] = {}
    if not receipt_file.is_file():
        error_type = "semantic_prelude_receipt_missing"
    else:
        evidence["receipt_sha256"] = _sha256(receipt_file)
        try:
            payload = json.loads(receipt_file.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("semantic prelude receipt must be an object")
            observed_id = str(payload.get("prelude_id") or "") or None
            raw_exit = payload.get("exit_code")
            if isinstance(raw_exit, bool) or not isinstance(raw_exit, int):
                raise ValueError("semantic prelude exit code must be an integer")
            if raw_exit < 0 or raw_exit > 255:
                raise ValueError("semantic prelude exit code is out of range")
            exit_code = raw_exit
            if observed_id != profile.semantic_prelude_id:
                error_type = "semantic_prelude_receipt_id_mismatch"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            error_type = "semantic_prelude_receipt_malformed"

    if not details_file.is_file():
        if error_type is None:
            error_type = "semantic_prelude_details_missing"
    else:
        evidence["details_sha256"] = _sha256(details_file)
        try:
            for raw_line in details_file.read_text(encoding="utf-8").splitlines():
                key, separator, value = raw_line.partition("=")
                if (
                    separator != "="
                    or not _PRELUDE_DETAIL_KEY.fullmatch(key)
                    or not _PRELUDE_DETAIL_VALUE.fullmatch(value)
                    or key in details
                ):
                    raise ValueError("semantic prelude detail is malformed")
                details[key] = value
            if not details:
                raise ValueError("semantic prelude details are empty")
        except (OSError, ValueError):
            if error_type is None:
                error_type = "semantic_prelude_details_malformed"
            details = {}
    evidence.update(
        {
            "observed_prelude_id": observed_id,
            "exit_code": exit_code,
            "details": dict(sorted(details.items())),
        }
    )
    valid = error_type is None
    return SemanticPreludeReceipt(
        required=True,
        valid=valid,
        succeeded=bool(valid and exit_code == 0),
        error_type=error_type,
        prelude_id=observed_id,
        exit_code=exit_code,
        details=details,
        receipt_hash=stable_hash(evidence),
    )


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
        runtime_key = offline_verifier_runtime_key(profile=profile)
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
            if labels.get("org.assumption-agent.verifier.profile") != profile.profile_hash:
                raise PermissionError("offline verifier volume profile mismatch")
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
                "-e",
                "OPENAI_API_KEY=offline-verifier-secret-canary",
                "-e",
                "ASSUMPTION_V2_API_KEY=offline-verifier-secret-canary",
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
                    "semantic_prelude_id": profile.semantic_prelude_id,
                    "probe_workspace_mode": profile.probe_workspace_mode,
                    "activation_blocker": profile.activation_blocker,
                    "raw_content_persisted": False,
                },
            )
        )


def offline_verifier_runtime_key(
    *,
    profile: OfflineVerifierProfile,
) -> str:
    return stable_hash(
        {
            "policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "runtime_scope": "profile_python_abi_v1",
            "profile_hash": profile.profile_hash,
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
    event_sink: EventSink | None = None,
    trace_id: str = "offline-verifier-prepare",
) -> Mapping[str, Any]:
    sink = event_sink or NullEventSink()
    _emit_preparation_event(
        sink,
        event="skilllearn_offline_verifier_preparation_started",
        trace_id=trace_id,
        profile=profile,
        payload={
            "base_image_tag_hash": stable_hash({"tag": base_image_tag}),
            "refresh_wheels": refresh_wheels,
        },
    )
    image = delegate.run(
        ["docker", "image", "inspect", base_image_tag],
        capture_output=True,
        text=True,
    )
    if int(getattr(image, "returncode", 1)) != 0:
        _emit_preparation_failure(
            sink,
            trace_id=trace_id,
            profile=profile,
            step="base_image_inspect",
            completed=image,
        )
        raise RuntimeError("offline verifier base image is unavailable")
    image_payload = json.loads(str(getattr(image, "stdout", "") or ""))[0]
    base_image_id = str(image_payload.get("Id") or "")
    if not base_image_id.startswith("sha256:"):
        raise RuntimeError("offline verifier base image has no immutable ID")
    runtime_key = offline_verifier_runtime_key(profile=profile)
    volume_name = offline_verifier_volume_name(runtime_key)
    cache_root = Path(
        os.environ.get(
            "ASSUMPTION_V2_OFFLINE_VERIFIER_CACHE",
            Path.home() / ".cache" / "assumption-agent-v2" / "offline-verifier",
        )
    ).expanduser().resolve()
    wheelhouse = cache_root / "wheelhouses" / profile.wheelhouse_key / "wheels"
    legacy_wheelhouse = cache_root / profile.profile_hash / "wheels"
    if not wheelhouse.exists() and legacy_wheelhouse.is_dir():
        wheelhouse.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(legacy_wheelhouse, wheelhouse)
    if refresh_wheels and wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    wheels = tuple(sorted(wheelhouse.glob("*.whl")))
    wheelhouse_reused = bool(wheels) and not refresh_wheels
    online_download_attempted = False
    runtime_cache = SkillLearnOfflineVerifierRuntimeCache(event_sink=sink)
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
    _emit_preparation_event(
        sink,
        event="skilllearn_offline_verifier_cache_checked",
        trace_id=trace_id,
        profile=profile,
        payload={
            "runtime_key": runtime_key,
            "runtime_reused": runtime_reused,
            "wheelhouse_key": profile.wheelhouse_key,
            "wheelhouse_reused": wheelhouse_reused,
            "wheel_count": len(wheels),
        },
    )
    if not runtime_reused:
        if not wheels:
            online_download_attempted = True
            _emit_preparation_event(
                sink,
                event="skilllearn_offline_verifier_wheel_download_started",
                trace_id=trace_id,
                profile=profile,
                payload={
                    "package_index_origin": TUNA_PYPI_INDEX_URL,
                    "requirements_hash": stable_hash(profile.requirements),
                    "python_version": profile.python_version,
                    "python_abi": profile.python_abi,
                    "platform": profile.platform,
                },
            )
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
                _emit_preparation_failure(
                    sink,
                    trace_id=trace_id,
                    profile=profile,
                    step="wheel_download",
                    completed=downloaded,
                )
                raise RuntimeError("offline verifier wheel download failed")
            wheels = tuple(sorted(wheelhouse.glob("*.whl")))
            wheelhouse_reused = False
            _emit_preparation_event(
                sink,
                event="skilllearn_offline_verifier_wheel_download_completed",
                trace_id=trace_id,
                profile=profile,
                payload={
                    "package_index_origin": TUNA_PYPI_INDEX_URL,
                    "wheel_count": len(wheels),
                    "wheel_total_bytes": sum(path.stat().st_size for path in wheels),
                    "wheel_set_hash": stable_hash(
                        [
                            {
                                "filename": path.name,
                                "size": path.stat().st_size,
                                "sha256": _sha256(path),
                            }
                            for path in wheels
                        ]
                    ),
                },
            )
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
            _emit_preparation_failure(
                sink,
                trace_id=trace_id,
                profile=profile,
                step="runtime_volume_create",
                completed=created,
            )
            raise RuntimeError("offline verifier volume creation failed")
        install_command = (
            "set -eu; rm -rf /runtime/site; mkdir -p /runtime/site; "
            "python3 -m pip install --no-index --no-cache-dir "
            "--disable-pip-version-check --find-links=/wheels "
            "--target=/runtime/site "
            + " ".join(profile.requirements)
        )
        _emit_preparation_event(
            sink,
            event="skilllearn_offline_verifier_install_started",
            trace_id=trace_id,
            profile=profile,
            payload={
                "runtime_key": runtime_key,
                "wheel_count": len(wheels),
                "container_network": "none",
            },
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
            _emit_preparation_failure(
                sink,
                trace_id=trace_id,
                profile=profile,
                step="runtime_wheel_install",
                completed=installed,
            )
            raise RuntimeError("offline verifier wheel installation failed")
        runtime_cache.ensure(
            profile=profile,
            base_image_tag=base_image_tag,
            base_image_id=base_image_id,
            delegate=delegate,
            trace_id="offline-verifier-prepare-verify",
        )
        _emit_preparation_event(
            sink,
            event="skilllearn_offline_verifier_install_completed",
            trace_id=trace_id,
            profile=profile,
            payload={
                "runtime_key": runtime_key,
                "container_network": "none",
                "probe_passed": True,
            },
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
        "semantic_prelude_id": profile.semantic_prelude_id,
        "probe_workspace_mode": profile.probe_workspace_mode,
        "activation_blocker": profile.activation_blocker,
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
    _emit_preparation_event(
        sink,
        event="skilllearn_offline_verifier_preparation_completed",
        trace_id=trace_id,
        profile=profile,
        payload={
            "runtime_key": runtime_key,
            "runtime_reused": runtime_reused,
            "wheelhouse_reused": wheelhouse_reused,
            "online_download_attempted": online_download_attempted,
            "wheel_count": len(wheels),
            "wheel_total_bytes": report["wheel_total_bytes"],
            "receipt_hash": report["receipt_hash"],
            "probe_passed": True,
        },
    )
    return report


def probe_offline_verifier_runtime(
    *,
    profile: OfflineVerifierProfile,
    base_image_tag: str,
    workspace: Path,
    tests_dir: Path,
    report_path: Path,
    delegate: Any = subprocess,
    event_sink: EventSink | None = None,
    trace_id: str = "offline-verifier-probe",
) -> Mapping[str, Any]:
    sink = event_sink or NullEventSink()
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
    runtime_key = offline_verifier_runtime_key(profile=profile)
    volume_name = offline_verifier_volume_name(runtime_key)
    _emit_preparation_event(
        sink,
        event="skilllearn_offline_verifier_probe_started",
        trace_id=trace_id,
        profile=profile,
        payload={
            "base_image_tag_hash": stable_hash({"tag": base_image_tag}),
            "workspace_hash": (
                _directory_hash(workspace)
                if profile.probe_workspace_mode == "empty_mount"
                else None
            ),
            "probe_workspace_mode": profile.probe_workspace_mode,
            "activation_blocker": profile.activation_blocker,
            "tests_hash": _directory_hash(tests_dir),
            "container_network": "none",
        },
    )
    SkillLearnOfflineVerifierRuntimeCache(event_sink=sink).ensure(
        profile=profile,
        base_image_tag=base_image_tag,
        base_image_id=base_image_id,
        delegate=delegate,
        trace_id=f"{trace_id}:runtime",
    )
    with tempfile.TemporaryDirectory(
        prefix="assumption-v2-verifier-probe-"
    ) as raw_logs_root:
        logs_root = Path(raw_logs_root)
        workspace_mount = (
            ["-v", f"{workspace}:/root"]
            if profile.probe_workspace_mode == "empty_mount"
            else []
        )
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
                *workspace_mount,
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
        semantic_prelude = inspect_semantic_prelude_receipt(
            profile=profile,
            verifier_dir=logs_root / "verifier",
        )
        report: dict[str, Any] = {
            "report_version": "offline_verifier_probe_receipt_v2",
            "policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "activation_blocker": profile.activation_blocker,
            "runtime_key": runtime_key,
            "runtime_volume_hash": stable_hash({"volume": volume_name}),
            "base_image_id": base_image_id,
            "container_network": "none",
            "runtime_mount_read_only": True,
            "original_online_test_script_executed": False,
            "model_secret_env_unset_before_verifier": True,
            "model_secret_env_canary_injected": True,
            "container_exit": int(getattr(completed, "returncode", 1)),
            "reward": reward,
            "test_count": test_count,
            "ctrf_sha256": _sha256(ctrf_file) if ctrf_file.is_file() else None,
            "probe_workspace_mode": profile.probe_workspace_mode,
            "workspace_mounted": profile.probe_workspace_mode == "empty_mount",
            "workspace_hash": (
                _directory_hash(workspace)
                if profile.probe_workspace_mode == "empty_mount"
                else None
            ),
            "tests_hash": _directory_hash(tests_dir),
            "semantic_prelude_required": semantic_prelude.required,
            "semantic_prelude_valid": semantic_prelude.valid,
            "semantic_prelude_succeeded": semantic_prelude.succeeded,
            "semantic_prelude_id": semantic_prelude.prelude_id,
            "semantic_prelude_exit_code": semantic_prelude.exit_code,
            "semantic_prelude_details": dict(semantic_prelude.details),
            "semantic_prelude_receipt_hash": semantic_prelude.receipt_hash,
            "probe_passed": (
                int(getattr(completed, "returncode", 1)) == 0
                and reward in {0, 1}
                and test_count > 0
                and semantic_prelude.valid
                and semantic_prelude.succeeded
            ),
            "raw_content_persisted": False,
        }
    report["receipt_hash"] = stable_hash(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit_preparation_event(
        sink,
        event="skilllearn_offline_verifier_probe_completed",
        trace_id=trace_id,
        profile=profile,
        payload={
            "runtime_key": runtime_key,
            "container_network": "none",
            "reward": reward,
            "test_count": test_count,
            "semantic_prelude_required": semantic_prelude.required,
            "semantic_prelude_valid": semantic_prelude.valid,
            "semantic_prelude_succeeded": semantic_prelude.succeeded,
            "semantic_prelude_id": semantic_prelude.prelude_id,
            "semantic_prelude_exit_code": semantic_prelude.exit_code,
            "semantic_prelude_details": dict(semantic_prelude.details),
            "semantic_prelude_receipt_hash": semantic_prelude.receipt_hash,
            "model_secret_env_canary_injected": True,
            "probe_passed": report["probe_passed"],
            "receipt_hash": report["receipt_hash"],
        },
    )
    return report


def _emit_preparation_event(
    sink: EventSink,
    *,
    event: str,
    trace_id: str,
    profile: OfflineVerifierProfile,
    payload: Mapping[str, Any],
) -> None:
    sink.emit(
        Event(
            event=event,
            stage="benchmark.skilllearn.offline_verifier_preparation",
            trace_id=trace_id,
            payload={
                "policy": OFFLINE_VERIFIER_POLICY_VERSION,
                "profile_id": profile.profile_id,
                "profile_hash": profile.profile_hash,
                "activation_blocker": profile.activation_blocker,
                **dict(payload),
                "secret_value_persisted": False,
                "raw_content_persisted": False,
            },
        )
    )


def _emit_preparation_failure(
    sink: EventSink,
    *,
    trace_id: str,
    profile: OfflineVerifierProfile,
    step: str,
    completed: Any,
) -> None:
    _emit_preparation_event(
        sink,
        event="skilllearn_offline_verifier_preparation_failed",
        trace_id=trace_id,
        profile=profile,
        payload={
            "step": step,
            **_process_failure_diagnostic(completed),
        },
    )


def _process_failure_diagnostic(completed: Any) -> dict[str, Any]:
    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    combined = "\n".join(value for value in (stdout, stderr) if value)
    signal_lines = [
        _DIAGNOSTIC_SECRET.sub("[REDACTED]", line.strip())
        for line in combined.splitlines()
        if _DIAGNOSTIC_SIGNAL.search(line)
    ]
    summary = "\n".join(signal_lines[-12:])[-2000:]
    return {
        "return_code": int(getattr(completed, "returncode", 1)),
        "process_output_hash": stable_hash({"stdout": stdout, "stderr": stderr}),
        "diagnostic_summary": summary or None,
        "diagnostic_summary_persisted": bool(summary),
    }


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
    parser.add_argument("--events", type=Path)
    args = parser.parse_args(argv)
    profile = next(
        (
            row
            for row in OFFLINE_VERIFIER_PROFILE_CATALOG
            if row.profile_id == args.profile
        ),
        None,
    )
    if profile is None:
        raise ValueError(f"unknown offline verifier profile: {args.profile}")
    if (args.probe_workspace is None) != (args.probe_tests is None):
        raise ValueError("offline verifier probe requires workspace and tests together")
    sink = JsonlEventSink(args.events) if args.events is not None else None
    if args.probe_workspace is not None:
        report = probe_offline_verifier_runtime(
            profile=profile,
            base_image_tag=args.base_image_tag,
            workspace=args.probe_workspace,
            tests_dir=args.probe_tests,
            report_path=args.report,
            event_sink=sink,
        )
    else:
        report = prepare_offline_verifier_runtime(
            profile=profile,
            base_image_tag=args.base_image_tag,
            report_path=args.report,
            refresh_wheels=args.refresh_wheels,
            event_sink=sink,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
