from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_PROFILE_CATALOG,
    OFFLINE_VERIFIER_PROFILES,
    OFFLINE_VERIFIER_POLICY_VERSION,
    DRUID_SECURITY_VERIFIER_PROFILE,
    POSTER_VERIFIER_PROFILE,
    WEIGHTED_GDP_VERIFIER_PROFILE,
    _process_failure_diagnostic,
    offline_verifier_activation_blocker_for_family,
    offline_verifier_catalog_profile_for_family,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
    offline_verifier_volume_name,
    prepare_offline_verifier_runtime,
    test_script_requires_offline_profile as _requires_offline_profile,
)
from assumption_agent.events import MemoryEventSink


class ExistingRuntimeDocker:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.base_image_id = "sha256:" + "a" * 64
        self.runtime_key = offline_verifier_runtime_key(
            profile=POSTER_VERIFIER_PROFILE
        )

    def run(self, args, *positional, **kwargs):
        command = list(args)
        self.commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"Id": self.base_image_id}]),
                stderr="",
            )
        if command[:3] == ["docker", "volume", "inspect"]:
            assert command[3] == offline_verifier_volume_name(self.runtime_key)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Labels": {
                                "org.assumption-agent.verifier.key": self.runtime_key,
                                "org.assumption-agent.verifier.policy": (
                                    OFFLINE_VERIFIER_POLICY_VERSION
                                ),
                                "org.assumption-agent.verifier.profile": (
                                    POSTER_VERIFIER_PROFILE.profile_hash
                                ),
                            }
                        }
                    ]
                ),
                stderr="",
            )
        if command[:2] == ["docker", "run"]:
            assert command[command.index("--network") + 1] == "none"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")


def test_existing_offline_runtime_never_redownloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ASSUMPTION_V2_OFFLINE_VERIFIER_CACHE", str(tmp_path / "cache"))
    delegate = ExistingRuntimeDocker()
    sink = MemoryEventSink()

    report = prepare_offline_verifier_runtime(
        profile=POSTER_VERIFIER_PROFILE,
        base_image_tag="poster-image:cached",
        report_path=tmp_path / "receipt.json",
        delegate=delegate,
        event_sink=sink,
        trace_id="existing-runtime",
    )

    assert report["runtime_reused"] is True
    assert report["online_download_attempted"] is False
    assert not any("pip" in command for command in delegate.commands)
    assert any(
        row["event"] == "skilllearn_offline_verifier_preparation_completed"
        and row["payload"]["runtime_reused"] is True
        and row["payload"]["online_download_attempted"] is False
        for row in sink.events
    )


def test_offline_profile_family_mapping_is_unique() -> None:
    families = [
        family
        for profile in OFFLINE_VERIFIER_PROFILES
        for family in profile.families
    ]

    catalog_families = [
        family
        for profile in OFFLINE_VERIFIER_PROFILE_CATALOG
        for family in profile.families
    ]

    assert len(OFFLINE_VERIFIER_PROFILES) == 7
    assert len(families) == 15
    assert len(set(families)) == len(families)
    assert len({profile.profile_id for profile in OFFLINE_VERIFIER_PROFILES}) == 7
    assert len({profile.profile_hash for profile in OFFLINE_VERIFIER_PROFILES}) == 7
    assert len(OFFLINE_VERIFIER_PROFILE_CATALOG) == 8
    assert len(catalog_families) == 16
    assert len(set(catalog_families)) == len(catalog_families)
    assert WEIGHTED_GDP_VERIFIER_PROFILE.probe_workspace_mode == "image_root"
    assert DRUID_SECURITY_VERIFIER_PROFILE.probe_workspace_mode == "image_root"
    assert DRUID_SECURITY_VERIFIER_PROFILE not in OFFLINE_VERIFIER_PROFILES
    assert offline_verifier_profile_for_family("fix-security-bug") is None
    assert (
        offline_verifier_catalog_profile_for_family("fix-security-bug")
        is DRUID_SECURITY_VERIFIER_PROFILE
    )
    assert offline_verifier_activation_blocker_for_family("fix-security-bug") == (
        "druid_maven_cache_incomplete"
    )
    assert "unset OPENAI_API_KEY" in DRUID_SECURITY_VERIFIER_PROFILE.verifier_command
    assert "ssconvert" in WEIGHTED_GDP_VERIFIER_PROFILE.verifier_command
    assert "start-single-server-small" in DRUID_SECURITY_VERIFIER_PROFILE.verifier_command


def test_process_failure_diagnostic_redacts_credentials() -> None:
    completed = SimpleNamespace(
        returncode=1,
        stdout="ERROR: token sk-12345678 was rejected\n",
        stderr="ERROR: https://user:password@example.invalid/simple failed\n",
    )

    diagnostic = _process_failure_diagnostic(completed)

    assert diagnostic["return_code"] == 1
    assert diagnostic["diagnostic_summary_persisted"] is True
    assert "sk-12345678" not in diagnostic["diagnostic_summary"]
    assert "user:password" not in diagnostic["diagnostic_summary"]
    assert diagnostic["diagnostic_summary"].count("[REDACTED]") == 2


def test_online_test_script_requires_a_local_profile(tmp_path: Path) -> None:
    online = tmp_path / "online.sh"
    local = tmp_path / "local.sh"
    online.write_text("uvx pytest /tests/test_outputs.py\n", encoding="utf-8")
    local.write_text("pytest /tests/test_outputs.py\n", encoding="utf-8")

    assert _requires_offline_profile(online) is True
    assert _requires_offline_profile(local) is False
