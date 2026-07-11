from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    POSTER_VERIFIER_PROFILE,
    offline_verifier_runtime_key,
    offline_verifier_volume_name,
    prepare_offline_verifier_runtime,
    test_script_requires_offline_profile as _requires_offline_profile,
)


class ExistingRuntimeDocker:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.base_image_id = "sha256:" + "a" * 64
        self.runtime_key = offline_verifier_runtime_key(
            profile=POSTER_VERIFIER_PROFILE,
            base_image_id=self.base_image_id,
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

    report = prepare_offline_verifier_runtime(
        profile=POSTER_VERIFIER_PROFILE,
        base_image_tag="poster-image:cached",
        report_path=tmp_path / "receipt.json",
        delegate=delegate,
    )

    assert report["runtime_reused"] is True
    assert report["online_download_attempted"] is False
    assert not any("pip" in command for command in delegate.commands)


def test_online_test_script_requires_a_local_profile(tmp_path: Path) -> None:
    online = tmp_path / "online.sh"
    local = tmp_path / "local.sh"
    online.write_text("uvx pytest /tests/test_outputs.py\n", encoding="utf-8")
    local.write_text("pytest /tests/test_outputs.py\n", encoding="utf-8")

    assert _requires_offline_profile(online) is True
    assert _requires_offline_profile(local) is False
