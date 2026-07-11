from __future__ import annotations

from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import docker_egress
from assumption_agent.benchmarks.docker_egress import (
    DockerEgressPolicy,
    RESTRICTED_FIREWALL_CHAIN,
)


def test_provider_only_policy_pins_dns_and_disables_online_package_clients() -> None:
    policy = DockerEgressPolicy.from_values(
        base_url="https://ruoli.dev/v1",
        allowed_ipv4s=("45.78.76.197", "45.78.76.197"),
    )

    assert policy.endpoint_origin == "https://ruoli.dev"
    assert policy.allowed_ipv4s == ("45.78.76.197",)
    args = policy.docker_run_args()
    assert args[args.index("--network") + 1] == "assumption-v2-restricted"
    assert args[args.index("--dns") + 1] == "127.0.0.1"
    assert args[args.index("--pull") + 1] == "never"
    assert "ruoli.dev:45.78.76.197" in args
    assert "PIP_NO_INDEX=1" in args
    assert "npm_config_offline=true" in args
    assert "HF_HUB_OFFLINE=1" in args
    assert policy.provenance()["ipv6_enabled"] is False


@pytest.mark.parametrize(
    ("base_url", "addresses"),
    [
        ("http://ruoli.dev", ("45.78.76.197",)),
        ("https://ruoli.dev:8443", ("45.78.76.197",)),
        ("https://ruoli.dev", ()),
        ("https://ruoli.dev", ("not-an-ip",)),
        ("https://ruoli.dev", ("2001:db8::1",)),
    ],
)
def test_provider_only_policy_rejects_unpinned_or_unsafe_routes(
    base_url: str,
    addresses: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        DockerEgressPolicy.from_values(
            base_url=base_url,
            allowed_ipv4s=addresses,
        )


def test_firewall_plan_allows_only_provider_then_rejects(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_iptables(args, *, check=True):
        command = list(args)
        commands.append(command)
        if command[0] == "-C":
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if command[:2] == ["-S", RESTRICTED_FIREWALL_CHAIN]:
            present = any(
                row[:2] == ["-A", RESTRICTED_FIREWALL_CHAIN]
                for row in commands
            )
            return SimpleNamespace(
                returncode=0 if present else 1,
                stdout="-A ASSUMPTION_V2_EGRESS -j REJECT\n" if present else "",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(docker_egress, "_iptables", fake_iptables)
    policy = DockerEgressPolicy.from_values(
        base_url="https://ruoli.dev",
        allowed_ipv4s=("45.78.76.197",),
    )

    rules_hash = docker_egress._ensure_firewall(policy)

    assert rules_hash
    allow = next(
        row
        for row in commands
        if row[:2] == ["-A", RESTRICTED_FIREWALL_CHAIN] and "ACCEPT" in row
    )
    assert "45.78.76.197/32" in allow
    assert "443" in allow
    assert any(
        row[:2] == ["-A", RESTRICTED_FIREWALL_CHAIN] and "REJECT" in row
        for row in commands
    )
    assert not any("0.0.0.0/0" in row for row in commands)
