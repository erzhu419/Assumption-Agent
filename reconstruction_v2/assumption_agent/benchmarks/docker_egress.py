from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from ..events import Event, EventSink, NullEventSink
from ..models import stable_hash


DOCKER_EGRESS_POLICY_VERSION = "docker_user_endpoint_allowlist_v1"
DEPENDENCY_CACHE_POLICY_VERSION = "fail_closed_prebuilt_only_v1"
PROVIDER_DNS_POLICY_VERSION = "pinned_hosts_no_external_dns_v1"
TRIAL_NETWORK_BUDGET_POLICY_VERSION = "docker_stats_hard_byte_cap_v1"
DEFAULT_TRIAL_NETWORK_BYTE_LIMIT = 32 * 1024 * 1024
RESTRICTED_NETWORK_NAME = "assumption-v2-restricted"
RESTRICTED_NETWORK_SUBNET = "172.29.0.0/24"
RESTRICTED_FIREWALL_CHAIN = "ASSUMPTION_V2_EGRESS"
FIREWALL_HELPER_IMAGE = "ubuntu:24.04"
_FIREWALL_LOCK_PATH = Path("/tmp/assumption-v2-docker-egress.lock")

_OFFLINE_ENVIRONMENT = (
    "PIP_NO_INDEX=1",
    "PIP_DISABLE_PIP_VERSION_CHECK=1",
    "PIP_DEFAULT_TIMEOUT=1",
    "UV_OFFLINE=1",
    "npm_config_offline=true",
    "npm_config_audit=false",
    "npm_config_fund=false",
    "npm_config_update_notifier=false",
    "HF_HUB_OFFLINE=1",
    "HF_DATASETS_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "CARGO_NET_OFFLINE=true",
    "GIT_TERMINAL_PROMPT=0",
)

_ensure_lock = threading.Lock()
_ensured_fingerprint: str | None = None


@dataclass(frozen=True)
class DockerEgressPolicy:
    endpoint_origin: str
    endpoint_host: str
    endpoint_port: int
    allowed_ipv4s: tuple[str, ...]
    network_name: str = RESTRICTED_NETWORK_NAME
    network_subnet: str = RESTRICTED_NETWORK_SUBNET

    @classmethod
    def from_env(cls) -> "DockerEgressPolicy":
        base_url = (
            os.environ.get("ASSUMPTION_V2_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).strip()
        allowed = os.environ.get("ASSUMPTION_V2_API_ALLOWED_IPV4S", "")
        return cls.from_values(base_url=base_url, allowed_ipv4s=allowed.split(","))

    @classmethod
    def from_values(
        cls,
        *,
        base_url: str,
        allowed_ipv4s: Sequence[str],
    ) -> "DockerEgressPolicy":
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("restricted model endpoint must be one HTTPS origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("restricted model endpoint contains unsupported URL fields")
        port = parsed.port or 443
        if port != 443:
            raise ValueError("restricted model endpoint must use TCP port 443")
        normalized_ips: list[str] = []
        for raw in allowed_ipv4s:
            value = str(raw).strip()
            if not value:
                continue
            address = ipaddress.ip_address(value)
            if not isinstance(address, ipaddress.IPv4Address):
                raise ValueError("restricted model endpoint currently requires IPv4")
            normalized_ips.append(str(address))
        normalized = tuple(sorted(set(normalized_ips)))
        if not normalized:
            raise ValueError("restricted model endpoint requires pinned IPv4 addresses")
        origin = f"https://{parsed.hostname}"
        return cls(
            endpoint_origin=origin,
            endpoint_host=parsed.hostname,
            endpoint_port=port,
            allowed_ipv4s=normalized,
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "policy": DOCKER_EGRESS_POLICY_VERSION,
                "dns_policy": PROVIDER_DNS_POLICY_VERSION,
                "endpoint_origin": self.endpoint_origin,
                "endpoint_port": self.endpoint_port,
                "allowed_ipv4s": self.allowed_ipv4s,
                "network_name": self.network_name,
                "network_subnet": self.network_subnet,
            }
        )

    def docker_run_args(self) -> list[str]:
        args = [
            "--pull",
            "never",
            "--network",
            self.network_name,
            "--dns",
            "127.0.0.1",
            "--dns-option",
            "attempts:1",
            "--dns-option",
            "timeout:1",
        ]
        for address in self.allowed_ipv4s:
            args.extend(["--add-host", f"{self.endpoint_host}:{address}"])
        for row in _OFFLINE_ENVIRONMENT:
            args.extend(["-e", row])
        return args

    def provenance(self) -> dict[str, Any]:
        return {
            "container_egress_policy": DOCKER_EGRESS_POLICY_VERSION,
            "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
            "provider_dns_policy": PROVIDER_DNS_POLICY_VERSION,
            "endpoint_origin": self.endpoint_origin,
            "endpoint_port": self.endpoint_port,
            "allowed_ipv4_count": len(self.allowed_ipv4s),
            "allowed_ipv4_set_hash": stable_hash(
                {"allowed_ipv4s": self.allowed_ipv4s}
            ),
            "network_name": self.network_name,
            "network_subnet": self.network_subnet,
            "firewall_chain": RESTRICTED_FIREWALL_CHAIN,
            "policy_fingerprint": self.fingerprint,
            "external_dns_enabled": False,
            "ipv6_enabled": False,
            "container_image_pull_enabled": False,
            "package_manager_online_mode_enabled": False,
        }

    def ensure(
        self,
        *,
        event_sink: EventSink | None = None,
        trace_id: str = "skilllearn-docker-egress",
    ) -> None:
        global _ensured_fingerprint
        sink = event_sink or NullEventSink()
        with _ensure_lock:
            if _ensured_fingerprint == self.fingerprint:
                return
            _FIREWALL_LOCK_PATH.touch(mode=0o600, exist_ok=True)
            with _FIREWALL_LOCK_PATH.open("r+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                _ensure_network(self)
                rules_hash = _ensure_firewall(self)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            _ensured_fingerprint = self.fingerprint
            sink.emit(
                Event(
                    event="skilllearn_container_egress_guard_ready",
                    stage="benchmark.skilllearn.network_isolation",
                    trace_id=trace_id,
                    payload={
                        **self.provenance(),
                        "firewall_rules_hash": rules_hash,
                        "secret_value_persisted": False,
                        "raw_content_persisted": False,
                    },
                )
            )


def validate_env_policy() -> dict[str, Any]:
    try:
        policy = DockerEgressPolicy.from_env()
    except (TypeError, ValueError) as exc:
        return {
            "valid": False,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
        }
    try:
        byte_limit = configured_trial_network_byte_limit()
    except ValueError as exc:
        return {
            "valid": False,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
        }
    return {
        "valid": True,
        **policy.provenance(),
        "trial_network_budget_policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
        "trial_network_byte_limit": byte_limit,
    }


def configured_trial_network_byte_limit() -> int:
    raw = os.environ.get(
        "ASSUMPTION_V2_TRIAL_NETWORK_BYTE_LIMIT",
        str(DEFAULT_TRIAL_NETWORK_BYTE_LIMIT),
    )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("trial network byte limit must be an integer") from exc
    if not 1024 * 1024 <= value <= 1024 * 1024 * 1024:
        raise ValueError("trial network byte limit must be between 1 MiB and 1 GiB")
    return value


def _ensure_network(policy: DockerEgressPolicy) -> None:
    inspected = _docker(
        ["network", "inspect", policy.network_name],
        check=False,
    )
    if inspected.returncode != 0:
        created = _docker(
            [
                "network",
                "create",
                "--driver",
                "bridge",
                "--subnet",
                policy.network_subnet,
                "--label",
                f"org.assumption-agent.egress.policy={DOCKER_EGRESS_POLICY_VERSION}",
                policy.network_name,
            ],
            check=False,
        )
        if created.returncode != 0:
            raise RuntimeError("restricted Docker network creation failed")
        inspected = _docker(["network", "inspect", policy.network_name])
    try:
        row = json.loads(inspected.stdout)[0]
        labels = row.get("Labels") or {}
        ipv6_enabled = bool(row.get("EnableIPv6"))
        subnets = {
            str(entry.get("Subnet") or "")
            for entry in row.get("IPAM", {}).get("Config", [])
            if isinstance(entry, dict)
        }
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("restricted Docker network metadata is malformed") from exc
    if labels.get("org.assumption-agent.egress.policy") != DOCKER_EGRESS_POLICY_VERSION:
        raise PermissionError("restricted Docker network policy label mismatch")
    if ipv6_enabled:
        raise PermissionError("restricted Docker network must not enable IPv6")
    if subnets != {policy.network_subnet}:
        raise PermissionError("restricted Docker network subnet mismatch")


def _ensure_firewall(policy: DockerEgressPolicy) -> str:
    if _iptables(["-S", "DOCKER-USER"], check=False).returncode != 0:
        raise RuntimeError("Docker DOCKER-USER firewall chain is unavailable")
    if _iptables(["-S", RESTRICTED_FIREWALL_CHAIN], check=False).returncode != 0:
        _iptables(["-N", RESTRICTED_FIREWALL_CHAIN])

    temporary_reject = ["-s", policy.network_subnet, "-j", "REJECT"]
    _iptables(["-I", "DOCKER-USER", "1", *temporary_reject])
    try:
        _iptables(["-F", RESTRICTED_FIREWALL_CHAIN])
        for address in policy.allowed_ipv4s:
            _iptables(
                [
                    "-A",
                    RESTRICTED_FIREWALL_CHAIN,
                    "-d",
                    f"{address}/32",
                    "-p",
                    "tcp",
                    "--dport",
                    str(policy.endpoint_port),
                    "-j",
                    "ACCEPT",
                ]
            )
        _iptables(
            [
                "-A",
                RESTRICTED_FIREWALL_CHAIN,
                "-j",
                "REJECT",
                "--reject-with",
                "icmp-port-unreachable",
            ]
        )
        jump = [
            "-s",
            policy.network_subnet,
            "-j",
            RESTRICTED_FIREWALL_CHAIN,
        ]
        _delete_all_rules("DOCKER-USER", jump)
        _iptables(["-I", "DOCKER-USER", "1", *jump])
        rules = _iptables(["-S", RESTRICTED_FIREWALL_CHAIN]).stdout
    finally:
        _delete_all_rules("DOCKER-USER", temporary_reject)
    return stable_hash({"rules": rules})


def _delete_all_rules(chain: str, rule: Sequence[str]) -> None:
    while _iptables(["-C", chain, *rule], check=False).returncode == 0:
        _iptables(["-D", chain, *rule])


def _docker(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise RuntimeError("Docker egress setup command failed")
    return result


def _iptables(
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = _docker(
        [
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "host",
            "--privileged",
            "-v",
            "/:/host:ro",
            FIREWALL_HELPER_IMAGE,
            "chroot",
            "/host",
            "/usr/sbin/iptables",
            "-w",
            "10",
            *args,
        ],
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError("Docker egress firewall command failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or install SkillLearn egress guard.")
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    policy = DockerEgressPolicy.from_env()
    if args.ensure:
        policy.ensure()
    print(json.dumps(policy.provenance(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
