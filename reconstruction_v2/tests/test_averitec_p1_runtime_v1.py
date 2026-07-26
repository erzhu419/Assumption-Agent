from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from assumption_agent.benchmarks.averitec_p1_runtime_v1 import (
    AveritecP1RuntimeError,
    _network_audit,
)


def test_network_audit_accepts_only_kernel_blocked_ip_family_attempts() -> None:
    with tempfile.TemporaryDirectory(
        prefix="averitec-network-", dir="/tmp"
    ) as temporary:
        path = Path(temporary) / "network.strace"
        path.write_text(
            "1 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = -1 "
            "EAFNOSUPPORT (Address family not supported by protocol)\n"
            "1 socket(AF_UNIX, SOCK_STREAM, 0) = 3\n",
            encoding="utf-8",
        )
        receipt = _network_audit(path)
        assert receipt["blocked_IP_family_syscall_count"] == 1
        assert receipt["nonblocked_IP_family_syscall_count"] == 0

        path.write_text(
            "1 socket(AF_INET, SOCK_STREAM, IPPROTO_IP) = 3\n",
            encoding="utf-8",
        )
        with pytest.raises(AveritecP1RuntimeError):
            _network_audit(path)
