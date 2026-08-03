from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hegel_machine import phase3_m25_container_ceremony_cli_v1 as cli
from hegel_machine import phase3_m25_formal_container_executor_v1 as executor
from hegel_machine.phase3_m25_formal_container_executor_v1 import (
    CeremonyReadinessV1,
)


def test_execution_status_reports_ready_basis_without_stale_blocker(
    tmp_path: Path, monkeypatch,
) -> None:
    basis_commit = "12" * 20
    output = tmp_path / "status.json"
    monkeypatch.setattr(
        executor,
        "build_qualified_formal_static_basis_v1",
        lambda _value: SimpleNamespace(
            blocking_gaps=(),
            implementation_inputs={
                "m3_execution_implementation_bindings_ready": True,
                "m3_execution_implementation_binding_roots": {
                    "python_implementation_binding_root": b"p" * 32,
                    "rust_implementation_binding_root": b"r" * 32,
                },
            },
        ),
    )
    monkeypatch.setattr(
        executor,
        "load_qualified_rust_bridge_dag_binary_binding_v1",
        lambda **_kwargs: ({}, "sha256:" + "11" * 32),
    )
    monkeypatch.setattr(
        executor,
        "load_actor_protocol_archive_qualification_v1",
        lambda _value: object(),
    )

    assert cli.main(
        [
            "execution-status",
            "--basis-commit",
            basis_commit,
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="ascii"))
    assert report["schema"] == "hegel-phase3-m25-execution-status/2"
    assert report["basis_commit"] == basis_commit
    assert report["ceremony_execution_enabled_for_basis"] is True
    assert report["blocking_prerequisites"] == []
    assert report["external_genesis_executed"] is False
    assert report["formal_gates_before"] == 14
    assert report["formal_gates_after"] == 14
    assert report["child_state"] == "NOT_RUN"
    assert report["m3_run_started"] is False
    assert report["qualification_side_effects_performed"] is True
    assert report["qualification_network_mode"] == "none"
    assert report["qualification_persistent_rust_binary_verified_or_written"] is True
    assert report["qualification_non_authoritative_roots_computed"] is True
    assert report["ceremony_actor_key_seed_marker_side_effects_performed"] is False
    assert report["formal_authority_or_gate_effect"] == "NONE"


def test_execution_status_normalizes_missing_basis_error(capsys) -> None:
    basis_commit = "ab" * 20

    assert cli.main(["execution-status", "--basis-commit", basis_commit]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert error["error_code"] == "FAIL_M25_STATIC_BASIS_COMMIT"
    assert "traceback" not in error["detail"].lower()


def test_execution_status_transports_exact_basis_specific_blockers(
    tmp_path: Path, monkeypatch,
) -> None:
    basis_commit = "34" * 20
    output = tmp_path / "blocked.json"
    blockers = ("FAIL_ONE", "FAIL_TWO")
    monkeypatch.setattr(
        cli,
        "inspect_formal_ceremony_readiness_v1",
        lambda value: CeremonyReadinessV1(
            basis_commit=value,
            ready=False,
            blockers=blockers,
        ),
    )

    assert cli.main(
        [
            "execution-status",
            "--basis-commit",
            basis_commit,
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="ascii"))
    assert report["ceremony_execution_enabled_for_basis"] is False
    assert [row["failure_code"] for row in report["blocking_prerequisites"]] == [
        "FAIL_ONE",
        "FAIL_TWO",
    ]
