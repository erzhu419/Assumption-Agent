from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from assumption_agent.benchmarks import tatqa_p18_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p18_implementation_freeze_v1 as freeze
from assumption_agent.benchmarks import tatqa_p18_public_canary_v1 as canary
from replication_runtime.tatqa_p18_v1 import formal_runtime


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _self_hashed(body: dict[str, object]) -> dict[str, object]:
    semantic = _canonical(body).rstrip(b"\n")
    return {**body, "self_sha256": hashlib.sha256(semantic).hexdigest()}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "p18@example.invalid")
    _git(root, "config", "user.name", "P18 Test")


def _network_preflight() -> dict[str, object]:
    return {
        "network_properties": list(formal_runtime.SYSTEMD_NETWORK_PROPERTIES),
        "returncode": 0,
        "stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }


def test_runtime_fingerprint_has_exact_five_tree_receipts_and_formal_schema(
    tmp_path: Path, monkeypatch
) -> None:
    assets: dict[str, Path] = {}
    for name in freeze.ASSET_NAMES:
        root = tmp_path / name
        root.mkdir()
        (root / "asset.bin").write_bytes(name.encode("ascii"))
        assets[name] = root
    output = tmp_path / "fingerprint.json"
    receipt = freeze.build_runtime_fingerprint(
        output_path=output,
        asset_roots=assets,
        runtime_inventory={"python": "3.11.9", "torch": "2.7", "GPU_count": 2},
        systemd_network_preflight=_network_preflight(),
        runtime_implementation_commit="a" * 40,
    )
    assert receipt["schema"] == formal_runtime.__dict__.get(
        "RUNTIME_FINGERPRINT_SCHEMA", freeze.RUNTIME_FINGERPRINT_SCHEMA
    )
    assert receipt["status"] == "verified_before_formal_source_open"
    assert set(receipt["asset_bindings"]) == set(freeze.ASSET_NAMES)
    for name, root in assets.items():
        assert receipt["asset_bindings"][name] == formal_runtime.tree_receipt(root)
    assert receipt["systemd_network_preflight"]["network_properties"] == [
        "IPAddressDeny=any",
        "RestrictAddressFamilies=AF_UNIX",
    ]
    assert receipt["formal_source_opened"] is False
    assert receipt["filesystem_isolation"] == canary.FILESYSTEM_ISOLATION
    assert receipt["api_environment_variables_exposed_to_workers"] == []
    assert output.read_bytes() == _canonical(receipt)


def _committed_freeze_fixture(
    root: Path, monkeypatch
) -> tuple[str, Path, Path, Path]:
    _init_repo(root)
    (root / "impl.py").write_text("FROZEN = True\n", encoding="utf-8")
    monkeypatch.setattr(
        acquisition, "REQUIRED_IMPLEMENTATION_PATHS", frozenset({"impl.py"})
    )
    monkeypatch.setattr(freeze, "REQUIRED_BINDING_PATHS", frozenset({"impl.py"}))
    monkeypatch.setattr(
        acquisition,
        "validate_production_canary_capability_receipts",
        lambda _value: None,
    )
    _git(root, "add", "impl.py")
    _git(root, "commit", "-qm", "runtime implementation")
    qualification_commit = _git(root, "rev-parse", "HEAD")
    fingerprint_body = {
        "schema": freeze.RUNTIME_FINGERPRINT_SCHEMA,
        "status": "verified_before_formal_source_open",
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        "source_custody_self_sha256": acquisition.CUSTODY_SELF_SHA256,
        "runtime_implementation_commit": qualification_commit,
        "formal_source_opened": False,
        "source_identifiers_answers_families_mappings_or_labels_present": False,
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
    }
    fingerprint = _self_hashed(fingerprint_body)
    fingerprint_path = root / "receipts/runtime.json"
    _write(fingerprint_path, fingerprint)
    canary_body = {
        "schema": canary.SCHEMA,
        "status": "qualified_before_formal_source_open",
        "qualified": True,
        "runtime_fingerprint_self_sha256": fingerprint["self_sha256"],
        "formal_source_opened": False,
        "hippo_canary_ran": True,
        "P1_retains_ordered_P0_top3": True,
        "P1_outside_P0_unit_count": 1,
        "typed_plan_worker_receipt_source": "capability_receipt_snapshot",
        "minilm_worker_receipt_source": "capability_receipt_snapshot",
        "hippo_worker_receipt_source": "capability_receipt_snapshot",
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
    }
    canary_receipt = _self_hashed(canary_body)
    canary_path = root / "receipts/canary.json"
    _write(canary_path, canary_receipt)
    terminal_body = {
        "schema": "tatqa_p18_runtime_qualification_v1_terminal_success_v1",
        "status": "qualified_before_formal_source_open",
        "runtime_fingerprint_self_sha256": fingerprint["self_sha256"],
        "production_canary_self_sha256": canary_receipt["self_sha256"],
        "formal_source_opened": False,
        "retry_requalification": 0,
    }
    terminal_path = root / "receipts/qualification-terminal.json"
    _write(terminal_path, _self_hashed(terminal_body))
    _git(root, "add", "receipts")
    _git(root, "commit", "-qm", "freeze evidence")
    return (
        _git(root, "rev-parse", "HEAD"),
        fingerprint_path,
        canary_path,
        terminal_path,
    )


def test_freeze_binds_exact_registry_and_committed_clean_receipts(
    tmp_path: Path, monkeypatch
) -> None:
    commit, _fingerprint, _canary, _terminal = _committed_freeze_fixture(
        tmp_path, monkeypatch
    )
    receipt = freeze.build_implementation_freeze(
        project_root=tmp_path,
        formal_implementation_commit=commit,
        runtime_fingerprint_relative="receipts/runtime.json",
        production_canary_relative="receipts/canary.json",
        runtime_qualification_terminal_relative=(
            "receipts/qualification-terminal.json"
        ),
        output_relative="manifests/freeze.json",
    )
    assert receipt["formal_implementation_commit"] == commit
    assert receipt["implementation_binding_registry_is_exact"] is True
    assert [row["relative_path"] for row in receipt["implementation_bindings"]] == [
        "impl.py"
    ]
    assert receipt["runtime_and_canary_committed_and_clean"] is True
    assert receipt["runtime_qualification_terminal_committed_and_clean"] is True
    assert receipt["implementation_bytes_unchanged_since_runtime_qualification"] is True
    assert receipt["formal_source_opened"] is False
    assert (tmp_path / "manifests/freeze.json").read_bytes() == _canonical(receipt)


def test_freeze_refuses_dirty_receipt_and_any_formal_source_state(
    tmp_path: Path, monkeypatch
) -> None:
    commit, fingerprint, _canary, _terminal = _committed_freeze_fixture(
        tmp_path, monkeypatch
    )
    fingerprint.write_bytes(fingerprint.read_bytes() + b" ")
    with pytest.raises(freeze.TatqaP18ImplementationFreezeError, match="differs"):
        freeze.build_implementation_freeze(
            project_root=tmp_path,
            formal_implementation_commit=commit,
            runtime_fingerprint_relative="receipts/runtime.json",
            production_canary_relative="receipts/canary.json",
            runtime_qualification_terminal_relative=(
                "receipts/qualification-terminal.json"
            ),
            output_relative="manifests/never.json",
        )
    _git(tmp_path, "checkout", "--", "receipts/runtime.json")
    source_root = tmp_path / acquisition.SOURCE_ROOT_RELATIVE
    source_root.mkdir(parents=True)
    with pytest.raises(freeze.TatqaP18ImplementationFreezeError, match="source"):
        freeze.build_implementation_freeze(
            project_root=tmp_path,
            formal_implementation_commit=commit,
            runtime_fingerprint_relative="receipts/runtime.json",
            production_canary_relative="receipts/canary.json",
            runtime_qualification_terminal_relative=(
                "receipts/qualification-terminal.json"
            ),
            output_relative="manifests/never.json",
        )


def test_freeze_refuses_production_byte_change_after_runtime_qualification(
    tmp_path: Path, monkeypatch
) -> None:
    _commit, _fingerprint, _canary, _terminal = _committed_freeze_fixture(
        tmp_path, monkeypatch
    )
    (tmp_path / "impl.py").write_text("FROZEN = False\n", encoding="utf-8")
    _git(tmp_path, "add", "impl.py")
    _git(tmp_path, "commit", "-qm", "forbidden production drift")
    formal_commit = _git(tmp_path, "rev-parse", "HEAD")

    with pytest.raises(
        freeze.TatqaP18ImplementationFreezeError,
        match="changed after runtime qualification",
    ):
        freeze.build_implementation_freeze(
            project_root=tmp_path,
            formal_implementation_commit=formal_commit,
            runtime_fingerprint_relative="receipts/runtime.json",
            production_canary_relative="receipts/canary.json",
            runtime_qualification_terminal_relative=(
                "receipts/qualification-terminal.json"
            ),
            output_relative="manifests/never.json",
        )


def test_acquisition_recomputes_c1_bytes_instead_of_trusting_freeze_boolean(
    tmp_path: Path, monkeypatch
) -> None:
    evidence_commit, _fingerprint, _canary, _terminal = _committed_freeze_fixture(
        tmp_path, monkeypatch
    )
    forged = freeze.build_implementation_freeze(
        project_root=tmp_path,
        formal_implementation_commit=evidence_commit,
        runtime_fingerprint_relative="receipts/runtime.json",
        production_canary_relative="receipts/canary.json",
        runtime_qualification_terminal_relative=(
            "receipts/qualification-terminal.json"
        ),
    )
    (tmp_path / "impl.py").write_text("FROZEN = False\n", encoding="utf-8")
    _git(tmp_path, "add", "impl.py")
    _git(tmp_path, "commit", "-qm", "post-qualification drift")
    formal_commit = _git(tmp_path, "rev-parse", "HEAD")
    forged["formal_implementation_commit"] = formal_commit
    forged["implementation_bindings"] = [
        {
            "relative_path": "impl.py",
            "sha256": hashlib.sha256(
                (tmp_path / "impl.py").read_bytes()
            ).hexdigest(),
        }
    ]
    forged["formal_implementation_tree_sha256"] = freeze._stable_hash(
        forged["implementation_bindings"]
    )
    body = dict(forged)
    del body["self_sha256"]
    forged["self_sha256"] = freeze._stable_hash(body)
    freeze_path = tmp_path / acquisition.IMPLEMENTATION_FREEZE_RELATIVE
    freeze_path.write_bytes(_canonical(forged))
    _git(tmp_path, "add", acquisition.IMPLEMENTATION_FREEZE_RELATIVE.as_posix())
    _git(tmp_path, "commit", "-qm", "forged freeze boolean")

    with pytest.raises(
        acquisition.TatqaP18AcquisitionError,
        match="changed after runtime qualification",
    ):
        acquisition._verify_freeze(tmp_path)


def test_runtime_fingerprint_rejects_inventory_secrets_and_network_drift(
    tmp_path: Path,
) -> None:
    assets = {}
    for name in freeze.ASSET_NAMES:
        root = tmp_path / name
        root.mkdir()
        (root / "x").write_text(name)
        assets[name] = root
    with pytest.raises(freeze.TatqaP18ImplementationFreezeError, match="forbidden"):
        freeze.build_runtime_fingerprint(
            output_path=tmp_path / "never.json",
            asset_roots=assets,
            runtime_inventory={"RUOLI_API_KEY": "must-not-be-recorded"},
            systemd_network_preflight=_network_preflight(),
        )
    bad = _network_preflight()
    bad["network_properties"] = ["IPAddressDeny=any"]
    with pytest.raises(freeze.TatqaP18ImplementationFreezeError, match="preflight"):
        freeze.build_runtime_fingerprint(
            output_path=tmp_path / "never.json",
            asset_roots=assets,
            runtime_inventory={"python": "3.11"},
            systemd_network_preflight=bad,
        )
