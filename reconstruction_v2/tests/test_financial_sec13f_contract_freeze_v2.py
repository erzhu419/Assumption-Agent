from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    TUNA_PYPI_INDEX_URL,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
    offline_verifier_volume_name,
)
from assumption_agent.models import stable_hash
from replication_runtime.financial_sec13f_contract_v2 import freeze
from replication_runtime.financial_sec13f_contract_v2.hygienic_materialize import (
    FAMILY,
    MATERIALIZATION_REPORT_NAME,
    MATERIALIZATION_VERSION,
    TREE_RECEIPT_VERSION,
    measurement_benchmark_tree_receipt_v2,
)
from replication_runtime.financial_sec13f_contract_v2.hygienic_prewarm import (
    OFFLINE_VERIFIER_PROFILE_ID,
    OFFLINE_VERIFIER_REQUIREMENTS,
    PREWARM_VERSION,
)
from replication_runtime.financial_sec13f_contract_v2.treatment import (
    CANDIDATE_IDENTITY_VERSION,
    FixedContractCandidateV2,
)
from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    sha256_file,
)


def _hash(label: str) -> str:
    return stable_hash({"financial-sec13f-freeze-test": label})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_git(project: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _candidate(project: Path) -> FixedContractCandidateV2:
    source = project / "candidates/financial_sec13f_contract_operator_v2"
    asset = project / "manifests/candidate-asset.json"
    asset_file_sha256 = sha256_file(asset)
    candidate_id = _hash("candidate-id")
    asset_manifest_hash = _hash("asset-manifest")
    operator_source_sha256 = _hash("operator-source")
    external_source_hash = _hash("external-source")
    recipe_id = stable_hash(
        {
            "identity_version": CANDIDATE_IDENTITY_VERSION,
            "candidate_id": candidate_id,
            "asset_manifest_hash": asset_manifest_hash,
            "asset_file_sha256": asset_file_sha256,
            "operator_source_sha256": operator_source_sha256,
            "external_skill_source_receipt_hash": external_source_hash,
        }
    )
    program_set_hash = stable_hash({"recipe_ids": [recipe_id]})
    base_treatment_id = stable_hash(
        {
            "identity_version": CANDIDATE_IDENTITY_VERSION,
            "recipe_id": recipe_id,
            "program_set_hash": program_set_hash,
            "operator_is_candidate_content": True,
            "post_agent_pre_verifier_treatment": True,
        }
    )
    return FixedContractCandidateV2(
        candidate_id=candidate_id,
        asset_manifest_hash=asset_manifest_hash,
        asset_file_sha256=asset_file_sha256,
        operator_source_sha256=operator_source_sha256,
        external_skill_source_receipt_hash=external_source_hash,
        recipe_id=recipe_id,
        program_set_hash=program_set_hash,
        base_treatment_id=base_treatment_id,
        candidate_skill_source=source,
        operator_asset_path=asset,
    )


def _measurement_view() -> dict[str, Any]:
    items = []
    for index in range(8):
        instruction = f"reconcile SEC 13F fixture {index}"
        items.append(
            {
                "item_id": f"financial-contract-{index}",
                "fold": index % 4,
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
            }
        )
    return {
        "measurement_view_hash": _hash("measurement-view"),
        "private_pack_hash": _hash("private-pack"),
        "measurement_item_count": 8,
        "sealed_item_count": 4,
        "measurement_items": items,
        "sources": {
            "previous": {"source_fingerprint": _hash("previous-source")},
            "current": {"source_fingerprint": _hash("current-source")},
        },
    }


class _FakePlanner:
    planner_hash = _hash("shared-planner")

    def __init__(self, *, asset_path: Path) -> None:
        self.asset_path = asset_path

    def build(
        self, instruction: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        instruction_sha256 = hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest()
        plan_hash = stable_hash(
            {"typed-plan-for-instruction": instruction_sha256}
        )
        receipt_hash = stable_hash(
            {"typed-extraction-for-plan": plan_hash}
        )
        return (
            {
                "instruction_sha256": instruction_sha256,
                "plan_hash": plan_hash,
            },
            {"plan_hash": plan_hash, "receipt_hash": receipt_hash},
        )


@dataclass(frozen=True)
class _FrozenFixture:
    project: Path
    value: dict[str, Any]
    candidate: FixedContractCandidateV2
    view_path: Path
    expected_output: Path


@pytest.fixture()
def frozen_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _FrozenFixture:
    project = tmp_path / "project"
    for relative, content in (
        ("assumption_agent/runtime.py", "# runtime closure fixture\n"),
        ("replication_runtime/runtime.py", "# runtime closure fixture\n"),
        (
            "candidates/financial_sec13f_contract_operator_v2/SKILL.md",
            "# fixed candidate fixture\n",
        ),
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _write_json(project / "manifests/candidate-asset.json", {"asset": "fixed"})
    prereg_path = project / "evidence/preregistration.json"
    acquisition_path = project / "evidence/acquisition.json"
    formation_path = project / "evidence/formation.json"
    view_path = project / "evidence/measurement-view.json"
    _write_json(
        prereg_path,
        {
            "prior_commitment_view": {
                "measurement_view_hash": _hash("prior-view")
            }
        },
    )
    _write_json(
        acquisition_path,
        {"archive_set_hash": _hash("archive-set")},
    )
    _write_json(formation_path, {"formation": "safe-receipt"})
    _write_json(view_path, {"measurement": "redacted-view"})

    _run_git(project, "init", "-q")
    _run_git(project, "config", "user.email", "freeze@example.invalid")
    _run_git(project, "config", "user.name", "Freeze Test")
    _run_git(project, "add", ".")
    _run_git(project, "commit", "-q", "-m", "committed freeze inputs")

    benchmark = project / "benchmark"
    expected_output = benchmark / "tasks/expected_output.json"
    expected_output.parent.mkdir(parents=True)
    expected_output.write_text(
        '{"answer":"EXPECTED-CONTENT-MUST-NOT-ENTER-FREEZE"}\n',
        encoding="utf-8",
    )
    materialization_path = benchmark / MATERIALIZATION_REPORT_NAME
    _write_json(materialization_path, {"report": "excluded from tree hash"})
    tree = measurement_benchmark_tree_receipt_v2(benchmark)
    view = _measurement_view()
    materialization = {
        "materialization_version": MATERIALIZATION_VERSION,
        "tree_receipt_version": TREE_RECEIPT_VERSION,
        "measurement_view_hash": view["measurement_view_hash"],
        "private_pack_hash": view["private_pack_hash"],
        "benchmark_tree_hash": tree["tree_hash"],
        "materialization_hash": _hash("materialization"),
        "items": [
            {"item_id": item["item_id"]}
            for item in view["measurement_items"]
        ],
        "period_source_receipts": {
            role: {
                "source_fingerprint": view["sources"][role][
                    "source_fingerprint"
                ]
            }
            for role in ("previous", "current")
        },
    }

    prewarm_path = project / "prewarm/measurement.prewarm.json"
    sidecar_path = project / "prewarm/offline-verifier.preparation.json"
    _write_json(prewarm_path, {"prewarm": "safe"})
    _write_json(sidecar_path, {"preparation": "safe"})

    protocol_path = project / freeze.V320_PROTOCOL_RELATIVE_PATH
    _write_json(protocol_path, {"protocol": "fixed"})
    provider_paths = {
        "identity": project / "provider/identity.json",
        "canary": project / "provider/canary.json",
        "events": project / "provider/events.json",
        "selection": project / "provider/selection.json",
    }
    for label, path in provider_paths.items():
        _write_json(path, {"provider-evidence": label})

    candidate = _candidate(project)
    monkeypatch.setattr(freeze, "verify_measurement_view", lambda _: view)
    monkeypatch.setattr(
        freeze,
        "validate_preregistration_v1",
        lambda _: _hash("preregistration"),
    )
    monkeypatch.setattr(
        freeze,
        "validate_acquisition_receipt_v1",
        lambda _value, *, preregistration: _hash("acquisition"),
    )
    monkeypatch.setattr(
        freeze,
        "_validate_safe_formation_receipt",
        lambda *_args, **_kwargs: _hash("formation"),
    )
    monkeypatch.setattr(freeze, "_load_materialization", lambda _: materialization)
    monkeypatch.setattr(freeze, "_validate_item_receipts", lambda **_: None)
    monkeypatch.setattr(freeze, "_validate_period_sources", lambda **_: None)

    def fake_prewarm(
        *,
        prewarm_path: Path,
        measurement_view: Mapping[str, Any],
        materialization: Mapping[str, Any],
        benchmark_tree_hash: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert measurement_view["measurement_view_hash"] == view[
            "measurement_view_hash"
        ]
        assert benchmark_tree_hash == materialization["benchmark_tree_hash"]
        return {}, {
            "relative_path": "",
            "file_sha256": sha256_file(prewarm_path),
            "prewarm_hash": _hash("prewarm"),
            "preparation_sidecar": {
                "relative_path": sidecar_path.name,
                "file_sha256": sha256_file(sidecar_path),
                "receipt_hash": _hash("preparation"),
            },
            "formal_execution_cache_only": True,
            "formal_verifier_network": "none",
        }

    monkeypatch.setattr(freeze, "_validate_prewarm", fake_prewarm)

    protocol = SimpleNamespace(
        id="fixture-protocol-v1",
        protocol_hash=_hash("protocol"),
        payload={"agent_id": "codex", "model": "gpt-5.4", "max_steps": 100},
        codex_agent_execution_policy=SimpleNamespace(
            policy_hash=_hash("execution-policy")
        ),
    )
    monkeypatch.setattr(
        freeze,
        "PaperProtocol",
        SimpleNamespace(read=lambda _path: protocol),
    )
    monkeypatch.setattr(
        freeze, "load_fixed_contract_candidate_v2", lambda _project: candidate
    )
    monkeypatch.setattr(
        freeze, "SharedFinancialSec13FContractPlannerV2", _FakePlanner
    )

    def provider_file(root: Path, supplied: str | Path) -> Path:
        path = Path(supplied)
        if not path.is_absolute():
            path = root / path
        return path.resolve(strict=True)

    def build_provider(
        *,
        project_root: str | Path,
        provider_label: str,
        identity_sidecar_path: str | Path,
        selected_canary_report_path: str | Path,
        selected_event_ledger_path: str | Path,
        selection_receipt_path: str | Path,
        env_file: str | Path | None = None,
    ) -> dict[str, Any]:
        del env_file
        root = Path(project_root).resolve(strict=True)
        files = {
            "identity": provider_file(root, identity_sidecar_path),
            "canary": provider_file(root, selected_canary_report_path),
            "events": provider_file(root, selected_event_ledger_path),
            "selection": provider_file(root, selection_receipt_path),
        }
        body = {
            "binding_version": "fixture-provider-binding-v1",
            "provider_label": provider_label,
            "provider_label_hash": stable_hash(
                {"provider_label": provider_label}
            ),
            "fallback_policy": "pro_only_after_complete_plus_unavailability",
            "identity_sidecar_relative_path": files["identity"].relative_to(
                root
            ).as_posix(),
            "identity_sidecar_file_sha256": sha256_file(files["identity"]),
            "identity_sidecar_hash": _hash("identity-sidecar"),
            "selected_canary_relative_path": files["canary"].relative_to(
                root
            ).as_posix(),
            "selected_canary_file_sha256": sha256_file(files["canary"]),
            "selected_event_ledger_relative_path": files["events"].relative_to(
                root
            ).as_posix(),
            "selected_event_ledger_file_sha256": sha256_file(files["events"]),
            "selection_receipt_relative_path": files["selection"].relative_to(
                root
            ).as_posix(),
            "selection_receipt_file_sha256": sha256_file(files["selection"]),
            "selection_receipt_hash": _hash("selection-receipt"),
            "model": "gpt-5.4",
            "api_origin": "https://ruoli.dev",
            "api_key_commitment_version": "fixture-key-hmac-v1",
            "api_key_hmac_sha256": _hash("provider-key-commitment"),
            "plus_transport_failure_before_pro_selection": (
                provider_label == "pro"
            ),
            "selected_provider_fixed_for_complete_batch": True,
            "mid_batch_provider_switch_authorized": False,
            "mid_batch_retry_authorized": False,
            "secret_value_persisted": False,
        }
        return {**body, "binding_hash": stable_hash(body)}

    def validate_provider(
        value: Mapping[str, Any],
        *,
        project_root: str | Path,
        env_file: str | Path | None = None,
    ) -> dict[str, Any]:
        expected = build_provider(
            project_root=project_root,
            provider_label=str(value.get("provider_label") or ""),
            identity_sidecar_path=str(
                value.get("identity_sidecar_relative_path") or ""
            ),
            selected_canary_report_path=str(
                value.get("selected_canary_relative_path") or ""
            ),
            selected_event_ledger_path=str(
                value.get("selected_event_ledger_relative_path") or ""
            ),
            selection_receipt_path=str(
                value.get("selection_receipt_relative_path") or ""
            ),
            env_file=env_file,
        )
        if dict(value) != expected:
            raise ValueError("provider drift")
        return expected

    monkeypatch.setattr(
        freeze, "build_execution_provider_binding_v1", build_provider
    )
    monkeypatch.setattr(
        freeze, "validate_execution_provider_binding_v1", validate_provider
    )

    value = freeze.build_execution_freeze_v2(
        project_root=project,
        preregistration_path="evidence/preregistration.json",
        acquisition_receipt_path="evidence/acquisition.json",
        formation_receipt_path="evidence/formation.json",
        measurement_view_path="evidence/measurement-view.json",
        benchmark_root="benchmark",
        materialization_report_path=(
            "benchmark/" + MATERIALIZATION_REPORT_NAME
        ),
        prewarm_path="prewarm/measurement.prewarm.json",
        paper_protocol_path=freeze.V320_PROTOCOL_RELATIVE_PATH,
        provider_identity_sidecar_path="provider/identity.json",
        provider_selection_path="provider/selection.json",
        selected_canary_path="provider/canary.json",
        selected_event_ledger_path="provider/events.json",
        provider_label="plus",
    )
    return _FrozenFixture(
        project=project,
        value=value,
        candidate=candidate,
        view_path=view_path,
        expected_output=expected_output,
    )


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("manifest_hash", None)
    value["manifest_hash"] = payload_hash(body)
    return value


def test_build_validate_and_load_bind_exact_safe_execution(
    frozen_fixture: _FrozenFixture,
) -> None:
    fixture = frozen_fixture
    observed = freeze.validate_execution_freeze_v2(
        fixture.value, project_root=fixture.project
    )
    assert observed == fixture.candidate
    safe_plan = fixture.value["plan"]["safe_payload"]
    assert safe_plan["physical_work_unit_count"] == 16
    assert safe_plan["raw_execution_count"] == 8
    assert safe_plan["candidate_execution_count"] == 8
    assert safe_plan["maximum_workers"] == 16
    assert safe_plan["retry_count"] == 0
    assert fixture.value["execution"]["outer_workers"] == 16
    assert fixture.value["execution"]["offline_evaluation_only"] is True
    assert fixture.value["provider"]["provider_label"] == "plus"
    assert len(fixture.value["provider"]["api_key_hmac_sha256"]) == 64
    assert fixture.value["typed_plan_set"]["item_count"] == 8
    serialized = json.dumps(fixture.value, sort_keys=True)
    assert "EXPECTED-CONTENT-MUST-NOT-ENTER-FREEZE" not in serialized
    assert '"api_key"' not in serialized

    freeze_path = fixture.project / "execution-freeze.json"
    _write_json(freeze_path, fixture.value)
    loaded, candidate = freeze.load_execution_freeze_v2(
        freeze_path, project_root=fixture.project
    )
    assert loaded == fixture.value
    assert candidate == fixture.candidate


def test_validator_rejects_runtime_extra_and_symlink(
    frozen_fixture: _FrozenFixture,
) -> None:
    fixture = frozen_fixture
    extra = fixture.project / "assumption_agent/unfrozen.py"
    extra.write_text("# unexpected runtime file\n", encoding="utf-8")
    with pytest.raises(freeze.ContractFreezeError):
        freeze.validate_execution_freeze_v2(
            fixture.value, project_root=fixture.project
        )
    extra.unlink()

    link = fixture.project / "replication_runtime/escape-link"
    link.symlink_to(fixture.project / "evidence/preregistration.json")
    with pytest.raises(freeze.ContractFreezeError, match="symbolic link"):
        freeze.validate_execution_freeze_v2(
            fixture.value, project_root=fixture.project
        )


def test_validator_rejects_committed_input_and_benchmark_drift(
    frozen_fixture: _FrozenFixture,
) -> None:
    fixture = frozen_fixture
    fixture.view_path.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(freeze.ContractFreezeError, match="committed binding"):
        freeze.validate_execution_freeze_v2(
            fixture.value, project_root=fixture.project
        )

    _run_git(fixture.project, "restore", "evidence/measurement-view.json")
    fixture.expected_output.write_text(
        '{"answer":"changed verifier bytes"}\n', encoding="utf-8"
    )
    with pytest.raises(freeze.ContractFreezeError, match="materialization"):
        freeze.validate_execution_freeze_v2(
            fixture.value, project_root=fixture.project
        )


@pytest.mark.parametrize("tamper", ["path", "plan", "extra", "secret"])
def test_self_consistent_manifest_tampering_fails_closed(
    frozen_fixture: _FrozenFixture,
    tamper: str,
) -> None:
    fixture = frozen_fixture
    value = copy.deepcopy(fixture.value)
    if tamper == "path":
        value["materialization"]["relative_path"] = "../escaped.json"
    elif tamper == "plan":
        value["plan"]["safe_payload"]["retry_count"] = 1
    elif tamper == "extra":
        value["unregistered_gate"] = True
    else:
        value["provider"]["api_key"] = "sk-forbidden-secret-value"
    _rehash(value)
    with pytest.raises(freeze.ContractFreezeError):
        freeze.validate_execution_freeze_v2(
            value, project_root=fixture.project
        )


def test_actual_prewarm_validator_accepts_download_only_in_preparation(
    tmp_path: Path,
) -> None:
    profile = offline_verifier_profile_for_family(FAMILY)
    assert profile is not None
    runtime_key = offline_verifier_runtime_key(profile=profile)
    cache_key = _hash("cache-key")
    image_id = "sha256:" + _hash("image-id")
    shared = {
        "cache_key": cache_key,
        "environment_hash": _hash("environment"),
        "source_environment_hash": _hash("source-environment"),
        "image_id": image_id,
        "agent_runtime_key": _hash("agent-runtime"),
        "agent_runtime_version": "codex-runtime-fixture-v1",
    }
    item_ids = [f"financial-contract-{index}" for index in range(8)]
    preparation_rows = [
        {
            "item_id": item_id,
            "item_id_hash": payload_hash({"item_id": item_id}),
            **shared,
            "prepared_before_formal_cache_check": True,
        }
        for item_id in item_ids
    ]
    formal_rows = [
        {
            "item_id": item_id,
            "item_id_hash": payload_hash({"item_id": item_id}),
            **shared,
            "prebuilt_cache_reused": True,
            "offline_verifier_profile_id": OFFLINE_VERIFIER_PROFILE_ID,
            "offline_verifier_profile_hash": profile.profile_hash,
            "offline_verifier_runtime_key": runtime_key,
            "offline_verifier_runtime_reused": True,
            "verifier_runtime_network": "none",
        }
        for item_id in item_ids
    ]
    wheel = {
        "filename": "pytest-8.4.1-py3-none-any.whl",
        "size": 123,
        "sha256": _hash("wheel"),
    }
    receipt_body = {
        "report_version": "offline_verifier_preparation_receipt_v2",
        "policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "profile_id": OFFLINE_VERIFIER_PROFILE_ID,
        "profile_hash": profile.profile_hash,
        "runtime_key": runtime_key,
        "runtime_volume_hash": stable_hash(
            {"volume": offline_verifier_volume_name(runtime_key)}
        ),
        "base_image_tag": "assumption-v2-item:" + cache_key[:24],
        "base_image_id": image_id,
        "python_version": profile.python_version,
        "python_abi": profile.python_abi,
        "platform": profile.platform,
        "semantic_prelude_id": profile.semantic_prelude_id,
        "probe_workspace_mode": profile.probe_workspace_mode,
        "activation_blocker": profile.activation_blocker,
        "package_index_origin": TUNA_PYPI_INDEX_URL,
        "docker_install_network": "none",
        "runtime_reused": False,
        "wheelhouse_reused": False,
        "online_download_attempted": True,
        "wheel_count": 1,
        "wheel_total_bytes": 123,
        "wheels": [wheel],
        "probe_passed": True,
        "raw_content_persisted": False,
    }
    receipt = {
        **receipt_body,
        "receipt_hash": payload_hash(receipt_body),
    }
    sidecar = tmp_path / "offline-verifier.preparation.json"
    _write_json(sidecar, receipt)
    tree_hash = _hash("benchmark-tree")
    materialization_hash = _hash("materialization")
    prewarm_body = {
        "prewarm_version": PREWARM_VERSION,
        "tree_receipt_version": TREE_RECEIPT_VERSION,
        "measurement_view_hash": _hash("view"),
        "materialization_hash": materialization_hash,
        "benchmark_tree_hash": tree_hash,
        "pre_prewarm_tree_hash": tree_hash,
        "post_prewarm_tree_hash": tree_hash,
        "benchmark_tree_unchanged": True,
        "python_dont_write_bytecode": True,
        "python_dont_write_bytecode_env": "1",
        "item_count": 8,
        "preparation_rows": preparation_rows,
        "preparation_row_set_hash": payload_hash(preparation_rows),
        "formal_cache_rows": formal_rows,
        "formal_cache_row_set_hash": payload_hash(formal_rows),
        "unique_image_id_hash": payload_hash({"image_id": image_id}),
        "unique_cache_key_hash": payload_hash({"cache_key": cache_key}),
        "offline_verifier_profile_id": OFFLINE_VERIFIER_PROFILE_ID,
        "offline_verifier_profile_hash": profile.profile_hash,
        "offline_verifier_requirements": list(OFFLINE_VERIFIER_REQUIREMENTS),
        "offline_verifier_requirements_hash": payload_hash(
            list(OFFLINE_VERIFIER_REQUIREMENTS)
        ),
        "offline_verifier_runtime_key": runtime_key,
        "offline_verifier_preparation": {
            "relative_path": sidecar.name,
            "file_sha256": sha256_file(sidecar),
            "receipt_hash": receipt["receipt_hash"],
            "network_allowed_only_during_preparation": True,
            "docker_install_network": "none",
            "probe_passed": True,
        },
        "formal_execution_cache_only": True,
        "formal_image_cache_only": True,
        "formal_offline_verifier_cache_only": True,
        "preparation_network_allowed": True,
        "formal_verifier_network": "none",
        "model_calls": 0,
        "online_judge_calls": 0,
        "sealed_task_count": 0,
        "sealed_content_accessed": False,
        "secret_value_persisted": False,
    }
    prewarm = {**prewarm_body, "prewarm_hash": payload_hash(prewarm_body)}
    prewarm_path = tmp_path / "measurement.prewarm.json"
    _write_json(prewarm_path, prewarm)
    view = {
        "measurement_view_hash": prewarm_body["measurement_view_hash"],
        "measurement_items": [{"item_id": item_id} for item_id in item_ids],
    }
    materialization = {"materialization_hash": materialization_hash}

    observed, binding = freeze._validate_prewarm(
        prewarm_path=prewarm_path,
        measurement_view=view,
        materialization=materialization,
        benchmark_tree_hash=tree_hash,
    )
    assert observed == prewarm
    assert binding["formal_execution_cache_only"] is True
    assert binding["formal_verifier_network"] == "none"
    assert binding["preparation_sidecar"]["receipt_hash"] == receipt[
        "receipt_hash"
    ]
