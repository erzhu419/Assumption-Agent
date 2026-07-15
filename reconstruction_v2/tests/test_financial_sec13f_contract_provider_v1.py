from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent.events import Event
from assumption_agent.models import stable_hash
from assumption_agent.secure_env import LOCKED_MODEL
from replication_runtime.financial_sec13f_contract_v2 import provider
from replication_runtime.financial_sec13f_contract_v2 import runner
from replication_runtime.financial_sec13f_contract_v2.treatment import (
    load_fixed_contract_candidate_v2,
)
from replication_runtime.financial_semantic_v2.pack import payload_hash


PROJECT = Path(__file__).resolve().parents[1]


def _hash(label: str) -> str:
    return stable_hash({"provider-test": label})


def _write_env(path: Path, *, key: str = "fixture-key-one") -> None:
    path.write_text(
        "\n".join(
            (
                "ASSUMPTION_V2_API_BASE=https://ruoli.dev/v1",
                f"ASSUMPTION_V2_API_KEY={key}",
                f"ASSUMPTION_V2_MODEL={LOCKED_MODEL}",
                "ASSUMPTION_V2_PROVIDER_CHAIN=openai_compatible",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _canary_payload() -> dict[str, Any]:
    return {
        "canary_version": provider.PROPOSAL_CANARY_VERSION,
        "model": LOCKED_MODEL,
        "provider_chain": ["openai_compatible"],
        "provider_chain_hash": stable_hash(
            {
                "providers": ["openai_compatible"],
                "model": LOCKED_MODEL,
            }
        ),
        "root_hypothesis_id": "fixture-root",
        "root_hypothesis_hash": _hash("root"),
        "recursive_node_count": 1,
        "recursive_depth": 0,
        "accepted": False,
        "accepted_program": None,
        "nodes": [
            {
                "hypothesis_id": "fixture-root",
                "hypothesis_hash": _hash("root"),
                "depth": 0,
                "passed": False,
                "checks": [],
                "child_id": None,
            }
        ],
        "api_key_present": True,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }


def _write_canary_files(report: Path, events: Path) -> None:
    report.write_text(
        json.dumps(_canary_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        Event(
            event="model_provider_selected",
            stage="model.provider_chain",
            trace_id="provider-test",
            payload={
                "provider": "openai_compatible",
                "model": LOCKED_MODEL,
                "request_hash": _hash("request"),
                "response_hash": _hash("response"),
            },
        ).to_dict()
    ]
    events.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _selection_receipt(
    sidecar: dict[str, Any], *, provider_label: str
) -> dict[str, Any]:
    selected = {
        "probe_kind": "complete_model_response_canary",
        "provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "canary_file_sha256": sidecar["canary_report_file_sha256"],
        "canary_payload_hash": sidecar["canary_payload_hash"],
        "model_response_received": True,
        "semantic_acceptance_used_for_provider_selection": False,
        "canary_semantic_accepted": False,
        "raw_canary_content_persisted": False,
    }
    if provider_label == "plus":
        order = ["plus_complete_model_response"]
        plus_probe = selected
        pro_response = None
    else:
        order = [
            "plus_pre_task_provider_unavailability",
            "pro_complete_model_response",
        ]
        plus_probe = {
            "probe_kind": "pre_task_provider_unavailability",
            "receipt_file_sha256": _hash("plus-failure-file"),
            "receipt_hash": _hash("plus-failure-receipt"),
            "failure_summary": {
                "transport_unavailable": True,
                "model_response_received": False,
            },
            "model_response_received": False,
            "raw_failure_content_persisted": False,
        }
        pro_response = selected
    body = {
        "selection_policy": provider.PROVIDER_SELECTION_POLICY,
        "selected_provider_label": provider_label,
        "selected_provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "probe_order": order,
        "plus_probe_receipt": plus_probe,
        "pro_model_response_receipt": pro_response,
        "selected_model_response_receipt": selected,
        "plus_semantic_acceptance_used_for_selection": False,
        "selection_completed_before_crossfit_task_calls": True,
        "crossfit_task_calls_before_selection": 0,
        "crossfit_model_calls_before_selection": 0,
        "selected_provider_fixed_for_complete_three_cell_batch": True,
        "mid_batch_provider_switch_authorized": False,
        "mid_batch_retry_authorized": False,
        "valid_failure_retry_authorized": False,
        "resampling_authorized": False,
        "secret_value_persisted": False,
        "raw_canary_content_persisted": False,
    }
    return {**body, "receipt_hash": stable_hash(body)}


def _provider_artifacts(
    root: Path, *, provider_label: str = "plus", key: str = "fixture-key-one"
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    env = root / f"{provider_label}.env"
    report = root / f"{provider_label}.canary.json"
    events = root / f"{provider_label}.events.jsonl"
    sidecar_path = root / f"{provider_label}.identity.json"
    selection_path = root / f"{provider_label}.selection.json"
    _write_env(env, key=key)
    _write_canary_files(report, events)
    sidecar = provider.build_provider_identity_sidecar_v1(
        provider_label=provider_label,
        canary_report_path=report,
        event_ledger_path=events,
        env_file=env,
    )
    provider.write_provider_identity_sidecar_v1(sidecar_path, sidecar)
    selection_path.write_text(
        json.dumps(
            _selection_receipt(sidecar, provider_label=provider_label),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return env, report, events, sidecar_path, {
        "sidecar": sidecar,
        "selection_path": selection_path,
    }


def test_sidecar_binds_hmac_key_without_persisting_secret(
    tmp_path: Path,
) -> None:
    secret = "fixture-key-value-that-must-not-be-persisted"
    env, report, events, _sidecar_path, values = _provider_artifacts(
        tmp_path,
        key=secret,
    )
    sidecar = values["sidecar"]
    serialized = json.dumps(sidecar, sort_keys=True)

    assert secret not in serialized
    assert "ASSUMPTION_V2_API_KEY" not in serialized
    assert sidecar["api_key_hmac_sha256"] == (
        provider.api_key_hmac_commitment_v1(secret)
    )
    assert (
        provider.validate_provider_identity_sidecar_v1(
            sidecar,
            canary_report_path=report,
            event_ledger_path=events,
            env_file=env,
            expected_provider_label="plus",
        )
        == sidecar["sidecar_hash"]
    )

    wrong_env = tmp_path / "wrong.env"
    _write_env(wrong_env, key="different-key")
    with pytest.raises(
        provider.ProviderIdentityError,
        match="current provider env differs",
    ):
        provider.validate_provider_identity_sidecar_v1(
            sidecar,
            canary_report_path=report,
            event_ledger_path=events,
            env_file=wrong_env,
        )


def test_controlled_canary_scrubs_aliases_and_launches_python_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = tmp_path / "plus.env"
    report = tmp_path / "plus.canary.json"
    events = tmp_path / "plus.events.jsonl"
    sidecar = tmp_path / "plus.identity.json"
    _write_env(env)
    for name in provider.PROVIDER_ENVIRONMENT_KEYS:
        monkeypatch.setenv(name, "ambient-value")
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        _write_canary_files(report, events)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(provider.subprocess, "run", fake_run)
    receipt = provider.run_controlled_provider_canary_v1(
        project_root=tmp_path,
        provider_label="plus",
        env_file=env,
        canary_report_path=report,
        event_ledger_path=events,
        sidecar_path=sidecar,
    )

    command = observed["command"]
    child_environment = observed["environment"]
    assert command[1:4] == ["-B", "-m", "assumption_agent.proposal_canary"]
    assert command.count("--env-file") == 1
    assert not (set(child_environment) & provider.PROVIDER_ENVIRONMENT_KEYS)
    assert child_environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert receipt["single_env_file_loaded"] is True
    assert sidecar.is_file()


def test_load_provider_environment_overrides_and_removes_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = tmp_path / "legacy.env"
    env.write_text(
        "RUOLI_BASE_URL=https://ruoli.dev/v1\n"
        "RUOLI_GPT_KEY=selected-key\n"
        f"ASSUMPTION_V2_MODEL={LOCKED_MODEL}\n",
        encoding="utf-8",
    )
    for name in provider.PROVIDER_ENVIRONMENT_KEYS:
        monkeypatch.setenv(name, "ambient-wrong-value")

    identity = provider.load_provider_environment_v1(env)

    assert identity["dotenv_override"] is True
    assert identity["api_key_hmac_sha256"] == (
        provider.api_key_hmac_commitment_v1("selected-key")
    )
    assert os.environ["ASSUMPTION_V2_API_KEY"] == "selected-key"
    assert os.environ["ASSUMPTION_V2_API_BASE"] == "https://ruoli.dev/v1"
    for name in (
        "RUOLI_GPT_KEY",
        "GPT5_API_KEY",
        "OPENAI_API_KEY",
        "RUOLI_BASE_URL",
        "GPT5_BASE_URL",
        "OPENAI_BASE_URL",
    ):
        assert name not in os.environ


@pytest.mark.parametrize("provider_label", ["plus", "pro"])
def test_execution_binding_supports_plus_and_authorized_pro(
    tmp_path: Path,
    provider_label: str,
) -> None:
    env, report, events, sidecar_path, values = _provider_artifacts(
        tmp_path,
        provider_label=provider_label,
    )
    selection = values["selection_path"]
    binding = provider.build_execution_provider_binding_v1(
        project_root=tmp_path,
        provider_label=provider_label,
        identity_sidecar_path=sidecar_path,
        selected_canary_report_path=report,
        selected_event_ledger_path=events,
        selection_receipt_path=selection,
        env_file=env,
    )
    verified = provider.validate_execution_provider_binding_v1(
        binding,
        project_root=tmp_path,
        env_file=env,
    )

    assert verified["provider_label"] == provider_label
    assert verified["plus_transport_failure_before_pro_selection"] is (
        provider_label == "pro"
    )
    assert verified["mid_batch_provider_switch_authorized"] is False

    wrong_env = tmp_path / f"{provider_label}.wrong.env"
    _write_env(wrong_env, key="wrong-current-key")
    with pytest.raises(provider.ProviderIdentityError):
        provider.validate_execution_provider_binding_v1(
            binding,
            project_root=tmp_path,
            env_file=wrong_env,
        )


def test_pro_binding_rejects_complete_plus_route(tmp_path: Path) -> None:
    env, report, events, sidecar_path, values = _provider_artifacts(
        tmp_path,
        provider_label="pro",
    )
    sidecar = values["sidecar"]
    selection_path = values["selection_path"]
    invalid = _selection_receipt(sidecar, provider_label="pro")
    selected = invalid["selected_model_response_receipt"]
    invalid["probe_order"] = ["plus_complete_model_response"]
    invalid["plus_probe_receipt"] = selected
    body = dict(invalid)
    body.pop("receipt_hash")
    invalid["receipt_hash"] = stable_hash(body)
    selection_path.write_text(
        json.dumps(invalid, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        provider.ProviderIdentityError,
        match="selection is not authorized",
    ):
        provider.build_execution_provider_binding_v1(
            project_root=tmp_path,
            provider_label="pro",
            identity_sidecar_path=sidecar_path,
            selected_canary_report_path=report,
            selected_event_ledger_path=events,
            selection_receipt_path=selection_path,
            env_file=env,
        )


def test_runner_freeze_validation_accepts_verified_pro_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = load_fixed_contract_candidate_v2(PROJECT)
    verified = {
        "provider_label": "pro",
        "binding_hash": _hash("provider-binding"),
        "identity_sidecar_hash": _hash("sidecar"),
        "selection_receipt_hash": _hash("selection"),
        "model": LOCKED_MODEL,
        "api_origin": "https://ruoli.dev",
        "api_key_commitment_version": provider.API_KEY_COMMITMENT_VERSION,
        "api_key_hmac_sha256": _hash("key-commitment"),
        "plus_transport_failure_before_pro_selection": True,
        "selected_provider_fixed_for_complete_batch": True,
        "mid_batch_provider_switch_authorized": False,
        "mid_batch_retry_authorized": False,
        "secret_value_persisted": False,
    }
    monkeypatch.setattr(
        runner,
        "validate_execution_provider_binding_v1",
        lambda *_args, **_kwargs: verified,
    )
    body = {
        "manifest_version": runner.EXECUTION_FREEZE_VERSION,
        "candidate": candidate.safe_payload(PROJECT),
        "provider": {"fixture": True},
        "execution_source_closure": {"closure_hash": _hash("closure")},
        "materialization": {"benchmark_tree_hash": _hash("tree")},
        "precomputed_plan_set_hash": _hash("plans"),
    }
    freeze = {**body, "manifest_hash": payload_hash(body)}

    freeze_hash, observed = runner._validate_execution_freeze_v2(
        freeze,
        project_root=PROJECT,
        candidate=candidate,
        env_file=tmp_path / "unused.env",
    )

    assert freeze_hash == freeze["manifest_hash"]
    assert observed["provider_label"] == "pro"
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert '"provider_label": provider_label' in source
    assert '"provider_label": "plus"' not in source
