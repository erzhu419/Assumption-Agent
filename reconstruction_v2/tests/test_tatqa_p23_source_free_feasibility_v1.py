from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    tatqa_p22_source_free_feasibility_v1 as base,
)
from assumption_agent.benchmarks import (
    tatqa_p23_source_free_feasibility_v1 as feasibility,
)


def _semantic_hash(value: object) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _exact_post_model_environment() -> dict[str, str]:
    environment = {
        name: f"bound-{name}"
        for name in base.formal_runtime.USER_SYSTEMD_POST_RUNTIME_ENVIRONMENT_VARIABLE_ALLOWLIST
    }
    environment.update(base.formal_runtime.USER_SYSTEMD_ENTRY_SAFE_VALUES)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUDA_MODULE_LOADING": "LAZY",
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
            "HOME": "/home/erzhu419",
            "KMP_DUPLICATE_LIB_OK": "True",
            "KMP_INIT_AT_FORK": "FALSE",
            "XDG_RUNTIME_DIR": "/run/user/1001",
        }
    )
    return environment


def test_exact_post_model_mutations_are_sealed_then_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _exact_post_model_environment()
    monkeypatch.setattr(feasibility, "os", SimpleNamespace(environ=environment))

    receipt = feasibility._normalize_post_minilm_environment()

    assert receipt["schema"] == feasibility.NORMALIZATION_SCHEMA
    assert receipt["status"] == (
        "verified_exact_deterministic_mutations_then_removed"
    )
    assert receipt["accepted_exact_mutations"] == [
        {"added_name": "KMP_DUPLICATE_LIB_OK", "exact_value": "True"},
        {"added_name": "KMP_INIT_AT_FORK", "exact_value": "FALSE"},
    ]
    assert receipt["raw_environment_values_or_credentials_recorded"] is False
    assert receipt["provider_or_api_credentials_read"] is False
    assert receipt["formal_TAT_QA_source_or_rows_accessed"] is False
    body = dict(receipt)
    declared = body.pop("self_sha256")
    assert declared == _semantic_hash(body)
    assert set(environment) == set(
        base.formal_runtime.USER_SYSTEMD_POST_RUNTIME_ENVIRONMENT_VARIABLE_ALLOWLIST
    )
    assert "KMP_DUPLICATE_LIB_OK" not in environment
    assert "KMP_INIT_AT_FORK" not in environment


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("unexpected", "1"),
        ("KMP_DUPLICATE_LIB_OK", "true"),
        ("KMP_INIT_AT_FORK", "TRUE"),
    ),
)
def test_normalization_fails_closed_without_partial_removal(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    value: str,
) -> None:
    environment = _exact_post_model_environment()
    if mutation == "unexpected":
        environment["UNEXPECTED"] = value
    else:
        environment[mutation] = value
    before = dict(environment)
    monkeypatch.setattr(feasibility, "os", SimpleNamespace(environ=environment))

    with pytest.raises(feasibility.TatqaP23SourceFreeFeasibilityError):
        feasibility._normalize_post_minilm_environment()

    assert environment == before


def test_p23_contract_is_exact_and_restored() -> None:
    original = {
        name: getattr(base, name) for name in feasibility._contract_bindings()
    }
    with feasibility._activate_p23_contract():
        assert base.VERSION == feasibility.VERSION
        assert base.EXPECTED_PROJECT_ROOT == feasibility.EXPECTED_PROJECT_ROOT
        assert base.EXPECTED_OUTER_UNIT == feasibility.EXPECTED_OUTER_UNIT
        assert base.REQUIRED_ENTRY_MODULE_NAME == feasibility.ENTRY_MODULE_NAME
        assert (
            base.EXPECTED_NORMALIZER_CALLABLE_NAME
            == feasibility.NORMALIZER_CALLABLE_NAME
        )
        assert base.REQUIRED_SNAPSHOT_PATHS == feasibility.REQUIRED_SNAPSHOT_PATHS
        binding = base._post_minilm_normalizer_binding(
            Path.cwd(), feasibility._normalize_post_minilm_environment
        )
        assert binding is not None
        assert binding["module"] == feasibility.ENTRY_MODULE_NAME
        assert binding["callable"] == feasibility.NORMALIZER_CALLABLE_NAME
    assert {
        name: getattr(base, name) for name in feasibility._contract_bindings()
    } == original


def test_wrapper_injects_only_preregistered_normalizer_and_restores_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    original_version = base.VERSION

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        assert base.VERSION == feasibility.VERSION
        assert base.REQUIRED_ENTRY_MODULE_NAME == feasibility.ENTRY_MODULE_NAME
        assert base.EXPECTED_FEASIBILITY_ROOT == (
            feasibility.EXPECTED_FEASIBILITY_ROOT
        )
        return {"status": "synthetic-pass"}

    monkeypatch.setattr(base, "run_source_free_feasibility", fake_run)
    result = feasibility.run_source_free_feasibility(
        project_root=Path("/fixed/project"),
        typed_runtime_python=Path("/typed/python"),
        hippo_runtime_python=Path("/hippo/python"),
        qwen_model=Path("/qwen"),
        minilm_asset_manifest=Path("/minilm.json"),
        minilm_model=Path("/minilm"),
        hippo_llm_model=Path("/hippo-llm"),
        hippo_embedding_model=Path("/hippo-embedding"),
        hipporag_source=Path("/hipporag"),
        hippo_attestation=Path("/attestation.json"),
        p21_runtime_fingerprint=Path("/fingerprint.json"),
        diagnostic_snapshot_commit="a" * 40,
    )

    assert result == {"status": "synthetic-pass"}
    assert observed["_post_minilm_environment_normalizer"] is (
        feasibility._normalize_post_minilm_environment
    )
    assert base.VERSION == original_version


def test_wrapper_translates_terminal_engine_error_and_restores_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_version = base.VERSION

    def fail(**_kwargs: object) -> dict[str, object]:
        raise base.TatqaP22SourceFreeFeasibilityError("terminal")

    monkeypatch.setattr(base, "run_source_free_feasibility", fail)
    with pytest.raises(
        feasibility.TatqaP23SourceFreeFeasibilityError,
        match="failed terminally",
    ):
        feasibility.run_source_free_feasibility(
            project_root="/project",
            typed_runtime_python="/typed",
            hippo_runtime_python="/hippo",
            qwen_model="/qwen",
            minilm_asset_manifest="/manifest",
            minilm_model="/minilm",
            hippo_llm_model="/llm",
            hippo_embedding_model="/embedding",
            hipporag_source="/source",
            hippo_attestation="/attestation",
            p21_runtime_fingerprint="/fingerprint",
            diagnostic_snapshot_commit="b" * 40,
        )
    assert base.VERSION == original_version


def test_snapshot_and_cli_have_closed_source_free_surface() -> None:
    assert feasibility.ENTRY_RELATIVE_PATH in (
        feasibility.REQUIRED_IMPLEMENTATION_PATHS
    )
    assert feasibility.TEST_RELATIVE_PATH in (
        feasibility.REQUIRED_IMPLEMENTATION_PATHS
    )
    assert len(feasibility.SOURCE_ISOLATION_SENTINEL_PATHS) == 6
    assert feasibility.REQUIRED_SNAPSHOT_PATHS == (
        feasibility.REQUIRED_IMPLEMENTATION_PATHS
        | feasibility.REQUIRED_EVIDENCE_PATHS
        | feasibility.SOURCE_ISOLATION_SENTINEL_PATHS
    )
    options = {
        action.dest
        for action in feasibility._parser()._actions
        if action.dest != "help"
    }
    assert options == {
        "project_root",
        "typed_runtime_python",
        "hippo_runtime_python",
        "qwen_model",
        "minilm_asset_manifest",
        "minilm_model",
        "hippo_llm_model",
        "hippo_embedding_model",
        "hipporag_source",
        "hippo_attestation",
        "p21_runtime_fingerprint",
        "diagnostic_snapshot_commit",
    }
    assert not any(
        "source" in option and option != "hipporag_source" for option in options
    )
