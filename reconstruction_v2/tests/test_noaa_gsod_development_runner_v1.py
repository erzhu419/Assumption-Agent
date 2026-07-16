from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import threading
import tempfile
from typing import Any, Mapping
from unittest import mock
import urllib.error
import urllib.request

import pytest

import replication_runtime.noaa_gsod_v1.development_runner as development_runner
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
)
from replication_runtime.noaa_gsod_v1 import oracle_sqlite, oracle_stdlib
from replication_runtime.noaa_gsod_v1.contract import (
    ORACLE_IDS,
    STUDY_ID,
    TASK_CONTRACT,
    canonical_json_bytes,
    payload_hash,
    sha256_file,
    with_self_hash,
)
from replication_runtime.noaa_gsod_v1.development_freeze import (
    CONTROLLER_PLAN_VERSION,
    DEVELOPMENT_ITEM_COUNT,
    ENDPOINT_IDENTITY_VERSION,
    MODEL_ID,
    MODEL_OUTPUT_TOKEN_BUDGET,
    MODEL_REQUEST_BODY_BYTE_BUDGET,
    MODEL_WORK_UNIT_COUNT,
    PRE_RUN_FREEZE_VERSION,
    ProviderIdentity,
    WORKER_PLAN_VERSION,
    WORK_UNIT_COUNT,
)
from replication_runtime.noaa_gsod_v1.development_runner import (
    DevelopmentRunnerError,
    ModelRequest,
    NoReplayError,
    ProviderCredential,
    ProviderProtocolError,
    ProviderTransportUnavailable,
    UrllibOpenAICompatibleTransport,
    endpoint_identity_hash,
    load_provider_credential,
    run_formal_development,
    run_synthetic_development_for_tests,
)
from replication_runtime.noaa_gsod_v1.development_implementation import (
    build_development_implementation_set,
)
from replication_runtime.noaa_gsod_v1.development_schemas import (
    DEVELOPMENT_SCHEMA_SET_HASH,
)
from replication_runtime.noaa_gsod_v1.pack import read_json, write_json
from replication_runtime.noaa_gsod_v1.typed_relational import (
    OPERATOR_VERSION,
    TypedRelationalProgram,
)


EXPECTED_OUTPUT = {
    "mean_daily_precip_mm": "50.80",
    "month": "02",
    "valid_day_count": 1,
}


def _program() -> TypedRelationalProgram:
    return TypedRelationalProgram(
        missing_tokens=("", "99.99"),
        year=2020,
        aggregation="mean",
        extreme="argmax",
        tie_break="earliest",
        unit_factor="25.4",
        rounding="ROUND_HALF_UP",
        decimal_places=2,
    )


def _write_env(path: Path, *, key: str) -> None:
    path.write_text(
        "\n".join(
            (
                "ASSUMPTION_V2_API_BASE=https://ruoli.dev",
                f"ASSUMPTION_V2_API_KEY={key}",
                f"ASSUMPTION_V2_MODEL={MODEL_ID}",
                "ASSUMPTION_V2_PROVIDER_CHAIN=openai_compatible",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _synthetic_execution_root(tmp_path: Path) -> dict[str, Any]:
    development = tmp_path / "development"
    inputs = development / "generation_inputs"
    operator_root = development / "operator"
    inputs.mkdir(parents=True)
    operator_root.mkdir()
    program = _program()
    envelope = {
        "formation_receipt_hash": payload_hash({"fixture": "formation"}),
        "operator_version": OPERATOR_VERSION,
        "program": program.to_dict(),
        "program_hash": program.program_hash,
        "raw_content_persisted": False,
    }
    operator_path = operator_root / "frozen_program.json"
    write_json(operator_path, envelope)

    items: list[dict[str, Any]] = []
    for ordinal in range(DEVELOPMENT_ITEM_COUNT):
        item_id = f"development_item_{ordinal:02d}"
        relative = f"generation_inputs/{item_id}.csv"
        source = development / relative
        source.write_text(
            "STATION,DATE,PRCP\n"
            f"DEVELOPMENT_STATION_{ordinal:02d},2020-01-01,1.00\n"
            f"DEVELOPMENT_STATION_{ordinal:02d},2020-02-01,2.00\n",
            encoding="utf-8",
        )
        source_hash = sha256_file(source)
        items.append(
            {
                "anonymous_item_id": item_id,
                "anonymized_station_token": f"DEVELOPMENT_STATION_{ordinal:02d}",
                "input_columns": ["STATION", "DATE", "PRCP"],
                "input_relative_path": relative,
                "input_sha256": source_hash,
                "ordinal": ordinal,
            }
        )

    shared_context = {
        "max_output_tokens": MODEL_OUTPUT_TOKEN_BUDGET,
        "model_request_body_byte_budget": MODEL_REQUEST_BODY_BYTE_BUDGET,
        "task_contract": TASK_CONTRACT,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
    }
    shared_hash = payload_hash(shared_context)
    work_units: list[dict[str, Any]] = []
    for item in items:
        common = {
            "anonymous_item_id": item["anonymous_item_id"],
            "attempts": 1,
            "input_sha256": item["input_sha256"],
            "model": MODEL_ID,
            "model_request_hash": None,
            "shared_context_hash": shared_hash,
        }
        work_units.extend(
            (
                {
                    **common,
                    "arm": "raw_model",
                    "execution_kind": "provider_model",
                    "model_response_contract": "canonical_task_json_only",
                    "post_model_local_operator": False,
                    "program_id": None,
                    "work_unit_id": f"{item['anonymous_item_id']}:raw_model",
                },
                {
                    **common,
                    "arm": "agent_typed_model",
                    "execution_kind": "provider_model_then_local_operator",
                    "model_response_contract": "opaque_frozen_program_id_only",
                    "post_model_local_operator": True,
                    "program_id": program.program_hash,
                    "work_unit_id": (
                        f"{item['anonymous_item_id']}:agent_typed_model"
                    ),
                },
                {
                    **common,
                    "arm": "operator_only_local",
                    "execution_kind": "local_operator",
                    "model": None,
                    "model_response_contract": None,
                    "post_model_local_operator": True,
                    "program_id": program.program_hash,
                    "work_unit_id": (
                        f"{item['anonymous_item_id']}:operator_only_local"
                    ),
                },
            )
        )
    worker = with_self_hash(
        {
            "batch_policy": {
                "attempts_per_work_unit": 1,
                "fallback_condition": "complete_plus_canary_unavailable",
                "fallback_scope": "entire_model_batch_pre_submission",
                "maximum_model_concurrency": MODEL_WORK_UNIT_COUNT,
                "mid_batch_provider_switch": False,
                "model_batch_count": 1,
                "model_batch_size": MODEL_WORK_UNIT_COUNT,
                "replays": 0,
                "resamples": 0,
                "retries": 0,
            },
            "development_item_count": DEVELOPMENT_ITEM_COUNT,
            "items": items,
            "operator_binding": {
                "frozen_program_file_sha256": sha256_file(operator_path),
                "frozen_program_relative_path": "operator/frozen_program.json",
                "operator_version": OPERATOR_VERSION,
                "program_envelope_hash": payload_hash(envelope),
                "program_id": program.program_hash,
            },
            "shared_context": shared_context,
            "study_id": STUDY_ID,
            "work_unit_counts": {
                "agent_typed_model": DEVELOPMENT_ITEM_COUNT,
                "model_total": MODEL_WORK_UNIT_COUNT,
                "operator_only_local": DEVELOPMENT_ITEM_COUNT,
                "raw_model": DEVELOPMENT_ITEM_COUNT,
                "total": WORK_UNIT_COUNT,
            },
            "work_units": work_units,
            "worker_plan_version": WORKER_PLAN_VERSION,
        },
        "worker_plan_hash",
    )
    write_json(development / "worker_plan.json", worker)
    staged_input_set_hash = payload_hash(
        [
            {
                "anonymous_item_id": item["anonymous_item_id"],
                "input_sha256": item["input_sha256"],
            }
            for item in items
        ]
    )
    source_view_binding = {
        "development_source_index_file_sha256": payload_hash(
            {"fixture": "source-index-file"}
        ),
        "development_source_index_hash": payload_hash(
            {"fixture": "source-index"}
        ),
        "development_source_receipt_file_sha256": payload_hash(
            {"fixture": "source-receipt-file"}
        ),
        "development_source_receipt_hash": payload_hash(
            {"fixture": "source-receipt"}
        ),
        "source_view_input_set_hash": staged_input_set_hash,
        "source_view_tree_hash": payload_hash({"fixture": "source-tree"}),
        "staged_input_set_hash": staged_input_set_hash,
    }
    controller = with_self_hash(
        {
            "controller_plan_version": CONTROLLER_PLAN_VERSION,
            "development_root": str(development.resolve()),
            "development_root_commitment": payload_hash(
                str(development.resolve())
            ),
            "generation_worker_plan_hash": worker["worker_plan_hash"],
            "post_join_verification": {
                "all_work_units_must_join_before_release": True,
                "expected_join_count": WORK_UNIT_COUNT,
                "gold_or_oracle_material_in_worker_plan": False,
                "offline_oracle_ids": list(ORACLE_IDS),
                "offline_oracle_release_phase": (
                    "after_all_generation_and_operator_join"
                ),
                "online_judge_calls": 0,
                "required_offline_oracle_calls": (
                    DEVELOPMENT_ITEM_COUNT * len(ORACLE_IDS)
                ),
            },
            "source_view_binding": source_view_binding,
            "study_id": STUDY_ID,
        },
        "controller_plan_hash",
    )
    write_json(development / "controller_plan.private.json", controller)

    plus_channel = "ruoli-plus-fixture"
    pro_channel = "ruoli-pro-fixture"
    provider_identity = ProviderIdentity(
        plus_channel_id=plus_channel,
        plus_endpoint_origin="https://ruoli.dev",
        pro_channel_id=pro_channel,
        pro_endpoint_origin="https://ruoli.dev",
    )
    bindings = {
        "acquisition_receipt_file_sha256": payload_hash(
            {"fixture": "acquisition-file"}
        ),
        "acquisition_receipt_hash": payload_hash({"fixture": "acquisition"}),
        "candidate_formation_receipt_file_sha256": payload_hash(
            {"fixture": "formation-file"}
        ),
        "candidate_formation_receipt_hash": payload_hash(
            {"fixture": "formation"}
        ),
        "candidate_program_file_sha256": sha256_file(operator_path),
        "candidate_program_id": program.program_hash,
        "controller_plan_hash": controller["controller_plan_hash"],
        "development_root_commitment": controller[
            "development_root_commitment"
        ],
        "development_schema_set_hash": DEVELOPMENT_SCHEMA_SET_HASH,
        "development_source_index_file_sha256": source_view_binding[
            "development_source_index_file_sha256"
        ],
        "development_source_index_hash": source_view_binding[
            "development_source_index_hash"
        ],
        "development_source_input_set_hash": staged_input_set_hash,
        "development_source_receipt_file_sha256": source_view_binding[
            "development_source_receipt_file_sha256"
        ],
        "development_source_receipt_hash": source_view_binding[
            "development_source_receipt_hash"
        ],
        "development_source_tree_hash": source_view_binding[
            "source_view_tree_hash"
        ],
        "implementation_set_hash": build_development_implementation_set()[
            "implementation_set_hash"
        ],
        "provider_identity_hash": provider_identity.identity_hash,
        "staged_input_set_hash": staged_input_set_hash,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
        "train_preparation_receipt_file_sha256": payload_hash(
            {"fixture": "train-file"}
        ),
        "train_preparation_receipt_hash": payload_hash({"fixture": "train"}),
        "worker_plan_hash": worker["worker_plan_hash"],
    }
    public_freeze = with_self_hash(
        {
            "binding_hashes": bindings,
            "call_ledger_at_freeze": {
                "model_calls": 0,
                "network_calls": 0,
                "offline_oracle_calls": 0,
                "online_judge_calls": 0,
                "operator_calls": 0,
                "scoring_calls": 0,
            },
            "content_boundary": {
                "development_gold_persisted_publicly": False,
                "development_raw_input_persisted_publicly": False,
                "development_station_identity_persisted_publicly": False,
                "model_answer_persisted_publicly": False,
                "private_controller_plan_persisted_publicly": False,
                "sealed_mapping_persisted_publicly": False,
                "source_view_private_index_persisted_publicly": False,
                "task_content_persisted_publicly": False,
                "trace_persisted_publicly": False,
            },
            "freeze_state": {
                "development_input_accessed": True,
                "development_input_staged": True,
                "generation_joined_count": 0,
                "generation_started": False,
                "gold_released": False,
                "launch_authorized": False,
                "model_request_hashes_precommitted": False,
                "operator_joined_count": 0,
                "scored": False,
                "sealed_runtime_accessed": False,
                "staged_item_count": DEVELOPMENT_ITEM_COUNT,
                "status": "pre_run_frozen_not_launched",
            },
            "performance_gate_added": False,
            "pre_run_freeze_version": PRE_RUN_FREEZE_VERSION,
            "provider_policy": {
                "endpoint_identity_version": ENDPOINT_IDENTITY_VERSION,
                "fallback_condition": "complete_plus_canary_unavailable",
                "fallback_scope": (
                    "entire_12_model_unit_batch_before_submission"
                ),
                "mid_batch_switch": False,
                "model": MODEL_ID,
                "primary_tier": "plus",
                "secondary_tier": "pro",
                "secret_hmac_precommit_phase": (
                    "runner_launch_precommit_before_any_model_submission"
                ),
            },
            "schedule": {
                "agent_typed_model_units": DEVELOPMENT_ITEM_COUNT,
                "attempts_per_unit": 1,
                "development_items": DEVELOPMENT_ITEM_COUNT,
                "maximum_model_concurrency": MODEL_WORK_UNIT_COUNT,
                "max_output_tokens": MODEL_OUTPUT_TOKEN_BUDGET,
                "model_request_body_byte_budget": MODEL_REQUEST_BODY_BYTE_BUDGET,
                "model_units": MODEL_WORK_UNIT_COUNT,
                "operator_only_local_units": DEVELOPMENT_ITEM_COUNT,
                "raw_model_units": DEVELOPMENT_ITEM_COUNT,
                "replays": 0,
                "resamples": 0,
                "retries": 0,
                "total_work_units": WORK_UNIT_COUNT,
            },
            "study_id": STUDY_ID,
        },
        "pre_run_freeze_hash",
    )
    public_path = tmp_path / "public-freeze.json"
    write_json(public_path, public_freeze)
    secure_env_root = tempfile.TemporaryDirectory(
        prefix="noaa-provider-fixture-", dir="/tmp"
    )
    plus_env = Path(secure_env_root.name) / "plus.env"
    pro_env = Path(secure_env_root.name) / "pro.env"
    _write_env(plus_env, key="fixture-plus-secret-value")
    _write_env(pro_env, key="fixture-pro-secret-value")
    return {
        "development": development,
        "output": development / "formal_run",
        "plus_channel": plus_channel,
        "plus_env": plus_env,
        "pro_channel": pro_channel,
        "pro_env": pro_env,
        "program": program,
        "public": public_path,
        "secure_env_root": secure_env_root,
        "worker": worker,
    }


class FakeTransport:
    def __init__(
        self,
        *,
        plus_canary: str = "complete transport response; semantics ignored",
        plus_unavailable: bool = False,
        plus_protocol_failure: bool = False,
        plus_http_status: int | None = None,
        malformed_raw_suffix: str | None = None,
        task_unavailable_suffix: str | None = None,
        wrong_raw_suffix: str | None = None,
    ) -> None:
        self.plus_canary = plus_canary
        self.plus_unavailable = plus_unavailable
        self.plus_protocol_failure = plus_protocol_failure
        self.plus_http_status = plus_http_status
        self.malformed_raw_suffix = malformed_raw_suffix
        self.task_unavailable_suffix = task_unavailable_suffix
        self.wrong_raw_suffix = wrong_raw_suffix
        self.calls: list[tuple[str, str]] = []
        self.request_bodies: list[dict[str, Any]] = []
        self.output_root: Path | None = None
        self.task_claim_counts_before_transport: list[int] = []
        self.task_claim_set_present_before_transport: list[bool] = []
        self._lock = threading.Lock()

    def complete(
        self, *, credential: ProviderCredential, request: ModelRequest
    ) -> str:
        with self._lock:
            self.calls.append((credential.channel_id, request.purpose))
            self.request_bodies.append(request.body())
            if (
                request.purpose != "provider_transport_canary"
                and self.output_root is not None
            ):
                self.task_claim_counts_before_transport.append(
                    len(
                        list(
                            (self.output_root / "worker_state").glob(
                                "*/claim.json"
                            )
                        )
                    )
                )
                self.task_claim_set_present_before_transport.append(
                    (self.output_root / "work.claim-set.json").is_file()
                )
        if request.purpose == "provider_transport_canary":
            if "plus" in credential.channel_id:
                if self.plus_http_status is not None:
                    error = urllib.error.HTTPError(
                        credential.api_base,
                        self.plus_http_status,
                        "synthetic status",
                        hdrs=None,
                        fp=None,
                    )
                    with mock.patch.object(
                        urllib.request, "urlopen", side_effect=error
                    ):
                        return UrllibOpenAICompatibleTransport().complete(
                            credential=credential,
                            request=request,
                        )
                if self.plus_unavailable:
                    raise ProviderTransportUnavailable("fixture unavailable")
                if self.plus_protocol_failure:
                    raise ProviderProtocolError("fixture protocol failure")
                return self.plus_canary
            return "complete Pro transport response"
        if self.task_unavailable_suffix and request.purpose.endswith(
            self.task_unavailable_suffix
        ):
            raise ProviderTransportUnavailable(
                "synthetic selected-provider task interruption"
            )
        if request.purpose.startswith("raw_model:"):
            output = EXPECTED_OUTPUT
            if self.wrong_raw_suffix and request.purpose.endswith(
                self.wrong_raw_suffix
            ):
                output = {
                    "mean_daily_precip_mm": "25.40",
                    "month": "01",
                    "valid_day_count": 1,
                }
            result = canonical_json_bytes(output).decode("utf-8")
            if self.malformed_raw_suffix and request.purpose.endswith(
                self.malformed_raw_suffix
            ):
                return json.dumps(EXPECTED_OUTPUT, sort_keys=True)
            return result
        if request.purpose.startswith("agent_typed_model:"):
            program_id = request.user_payload["output_schema"]["properties"][
                "program_id"
            ]["enum"][0]
            return canonical_json_bytes({"program_id": program_id}).decode("utf-8")
        raise AssertionError("unexpected fake request")


def _run(fixture: Mapping[str, Any], transport: FakeTransport, **kwargs: Any):
    transport.output_root = Path(fixture["output"])
    return run_synthetic_development_for_tests(
        development_root=fixture["development"],
        public_freeze_path=fixture["public"],
        output_root=fixture["output"],
        plus_env_file=fixture["plus_env"],
        pro_env_file=fixture["pro_env"],
        plus_channel_id=fixture["plus_channel"],
        pro_channel_id=fixture["pro_channel"],
        transport=transport,
        **kwargs,
    )


def test_live_implementation_drift_fails_before_output_or_transport(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    frozen = read_json(fixture["public"])
    frozen.pop("pre_run_freeze_hash")
    frozen["binding_hashes"]["implementation_set_hash"] = "f" * 64
    write_json(
        fixture["public"],
        with_self_hash(frozen, "pre_run_freeze_hash"),
    )
    transport = FakeTransport()

    with pytest.raises(
        DevelopmentRunnerError,
        match="live development implementation differs",
    ):
        _run(fixture, transport)

    assert transport.calls == []
    assert not fixture["output"].exists()


def test_formal_entrypoint_forbids_injection_and_binds_exact_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = inspect.signature(run_formal_development).parameters
    assert "transport" not in parameters
    assert "oracle_functions" not in parameters

    fixture = _synthetic_execution_root(tmp_path)
    fake = FakeTransport()
    fake.output_root = Path(fixture["output"])

    def complete(
        _self: UrllibOpenAICompatibleTransport,
        *,
        credential: ProviderCredential,
        request: ModelRequest,
    ) -> str:
        return fake.complete(credential=credential, request=request)

    monkeypatch.setattr(UrllibOpenAICompatibleTransport, "complete", complete)
    report = run_formal_development(
        development_root=fixture["development"],
        public_freeze_path=fixture["public"],
        output_root=fixture["output"],
        plus_env_file=fixture["plus_env"],
        pro_env_file=fixture["pro_env"],
        plus_channel_id=fixture["plus_channel"],
        pro_channel_id=fixture["pro_channel"],
    )

    assert report["dependency_injection_used"] is False
    assert report["formal_evidence"] is True
    assert report["execution_integrity_valid"] is True
    assert report["paired_evidence_complete"] is True
    assert report["formal_evidence_valid"] is True
    assert report["evidence_valid"] is True
    assert report["execution_dependency_ids"] == {
        "local_operator": OPERATOR_VERSION,
        "model_transport": "urllib_openai_compatible_chat_completions_v1",
        "offline_oracles": list(ORACLE_IDS),
    }


def test_plus_complete_transport_runs_one_18_future_batch_then_dual_oracle(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    transport = FakeTransport(plus_canary="not JSON and still complete")
    oracle_release_counts: list[int] = []

    def checked_stdlib(path: str | Path) -> dict[str, Any]:
        oracle_release_counts.append(
            len(
                list(
                    (fixture["output"] / "worker_state").glob(
                        "*/terminal.private.json"
                    )
                )
            )
        )
        return oracle_stdlib.evaluate(path)

    def checked_sqlite(path: str | Path) -> dict[str, Any]:
        oracle_release_counts.append(
            len(
                list(
                    (fixture["output"] / "worker_state").glob(
                        "*/terminal.private.json"
                    )
                )
            )
        )
        return oracle_sqlite.evaluate(path)

    report = _run(
        fixture,
        transport,
        oracle_functions=(
            (oracle_stdlib.ORACLE_ID, checked_stdlib),
            (oracle_sqlite.ORACLE_ID, checked_sqlite),
        ),
    )

    assert report["execution_completed"] is True
    assert report["dependency_injection_used"] is True
    assert report["formal_evidence"] is False
    assert report["evidence_valid"] is False
    assert report["formal_evidence_valid"] is False
    assert report["execution_integrity_valid"] is True
    assert report["paired_evidence_complete"] is True
    assert report["execution_dependency_ids"] == {
        "local_operator": OPERATOR_VERSION,
        "model_transport": "synthetic_injected_unattested_v1",
        "offline_oracles": ["synthetic_injected_unattested_v1"],
    }
    assert report["selected_provider_label"] == "plus"
    assert report["joined_work_unit_count"] == WORK_UNIT_COUNT
    assert report["concurrency"]["configured_work_concurrency"] == 18
    assert report["concurrency"]["observed_maximum_model_calls"] == 12
    assert report["concurrency"][
        "all_claims_persisted_before_work_start"
    ] is True
    assert report["call_ledger"] == {
        "canary_model_calls": 1,
        "offline_oracle_calls": 12,
        "online_judge_calls": 0,
        "operator_calls": 12,
        "replays": 0,
        "resamples": 0,
        "retries": 0,
        "scoring_model_calls": 0,
        "task_model_calls": 12,
        "total_model_calls": 13,
    }
    assert report["offline_evaluation"]["arm_exact_success_counts"] == {
        "agent_typed_model": 6,
        "operator_only_local": 6,
        "raw_model": 6,
    }
    assert report["offline_evaluation"]["pairwise_item_counts"] == {
        "agent_typed_minus_operator_only": {
            "complete_pair_count": 6,
            "gain_count": 0,
            "harm_count": 0,
            "incomplete_pair_count": 0,
            "paired_net_gain": 0,
            "tie_count": 6,
        },
        "agent_typed_minus_raw": {
            "complete_pair_count": 6,
            "gain_count": 0,
            "harm_count": 0,
            "incomplete_pair_count": 0,
            "paired_net_gain": 0,
            "tie_count": 6,
        },
        "operator_only_minus_raw": {
            "complete_pair_count": 6,
            "gain_count": 0,
            "harm_count": 0,
            "incomplete_pair_count": 0,
            "paired_net_gain": 0,
            "tie_count": 6,
        },
    }
    assert oracle_release_counts == [18] * 12
    assert len(transport.calls) == 13
    assert "response_format" not in transport.request_bodies[0]
    assert all(
        body.get("response_format") == {"type": "json_object"}
        for body in transport.request_bodies[1:]
    )
    assert transport.task_claim_counts_before_transport == [18] * 12
    assert transport.task_claim_set_present_before_transport == [True] * 12
    assert not any("pro" in channel for channel, _purpose in transport.calls)
    precommit = read_json(
        fixture["output"] / "provider.identity.precommit.json"
    )
    assert precommit["provider_registration_contract"] == {
        "api_origin": "https://ruoli.dev",
        "model": MODEL_ID,
        "provider_labels": ["plus", "pro"],
    }
    assert precommit["providers"]["plus"]["provider_label"] == "plus"
    assert precommit["providers"]["pro"]["provider_label"] == "pro"
    assert (
        precommit["providers"]["plus"]["api_key_hmac_sha256"]
        != precommit["providers"]["pro"]["api_key_hmac_sha256"]
    )
    serialized = canonical_json_bytes(report)
    for forbidden in (
        b"fixture-plus-secret-value",
        b"fixture-pro-secret-value",
        b"input_csv_utf8",
        b"mean_daily_precip_mm",
        b"gold_commitment",
        b"model_response_hash",
    ):
        assert forbidden not in serialized


def test_complete_request_body_byte_budget_is_enforced_before_batch_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    transport = FakeTransport()
    monkeypatch.setattr(
        development_runner,
        "MODEL_REQUEST_BODY_BYTE_BUDGET",
        1,
    )

    with pytest.raises(DevelopmentRunnerError, match="request body exceeds"):
        _run(fixture, transport)

    assert transport.calls == [
        (fixture["plus_channel"], "provider_transport_canary")
    ]
    assert not (fixture["output"] / "batch.launch.json").exists()
    assert not (fixture["output"] / "worker_state").exists()


def test_plus_canary_401_authorizes_whole_batch_pro_fallback(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    transport = FakeTransport(plus_http_status=401)
    report = _run(fixture, transport)

    assert report["selected_provider_label"] == "pro"
    assert report["call_ledger"]["canary_model_calls"] == 2
    task_calls = [row for row in transport.calls if row[1] != "provider_transport_canary"]
    assert len(task_calls) == 12
    assert all("pro" in channel for channel, _purpose in task_calls)
    assert report["call_ledger"]["retries"] == 0
    assert report["call_ledger"]["replays"] == 0
    assert report["call_ledger"]["resamples"] == 0


def test_http_400_plus_canary_failure_never_probes_pro_or_reads_tasks(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    transport = FakeTransport(plus_http_status=400)
    with pytest.raises(ProviderProtocolError):
        _run(fixture, transport)

    assert transport.calls == [
        (fixture["plus_channel"], "provider_transport_canary")
    ]
    assert not (fixture["output"] / "batch.launch.json").exists()
    assert not (fixture["output"] / "worker_state").exists()


def test_incomplete_atomic_claim_blocks_replay_and_oracle_release(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    output = fixture["output"]
    output.mkdir()
    unit = fixture["worker"]["work_units"][0]
    unit_hash = payload_hash(unit)
    atomic_write_hashed_json_v2(
        output / "worker_state" / unit_hash / "claim.json",
        {"fixture": "crash-after-claim"},
        hash_field="claim_hash",
    )
    transport = FakeTransport()

    def forbidden_oracle(_path: str | Path) -> dict[str, Any]:
        raise AssertionError("oracle released before a complete 18-unit join")

    for _attempt in range(2):
        with pytest.raises(NoReplayError):
            _run(
                fixture,
                transport,
                oracle_functions=(
                    (oracle_stdlib.ORACLE_ID, forbidden_oracle),
                    (oracle_sqlite.ORACLE_ID, forbidden_oracle),
                ),
            )
    assert transport.calls == []
    assert not (output / "evaluation.private.json").exists()
    assert not (output / "development.report.json").exists()


def test_invalid_raw_contract_is_one_terminal_failure_not_a_retry(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    transport = FakeTransport(
        malformed_raw_suffix="development_item_00:raw_model"
    )
    report = _run(fixture, transport)

    assert report["call_ledger"]["task_model_calls"] == 12
    assert report["call_ledger"]["retries"] == 0
    assert report["offline_evaluation"]["arm_contract_valid_counts"] == {
        "agent_typed_model": 6,
        "operator_only_local": 6,
        "raw_model": 5,
    }
    assert report["offline_evaluation"]["arm_exact_success_counts"][
        "raw_model"
    ] == 5
    paired = report["offline_evaluation"]["pairwise_item_counts"]
    assert paired["agent_typed_minus_raw"] == {
        "complete_pair_count": 6,
        "gain_count": 1,
        "harm_count": 0,
        "incomplete_pair_count": 0,
        "paired_net_gain": 1,
        "tie_count": 5,
    }
    assert paired["operator_only_minus_raw"] == {
        "complete_pair_count": 6,
        "gain_count": 1,
        "harm_count": 0,
        "incomplete_pair_count": 0,
        "paired_net_gain": 1,
        "tie_count": 5,
    }


def test_item_paired_gain_counts_only_two_valid_outputs(tmp_path: Path) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    transport = FakeTransport(
        wrong_raw_suffix="development_item_00:raw_model"
    )
    report = _run(fixture, transport)

    paired = report["offline_evaluation"]["pairwise_item_counts"]
    assert paired["agent_typed_minus_raw"] == {
        "complete_pair_count": 6,
        "gain_count": 1,
        "harm_count": 0,
        "incomplete_pair_count": 0,
        "paired_net_gain": 1,
        "tie_count": 5,
    }
    assert paired["operator_only_minus_raw"] == paired[
        "agent_typed_minus_raw"
    ]


def test_mid_batch_task_failure_never_switches_provider(tmp_path: Path) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    transport = FakeTransport(
        task_unavailable_suffix="development_item_00:raw_model"
    )
    report = _run(fixture, transport)

    assert report["selected_provider_label"] == "plus"
    assert not any("pro" in channel for channel, _purpose in transport.calls)
    assert report["call_ledger"]["task_model_calls"] == 12
    assert report["call_ledger"]["retries"] == 0
    assert report["offline_evaluation"]["arm_contract_valid_counts"][
        "raw_model"
    ] == 5


def test_complete_terminal_grid_resumes_only_offline_aggregate(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    _run(fixture, FakeTransport())
    (fixture["output"] / "development.report.json").unlink()

    recovery_transport = FakeTransport()
    report = _run(fixture, recovery_transport)

    assert recovery_transport.calls == []
    assert report["recovered_work_unit_count"] == WORK_UNIT_COUNT
    assert report["concurrency"][
        "recovered_from_complete_terminal_grid"
    ] is True
    assert report["concurrency"][
        "all_futures_submitted_before_results_read"
    ] is False
    assert report["offline_evaluation"]["oracle_call_count"] == 12


def test_identical_plus_and_pro_key_commitments_fail_before_transport(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_execution_root(tmp_path)
    _write_env(
        fixture["pro_env"], key="fixture-plus-secret-value"
    )
    transport = FakeTransport()
    with pytest.raises(DevelopmentRunnerError, match="distinct API-key HMAC"):
        _run(fixture, transport)

    assert transport.calls == []
    assert not (
        fixture["output"] / "provider.identity.precommit.json"
    ).exists()


def test_env_requires_0600_exact_allowlist_and_never_echoes_secret(
    tmp_path: Path,
) -> None:
    del tmp_path
    with tempfile.TemporaryDirectory(
        prefix="noaa-env-security-", dir="/tmp"
    ) as secure_root:
        env = Path(secure_root) / "bad.env"
        secret = "fixture-secret-that-must-not-appear"
        _write_env(env, key=secret)
        os.chmod(env, 0o644)
        with pytest.raises(DevelopmentRunnerError) as permissions_error:
            load_provider_credential(env, channel_id="ruoli-plus-fixture")
        assert secret not in str(permissions_error.value)

        os.chmod(env, 0o600)
        with env.open("a", encoding="utf-8") as handle:
            handle.write("UNAPPROVED_KEY=value\n")
        with pytest.raises(DevelopmentRunnerError) as allowlist_error:
            load_provider_credential(env, channel_id="ruoli-plus-fixture")
        assert secret not in str(allowlist_error.value)

    assert endpoint_identity_hash("https://ruoli.dev") == payload_hash(
        {
            "canonical_origin": "https://ruoli.dev",
            "endpoint_identity_version": ENDPOINT_IDENTITY_VERSION,
        }
    )
