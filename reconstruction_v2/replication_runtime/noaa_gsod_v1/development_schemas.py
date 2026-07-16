from __future__ import annotations

from typing import Any

from .contract import payload_hash


DEVELOPMENT_ITEM_FIELDS = frozenset(
    {
        "anonymous_item_id",
        "anonymized_station_token",
        "input_columns",
        "input_relative_path",
        "input_sha256",
        "ordinal",
    }
)
WORK_UNIT_FIELDS = frozenset(
    {
        "anonymous_item_id",
        "arm",
        "attempts",
        "execution_kind",
        "input_sha256",
        "model",
        "model_request_hash",
        "model_response_contract",
        "post_model_local_operator",
        "program_id",
        "shared_context_hash",
        "work_unit_id",
    }
)
BATCH_POLICY_FIELDS = frozenset(
    {
        "attempts_per_work_unit",
        "fallback_condition",
        "fallback_scope",
        "maximum_model_concurrency",
        "mid_batch_provider_switch",
        "model_batch_count",
        "model_batch_size",
        "replays",
        "resamples",
        "retries",
    }
)
OPERATOR_BINDING_FIELDS = frozenset(
    {
        "frozen_program_file_sha256",
        "frozen_program_relative_path",
        "operator_version",
        "program_envelope_hash",
        "program_id",
    }
)
SHARED_CONTEXT_FIELDS = frozenset(
    {
        "max_output_tokens",
        "model_request_body_byte_budget",
        "task_contract",
        "task_contract_hash",
    }
)
WORK_UNIT_COUNTS_FIELDS = frozenset(
    {
        "agent_typed_model",
        "model_total",
        "operator_only_local",
        "raw_model",
        "total",
    }
)
WORKER_PLAN_FIELDS = frozenset(
    {
        "batch_policy",
        "development_item_count",
        "items",
        "operator_binding",
        "shared_context",
        "study_id",
        "work_unit_counts",
        "work_units",
        "worker_plan_hash",
        "worker_plan_version",
    }
)
SOURCE_VIEW_BINDING_FIELDS = frozenset(
    {
        "development_source_index_file_sha256",
        "development_source_index_hash",
        "development_source_receipt_file_sha256",
        "development_source_receipt_hash",
        "source_view_input_set_hash",
        "source_view_tree_hash",
        "staged_input_set_hash",
    }
)
POST_JOIN_VERIFICATION_FIELDS = frozenset(
    {
        "all_work_units_must_join_before_release",
        "expected_join_count",
        "gold_or_oracle_material_in_worker_plan",
        "offline_oracle_ids",
        "offline_oracle_release_phase",
        "online_judge_calls",
        "required_offline_oracle_calls",
    }
)
CONTROLLER_PLAN_FIELDS = frozenset(
    {
        "controller_plan_hash",
        "controller_plan_version",
        "development_root",
        "development_root_commitment",
        "generation_worker_plan_hash",
        "post_join_verification",
        "source_view_binding",
        "study_id",
    }
)
PUBLIC_BINDING_HASH_FIELDS = frozenset(
    {
        "acquisition_receipt_file_sha256",
        "acquisition_receipt_hash",
        "candidate_formation_receipt_file_sha256",
        "candidate_formation_receipt_hash",
        "candidate_program_file_sha256",
        "candidate_program_id",
        "controller_plan_hash",
        "development_root_commitment",
        "development_schema_set_hash",
        "development_source_index_file_sha256",
        "development_source_index_hash",
        "development_source_input_set_hash",
        "development_source_receipt_file_sha256",
        "development_source_receipt_hash",
        "development_source_tree_hash",
        "implementation_set_hash",
        "provider_identity_hash",
        "staged_input_set_hash",
        "task_contract_hash",
        "train_preparation_receipt_file_sha256",
        "train_preparation_receipt_hash",
        "worker_plan_hash",
    }
)
CALL_LEDGER_FIELDS = frozenset(
    {
        "model_calls",
        "network_calls",
        "offline_oracle_calls",
        "online_judge_calls",
        "operator_calls",
        "scoring_calls",
    }
)
CONTENT_BOUNDARY_FIELDS = frozenset(
    {
        "development_gold_persisted_publicly",
        "development_raw_input_persisted_publicly",
        "development_station_identity_persisted_publicly",
        "model_answer_persisted_publicly",
        "private_controller_plan_persisted_publicly",
        "sealed_mapping_persisted_publicly",
        "source_view_private_index_persisted_publicly",
        "task_content_persisted_publicly",
        "trace_persisted_publicly",
    }
)
FREEZE_STATE_FIELDS = frozenset(
    {
        "development_input_accessed",
        "development_input_staged",
        "generation_joined_count",
        "generation_started",
        "gold_released",
        "launch_authorized",
        "model_request_hashes_precommitted",
        "operator_joined_count",
        "scored",
        "sealed_runtime_accessed",
        "staged_item_count",
        "status",
    }
)
PROVIDER_POLICY_FIELDS = frozenset(
    {
        "endpoint_identity_version",
        "fallback_condition",
        "fallback_scope",
        "mid_batch_switch",
        "model",
        "primary_tier",
        "secondary_tier",
        "secret_hmac_precommit_phase",
    }
)
SCHEDULE_FIELDS = frozenset(
    {
        "agent_typed_model_units",
        "attempts_per_unit",
        "development_items",
        "maximum_model_concurrency",
        "max_output_tokens",
        "model_request_body_byte_budget",
        "model_units",
        "operator_only_local_units",
        "raw_model_units",
        "replays",
        "resamples",
        "retries",
        "total_work_units",
    }
)
PUBLIC_FREEZE_FIELDS = frozenset(
    {
        "binding_hashes",
        "call_ledger_at_freeze",
        "content_boundary",
        "freeze_state",
        "performance_gate_added",
        "pre_run_freeze_hash",
        "pre_run_freeze_version",
        "provider_policy",
        "schedule",
        "study_id",
    }
)

DEVELOPMENT_SCHEMA_SET: dict[str, Any] = {
    "schema_version": "noaa_gsod_formal_development_schema_set_v2",
    "development_item": {
        "additionalProperties": False,
        "required": sorted(DEVELOPMENT_ITEM_FIELDS),
        "type": "object",
    },
    "work_unit": {
        "additionalProperties": False,
        "required": sorted(WORK_UNIT_FIELDS),
        "type": "object",
    },
    "batch_policy": {
        "additionalProperties": False,
        "required": sorted(BATCH_POLICY_FIELDS),
        "type": "object",
    },
    "operator_binding": {
        "additionalProperties": False,
        "required": sorted(OPERATOR_BINDING_FIELDS),
        "type": "object",
    },
    "shared_context": {
        "additionalProperties": False,
        "required": sorted(SHARED_CONTEXT_FIELDS),
        "type": "object",
    },
    "work_unit_counts": {
        "additionalProperties": False,
        "required": sorted(WORK_UNIT_COUNTS_FIELDS),
        "type": "object",
    },
    "worker_plan": {
        "additionalProperties": False,
        "required": sorted(WORKER_PLAN_FIELDS),
        "type": "object",
    },
    "source_view_binding": {
        "additionalProperties": False,
        "required": sorted(SOURCE_VIEW_BINDING_FIELDS),
        "type": "object",
    },
    "post_join_verification": {
        "additionalProperties": False,
        "required": sorted(POST_JOIN_VERIFICATION_FIELDS),
        "type": "object",
    },
    "controller_plan": {
        "additionalProperties": False,
        "required": sorted(CONTROLLER_PLAN_FIELDS),
        "type": "object",
    },
    "public_binding_hashes": {
        "additionalProperties": False,
        "required": sorted(PUBLIC_BINDING_HASH_FIELDS),
        "type": "object",
    },
    "call_ledger": {
        "additionalProperties": False,
        "required": sorted(CALL_LEDGER_FIELDS),
        "type": "object",
    },
    "content_boundary": {
        "additionalProperties": False,
        "required": sorted(CONTENT_BOUNDARY_FIELDS),
        "type": "object",
    },
    "freeze_state": {
        "additionalProperties": False,
        "required": sorted(FREEZE_STATE_FIELDS),
        "type": "object",
    },
    "provider_policy": {
        "additionalProperties": False,
        "required": sorted(PROVIDER_POLICY_FIELDS),
        "type": "object",
    },
    "schedule": {
        "additionalProperties": False,
        "required": sorted(SCHEDULE_FIELDS),
        "type": "object",
    },
    "public_pre_run_freeze": {
        "additionalProperties": False,
        "required": sorted(PUBLIC_FREEZE_FIELDS),
        "type": "object",
    },
}
DEVELOPMENT_SCHEMA_SET_HASH = payload_hash(DEVELOPMENT_SCHEMA_SET)
