from __future__ import annotations

from typing import Any

from .contract import payload_hash


TRAIN_VIEW_ITEM_FIELDS = frozenset(
    {
        "anonymized_station_token",
        "input_relative_path",
        "input_sha256",
        "input_columns",
        "oracle_consensus",
        "oracle_consensus_hash",
        "ordinal",
        "train_item_hash",
        "train_item_id",
    }
)
TRAIN_VIEW_FIELDS = frozenset(
    {
        "candidate_imports",
        "items",
        "model_calls",
        "network_calls",
        "online_judge_calls",
        "oracle_consensus_ids",
        "partition",
        "role",
        "scoring_calls",
        "source_private_pack_hash",
        "study_id",
        "task_contract",
        "task_contract_hash",
        "train_item_count",
        "train_view_hash",
        "train_view_version",
        "typed_operator_formed",
    }
)
TRAIN_PREPARATION_RECEIPT_FIELDS = frozenset(
    {
        "candidate_imports",
        "content_boundary",
        "model_calls",
        "network_calls",
        "online_judge_calls",
        "oracle_verification_calls",
        "preparation_receipt_hash",
        "preparation_receipt_version",
        "schema_set_hash",
        "scoring_calls",
        "source_private_pack_hash",
        "study_id",
        "task_contract_hash",
        "train_consensus_set_hash",
        "train_input_set_hash",
        "train_item_count",
        "train_view_hash",
        "train_view_git_persisted",
        "typed_operator_formed",
    }
)

TRAIN_SCHEMA_SET: dict[str, Any] = {
    "schema_version": "noaa_gsod_train_export_schema_set_v1",
    "train_view_item": {
        "additionalProperties": False,
        "required": sorted(TRAIN_VIEW_ITEM_FIELDS),
        "type": "object",
    },
    "train_view": {
        "additionalProperties": False,
        "required": sorted(TRAIN_VIEW_FIELDS),
        "type": "object",
    },
    "train_preparation_receipt": {
        "additionalProperties": False,
        "required": sorted(TRAIN_PREPARATION_RECEIPT_FIELDS),
        "type": "object",
    },
}
TRAIN_SCHEMA_SET_HASH = payload_hash(TRAIN_SCHEMA_SET)
