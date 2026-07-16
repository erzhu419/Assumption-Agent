from __future__ import annotations

from typing import Any

from .contract import payload_hash


ORACLE_RESULT_FIELDS = frozenset(
    {"mean_daily_precip_mm", "month", "valid_day_count"}
)
PRIVATE_ITEM_FIELDS = frozenset(
    {
        "gold_commitment",
        "item_commitment",
        "item_id",
        "oracle_outputs",
        "ordinal",
        "partition",
        "raw_csv_relative_path",
        "raw_csv_sha256",
        "station_id",
        "station_metadata_commitment",
        "task_payload_commitment",
    }
)
PRIVATE_PACK_FIELDS = frozenset(
    {
        "acquisition_statistics",
        "candidate_imports",
        "ground_truth_persisted_only_in_private_root",
        "items",
        "model_calls",
        "online_judge_calls",
        "oracle_ids",
        "pack_hash",
        "pack_version",
        "partition_counts",
        "scoring_calls",
        "selection_policy",
        "source_commitments",
        "study_id",
        "task_contract",
        "task_contract_hash",
        "year",
    }
)
PUBLIC_RECEIPT_FIELDS = frozenset(
    {
        "acquisition_statistics",
        "candidate_imports",
        "content_boundary",
        "item_commitments_by_partition",
        "model_calls",
        "network_calls",
        "online_judge_calls",
        "oracle_ids",
        "pack_hash",
        "partition_counts",
        "private_pack_git_persisted",
        "public_receipt_version",
        "receipt_hash",
        "resampling_used",
        "schema_set_hash",
        "scoring_calls",
        "selection_policy_hash",
        "source_commitments",
        "source_urls",
        "study_id",
        "task_contract_hash",
        "typed_operator_formed",
        "year",
    }
)

# A dependency-free, JSON-serializable schema declaration.  Runtime validation
# in pack.py is intentionally stricter than merely accepting these types.
SCHEMA_SET: dict[str, Any] = {
    "schema_version": "noaa_gsod_auto_typed_operator_schema_set_v1",
    "oracle_result": {
        "additionalProperties": False,
        "properties": {
            "mean_daily_precip_mm": "fixed-point string /^[0-9]+\\.[0-9]{2}$/",
            "month": "string /^(0[1-9]|1[0-2])$/",
            "valid_day_count": "integer >= 1",
        },
        "required": sorted(ORACLE_RESULT_FIELDS),
        "type": "object",
    },
    "private_item": {
        "additionalProperties": False,
        "required": sorted(PRIVATE_ITEM_FIELDS),
        "type": "object",
    },
    "private_pack": {
        "additionalProperties": False,
        "required": sorted(PRIVATE_PACK_FIELDS),
        "type": "object",
    },
    "public_receipt": {
        "additionalProperties": False,
        "required": sorted(PUBLIC_RECEIPT_FIELDS),
        "type": "object",
    },
}
SCHEMA_SET_HASH = payload_hash(SCHEMA_SET)
