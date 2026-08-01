"""Deterministic same-study SCAR parser continuation r1.

This executable consumes only the immutable v1 prediction archive, the
immutable official mapping-label pack, the immutable negative formal result,
and the four frozen v2 repair manifests.  It never opens the source archive,
loads a model, regenerates a candidate, calls the v1 scorer, uses the network,
or claims confirmatory authority.

The output is deliberately split into a private, per-item archive and a safe
aggregate.  Both are strict, self-sealed JSON objects.  The old formal result
is an input binding and is never replaced or reinterpreted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy._core import _multiarray_umath as _numpy_multiarray_umath
from numpy.linalg import lapack_lite as _numpy_lapack_lite
from numpy.linalg import _umath_linalg as _numpy_umath_linalg

from assumption_agent import gscl_scar_cssm_repair_contract_v2 as contract
from assumption_agent import gscl_scar_cssm_repair_mechanisms_v2 as mechanisms


VERSION = "gscl_scar_cssm_same_study_repair_parser_continuation_r1_v2"
STUDY_ID = "GSCL_SCAR_CSSM_INTRINSIC_FORMAL_V1"

PREDICTION_SCHEMA = "gscl_scar_cssm_score_v1.prediction_pack.v1"
LABEL_SCHEMA = "gscl_scar_cssm_source_v1.label_pack.v1"
FORMAL_RESULT_SCHEMA = "gscl_scar_cssm_intrinsic_formal_result_v1"
ARM_SPEC_SCHEMA = "gscl_scar_cssm_same_study_repair_arm_spec_v2"
ANALYSIS_SPEC_SCHEMA = (
    "gscl_scar_cssm_same_study_repair_development_analysis_spec_v2"
)
ORACLE_SPEC_SCHEMA = "gscl_scar_cssm_same_study_repair_oracle_diagnostic_v2"
BINDING_SCHEMA = (
    "gscl_scar_cssm_same_study_repair_parser_continuation_r1_binding_v2"
)
AMENDMENT_SCHEMA = (
    "gscl_scar_cssm_same_study_repair_parser_continuation_amendment_r1_v2"
)
AMENDMENT_RELATIVE_PATH = (
    "manifests/"
    "gscl_scar_cssm_same_study_repair_parser_continuation_amendment_r1_v2.json"
)
AMENDMENT_HASH_DOMAIN = (
    "ASSUMPTION_AGENT/GSCL_SCAR_CSSM/"
    "SAME_STUDY_REPAIR_PARSER_CONTINUATION_AMENDMENT_R1/V2"
)
AMENDMENT_VERSION = "v2_parser_continuation_r1"
AMENDMENT_STATUS = "AUTHORIZED_APPEND_ONLY_SAME_STUDY_PARSER_CONTINUATION_R1"
PRIVATE_SCHEMA = f"{VERSION}.private_result.v1"
SAFE_SCHEMA = f"{VERSION}.safe_aggregate.v1"

PRIVATE_FILENAME = "repair_parser_continuation_r1.private.json"
SAFE_FILENAME = "repair_parser_continuation_r1.safe.json"
ATTEMPT_INTENT_FILENAME = "attempt.intent.safe.json"
ATTEMPT_INTENT_SCHEMA = f"{VERSION}.attempt_intent.safe.v1"

PRIMARY_ITEM_COUNT = 362
AMBIGUOUS_ITEM_COUNT = 29
TOTAL_ITEM_COUNT = PRIMARY_ITEM_COUNT + AMBIGUOUS_ITEM_COUNT
FOLD_COUNT = 5
PRIMARY_BOOTSTRAP_SEED = 18_391_702_929_142_623_763
PRIMARY_BOOTSTRAP_REPLICATES = 100_000
ORACLE_BOOTSTRAP_SEED = 20_260_801
ORACLE_BOOTSTRAP_REPLICATES = 10_000
FOLD_DOMAIN = (
    "ASSUMPTION_AGENT/GSCL_SCAR_CSSM/SAME_STUDY_REPAIR_FOLD_ASSIGNMENT/V2"
)
PRIVATE_SEAL_DOMAIN = (
    "ASSUMPTION_AGENT/GSCL_SCAR_CSSM/"
    "SAME_STUDY_REPAIR_PARSER_CONTINUATION_R1_PRIVATE_RESULT/V2"
)
SAFE_SEAL_DOMAIN = (
    "ASSUMPTION_AGENT/GSCL_SCAR_CSSM/"
    "SAME_STUDY_REPAIR_PARSER_CONTINUATION_R1_SAFE_AGGREGATE/V2"
)
ATTEMPT_INTENT_SEAL_DOMAIN = (
    "ASSUMPTION_AGENT/GSCL_SCAR_CSSM/"
    "SAME_STUDY_REPAIR_PARSER_CONTINUATION_R1_ATTEMPT_INTENT/V2"
)

_REQUIRED_RUNTIME_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": (
        "/var/tmp/gscl_scar_cssm_same_study_repair_parser_continuation_r1_v2_"
        "deployment/"
        "reconstruction_v2"
    ),
    "VECLIB_MAXIMUM_THREADS": "1",
}
_EXPECTED_LAUNCH_UNIT_PATH = (
    "/var/tmp/gscl_scar_cssm_same_study_repair_parser_continuation_r1_v2_"
    "deployment/"
    "gscl-scar-cssm-same-study-repair-parser-continuation-r1-v2.service"
)
_FAILED_OUTPUT_ROOT = (
    "/var/tmp/gscl_scar_cssm_intrinsic_formal_20260801/"
    "v1_same_study_repair_v2"
)
_CONTINUATION_OUTPUT_ROOT = (
    "/var/tmp/gscl_scar_cssm_intrinsic_formal_20260801/"
    "v1_same_study_repair_parser_continuation_r1_v2"
)
_PARENT_RUNNER_RELATIVE_PATH = (
    "assumption_agent/benchmarks/gscl_scar_cssm_same_study_repair_development_v2.py"
)
_PARENT_RUNNER_FILE_SHA256 = (
    "cf4cea0b9ee3101726ea5048ba497dab12cd644653dbdbcd0cbb0266eea4c9d9"
)
_PARENT_BINDING_RELATIVE_PATH = (
    "manifests/gscl_scar_cssm_same_study_repair_binding_v2.json"
)
_PARENT_UNIT_RELATIVE_PATH = (
    "manifests/gscl_scar_cssm_same_study_repair_v2.service"
)
_PARENT_ATTEMPT_SCHEMA = (
    "gscl_scar_cssm_same_study_repair_development_v2.attempt_intent.safe.v1"
)
_BLAS_LIBRARY_PREFIXES = (
    "libblas.",
    "libblas-",
    "libblis",
    "liblapack.",
    "liblapack-",
    "libmkl",
    "libopenblas",
    "libscipy_openblas",
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_ITEM_TOKEN = re.compile(r"scar-item-v1-[0-9a-f]{64}\Z")
_SLOT_TOKEN = re.compile(r"scar-slot-v1-[0-9a-f]{64}\Z")
_OPERATOR = re.compile(
    r"ori_(?:keep|inv)\.pol_(?:keep|inv)\.slots_(?:identity|reverse)\Z"
)

_ROOT_PREDICTION_KEYS = frozenset(
    {
        "arm_ids",
        "items",
        "schema",
        "self_sha256",
        "source_action_commitment_sha256",
        "study_id",
        "variant_names",
    }
)
_ROOT_LABEL_KEYS = frozenset(
    {
        "action_commitment_sha256",
        "cross_binding_hmac_sha256",
        "items",
        "label_commitment_sha256",
        "schema",
        "self_sha256",
        "source_sha256",
        "source_size_bytes",
        "study_id",
        "variant_names",
    }
)
_PACK_FINAL_KEYS = frozenset(
    {
        "action_commitment_sha256",
        "cross_binding_hmac_sha256",
        "label_commitment_sha256",
        "self_sha256",
    }
)
_PROPOSAL_KEYS = frozenset(
    {
        "flat_structural_score",
        "injective_verified",
        "length2_composition_verified",
        "length2_path_matched",
        "length2_path_total",
        "operator_id",
        "origins",
        "proposal_hash",
        "semantic_score",
        "target_indices",
        "typed_incidence_matched",
        "typed_incidence_total",
        "typed_incidence_verified",
    }
)
_BINDER_KEYS = frozenset(
    {
        "coverage_disposition",
        "dropped_edge_count",
        "endpoint_count",
        "retained_edge_count",
        "self_loop_count",
        "unbound_count",
        "zero_degree_count",
    }
)
_CHOICE_KEYS = frozenset({"arm", "disposition", "proposal_hash", "reason_ids"})
_ARM_IDS = (
    "semantic_only",
    "flat_structural",
    "full_no_composition",
    "full_with_length2_composition",
    "full_with_length2_composition_target_color_shuffle",
)
_VARIANT_NAMES = ("base", "system_swap")
_EXPECTED_ARITY = {2: 31, 3: 187, 4: 77, 5: 32, 6: 12, 7: 12, 8: 8, 9: 2, 10: 1}
_EXPECTED_DOMAIN_RELATION = {"cross": 254, "intra": 108}
_FEATURE_IDS = (
    "f01_arity_scaled",
    "f02_semantic_score_scaled",
    "f03_semantic_gap_from_S0_scaled",
    "f04_flat_structural_score_per_slot",
    "f05_typed_incidence_match_rate",
    "f06_typed_incidence_total_per_slot",
    "f07_zero_incidence_support",
    "f08_semantic_origin",
    "f09_structure_origin",
    "f10_orientation_inverting",
    "f11_polarity_inverting",
    "f12_positional_slot_reversal",
    "f13_two_side_mean_retained_edges_per_slot",
    "f14_two_side_mean_dropped_fraction",
    "f15_two_side_mean_unbound_endpoint_fraction",
    "f16_two_side_mean_zero_degree_fraction",
)


class SameStudyRepairDevelopmentError(RuntimeError):
    """Stable fail-closed error for archive, binding, or output violations."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


@dataclass(frozen=True, slots=True)
class Candidate:
    proposal_hash: str
    mapping: tuple[tuple[str, str], ...]
    semantic_score: int
    features: tuple[float, ...]
    null_features: tuple[float, ...]
    target_delta: Fraction | None
    exact_against_gold: bool | None


@dataclass(frozen=True, slots=True)
class PrimaryItem:
    item_token: str
    arity: int
    domain_relation: str
    stratum: str
    fold: int
    baseline: tuple[tuple[str, str], ...]
    baseline_swap: tuple[tuple[str, str], ...]
    gold_base: tuple[tuple[str, str], ...]
    gold_swap: tuple[tuple[str, str], ...]
    baseline_f1: Fraction | None
    baseline_exact: bool | None
    candidates: tuple[Candidate, ...]
    common_v1_base: tuple[tuple[str, str], ...]
    common_v1_swap: tuple[tuple[str, str], ...]
    common_v1_answered_base: bool
    common_v1_answered_swap: bool
    common_v1_f1: Fraction | None


@dataclass(frozen=True, slots=True)
class AppliedItem:
    item: PrimaryItem
    threshold: float
    model_commitment: str | None
    selected: Candidate | None
    selected_score: float | None
    output_base: tuple[tuple[str, str], ...]
    output_swap: tuple[tuple[str, str], ...]
    output_f1: Fraction
    oracle: Candidate | None
    oracle_f1: Fraction


def _fail(issue_id: str) -> None:
    raise SameStudyRepairDevelopmentError(issue_id)


def _pairs_hook(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            _fail("SCAR_REPAIR_JSON_DUPLICATE_KEY")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SameStudyRepairDevelopmentError(
            "SCAR_REPAIR_JSON_CANONICALIZATION_INVALID"
        ) from exc


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json_once(path: Path, *, issue_id: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_hook)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc
    if type(value) is not dict:
        _fail(issue_id)
    # This rejects NaN/Infinity and values unsupported by the commitment wire.
    _canonical_bytes(value)
    return value, _file_sha256(raw)


def _require_hex64(value: Any, issue_id: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _fail(issue_id)
    return value


def _validate_legacy_self_seal(value: Mapping[str, Any], issue_id: str) -> str:
    if type(value) is not dict:
        _fail(issue_id)
    claimed = _require_hex64(value.get("self_sha256"), issue_id)
    body = dict(value)
    body.pop("self_sha256")
    if not hmac.compare_digest(claimed, _object_sha256(body)):
        _fail(issue_id)
    return claimed


def _require_exact_keys(value: Any, keys: frozenset[str], issue_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _fail(issue_id)
    return value


def _require_study_schema(
    value: Mapping[str, Any], *, schema: str, issue_id: str
) -> None:
    if value.get("schema") != schema or value.get("study_id") != STUDY_ID:
        _fail(issue_id)


def _validate_manifest(
    path: Path, *, schema: str, issue_id: str
) -> tuple[dict[str, Any], str, str]:
    value, file_hash = _read_json_once(path, issue_id=issue_id)
    _require_study_schema(value, schema=schema, issue_id=issue_id)
    self_hash = _validate_legacy_self_seal(value, issue_id)
    if value.get("authority") != "POSTHOC_DEVELOPMENT_ONLY":
        _fail(issue_id)
    return value, file_hash, self_hash


def _validate_frozen_manifests(
    *,
    arm_spec_path: Path,
    analysis_spec_path: Path,
    oracle_spec_path: Path,
    binding_path: Path,
    formal_result_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    arm, arm_file, arm_self = _validate_manifest(
        arm_spec_path, schema=ARM_SPEC_SCHEMA, issue_id="SCAR_REPAIR_ARM_SPEC_INVALID"
    )
    analysis, analysis_file, analysis_self = _validate_manifest(
        analysis_spec_path,
        schema=ANALYSIS_SPEC_SCHEMA,
        issue_id="SCAR_REPAIR_ANALYSIS_SPEC_INVALID",
    )
    oracle, oracle_file, oracle_self = _validate_manifest(
        oracle_spec_path,
        schema=ORACLE_SPEC_SCHEMA,
        issue_id="SCAR_REPAIR_ORACLE_SPEC_INVALID",
    )
    binding, binding_file, binding_self = _validate_manifest(
        binding_path,
        schema=BINDING_SCHEMA,
        issue_id="SCAR_REPAIR_BINDING_INVALID",
    )
    formal, formal_file = _read_json_once(
        formal_result_path, issue_id="SCAR_REPAIR_FORMAL_RESULT_INVALID"
    )
    _require_study_schema(
        formal, schema=FORMAL_RESULT_SCHEMA, issue_id="SCAR_REPAIR_FORMAL_RESULT_INVALID"
    )
    formal_self = _validate_legacy_self_seal(
        formal, "SCAR_REPAIR_FORMAL_RESULT_INVALID"
    )
    if formal.get("status") != (
        "PROTOCOL_VALID_PRIMARY_FAIL_GENERALIZED_COUNTERPOINT_OPERATIONALIZATION_NEGATIVE"
    ):
        _fail("SCAR_REPAIR_FORMAL_RESULT_INVALID")
    cohort = formal.get("formal_result", {}).get("cohort", {})
    old_primary = formal.get("formal_result", {}).get("primary", {})
    if (
        cohort.get("complete_primary_item_count") != PRIMARY_ITEM_COUNT
        or cohort.get("ambiguous_premodel_typed_failure_count") != AMBIGUOUS_ITEM_COUNT
        or formal.get("interpretation", {}).get("implementation_or_infrastructure_invalid")
        is not False
        or old_primary.get("disposition") != "FAIL"
        or old_primary.get("paired_item_count") != PRIMARY_ITEM_COUNT
        or old_primary.get("comparator_arm") != "semantic_only"
        or old_primary.get("mechanism_arm")
        != "full_with_length2_composition"
    ):
        _fail("SCAR_REPAIR_FORMAL_RESULT_INVALID")

    arm_feature = arm.get("feature_contract")
    candidate_contract = arm.get("candidate_contract")
    u0_contract = arm.get("arm_contract", {}).get("U0_UNION_SEMANTIC_RERANK")
    if (
        type(arm_feature) is not dict
        or arm_feature.get("feature_count") != contract.FEATURE_WIDTH
        or tuple(row.get("id") for row in arm_feature.get("exact_order", []))
        != _FEATURE_IDS
        or type(candidate_contract) is not dict
        or candidate_contract.get("candidate_generation_may_be_rerun") is not False
        or candidate_contract.get("k_or_search_budget_expansion_allowed") is not False
        or candidate_contract.get("base_system_swap_contract", {}).get(
            "variant_specific_independent_selection_allowed"
        )
        is not False
        or type(u0_contract) is not dict
        or u0_contract.get("feature_ids")
        != ["f02_semantic_score_scaled", "f03_semantic_gap_from_S0_scaled"]
        or u0_contract.get("verdict_authority") is not False
    ):
        _fail("SCAR_REPAIR_ARM_SPEC_INVALID")

    population = analysis.get("analysis_population")
    crossfit = analysis.get("crossfit_contract")
    estimator = analysis.get("estimator_contract")
    if (
        type(population) is not dict
        or population.get("complete_primary_item_count") != PRIMARY_ITEM_COUNT
        or population.get("fresh_or_sealed_item_count") != 0
        or type(crossfit) is not dict
        or crossfit.get("fold_assignment", {}).get("fold_count") != FOLD_COUNT
        or type(estimator) is not dict
        or estimator.get("regularization_alpha") != {"denominator": 1, "numerator": 1}
        or crossfit.get("threshold_grid_low_to_high")
        != [
            {"denominator": 1, "numerator": 0},
            {"denominator": 32, "numerator": 1},
            {"denominator": 16, "numerator": 1},
            {"denominator": 8, "numerator": 1},
            {"denominator": 4, "numerator": 1},
            {"kind": "ALL_NOOP"},
        ]
        or crossfit.get("threshold_selection", {}).get("eligibility_floor")
        .get("numerator")
        != 99
        or crossfit.get("threshold_selection", {}).get("eligibility_floor")
        .get("denominator")
        != 100
        or crossfit.get("fold_assignment", {}).get("hash_payload", {}).get(
            "formal_result_self_sha256"
        )
        != formal_self
        or crossfit.get("fold_assignment", {}).get("hash_payload", {}).get(
            "hash_domain"
        )
        != FOLD_DOMAIN
    ):
        _fail("SCAR_REPAIR_ANALYSIS_SPEC_INVALID")
    outcome = analysis.get("outcome_contract", {})
    bootstrap = outcome.get("bootstrap", {})
    if (
        bootstrap.get("replicate_count") != PRIMARY_BOOTSTRAP_REPLICATES
        or bootstrap.get("seed") != PRIMARY_BOOTSTRAP_SEED
        or bootstrap.get("one_sided_lower_bound_zero_based_sorted_index")
        != 4_999
        or outcome.get("minimum_important_difference")
        != {"denominator": 100, "numerator": 1}
    ):
        _fail("SCAR_REPAIR_ANALYSIS_SPEC_INVALID")

    oracle_population = oracle.get("cohort_and_quota_contract", {})
    oracle_bootstrap = oracle.get("mapping_oracle_headroom", {}).get(
        "bootstrap", {}
    )
    if (
        oracle_population.get("complete_primary_item_count") != PRIMARY_ITEM_COUNT
        or oracle_population.get("fresh_sealed_confirmatory_item_count") != 0
        or oracle.get("gold_authority", {}).get("law_family_labels_available")
        is not False
        or oracle_bootstrap.get("item_resamples") != ORACLE_BOOTSTRAP_REPLICATES
        or oracle_bootstrap.get("seed") != ORACLE_BOOTSTRAP_SEED
        or oracle_bootstrap.get("equal_tail_lower_bound_zero_based_sorted_index")
        != 249
    ):
        _fail("SCAR_REPAIR_ORACLE_SPEC_INVALID")

    repair_refs = binding.get("repair_spec_bindings")
    formal_ref = binding.get("formal_lineage_bindings", {}).get("formal_result")
    expected_refs = {
        "arm_spec": (arm_file, arm_self),
        "development_analysis_spec": (analysis_file, analysis_self),
        "oracle_diagnostic_spec": (oracle_file, oracle_self),
    }
    if type(repair_refs) is not dict or type(formal_ref) is not dict:
        _fail("SCAR_REPAIR_BINDING_INVALID")
    for key, (file_hash, self_hash) in expected_refs.items():
        row = repair_refs.get(key)
        if (
            type(row) is not dict
            or row.get("file_sha256") != file_hash
            or row.get("self_sha256") != self_hash
        ):
            _fail("SCAR_REPAIR_BINDING_STALE")
    if (
        formal_ref.get("file_sha256") != formal_file
        or formal_ref.get("self_sha256") != formal_self
        or binding.get("append_only_runtime_contract", {}).get(
            "archived_candidate_generation_extractor_binder_or_neural_model_rerun_allowed"
        )
        is not False
        or binding.get("append_only_runtime_contract", {}).get("network_access_allowed")
        is not False
        or binding.get("claim_contract", {}).get("confirmatory_authority") is not False
    ):
        _fail("SCAR_REPAIR_BINDING_INVALID")
    if analysis.get("arm_binding", {}).get("self_sha256") != arm_self:
        _fail("SCAR_REPAIR_ANALYSIS_SPEC_INVALID")
    if oracle.get("arm_binding", {}).get("self_sha256") != arm_self:
        _fail("SCAR_REPAIR_ORACLE_SPEC_INVALID")

    roots = {
        "arm_spec_self_sha256": arm_self,
        "analysis_spec_self_sha256": analysis_self,
        "binding_file_sha256": binding_file,
        "binding_self_sha256": binding_self,
        "formal_result_self_sha256": formal_self,
        "oracle_spec_self_sha256": oracle_self,
    }
    return binding, roots


def _read_strict_canonical_json_once(
    path: Path, *, issue_id: str
) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs_hook)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc
    if type(value) is not dict or _canonical_bytes(value) != raw:
        _fail(issue_id)
    return value, _file_sha256(raw)


def _validate_parser_continuation_authority(
    amendment_path: Path,
    *,
    binding: Mapping[str, Any],
) -> dict[str, str]:
    issue = "SCAR_REPAIR_PARSER_CONTINUATION_AUTHORITY_INVALID"
    expected_amendment_path = (
        Path(_REQUIRED_RUNTIME_ENVIRONMENT["PYTHONPATH"])
        / AMENDMENT_RELATIVE_PATH
    )
    if str(amendment_path) != str(expected_amendment_path):
        _fail(issue)
    amendment, amendment_file_sha = _read_strict_canonical_json_once(
        amendment_path, issue_id=issue
    )
    amendment_self = _validate_legacy_self_seal(amendment, issue)
    reference = binding.get("parser_continuation_authority_binding")
    if (
        type(reference) is not dict
        or set(reference) != {"file_sha256", "relative_path", "self_sha256"}
        or reference.get("relative_path") != AMENDMENT_RELATIVE_PATH
        or reference.get("file_sha256") != amendment_file_sha
        or reference.get("self_sha256") != amendment_self
    ):
        _fail(issue)
    if (
        set(amendment)
        != {
            "authority",
            "hash_domain",
            "lineage",
            "parent_failed_attempt",
            "parser_amendment",
            "qualification_contract",
            "schema",
            "self_sha256",
            "status",
            "stopping_rule",
            "study_id",
            "version",
        }
        or amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("study_id") != STUDY_ID
        or amendment.get("authority") != "APPEND_ONLY_PARSER_CONTINUATION_ONLY"
        or amendment.get("hash_domain") != AMENDMENT_HASH_DOMAIN
        or amendment.get("status") != AMENDMENT_STATUS
        or amendment.get("version") != AMENDMENT_VERSION
    ):
        _fail(issue)

    lineage = amendment.get("lineage")
    if lineage != {
        "deployment_root": str(Path(_REQUIRED_RUNTIME_ENVIRONMENT["PYTHONPATH"]).parent),
        "failed_output_root": _FAILED_OUTPUT_ROOT,
        "new_cohort": False,
        "new_source": False,
        "new_study": False,
        "old_attempt_immutable": True,
        "output_root": _CONTINUATION_OUTPUT_ROOT,
        "same_study": True,
    }:
        _fail(issue)
    if (
        binding.get("append_only_runtime_contract", {}).get(
            "append_only_output_root"
        )
        != _CONTINUATION_OUTPUT_ROOT
    ):
        _fail(issue)

    parser_amendment = amendment.get("parser_amendment")
    if (
        type(parser_amendment) is not dict
        or set(parser_amendment)
        != {
            "accepted_gold_contract",
            "allowed_change",
            "all_other_frozen_implementation_and_effect_contracts_unchanged",
            "former_invalid_requirement",
            "gold_pair_order_is_semantically_irrelevant",
            "scope",
        }
        or type(parser_amendment.get("accepted_gold_contract")) is not str
        or not parser_amendment["accepted_gold_contract"]
        or parser_amendment.get("allowed_change")
        != "GOLD_PAIR_LIST_ORDER_VALIDATOR_TO_SET_BIJECTION_AND_INVERSE_ONLY"
        or parser_amendment.get(
            "all_other_frozen_implementation_and_effect_contracts_unchanged"
        )
        is not True
        or type(parser_amendment.get("former_invalid_requirement")) is not str
        or not parser_amendment["former_invalid_requirement"]
        or parser_amendment.get("gold_pair_order_is_semantically_irrelevant")
        is not True
        or parser_amendment.get("scope")
        != "gold_base_and_gold_system_swap_only"
    ):
        _fail(issue)

    qualification = amendment.get("qualification_contract")
    if (
        type(qualification) is not dict
        or set(qualification)
        != {
            "access_limits",
            "allowed_checks",
            "construct_effect_targets",
            "forbidden_computations",
            "receipt_contract",
            "status",
        }
        or qualification.get("access_limits")
        != {"label_pack_read_count": 1, "prediction_pack_read_count": 1}
        or qualification.get("allowed_checks")
        != ["private_pack_schema", "slot_graph", "proposal", "null_feature"]
        or qualification.get("construct_effect_targets") is not False
        or qualification.get("forbidden_computations")
        != [
            "pair_f1",
            "target_delta",
            "exact_metric",
            "oracle",
            "fit",
            "threshold_selection",
            "bootstrap",
            "arm_aggregate",
        ]
        or qualification.get("receipt_contract")
        != {
            "allowed_ambiguous_item_count": AMBIGUOUS_ITEM_COUNT,
            "allowed_primary_item_count": PRIMARY_ITEM_COUNT,
            "allowed_statuses": ["PASS", "FAIL"],
            "content_or_identifier_disclosure_allowed": False,
            "other_metric_or_effect_output_allowed": False,
            "pack_access_counts_required": True,
        }
        or qualification.get("status")
        != "PRIVATE_SCHEMA_QUALIFICATION_NOT_EFFECT_MEASUREMENT"
    ):
        _fail(issue)
    if amendment.get("stopping_rule") != {
        "continuation_attempt_limit": 1,
        "continuation_failure_or_completion_is_terminal": True,
        "old_attempt_restart_retry_replay_or_replacement_allowed": False,
        "online_or_api_evaluation_allowed": False,
        "qualification_attempt_limit": 1,
        "qualification_failure_closes_continuation": True,
        "retry_replay_resample_candidate_or_gate_change_allowed": False,
    }:
        _fail(issue)

    parent = amendment.get("parent_failed_attempt")
    if type(parent) is not dict or set(parent) != {
        "binding",
        "effect_execution_counts",
        "error_code",
        "execution",
        "result_artifacts",
        "runner",
        "safe_diagnostic",
        "sentinel",
        "unit",
    }:
        _fail(issue)
    parent_binding = parent.get("binding")
    parent_runner = parent.get("runner")
    parent_unit = parent.get("unit")
    sentinel_ref = parent.get("sentinel")
    if (
        type(parent_binding) is not dict
        or set(parent_binding) != {"file_sha256", "relative_path", "self_sha256"}
        or parent_binding.get("relative_path") != _PARENT_BINDING_RELATIVE_PATH
        or type(parent_runner) is not dict
        or set(parent_runner) != {"file_sha256", "relative_path"}
        or parent_runner
        != {
            "file_sha256": _PARENT_RUNNER_FILE_SHA256,
            "relative_path": _PARENT_RUNNER_RELATIVE_PATH,
        }
        or type(parent_unit) is not dict
        or set(parent_unit) != {"file_sha256", "relative_path"}
        or parent_unit.get("relative_path") != _PARENT_UNIT_RELATIVE_PATH
        or type(sentinel_ref) is not dict
        or set(sentinel_ref) != {"absolute_path", "file_sha256", "self_sha256"}
    ):
        _fail(issue)
    for row, keys in (
        (parent_binding, ("file_sha256", "self_sha256")),
        (parent_unit, ("file_sha256",)),
        (sentinel_ref, ("file_sha256", "self_sha256")),
    ):
        for key in keys:
            _require_hex64(row.get(key), issue)
    expected_sentinel_path = str(Path(_FAILED_OUTPUT_ROOT) / ATTEMPT_INTENT_FILENAME)
    if sentinel_ref.get("absolute_path") != expected_sentinel_path:
        _fail(issue)
    failed_root = Path(_FAILED_OUTPUT_ROOT)
    _require_exact_mode(
        failed_root, expected=0o700, directory=True, issue_id=issue
    )
    _require_exact_mode(
        Path(expected_sentinel_path),
        expected=0o600,
        directory=False,
        issue_id=issue,
    )
    try:
        failed_entries = {row.name for row in failed_root.iterdir()}
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(issue) from exc
    if failed_entries != {ATTEMPT_INTENT_FILENAME}:
        _fail(issue)
    if (
        parent.get("error_code") != "SCAR_REPAIR_GOLD_INVALID"
        or parent.get("execution")
        != {
            "exec_main_status": 2,
            "invocation_id": "d64740d119f242ec97f9997c1aa4c816",
            "nrestarts": 0,
        }
        or parent.get("result_artifacts")
        != {"private_result_exists": False, "safe_result_exists": False}
        or parent.get("effect_execution_counts")
        != {
            "bootstrap_count": 0,
            "effect_fit_count": 0,
            "oof_prediction_count": 0,
            "result_count": 0,
            "score_count": 0,
            "threshold_selection_count": 0,
        }
        or parent.get("safe_diagnostic")
        != {
            "base_and_system_swap_set_bijection_and_inverse_valid_item_count": 362,
            "base_gold_order_exact_item_count": 51,
            "private_item_identifiers_disclosed": False,
            "system_swap_gold_order_exact_item_count": 57,
            "total_primary_item_count": 362,
        }
    ):
        _fail(issue)

    sentinel, sentinel_file_sha = _read_strict_canonical_json_once(
        Path(expected_sentinel_path), issue_id=issue
    )
    sentinel_self = _validate_legacy_self_seal(sentinel, issue)
    if (
        sentinel_file_sha != sentinel_ref.get("file_sha256")
        or sentinel_self != sentinel_ref.get("self_sha256")
        or sentinel.get("schema") != _PARENT_ATTEMPT_SCHEMA
        or sentinel.get("study_id") != STUDY_ID
        or sentinel.get("version")
        != "gscl_scar_cssm_same_study_repair_development_v2"
        or sentinel.get("execution_limit") != 1
        or sentinel.get("content_free_attempt_evidence") is not True
        or sentinel.get("private_input_access_counts_at_claim")
        != {"label_pack": 0, "prediction_pack": 0}
        or sentinel.get("binding_file_sha256")
        != parent_binding.get("file_sha256")
        or sentinel.get("binding_self_sha256")
        != parent_binding.get("self_sha256")
        or sentinel.get("runner_file_sha256") != _PARENT_RUNNER_FILE_SHA256
    ):
        _fail(issue)
    return {
        "parent_failed_attempt_binding_file_sha256": parent_binding["file_sha256"],
        "parent_failed_attempt_binding_self_sha256": parent_binding["self_sha256"],
        "parent_failed_attempt_root_sha256": sentinel_self,
        "parent_failed_attempt_sentinel_file_sha256": sentinel_file_sha,
        "parser_continuation_amendment_file_sha256": amendment_file_sha,
        "parser_continuation_authority_root_sha256": amendment_self,
    }


def _validate_pack_self(value: dict[str, Any], issue_id: str) -> str:
    return _validate_legacy_self_seal(value, issue_id)


def _validate_label_commitment(label_pack: dict[str, Any]) -> None:
    core = {key: value for key, value in label_pack.items() if key not in _PACK_FINAL_KEYS}
    if not hmac.compare_digest(
        _require_hex64(label_pack.get("label_commitment_sha256"), "SCAR_REPAIR_LABEL_PACK_INVALID"),
        _object_sha256(core),
    ):
        _fail("SCAR_REPAIR_LABEL_PACK_INVALID")


def _validate_pack_roots(
    prediction: dict[str, Any], label: dict[str, Any]
) -> tuple[str, str]:
    _require_exact_keys(prediction, _ROOT_PREDICTION_KEYS, "SCAR_REPAIR_PREDICTION_PACK_INVALID")
    _require_exact_keys(label, _ROOT_LABEL_KEYS, "SCAR_REPAIR_LABEL_PACK_INVALID")
    _require_study_schema(
        prediction, schema=PREDICTION_SCHEMA, issue_id="SCAR_REPAIR_PREDICTION_PACK_INVALID"
    )
    _require_study_schema(label, schema=LABEL_SCHEMA, issue_id="SCAR_REPAIR_LABEL_PACK_INVALID")
    prediction_self = _validate_pack_self(prediction, "SCAR_REPAIR_PREDICTION_PACK_INVALID")
    label_self = _validate_pack_self(label, "SCAR_REPAIR_LABEL_PACK_INVALID")
    _validate_label_commitment(label)
    if (
        prediction.get("arm_ids") != list(_ARM_IDS)
        or prediction.get("variant_names") != list(_VARIANT_NAMES)
        or label.get("variant_names") != list(_VARIANT_NAMES)
        or prediction.get("source_action_commitment_sha256")
        != label.get("action_commitment_sha256")
        or type(prediction.get("items")) is not list
        or type(label.get("items")) is not list
        or len(prediction["items"]) != TOTAL_ITEM_COUNT
        or len(label["items"]) != TOTAL_ITEM_COUNT
    ):
        _fail("SCAR_REPAIR_PACK_CROSS_BINDING_INVALID")
    for key in (
        "action_commitment_sha256",
        "cross_binding_hmac_sha256",
        "label_commitment_sha256",
        "source_sha256",
    ):
        _require_hex64(label.get(key), "SCAR_REPAIR_LABEL_PACK_INVALID")
    if type(label.get("source_size_bytes")) is not int or label["source_size_bytes"] <= 0:
        _fail("SCAR_REPAIR_LABEL_PACK_INVALID")
    return prediction_self, label_self


def _hash_runtime_file(path: Path, issue_id: str) -> str:
    try:
        return _file_sha256(path.read_bytes())
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc


def _runtime_file_binding(value: Any, issue_id: str) -> dict[str, str]:
    if type(value) is not str or not value:
        _fail(issue_id)
    path = Path(value)
    if not path.is_absolute():
        _fail(issue_id)
    return {
        "absolute_path": str(path),
        "file_sha256": _hash_runtime_file(path, issue_id),
    }


def _loaded_blas_library_paths(issue_id: str) -> tuple[Path, ...]:
    """Return every currently mapped BLAS/LAPACK implementation library."""

    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc
    paths: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        candidate = Path(fields[5])
        basename = candidate.name.casefold()
        if basename.startswith(_BLAS_LIBRARY_PREFIXES):
            paths.add(candidate)
    if not paths:
        _fail(issue_id)
    return tuple(sorted(paths, key=str))


def _runtime_dependency_snapshot(issue_id: str) -> dict[str, Any]:
    numpy_init = _runtime_file_binding(np.__file__, issue_id)
    multiarray = _runtime_file_binding(_numpy_multiarray_umath.__file__, issue_id)
    lapack_lite = _runtime_file_binding(_numpy_lapack_lite.__file__, issue_id)
    umath_linalg = _runtime_file_binding(_numpy_umath_linalg.__file__, issue_id)
    blas_libraries = [
        _runtime_file_binding(str(path), issue_id)
        for path in _loaded_blas_library_paths(issue_id)
    ]
    return {
        "loaded_blas_shared_libraries": blas_libraries,
        "numpy_lapack_lite": lapack_lite,
        "numpy_multiarray_umath": multiarray,
        "numpy_package_init": numpy_init,
        "numpy_umath_linalg": umath_linalg,
        "numpy_version": np.__version__,
    }


def _validate_static_implementation_closure(
    binding: Mapping[str, Any],
    *,
    prediction_pack_path: Path,
    label_pack_path: Path,
) -> dict[str, Any]:
    issue = "SCAR_REPAIR_IMPLEMENTATION_CLOSURE_INVALID"
    closure = binding.get("implementation_closure_binding")
    expected_keys = {
        "contract_module",
        "execution_authorized",
        "implementation_closure_sha256",
        "input_archive_set_commitment_sha256",
        "input_archives",
        "launch_unit",
        "mechanisms_module",
        "python_executable",
        "runner",
        "runtime_dependencies",
        "runtime_dependency_binding_sha256",
        "runtime_environment",
        "runtime_environment_binding_sha256",
        "status",
    }
    if type(closure) is not dict or set(closure) != expected_keys:
        _fail(issue)
    claimed_closure = _require_hex64(
        closure.get("implementation_closure_sha256"), issue
    )
    closure_body = dict(closure)
    closure_body.pop("implementation_closure_sha256")
    if (
        closure.get("execution_authorized") is not True
        or closure.get("status")
        != "FROZEN_EXACT_IMPLEMENTATION_RUNTIME_AND_INPUT_CLOSURE"
        or not hmac.compare_digest(claimed_closure, contract.content_hash(closure_body))
    ):
        _fail(issue)

    module_rows = (
        (
            "contract_module",
            "assumption_agent/gscl_scar_cssm_repair_contract_v2.py",
            Path(str(contract.__file__)),
        ),
        (
            "mechanisms_module",
            "assumption_agent/gscl_scar_cssm_repair_mechanisms_v2.py",
            Path(str(mechanisms.__file__)),
        ),
        (
            "runner",
            "assumption_agent/benchmarks/"
            "gscl_scar_cssm_same_study_repair_parser_continuation_r1_v2.py",
            Path(__file__),
        ),
    )
    for key, relative_path, actual_path in module_rows:
        row = closure.get(key)
        if (
            type(row) is not dict
            or set(row) != {"file_sha256", "relative_path"}
            or row.get("relative_path") != relative_path
            or row.get("file_sha256") != _hash_runtime_file(actual_path, issue)
        ):
            _fail(issue)

    python_row = closure.get("python_executable")
    if (
        type(python_row) is not dict
        or set(python_row) != {"absolute_path", "file_sha256"}
        or python_row.get("absolute_path") != sys.executable
        or python_row.get("file_sha256")
        != _hash_runtime_file(Path(sys.executable), issue)
    ):
        _fail(issue)
    launch_unit = closure.get("launch_unit")
    if (
        type(launch_unit) is not dict
        or set(launch_unit) != {"absolute_path", "file_sha256"}
        or launch_unit.get("absolute_path") != _EXPECTED_LAUNCH_UNIT_PATH
        or launch_unit
        != _runtime_file_binding(launch_unit.get("absolute_path"), issue)
    ):
        _fail(issue)

    runtime_environment = closure.get("runtime_environment")
    actual_runtime_environment = {
        key: os.environ.get(key) for key in _REQUIRED_RUNTIME_ENVIRONMENT
    }
    if (
        runtime_environment != _REQUIRED_RUNTIME_ENVIRONMENT
        or actual_runtime_environment != _REQUIRED_RUNTIME_ENVIRONMENT
        or closure.get("runtime_environment_binding_sha256")
        != contract.content_hash(runtime_environment)
    ):
        _fail(issue)
    dependencies = closure.get("runtime_dependencies")
    if (
        dependencies != _runtime_dependency_snapshot(issue)
        or closure.get("runtime_dependency_binding_sha256")
        != contract.content_hash(dependencies)
    ):
        _fail(issue)

    input_archives = closure.get("input_archives")
    if type(input_archives) is not dict or set(input_archives) != {
        "label_pack",
        "prediction_pack",
    }:
        _fail(issue)
    for key, path in (
        ("label_pack", label_pack_path),
        ("prediction_pack", prediction_pack_path),
    ):
        row = input_archives.get(key)
        if (
            type(row) is not dict
            or set(row) != {"absolute_path", "file_sha256", "self_sha256"}
            or row.get("absolute_path") != str(path)
        ):
            _fail(issue)
        _require_hex64(row.get("file_sha256"), issue)
        _require_hex64(row.get("self_sha256"), issue)
    if closure.get("input_archive_set_commitment_sha256") != contract.content_hash(
        input_archives
    ):
        _fail(issue)
    return closure


def _validate_input_implementation_closure(
    closure: Mapping[str, Any],
    *,
    prediction_pack_path: Path,
    label_pack_path: Path,
    prediction_file_sha: str,
    label_file_sha: str,
    prediction_self: str,
    label_self: str,
) -> None:
    issue = "SCAR_REPAIR_IMPLEMENTATION_CLOSURE_INVALID"

    input_archives = {
        "label_pack": {
            "absolute_path": str(label_pack_path),
            "file_sha256": label_file_sha,
            "self_sha256": label_self,
        },
        "prediction_pack": {
            "absolute_path": str(prediction_pack_path),
            "file_sha256": prediction_file_sha,
            "self_sha256": prediction_self,
        },
    }
    if (
        closure.get("input_archives") != input_archives
        or closure.get("input_archive_set_commitment_sha256")
        != contract.content_hash(input_archives)
    ):
        _fail(issue)


def _as_pairs(value: Any, *, issue_id: str) -> tuple[tuple[str, str], ...]:
    if type(value) is not list:
        _fail(issue_id)
    rows: list[tuple[str, str]] = []
    for row in value:
        if (
            type(row) is not list
            or len(row) != 2
            or any(type(cell) is not str or not cell for cell in row)
        ):
            _fail(issue_id)
        rows.append((row[0], row[1]))
    if len(rows) != len(set(rows)):
        _fail(issue_id)
    return tuple(rows)


def _archived_arm_pairs(
    value: Any, *, issue_id: str
) -> tuple[tuple[tuple[str, str], ...], bool]:
    if type(value) is not dict or set(value) != {"disposition", "error_code", "pairs"}:
        _fail(issue_id)
    if value["disposition"] == "ANSWER":
        if value["error_code"] is not None:
            _fail(issue_id)
        return _as_pairs(value["pairs"], issue_id=issue_id), True
    if value["disposition"] == "ABSTAIN" and value["pairs"] is None:
        return (), False
    _fail(issue_id)


def _inverse(pairs: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple((right, left) for left, right in pairs)


def _forced_swap_output(
    item: PrimaryItem,
    base_pairs: Sequence[tuple[str, str]],
    *,
    no_op: bool,
) -> tuple[tuple[str, str], ...]:
    if no_op:
        return item.baseline_swap
    inverse = {right: left for left, right in base_pairs}
    order = tuple(right for right, _ in item.baseline_swap)
    if set(inverse) != set(order):
        _fail("SCAR_REPAIR_SWAP_INVERSE_INVALID")
    result = tuple((right, inverse[right]) for right in order)
    if set(result) != set(_inverse(base_pairs)):
        _fail("SCAR_REPAIR_SWAP_INVERSE_INVALID")
    return result


def _validate_full_bijection(
    pairs: Sequence[tuple[str, str]],
    *,
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    issue_id: str,
) -> None:
    if (
        len(pairs) != len(left_ids)
        or tuple(left for left, _ in pairs) != tuple(left_ids)
        or {right for _, right in pairs} != set(right_ids)
    ):
        _fail(issue_id)


def _validate_gold_bijection(
    pairs: Sequence[tuple[str, str]],
    *,
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    issue_id: str,
) -> None:
    """Validate gold as a bijection without imposing prediction wire order."""

    if (
        len(pairs) != len(left_ids)
        or {left for left, _ in pairs} != set(left_ids)
        or {right for _, right in pairs} != set(right_ids)
    ):
        _fail(issue_id)


def _validate_slots(value: Any, *, arity: int, issue_id: str) -> tuple[str, ...]:
    if type(value) is not list or len(value) != arity:
        _fail(issue_id)
    rows: list[str] = []
    for row in value:
        if type(row) is not dict or set(row) != {
            "evidence_binding_sha256",
            "normalized_label_sha256",
            "slot_id",
        }:
            _fail(issue_id)
        slot_id = row.get("slot_id")
        if type(slot_id) is not str or _SLOT_TOKEN.fullmatch(slot_id) is None:
            _fail(issue_id)
        _require_hex64(row.get("evidence_binding_sha256"), issue_id)
        _require_hex64(row.get("normalized_label_sha256"), issue_id)
        rows.append(slot_id)
    if len(rows) != len(set(rows)):
        _fail(issue_id)
    return tuple(rows)


def _validate_proposal(value: Any, *, arity: int) -> dict[str, Any]:
    row = _require_exact_keys(value, _PROPOSAL_KEYS, "SCAR_REPAIR_PROPOSAL_INVALID")
    proposal_hash = _require_hex64(row.get("proposal_hash"), "SCAR_REPAIR_PROPOSAL_INVALID")
    body = dict(row)
    body.pop("proposal_hash")
    if not hmac.compare_digest(proposal_hash, _object_sha256(body)):
        _fail("SCAR_REPAIR_PROPOSAL_INVALID")
    indices = row.get("target_indices")
    origins = row.get("origins")
    if (
        row.get("injective_verified") is not True
        or type(indices) is not list
        or len(indices) != arity
        or any(type(index) is not int for index in indices)
        or sorted(indices) != list(range(arity))
        or type(origins) is not list
        or not origins
        or len(origins) != len(set(origins))
        or not set(origins) <= {"semantic_kbest", "structure_kbest"}
        or type(row.get("operator_id")) is not str
        or _OPERATOR.fullmatch(row["operator_id"]) is None
    ):
        _fail("SCAR_REPAIR_PROPOSAL_INVALID")
    for key in (
        "flat_structural_score",
        "length2_path_matched",
        "length2_path_total",
        "semantic_score",
        "typed_incidence_matched",
        "typed_incidence_total",
    ):
        if type(row.get(key)) is not int:
            _fail("SCAR_REPAIR_PROPOSAL_INVALID")
    if (
        row["typed_incidence_matched"] < 0
        or row["typed_incidence_total"] < row["typed_incidence_matched"]
        or row["length2_path_matched"] < 0
        or row["length2_path_total"] < row["length2_path_matched"]
        or type(row.get("typed_incidence_verified")) is not bool
        or type(row.get("length2_composition_verified")) is not bool
    ):
        _fail("SCAR_REPAIR_PROPOSAL_INVALID")
    return row


def _mapping_from_proposal(
    proposal: Mapping[str, Any], left_ids: Sequence[str], right_ids: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (left_id, right_ids[target_index])
        for left_id, target_index in zip(
            left_ids, proposal["target_indices"], strict=True
        )
    )


def _binder_row(value: Any) -> dict[str, Any]:
    row = _require_exact_keys(value, _BINDER_KEYS, "SCAR_REPAIR_BINDER_INVALID")
    if row.get("coverage_disposition") not in {
        "COMPLETE_SELECTED_SET",
        "PARTIAL_SELECTED_SET",
        "EMPTY_ABSTENTION",
    }:
        _fail("SCAR_REPAIR_BINDER_INVALID")
    for key in _BINDER_KEYS - {"coverage_disposition"}:
        if type(row.get(key)) is not int or row[key] < 0:
            _fail("SCAR_REPAIR_BINDER_INVALID")
    return row


def _proposal_for_choice(mapping_receipt: Any, *, arity: int) -> dict[str, Any]:
    if type(mapping_receipt) is not dict:
        _fail("SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
    choices = mapping_receipt.get("choices")
    proposals = mapping_receipt.get("proposals")
    if type(choices) is not list or type(proposals) is not list:
        _fail("SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
    selected_hash: str | None = None
    for choice in choices:
        row = _require_exact_keys(choice, _CHOICE_KEYS, "SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
        if row.get("arm") == "semantic_only":
            if (
                selected_hash is not None
                or row.get("disposition") != "SELECTED"
                or row.get("reason_ids") != []
            ):
                _fail("SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
            selected_hash = _require_hex64(
                row.get("proposal_hash"), "SCAR_REPAIR_MAPPING_RECEIPT_INVALID"
            )
    if selected_hash is None:
        _fail("SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
    normalized = [_validate_proposal(row, arity=arity) for row in proposals]
    selected = [row for row in normalized if row["proposal_hash"] == selected_hash]
    if len(selected) != 1:
        _fail("SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
    return selected[0]


def _candidate_feature_input(
    *,
    proposal: Mapping[str, Any],
    baseline_semantic_score: int,
    arity: int,
    left_binder: Mapping[str, Any],
    right_binder: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "arity": arity,
        "baseline_semantic_score": baseline_semantic_score,
        "proposal": {
            "selected_operator": proposal["operator_id"],
            "semantic_origin_count": int("semantic_kbest" in proposal["origins"]),
            "structural_origin_count": int("structure_kbest" in proposal["origins"]),
            "incidence_match_count": proposal["typed_incidence_matched"],
            "incidence_total_count": proposal["typed_incidence_total"],
            "length2_path_count": proposal["length2_path_matched"],
            "length2_path_total_count": proposal["length2_path_total"],
            "typed_incidence_verified": proposal["typed_incidence_verified"],
            "length2_composition_verified": proposal[
                "length2_composition_verified"
            ],
            "proposal_hash": proposal["proposal_hash"],
            "semantic_score": proposal["semantic_score"],
            "flat_structural_score": proposal["flat_structural_score"],
        },
        "left_binder": dict(left_binder),
        "right_binder": dict(right_binder),
    }


def _arity_bucket(arity: int) -> str:
    return f"ARITY_{arity}" if arity in {2, 3, 4} else "ARITY_5_PLUS"


def _assign_stratified_folds(
    rows: Sequence[tuple[str, str, int]], formal_result_self_sha256: str
) -> tuple[dict[str, int], dict[str, str]]:
    try:
        assigned_rows = mechanisms.assign_stratified_folds(
            tuple(
                mechanisms.StratifiedFoldRow(
                    canonical_item_id=item_token,
                    domain_relation=f"{domain_relation}_domain",
                    arity=arity,
                )
                for item_token, domain_relation, arity in rows
            ),
            formal_result_self_sha256=formal_result_self_sha256,
        )
    except mechanisms.ScarRepairMechanismError as exc:
        raise SameStudyRepairDevelopmentError(exc.issue_id) from exc
    assigned = {row.canonical_item_id: row.fold_index for row in assigned_rows}
    strata = {row.canonical_item_id: row.stratum for row in assigned_rows}
    if len(assigned) != len(rows) or len(strata) != len(rows):
        _fail("SCAR_REPAIR_FOLD_ASSIGNMENT_INVALID")
    return assigned, strata


def _validate_label_item(value: Any) -> tuple[str, dict[str, Any], dict[str, Any]]:
    row = _require_exact_keys(
        value, frozenset({"gold_pairs", "item_token", "strata"}), "SCAR_REPAIR_LABEL_ITEM_INVALID"
    )
    token = row.get("item_token")
    strata = row.get("strata")
    gold = row.get("gold_pairs")
    if (
        type(token) is not str
        or _ITEM_TOKEN.fullmatch(token) is None
        or type(strata) is not dict
        or set(strata)
        != {"arity", "cohort", "domain_relation", "system_a_domain", "system_b_domain"}
        or type(gold) is not dict
        or tuple(gold) != _VARIANT_NAMES
    ):
        _fail("SCAR_REPAIR_LABEL_ITEM_INVALID")
    return token, strata, gold


def _prediction_index(value: Sequence[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in value:
        if type(item) is not dict or set(item) != {
            "diagnostics",
            "execution",
            "item_token",
            "private_mechanism_receipts",
            "proposal_pools",
            "variants",
        }:
            _fail("SCAR_REPAIR_PREDICTION_ITEM_INVALID")
        token = item.get("item_token")
        if type(token) is not str or _ITEM_TOKEN.fullmatch(token) is None or token in result:
            _fail("SCAR_REPAIR_PREDICTION_ITEM_INVALID")
        result[token] = item
    return result


def _build_primary_items(
    prediction_pack: dict[str, Any],
    label_pack: dict[str, Any],
    *,
    formal_result_self_sha256: str,
    construct_effect_targets: bool,
) -> tuple[PrimaryItem, ...]:
    if type(construct_effect_targets) is not bool:
        _fail("SCAR_REPAIR_EFFECT_TARGET_MODE_INVALID")
    predictions = _prediction_index(prediction_pack["items"])
    labels: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    arity_counts: dict[int, int] = {}
    domain_counts: dict[str, int] = {}
    ambiguous_count = 0
    ambiguous_tokens: list[str] = []
    fold_inputs: list[tuple[str, str, int]] = []
    for raw in label_pack["items"]:
        token, strata, gold = _validate_label_item(raw)
        if token in labels:
            _fail("SCAR_REPAIR_LABEL_ITEM_INVALID")
        labels[token] = (strata, gold)
        cohort = strata.get("cohort")
        if cohort == "ambiguous_secondary":
            ambiguous_count += 1
            ambiguous_tokens.append(token)
            continue
        if cohort != "primary_unique_slot":
            _fail("SCAR_REPAIR_LABEL_COHORT_INVALID")
        arity = strata.get("arity")
        relation = strata.get("domain_relation")
        if type(arity) is not int or arity not in _EXPECTED_ARITY or relation not in {"cross", "intra"}:
            _fail("SCAR_REPAIR_LABEL_COHORT_INVALID")
        arity_counts[arity] = arity_counts.get(arity, 0) + 1
        domain_counts[relation] = domain_counts.get(relation, 0) + 1
        fold_inputs.append((token, relation, arity))
    if (
        set(labels) != set(predictions)
        or len(labels) != TOTAL_ITEM_COUNT
        or ambiguous_count != AMBIGUOUS_ITEM_COUNT
        or arity_counts != _EXPECTED_ARITY
        or domain_counts != _EXPECTED_DOMAIN_RELATION
    ):
        _fail("SCAR_REPAIR_LABEL_COHORT_INVALID")
    for token in ambiguous_tokens:
        archived = predictions[token]
        execution = archived.get("execution")
        receipts = archived.get("private_mechanism_receipts")
        pools = archived.get("proposal_pools")
        if (
            type(execution) is not dict
            or execution.get("structural_status") != "TYPED_FAILURE"
            or execution.get("document_call_count") != 0
            or type(execution.get("error_code")) is not str
            or type(receipts) is not dict
            or receipts.get("availability") != "PREMODEL_TYPED_FAILURE"
            or type(pools) is not dict
            or any(
                pools.get(variant) != {"semantic_kbest": [], "structure_kbest": []}
                for variant in _VARIANT_NAMES
            )
        ):
            _fail("SCAR_REPAIR_AMBIGUOUS_ARCHIVE_INVALID")
    if construct_effect_targets:
        folds, frozen_strata = _assign_stratified_folds(
            fold_inputs, formal_result_self_sha256
        )
    else:
        # Qualification validates the private parser topology only.  These
        # sentinels cannot enter fitting and deliberately avoid even the
        # effect-analysis fold assignment.
        folds = {token: -1 for token, _, _ in fold_inputs}
        frozen_strata = {token: "EFFECT_ANALYSIS_NOT_CONSTRUCTED" for token, _, _ in fold_inputs}

    output: list[PrimaryItem] = []
    for token, relation, arity in sorted(fold_inputs):
        prediction = predictions[token]
        strata, gold_payload = labels[token]
        if (
            prediction.get("execution")
            != {
                "document_call_count": 2,
                "error_code": None,
                "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
            }
            or prediction.get("private_mechanism_receipts", {}).get("availability")
            != "COMPLETE"
        ):
            _fail("SCAR_REPAIR_PRIMARY_ARCHIVE_INCOMPLETE")
        receipts = prediction["private_mechanism_receipts"]
        sides = receipts.get("sides")
        mapping_variants = receipts.get("variants")
        if type(sides) is not dict or set(sides) != {"left", "right"} or type(mapping_variants) is not dict:
            _fail("SCAR_REPAIR_PRIVATE_RECEIPT_INVALID")
        left_ids = _validate_slots(
            sides.get("left", {}).get("slot_graph", {}).get("slots"),
            arity=arity,
            issue_id="SCAR_REPAIR_LEFT_GRAPH_INVALID",
        )
        right_ids = _validate_slots(
            sides.get("right", {}).get("slot_graph", {}).get("slots"),
            arity=arity,
            issue_id="SCAR_REPAIR_RIGHT_GRAPH_INVALID",
        )
        if set(left_ids) & set(right_ids):
            _fail("SCAR_REPAIR_GRAPH_SLOT_COLLISION")
        base_mapping = mapping_variants.get("base")
        if type(base_mapping) is not dict or set(base_mapping) != {
            "semantic_mapping",
            "structural_mapping",
            "target_color_shuffle_mapping",
        }:
            _fail("SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
        s0_proposal = _proposal_for_choice(base_mapping["semantic_mapping"], arity=arity)
        baseline = _mapping_from_proposal(s0_proposal, left_ids, right_ids)
        _validate_full_bijection(
            baseline, left_ids=left_ids, right_ids=right_ids, issue_id="SCAR_REPAIR_S0_INVALID"
        )
        archived_s0 = _as_pairs(
            prediction.get("variants", {}).get("base", {}).get("arms", {}).get("semantic_only", {}).get("pairs"),
            issue_id="SCAR_REPAIR_S0_INVALID",
        )
        archived_swap = _as_pairs(
            prediction.get("variants", {}).get("system_swap", {}).get("arms", {}).get("semantic_only", {}).get("pairs"),
            issue_id="SCAR_REPAIR_S0_INVALID",
        )
        if baseline != archived_s0 or set(archived_swap) != set(_inverse(baseline)):
            _fail("SCAR_REPAIR_S0_INVALID")
        gold_base = _as_pairs(gold_payload["base"], issue_id="SCAR_REPAIR_GOLD_INVALID")
        gold_swap = _as_pairs(gold_payload["system_swap"], issue_id="SCAR_REPAIR_GOLD_INVALID")
        _validate_gold_bijection(
            gold_base, left_ids=left_ids, right_ids=right_ids, issue_id="SCAR_REPAIR_GOLD_INVALID"
        )
        if set(gold_swap) != set(_inverse(gold_base)):
            _fail("SCAR_REPAIR_GOLD_INVALID")
        baseline_f1: Fraction | None = None
        baseline_exact: bool | None = None
        if construct_effect_targets:
            baseline_f1 = contract.pair_f1(baseline, gold_base)
            if contract.pair_f1(archived_swap, gold_swap) != baseline_f1:
                _fail("SCAR_REPAIR_S0_VARIANT_SCORE_MISMATCH")
            baseline_exact = set(baseline) == set(gold_base)

        common_base_arm = (
            prediction.get("variants", {})
            .get("base", {})
            .get("arms", {})
            .get("full_with_length2_composition")
        )
        common_swap_arm = (
            prediction.get("variants", {})
            .get("system_swap", {})
            .get("arms", {})
            .get("full_with_length2_composition")
        )
        common_v1_base, common_answered_base = _archived_arm_pairs(
            common_base_arm, issue_id="SCAR_REPAIR_COMMON_V1_CONTROL_INVALID"
        )
        common_v1_swap, common_answered_swap = _archived_arm_pairs(
            common_swap_arm, issue_id="SCAR_REPAIR_COMMON_V1_CONTROL_INVALID"
        )
        if common_answered_base:
            _validate_full_bijection(
                common_v1_base,
                left_ids=left_ids,
                right_ids=right_ids,
                issue_id="SCAR_REPAIR_COMMON_V1_CONTROL_INVALID",
            )
        if common_answered_swap:
            _validate_full_bijection(
                common_v1_swap,
                left_ids=right_ids,
                right_ids=left_ids,
                issue_id="SCAR_REPAIR_COMMON_V1_CONTROL_INVALID",
            )
        common_v1_f1: Fraction | None = None
        if construct_effect_targets:
            common_v1_f1 = (
                contract.pair_f1(common_v1_base, gold_base)
                + contract.pair_f1(common_v1_swap, gold_swap)
            ) / 2

        diagnostics = prediction.get("diagnostics", {}).get("base")
        if (
            type(diagnostics) is not dict
            or diagnostics.get("structural_diagnostics_available") is not True
        ):
            _fail("SCAR_REPAIR_DIAGNOSTICS_INVALID")
        left_binder = _binder_row(diagnostics.get("left_binder"))
        right_binder = _binder_row(diagnostics.get("right_binder"))
        structural = base_mapping["structural_mapping"]
        proposals = structural.get("proposals") if type(structural) is dict else None
        if type(proposals) is not list or not proposals:
            _fail("SCAR_REPAIR_MAPPING_RECEIPT_INVALID")
        try:
            null_means = {
                row.proposal_hash: row
                for row in mechanisms.build_null_package_mean(
                    token,
                    sides["left"]["slot_graph"],
                    sides["right"]["slot_graph"],
                    proposals,
                )
            }
        except mechanisms.ScarRepairMechanismError as exc:
            raise SameStudyRepairDevelopmentError(exc.issue_id) from exc
        if len(null_means) != len(proposals):
            _fail("SCAR_REPAIR_NULL_PACKAGE_INCOMPLETE")
        seen_hashes: set[str] = set()
        candidates: list[Candidate] = []
        for raw_proposal in proposals:
            proposal = _validate_proposal(raw_proposal, arity=arity)
            proposal_hash = proposal["proposal_hash"]
            if proposal_hash in seen_hashes:
                _fail("SCAR_REPAIR_PROPOSAL_DUPLICATE")
            seen_hashes.add(proposal_hash)
            mapping = _mapping_from_proposal(proposal, left_ids, right_ids)
            if mapping == baseline:
                continue
            feature_input = _candidate_feature_input(
                proposal=proposal,
                baseline_semantic_score=s0_proposal["semantic_score"],
                arity=arity,
                left_binder=left_binder,
                right_binder=right_binder,
            )
            features = contract.extract_archived_features(feature_input)
            null_row = null_means[proposal_hash]
            null_features_list = list(features)
            null_features_list[3:7] = [
                float(null_row.f04_flat_structural_score_per_slot),
                float(null_row.f05_typed_incidence_match_rate),
                float(null_row.f06_typed_incidence_total_per_slot),
                float(null_row.f07_zero_incidence_support),
            ]
            null_features = tuple(null_features_list)
            target_delta: Fraction | None = None
            exact_against_gold: bool | None = None
            if construct_effect_targets:
                assert baseline_f1 is not None
                candidate_f1 = contract.pair_f1(mapping, gold_base)
                target_delta = candidate_f1 - baseline_f1
                exact_against_gold = set(mapping) == set(gold_base)
            candidates.append(
                Candidate(
                    proposal_hash=proposal_hash,
                    mapping=mapping,
                    semantic_score=proposal["semantic_score"],
                    features=features,
                    null_features=null_features,
                    target_delta=target_delta,
                    exact_against_gold=exact_against_gold,
                )
            )
        stratum = frozen_strata[token]
        output.append(
            PrimaryItem(
                item_token=token,
                arity=arity,
                domain_relation=relation,
                stratum=stratum,
                fold=folds[token],
                baseline=baseline,
                baseline_swap=archived_swap,
                gold_base=gold_base,
                gold_swap=gold_swap,
                baseline_f1=baseline_f1,
                baseline_exact=baseline_exact,
                candidates=tuple(candidates),
                common_v1_base=common_v1_base,
                common_v1_swap=common_v1_swap,
                common_v1_answered_base=common_answered_base,
                common_v1_answered_swap=common_answered_swap,
                common_v1_f1=common_v1_f1,
            )
        )
    if len(output) != PRIMARY_ITEM_COUNT:
        _fail("SCAR_REPAIR_PRIMARY_COHORT_INVALID")
    return tuple(output)


def _training_rows(
    items: Iterable[PrimaryItem], *, feature_mode: str
) -> tuple[list[tuple[float, ...]], list[float], list[float]]:
    features: list[tuple[float, ...]] = []
    targets: list[float] = []
    weights: list[float] = []
    for item in items:
        count = len(item.candidates)
        if not count:
            continue
        weight = 1.0 / count
        for candidate in item.candidates:
            if candidate.target_delta is None:
                _fail("SCAR_REPAIR_EFFECT_TARGETS_NOT_CONSTRUCTED")
            features.append(_candidate_features(candidate, feature_mode))
            targets.append(float(candidate.target_delta))
            weights.append(weight)
    return features, targets, weights


def _candidate_features(candidate: Candidate, feature_mode: str) -> tuple[float, ...]:
    if feature_mode == "U1":
        return candidate.features
    if feature_mode == "U0":
        return candidate.features[1:3]
    if feature_mode == "U1_NULL_PACKAGE":
        return candidate.null_features
    _fail("SCAR_REPAIR_FEATURE_MODE_INVALID")


def _fit(
    items: Iterable[PrimaryItem], *, feature_mode: str
) -> contract.StandardizedRidgeModel:
    if feature_mode == "U1_NULL_PACKAGE":
        _fail("SCAR_REPAIR_NULL_REFIT_FORBIDDEN")
    features, targets, weights = _training_rows(items, feature_mode=feature_mode)
    if not features:
        _fail("SCAR_REPAIR_NO_TRAINING_ROWS")
    return contract.fit_standardized_ridge(
        features, targets, sample_weights=weights
    )


def _best_candidate(
    item: PrimaryItem,
    model: contract.StandardizedRidgeModel,
    *,
    feature_mode: str,
) -> tuple[Candidate | None, float | None]:
    scored = [
        (
            candidate,
            float(model.predict(_candidate_features(candidate, feature_mode))),
        )
        for candidate in item.candidates
    ]
    if not scored:
        return None, None
    try:
        ranked = mechanisms.rank_candidates(
            tuple(
                mechanisms.CandidateRankInput(
                    payload=candidate,
                    predicted_delta=score,
                    semantic_score=candidate.semantic_score,
                    proposal_hash=candidate.proposal_hash,
                )
                for candidate, score in scored
            )
        )
    except mechanisms.ScarRepairMechanismError as exc:
        raise SameStudyRepairDevelopmentError(exc.issue_id) from exc
    chosen = ranked[0]
    if not isinstance(chosen.payload, Candidate):
        _fail("SCAR_REPAIR_CANDIDATE_RANK_INVALID")
    return chosen.payload, chosen.predicted_delta


def _threshold_examples(
    predictions: Sequence[tuple[PrimaryItem, Candidate | None, float | None]]
) -> list[contract.ThresholdExample]:
    rows: list[contract.ThresholdExample] = []
    for item, candidate, score in predictions:
        rows.append(
            contract.ThresholdExample(
                selector_score=-1 if score is None else score,
                utility_delta=Fraction(0) if candidate is None else candidate.target_delta,
                old_success_count=int(item.baseline_exact),
                override_preserved_count=(
                    int(item.baseline_exact)
                    if candidate is None
                    else int(item.baseline_exact and candidate.exact_against_gold)
                ),
            )
        )
    return rows


def _select_threshold(
    predictions: Sequence[tuple[PrimaryItem, Candidate | None, float | None]]
) -> float:
    selection = contract.select_override_threshold(
        _threshold_examples(predictions),
        thresholds=contract.THRESHOLD_GRID,
        minimum_preservation=Fraction(99, 100),
    )
    if not isinstance(selection, contract.ThresholdSelection):
        _fail("SCAR_REPAIR_THRESHOLD_RESULT_INVALID")
    return float(selection.threshold)


def _oracle(item: PrimaryItem) -> tuple[Candidate | None, Fraction]:
    best: Candidate | None = None
    best_f1 = item.baseline_f1
    for candidate in item.candidates:
        score = item.baseline_f1 + candidate.target_delta
        if score > best_f1 or (
            score == best_f1
            and best is not None
            and candidate.proposal_hash < best.proposal_hash
        ):
            best = candidate
            best_f1 = score
    # The frozen tie rule prefers S0 over every equal-scoring alternative.
    if best_f1 == item.baseline_f1:
        return None, item.baseline_f1
    return best, best_f1


def _nested_crossfit(
    items: Sequence[PrimaryItem],
    *,
    feature_mode: str,
) -> tuple[
    tuple[AppliedItem, ...],
    list[dict[str, Any]],
    list[contract.StandardizedRidgeModel | None],
    int,
]:
    applied: list[AppliedItem] = []
    fold_receipts: list[dict[str, Any]] = []
    outer_models: list[contract.StandardizedRidgeModel | None] = []
    failure_count = 0
    for outer_fold in range(FOLD_COUNT):
        outer_train = [row for row in items if row.fold != outer_fold]
        outer_test = [row for row in items if row.fold == outer_fold]
        inner_predictions: list[tuple[PrimaryItem, Candidate | None, float | None]] = []
        inner_commitments: list[str] = []
        failure: str | None = None
        try:
            for inner_fold in range(FOLD_COUNT):
                if inner_fold == outer_fold:
                    continue
                inner_train = [
                    row for row in items if row.fold not in {outer_fold, inner_fold}
                ]
                inner_test = [row for row in items if row.fold == inner_fold]
                inner_model = _fit(inner_train, feature_mode=feature_mode)
                inner_commitments.append(inner_model.commitment)
                for row in inner_test:
                    candidate, score = _best_candidate(
                        row, inner_model, feature_mode=feature_mode
                    )
                    inner_predictions.append((row, candidate, score))
            if len(inner_predictions) != len(outer_train):
                _fail("SCAR_REPAIR_INNER_OOF_INCOMPLETE")
            threshold = _select_threshold(inner_predictions)
            outer_model = _fit(outer_train, feature_mode=feature_mode)
        except (contract.ScarRepairContractError, SameStudyRepairDevelopmentError) as exc:
            failure_count += 1
            failure = getattr(exc, "issue_id", type(exc).__name__)
            threshold = math.inf
            outer_model = None
        outer_models.append(outer_model)

        for row in outer_test:
            candidate: Candidate | None = None
            score: float | None = None
            if outer_model is not None:
                candidate, score = _best_candidate(
                    row, outer_model, feature_mode=feature_mode
                )
            selected = (
                candidate
                if candidate is not None
                and score is not None
                and math.isfinite(threshold)
                and score > threshold
                else None
            )
            output_base = row.baseline if selected is None else selected.mapping
            output_swap = _forced_swap_output(
                row, output_base, no_op=selected is None
            )
            if selected is None and (
                output_base != row.baseline or output_swap != row.baseline_swap
            ):
                _fail("SCAR_REPAIR_BYTE_EXACT_NO_OP_INVALID")
            output_f1 = (
                contract.pair_f1(output_base, row.gold_base)
                + contract.pair_f1(output_swap, row.gold_swap)
            ) / 2
            oracle, oracle_f1 = _oracle(row)
            applied.append(
                AppliedItem(
                    item=row,
                    threshold=threshold,
                    model_commitment=(None if outer_model is None else outer_model.commitment),
                    selected=selected,
                    selected_score=score,
                    output_base=output_base,
                    output_swap=output_swap,
                    output_f1=output_f1,
                    oracle=oracle,
                    oracle_f1=oracle_f1,
                )
            )
        fold_receipts.append(
            {
                "failure_issue_id": failure,
                "feature_mode": feature_mode,
                "inner_model_commitments": inner_commitments,
                "outer_fold": outer_fold,
                "outer_model_commitment": (
                    None if outer_model is None else outer_model.commitment
                ),
                "test_item_count": len(outer_test),
                "threshold": _threshold_wire(threshold),
                "training_item_count": len(outer_train),
            }
        )
    if len(applied) != len(items) or len({row.item.item_token for row in applied}) != len(items):
        _fail("SCAR_REPAIR_OUTER_OOF_INCOMPLETE")
    return tuple(sorted(applied, key=lambda row: row.item.item_token)), fold_receipts, outer_models, failure_count


def _full_data_artifact(
    items: Sequence[PrimaryItem],
    outer_models: Sequence[contract.StandardizedRidgeModel | None],
    *,
    feature_mode: str,
) -> dict[str, Any]:
    if len(outer_models) != FOLD_COUNT or any(model is None for model in outer_models):
        return {
            "failure_issue_id": "SCAR_REPAIR_OUTER_FIT_FAILURE_PROPAGATED",
            "feature_mode": feature_mode,
            "status": "ALL_NOOP",
            "threshold": {"kind": "ALL_NOOP"},
        }
    predictions: list[tuple[PrimaryItem, Candidate | None, float | None]] = []
    for fold, model in enumerate(outer_models):
        assert model is not None
        for row in items:
            if row.fold == fold:
                candidate, score = _best_candidate(
                    row, model, feature_mode=feature_mode
                )
                predictions.append((row, candidate, score))
    try:
        threshold = _select_threshold(predictions)
        model = _fit(items, feature_mode=feature_mode)
    except (contract.ScarRepairContractError, SameStudyRepairDevelopmentError) as exc:
        return {
            "failure_issue_id": getattr(exc, "issue_id", type(exc).__name__),
            "feature_mode": feature_mode,
            "status": "ALL_NOOP",
            "threshold": {"kind": "ALL_NOOP"},
        }
    return {
        "model": model.payload(),
        "model_commitment": model.commitment,
        "feature_mode": feature_mode,
        "status": "POSTHOC_DEVELOPMENT_ARTIFACT_ONLY",
        "threshold": _threshold_wire(threshold),
    }


def _apply_frozen_models_without_refit(
    reference: Sequence[AppliedItem],
    outer_models: Sequence[contract.StandardizedRidgeModel | None],
    *,
    feature_mode: str,
) -> tuple[AppliedItem, ...]:
    """Apply U1's fitted objects to another feature view without retuning."""

    if feature_mode != "U1_NULL_PACKAGE" or len(outer_models) != FOLD_COUNT:
        _fail("SCAR_REPAIR_NULL_APPLICATION_INVALID")
    output: list[AppliedItem] = []
    for reference_row in reference:
        item = reference_row.item
        model = outer_models[item.fold]
        candidate: Candidate | None = None
        score: float | None = None
        if model is not None:
            candidate, score = _best_candidate(
                item, model, feature_mode=feature_mode
            )
        selected = (
            candidate
            if candidate is not None
            and score is not None
            and math.isfinite(reference_row.threshold)
            and score > reference_row.threshold
            else None
        )
        output_base = item.baseline if selected is None else selected.mapping
        output_swap = _forced_swap_output(
            item, output_base, no_op=selected is None
        )
        if selected is None and (
            output_base != item.baseline or output_swap != item.baseline_swap
        ):
            _fail("SCAR_REPAIR_BYTE_EXACT_NO_OP_INVALID")
        output_f1 = (
            contract.pair_f1(output_base, item.gold_base)
            + contract.pair_f1(output_swap, item.gold_swap)
        ) / 2
        oracle, oracle_f1 = _oracle(item)
        output.append(
            AppliedItem(
                item=item,
                threshold=reference_row.threshold,
                model_commitment=(None if model is None else model.commitment),
                selected=selected,
                selected_score=score,
                output_base=output_base,
                output_swap=output_swap,
                output_f1=output_f1,
                oracle=oracle,
                oracle_f1=oracle_f1,
            )
        )
    return tuple(sorted(output, key=lambda row: row.item.item_token))


def _fraction_wire(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _float_wire(value: float | None) -> str | None:
    if value is None:
        return None
    if not math.isfinite(value):
        _fail("SCAR_REPAIR_NONFINITE_OUTPUT")
    return value.hex()


def _threshold_wire(value: float) -> dict[str, Any]:
    return {"kind": "ALL_NOOP"} if math.isinf(value) else {"float64_hex": value.hex()}


def _bootstrap_wire(result: contract.PairedBootstrapResult) -> dict[str, Any]:
    return {
        "lower_quantile_zero_based_index": result.lower_quantile_zero_based_index,
        "observed_mean_delta": _fraction_wire(result.observed_mean_delta),
        "one_sided_lower_bound": _fraction_wire(result.one_sided_lower_bound),
        "replicate_count": result.replicate_count,
        "seed": result.seed,
    }


def _mean(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def _seal(domain: str, body: dict[str, Any]) -> dict[str, Any]:
    if body.get("hash_domain") != domain:
        _fail("SCAR_REPAIR_OUTPUT_HASH_DOMAIN_INVALID")
    sealed = contract.seal_payload(body)
    if not isinstance(sealed, dict):
        _fail("SCAR_REPAIR_OUTPUT_SEAL_INVALID")
    return sealed


def _result_payloads(
    *,
    applied: Sequence[AppliedItem],
    null_applied: Sequence[AppliedItem],
    u0_applied: Sequence[AppliedItem],
    u0_failure_count: int,
    fold_receipts: Sequence[dict[str, Any]],
    full_data_artifact: dict[str, Any],
    roots: Mapping[str, str],
    attempt_binding: Mapping[str, str],
    prediction_self: str,
    label_self: str,
    prediction_file_sha: str,
    label_file_sha: str,
    failure_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        tuple(row.item.item_token for row in applied)
        != tuple(row.item.item_token for row in null_applied)
        or tuple(row.item.item_token for row in applied)
        != tuple(row.item.item_token for row in u0_applied)
    ):
        _fail("SCAR_REPAIR_CONTROL_ITEM_ORDER_INVALID")
    s0_values = [row.item.baseline_f1 for row in applied]
    u1_values = [row.output_f1 for row in applied]
    null_values = [row.output_f1 for row in null_applied]
    u0_values = [row.output_f1 for row in u0_applied]
    common_v1_values = [row.item.common_v1_f1 for row in applied]
    oracle_values = [row.oracle_f1 for row in applied]
    primary_bootstrap = contract.paired_bootstrap_mean_delta(
        u1_values,
        s0_values,
        seed=PRIMARY_BOOTSTRAP_SEED,
        replicate_count=PRIMARY_BOOTSTRAP_REPLICATES,
        alpha=Fraction(1, 20),
    )
    oracle_bootstrap = contract.paired_bootstrap_mean_delta(
        oracle_values,
        s0_values,
        seed=ORACLE_BOOTSTRAP_SEED,
        replicate_count=ORACLE_BOOTSTRAP_REPLICATES,
        alpha=Fraction(1, 40),
    )
    preservation = contract.old_success_preservation(
        [row.item.baseline for row in applied]
        + [row.item.baseline_swap for row in applied],
        [row.output_base for row in applied] + [row.output_swap for row in applied],
        [row.item.gold_base for row in applied] + [row.item.gold_swap for row in applied],
    )
    implementation_valid = failure_count == 0
    verdict = contract.decide_repair_development_verdict(
        implementation_valid=implementation_valid,
        old_success_preservation=preservation.fraction,
        minimum_old_success_preservation=Fraction(98, 100),
        primary_ci_lower_bound=primary_bootstrap.one_sided_lower_bound,
        minimum_practically_important_gain=Fraction(1, 100),
    )
    override_count = sum(row.selected is not None for row in applied)
    deltas = [successor - baseline for successor, baseline in zip(u1_values, s0_values, strict=True)]
    positive_count = sum(row > 0 for row in deltas)
    negative_count = sum(row < 0 for row in deltas)
    zero_count = len(deltas) - positive_count - negative_count
    baseline_exact_count = sum(row.item.baseline_exact for row in applied)
    preserved_exact_count = sum(
        row.item.baseline_exact and set(row.output_base) == set(row.item.gold_base)
        for row in applied
    )

    private_records = []
    for row, null_row, u0_row in zip(
        applied, null_applied, u0_applied, strict=True
    ):
        private_records.append(
            {
                "arity": row.item.arity,
                "baseline_base_pairs": [list(pair) for pair in row.item.baseline],
                "baseline_system_swap_pairs": [
                    list(pair) for pair in row.item.baseline_swap
                ],
                "baseline_f1": _fraction_wire(row.item.baseline_f1),
                "candidate_count": len(row.item.candidates),
                "domain_relation": row.item.domain_relation,
                "fold": row.item.fold,
                "gold_base_pairs": [list(pair) for pair in row.item.gold_base],
                "gold_system_swap_pairs": [list(pair) for pair in row.item.gold_swap],
                "item_token": row.item.item_token,
                "model_commitment": row.model_commitment,
                "common_input_v1_answered_base": row.item.common_v1_answered_base,
                "common_input_v1_answered_system_swap": (
                    row.item.common_v1_answered_swap
                ),
                "common_input_v1_base_pairs": [
                    list(pair) for pair in row.item.common_v1_base
                ],
                "common_input_v1_f1": _fraction_wire(row.item.common_v1_f1),
                "common_input_v1_system_swap_pairs": [
                    list(pair) for pair in row.item.common_v1_swap
                ],
                "null_output_base_pairs": [
                    list(pair) for pair in null_row.output_base
                ],
                "null_output_f1": _fraction_wire(null_row.output_f1),
                "null_output_system_swap_pairs": [
                    list(pair) for pair in null_row.output_swap
                ],
                "null_selected_proposal_hash": (
                    None
                    if null_row.selected is None
                    else null_row.selected.proposal_hash
                ),
                "null_selected_score_float64_hex": _float_wire(
                    null_row.selected_score
                ),
                "oracle_f1": _fraction_wire(row.oracle_f1),
                "oracle_proposal_hash": (
                    None if row.oracle is None else row.oracle.proposal_hash
                ),
                "output_base_pairs": [list(pair) for pair in row.output_base],
                "output_f1": _fraction_wire(row.output_f1),
                "output_system_swap_pairs": [list(pair) for pair in row.output_swap],
                "selected_proposal_hash": (
                    None if row.selected is None else row.selected.proposal_hash
                ),
                "selected_score_float64_hex": _float_wire(row.selected_score),
                "stratum": row.item.stratum,
                "threshold": _threshold_wire(row.threshold),
                "U0_output_base_pairs": [
                    list(pair) for pair in u0_row.output_base
                ],
                "U0_output_f1": _fraction_wire(u0_row.output_f1),
                "U0_output_system_swap_pairs": [
                    list(pair) for pair in u0_row.output_swap
                ],
                "U0_selected_proposal_hash": (
                    None if u0_row.selected is None else u0_row.selected.proposal_hash
                ),
                "U0_selected_score_float64_hex": _float_wire(
                    u0_row.selected_score
                ),
                "U0_threshold": _threshold_wire(u0_row.threshold),
            }
        )
    common = {
        "access_counts": {
            "api": 0,
            "bound_manifest": 4,
            "formal_result_manifest": 1,
            "implementation_module_file": 3,
            "label_pack": 1,
            "model": 0,
            "network": 0,
            "online_evaluator": 0,
            "parent_failed_attempt_intent": 1,
            "parser_continuation_amendment": 1,
            "prediction_pack": 1,
            "python_executable_file": 1,
            "scorer": 0,
            "source": 0,
        },
        "authority": "POSTHOC_DEVELOPMENT_ONLY",
        "attempt_intent_binding": dict(attempt_binding),
        "binding_roots": dict(roots),
        "formal_negative_result_changed": False,
        "input_commitments": {
            "label_pack_file_sha256": label_file_sha,
            "label_pack_self_sha256": label_self,
            "prediction_pack_file_sha256": prediction_file_sha,
            "prediction_pack_self_sha256": prediction_self,
        },
        "study_id": STUDY_ID,
        "version": VERSION,
    }
    private_body = {
        **common,
        "fold_receipts": list(fold_receipts),
        "full_data_development_artifact": full_data_artifact,
        "hash_domain": PRIVATE_SEAL_DOMAIN,
        "primary_bootstrap": _bootstrap_wire(primary_bootstrap),
        "oracle_bootstrap": _bootstrap_wire(oracle_bootstrap),
        "records": private_records,
        "schema": PRIVATE_SCHEMA,
        "status": verdict,
    }
    private = _seal(PRIVATE_SEAL_DOMAIN, private_body)

    safe_body = {
        **common,
        "aggregates": {
            "S0_mean_item_pair_f1": _fraction_wire(_mean(s0_values)),
            "U1_mean_item_pair_f1": _fraction_wire(_mean(u1_values)),
            "U1_NULL_PACKAGE_mean_item_pair_f1": _fraction_wire(
                _mean(null_values)
            ),
            "U1_minus_U1_NULL_PACKAGE_mean_item_pair_f1": _fraction_wire(
                _mean(
                    [
                        actual - null
                        for actual, null in zip(
                            u1_values, null_values, strict=True
                        )
                    ]
                )
            ),
            "U1_NULL_PACKAGE_override_count": sum(
                row.selected is not None for row in null_applied
            ),
            "U0_mean_item_pair_f1": _fraction_wire(_mean(u0_values)),
            "U0_failure_count": u0_failure_count,
            "U0_override_count": sum(
                row.selected is not None for row in u0_applied
            ),
            "COMMON_INPUT_V1_HARD_SELECTOR_mean_item_pair_f1": (
                _fraction_wire(_mean(common_v1_values))
            ),
            "COMMON_INPUT_V1_HARD_SELECTOR_answered_base_count": sum(
                row.item.common_v1_answered_base for row in applied
            ),
            "COMMON_INPUT_V1_HARD_SELECTOR_answered_system_swap_count": sum(
                row.item.common_v1_answered_swap for row in applied
            ),
            "base_system_swap_consistency_count": len(applied),
            "byte_exact_no_op_count": len(applied) - override_count,
            "failure_count": failure_count,
            "mapping_oracle_mean_item_pair_f1": _fraction_wire(_mean(oracle_values)),
            "mapping_oracle_mean_headroom": _fraction_wire(
                _mean([oracle - baseline for oracle, baseline in zip(oracle_values, s0_values, strict=True)])
            ),
            "negative_switch_count": negative_count,
            "old_success_pair_count": preservation.old_success_count,
            "old_success_preservation": _fraction_wire(preservation.fraction),
            "override_count": override_count,
            "paired_item_count": len(applied),
            "positive_switch_count": positive_count,
            "S0_strict_exact_item_count": baseline_exact_count,
            "S0_strict_exact_preserved_item_count": preserved_exact_count,
            "zero_switch_count": zero_count,
        },
        "claim_limit": (
            "same_consumed_cohort_posthoc_crossfit_selector_and_fixed_pool_oracle_headroom_only"
        ),
        "hash_domain": SAFE_SEAL_DOMAIN,
        "mapping_oracle_bootstrap": _bootstrap_wire(oracle_bootstrap),
        "primary_bootstrap": _bootstrap_wire(primary_bootstrap),
        "private_item_slot_pair_or_proposal_identifiers_disclosed": False,
        "schema": SAFE_SCHEMA,
        "status": verdict,
    }
    safe = _seal(SAFE_SEAL_DOMAIN, safe_body)
    return private, safe


def _fsync_directory(path: Path, issue_id: str) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(descriptor)
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_exact_mode(
    path: Path, *, expected: int, directory: bool, issue_id: str
) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(issue_id) from exc
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != expected:
        _fail(issue_id)


def _claim_single_attempt(
    path: Path,
    *,
    roots: Mapping[str, str],
    implementation_closure: Mapping[str, Any],
) -> dict[str, str]:
    """Atomically consume the sole execution before any private pack read."""

    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise SameStudyRepairDevelopmentError(
            "SCAR_REPAIR_OUTPUT_ROOT_ALREADY_CLAIMED"
        ) from exc
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(
            "SCAR_REPAIR_OUTPUT_ROOT_CLAIM_FAILED"
        ) from exc
    try:
        os.chmod(path, 0o700, follow_symlinks=False)
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(
            "SCAR_REPAIR_OUTPUT_ROOT_CLAIM_FAILED"
        ) from exc
    _require_exact_mode(
        path,
        expected=0o700,
        directory=True,
        issue_id="SCAR_REPAIR_OUTPUT_ROOT_CLAIM_FAILED",
    )
    _fsync_directory(path.parent, "SCAR_REPAIR_OUTPUT_ROOT_CLAIM_FAILED")

    body = {
        "authority": "POSTHOC_DEVELOPMENT_ONLY",
        "binding_file_sha256": roots["binding_file_sha256"],
        "binding_self_sha256": roots["binding_self_sha256"],
        "content_free_attempt_evidence": True,
        "execution_limit": 1,
        "hash_domain": ATTEMPT_INTENT_SEAL_DOMAIN,
        "implementation_closure_sha256": implementation_closure[
            "implementation_closure_sha256"
        ],
        "input_archive_set_commitment_sha256": implementation_closure[
            "input_archive_set_commitment_sha256"
        ],
        "private_input_access_counts_at_claim": {
            "label_pack": 0,
            "prediction_pack": 0,
        },
        "runner_file_sha256": implementation_closure["runner"]["file_sha256"],
        "schema": ATTEMPT_INTENT_SCHEMA,
        "status": "SINGLE_APPEND_ONLY_ATTEMPT_CLAIMED_BEFORE_PRIVATE_INPUT",
        "study_id": STUDY_ID,
        "version": VERSION,
    }
    intent = _seal(ATTEMPT_INTENT_SEAL_DOMAIN, body)
    wire = _canonical_bytes(intent)
    intent_path = path / ATTEMPT_INTENT_FILENAME
    descriptor: int | None = None
    try:
        descriptor = os.open(
            intent_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(wire)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path, "SCAR_REPAIR_ATTEMPT_INTENT_WRITE_FAILED")
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        # Never remove the claimed root or any partial intent: their presence
        # is the permanent, fail-closed evidence that the sole attempt began.
        raise SameStudyRepairDevelopmentError(
            "SCAR_REPAIR_ATTEMPT_INTENT_WRITE_FAILED"
        ) from exc
    _require_exact_mode(
        intent_path,
        expected=0o600,
        directory=False,
        issue_id="SCAR_REPAIR_ATTEMPT_INTENT_WRITE_FAILED",
    )
    return {
        "file_sha256": _file_sha256(wire),
        "filename": ATTEMPT_INTENT_FILENAME,
        "self_sha256": intent["self_sha256"],
    }


def _validate_claimed_root_for_results(
    path: Path, attempt_binding: Mapping[str, str]
) -> None:
    issue = "SCAR_REPAIR_ATTEMPT_INTENT_CHANGED"
    _require_exact_mode(path, expected=0o700, directory=True, issue_id=issue)
    _require_exact_mode(
        path / ATTEMPT_INTENT_FILENAME,
        expected=0o600,
        directory=False,
        issue_id=issue,
    )
    try:
        entries = {row.name for row in path.iterdir()}
        raw = (path / ATTEMPT_INTENT_FILENAME).read_bytes()
    except OSError as exc:
        raise SameStudyRepairDevelopmentError(issue) from exc
    if (
        entries != {ATTEMPT_INTENT_FILENAME}
        or attempt_binding.get("filename") != ATTEMPT_INTENT_FILENAME
        or attempt_binding.get("file_sha256") != _file_sha256(raw)
    ):
        _fail(issue)
    try:
        parsed = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs_hook)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SameStudyRepairDevelopmentError(issue) from exc
    if (
        type(parsed) is not dict
        or _canonical_bytes(parsed) != raw
        or parsed.get("self_sha256") != attempt_binding.get("self_sha256")
    ):
        _fail(issue)
    try:
        validated = contract.validate_self_seal(
            parsed, expected_schema=ATTEMPT_INTENT_SCHEMA
        )
    except contract.ScarRepairContractError as exc:
        raise SameStudyRepairDevelopmentError(issue) from exc
    if not isinstance(validated, Mapping):
        _fail(issue)


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> str:
    wire = _canonical_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(wire)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise SameStudyRepairDevelopmentError("SCAR_REPAIR_OUTPUT_WRITE_FAILED") from exc
    return _file_sha256(wire)


def _validate_bound_private_input_paths(
    binding: Mapping[str, Any],
    *,
    prediction_pack_path: Path,
    label_pack_path: Path,
) -> None:
    old_roots = binding["old_remote_roots_read_only"]
    expected_prediction = (
        Path(old_roots["private_result_archive_root"])
        / "control"
        / "prediction_pack.private.json"
    )
    if str(prediction_pack_path) != str(expected_prediction):
        _fail("SCAR_REPAIR_PREDICTION_PATH_BINDING_MISMATCH")
    if str(label_pack_path) != old_roots["prepared_label_pack"]:
        _fail("SCAR_REPAIR_LABEL_PATH_BINDING_MISMATCH")


def qualify_private_input_schema_only(
    *,
    prediction_pack_path: Path,
    label_pack_path: Path,
    formal_result_path: Path,
    arm_spec_path: Path,
    analysis_spec_path: Path,
    oracle_spec_path: Path,
    binding_path: Path,
    continuation_amendment_path: Path,
    _access_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Qualify the corrected parser path without consuming a formal attempt.

    This reuses the parser with effect-target construction disabled, giving
    maximum schema and cross-binding coverage without calling pair-F1 or
    constructing target deltas.  It neither fits nor selects a model,
    thresholds, bootstraps, writes an output, creates an attempt root, nor
    discloses any effect value or per-item content.
    """

    access_counts = (
        _access_counts
        if _access_counts is not None
        else {
            "bootstrap": 0,
            "fit": 0,
            "label_pack": 0,
            "metric_aggregate": 0,
            "oracle": 0,
            "output_write": 0,
            "prediction_pack": 0,
            "score": 0,
            "threshold_selection": 0,
        }
    )
    binding, roots = _validate_frozen_manifests(
        arm_spec_path=arm_spec_path,
        analysis_spec_path=analysis_spec_path,
        oracle_spec_path=oracle_spec_path,
        binding_path=binding_path,
        formal_result_path=formal_result_path,
    )
    amendment_roots = _validate_parser_continuation_authority(
        continuation_amendment_path,
        binding=binding,
    )
    roots.update(amendment_roots)
    _validate_bound_private_input_paths(
        binding,
        prediction_pack_path=prediction_pack_path,
        label_pack_path=label_pack_path,
    )
    implementation_closure = _validate_static_implementation_closure(
        binding,
        prediction_pack_path=prediction_pack_path,
        label_pack_path=label_pack_path,
    )
    access_counts["prediction_pack"] += 1
    prediction, prediction_file_sha = _read_json_once(
        prediction_pack_path, issue_id="SCAR_REPAIR_PREDICTION_PACK_INVALID"
    )
    access_counts["label_pack"] += 1
    label, label_file_sha = _read_json_once(
        label_pack_path, issue_id="SCAR_REPAIR_LABEL_PACK_INVALID"
    )
    prediction_self, label_self = _validate_pack_roots(prediction, label)
    _validate_input_implementation_closure(
        implementation_closure,
        prediction_pack_path=prediction_pack_path,
        label_pack_path=label_pack_path,
        prediction_file_sha=prediction_file_sha,
        label_file_sha=label_file_sha,
        prediction_self=prediction_self,
        label_self=label_self,
    )
    items = _build_primary_items(
        prediction,
        label,
        formal_result_self_sha256=roots["formal_result_self_sha256"],
        construct_effect_targets=False,
    )
    if len(items) != PRIMARY_ITEM_COUNT:
        _fail("SCAR_REPAIR_PRIMARY_COHORT_INVALID")
    return {
        "access_counts": dict(access_counts),
        "ambiguous_item_count": AMBIGUOUS_ITEM_COUNT,
        "attempt_created": False,
        "effect_aggregate_disclosed": False,
        "formal_verdict_authority": False,
        "internal_target_construction_occurred": False,
        "primary_item_count": len(items),
        "qualification_kind": (
            "PRIVATE_SCHEMA_QUALIFICATION_NOT_EFFECT_MEASUREMENT"
        ),
        "status": "PASS",
        "study_id": STUDY_ID,
        "version": VERSION,
    }


def run_same_study_repair_parser_continuation_r1_v2(
    *,
    prediction_pack_path: Path,
    label_pack_path: Path,
    formal_result_path: Path,
    arm_spec_path: Path,
    analysis_spec_path: Path,
    oracle_spec_path: Path,
    binding_path: Path,
    continuation_amendment_path: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run the sole append-only parser continuation over archived inputs."""

    binding, roots = _validate_frozen_manifests(
        arm_spec_path=arm_spec_path,
        analysis_spec_path=analysis_spec_path,
        oracle_spec_path=oracle_spec_path,
        binding_path=binding_path,
        formal_result_path=formal_result_path,
    )
    amendment_roots = _validate_parser_continuation_authority(
        continuation_amendment_path,
        binding=binding,
    )
    roots.update(amendment_roots)
    runtime = binding["append_only_runtime_contract"]
    declared_output = Path(runtime["append_only_output_root"])
    resolved_output = declared_output if output_root is None else output_root
    if str(resolved_output) != str(declared_output):
        _fail("SCAR_REPAIR_OUTPUT_ROOT_BINDING_MISMATCH")
    _validate_bound_private_input_paths(
        binding,
        prediction_pack_path=prediction_pack_path,
        label_pack_path=label_pack_path,
    )

    implementation_closure = _validate_static_implementation_closure(
        binding,
        prediction_pack_path=prediction_pack_path,
        label_pack_path=label_pack_path,
    )
    attempt_binding = _claim_single_attempt(
        resolved_output,
        roots=roots,
        implementation_closure=implementation_closure,
    )

    prediction, prediction_file_sha = _read_json_once(
        prediction_pack_path, issue_id="SCAR_REPAIR_PREDICTION_PACK_INVALID"
    )
    label, label_file_sha = _read_json_once(
        label_pack_path, issue_id="SCAR_REPAIR_LABEL_PACK_INVALID"
    )
    prediction_self, label_self = _validate_pack_roots(prediction, label)
    _validate_input_implementation_closure(
        implementation_closure,
        prediction_pack_path=prediction_pack_path,
        label_pack_path=label_pack_path,
        prediction_file_sha=prediction_file_sha,
        label_file_sha=label_file_sha,
        prediction_self=prediction_self,
        label_self=label_self,
    )
    items = _build_primary_items(
        prediction,
        label,
        formal_result_self_sha256=roots["formal_result_self_sha256"],
        construct_effect_targets=True,
    )
    applied, fold_receipts, outer_models, failure_count = _nested_crossfit(
        items, feature_mode="U1"
    )
    null_applied = _apply_frozen_models_without_refit(
        applied, outer_models, feature_mode="U1_NULL_PACKAGE"
    )
    u0_applied, u0_fold_receipts, u0_models, u0_failure_count = _nested_crossfit(
        items, feature_mode="U0"
    )
    full_artifact = _full_data_artifact(
        items, outer_models, feature_mode="U1"
    )
    u0_full_artifact = _full_data_artifact(
        items, u0_models, feature_mode="U0"
    )
    full_artifact = {
        "U0_descriptive_secondary": u0_full_artifact,
        "U0_fold_receipts": u0_fold_receipts,
        "U1_primary": full_artifact,
    }
    private, safe = _result_payloads(
        applied=applied,
        null_applied=null_applied,
        u0_applied=u0_applied,
        u0_failure_count=u0_failure_count,
        fold_receipts=fold_receipts,
        full_data_artifact=full_artifact,
        roots=roots,
        attempt_binding=attempt_binding,
        prediction_self=prediction_self,
        label_self=label_self,
        prediction_file_sha=prediction_file_sha,
        label_file_sha=label_file_sha,
        failure_count=failure_count,
    )
    _validate_claimed_root_for_results(resolved_output, attempt_binding)
    private_path = resolved_output / PRIVATE_FILENAME
    private_file_sha = _atomic_write_new(private_path, private)
    safe["private_result_binding"] = {
        "file_sha256": private_file_sha,
        "filename": PRIVATE_FILENAME,
        "self_sha256": private["self_sha256"],
    }
    # Adding the private binding changes the safe body, so seal it exactly once
    # after the append-only private file exists.
    safe_body = dict(safe)
    safe_body.pop("self_sha256")
    safe = _seal(SAFE_SEAL_DOMAIN, safe_body)
    safe_path = resolved_output / SAFE_FILENAME
    safe_file_sha = _atomic_write_new(safe_path, safe)
    return {
        "private_file_sha256": private_file_sha,
        "private_path": str(private_path),
        "safe_file_sha256": safe_file_sha,
        "safe_path": str(safe_path),
        "status": safe["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-pack", type=Path, required=True)
    parser.add_argument("--label-pack", type=Path, required=True)
    parser.add_argument("--formal-result", type=Path, required=True)
    parser.add_argument("--arm-spec", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--oracle-spec", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--continuation-amendment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--qualify-private-schema-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    qualification_access_counts = {
        "bootstrap": 0,
        "fit": 0,
        "label_pack": 0,
        "metric_aggregate": 0,
        "oracle": 0,
        "output_write": 0,
        "prediction_pack": 0,
        "score": 0,
        "threshold_selection": 0,
    }
    try:
        if args.qualify_private_schema_only:
            if args.output_root is not None:
                _fail("SCAR_REPAIR_QUALIFICATION_OUTPUT_ROOT_FORBIDDEN")
            result = qualify_private_input_schema_only(
                prediction_pack_path=args.prediction_pack,
                label_pack_path=args.label_pack,
                formal_result_path=args.formal_result,
                arm_spec_path=args.arm_spec,
                analysis_spec_path=args.analysis_spec,
                oracle_spec_path=args.oracle_spec,
                binding_path=args.binding,
                continuation_amendment_path=args.continuation_amendment,
                _access_counts=qualification_access_counts,
            )
        else:
            result = run_same_study_repair_parser_continuation_r1_v2(
                prediction_pack_path=args.prediction_pack,
                label_pack_path=args.label_pack,
                formal_result_path=args.formal_result,
                arm_spec_path=args.arm_spec,
                analysis_spec_path=args.analysis_spec,
                oracle_spec_path=args.oracle_spec,
                binding_path=args.binding,
                continuation_amendment_path=args.continuation_amendment,
                output_root=args.output_root,
            )
    except (SameStudyRepairDevelopmentError, contract.ScarRepairContractError) as exc:
        if args.qualify_private_schema_only:
            print(
                _canonical_bytes(
                    {
                        "access_counts": qualification_access_counts,
                        "issue_id": getattr(exc, "issue_id", type(exc).__name__),
                        "qualification_kind": (
                            "PRIVATE_SCHEMA_QUALIFICATION_NOT_EFFECT_MEASUREMENT"
                        ),
                        "status": "FAIL",
                        "study_id": STUDY_ID,
                        "version": VERSION,
                    }
                ).decode("ascii")
            )
            return 2
        print(getattr(exc, "issue_id", type(exc).__name__), file=sys.stderr)
        return 2
    print(_canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SameStudyRepairDevelopmentError",
    "main",
    "qualify_private_input_schema_only",
    "run_same_study_repair_parser_continuation_r1_v2",
]
