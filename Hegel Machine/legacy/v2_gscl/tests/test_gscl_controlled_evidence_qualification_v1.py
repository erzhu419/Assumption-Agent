from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from assumption_agent import gscl_baselines_v1 as baselines
from assumption_agent.benchmarks import (
    gscl_controlled_evidence_qualification_v1 as qualification,
)
from assumption_agent.benchmarks.gscl_controlled_evidence_corpus_v1 import (
    build_controlled_roots,
    render_controlled_views,
)
from assumption_agent.generalized_structural_correspondence_v1 import (
    build_gscl_schema_registry_v1,
    strict_content_hash,
)
from assumption_agent.gscl_evidence_extractor_v1 import (
    bind_structural_episode,
    extract_structural_episode,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return qualification.run_controlled_evidence_qualification()


def test_controlled_qualification_passes_without_claiming_effect_or_public_freeze(
    receipt: dict[str, Any],
) -> None:
    assert receipt["status"] == "PASS_CONTROLLED_EVIDENCE_PATH"
    assert receipt["new_study"] is False
    assert receipt["formal_result"] is False
    assert receipt["efficacy_evidence"] is False
    assert receipt["public_intrinsic_measurement"] is False
    assert receipt["controlled_implementation_ready"] is True
    assert receipt["source_qualification_ready"] is True
    assert (
        receipt["source_qualification_scope"]
        == "controlled_synthetic_atomic_fact_corpus_only"
    )
    assert receipt["public_source_qualified"] is False
    assert receipt["runtime_access_audited"] is False
    assert receipt["public_intrinsic_freeze_ready"] is False
    assert receipt["issue_count"] == 0
    assert receipt["issue_ids"] == []


def test_case_composition_operator_receipts_and_serialization_units_are_exact(
    receipt: dict[str, Any],
) -> None:
    construction = receipt["construction_diagnostics"]
    assert construction["case_count"] == 25
    assert construction["primary_case_count"] == 10
    assert construction["paired_negative_case_count"] == 10
    assert construction["paired_missingness_control_count"] == 5
    assert construction["paired_law_group_count"] == 5
    assert construction["operator_receipt_count"] == 15
    assert construction["valid_operator_receipt_count"] == 15
    assert construction["serialization_view_count"] == 100
    assert construction["serialization_views_per_case"] == 4
    assert (
        construction[
            "serialization_views_are_not_independent_cases"
        ]
        is True
    )
    assert construction["natural_language_paraphrase_claimed"] is False
    assert (
        construction["atomic_fact_overlap_is_descriptive_only"]
        is True
    )
    overlap = construction["atomic_fact_overlap_microunits"]
    assert overlap["primary_pair_groups"]["count"] == 5
    assert overlap["paired_counterfactual_negatives"]["count"] == 10
    assert overlap["paired_missingness_controls"]["count"] == 5


def test_controlled_full_lane_and_correspondence_are_exact(
    receipt: dict[str, Any],
) -> None:
    lane = receipt["full_lane"]
    assert lane["case_count"] == 25
    assert lane["primary_case_total"] == 10
    assert lane["paired_negative_case_total"] == 10
    assert lane["paired_missingness_control_total"] == 5
    assert lane["paired_law_group_count"] == 5
    assert lane["serialization_view_count"] == 100
    assert lane["extractor_success_serialization_views"] == 100
    assert lane["unique_binding_serialization_views"] == 100
    assert (
        lane[
            "required_value_missing_bound_unknown_serialization_views"
        ]
        == 4
    )
    assert lane["exact_denotation_serialization_views"] == 100
    assert lane["primary_cases_accepted"] == 10
    assert lane["paired_negative_cases_rejected"] == 10
    assert lane["paired_missingness_controls_abstained"] == 5
    assert lane["serialization_denotation_invariant_cases"] == 25
    assert lane["serialization_disposition_invariant_cases"] == 25
    assert lane["same_process_byte_exact_replay"] is True

    correspondence = receipt["correspondence_diagnostics"]
    assert correspondence["paired_law_group_count"] == 5
    assert (
        correspondence["primary_correspondence_groups_accepted"]
        == 5
    )
    assert correspondence["primary_correspondence_group_total"] == 5
    assert (
        correspondence["paired_negative_correspondences_rejected"]
        == 10
    )
    assert (
        correspondence["paired_negative_correspondence_total"]
        == 10
    )
    assert correspondence["atomic_fact_overlap_is_not_a_gate"] is True


def test_baselines_use_lossless_common_bytes_and_remain_descriptive(
    receipt: dict[str, Any],
) -> None:
    baselines = receipt["baseline_diagnostics"]
    roots = build_controlled_roots()
    views = render_controlled_views(roots)
    expected_common_input_commitment = strict_content_hash(
        [
            {
                "source_sha256": hashlib.sha256(
                    view.source_bytes
                ).hexdigest(),
                "source_size": len(view.source_bytes),
            }
            for view in sorted(views, key=lambda row: row.view_id)
        ]
    )
    assert baselines["common_input_serialization_view_count"] == 100
    assert baselines["lossless_utf8_roundtrip_count"] == 100
    assert baselines["common_input_hash_algorithm"] == "sha256_raw_bytes"
    assert (
        baselines["common_input_commitment"]
        == expected_common_input_commitment
    )
    assert baselines["baseline_performance_is_qualification_gate"] is False
    assert baselines["baseline_performance_is_effect_gate"] is False
    assert (
        baselines[
            "descriptive_performance_values_are_not_acceptance_checks"
        ]
        is True
    )

    semantic = baselines["semantic_only"]
    assert semantic["actual_chunk_batch_replay_exact"] is True
    assert semantic["maximum_sequence_length"] == 256
    assert semantic["source_text_count"] == 100
    assert semantic["source_texts_requiring_chunking"] == 100
    assert semantic["source_chunk_count"] > 100
    assert 0 < semantic["maximum_chunk_token_count"] <= 256
    assert semantic["truncated_chunk_count"] == 0
    assert semantic["full_token_coverage"] is True
    assert len(semantic["chunk_plan_commitment"]) == 64
    assert semantic["threshold_tuned"] is False

    legacy = baselines["legacy_keyword"]
    assert legacy["mapped_top1_accuracy"]["total"] == 100
    assert (
        0
        <= legacy["mapped_top1_accuracy"]["correct"]
        <= legacy["mapped_top1_accuracy"]["total"]
    )
    assert 0 <= legacy["mapped_top1_prediction_count"] <= 100
    assert (
        0
        <= legacy["accepted_accuracy"]["correct"]
        <= legacy["accepted_accuracy"]["total"]
        <= 100
    )
    assert legacy["compatible_law_family_coverage"]["total"] == 5
    assert (
        0
        <= legacy["compatible_law_family_coverage"]["covered"]
        <= legacy["compatible_law_family_coverage"]["total"]
    )
    assert len(legacy["prediction_commitment"]) == 64
    assert legacy["same_process_replay_exact"] is True
    assert legacy["actual_recomputation_verified"] is True
    assert legacy["actual_prediction_compute_count"] == 200
    assert legacy["expected_prediction_compute_count"] == 200
    assert legacy["new_markers_added"] is False

    flat = baselines["flat_label_no_verifier"]
    assert flat["independent_extractor_binder_reexecution"] is True
    assert flat["paired_negative_serialization_view_total"] == 40
    assert (
        0
        <= flat["paired_negative_false_accepts"]
        <= flat["paired_negative_serialization_view_total"]
    )
    assert flat["paired_missingness_serialization_view_total"] == 20
    assert (
        0
        <= flat["paired_missingness_false_accepts"]
        <= flat["paired_missingness_serialization_view_total"]
    )
    assert len(flat["prediction_commitment"]) == 64
    assert (
        flat["residual_or_hard_negative_verifier_called"]
        is False
    )
    closure = baselines["implementation_closure"]
    assert closure["verified"] is True
    assert baselines["implementation_closure_stable_during_run"] is True
    assert len(closure["closure_hash"]) == 64


def test_baseline_implementation_closure_binds_transitive_live_origins(
) -> None:
    closure, issues = (
        baselines.build_baseline_implementation_closure()
    )
    assert issues == ()
    assert closure["verified"] is True
    assert closure["issue_count"] == 0
    body = dict(closure)
    closure_hash = body.pop("closure_hash")
    assert strict_content_hash(body) == closure_hash

    required_modules = {
        "assumption_agent.gscl_baselines_v1",
        "replication_runtime.qasper_minilm_v1.binding",
        "assumption_os.structural_patterns",
        "assumption_os.graph_memory",
        "assumption_os.formal_mapping",
    }
    module_rows = {
        row["module_name"]: row for row in closure["modules"]
    }
    assert required_modules <= set(module_rows)
    workspace = Path(__file__).resolve().parents[2]
    for row in module_rows.values():
        assert (
            row["actual_import_origin"]
            == row["expected_import_origin"]
        )
        path = workspace / row["actual_import_origin"]
        assert path.is_file()
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == row["implementation_sha256"]
        )
        assert path.stat().st_size == row["implementation_size"]
    assert set(closure["lanes"]) == {
        "legacy_keyword",
        "semantic_only",
    }
    assert all(
        len(row["closure_hash"]) == 64
        for row in closure["lanes"].values()
    )


def test_legacy_replay_recomputes_every_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = render_controlled_views(build_controlled_roots())[:3]
    item_texts = {
        view.view_id: view.source_bytes.decode("utf-8")
        for view in views
    }
    calls = {"search": 0}
    original = baselines.search_structural_patterns

    def counted_search(*args: object, **kwargs: object) -> object:
        calls["search"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        baselines, "search_structural_patterns", counted_search
    )
    execution_audit: dict[str, int] = {}
    first = baselines.run_legacy_keyword_baseline(
        item_texts, execution_audit=execution_audit
    )
    second = baselines.run_legacy_keyword_baseline(
        item_texts, execution_audit=execution_audit
    )
    assert calls["search"] == 2 * len(item_texts)
    assert (
        execution_audit["prediction_compute_calls"]
        == 2 * len(item_texts)
    )
    assert [row.safe_payload() for row in first] == [
        row.safe_payload() for row in second
    ]


def test_flat_lane_independently_reexecutes_without_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = build_controlled_roots()
    views = render_controlled_views(roots)
    registry = build_gscl_schema_registry_v1(
        build_universal_assumption_ontology_v1()
    )
    calls = {"extractor": 0, "binder": 0, "residual": 0}

    def forbidden_residual(*args: object, **kwargs: object) -> None:
        calls["residual"] += 1
        raise AssertionError("flat lane called evaluate_bound_law")

    def counted_extractor(
        source_bytes: bytes,
        media_type: str,
        *,
        registry: object,
    ) -> object:
        calls["extractor"] += 1
        return extract_structural_episode(
            source_bytes, media_type, registry=registry
        )

    def counted_binder(
        registry: object, extraction: object
    ) -> object:
        calls["binder"] += 1
        return bind_structural_episode(registry, extraction)

    monkeypatch.setattr(
        qualification, "evaluate_bound_law", forbidden_residual
    )
    flat, issues = qualification.run_flat_label_no_verifier_lane(
        roots=roots,
        views=views,
        registry=registry,
        extractor_fn=counted_extractor,
        binder_fn=counted_binder,
    )
    assert issues == ()
    assert calls == {
        "extractor": 100,
        "binder": 100,
        "residual": 0,
    }
    assert flat["family_total_serialization_views"] == 100
    assert (
        0
        <= flat["accepted_serialization_views"]
        <= flat["family_total_serialization_views"]
    )
    assert len(flat["prediction_commitment"]) == 64


def _walk_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_walk_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_walk_mapping_keys(item))
    return keys


def test_safe_receipt_has_allowlisted_aggregates_and_no_raw_ngram_leak(
    receipt: dict[str, Any],
) -> None:
    expected_top_level = {
        "qualification_version",
        "qualification_contract_hash",
        "qualification_scope",
        "status",
        "new_study",
        "formal_result",
        "efficacy_evidence",
        "public_intrinsic_measurement",
        "controlled_implementation_ready",
        "source_qualification_ready",
        "source_qualification_scope",
        "public_source_qualified",
        "public_intrinsic_freeze_ready",
        "public_intrinsic_freeze_blocker",
        "extractor_contract_hash",
        "extractor_implementation_sha256",
        "baseline_contract_hash",
        "corpus_version",
        "raw_pack_hash",
        "gold_pack_hash",
        "ontology_hash",
        "registry_hash",
        "construction_diagnostics",
        "full_lane",
        "correspondence_diagnostics",
        "baseline_diagnostics",
        "declared_capability_surface",
        "runtime_access_audited",
        "issue_count",
        "issue_ids",
        "issue_commitment",
        "self_hash",
    }
    assert set(receipt) == expected_top_level
    forbidden_detail_keys = {
        "view_id",
        "root_id",
        "item_key",
        "source_sha256",
        "predicted_law_id",
        "gold_law_id",
        "law_id",
        "rows",
        "law_rows",
        "predictions",
        "pair_similarities",
        "source_bytes",
        "per_item_score",
    }
    assert not (
        _walk_mapping_keys(receipt) & forbidden_detail_keys
    )

    body = dict(receipt)
    self_hash = body.pop("self_hash")
    assert strict_content_hash(body) == self_hash
    encoded = json.dumps(
        receipt, ensure_ascii=True, sort_keys=True
    )
    for view in render_controlled_views(build_controlled_roots()):
        for line in view.source_bytes.decode("utf-8").splitlines():
            for start in range(0, max(1, len(line) - 31), 11):
                fragment = line[start : start + 32]
                if len(fragment) == 32:
                    assert fragment not in encoded
