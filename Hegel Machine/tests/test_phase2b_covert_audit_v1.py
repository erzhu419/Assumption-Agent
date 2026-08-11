from __future__ import annotations

from dataclasses import replace
from inspect import signature

import pytest

import hegel_machine.phase2b_covert_audit_v1 as covert
from hegel_machine.phase2b_covert_audit_v1 import (
    AnswerabilityLabel,
    AuditorLabels,
    AuditorPermutationStrata,
    CovertAuditDisposition,
    DEFAULT_COVERT_AUDIT_POLICY,
    EnvelopeAuditRow,
    InvariantDecisionRow,
    InvarianceAuditBatch,
    InvarianceKind,
    RecognizerInvariantReceipt,
    SEMANTICS_ID,
    extract_frozen_byte_features,
    run_frozen_envelope_covert_audit_mechanics,
    validate_frozen_covert_audit_structure,
)
from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze


def _row(index: int, fill: int = 0) -> EnvelopeAuditRow:
    labels = AuditorLabels(
        "F01",
        "binding-a",
        "root",
        AnswerabilityLabel.ANSWERABLE,
        "joint-a",
        "main",
    )
    strata = AuditorPermutationStrata(
        ("main", "root"),
        ("main", "F01"),
        ("main", "F01", "root"),
        ("F01", "root"),
        ("main",),
    )
    return EnvelopeAuditRow(index, bytes([fill]) * 65_536, labels, strata)


def _decision(index: int, residual: bytes = b"residual") -> InvariantDecisionRow:
    return InvariantDecisionRow(
        index,
        "SELECT",
        "F01",
        (("lhs", "entity-a"),),
        ("root",),
        residual,
    )


def _batch(count: int = 3) -> InvarianceAuditBatch:
    canonical = tuple(_decision(index) for index in range(count))
    baseline = RecognizerInvariantReceipt(InvarianceKind.BASELINE, 0, canonical)
    renaming = tuple(
        RecognizerInvariantReceipt(InvarianceKind.GLOBAL_RENAMING, index, canonical)
        for index in range(32)
    )
    case_order = tuple(
        RecognizerInvariantReceipt(
            InvarianceKind.CASE_ORDER,
            index,
            canonical[index % count :] + canonical[: index % count],
        )
        for index in range(32)
    )
    observation = tuple(
        RecognizerInvariantReceipt(InvarianceKind.OBSERVATION_ORDER, index, canonical)
        for index in range(16)
    )
    return InvarianceAuditBatch(baseline, renaming, case_order, observation)


def _statistical_rows() -> tuple[EnvelopeAuditRow, ...]:
    rows = []
    index = 0
    for family_index in range(2):
        for scale_index in range(2):
            for replicate in range(2):
                family = f"F0{family_index + 1}"
                scale = ("root", "child")[scale_index]
                labels = AuditorLabels(
                    family,
                    f"binding-{replicate}",
                    scale,
                    (
                        AnswerabilityLabel.ANSWERABLE,
                        AnswerabilityLabel.ABSTAIN,
                    )[replicate],
                    f"joint-{replicate}",
                    "main",
                )
                strata = AuditorPermutationStrata(
                    ("main", scale),
                    ("main", family),
                    ("main", family, scale),
                    (family, scale),
                    ("main",),
                )
                envelope = bytearray(65_536)
                envelope[0] = family_index
                envelope[-1] = (scale_index << 1) | replicate
                rows.append(EnvelopeAuditRow(index, bytes(envelope), labels, strata))
                index += 1
    return tuple(rows)


@pytest.fixture(scope="module")
def complete_mechanics_receipt():
    rows = _statistical_rows()
    return rows, run_frozen_envelope_covert_audit_mechanics(
        rows,
        _batch(len(rows)),
    )


def test_policy_is_bound_to_exact_freeze_and_api_has_no_caller_policy_seed_or_root():
    frozen = frozen_phase2b_exact_freeze().covert_channel_audit
    policy = DEFAULT_COVERT_AUDIT_POLICY
    assert policy.envelope_bytes == frozen.envelope_bytes == 65_536
    assert policy.label_permutations == frozen.label_permutations == 10_000
    assert policy.unique_id_feature_family == frozen.unique_id_feature_family == (
        "128_individual_bits",
        "first_8_16_32_bits",
        "last_8_16_32_bits",
        "hamming_weight",
        "integer_mod_3_5_7_11_13",
        "hex_character_histogram",
    )
    assert policy.global_renamings == 32
    assert tuple(signature(validate_frozen_covert_audit_structure).parameters) == (
        "rows",
        "invariance",
    )
    assert tuple(signature(run_frozen_envelope_covert_audit_mechanics).parameters) == (
        "rows",
        "invariance",
    )
    with pytest.raises(ValueError, match="policy drift"):
        replace(
            policy,
            unique_id_feature_family=policy.unique_id_feature_family[:-1],
        )


def test_structural_validation_returns_only_non_authoritative_mechanics_receipt():
    rows = tuple(_row(index, index) for index in range(3))
    receipt = validate_frozen_covert_audit_structure(rows, _batch())
    assert receipt.row_count == 3
    assert receipt.feature_count_per_row > 20
    assert len(receipt.invariant_receipt_content_ids) == 1 + 32 + 32 + 16
    assert receipt.exact_invariant_comparisons == 3 * (32 + 32 + 16) * 5
    assert receipt.claim_level == "NON_AUTHORITATIVE_MECHANICS_ONLY"
    assert receipt.formal_audit_evidence is False
    assert receipt.sealed_holdout_eligible is False
    assert receipt.statistics_executed is False


def test_envelope_and_input_container_types_fail_closed():
    with pytest.raises(TypeError, match="exact bytes"):
        EnvelopeAuditRow(0, bytearray(65_536), _row(0).labels, _row(0).strata)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly 65,536"):
        replace(_row(0), envelope=b"short")
    with pytest.raises(TypeError, match="exact tuple"):
        validate_frozen_covert_audit_structure([_row(0)], _batch(1))  # type: ignore[arg-type]


def test_rows_require_contiguous_ordered_all_row_coverage():
    with pytest.raises(ValueError, match="contiguous, ordered"):
        validate_frozen_covert_audit_structure((_row(0), _row(2)), _batch(2))


def test_strata_must_match_auditor_labels_and_frozen_target_table():
    row = _row(0)
    with pytest.raises(ValueError, match="frozen target table"):
        replace(
            row,
            strata=replace(row.strata, family=("main", "wrong-scale")),
        )
    with pytest.raises(ValueError, match="frozen target table"):
        replace(row, labels=replace(row.labels, case_type="caller-rewritten"))
    with pytest.raises(ValueError, match="frozen target table"):
        replace(
            row,
            strata=replace(row.strata, joint_class=("caller-rewritten",)),
        )
    with pytest.raises(TypeError, match="binary enum"):
        replace(row, labels=replace(row.labels, answerable="answerable"))  # type: ignore[arg-type]


def test_feature_extraction_recomputes_from_full_fixed_envelope_deterministically():
    zero = extract_frozen_byte_features(_row(0, 0))
    ones = extract_frozen_byte_features(_row(0, 255))
    assert zero == extract_frozen_byte_features(_row(0, 0))
    assert dict(zero.values)["envelope_hamming_weight"] == 0
    assert dict(ones.values)["envelope_hamming_weight"] == 65_536 * 8
    assert dict(ones.values)["envelope_xor_u8"] == 0


def test_feature_extractor_implements_every_frozen_unique_id_feature_family_member():
    features = dict(extract_frozen_byte_features(_row(0, 0x80)).values)
    assert len(features) == 324
    assert features["prefix_128_individual_bit_000"] == 1
    assert features["prefix_128_individual_bit_001"] == 0
    assert features["suffix_128_individual_bit_127"] == 0
    assert features["prefix_128_first_8_bits"] == 0x80
    assert features["prefix_128_first_16_bits"] == 0x8080
    assert features["prefix_128_last_32_bits"] == 0x80808080
    assert features["prefix_128_hamming_weight"] == 16
    assert features["prefix_128_integer_mod_13"] == int.from_bytes(
        bytes([0x80]) * 16,
        "big",
    ) % 13
    assert features["prefix_128_hex_histogram_8"] == 16
    assert features["prefix_128_hex_histogram_0"] == 16


def test_invariance_count_coverage_and_full_row_mismatch_fail_closed():
    rows = tuple(_row(index) for index in range(3))
    batch = _batch()
    with pytest.raises(ValueError, match="receipt count"):
        validate_frozen_covert_audit_structure(
            rows,
            replace(batch, global_renamings=batch.global_renamings[:-1]),
        )
    bad_rows = tuple(_decision(index, b"changed" if index == 1 else b"residual") for index in range(3))
    bad = RecognizerInvariantReceipt(InvarianceKind.OBSERVATION_ORDER, 0, bad_rows)
    with pytest.raises(ValueError, match="differs from the canonical baseline"):
        validate_frozen_covert_audit_structure(
            rows,
            replace(
                batch,
                observation_order_permutations=(bad,) + batch.observation_order_permutations[1:],
            ),
        )


def test_receipt_content_id_is_computed_not_caller_supplied_and_commits_full_vector():
    first = RecognizerInvariantReceipt(InvarianceKind.BASELINE, 0, (_decision(0),))
    same = RecognizerInvariantReceipt(InvarianceKind.BASELINE, 0, (_decision(0),))
    changed = RecognizerInvariantReceipt(InvarianceKind.BASELINE, 0, (_decision(0, b"different"),))
    assert first.content_id == same.content_id
    assert first.content_id != changed.content_id
    with pytest.raises(TypeError):
        RecognizerInvariantReceipt(  # type: ignore[call-arg]
            InvarianceKind.BASELINE,
            0,
            (_decision(0),),
            "caller-root",
        )


def test_input_roots_are_computed_internally_and_length_caps_precede_item_scans():
    assert _row(0, 0).content_id != _row(0, 1).content_id
    with pytest.raises(TypeError):
        EnvelopeAuditRow(  # type: ignore[call-arg]
            0,
            bytes(65_536),
            _row(0).labels,
            _row(0).strata,
            "caller-root",
        )
    oversized = (object(),) * (DEFAULT_COVERT_AUDIT_POLICY.maximum_rows + 1)
    with pytest.raises(ValueError, match="row count exceeds"):
        validate_frozen_covert_audit_structure(oversized, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="row count exceeds"):
        RecognizerInvariantReceipt(  # type: ignore[arg-type]
            InvarianceKind.BASELINE,
            0,
            oversized,
        )


def test_authoritative_entry_recomputes_cached_input_and_receipt_roots():
    row = _row(0)
    object.__setattr__(row, "envelope", bytes([1]) * 65_536)
    with pytest.raises(ValueError, match="audit row cached content id drift"):
        validate_frozen_covert_audit_structure((row,), _batch(1))

    rows = tuple(_row(index) for index in range(2))
    batch = _batch(2)
    object.__setattr__(batch.baseline.rows[0], "decision", "MUTATED")
    with pytest.raises(ValueError, match="baseline cached content id drift"):
        validate_frozen_covert_audit_structure(rows, batch)


def test_insufficient_categories_or_strata_abstain_without_zero_leakage_substitute():
    receipt = run_frozen_envelope_covert_audit_mechanics(
        tuple(_row(index) for index in range(3)),
        _batch(),
    )
    assert receipt.disposition is CovertAuditDisposition.ABSTAIN
    assert receipt.results == ()
    assert receipt.permutations_executed_per_target == 0
    assert receipt.permutation_schedule_root is None
    assert receipt.envelope_feature_gate_acceptable is False
    assert receipt.formal_audit_evidence is False
    assert {item.reason for item in receipt.sufficiency} == {
        "TARGET_HAS_FEWER_THAN_TWO_CLASSES"
    }


def test_insufficient_cv_and_frozen_stratum_each_abstain_with_explicit_reason():
    rows = _statistical_rows()
    rare_joint = tuple(
        replace(
            row,
            labels=replace(
                row.labels,
                joint_class="rare" if row.auditor_row_id == 0 else "common",
            ),
        )
        for row in rows
    )
    cv_receipt = run_frozen_envelope_covert_audit_mechanics(
        rare_joint,
        _batch(len(rows)),
    )
    assert cv_receipt.disposition is CovertAuditDisposition.ABSTAIN
    assert {
        item.target: item.reason for item in cv_receipt.sufficiency
    }["joint_decision_class"] == "LOO_CV_CLASS_HAS_FEWER_THAN_TWO_ROWS"

    homogeneous_binding = tuple(
        replace(
            row,
            labels=replace(row.labels, binding="binding-0"),
        )
        if row.labels.family == "F01" and row.labels.scale == "root"
        else row
        for row in rows
    )
    strata_receipt = run_frozen_envelope_covert_audit_mechanics(
        homogeneous_binding,
        _batch(len(rows)),
    )
    assert strata_receipt.disposition is CovertAuditDisposition.ABSTAIN
    assert {
        item.target: item.reason for item in strata_receipt.sufficiency
    }["binding"] == "PERMUTATION_STRATUM_HAS_FEWER_THAN_TWO_TARGET_CLASSES"


def test_metric_and_holm_semantics_are_explicit_and_deterministic():
    labels = ("a", "a", "b", "b")
    values = (0, 0, 1, 1)
    assert covert._normalized_mutual_information(values, labels) == pytest.approx(1.0)
    assert covert._leave_one_out_balanced_accuracy_advantage(
        values,
        labels,
    ) == pytest.approx(0.5)
    assert covert._holm_adjust((0.001, 0.02, 0.03)) == pytest.approx(
        (0.003, 0.04, 0.04)
    )
    assert SEMANTICS_ID.startswith("phase2b_covert_audit_semantics_")


def test_nmi_and_combined_stat_are_bit_identical_under_row_reordering():
    values = (0, 0, 0, 1, 1, 1)
    labels = ("a", "a", "b", "a", "b", "b")
    order = (5, 2, 4, 0, 3, 1)
    reordered_values = tuple(values[index] for index in order)
    reordered_labels = tuple(labels[index] for index in order)
    first_nmi = covert._normalized_mutual_information(values, labels)
    second_nmi = covert._normalized_mutual_information(
        reordered_values,
        reordered_labels,
    )
    first_advantage = covert._leave_one_out_balanced_accuracy_advantage(
        values,
        labels,
    )
    second_advantage = covert._leave_one_out_balanced_accuracy_advantage(
        reordered_values,
        reordered_labels,
    )
    assert first_nmi.hex() == second_nmi.hex()
    assert first_advantage.hex() == second_advantage.hex()
    assert covert._combined_leak_statistic(
        first_nmi,
        first_advantage,
    ).hex() == covert._combined_leak_statistic(
        second_nmi,
        second_advantage,
    ).hex()


def test_statistical_work_budget_abstains_before_10000_permutations():
    count = DEFAULT_COVERT_AUDIT_POLICY.maximum_statistical_rows + 1
    rows = tuple(_row(index, index % 2) for index in range(count))
    receipt = run_frozen_envelope_covert_audit_mechanics(rows, _batch(count))
    assert receipt.disposition is CovertAuditDisposition.ABSTAIN
    assert receipt.permutations_executed_per_target == 0
    assert receipt.estimated_statistical_work_units > (
        receipt.maximum_statistical_work_units
    )
    assert {item.reason for item in receipt.sufficiency} == {
        "STATISTICAL_WORK_BUDGET_EXCEEDED"
    }


def test_full_frozen_10000_permutation_audit_uses_one_global_holm_family(
    complete_mechanics_receipt,
):
    rows, receipt = complete_mechanics_receipt
    assert receipt.disposition is CovertAuditDisposition.STATISTICS_COMPLETE
    assert receipt.permutations_requested == 10_000
    assert receipt.permutations_executed_per_target == 10_000
    assert receipt.permutation_schedule_root is not None
    assert receipt.permutation_schedule_root == (
        "phase2b_covert_permutation_schedule_"
        "479cd5314d1011040eda23269e0c7721d432f115172bd6fce23420a6f850b42a"
    )
    assert receipt.permutation_schedule_domain_root == (
        "phase2b_covert_permutation_domain_"
        "2e4ef71d35f6a92ace327f842e70d90240b84d09006f4794deaa7c90640b8362"
    )
    assert receipt.semantics_id == (
        "phase2b_covert_audit_semantics_"
        "793b46201e01e67feef6268009542319cb7bf5204b90a654c089c8f8edb33a86"
    )
    assert receipt.holm_hypothesis_count == 5 * len(
        extract_frozen_byte_features(rows[0]).values
    )
    assert len(receipt.results) == receipt.holm_hypothesis_count
    assert {result.target for result in receipt.results} == {
        "family",
        "binding",
        "scale",
        "answerable_vs_abstain",
        "joint_decision_class",
    }
    assert all(
        abs(result.raw_permutation_p * 10_001 - round(result.raw_permutation_p * 10_001))
        < 1e-9
        for result in receipt.results
    )
    assert receipt.formal_audit_evidence is False
    assert receipt.sealed_holdout_eligible is False


def test_forged_target_feature_receipts_fail_closed(complete_mechanics_receipt):
    _, receipt = complete_mechanics_receipt
    result = receipt.results[0]
    with pytest.raises(TypeError, match="exceedance count"):
        replace(
            result,
            permutation_exceedance_count=(
                DEFAULT_COVERT_AUDIT_POLICY.label_permutations + 1
            ),
        )
    forged_raw_p = 0.5 if result.raw_permutation_p != 0.5 else 0.75
    with pytest.raises(ValueError, match="does not match its exact count"):
        replace(result, raw_permutation_p=forged_raw_p)
    with pytest.raises(ValueError, match="gate booleans"):
        replace(result, nmi_within_limit=not result.nmi_within_limit)


def test_forged_complete_mechanics_receipts_fail_closed(
    complete_mechanics_receipt,
):
    _, receipt = complete_mechanics_receipt
    swapped = (
        receipt.results[1],
        receipt.results[0],
    ) + receipt.results[2:]
    with pytest.raises(ValueError, match="missing frozen statistics"):
        replace(receipt, results=swapped)

    forged_result_p = (
        0.5
        if receipt.results[0].holm_adjusted_p != 0.5
        else 0.75
    )
    forged_result = replace(
        receipt.results[0],
        holm_adjusted_p=forged_result_p,
        adjusted_p_within_limit=(
            forged_result_p >= DEFAULT_COVERT_AUDIT_POLICY.family_wise_alpha
        ),
    )
    forged_holm_results = (forged_result,) + receipt.results[1:]
    with pytest.raises(ValueError, match="Holm adjustment drift"):
        replace(receipt, results=forged_holm_results)

    with pytest.raises(ValueError, match="aggregate gate drift"):
        replace(
            receipt,
            envelope_feature_gate_acceptable=(
                not receipt.envelope_feature_gate_acceptable
            ),
        )
    with pytest.raises(ValueError, match="missing frozen statistics"):
        replace(
            receipt,
            sufficiency=(
                replace(receipt.sufficiency[0], sufficient=False),
            )
            + receipt.sufficiency[1:],
        )
    with pytest.raises(TypeError, match="exact booleans"):
        replace(receipt, envelope_feature_gate_acceptable=1)
    with pytest.raises(ValueError, match="work formula"):
        replace(
            receipt,
            estimated_statistical_work_units=(
                receipt.estimated_statistical_work_units + 1
            ),
        )
    with pytest.raises(ValueError, match="missing frozen statistics"):
        replace(
            receipt,
            holm_hypothesis_count=receipt.holm_hypothesis_count - 1,
        )
    with pytest.raises(TypeError, match="structural receipt id"):
        replace(receipt, structural_receipt_content_id="caller-root")
    with pytest.raises(TypeError, match="domain root"):
        replace(receipt, permutation_schedule_domain_root="caller-root")
    with pytest.raises(TypeError, match="schedule root"):
        replace(receipt, permutation_schedule_root="caller-root")
    with pytest.raises(ValueError, match="result tuple exceeds"):
        replace(receipt, results=receipt.results + (object(),))
    with pytest.raises(ValueError, match="sufficiency coverage"):
        replace(receipt, sufficiency=receipt.sufficiency + (object(),))
    CovertAuditDisposition,
