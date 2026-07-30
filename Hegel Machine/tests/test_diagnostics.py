from dataclasses import replace

import pytest

from hegel_machine.diagnostics import (
    InadequacyVerdict,
    OntologyInadequacyReport,
    ResidualProfile,
    diagnose_ontology_inadequacy,
)


def persistent_profile() -> ResidualProfile:
    return ResidualProfile(
        profile_id="persistent",
        refit_gain=0.01,
        uncertainty_coverage=0.10,
        scope_repair_gain=0.01,
        mixture_gain=0.01,
        low_order_composition_gain=0.01,
        idealization_gain=0.01,
        robustification_tail_gain=0.01,
        added_probe_gain=0.01,
        cross_seed_stability=0.90,
        structural_coherence=0.90,
        uncertainty_excess=0.90,
        compression_gain=0.30,
        preregistered_prediction_gain=0.30,
        case_count=10,
        outlier_fraction=0.10,
    )


def test_cheaper_parameter_repair_blocks_invention():
    report = diagnose_ontology_inadequacy(
        replace(persistent_profile(), refit_gain=0.50)
    )
    assert report.verdict is InadequacyVerdict.PARAMETER_DEFECT
    assert not report.language_extension_allowed
    assert report.checked_steps == (InadequacyVerdict.PARAMETER_DEFECT,)


def test_probe_repair_is_checked_before_ontology():
    report = diagnose_ontology_inadequacy(
        replace(persistent_profile(), added_probe_gain=0.50)
    )
    assert report.verdict is InadequacyVerdict.PROBE_DEFECT
    assert not report.language_extension_allowed


def test_only_persistent_structured_residual_opens_language():
    report = diagnose_ontology_inadequacy(persistent_profile())
    assert report.verdict is InadequacyVerdict.ONTOLOGY_DEFECT
    assert report.residual_is_persistent
    assert report.language_extension_allowed


def test_single_outlier_never_opens_language():
    report = diagnose_ontology_inadequacy(
        replace(persistent_profile(), case_count=1, outlier_fraction=1.0)
    )
    assert report.verdict is InadequacyVerdict.INSUFFICIENT_EVIDENCE
    assert not report.language_extension_allowed


def test_direct_ontology_report_cannot_skip_the_diagnostic_ladder():
    with pytest.raises(ValueError, match="full ladder"):
        OntologyInadequacyReport(
            "profile",
            InadequacyVerdict.ONTOLOGY_DEFECT,
            (InadequacyVerdict.ONTOLOGY_DEFECT,),
            True,
            True,
            ("self-reported",),
            "profile_hash",
            "threshold_hash",
        )
