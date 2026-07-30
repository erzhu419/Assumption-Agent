from __future__ import annotations

from collections import Counter
import json

from assumption_agent.meta_assumption import (
    AssumptionRole,
    CompilerTarget,
    UniversalAssumptionOntology,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    T01,
    T02,
    T03,
    T04,
    T05,
    T06,
    T07,
    T08,
    T09,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
    T16,
    T17,
    T18,
    T19,
    T20,
    T21,
    T22,
    VERSION,
    build_universal_assumption_ontology_v1,
)


TEMPLATE_IDS = (
    T01,
    T02,
    T03,
    T04,
    T05,
    T06,
    T07,
    T08,
    T09,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
    T16,
    T17,
    T18,
    T19,
    T20,
    T21,
    T22,
)


def test_v1_catalog_is_complete_and_valid() -> None:
    ontology = build_universal_assumption_ontology_v1()

    assert ontology.version == VERSION
    assert ontology.validate() == ()
    assert len(ontology.roots) == 6
    assert len(ontology.templates) == 22
    assert len(ontology.legacy_aliases) == 13
    assert tuple(row.template_id for row in ontology.templates) == TEMPLATE_IDS
    assert len(set(TEMPLATE_IDS)) == 22


def test_v1_hash_is_deterministic_and_order_canonical() -> None:
    first = build_universal_assumption_ontology_v1()
    second = build_universal_assumption_ontology_v1()
    reordered = UniversalAssumptionOntology(
        version=first.version,
        roots=tuple(reversed(first.roots)),
        templates=tuple(reversed(first.templates)),
        legacy_aliases=tuple(reversed(first.legacy_aliases)),
    )

    assert first.ontology_hash == second.ontology_hash
    assert first.safe_payload() == second.safe_payload()
    assert reordered.validate() == ()
    assert reordered.ontology_hash == first.ontology_hash
    assert reordered.safe_payload() == first.safe_payload()
    assert all(
        first.require_template(template_id).template_hash
        == second.require_template(template_id).template_hash
        for template_id in TEMPLATE_IDS
    )


def test_all_legacy_13_aliases_resolve_without_becoming_a_second_catalog() -> None:
    ontology = build_universal_assumption_ontology_v1()
    expected_aliases = tuple(
        f"legacy13.v0.{ordinal:02d}_{suffix}"
        for ordinal, suffix in (
            (1, "symmetry_equivariance"),
            (2, "locality_markov"),
            (3, "manifold_intrinsic_dimension"),
            (4, "low_rank_separability"),
            (5, "monotonic_shape"),
            (6, "diminishing_returns_submodularity"),
            (7, "conservation_balance"),
            (8, "stability_contractivity"),
            (9, "exchangeability_hierarchical_bayes"),
            (10, "maximum_entropy"),
            (11, "mdl_occam"),
            (12, "information_bottleneck"),
            (13, "pac_bayes"),
        )
    )
    template_ids = frozenset(TEMPLATE_IDS)

    assert tuple(row.alias_id for row in ontology.legacy_aliases) == expected_aliases
    assert all(
        set(alias.target_template_ids) <= template_ids
        for alias in ontology.legacy_aliases
    )
    assert ontology.legacy_aliases[2].target_template_ids == (T04, T10)
    assert ontology.legacy_aliases[8].target_template_ids == (T14, T07)
    assert ontology.legacy_aliases[12].target_template_ids == (T01, T21)


def test_every_template_is_falsifiable_train_only_and_fail_closed() -> None:
    ontology = build_universal_assumption_ontology_v1()

    for template in ontology.templates:
        assert template.support_signatures
        assert template.counter_signatures
        assert not (
            set(template.support_signatures)
            & set(template.counter_signatures)
        )
        assert template.probe_plan.train_only is True
        assert template.probe_plan.max_evaluations == 1
        assert template.probe_plan.support_rule_id
        assert template.probe_plan.counter_rule_id
        assert (
            template.probe_plan.support_rule_id
            != template.probe_plan.counter_rule_id
        )
        assert template.not_applicable_conditions
        assert template.compiler_targets


def test_parent_taxonomy_and_epistemic_roles_are_not_conflated() -> None:
    ontology = build_universal_assumption_ontology_v1()
    by_id = {row.template_id: row for row in ontology.templates}
    primary_counts = Counter(
        row.primary_parent_id for row in ontology.templates
    )

    # Root membership is 4+5+4+4+2+3.  The first five roots therefore
    # contain 19 leaves, while epistemic role assignment remains 18+4:
    # t19 is rooted under uncertainty but is a governance/decision rule.
    assert primary_counts == {
        R1: 4,
        R2: 5,
        R3: 4,
        R4: 4,
        R5: 2,
        R6: 3,
    }
    assert sum(primary_counts[root_id] for root_id in (R1, R2, R3, R4, R5)) == 19
    assert primary_counts[R6] == 3

    structural_roles = {
        AssumptionRole.WORLD_CLAIM,
        AssumptionRole.REPRESENTATION_PRIOR,
        AssumptionRole.REGULARIZER,
    }
    governance_roles = {
        AssumptionRole.GOVERNANCE_RULE,
        AssumptionRole.DECISION_RULE,
    }
    assert all(set(by_id[template_id].roles) & structural_roles for template_id in TEMPLATE_IDS[:18])
    assert all(
        set(by_id[template_id].roles) <= governance_roles
        and AssumptionRole.GOVERNANCE_RULE in by_id[template_id].roles
        for template_id in TEMPLATE_IDS[18:]
    )
    assert by_id[T19].primary_parent_id == R5
    assert R6 in by_id[T19].parent_ids
    assert by_id[T19].roles == (
        AssumptionRole.GOVERNANCE_RULE,
        AssumptionRole.DECISION_RULE,
    )


def test_catalog_uses_support_not_unearned_bayesian_language() -> None:
    ontology = build_universal_assumption_ontology_v1()
    serialized = json.dumps(
        ontology.safe_payload(),
        ensure_ascii=False,
        sort_keys=True,
    ).lower()

    assert "posterior" not in serialized


def test_gap_study_templates_authorize_the_frozen_compiler_targets() -> None:
    ontology = build_universal_assumption_ontology_v1()

    for template_id in (T02, T05, T08, T18):
        assert (
            CompilerTarget.POLICY_PROGRAM
            in ontology.require_template(template_id).compiler_targets
        )
    assert (
        CompilerTarget.NO_DIRECT_TREATMENT
        in ontology.require_template(T19).compiler_targets
    )
    assert ontology.require_template(T20).compiler_targets == (
        CompilerTarget.NO_DIRECT_TREATMENT,
    )
    assert ontology.require_template(T21).compiler_targets == (
        CompilerTarget.NO_DIRECT_TREATMENT,
    )
