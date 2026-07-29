from __future__ import annotations

from dataclasses import dataclass, replace
import json

import pytest

from assumption_agent.evaluation import hypothesis_program_behavior_hash
from assumption_agent.meta_assumption import (
    AssumptionRole,
    CompiledTreatment,
    CompilerTrustAnchor,
    CompilerTarget,
    CompilationReceipt,
    DiagnosticProbePlan,
    HypothesisClaim,
    HypothesisSpaceCompilerRegistry,
    LegacyAssumptionAlias,
    MetaAssumptionTemplate,
    OntologyRoot,
    ProbeDisposition,
    ProbeEvidenceBundle,
    ProbeObservationStatistic,
    ProbeReceipt,
    ProbeTrustAnchor,
    ProbeVerificationResult,
    ProbeVerifierRegistry,
    RecipeActionBinding,
    TreatmentDisposition,
    UniversalAssumptionOntology,
    action_node_semantics_hash,
    verify_compilation_receipt,
)
from assumption_agent.models import (
    ActionNode,
    ExpectedEffect,
    FeaturePredicate,
    HypothesisKind,
    HypothesisProgram,
    HypothesisStatus,
    SplitName,
    TriggerSpec,
    VerifierContract,
    stable_hash,
)


def _hash(label: str) -> str:
    return stable_hash({"fixture": label})


def _probe_plan() -> DiagnosticProbePlan:
    return DiagnosticProbePlan(
        probe_id="probe.sparse.v1",
        observable_ids=("changed-selection-count", "net-train-utility"),
        support_rule_id="rule.net-positive.v1",
        counter_rule_id="rule.net-negative.v1",
        max_evaluations=4,
    )


def _ontology() -> UniversalAssumptionOntology:
    root = OntologyRoot(
        root_id="compression.sufficient-representation",
        title="Compression and sufficient representation",
        description="Prefer the smallest representation that retains decisions.",
    )
    template = MetaAssumptionTemplate(
        template_id="sparsity.v1",
        primary_parent_id=root.root_id,
        parent_ids=(root.root_id,),
        roles=(
            AssumptionRole.WORLD_CLAIM,
            AssumptionRole.REPRESENTATION_PRIOR,
        ),
        claim_schema="sparse-beneficial-action-schema-v1",
        admissible_variable_types=("integer-cardinality", "relation-role"),
        support_signatures=("few-actions-positive",),
        counter_signatures=("dense-actions-required",),
        probe_plan=_probe_plan(),
        compiler_targets=(
            CompilerTarget.POLICY_PROGRAM,
            CompilerTarget.EVALUATOR_ARTIFACT,
            CompilerTarget.NO_DIRECT_TREATMENT,
        ),
        not_applicable_conditions=("no-action-candidate-space",),
        invariances=("item-order",),
    )
    return UniversalAssumptionOntology(
        version="universal-assumption-ontology.v1",
        roots=(root,),
        templates=(template,),
        legacy_aliases=(
            LegacyAssumptionAlias(
                alias_id="legacy.sparsity",
                target_template_ids=(template.template_id,),
            ),
        ),
    )


def _claim(
    ontology: UniversalAssumptionOntology,
    *,
    formation_split: SplitName = SplitName.TRAIN,
) -> HypothesisClaim:
    return HypothesisClaim(
        claim_id="claim.sparse-action.v1",
        ontology_hash=ontology.ontology_hash,
        template_ids=("sparsity.v1",),
        scope_hash=_hash("scope"),
        mechanism_statement=(
            "Only a sparse subset of candidate actions has positive TRAIN utility."
        ),
        bound_variable_types=("integer-cardinality", "relation-role"),
        observable_predictions=("few-actions-positive",),
        counter_predictions=("dense-actions-required",),
        competing_claim_ids=("claim.dense-action.v1",),
        description_length_bits=96,
        evidence_receipt_hashes=(_hash("formation-evidence"),),
        formation_split=formation_split,
    )


@dataclass
class _ProbeVerifier:
    verifier_id: str
    verifier_version: str
    implementation_hash: str
    probe_id: str
    support_rule_id: str
    counter_rule_id: str

    def match_signatures(
        self,
        *,
        template: MetaAssumptionTemplate,
        claim: HypothesisClaim,
        evidence: ProbeEvidenceBundle,
    ) -> ProbeVerificationResult:
        del claim
        support_votes = 0
        counter_votes = 0
        for row in evidence.observation_statistics:
            values = dict(row.statistic_values)
            support_votes += int(values["support-vote"])
            counter_votes += int(values["counter-vote"])
        if support_votes > counter_votes:
            return ProbeVerificationResult(
                observed_support_signature_ids=(
                    template.support_signatures[0],
                )
            )
        if counter_votes > support_votes:
            return ProbeVerificationResult(
                observed_counter_signature_ids=(
                    template.counter_signatures[0],
                )
            )
        if support_votes:
            return ProbeVerificationResult(
                observed_support_signature_ids=(
                    template.support_signatures[0],
                ),
                observed_counter_signature_ids=(
                    template.counter_signatures[0],
                ),
            )
        return ProbeVerificationResult()


def _probe_trust_anchor(
    template: MetaAssumptionTemplate,
) -> ProbeTrustAnchor:
    return ProbeTrustAnchor(
        verifier_id=f"verifier.{template.template_id}",
        verifier_version="probe-verifier-version.v1",
        implementation_hash=_hash(
            f"probe-verifier:{template.template_id}"
        ),
        probe_id=template.probe_plan.probe_id,
        support_rule_id=template.probe_plan.support_rule_id,
        counter_rule_id=template.probe_plan.counter_rule_id,
    )


def _probe_verifier_registry(
    ontology: UniversalAssumptionOntology,
    template_ids: tuple[str, ...],
) -> ProbeVerifierRegistry:
    registry = ProbeVerifierRegistry()
    for template_id in template_ids:
        template = ontology.require_template(template_id)
        anchor = _probe_trust_anchor(template)
        registry.register(
            _ProbeVerifier(**anchor.safe_payload()),
            trust_anchor=anchor,
        )
    return registry


def _probe_evidence(
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
    *,
    template_id: str = "sparsity.v1",
    disposition: ProbeDisposition = ProbeDisposition.SUPPORTED,
) -> ProbeEvidenceBundle:
    template = ontology.require_template(template_id)
    if disposition is ProbeDisposition.SUPPORTED:
        votes = ((0, 1), (0, 1), (0, 0))
    elif disposition is ProbeDisposition.FALSIFIED:
        votes = ((1, 0), (1, 0), (0, 0))
    else:
        votes = ((0, 1), (1, 0), (0, 0))
    rows = tuple(
        sorted(
            (
                ProbeObservationStatistic(
                    observation_hash=_hash(
                        f"{template_id}-observation-{index}"
                    ),
                    statistic_values=(
                        ("counter-vote", str(counter_vote)),
                        ("support-vote", str(support_vote)),
                    ),
                )
                for index, (counter_vote, support_vote) in enumerate(votes)
            ),
            key=lambda row: row.observation_hash,
        )
    )
    evidence = ProbeEvidenceBundle(
        bundle_id="probe-evidence.pending",
        ontology_hash=ontology.ontology_hash,
        claim_hash=claim.claim_hash,
        template_id=template_id,
        probe_plan_hash=template.probe_plan.plan_hash,
        train_split_hash=_hash("train-split"),
        observation_statistics=rows,
        source_payload_access_count=len(rows),
    )
    return replace(evidence, bundle_id=evidence.expected_bundle_id)


def _probe(
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
    *,
    disposition: ProbeDisposition = ProbeDisposition.SUPPORTED,
    formation_split: SplitName = SplitName.TRAIN,
    validation_or_test_access_count: int = 0,
) -> ProbeReceipt:
    evidence = _probe_evidence(
        ontology,
        claim,
        disposition=disposition,
    )
    registry = _probe_verifier_registry(
        ontology, ("sparsity.v1",)
    )
    receipt = registry.issue_receipt(
        ontology=ontology,
        claim=claim,
        evidence=evidence,
    )
    receipt = replace(
        receipt,
        formation_split=formation_split,
        validation_or_test_access_count=validation_or_test_access_count,
    )
    return replace(receipt, receipt_id=receipt.expected_receipt_id)


def _two_template_ontology() -> UniversalAssumptionOntology:
    ontology = _ontology()
    root = ontology.roots[0]
    locality_plan = DiagnosticProbePlan(
        probe_id="probe.locality.v1",
        observable_ids=("local-neighborhood-gain",),
        support_rule_id="rule.local-positive.v1",
        counter_rule_id="rule.nonlocal-positive.v1",
        max_evaluations=4,
    )
    locality = MetaAssumptionTemplate(
        template_id="locality.v1",
        primary_parent_id=root.root_id,
        parent_ids=(root.root_id,),
        roles=(AssumptionRole.WORLD_CLAIM,),
        claim_schema="local-relation-schema-v1",
        admissible_variable_types=("relation-role",),
        support_signatures=("local-neighborhood-gain",),
        counter_signatures=("global-context-required",),
        probe_plan=locality_plan,
        compiler_targets=(
            CompilerTarget.POLICY_PROGRAM,
            CompilerTarget.NO_DIRECT_TREATMENT,
        ),
        not_applicable_conditions=("no-relation-structure",),
    )
    return replace(
        ontology,
        templates=(locality, *ontology.templates),
        legacy_aliases=(
            *ontology.legacy_aliases,
            LegacyAssumptionAlias(
                alias_id="legacy.locality",
                target_template_ids=(locality.template_id,),
            ),
        ),
    )


def _two_template_claim(
    ontology: UniversalAssumptionOntology,
) -> HypothesisClaim:
    return replace(
        _claim(ontology),
        template_ids=("locality.v1", "sparsity.v1"),
        claim_id="claim.local-sparse-action.v1",
        mechanism_statement=(
            "A sparse subset of local relation actions has positive TRAIN utility."
        ),
        observable_predictions=(
            "few-actions-positive",
            "local-neighborhood-gain",
        ),
        counter_predictions=(
            "dense-actions-required",
            "global-context-required",
        ),
    )


def _probe_for_template(
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
    template_id: str,
) -> ProbeReceipt:
    evidence = _probe_evidence(
        ontology,
        claim,
        template_id=template_id,
    )
    registry = _probe_verifier_registry(ontology, (template_id,))
    return registry.issue_receipt(
        ontology=ontology,
        claim=claim,
        evidence=evidence,
    )


def _program(
    *,
    metric: str = "task_success",
    status: HypothesisStatus = HypothesisStatus.CANDIDATE,
) -> HypothesisProgram:
    return HypothesisProgram(
        id="ontology-policy-v1",
        kind=HypothesisKind.POLICY,
        statement="Apply the frozen sparse action selector.",
        trigger=TriggerSpec(
            all_of=(FeaturePredicate("family", "eq", "relation-a"),)
        ),
        anti_trigger=TriggerSpec(),
        action_graph=(
            ActionNode(
                id="set-threshold",
                operation="set_parameter",
                target="selection.minimum_confidence",
                value=0.7,
            ),
        ),
        expected_effect=ExpectedEffect(metric=metric),
        verifier=VerifierContract(
            checks=("frozen-local-check",),
            anchor_id="offline-anchor-v1",
            repair_on_failure=False,
            max_repair_depth=0,
        ),
        evaluator_epoch="evaluator-epoch-v1",
        status=status,
    )


@dataclass
class _Compiler:
    treatment: CompiledTreatment
    compiler_id: str = "compiler.sparse.v1"
    compiler_version: str = "compiler-version.v1"
    implementation_hash: str = _hash("compiler-implementation")
    primary_metric: str = "task_success"

    def compile(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        probes: tuple[ProbeReceipt, ...],
    ) -> CompiledTreatment:
        assert ontology.ontology_hash == claim.ontology_hash
        assert probes
        return self.treatment


def _trust_anchor() -> CompilerTrustAnchor:
    return CompilerTrustAnchor(
        compiler_id="compiler.sparse.v1",
        compiler_version="compiler-version.v1",
        implementation_hash=_hash("compiler-implementation"),
        primary_metric="task_success",
    )


def _compiler_registry(
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
) -> HypothesisSpaceCompilerRegistry:
    return HypothesisSpaceCompilerRegistry(
        probe_verifier_registry=_probe_verifier_registry(
            ontology, claim.template_ids
        )
    )


def _evidence_for_probe(
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
    probe: ProbeReceipt,
) -> ProbeEvidenceBundle:
    return _probe_evidence(
        ontology,
        claim,
        template_id=probe.template_id,
        disposition=probe.disposition,
    )


def _evidence_for_probes(
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
    probes: tuple[ProbeReceipt, ...],
) -> tuple[ProbeEvidenceBundle, ...]:
    return tuple(
        _evidence_for_probe(ontology, claim, probe)
        for probe in probes
    )


def _active_treatment(
    *,
    metric: str = "task_success",
) -> CompiledTreatment:
    program = _program(metric=metric)
    action = program.action_graph[0]
    return CompiledTreatment(
        disposition=TreatmentDisposition.ACTIVE_PROGRAM,
        program=program,
        recipe_ids=("recipe.sparse.v1",),
        recipe_action_bindings=(
            RecipeActionBinding(
                recipe_id="recipe.sparse.v1",
                action_id=action.id,
                action_semantics_hash=action_node_semantics_hash(action),
            ),
        ),
    )


def _with_expected_receipt_id(
    receipt: CompilationReceipt,
) -> CompilationReceipt:
    pending = replace(receipt, receipt_id="compilation.pending")
    return replace(pending, receipt_id=pending.expected_receipt_id)


def _with_expected_probe_receipt_id(
    receipt: ProbeReceipt,
) -> ProbeReceipt:
    pending = replace(receipt, receipt_id="probe-receipt.pending")
    return replace(pending, receipt_id=pending.expected_receipt_id)


def _compile_active() -> tuple[
    UniversalAssumptionOntology,
    HypothesisClaim,
    ProbeReceipt,
    CompiledTreatment,
    CompilationReceipt,
    CompilerTrustAnchor,
]:
    ontology = _ontology()
    claim = _claim(ontology)
    probe = _probe(ontology, claim)
    treatment = _active_treatment()
    trust_anchor = _trust_anchor()
    registry = _compiler_registry(ontology, claim)
    registry.register(_Compiler(treatment), trust_anchor=trust_anchor)
    compiled, receipt = registry.compile(
        compiler_id="compiler.sparse.v1",
        compiler_version="compiler-version.v1",
        ontology=ontology,
        claim=claim,
        probes=(probe,),
        probe_evidence_bundles=(
            _evidence_for_probe(ontology, claim, probe),
        ),
    )
    return ontology, claim, probe, compiled, receipt, trust_anchor


def test_valid_ontology_claim_probe_and_active_compilation_path() -> None:
    ontology, claim, probe, treatment, receipt, trust_anchor = (
        _compile_active()
    )

    assert ontology.validate() == ()
    assert claim.validate(ontology) == ()
    assert probe.validate(ontology=ontology, claim=claim) == ()
    assert treatment.validate() == ()
    assert (
        receipt.validate(
            ontology=ontology,
            claim=claim,
            probes=(probe,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (probe,)
            ),
            probe_verifier_registry=_probe_verifier_registry(
                ontology, claim.template_ids
            ),
            treatment=treatment,
            trust_anchor=trust_anchor,
        )
        == ()
    )
    assert receipt.compiler_target is CompilerTarget.POLICY_PROGRAM
    assert receipt.treatment_behavior_hash == hypothesis_program_behavior_hash(
        treatment.program  # type: ignore[arg-type]
    )
    serialized = json.dumps(
        {
            "ontology": ontology.safe_payload(),
            "claim": claim.safe_payload(),
            "probe": probe.safe_payload(),
            "treatment": treatment.safe_payload(),
            "compilation": receipt.safe_payload(),
        },
        sort_keys=True,
    )
    assert "posterior" not in serialized


@pytest.mark.parametrize("split", [SplitName.VALIDATION, SplitName.TEST])
def test_claim_and_probe_fail_closed_on_non_train_formation(
    split: SplitName,
) -> None:
    ontology = _ontology()
    leaked_claim = _claim(ontology, formation_split=split)
    assert "hypothesis_claim_not_train_formed" in leaked_claim.validate(
        ontology
    )

    claim = _claim(ontology)
    leaked_probe = _probe(ontology, claim, formation_split=split)
    assert "probe_receipt_not_train_only" in leaked_probe.validate(
        ontology=ontology,
        claim=claim,
    )


def test_probe_rejects_heldout_access_even_when_declared_train() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    leaked = _probe(
        ontology,
        claim,
        validation_or_test_access_count=1,
    )

    assert "probe_receipt_heldout_accessed" in leaked.validate(
        ontology=ontology,
        claim=claim,
    )
    registry = _compiler_registry(ontology, claim)
    registry.register(
        _Compiler(
            CompiledTreatment(
                disposition=TreatmentDisposition.PRESERVE_BASELINE
            )
        ),
        trust_anchor=_trust_anchor(),
    )
    with pytest.raises(PermissionError, match="probe receipt is invalid"):
        registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(leaked,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (leaked,)
            ),
        )


def test_probe_rejects_online_or_api_evaluation() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    leaked = replace(
        _probe(ontology, claim),
        online_or_api_evaluation_count=1,
    )

    issues = leaked.validate(ontology=ontology, claim=claim)
    assert "probe_receipt_online_evaluation_used" in issues
    registry = _compiler_registry(ontology, claim)
    registry.register(
        _Compiler(
            CompiledTreatment(
                disposition=TreatmentDisposition.PRESERVE_BASELINE
            )
        ),
        trust_anchor=_trust_anchor(),
    )
    with pytest.raises(PermissionError, match="probe receipt is invalid"):
        registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(leaked,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (leaked,)
            ),
        )


def test_zero_observation_probe_is_invalid_and_cannot_compile() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    empty = replace(
        _probe(ontology, claim),
        observation_hashes=(),
        support_count=0,
        counter_count=0,
        observation_count=0,
        budget_used=0,
        disposition=ProbeDisposition.INCONCLUSIVE,
    )

    issues = empty.validate(ontology=ontology, claim=claim)
    assert "probe_receipt_observations_empty" in issues
    registry = _compiler_registry(ontology, claim)
    registry.register(
        _Compiler(
            CompiledTreatment(
                disposition=TreatmentDisposition.PRESERVE_BASELINE
            )
        ),
        trust_anchor=_trust_anchor(),
    )
    with pytest.raises(PermissionError, match="probe receipt is invalid"):
        registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(empty,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (empty,)
            ),
        )


def test_claim_variable_types_must_be_covered_without_cross_template_false_rejection() -> None:
    ontology = _two_template_ontology()
    claim = _two_template_claim(ontology)
    assert claim.validate(ontology) == ()

    unknown = replace(
        claim,
        bound_variable_types=(
            *claim.bound_variable_types,
            "unknown-variable-type",
        ),
    )
    assert (
        "hypothesis_claim_bound_variable_not_admissible"
        in unknown.validate(ontology)
    )
    leaves_one_template_unbound = replace(
        claim,
        bound_variable_types=("integer-cardinality",),
    )
    assert (
        "hypothesis_claim_template_unbound"
        in leaves_one_template_unbound.validate(ontology)
    )


def test_claim_predictions_cover_each_selected_template_signature() -> None:
    ontology = _two_template_ontology()
    claim = _two_template_claim(ontology)
    assert claim.validate(ontology) == ()

    support_uncovered = replace(
        claim,
        observable_predictions=("few-actions-positive",),
    )
    assert (
        "hypothesis_claim_template_support_signature_uncovered"
        in support_uncovered.validate(ontology)
    )
    counter_uncovered = replace(
        claim,
        counter_predictions=("dense-actions-required",),
    )
    assert (
        "hypothesis_claim_template_counter_signature_uncovered"
        in counter_uncovered.validate(ontology)
    )


def test_probe_counts_are_derived_from_authorized_observed_signatures() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    probe = _probe(ontology, claim)

    self_reported_count = replace(probe, support_count=2)
    assert (
        "probe_receipt_count_contract_invalid"
        in self_reported_count.validate(ontology=ontology, claim=claim)
    )
    unauthorized_support = replace(
        probe,
        observed_support_signature_ids=("unclaimed-support",),
    )
    assert (
        "probe_receipt_support_signature_not_authorized"
        in unauthorized_support.validate(ontology=ontology, claim=claim)
    )
    unauthorized_counter = replace(
        probe,
        disposition=ProbeDisposition.FALSIFIED,
        support_count=0,
        counter_count=1,
        observed_support_signature_ids=(),
        observed_counter_signature_ids=("unclaimed-counter",),
    )
    assert (
        "probe_receipt_counter_signature_not_authorized"
        in unauthorized_counter.validate(ontology=ontology, claim=claim)
    )
    assert unauthorized_support.receipt_hash != probe.receipt_hash


def test_random_observation_receipt_cannot_recompile_against_committed_evidence() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    evidence = _probe_evidence(ontology, claim)
    verifier_registry = _probe_verifier_registry(
        ontology, claim.template_ids
    )
    receipt = verifier_registry.issue_receipt(
        ontology=ontology,
        claim=claim,
        evidence=evidence,
    )
    forged = _with_expected_probe_receipt_id(
        replace(
            receipt,
            observation_hashes=tuple(
                sorted(_hash(f"forged-observation-{index}") for index in range(3))
            ),
        )
    )
    assert forged.validate(ontology=ontology, claim=claim) == ()
    compiler_registry = HypothesisSpaceCompilerRegistry(
        probe_verifier_registry=verifier_registry
    )
    compiler_registry.register(
        _Compiler(_active_treatment()),
        trust_anchor=_trust_anchor(),
    )

    with pytest.raises(PermissionError, match="trusted_recomputation"):
        compiler_registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(forged,),
            probe_evidence_bundles=(evidence,),
        )


def test_self_declared_support_cannot_override_trusted_counter_evidence() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    evidence = _probe_evidence(
        ontology,
        claim,
        disposition=ProbeDisposition.FALSIFIED,
    )
    verifier_registry = _probe_verifier_registry(
        ontology, claim.template_ids
    )
    counter_receipt = verifier_registry.issue_receipt(
        ontology=ontology,
        claim=claim,
        evidence=evidence,
    )
    forged_support = _with_expected_probe_receipt_id(
        replace(
            counter_receipt,
            support_count=1,
            counter_count=0,
            disposition=ProbeDisposition.SUPPORTED,
            observed_support_signature_ids=("few-actions-positive",),
            observed_counter_signature_ids=(),
        )
    )
    assert forged_support.validate(ontology=ontology, claim=claim) == ()
    compiler_registry = HypothesisSpaceCompilerRegistry(
        probe_verifier_registry=verifier_registry
    )
    compiler_registry.register(
        _Compiler(_active_treatment()),
        trust_anchor=_trust_anchor(),
    )

    with pytest.raises(PermissionError, match="trusted_recomputation"):
        compiler_registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(forged_support,),
            probe_evidence_bundles=(evidence,),
        )


def test_probe_statistic_commitment_tampering_is_rejected() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    evidence = _probe_evidence(ontology, claim)
    verifier_registry = _probe_verifier_registry(
        ontology, claim.template_ids
    )
    receipt = verifier_registry.issue_receipt(
        ontology=ontology,
        claim=claim,
        evidence=evidence,
    )
    first = evidence.observation_statistics[0]
    tampered_evidence = replace(
        evidence,
        observation_statistics=(
            replace(
                first,
                statistic_values=(
                    ("counter-vote", "1"),
                    ("support-vote", "0"),
                ),
            ),
            *evidence.observation_statistics[1:],
        ),
    )

    issues = verifier_registry.verify_receipt(
        receipt,
        ontology=ontology,
        claim=claim,
        evidence=tampered_evidence,
    )
    assert "probe_receipt_trusted_verification_failed" in issues
    assert tampered_evidence.evidence_bundle_hash != evidence.evidence_bundle_hash


def test_probe_verifier_is_bound_to_an_explicit_harness_anchor() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    template = ontology.require_template("sparsity.v1")
    anchor = _probe_trust_anchor(template)
    verifier = _ProbeVerifier(**anchor.safe_payload())
    registry = ProbeVerifierRegistry()
    with pytest.raises(TypeError, match="trust_anchor"):
        registry.register(verifier)  # type: ignore[call-arg]
    registry.register(verifier, trust_anchor=anchor)
    verifier.implementation_hash = _hash("mutated-probe-verifier")

    with pytest.raises(PermissionError, match="no longer matches"):
        registry.issue_receipt(
            ontology=ontology,
            claim=claim,
            evidence=_probe_evidence(ontology, claim),
        )


def test_content_hashes_expose_payload_tampering() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    probe = _probe(ontology, claim)

    ontology_payload = ontology.safe_payload()
    declared_ontology_hash = ontology_payload.pop("ontology_hash")
    ontology_payload["templates"][0]["claim_schema"] = "tampered"
    assert stable_hash(ontology_payload) != declared_ontology_hash

    claim_payload = claim.safe_payload()
    claim_payload["mechanism_statement"] = "tampered"
    assert stable_hash(claim_payload) != claim.claim_hash

    probe_payload = probe.safe_payload()
    probe_payload["support_count"] = 0
    assert stable_hash(probe_payload) != probe.receipt_hash


def test_program_behavior_hash_ignores_status_id_and_lineage_only() -> None:
    original = _program()
    promoted = replace(original, status=HypothesisStatus.PROMOTED)
    renamed_with_lineage = replace(
        original,
        id="ontology-policy-renamed",
        parent_id="ontology-policy-parent",
        lineage=("ontology-policy-parent",),
        created_from_transition_ids=("transition-v1",),
        status=HypothesisStatus.REJECTED,
    )

    expected = hypothesis_program_behavior_hash(original)
    assert hypothesis_program_behavior_hash(promoted) == expected
    assert hypothesis_program_behavior_hash(renamed_with_lineage) == expected
    executable_change = replace(
        original,
        expected_effect=replace(
            original.expected_effect,
            maximum_cost_ratio=2.0,
        ),
    )
    assert hypothesis_program_behavior_hash(executable_change) != expected


def test_primary_metric_mismatch_fails_closed_after_compiler_output() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    probe = _probe(ontology, claim)
    registry = _compiler_registry(ontology, claim)
    registry.register(
        _Compiler(
            treatment=_active_treatment(metric="different_metric"),
            primary_metric="task_success",
        ),
        trust_anchor=_trust_anchor(),
    )

    with pytest.raises(PermissionError, match="primary_metric_mismatch"):
        registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(probe,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (probe,)
            ),
        )


def test_falsified_probe_cannot_be_compiled() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    falsified = _probe(
        ontology,
        claim,
        disposition=ProbeDisposition.FALSIFIED,
    )
    assert falsified.validate(ontology=ontology, claim=claim) == ()
    registry = _compiler_registry(ontology, claim)
    registry.register(
        _Compiler(
            CompiledTreatment(
                disposition=TreatmentDisposition.PRESERVE_BASELINE
            )
        ),
        trust_anchor=_trust_anchor(),
    )

    with pytest.raises(PermissionError, match="falsified claim"):
        registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(falsified,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (falsified,)
            ),
        )


def test_inconclusive_probe_can_only_compile_preserve_baseline() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    inconclusive = _probe(
        ontology,
        claim,
        disposition=ProbeDisposition.INCONCLUSIVE,
    )
    assert inconclusive.validate(ontology=ontology, claim=claim) == ()

    active_registry = _compiler_registry(ontology, claim)
    active_registry.register(
        _Compiler(_active_treatment()),
        trust_anchor=_trust_anchor(),
    )
    with pytest.raises(PermissionError, match="requires supported probes"):
        active_registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(inconclusive,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (inconclusive,)
            ),
        )

    noop_registry = _compiler_registry(ontology, claim)
    noop_registry.register(
        _Compiler(
            CompiledTreatment(
                disposition=TreatmentDisposition.PRESERVE_BASELINE
            )
        ),
        trust_anchor=_trust_anchor(),
    )
    treatment, receipt = noop_registry.compile(
        compiler_id="compiler.sparse.v1",
        compiler_version="compiler-version.v1",
        ontology=ontology,
        claim=claim,
        probes=(inconclusive,),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (inconclusive,)
        ),
    )
    assert treatment.program is None
    assert receipt.compiler_target is CompilerTarget.NO_DIRECT_TREATMENT

    evaluator_registry = _compiler_registry(ontology, claim)
    evaluator_registry.register(
        _Compiler(
            CompiledTreatment(
                disposition=TreatmentDisposition.EVALUATOR_ARTIFACT,
                evaluator_artifact_hash=_hash("inconclusive-evaluator"),
            )
        ),
        trust_anchor=_trust_anchor(),
    )
    with pytest.raises(PermissionError, match="requires supported probes"):
        evaluator_registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(inconclusive,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (inconclusive,)
            ),
        )


def test_incomplete_multitemplate_probe_coverage_cannot_compile() -> None:
    ontology = _two_template_ontology()
    claim = _two_template_claim(ontology)
    locality_probe = _probe_for_template(ontology, claim, "locality.v1")
    registry = _compiler_registry(ontology, claim)
    registry.register(
        _Compiler(
            CompiledTreatment(
                disposition=TreatmentDisposition.PRESERVE_BASELINE
            )
        ),
        trust_anchor=_trust_anchor(),
    )

    with pytest.raises(PermissionError, match="coverage is incomplete"):
        registry.compile(
            compiler_id="compiler.sparse.v1",
            compiler_version="compiler-version.v1",
            ontology=ontology,
            claim=claim,
            probes=(locality_probe,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (locality_probe,)
            ),
        )


def test_preserve_baseline_is_a_real_noop_without_a_fabricated_program() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    probe = _probe(ontology, claim)
    no_op = CompiledTreatment(
        disposition=TreatmentDisposition.PRESERVE_BASELINE
    )
    registry = _compiler_registry(ontology, claim)
    registry.register(_Compiler(no_op), trust_anchor=_trust_anchor())

    treatment, receipt = registry.compile(
        compiler_id="compiler.sparse.v1",
        compiler_version="compiler-version.v1",
        ontology=ontology,
        claim=claim,
        probes=(probe,),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (probe,)
        ),
    )

    assert treatment.program is None
    assert treatment.recipe_ids == ()
    assert treatment.safe_payload()["program_payload_persisted"] is False
    assert receipt.compiler_target is CompilerTarget.NO_DIRECT_TREATMENT
    disguised_noop = replace(no_op, program=_program())
    assert "compiled_noop_contains_active_treatment" in disguised_noop.validate()


def test_evaluator_artifact_contract_contains_no_program() -> None:
    ontology = _ontology()
    claim = _claim(ontology)
    probe = _probe(ontology, claim)
    artifact = CompiledTreatment(
        disposition=TreatmentDisposition.EVALUATOR_ARTIFACT,
        evaluator_artifact_hash=_hash("frozen-evaluator-artifact"),
    )
    registry = _compiler_registry(ontology, claim)
    registry.register(_Compiler(artifact), trust_anchor=_trust_anchor())

    treatment, receipt = registry.compile(
        compiler_id="compiler.sparse.v1",
        compiler_version="compiler-version.v1",
        ontology=ontology,
        claim=claim,
        probes=(probe,),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (probe,)
        ),
    )

    assert treatment.validate() == ()
    assert treatment.program is None
    assert treatment.recipe_ids == ()
    assert receipt.compiler_target is CompilerTarget.EVALUATOR_ARTIFACT
    assert receipt.treatment_behavior_hash == artifact.evaluator_artifact_hash
    malformed = replace(artifact, program=_program())
    assert "compiled_evaluator_contains_program" in malformed.validate()


def test_compiler_registry_rejects_conflicting_implementation_binding() -> None:
    registry = HypothesisSpaceCompilerRegistry(
        probe_verifier_registry=ProbeVerifierRegistry()
    )
    no_op = CompiledTreatment(
        disposition=TreatmentDisposition.PRESERVE_BASELINE
    )
    registry.register(_Compiler(no_op), trust_anchor=_trust_anchor())

    with pytest.raises(PermissionError, match="registry conflict"):
        registry.register(
            _Compiler(
                no_op,
                implementation_hash=_hash("other-implementation"),
            ),
            trust_anchor=replace(
                _trust_anchor(),
                implementation_hash=_hash("other-implementation"),
            ),
        )


def test_compiler_registry_binds_an_independent_trust_anchor() -> None:
    registry = HypothesisSpaceCompilerRegistry(
        probe_verifier_registry=ProbeVerifierRegistry()
    )
    compiler = _Compiler(
        CompiledTreatment(
            disposition=TreatmentDisposition.PRESERVE_BASELINE
        )
    )
    anchor = _trust_anchor()
    with pytest.raises(TypeError, match="trust_anchor"):
        registry.register(compiler)  # type: ignore[call-arg]
    registry.register(compiler, trust_anchor=anchor)

    assert registry.require_trust_anchor(
        anchor.compiler_id, anchor.compiler_version
    ) == anchor
    with pytest.raises(PermissionError, match="does not match trust anchor"):
        registry.register(
            compiler,
            trust_anchor=replace(
                anchor,
                implementation_hash=_hash("untrusted-implementation"),
            ),
        )


def test_registered_compiler_identity_drift_fails_closed() -> None:
    registry = HypothesisSpaceCompilerRegistry(
        probe_verifier_registry=ProbeVerifierRegistry()
    )
    compiler = _Compiler(
        CompiledTreatment(
            disposition=TreatmentDisposition.PRESERVE_BASELINE
        )
    )
    anchor = _trust_anchor()
    registry.register(compiler, trust_anchor=anchor)
    compiler.primary_metric = "mutated_metric"
    ontology = _ontology()
    claim = _claim(ontology)

    with pytest.raises(PermissionError, match="no longer matches"):
        registry.compile(
            compiler_id=anchor.compiler_id,
            compiler_version=anchor.compiler_version,
            ontology=ontology,
            claim=claim,
            probes=(_probe(ontology, claim),),
            probe_evidence_bundles=(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("compiler_id", "compiler.forged.v1"),
        ("compiler_version", "compiler-version.forged"),
        ("compiler_implementation_hash", _hash("forged-compiler")),
        ("primary_metric", "forged_metric"),
    ],
)
def test_independent_anchor_rejects_compiler_receipt_tampering(
    field: str,
    value: str,
) -> None:
    ontology, claim, probe, treatment, receipt, trust_anchor = (
        _compile_active()
    )
    tampered = _with_expected_receipt_id(
        replace(receipt, **{field: value})
    )

    with pytest.raises(PermissionError, match="verification failed"):
        verify_compilation_receipt(
            tampered,
            ontology=ontology,
            claim=claim,
            probes=(probe,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (probe,)
            ),
            probe_verifier_registry=_probe_verifier_registry(
                ontology, claim.template_ids
            ),
            treatment=treatment,
            trust_anchor=trust_anchor,
        )


def test_compilation_receipt_id_is_content_bound_and_verified() -> None:
    ontology, claim, probe, treatment, receipt, trust_anchor = (
        _compile_active()
    )
    assert receipt.receipt_id == receipt.expected_receipt_id
    forged_id = replace(receipt, receipt_id="compilation.forged")

    with pytest.raises(PermissionError, match="verification failed"):
        verify_compilation_receipt(
            forged_id,
            ontology=ontology,
            claim=claim,
            probes=(probe,),
            probe_evidence_bundles=_evidence_for_probes(
                ontology, claim, (probe,)
            ),
            probe_verifier_registry=_probe_verifier_registry(
                ontology, claim.template_ids
            ),
            treatment=treatment,
            trust_anchor=trust_anchor,
        )


def test_active_treatment_requires_exact_recipe_action_semantics_binding() -> None:
    treatment = _active_treatment()
    assert treatment.validate() == ()
    binding = treatment.recipe_action_bindings[0]

    missing = replace(treatment, recipe_action_bindings=())
    assert (
        "compiled_treatment_recipe_action_coverage_invalid"
        in missing.validate()
    )
    wrong_recipe = replace(
        treatment,
        recipe_action_bindings=(
            replace(binding, recipe_id="recipe.other.v1"),
        ),
    )
    assert (
        "compiled_treatment_recipe_action_coverage_invalid"
        in wrong_recipe.validate()
    )
    wrong_action = replace(
        treatment,
        recipe_action_bindings=(
            replace(binding, action_id="missing-action"),
        ),
    )
    assert (
        "recipe_action_binding_action_missing"
        in wrong_action.validate()
    )
    wrong_semantics = replace(
        treatment,
        recipe_action_bindings=(
            replace(
                binding,
                action_semantics_hash=_hash("forged-action-semantics"),
            ),
        ),
    )
    assert (
        "recipe_action_binding_semantics_mismatch"
        in wrong_semantics.validate()
    )


def test_noop_and_evaluator_reject_recipe_action_bindings() -> None:
    binding = _active_treatment().recipe_action_bindings
    no_op = CompiledTreatment(
        disposition=TreatmentDisposition.PRESERVE_BASELINE,
        recipe_action_bindings=binding,
    )
    assert "compiled_noop_contains_active_treatment" in no_op.validate()

    evaluator = CompiledTreatment(
        disposition=TreatmentDisposition.EVALUATOR_ARTIFACT,
        evaluator_artifact_hash=_hash("evaluator"),
        recipe_action_bindings=binding,
    )
    assert "compiled_evaluator_contains_program" in evaluator.validate()


def test_compilation_receipt_binds_recipe_action_binding_hashes() -> None:
    ontology, claim, probe, treatment, receipt, trust_anchor = (
        _compile_active()
    )
    tampered = _with_expected_receipt_id(
        replace(
            receipt,
            recipe_action_binding_hashes=(
                _hash("forged-recipe-action-binding"),
            ),
        )
    )

    issues = tampered.validate(
        ontology=ontology,
        claim=claim,
        probes=(probe,),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (probe,)
        ),
        probe_verifier_registry=_probe_verifier_registry(
            ontology, claim.template_ids
        ),
        treatment=treatment,
        trust_anchor=trust_anchor,
    )
    assert "compilation_receipt_recipe_action_bindings_mismatch" in issues


def test_compilation_receipt_tampering_is_detected() -> None:
    ontology, claim, probe, treatment, receipt, trust_anchor = (
        _compile_active()
    )
    tampered = replace(
        receipt,
        treatment_behavior_hash=_hash("forged-treatment"),
    )

    issues = tampered.validate(
        ontology=ontology,
        claim=claim,
        probes=(probe,),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (probe,)
        ),
        probe_verifier_registry=_probe_verifier_registry(
            ontology, claim.template_ids
        ),
        treatment=treatment,
        trust_anchor=trust_anchor,
    )
    assert "compilation_receipt_behavior_hash_mismatch" in issues
    assert tampered.receipt_hash != receipt.receipt_hash


def test_ontology_and_compilation_hashes_are_order_deterministic() -> None:
    ontology = _two_template_ontology()
    reordered = replace(
        ontology,
        roots=tuple(reversed(ontology.roots)),
        templates=tuple(reversed(ontology.templates)),
        legacy_aliases=tuple(reversed(ontology.legacy_aliases)),
    )
    assert ontology.validate() == ()
    assert reordered.validate() == ()
    assert reordered.ontology_hash == ontology.ontology_hash
    assert reordered.safe_payload() == ontology.safe_payload()

    claim = _two_template_claim(ontology)
    locality_probe = _probe_for_template(ontology, claim, "locality.v1")
    sparsity_probe = _probe_for_template(ontology, claim, "sparsity.v1")
    treatment = _active_treatment()
    registry = _compiler_registry(ontology, claim)
    registry.register(
        _Compiler(treatment),
        trust_anchor=_trust_anchor(),
    )
    first_treatment, first_receipt = registry.compile(
        compiler_id="compiler.sparse.v1",
        compiler_version="compiler-version.v1",
        ontology=ontology,
        claim=claim,
        probes=(locality_probe, sparsity_probe),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (locality_probe, sparsity_probe)
        ),
    )
    second_treatment, second_receipt = registry.compile(
        compiler_id="compiler.sparse.v1",
        compiler_version="compiler-version.v1",
        ontology=ontology,
        claim=claim,
        probes=(sparsity_probe, locality_probe),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (sparsity_probe, locality_probe)
        ),
    )

    assert first_treatment == second_treatment
    assert first_receipt.safe_payload() == second_receipt.safe_payload()
    assert first_receipt.receipt_hash == second_receipt.receipt_hash


def test_compilation_receipt_primary_metric_tampering_is_detected() -> None:
    ontology, claim, probe, treatment, receipt, trust_anchor = (
        _compile_active()
    )
    tampered = replace(receipt, primary_metric="other-metric")

    issues = tampered.validate(
        ontology=ontology,
        claim=claim,
        probes=(probe,),
        probe_evidence_bundles=_evidence_for_probes(
            ontology, claim, (probe,)
        ),
        probe_verifier_registry=_probe_verifier_registry(
            ontology, claim.template_ids
        ),
        treatment=treatment,
        trust_anchor=trust_anchor,
    )
    assert "compilation_receipt_primary_metric_mismatch" in issues
    assert tampered.receipt_hash != receipt.receipt_hash
