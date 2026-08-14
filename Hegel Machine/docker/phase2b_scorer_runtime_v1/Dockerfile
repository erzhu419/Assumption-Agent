FROM python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3

ENV HOME=/nonexistent \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/hegel/src

WORKDIR /opt/hegel

COPY --chown=65534:65534 src/hegel_machine/bootstrap.py /opt/hegel/src/hegel_machine/bootstrap.py
COPY --chown=65534:65534 src/hegel_machine/hashing.py /opt/hegel/src/hegel_machine/hashing.py
COPY --chown=65534:65534 src/hegel_machine/laws.py /opt/hegel/src/hegel_machine/laws.py
COPY --chown=65534:65534 src/hegel_machine/milestones.py /opt/hegel/src/hegel_machine/milestones.py
COPY --chown=65534:65534 src/hegel_machine/ontology.py /opt/hegel/src/hegel_machine/ontology.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_actual_unsealed_960_replay_input_contract_v2.py /opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_replay_input_contract_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2.py /opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2.py /opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py /opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2.py /opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_adapter.py /opt/hegel/src/hegel_machine/phase2b_adapter.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_c1_container_actor_profile_v1.py /opt/hegel/src/hegel_machine/phase2b_c1_container_actor_profile_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_c1_container_actor_profile_v2.py /opt/hegel/src/hegel_machine/phase2b_c1_container_actor_profile_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_covert_audit_v1.py /opt/hegel/src/hegel_machine/phase2b_covert_audit_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_exact_bridge_v1.py /opt/hegel/src/hegel_machine/phase2b_exact_bridge_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py /opt/hegel/src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_exact_transform_semantics_v1.py /opt/hegel/src/hegel_machine/phase2b_exact_transform_semantics_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_formal_unsealed_prediction_scoring_contract_v2.py /opt/hegel/src/hegel_machine/phase2b_formal_unsealed_prediction_scoring_contract_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_freeze_v1.py /opt/hegel/src/hegel_machine/phase2b_freeze_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_protocol.py /opt/hegel/src/hegel_machine/phase2b_protocol.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_projection_compiler.py /opt/hegel/src/hegel_machine/phase2b_projection_compiler.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_input_archive_v1.py /opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_input_archive_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_prediction_archive_v1.py /opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_prediction_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_runner.py /opt/hegel/src/hegel_machine/phase2b_runner.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_scorer_bundle_v1.py /opt/hegel/src/hegel_machine/phase2b_scorer_bundle_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_scorer_runtime_cli_v1.py /opt/hegel/src/hegel_machine/phase2b_scorer_runtime_cli_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_selector.py /opt/hegel/src/hegel_machine/phase2b_selector.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_strict_recognizer_cli_v2.py /opt/hegel/src/hegel_machine/phase2b_strict_recognizer_cli_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_batch_v1.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_batch_v2.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_typed_replay_v1.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_v1.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_uncertainty_compiler.py /opt/hegel/src/hegel_machine/phase2b_uncertainty_compiler.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_unsealed_960_available_gate_mechanics_v2.py /opt/hegel/src/hegel_machine/phase2b_unsealed_960_available_gate_mechanics_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py /opt/hegel/src/hegel_machine/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_unsealed_prediction_evaluator_v1.py /opt/hegel/src/hegel_machine/phase2b_unsealed_prediction_evaluator_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_unsealed_prediction_evaluator_v2.py /opt/hegel/src/hegel_machine/phase2b_unsealed_prediction_evaluator_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_wire.py /opt/hegel/src/hegel_machine/phase2b_wire.py
COPY --chown=65534:65534 src/hegel_machine/phase3_dsl_v1.py /opt/hegel/src/hegel_machine/phase3_dsl_v1.py
COPY --chown=65534:65534 src/hegel_machine/schema.py /opt/hegel/src/hegel_machine/schema.py

# The package boundary and entrypoint are inert source files copied without a
# build-time interpreter or package-manager step.
COPY --chown=65534:65534 docker/phase2b_scorer_runtime_v1/hegel_machine_init.py /opt/hegel/src/hegel_machine/__init__.py
COPY --chown=65534:65534 docker/phase2b_scorer_runtime_v1/phase2b-scorer /opt/hegel/bin/phase2b-scorer

USER 65534:65534

LABEL org.opencontainers.image.title="Hegel Machine Phase2B V1 scorer runtime" \
      org.opencontainers.image.description="Non-authoritative local Docker scorer mechanics; no actual scoring, formal gate, or C1 evidence" \
      org.opencontainers.image.base.digest="sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3" \
      org.hegel-machine.phase2b.actor-profile.content-id="phase2b_c1_container_actor_profile_v2_b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906" \
      org.hegel-machine.phase2b.actor-profile.canonical-sha256="8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed" \
      org.hegel-machine.phase2b.scoring-source.sha256="c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11" \
      org.hegel-machine.phase2b.scorer-source-root.sha256="34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"

ENTRYPOINT ["/opt/hegel/bin/phase2b-scorer"]
CMD ["--input", "/phase2b/input/phase2b_scorer_bundle_v1.bin", "--output", "/phase2b/output"]
