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
COPY --chown=65534:65534 src/hegel_machine/phase2b_adapter.py /opt/hegel/src/hegel_machine/phase2b_adapter.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_exact_bridge_v1.py /opt/hegel/src/hegel_machine/phase2b_exact_bridge_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py /opt/hegel/src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_exact_transform_semantics_v1.py /opt/hegel/src/hegel_machine/phase2b_exact_transform_semantics_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_freeze_v1.py /opt/hegel/src/hegel_machine/phase2b_freeze_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_protocol.py /opt/hegel/src/hegel_machine/phase2b_protocol.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_input_archive_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_prediction_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py /opt/hegel/src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_runner.py /opt/hegel/src/hegel_machine/phase2b_runner.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_batch_v1.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_batch_v2.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_trusted_wire_v1.py /opt/hegel/src/hegel_machine/phase2b_trusted_wire_v1.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_uncertainty_compiler.py /opt/hegel/src/hegel_machine/phase2b_uncertainty_compiler.py
COPY --chown=65534:65534 src/hegel_machine/phase2b_wire.py /opt/hegel/src/hegel_machine/phase2b_wire.py
COPY --chown=65534:65534 src/hegel_machine/phase3_dsl_v1.py /opt/hegel/src/hegel_machine/phase3_dsl_v1.py
COPY --chown=65534:65534 src/hegel_machine/schema.py /opt/hegel/src/hegel_machine/schema.py

# The repository initializer eagerly imports unrelated Phase-3 modules.  The
# recognizer image instead carries this deliberately inert package boundary.
COPY --chown=65534:65534 docker/phase2b_recognizer_runtime_v2/hegel_machine_init.py /opt/hegel/src/hegel_machine/__init__.py
COPY --chown=65534:65534 docker/phase2b_recognizer_runtime_v2/phase2b-recognizer /opt/hegel/bin/phase2b-recognizer

USER 65534:65534

LABEL org.opencontainers.image.title="Hegel Machine Phase2B V2 recognizer runtime" \
      org.opencontainers.image.description="Non-authoritative owner-accepted Docker recognizer mechanics; no answer, scorer, or actual-960 evidence" \
      org.opencontainers.image.base.digest="sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"

ENTRYPOINT ["/opt/hegel/bin/phase2b-recognizer"]
CMD ["--input", "/phase2b/input", "--output", "/phase2b/output"]
