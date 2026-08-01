from dataclasses import replace

import pytest

from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from hegel_machine.phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    ExecutionFreezeManifest,
    frozen_phase2b_protocol,
)
from hegel_machine.phase2b_runner import (
    ExternalRuntimeAttestation,
    REQUIRED_RECOGNIZER_MODULES,
    audit_recognizer_image_modules,
    build_oci_run_spec,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _freeze_manifest():
    protocol = frozen_phase2b_protocol()
    exact_freeze = frozen_phase2b_exact_freeze()
    spec_id_by_kind = {
        BaselineKind(spec.baseline_id): spec.content_id
        for spec in exact_freeze.baselines
    }
    baselines = tuple(
        BaselineRegistration(
            kind=kind,
            baseline_spec_id=spec_id_by_kind[kind],
            implementation_id=f"baseline_{kind.value}",
            artifact_sha256=digest,
            frozen_before_holdout_generation=True,
        )
        for kind, digest in zip(BaselineKind, (SHA_A, SHA_B, SHA_C), strict=True)
    )
    return ExecutionFreezeManifest(
        protocol_id=protocol.protocol_id,
        exact_freeze_id=exact_freeze.freeze_id,
        git_commit="1" * 40,
        recognizer_image_digest="sha256:" + SHA_A,
        configuration_sha256=SHA_B,
        theory_version_id="theory_v1",
        adapter_implementation_sha256=SHA_C,
        selector_implementation_sha256=SHA_D,
        verifier_registry_sha256=SHA_A,
        baseline_registrations=baselines,
        isolation_profile_id=protocol.isolation_profile.profile_id,
    )


def _run_spec():
    return build_oci_run_spec(
        freeze_manifest=_freeze_manifest(),
        input_host_directory="/sealed/phase2b/input",
        output_host_directory="/sealed/phase2b/output",
        repository_host_directory="/workspace/assumption-agent",
        input_manifest_sha256=SHA_D,
    )


def test_oci_run_spec_has_no_network_read_only_root_and_only_two_mounts():
    spec = _run_spec()
    argv = spec.argv("podman")
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    mounts = tuple(item for item in argv if item.startswith("--mount="))
    assert len(mounts) == 2
    assert "dst=/phase2b/input,readonly" in mounts[0]
    assert "dst=/phase2b/output" in mounts[1]
    assert all("assumption-agent" not in item for item in mounts)
    assert all("answer" not in item.casefold() for item in argv)
    with pytest.raises(ValueError, match="frozen contract"):
        replace(spec, entrypoint=("python", "-c", "pass"))


def test_run_spec_rejects_repository_or_overlapping_mounts():
    manifest = _freeze_manifest()
    with pytest.raises(ValueError, match="outside the repository"):
        build_oci_run_spec(
            freeze_manifest=manifest,
            input_host_directory="/workspace/assumption-agent/sealed-input",
            output_host_directory="/sealed/output",
            repository_host_directory="/workspace/assumption-agent",
            input_manifest_sha256=SHA_D,
        )
    with pytest.raises(ValueError, match="cannot overlap"):
        build_oci_run_spec(
            freeze_manifest=manifest,
            input_host_directory="/sealed/run",
            output_host_directory="/sealed/run/output",
            repository_host_directory="/workspace/assumption-agent",
            input_manifest_sha256=SHA_D,
        )


def test_recognizer_image_inventory_is_allowlisted_and_excludes_fixture_code():
    clean = audit_recognizer_image_modules(tuple(sorted(REQUIRED_RECOGNIZER_MODULES)))
    assert clean.passed
    contaminated = audit_recognizer_image_modules(
        tuple(sorted(REQUIRED_RECOGNIZER_MODULES | {"hegel_machine.phase2_exit"}))
    )
    assert not contaminated.passed
    assert contaminated.forbidden_modules_present == ("hegel_machine.phase2_exit",)
    assert contaminated.unexpected_modules_present == (
        "hegel_machine.phase2_exit",
    )
    oracle = audit_recognizer_image_modules(
        tuple(sorted(REQUIRED_RECOGNIZER_MODULES | {"evil.answer_oracle"}))
    )
    assert not oracle.passed
    assert oracle.unexpected_modules_present == ("evil.answer_oracle",)
    incomplete = audit_recognizer_image_modules(("hegel_machine.hashing",))
    assert not incomplete.passed
    assert "hegel_machine.phase2b_selector" in incomplete.missing_required_modules


def test_external_attestation_is_bound_to_run_and_fails_closed():
    spec = _run_spec()
    attestation = ExternalRuntimeAttestation(
        run_spec_id=spec.run_spec_id,
        runtime_name="podman",
        runtime_version="5.0.0",
        external_attestor_id="independent_custodian",
        detached_attestation_sha256=SHA_A,
        prediction_archive_sha256=SHA_B,
        audit_archive_sha256=SHA_C,
        freeze_manifest_id=spec.freeze_manifest_id,
        protocol_id=spec.protocol_id,
        input_manifest_sha256=spec.input_manifest_sha256,
        prediction_case_count=720,
        exit_code=0,
        timed_out=False,
        output_size_bytes=1024,
        observed_network_disabled=True,
        observed_read_only_root=True,
        observed_repository_absent=True,
        observed_answer_manifest_absent=True,
    )
    attestation.validate(spec)
    with pytest.raises(ValueError, match="missing isolation"):
        replace(attestation, observed_network_disabled=False).validate(spec)
    with pytest.raises(ValueError, match="did not complete"):
        replace(attestation, exit_code=1).validate(spec)
    with pytest.raises(ValueError, match="different run"):
        replace(attestation, run_spec_id="wrong").validate(spec)
    with pytest.raises(ValueError, match="exactly 720"):
        replace(attestation, prediction_case_count=0).validate(spec)
    with pytest.raises(ValueError, match="empty output"):
        replace(attestation, output_size_bytes=0).validate(spec)
