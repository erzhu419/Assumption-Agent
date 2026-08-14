from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

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

EXPECTED_V2_RECOGNIZER_RUNTIME_MODULES = frozenset(
    {
        "hegel_machine.bootstrap",
        "hegel_machine.hashing",
        "hegel_machine.laws",
        "hegel_machine.milestones",
        "hegel_machine.ontology",
        "hegel_machine.phase2b_adapter",
        "hegel_machine.phase2b_exact_bridge_v1",
        "hegel_machine.phase2b_exact_derived_witness_bridge_v1",
        "hegel_machine.phase2b_exact_transform_semantics_v1",
        "hegel_machine.phase2b_freeze_v1",
        "hegel_machine.phase2b_protocol",
        "hegel_machine.phase2b_recognizer_input_archive_v2",
        "hegel_machine.phase2b_recognizer_prediction_archive_v2",
        "hegel_machine.phase2b_recognizer_prediction_v2",
        "hegel_machine.phase2b_recognizer_runtime_cli_v2",
        "hegel_machine.phase2b_runner",
        "hegel_machine.phase2b_trusted_wire_batch_v1",
        "hegel_machine.phase2b_trusted_wire_batch_v2",
        "hegel_machine.phase2b_trusted_wire_typed_authority_v1",
        "hegel_machine.phase2b_trusted_wire_typed_authority_v2",
        "hegel_machine.phase2b_trusted_wire_typed_replay_v2",
        "hegel_machine.phase2b_trusted_wire_v1",
        "hegel_machine.phase2b_uncertainty_compiler",
        "hegel_machine.phase2b_wire",
        "hegel_machine.phase3_dsl_v1",
        "hegel_machine.schema",
    }
)


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
    argv = spec.argv("/usr/bin/docker")
    assert argv == (
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "run",
        "--rm",
        "--no-healthcheck",
        "--pull=never",
        "--network=none",
        "--ipc=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        (
            "--security-opt=seccomp=/workspace/assumption-agent/"
            "config/phase3_internal_actor_seccomp_v1.json"
        ),
        "--entrypoint=/opt/hegel/bin/phase2b-recognizer",
        "--cpus=2.0",
        "--memory=2g",
        "--pids-limit=128",
        "--memory-swap=2g",
        "--ulimit=nofile=256:256",
        "--env=PYTHONHASHSEED=0",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONPATH=/opt/hegel/src",
        "--env=HOME=/nonexistent",
        "--env=NO_PROXY=*",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=268435456",
        (
            "--mount=type=bind,src=/sealed/phase2b/input,"
            "dst=/phase2b/input,readonly"
        ),
        (
            "--mount=type=bind,src=/sealed/phase2b/output,"
            "dst=/phase2b/output"
        ),
        "sha256:" + SHA_A,
        "--input",
        "/phase2b/input",
        "--output",
        "/phase2b/output",
    )
    assert spec.seccomp_profile_sha256 == (
        "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
    )
    mounts = tuple(item for item in argv if item.startswith("--mount="))
    assert len(mounts) == 2
    assert "dst=/phase2b/input,readonly" in mounts[0]
    assert "dst=/phase2b/output" in mounts[1]
    assert all("assumption-agent" not in item for item in mounts)
    assert all("answer" not in item.casefold() for item in argv)
    for forbidden_runtime in ("docker", "podman", "/usr/bin/podman"):
        with pytest.raises(ValueError, match="exact /usr/bin/docker"):
            spec.argv(forbidden_runtime)
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
    assert REQUIRED_RECOGNIZER_MODULES == EXPECTED_V2_RECOGNIZER_RUNTIME_MODULES
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
    assert (
        "hegel_machine.phase2b_recognizer_runtime_cli_v2"
        in incomplete.missing_required_modules
    )
    scorer = audit_recognizer_image_modules(
        tuple(
            sorted(
                REQUIRED_RECOGNIZER_MODULES
                | {
                    "hegel_machine."
                    "phase2b_unsealed_960_prediction_scoring_mechanics_v2"
                }
            )
        )
    )
    assert not scorer.passed
    assert scorer.forbidden_modules_present == (
        "hegel_machine.phase2b_unsealed_960_prediction_scoring_mechanics_v2",
    )


def test_required_inventory_matches_fresh_runtime_import_closure_with_minimal_initializer():
    source_directory = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_directory)
    environment["HEGEL_RECOGNIZER_SOURCE"] = str(source_directory)
    script = """
import json
import os
import sys
import types

# The repository's eager package initializer imports code that is deliberately
# forbidden in the recognizer image.  The image must install an equivalent
# minimal initializer; this stub exercises precisely that image contract.
package = types.ModuleType('hegel_machine')
package.__package__ = 'hegel_machine'
package.__path__ = [
    os.path.join(os.environ['HEGEL_RECOGNIZER_SOURCE'], 'hegel_machine')
]
sys.modules['hegel_machine'] = package

import hegel_machine.phase2b_recognizer_runtime_cli_v2
from hegel_machine.phase2b_runner import REQUIRED_RECOGNIZER_MODULES
loaded = {name for name in sys.modules if name.startswith('hegel_machine.')}
print(json.dumps({
    'minimal_initializer': getattr(sys.modules['hegel_machine'], '__file__', None) is None,
    'missing': sorted(REQUIRED_RECOGNIZER_MODULES - loaded),
    'unexpected': sorted(loaded - REQUIRED_RECOGNIZER_MODULES),
}, sort_keys=True))
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=source_directory.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "minimal_initializer": True,
        "missing": [],
        "unexpected": [],
    }


def test_external_attestation_is_bound_to_run_and_fails_closed():
    spec = _run_spec()
    attestation = ExternalRuntimeAttestation(
        run_spec_id=spec.run_spec_id,
        runtime_name="docker",
        runtime_version="5.0.0",
        external_attestor_id="independent_custodian",
        detached_attestation_sha256=SHA_A,
        prediction_archive_sha256=SHA_B,
        audit_archive_sha256=SHA_C,
        freeze_manifest_id=spec.freeze_manifest_id,
        protocol_id=spec.protocol_id,
        input_manifest_sha256=spec.input_manifest_sha256,
        prediction_case_count=960,
        exit_code=0,
        timed_out=False,
        output_size_bytes=1024,
        observed_network_disabled=True,
        observed_read_only_root=True,
        observed_repository_absent=True,
        observed_answer_manifest_absent=True,
    )
    attestation.validate(spec)
    with pytest.raises(ValueError, match="frozen Docker runtime"):
        replace(attestation, runtime_name="podman").validate(spec)
    with pytest.raises(ValueError, match="missing isolation"):
        replace(attestation, observed_network_disabled=False).validate(spec)
    with pytest.raises(ValueError, match="did not complete"):
        replace(attestation, exit_code=1).validate(spec)
    with pytest.raises(ValueError, match="different run"):
        replace(attestation, run_spec_id="wrong").validate(spec)
    with pytest.raises(ValueError, match="exactly 960"):
        replace(attestation, prediction_case_count=720).validate(spec)
    with pytest.raises(ValueError, match="empty output"):
        replace(attestation, output_size_bytes=0).validate(spec)
