from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

from hegel_machine.phase2b_runner import REQUIRED_RECOGNIZER_MODULES


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRECTORY = ROOT / "docker" / "phase2b_recognizer_runtime_v2"
DOCKERFILE = IMAGE_DIRECTORY / "Dockerfile"
DOCKERIGNORE = IMAGE_DIRECTORY / "Dockerfile.dockerignore"
INITIALIZER = IMAGE_DIRECTORY / "hegel_machine_init.py"
ENTRYPOINT = IMAGE_DIRECTORY / "phase2b-recognizer"
BUILD_RECEIPT = ROOT / "artifacts" / "phase2b_recognizer_runtime_image_build_v2.json"
BASE_DIGEST = (
    "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)


def _module_source_paths() -> frozenset[str]:
    return frozenset(
        "src/" + module.replace(".", "/") + ".py"
        for module in REQUIRED_RECOGNIZER_MODULES
    )


def test_dockerfile_uses_pinned_offline_base_and_exact_module_closure() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert source.startswith(f"FROM python@{BASE_DIGEST}\n")
    assert not any(
        token in source.casefold()
        for token in ("apt-get", "pip install", "curl ", "wget ", "git clone")
    )
    copied = frozenset(
        match.group(1)
        for match in re.finditer(
            r"^COPY .*? (src/hegel_machine/[a-zA-Z0-9_]+\.py) ",
            source,
            flags=re.MULTILINE,
        )
    )
    assert copied == _module_source_paths()
    assert " src/hegel_machine/__init__.py " not in source
    assert "\nRUN " not in source


def test_dockerignore_exposes_only_the_exact_build_inputs() -> None:
    lines = tuple(
        line
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line and line != "**"
    )
    source_rules = frozenset(
        line.removeprefix("!")
        for line in lines
        if line.startswith("!src/hegel_machine/") and line.endswith(".py")
    )
    assert source_rules == _module_source_paths()
    assert "!src/hegel_machine/__init__.py" not in lines
    assert all("answer" not in line.casefold() for line in lines)
    assert all("scor" not in line.casefold() for line in lines)


def test_minimal_initializer_is_inert_and_entrypoint_is_fixed() -> None:
    initializer_tree = ast.parse(INITIALIZER.read_text(encoding="utf-8"))
    initializer_imports = tuple(
        node
        for node in ast.walk(initializer_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        )
    )
    assert initializer_imports == ()

    entrypoint_source = ENTRYPOINT.read_text(encoding="utf-8")
    assert entrypoint_source.startswith("#!/usr/local/bin/python3\n")
    entrypoint_tree = ast.parse(entrypoint_source)
    imports = tuple(
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(entrypoint_tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )
    assert imports == (
        ("hegel_machine.phase2b_recognizer_runtime_cli_v2", ("main",)),
    )


def test_runtime_command_is_nonroot_readonly_and_not_an_evaluator() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert "USER 65534:65534" in source
    assert 'ENTRYPOINT ["/opt/hegel/bin/phase2b-recognizer"]' in source
    assert 'CMD ["--input", "/phase2b/input", "--output", "/phase2b/output"]' in source
    assert "HEALTHCHECK" not in source
    assert "VOLUME" not in source
    assert "no answer, scorer, or actual-960 evidence" in source
    forbidden = (
        "phase2b_unsealed_960_prediction_scoring_mechanics_v2.py",
        "phase2b_actual_unsealed_960",
        "phase2b_evaluator",
        "phase2_exit",
    )
    assert all(name not in source for name in forbidden)


def test_checked_build_receipt_recomputes_inputs_and_preserves_nonclaims() -> None:
    payload = BUILD_RECEIPT.read_bytes()
    receipt = json.loads(payload)
    assert payload == (
        json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("ascii")
    assert receipt["claim_level"] == (
        "NON_AUTHORITATIVE_LOCAL_DOCKER_IMAGE_BUILD_AND_NEGATIVE_SMOKE_ONLY"
    )
    assert receipt["disposition"] == (
        "LOCAL_IMAGE_BUILD_AND_NEGATIVE_SMOKE_COMPLETE_NOT_ACTUAL_REPLAY"
    )
    assert receipt["build"]["context_mode"] == "EXACT_29_FILE_USTAR_STREAM"
    assert receipt["build"]["dockerfile_run_instruction_count"] == 0
    assert receipt["build"]["custom_build_seccomp_applied"] is False
    assert receipt["qualification"]["runtime_custom_seccomp_applied"] is True
    assert receipt["qualification"]["runtime_import_closure_exact"] is True
    assert receipt["qualification"]["negative_smoke_exit_code"] == 2

    build_inputs = receipt["build_inputs"]
    paths = {
        "dockerfile_sha256": DOCKERFILE,
        "dockerignore_sha256": DOCKERIGNORE,
        "minimal_initializer_sha256": INITIALIZER,
        "entrypoint_sha256": ENTRYPOINT,
        "runtime_seccomp_sha256": (
            ROOT / "config" / "phase3_internal_actor_seccomp_v1.json"
        ),
    }
    for field, path in paths.items():
        assert build_inputs[field] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozenset(build_inputs["modules"]) == REQUIRED_RECOGNIZER_MODULES
    for module, digest in build_inputs["modules"].items():
        path = ROOT / "src" / (module.replace(".", "/") + ".py")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    claims = receipt["claims"]
    assert {
        key for key, value in claims.items() if value is True
    } == {
        "exact_source_context_verified",
        "local_image_built",
        "minimal_package_boundary_verified",
        "profile_control_negative_smoke_executed",
        "runtime_image_config_verified",
        "runtime_module_closure_verified",
    }
    assert all(
        claims[name] is False
        for name in (
            "actual_input_archive_supplied",
            "execution_freeze_manifest_supplied",
            "prediction_archive_materialized",
            "fresh_role_keys_generated",
            "signed_execution_registry_materialized",
            "custodian_ceremony_executed",
            "store_receipt_materialized",
            "signed_monotonic_time_materialized",
            "actual_960_recognizer_executed",
            "scoring_performed",
            "formal_gate_evaluation_performed",
            "effect_evidence",
            "formal_c1_report_verified",
        )
    )

    receipt_id = receipt.pop("receipt_id")
    wire = json.dumps(
        receipt,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert receipt_id == (
        "phase2b_recognizer_runtime_image_build_receipt_v2_"
        + hashlib.sha256(
            b"hegel-machine-phase2b-recognizer-runtime-image-build-receipt-v2\0"
            + wire
        ).hexdigest()
    )
