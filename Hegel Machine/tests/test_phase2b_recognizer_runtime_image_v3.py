from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if SOURCE.as_posix() not in sys.path:
    sys.path.insert(0, SOURCE.as_posix())

from hegel_machine.phase2b_recognizer_runtime_cli_v2 import (
    RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2,
)
from hegel_machine.phase2b_runner import REQUIRED_RECOGNIZER_MODULES


TOOLS = ROOT / "tools"
if TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, TOOLS.as_posix())

import build_phase2b_recognizer_runtime_image_v3 as builder  # noqa: E402


IMAGE_DIRECTORY = ROOT / "docker" / "phase2b_recognizer_runtime_v3"
DOCKERFILE = IMAGE_DIRECTORY / "Dockerfile"
DOCKERIGNORE = IMAGE_DIRECTORY / "Dockerfile.dockerignore"
INITIALIZER = IMAGE_DIRECTORY / "hegel_machine_init.py"
ENTRYPOINT = IMAGE_DIRECTORY / "phase2b-recognizer"
OLD_BUILD_RECEIPT = (
    ROOT / "artifacts" / "phase2b_recognizer_runtime_image_build_v2.json"
)


def _module_source_paths() -> frozenset[str]:
    return frozenset(
        "src/" + module.replace(".", "/") + ".py"
        for module in REQUIRED_RECOGNIZER_MODULES
    )


def _copy_pairs() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("COPY "):
            continue
        parts = line.split()
        source = next(item for item in parts[1:] if not item.startswith("--"))
        pairs.append((parts[-1], source))
    return tuple(sorted(pairs))


def _independent_source_root() -> str:
    pairs = _copy_pairs()
    preimage = bytearray(
        b"HEGEL/PHASE2B/RECOGNIZER_RUNTIME_SOURCE_ROOT/V1\x00"
    )
    preimage.extend(struct.pack(">I", len(pairs)))
    for installed, context_path in pairs:
        payload = (ROOT / context_path).read_bytes()
        encoded = installed.encode("ascii")
        preimage.extend(struct.pack(">I", len(encoded)))
        preimage.extend(encoded)
        preimage.extend(struct.pack(">Q", len(payload)))
        preimage.extend(hashlib.sha256(payload).digest())
    return hashlib.sha256(preimage).hexdigest()


def test_v3_dockerfile_uses_pinned_offline_base_and_exact_28_entry_closure() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert source.startswith(f"FROM python@{builder.BASE_IMAGE_DIGEST_V3}\n")
    assert "\nRUN " not in source
    assert not any(
        token in source.casefold()
        for token in ("apt-get", "pip install", "curl ", "wget ", "git clone")
    )
    copied_modules = frozenset(
        match.group(1)
        for match in re.finditer(
            r"^COPY .*? (src/hegel_machine/[a-zA-Z0-9_]+\.py) ",
            source,
            flags=re.MULTILINE,
        )
    )
    assert copied_modules == _module_source_paths()
    assert len(copied_modules) == 26
    assert len(_copy_pairs()) == 28
    assert " src/hegel_machine/__init__.py " not in source
    assert "phase2b_scorer_bundle_v1.py" not in source
    assert "phase2b_unsealed_960_prediction_scoring_mechanics_v2.py" not in source


def test_v3_dockerignore_exposes_only_the_exact_30_context_files() -> None:
    lines = tuple(DOCKERIGNORE.read_text(encoding="utf-8").splitlines())
    assert lines[0] == "**"
    source_rules = frozenset(
        line.removeprefix("!")
        for line in lines
        if line.startswith("!src/hegel_machine/") and line.endswith(".py")
    )
    assert source_rules == _module_source_paths()
    assert "!src/hegel_machine/__init__.py" not in lines
    assert {
        "!docker/phase2b_recognizer_runtime_v3/Dockerfile",
        "!docker/phase2b_recognizer_runtime_v3/Dockerfile.dockerignore",
        "!docker/phase2b_recognizer_runtime_v3/hegel_machine_init.py",
        "!docker/phase2b_recognizer_runtime_v3/phase2b-recognizer",
    } <= set(lines)
    assert all("answer" not in line.casefold() for line in lines)
    assert all("scorer" not in line.casefold() for line in lines)


def test_v3_source_manifest_and_independent_role_domain_root_are_exact() -> None:
    pairs = _copy_pairs()
    assert pairs == tuple(
        (installed, context)
        for installed, context, _digest in builder.FROZEN_SOURCE_MANIFEST_V3
    )
    assert tuple(installed for installed, _context in pairs) == tuple(
        sorted(installed for installed, _context in pairs)
    )
    assert builder.RECOGNIZER_SOURCE_ROOT_DOMAIN_V3 == (
        b"HEGEL/PHASE2B/RECOGNIZER_RUNTIME_SOURCE_ROOT/V1\x00"
    )
    assert builder.RECOGNIZER_SOURCE_ROOT_DOMAIN_V3 != (
        b"HEGEL/PHASE2B/SCORER_RUNTIME_SOURCE_ROOT/V1\x00"
    )
    assert _independent_source_root() == (
        "d71f96926156b78bbd42d016b04c40aaa3e0abc5dad2510adc05c23e8dd54f54"
    )
    assert _independent_source_root() == builder.RECOGNIZER_SOURCE_ROOT_SHA256_V3
    for installed, context, expected_digest in builder.FROZEN_SOURCE_MANIFEST_V3:
        assert hashlib.sha256((ROOT / context).read_bytes()).hexdigest() == (
            expected_digest
        )
        assert installed.startswith("/opt/hegel/")


def test_v3_labels_bind_profile_authority_and_recognizer_root_without_claim_upgrade() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    for value in (
        builder.PROFILE_CONTENT_ID_V3,
        builder.PROFILE_CANONICAL_SHA256_V3,
        builder.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3,
        builder.RECOGNIZER_SOURCE_ROOT_SHA256_V3,
    ):
        assert value in source
    assert builder.FROZEN_IMAGE_LABELS_V3 == {
        "org.opencontainers.image.base.digest": builder.BASE_IMAGE_DIGEST_V3,
        "org.opencontainers.image.description": (
            builder.RECOGNIZER_RUNTIME_IMAGE_DESCRIPTION_V3
        ),
        "org.opencontainers.image.title": builder.RECOGNIZER_RUNTIME_IMAGE_TITLE_V3,
        "org.hegel-machine.phase2b.actor-profile.canonical-sha256": (
            builder.PROFILE_CANONICAL_SHA256_V3
        ),
        "org.hegel-machine.phase2b.actor-profile.content-id": (
            builder.PROFILE_CONTENT_ID_V3
        ),
        "org.hegel-machine.phase2b.owner-accepted-authority-requirement.id": (
            builder.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
        ),
        "org.hegel-machine.phase2b.recognizer-source-root.sha256": (
            builder.RECOGNIZER_SOURCE_ROOT_SHA256_V3
        ),
    }
    assert "Non-authoritative" in source
    assert "no actual-960 execution, scoring, gate, or C1 evidence" in source


def test_v3_default_runtime_config_is_fixed_and_buildkit_shape_is_exact_seven() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert "USER 65534:65534" in source
    assert 'ENTRYPOINT ["/opt/hegel/bin/phase2b-recognizer"]' in source
    assert 'CMD ["--input", "/phase2b/input", "--output", "/phase2b/output"]' in source
    assert "HEALTHCHECK" not in source
    assert "VOLUME" not in source
    assert builder.FROZEN_IMAGE_ARGS_ESCAPED_V3 is True
    assert builder.FROZEN_IMAGE_CONFIG_FIELDS_V3 == frozenset(
        {
            "ArgsEscaped",
            "Cmd",
            "Entrypoint",
            "Env",
            "Labels",
            "User",
            "WorkingDir",
        }
    )
    assert builder.FROZEN_ENTRYPOINT_V3 == (
        "/opt/hegel/bin/phase2b-recognizer",
    )
    assert builder.FROZEN_COMMAND_V3 == (
        "--input",
        "/phase2b/input",
        "--output",
        "/phase2b/output",
    )


def test_v3_initializer_is_inert_and_default_entrypoint_is_exact_executable() -> None:
    initializer_tree = ast.parse(INITIALIZER.read_text(encoding="utf-8"))
    assert tuple(
        node
        for node in ast.walk(initializer_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        )
    ) == ()
    entrypoint_source = ENTRYPOINT.read_text(encoding="utf-8")
    assert entrypoint_source.startswith("#!/usr/local/bin/python3\n")
    tree = ast.parse(entrypoint_source)
    imports = tuple(
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )
    assert imports == (
        ("hegel_machine.phase2b_recognizer_runtime_cli_v2", ("main",)),
    )
    assert stat.S_IMODE(ENTRYPOINT.stat().st_mode) == 0o755


def test_negative_missing_input_wire_matches_runtime_and_historical_v2_observation() -> None:
    expected = (
        json.dumps(
            {
                "disposition": "ABSTAIN",
                "reason": RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    assert expected == builder.NEGATIVE_SMOKE_STDERR_V3
    assert hashlib.sha256(expected).hexdigest() == (
        "6bf82d6b88c57325f3b3c67c1db820d861834b2f75beccd188f23fda0fa2b282"
    )
    historical = json.loads(OLD_BUILD_RECEIPT.read_bytes())
    # V2 recorded the line without its transport LF; V3 freezes the exact CLI wire.
    assert (
        historical["qualification"]["negative_smoke_stderr"].encode("ascii")
        + b"\n"
        == expected
    )


def test_v3_build_inputs_are_content_frozen_but_no_build_receipt_is_claimed_here() -> None:
    assert hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest() == (
        builder.DOCKERFILE_SHA256_V3
    )
    assert hashlib.sha256(DOCKERIGNORE.read_bytes()).hexdigest() == (
        builder.DOCKERIGNORE_SHA256_V3
    )
    assert hashlib.sha256(INITIALIZER.read_bytes()).hexdigest() == (
        builder.INITIALIZER_SHA256_V3
    )
    assert hashlib.sha256(ENTRYPOINT.read_bytes()).hexdigest() == (
        builder.ENTRYPOINT_SHA256_V3
    )
    assert builder.RECOGNIZER_RUNTIME_IMAGE_BUILD_CLAIM_LEVEL_V3 == (
        "NON_AUTHORITATIVE_LOCAL_DOCKER_IMAGE_BUILD_AND_NEGATIVE_SMOKE_ONLY"
    )
    assert builder.RECOGNIZER_RUNTIME_IMAGE_BUILD_DISPOSITION_V3 == (
        "LOCAL_RECOGNIZER_IMAGE_BUILD_AND_NEGATIVE_SMOKE_COMPLETE_NOT_ACTUAL_960"
    )
