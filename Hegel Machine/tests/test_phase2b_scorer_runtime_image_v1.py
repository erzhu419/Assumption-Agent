from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys

import hegel_machine.phase2b_scorer_host_executor_v1 as executor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "hegel_machine"
IMAGE = ROOT / "docker" / "phase2b_scorer_runtime_v1"
DOCKERFILE = IMAGE / "Dockerfile"
DOCKERIGNORE = IMAGE / "Dockerfile.dockerignore"
INITIALIZER = IMAGE / "hegel_machine_init.py"
ENTRYPOINT = IMAGE / "phase2b-scorer"
BASE_DIGEST = (
    "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
PROFILE_CONTENT_ID = (
    "phase2b_c1_container_actor_profile_v2_"
    "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
)
PROFILE_CANONICAL_SHA256 = (
    "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
)
SCORING_SHA256 = (
    "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
)
SCORER_SOURCE_ROOT_SHA256 = (
    "34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"
)


def _relative_dependencies(filename: str) -> frozenset[str]:
    seen: set[str] = set()
    pending = [filename]
    while pending:
        selected = pending.pop()
        if selected in seen:
            continue
        path = SOURCE_ROOT / selected
        assert path.is_file(), selected
        seen.add(selected)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            candidates: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module is not None:
                    candidates.append(node.module + ".py")
                else:
                    candidates.extend(alias.name + ".py" for alias in node.names)
            for candidate in candidates:
                if (SOURCE_ROOT / candidate).is_file() and candidate not in seen:
                    pending.append(candidate)
    return frozenset(seen)


def _copied_modules() -> frozenset[str]:
    return frozenset(
        match.group(1)
        for match in re.finditer(
            r"^COPY .*? src/hegel_machine/([A-Za-z0-9_]+\.py) ",
            DOCKERFILE.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def _installed_payloads() -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for installed in executor.SCORER_SOURCE_ROOT_ORDERED_ENTRY_PATHS_V1:
        if installed == "/opt/hegel/bin/phase2b-scorer":
            source = ENTRYPOINT
        elif installed == "/opt/hegel/src/hegel_machine/__init__.py":
            source = INITIALIZER
        else:
            source = SOURCE_ROOT / Path(installed).name
        values[installed] = source.read_bytes()
    return values


def _independent_source_root(payloads: dict[str, bytes]) -> str:
    preimage = bytearray(b"HEGEL/PHASE2B/SCORER_RUNTIME_SOURCE_ROOT/V1\x00")
    preimage.extend(struct.pack(">I", len(payloads)))
    for path, payload in payloads.items():
        encoded = path.encode("ascii")
        preimage.extend(struct.pack(">I", len(encoded)))
        preimage.extend(encoded)
        preimage.extend(struct.pack(">Q", len(payload)))
        preimage.extend(hashlib.sha256(payload).digest())
    return hashlib.sha256(preimage).hexdigest()


def test_dockerfile_is_pinned_no_run_nonroot_and_exact_closure() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert source.startswith(f"FROM python@{BASE_DIGEST}\n")
    assert "\nRUN " not in source
    assert not any(
        token in source.casefold()
        for token in ("apt-get", "pip install", "curl ", "wget ", "git clone")
    )
    assert _copied_modules() == _relative_dependencies(
        "phase2b_scorer_runtime_cli_v1.py"
    )
    assert " src/hegel_machine/__init__.py " not in source
    assert "USER 65534:65534" in source
    assert 'ENTRYPOINT ["/opt/hegel/bin/phase2b-scorer"]' in source
    assert (
        'CMD ["--input", "/phase2b/input/phase2b_scorer_bundle_v1.bin", '
        '"--output", "/phase2b/output"]'
    ) in source
    assert "VOLUME" not in source and "EXPOSE" not in source


def test_buildkit_exec_form_image_config_normalization_is_exact_seven() -> None:
    assert executor.SCORER_RUNTIME_IMAGE_ARGS_ESCAPED_V1 is True
    assert executor.SCORER_RUNTIME_IMAGE_CONFIG_FIELDS_V1 == {
        "ArgsEscaped",
        "Cmd",
        "Entrypoint",
        "Env",
        "Labels",
        "User",
        "WorkingDir",
    }
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["/opt/hegel/bin/phase2b-scorer"]' in source
    assert (
        'CMD ["--input", "/phase2b/input/phase2b_scorer_bundle_v1.bin", '
        '"--output", "/phase2b/output"]'
    ) in source


def test_image_labels_bind_profile_scoring_source_and_nonclaims() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert PROFILE_CONTENT_ID in source
    assert PROFILE_CANONICAL_SHA256 in source
    assert SCORING_SHA256 in source
    assert SCORER_SOURCE_ROOT_SHA256 in source
    assert (
        "Non-authoritative local Docker scorer mechanics; no actual scoring, "
        "formal gate, or C1 evidence"
    ) in source


def test_dockerignore_is_exact_same_source_closure() -> None:
    lines = tuple(DOCKERIGNORE.read_text(encoding="utf-8").splitlines())
    exposed = frozenset(
        Path(line.removeprefix("!")).name
        for line in lines
        if line.startswith("!src/hegel_machine/") and line.endswith(".py")
    )
    assert exposed == _copied_modules()
    assert "!src/hegel_machine/__init__.py" not in lines
    assert lines[0] == "**"
    assert {
        "!docker/phase2b_scorer_runtime_v1/Dockerfile",
        "!docker/phase2b_scorer_runtime_v1/Dockerfile.dockerignore",
        "!docker/phase2b_scorer_runtime_v1/hegel_machine_init.py",
        "!docker/phase2b_scorer_runtime_v1/phase2b-scorer",
    } <= set(lines)


def test_source_root_wire_uses_exact_installed_paths_and_payload_bytes() -> None:
    payloads = _installed_payloads()
    assert tuple(payloads) == tuple(sorted(payloads))
    assert len(payloads) == len(_copied_modules()) + 2
    assert tuple(payloads) == executor.SCORER_SOURCE_ROOT_ORDERED_ENTRY_PATHS_V1
    independently_computed = _independent_source_root(payloads)
    assert independently_computed == SCORER_SOURCE_ROOT_SHA256
    assert independently_computed == executor.SCORER_SOURCE_ROOT_SHA256_V1
    assert (
        executor.compute_scorer_source_root_sha256_v1(payloads)
        == independently_computed
    )


def test_inert_initializer_and_fixed_executable_entrypoint() -> None:
    initializer_tree = ast.parse(INITIALIZER.read_text(encoding="utf-8"))
    assert tuple(
        node
        for node in ast.walk(initializer_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        )
    ) == ()
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/local/bin/python3\n")
    tree = ast.parse(source)
    imports = tuple(
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )
    assert imports == (("hegel_machine.phase2b_scorer_runtime_cli_v1", ("main",)),)
    assert stat.S_IMODE(ENTRYPOINT.stat().st_mode) == 0o755


def test_fresh_inert_package_imports_exact_scorer_closure(
    tmp_path: Path,
) -> None:
    isolated = tmp_path / "isolated"
    package = isolated / "hegel_machine"
    package.mkdir(parents=True)
    shutil.copy2(INITIALIZER, package / "__init__.py")
    for filename in _copied_modules():
        shutil.copy2(SOURCE_ROOT / filename, package / filename)
    command = (
        sys.executable,
        "-I",
        "-c",
        (
            "import sys;"
            f"sys.path.insert(0,{isolated.as_posix()!r});"
            "import hegel_machine.phase2b_scorer_runtime_cli_v1 as value;"
            "assert value.PHASE2B_SCORER_BUNDLE_V1_FILENAME == "
            "'phase2b_scorer_bundle_v1.bin'"
        ),
    )
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout == b"" and completed.stderr == b""
