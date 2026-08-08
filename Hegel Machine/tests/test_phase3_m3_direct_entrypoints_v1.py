from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    PROJECT_ROOT / "src/hegel_machine/phase3_m3_start_entrypoint_v1.py",
    PROJECT_ROOT
    / "src/hegel_machine/phase3_m3_formal_execution_entrypoint_v1.py",
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_direct_entrypoint_requires_isolated_source_only_launch(
    entrypoint: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="hegel-m3-entrypoint-cache-", dir="/tmp"
    ) as raw:
        cache = Path(raw).resolve()
        cache.chmod(0o700)
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-X",
                f"pycache_prefix={cache}",
                entrypoint.as_posix(),
                "--help",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
        )
        assert completed.returncode == 0, completed.stderr.decode(
            "utf-8", "replace"
        )
        assert b"usage:" in completed.stdout
        assert not tuple(cache.iterdir())

    unsafe = subprocess.run(
        ["/usr/bin/python3", entrypoint.as_posix(), "--help"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    assert unsafe.returncode != 0
    assert b"requires python -I -S -B" in unsafe.stderr


@pytest.mark.parametrize(
    "module",
    (
        "hegel_machine.phase3_m3_start_cli_v1",
        "hegel_machine.phase3_m3_formal_execution_cli_v1",
    ),
)
def test_package_module_launch_is_rejected_before_cli_import(module: str) -> None:
    completed = subprocess.run(
        ["/usr/bin/python3", "-m", module, "--help"],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": (PROJECT_ROOT / "src").as_posix(),
        },
    )
    assert completed.returncode != 0
    assert b"require their committed direct entrypoints" in completed.stderr
