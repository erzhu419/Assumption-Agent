from __future__ import annotations

import os
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import paper_controls
from assumption_agent.events import NullEventSink
from assumption_agent.models import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_paper_pipeline.sh"
MANIFEST = ROOT / "manifests" / "skilllearnbench_instance_holdout_offline_ready_v1.json"


def _run_with_stubbed_commands(
    tmp_path: Path,
    *,
    protocol: Path,
    action: str,
) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "captured.txt"
    python = bin_dir / "python3"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == '-c' ]]; then exec /usr/bin/python3 \"$@\"; fi\n"
        f"printf '%q ' \"$@\" > {shlex.quote(str(capture))}\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    docker = bin_dir / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PROTOCOL": str(protocol),
        "MANIFEST": str(MANIFEST),
        "RUN_ROOT": str(tmp_path / "run"),
        "BENCHMARK_ROOT": str(tmp_path / "benchmark"),
        "ENV_FILE": str(tmp_path / ".env"),
        "TASK_INPUT_CACHE_ROOT": str(tmp_path / "closure-cache"),
    }
    subprocess.run(
        ["bash", str(PIPELINE), action],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return shlex.split(capture.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("protocol_name", "expected_version"),
    (
        (
            "skilllearn_paper_protocol_v3_18r1_ruoli_gpt54mini.json",
            "all_manifest_images_and_offline_verifiers_v4",
        ),
        (
            "skilllearn_paper_protocol_v3_19_ruoli_gpt54mini.json",
            "all_manifest_images_offline_verifiers_and_public_inputs_v5",
        ),
    ),
)
def test_pipeline_passes_each_protocols_frozen_prewarm_version(
    tmp_path: Path,
    protocol_name: str,
    expected_version: str,
) -> None:
    tokens = _run_with_stubbed_commands(
        tmp_path,
        protocol=ROOT / "manifests" / protocol_name,
        action="prewarm",
    )
    assert tokens[:2] == ["-m", "assumption_agent.benchmarks.prewarm"]
    assert tokens[tokens.index("--prewarm-version") + 1] == expected_version
    assert tokens[tokens.index("--protocol") + 1].endswith(protocol_name)
    assert tokens[tokens.index("--task-input-cache-root") + 1] == str(
        tmp_path / "closure-cache"
    )


def test_pipeline_exposes_maximally_parallel_frozen_image_preparation(
    tmp_path: Path,
) -> None:
    protocol = (
        ROOT / "manifests" / "skilllearn_paper_protocol_v3_19_ruoli_gpt54mini.json"
    )
    tokens = _run_with_stubbed_commands(
        tmp_path,
        protocol=protocol,
        action="prepare-images",
    )
    assert tokens[0] == "scripts/prepare_codex_agent_runtime.py"
    assert tokens[tokens.index("--scope") + 1] == "all"
    assert tokens[tokens.index("--parallel-workers") + 1] == "0"
    assert tokens[tokens.index("--protocol") + 1] == str(protocol)
    assert "--allow-network-download" in tokens


def test_v5_prewarm_cli_fails_before_docker_on_frozen_receipt_drift(
    tmp_path: Path,
) -> None:
    source_protocol = (
        ROOT / "manifests" / "skilllearn_paper_protocol_v3_19_ruoli_gpt54mini.json"
    )
    payload = json.loads(source_protocol.read_text(encoding="utf-8"))
    payload["execution"]["task_input_closure_source"][
        "preparation_receipt_hash"
    ] = "0" * 64
    protocol = tmp_path / "drifted-protocol.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    completed = subprocess.run(
        [
            "/usr/bin/python3",
            "-m",
            "assumption_agent.benchmarks.prewarm",
            "--root",
            str(tmp_path / "benchmark-does-not-need-to-exist"),
            "--manifest",
            str(MANIFEST),
            "--protocol",
            str(protocol),
            "--project-root",
            str(ROOT),
            "--env-file",
            str(env_file),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--out",
            str(tmp_path / "receipt.json"),
            "--prewarm-version",
            "all_manifest_images_offline_verifiers_and_public_inputs_v5",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "task input preparation receipt contract mismatch" in completed.stderr
    assert not (tmp_path / "events.jsonl").exists()


def test_controls_construct_the_closure_cache_from_validated_v5_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_hash = stable_hash({"item_id": "organize-messy-files-1"})
    payload = {
        "items": [
            {
                "item_id_hash": item_hash,
                "task_input_closure_required": True,
            }
        ]
    }
    frozen = SimpleNamespace(freeze_hash="frozen")
    captured = {}

    def fake_cache(benchmark_root, **kwargs):
        captured["benchmark_root"] = benchmark_root
        captured.update(kwargs)
        return "closure-cache"

    monkeypatch.setattr(
        paper_controls,
        "FrozenTaskInputPrebuiltImageCache",
        fake_cache,
    )
    built = paper_controls._build_control_prebuilt_cache(
        benchmark_root=tmp_path / "benchmark",
        event_sink=NullEventSink(),
        frozen_task_inputs=frozen,
        prewarm_payload=payload,
        task_input_cache_root=tmp_path / "task-inputs",
    )
    assert built == "closure-cache"
    assert captured["frozen_task_inputs"] is frozen
    assert captured["expected_prewarm_rows"] == {
        item_hash: payload["items"][0]
    }
    assert captured["task_input_cache_root"] == tmp_path / "task-inputs"
