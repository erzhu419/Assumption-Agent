from __future__ import annotations

import copy
import hashlib
import os
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import paper_controls
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
)
from assumption_agent.events import NullEventSink
from assumption_agent.models import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_paper_pipeline.sh"
MANIFEST = ROOT / "manifests" / "skilllearnbench_instance_holdout_offline_ready_v1.json"
V319_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_19_ruoli_gpt54mini.json"
)
V320_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json"
)
PORTABLE_PREREGISTRATION = (
    ROOT / "manifests" / "skilllearn_typed_portable_integration_v1.json"
)
PORTABLE_RESULT = (
    ROOT / "manifests" / "skilllearn_typed_portable_integration_result_v1.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        (
            "skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json",
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


def test_v320_binds_the_exact_portable_integration_result() -> None:
    protocol = PaperProtocol.read(V320_PROTOCOL)
    preregistration = json.loads(
        PORTABLE_PREREGISTRATION.read_text(encoding="utf-8")
    )
    result = json.loads(PORTABLE_RESULT.read_text(encoding="utf-8"))
    execution = protocol.payload["execution"]
    source = execution["typed_selection_snapshot_source"]

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.20.0"
    assert execution["portable_capability_compiler_mode"] == (
        PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
    )
    assert source == {
        "preregistration": (
            "manifests/skilllearn_typed_portable_integration_v1.json"
        ),
        "preregistration_file_sha256": _file_sha256(
            PORTABLE_PREREGISTRATION
        ),
        "source_run_root": preregistration["source_run_root"],
        "source_train_receipt": preregistration["source_train_receipt"],
        "source_train_receipt_file_sha256": preregistration[
            "source_train_receipt_file_sha256"
        ],
        "integration_result_receipt": (
            "manifests/skilllearn_typed_portable_integration_result_v1.json"
        ),
        "integration_result_receipt_file_sha256": _file_sha256(
            PORTABLE_RESULT
        ),
        "snapshot_ledger_hash": result["portable_projection"][
            "snapshot_ledger_hash"
        ],
    }
    assert source["snapshot_ledger_hash"] == (
        "d560903a5df0da0a464b3636ef2f80bd86cba3f5230de53f5da6f3acc4597bbf"
    )
    assert result["integration_passed"] is True
    assert result["fresh_development_protocol_freeze_eligible"] is True
    assert execution["development_prewarm"] == (
        "all_manifest_images_offline_verifiers_and_public_inputs_v5"
    )
    assert execution["model_inference_slots"] == 48
    assert protocol.payload["phases"]["development"][
        "parallel_workers"
    ] == 38


def test_v320_changes_only_the_portable_integration_binding_from_v319() -> None:
    v319 = copy.deepcopy(PaperProtocol.read(V319_PROTOCOL).payload)
    v320 = copy.deepcopy(PaperProtocol.read(V320_PROTOCOL).payload)
    for payload in (v319, v320):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
        payload["execution"]["typed_selection_snapshot_source"] = (
            "<typed-selection-source>"
        )
    v320["execution"].pop("portable_capability_compiler_mode")

    assert v320 == v319


def test_pipeline_defaults_to_the_fresh_v320_development_root() -> None:
    source = PIPELINE.read_text(encoding="utf-8")
    assert (
        'PROTOCOL="${PROTOCOL:-manifests/'
        'skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json}"'
    ) in source
    assert (
        'RUN_ROOT="${RUN_ROOT:-artifacts/'
        'paper_primary_v3_20_offline86_ruoli_gpt54mini_'
        'outer38_model48_portable01}"'
    ) in source


def _run_v320_pipeline_with_recording_stubs(
    tmp_path: Path,
    *,
    action: str,
) -> list[list[str]]:
    bin_dir = tmp_path / "recording-bin"
    bin_dir.mkdir()
    capture = tmp_path / "all-commands.txt"
    python = bin_dir / "python3"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == \'-c\' ]]; then exec /usr/bin/python3 "$@"; fi\n'
        f"printf '%q ' \"$@\" >> {shlex.quote(str(capture))}\n"
        f"printf '\\n' >> {shlex.quote(str(capture))}\n"
        "previous=''\n"
        "for argument in \"$@\"; do\n"
        "  case \"${previous}\" in\n"
        "    --out|--archive-out|--paired-no-recursive-out|"
        "--paired-no-recursive-archive-out)\n"
        "      mkdir -p \"$(dirname \"${argument}\")\"\n"
        "      printf '{\"incumbent_id\":null,\"nodes\":{}}\\n' "
        "> \"${argument}\"\n"
        "      ;;\n"
        "  esac\n"
        "  previous=\"${argument}\"\n"
        "done\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    docker = bin_dir / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PROTOCOL": str(V320_PROTOCOL),
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
    if not capture.exists():
        return []
    return [
        shlex.split(line)
        for line in capture.read_text(encoding="utf-8").splitlines()
    ]


def test_v320_skips_the_stale_proposal_diagnostic_action(
    tmp_path: Path,
) -> None:
    assert _run_v320_pipeline_with_recording_stubs(
        tmp_path,
        action="proposal-diagnostic",
    ) == []


def test_v320_all_development_runs_lock_prewarm_and_develop_directly(
    tmp_path: Path,
) -> None:
    commands = _run_v320_pipeline_with_recording_stubs(
        tmp_path,
        action="all-development",
    )
    modules = [
        tokens[tokens.index("-m") + 1]
        for tokens in commands
        if "-m" in tokens
    ]

    assert modules == [
        "assumption_agent.benchmarks.paper_protocol",
        "assumption_agent.benchmarks.prewarm",
        "assumption_agent.benchmarks.docker_egress",
        "assumption_agent.benchmarks.skilllearn_experiment",
    ]
    assert "assumption_agent.benchmarks.train_proposal_diagnostic" not in modules
    assert "assumption_agent.benchmarks.preflight" not in modules
    experiment = commands[-1]
    assert experiment[experiment.index("--out") + 1].endswith(
        "/development_recursive.report.json"
    )
    assert not any("smoke_recursive" in token for token in experiment)


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
