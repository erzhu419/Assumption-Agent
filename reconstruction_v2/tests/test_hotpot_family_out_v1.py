from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import threading
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from assumption_agent.benchmarks import hotpot_family_out_acquisition_v1 as acquisition
from assumption_agent.benchmarks import hotpot_family_out_runner_v1 as runner
from assumption_agent.benchmarks.musique_official_core_comparison_v1 import (
    generate_selection_secret,
)
from assumption_agent.benchmarks.musique_typed_retriever_formation_v1 import (
    TypedRetrievalProgram,
)
from assumption_agent.models import stable_hash


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Hotpot Test",
            "-c",
            "user.email=hotpot-test@example.invalid",
            *arguments,
        ],
        check=True,
        capture_output=True,
    )


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (project / "impl.py").write_text("VERSION = 1\n", encoding="utf-8")
    _git(project, "add", ".gitignore", "impl.py")
    _git(project, "commit", "-q", "-m", "initial")
    return project


def _source_row(index: int, *, item_id: str | None = None, valid: bool = True) -> dict:
    titles = [f"Title {index}-{position}" for position in range(6)]
    sentences = [
        [f"Sentence {index}-{position} contains token {index}-{position}."]
        for position in range(6)
    ]
    if not valid:
        sentences[3] = []
    return {
        "id": item_id or f"hotpot-{index:03d}",
        "question": f"Which records connect token {index}?",
        "answer": f"answer-{index}",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": {
            "title": [titles[0], titles[5]],
            "sent_id": [0, 0],
        },
        "context": {"title": titles, "sentences": sentences},
    }


def _write_self_hashed(path: Path, payload: Mapping[str, Any], hash_field: str) -> None:
    acquisition._write_json_exclusive(
        path,
        payload,
        hash_field=hash_field,
        mode=0o644,
    )


def test_capability_receipt_checks_network_namespace_and_writable_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Completed:
        def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = stderr

    calls: list[tuple[str, ...]] = []

    def run(arguments: list[str], **_kwargs: Any) -> Completed:
        calls.append(tuple(arguments))
        if arguments[-1] == "--version":
            return Completed(stdout=(acquisition.BWRAP_VERSION + "\n").encode())
        assert arguments[1:9] == [
            "--unshare-net",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
        ]
        bind = arguments.index("--bind")
        assert arguments[bind + 1] == arguments[bind + 2]
        assert Path(arguments[bind + 1]).is_dir()
        assert arguments[-1] == "/bin/true"
        return Completed()

    monkeypatch.setattr(runner, "_sha256_file", lambda _path: runner.BWRAP_SHA256)
    monkeypatch.setattr(runner.subprocess, "run", run)
    output = tmp_path / "capability.json"
    receipt = runner.build_capability_receipt(output)
    assert receipt["probe_contract_sha256"] == stable_hash(
        {"argv_without_binary": list(acquisition.BWRAP_PROBE_TEMPLATE_ARGS)}
    )
    assert receipt["benchmark_rows_read"] == receipt["model_calls"] == 0
    assert len(calls) == 2
    runner.verify_capability_receipt(output)
    assert acquisition.capability_binding(output)["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]

    tampered = dict(receipt)
    tampered["bwrap_version"] = "bubblewrap synthetic drift"
    body = dict(tampered)
    body.pop("receipt_sha256")
    tampered["receipt_sha256"] = stable_hash(body)
    drift = tmp_path / "capability-drift.json"
    drift.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(Exception, match="drifted|did not pass"):
        acquisition.capability_binding(drift)


def test_preregister_then_committed_one_shot_acquisition_is_content_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    artifacts = project / "artifacts"
    artifacts.mkdir()
    source = artifacts / "source.parquet"
    rows = [_source_row(index) for index in range(14)]
    # A malformed duplicate still makes its structurally valid twin globally
    # non-unique and therefore ineligible.
    rows.append(_source_row(99, item_id="hotpot-000", valid=False))
    pq.write_table(pa.Table.from_pylist(rows), source)

    monkeypatch.setattr(acquisition, "SOURCE_SIZE", source.stat().st_size)
    monkeypatch.setattr(acquisition, "SOURCE_SHA256", acquisition._sha256_file(source))
    monkeypatch.setattr(acquisition, "SOURCE_ROW_COUNT", len(rows))
    monkeypatch.setattr(acquisition, "IMPLEMENTATION_RELATIVE_FILES", ("impl.py",))
    monkeypatch.setattr(
        acquisition,
        "official_repository_receipt",
        lambda _path: {
            "repository": acquisition.OFFICIAL_REPOSITORY,
            "commit": acquisition.OFFICIAL_REPOSITORY_COMMIT,
            "readme_sha256": acquisition.OFFICIAL_README_SHA256,
            "declared_original_url": acquisition.ORIGINAL_DECLARED_URL,
            "role": "schema_and_original_source_declaration_only",
        },
    )
    p_binding = {
        "formation_receipt_file_sha256": "1" * 64,
        "formation_receipt_hash": "2" * 64,
        "frozen_program_file_sha256": "3" * 64,
        "frozen_program_envelope_hash": "4" * 64,
        "program_hash": "5" * 64,
        "formed_on_block_id_hash": stable_hash({"block": "F1"}),
    }
    capability_binding = {
        "file_sha256": "6" * 64,
        "receipt_sha256": "7" * 64,
        "bwrap_file_sha256": acquisition.BWRAP_SHA256,
        "probe_contract_sha256": stable_hash(
            {"argv_without_binary": list(acquisition.BWRAP_PROBE_TEMPLATE_ARGS)}
        ),
    }
    monkeypatch.setattr(acquisition, "p_program_binding", lambda **_kwargs: p_binding)
    monkeypatch.setattr(
        acquisition, "capability_binding", lambda _path: capability_binding
    )

    secret = artifacts / "selection.key"
    generate_selection_secret(project=project, output=secret)
    public = project / "manifests"
    public.mkdir()
    prereg_path = public / "hotpot.prereg.json"
    prereg = acquisition.build_preregistration(
        project=project,
        official_repository=project,
        selection_secret_path=secret,
        capability_receipt_path=project / "unused-capability.json",
        p_formation_receipt_path=project / "unused-formation.json",
        p_frozen_program_path=project / "unused-program.json",
    )
    assert prereg["safety"]["dataset_rows_read"] == 0
    _write_self_hashed(prereg_path, prereg, "preregistration_sha256")
    _git(project, "add", "manifests/hotpot.prereg.json")
    _git(project, "commit", "-q", "-m", "freeze preregistration")

    private_pack = artifacts / "family" / "pack.jsonl"
    private_locator = artifacts / "family" / "locator.json"
    receipt = acquisition.acquire_private_pack(
        project=project,
        preregistration_path=prereg_path,
        official_repository=project,
        selection_secret_path=secret,
        capability_receipt_path=project / "unused-capability.json",
        p_formation_receipt_path=project / "unused-formation.json",
        p_frozen_program_path=project / "unused-program.json",
        source_parquet_path=source,
        private_pack_path=private_pack,
        private_locator_path=private_locator,
    )
    assert receipt["counts"] == {
        "source_rows": 15,
        "structurally_valid_rows": 14,
        "eligible_unique_id_rows": 13,
        "selected_rows": acquisition.SAMPLE_COUNT,
    }
    assert receipt["prospective_ordering"][
        "preregistration_committed_before_source_row_open"
    ] is True
    assert receipt["prospective_ordering"][
        "acquisition_consumed_before_source_row_open"
    ] is True
    assert receipt["preregistration_custody"]["preregistration_file_sha256"] == (
        receipt["preregistration_custody"]["preregistration_head_blob_sha256"]
    )
    serialized = json.dumps(receipt, sort_keys=True)
    for private_value in (
        "Which records",
        "hotpot-",
        "Sentence",
        str(private_pack),
        str(private_locator),
        '"support_indices"',
    ):
        assert private_value not in serialized

    public_receipt = public / "hotpot.acquisition.json"
    _write_self_hashed(public_receipt, receipt, "acquisition_sha256")
    loaded, _raw = runner._load_acquisition(public_receipt)
    assert loaded["counts"]["selected_rows"] == acquisition.SAMPLE_COUNT

    with pytest.raises(FileExistsError, match="already consumed"):
        acquisition.acquire_private_pack(
            project=project,
            preregistration_path=prereg_path,
            official_repository=project,
            selection_secret_path=secret,
            capability_receipt_path=project / "unused-capability.json",
            p_formation_receipt_path=project / "unused-formation.json",
            p_frozen_program_path=project / "unused-program.json",
            source_parquet_path=source,
            private_pack_path=artifacts / "alternate" / "pack.jsonl",
            private_locator_path=artifacts / "alternate" / "locator.json",
        )


def test_committed_preregistration_receipt_works_from_nested_project(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    project = repository / "nested" / "project"
    project.mkdir(parents=True)
    prereg = project / "manifests" / "prereg.json"
    prereg.parent.mkdir()
    prereg.write_text('{"frozen":true}\n', encoding="utf-8")
    _git(repository, "add", "nested/project/manifests/prereg.json")
    _git(repository, "commit", "-q", "-m", "nested preregistration")
    receipt = acquisition.committed_public_file_receipt(
        project=project,
        path=prereg,
    )
    expected = hashlib.sha256(prereg.read_bytes()).hexdigest()
    assert receipt["preregistration_file_sha256"] == expected
    assert receipt["preregistration_head_blob_sha256"] == expected


class _PreparedRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._safe = {
            "attestation_receipt_sha256": "a" * 64,
            "binding_sha256": "b" * 64,
        }

    @property
    def safe_binding(self) -> dict[str, Any]:
        return dict(self._safe)

    def fresh_reverify(self) -> dict[str, Any]:
        self.events.append("postflight")
        return dict(self._safe)


def _private_pack(project: Path) -> tuple[Path, str, str]:
    path = project / "artifacts" / "runner" / "pack.jsonl"
    path.parent.mkdir(parents=True)
    rows = []
    for ordinal in range(acquisition.SAMPLE_COUNT):
        corpus = [
            {
                "idx": index,
                "is_supporting": index in (0, 5),
                "text": f"private paragraph {ordinal}-{index}",
                "title": f"private title {ordinal}-{index}",
            }
            for index in range(6)
        ]
        rows.append(
            {
                "schema": acquisition.PRIVATE_ROW_SCHEMA,
                "item_id": f"private-id-{ordinal}",
                "question": f"private question {ordinal}",
                "corpus": corpus,
                "support_indices": [0, 5],
                "source_row_sha256": stable_hash({"source": ordinal}),
            }
        )
    raw = b"".join(acquisition._canonical_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), stable_hash(
        [stable_hash(row) for row in rows]
    )


def test_formal_runner_freezes_before_open_then_runs_one_36_way_offline_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    private_pack, pack_hash, item_set_hash = _private_pack(project)
    acquisition_receipt = {
        "acquisition_sha256": "c" * 64,
        "commitments": {
            "private_pack_file_sha256": pack_hash,
            "item_commitment_set_sha256": item_set_hash,
        },
    }
    acquisition_raw = b"synthetic acquisition receipt"
    program = TypedRetrievalProgram(
        seed_algorithm="bm25",
        title_weight=2,
        text_weight=1,
        expansion_mode="entity_token_one_hop",
        expansion_weight=1,
    )
    p_binding = {
        "formation_receipt_file_sha256": "1" * 64,
        "formation_receipt_hash": "2" * 64,
        "frozen_program_file_sha256": "3" * 64,
        "frozen_program_envelope_hash": "4" * 64,
        "program_hash": program.program_hash,
        "formed_on_block_id_hash": stable_hash({"block": "F1"}),
    }
    capability_raw = b"synthetic capability receipt"
    capability = {
        "receipt_sha256": "5" * 64,
        "bwrap_file_sha256": runner.BWRAP_SHA256,
        "probe_contract_sha256": runner._probe_contract_hash(),
    }
    events: list[str] = []
    prepared = _PreparedRuntime(events)
    implementation = {
        "schema": runner.IMPLEMENTATION_SCHEMA,
        "files": [{"path": "impl.py", "sha256": "6" * 64}],
        "set_sha256": "7" * 64,
    }
    monkeypatch.setattr(
        runner, "_load_acquisition", lambda _path: (acquisition_receipt, acquisition_raw)
    )
    monkeypatch.setattr(runner, "_p_program", lambda **_kwargs: (program, p_binding))
    monkeypatch.setattr(
        runner,
        "verify_capability_receipt",
        lambda _path: (capability, capability_raw),
    )
    monkeypatch.setattr(runner, "current_implementation_binding", lambda _project: implementation)
    monkeypatch.setattr(runner, "_prepare", lambda _project, _runtime: prepared)
    monkeypatch.setattr(
        runner,
        "_probe_bubblewrap",
        lambda: {
            "bwrap_file_sha256": runner.BWRAP_SHA256,
            "bwrap_version": runner.BWRAP_VERSION,
            "probe_contract_sha256": runner._probe_contract_hash(),
            "probe_returncode": 0,
            "probe_stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "probe_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )

    runtime = project / "artifacts" / "runtime"
    runtime.mkdir(parents=True)
    runtime_python = runtime / "python"
    runtime_python.write_text("synthetic", encoding="utf-8")
    llm = runtime / "llm"
    embedding = runtime / "embedding"
    llm.mkdir()
    embedding.mkdir()
    base = runtime / "base.json"
    attestation = runtime / "attestation.json"
    base.write_text("{}", encoding="utf-8")
    attestation.write_text("{}", encoding="utf-8")
    execution_root = project / "artifacts" / "runner" / "formal-root"
    freeze_path = project / "freeze.json"
    common = {
        "project_root": project,
        "acquisition_receipt_path": project / "unused-acquisition.json",
        "p_formation_receipt_path": project / "unused-formation.json",
        "p_frozen_program_path": project / "unused-program.json",
        "capability_receipt_path": project / "unused-capability.json",
        "runtime_python": runtime_python,
        "local_llm_model": llm,
        "local_embedding_model": embedding,
        "base_binding_receipt_path": base,
        "attestation_receipt_path": attestation,
        "execution_root": execution_root,
    }
    runner.build_pre_run_freeze(
        **common,
        authorization_hash=stable_hash({"authorization": "family-out-test"}),
        output_path=freeze_path,
    )
    assert "private_pack_path" not in inspect.signature(
        runner.build_pre_run_freeze
    ).parameters
    assert execution_root.exists() is False
    frozen_text = freeze_path.read_text(encoding="utf-8")
    assert "private question" not in frozen_text
    assert "support_indices" not in frozen_text

    calls: list[str] = []
    lock = threading.Lock()

    def record(arm: str, item: runner.HotpotRetrievalItem) -> None:
        assert not hasattr(item, "support_indices")
        assert not hasattr(item, "item_id")
        assert not hasattr(item, "item_commitment_sha256")
        with lock:
            calls.append(arm)
            events.append("retrieve")

    def raw(item: runner.HotpotRetrievalItem) -> tuple[int, ...]:
        record("RAW", item)
        return (0, 1, 2, 3, 4)

    def p(_program: Any, item: runner.HotpotRetrievalItem) -> tuple[int, ...]:
        record("P", item)
        return (1, 2, 3, 4, 5)

    def official(
        _runtime: Any, item: runner.HotpotRetrievalItem, _root: Path
    ) -> tuple[int, ...]:
        record("official", item)
        return (0, 1, 2, 3, 5)

    monkeypatch.setattr(runner, "_raw", raw)
    monkeypatch.setattr(runner, "_p", p)
    monkeypatch.setattr(runner, "_official", official)
    score = runner._score

    def score_after_postflight(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert events.count("retrieve") == runner.WORK_UNIT_COUNT
        assert events[-1] == "postflight"
        events.append("score")
        return score(*args, **kwargs)

    monkeypatch.setattr(runner, "_score", score_after_postflight)
    monkeypatch.setattr(runner, "_CLEAN_MODULE_CLI_ACTIVE", True)
    report = runner.execute_formal(
        **common,
        pre_run_freeze_path=freeze_path,
        private_pack_path=private_pack,
    )
    assert len(calls) == runner.WORK_UNIT_COUNT == 36
    assert calls.count("RAW") == calls.count("P") == calls.count("official") == 12
    assert events[-2:] == ["postflight", "score"]
    assert report["execution"]["configured_maximum_concurrency"] == 36
    assert report["execution"]["observed_start_barrier_party_count"] == 36
    assert report["execution"][
        "all_work_units_released_from_single_start_barrier"
    ] is True
    assert report["measurement"]["arm_metrics"]["canonical_RAW"][
        "support_hit_count"
    ] == 12
    assert report["measurement"]["arm_metrics"]["frozen_P"][
        "support_hit_count"
    ] == 12
    assert report["measurement"]["arm_metrics"]["official_HippoRAG"][
        "support_hit_count"
    ] == 24
    public_text = (execution_root / runner.REPORT_FILENAME).read_text(encoding="utf-8")
    for private_value in (
        "private question",
        "private title",
        "private paragraph",
        "private-id",
        '"support_indices"',
        str(private_pack),
    ):
        assert private_value not in public_text

    prior = len(calls)
    with pytest.raises(runner.HotpotFamilyOutRunnerError, match="replay is forbidden"):
        runner.execute_formal(
            **common,
            pre_run_freeze_path=freeze_path,
            private_pack_path=private_pack,
        )
    assert len(calls) == prior
