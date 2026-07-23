from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import birco_p1_private_selection_v1 as selection
from replication_runtime.birco_gpt54_semantic_v1 import contract as semantic
from replication_runtime.birco_gpt54_semantic_v1 import worker as semantic_worker
from replication_runtime.birco_official_hipporag_v1 import contract as hippo
from replication_runtime.birco_p1_formal_v1 import runner


SECRET = "plus-secret-value-that-must-never-be-persisted"


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def _write_json(path: Path, value: object) -> None:
    _write_private(path, runner._canonical_bytes(value))


def _plus_env(path: Path) -> None:
    _write_private(
        path,
        (
            "ASSUMPTION_V2_API_BASE=https://ruoli.dev\n"
            f"ASSUMPTION_V2_API_KEY={SECRET}\n"
            "ASSUMPTION_V2_MODEL=gpt-5.4-mini\n"
            "BIRCO_P1_PROVIDER_LABEL=plus\n"
        ).encode(),
    )


def _python_symlink(tmp_path: Path) -> Path:
    lexical = tmp_path / "venv" / "bin" / "python"
    lexical.parent.mkdir(parents=True)
    lexical.symlink_to(Path(sys.executable).resolve())
    return lexical


def _semantic_terminal(
    *, mode: str, payload: dict[str, object], provider: semantic_worker.Provider
) -> dict[str, object]:
    assert mode == "plan"
    body = {
        "action": {
            "plan": semantic.deterministic_plan_totalizer(
                str(payload["query"])
            ).payload()
        },
        "attempt_count": 1,
        "generation_valid": False,
        "input_sha256": semantic.semantic_hash(payload),
        "mode": mode,
        "model_request_sha256": "1" * 64,
        "provider": provider.safe_identity(),
        "raw_completion_persisted": False,
        "response_sha256": None,
        "retry_replay_resample_or_provider_switch_count": 0,
        "schema": semantic.TERMINAL_OUTPUT_SCHEMA,
        "terminal_category": "transport_unavailable",
        "transport": semantic_worker.TRANSPORT_ID,
        "transport_succeeded": False,
        "work_id": payload["work_id"],
    }
    return {**body, "self_sha256": semantic.semantic_hash(body)}


def test_semantic_executor_keeps_secret_out_of_argv_artifacts_and_reuses_terminal(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    credential = tmp_path / "plus.env"
    _plus_env(credential)
    lexical_python = _python_symlink(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert command[0] == str(lexical_python)
        assert SECRET not in "\n".join(command)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["ASSUMPTION_V2_API_KEY"] == SECRET
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        payload = json.loads(input_path.read_text(encoding="ascii"))
        provider = semantic_worker.Provider(
            api_base=environment["ASSUMPTION_V2_API_BASE"],
            api_origin=semantic_worker.PROVIDER_ORIGIN,
            api_key=environment["ASSUMPTION_V2_API_KEY"],
            model=environment["ASSUMPTION_V2_MODEL"],
            label=environment["BIRCO_P1_PROVIDER_LABEL"],
        )
        _write_json(
            output_path.with_name(output_path.name + ".attempt.json"),
            semantic_worker._attempt_claim(
                mode="plan", payload=payload, provider=provider
            ),
        )
        _write_json(
            output_path,
            _semantic_terminal(mode="plan", payload=payload, provider=provider),
        )
        return SimpleNamespace(returncode=0, stdout=b"safe", stderr=b"")

    executor = runner.SemanticExecutor(
        project_root=project,
        runtime_root=project / "runtime",
        credential_env_path=credential,
        python_executable=lexical_python,
        subprocess_runner=fake_run,
    )
    payload = semantic.planner_input(
        work_id="birco-work-v1-" + "a" * 64,
        objective="Find the relevant document.",
        query="A required clause and an excluded clause.",
    )
    first = executor(mode="plan", payload=payload)
    credential.write_text("credential file drifted after prerun\n", encoding="utf-8")
    second = executor(mode="plan", payload=payload)
    assert first == second
    assert len(calls) == 1
    for artifact in (project / "runtime").rglob("*"):
        if artifact.is_file():
            assert SECRET.encode() not in artifact.read_bytes()


def _fake_executable(path: Path) -> Path:
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o700)
    return path


def _hippo_payload(work_suffix: str = "a") -> dict[str, object]:
    documents = [
        {"ordinal": ordinal, "text": f"candidate document {ordinal}"}
        for ordinal in range(10)
    ]
    objective = "Retrieve the best candidate."
    query = "query text"
    return {
        "common_projection_sha256": hippo.common_projection_sha256(
            objective=objective, query=query, documents=documents
        ),
        "documents": documents,
        "objective": objective,
        "query": query,
        "schema": hippo.INPUT_SCHEMA,
        "work_id": "birco-work-v1-" + work_suffix * 64,
    }


def _hippo_policy(slot: int) -> dict[str, object]:
    return {
        "model_alias_cwd_relative": "models",
        "llm_model_alias": "smollm2",
        "embedding_model_alias": "minilm",
        "aliases_are_single_relative_components": True,
        "subprocess_cwd_is_model_alias_cwd": True,
        "absolute_model_path_argument_count": 0,
        "logical_slot_count": 4,
        "gpu_assignment": ["0", "1", "0", "1"],
        "maximum_processes_per_gpu": 2,
        "cpu_threads_per_process": 2,
        "logical_slot_ordinal": slot,
        "visible_gpu": ("0", "1", "0", "1")[slot],
    }


def _hippo_base_policy() -> dict[str, object]:
    value = _hippo_policy(0)
    value.pop("logical_slot_ordinal")
    value.pop("visible_gpu")
    return value


def _hippo_layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    model_cwd = project / "models"
    model_cwd.mkdir(parents=True)
    targets = tmp_path / "targets"
    for alias in ("smollm2", "minilm"):
        target = targets / alias
        target.mkdir(parents=True)
        (model_cwd / alias).symlink_to(target, target_is_directory=True)
    fake_strace = _fake_executable(tmp_path / "strace")
    fake_env = _fake_executable(tmp_path / "env")
    return project, model_cwd, fake_strace, fake_env


def test_hippo_executor_uses_env_i_network_injection_exact_cwd_and_short_aliases(
    tmp_path: Path,
) -> None:
    project, model_cwd, fake_strace, fake_env = _hippo_layout(tmp_path)
    lexical_python = _python_symlink(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        assert kwargs["cwd"] == model_cwd.resolve()
        assert kwargs["env"] == {}
        assert command[0] == str(fake_strace)
        assert command[command.index("-o") + 2] == str(fake_env)
        assert command[command.index(str(fake_env)) + 1] == "-i"
        assert "trace=%network" in command
        assert "inject=%network:error=EPERM" in command
        assert command[command.index("--llm-model") + 1] == "smollm2"
        assert command[command.index("--embedding-model") + 1] == "minilm"
        assert str(lexical_python) in command
        output_path = Path(command[command.index("--output") + 1])
        index_path = Path(command[command.index("--index-root") + 1])
        input_path = Path(command[command.index("--input") + 1])
        audit_path = Path(command[command.index("-o") + 1])
        payload = json.loads(input_path.read_text(encoding="ascii"))
        index_path.mkdir(mode=0o700)
        output = hippo.output_payload(
            work_id=payload["work_id"],
            common_projection_sha256=payload["common_projection_sha256"],
            candidate_count=len(payload["documents"]),
            rank_ordinals=list(range(len(payload["documents"]))),
            graph_nodes=12,
            graph_edges=13,
        )
        _write_private(output_path, hippo.canonical_json_bytes(output))
        audit_path.write_text(
            "socket(AF_INET, SOCK_STREAM, IPPROTO_IP) = -1 EPERM "
            "(Operation not permitted) (INJECTED)\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout=b"safe", stderr=b"")

    executor = runner.HippoExecutor(
        project_root=project,
        runtime_root=project / "runtime",
        python_executable=lexical_python,
        strace_executable=fake_strace,
        env_executable=fake_env,
        subprocess_runner=fake_run,
    )
    result = executor(payload=_hippo_payload(), runtime_policy=_hippo_policy(2))
    recovered = executor(payload=_hippo_payload(), runtime_policy=_hippo_policy(2))
    assert result == recovered
    assert len(commands) == 1
    receipt = result["runtime_receipt"]
    assert set(receipt) == runner.HippoExecutor._CONTROLLER_RECEIPT_FIELDS
    assert receipt["logical_slot_ordinal"] == 2
    assert receipt["visible_gpu"] == "0"
    audit_receipt = next(
        (project / "runtime").rglob("runtime_audit_receipt.json")
    )
    audit = json.loads(audit_receipt.read_text(encoding="ascii"))
    assert audit["external_network_call_count"] == 0
    assert audit["denied_network_syscall_count"] == 1
    assert audit["configured_gpu_assignment"] == ["0", "1", "0", "1"]


def test_hippo_executor_fixed_four_slot_gpu_assignment_and_two_per_gpu_cap(
    tmp_path: Path,
) -> None:
    project, _model_cwd, fake_strace, fake_env = _hippo_layout(tmp_path)
    lexical_python = _python_symlink(tmp_path)
    lock = threading.Lock()
    active = {"0": 0, "1": 0}
    peak = {"0": 0, "1": 0}
    seen: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        gpu_row = next(row for row in command if row.startswith("CUDA_VISIBLE_DEVICES="))
        gpu = gpu_row.split("=", 1)[1]
        with lock:
            active[gpu] += 1
            peak[gpu] = max(peak[gpu], active[gpu])
            seen.append(gpu)
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        index_path = Path(command[command.index("--index-root") + 1])
        audit_path = Path(command[command.index("-o") + 1])
        payload = json.loads(input_path.read_text(encoding="ascii"))
        index_path.mkdir(mode=0o700)
        output = hippo.output_payload(
            work_id=payload["work_id"],
            common_projection_sha256=payload["common_projection_sha256"],
            candidate_count=10,
            rank_ordinals=list(range(10)),
            graph_nodes=1,
            graph_edges=1,
        )
        _write_private(output_path, hippo.canonical_json_bytes(output))
        audit_path.write_bytes(b"")
        with lock:
            active[gpu] -= 1
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    executor = runner.HippoExecutor(
        project_root=project,
        runtime_root=project / "runtime",
        python_executable=lexical_python,
        strace_executable=fake_strace,
        env_executable=fake_env,
        subprocess_runner=fake_run,
    )
    suffixes = ("a", "b", "c", "d")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                executor,
                payload=_hippo_payload(suffix),
                runtime_policy=_hippo_policy(slot),
            )
            for slot, suffix in enumerate(suffixes)
        ]
        [future.result() for future in futures]
    assert sorted(seen) == ["0", "0", "1", "1"]
    assert peak["0"] <= 2
    assert peak["1"] <= 2


def test_hippo_executor_fails_closed_if_strace_reports_successful_network_call(
    tmp_path: Path,
) -> None:
    project, _model_cwd, fake_strace, fake_env = _hippo_layout(tmp_path)
    lexical_python = _python_symlink(tmp_path)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        index_path = Path(command[command.index("--index-root") + 1])
        audit_path = Path(command[command.index("-o") + 1])
        payload = json.loads(input_path.read_text(encoding="ascii"))
        index_path.mkdir(mode=0o700)
        output = hippo.output_payload(
            work_id=payload["work_id"],
            common_projection_sha256=payload["common_projection_sha256"],
            candidate_count=10,
            rank_ordinals=list(range(10)),
            graph_nodes=1,
            graph_edges=1,
        )
        _write_private(output_path, hippo.canonical_json_bytes(output))
        audit_path.write_text(
            "connect(3, {sa_family=AF_INET}, 16) = 0\n", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    executor = runner.HippoExecutor(
        project_root=project,
        runtime_root=project / "runtime",
        python_executable=lexical_python,
        strace_executable=fake_strace,
        env_executable=fake_env,
        subprocess_runner=fake_run,
    )
    with pytest.raises(runner.BircoP1FormalRuntimeError, match="not denied"):
        executor(payload=_hippo_payload(), runtime_policy=_hippo_policy(0))


def test_network_audit_rejects_strace_warning(tmp_path: Path) -> None:
    audit = tmp_path / "network.strace"
    audit.write_text(
        "strace: ptrace(PTRACE_SEIZE, 123): Operation not permitted\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.BircoP1FormalRuntimeError, match="strace reported"):
        runner.HippoExecutor._audit_network(audit)


def test_qrel_opener_writes_authorization_then_opens_and_never_opens_f(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "selection"
    output_root.mkdir()
    control_root = tmp_path / "control"
    calls: list[str] = []
    authorization_sha = "a" * 64

    def write_authorization(path: Path, **_kwargs: object) -> dict[str, str]:
        calls.append("write")
        _write_json(path, {"authorization_sha256": authorization_sha})
        return {"authorization_sha256": authorization_sha}

    def open_qrels(**kwargs: object) -> dict[str, object]:
        calls.append("open")
        assert kwargs["expected_authorization_sha256"] == authorization_sha
        return {"sealed": True}

    monkeypatch.setattr(selection, "write_block_open_authorization", write_authorization)
    monkeypatch.setattr(selection, "open_block_qrels", open_qrels)
    opener = runner.QrelOpener(output_root=output_root, control_root=control_root)
    assert opener(
        block="A_form",
        action_archive_sha256s=("b" * 64,),
        promotion_sha256=None,
    ) == {"sealed": True}
    assert calls == ["write", "open"]
    with pytest.raises(runner.BircoP1FormalRuntimeError, match="permanently sealed"):
        opener(
            block="F_search",
            action_archive_sha256s=("b" * 64,),
            promotion_sha256=None,
        )
    assert calls == ["write", "open"]


def test_qrel_opener_recovers_existing_open_marker_without_second_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "selection"
    output_root.mkdir()
    control_root = tmp_path / "control"
    opener = runner.QrelOpener(output_root=output_root, control_root=control_root)
    archive_sha = "a" * 64
    authorization_sha = "b" * 64
    acquisition_sha = "c" * 64
    action_sha = "d" * 64
    qrel_sha = "e" * 64
    authorization = {
        "action_archive_sha256s": [archive_sha],
        "A_hold_promotion_sha256": None,
        "authorization_sha256": authorization_sha,
    }
    _write_json(opener._authorization_path("A_hold"), authorization)
    marker = selection.self_hashed(
        {
            "schema": f"{selection.VERSION}_qrel_open_marker_v1",
            "version": selection.VERSION,
            "study_id": selection.STUDY_ID,
            "status": "authorization_consumed_immediately_before_numeric_qrel_open",
            "block": "A_hold",
            "acquisition_sha256": acquisition_sha,
            "authorization_sha256": authorization_sha,
            "same_block_second_open_authorized": False,
        },
        "open_marker_sha256",
    )
    marker_path = output_root / selection.QREL_OPEN_MARKER_FILENAMES["A_hold"]
    _write_json(marker_path, marker)

    monkeypatch.setattr(
        selection,
        "_load_public_receipt",
        lambda _root: {"acquisition_sha256": acquisition_sha},
    )
    monkeypatch.setattr(
        selection,
        "_pack_binding",
        lambda _receipt, *, block, role: {
            "semantic_sha256": action_sha if role == "action" else qrel_sha
        },
    )
    monkeypatch.setattr(
        selection,
        "_validate_block_authorization",
        lambda *_args, **_kwargs: authorization_sha,
    )
    qrel_pack = {"qrel_pack_sha256": qrel_sha, "sealed": True}
    monkeypatch.setattr(
        selection, "_read_bound_private_pack", lambda *_args, **_kwargs: qrel_pack
    )
    monkeypatch.setattr(
        selection, "_validate_qrel_pack", lambda *_args, **_kwargs: qrel_sha
    )
    monkeypatch.setattr(
        selection,
        "write_block_open_authorization",
        lambda *_args, **_kwargs: pytest.fail("must not write a second authorization"),
    )
    monkeypatch.setattr(
        selection,
        "open_block_qrels",
        lambda **_kwargs: pytest.fail("must not write a second open marker"),
    )
    before = marker_path.read_bytes()
    assert opener(
        block="A_hold",
        action_archive_sha256s=(archive_sha,),
        promotion_sha256=None,
    ) == qrel_pack
    assert marker_path.read_bytes() == before


def test_cli_failure_stdout_and_terminal_artifact_do_not_contain_exception_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    selection_root = project / "selection"
    selection_root.mkdir()
    (selection_root / "receipt.json").write_text("{}\n", encoding="ascii")
    credential = tmp_path / "plus.env"
    _plus_env(credential)
    _environment, provider_identity = runner._parse_plus_environment(credential)
    freeze_body = {
        "hipporag_runtime_policy": _hippo_base_policy(),
        "provider_identity": provider_identity,
        "selection_receipt_binding": {"relative_path": "selection/receipt.json"},
    }
    freeze = {**freeze_body, "self_sha256": runner._stable_hash(freeze_body)}
    freeze_path = project / "freeze.json"
    freeze_path.write_bytes(runner._canonical_bytes(freeze))
    file_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    control = tmp_path / "control"

    monkeypatch.setattr(
        runner,
        "SemanticExecutor",
        lambda **_kwargs: SimpleNamespace(provider_identity=provider_identity),
    )

    class FakeHippoExecutor:
        def preflight_base_policy(self, value: object) -> None:
            assert value == _hippo_base_policy()

    monkeypatch.setattr(
        runner, "HippoExecutor", lambda **_kwargs: FakeHippoExecutor()
    )
    monkeypatch.setattr(runner, "QrelOpener", lambda **_kwargs: object())

    class FailingController:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self) -> dict[str, object]:
            raise RuntimeError("do not persist " + SECRET)

    monkeypatch.setattr(runner.controller, "FormalController", FailingController)
    code = runner.main(
        [
            "--project-root",
            str(project),
            "--control-root",
            str(control),
            "--execution-freeze",
            str(freeze_path),
            "--execution-freeze-file-sha256",
            file_sha,
            "--execution-freeze-self-sha256",
            freeze["self_sha256"],
            "--plus-env",
            str(credential),
        ]
    )
    assert code == 1
    stdout = capsys.readouterr().out
    assert SECRET not in stdout
    failure = control / "runtime_terminal_failure.json"
    assert failure.exists()
    assert SECRET.encode() not in failure.read_bytes()


def test_cli_rejects_credential_identity_before_controller_or_stage_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    selection_root = project / "selection"
    selection_root.mkdir()
    (selection_root / "receipt.json").write_text("{}\n", encoding="ascii")
    credential = tmp_path / "plus.env"
    _plus_env(credential)
    _environment, provider_identity = runner._parse_plus_environment(credential)
    drifted_identity = dict(provider_identity)
    drifted_identity["api_key_hmac_sha256"] = "0" * 64
    freeze_body = {
        "provider_identity": drifted_identity,
        "selection_receipt_binding": {"relative_path": "selection/receipt.json"},
    }
    freeze = {**freeze_body, "self_sha256": runner._stable_hash(freeze_body)}
    freeze_path = project / "freeze.json"
    freeze_path.write_bytes(runner._canonical_bytes(freeze))
    file_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    control = tmp_path / "control"

    real_semantic_executor = runner.SemanticExecutor

    class SemanticWithoutProcess(real_semantic_executor):
        pass

    monkeypatch.setattr(runner, "SemanticExecutor", SemanticWithoutProcess)

    class MustNotConstruct:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("Hippo/qrel/controller must not be constructed")

    monkeypatch.setattr(runner, "HippoExecutor", MustNotConstruct)
    monkeypatch.setattr(runner, "QrelOpener", MustNotConstruct)
    monkeypatch.setattr(runner.controller, "FormalController", MustNotConstruct)
    assert (
        runner.main(
            [
                "--project-root",
                str(project),
                "--control-root",
                str(control),
                "--execution-freeze",
                str(freeze_path),
                "--execution-freeze-file-sha256",
                file_sha,
                "--execution-freeze-self-sha256",
                freeze["self_sha256"],
                "--plus-env",
                str(credential),
            ]
        )
        == 1
    )
    assert not (control / "stages").exists()
    assert not list(control.rglob("*.claim.json"))
