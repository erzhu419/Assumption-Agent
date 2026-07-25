from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks import mmqa_p1_action_integration_v1 as integration
from assumption_agent.benchmarks import mmqa_p1_block_coordinate_worker_v1 as worker
from assumption_agent.benchmarks import mmqa_p1_formal_action_runtime_v1 as runtime
from assumption_agent.benchmarks import mmqa_p1_formal_controller_v1 as controller
from assumption_agent.benchmarks import mmqa_p1_local_action_executor_v1 as local
from assumption_agent.benchmarks import mmqa_p1_local_runtime_preflight_v1 as preflight
from assumption_agent.benchmarks import mmqa_p1_official_hipporag_block_v1 as hippo
from assumption_agent.benchmarks import mmqa_p1_remote_outer_lifecycle_v1 as outer


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _work_id(label: str) -> str:
    return "mmqa-work-v1-" + hashlib.sha256(label.encode("ascii")).hexdigest()


def _item(label: str) -> integration.AnonymousWorkItem:
    return integration.validate_anonymous_work_item(
        {
            "schema": integration.ANONYMOUS_WORK_ITEM_SCHEMA,
            "question": f"Which Aurora relation resolves {label} in 2012?",
            "rows": [
                {
                    "ordinal": 0,
                    "serialized_content": "Aurora relation row 2012",
                },
                {
                    "ordinal": 1,
                    "serialized_content": "Unrelated row",
                },
                {
                    "ordinal": 2,
                    "serialized_content": "Third row",
                },
            ],
            "texts": [
                {
                    "ordinal": 3,
                    "serialized_content": "Aurora relation text",
                },
                {
                    "ordinal": 4,
                    "serialized_content": "Unrelated text",
                },
                {
                    "ordinal": 5,
                    "serialized_content": "Third text",
                },
            ],
            "exact_row_text_links": [
                {"row_ordinal": 0, "text_ordinal": 3},
                {"row_ordinal": 1, "text_ordinal": 4},
                {"row_ordinal": 2, "text_ordinal": 5},
            ],
        }
    )


def _official_paths(tmp_path: Path) -> hippo.FreshComparatorRuntimePaths:
    base = tmp_path / "official"
    return hippo.FreshComparatorRuntimePaths(
        runtime_python=str(base / "venv/bin/python"),
        pyvenv_cfg=str(base / "venv/pyvenv.cfg"),
        overlay_root=str(base / "overlay"),
        hipporag_source_root=str(base / "source"),
        p16_site_root=str(base / "site"),
        local_llm_model=str(base / "llm"),
        local_embedding_model=str(base / "embedding"),
    )


def _config(tmp_path: Path) -> runtime.FormalActionRuntimeConfig:
    project = tmp_path / "project"
    project.mkdir(mode=0o700)
    (project / "artifacts").mkdir(mode=0o700)
    (project / "manifests").mkdir(mode=0o700)
    typed = project / "typed-python"
    minilm = project / "minilm"
    cross_encoder = project / "cross-encoder"
    nvidia_smi = project / "nvidia-smi"
    systemd_run = project / "systemd-run"
    for path in (typed, nvidia_smi, systemd_run):
        path.write_bytes(path.name.encode("ascii"))
        path.chmod(0o700)
    minilm.mkdir(mode=0o700)
    cross_encoder.mkdir(mode=0o700)
    return runtime.FormalActionRuntimeConfig(
        project_root=project,
        execution_freeze_self_sha256="1" * 64,
        implementation_freeze_self_sha256="2" * 64,
        local_preflight_receipt=(
            project / outer.PREFLIGHT_RECEIPT_RELATIVE
        ),
        local_preflight_self_sha256="3" * 64,
        typed_python=typed,
        typed_python_resolved_sha256=hashlib.sha256(
            typed.read_bytes()
        ).hexdigest(),
        minilm_model=minilm,
        minilm_required_tree_sha256=preflight.MINILM_REQUIRED_TREE_SHA256,
        cross_encoder_model=cross_encoder,
        cross_encoder_required_tree_sha256=preflight.CE_REQUIRED_TREE_SHA256,
        nvidia_smi=nvidia_smi,
        systemd_run=systemd_run,
        systemd_run_resolved_sha256=hashlib.sha256(
            systemd_run.read_bytes()
        ).hexdigest(),
        systemd_isolation_disposition_sha256=(
            runtime._semantic_hash(  # noqa: SLF001
                outer._transient_unit_contract(project)  # noqa: SLF001
            )
        ),
        runtime_module_sha256=hashlib.sha256(
            Path(runtime.__file__).read_bytes()
        ).hexdigest(),
        official_preflight_receipt=(
            project / "manifests/official-preflight.json"
        ),
        official_preflight_receipt_sha256="a" * 64,
        official_runtime_paths=_official_paths(tmp_path),
        official_expected_package_versions={
            "torch": "synthetic",
            "sentence-transformers": "synthetic",
            "transformers": "synthetic",
        },
        official_expected_module_import_roots={
            "torch": "overlay_root",
            "sentence_transformers": "overlay_root",
            "transformers": "overlay_root",
            "hipporag": "hipporag_source_root",
        },
        selection_acquisition_sha256="6" * 64,
        controller_arguments=(
            "--execution-freeze-self-sha256",
            "1" * 64,
            "--local-preflight-self-sha256",
            "3" * 64,
            "--selection-acquisition-sha256",
            "6" * 64,
            "--official-preflight-receipt-sha256",
            "a" * 64,
            "--official-runtime-python",
            str(_official_paths(tmp_path).runtime_python),
            "--official-pyvenv-cfg",
            str(_official_paths(tmp_path).pyvenv_cfg),
            "--official-overlay-root",
            str(_official_paths(tmp_path).overlay_root),
            "--official-hipporag-source-root",
            str(_official_paths(tmp_path).hipporag_source_root),
            "--official-p16-site-root",
            str(_official_paths(tmp_path).p16_site_root),
            "--official-local-llm-model",
            str(_official_paths(tmp_path).local_llm_model),
            "--official-local-embedding-model",
            str(_official_paths(tmp_path).local_embedding_model),
        ),
    )


class FakeCoordinateProcessRunner:
    def __init__(self, *, fail_role: str | None = None) -> None:
        self.fail_role = fail_role
        self.calls: list[dict[str, Any]] = []
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        role = command[command.index("--role") + 1]
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls.append(
                {
                    "role": role,
                    "command": tuple(command),
                    "env": dict(kwargs["env"]),
                }
            )
        try:
            self.barrier.wait(timeout=5)
            if role == self.fail_role:
                return SimpleNamespace(
                    returncode=17, stdout=b"failed", stderr=b"synthetic"
                )
            input_path = Path(command[command.index("--input-block") + 1])
            output_path = Path(command[command.index("--output") + 1])
            model_path = command[command.index("--model") + 1]
            runtime_identity = command[
                command.index("--local-runtime-identity-sha256") + 1
            ]
            items = worker.load_anonymous_block(input_path)
            rows = tuple(
                worker.CoordinateRow(
                    item.work_id,
                    unit.ordinal,
                    (
                        0.10 + 0.01 * unit.ordinal
                        if role == worker.ROLE_MINILM
                        else 0.80 - 0.01 * unit.ordinal
                    ),
                )
                for item in items
                for unit in item.work_item.units
            )
            unit_count = len(rows)
            inputs = (
                unit_count + len(items)
                if role == worker.ROLE_MINILM
                else unit_count
            )
            archive = worker.BlockCoordinateArchive(
                role=role,
                device=worker.ROLE_DEVICE[role],
                model_id=worker.ROLE_MODEL_ID[role],
                required_tree_sha256=worker.ROLE_REQUIRED_TREE_SHA256[role],
                model_path_sha256=hashlib.sha256(
                    model_path.encode("utf-8")
                ).hexdigest(),
                local_runtime_identity_sha256=runtime_identity,
                anonymous_block_sha256=str(
                    worker.anonymous_block_payload(items)["block_sha256"]
                ),
                item_count=len(items),
                unit_count=unit_count,
                inference_input_count=inputs,
                frozen_batch_size=worker.ROLE_BATCH_SIZE[role],
                frozen_max_length=worker.ROLE_MAX_LENGTH[role],
                model_initialization_count=1,
                batch_call_count=(
                    inputs + worker.ROLE_BATCH_SIZE[role] - 1
                )
                // worker.ROLE_BATCH_SIZE[role],
                rows=rows,
            )
            worker.write_private_coordinate_archive(output_path, archive)
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        finally:
            with self.lock:
                self.active -= 1


def _runtime_binding(tmp_path: Path) -> hippo.FreshComparatorRuntimeBinding:
    return hippo.FreshComparatorRuntimeBinding(
        paths=_official_paths(tmp_path),
        receipt_sha256="a" * 64,
        filesystem_binding_sha256="b" * 64,
        runtime_probe_sha256="c" * 64,
        address_family_isolation_probe_sha256="d" * 64,
        public_synthetic_output_sha256="e" * 64,
        _capability=hippo._FRESH_BINDING_CAPABILITY,  # noqa: SLF001
    )


def _candidate_payload() -> local.CandidateRestrictedHippoRAGPayload:
    texts = tuple(f"synthetic candidate {index}" for index in range(6))
    ordinals = tuple(range(6))
    return local.CandidateRestrictedHippoRAGPayload(
        query="Which synthetic candidate is relevant?",
        logical_source_ordinals=ordinals,
        exact_sentence_texts=texts,
        closure_ordinal_bytes_sha256=hashlib.sha256(
            json.dumps(
                list(ordinals),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        exact_text_quotient_count=len(texts),
    )


class FakeHippoBlock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        items: tuple[hippo.AHoldHippoItem, ...],
        *,
        runtime_binding: hippo.FreshComparatorRuntimeBinding,
        stage_parent: Path,
    ) -> hippo.OfficialTerminalArchive:
        self.calls += 1
        Path(stage_parent).mkdir(mode=0o700)
        rows = tuple(
            hippo.OfficialTerminalRow(
                work_id=item.work_id,
                top5_source_ordinals=tuple(
                    item.payload.logical_source_ordinals[:5]
                ),
                candidate_payload_sha256=item.candidate_payload_sha256,
                closure_ordinal_bytes_sha256=(
                    item.payload.closure_ordinal_bytes_sha256
                ),
                worker_output_sha256=hashlib.sha256(
                    item.work_id.encode("ascii")
                ).hexdigest(),
            )
            for item in items
        )
        return hippo.OfficialTerminalArchive(
            runtime_binding_sha256=runtime_binding.binding_sha256,
            address_family_isolation_probe_sha256=(
                runtime_binding.address_family_isolation_probe_sha256
            ),
            A_hold_input_sha256=hippo.ahold_input_sha256(items),
            rows=rows,
        )


def test_coordinate_provider_runs_exactly_two_concurrent_safe_clis(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    private = config.project_root / "private"
    private.mkdir(mode=0o700)
    process = FakeCoordinateProcessRunner()
    provider = runtime.FormalCoordinateProvider(
        config=config,
        private_root=private,
        local_runtime_identity_sha256="4" * 64,
        process_runner=process,
    )
    work_id = _work_id("coordinate")
    item = _item("coordinate")
    result = provider(block="A_form", items={work_id: item})

    assert len(process.calls) == 2
    assert process.maximum_active == 2
    assert {call["role"] for call in process.calls} == set(worker.ROLES)
    for call in process.calls:
        assert not any(
            fragment in key.upper()
            for key in call["env"]
            for fragment in ("OPENAI", "RUOLI", "API_KEY", "PROXY")
        )
        assert call["env"]["HF_HUB_OFFLINE"] == "1"
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    coordinates = result[work_id]
    assert len(coordinates) == len(item.units)
    for unit, row in zip(item.units, coordinates, strict=True):
        assert (
            row.entity_anchor,
            row.relation_anchor,
            row.numeric_or_temporal_anchor,
        ) == local.deterministic_anchor_flags(
            item.question, unit.serialized_content
        )
    for path in (private / "coordinates" / "A_form").glob("*.json"):
        assert _mode(path) == 0o600
    with pytest.raises(
        runtime.MmqaP1FormalActionRuntimeError, match="replayed"
    ):
        provider(block="A_form", items={work_id: item})
    assert len(process.calls) == 2


def test_controller_argument_template_restores_exactly_three_markers(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    template = runtime.controller_argument_template(config)

    assert template.count(runtime.EXECUTION_FREEZE_ARGUMENT_MARKER) == 1
    assert template.count(runtime.LOCAL_PREFLIGHT_ARGUMENT_MARKER) == 1
    assert template.count(runtime.SELECTION_ACQUISITION_ARGUMENT_MARKER) == 1
    assert "1" * 64 not in template
    assert "3" * 64 not in template
    assert "6" * 64 not in template
    for flag, field in runtime._OFFICIAL_RUNTIME_ARGUMENT_FIELDS.items():  # noqa: SLF001
        assert template.count(flag) == 1
        position = template.index(flag)
        assert template[position + 1] == str(
            getattr(config.official_runtime_paths, field)
        )
    assert template.count("--official-preflight-receipt-sha256") == 1
    receipt_position = template.index(
        "--official-preflight-receipt-sha256"
    )
    assert template[receipt_position + 1] == (
        config.official_preflight_receipt_sha256
    )

    with pytest.raises(
        runtime.MmqaP1FormalActionRuntimeError,
        match="must occur once",
    ):
        replace(
            config,
            controller_arguments=(
                *config.controller_arguments,
                "--selection-acquisition-sha256",
                config.selection_acquisition_sha256,
            ),
        )


def test_runtime_schema_matches_outer_execution_freeze_contract(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    assert runtime.SYSTEMD_EXECUTION_POLICY == outer.EXECUTION_POLICY
    assert (
        runtime.SYSTEMD_PARENT_NETWORK_CONTRACT
        == outer.OUTER_NETWORK_ISOLATION_CONTRACT
    )
    assert runtime._formal_action_transient_unit_contract(  # noqa: SLF001
        config
    ) == outer._transient_unit_contract(config.project_root)  # noqa: SLF001
    assert runtime._official_preflight_receipt_contract(  # noqa: SLF001
        config
    ) == {
        "relative_path": "manifests/official-preflight.json",
        "schema": hippo.FRESH_PREFLIGHT_SCHEMA,
        "self_hash_field": "receipt_sha256",
        "self_sha256": config.official_preflight_receipt_sha256,
        "status": "passed_public_synthetic_candidate_only_fresh_runtime",
    }
    official = runtime._official_runtime_contract(config)  # noqa: SLF001
    assert official["path_binding"] == config.official_runtime_paths.path_binding()
    assert tuple(official["paths"]) == tuple(
        runtime._OFFICIAL_RUNTIME_ARGUMENT_FIELDS.values()  # noqa: SLF001
    )
    assert len(official["paths"]) == 7


def test_local_preflight_revalidates_active_address_family_probe_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    probe = {
        "AF_INET6_socket_creation_errno": "EAFNOSUPPORT",
        "AF_INET_socket_creation_errno": "EAFNOSUPPORT",
        "address_family_isolation_contract": (
            preflight.ADDRESS_FAMILY_ISOLATION_CONTRACT
        ),
        "denied_family_count": 2,
        "probe_count": 2,
        "status": "AF_INET_and_AF_INET6_socket_creation_denied",
    }
    body = {
        "schema": preflight.RECEIPT_SCHEMA,
        "status": (
            "passed_public_synthetic_non_scoring_runtime_action_preflight"
        ),
        "study_design_self_sha256": runtime.STUDY_DESIGN_SELF_SHA256,
        "address_family_isolation_probe": probe,
        "address_family_isolation_probe_sha256": runtime._semantic_hash(  # noqa: SLF001
            probe
        ),
        "asset_bindings": {
            "minilm_required_tree_sha256": (
                config.minilm_required_tree_sha256
            ),
            "cross_encoder_required_tree_sha256": (
                config.cross_encoder_required_tree_sha256
            ),
        },
        "runtime_binding": {
            "typed_runtime_identity_sha256": "4" * 64,
            "typed_runtime_lexical_path_sha256": hashlib.sha256(
                os.fsencode(str(config.typed_python))
            ).hexdigest(),
        },
        "claim_boundary": {
            "api_or_provider_call_count": 0,
            "formal_HippoRAG_call_count": 0,
            "formal_MMQA_source_or_row_access_count": 0,
            "label_or_score_access_count": 0,
            "online_evaluator_call_count": 0,
            "retry_replay_or_resample_count": 0,
        },
        "concurrency": {
            "cross_encoder_physical_gpu_1_process_cap": 1,
            "minilm_physical_gpu_0_process_cap": 1,
            "model_process_co_residency": False,
            "typed_gpu_process_cap": 2,
        },
    }
    receipt = runtime._self_hashed(body)  # noqa: SLF001
    runtime._write_once(config.local_preflight_receipt, receipt)  # noqa: SLF001
    arguments = list(config.controller_arguments)
    local_position = arguments.index("--local-preflight-self-sha256")
    arguments[local_position + 1] = receipt["self_sha256"]
    config = replace(
        config,
        local_preflight_self_sha256=receipt["self_sha256"],
        controller_arguments=tuple(arguments),
    )
    monkeypatch.setattr(
        preflight,
        "_verify_typed_python",
        lambda _path: {
            "executable_resolved_file_sha256": (
                config.typed_python_resolved_sha256
            )
        },
    )
    monkeypatch.setattr(
        preflight,
        "_verify_minilm_asset",
        lambda _path: {
            "required_tree_sha256": config.minilm_required_tree_sha256
        },
    )
    monkeypatch.setattr(
        preflight,
        "_verify_ce_asset",
        lambda _path: {
            "required_tree_sha256": config.cross_encoder_required_tree_sha256
        },
    )
    calls = 0

    def isolation_inspector() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return probe

    result = runtime.validate_local_preflight_once(
        config,
        isolation_inspector=isolation_inspector,
    )

    assert calls == 1
    assert result["address_family_isolation_revalidation_count"] == 1
    assert result["address_family_isolation_probe_sha256"] == (
        runtime._semantic_hash(probe)  # noqa: SLF001
    )


def test_official_preflight_rejects_another_valid_self_hashed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    replacement = runtime._self_hashed(  # noqa: SLF001
        {
            "schema": hippo.FRESH_PREFLIGHT_SCHEMA,
            "status": (
                "passed_public_synthetic_candidate_only_fresh_runtime"
            ),
            "expected_package_versions": dict(
                config.official_expected_package_versions
            ),
            "expected_module_import_roots": dict(
                config.official_expected_module_import_roots
            ),
            "replacement_receipt": True,
        },
        "receipt_sha256",
    )
    assert replacement["receipt_sha256"] != (
        config.official_preflight_receipt_sha256
    )
    runtime._write_once(  # noqa: SLF001
        config.official_preflight_receipt,
        replacement,
    )
    monkeypatch.setattr(
        hippo,
        "validate_fresh_preflight_receipt",
        lambda *_args, **_kwargs: pytest.fail(
            "replacement receipt reached live validation"
        ),
    )

    with pytest.raises(
        runtime.MmqaP1FormalActionRuntimeError,
        match="expected hash drifted",
    ):
        runtime.validate_official_preflight_once(config)


def test_coordinate_failure_has_no_retry_or_oom_resize(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    private = config.project_root / "private"
    private.mkdir(mode=0o700)
    process = FakeCoordinateProcessRunner(
        fail_role=worker.ROLE_CROSS_ENCODER
    )
    provider = runtime.FormalCoordinateProvider(
        config=config,
        private_root=private,
        local_runtime_identity_sha256="4" * 64,
        process_runner=process,
    )
    with pytest.raises(
        runtime.MmqaP1FormalActionRuntimeError,
        match="no retry permitted",
    ):
        provider(
            block="A_form",
            items={_work_id("failure"): _item("failure")},
        )
    assert len(process.calls) == 2
    with pytest.raises(runtime.MmqaP1FormalActionRuntimeError):
        provider(
            block="A_form",
            items={_work_id("failure"): _item("failure")},
        )
    assert len(process.calls) == 2


def test_official_executor_seals_rereads_one_cpu_block(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    block_runner = FakeHippoBlock()
    executor = runtime.FormalHippoExecutor(
        runtime_binding=_runtime_binding(tmp_path),
        private_root=private,
        block_runner=block_runner,
    )
    payload = _candidate_payload()
    payloads = {
        _work_id(f"hippo-{index}"): payload for index in range(45)
    }
    result = executor(block="A_hold", payloads=payloads)

    assert block_runner.calls == 1
    assert set(result) == set(payloads)
    assert all(tuple(value) == (0, 1, 2, 3, 4) for value in result.values())
    terminal = (
        private
        / "official_A_hold"
        / runtime.HIPPORAG_TERMINAL_FILENAME
    )
    assert terminal.is_file()
    assert _mode(terminal) == 0o600
    assert executor.safe_summary()["max_workers"] == 4
    with pytest.raises(
        runtime.MmqaP1FormalActionRuntimeError, match="replayed"
    ):
        executor(block="A_hold", payloads=payloads)
    assert block_runner.calls == 1


def test_active_systemd_probe_requires_errno97_nnp_and_umask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.EAFNOSUPPORT, "Address family not supported")

    original = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == Path("/proc/self/status"):
            return "Name:\tpython\nUmask:\t0077\nNoNewPrivs:\t1\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runtime.socket, "socket", denied)
    monkeypatch.setattr(Path, "read_text", read_text)
    receipt = runtime.verify_systemd_isolation_once(
        _config(tmp_path),
        process_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"0\n1\n",
            stderr=b"",
        ),
    )
    assert receipt["probe_count"] == 6
    assert receipt["denied_family_count"] == 2
    assert receipt["NoNewPrivs_verified"] is True
    assert receipt["UMask_0077_verified"] is True
    assert receipt["gpu_0_visible"] is True
    assert receipt["gpu_1_visible"] is True


def test_complete_runtime_binds_safe_terminals_without_source_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    process = FakeCoordinateProcessRunner()
    block_runner = FakeHippoBlock()
    binding = _runtime_binding(tmp_path)
    calls = {
        "isolation": 0,
        "freeze": 0,
        "local": 0,
        "official": 0,
        "controller": 0,
    }
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("RUOLI_TOKEN", "must-not-be-forwarded")

    def isolation_verifier(
        observed: runtime.FormalActionRuntimeConfig,
    ) -> dict[str, Any]:
        calls["isolation"] += 1
        assert observed is config
        return {
            "status": "synthetic_outer_systemd_isolation_verified",
            "outer_network_isolation_contract_sha256": (
                runtime.SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256
            ),
            "formal_action_transient_unit_contract_sha256": (
                config.systemd_isolation_disposition_sha256
            ),
            "NoNewPrivs_verified": True,
            "UMask_0077_verified": True,
            "gpu_0_visible": True,
            "gpu_1_visible": True,
            "probe_binding_sha256": "9" * 64,
            "verification_count": 1,
        }

    def freeze_verifier(
        observed: runtime.FormalActionRuntimeConfig,
    ) -> dict[str, Any]:
        calls["freeze"] += 1
        assert observed is config
        return {
            "status": "synthetic_freezes_verified",
            "controller_argument_template_sha256": (
                runtime._semantic_hash(  # noqa: SLF001
                    list(runtime.controller_argument_template(config))
                )
            ),
            "verification_count": 1,
        }

    def local_verifier(
        observed: runtime.FormalActionRuntimeConfig,
    ) -> dict[str, Any]:
        calls["local"] += 1
        assert observed is config
        return {
            "status": "synthetic_local_preflight_verified",
            "typed_runtime_identity_sha256": "4" * 64,
            "verification_count": 1,
        }

    def official_verifier(
        observed: runtime.FormalActionRuntimeConfig,
    ) -> hippo.FreshComparatorRuntimeBinding:
        calls["official"] += 1
        assert observed is config
        return binding

    def controller_runner(**kwargs: Any) -> dict[str, Any]:
        calls["controller"] += 1
        assert kwargs["expected_selection_acquisition_sha256"] == "6" * 64
        coordinate_provider = kwargs["coordinate_provider"]
        hippo_executor = kwargs["hippo_executor"]
        coordinate_provider(
            block="A_form",
            items={_work_id("runtime-coordinate"): _item("runtime-coordinate")},
        )
        payload = _candidate_payload()
        hippo_executor(
            block="A_hold",
            payloads={
                _work_id(f"runtime-hippo-{index}"): payload
                for index in range(45)
            },
        )
        control_root = Path(kwargs["control_root"])
        controller._ensure_private_directory(control_root)  # noqa: SLF001
        body = {
            "schema": (
                f"{controller.VERSION}_hash_safe_final_receipt_v1"
            ),
            "version": controller.VERSION,
            "study_id": controller.STUDY_ID,
            "status": "lifecycle_complete_promoted_M_scored",
            "A_hold": {
                "promoted": True,
                "reality_primary_passed": False,
            },
            "M_search": {
                "authorized": True,
                "gold_opened": True,
                "L5_passed": True,
            },
        }
        terminal = controller.self_hashed(body, "final_sha256")
        controller._write_once(  # noqa: SLF001
            control_root / controller.FINAL_RECEIPT_FILENAME,
            terminal,
        )
        return terminal

    wrapper = runtime.run_formal_action_runtime(
        config,
        isolation_verifier=isolation_verifier,
        freeze_verifier=freeze_verifier,
        local_preflight_verifier=local_verifier,
        official_preflight_verifier=official_verifier,
        coordinate_process_runner=process,
        official_block_runner=block_runner,
        controller_runner=controller_runner,
    )

    assert calls == {
        "isolation": 1,
        "freeze": 1,
        "local": 1,
        "official": 1,
        "controller": 1,
    }
    assert wrapper["status"] == runtime.WRAPPER_STATUS
    assert wrapper["outcomes"] == {
        "promotion_passed": True,
        "reality_primary_passed": False,
        "M_search_authorized": True,
        "M_search_gold_opened": True,
        "L5_passed": True,
    }
    official_safe = wrapper["official_A_hold"]
    assert official_safe["runtime_binding_sha256"] == binding.binding_sha256
    assert official_safe["archive_sha256"]
    assert official_safe["archive_file_sha256"]
    assert official_safe["A_hold_input_sha256"]
    assert official_safe["item_launcher_call_count"] == 45
    assert official_safe["bwrap_call_count"] == 0
    assert wrapper["claim_boundary"]["nested_bwrap_launch_count"] == 0
    assert process.maximum_active == 2
    assert block_runner.calls == 1
    assert all(
        not any(
            fragment in key.upper()
            for key in call["env"]
            for fragment in ("OPENAI", "RUOLI", "API_KEY")
        )
        for call in process.calls
    )

    root = config.project_root / runtime.RUNTIME_ROOT_RELATIVE
    terminal_path = root / runtime.WRAPPER_TERMINAL_FILENAME
    assert terminal_path.is_file()
    assert _mode(terminal_path) == 0o600
    for path in root.rglob("*"):
        if path.is_file():
            assert _mode(path) == 0o600
        elif path.is_dir():
            assert _mode(path) == 0o700
