from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import errno
import hashlib
import inspect
import json
from pathlib import Path
import stat
import threading
import time
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import mmqa_p1_local_action_executor_v1 as action
from assumption_agent.benchmarks import mmqa_p1_official_hipporag_block_v1 as hippo


def _work_id(index: int) -> str:
    return f"mmqa-work-v1-{index:064x}"


def _payload(index: int) -> action.CandidateRestrictedHippoRAGPayload:
    ordinals = tuple(index * 10 + offset for offset in range(6))
    texts = tuple(
        f"Public synthetic candidate item {index} sentence {offset}."
        for offset in range(6)
    )
    ordinal_bytes = json.dumps(
        list(ordinals), separators=(",", ":")
    ).encode("ascii")
    return action.CandidateRestrictedHippoRAGPayload(
        query=f"Which synthetic candidate belongs to item {index}?",
        logical_source_ordinals=ordinals,
        exact_sentence_texts=texts,
        closure_ordinal_bytes_sha256=hashlib.sha256(
            ordinal_bytes
        ).hexdigest(),
        exact_text_quotient_count=6,
    )


def _items() -> tuple[hippo.AHoldHippoItem, ...]:
    return tuple(
        hippo.AHoldHippoItem(_work_id(index), _payload(index))
        for index in range(hippo.A_HOLD_ITEM_COUNT)
    )


def _paths() -> hippo.FreshComparatorRuntimePaths:
    return hippo.FreshComparatorRuntimePaths(
        runtime_python="/runtime/venv/bin/python",
        pyvenv_cfg="/runtime/venv/pyvenv.cfg",
        overlay_root="/runtime/mmqa_overlay",
        hipporag_source_root="/runtime/HippoRAG",
        p16_site_root="/runtime/p16_site",
        local_llm_model="/runtime/models/smollm2",
        local_embedding_model="/runtime/models/minilm",
    )


def _filesystem() -> dict[str, str]:
    return {
        "runtime_python": "1" * 64,
        "pyvenv_cfg": "2" * 64,
        "overlay_root": "3" * 64,
        "hipporag_source_root": "4" * 64,
        "p16_site_root": "5" * 64,
        "local_llm_model": "6" * 64,
        "local_embedding_model": "7" * 64,
        "eraser_adapter_file": hippo.ERASER_ADAPTER_FILE_SHA256,
        "eraser_worker_file": hippo.ERASER_WORKER_FILE_SHA256,
        "eraser_contract_file": hippo.ERASER_CONTRACT_FILE_SHA256,
    }


def _versions() -> dict[str, str]:
    return {
        "torch": "2.4.1+cu118",
        "sentence-transformers": "3.1.1",
        "transformers": "4.45.2",
    }


def _roots() -> dict[str, str]:
    return {
        "torch": "p16_site_root",
        "sentence_transformers": "p16_site_root",
        "transformers": "p16_site_root",
        "hipporag": "hipporag_source_root",
    }


def _runtime_probe() -> dict[str, object]:
    return {
        "python_version": "3.10.14",
        "package_versions": _versions(),
        "module_origins": {
            "torch": "/runtime/p16_site/torch/__init__.py",
            "sentence_transformers": (
                "/runtime/p16_site/sentence_transformers/__init__.py"
            ),
            "transformers": "/runtime/p16_site/transformers/__init__.py",
            "hipporag": "/runtime/HippoRAG/src/hipporag/__init__.py",
        },
        "module_import_roots": _roots(),
        "sys_path_sha256": "8" * 64,
        "cuda_visible_devices": "",
        "cpu_thread_env": dict(hippo.CPU_THREAD_ENV),
        "address_family_isolation_probe": _isolation_probe(),
    }


def _isolation_probe() -> dict[str, object]:
    return {
        "schema": hippo.ADDRESS_FAMILY_ISOLATION_PROBE_SCHEMA,
        "required_RestrictAddressFamilies": ["AF_UNIX"],
        "AF_INET_socket_creation_errno": "EAFNOSUPPORT",
        "AF_INET6_socket_creation_errno": "EAFNOSUPPORT",
        "all_inet_socket_creation_denied": True,
        "private_network_namespace_claimed": False,
        "probe_count": 2,
    }


def _binding(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    hippo.FreshComparatorRuntimeBinding,
    dict[str, int],
]:
    calls = {"filesystem": 0, "runtime": 0, "canary": 0, "isolation": 0}

    def filesystem(paths: hippo.FreshComparatorRuntimePaths) -> object:
        assert paths == _paths()
        calls["filesystem"] += 1
        return _filesystem()

    def runtime(
        paths: hippo.FreshComparatorRuntimePaths,
        *,
        package_names: tuple[str, ...],
        module_names: tuple[str, ...],
    ) -> object:
        assert paths == _paths()
        assert package_names == tuple(sorted(_versions()))
        assert module_names == tuple(sorted(_roots()))
        calls["runtime"] += 1
        return _runtime_probe()

    def canary(**kwargs: object) -> object:
        calls["canary"] += 1
        assert kwargs["runtime_paths"] == _paths()
        assert kwargs["timeout_seconds"] == hippo.ITEM_TIMEOUT_SECONDS
        assert isinstance(
            kwargs["payload"],
            action.CandidateRestrictedHippoRAGPayload,
        )
        assert not kwargs["work_root"].exists()  # type: ignore[union-attr]
        return (5, 2, 0, 1, 3)

    def isolation() -> object:
        calls["isolation"] += 1
        return _isolation_probe()

    receipt, binding = hippo.build_fresh_comparator_preflight(
        paths=_paths(),
        expected_package_versions=_versions(),
        expected_module_import_roots=_roots(),
        canary_stage_parent=tmp_path / "fresh-canary-stage",
        filesystem_inspector=filesystem,
        runtime_inspector=runtime,
        canary_launcher=canary,
        isolation_inspector=isolation,
    )
    return receipt, binding, calls


def _run_archive(
    tmp_path: Path,
    *,
    launcher=None,
) -> tuple[
    hippo.OfficialTerminalArchive,
    hippo.FreshComparatorRuntimeBinding,
]:
    _receipt, binding, _calls = _binding(tmp_path)
    if launcher is None:
        launcher = lambda **_kwargs: (5, 4, 3, 2, 1)
    archive = hippo.run_ahold_official_hipporag_block(
        _items(),
        runtime_binding=binding,
        stage_parent=tmp_path / "A_hold-official-stage",
        item_launcher=launcher,
        _isolation_inspector=_isolation_probe,
    )
    return archive, binding


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    body = {
        key: value for key, value in payload.items() if key != "archive_sha256"
    }
    payload["archive_sha256"] = hippo._semantic_hash(body)  # noqa: SLF001
    return payload


def _all_keys(value: object) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        output.update(str(key) for key in value)
        for nested in value.values():
            output.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            output.update(_all_keys(nested))
    return output


def test_gpu_design_conflict_is_explicitly_disposed_as_cpu_without_manifest_edit() -> None:
    assert hippo.MAX_WORKERS == 4
    assert hippo.ATTESTED_CUDA_VISIBLE_DEVICES == ""
    assert hippo.DESIGN_TWO_PER_GPU_APPLIED is False
    assert hippo.DESIGN_CONCURRENCY_CONFLICT_REQUIRES_UNIFIED_DISPOSITION is True
    assert hippo.EXECUTION_DEVICE_DISPOSITION.startswith("four_way_cpu")
    assert hippo.CPU_THREAD_ENV == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    source = inspect.getsource(hippo.production_item_launcher)
    assert "verify_formal_runtime_attestation_v3" not in source
    assert "run_item_local_official_hipporag_v1" not in source
    assert "run_fresh_bound_item" in source
    assert hippo.NETWORK_ISOLATION_POLICY == (
        "parent_systemd_RestrictAddressFamilies_AF_UNIX_inherited_v1"
    )
    assert hippo.ADDRESS_FAMILY_ISOLATION_CONTRACT[
        "private_network_namespace_claimed"
    ] is False
    assert "bwrap" not in inspect.getsource(hippo.production_runtime_inspector)
    assert "bwrap" not in inspect.getsource(hippo._launch_fresh_bound_worker)  # noqa: SLF001


def test_effective_address_family_probe_checks_both_inet_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, int]] = []

    def deny(family: int, kind: int) -> object:
        observed.append((family, kind))
        raise OSError(errno.EAFNOSUPPORT, "address family not supported")

    monkeypatch.setattr(hippo.socket, "socket", deny)
    assert hippo.production_address_family_isolation_probe() == (
        _isolation_probe()
    )
    assert observed == [
        (hippo.socket.AF_INET, hippo.socket.SOCK_STREAM),
        (hippo.socket.AF_INET6, hippo.socket.SOCK_STREAM),
    ]


@pytest.mark.parametrize("failure", ["allowed", "wrong_errno"])
def test_effective_address_family_probe_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    class AllowedSocket:
        def close(self) -> None:
            return None

    def socket_factory(_family: int, _kind: int) -> object:
        if failure == "allowed":
            return AllowedSocket()
        raise OSError(errno.EPERM, "wrong effective policy")

    monkeypatch.setattr(hippo.socket, "socket", socket_factory)
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError,
        match="succeeded|not EAFNOSUPPORT",
    ):
        hippo.production_address_family_isolation_probe()


def test_runtime_inspector_is_direct_and_child_probe_must_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    child = {
        "python_version": "3.10.14",
        "package_versions": _versions(),
        "module_origins": {
            "torch": "/runtime/p16_site/torch/__init__.py",
            "sentence_transformers": (
                "/runtime/p16_site/sentence_transformers/__init__.py"
            ),
            "transformers": "/runtime/p16_site/transformers/__init__.py",
            "hipporag": "/runtime/HippoRAG/src/hipporag/__init__.py",
        },
        "sys_path": [
            "/runtime/mmqa_overlay",
            "/runtime/HippoRAG",
            "/runtime/p16_site",
        ],
        "cuda_visible_devices": "",
        "cpu_thread_env": dict(hippo.CPU_THREAD_ENV),
        "address_family_isolation_probe": _isolation_probe(),
    }

    def run(command: object, **kwargs: object) -> object:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=hippo._canonical_json_bytes(child),  # noqa: SLF001
            stderr=b"",
        )

    monkeypatch.setattr(
        hippo,
        "production_address_family_isolation_probe",
        _isolation_probe,
    )
    monkeypatch.setattr(hippo.subprocess, "run", run)
    result = hippo.production_runtime_inspector(
        _paths(),
        package_names=tuple(sorted(_versions())),
        module_names=tuple(sorted(_roots())),
    )
    command = captured["command"]
    assert command[0] == _paths().runtime_python
    assert "/usr/bin/bwrap" not in command
    assert "--unshare-net" not in command
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result["address_family_isolation_probe"] == _isolation_probe()
    assert result["module_import_roots"] == _roots()


def test_fresh_preflight_binds_actual_overlay_site_versions_and_one_canary(
    tmp_path: Path,
) -> None:
    receipt, binding, calls = _binding(tmp_path)
    assert calls == {
        "filesystem": 1,
        "runtime": 1,
        "canary": 1,
        "isolation": 1,
    }
    assert receipt["status"] == (
        "passed_public_synthetic_candidate_only_fresh_runtime"
    )
    assert receipt["filesystem_binding"] == _filesystem()
    assert receipt["runtime_probe"] == _runtime_probe()
    assert receipt["expected_package_versions"] == _versions()
    assert receipt["expected_module_import_roots"] == _roots()
    assert receipt["benchmark_rows_read"] == 0
    assert receipt["scores_computed"] == 0
    assert receipt["address_family_isolation_contract"] == (
        hippo.ADDRESS_FAMILY_ISOLATION_CONTRACT
    )
    assert receipt["address_family_isolation_probe"] == _isolation_probe()
    assert receipt["worker_subprocess_inherits_parent_restriction"] is True
    assert receipt["bwrap_call_count"] == 0
    assert receipt["cuda_visible_devices"] == ""
    assert receipt["cpu_thread_env"] == hippo.CPU_THREAD_ENV
    assert receipt["retry_count"] == 0
    assert len(binding.binding_sha256) == 64
    assert stat.S_IMODE(
        (tmp_path / "fresh-canary-stage").stat().st_mode
    ) == 0o700


def test_fresh_preflight_receipt_is_one_shot_0600_and_revalidates_files(
    tmp_path: Path,
) -> None:
    receipt, binding, _calls = _binding(tmp_path)
    path = tmp_path / "fresh-comparator-runtime.private.json"
    file_sha = hippo.write_fresh_preflight_receipt(path, receipt)
    assert len(file_sha) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes() == hippo._canonical_json_bytes(  # noqa: SLF001
        receipt, newline=True
    )
    observed: list[object] = []
    verified = hippo.validate_fresh_preflight_receipt(
        json.loads(path.read_text("ascii")),
        paths=_paths(),
        filesystem_inspector=lambda paths: (
            observed.append(paths) or _filesystem()
        ),
        isolation_inspector=_isolation_probe,
    )
    assert observed == [_paths()]
    assert verified.binding_sha256 == binding.binding_sha256
    loaded = hippo.load_fresh_preflight_binding(
        path,
        paths=_paths(),
        expected_receipt_sha256=str(receipt["receipt_sha256"]),
        filesystem_inspector=lambda _paths: _filesystem(),
        isolation_inspector=_isolation_probe,
    )
    assert loaded.binding_sha256 == binding.binding_sha256
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError, match="exists"
    ):
        hippo.write_fresh_preflight_receipt(path, receipt)


def test_fresh_preflight_receipt_revalidation_requires_current_effective_denial(
    tmp_path: Path,
) -> None:
    receipt, _binding_value, _calls = _binding(tmp_path)
    allowed = {
        **_isolation_probe(),
        "AF_INET_socket_creation_errno": "ALLOWED",
        "all_inet_socket_creation_denied": False,
    }
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError,
        match="isolation revalidation failed",
    ):
        hippo.validate_fresh_preflight_receipt(
            receipt,
            paths=_paths(),
            isolation_inspector=lambda: allowed,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cuda_visible_devices", "0", "policy"),
        (
            "package_versions",
            {
                "torch": "2.5.1",
                "sentence-transformers": "5.4.1",
                "transformers": "4.45.2",
            },
            "versions",
        ),
        (
            "module_import_roots",
            {
                **_roots(),
                "torch": "overlay_root",
            },
            "versions",
        ),
        (
            "cpu_thread_env",
            {**hippo.CPU_THREAD_ENV, "OMP_NUM_THREADS": "8"},
            "policy",
        ),
    ],
)
def test_fresh_preflight_fails_closed_on_old_or_drifted_runtime(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    probe = _runtime_probe()
    probe[field] = value
    with pytest.raises(hippo.MmqaP1OfficialHippoRAGBlockError, match=message):
        hippo.build_fresh_comparator_preflight(
            paths=_paths(),
            expected_package_versions=_versions(),
            expected_module_import_roots=_roots(),
            canary_stage_parent=tmp_path / f"bad-{field}",
            filesystem_inspector=lambda _paths: _filesystem(),
            runtime_inspector=lambda *_args, **_kwargs: probe,
            canary_launcher=lambda **_kwargs: (0, 1, 2, 3, 4),
            isolation_inspector=_isolation_probe,
        )


def test_fresh_binding_cannot_be_forged_directly() -> None:
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError, match="preflight"
    ):
        hippo.FreshComparatorRuntimeBinding(
            paths=_paths(),
            receipt_sha256="1" * 64,
            filesystem_binding_sha256="2" * 64,
            runtime_probe_sha256="3" * 64,
            address_family_isolation_probe_sha256="4" * 64,
            public_synthetic_output_sha256="5" * 64,
            _capability=object(),
        )


def test_canary_stage_cannot_mutate_a_bound_runtime_tree() -> None:
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError, match="mutate"
    ):
        hippo.build_fresh_comparator_preflight(
            paths=_paths(),
            expected_package_versions=_versions(),
            expected_module_import_roots=_roots(),
            canary_stage_parent="/runtime/mmqa_overlay/canary",
            filesystem_inspector=lambda _paths: _filesystem(),
            runtime_inspector=lambda *_args, **_kwargs: _runtime_probe(),
            canary_launcher=lambda **_kwargs: (0, 1, 2, 3, 4),
        )


def test_runtime_requires_exact_complete_ahold_candidate_payloads() -> None:
    items = _items()
    assert hippo.validate_ahold_items(items) == items
    with pytest.raises(hippo.MmqaP1OfficialHippoRAGBlockError, match="45-item"):
        hippo.validate_ahold_items(items[:-1])
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError, match="candidate-restricted"
    ):
        hippo.AHoldHippoItem(_work_id(99), {})  # type: ignore[arg-type]
    duplicated = (*items[:-1], items[0])
    with pytest.raises(hippo.MmqaP1OfficialHippoRAGBlockError, match="duplicate"):
        hippo.validate_ahold_items(duplicated)


def test_complete_block_uses_one_four_worker_pool_and_each_launcher_once(
    tmp_path: Path,
) -> None:
    _receipt, binding, _calls = _binding(tmp_path)
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    seen: dict[str, int] = {}
    pool_config: list[dict[str, object]] = []

    def launcher(**kwargs: object) -> object:
        nonlocal active, maximum_active
        payload = kwargs["payload"]
        assert isinstance(payload, action.CandidateRestrictedHippoRAGPayload)
        assert kwargs["runtime_binding"] is binding
        assert kwargs["timeout_seconds"] == 900
        assert not kwargs["work_root"].exists()  # type: ignore[union-attr]
        key = payload.query
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            seen[key] = seen.get(key, 0) + 1
        time.sleep(0.003)
        with lock:
            active -= 1
        return (5, 4, 3, 2, 1)

    def pool_factory(**kwargs: object) -> ThreadPoolExecutor:
        pool_config.append(kwargs)
        return ThreadPoolExecutor(**kwargs)  # type: ignore[arg-type]

    archive = hippo.run_ahold_official_hipporag_block(
        _items(),
        runtime_binding=binding,
        stage_parent=tmp_path / "formal-A_hold-stage",
        item_launcher=launcher,
        _executor_factory=pool_factory,
        _isolation_inspector=_isolation_probe,
    )
    assert pool_config == [
        {
            "max_workers": 4,
            "thread_name_prefix": "mmqa-p1-official-cpu",
        }
    ]
    assert maximum_active == 4
    assert len(seen) == 45
    assert set(seen.values()) == {1}
    assert tuple(row.work_id for row in archive.rows) == tuple(
        item.work_id for item in _items()
    )
    for index, row in enumerate(archive.rows):
        assert row.top5_source_ordinals == (
            index * 10 + 5,
            index * 10 + 4,
            index * 10 + 3,
            index * 10 + 2,
            index * 10 + 1,
        )


def test_formal_block_rechecks_isolation_before_stage_or_launcher(
    tmp_path: Path,
) -> None:
    _receipt, binding, _calls = _binding(tmp_path)
    stage = tmp_path / "must-not-be-created"
    launched = 0

    def launcher(**_kwargs: object) -> object:
        nonlocal launched
        launched += 1
        return (0, 1, 2, 3, 4)

    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError,
        match="RestrictAddressFamilies",
    ):
        hippo.run_ahold_official_hipporag_block(
            _items(),
            runtime_binding=binding,
            stage_parent=stage,
            item_launcher=launcher,
            _isolation_inspector=lambda: {
                **_isolation_probe(),
                "AF_INET6_socket_creation_errno": "ALLOWED",
            },
        )
    assert launched == 0
    assert not stage.exists()


def test_failure_or_oom_is_terminal_and_never_retried(tmp_path: Path) -> None:
    _receipt, binding, _calls = _binding(tmp_path)
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def launcher(**kwargs: object) -> object:
        payload = kwargs["payload"]
        key = payload.query  # type: ignore[union-attr]
        with lock:
            counts[key] = counts.get(key, 0) + 1
        if "item 2?" in key:
            raise RuntimeError("synthetic OOM")
        time.sleep(0.002)
        return (0, 1, 2, 3, 4)

    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError, match="no retry"
    ):
        hippo.run_ahold_official_hipporag_block(
            _items(),
            runtime_binding=binding,
            stage_parent=tmp_path / "failed-A_hold-stage",
            item_launcher=launcher,
            _isolation_inspector=_isolation_probe,
        )
    assert counts
    assert set(counts.values()) == {1}
    assert (tmp_path / "failed-A_hold-stage").exists()


@pytest.mark.parametrize(
    "result",
    [
        (0, 1, 2, 3),
        (0, 1, 2, 3, 3),
        (0, 1, 2, 3, 99),
        {"ordinals": [0, 1, 2, 3, 4]},
    ],
)
def test_launcher_terminal_is_strict_ordinal_only(
    tmp_path: Path, result: object
) -> None:
    _receipt, binding, _calls = _binding(tmp_path)
    with pytest.raises(hippo.MmqaP1OfficialHippoRAGBlockError):
        hippo.run_ahold_official_hipporag_block(
            _items(),
            runtime_binding=binding,
            stage_parent=tmp_path / f"bad-terminal-{len(str(result))}",
            item_launcher=lambda **_kwargs: result,
            _isolation_inspector=_isolation_probe,
        )


def test_item_work_root_must_be_destroyed(tmp_path: Path) -> None:
    _receipt, binding, _calls = _binding(tmp_path)

    def launcher(**kwargs: object) -> object:
        kwargs["work_root"].mkdir()  # type: ignore[union-attr]
        return (0, 1, 2, 3, 4)

    with pytest.raises(hippo.MmqaP1OfficialHippoRAGBlockError, match="destroy"):
        hippo.run_ahold_official_hipporag_block(
            _items(),
            runtime_binding=binding,
            stage_parent=tmp_path / "dirty-root-stage",
            item_launcher=launcher,
            _isolation_inspector=_isolation_probe,
        )


def test_terminal_archive_is_hash_and_ordinal_only_and_records_cpu_policy(
    tmp_path: Path,
) -> None:
    archive, _binding_value = _run_archive(tmp_path)
    payload = archive.payload()
    forbidden = {
        "question",
        "text",
        "content",
        "source",
        "gold",
        "answer",
        "support",
        "family",
        "qid",
        "score",
    }
    assert not (_all_keys(payload) & forbidden)
    rendered = json.dumps(payload, sort_keys=True)
    assert "Public synthetic candidate" not in rendered
    assert "/runtime/" not in rendered
    assert payload["max_workers"] == 4
    assert payload["attested_cuda_visible_devices"] == ""
    assert payload["cpu_thread_env_sha256"] == hippo._semantic_hash(  # noqa: SLF001
        hippo.CPU_THREAD_ENV
    )
    assert payload["item_launcher_call_count"] == 45
    assert payload["fresh_isolated_index_count"] == 45
    assert payload["network_isolation_policy"] == hippo.NETWORK_ISOLATION_POLICY
    assert payload["address_family_isolation_contract_sha256"] == (
        hippo._semantic_hash(hippo.ADDRESS_FAMILY_ISOLATION_CONTRACT)  # noqa: SLF001
    )
    assert payload["parent_address_family_restriction_inherited_count"] == 45
    assert payload["bwrap_call_count"] == 0
    assert payload["retry_replay_resample_count"] == 0
    assert payload["online_evaluator_call_count"] == 0
    assert all(set(row) == hippo._ROW_FIELDS for row in payload["rows"])  # noqa: SLF001


def test_terminal_archive_is_exclusive_canonical_mode_0600(
    tmp_path: Path,
) -> None:
    archive, _binding_value = _run_archive(tmp_path)
    path = tmp_path / "A_hold.hipporag-terminal.private.json"
    file_sha = hippo.write_private_terminal_archive(path, archive)
    assert len(file_sha) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes() == hippo._canonical_json_bytes(  # noqa: SLF001
        archive.payload(), newline=True
    )
    assert hippo.load_private_terminal_archive(path).payload() == archive.payload()
    with pytest.raises(hippo.MmqaP1OfficialHippoRAGBlockError, match="exists"):
        hippo.write_private_terminal_archive(path, archive)


def test_terminal_archive_reorder_or_cross_item_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    archive, _binding_value = _run_archive(tmp_path)
    payload = copy.deepcopy(archive.payload())
    payload["rows"][0], payload["rows"][1] = (  # type: ignore[index]
        payload["rows"][1],  # type: ignore[index]
        payload["rows"][0],  # type: ignore[index]
    )
    _rehash(payload)
    parsed = hippo.parse_terminal_archive_payload(payload)
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError, match="reordered"
    ):
        hippo.validate_terminal_archive_for_items(parsed, _items())

    payload = copy.deepcopy(archive.payload())
    payload["rows"][0]["top5_source_ordinals"] = [10, 11, 12, 13, 14]  # type: ignore[index]
    _rehash(payload)
    parsed = hippo.parse_terminal_archive_payload(payload)
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError, match="cross-item|outside"
    ):
        hippo.validate_terminal_archive_for_items(parsed, _items())


def test_fresh_worker_launch_is_direct_and_inherits_address_family_restriction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def run(command: object, **kwargs: object) -> object:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(hippo.subprocess, "run", run)
    monkeypatch.setattr(
        hippo,
        "production_address_family_isolation_probe",
        _isolation_probe,
    )
    writable = tmp_path / "work"
    writable.mkdir()
    hippo._launch_fresh_bound_worker(  # noqa: SLF001
        paths=_paths(),
        input_path=writable / "in.json",
        output_path=writable / "out.json",
        index_root=writable / "index",
        writable_root=writable,
        timeout_seconds=900,
    )
    command = captured["command"]
    assert command[0] == _paths().runtime_python
    assert "--unshare-net" not in command
    assert "/usr/bin/bwrap" not in command
    environment = captured["env"]
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    for key, value in hippo.CPU_THREAD_ENV.items():
        assert environment[key] == value
    assert captured["timeout"] == 900
    assert captured["cwd"] == writable
    assert captured["stdin"] is hippo.subprocess.DEVNULL


def test_fresh_bound_item_reuses_ordinal_worker_semantics_and_destroys_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def launch(**kwargs: object) -> None:
        observed.update(kwargs)
        input_payload = json.loads(
            kwargs["input_path"].read_text("utf-8")  # type: ignore[union-attr]
        )
        assert set(input_payload) == {"query", "schema", "sentence_texts"}
        kwargs["output_path"].write_bytes(b"[5,2,0,1,3]\n")  # type: ignore[union-attr]

    monkeypatch.setattr(hippo, "_launch_fresh_bound_worker", launch)
    work_root = tmp_path / "one-item.work"
    result = hippo.run_fresh_bound_item(
        payload=_payload(0),
        runtime_paths=_paths(),
        work_root=work_root,
    )
    assert result == (5, 2, 0, 1, 3)
    assert observed["paths"] == _paths()
    assert observed["timeout_seconds"] == 900
    assert not work_root.exists()


def test_preflight_cli_freezes_explicit_inputs_and_emits_only_safe_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt, binding, _calls = _binding(tmp_path)
    captured: dict[str, object] = {}

    def builder(**kwargs: object) -> object:
        captured.update(kwargs)
        return receipt, binding

    output = tmp_path / "formal" / "fresh-comparator.private.json"
    argv = [
        "--runtime-python",
        _paths().runtime_python,
        "--pyvenv-cfg",
        _paths().pyvenv_cfg,
        "--overlay-root",
        _paths().overlay_root,
        "--hipporag-source-root",
        _paths().hipporag_source_root,
        "--p16-site-root",
        _paths().p16_site_root,
        "--local-llm-model",
        _paths().local_llm_model,
        "--local-embedding-model",
        _paths().local_embedding_model,
        "--expected-package-versions-json",
        json.dumps(_versions(), sort_keys=True),
        "--expected-module-import-roots-json",
        json.dumps(_roots(), sort_keys=True),
        "--canary-stage-parent",
        str(tmp_path / "cli-canary"),
        "--output",
        str(output),
    ]
    assert hippo._preflight_main(  # noqa: SLF001
        argv,
        builder=builder,
        writer=hippo.write_fresh_preflight_receipt,
    ) == 0
    safe_raw = capsys.readouterr().out.strip()
    safe = json.loads(safe_raw)
    assert captured == {
        "paths": _paths(),
        "expected_package_versions": _versions(),
        "expected_module_import_roots": _roots(),
        "canary_stage_parent": tmp_path / "cli-canary",
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert output.read_bytes() == hippo._canonical_json_bytes(  # noqa: SLF001
        receipt, newline=True
    )
    assert safe["preflight_receipt_sha256"] == receipt["receipt_sha256"]
    assert safe["address_family_isolation_probe_sha256"] == receipt[
        "address_family_isolation_probe_sha256"
    ]
    assert safe["formal_source_read_count"] == 0
    assert safe["benchmark_rows_read"] == 0
    assert safe["scores_computed"] == 0
    assert safe["bwrap_call_count"] == 0
    assert safe["self_sha256"] == hippo._semantic_hash(  # noqa: SLF001
        {key: value for key, value in safe.items() if key != "self_sha256"}
    )
    assert "/runtime/" not in safe_raw
    assert "Synthetic Alpha" not in safe_raw
    with pytest.raises(
        hippo.MmqaP1OfficialHippoRAGBlockError,
        match="exists",
    ):
        hippo._preflight_main(  # noqa: SLF001
            argv,
            builder=builder,
            writer=hippo.write_fresh_preflight_receipt,
        )
