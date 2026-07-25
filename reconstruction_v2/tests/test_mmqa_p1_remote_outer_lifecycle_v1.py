from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import mmqa_p1_remote_outer_lifecycle_v1 as r


def _synthetic_outer_network_probe() -> dict[str, object]:
    return {
        "AF_INET6_socket_creation_errno": "EAFNOSUPPORT",
        "AF_INET_socket_creation_errno": "EAFNOSUPPORT",
        "denied_family_count": 2,
        "outer_network_isolation_contract": (
            r.OUTER_NETWORK_ISOLATION_CONTRACT
        ),
        "probe_count": 2,
        "schema": f"{r.VERSION}_outer_address_family_probe_v1",
        "status": "AF_INET_and_AF_INET6_socket_creation_denied",
    }


def _run_outer(
    config: r.OuterLifecycleConfig,
    stages: r.LifecycleStages,
) -> object:
    return r.run_outer_lifecycle(
        config,
        stages,
        outer_network_probe=_synthetic_outer_network_probe,
    )


def _write_receipt(
    project: Path,
    *,
    relative: Path,
    schema: str,
    status: str = "complete",
    self_hash_field: str = "self_sha256",
    mode: int = 0o600,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": schema,
        "status": status,
        "study_id": r.STUDY_ID,
    }
    if extra:
        body.update(extra)
    value = dict(body)
    value[self_hash_field] = r._semantic_hash(body)
    path = project / relative
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(r._canonical_bytes(value))
    os.chmod(path, mode)
    return value


def _with_canonical_controller_arguments(
    config: r.OuterLifecycleConfig,
) -> r.OuterLifecycleConfig:
    return replace(
        config,
        controller_arguments=r._canonical_formal_runtime_arguments(
            config,
            execution_freeze_sha256=config.execution_freeze_self_sha256,
        ),
    )


def _config(project: Path) -> r.OuterLifecycleConfig:
    official_body = {
        "prior_binding_count": 2,
        "schema": r.OFFICIAL_HIPPO_PREFLIGHT_SCHEMA,
        "status": r.OFFICIAL_HIPPO_PREFLIGHT_STATUS,
        "study_id": r.STUDY_ID,
    }
    config = r.OuterLifecycleConfig(
        project_root=project,
        execution_freeze_self_sha256="1" * 64,
        implementation_freeze_self_sha256="2" * 64,
        typed_python=project / "runtime/bin/python",
        minilm_model=project / "models/minilm",
        cross_encoder_model=project / "models/cross-encoder",
        nvidia_smi=Path("/usr/bin/nvidia-smi"),
        systemd_run_sha256="3" * 64,
        env_executable_sha256="8" * 64,
        controller_executable=project / "runtime/bin/python",
        controller_executable_sha256="4" * 64,
        controller_module=project / "controller.py",
        controller_module_sha256="5" * 64,
        controller_arguments=(),
        official_hippo_receipt_relative=Path(
            "manifests/synthetic.02.official_hippo.json"
        ),
        official_hippo_receipt_schema=(
            r.OFFICIAL_HIPPO_PREFLIGHT_SCHEMA
        ),
        official_hippo_receipt_status=r.OFFICIAL_HIPPO_PREFLIGHT_STATUS,
        official_hippo_receipt_self_sha256=r._semantic_hash(official_body),
        official_runtime_python=Path("/synthetic/official/bin/python"),
        official_pyvenv_cfg=Path("/synthetic/official/pyvenv.cfg"),
        official_overlay_root=Path("/synthetic/official/overlay"),
        official_hipporag_source_root=Path(
            "/synthetic/official/HippoRAG/src"
        ),
        official_p16_site_root=Path("/synthetic/official/p16_site"),
        official_local_llm_model=Path("/synthetic/models/llm"),
        official_local_embedding_model=Path(
            "/synthetic/models/embedding"
        ),
        official_expected_package_versions={
            "sentence-transformers": "3.1.1",
            "torch": "2.4.1+cu118",
            "transformers": "4.45.2",
        },
        official_expected_module_import_roots={
            "hipporag": "hipporag_source_root",
            "sentence_transformers": "p16_site_root",
            "torch": "p16_site_root",
            "transformers": "p16_site_root",
        },
        controller_receipt_relative=Path(
            "manifests/mmqa_p1_formal_action_terminal_v1.json"
        ),
        controller_receipt_schema="mmqa_p1_formal_action_terminal_v1",
        controller_receipt_status="formal_action_complete",
        controller_timeout_seconds=600,
    )
    return _with_canonical_controller_arguments(config)


@pytest.mark.parametrize(
    ("case", "flag"),
    (
        ("missing", "--implementation-freeze-self-sha256"),
        ("misplaced", "--project"),
        ("official_path", "--official-runtime-python"),
        ("official_path", "--official-pyvenv-cfg"),
        ("official_path", "--official-overlay-root"),
        ("official_path", "--official-hipporag-source-root"),
        ("official_path", "--official-p16-site-root"),
        ("official_path", "--official-local-llm-model"),
        ("official_path", "--official-local-embedding-model"),
        ("official_hash", "--official-preflight-receipt-sha256"),
        ("version", "--official-package-version"),
        ("import_root", "--official-module-import-root"),
        ("systemd", "--systemd-run-resolved-sha256"),
    ),
)
def test_pre_source_config_rejects_formal_runtime_argv_drift(
    tmp_path: Path,
    case: str,
    flag: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project)
    arguments = list(config.controller_arguments)
    position = arguments.index(flag)
    if case == "missing":
        del arguments[position : position + 2]
    elif case == "misplaced":
        pair = arguments[position : position + 2]
        del arguments[position : position + 2]
        arguments.extend(pair)
    elif case in {"official_hash", "systemd"}:
        arguments[position + 1] = "b" * 64
    else:
        arguments[position + 1] += "-drift"

    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="exact split argv contract drifted",
    ):
        r._validate_config(
            replace(config, controller_arguments=tuple(arguments))
        )


def test_outer_template_matches_formal_runtime_template_exactly(
    tmp_path: Path,
) -> None:
    from assumption_agent.benchmarks import (
        mmqa_p1_formal_action_runtime_v1 as formal_runtime,
    )
    from assumption_agent.benchmarks import (
        mmqa_p1_local_runtime_preflight_v1 as local_preflight,
    )

    project = tmp_path / "project"
    project.mkdir()
    config = _config(project)
    local_sha256 = "d" * 64
    selection_sha256 = "e" * 64
    materialized = tuple(
        {
            r.LOCAL_PREFLIGHT_SELF_SHA256_PLACEHOLDER: local_sha256,
            r.SELECTION_ACQUISITION_SHA256_PLACEHOLDER: selection_sha256,
        }.get(argument, argument)
        for argument in config.controller_arguments
    )
    runtime_config = formal_runtime.FormalActionRuntimeConfig(
        project_root=project,
        execution_freeze_self_sha256=config.execution_freeze_self_sha256,
        implementation_freeze_self_sha256=(
            config.implementation_freeze_self_sha256
        ),
        local_preflight_receipt=project / r.PREFLIGHT_RECEIPT_RELATIVE,
        local_preflight_self_sha256=local_sha256,
        typed_python=config.typed_python,
        typed_python_resolved_sha256=(
            config.controller_executable_sha256
        ),
        minilm_model=config.minilm_model,
        minilm_required_tree_sha256=(
            local_preflight.MINILM_REQUIRED_TREE_SHA256
        ),
        cross_encoder_model=config.cross_encoder_model,
        cross_encoder_required_tree_sha256=(
            local_preflight.CE_REQUIRED_TREE_SHA256
        ),
        nvidia_smi=config.nvidia_smi,
        systemd_run=r.SYSTEMD_RUN_PATH,
        systemd_run_resolved_sha256=config.systemd_run_sha256,
        systemd_isolation_disposition_sha256=r._semantic_hash(
            r._transient_unit_contract(project)
        ),
        runtime_module_sha256=config.controller_module_sha256,
        official_preflight_receipt=(
            project / config.official_hippo_receipt_relative
        ),
        official_preflight_receipt_sha256=(
            config.official_hippo_receipt_self_sha256
        ),
        official_runtime_paths=r._official_runtime_paths(config),
        official_expected_package_versions=(
            config.official_expected_package_versions
        ),
        official_expected_module_import_roots=(
            config.official_expected_module_import_roots
        ),
        selection_acquisition_sha256=selection_sha256,
        controller_arguments=materialized,
    )

    assert formal_runtime.controller_argument_template(runtime_config) == (
        r._controller_argument_template(config)
    )


def _fake_stages(
    project: Path,
    calls: list[str],
    *,
    fail_stage: str | None = None,
    preflight_touches_source: bool = False,
) -> r.LifecycleStages:
    runners: dict[str, r.StageRunner] = {}
    binding_count_before = (0, 2, 4, 5, 6, 7, 8)

    for ordinal, stage in enumerate(r.STAGE_ORDER, start=1):
        def runner(
            context: r.StageContext,
            *,
            stage: str = stage,
            ordinal: int = ordinal,
        ) -> tuple[r.ReceiptSpec, ...]:
            calls.append(stage)
            assert len(context.bindings) == binding_count_before[ordinal - 1]
            if stage == fail_stage:
                raise RuntimeError(
                    "SECRET_ACTION_GOLD_LEDGER_OPENAI_API_KEY_RUOLI"
                )
            if (
                stage == "public_synthetic_local_runtime_preflight"
                and preflight_touches_source
            ):
                (project / r.SOURCE_ROOT_RELATIVE).mkdir(
                    mode=0o700, parents=True
                )
            names = tuple(
                sorted(r.EXPECTED_STAGE_BINDING_NAMES[stage])
            )
            specs: list[r.ReceiptSpec] = []
            for name_index, name in enumerate(names, start=1):
                if name == "official_hipporag_runtime_binding_canary":
                    relative = (
                        context.config.official_hippo_receipt_relative
                    )
                    schema = context.config.official_hippo_receipt_schema
                    status = context.config.official_hippo_receipt_status
                else:
                    relative = Path("manifests") / (
                        f"synthetic.{ordinal:02d}.{name_index:02d}.json"
                    )
                    schema = (
                        f"synthetic_{ordinal:02d}_{name_index:02d}_receipt_v1"
                    )
                    status = "complete"
                _write_receipt(
                    project,
                    relative=relative,
                    schema=schema,
                    status=status,
                    self_hash_field=(
                        "receipt_sha256"
                        if name
                        == "official_hipporag_runtime_binding_canary"
                        else (
                            "acquisition_sha256"
                            if name == "private_selection_public_receipt"
                            else "self_sha256"
                        )
                    ),
                    extra={
                        "prior_binding_count": len(context.bindings)
                    },
                )
                specs.append(
                    r.ReceiptSpec(
                        name=name,
                        relative_path=relative,
                        expected_schema=schema,
                        expected_status=status,
                        self_hash_field=(
                            "receipt_sha256"
                            if name
                            == "official_hipporag_runtime_binding_canary"
                            else (
                                "acquisition_sha256"
                                if name
                                == "private_selection_public_receipt"
                                else "self_sha256"
                            )
                        ),
                        required_mode=0o600,
                    )
                )
            return tuple(specs)

        runners[stage] = runner
    return r.LifecycleStages(**runners)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("ascii"))
    assert isinstance(value, dict)
    return value


def _assert_self_hash(value: dict[str, object]) -> None:
    body = dict(value)
    claimed = body.pop("self_sha256")
    assert claimed == r._semantic_hash(body)


def test_injected_stages_execute_exactly_once_in_frozen_order(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir(mode=0o700)
    calls: list[str] = []
    result = dict(
        _run_outer(
            _config(project), _fake_stages(project, calls)
        )
    )

    assert calls == list(r.STAGE_ORDER)
    assert result["stage_count"] == len(r.STAGE_ORDER)
    assert result["status"] == (
        "outer_lifecycle_complete_formal_action_terminal_bound"
    )
    assert result["restart_retry_replay_resample_provider_or_model_switch_count"] == 0
    assert result["api_or_ruoli_environment_read_or_forward_count"] == 0
    _assert_self_hash(result)

    outer_root = project / r.OUTER_ROOT_RELATIVE
    marker = _read_json(outer_root / r.OUTER_MARKER_FILENAME)
    _assert_self_hash(marker)
    assert marker["stage_order"] == list(r.STAGE_ORDER)
    assert marker["outer_restart_retry_replay_or_resample_count"] == 0
    prior_completion: dict[str, object] | None = None
    for ordinal, stage in enumerate(r.STAGE_ORDER, start=1):
        stage_marker_path = (
            outer_root
            / r.STAGE_DIRECTORY_NAME
            / f"{ordinal:02d}.{stage}.attempt.json"
        )
        completion_path = (
            outer_root
            / r.STAGE_DIRECTORY_NAME
            / f"{ordinal:02d}.{stage}.complete.json"
        )
        stage_marker = _read_json(stage_marker_path)
        completion = _read_json(completion_path)
        _assert_self_hash(stage_marker)
        _assert_self_hash(completion)
        assert stat_mode(stage_marker_path) == 0o600
        assert stat_mode(completion_path) == 0o600
        assert stage_marker["stage_ordinal"] == ordinal
        assert completion["stage_marker_self_sha256"] == stage_marker[
            "self_sha256"
        ]
        if prior_completion is None:
            assert stage_marker["prior_stage_completion_self_sha256"] is None
        else:
            assert stage_marker["prior_stage_completion_self_sha256"] == (
                prior_completion["self_sha256"]
            )
        bindings = completion["aggregate_receipt_bindings"]
        expected_names = r.EXPECTED_STAGE_BINDING_NAMES[stage]
        assert isinstance(bindings, list)
        assert len(bindings) == len(expected_names)
        assert {binding["name"] for binding in bindings} == set(
            expected_names
        )
        prior_completion = completion
    assert not (outer_root / r.OUTER_FAILURE_FILENAME).exists()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_consumed_outer_root_blocks_all_restart_and_stage_calls(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []
    config = _config(project)
    _run_outer(config, _fake_stages(project, calls))
    before = list(calls)

    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError, match="already consumed"
    ):
        _run_outer(config, _fake_stages(project, calls))
    assert calls == before


@pytest.mark.parametrize("failure_index", range(len(r.STAGE_ORDER)))
def test_any_stage_failure_is_sanitized_terminal_and_never_retried(
    tmp_path: Path,
    failure_index: int,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []
    failing = r.STAGE_ORDER[failure_index]
    config = _config(project)
    stages = _fake_stages(project, calls, fail_stage=failing)

    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="failed closed",
    ):
        _run_outer(config, stages)
    assert calls == list(r.STAGE_ORDER[: failure_index + 1])

    terminal_path = (
        project / r.OUTER_ROOT_RELATIVE / r.OUTER_FAILURE_FILENAME
    )
    terminal_raw = terminal_path.read_bytes()
    terminal = _read_json(terminal_path)
    _assert_self_hash(terminal)
    assert terminal["failure_stage"] == failing
    assert terminal["completed_stage_count"] == failure_index
    assert terminal["status"] == (
        "terminal_failure_outer_attempt_consumed_no_restart"
    )
    assert (
        terminal[
            "restart_retry_replay_resample_provider_or_model_switch_count"
        ]
        == 0
    )
    for forbidden in (
        b"SECRET_ACTION",
        b"GOLD_LEDGER",
        b"OPENAI_API_KEY",
        b"RUOLI",
    ):
        assert forbidden not in terminal_raw

    before = list(calls)
    with pytest.raises(r.MMQAP1RemoteOuterLifecycleError):
        _run_outer(config, stages)
    assert calls == before


def test_preexisting_source_state_stops_before_preflight_call(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / r.SOURCE_ROOT_RELATIVE).mkdir(mode=0o700, parents=True)
    calls: list[str] = []

    with pytest.raises(r.MMQAP1RemoteOuterLifecycleError):
        _run_outer(
            _config(project), _fake_stages(project, calls)
        )
    assert calls == [r.STAGE_ORDER[0]]
    terminal = _read_json(
        project / r.OUTER_ROOT_RELATIVE / r.OUTER_FAILURE_FILENAME
    )
    assert terminal["failure_stage"] == r.STAGE_ORDER[1]
    assert terminal["completed_stage_count"] == 1


def test_preflight_that_touches_source_stops_before_acquisition(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []

    with pytest.raises(r.MMQAP1RemoteOuterLifecycleError):
        _run_outer(
            _config(project),
            _fake_stages(
                project, calls, preflight_touches_source=True
            ),
        )
    assert calls == list(r.STAGE_ORDER[:2])
    outer_root = project / r.OUTER_ROOT_RELATIVE
    assert not (
        outer_root
        / r.STAGE_DIRECTORY_NAME
        / f"02.{r.STAGE_ORDER[1]}.complete.json"
    ).exists()
    assert not (
        outer_root
        / r.STAGE_DIRECTORY_NAME
        / f"03.{r.STAGE_ORDER[2]}.attempt.json"
    ).exists()


def test_local_only_preflight_receipt_cannot_unlock_source_acquisition(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []
    normal = _fake_stages(project, calls)

    def local_only(
        context: r.StageContext,
    ) -> tuple[r.ReceiptSpec, ...]:
        calls.append(r.STAGE_ORDER[1])
        relative = Path("manifests/local-only-preflight.json")
        _write_receipt(
            project,
            relative=relative,
            schema="synthetic_local_only_preflight_v1",
        )
        return (
            r.ReceiptSpec(
                name="public_synthetic_local_runtime_preflight",
                relative_path=relative,
                expected_schema="synthetic_local_only_preflight_v1",
                expected_status="complete",
                required_mode=0o600,
            ),
        )

    stages = replace(
        normal,
        public_synthetic_local_runtime_preflight=local_only,
    )
    with pytest.raises(r.MMQAP1RemoteOuterLifecycleError):
        _run_outer(_config(project), stages)
    assert calls == list(r.STAGE_ORDER[:2])
    assert not (
        project
        / r.OUTER_ROOT_RELATIVE
        / r.STAGE_DIRECTORY_NAME
        / f"03.{r.STAGE_ORDER[2]}.attempt.json"
    ).exists()


def test_production_preflight_accepts_injected_official_canary_second(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project)
    order: list[str] = []

    def local(
        _context: r.StageContext,
    ) -> tuple[r.ReceiptSpec, ...]:
        order.append("local")
        relative = Path("manifests/local-preflight.json")
        _write_receipt(
            project,
            relative=relative,
            schema="synthetic_local_preflight_v1",
        )
        return (
            r.ReceiptSpec(
                "public_synthetic_local_runtime_preflight",
                relative,
                "synthetic_local_preflight_v1",
                "complete",
            ),
        )

    def official(
        _context: r.StageContext,
    ) -> tuple[r.ReceiptSpec, ...]:
        order.append("official")
        _write_receipt(
            project,
            relative=config.official_hippo_receipt_relative,
            schema=config.official_hippo_receipt_schema,
            status=config.official_hippo_receipt_status,
            self_hash_field="receipt_sha256",
            extra={"prior_binding_count": 2},
        )
        return (
            r.ReceiptSpec(
                "official_hipporag_runtime_binding_canary",
                config.official_hippo_receipt_relative,
                config.official_hippo_receipt_schema,
                config.official_hippo_receipt_status,
                self_hash_field="receipt_sha256",
                expected_self_sha256=(
                    config.official_hippo_receipt_self_sha256
                ),
                required_mode=0o600,
            ),
        )

    monkeypatch.setattr(r, "_preflight_stage", local)
    monkeypatch.setattr(
        r,
        "_live_validate_official_hippo_preflight",
        lambda _context, _spec: "f" * 64,
    )
    stages = r.production_stages(official_hippo_preflight=official)
    specs = stages.public_synthetic_local_runtime_preflight(
        r.StageContext(
            config=config,
            bindings={
                "execution_freeze": SimpleNamespace(),
                "implementation_freeze": SimpleNamespace(),
            },
        )
    )
    assert order == ["local", "official"]
    assert {spec.name for spec in specs} == (
        r.REQUIRED_PREFLIGHT_BINDING_NAMES
    )


def test_official_live_preflight_rehashes_frozen_external_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from assumption_agent.benchmarks import (
        mmqa_p1_official_hipporag_block_v1 as official,
    )

    project = tmp_path / "project"
    project.mkdir()
    config = _config(project)
    receipt_sha256 = "a" * 64
    value = {
        "expected_module_import_roots": dict(
            config.official_expected_module_import_roots
        ),
        "expected_package_versions": dict(
            config.official_expected_package_versions
        ),
    }
    receipt_binding = r.ReceiptBinding(
        "official_hipporag_runtime_binding_canary",
        config.official_hippo_receipt_relative.as_posix(),
        config.official_hippo_receipt_schema,
        config.official_hippo_receipt_status,
        "receipt_sha256",
        receipt_sha256,
        "b" * 64,
        1,
        "0600",
    )
    monkeypatch.setattr(
        r,
        "_load_receipt_value_and_binding",
        lambda _project, _spec: (value, receipt_binding),
    )
    calls: list[dict[str, object]] = []

    def live_loader(path, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"path": path, **kwargs})
        return SimpleNamespace(
            receipt_sha256=receipt_sha256,
            binding_sha256="c" * 64,
        )

    monkeypatch.setattr(
        official, "load_fresh_preflight_binding", live_loader
    )
    spec = r.ReceiptSpec(
        "official_hipporag_runtime_binding_canary",
        config.official_hippo_receipt_relative,
        config.official_hippo_receipt_schema,
        config.official_hippo_receipt_status,
        self_hash_field="receipt_sha256",
        required_mode=0o600,
    )
    observed = r._live_validate_official_hippo_preflight(
        r.StageContext(config=config, bindings={}), spec
    )
    assert observed == "c" * 64
    assert len(calls) == 1
    assert calls[0]["paths"].path_binding() == (
        r._official_runtime_paths(config).path_binding()
    )
    assert calls[0]["filesystem_inspector"] is (
        official.production_filesystem_inspector
    )
    assert calls[0]["isolation_inspector"] is (
        official.production_address_family_isolation_probe
    )


def test_official_filesystem_drift_stops_before_acquisition_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project)
    calls: list[str] = []
    normal = _fake_stages(project, calls)

    def local(_context: r.StageContext) -> tuple[r.ReceiptSpec, ...]:
        relative = Path("manifests/local-preflight.json")
        _write_receipt(
            project,
            relative=relative,
            schema="synthetic_local_preflight_v1",
        )
        return (
            r.ReceiptSpec(
                "public_synthetic_local_runtime_preflight",
                relative,
                "synthetic_local_preflight_v1",
                "complete",
            ),
        )

    def official(_context: r.StageContext) -> tuple[r.ReceiptSpec, ...]:
        relative = config.official_hippo_receipt_relative
        _write_receipt(
            project,
            relative=relative,
            schema=config.official_hippo_receipt_schema,
            status=config.official_hippo_receipt_status,
            self_hash_field="receipt_sha256",
        )
        return (
            r.ReceiptSpec(
                "official_hipporag_runtime_binding_canary",
                relative,
                config.official_hippo_receipt_schema,
                config.official_hippo_receipt_status,
                self_hash_field="receipt_sha256",
                required_mode=0o600,
            ),
        )

    monkeypatch.setattr(r, "_preflight_stage", local)
    monkeypatch.setattr(
        r,
        "_live_validate_official_hippo_preflight",
        lambda _context, _spec: (_ for _ in ()).throw(
            r.MMQAP1RemoteOuterLifecycleError(
                "official external filesystem drifted"
            )
        ),
    )
    production = r.production_stages(
        official_hippo_preflight=official
    )
    stages = replace(
        normal,
        public_synthetic_local_runtime_preflight=(
            production.public_synthetic_local_runtime_preflight
        ),
    )
    with pytest.raises(r.MMQAP1RemoteOuterLifecycleError):
        _run_outer(config, stages)
    assert calls == [r.STAGE_ORDER[0]]
    assert not (project / r.DOWNLOAD_RECEIPT_RELATIVE).exists()


def test_invalid_stage_receipt_stops_before_next_stage(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []
    normal = _fake_stages(project, calls)

    def invalid(_context: r.StageContext) -> tuple[r.ReceiptSpec, ...]:
        calls.append(r.STAGE_ORDER[0])
        relative = Path("manifests/invalid.json")
        path = project / relative
        _write_receipt(
            project, relative=relative, schema="synthetic_invalid_v1"
        )
        value = _read_json(path)
        value["self_sha256"] = "0" * 64
        path.write_bytes(r._canonical_bytes(value))
        os.chmod(path, 0o600)
        return (
            r.ReceiptSpec(
                name="invalid_receipt",
                relative_path=relative,
                expected_schema="synthetic_invalid_v1",
                expected_status="complete",
                required_mode=0o600,
            ),
        )

    stages = r.LifecycleStages(
        verify_execution_and_implementation_freezes=invalid,
        public_synthetic_local_runtime_preflight=(
            normal.public_synthetic_local_runtime_preflight
        ),
        authorized_source_acquisition=normal.authorized_source_acquisition,
        source_qualification_freeze=normal.source_qualification_freeze,
        aggregate_source_qualification=(
            normal.aggregate_source_qualification
        ),
        private_one_shot_selection=normal.private_one_shot_selection,
        post_selection_network_denied_formal_action=(
            normal.post_selection_network_denied_formal_action
        ),
    )
    with pytest.raises(r.MMQAP1RemoteOuterLifecycleError):
        _run_outer(_config(project), stages)
    assert calls == [r.STAGE_ORDER[0]]


def test_outer_refuses_private_action_gold_or_ledger_receipt_path(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []
    secret = b"SECRET_PRIVATE_ACTION_CONTENT"
    relative = Path(
        "artifacts/mmqa_p1_private_selection_v1/"
        "A_hold.gold.sealed.private.json"
    )

    def forbidden(_context: r.StageContext) -> tuple[r.ReceiptSpec, ...]:
        calls.append(r.STAGE_ORDER[0])
        path = project / relative
        path.parent.mkdir(mode=0o700, parents=True)
        path.write_bytes(secret)
        os.chmod(path, 0o600)
        return (
            r.ReceiptSpec(
                name="forbidden_private_pack",
                relative_path=relative,
                expected_schema="synthetic_private_v1",
            ),
        )

    fallback = _fake_stages(project, calls)
    stages = r.LifecycleStages(
        verify_execution_and_implementation_freezes=forbidden,
        public_synthetic_local_runtime_preflight=(
            fallback.public_synthetic_local_runtime_preflight
        ),
        authorized_source_acquisition=(
            fallback.authorized_source_acquisition
        ),
        source_qualification_freeze=fallback.source_qualification_freeze,
        aggregate_source_qualification=(
            fallback.aggregate_source_qualification
        ),
        private_one_shot_selection=fallback.private_one_shot_selection,
        post_selection_network_denied_formal_action=(
            fallback.post_selection_network_denied_formal_action
        ),
    )
    with pytest.raises(r.MMQAP1RemoteOuterLifecycleError):
        _run_outer(_config(project), stages)
    terminal_raw = (
        project / r.OUTER_ROOT_RELATIVE / r.OUTER_FAILURE_FILENAME
    ).read_bytes()
    assert secret not in terminal_raw


def _frozen_production_fixture(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[r.OuterLifecycleConfig, Path]:
    typed_python = project / "runtime/bin/python"
    nvidia_smi = project / "runtime/bin/nvidia-smi"
    controller_module = project / "formal_controller.py"
    official_adapter = project / "official_hippo_adapter.py"
    minilm = project / "models/minilm"
    cross_encoder = project / "models/cross-encoder"
    for path, raw in (
        (typed_python, b"SYNTHETIC_TYPED_PYTHON"),
        (nvidia_smi, b"SYNTHETIC_NVIDIA_SMI"),
        (controller_module, b"SYNTHETIC_CONTROLLER"),
        (official_adapter, b"SYNTHETIC_OFFICIAL_HIPPO"),
    ):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(raw)
        os.chmod(path, 0o600)
    minilm.mkdir(mode=0o700, parents=True)
    cross_encoder.mkdir(mode=0o700, parents=True)
    disposition_source = (
        Path(r.__file__).resolve().parents[2]
        / r.RUNTIME_DISPOSITION_RELATIVE
    )
    disposition_value = json.loads(
        disposition_source.read_text("utf-8")
    )
    _write_value(
        project / r.RUNTIME_DISPOSITION_RELATIVE,
        disposition_value,
    )
    os.chmod(project / r.RUNTIME_DISPOSITION_RELATIVE, 0o644)

    config = _with_canonical_controller_arguments(
        replace(
            _config(project),
            typed_python=typed_python,
            minilm_model=minilm,
            cross_encoder_model=cross_encoder,
            nvidia_smi=nvidia_smi,
            controller_executable=typed_python,
            controller_executable_sha256=hashlib.sha256(
                typed_python.read_bytes()
            ).hexdigest(),
            controller_module=controller_module,
            controller_module_sha256=hashlib.sha256(
                controller_module.read_bytes()
            ).hexdigest(),
        )
    )
    monkeypatch.setattr(r, "REQUIRED_IMPLEMENTATION_RELATIVES", frozenset())
    real_hasher = r._sha256_regular_file

    def hasher(path: Path, *, resolve_symlink: bool = False) -> str:
        if path == r.SYSTEMD_RUN_PATH:
            return config.systemd_run_sha256
        if path == r.ENV_PATH:
            return config.env_executable_sha256
        return real_hasher(path, resolve_symlink=resolve_symlink)

    monkeypatch.setattr(r, "_sha256_regular_file", hasher)

    implementation_body = {
        "implementation_files": {
            controller_module.relative_to(project).as_posix(): (
                config.controller_module_sha256
            ),
            official_adapter.relative_to(project).as_posix(): hashlib.sha256(
                official_adapter.read_bytes()
            ).hexdigest(),
            r.RUNTIME_DISPOSITION_RELATIVE.as_posix(): hashlib.sha256(
                (project / r.RUNTIME_DISPOSITION_RELATIVE).read_bytes()
            ).hexdigest(),
        },
        "schema": "mmqa_p1_implementation_freeze_v1",
        "status": "frozen_before_execution_freeze",
        "study_id": r.STUDY_ID,
    }
    implementation = r._self_hashed(implementation_body)
    _write_value(
        project / r.IMPLEMENTATION_FREEZE_RELATIVE,
        implementation,
    )
    config = _with_canonical_controller_arguments(
        replace(
            config,
            implementation_freeze_self_sha256=implementation["self_sha256"],
        )
    )
    runtime_rows = {
        "controller_executable": {
            "lexical_path_sha256": r._lexical_path_sha256(typed_python),
            "resolved_file_sha256": (
                config.controller_executable_sha256
            ),
        },
        "controller_module": {
            "file_sha256": config.controller_module_sha256,
            "project_relative_path": controller_module.relative_to(
                project
            ).as_posix(),
        },
        "cross_encoder_model": {
            "lexical_path_sha256": r._lexical_path_sha256(cross_encoder)
        },
        "env_executable": {
            "lexical_path_sha256": r._lexical_path_sha256(r.ENV_PATH),
            "resolved_file_sha256": config.env_executable_sha256,
        },
        "minilm_model": {
            "lexical_path_sha256": r._lexical_path_sha256(minilm)
        },
        "nvidia_smi": {
            "lexical_path_sha256": r._lexical_path_sha256(nvidia_smi),
            "resolved_file_sha256": hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
        },
        "systemd_run": {
            "lexical_path_sha256": r._lexical_path_sha256(
                r.SYSTEMD_RUN_PATH
            ),
            "resolved_file_sha256": config.systemd_run_sha256,
        },
        "typed_python": {
            "lexical_path_sha256": r._lexical_path_sha256(typed_python),
            "resolved_file_sha256": hashlib.sha256(
                typed_python.read_bytes()
            ).hexdigest(),
        },
    }
    execution_body = {
        "controller_argument_template_sha256": r._semantic_hash(
            list(r._controller_argument_template(config))
        ),
        "download_authorization_self_sha256": (
            r.EXPECTED_AUTHORIZATION_SELF_SHA256
        ),
        "execution_policy": r.EXECUTION_POLICY,
        "formal_child_environment_sha256": r._semantic_hash(
            r._formal_child_environment(project)
        ),
        "formal_action_transient_unit_contract": (
            r._transient_unit_contract(project)
        ),
        "formal_controller_receipt_contract": {
            "relative_path": config.controller_receipt_relative.as_posix(),
            "schema": config.controller_receipt_schema,
            "status": config.controller_receipt_status,
        },
        "implementation_freeze_self_sha256": (
            config.implementation_freeze_self_sha256
        ),
        "official_hipporag_adapter_relative_path": (
            official_adapter.relative_to(project).as_posix()
        ),
        "official_hipporag_preflight_receipt_contract": {
            "relative_path": (
                config.official_hippo_receipt_relative.as_posix()
            ),
            "schema": config.official_hippo_receipt_schema,
            "self_hash_field": "receipt_sha256",
            "self_sha256": config.official_hippo_receipt_self_sha256,
            "status": config.official_hippo_receipt_status,
        },
        "official_hipporag_runtime_contract": (
            r._official_runtime_contract(config)
        ),
        "preexecution_runtime_disposition_self_sha256": (
            r.EXPECTED_RUNTIME_DISPOSITION_SELF_SHA256
        ),
        "outer_network_isolation_contract": (
            r.OUTER_NETWORK_ISOLATION_CONTRACT
        ),
        "runtime_path_bindings": runtime_rows,
        "schema": "mmqa_p1_execution_freeze_v1",
        "source_custody_self_sha256": r.EXPECTED_CUSTODY_SELF_SHA256,
        "stage_order": list(r.STAGE_ORDER),
        "status": "frozen_before_outer_one_shot",
        "study_design_self_sha256": r.EXPECTED_DESIGN_SELF_SHA256,
        "study_id": r.STUDY_ID,
        "source_acquisition_child_environment_sha256": r._semantic_hash(
            r._source_acquisition_child_environment(project)
        ),
        "source_acquisition_transient_unit_contract": (
            r._source_acquisition_transient_unit_contract(project)
        ),
        "systemd_client_environment_sha256": r._semantic_hash(
            r._systemd_client_environment()
        ),
    }
    execution = r._self_hashed(execution_body)
    _write_value(project / r.EXECUTION_FREEZE_RELATIVE, execution)
    prior_execution_sha256 = config.execution_freeze_self_sha256
    config = replace(
        config,
        execution_freeze_self_sha256=execution["self_sha256"],
        controller_arguments=tuple(
            execution["self_sha256"]
            if argument == prior_execution_sha256
            else argument
            for argument in config.controller_arguments
        ),
    )
    return config, official_adapter


def _write_value(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(r._canonical_bytes(value))
    os.chmod(path, 0o600)


def test_production_freeze_live_verifies_inventory_and_execution_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config, _official_adapter = _frozen_production_fixture(
        project, monkeypatch
    )
    specs = r._freeze_stage(
        r.StageContext(config=config, bindings={})
    )
    bindings = [
        r._load_receipt_binding(project, spec) for spec in specs
    ]
    assert [binding.name for binding in bindings] == [
        "execution_freeze",
        "implementation_freeze",
    ]
    assert [binding.self_sha256 for binding in bindings] == [
        config.execution_freeze_self_sha256,
        config.implementation_freeze_self_sha256,
    ]


def test_implementation_dependency_drift_fails_freeze_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config, official_adapter = _frozen_production_fixture(
        project, monkeypatch
    )
    official_adapter.write_bytes(
        official_adapter.read_bytes() + b"DRIFT"
    )
    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="dependency file drifted",
    ):
        r._freeze_stage(r.StageContext(config=config, bindings={}))


def test_implementation_inventory_rejects_symlinked_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "real.py"
    target.write_bytes(b"BOUND_IMPLEMENTATION")
    alias = project / "alias.py"
    alias.symlink_to(target)
    manifest = {
        "implementation_files": {
            "alias.py": hashlib.sha256(target.read_bytes()).hexdigest()
        }
    }
    monkeypatch.setattr(r, "REQUIRED_IMPLEMENTATION_RELATIVES", frozenset())
    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="symlinked",
    ):
        r._verify_frozen_inventory(project, manifest)


def test_execution_freeze_cannot_relax_retry_or_network_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config, _official_adapter = _frozen_production_fixture(
        project, monkeypatch
    )
    path = project / r.EXECUTION_FREEZE_RELATIVE
    value = _read_json(path)
    value.pop("self_sha256")
    policy = dict(value["execution_policy"])
    policy["post_selection_network_allowed"] = True
    value["execution_policy"] = policy
    drifted = r._self_hashed(value)
    _write_value(path, drifted)
    prior_execution_sha256 = config.execution_freeze_self_sha256
    config = replace(
        config,
        execution_freeze_self_sha256=drifted["self_sha256"],
        controller_arguments=tuple(
            drifted["self_sha256"]
            if argument == prior_execution_sha256
            else argument
            for argument in config.controller_arguments
        ),
    )
    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="policy or upstream binding drifted",
    ):
        r._freeze_stage(r.StageContext(config=config, bindings={}))


def test_formal_child_environment_never_copies_api_or_ruoli_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET_OPENAI")
    monkeypatch.setenv("RUOLI_API_KEY", "SECRET_RUOLI")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-proxy")
    environment = r._formal_child_environment(tmp_path)
    assert set(environment) == r._FIXED_CHILD_ENV_NAMES
    serialized = json.dumps(environment, sort_keys=True)
    for forbidden in (
        "OPENAI",
        "RUOLI",
        "SECRET_OPENAI",
        "SECRET_RUOLI",
        "secret-proxy",
        "HTTPS_PROXY",
    ):
        assert forbidden not in serialized


def test_systemd_client_environment_is_derived_without_parent_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "secret-parent-bus")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/secret-parent-runtime")
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET_OPENAI")
    monkeypatch.setenv("RUOLI_API_KEY", "SECRET_RUOLI")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-proxy")
    environment = r._systemd_client_environment()
    runtime = f"/run/user/{os.getuid()}"
    assert environment == {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": runtime,
    }
    serialized = json.dumps(environment, sort_keys=True)
    for forbidden in (
        "secret-parent",
        "OPENAI",
        "RUOLI",
        "SECRET_OPENAI",
        "SECRET_RUOLI",
        "secret-proxy",
        "HTTPS_PROXY",
    ):
        assert forbidden not in serialized


def test_transient_contract_requires_both_inet_denial_probes_without_private_network(
    tmp_path: Path,
) -> None:
    contract = r._transient_unit_contract(tmp_path)
    assert contract["AF_INET_and_AF_INET6_denial_probe_required"] is True
    assert contract["properties"] == list(
        r._resolved_unit_properties(tmp_path)
    )
    assert "RestrictAddressFamilies=AF_UNIX" in contract["properties"]
    assert not any(
        "PrivateNetwork" in value for value in contract["properties"]
    )
    assert (
        r.EXECUTION_POLICY[
            "post_selection_AF_INET_and_AF_INET6_denial_probe_required"
        ]
        is True
    )


def test_outer_network_probe_requires_errno97_for_both_inet_families() -> None:
    observed: list[int] = []

    def denied(family: int, _kind: int) -> object:
        observed.append(family)
        raise OSError(97, "Address family not supported")

    receipt = r._verify_outer_inet_denial_once(socket_factory=denied)
    assert observed == [r.socket.AF_INET, r.socket.AF_INET6]
    assert receipt["probe_count"] == 2
    assert receipt["AF_INET_socket_creation_errno"] == "EAFNOSUPPORT"
    assert receipt["AF_INET6_socket_creation_errno"] == "EAFNOSUPPORT"
    assert (
        receipt["outer_network_isolation_contract"]
        == r.OUTER_NETWORK_ISOLATION_CONTRACT
    )


def test_outer_network_probe_fails_if_one_inet_socket_is_available() -> None:
    class OpenSocket:
        def close(self) -> None:
            pass

    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="isolation is absent",
    ):
        r._verify_outer_inet_denial_once(
            socket_factory=lambda _family, _kind: OpenSocket()
        )


def test_available_inet_family_stops_before_every_stage_and_download(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []

    def available() -> dict[str, object]:
        raise r.MMQAP1RemoteOuterLifecycleError(
            "outer address-family isolation is absent"
        )

    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="isolation is absent",
    ):
        r.run_outer_lifecycle(
            _config(project),
            _fake_stages(project, calls),
            outer_network_probe=available,
        )
    assert calls == []
    assert not (project / r.DOWNLOAD_RECEIPT_RELATIVE).exists()
    assert not (project / r.OUTER_ROOT_RELATIVE).exists()


def test_acquisition_uses_one_hardened_network_sibling_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    executable = project / "runtime/bin/python"
    module = project / r.SOURCE_ACQUISITION_MODULE_RELATIVE
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    executable.write_bytes(b"EXEC")
    module.write_bytes(b"MODULE")
    config = _config(project)
    monkeypatch.setattr(
        r,
        "_verify_source_acquisition_capability",
        lambda _config, _project: (executable, module),
    )
    calls: list[dict[str, object]] = []

    def process_runner(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"command": list(command), **kwargs})
        _write_receipt(
            project,
            relative=r.DOWNLOAD_RECEIPT_RELATIVE,
            schema="mmqa_p1_source_download_receipt_v1",
            status=(
                "four_fixed_sources_downloaded_identity_verified_not_parsed"
            ),
        )
        return SimpleNamespace(returncode=0)

    specs = r._acquisition_stage(
        r.StageContext(config=config, bindings={}),
        process_runner=process_runner,
    )
    assert len(calls) == 1
    command = calls[0]["command"]
    assert isinstance(command, list)
    assert command[:5] == [
        "/usr/bin/systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--quiet",
    ]
    assert f"--unit={r.SOURCE_ACQUISITION_UNIT_NAME}" in command
    properties = {
        value.removeprefix("--property=")
        for value in command
        if value.startswith("--property=")
    }
    assert properties == set(
        r._resolved_source_acquisition_unit_properties(project)
    )
    assert (
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6"
        in properties
    )
    assert "/usr/bin/env" in command
    assert "-i" in command
    assert command[-4:] == [
        str(executable),
        str(module),
        "--project",
        str(project),
    ]
    assert calls[0]["env"] == r._systemd_client_environment()
    assert calls[0]["timeout"] == r.SOURCE_ACQUISITION_TIMEOUT_SECONDS
    assert specs[0].relative_path == r.DOWNLOAD_RECEIPT_RELATIVE


def test_post_selection_launches_one_fixed_systemd_transient_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    executable = project / "runtime/bin/python"
    module = project / "formal_controller.py"
    executable.parent.mkdir(mode=0o700, parents=True)
    executable.write_bytes(b"SYNTHETIC_EXECUTABLE")
    module.write_bytes(b"SYNTHETIC_CONTROLLER_MODULE")
    os.chmod(executable, 0o700)
    os.chmod(module, 0o600)
    config = _config(project)
    config = _with_canonical_controller_arguments(
        replace(
            config,
            controller_executable=executable,
            controller_executable_sha256=hashlib.sha256(
                executable.read_bytes()
            ).hexdigest(),
            controller_module=module,
            controller_module_sha256=hashlib.sha256(
                module.read_bytes()
            ).hexdigest(),
        )
    )
    real_hasher = r._sha256_regular_file

    def hasher(path: Path, *, resolve_symlink: bool = False) -> str:
        if path == r.SYSTEMD_RUN_PATH:
            return config.systemd_run_sha256
        if path == r.ENV_PATH:
            return config.env_executable_sha256
        return real_hasher(path, resolve_symlink=resolve_symlink)

    monkeypatch.setattr(r, "_sha256_regular_file", hasher)
    calls: list[dict[str, object]] = []

    def process_runner(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"command": list(command), **kwargs})
        _write_receipt(
            project,
            relative=config.controller_receipt_relative,
            schema=config.controller_receipt_schema,
            status=config.controller_receipt_status,
        )
        return SimpleNamespace(returncode=0)

    selection = r.ReceiptBinding(
        name="private_selection_public_receipt",
        relative_path=r.SELECTION_RECEIPT_RELATIVE.as_posix(),
        schema="mmqa_p1_private_selection_v1_public_receipt_v1",
        status="private_one_shot_selection_complete",
        self_hash_field="acquisition_sha256",
        self_sha256="6" * 64,
        file_sha256="7" * 64,
        size_bytes=123,
        mode_octal="0644",
    )
    local = r.ReceiptBinding(
        name="public_synthetic_local_runtime_preflight",
        relative_path=r.PREFLIGHT_RECEIPT_RELATIVE.as_posix(),
        schema="mmqa_p1_local_runtime_preflight_v1_receipt_v1",
        status="passed_public_synthetic_non_scoring_runtime_action_preflight",
        self_hash_field="self_sha256",
        self_sha256="8" * 64,
        file_sha256="9" * 64,
        size_bytes=321,
        mode_octal="0600",
    )
    bindings = {
        "public_synthetic_local_runtime_preflight": local,
        "private_selection_public_receipt": selection,
    }
    specs = r._post_selection_stage(
        r.StageContext(
            config=config,
            bindings=bindings,
        ),
        process_runner=process_runner,
    )
    assert len(calls) == 1
    call = calls[0]
    command = call["command"]
    assert isinstance(command, list)
    assert command[:5] == [
        "/usr/bin/systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--quiet",
    ]
    assert command.count(f"--unit={r.FORMAL_ACTION_UNIT_NAME}") == 1
    assert command.count(f"--working-directory={project}") == 1
    properties = {
        value.removeprefix("--property=")
        for value in command
        if value.startswith("--property=")
    }
    assert properties == set(r._resolved_unit_properties(project))
    assert "RestrictAddressFamilies=AF_UNIX" in properties
    assert "NoNewPrivileges=yes" in properties
    assert "PrivateTmp=yes" in properties
    assert "ProtectSystem=strict" in properties
    assert "ProtectHome=read-only" in properties
    assert f"ReadWritePaths={project}" in properties
    assert "UMask=0077" in properties
    assert not any("PrivateNetwork" in value for value in command)
    separator = command.index("--")
    assert command[separator + 1 : separator + 3] == [
        "/usr/bin/env",
        "-i",
    ]
    child_environment = r._formal_child_environment(project)
    assert command[
        separator + 3 : separator + 3 + len(child_environment)
    ] == [
        f"{name}={child_environment[name]}"
        for name in sorted(child_environment)
    ]
    materialized_arguments, substitutions = (
        r._materialize_controller_arguments(config, bindings)
    )
    assert len(substitutions) == 2
    assert command[-(len(materialized_arguments) + 2)] == str(
        executable
    )
    assert command[-(len(materialized_arguments) + 1)] == str(module)
    assert tuple(command[-len(materialized_arguments) :]) == (
        materialized_arguments
    )
    assert r.LOCAL_PREFLIGHT_SELF_SHA256_PLACEHOLDER not in command
    assert r.SELECTION_ACQUISITION_SHA256_PLACEHOLDER not in command
    assert local.self_sha256 in command
    assert selection.self_sha256 in command
    assert call["stdout"] == r.subprocess.DEVNULL
    assert call["stderr"] == r.subprocess.DEVNULL
    assert call["check"] is False
    environment = call["env"]
    assert isinstance(environment, dict)
    assert environment == r._systemd_client_environment()
    assert set(environment) == r._FIXED_SYSTEMD_CLIENT_ENV_NAMES
    assert not any(
        forbidden in name
        for name in environment
        for forbidden in ("OPENAI", "RUOLI", "API_KEY", "PROXY")
    )
    binding = r._load_receipt_binding(project, specs[0])
    assert binding.name == "post_selection_formal_action_terminal"


def test_post_selection_nonzero_exit_is_terminal_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    executable = project / "python"
    module = project / "controller.py"
    executable.write_bytes(b"EXEC")
    module.write_bytes(b"MODULE")
    config = _config(project)
    config = _with_canonical_controller_arguments(
        replace(
            config,
            controller_executable=executable,
            controller_executable_sha256=hashlib.sha256(b"EXEC").hexdigest(),
            controller_module=module,
            controller_module_sha256=hashlib.sha256(b"MODULE").hexdigest(),
        )
    )
    real_hasher = r._sha256_regular_file

    def hasher(path: Path, *, resolve_symlink: bool = False) -> str:
        if path == r.SYSTEMD_RUN_PATH:
            return config.systemd_run_sha256
        if path == r.ENV_PATH:
            return config.env_executable_sha256
        return real_hasher(path, resolve_symlink=resolve_symlink)

    monkeypatch.setattr(r, "_sha256_regular_file", hasher)
    call_count = 0

    def fail_once(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(returncode=17)

    selection = r.ReceiptBinding(
        "private_selection_public_receipt",
        r.SELECTION_RECEIPT_RELATIVE.as_posix(),
        "selection",
        "complete",
        "acquisition_sha256",
        "1" * 64,
        "2" * 64,
        1,
        "0644",
    )
    local = r.ReceiptBinding(
        "public_synthetic_local_runtime_preflight",
        r.PREFLIGHT_RECEIPT_RELATIVE.as_posix(),
        "local-preflight",
        "complete",
        "self_sha256",
        "3" * 64,
        "4" * 64,
        1,
        "0600",
    )
    with pytest.raises(
        r.MMQAP1RemoteOuterLifecycleError,
        match="exited unsuccessfully",
    ):
        r._post_selection_stage(
            r.StageContext(
                config=config,
                bindings={
                    "public_synthetic_local_runtime_preflight": local,
                    "private_selection_public_receipt": selection
                },
            ),
            process_runner=fail_once,
        )
    assert call_count == 1
