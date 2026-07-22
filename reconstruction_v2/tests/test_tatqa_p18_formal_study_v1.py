from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import tatqa_p18_formal_controller_v1 as controller
from assumption_agent.benchmarks import tatqa_p18_formal_study_v1 as study


SHA = "7" * 64


@pytest.fixture
def linux_tmp_path() -> Path:
    root = Path(tempfile.mkdtemp(prefix="tatqa-p18-study-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _config(tmp_path: Path) -> study.FormalStudyConfig:
    project = tmp_path / "project"
    project.mkdir()
    return study.FormalStudyConfig(
        project_root=project,
        control_root=tmp_path / "control" / "execution",
        work_root=tmp_path / "runtime" / "work",
        runtime_python=tmp_path / "runtime" / "python",
        qwen_model=tmp_path / "assets" / "qwen",
        minilm_asset_manifest=tmp_path / "assets" / "minilm.json",
        minilm_model=tmp_path / "assets" / "minilm",
        hippo_llm_model=tmp_path / "assets" / "hippo-llm",
        hippo_embedding_model=tmp_path / "assets" / "hippo-embedding",
        hipporag_source=tmp_path / "runtime" / "HippoRAG",
        hippo_attestation=tmp_path / "assets" / "hippo-attestation.json",
        runtime_fingerprint=tmp_path / "manifests" / "fingerprint.json",
        production_canary=tmp_path / "manifests" / "canary.json",
    )


def _invalid_disposition() -> controller.FormalDisposition:
    return controller.FormalDisposition(
        status="implementation_or_runtime_invalid",
        primary_evaluated=False,
        primary_value=None,
        efficacy="unknown",
        a_hold_promoted=False,
        epoch_transition_count=0,
        m_view_released=False,
        m_labels_released=False,
        replay_authorized=False,
        failure_stage="injected_controller_terminal",
        failure_type_sha256="8" * 64,
    )


def _dependencies(events: list[str]) -> study.FormalStudyDependencies:
    class Runtime:
        pass

    class Custody:
        pass

    class Lifecycle:
        def run(self):
            events.append("run")
            return _invalid_disposition()

    def verify_freeze(project, **kwargs):
        events.append("freeze")
        assert project.name == "project"
        assert kwargs["runtime_fingerprint_path"].name == "fingerprint.json"
        assert kwargs["production_canary_path"].name == "canary.json"
        return {"self_sha256": "6" * 64}

    def verify(paths):
        events.append("fingerprint")
        assert paths.hipporag_source.name == "HippoRAG"
        return {"self_sha256": SHA}

    def network():
        events.append("network")
        return {
            "network_properties": [
                "IPAddressDeny=any",
                "RestrictAddressFamilies=AF_UNIX",
            ],
            "returncode": 0,
            "stderr_sha256": "1" * 64,
            "stdout_sha256": "2" * 64,
        }

    def minilm(paths):
        events.append("minilm")
        return object()

    def typed(paths):
        events.append("typed")
        return object()

    def hippo(paths):
        events.append("hippo")
        return object()

    def runtime(**kwargs):
        events.append("runtime")
        assert set(kwargs) == {
            "control_root",
            "receipt_paths",
            "typed_plan_runner",
            "minilm_encoder",
            "hippo_runner",
        }
        return Runtime()

    def custody(**kwargs):
        events.append("custody")
        assert set(kwargs) == {"project_root", "runtime", "control_root"}
        return Custody()

    def lifecycle(**kwargs):
        events.append("controller")
        assert set(kwargs) == {"acquisition", "runtime"}
        return Lifecycle()

    return study.FormalStudyDependencies(
        verify_implementation_freeze=verify_freeze,
        verify_runtime_fingerprint=verify,
        systemd_network_preflight=network,
        minilm_encoder_factory=minilm,
        typed_plan_runner_factory=typed,
        hippo_runner_factory=hippo,
        runtime_adapter_factory=runtime,
        acquisition_adapter_factory=custody,
        controller_factory=lifecycle,
    )


def test_injected_entry_orders_preflight_before_models_and_persists_terminal(
    linux_tmp_path: Path,
) -> None:
    events: list[str] = []
    config = _config(linux_tmp_path)

    result = study.run_formal_study(config, dependencies=_dependencies(events))

    assert events == [
        "freeze",
        "fingerprint",
        "network",
        "minilm",
        "typed",
        "hippo",
        "runtime",
        "custody",
        "controller",
        "run",
    ]
    path = Path(result["path"])
    assert path == config.control_root / study.FINAL_DISPOSITION_FILENAME
    assert path.stat().st_mode & 0o777 == 0o600
    assert config.control_root.stat().st_mode & 0o777 == 0o700
    assert config.work_root.stat().st_mode & 0o777 == 0o700
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["file_sha256"]
    reopened = study.load_final_disposition(path)
    assert reopened == result["disposition"]
    assert reopened["status"] == "implementation_or_runtime_invalid"
    assert reopened["runtime_fingerprint_self_sha256"] == SHA
    assert reopened["controller_disposition"]["failure_stage"] == (
        "injected_controller_terminal"
    )
    assert reopened["offline_artifacts"] == {
        "A_form_archive": None,
        "A_form_fit": None,
        "A_hold_archive": None,
        "A_hold_score": None,
        "F_search_archive": None,
        "M_search_archive": None,
        "M_search_score": None,
        "epoch_authorization": None,
        "E1_model": None,
        "policy_freeze": None,
        "runtime_preflight": None,
    }
    assert path.read_bytes() == study._canonical_bytes(reopened)


def test_bootstrap_failure_is_one_terminal_and_cannot_be_replayed(
    linux_tmp_path: Path,
) -> None:
    config = _config(linux_tmp_path)
    events: list[str] = []
    base = _dependencies(events)

    def fail_network():
        events.append("network")
        raise OSError("content must not be persisted")

    deps = study.FormalStudyDependencies(
        verify_implementation_freeze=base.verify_implementation_freeze,
        verify_runtime_fingerprint=base.verify_runtime_fingerprint,
        systemd_network_preflight=fail_network,
        minilm_encoder_factory=base.minilm_encoder_factory,
        typed_plan_runner_factory=base.typed_plan_runner_factory,
        hippo_runner_factory=base.hippo_runner_factory,
        runtime_adapter_factory=base.runtime_adapter_factory,
        acquisition_adapter_factory=base.acquisition_adapter_factory,
        controller_factory=base.controller_factory,
    )
    result = study.run_formal_study(config, dependencies=deps)
    path = Path(result["path"])
    original = path.read_bytes()

    assert events == ["freeze", "fingerprint", "network"]
    assert not config.work_root.exists()
    disposition = result["disposition"]["controller_disposition"]
    assert disposition["status"] == "implementation_or_runtime_invalid"
    assert disposition["primary_evaluated"] is False
    assert disposition["efficacy"] == "unknown"
    assert disposition["failure_stage"] == "systemd_network_preflight"
    assert "content must not be persisted" not in original.decode("ascii")

    with pytest.raises(study.TatqaP18FormalStudyAlreadyConsumed):
        study.run_formal_study(config, dependencies=deps)
    assert path.read_bytes() == original
    assert events == ["freeze", "fingerprint", "network"]


def test_unproved_worker_closure_cannot_be_persisted_as_terminal(
    linux_tmp_path: Path,
) -> None:
    config = _config(linux_tmp_path)
    events: list[str] = []
    base = _dependencies(events)

    class UnclosedRuntime:
        def abort_all_inference(self) -> None:
            events.append("abort")
            raise RuntimeError("worker stop failed")

        def verify_all_inference_closed(self) -> None:
            events.append("verify-closed")
            raise RuntimeError("worker remains active")

    runtime = UnclosedRuntime()

    class RaisingLifecycle:
        def run(self):
            events.append("run-raises")
            raise RuntimeError("controller escaped")

    deps = study.FormalStudyDependencies(
        verify_implementation_freeze=base.verify_implementation_freeze,
        verify_runtime_fingerprint=base.verify_runtime_fingerprint,
        systemd_network_preflight=base.systemd_network_preflight,
        minilm_encoder_factory=base.minilm_encoder_factory,
        typed_plan_runner_factory=base.typed_plan_runner_factory,
        hippo_runner_factory=base.hippo_runner_factory,
        runtime_adapter_factory=lambda **_kwargs: runtime,
        acquisition_adapter_factory=lambda **_kwargs: object(),
        controller_factory=lambda **_kwargs: RaisingLifecycle(),
    )

    with pytest.raises(
        study.TatqaP18FormalStudyError,
        match="closure could not be proved",
    ):
        study.run_formal_study(config, dependencies=deps)
    assert events[-3:] == ["run-raises", "abort", "verify-closed"]
    assert not (
        config.control_root / study.FINAL_DISPOSITION_FILENAME
    ).exists()


def test_dirty_or_mismatched_freeze_fails_before_fingerprint_or_models(
    linux_tmp_path: Path,
) -> None:
    config = _config(linux_tmp_path)
    events: list[str] = []
    base = _dependencies(events)

    def fail_freeze(*_args, **_kwargs):
        events.append("freeze-rejected")
        raise RuntimeError("uncommitted production member")

    deps = study.FormalStudyDependencies(
        verify_implementation_freeze=fail_freeze,
        verify_runtime_fingerprint=base.verify_runtime_fingerprint,
        systemd_network_preflight=base.systemd_network_preflight,
        minilm_encoder_factory=base.minilm_encoder_factory,
        typed_plan_runner_factory=base.typed_plan_runner_factory,
        hippo_runner_factory=base.hippo_runner_factory,
        runtime_adapter_factory=base.runtime_adapter_factory,
        acquisition_adapter_factory=base.acquisition_adapter_factory,
        controller_factory=base.controller_factory,
    )

    result = study.run_formal_study(config, dependencies=deps)

    assert events == ["freeze-rejected"]
    assert result["disposition"]["controller_disposition"]["failure_stage"] == (
        "implementation_freeze"
    )
    assert not config.work_root.exists()


def test_terminal_tamper_is_rejected(linux_tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(linux_tmp_path)
    result = study.run_formal_study(config, dependencies=_dependencies(events))
    path = Path(result["path"])
    value = json.loads(path.read_text(encoding="ascii"))
    value["status"] = "valid_primary_true"
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(study.TatqaP18FormalStudyError):
        study.load_final_disposition(path)


def test_offline_artifacts_persist_full_a_form_fit_and_e1_model() -> None:
    class Payload:
        def __init__(self, value):
            self.value = value

        def payload(self):
            return dict(self.value)

    model = Payload({"schema": "E1", "coefficients": [1, 2, 3]})
    fit = Payload({"schema": "A_form_fit", "training_rows": ["bound"]})
    fit.model = model
    disposition = SimpleNamespace(
        preflight=None,
        a_form_fit=fit,
        a_form_archive=None,
        f_search_archive=None,
        policy_freeze=None,
        a_hold_archive=None,
        a_hold_score=None,
        epoch_authorization=None,
        m_search_archive=None,
        m_search_score=None,
    )

    artifacts = study._disposition_artifacts(disposition)

    assert artifacts["A_form_fit"] == fit.payload()
    assert artifacts["E1_model"] == model.payload()


def test_cli_surface_is_explicit_and_has_no_secret_or_source_arguments(
    tmp_path: Path,
) -> None:
    arguments = [
        "--project-root",
        str(tmp_path / "project"),
        "--control-root",
        str(tmp_path / "control"),
        "--work-root",
        str(tmp_path / "work"),
        "--runtime-python",
        str(tmp_path / "python"),
        "--qwen-model",
        str(tmp_path / "qwen"),
        "--minilm-asset-manifest",
        str(tmp_path / "minilm.json"),
        "--minilm-model",
        str(tmp_path / "minilm"),
        "--hippo-llm-model",
        str(tmp_path / "llm"),
        "--hippo-embedding-model",
        str(tmp_path / "embedding"),
        "--hipporag-source",
        str(tmp_path / "HippoRAG"),
        "--hippo-attestation",
        str(tmp_path / "attestation.json"),
        "--runtime-fingerprint",
        str(tmp_path / "fingerprint.json"),
        "--production-canary",
        str(tmp_path / "canary.json"),
    ]
    parsed = study._parser().parse_args(arguments)
    assert parsed.hipporag_source == tmp_path / "HippoRAG"
    option_strings = {
        option
        for action in study._parser()._actions
        for option in action.option_strings
    }
    assert not any(
        token in option.lower()
        for option in option_strings
        for token in ("api", "key", "secret", "label", "answer", "source-file")
    )
