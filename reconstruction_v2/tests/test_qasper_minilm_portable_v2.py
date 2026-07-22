from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from replication_runtime.qasper_minilm_portable_v2 import (
    PORTABLE_CANARY_SCHEMA,
    PORTABLE_ROW_L2_NORM_ATOL,
    PortableMiniLMError,
    PortableOfflineMiniLMEncoder,
    run_portable_startup_canary,
)
from replication_runtime.qasper_minilm_portable_v2 import binding
from replication_runtime.qasper_minilm_v1 import binding as frozen_v1


def _matrix(row_count: int = 256) -> np.ndarray:
    value = np.zeros((row_count, frozen_v1.EMBEDDING_DIMENSION), dtype=np.float32)
    value[:, 0] = 1.0
    if row_count > 1:
        value[1, 0] = 0.0
        value[1, 1] = 1.0
    return value


class _FakeEncoder:
    def __init__(self, matrices: list[np.ndarray] | None = None) -> None:
        self.calls = 0
        self.matrices = matrices

    def encode(self, texts: object) -> np.ndarray:
        assert tuple(texts) == frozen_v1.synthetic_canary_texts()
        index = self.calls
        self.calls += 1
        source = _matrix() if self.matrices is None else self.matrices[index]
        return source.copy()


def test_portable_canary_accepts_only_structural_repeat_and_marks_hashes_non_normative() -> None:
    encoder = _FakeEncoder()
    receipt = run_portable_startup_canary(encoder)
    assert encoder.calls == 2
    assert receipt["schema"] == PORTABLE_CANARY_SCHEMA
    assert receipt["embedding_shape"] == [256, 384]
    assert receipt["embedding_dtype"] == "float32"
    assert receipt["repeat_count"] == 2
    assert receipt["repeat_byte_exact"] is True
    assert receipt["repeat_elementwise_exact"] is True
    assert receipt["at_least_two_distinct_vectors"] is True
    assert receipt["maximum_observed_row_l2_norm_error"] == 0.0
    assert receipt["per_row_l2_norm_maximum_error"] == 1e-5
    hashes = receipt["observed_output_hashes"]
    assert hashes["normative_acceptance"] is False
    assert hashes["compared_to_expected_or_allowlist"] is False
    assert len(hashes["float32_little_endian_c_order_sha256"]) == 64
    assert len(hashes["quantized_embedding_matrix_sha256"]) == 64
    assert receipt["formal_QASPER_source_or_rows_accessed"] is False
    assert receipt["formal_TAT_QA_source_or_rows_accessed"] is False
    assert receipt["qasper_rows_or_archives_accessed_by_canary"] is False
    assert receipt["tatqa_rows_or_archives_accessed_by_canary"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.astype(np.float64), "dtype"),
        (lambda value: value[:, :-1], "shape"),
        (
            lambda value: np.where(
                np.indices(value.shape)[1] == 0, np.float32(np.nan), value
            ).astype(np.float32),
            "non-finite",
        ),
        (lambda value: value * np.float32(0.99), "norm"),
        (
            lambda value: np.repeat(value[:1], repeats=len(value), axis=0),
            "collapsed",
        ),
    ],
)
def test_portable_canary_rejects_invalid_matrix(mutate: object, message: str) -> None:
    invalid = mutate(_matrix())
    encoder = _FakeEncoder([invalid, invalid])
    with pytest.raises(PortableMiniLMError, match=message):
        run_portable_startup_canary(encoder)


def test_portable_canary_rejects_nonexact_second_encode() -> None:
    first = _matrix()
    second = first.copy()
    second[1, 0] = np.float32(1e-7)
    second[1, 1] = np.float32(np.sqrt(1.0 - float(second[1, 0]) ** 2))
    with pytest.raises(PortableMiniLMError, match="byte/element exact"):
        run_portable_startup_canary(_FakeEncoder([first, second]))


def test_encoder_reuses_v1_binding_and_never_invokes_v1_hash_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def verify(**kwargs: object) -> dict[str, object]:
        calls.append(("verify", kwargs))
        return {"status": "verified_offline_immutable_qasper_minilm_runtime"}

    def configure() -> None:
        calls.append("offline")

    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            calls.append(("encode", kwargs))
            return _matrix(len(texts))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the v1 expected-output-hash canary must not run")

    monkeypatch.setattr(frozen_v1, "verify_runtime_binding", verify)
    monkeypatch.setattr(frozen_v1, "_configure_offline_environment", configure)
    monkeypatch.setattr(frozen_v1, "run_synthetic_canary", forbidden)
    monkeypatch.setattr(binding, "_load_exact_v1_model", lambda **kwargs: FakeModel())

    encoder = PortableOfflineMiniLMEncoder(
        asset_manifest_path="public-manifest.json",
        model_root="public-model",
    )
    assert encoder.runtime_receipt["status"].startswith("verified_offline")
    assert encoder.canary_receipt["status"].startswith("passed_portable")
    assert calls[0][0] == "verify"
    assert calls[1] == "offline"
    encode_kwargs = calls[2][1]
    assert encode_kwargs == {
        "batch_size": frozen_v1.BATCH_SIZE,
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "device": "cpu",
        "normalize_embeddings": True,
        "precision": "float32",
        "show_progress_bar": False,
    }
    with pytest.raises(PortableMiniLMError, match="cannot be skipped"):
        PortableOfflineMiniLMEncoder(
            asset_manifest_path="public-manifest.json",
            model_root="public-model",
            run_canary=False,
        )


def test_exact_model_constructor_keeps_v1_cpu_float32_offline_safety(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}
    float32 = object()

    class Parameter:
        device = SimpleNamespace(type="cpu")
        dtype = float32

    class FakeModel:
        training = True

        def __init__(self, path: str, **kwargs: object) -> None:
            calls["path"] = path
            calls["kwargs"] = kwargs

        def float(self) -> None:
            calls["float"] = True

        def eval(self) -> None:
            self.training = False
            calls["eval"] = True

        def parameters(self) -> tuple[Parameter, ...]:
            return (Parameter(),)

    torch = ModuleType("torch")
    torch.float32 = float32
    torch.set_num_threads = lambda value: calls.setdefault("threads", value)
    torch.manual_seed = lambda value: calls.setdefault("seed", value)
    torch.use_deterministic_algorithms = lambda value: calls.setdefault(
        "deterministic", value
    )
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setattr(
        frozen_v1, "_reject_symlink_components", lambda path, field: Path(path).absolute()
    )

    model = binding._load_exact_v1_model(model_root=tmp_path)
    assert model.max_seq_length == frozen_v1.MAXIMUM_SEQUENCE_LENGTH
    assert calls["threads"] == 1
    assert calls["seed"] == 0
    assert calls["deterministic"] is True
    assert calls["float"] is True and calls["eval"] is True
    kwargs = calls["kwargs"]
    assert kwargs["device"] == "cpu"
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False
    assert kwargs["model_kwargs"] == {
        "local_files_only": True,
        "torch_dtype": float32,
        "use_safetensors": True,
    }
    assert kwargs["config_kwargs"] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }


def test_norm_tolerance_is_exactly_portable_bound() -> None:
    assert PORTABLE_ROW_L2_NORM_ATOL == 1e-5
