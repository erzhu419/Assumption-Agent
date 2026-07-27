from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import replication_runtime.dstc9_official_hipporag_v1.adapter as adapter
import replication_runtime.dstc9_official_hipporag_v1.contract as contract
import replication_runtime.dstc9_official_hipporag_v1.runtime_binding as binding
import replication_runtime.dstc9_official_hipporag_v1.worker as worker


STUDY_ID = "DSTC9_TRACK1_SYNTHETIC_SOURCE_FREE_V1"


def _units(count: int = contract.CORPUS_SIZE) -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "text": (
                f"Synthetic DSTC9 FAQ snippet {ordinal}. "
                f"Unique offline token D9-{ordinal}."
            ),
        }
        for ordinal in range(count)
    ]


def _queries(count: int = 17) -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "query_text": f"Exact synthetic dialogue query {ordinal}?",
            "work_id": f"opaque-work-{ordinal:04d}",
        }
        for ordinal in range(count)
    ]


def _corpus_input(
    units: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return contract.make_corpus_input(
        study_id=STUDY_ID,
        units=_units() if units is None else units,
    )


def _query_input(count: int = 17) -> dict[str, object]:
    return contract.make_query_input(
        study_id=STUDY_ID,
        queries=_queries(count),
    )


def _recompute_self_hash(payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("self_sha256", None)
    payload["self_sha256"] = contract.stable_hash(body)


def _exact_worker_environment(root: Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "0",
        "HOME": str(root / "home"),
        "HF_HOME": str(root / "cache"),
        "HF_HUB_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(root / "tmp" / "pycache"),
        "PYTHONNOUSERSITE": "1",
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


@dataclass
class _Store:
    documents: list[str] | None = None
    index_calls: int = 0
    index_root: Path | None = None


class _FakeCore:
    def __init__(self, store: _Store, *, build_mode: bool) -> None:
        self.store = store
        self.build_mode = build_mode
        self.retrieve_calls: list[tuple[tuple[str, ...], int]] = []

    def index(self, documents: list[str]) -> None:
        if not self.build_mode:
            raise AssertionError("reopened core must never index")
        self.store.index_calls += 1
        self.store.documents = list(documents)
        assert self.store.index_root is not None
        self.store.index_root.mkdir(parents=True, exist_ok=True)
        (self.store.index_root / "synthetic.index").write_bytes(
            b"synthetic source-free DSTC9 official index"
        )

    def retrieve(self, queries: list[str], *, num_to_retrieve: int) -> list[object]:
        assert self.store.documents is not None
        assert num_to_retrieve == contract.CORPUS_SIZE
        self.retrieve_calls.append((tuple(queries), num_to_retrieve))
        unique = list(reversed(tuple(dict.fromkeys(self.store.documents))))
        scores = [1.0] * len(unique)
        return [
            SimpleNamespace(docs=list(unique), doc_scores=list(scores))
            for _query in queries
        ]


def test_self_hashed_inputs_are_exact_source_free_projections() -> None:
    corpus_payload = _corpus_input()
    query_payload = _query_input(3)
    corpus = contract.validate_corpus_input(corpus_payload)
    queries = contract.validate_query_input(
        query_payload, expected_study_id=STUDY_ID
    )

    assert set(corpus_payload) == contract.CORPUS_INPUT_KEYS
    assert set(query_payload) == contract.QUERY_INPUT_KEYS
    assert len(corpus.units) == contract.CORPUS_SIZE == 2900
    assert len(queries.queries) == 3
    assert contract.serialize_corpus(corpus.units)[7] == _units()[7]["text"]
    assert contract.serialize_queries(queries.queries) == tuple(
        row["query_text"] for row in _queries(3)
    )
    assert contract.corpus_input_projection(corpus) == corpus_payload
    assert contract.query_input_projection(queries) == query_payload

    tampered = dict(corpus_payload)
    tampered["study_id"] = "A_DIFFERENT_STUDY"
    with pytest.raises(contract.Dstc9OfficialHippoRAGError, match="self hash"):
        contract.validate_corpus_input(tampered)

    with pytest.raises(contract.Dstc9OfficialHippoRAGError, match="different study"):
        contract.validate_query_input(
            query_payload, expected_study_id="A_DIFFERENT_STUDY"
        )


@pytest.mark.parametrize(
    ("container", "forbidden_key"),
    [
        ("unit", "domain"),
        ("unit", "entity_id"),
        ("unit", "doc_id"),
        ("query", "family"),
        ("query", "qrel"),
        ("query", "answer"),
        ("query", "score"),
        ("query", "label"),
    ],
)
def test_forbidden_source_label_and_score_fields_fail_closed(
    container: str, forbidden_key: str
) -> None:
    if container == "unit":
        payload = _corpus_input()
        units = payload["units"]
        assert isinstance(units, list)
        assert isinstance(units[0], dict)
        units[0][forbidden_key] = "forbidden"
        _recompute_self_hash(payload)
        validator = contract.validate_corpus_input
    else:
        payload = _query_input(1)
        queries = payload["queries"]
        assert isinstance(queries, list)
        assert isinstance(queries[0], dict)
        queries[0][forbidden_key] = "forbidden"
        _recompute_self_hash(payload)
        validator = contract.validate_query_input
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError, match="forbidden"
    ):
        validator(payload)


def test_exact_cardinality_contiguous_ordinals_and_query_bound() -> None:
    assert (
        contract.MIN_CORPUS_SIZE
        == contract.MAX_CORPUS_SIZE
        == contract.CORPUS_SIZE
        == 2900
    )
    for count in (2899, 2901):
        with pytest.raises(
            contract.Dstc9OfficialHippoRAGError, match="exactly 2900"
        ):
            contract.validate_corpus(_units(count))

    bad_units = _units()
    bad_units[7]["ordinal"] = 8
    with pytest.raises(contract.Dstc9OfficialHippoRAGError, match="contiguous"):
        contract.validate_corpus(bad_units)

    assert len(contract.validate_queries(_queries(256))) == 256
    with pytest.raises(contract.Dstc9OfficialHippoRAGError, match="1..256"):
        contract.validate_queries(_queries(257))

    bad_queries = _queries(2)
    bad_queries[1]["ordinal"] = 0
    with pytest.raises(contract.Dstc9OfficialHippoRAGError, match="contiguous"):
        contract.validate_queries(bad_queries)
    duplicate_work = _queries(2)
    duplicate_work[1]["work_id"] = duplicate_work[0]["work_id"]
    with pytest.raises(contract.Dstc9OfficialHippoRAGError, match="unique"):
        contract.validate_queries(duplicate_work)


def test_duplicate_text_expansion_ties_and_md5_collision_fail_before_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate_units = _units()
    duplicate_units[-1]["text"] = duplicate_units[0]["text"]
    corpus = contract.validate_corpus_input(_corpus_input(duplicate_units))
    documents = contract.serialize_corpus(corpus.units)
    mapping: dict[str, list[int]] = {}
    for document, row in zip(documents, corpus.units):
        mapping.setdefault(document, []).append(row.ordinal)
    unique = list(reversed(tuple(mapping)))
    scores = [2.0 if text == documents[0] else 1.0 for text in unique]
    assert contract.stable_top_five_from_official_result(
        retrieved_documents=unique,
        retrieved_scores=scores,
        document_to_ordinals=mapping,
    ) == (0, contract.CORPUS_SIZE - 1, 1, 2, 3)

    all_tied = contract.stable_top_five_from_official_result(
        retrieved_documents=unique,
        retrieved_scores=[1.0] * len(unique),
        document_to_ordinals=mapping,
    )
    assert all_tied == (0, 1, 2, 3, 4)

    class _CollidingMd5:
        def hexdigest(self) -> str:
            return "0" * 32

    monkeypatch.setattr(
        contract.hashlib, "md5", lambda _raw: _CollidingMd5()
    )
    index_root = tmp_path / "must-not-exist"
    store = _Store(index_root=index_root)
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError,
        match="distinct corpus texts collide",
    ):
        worker.build_index_with_core(
            core=_FakeCore(store, build_mode=True),
            corpus_input=_corpus_input(),
            index_root=index_root,
            runtime_attestation_receipt_sha256="a" * 64,
        )
    assert store.index_calls == 0
    assert not index_root.exists()


def test_build_once_reopen_retrieve_has_zero_retrieve_index_calls(
    tmp_path: Path,
) -> None:
    corpus_payload = _corpus_input()
    query_payload = _query_input(17)
    index_root = tmp_path / "index"
    store = _Store(index_root=index_root)
    build_receipt = worker.build_index_with_core(
        core=_FakeCore(store, build_mode=True),
        corpus_input=corpus_payload,
        index_root=index_root,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    assert store.index_calls == 1
    assert len(store.documents or ()) == 2900
    assert build_receipt["corpus_count"] == 2900
    assert build_receipt["index_call_count"] == 1
    assert build_receipt["retry_count"] == 0
    assert build_receipt["dynamic_resize_count"] == 0
    assert build_receipt["cuda_visible_devices"] == "0"
    assert build_receipt["logical_cuda_device"] == "cuda:0"

    reopened = _FakeCore(store, build_mode=False)
    indices, batch_sizes = worker.retrieve_batches_with_core(
        core=reopened,
        corpus_input=corpus_payload,
        query_input=query_payload,
    )
    assert indices == ((0, 1, 2, 3, 4),) * 17
    assert batch_sizes == (8, 8, 1)
    assert [len(call[0]) for call in reopened.retrieve_calls] == [8, 8, 1]
    assert all(call[1] == 2900 for call in reopened.retrieve_calls)
    assert store.index_calls == 1


def test_output_is_canonical_ordinals_only_without_text_or_scores(
    tmp_path: Path,
) -> None:
    corpus_payload = _corpus_input()
    query_payload = _query_input(2)
    corpus = contract.validate_corpus_input(corpus_payload)
    queries = contract.validate_query_input(
        query_payload, expected_study_id=STUDY_ID
    )
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "index.bin").write_bytes(b"synthetic")
    snapshot = contract.snapshot_index_tree(index_root)
    build_receipt = contract.make_build_receipt(
        corpus,
        index_snapshot=snapshot,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    indices = ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0))
    retrieval_receipt = contract.make_retrieval_receipt(
        corpus_input=corpus,
        query_input=queries,
        indices=indices,
        batch_sizes=[2],
        build_receipt=build_receipt,
        index_snapshot_before=snapshot,
        index_snapshot_after=snapshot,
    )
    output = {
        "receipt": retrieval_receipt,
        "retrieved_ordinals": [list(row) for row in indices],
        "schema": contract.RETRIEVAL_OUTPUT_SCHEMA,
    }
    raw = contract.canonical_json_bytes(output)
    parsed = contract.parse_retrieval_output(
        raw,
        query_input=queries,
        expected_build_receipt=build_receipt,
        expected_index_snapshot_after=snapshot,
    )
    assert parsed.indices == indices
    assert b"Exact synthetic dialogue query" not in raw
    assert b"opaque-work" not in raw
    assert b"Synthetic DSTC9 FAQ snippet" not in raw
    assert b'"score"' not in raw
    assert b'"text"' not in raw

    leaked = dict(output)
    leaked["scores"] = [[1.0] * 5, [1.0] * 5]
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError, match="only ordinals"
    ):
        contract.parse_retrieval_output(
            contract.canonical_json_bytes(leaked),
            query_input=queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=snapshot,
        )

    duplicate = json.loads(raw)
    duplicate["retrieved_ordinals"][0] = [0, 0, 1, 2, 3]
    with pytest.raises(contract.Dstc9OfficialHippoRAGError, match="duplicate"):
        contract.parse_retrieval_output(
            contract.canonical_json_bytes(duplicate),
            query_input=queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=snapshot,
        )


def test_worker_environment_and_logical_cuda0_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _exact_worker_environment(tmp_path)
    assert frozenset(environment) == contract.WORKER_ENVIRONMENT_KEYS
    monkeypatch.setattr(
        worker.sys,
        "pycache_prefix",
        environment["PYTHONPYCACHEPREFIX"],
    )
    worker._validate_effective_environment(environment)

    contaminated = dict(environment)
    contaminated["OPENAI_API_KEY"] = "forbidden"
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError, match="environment contract"
    ):
        worker._validate_effective_environment(contaminated)
    wrong_gpu = dict(environment)
    wrong_gpu["CUDA_VISIBLE_DEVICES"] = "1"
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError, match="environment contract"
    ):
        worker._validate_effective_environment(wrong_gpu)
    monkeypatch.setattr(worker.sys, "pycache_prefix", "/wrong/prefix")
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError, match="environment contract"
    ):
        worker._validate_effective_environment(environment)

    class _Cuda:
        def __init__(self, count: int = 1) -> None:
            self.count = count

        @staticmethod
        def is_available() -> bool:
            return True

        def device_count(self) -> int:
            return self.count

        @staticmethod
        def set_device(_ordinal: int) -> None:
            return None

        @staticmethod
        def current_device() -> int:
            return 0

    class _Torch:
        def __init__(self, count: int = 1) -> None:
            self.cuda = _Cuda(count)

        @staticmethod
        def empty(_count: int, *, device: str) -> object:
            return SimpleNamespace(device=device)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    worker._validate_logical_cuda0(_Torch())
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError, match="logical cuda:0"
    ):
        worker._validate_logical_cuda0(_Torch(count=2))


def test_isolated_xoption_overrides_ignored_pycache_environment(
    tmp_path: Path,
) -> None:
    ignored = tmp_path / "ignored-by-isolated-mode"
    bound = tmp_path / "fresh-work/tmp/pycache"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={bound}",
            "-c",
            (
                "import json,os,sys;"
                "print(json.dumps({"
                "'env':os.environ['PYTHONPYCACHEPREFIX'],"
                "'prefix':sys.pycache_prefix,"
                "'write':sys.dont_write_bytecode"
                "},sort_keys=True))"
            ),
        ],
        check=False,
        capture_output=True,
        env={
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONPYCACHEPREFIX": str(ignored),
        },
        text=True,
    )
    assert completed.returncode == 0
    observed = json.loads(completed.stdout)
    assert observed == {
        "env": str(ignored),
        "prefix": str(bound),
        "write": True,
    }


def test_official_core_config_is_gpu0_offline_fixed_and_never_resized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        worker, "_validate_logical_cuda0", lambda: calls.append("cuda:0")
    )

    class _BaseConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _HippoRAG:
        def __init__(self, *, global_config: _BaseConfig) -> None:
            self.global_config = global_config
            self.llm_model = SimpleNamespace(
                llm_config=SimpleNamespace(generate_params={})
            )

    hipporag_module = ModuleType("hipporag")
    hipporag_module.HippoRAG = _HippoRAG  # type: ignore[attr-defined]
    utils_module = ModuleType("hipporag.utils")
    config_module = ModuleType("hipporag.utils.config_utils")
    config_module.BaseConfig = _BaseConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hipporag", hipporag_module)
    monkeypatch.setitem(sys.modules, "hipporag.utils", utils_module)
    monkeypatch.setitem(
        sys.modules, "hipporag.utils.config_utils", config_module
    )

    core = worker._build_official_core(
        save_dir=tmp_path / "index",
        llm_model=tmp_path / "llm",
        embedding_model=tmp_path / "embedding",
        force_index_from_scratch=False,
        corpus_count=2900,
    )
    assert calls == ["cuda:0"]
    config = core.global_config.kwargs
    assert config["retrieval_top_k"] == 2900
    assert config["qa_top_k"] == 5
    assert config["max_retry_attempts"] == 0
    assert config["force_index_from_scratch"] is False
    assert str(config["llm_name"]).startswith("Transformers/")
    assert str(config["embedding_model_name"]).startswith("Transformers/")
    assert core.llm_model.llm_config.generate_params["max_tokens"] == 4


def test_adapter_uses_only_the_p17_reused_closure_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    expected_receipt = {
        "project_root": str(tmp_path / "exact-p17-project"),
        "schema": binding.SCHEMA,
    }

    def _verify(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return expected_receipt

    monkeypatch.setattr(
        adapter, "verify_p17_reused_closure_binding", _verify
    )
    result = adapter._validated_runtime(
        expected_study_id=STUDY_ID,
        worker_project_root=tmp_path / "formal-project",
        current_hardware_binding_path=tmp_path / "hardware.json",
        runtime_python=tmp_path / "python",
        local_llm_model=tmp_path / "smollm",
        local_embedding_model=tmp_path / "minilm",
        runtime_fingerprint_path=tmp_path / "fingerprint.json",
    )

    assert result[-1] == expected_receipt
    assert result[0] == tmp_path / "exact-p17-project"
    assert calls == [
        {
            "current_hardware_binding_path": (
                tmp_path / "hardware.json"
            ).absolute(),
            "expected_study_id": STUDY_ID,
            "local_embedding_model": (tmp_path / "minilm").absolute(),
            "local_llm_model": (tmp_path / "smollm").absolute(),
            "runtime_fingerprint_path": (
                tmp_path / "fingerprint.json"
            ).absolute(),
            "runtime_python": (tmp_path / "python").absolute(),
            "worker_project_root": (
                tmp_path / "formal-project"
            ).absolute(),
        }
    ]
    adapter_source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "verify_formal_runtime_attestation_v3" not in adapter_source
    assert "musique_official_hipporag_v1" not in adapter_source
    assert not hasattr(adapter, "verify_formal_runtime_attestation_v3")


def test_committed_p17_fingerprint_matches_the_reused_closure_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint_path = (
        Path(__file__).parents[1]
        / "manifests/bright_p17_remote_runtime_fingerprint_v1.json"
    )
    monkeypatch.setattr(binding, "P17_FINGERPRINT_PATH", fingerprint_path)
    fingerprint = binding._load_exact_fingerprint(fingerprint_path)
    binding._verify_fingerprint_contract(fingerprint)
    assert fingerprint["self_sha256"] == binding.FINGERPRINT_SELF_SHA256
    assert (
        fingerprint["runtime_inventory_receipt"]
        == binding.EXPECTED_RUNTIME_INVENTORY
    )


def test_committed_minilm_manifest_matches_without_reading_model_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "manifests/qasper_minilm_runtime_asset_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    declared_rows = manifest["local_binding"]["snapshot_files"]
    rows = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "size_bytes": row["size"],
        }
        for row in declared_rows
    ]
    monkeypatch.setattr(binding, "P17_MINILM_MANIFEST", manifest_path)
    monkeypatch.setattr(
        binding,
        "_tree_rows",
        lambda _root, _field_name: rows,
    )
    receipt = binding._verify_minilm_manifest_and_tree()
    assert receipt["generic_tree"] == binding.EXPECTED_ASSET_TREES["MiniLM"]
    assert (
        receipt["normative_tree_sha256"]
        == binding.MINILM_NORMATIVE_TREE_SHA256
    )


def test_hipporag_source_tree_binds_only_normative_files_and_diagnoses_pyc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "hipporag-source"
    source_file = source_root / "hipporag/HippoRAG.py"
    init_file = source_root / "hipporag/__init__.py"
    cache_file = (
        source_root
        / "hipporag/__pycache__/HippoRAG.cpython-310.pyc"
    )
    cache_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"# synthetic repaired source\n")
    init_file.write_bytes(b"# synthetic package\n")
    cache_file.write_bytes(b"synthetic bytecode generation one")
    normative_rows = [
        row
        for row in binding._tree_rows(source_root, "synthetic source")
        if "__pycache__" not in Path(str(row["path"])).parts
        and Path(str(row["path"])).suffix != ".pyc"
    ]
    normative = {
        "file_count": len(normative_rows),
        "size_bytes": sum(
            int(row["size_bytes"]) for row in normative_rows
        ),
        "tree_sha256": binding.stable_hash(normative_rows),
    }
    monkeypatch.setattr(binding, "P17_HIPPORAG_SOURCE", source_root)
    monkeypatch.setattr(
        binding, "HIPPORAG_SOURCE_NORMATIVE_TREE", normative
    )

    first = binding._hipporag_source_tree_receipt()
    assert first["normative_tree"] == normative
    diagnostic = first["source_local_bytecode_diagnostic"]
    assert diagnostic["file_count"] == 1
    cache_file.write_bytes(b"synthetic bytecode generation two is larger")
    second = binding._hipporag_source_tree_receipt()
    assert second["normative_tree"] == normative
    assert (
        second["source_local_bytecode_diagnostic"]["tree_sha256"]
        != diagnostic["tree_sha256"]
    )

    rogue_pyc = source_root / "hipporag/rogue.pyc"
    rogue_pyc.write_bytes(b"not beneath pycache")
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match="non-cache content",
    ):
        binding._hipporag_source_tree_receipt()
    rogue_pyc.unlink()
    rogue_cache_file = source_root / "hipporag/__pycache__/rogue.txt"
    rogue_cache_file.write_bytes(b"not bytecode")
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match="non-cache content",
    ):
        binding._hipporag_source_tree_receipt()


def test_tree_rows_use_relative_posix_string_order(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    package_file = source_root / "hipporag/HippoRAG.py"
    metadata_file = source_root / "hipporag.egg-info/PKG-INFO"
    package_file.parent.mkdir(parents=True)
    metadata_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"package")
    metadata_file.write_bytes(b"metadata")

    rows = binding._tree_rows(source_root, "synthetic source")

    assert [row["path"] for row in rows] == [
        "hipporag.egg-info/PKG-INFO",
        "hipporag/HippoRAG.py",
    ]


def test_runtime_identity_excludes_only_source_bytecode_diagnostic() -> None:
    source = {
        "historical_P17_aggregate_lineage": {
            "acceptance_role": "lineage_only_not_live_identity",
            **binding.EXPECTED_ASSET_TREES["HippoRAG_source"],
        },
        "normative_tree": dict(binding.HIPPORAG_SOURCE_NORMATIVE_TREE),
        "repaired_file_sha256": binding.REPAIRED_SOURCE_FILE_SHA256,
        "source_local_bytecode_diagnostic": {
            "acceptance_role": "diagnostic_only_not_live_identity",
            "allowed_shape": "only___pycache___descendant_dot_pyc_v1",
            "file_count": 36,
            "size_bytes": 195_884,
            "tree_sha256": "a" * 64,
        },
        "source_root": str(binding.P17_HIPPORAG_SOURCE),
    }
    body = {
        "assets": {"HippoRAG_source": source},
        "hardware": "bound",
        "schema": binding.SCHEMA,
    }
    first = {**body, "self_sha256": binding.stable_hash(body)}
    changed_body = json.loads(json.dumps(body))
    changed_body["assets"]["HippoRAG_source"][
        "source_local_bytecode_diagnostic"
    ] = {
        "acceptance_role": "diagnostic_only_not_live_identity",
        "allowed_shape": "only___pycache___descendant_dot_pyc_v1",
        "file_count": 37,
        "size_bytes": 195_944,
        "tree_sha256": "b" * 64,
    }
    changed = {
        **changed_body,
        "self_sha256": binding.stable_hash(changed_body),
    }
    assert (
        binding.runtime_binding_acceptance_identity(first)
        == binding.runtime_binding_acceptance_identity(changed)
    )

    changed_body["hardware"] = "drifted"
    drifted = {
        **changed_body,
        "self_sha256": binding.stable_hash(changed_body),
    }
    assert (
        binding.runtime_binding_acceptance_identity(first)
        != binding.runtime_binding_acceptance_identity(drifted)
    )


def test_current_hardware_receipt_is_pre_canary_and_not_old_p17_host_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_project_root = tmp_path / "formal/reconstruction_v2"
    worker_project_root.mkdir(parents=True)
    receipt_path = tmp_path / "formal/receipts/hardware.json"
    receipt_path.parent.mkdir()
    payload = binding.make_current_study_hardware_binding(
        study_id="DSTC9_SYNTHETIC_V1",
        capture_id="PRE_CANARY_HARDWARE_V1",
        gpus=binding.EXPECTED_GPU_ROWS,
        nvidia_driver_version="595.84",
        kernel_release="7.0.0-28-generic",
    )
    receipt_path.write_bytes(binding.canonical_json_bytes(payload))
    assert "source_free_canary_receipt_sha256" not in json.dumps(payload)
    assert payload["status"] == binding.CURRENT_HARDWARE_STATUS
    assert (
        payload["source_free_boundary"]["formal_source_open_count"] == 0
    )
    assert (
        payload["source_free_boundary"][
            "old_P17_driver_or_kernel_used_as_requirement"
        ]
        is False
    )
    monkeypatch.setattr(
        binding,
        "_probe_current_hardware",
        lambda: dict(payload["hardware"]),
    )
    verified = binding.verify_current_study_hardware_binding(
        path=receipt_path,
        worker_project_root=worker_project_root,
        expected_study_id="DSTC9_SYNTHETIC_V1",
    )
    assert verified["hardware"] == payload["hardware"]

    changed = dict(payload["hardware"])
    changed["kernel_release"] = "a-different-current-kernel"
    monkeypatch.setattr(
        binding, "_probe_current_hardware", lambda: changed
    )
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match="no longer matches",
    ):
        binding.verify_current_study_hardware_binding(
            path=receipt_path,
            worker_project_root=worker_project_root,
            expected_study_id="DSTC9_SYNTHETIC_V1",
        )


def test_worker_code_binding_uses_deployed_formal_root_not_p17() -> None:
    worker_project_root = Path(__file__).parents[1]
    receipt = binding._verify_worker_project_root(worker_project_root)
    assert receipt["project_root"] == str(worker_project_root)
    assert receipt["project_root"] != str(binding.P17_PROJECT_ROOT)
    assert [row["path"] for row in receipt["files"]] == list(
        binding.WORKER_CODE_RELATIVE_FILES
    )


def test_isolated_bootstrap_reaches_only_the_formal_worker_root(
    tmp_path: Path,
) -> None:
    formal_root = tmp_path / "formal/reconstruction_v2"
    package = (
        formal_root
        / "replication_runtime/dstc9_official_hipporag_v1"
    )
    package.mkdir(parents=True)
    (formal_root / "replication_runtime/__init__.py").write_text(
        "", encoding="utf-8"
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text(
        "import json\n"
        "def main(argv):\n"
        " print(json.dumps({'argv':argv,'origin':__file__},sort_keys=True))\n"
        " return 0\n",
        encoding="utf-8",
    )
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    isolated_probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            (
                "import importlib.util;"
                "raise SystemExit(0 if "
                "importlib.util.find_spec("
                "'replication_runtime.dstc9_official_hipporag_v1.worker'"
                ") is None else 9)"
            ),
        ],
        check=False,
        capture_output=True,
        cwd=empty_cwd,
        env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        text=True,
    )
    assert isolated_probe.returncode != 9

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            adapter.WORKER_BOOTSTRAP_SCRIPT,
            str(formal_root),
            "--synthetic",
            "one",
        ],
        check=False,
        capture_output=True,
        cwd=empty_cwd,
        env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        text=True,
    )
    assert completed.returncode == 0
    terminal = json.loads(completed.stdout)
    assert terminal["argv"] == ["--synthetic", "one"]
    assert Path(terminal["origin"]) == package / "worker.py"


def test_exact_fingerprint_path_and_file_hash_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_root = tmp_path / "p17"
    fingerprint_path = (
        remote_root
        / "runtime/reconstruction_v2/manifests/"
        "bright_p17_remote_runtime_fingerprint_v1.json"
    )
    fingerprint_path.parent.mkdir(parents=True)
    body = {
        "remote_root": str(remote_root),
        "schema": binding.FINGERPRINT_SCHEMA,
        "status": binding.FINGERPRINT_STATUS,
    }
    payload = {**body, "self_sha256": binding.stable_hash(body)}
    raw = binding.canonical_json_bytes(payload)
    fingerprint_path.write_bytes(raw)
    monkeypatch.setattr(binding, "P17_REMOTE_ROOT", remote_root)
    monkeypatch.setattr(binding, "P17_FINGERPRINT_PATH", fingerprint_path)
    monkeypatch.setattr(
        binding,
        "FINGERPRINT_FILE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        binding, "FINGERPRINT_SELF_SHA256", payload["self_sha256"]
    )

    assert binding._load_exact_fingerprint(fingerprint_path) == payload
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match="fingerprint path drifted",
    ):
        binding._load_exact_fingerprint(tmp_path / "other.json")

    fingerprint_path.write_bytes(raw.replace(b"p17", b"q17", 1))
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match="file hash drifted",
    ):
        binding._load_exact_fingerprint(fingerprint_path)


def test_runtime_python_and_pth_hash_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        binding.stable_hash(binding.EXPECTED_PTH_ROWS)
        == binding.EXPECTED_PTH_SET_SHA256
    )
    runtime_python = tmp_path / "venv/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"synthetic-p17-python")
    runtime_python.chmod(0o700)
    monkeypatch.setattr(binding, "P17_RUNTIME_PYTHON", runtime_python)
    monkeypatch.setattr(
        binding,
        "RUNTIME_PYTHON_TARGET_SIZE",
        runtime_python.stat().st_size,
    )
    monkeypatch.setattr(
        binding,
        "RUNTIME_PYTHON_TARGET_SHA256",
        hashlib.sha256(runtime_python.read_bytes()).hexdigest(),
    )
    assert (
        binding._verify_runtime_python(runtime_python)[
            "resolved_target_size_bytes"
        ]
        == len(b"synthetic-p17-python")
    )
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match="lexical path drifted",
    ):
        binding._verify_runtime_python(tmp_path / "other-python")
    runtime_python.write_bytes(b"synthetic-p17-pythoN")
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match="target drifted",
    ):
        binding._verify_runtime_python(runtime_python)

    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    pth = site_packages / "exact.pth"
    pth.write_bytes(b"/synthetic/P17/project\n")
    rows = [
        {
            "name": pth.name,
            "sha256": hashlib.sha256(pth.read_bytes()).hexdigest(),
            "size_bytes": pth.stat().st_size,
        }
    ]
    monkeypatch.setattr(binding, "P17_VENV_SITE_PACKAGES", site_packages)
    monkeypatch.setattr(binding, "EXPECTED_PTH_ROWS", rows)
    monkeypatch.setattr(
        binding, "EXPECTED_PTH_SET_SHA256", binding.stable_hash(rows)
    )
    assert binding._verify_pth_topology() == rows
    pth.write_bytes(b"/synthetic/P17/projecT\n")
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match=r"\.pth binding drifted",
    ):
        binding._verify_pth_topology()


def test_worker_provenance_requires_all_exact_pth_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    worker_project_root = tmp_path / "formal-project"
    p16_site = tmp_path / "p16-site"
    source_root = tmp_path / "hipporag-source"
    for path in (
        project_root,
        worker_project_root,
        p16_site,
        source_root,
    ):
        path.mkdir()
    receipt = {
        "project_root": str(project_root),
        "schema": binding.SCHEMA,
    }
    receipt_path = tmp_path / "binding.json"
    receipt_raw = binding.canonical_json_bytes(receipt)
    receipt_path.write_bytes(receipt_raw)
    monkeypatch.setattr(binding, "P17_PROJECT_ROOT", project_root)
    monkeypatch.setattr(binding, "P17_P16_SITE", p16_site)
    monkeypatch.setattr(binding, "P17_HIPPORAG_SOURCE", source_root)
    monkeypatch.setattr(
        binding, "P17_RUNTIME_PYTHON", Path(sys.executable).absolute()
    )
    monkeypatch.setattr(
        binding,
        "verify_p17_reused_closure_binding",
        lambda **_kwargs: receipt,
    )
    monkeypatch.setattr(
        binding,
        "runtime_binding_acceptance_identity",
        lambda _value: "same-synthetic-runtime",
    )
    with pytest.raises(
        binding.Dstc9P17RuntimeBindingError,
        match=r"required P17 base sys\.path provenance",
    ):
        binding.verify_worker_runtime_provenance(
            binding_receipt_path=receipt_path,
            binding_receipt_file_sha256=hashlib.sha256(
                receipt_raw
            ).hexdigest(),
            p17_project_root=project_root,
            worker_project_root=worker_project_root,
            current_hardware_binding_path=tmp_path / "hardware.json",
            expected_study_id=STUDY_ID,
            runtime_fingerprint_path=tmp_path / "fingerprint.json",
            runtime_python=Path(sys.executable),
            local_llm_model=tmp_path / "llm",
            local_embedding_model=tmp_path / "embedding",
            effective_sys_path=(str(worker_project_root),),
        )


def test_worker_attests_p17_base_sys_path_separately_from_formal_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "p17-project"
    worker_project_root = tmp_path / "formal-project"
    p16_site = tmp_path / "p16-site"
    source_root = tmp_path / "hipporag-source"
    hippo_init = source_root / "hipporag/__init__.py"
    for path in (project_root, worker_project_root, p16_site):
        path.mkdir(parents=True)
    hippo_init.parent.mkdir(parents=True)
    hippo_init.write_bytes(b"# synthetic hipporag package\n")
    receipt = {
        "current_hardware_binding": {
            "hardware": {
                "GPUs": binding.EXPECTED_GPU_ROWS,
                "NVIDIA_driver_version": "595.84",
                "kernel_release": "7.0.0-28-generic",
            }
        },
        "project_root": str(project_root),
        "schema": binding.SCHEMA,
    }
    receipt_path = tmp_path / "binding.json"
    receipt_raw = binding.canonical_json_bytes(receipt)
    receipt_path.write_bytes(receipt_raw)
    monkeypatch.setattr(binding, "P17_PROJECT_ROOT", project_root)
    monkeypatch.setattr(binding, "P17_P16_SITE", p16_site)
    monkeypatch.setattr(binding, "P17_HIPPORAG_SOURCE", source_root)
    monkeypatch.setattr(binding, "P17_HIPPORAG_INIT", hippo_init)
    monkeypatch.setattr(
        binding,
        "HIPPORAG_INIT_FILE_SHA256",
        hashlib.sha256(hippo_init.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        binding, "P17_RUNTIME_PYTHON", Path(sys.executable).absolute()
    )
    monkeypatch.setattr(
        binding,
        "verify_p17_reused_closure_binding",
        lambda **_kwargs: receipt,
    )
    monkeypatch.setattr(
        binding,
        "runtime_binding_acceptance_identity",
        lambda _value: "same-synthetic-runtime",
    )
    observed_inventory_paths: list[tuple[str, ...]] = []

    def _inventory(paths: object = None) -> dict[str, object]:
        assert isinstance(paths, tuple)
        observed_inventory_paths.append(paths)
        return dict(binding.EXPECTED_RUNTIME_INVENTORY)

    monkeypatch.setattr(binding, "_runtime_inventory_receipt", _inventory)
    monkeypatch.setattr(
        binding,
        "_active_distribution_version",
        lambda name, _paths: binding.EXPECTED_ACTIVE_DISTRIBUTIONS[name],
    )
    monkeypatch.setattr(
        binding,
        "_probe_current_hardware",
        lambda: receipt["current_hardware_binding"]["hardware"],
    )
    monkeypatch.setattr(
        binding.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(hippo_init)),
    )
    base_sys_path = (
        str(project_root),
        str(p16_site),
        str(source_root),
    )
    result = binding.verify_worker_runtime_provenance(
        binding_receipt_path=receipt_path,
        binding_receipt_file_sha256=hashlib.sha256(
            receipt_raw
        ).hexdigest(),
        p17_project_root=project_root,
        worker_project_root=worker_project_root,
        current_hardware_binding_path=tmp_path / "hardware.json",
        expected_study_id=STUDY_ID,
        runtime_fingerprint_path=tmp_path / "fingerprint.json",
        runtime_python=Path(sys.executable),
        local_llm_model=tmp_path / "llm",
        local_embedding_model=tmp_path / "embedding",
        effective_sys_path=(str(worker_project_root), *base_sys_path),
    )
    assert result == receipt
    assert observed_inventory_paths == [base_sys_path]


def test_worker_binding_failure_precedes_build_or_model_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        worker, "_validate_effective_environment", lambda: None
    )

    def _fail_binding(**_kwargs: object) -> None:
        events.append("binding")
        raise binding.Dstc9P17RuntimeBindingError("synthetic drift")

    def _unexpected_build(_arguments: object) -> dict[str, object]:
        events.append("build")
        return {}

    monkeypatch.setattr(
        worker, "verify_worker_runtime_provenance", _fail_binding
    )
    monkeypatch.setattr(worker, "_run_build", _unexpected_build)
    with pytest.raises(
        contract.Dstc9OfficialHippoRAGError,
        match="provenance failed",
    ):
        worker.main(
            [
                "--stage",
                "build",
                "--study-id",
                STUDY_ID,
                "--p17-project-root",
                str(tmp_path / "project"),
                "--worker-project-root",
                str(tmp_path / "formal-project"),
                "--current-hardware-binding",
                str(tmp_path / "hardware.json"),
                "--runtime-python",
                str(tmp_path / "python"),
                "--runtime-fingerprint",
                str(tmp_path / "fingerprint.json"),
                "--runtime-binding-receipt",
                str(tmp_path / "binding.json"),
                "--runtime-binding-receipt-sha256",
                "a" * 64,
                "--corpus-input",
                str(tmp_path / "corpus.json"),
                "--output",
                str(tmp_path / "output.json"),
                "--index-root",
                str(tmp_path / "index"),
                "--llm-model",
                str(tmp_path / "llm"),
                "--embedding-model",
                str(tmp_path / "embedding"),
            ]
        )
    assert events == ["binding"]


def test_adapter_worker_command_clears_environment_denies_network_and_binds_gpu0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(adapter, "_preflight_systemd_transport", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-propagate")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-propagate")

    def _run(command: list[str], **kwargs: object) -> object:
        calls.append((list(command), dict(kwargs)))
        terminal = {
            "batch_count": 2,
            "index_call_count": 0,
            "query_count": 9,
            "stage": "retrieve",
            "status": "passed",
        }
        return SimpleNamespace(
            returncode=0,
            stdout=(json.dumps(terminal) + "\n").encode("ascii"),
            stderr=b"",
        )

    monkeypatch.setattr(adapter.subprocess, "run", _run)
    runtime_python = Path(sys.executable)
    adapter._launch_worker(
        stage="retrieve",
        study_id=STUDY_ID,
        p17_project_root=tmp_path / "p17-project",
        worker_project_root=Path(__file__).parents[1],
        current_hardware_binding_path=tmp_path / "hardware.json",
        runtime_python=runtime_python,
        local_llm_model=tmp_path / "llm",
        local_embedding_model=tmp_path / "embedding",
        runtime_fingerprint_path=tmp_path / "fingerprint.json",
        runtime_binding_receipt_path=tmp_path / "binding.json",
        corpus_input=tmp_path / "corpus.json",
        query_input=tmp_path / "queries.json",
        build_receipt=tmp_path / "build.json",
        output_path=tmp_path / "output.json",
        index_root=tmp_path / "index",
        writable_root=tmp_path / "work",
        timeout_seconds=10,
        runtime_binding_receipt_sha256="a" * 64,
        expected_corpus_count=2900,
        expected_query_count=9,
    )
    assert len(calls) == 1
    command, kwargs = calls[0]
    joined = "\n".join(command)
    assert "--ignore-environment" in command
    assert "-I" in command
    assert "-X" in command
    assert (
        f"pycache_prefix={tmp_path / 'work/tmp/pycache'}"
        in command
    )
    assert command.index("-X") < command.index("-c")
    assert "-c" in command
    assert "-m" not in command
    assert "CUDA_VISIBLE_DEVICES=0" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert (
        f"PYTHONPYCACHEPREFIX={tmp_path / 'work/tmp/pycache'}"
        in command
    )
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert "TOKENIZERS_PARALLELISM=false" in command
    assert "IPAddressDeny=any" in command
    assert "RestrictAddressFamilies=AF_UNIX" in command
    assert (
        "replication_runtime.dstc9_official_hipporag_v1.worker"
        in joined
    )
    assert "--p17-project-root" in command
    assert "--worker-project-root" in command
    assert "--current-hardware-binding" in command
    assert "--runtime-fingerprint" in command
    assert "--runtime-binding-receipt" in command
    assert "--runtime-binding-receipt-sha256" in command
    assert "--runtime-attestation-receipt-sha256" not in command
    assert "OPENAI_API_KEY" not in joined
    assert "ANTHROPIC_API_KEY" not in joined
    launcher_environment = kwargs["env"]
    assert isinstance(launcher_environment, dict)
    assert "OPENAI_API_KEY" not in launcher_environment
    assert "ANTHROPIC_API_KEY" not in launcher_environment
    assert "CUDA_VISIBLE_DEVICES" not in launcher_environment
    assert kwargs["timeout"] == 10
