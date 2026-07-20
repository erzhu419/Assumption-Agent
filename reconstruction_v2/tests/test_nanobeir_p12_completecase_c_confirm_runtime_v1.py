import hashlib
from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_c_confirm_runtime_v1 as runtime,
)


def test_pack_schema_adapter_maps_only_completecase_envelopes() -> None:
    def original(_base, _binding, _name):
        return {"schema": runtime.acquisition.VIEW_SCHEMA, "value": 1}

    value = runtime._replace_pack_schema(original, Path("/tmp"), {}, "view")
    assert value == {"schema": "nanobeir_p11_private_view_v1", "value": 1}


def test_cached_runner_reuses_exact_bytes_without_entering_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"frozen-output")
    item_root = tmp_path / "item_000"
    item_root.mkdir()
    row = {
        "base_pool": list(range(32)),
        "graph_edge_count": 9,
        "graph_node_count": 41,
        "source_output_file_sha256": hashlib.sha256(b"frozen-output").hexdigest(),
        "source_output_relative_path": "source.json",
        "source_screen_ordinal": 17,
        "source_stderr_sha256": "e" * 64,
        "source_stdout_sha256": "a" * 64,
        "top_rows": list(range(10)),
    }
    monkeypatch.setattr(
        runtime.p11_runtime.train.hippo_contract,
        "parse_output",
        lambda _payload: {
            "graph_edge_count": 9,
            "graph_node_count": 41,
            "top_ordinals": list(range(10)),
        },
    )

    class Counter:
        def enter(self):
            raise AssertionError("counter must not be entered")

        def leave(self):
            raise AssertionError("counter must not be left")

    runner = runtime.cached_hipporag_runner(base=tmp_path, cached={0: row})
    result = runner(
        base=tmp_path,
        item_root=item_root,
        candidate_rows=list(range(32)),
        patched_source=tmp_path / "unused.py",
        semaphore=object(),
        counter=Counter(),
    )
    assert result["comparator_relaunch_count"] == 0
    assert result["top_rows"] == list(range(10))
    assert (item_root / "reused.screen.output.json").is_file()


def test_cached_runner_rejects_recomputed_base_pool_drift(tmp_path: Path) -> None:
    item_root = tmp_path / "item_000"
    item_root.mkdir()
    runner = runtime.cached_hipporag_runner(
        base=tmp_path,
        cached={0: {"base_pool": list(range(32))}},
    )
    with pytest.raises(
        runtime.p11_runtime.NanoBEIRCConfirmError,
        match="base pool drifted",
    ):
        runner(
            base=tmp_path,
            item_root=item_root,
            candidate_rows=list(reversed(range(32))),
            patched_source=tmp_path / "unused.py",
            semaphore=object(),
            counter=object(),
        )


def test_formal_refuses_consumed_root_before_private_access(tmp_path: Path) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / runtime.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(runtime.OneShotRefusal, match="root already exists"):
        runtime.run_formal(tmp_path)


def test_runtime_item_counts_match_frozen_design() -> None:
    assert runtime.ITEMS_PER_FAMILY == 10
    assert runtime.ITEM_COUNT == 30


def test_nested_mature_controller_patch_is_compatible_and_restored(
    tmp_path: Path,
) -> None:
    base = tmp_path / "reconstruction_v2"
    path = base / runtime.ACQUISITION_RESULT_RELATIVE
    path.parent.mkdir(parents=True)
    acquisition_result = runtime.acquisition.self_hashed(
        {"schema": runtime.acquisition.SCHEMA}
    )
    path.write_bytes(runtime.acquisition.canonical_json_bytes(acquisition_result))
    original_count = runtime.p11_runtime.ITEM_COUNT
    with runtime._patched_runtime(
        base=base, acquisition_result=acquisition_result, cached={}
    ):
        assert runtime.p11_runtime.ITEM_COUNT == 30
        with runtime.p12_runtime._patched_controller():
            assert runtime.p11_runtime.acquisition is runtime.acquisition
    assert runtime.p11_runtime.ITEM_COUNT == original_count
