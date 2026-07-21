from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_c_confirm_runtime_v1 as runtime,
)


def test_slot_rename_targets_p13_without_mutating_candidate_name() -> None:
    value = {
        "P11_rows": [1],
        "candidate": runtime.CANDIDATE_NAME,
        "status": "same_source_P11_stops",
    }
    assert runtime._rename_p11_slot_to_p13(value) == {
        "P13_rows": [1],
        "candidate": runtime.CANDIDATE_NAME,
        "status": "same_source_P13_stops",
    }


def test_p13_totalizer_avoids_recursive_p12_dispatch() -> None:
    original = runtime.p12_runtime.totalize_qwen_output
    wrapped = runtime._p13_totalizer(original)
    output = {
        "items": [
            {
                "completion_sha256": "a" * 64,
                "completion_token_count": 1,
                "expansions": [],
                "generation_valid": False,
                "ordinal": 0,
            }
        ],
        "schema": "bright_query_generator_v1_output",
    }

    class Item:
        query = "q" * 1000

    runtime.p12_runtime.totalize_qwen_output = wrapped
    try:
        projected, audit = wrapped(output, [Item()])
    finally:
        runtime.p12_runtime.totalize_qwen_output = original
    assert projected["items"][0]["expansions"][1].startswith("relation: ")
    assert audit["candidate"] == runtime.CANDIDATE_NAME


def test_shared_source_filter_loads_expected_corpora() -> None:
    base = Path(__file__).resolve().parents[1]
    corpora = runtime.load_corpora(base)
    assert {family: len(value.ids) for family, value in corpora.items()} == {
        "NanoFiQA2018": 4571,
        "NanoNFCorpus": 2953,
        "NanoTouche2020": 5745,
    }


def test_outer_compatibility_context_restores_mature_runtime() -> None:
    original_acquisition = runtime.mature.acquisition
    original_loader = runtime.p11_runtime.load_corpora
    original_totalizer = runtime.p12_runtime.totalize_qwen_output
    with runtime._patched_mature_runtime():
        assert runtime.mature.acquisition is runtime.acquisition
        assert runtime.p11_runtime.load_corpora is runtime.load_corpora
        assert runtime.p12_runtime.totalize_qwen_output is not original_totalizer
    assert runtime.mature.acquisition is original_acquisition
    assert runtime.p11_runtime.load_corpora is original_loader
    assert runtime.p12_runtime.totalize_qwen_output is original_totalizer


def test_formal_refuses_consumed_root_before_private_access(
    tmp_path: Path,
) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / runtime.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(runtime.OneShotRefusal, match="root already exists"):
        runtime.run_formal(tmp_path)
