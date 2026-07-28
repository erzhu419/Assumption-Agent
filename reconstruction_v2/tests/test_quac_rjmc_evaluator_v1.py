from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from assumption_agent.benchmarks import quac_rjmc_evaluator_v1 as core


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "qualify_quac_rjmc_source_free_v1.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qualify_quac_rjmc_source_free_v1_for_test", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)


def _item(topology: str = "pair_complement") -> core.ListwiseTrainingItem:
    item, _required, _topology = qualification.make_item(
        item_id=f"test-{topology}",
        topology=topology,
        component=0,
        perturbation=0.019,
    )
    return item


def test_complete_state_space_is_raw_plus_every_one_and_two_replacement() -> None:
    item = _item()
    states = item.states
    assert len(states) == core.complete_state_count(3) == 46
    assert states[0].replacements == 0
    assert sum(state.replacements == 1 for state in states) == 15
    assert sum(state.replacements == 2 for state in states) == 30
    assert len({state.unit_ids for state in states}) == len(states)
    assert states[1:] == tuple(
        sorted(states[1:], key=lambda row: (row.replacements, row.unit_ids))
    )

    assert "candidates" not in inspect.signature(
        core.enumerate_complete_states
    ).parameters
    with pytest.raises(TypeError, match="candidates"):
        core.enumerate_complete_states(
            item.graph,
            raw_top5=item.raw_top5,
            candidates=("u00", "u05", "u06"),
        )

    five = core.RelationalGraph(units=item.graph.units[:5], edges=())
    six = core.RelationalGraph(units=item.graph.units[:6], edges=())
    assert len(core.enumerate_complete_states(five, raw_top5=item.raw_top5)) == 1
    assert len(core.enumerate_complete_states(six, raw_top5=item.raw_top5)) == 6
    assert core.complete_state_count(0) == 1
    assert core.complete_state_count(1) == 6


def test_comparator_is_antisymmetric_permutation_invariant_and_raw_zero() -> None:
    item = _item()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991)
        model = core.RelationalSetComparator(width=8)
    left = item.states[-1].unit_ids
    raw = item.raw_top5
    forward = core.compare_sets(model, item.graph, left=left, right=raw)
    reverse = core.compare_sets(model, item.graph, left=raw, right=left)
    permuted = core.compare_sets(
        model,
        item.graph,
        left=tuple(reversed(left)),
        right=tuple(reversed(raw)),
    )
    assert forward == -reverse
    assert forward == permuted
    assert core.compare_sets(model, item.graph, left=raw, right=raw) == 0.0


def test_evaluator_input_contract_has_no_gold_family_split_or_hipporag_feature() -> None:
    forbidden = {
        "gold",
        "required",
        "answer",
        "family",
        "split",
        "hipporag",
        "qrel",
    }
    assert not forbidden.intersection(core.EvidenceUnit.__dataclass_fields__)
    assert not forbidden.intersection(core.RelationalGraph.__dataclass_fields__)
    for function in (
        core.compare_sets,
        core.score_states,
        core.select_e0_proof_coverage,
        core.fit_component_jackknife,
    ):
        assert not forbidden.intersection(
            name.casefold() for name in inspect.signature(function).parameters
        )
    for function in (
        core.score_states,
        core.select_e0_proof_coverage,
        core.JackknifeMinimaxComparator.score_matrix,
        core.JackknifeMinimaxComparator.minimax_scores,
        core.JackknifeMinimaxComparator.select,
    ):
        parameters = inspect.signature(function).parameters
        assert "states" not in parameters
        assert "candidates" not in parameters
    assert "candidates" not in core.ListwiseTrainingItem.__dataclass_fields__


def test_large_candidate_space_scores_in_bounded_batches(monkeypatch) -> None:
    units = tuple(
        core.EvidenceUnit(
            unit_id=f"u{ordinal:02d}",
            node_features=(
                (ordinal % 7) / 7,
                (ordinal % 5) / 5,
                (ordinal % 3) / 3,
                (ordinal % 11) / 11,
            ),
            dialogue_facets=tuple(
                int((ordinal + facet) % 4 == 0) for facet in range(4)
            ),
        )
        for ordinal in range(37)
    )
    graph = core.RelationalGraph(units=units, edges=())
    raw = tuple(f"u{ordinal:02d}" for ordinal in range(5))
    assert len(core.enumerate_complete_states(graph, raw_top5=raw)) == 5121

    observed_batch_sizes: list[int] = []
    original = core._compile_comparisons

    def bounded_compile(graph, *, left_sets, right_set):
        observed_batch_sizes.append(len(left_sets))
        return original(graph, left_sets=left_sets, right_set=right_set)

    monkeypatch.setattr(core, "_compile_comparisons", bounded_compile)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1441)
        model = core.RelationalSetComparator(width=4)
    scores = core.score_states(
        model, graph, raw_top5=raw, state_batch_size=64
    )
    assert scores.shape == (5121,)
    assert scores[0] == 0.0
    assert max(observed_batch_sizes) <= 64
    assert len(observed_batch_sizes) > 1
    assert 0 <= core.select_e0_proof_coverage(graph, raw_top5=raw) < 5121


def test_jackknife_fit_streams_each_item_in_bounded_batches(monkeypatch) -> None:
    rows = []
    for component in range(core.COMPONENT_COUNT):
        for replicate in range(2):
            item, _required, _topology = qualification.make_item(
                item_id=f"stream-c{component}-r{replicate}",
                topology="pair_complement",
                component=component,
                perturbation=0.003 * component + 0.001 * replicate,
            )
            rows.append(item)

    observed_batch_sizes: list[int] = []
    original = core._compile_comparisons

    def bounded_compile(graph, *, left_sets, right_set):
        observed_batch_sizes.append(len(left_sets))
        return original(graph, left_sets=left_sets, right_set=right_set)

    monkeypatch.setattr(core, "_compile_comparisons", bounded_compile)
    ensemble = core.fit_component_jackknife(
        rows,
        config=core.FitConfig(epochs=1, state_batch_size=7),
    )
    assert len(ensemble.heads) == 5
    assert observed_batch_sizes
    assert max(observed_batch_sizes) <= 7


def test_source_free_development_qualification_passes_all_four_topologies() -> None:
    receipt = qualification.qualify()
    assert (
        receipt["status"]
        == "passed_nonformal_source_free_development_qualification"
    )
    assert receipt["formal_result"] is False
    assert receipt["component_jackknife_head_count"] == 5
    assert receipt["complete_state_count_for_three_candidates"] == 46
    assert receipt["antisymmetric"] is True
    assert receipt["permutation_invariant"] is True
    assert receipt["RAW_structural_zero"] is True
    assert receipt["same_process_repeat_exact"] is True
    assert receipt["QuAC_source_payload_access_count"] == 0
    assert receipt["prior_private_source_access_count"] == 0
    assert receipt["online_or_API_evaluation_count"] == 0
    for block_name in ("A_hold", "M_search"):
        block = receipt[block_name]
        assert block["E1_minus_E0"] > 0
        assert all(
            block["topology_delta"][topology] > 0
            for topology in qualification.TOPOLOGIES
        )
        assert block["topology_raw_harm"]["retention_trap"] == 0
        assert block["topology_raw_harm"]["null_shift"] == 0
        assert block["topology_required_complete"]["pair_complement"] is True
    assert receipt["A_hold"]["promotion_passed"] is True
    assert (
        receipt["M_search"]["structural_variant"]
        == "extra_distractor_and_two_new_edges"
    )
