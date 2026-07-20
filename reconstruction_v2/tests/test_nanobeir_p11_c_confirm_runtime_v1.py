from __future__ import annotations

from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_c_confirm_runtime_v1 as runtime,
)
from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import (
    backport,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "reconstruction_v2"


def test_acquisition_and_view_binding() -> None:
    acquisition = runtime._load_acquisition(BASE)
    items = runtime.load_views(BASE, acquisition)
    assert len(items) == runtime.ITEM_COUNT
    assert [item.ordinal for item in items] == list(range(runtime.ITEM_COUNT))
    for family in runtime.acquisition.FAMILIES:
        family_items = [item for item in items if item.family == family]
        assert len(family_items) == runtime.ITEMS_PER_FAMILY
        assert [item.family_ordinal for item in family_items] == list(
            range(runtime.ITEMS_PER_FAMILY)
        )


def test_corpora_are_projected_and_identity_unique() -> None:
    corpora = runtime.load_corpora(BASE)
    assert {family: len(corpus.ids) for family, corpus in corpora.items()} == {
        "NanoClimateFEVER": 3408,
        "NanoDBPedia": 6045,
        "NanoHotpotQA": 5090,
    }
    for corpus in corpora.values():
        assert len(corpus.ids) == len(set(corpus.ids)) == len(corpus.contents)
        assert all(0 < len(text) <= 3000 for text in corpus.contents)


def _items() -> tuple[runtime.RuntimeItem, ...]:
    return tuple(
        runtime.RuntimeItem(
            ordinal=ordinal,
            family=family,
            family_ordinal=within,
            item_key=f"k{ordinal}",
            query=f"q{ordinal}",
            source_query_id=f"s{ordinal}",
        )
        for ordinal, (family, within) in enumerate(
            (family, within)
            for family in runtime.acquisition.FAMILIES
            for within in range(runtime.ITEMS_PER_FAMILY)
        )
    )


def test_primary_requires_both_comparisons_and_every_family_positive() -> None:
    items = _items()
    passed, comparisons = runtime.primary_decision(
        items=items,
        arm_scores={
            "P11": [3] * runtime.ITEM_COUNT,
            "RAW": [2] * runtime.ITEM_COUNT,
            "HippoRAG": [1] * runtime.ITEM_COUNT,
        },
    )
    assert passed is True
    assert all(
        value > 0
        for row in comparisons.values()
        for value in row["family_net_integer_ndcg"].values()
    )


def test_primary_rejects_aggregate_positive_with_one_negative_family() -> None:
    items = _items()
    p11 = [10] * runtime.ITEM_COUNT
    raw = [0] * runtime.ITEM_COUNT
    for index, item in enumerate(items):
        if item.family == "NanoClimateFEVER":
            raw[index] = 11
    passed, comparisons = runtime.primary_decision(
        items=items,
        arm_scores={"P11": p11, "RAW": raw, "HippoRAG": [0] * runtime.ITEM_COUNT},
    )
    assert comparisons["P11_minus_RAW"]["net_integer_ndcg"] > 0
    assert comparisons["P11_minus_RAW"]["family_net_integer_ndcg"]["NanoClimateFEVER"] < 0
    assert passed is False


def test_label_loader_fails_before_any_label_read(tmp_path: Path) -> None:
    acquisition = runtime._load_acquisition(BASE)
    items = runtime.load_views(BASE, acquisition)
    missing = tmp_path / "missing.actions.json"
    with pytest.raises(runtime.NanoBEIRCConfirmError, match="action seal"):
        runtime.load_labels_after_action_seal(
            base=BASE,
            acquisition_result=acquisition,
            items=items,
            action_path=missing,
            expected_action_sha256="0" * 64,
        )


def test_hardened_source_materialization_is_exact(tmp_path: Path) -> None:
    path = runtime._materialize_hardened_source(BASE, tmp_path)
    assert runtime.acquisition.file_sha256(path) == backport.PATCHED_SOURCE_SHA256
    baseline = (
        BASE
        / runtime.hardening_qualification.BASELINE_REPO_RELATIVE
        / runtime.hardening_qualification.BASELINE_SOURCE_WITHIN_REPO
    )
    assert path.read_bytes() == backport.apply_fixed_backport(baseline.read_bytes())


def test_paired_counts() -> None:
    observed = runtime._paired([3, 2, 1, 0], [2, 2, 2, 0])
    assert observed == {
        "gain": 1,
        "harm": 1,
        "net_integer_ndcg": 0,
        "tie": 2,
    }


def test_formal_is_one_shot_before_private_view_access(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = project / "reconstruction_v2" / runtime.RUN_ROOT_RELATIVE
    root.mkdir(parents=True)
    with pytest.raises(runtime.OneShotRefusal):
        runtime.run_formal(project)
