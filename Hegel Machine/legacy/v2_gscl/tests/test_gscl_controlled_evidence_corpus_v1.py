from dataclasses import replace
import hashlib

import pytest

from assumption_agent.benchmarks.gscl_controlled_evidence_corpus_v1 import (
    ControlledView,
    JSONL_MEDIA_TYPE,
    PAIRED_CONTROL_ROOT_COUNT,
    PAIRED_NEGATIVE_ROOT_COUNT,
    PRIMARY_ROOT_COUNT,
    ROOT_COUNT,
    VIEW_COUNT,
    atomic_record_diff_paths,
    build_controlled_roots,
    controlled_view_gold_linkage,
    gold_pack_contract,
    raw_pack_contract,
    render_controlled_views,
    validate_no_runtime_answer_leak,
    validate_pair_operator_receipts,
)


def test_paired_case_composition_and_operator_receipts() -> None:
    roots = build_controlled_roots()
    assert len(roots) == ROOT_COUNT == 25
    assert (
        sum(root.pair_role == "primary" for root in roots)
        == PRIMARY_ROOT_COUNT
        == 10
    )
    assert (
        sum(
            root.pair_role == "counterfactual_negative"
            for root in roots
        )
        == PAIRED_NEGATIVE_ROOT_COUNT
        == 10
    )
    assert (
        sum(
            root.pair_role == "missingness_control"
            for root in roots
        )
        == PAIRED_CONTROL_ROOT_COUNT
        == 5
    )
    assert validate_pair_operator_receipts(roots) == ()
    roots_by_id = {root.root_id: root for root in roots}
    for root in roots:
        assert all(record["kind"] != "note" for record in root.records)
        if root.pair_role == "primary":
            continue
        primary = roots_by_id[root.paired_primary_root_id]
        receipt = root.operator_receipt
        assert receipt is not None
        assert (
            atomic_record_diff_paths(primary.records, root.records)
            == receipt.allowed_diff_paths
            == receipt.observed_diff_paths
        )
        assert receipt.non_target_fields_unchanged


def test_raw_records_are_atomic_law_agnostic_and_reuse_primitives() -> None:
    roots = build_controlled_roots()
    allowed_kinds = {
        "assertion",
        "edge",
        "map",
        "node",
        "observation",
        "subset_outcome",
        "transfer",
    }
    laws_by_kind: dict[str, set[str]] = {}
    for root in roots:
        for record in root.records:
            assert record["kind"] in allowed_kinds
            assert "gscl" not in str(record).lower()
            laws_by_kind.setdefault(record["kind"], set()).add(
                root.law_id
            )
    assert len(laws_by_kind["node"]) == 5
    assert len(laws_by_kind["assertion"]) == 5
    assert len(laws_by_kind["map"]) == 5
    assert sum(
        any(record["id"] == "map.decoy" for record in root.records)
        for root in roots
    ) >= 4


def test_four_views_are_serialization_aliases_and_leak_free() -> None:
    roots = build_controlled_roots()
    views = render_controlled_views(roots)
    links = controlled_view_gold_linkage(roots)
    assert len(views) == VIEW_COUNT == 100
    assert {
        row.view_kind for row in links
    } == {
        "json_canonical",
        "json_alias",
        "line_canonical",
        "line_alias",
    }
    assert validate_no_runtime_answer_leak(views) == ()
    assert len(links) == len(views)
    assert {row.item_id for row in links} == {
        row.item_id for row in views
    }
    contract = raw_pack_contract(views)
    assert contract["case_count_semantics"] == "paired_not_independent"
    assert contract["natural_language_paraphrase_claimed"] is False
    assert all(
        set(row)
        == {
            "item_id",
            "media_type",
            "source_sha256",
            "source_size",
            "record_count",
            "record_spans_commitment",
        }
        for row in contract["rows"]
    )
    assert all(
        "root_id" not in row
        and "pair_id" not in row
        and "view_kind" not in row
        for row in contract["rows"]
    )
    assert all(
        "record_spans_commitment" in row for row in contract["rows"]
    )


def test_raw_contract_recomputes_source_hash_and_detects_tamper() -> None:
    view = render_controlled_views(build_controlled_roots())[0]
    original_hash = raw_pack_contract([view])["raw_pack_hash"]
    changed_bytes = view.source_bytes + b"\n"
    stale = replace(view, source_bytes=changed_bytes)
    with pytest.raises(ValueError, match="declared source hash mismatch"):
        raw_pack_contract([stale])
    re_declared = replace(
        stale,
        source_sha256=hashlib.sha256(changed_bytes).hexdigest(),
    )
    with pytest.raises(ValueError, match="opaque item id mismatch"):
        raw_pack_contract([re_declared])
    reidentified = replace(
        re_declared,
        item_id=(
            "item."
            + hashlib.sha256(
                re_declared.media_type.encode("utf-8")
                + b"\0"
                + changed_bytes
            ).hexdigest()
        ),
    )
    assert raw_pack_contract([reidentified])["raw_pack_hash"] != original_hash


def test_leak_scanner_decodes_json_unicode_escapes() -> None:
    source = (
        b'{"kind":"assertion","id":"assertion.test",'
        b'"attrs":{"value":"input\\u005fbefore"}}\n'
    )
    source_hash = hashlib.sha256(source).hexdigest()
    line_length = len(source.rstrip(b"\n"))
    view = ControlledView(
        item_id="item.input_before",
        media_type=JSONL_MEDIA_TYPE,
        source_bytes=source,
        source_sha256=source_hash,
        record_spans={"assertion.test": (0, line_length)},
    )
    assert "raw_answer_fragment_leak.input_before" in (
        validate_no_runtime_answer_leak([view])
    )


def test_raw_schema_signatures_do_not_name_a_law_family() -> None:
    roots = build_controlled_roots()
    laws_by_signature: dict[tuple[object, ...], set[str]] = {}
    for root in roots:
        for record in root.records:
            attrs = record["attrs"]
            signature = (
                record["kind"],
                tuple(sorted(attrs)),
                attrs.get("mode"),
                attrs.get("predicate"),
            )
            laws_by_signature.setdefault(signature, set()).add(
                root.law_id
            )
    assert laws_by_signature
    assert all(
        len(law_ids) >= 2
        for law_ids in laws_by_signature.values()
    )


def test_operator_receipt_tamper_cannot_override_frozen_specs() -> None:
    roots = build_controlled_roots()
    paired = next(
        root for root in roots if root.pair_role != "primary"
    )
    assert paired.operator_receipt is not None
    bad_receipt = replace(
        paired.operator_receipt,
        allowed_diff_paths=("/records/unfrozen",),
        observed_diff_paths=("/records/unfrozen",),
        operator_implementation_sha256="0" * 64,
        frozen_target_spec_hash="0" * 64,
    )
    tampered = replace(paired, operator_receipt=bad_receipt)
    tampered_roots = tuple(
        tampered if row.root_id == paired.root_id else row
        for row in roots
    )
    assert any(
        issue.startswith("operator_receipt_invalid.")
        for issue in validate_pair_operator_receipts(tampered_roots)
    )


def test_gold_contract_commits_pair_topology_and_operator_contract() -> None:
    contract = gold_pack_contract(build_controlled_roots())
    assert contract["case_count"] == 25
    assert contract["primary_root_count"] == 10
    assert contract["paired_negative_count"] == 10
    assert contract["paired_missingness_control_count"] == 5
    assert len(contract["pair_rows"]) == 15
    assert len(contract["operator_contract_hash"]) == 64
    assert len(contract["gold_pack_hash"]) == 64
