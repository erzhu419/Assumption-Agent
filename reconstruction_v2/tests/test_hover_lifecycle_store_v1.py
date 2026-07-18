from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from assumption_agent.benchmarks import hover_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks import hover_lifecycle_store_v1 as store
from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
    ActionTrace,
    CausalSignature,
    CoverageSignature,
    INTEGER_SCALE,
    recompute_action_trace_sha256,
)


ZERO = "0" * 64
SENTINEL_CONTENT = "synthetic committed acquisition only; no official assets\n"


def _raw(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"


def _write(path: Path, value: object, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_raw(value))
    path.chmod(mode)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _synthetic_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / store._SYNTHETIC_SENTINEL).write_text(SENTINEL_CONTENT, encoding="ascii")
    corpus = {
        "schema": "synthetic_corpus",
        "articles": [
            {"article_id": index, "title": f"t{index}", "body": f"b{index}"}
            for index in range(store.CORPUS_SIZE)
        ],
    }
    corpus["corpus_view_sha256"] = store.stable_hash(corpus)
    _write(root / "private/corpus.json", corpus, 0o600)
    views: dict[str, dict[str, Any]] = {}
    for block in store.BLOCK_ORDER:
        view = {
            "schema": "synthetic_claim_view",
            "block": block,
            "item_count": store.BLOCK_COUNTS[block],
            "items": [
                {
                    "schema": "synthetic_claim",
                    "block": block,
                    "ordinal": ordinal,
                    "claim": f"{block} synthetic claim {ordinal}",
                }
                for ordinal in range(store.BLOCK_COUNTS[block])
            ],
        }
        view["block_view_sha256"] = store.stable_hash(view)
        views[block] = view
        _write(root / f"private/{block}.json", view, 0o600)
    late_labels: dict[str, dict[str, Any]] = {}
    for block in ("A_hold", "M_search"):
        label_items = []
        for ordinal, view_item in enumerate(views[block]["items"]):
            hop = 2 + ordinal // 10
            label_items.append(
                {
                    "schema": acquisition.LABEL_ITEM_SCHEMA,
                    "block": block,
                    "ordinal": ordinal,
                    "view_sha256": store.stable_hash(view_item),
                    "identity_commitment_sha256": hashlib.sha256(
                        f"{block}:identity:{ordinal}".encode("ascii")
                    ).hexdigest(),
                    "source_record_commitment_sha256": hashlib.sha256(
                        f"{block}:source:{ordinal}".encode("ascii")
                    ).hexdigest(),
                    "hop_stratum": f"{hop}_hop",
                    "gold_article_ids": list(range(5, 5 + hop)),
                }
            )
        labels = acquisition.with_self_hash(
            {
                "schema": acquisition.BLOCK_LABEL_SCHEMA,
                "version": acquisition.VERSION,
                "block": block,
                "item_count": store.BLOCK_COUNTS[block],
                "source_or_verdict_payload_included": False,
                "items": label_items,
            },
            "block_labels_sha256",
        )
        acquisition.validate_block_labels(labels, expected_block=block)
        late_labels[block] = labels
        _write(root / f"private/{block}.labels.json", labels, 0o600)
    receipt_body = {
        "schema": "synthetic_acquisition_receipt",
        "status": "committed_private_pack",
        "corpus_semantic_sha256": corpus["corpus_view_sha256"],
        "block_view_semantic_sha256s": {
            block: view["block_view_sha256"] for block, view in views.items()
        },
        "late_label_semantic_sha256s": {
            block: labels["block_labels_sha256"]
            for block, labels in late_labels.items()
        },
        "F_search_label_pack_created": False,
    }
    receipt = {
        **receipt_body,
        "acquisition_sha256": store.stable_hash(receipt_body),
    }
    _write(root / "synthetic_acquisition_receipt.json", receipt, 0o644)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "add", "synthetic_acquisition_receipt.json", "private")
    _git(root, "commit", "-qm", "synthetic acquisition freeze")

    def load_context(*, project: Path, block: str) -> store.StageAcquisitionContext:
        actual = Path(project).resolve()
        assert actual == root.resolve()
        assert (actual / store._SYNTHETIC_SENTINEL).read_text(
            encoding="ascii"
        ) == SENTINEL_CONTENT
        head = _git(actual, "rev-parse", "HEAD")
        receipt_path = actual / "synthetic_acquisition_receipt.json"
        receipt_raw = receipt_path.read_bytes()
        assert _git(
            actual, "show", "HEAD:synthetic_acquisition_receipt.json"
        ).encode() == receipt_raw.rstrip(b"\n")
        blob = _git(actual, "rev-parse", "HEAD:synthetic_acquisition_receipt.json")
        current_receipt = json.loads(receipt_raw)
        corpus_raw = (actual / "private/corpus.json").read_bytes()
        current_corpus = json.loads(corpus_raw)
        view_raw = (actual / f"private/{block}.json").read_bytes()
        current_view = json.loads(view_raw)
        late_label_payload = None
        if block in {"A_hold", "M_search"}:
            late_label_payload = json.loads(
                (actual / f"private/{block}.labels.json").read_bytes()
            )
        return store.StageAcquisitionContext(
            acquisition_sha256=current_receipt["acquisition_sha256"],
            acquisition_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            acquisition_git_head=head,
            acquisition_git_blob_sha1=blob,
            corpus_file_sha256=hashlib.sha256(corpus_raw).hexdigest(),
            corpus_semantic_sha256=current_corpus["corpus_view_sha256"],
            block=block,
            view_file_sha256=hashlib.sha256(view_raw).hexdigest(),
            view_semantic_sha256=current_view["block_view_sha256"],
            view_items=tuple(current_view["items"]),
            f_search_label_pack_created=current_receipt[
                "F_search_label_pack_created"
            ],
            late_labels=late_label_payload,
        )

    # This is the only production seam replaced in synthetic tests.  It is
    # guarded by a root-local sentinel and verifies a real committed receipt.
    monkeypatch.setattr(store, "_load_stage_acquisition_context", load_context)
    return root


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _relevance() -> tuple[int, ...]:
    return tuple(store.CORPUS_SIZE - index for index in range(store.CORPUS_SIZE))


def _trace(
    *, block: str, ordinal: int, action_id: str, action_index: int, relevance_sha: str
) -> ActionTrace:
    claim = f"{block} synthetic claim {ordinal}"
    query_sha = hashlib.sha256(claim.casefold().encode("utf-8")).hexdigest()
    start = action_index * 5
    output = tuple(range(start, start + 5))
    necessary = 4 if action_index == 1 else 0
    e0 = (10 - action_index, 0, 0)
    causal = CausalSignature(
        necessary_count=necessary,
        necessary_fraction=Fraction(necessary, 4),
        minimum_leave_one_out_loss=Fraction(0),
        minimum_replacement_loss=Fraction(0),
        path_connectivity=Fraction(0),
    )
    trace = ActionTrace(
        action_id=action_id,
        output_top5=output,  # type: ignore[arg-type]
        core=output[:4],  # type: ignore[arg-type]
        core_quality=(0,),
        coverage=CoverageSignature(
            covered=1,
            total=1,
            value=Fraction(1),
            slot_keys=("synthetic-slot",),
            covered_slot_keys=("synthetic-slot",),
        ),
        causal=causal,
        e0_key=e0,
        e1_key=(
            causal.necessary_fraction,
            causal.minimum_leave_one_out_loss,
            causal.path_connectivity,
            *e0,
        ),
        ordered_pair_scan_count=store.CORPUS_SIZE * (store.CORPUS_SIZE - 1),
        extension_scan_count=(store.CORPUS_SIZE - 2) + (store.CORPUS_SIZE - 3),
        graph_sha256=_sha("graph"),
        plan_sha256=_sha(f"plan:{block}:{ordinal}"),
        query_sha256=query_sha,
        relevance_sha256=relevance_sha,
        trace_sha256=ZERO,
    )
    return replace(trace, trace_sha256=recompute_action_trace_sha256(trace))


def _record(block: str, ordinal: int) -> dict[str, Any]:
    relevance = _relevance()
    relevance_sha = store.stable_hash(
        {"integer_scale": INTEGER_SCALE, "values": list(relevance)}
    )
    return store.build_stage_output_record(
        block=block,
        ordinal=ordinal,
        view_sha256=store.stable_hash(
            {
                "schema": "synthetic_claim",
                "block": block,
                "ordinal": ordinal,
                "claim": f"{block} synthetic claim {ordinal}",
            }
        ),
        dense_relevance_ints=relevance,
        raw_top5=(0, 1, 2, 3, 4),
        hipporag_top5=(20, 21, 22, 23, 24),
        action_traces=tuple(
            _trace(
                block=block,
                ordinal=ordinal,
                action_id=action_id,
                action_index=action_index,
                relevance_sha=relevance_sha,
            )
            for action_index, action_id in enumerate(store.AGENT_ACTION_IDS)
        ),
    )


def _records(block: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        _record(block, ordinal)
        for ordinal in range(store.BLOCK_COUNTS[block])
    )


def _runtime() -> dict[str, str]:
    return {
        field: _sha(field if field != "graph_sha256" else "graph")
        for field in store.STAGE_RUNTIME_BINDING_KEYS
    }


def _write_archive(root: Path, block: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return store.create_stage_output_archive_once(
        project=root,
        block=block,
        records=_records(block),
        stage_runtime_binding=_runtime(),
    )


def _freeze_a_form(root: Path) -> dict[str, Any]:
    archive, _ = store.load_stage_output_archive(project=root, block="A_form")
    e0, e1, _ = store._recompute_policies(archive, block="A_form")
    return store.create_a_form_evaluator_freeze_once(
        project=root, e0_policy=e0, e1_policy=e1
    )


def _freeze_f(root: Path) -> dict[str, Any]:
    archive, _ = store.load_stage_output_archive(project=root, block="F_search")
    e0, e1, _ = store._recompute_policies(archive, block="F_search")
    return store.create_f_search_policy_freeze_once(
        project=root, e0_policy=e0, e1_policy=e1
    )


def test_read_only_preflight_and_complete_record_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_project(tmp_path, monkeypatch)
    before = {path.relative_to(root) for path in root.rglob("*")}
    paths = store.preflight_lifecycle_outputs_absent(root)
    assert len(paths) == 10
    assert before == {path.relative_to(root) for path in root.rglob("*")}
    record = _record("A_hold", 0)
    assert len(record["dense_relevance_ints"]) == 609
    assert len(record["agent_action_traces"]) == 6
    broken = dict(record)
    broken["raw_output"] = dict(record["raw_output"])
    broken["raw_output"]["output_top5"] = [1, 0, 2, 3, 4]
    with pytest.raises(store.HoVerLifecycleStoreError):
        store.validate_stage_output_record(
            record=broken,
            block="A_hold",
            ordinal=0,
            view_sha256=record["view_sha256"],
        )


def test_full_ordered_lifecycle_and_replay_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_project(tmp_path, monkeypatch)
    _write_archive(root, "A_form")
    archive, _ = store.load_stage_output_archive(project=root, block="A_form")
    e0, e1, _ = store._recompute_policies(archive, block="A_form")
    with pytest.raises(store.HoVerLifecycleStoreError):
        store.create_a_form_evaluator_freeze_once(
            project=root, e0_policy=e0, e1_policy=e1
        )
    store.create_action_seal_once(project=root, block="A_form")
    freeze_a = store.create_a_form_evaluator_freeze_once(
        project=root, e0_policy=e0, e1_policy=e1
    )
    assert freeze_a["selection_purpose"] == "diagnostic_only_not_F_policy"

    _write_archive(root, "F_search")
    freeze_f = _freeze_f(root)
    assert freeze_f["e0_action_id"] != freeze_f["e1_action_id"]
    _write_archive(root, "A_hold")
    seal = store.create_action_seal_once(project=root, block="A_hold")
    exact_report = store.recompute_a_hold_outcome_report(project=root)
    assert exact_report["promoted"] is True
    assert exact_report["primary_passed"] is False
    plausible_but_false = json.loads(json.dumps(exact_report))
    plausible_but_false["promotion_delta_total"] = [59, 1]
    with pytest.raises(store.HoVerLifecycleStoreError, match="differs"):
        store.create_a_hold_promotion_once(
            project=root, outcome_report=plausible_but_false
        )
    assert not (root / store.PROMOTION_RELATIVE).exists()
    with pytest.raises(store.HoVerLifecycleStoreError):
        store.create_stage_output_archive_once(
            project=root,
            block="M_search",
            records=(),
            stage_runtime_binding=_runtime(),
        )
    promotion = store.create_a_hold_promotion_once(
        project=root, outcome_report=exact_report
    )
    assert promotion["a_hold_action_seal_sha256"] == seal["action_seal_sha256"]
    _write_archive(root, "M_search")
    store.create_action_seal_once(project=root, block="M_search")
    m_report = store.recompute_m_search_outcome_report(project=root)
    assert m_report["l5_passed"] is True
    assert m_report["l5_delta_total"] == [60, 1]
    assert store.validate_m_search_outcome_report(
        project=root, outcome_report=m_report, l5_passed=True
    ) == m_report
    plausible_false_m = json.loads(json.dumps(m_report))
    plausible_false_m["e1_minus_hippo_delta_total"] = [59, 1]
    with pytest.raises(store.HoVerLifecycleStoreError, match="differs"):
        store.validate_m_search_outcome_report(
            project=root,
            outcome_report=plausible_false_m,
            l5_passed=True,
        )
    with pytest.raises(store.HoVerLifecycleStoreError, match="differs"):
        store.validate_m_search_outcome_report(
            project=root, outcome_report=m_report, l5_passed=False
        )

    with pytest.raises(store.HoVerLifecycleStoreError, match="already exists"):
        store.create_a_hold_promotion_once(
            project=root, outcome_report=exact_report
        )
    with pytest.raises(store.HoVerLifecycleStoreError, match="existing paths"):
        store.preflight_lifecycle_outputs_absent(root)


def test_cross_binding_and_canonical_policy_evidence_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_project(tmp_path, monkeypatch)
    _write_archive(root, "A_form")
    store.create_action_seal_once(project=root, block="A_form")
    archive, _ = store.load_stage_output_archive(project=root, block="A_form")
    e0, e1, _ = store._recompute_policies(archive, block="A_form")
    wrong_e0 = replace(e0, action_id=e1.action_id)
    with pytest.raises(store.HoVerLifecycleStoreError):
        store.create_a_form_evaluator_freeze_once(
            project=root, e0_policy=wrong_e0, e1_policy=e1
        )
    _freeze_a_form(root)

    path = root / store.STAGE_OUTPUT_ARCHIVE_RELATIVES["A_form"]
    payload = json.loads(path.read_bytes())
    payload["block_view_semantic_sha256"] = _sha("wrong-view")
    body = dict(payload)
    del body["stage_output_archive_sha256"]
    payload["stage_output_archive_sha256"] = store.stable_hash(body)
    _write(path, payload, 0o600)
    with pytest.raises(store.HoVerLifecycleStoreError, match="archive binding"):
        store.load_stage_output_archive(project=root, block="A_form")
