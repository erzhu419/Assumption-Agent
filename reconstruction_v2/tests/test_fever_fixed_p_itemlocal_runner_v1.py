from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import threading
import time
from typing import Any, Mapping, Sequence

import pytest

from assumption_agent.models import stable_hash
from assumption_agent.benchmarks import fever_fixed_p_itemlocal_acquisition_v1 as acquisition
from assumption_agent.benchmarks import fever_fixed_p_itemlocal_runner_v1 as runner


def _write_private(path: Path, payload: Mapping[str, Any]) -> str:
    raw = runner.canonical_bytes(payload)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _action_payload(*, inconsistent_first_rank: bool = False) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for ordinal in range(runner.ITEM_COUNT):
        documents = [
            {
                "doc_id": doc_id,
                "line_number": ordinal * runner.DOCUMENTS_PER_ITEM + doc_id,
                # Repeated page titles deliberately exercise the FEVER sentence-unit
                # transfer while page_id+line_number remains unique.
                "page_id": f"Page_{ordinal}_{doc_id // 4}",
                "sentence_text": (
                    f"claimtoken{ordinal} evidence token {doc_id}"
                    if doc_id < 5
                    else f"neutral token {ordinal} {doc_id}"
                ),
            }
            for doc_id in range(runner.DOCUMENTS_PER_ITEM)
        ]
        scores = [runner.DOCUMENTS_PER_ITEM - doc_id for doc_id in range(32)]
        rank = list(range(runner.DOCUMENTS_PER_ITEM))
        if inconsistent_first_rank and ordinal == 0:
            rank[0], rank[1] = rank[1], rank[0]
        body = {
            "schema": runner.ACTION_ITEM_SCHEMA,
            "ordinal": ordinal,
            "item_id_hash": stable_hash({"private_item": ordinal}),
            "claim": f"claimtoken{ordinal}",
            "documents": documents,
            "bm25_scores": scores,
            "bm25_rank": rank,
        }
        items.append({**body, "action_item_sha256": stable_hash(body)})
    body = {
        "schema": runner.ACTION_PACK_SCHEMA,
        "version": "v1",
        "item_count": runner.ITEM_COUNT,
        "document_count_per_item": runner.DOCUMENTS_PER_ITEM,
        "labels_included": False,
        "items": items,
    }
    return {**body, "pack_sha256": stable_hash(body)}


def _label_payload(action_payload: Mapping[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for ordinal, action in enumerate(action_payload["items"]):
        body = {
            "schema": runner.LABEL_ITEM_SCHEMA,
            "ordinal": ordinal,
            "item_id_hash": action["item_id_hash"],
            "action_item_sha256": action["action_item_sha256"],
            "source_label": "SUPPORTS" if ordinal < 64 else "REFUTES",
            "gold_indices": [0, 1],
        }
        items.append({**body, "label_item_sha256": stable_hash(body)})
    body = {
        "schema": runner.LABEL_PACK_SCHEMA,
        "version": "v1",
        "item_count": runner.ITEM_COUNT,
        "gold_contract": "one_preselected_gold_evidence_set_per_item",
        "items": items,
    }
    return {**body, "pack_sha256": stable_hash(body)}


def _load_packs(tmp_path: Path) -> tuple[runner.ActionPack, runner.LabelPack]:
    action_payload = _action_payload()
    action_path = tmp_path / "action_pack.json"
    action_file_hash = _write_private(action_path, action_payload)
    action_set_hash = stable_hash(
        [row["action_item_sha256"] for row in action_payload["items"]]
    )
    action_pack = runner.load_action_pack(
        action_path,
        expected_file_sha256=action_file_hash,
        expected_item_commitment_set_sha256=action_set_hash,
    )
    label_payload = _label_payload(action_payload)
    label_path = tmp_path / "label_pack.json"
    label_file_hash = _write_private(label_path, label_payload)
    label_set_hash = stable_hash(
        [row["label_item_sha256"] for row in label_payload["items"]]
    )
    label_pack = runner.load_label_pack(
        label_path,
        expected_file_sha256=label_file_hash,
        expected_item_commitment_set_sha256=label_set_hash,
    )
    return action_pack, label_pack


class _FakeRuntime:
    def __init__(self, *, fail_ordinal: int | None = None) -> None:
        self.fail_ordinal = fail_ordinal
        self.calls = 0
        self.postflight_calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self._binding = {
            "schema": "fake_offline_runtime_for_unit_test",
            "binding_sha256": stable_hash({"fake": "binding"}),
        }

    @property
    def safe_binding(self) -> Mapping[str, Any]:
        return dict(self._binding)

    def retrieve(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]:
        # This fake's narrow signature also proves that BM25 fields cannot be
        # forwarded to the official arm.
        assert question.startswith("claimtoken")
        assert len(paragraphs) == runner.DOCUMENTS_PER_ITEM
        assert all(set(row) == {"idx", "title", "paragraph_text"} for row in paragraphs)
        ordinal = int(work_root.name.rsplit("_", 1)[1])
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.0005)
            if ordinal == self.fail_ordinal:
                raise RuntimeError("deliberate fake runtime failure")
            return (0, 1, 2, 3, 4)
        finally:
            with self._lock:
                self.active -= 1

    def fresh_reverify(self) -> Mapping[str, Any]:
        self.postflight_calls += 1
        return dict(self._binding)


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def test_runner_required_freeze_paths_are_acquisition_bound() -> None:
    assert runner._REQUIRED_FREEZE_PATHS.issubset(
        set(acquisition.REQUIRED_FREEZE_PATHS)
    )


def test_exact_program_and_strict_bm25_score_rank_contract(tmp_path: Path) -> None:
    assert runner.exact_fixed_p().program_hash == runner.FIXED_P_PROGRAM_SHA256
    payload = _action_payload(inconsistent_first_rank=True)
    path = tmp_path / "bad_action.json"
    file_hash = _write_private(path, payload)
    set_hash = stable_hash(
        [row["action_item_sha256"] for row in payload["items"]]
    )
    with pytest.raises(runner.FeverFixedPRunnerError, match="score/rank"):
        runner.load_action_pack(
            path,
            expected_file_sha256=file_hash,
            expected_item_commitment_set_sha256=set_hash,
        )


def test_all_actions_seal_before_late_labels_and_score_set_aware(
    tmp_path: Path,
) -> None:
    action_pack, label_pack = _load_packs(tmp_path)
    runtime = _FakeRuntime()
    seal_path = tmp_path / "formal.action.seal.json"
    callback_calls = 0

    def late_labels() -> runner.LabelPack:
        nonlocal callback_calls
        callback_calls += 1
        assert seal_path.is_file()
        assert stat.S_IMODE(seal_path.stat().st_mode) == 0o600
        seal = json.loads(seal_path.read_bytes())
        assert seal["labels_opened_before_action_seal"] is False
        assert seal["work_unit_count"] == 384
        assert len(seal["action_rows"]) == runner.ITEM_COUNT
        return label_pack

    outcome = runner.run_measurement(
        action_pack,
        late_label_loader=late_labels,
        program=runner.exact_fixed_p(),
        runtime=runtime,
        work_root=tmp_path / "work",
        action_seal_path=seal_path,
        acquisition_sha256="a" * 64,
    )
    assert callback_calls == 1
    assert runtime.calls == runner.ITEM_COUNT
    assert runtime.postflight_calls == 1
    assert runtime.max_active <= runner.OFFICIAL_CONCURRENCY_CAP
    official = outcome.arm_metrics["official_HippoRAG"]
    assert official["overall"] == {
        "set_aware_support_hit_count_at_5": 256,
        "set_aware_support_total": 256,
        "micro_set_aware_support_recall_at_5": 1.0,
        "complete_item_count": 128,
        "item_count": 128,
    }
    assert official["by_source_label"]["SUPPORTS"]["complete_item_count"] == 64
    assert official["by_source_label"]["REFUTES"]["complete_item_count"] == 64

    public = runner.aggregate_result_body(outcome)
    forbidden = {
        "action_item_sha256",
        "bm25_rank",
        "bm25_scores",
        "claim",
        "documents",
        "gold_indices",
        "item_id_hash",
        "label_item_sha256",
        "rankings",
        "scores",
        "p_value",
        "promotion",
        "gate",
        "stage",
    }
    assert not (_walk_keys(public) & forbidden)
    assert public["public_payload_contract"]["aggregate_only"] is True


def test_action_failure_joins_official_wave_postflights_and_never_opens_labels(
    tmp_path: Path,
) -> None:
    action_pack, label_pack = _load_packs(tmp_path)
    runtime = _FakeRuntime(fail_ordinal=7)
    label_calls = 0

    def forbidden_labels() -> runner.LabelPack:
        nonlocal label_calls
        label_calls += 1
        return label_pack

    seal_path = tmp_path / "must_not_exist.action.seal.json"
    with pytest.raises(runner.FeverFixedPRunnerError, match="action barrier"):
        runner.run_measurement(
            action_pack,
            late_label_loader=forbidden_labels,
            program=runner.exact_fixed_p(),
            runtime=runtime,
            work_root=tmp_path / "failed_work",
            action_seal_path=seal_path,
        )
    assert runtime.calls == runner.ITEM_COUNT
    assert runtime.postflight_calls == 1
    assert label_calls == 0
    assert not seal_path.exists()


def test_label_pack_requires_fixed_single_set_and_balanced_strata(tmp_path: Path) -> None:
    action_payload = _action_payload()
    label_payload = _label_payload(action_payload)
    label_payload["items"][0]["source_label"] = "SUPPORTS"
    label_payload["items"][64]["source_label"] = "SUPPORTS"
    for row in (label_payload["items"][0], label_payload["items"][64]):
        body = dict(row)
        body.pop("label_item_sha256")
        row["label_item_sha256"] = stable_hash(body)
    body = dict(label_payload)
    body.pop("pack_sha256")
    label_payload["pack_sha256"] = stable_hash(body)
    path = tmp_path / "imbalanced_labels.json"
    file_hash = _write_private(path, label_payload)
    set_hash = stable_hash(
        [row["label_item_sha256"] for row in label_payload["items"]]
    )
    with pytest.raises(runner.FeverFixedPRunnerError, match="64/64"):
        runner.load_label_pack(
            path,
            expected_file_sha256=file_hash,
            expected_item_commitment_set_sha256=set_hash,
        )
